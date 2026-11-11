"""``driver.transport``: tcp, unix and direct reach the same policy, alike.

A policy the run serves itself (``driver.policy``) is reached over a TCP port, a
Unix socket, or by calling its servicer directly; whichever it is, the policy must
see the same requests and the run the same answers and the same errors.
"""

from __future__ import annotations

import shutil
import stat
import tempfile
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, Iterator, List

import grpc
import numpy as np
import pytest
import yaml

from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.protocol import (
    CarlaRendererData,
    egodriver_pb2_grpc,
    pack_lidar_sweep,
    pack_renderer_data,
)

from autoware_carla_scenario.driver import egodriver_client
from autoware_carla_scenario.driver.base import (
    DRIVER_TRANSPORTS,
    DriverClientConfig,
    EgoObservation,
)
from autoware_carla_scenario.driver.egodriver_client import EgoDriverGrpcClient
from autoware_carla_scenario.driver.geometry import Pose, Trajectory

from .test_egodriver_client import _StubPolicy

_DEFAULT_YAML = (
    Path(egodriver_client.__file__).parents[1]
    / "examples"
    / "conf"
    / "driver"
    / "default.yaml"
)


class _Records(BaseDriver):
    """Deterministic, and keeps everything the servicer handed it."""

    name = "records"

    def __init__(self) -> None:
        self.seen: List[Any] = []

    def on_session_start(self, session) -> None:  # noqa: ANN001
        self.seen.append(("start", session.uuid, session.seed, sorted(session.cameras)))

    def on_route(self, session) -> None:  # noqa: ANN001
        self.seen.append(
            (
                "route",
                session.route_timestamp_us,
                session.route_waypoints_in_rig.tobytes(),
            )
        )

    def drive(self, ctx: DriveContext) -> DriveResult:
        renderer = ctx.renderer_data.SerializeToString() if ctx.renderer_data else b""
        poses = ctx.session.ego_trajectory
        self.seen.append(
            (
                "drive",
                ctx.time_now_us,
                ctx.time_query_us,
                renderer,
                list(poses.timestamps_us),
            )
        )
        plan = Trajectory.empty()
        for k in range(1, 6):
            plan.append(
                ctx.time_now_us + k * 100_000,
                Pose.from_xyz_yaw(0.8 * k, 0.05 * k, 0.0, 0.01 * k),
            )
        return DriveResult(trajectory_in_rig=plan, debug_scalars={"k": 1.0})


class _Refuses(_Records):
    name = "refuses"

    def on_session_start(self, session) -> None:  # noqa: ANN001
        raise ValueError("needs a camera called 'nope'")


@pytest.fixture
def policies(monkeypatch: pytest.MonkeyPatch) -> Iterator[Dict[str, BaseDriver]]:
    """``driver.policy`` names resolved to test policies, loaded afresh per test."""
    loaded: Dict[str, BaseDriver] = {}

    def load(spec: str) -> BaseDriver:
        loaded[spec] = _Refuses() if spec.startswith("refuses") else _Records()
        return loaded[spec]

    monkeypatch.setattr(egodriver_client, "load_policy", load)
    caches = (
        egodriver_client._policy,
        egodriver_client._serve_in_process,
        egodriver_client._serve_on_unix_socket,
        egodriver_client._in_process_servicer,
    )
    for cache in caches:
        cache.cache_clear()
    yield loaded
    for cache in caches:
        cache.cache_clear()


def _observation(timestamp_us: int) -> EgoObservation:
    return EgoObservation(
        timestamp_us=timestamp_us,
        pose=Pose.from_xyz_yaw(10.0 + timestamp_us * 1e-5, 20.0, 0.0, 0.3),
        linear_velocity=np.array([5.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.1]),
        linear_acceleration=np.array([1.0, 0.0, 0.0]),
        speed_mps=5.0,
    )


def _renderer_data() -> bytes:
    points = np.random.default_rng(3).normal(size=(50_000, 4)).astype(np.float32)
    sweep = pack_lidar_sweep("lidar_top", 50_000, points, Pose.identity().to_proto())
    return pack_renderer_data(CarlaRendererData(map_id="Town10HD_Opt", lidar=[sweep]))


def _session(transport: str, spec: str) -> List[Any]:
    """Two policy steps through ``EgoDriverGrpcClient``; what the run got back."""
    config = DriverClientConfig(
        address="nowhere.invalid:1",
        policy=spec,
        transport=transport,
        cameras=(),
        timeout_s=10.0,
    )
    client = EgoDriverGrpcClient(config)
    answers: List[Any] = []
    try:
        client.start_session("session-1", "scene")
        for step in range(2):
            now = (step + 1) * 100_000
            route = np.array([[2.0 * k, 0.1 * k, 0.0] for k in range(10)]) + step
            client.submit_route(now, route)
            client.submit_egomotion_observation(
                [_observation(now - 50_000), _observation(now)]
            )
            outcome = client.drive(now, now + 100_000, _renderer_data())
            answers.append(
                (
                    outcome.trajectory.to_proto().SerializeToString(),
                    outcome.terminate_session,
                    outcome.policy_name,
                    outcome.debug_scalars,
                    outcome.turn_indicator,
                )
            )
    finally:
        client.close_session()
    return answers


# -- configuration -----------------------------------------------------------------


def test_tcp_is_the_default_everywhere() -> None:
    assert DRIVER_TRANSPORTS == ("tcp", "unix", "direct")
    assert DriverClientConfig().transport == "tcp"
    assert yaml.safe_load(_DEFAULT_YAML.read_text())["driver"]["transport"] == "tcp"
    assert (
        DriverClientConfig.from_mapping(
            {"transport": "direct", "policy": "p"}
        ).transport
        == "direct"
    )


def test_an_unknown_transport_is_refused() -> None:
    with pytest.raises(
        ValueError, match="driver.transport must be one of tcp, unix, direct, got 'udp'"
    ):
        DriverClientConfig(transport="udp")


def test_direct_is_only_for_a_policy_the_run_serves() -> None:
    with pytest.raises(
        ValueError, match="driver.transport=direct calls a policy this run serves"
    ):
        DriverClientConfig(transport="direct")


def test_unix_without_a_policy_dials_a_unix_address() -> None:
    with pytest.raises(ValueError, match="must name a Unix socket"):
        DriverClientConfig(transport="unix", address="localhost:50051")
    DriverClientConfig(transport="unix", address="unix:///run/policy.sock")
    DriverClientConfig(transport="unix", policy="route_follower")  # serves its own


# -- the same policy, however reached -------------------------------------------------


def test_every_transport_hands_the_policy_and_the_run_the_same(
    policies: Dict[str, BaseDriver],
) -> None:
    answers = {t: _session(t, f"records-{t}") for t in DRIVER_TRANSPORTS}
    seen = {t: policies[f"records-{t}"].seen for t in DRIVER_TRANSPORTS}  # type: ignore[attr-defined]
    for transport in DRIVER_TRANSPORTS:
        assert answers[transport] == answers["tcp"], transport
        assert seen[transport] == seen["tcp"], transport
    assert [entry[0] for entry in seen["tcp"]] == [
        "start",
        "route",
        "drive",
        "route",
        "drive",
    ]
    assert len(Trajectory.from_proto(_proto(answers["tcp"][0][0])).poses) == 6


def _proto(raw: bytes):  # noqa: ANN202
    from carla_driver_interface.protocol import common_pb2

    return common_pb2.Trajectory.FromString(raw)


def test_every_transport_reports_a_refused_session_alike(
    policies: Dict[str, BaseDriver],
) -> None:
    errors = {}
    for transport in DRIVER_TRANSPORTS:
        with pytest.raises(grpc.RpcError) as error:
            _session(transport, f"refuses-{transport}")
        errors[transport] = (error.value.code(), error.value.details())
    assert set(errors.values()) == {
        (grpc.StatusCode.FAILED_PRECONDITION, "refuses: needs a camera called 'nope'")
    }


def test_direct_keeps_one_policy_for_every_scenario(
    policies: Dict[str, BaseDriver],
) -> None:
    first = _session("direct", "records")
    second = _session("direct", "records")  # the next scenario: a fresh client
    assert first == second
    assert list(policies) == ["records"]  # loaded once
    starts = [entry for entry in policies["records"].seen if entry[0] == "start"]  # type: ignore[attr-defined]
    assert len(starts) == 2


def test_the_runs_unix_socket_is_its_own(policies: Dict[str, BaseDriver]) -> None:
    address = egodriver_client._serve_on_unix_socket("records")
    path = Path(address.removeprefix("unix://"))
    assert path.is_socket()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    # $TMPDIR, or /tmp when $TMPDIR would make the path too long for a socket.
    assert path.parent.parent in (Path(tempfile.gettempdir()), Path("/tmp"))
    # A second run in this process (another scenario) reuses it.
    assert egodriver_client._serve_on_unix_socket("records") == address


@pytest.fixture
def short_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """*tmp_path* as ``$TMPDIR``, with the unix transport's socket path check out of the way.

    The socket path limit is lifted so a long *tmp_path* never falls back to a real
    ``/tmp`` directory.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(egodriver_client, "_MAX_SOCKET_PATH", 1 << 16)
    yield tmp_path


def test_a_policy_that_fails_to_load_leaves_no_socket_directory(
    policies: Dict[str, BaseDriver],
    short_tmpdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad(spec: str) -> BaseDriver:
        raise ImportError("no such policy")

    monkeypatch.setattr(egodriver_client, "load_policy", bad)
    for _ in range(3):  # every case of a batch retries: lru_cache keeps no exception
        with pytest.raises(ImportError, match="no such policy"):
            egodriver_client._serve_on_unix_socket("foo:Bar")
    assert list(short_tmpdir.iterdir()) == []


@pytest.mark.parametrize("failing", ["build", "start"])
def test_a_server_that_fails_to_start_leaves_no_socket_directory(
    policies: Dict[str, BaseDriver],
    short_tmpdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
) -> None:
    class _Server:
        def start(self) -> None:
            raise RuntimeError("cannot bind")

    def build(policy: BaseDriver, **kwargs: Any) -> Any:
        assert Path(kwargs["host"].removeprefix("unix://")).parent.is_dir()
        if failing == "build":
            raise RuntimeError("cannot bind")
        return _Server(), 0

    monkeypatch.setattr(egodriver_client, "build_server", build)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="cannot bind"):
            egodriver_client._serve_on_unix_socket("records")
    assert list(short_tmpdir.iterdir()) == []
    assert list(policies) == ["records"]  # the policy itself loaded, once


def test_unix_dials_a_policy_in_another_process() -> None:
    """A policy serving on a Unix socket is dialled at driver.address; any release serves one."""
    stub = _StubPolicy()
    directory = tempfile.mkdtemp(prefix="acs-test-")
    address = f"unix://{directory}/policy.sock"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(stub, server)
    server.add_insecure_port(address)
    server.start()
    config = DriverClientConfig(
        address=address, transport="unix", cameras=(), timeout_s=10.0
    )
    client = EgoDriverGrpcClient(config)
    try:
        client.start_session("session-1", "scene")
        outcome = client.drive(0, 100_000)
    finally:
        client.close_session()
        # stop() unlinks the socket asynchronously: wait for it, then remove
        # whatever is left, so the directory is never "not empty".
        server.stop(grace=None).wait()
        shutil.rmtree(directory, ignore_errors=True)
    assert len(outcome.trajectory.poses) == 4
    assert [s.session_uuid for s in stub.sessions] == ["session-1"]
