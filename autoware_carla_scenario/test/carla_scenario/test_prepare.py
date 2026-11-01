"""Backends a run needs started before it (autoware_carla_scenario.prepare).

No docker runs here: the backends are fakes, and docker's command line is
recorded rather than run.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Mapping, Optional

import pytest

from autoware_carla_scenario import SmartPedestrianControllerAction
from autoware_carla_scenario.pedestrian.ardy import DEFAULT_IMAGE, ArdyBackend
from autoware_carla_scenario.prepare import (
    Backend,
    BackendPreparer,
    InsufficientResourcesError,
    Resources,
)
from autoware_carla_scenario.prepare import docker as docker_module
from autoware_carla_scenario.prepare import host as host_module
from autoware_carla_scenario.prepare import pool as pool_module
from autoware_carla_scenario.prepare.declare import (
    action_backends,
    need_prepare,
    required_backends,
    scenario_backends,
)
from autoware_carla_scenario.prepare.host import Gpu, HostResources, assign_gpus
from autoware_carla_scenario.prepare.pool import BackendPool, running_backend


class _Fake:
    """A backend that records what was done to it."""

    backend_name: ClassVar[str] = "fake"
    built: ClassVar[list["_Fake"]] = []

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)
        self.started: Optional[tuple[Optional[Path], Optional[int]]] = None
        self.closed = 0
        _Fake.built.append(self)

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "_Fake":
        return cls(settings)

    @property
    def name(self) -> str:
        return self.backend_name

    def resources(self) -> Resources:
        return Resources(
            gpu_memory_mb=int(self.settings.get("gpu", 0)),
            memory_mb=int(self.settings.get("ram", 0)),
        )

    def start(self, log_dir: Optional[Path] = None, gpu: Optional[int] = None) -> None:
        if self.settings.get("fail"):
            raise RuntimeError("would not start")
        self.started = (log_dir, gpu)

    def address(self) -> str:
        return "127.0.0.1:1"

    def close(self) -> None:
        self.closed += 1


class _Other(_Fake):
    backend_name: ClassVar[str] = "other"


@pytest.fixture(autouse=True)
def _clean() -> Any:
    _Fake.built.clear()
    yield
    pool_module._ACTIVE = None


def test_the_contract_is_what_the_pool_and_the_backends_are() -> None:
    assert isinstance(BackendPool(), BackendPreparer)
    assert isinstance(_Fake({}), Backend)
    assert isinstance(ArdyBackend.from_settings({}), Backend)


class TestNeedPrepare:
    def test_a_class_declares_the_backends_it_needs(self) -> None:
        @need_prepare(_Fake)
        class Action:
            pass

        assert required_backends(Action) == (_Fake,)
        assert required_backends(Action()) == (_Fake,)

    def test_a_subclass_needs_its_bases_backends_and_its_own(self) -> None:
        @need_prepare(_Fake)
        class Base:
            pass

        @need_prepare(_Other, _Fake)
        class Derived(Base):
            pass

        assert required_backends(Derived) == (_Fake, _Other)
        assert required_backends(Base) == (_Fake,)

    def test_an_undecorated_class_needs_nothing(self) -> None:
        assert required_backends(object()) == ()

    def test_only_a_backend_class_is_taken(self) -> None:
        with pytest.raises(TypeError, match="backend_name"):
            need_prepare(object)

    def test_the_smart_pedestrian_needs_ardy(self) -> None:
        assert required_backends(SmartPedestrianControllerAction) == (ArdyBackend,)

    def test_actions_are_scanned_each_backend_once(self) -> None:
        @need_prepare(_Fake)
        class A:
            pass

        @need_prepare(_Other, _Fake)
        class B:
            pass

        assert action_backends([A(), B(), A()]) == (_Fake, _Other)

    def test_a_declarative_scenarios_actions_are_read_before_setup(self) -> None:
        scenario = SimpleNamespace(
            compiled=SimpleNamespace(
                actions=[
                    SimpleNamespace(
                        spec=SimpleNamespace(target="..actions:WalkStraightAction")
                    ),
                    SimpleNamespace(
                        spec=SimpleNamespace(
                            target="..actions:SmartPedestrianControllerAction"
                        )
                    ),
                ]
            )
        )
        assert scenario_backends(scenario) == (ArdyBackend,)


class TestBackendPool:
    def test_each_backend_starts_once_with_its_settings(self, tmp_path: Path) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        pool = BackendPool({"fake": {"x": 1}}, check_resources=False)
        pool.prepare_scenarios([Scenario(), Scenario()], log_dir=tmp_path)
        pool.prepare_actions([Scenario()], log_dir=tmp_path)
        assert len(_Fake.built) == 1
        assert _Fake.built[0].settings == {"x": 1}
        assert _Fake.built[0].started == (tmp_path, None)
        assert pool.backend("fake") is _Fake.built[0]

    def test_close_stops_every_backend(self) -> None:
        @need_prepare(_Fake, _Other)
        class Scenario:
            pass

        pool = BackendPool(check_resources=False)
        pool.prepare_scenarios([Scenario()])
        pool.close()
        pool.close()
        assert [b.closed for b in _Fake.built] == [1, 1]
        assert pool.backend("fake") is None

    def test_a_host_without_room_refuses_before_anything_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        monkeypatch.setattr(
            pool_module,
            "probe_host",
            lambda: HostResources(gpus=(Gpu(0, "GPU-a", 8000),), memory_mb=64000),
        )
        pool = BackendPool({"fake": {"gpu": 24000, "ram": 1000}})
        with pytest.raises(InsufficientResourcesError, match="24000 MiB of GPU memory"):
            pool.prepare_scenarios([Scenario()])
        assert _Fake.built[0].started is None

    def test_a_backend_is_started_on_the_gpu_chosen_for_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        monkeypatch.setattr(
            pool_module,
            "probe_host",
            lambda: HostResources(
                gpus=(Gpu(0, "GPU-a", 8000), Gpu(1, "GPU-b", 40000)), memory_mb=64000
            ),
        )
        BackendPool({"fake": {"gpu": 24000}}).prepare_scenarios([Scenario()])
        assert _Fake.built[0].started == (None, 1)

    def test_a_backend_that_fails_to_start_is_closed(self) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        pool = BackendPool({"fake": {"fail": True}}, check_resources=False)
        with pytest.raises(RuntimeError, match="would not start"):
            pool.prepare_scenarios([Scenario()])
        assert _Fake.built[0].closed == 1
        assert pool.backend("fake") is None

    def test_an_action_finds_the_backend_its_queue_started(self) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        pool = BackendPool(check_resources=False)
        pool.prepare_scenarios([Scenario()])
        assert running_backend(_Fake) is _Fake.built[0]
        pool.close()
        assert pool_module._ACTIVE is None

    def test_an_action_starts_a_backend_no_queue_prepared(self) -> None:
        pool = BackendPool(check_resources=False)
        pool_module._ACTIVE = pool
        backend = running_backend(_Fake)
        assert backend is _Fake.built[0]
        assert pool.backend("fake") is backend


class TestAssignGpus:
    def test_host_memory_is_added_up(self) -> None:
        host = HostResources(gpus=(), memory_mb=10000)
        with pytest.raises(InsufficientResourcesError, match="12000 MiB needed"):
            assign_gpus(
                [("a", Resources(memory_mb=6000)), ("b", Resources(memory_mb=6000))],
                host,
            )

    def test_a_gpu_is_needed_when_none_is_visible(self) -> None:
        with pytest.raises(InsufficientResourcesError, match="none is visible"):
            assign_gpus([("a", Resources(gpu_memory_mb=1))], HostResources((), 1000))

    def test_two_backends_share_a_gpu_with_room_for_both(self) -> None:
        host = HostResources(gpus=(Gpu(0, "GPU-a", 50000),), memory_mb=1000)
        placed = assign_gpus(
            [
                ("a", Resources(gpu_memory_mb=24000)),
                ("b", Resources(gpu_memory_mb=24000)),
            ],
            host,
        )
        assert placed == {"a": host.gpus[0], "b": host.gpus[0]}

    def test_two_backends_that_do_not_fit_one_gpu_are_refused(self) -> None:
        host = HostResources(gpus=(Gpu(0, "GPU-a", 40000),), memory_mb=1000)
        with pytest.raises(InsufficientResourcesError, match="b: 24000"):
            assign_gpus(
                [
                    ("a", Resources(gpu_memory_mb=24000)),
                    ("b", Resources(gpu_memory_mb=24000)),
                ],
                host,
            )

    def test_a_backend_set_to_a_gpu_is_checked_on_it(self) -> None:
        host = HostResources(
            gpus=(Gpu(0, "GPU-a", 50000), Gpu(1, "GPU-b", 1000)), memory_mb=1000
        )
        with pytest.raises(InsufficientResourcesError, match="set to .1."):
            assign_gpus([("a", Resources(gpu_memory_mb=24000, gpu=1))], host)
        with pytest.raises(InsufficientResourcesError, match="GPU 2, which is not"):
            assign_gpus([("a", Resources(gpu_memory_mb=1, gpu=2))], host)
        placed = assign_gpus([("a", Resources(gpu_memory_mb=24000, gpu=0))], host)
        assert placed == {"a": host.gpus[0]}

    def test_only_the_visible_gpus_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpus = (Gpu(0, "GPU-aaaa", 1), Gpu(1, "GPU-bbbb", 2))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        assert host_module._visible(gpus) == (gpus[1],)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aa")
        assert host_module._visible(gpus) == (gpus[0],)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        assert host_module._visible(gpus) == gpus


class _Docker:
    """``subprocess.run`` of docker, recorded."""

    def __init__(self, listing: str = "", image_present: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.listing = listing
        self.image_present = image_present

    def __call__(self, argv: list[str], **_: Any) -> Any:
        self.calls.append(list(argv))
        out, code = "", 0
        if argv[1] == "info":
            out = '{"nvidia":{},"runc":{}}'
        elif argv[1] == "ps":
            out = self.listing
        elif argv[1:3] == ["image", "inspect"]:
            code = 0 if self.image_present else 1
        elif argv[1] == "run":
            out = "c0ffee\n"
        elif argv[1] == "inspect":
            out = "true\n"
        return subprocess.CompletedProcess(argv, code, out, "")


class _Answering(ArdyBackend):
    def ready(self, timeout_s: float) -> bool:
        return True


class TestDockerBackend:
    def test_the_container_runs_on_its_gpu_with_its_port_cache_and_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _Docker()
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)
        monkeypatch.setenv("HF_TOKEN", "secret")
        backend = _Answering.from_settings({"hf_home": "~/hf", "model": "core"})
        backend.start(None, gpu=1)
        (run,) = [c for c in recorded.calls if c[1] == "run"]
        assert run[run.index("--gpus") + 1] == "device=1"
        published = run[run.index("--publish") + 1]
        assert published == f"127.0.0.1:{backend.address().split(':')[1]}:50061"
        assert f"{os.path.expanduser('~/hf')}:/hf" in run
        assert "ARDY_MODEL=core" in run
        # Passed on by name: the token never shows in the command line.
        assert "HF_TOKEN" in run and "secret" not in " ".join(run)
        assert run[-1] == DEFAULT_IMAGE
        assert "--rm" not in run
        backend.close()
        assert ["docker", "stop", "--time", "10", "c0ffee"] in recorded.calls
        assert ["docker", "rm", "--force", "c0ffee"] in recorded.calls

    @pytest.mark.parametrize(
        "gpus, expected",
        [
            ("none", Resources(memory_mb=16000)),
            ("device=1", Resources(gpu_memory_mb=24000, memory_mb=16000, gpu=1)),
            ("auto", Resources(gpu_memory_mb=24000, memory_mb=16000)),
        ],
    )
    def test_what_a_container_takes_follows_its_gpus(
        self, gpus: str, expected: Resources
    ) -> None:
        assert ArdyBackend.from_settings({"gpus": gpus}).resources() == expected

    def test_a_missing_image_is_pulled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = _Docker(image_present=False)
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)
        _Answering.from_settings({}).start()
        assert ["docker", "pull", DEFAULT_IMAGE] in recorded.calls

    def test_only_a_stopped_runs_leftover_is_removed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = docker_module.socket.gethostname()
        recorded = _Docker(
            listing=(
                f"dead {host}:999999999\n"
                f"alive {host}:{os.getpid()}\n"
                f"elsewhere other-host:999999999\n"
            )
        )
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)
        _Answering.from_settings({}).start()
        removed = [c[-1] for c in recorded.calls if c[1] == "rm"]
        assert removed == ["dead"]


class TestQueue:
    def test_a_scenario_added_after_the_start_is_prepared_then(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock

        from autoware_carla_scenario.scenario_queue import ScenarioQueue

        prepared: list[list[object]] = []
        preparer = SimpleNamespace(
            prepare_scenarios=lambda scenarios, log_dir=None: prepared.append(
                list(scenarios)
            )
        )
        queue = ScenarioQueue(
            server=MagicMock(),
            output_dir=tmp_path,
            backends=preparer,  # type: ignore[arg-type]
        )
        before, after = MagicMock(), MagicMock()
        queue.add(before)
        assert prepared == []  # the start prepares it, with the others
        queue._runner = MagicMock()
        queue.add(after)
        assert prepared == [[after]]
