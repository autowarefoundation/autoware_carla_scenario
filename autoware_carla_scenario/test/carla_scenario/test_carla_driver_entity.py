"""Unit tests for the ego entity driven by an external policy.

The CARLA world and the ego actor are mocked, and the driver client is replaced by a
recording fake, so the whole driver loop is exercised without a simulator or a policy
server.
"""

from __future__ import annotations

import importlib.util
import math
from concurrent import futures
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import grpc
import numpy as np
import pytest

from autoware_carla_scenario.autoware_bridge import (
    AutowareBridgeConfig,
    BridgePose,
    FakeAutowareBridge,
)
from autoware_carla_scenario.driver.base import (
    BaseEgoDriverClient,
    DriveOutcome,
    DriverCameraConfig,
    DriverClientConfig,
    EgoObservation,
)
from carla_driver_interface.driver import (
    BaseDriver,
    DriveContext,
    DriveResult,
    TurnIndicator,
)
from carla_driver_interface.protocol import carla_driver_pb2
from carla_driver_interface.protocol import (
    add_CarlaVehicleServiceServicer_to_server,
    add_EgodriverServiceServicer_to_server,
)
from carla_driver_interface.server import serving
from carla_driver_interface.service import EgodriverServicer
from autoware_carla_scenario.driver.geometry import Pose, Trajectory
from autoware_carla_scenario.ego_plan import PlanMarkers
from autoware_carla_scenario.entity.carla_driver_entity import CarlaDriverEntity
from autoware_carla_scenario.t4 import withdraw_driver_sensors

from ._vehicle_physics import mkz_physics, mkz_powertrain


@pytest.fixture(autouse=True)
def _withdraw_driver_sensors():
    """Most tests here start an entity and never end it; its offer of sensors to
    a T4 dataset is process-global and must not reach another test's recording."""
    yield
    withdraw_driver_sensors()


_TICK_S = 0.05


class _FakeDriverClient(BaseEgoDriverClient):
    """Records every call and answers ``drive`` with a straight-ahead plan."""

    def __init__(
        self,
        config: DriverClientConfig,
        *,
        terminate_after: Optional[int] = None,
        turn_indicators: Sequence[int] = (),
    ) -> None:
        super().__init__(config)
        self.started: List[Tuple[str, str]] = []
        self.routes: List[np.ndarray] = []
        self.images: List[Tuple[str, bytes]] = []
        self.egomotion: List[List[EgoObservation]] = []
        self.ground_truths: List[Tuple[int, Trajectory]] = []
        self.drives: List[Tuple[int, int]] = []
        self.renderer_payloads: List[bytes] = []
        self.closed = 0
        self._terminate_after = terminate_after
        #: The turn indicator each drive answers, in order; then none.
        self._turn_indicators = list(turn_indicators)

    def start_session(self, session_uuid: str, scene_id: str) -> None:
        self.started.append((session_uuid, scene_id))

    def submit_route(self, timestamp_us: int, waypoints_in_rig) -> None:
        self.routes.append(np.asarray(waypoints_in_rig))

    def submit_image_observation(
        self,
        logical_id: str,
        frame_start_us: int,
        frame_end_us: int,
        image_bytes: bytes,
    ) -> None:
        self.images.append((logical_id, image_bytes))

    def submit_egomotion_observation(
        self, observations: Sequence[EgoObservation]
    ) -> None:
        self.egomotion.append(list(observations))

    def submit_recording_ground_truth(
        self, timestamp_us: int, trajectory_in_rig: Trajectory
    ) -> None:
        self.ground_truths.append((timestamp_us, trajectory_in_rig))

    def drive(
        self,
        time_now_us: int,
        time_query_us: int,
        renderer_data: bytes = b"",
    ) -> DriveOutcome:
        self.drives.append((time_now_us, time_query_us))
        self.renderer_payloads.append(renderer_data)
        plan = Trajectory.empty()
        for index in range(1, 5):
            plan.append(
                time_now_us + index * 100_000,
                Pose.from_xyz_yaw(index * 0.8, 0.0, 0.0, 0.0),
            )
        terminate = (
            self._terminate_after is not None
            and len(self.drives) >= self._terminate_after
        )
        step = len(self.drives) - 1
        return DriveOutcome(
            trajectory=plan,
            terminate_session=terminate,
            turn_indicator=(
                self._turn_indicators[step] if step < len(self._turn_indicators) else 0
            ),
        )

    def close_session(self) -> None:
        self.closed += 1


class _EmptyActors(list):
    """Stands in for CARLA's ActorList when the world holds only the ego."""

    def filter(self, pattern: str) -> List[MagicMock]:
        return []


def _actor(x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0) -> MagicMock:
    """Return a mock CARLA vehicle actor with usable kinematics."""
    actor = MagicMock()
    actor.get_transform.return_value = SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=0.0),
        rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw_deg),
    )
    actor.get_location.return_value = SimpleNamespace(x=x, y=y, z=0.0)
    actor.get_velocity.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    actor.get_acceleration.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    actor.get_angular_velocity.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    actor.get_physics_control.return_value = mkz_physics()
    actor.get_control.return_value = SimpleNamespace(gear=1)
    return actor


def _world_with_route() -> MagicMock:
    """Return a mock world whose road graph walks straight along +x."""
    world = MagicMock()

    def _waypoint_at(distance: float) -> MagicMock:
        waypoint = MagicMock()
        waypoint.transform.location = SimpleNamespace(x=distance, y=0.0, z=0.0)
        waypoint.next.side_effect = lambda step: [_waypoint_at(distance + step)]
        return waypoint

    world.get_map.return_value.get_waypoint.return_value = _waypoint_at(0.0)
    world.get_map.return_value.name = "Town10HD_Opt"
    return world


def _entity(
    *,
    cameras: Tuple[DriverCameraConfig, ...] = (),
    terminate_after: Optional[int] = None,
    turn_indicators: Sequence[int] = (),
    **config_overrides,
) -> Tuple[CarlaDriverEntity, _FakeDriverClient]:
    """Return an entity wired to a fake client, with the ego actor already 'spawned'."""
    config_overrides.setdefault("send_renderer_data", False)
    config = DriverClientConfig(
        cameras=cameras, rear_axle_offset_m=-1.4, **config_overrides
    )
    client = _FakeDriverClient(
        config, terminate_after=terminate_after, turn_indicators=turn_indicators
    )
    entity = CarlaDriverEntity(config, client=client)
    entity._vehicle = _actor()  # noqa: SLF001 - stands in for spawn()
    return entity, client


# ---------------------------------------------------------------------------
# Policy flags
# ---------------------------------------------------------------------------


def test_traffic_manager_is_disabled() -> None:
    """ScenarioRunner reads this flag to skip set_autopilot on the ego."""
    assert CarlaDriverEntity.use_autopilot is False


# ---------------------------------------------------------------------------
# Session start
# ---------------------------------------------------------------------------


def test_the_map_is_fetched_once() -> None:
    """CARLA rebuilds the map on every get_map() call; the walk must not pay that."""
    entity, _ = _entity(policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)
    for tick in range(4):
        entity.on_tick(world, tick * _TICK_S)

    assert world.get_map.call_count == 1


def test_the_route_rolls_forward_with_the_ego() -> None:
    """Re-sending the route each policy step keeps the horizon ahead of the vehicle."""
    entity, client = _entity(policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)

    for tick in range(3):
        entity.on_tick(world, tick * _TICK_S)

    # One at session start plus one per policy step.
    assert len(client.routes) == 4


def test_start_opens_a_session_and_sends_a_route() -> None:
    entity, client = _entity()
    world = _world_with_route()

    entity.on_scenario_start(world)

    assert len(client.started) == 1
    session_uuid, scene_id = client.started[0]
    assert session_uuid
    assert scene_id == "Town10HD_Opt"

    assert len(client.routes) == 1
    route = client.routes[0]
    assert route.shape[1] == 3
    # Waypoints are in the rig frame, so they start just ahead of the rear-axle origin.
    assert route[0][0] == pytest.approx(1.4, abs=1e-6)
    assert np.allclose(route[:, 1], 0.0, atol=1e-6)


def test_ground_truth_reaches_the_policy() -> None:
    """The payload carries traffic lights and other vehicles; the contract does not."""
    entity, client = _entity(send_renderer_data=True, policy_timestep_s=0.05)
    world = _world_with_route()
    world.get_actors.return_value = _EmptyActors()
    world.get_snapshot.return_value.frame = 1
    world.get_weather.return_value = SimpleNamespace(cloudiness=0.0)

    entity.on_scenario_start(world)
    entity.on_tick(world, 0.0)

    assert client.renderer_payloads
    payload = carla_driver_pb2.CarlaRendererData()
    payload.ParseFromString(client.renderer_payloads[0])
    assert payload.snapshot_timestamp_us == 50_000
    assert payload.map_name == "Town10HD_Opt"


def test_ground_truth_can_be_switched_off() -> None:
    """Off is indistinguishable from 'nothing to report' on the policy side."""
    entity, client = _entity(send_renderer_data=False, policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_tick(world, 0.0)

    assert client.renderer_payloads == [b""]


def test_start_before_spawn_raises() -> None:
    config = DriverClientConfig(cameras=())
    entity = CarlaDriverEntity(config, client=_FakeDriverClient(config))
    with pytest.raises(RuntimeError, match="before spawn"):
        entity.on_scenario_start(_world_with_route())


def test_a_session_that_fails_to_start_leaves_no_sensors_on_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario import t4

    entity, client = _entity()

    def _refuse(session_uuid: str, scene_id: str) -> None:
        raise ConnectionError("policy unreachable")

    monkeypatch.setattr(client, "start_session", _refuse)
    with pytest.raises(ConnectionError, match="policy unreachable"):
        entity.on_scenario_start(_world_with_route())

    assert t4._driver_rear_axle is None


def test_cameras_are_attached_and_streamed(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoware_carla_scenario.sensor.carla_camera import CarlaCameraSensor

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(CarlaCameraSensor, "get_image", lambda self: frame)

    entity, client = _entity(cameras=(DriverCameraConfig(logical_id="front"),))
    world = _world_with_route()

    entity.on_scenario_start(world)
    assert world.spawn_actor.call_count == 1

    entity.on_tick(world, 0.0)
    assert [logical_id for logical_id, _ in client.images] == ["front"]
    # The frame is JPEG encoded on the way out.
    assert client.images[0][1].startswith(b"\xff\xd8")


def test_a_camera_with_no_frame_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoware_carla_scenario.sensor.carla_camera import CarlaCameraSensor

    monkeypatch.setattr(CarlaCameraSensor, "get_image", lambda self: None)

    entity, client = _entity(cameras=(DriverCameraConfig(logical_id="front"),))
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_tick(world, 0.0)

    assert client.images == []
    # The policy is still asked to plan.
    assert len(client.drives) == 1


# ---------------------------------------------------------------------------
# Tick loop
# ---------------------------------------------------------------------------


def test_control_is_applied_every_tick() -> None:
    entity, _ = _entity()
    world = _world_with_route()
    entity.on_scenario_start(world)

    actor = entity.actor
    assert actor is not None
    for tick in range(6):
        entity.on_tick(world, tick * _TICK_S)

    assert actor.apply_control.call_count == 6


def test_the_policy_is_queried_at_the_configured_cadence() -> None:
    """0.1 s of policy timestep over a 0.05 s tick means every other tick."""
    entity, client = _entity(policy_timestep_s=0.1)
    world = _world_with_route()
    entity.on_scenario_start(world)

    for tick in range(6):
        entity.on_tick(world, tick * _TICK_S)

    assert len(client.drives) == 3
    assert [now for now, _ in client.drives] == [50_000, 150_000, 250_000]
    # Each request tells the policy where the runtime is heading next.
    assert [query for _, query in client.drives] == [150_000, 250_000, 350_000]


def test_every_tick_queries_when_the_timestep_matches_the_tick() -> None:
    entity, client = _entity(policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)

    for tick in range(4):
        entity.on_tick(world, tick * _TICK_S)

    assert len(client.drives) == 4


def test_observations_accompany_every_policy_step() -> None:
    entity, client = _entity(policy_timestep_s=0.1)
    world = _world_with_route()
    entity.on_scenario_start(world)

    for tick in range(4):
        entity.on_tick(world, tick * _TICK_S)

    assert len(client.egomotion) == len(client.drives) == 2
    assert client.egomotion[0][-1].timestamp_us == 50_000


def test_egomotion_carries_every_tick_since_the_last_step() -> None:
    """Like the upstream runtime, the history holds the ticks between policy steps."""
    entity, client = _entity(policy_timestep_s=0.1)
    world = _world_with_route()
    entity.on_scenario_start(world)

    for tick in range(4):
        entity.on_tick(world, tick * _TICK_S)

    timestamps = [
        [observation.timestamp_us for observation in batch]
        for batch in client.egomotion
    ]
    # The first step also carries the state at session start; each later one starts
    # right after the previous step, so no tick is sent twice or skipped.
    assert timestamps == [[0, 50_000], [100_000, 150_000]]


def test_the_reference_trajectory_is_off_by_default() -> None:
    entity, client = _entity(policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_tick(world, 0.0)

    assert client.ground_truths == []


def test_the_route_goes_out_as_the_reference_trajectory() -> None:
    entity, client = _entity(send_ground_truth=True, policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_tick(world, 0.0)

    # Sent alongside every route: once at session start, once per policy step.
    assert len(client.ground_truths) == len(client.routes) == 2
    timestamp_us, reference = client.ground_truths[-1]
    assert timestamp_us == 50_000
    assert np.allclose(reference.positions, client.routes[-1])
    assert reference.timestamps_us[:3] == [50_000, 100_000, 150_000]
    # The route runs straight ahead, so every pose faces forward.
    assert all(pose.yaw == pytest.approx(0.0, abs=1e-9) for pose in reference.poses)


def test_ticks_before_the_session_opens_do_nothing() -> None:
    entity, client = _entity()
    world = _world_with_route()
    actor = entity.actor
    assert actor is not None

    entity.on_tick(world, 0.0)

    assert client.drives == []
    assert actor.apply_control.call_count == 0


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def test_policy_termination_is_reported() -> None:
    entity, _ = _entity(terminate_after=2, policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)

    entity.on_tick(world, 0.0)
    assert entity.termination_requested is False

    entity.on_tick(world, _TICK_S)
    assert entity.termination_requested is True


# CARLA's VehicleLightState bits.
_LOW_BEAM, _RIGHT_BLINKER, _LEFT_BLINKER = 0x2, 0x10, 0x20


def _light_states(actor: MagicMock, initial: int) -> List[int]:
    """Have *actor* keep its light state as CARLA does; the states it was set to."""
    state = [initial]
    set_to: List[int] = []

    def _set(value: int) -> None:
        state[0] = int(value)
        set_to.append(int(value))

    actor.get_light_state.side_effect = lambda: state[0]
    actor.set_light_state.side_effect = _set
    return set_to


def test_the_policy_s_turn_indicator_lights_the_ego() -> None:
    entity, _ = _entity(
        policy_timestep_s=0.05,
        turn_indicators=(
            TurnIndicator.ENABLE_LEFT,
            TurnIndicator.ENABLE_LEFT,
            TurnIndicator.NO_COMMAND,
            TurnIndicator.ENABLE_RIGHT,
            TurnIndicator.DISABLE,
        ),
    )
    actor = entity.actor
    assert actor is not None
    set_to = _light_states(actor, _LOW_BEAM)
    world = _world_with_route()
    entity.on_scenario_start(world)

    shown = []
    for tick in range(5):
        entity.on_tick(world, tick * _TICK_S)
        shown.append(entity.turn_indicator)

    # Held until the policy asks for another; the headlights stay on throughout.
    assert shown == [
        TurnIndicator.ENABLE_LEFT,
        TurnIndicator.ENABLE_LEFT,
        TurnIndicator.ENABLE_LEFT,
        TurnIndicator.ENABLE_RIGHT,
        TurnIndicator.DISABLE,
    ]
    assert set_to == [
        _LOW_BEAM | _LEFT_BLINKER,
        _LOW_BEAM | _RIGHT_BLINKER,
        _LOW_BEAM,
    ]


def test_the_policy_s_next_command_puts_back_what_a_scenario_changed() -> None:
    """As docs/driver_interface.md says: an action on the ego holds until then."""
    entity, _ = _entity(
        policy_timestep_s=0.05,
        turn_indicators=(TurnIndicator.ENABLE_LEFT, TurnIndicator.ENABLE_LEFT),
    )
    actor = entity.actor
    assert actor is not None
    set_to = _light_states(actor, 0)
    world = _world_with_route()
    entity.on_scenario_start(world)

    entity.on_tick(world, 0.0)
    actor.set_light_state(_RIGHT_BLINKER)  # a TurnIndicatorAction on the ego
    entity.on_tick(world, _TICK_S)

    assert set_to == [_LEFT_BLINKER, _RIGHT_BLINKER, _LEFT_BLINKER]


def test_a_policy_that_never_signals_leaves_the_lights_alone() -> None:
    entity, _ = _entity(policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)
    for tick in range(3):
        entity.on_tick(world, tick * _TICK_S)

    actor = entity.actor
    assert actor is not None
    actor.set_light_state.assert_not_called()
    assert entity.turn_indicator == TurnIndicator.NO_COMMAND


def test_an_unknown_turn_indicator_is_ignored() -> None:
    entity, _ = _entity(policy_timestep_s=0.05, turn_indicators=(7,))
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_tick(world, 0.0)

    actor = entity.actor
    assert actor is not None
    actor.set_light_state.assert_not_called()
    assert entity.turn_indicator == TurnIndicator.NO_COMMAND


class _SignalsLeft(BaseDriver):
    name = "signals_left"

    def drive(self, ctx: DriveContext) -> DriveResult:
        return DriveResult(
            trajectory_in_rig=Trajectory.empty(),
            turn_indicator=TurnIndicator.ENABLE_LEFT,
        )


def test_the_turn_indicator_arrives_over_the_wire() -> None:
    with serving(_SignalsLeft(), port=0, host="127.0.0.1") as port:
        config = DriverClientConfig(
            address=f"127.0.0.1:{port}",
            cameras=(),
            rear_axle_offset_m=-1.4,
            send_renderer_data=False,
            policy_timestep_s=0.05,
            timeout_s=10.0,
        )
        entity = CarlaDriverEntity(config)
        entity._vehicle = _actor()  # noqa: SLF001 - stands in for spawn()
        set_to = _light_states(entity._vehicle, 0)  # noqa: SLF001
        world = _world_with_route()
        entity.on_scenario_start(world)
        entity.on_tick(world, 0.0)
        entity.on_scenario_end(world)

    assert entity.turn_indicator == TurnIndicator.ENABLE_LEFT
    assert set_to == [_LEFT_BLINKER]


def test_a_policy_without_the_vehicle_service_commands_nothing() -> None:
    """A policy server older than CarlaVehicleService still drives; it signals nothing."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_EgodriverServiceServicer_to_server(EgodriverServicer(_SignalsLeft()), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        config = DriverClientConfig(
            address=f"127.0.0.1:{port}",
            cameras=(),
            rear_axle_offset_m=-1.4,
            send_renderer_data=False,
            policy_timestep_s=0.05,
            timeout_s=10.0,
        )
        entity = CarlaDriverEntity(config)
        entity._vehicle = _actor()  # noqa: SLF001 - stands in for spawn()
        world = _world_with_route()
        entity.on_scenario_start(world)
        for tick in range(2):
            entity.on_tick(world, tick * _TICK_S)
        entity.on_scenario_end(world)
    finally:
        server.stop(grace=None)

    assert entity.drive_count == 2
    assert entity.turn_indicator == TurnIndicator.NO_COMMAND
    entity._vehicle.set_light_state.assert_not_called()  # noqa: SLF001


class _VehicleServiceDown(EgodriverServicer):
    """Plans, but its CarlaVehicleService fails."""

    def get_vehicle_command(self, request, context):  # type: ignore[no-untyped-def]
        context.abort(grpc.StatusCode.INTERNAL, "indicator head crashed")


def test_a_failing_vehicle_command_keeps_the_plan() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = _VehicleServiceDown(_SignalsLeft())
    add_EgodriverServiceServicer_to_server(servicer, server)
    add_CarlaVehicleServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        config = DriverClientConfig(
            address=f"127.0.0.1:{port}",
            cameras=(),
            rear_axle_offset_m=-1.4,
            send_renderer_data=False,
            policy_timestep_s=0.05,
            timeout_s=10.0,
        )
        entity = CarlaDriverEntity(config)
        entity._vehicle = _actor()  # noqa: SLF001 - stands in for spawn()
        world = _world_with_route()
        entity.on_scenario_start(world)
        for tick in range(2):
            entity.on_tick(world, tick * _TICK_S)
        entity.on_scenario_end(world)
    finally:
        server.stop(grace=None)

    assert entity.drive_count == 2
    assert entity.turn_indicator == TurnIndicator.NO_COMMAND


def test_a_plain_ego_never_requests_termination() -> None:
    from autoware_carla_scenario.entity.ego import EgoVehicle

    assert EgoVehicle().termination_requested is False


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def test_teardown_closes_the_session_and_destroys_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.sensor.carla_camera import CarlaCameraSensor

    destroyed: List[int] = []
    monkeypatch.setattr(CarlaCameraSensor, "get_image", lambda self: None)
    monkeypatch.setattr(
        CarlaCameraSensor, "destroy", lambda self: destroyed.append(id(self))
    )

    entity, client = _entity(cameras=(DriverCameraConfig(logical_id="front"),))
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_scenario_end(world)

    assert client.closed == 1
    assert len(destroyed) == 1


def test_teardown_is_safe_when_the_policy_is_gone() -> None:
    """A dead policy must not abort the rest of the scenario cleanup."""
    entity, client = _entity()
    world = _world_with_route()
    entity.on_scenario_start(world)

    def _boom() -> None:
        raise ConnectionError("policy vanished")

    client.close_session = _boom  # type: ignore[method-assign]

    entity.on_scenario_end(world)  # must not raise


def test_teardown_without_a_session_is_a_noop() -> None:
    entity, client = _entity()
    entity.on_scenario_end(_world_with_route())
    assert client.closed == 0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_debug_state_reports_loop_progress() -> None:
    entity, _ = _entity(policy_timestep_s=0.05)
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_tick(world, 0.0)

    state = entity.debug_state()
    assert state["drive_count"] == 1.0
    assert state["plan_length"] == 4.0
    assert state["sim_time_us"] == pytest.approx(50_000.0)
    assert state["rear_axle_offset_m"] == pytest.approx(-1.4)


def test_yawed_ego_reports_a_rig_frame_route() -> None:
    """The route is expressed in the rig frame, so a yawed ego still sees it ahead."""
    entity, client = _entity()
    entity._vehicle = _actor(x=5.0, y=-5.0, yaw_deg=-90.0)  # noqa: SLF001
    world = _world_with_route()

    entity.on_scenario_start(world)

    route = client.routes[0]
    # CARLA yaw -90 deg is +90 deg in the right-handed frame: the ego faces +y, while
    # the road runs along +x, so the road is off to the ego's right (negative y in rig).
    assert route[-1][1] < 0.0
    assert math.isfinite(float(route[-1][0]))


# ---------------------------------------------------------------------------
# Warm-up run onto the spawn pose
# ---------------------------------------------------------------------------


class _Lane:
    """A straight lane along +x; ``previous`` walks back down it."""

    @staticmethod
    def waypoint_at(x: float) -> MagicMock:
        waypoint = MagicMock()
        waypoint.transform = SimpleNamespace(
            location=SimpleNamespace(x=x, y=0.0, z=0.0),
            rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0),
        )
        waypoint.next.side_effect = lambda step: [_Lane.waypoint_at(x + step)]
        waypoint.previous.side_effect = lambda step: [_Lane.waypoint_at(x - step)]
        return waypoint


class _Actors(list):
    def filter(self, pattern: str) -> list:
        prefix = pattern.rstrip("*")
        return [actor for actor in self if actor.type_id.startswith(prefix)]


def _placed(actor: MagicMock) -> None:
    """Make the actor go where it is placed, and move by the velocity it is set,
    one tick's worth (the run-up reads its pose back on the next tick)."""

    def _set(transform) -> None:  # noqa: ANN001
        actor.get_transform.return_value = transform
        actor.get_location.return_value = transform.location

    def _move(velocity) -> None:  # noqa: ANN001
        at = actor.get_transform.return_value
        _set(
            SimpleNamespace(
                location=SimpleNamespace(
                    x=at.location.x + velocity.x * _TICK_S,
                    y=at.location.y + velocity.y * _TICK_S,
                    z=at.location.z,
                ),
                rotation=at.rotation,
            )
        )

    actor.set_transform.side_effect = _set
    actor.set_target_velocity.side_effect = _move
    actor.get_velocity.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    actor.get_angular_velocity.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    actor.get_physics_control.return_value = mkz_physics()
    actor.get_control.return_value = SimpleNamespace(gear=1)


def _warmup_world(others: Sequence[MagicMock] = ()) -> Tuple[MagicMock, MagicMock]:
    world = MagicMock()
    world.get_map.return_value.get_waypoint.side_effect = (
        lambda location, **_: _Lane.waypoint_at(location.x)
    )
    world.get_map.return_value.name = "Town10HD_Opt"
    light = MagicMock(type_id="traffic.traffic_light", id=99)
    light.is_frozen.return_value = False
    world.get_actors.return_value = _Actors([*others, light])
    return world, light


def _warming_entity(speed_mps: float = 5.0, warmup_s: float = 1.0):  # noqa: ANN202
    entity, client = _entity(warmup_s=warmup_s, policy_timestep_s=0.1)
    actor = entity.actor
    assert actor is not None
    actor.id, actor.type_id = 1, "vehicle.lincoln.mkz"
    _placed(actor)
    entity.set_initial_speeds({1: speed_mps})
    return entity, client, actor


def test_a_warm_up_carries_the_ego_onto_its_spawn_pose() -> None:
    entity, client, actor = _warming_entity(speed_mps=5.0, warmup_s=1.0)
    world, _ = _warmup_world()
    entity.on_scenario_start(world)

    xs = [actor.set_transform.call_args.args[0].location.x]
    ticks = 0
    while not entity.is_initialized:
        entity.on_tick(world, 0.0)
        ticks += 1

    assert ticks == 20
    # Placed once, at the start of the run-up; moved by its velocity after that,
    # aimed each tick a quarter metre on and, at the end, at the spawn pose.
    assert xs[0] == pytest.approx(-5.0)
    assert actor.get_transform().location.x == pytest.approx(
        0.25
    )  # spawn pose + a tick
    actor.set_transform.assert_called_once()
    aims = [call.args[0] for call in actor.set_target_velocity.call_args_list]
    assert [v.x for v in aims[:-1]] == pytest.approx([5.0] * 20)
    assert (aims[-1].x, aims[-1].y) == pytest.approx((5.0, 0.0))
    # The policy planned all the way, but the run-up drove.
    assert len(client.drives) == 10
    # Released onto the throttle that holds 5 m/s in gear, never the follower's;
    # physics never switched off.
    hold, _ = mkz_powertrain().pedals(0.0, 5.0, 1)
    controls = [call.args[0] for call in actor.apply_control.call_args_list]
    assert all(control.throttle == pytest.approx(hold) for control in controls)
    assert controls[-1].brake == 0.0 and not controls[-1].hand_brake
    actor.set_simulate_physics.assert_not_called()


def test_the_policy_sees_the_ego_moving_through_the_run_up() -> None:
    entity, client, _ = _warming_entity(speed_mps=5.0, warmup_s=0.5)
    world, _ = _warmup_world()
    entity.on_scenario_start(world)
    while not entity.is_initialized:
        entity.on_tick(world, 0.0)

    observations = [obs for batch in client.egomotion for obs in batch]
    assert observations
    assert all(obs.speed_mps == pytest.approx(5.0) for obs in observations)


def _other(type_id: str, actor_id: int, x: float, y: float = 0.0) -> MagicMock:
    import typesafe_carla.carla as carla  # noqa: PLC0415

    other = MagicMock(type_id=type_id, id=actor_id)
    other.get_transform.return_value = carla.Transform(
        carla.Location(x=x, y=y, z=0.0), carla.Rotation()
    )
    _placed(other)
    return other


def test_traffic_runs_up_onto_the_first_frame_with_the_ego() -> None:
    """A moving vehicle runs up along its lane, a walking pedestrian in a straight
    line, and a parked vehicle stays put; each arrives on its first-frame pose."""
    car = _other("vehicle.audi.tt", 2, x=30.0, y=3.5)
    parked = _other("vehicle.audi.tt", 3, x=50.0)
    walker = _other("walker.pedestrian.0001", 4, x=20.0, y=6.0)
    entity, _, _ = _warming_entity(speed_mps=5.0, warmup_s=1.0)
    entity.set_initial_speeds({1: 5.0, 2: 8.0, 4: 1.5})
    world, light = _warmup_world([car, parked, walker])
    entity.on_scenario_start(world)

    start = car.set_transform.call_args.args[0]
    assert (start.location.x, start.location.y) == pytest.approx((22.0, 3.5))
    first_step = walker.set_transform.call_args.args[0]
    assert (first_step.location.x, first_step.location.y) == pytest.approx((18.5, 6.0))
    parked.set_transform.assert_not_called()
    assert entity.carried_actor_ids == {1, 2}
    light.freeze.assert_called_once_with(True)

    while not entity.is_initialized:
        entity.on_tick(world, 0.0)

    arrived = walker.set_transform.call_args.args[0]
    assert (arrived.location.x, arrived.location.y) == pytest.approx((20.0, 6.0))
    assert car.get_transform().location.x == pytest.approx(30.0 + 8.0 * _TICK_S)
    final = car.set_target_velocity.call_args.args[0]
    assert (final.x, final.y) == pytest.approx((8.0, 0.0))
    light.freeze.assert_called_with(False)
    assert entity.carried_actor_ids == frozenset()


def test_without_a_warm_up_the_ego_is_ready_at_once() -> None:
    entity, client = _entity()
    entity.on_scenario_start(_world_with_route())
    assert entity.is_initialized


def test_the_run_up_keeps_the_spawn_pose_s_offset_from_its_lane() -> None:
    """Spawned 0.8 m to the side of the centre line, the ego runs up 0.8 m to the side."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    entity, _, actor = _warming_entity(speed_mps=5.0, warmup_s=1.0)
    actor.get_transform.return_value = carla.Transform(
        carla.Location(x=0.0, y=0.8, z=0.0), carla.Rotation()
    )
    world, _ = _warmup_world()
    entity.on_scenario_start(world)

    first = actor.set_transform.call_args.args[0]
    assert (first.location.x, first.location.y) == pytest.approx((-5.0, 0.8))


# ---------------------------------------------------------------------------
# A policy served by the scenario itself
# ---------------------------------------------------------------------------

_NEEDS_IN_PROCESS = pytest.mark.skipif(
    importlib.util.find_spec("carla_driver_interface.inprocess") is None,
    reason="carla-driver-interface < 2.2 has no in-process transports",
)


def _served_plans(transport: str) -> list:
    config = DriverClientConfig(
        address="nowhere.invalid:1",  # not dialled: the run serves its own policy
        policy="constant_speed",
        transport=transport,
        cameras=(),
        rear_axle_offset_m=-1.4,
        send_renderer_data=False,
        policy_timestep_s=0.05,
        timeout_s=10.0,
    )
    plans = []
    for _ in range(2):  # a second scenario reuses the policy the first started
        entity = CarlaDriverEntity(config)
        entity._vehicle = _actor()  # noqa: SLF001 - stands in for spawn()
        world = _world_with_route()
        entity.on_scenario_start(world)
        entity.on_tick(world, 0.0)
        assert len(entity._plan) > 0  # noqa: SLF001 - the served policy planned
        plans.append(entity._plan.to_proto().SerializeToString())  # noqa: SLF001
        entity.on_scenario_end(world)
    return plans


@pytest.mark.parametrize(
    "transport",
    [
        "tcp",
        pytest.param("unix", marks=_NEEDS_IN_PROCESS),
        pytest.param("direct", marks=_NEEDS_IN_PROCESS),
    ],
)
def test_a_named_policy_is_served_by_the_run(transport: str) -> None:
    """driver.policy serves the policy in this process; nothing else has to run.

    However the run reaches it (driver.transport), it plans the same.
    """
    assert _served_plans(transport) == _served_plans("tcp")


# ---------------------------------------------------------------------------
# A stack serving the policy's world, over the bridge
# ---------------------------------------------------------------------------

_GOAL = BridgePose.from_yaw(x=40.0, y=0.0, z=0.0, yaw=0.0)


@pytest.fixture
def _map_frame_without_a_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert CARLA poses to the map frame without a loaded map (see the Autoware tests)."""

    def _flip_y(pose):
        return BridgePose.from_yaw(
            x=pose.x, y=-pose.y, z=pose.z, yaw=math.radians(-pose.yaw)
        )

    monkeypatch.setattr(
        "autoware_carla_scenario.entity.bridged_ego.to_map_frame", _flip_y
    )


def _bridged_entity(
    bridge: FakeAutowareBridge, *, goal: Optional[BridgePose] = _GOAL, **bridge_config
) -> Tuple[CarlaDriverEntity, _FakeDriverClient]:
    """An entity wired to a fake client and *bridge*, its mission set."""
    config = DriverClientConfig(
        cameras=(),
        rear_axle_offset_m=-1.4,
        send_renderer_data=False,
        policy_timestep_s=0.05,
    )
    client = _FakeDriverClient(config)
    bridge_config.setdefault("base_link_offset_m", -1.4)
    entity = CarlaDriverEntity(
        config,
        client=client,
        bridge=bridge,
        bridge_config=AutowareBridgeConfig(**bridge_config),
    )
    entity._vehicle = _actor()  # noqa: SLF001 - stands in for spawn()
    if goal is not None:
        assert entity.handshake is not None
        entity.handshake.set_mission(None, goal)
    return entity, client


def test_without_a_bridge_the_policy_needs_no_goal() -> None:
    entity, _ = _entity()
    assert entity.handshake is None
    assert entity.requires_goal is False


def test_a_bridge_makes_the_goal_required() -> None:
    entity, _ = _bridged_entity(FakeAutowareBridge())
    assert entity.requires_goal is True
    # Only this instance: the class still runs a policy alone.
    assert CarlaDriverEntity.requires_goal is False


def test_a_launcher_needs_a_bridge() -> None:
    with pytest.raises(ValueError, match="needs a bridge"):
        CarlaDriverEntity(
            DriverClientConfig(cameras=()),
            client=MagicMock(),
            launcher=MagicMock(),
        )


@pytest.mark.usefixtures("_map_frame_without_a_map")
def test_a_bridged_ego_without_a_goal_is_refused() -> None:
    entity, _ = _bridged_entity(FakeAutowareBridge(), goal=None)
    with pytest.raises(ValueError, match="has no goal"):
        entity.on_scenario_start(_world_with_route())


@pytest.mark.usefixtures("_map_frame_without_a_map")
def test_the_policy_waits_for_the_bridged_stack() -> None:
    bridge = FakeAutowareBridge(ready_after=3)
    entity, client = _bridged_entity(bridge)
    world = _world_with_route()

    entity.on_scenario_start(world)
    # The mission is handed over at once; the policy has not started.
    assert bridge.configured_goal is not None
    assert bridge.configured_initial_pose is not None
    assert client.routes == []
    assert not entity.is_initialized

    for tick in range(3):
        entity.on_tick(world, tick * _TICK_S)
        assert not entity.is_initialized
        assert client.drives == []

    # The tick the stack turns ready starts the policy; it drives from the next.
    entity.on_tick(world, 3 * _TICK_S)
    assert entity.is_initialized
    assert len(client.routes) == 1
    assert client.drives == []

    entity.on_tick(world, 4 * _TICK_S)
    assert len(client.drives) == 1


@pytest.mark.usefixtures("_map_frame_without_a_map")
def test_a_stack_that_never_comes_up_ends_the_run() -> None:
    entity, client = _bridged_entity(
        FakeAutowareBridge(ready_after=100), ready_timeout_ticks=2
    )
    world = _world_with_route()
    entity.on_scenario_start(world)
    for tick in range(4):
        entity.on_tick(world, tick * _TICK_S)

    assert entity.termination_requested
    assert entity.termination_reason == (
        "The driver policy stack not ready after 2 ticks"
    )
    assert client.drives == []


@pytest.mark.usefixtures("_map_frame_without_a_map")
def test_teardown_closes_the_bridge() -> None:
    bridge = FakeAutowareBridge()
    entity, client = _bridged_entity(bridge)
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_scenario_end(world)

    assert bridge.closed
    assert client.closed == 1


@pytest.mark.usefixtures("_map_frame_without_a_map")
def test_a_retry_waits_for_the_stack_again() -> None:
    bridge = FakeAutowareBridge()
    entity, _ = _bridged_entity(bridge)
    world = _world_with_route()
    entity.on_scenario_start(world)
    entity.on_tick(world, 0.0)
    assert entity.is_initialized
    entity.on_scenario_end(world)

    entity.on_scenario_start(world)
    assert not entity.is_initialized


def _run_attempt(entity: CarlaDriverEntity, world: MagicMock, ticks: int) -> None:
    """One attempt of a scenario: start, *ticks* ticks, end."""
    entity.on_scenario_start(world)
    for tick in range(ticks):
        entity.on_tick(world, tick * _TICK_S)
    entity.on_scenario_end(world)


def test_a_retry_starts_its_session_clock_at_zero() -> None:
    # A retried scenario reuses the entity (#672): its new session must see
    # the clock and the policy steps the first attempt saw, not continue them.
    entity, client = _entity(policy_timestep_s=0.1)
    world = _world_with_route()

    _run_attempt(entity, world, 40)
    first = list(client.drives)
    first_count = entity.drive_count
    client.drives.clear()

    _run_attempt(entity, world, 40)

    assert client.drives == first
    assert entity.drive_count == first_count


def test_a_retry_shows_no_plan_before_its_first_policy_step() -> None:
    entity, _ = _entity()
    world = _world_with_route()
    _run_attempt(entity, world, 3)

    entity.on_scenario_start(world)

    assert not entity.debug_markers()


def _composed(overrides: List[str]):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    conf = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "autoware_carla_scenario"
        / "examples"
        / "conf"
    )
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_run_builds_a_policy_ego_without_a_bridge_by_default() -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    entity = build_ego_entity(_composed(["ego.entity=carla_driver"]))
    assert isinstance(entity, CarlaDriverEntity)
    assert entity.handshake is None


def test_run_gives_the_policy_ego_a_bridge_from_its_config() -> None:
    from autoware_carla_scenario.autoware_bridge import GrpcAutowareBridgeServer
    from autoware_carla_scenario.examples.run import build_ego_entity

    entity = build_ego_entity(
        _composed(
            [
                "ego.entity=carla_driver",
                "driver.bridge={address:'localhost:50099',ready_timeout_ticks:40}",
            ]
        )
    )
    assert isinstance(entity, CarlaDriverEntity)
    assert entity.handshake is not None
    assert isinstance(entity.handshake.bridge, GrpcAutowareBridgeServer)
    assert entity.handshake.config.address == "localhost:50099"
    assert entity.handshake.config.ready_timeout_ticks == 40
    assert entity.requires_goal is True


def test_run_draws_the_lidars_fog_with_the_servers_scale() -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity, server_extra_args

    lidars = "driver.lidars=[{logical_id:top},{logical_id:rear,fog_density_scale:0.05}]"
    cfg = _composed(
        ["ego.entity=carla_driver", lidars, "server.fog_density_scale=0.01"]
    )
    entity = build_ego_entity(cfg)
    assert isinstance(entity, CarlaDriverEntity)
    # A LiDAR setting its own scale keeps it.
    assert [lidar.fog_density_scale for lidar in entity.config.lidars] == [0.01, 0.05]
    assert server_extra_args(cfg) == ["-ExecCmds=carla.Weather.FogDensityScale 0.01"]

    cfg = _composed(["ego.entity=carla_driver", lidars])
    entity = build_ego_entity(cfg)
    assert isinstance(entity, CarlaDriverEntity)
    # Unset: the server's own scale, and nothing passed to the server.
    assert [lidar.fog_density_scale for lidar in entity.config.lidars] == [0.001, 0.05]
    assert server_extra_args(cfg) == []


def test_run_refuses_an_unknown_bridge_setting() -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    with pytest.raises(ValueError, match="adress"):
        build_ego_entity(
            _composed(["ego.entity=carla_driver", "driver.bridge={adress:'x'}"])
        )


def test_the_plan_is_shown_as_the_ego_s_debug_markers() -> None:
    entity, _ = _entity()
    world = _world_with_route()
    entity.on_scenario_start(world)
    assert not entity.debug_markers()

    entity.on_tick(world, _TICK_S)

    markers = entity.debug_markers()
    assert len(markers.polylines) == 1
    # The fake policy plans 0.8 m steps straight along +x.
    xs = [x for x, _, _ in markers.polylines[0].points]
    assert xs == pytest.approx([0.8, 1.6, 2.4, 3.2])
    # Until the policy plans again, the same markers: one per plan, not tick.
    assert entity.debug_markers() is markers


def test_the_markers_carry_the_plan_timed_in_the_world() -> None:
    entity, _ = _entity(turn_indicators=(2,))
    world = _world_with_route()
    entity.on_scenario_start(world)

    entity.on_tick(world, _TICK_S)

    markers = entity.debug_markers()
    assert isinstance(markers, PlanMarkers)
    plan = markers.plan
    assert plan.step == 1
    # Planned on the first tick; the fake policy times its points 0.1 s apart.
    assert plan.planned_at_us == int(round(_TICK_S * 1_000_000))
    assert [p[0] for p in plan.points] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert [p[1] for p in plan.points] == pytest.approx([0.8, 1.6, 2.4, 3.2])
    assert plan.turn_indicator == 2


# ---------------------------------------------------------------------------
# The road ahead, routed to the goal
# ---------------------------------------------------------------------------


@pytest.fixture
def _lane_routes(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Lanelets 10 -> 11 -> 12 (the turn) -> 13, and 20 nowhere near them; each
    lanelet on the OpenDRIVE road ``r<id>``."""
    import autoware_carla_scenario.coordinate as coordinate
    from autoware_carla_scenario.coordinate.map_manager import MapManager
    from autoware_carla_scenario.odd import route_plan

    class _Layer(dict):
        def exists(self, lanelet_id: int) -> bool:
            return lanelet_id in self

    layer = _Layer({i: SimpleNamespace(id=i) for i in (10, 11, 12, 13, 20)})
    chain = [10, 11, 12, 13]

    def shortest(a: SimpleNamespace, b: SimpleNamespace) -> Optional[list]:
        if a.id in chain and b.id in chain and chain.index(a.id) < chain.index(b.id):
            return [layer[i] for i in chain[chain.index(a.id) : chain.index(b.id) + 1]]
        return None

    mm = SimpleNamespace(
        lanelet_map=SimpleNamespace(laneletLayer=layer),
        routing_graph=SimpleNamespace(shortestPath=shortest),
        # No CARLA map held: the run reads the world's.
        carla_map=None,
    )
    monkeypatch.setattr(MapManager, "get_instance", classmethod(lambda cls: mm))
    monkeypatch.setattr(
        coordinate, "to_lanelet2", lambda pose: SimpleNamespace(lanelet_id=10)
    )
    monkeypatch.setattr(
        coordinate, "route_road_ids", lambda ids: [f"r{i}" for i in ids]
    )
    monkeypatch.setattr(
        route_plan, "_lengths", lambda lanelets: [20.0 for _ in lanelets]
    )
    return layer


def test_the_road_ahead_turns_where_the_route_to_the_goal_does(
    _lane_routes: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.coordinate import Lanelet2Pose
    from autoware_carla_scenario.entity import carla_driver_entity as module

    walked: list = []
    real = module.route_waypoints_in_rig

    def walk(*args, **kwargs):
        walked.append(kwargs.get("route_roads"))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "route_waypoints_in_rig", walk)
    entity, _ = _entity()
    world = _world_with_route()
    entity.route_to(world, Lanelet2Pose(lanelet_id=13, s=0.0))
    entity.on_scenario_start(world)

    assert walked == [{"r10", "r11", "r12", "r13"}]


def test_the_route_runs_through_the_waypoints(_lane_routes: dict) -> None:
    from autoware_carla_scenario.coordinate import Lanelet2Pose

    entity, _ = _entity()
    entity.route_to(
        _world_with_route(),
        Lanelet2Pose(lanelet_id=13, s=0.0),
        waypoints=[Lanelet2Pose(lanelet_id=12, s=5.0)],
    )

    assert entity._route_roads == {"r10", "r11", "r12", "r13"}  # noqa: SLF001


def test_a_goal_no_lane_reaches_leaves_the_first_way(
    _lane_routes: dict, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.coordinate import Lanelet2Pose

    entity, _ = _entity()
    with caplog.at_level("WARNING"):
        entity.route_to(_world_with_route(), Lanelet2Pose(lanelet_id=20, s=0.0))

    assert entity._route_roads is None  # noqa: SLF001
    assert "no route from lanelet 10 to lanelet 20" in caplog.text


def test_without_a_goal_the_road_ahead_takes_the_first_way() -> None:
    entity, _ = _entity()

    assert entity._route_roads is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# T4Scenes (driver.t4_scene)
# ---------------------------------------------------------------------------


class _FakeLiveScene:
    """What the entity asks of its LiveT4Scene, recorded."""

    def __init__(self) -> None:
        self.started = 0
        self.ticks: list = []
        self.asked: list = []
        self.map_resource = None

    def start(self, world) -> None:
        self.started += 1

    def record(self, world, elapsed, extras) -> None:
        self.ticks.append((elapsed, extras))

    def scene(self, stem, **kwargs):
        from autoware_carla_scenario.t4_scene._proto import t4_scene_pb2

        self.asked.append(kwargs)
        return t4_scene_pb2.T4Scene(current_index=len(self.ticks) - 1)


class _FakeRenderer:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def build(self, timestamp_us, pose, lidar=(), t4_scene=b"") -> bytes:
        data = carla_driver_pb2.CarlaRendererData(snapshot_timestamp_us=timestamp_us)
        data.t4_scene = t4_scene
        return data.SerializeToString()


def test_every_drive_carries_the_run_so_far_as_a_t4_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.entity import carla_driver_entity as module
    from autoware_carla_scenario.t4_scene._proto import t4_scene_pb2

    live = _FakeLiveScene()
    monkeypatch.setattr(module, "_live_t4_scene", lambda config, offset: live)
    monkeypatch.setattr(module, "RendererDataBuilder", _FakeRenderer)
    monkeypatch.setattr(CarlaDriverEntity, "_write_map", lambda self: "town-1")
    monkeypatch.setattr(module, "_tire_steering_angle", lambda actor: 0.02)
    sent: list = []

    def road_ahead_in_map(self: CarlaDriverEntity) -> Any:
        sent.append(self._road_ahead)  # noqa: SLF001
        return "the road ahead", "where the ego was"

    monkeypatch.setattr(CarlaDriverEntity, "_road_ahead_in_map", road_ahead_in_map)
    entity, client = _entity(
        policy_timestep_s=0.1,
        send_renderer_data=True,
        map_dir="/maps",
        t4_scene=True,
    )
    entity._goal_in_map = (1.0, 2.0, 0.5)  # noqa: SLF001
    world = _world_with_route()
    entity.on_scenario_start(world)
    for tick in range(3):
        entity.on_tick(world, (tick + 1) * _TICK_S)

    assert live.started == 1
    # Every tick, with what the policy is told of the ego.
    assert len(live.ticks) == 3
    assert live.ticks[-1][1].tire_steering_angle == 0.02
    # Every drive (ticks 1 and 3 at 0.1 s), with the route and goal.
    assert len(client.renderer_payloads) == 2
    for payload, ticks in zip(client.renderer_payloads, (1, 3)):
        data = carla_driver_pb2.CarlaRendererData.FromString(payload)
        scene = t4_scene_pb2.T4Scene.FromString(data.t4_scene)
        assert scene.current_index == ticks - 1
    # The road ahead the policy was sent this step, on the scene's lanelets.
    assert live.asked[-1]["road_ahead"] == "the road ahead"
    assert live.asked[-1]["road_ahead_origin"] == "where the ego was"
    assert all(road is not None and len(road[1]) for road in sent)
    assert live.asked[-1]["goal_pose"] == (1.0, 2.0, 0.5)
    entity.on_scenario_end(world)
    assert entity._t4 is None  # noqa: SLF001


def test_route_to_keeps_the_routes_roads_and_the_goal_for_the_scenes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.coordinate import Lanelet2Pose
    from autoware_carla_scenario.entity import carla_driver_entity as module

    monkeypatch.setattr(module, "_route_lanelets", lambda start, stops: [10, 11, 12])
    monkeypatch.setattr(module, "_roads_of", lambda ids: {f"r{i}" for i in ids})
    monkeypatch.setattr(module, "_in_map_frame", lambda pose: (5.0, 6.0, 0.25))
    entity, _ = _entity()
    entity.route_to(_world_with_route(), Lanelet2Pose(lanelet_id=12, s=0.0))

    assert entity._route_roads == {"r10", "r11", "r12"}  # noqa: SLF001
    assert entity._goal_in_map == (5.0, 6.0, 0.25)  # noqa: SLF001


def test_the_road_aheads_far_end_heads_along_its_last_half_metre() -> None:
    from autoware_carla_scenario.entity.carla_driver_entity import _far_end

    road = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [10.0, 10.1]])
    end = _far_end(road)
    assert end is not None
    x, y, yaw = end
    assert (x, y) == (10.0, 10.1) and yaw == pytest.approx(math.pi / 2)
    assert _far_end(np.array([[0.0, 0.0], [0.1, 0.0]])) is None
    assert _far_end(None) is None


def test_the_road_ahead_is_put_on_the_map_as_the_policy_was_told_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.coordinate.map_manager import MapManager

    mm = SimpleNamespace(mgrs_offset=(100.0, 200.0))
    monkeypatch.setattr(MapManager, "get_instance", classmethod(lambda cls: mm))
    entity, _ = _entity()
    pose = Pose.from_xyz_yaw(109.887769185, 0.810520982, 0.0, math.pi / 2)
    entity._road_ahead = (pose, np.array([[1.4000001, 0.0, 0.0]]))  # noqa: SLF001
    points, origin = entity._road_ahead_in_map()  # noqa: SLF001
    # Single precision, as the egodriver observations carry them.
    x32, y32 = float(np.float32(109.887769185)), float(np.float32(0.810520982))
    np.testing.assert_allclose(origin, [x32 + 100.0, y32 + 200.0], rtol=0, atol=1e-9)
    assert origin[0] != 109.887769185 + 100.0
    np.testing.assert_allclose(
        points,
        [[x32 + 100.0, y32 + float(np.float32(1.4000001)) + 200.0]],
        rtol=0,
        atol=1e-6,
    )
