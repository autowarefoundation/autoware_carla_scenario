"""Unit tests for the ego entity driven by ROS 2 nodes over DDS.

The CARLA world and the ego actor are mocked and the DDS transport is the
in-memory fake, so the loop -- publish, take commands, arbitrate, apply -- runs
without a simulator or a DDS domain.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest

from autoware_carla_scenario.driver.geometry import Pose
from autoware_carla_scenario.entity.ros2_entity import Ros2EgoEntity
from autoware_carla_scenario.ros2 import (
    CommandSourceConfig,
    Ros2CameraConfig,
    ControlCommand,
    FakeRos2Transport,
    Ros2EgoConfig,
    SteerAccelCommand,
    TrajectoryCommand,
    TrajectoryCommandPoint,
)

from ._vehicle_physics import mkz_physics

_TICK_S = 0.05

_SOURCES = (
    CommandSourceConfig(
        name="vp",
        kind="steer_accel",
        steering_topic="/vehicle/steering_cmd",
        acceleration_topic="/vehicle/throttle_cmd",
        priority=0,
        stale_s=0.3,
    ),
    CommandSourceConfig(
        name="si",
        kind="control",
        topic="/control/trajectory_follower/control_cmd",
        priority=10,
        stale_s=0.3,
    ),
    CommandSourceConfig(
        name="planner",
        kind="trajectory",
        topic="/planning/scenario_planning/trajectory",
        priority=-1,
        stale_s=0.3,
    ),
)


class _Clock:
    """A world whose snapshot reports a settable simulation time."""

    def __init__(self) -> None:
        self.now_s = 0.0
        self.world = MagicMock()
        self.world.get_snapshot.side_effect = lambda: SimpleNamespace(
            timestamp=SimpleNamespace(elapsed_seconds=self.now_s)
        )

    def advance(self) -> None:
        self.now_s += _TICK_S


def _actor(speed_mps: float = 5.0, wheel_deg: float = 0.0) -> MagicMock:
    """A mock ego vehicle heading +x in CARLA's world at *speed_mps*."""
    actor = MagicMock()
    actor.get_transform.return_value = SimpleNamespace(
        location=SimpleNamespace(x=10.0, y=-2.0, z=0.0),
        rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0),
    )
    actor.get_velocity.return_value = SimpleNamespace(x=speed_mps, y=0.0, z=0.0)
    actor.get_acceleration.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    actor.get_angular_velocity.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    actor.get_physics_control.return_value = mkz_physics()
    actor.get_control.return_value = SimpleNamespace(gear=1)
    actor.get_wheel_steer_angle.return_value = wheel_deg
    return actor


def _started(
    *,
    on_stale: str = "hold",
    actor: MagicMock | None = None,
    cameras: tuple[Ros2CameraConfig, ...] = (),
) -> tuple[Ros2EgoEntity, FakeRos2Transport, MagicMock, _Clock]:
    config = Ros2EgoConfig(
        sources=_SOURCES, on_stale=on_stale, rear_axle_offset_m=-1.4, cameras=cameras
    )
    transport = FakeRos2Transport(config)
    entity = Ros2EgoEntity(config, transport=transport)
    actor = actor if actor is not None else _actor()
    entity._vehicle = actor
    clock = _Clock()
    entity.on_scenario_start(clock.world)
    return entity, transport, actor, clock


def _tick(entity: Ros2EgoEntity, clock: _Clock) -> None:
    clock.advance()
    entity.on_tick(clock.world, clock.now_s)


def _last_ackermann(actor: MagicMock):
    return actor.apply_ackermann_control.call_args.args[0]


def _last_control(actor: MagicMock):
    return actor.apply_control.call_args.args[0]


# ---------------------------------------------------------------------------
# Lifecycle and state
# ---------------------------------------------------------------------------


def test_start_opens_the_transport_and_publishes_the_first_state() -> None:
    entity, transport, _, _ = _started()
    assert transport.opened == 1
    assert len(transport.published) == 1
    assert entity.use_autopilot is False


def test_start_before_spawn_is_an_error() -> None:
    entity = Ros2EgoEntity(transport=FakeRos2Transport(Ros2EgoConfig()))
    with pytest.raises(RuntimeError, match="before spawn"):
        entity.on_scenario_start(_Clock().world)


def test_every_tick_publishes_the_state_in_the_local_frame() -> None:
    entity, transport, _, clock = _started(actor=_actor(speed_mps=5.0, wheel_deg=10.0))
    _tick(entity, clock)
    state = transport.published[-1]
    assert state.stamp_us == 50_000
    # CARLA's y=-2 is y=+2 right-handed, and base_link sits 1.4 m behind the
    # actor origin.
    assert state.pose.position[0] == pytest.approx(8.6)
    assert state.pose.position[1] == pytest.approx(2.0)
    assert state.speed_mps == pytest.approx(5.0)
    assert state.linear_velocity[0] == pytest.approx(5.0)
    # CARLA's wheels at +10 deg (right) are Autoware's -10 deg.
    assert state.steering_tire_angle_rad == pytest.approx(-math.radians(10.0))


def test_end_closes_the_transport_once() -> None:
    entity, transport, _, clock = _started()
    entity.on_scenario_end(clock.world)
    entity.on_scenario_end(clock.world)
    assert transport.closed == 1
    # A tick after the end is ignored.
    published = len(transport.published)
    _tick(entity, clock)
    assert len(transport.published) == published


def test_a_failing_close_does_not_raise() -> None:
    entity, transport, _, clock = _started()
    transport.close = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    entity.on_scenario_end(clock.world)


# ---------------------------------------------------------------------------
# Applying commands
# ---------------------------------------------------------------------------


def test_the_ego_brakes_before_any_command() -> None:
    entity, _, actor, clock = _started()
    _tick(entity, clock)
    assert entity.active_source is None
    assert _last_control(actor).brake == pytest.approx(1.0)
    actor.apply_ackermann_control.assert_not_called()


def test_steer_accel_goes_to_the_ackermann_controller_as_visionpilot_hands_it() -> None:
    entity, transport, actor, clock = _started(actor=_actor(speed_mps=5.0))
    transport.deliver("vp", SteerAccelCommand(steering_rad=0.1, acceleration_mps2=0.5))
    _tick(entity, clock)
    assert entity.active_source == "vp"
    control = _last_ackermann(actor)
    # carla_control_publisher: steer as is, speed = v + a * 2.0.
    assert control.steer == pytest.approx(0.1)
    assert control.speed == pytest.approx(6.0)
    assert control.acceleration == pytest.approx(0.5)
    assert control.jerk == pytest.approx(10.0)


def test_a_negative_target_speed_is_clamped_to_zero() -> None:
    entity, transport, actor, clock = _started(actor=_actor(speed_mps=1.0))
    transport.deliver("vp", SteerAccelCommand(0.0, -3.0))
    _tick(entity, clock)
    assert _last_ackermann(actor).speed == pytest.approx(0.0)


def test_a_control_command_is_steered_in_carlas_sense() -> None:
    entity, transport, actor, clock = _started()
    transport.deliver("si", ControlCommand(0.2, 4.0, -3.0, jerk_mps3=None))
    _tick(entity, clock)
    control = _last_ackermann(actor)
    assert control.steer == pytest.approx(-0.2)
    assert control.speed == pytest.approx(4.0)
    assert control.acceleration == pytest.approx(-3.0)
    assert control.jerk == pytest.approx(10.0)


def test_a_control_command_keeps_its_own_jerk_when_it_defines_one() -> None:
    entity, transport, actor, clock = _started()
    transport.deliver("si", ControlCommand(0.2, 4.0, -3.0, jerk_mps3=2.5))
    _tick(entity, clock)
    assert _last_ackermann(actor).jerk == pytest.approx(2.5)


def test_the_safety_island_overrides_visionpilot_while_fresh() -> None:
    entity, transport, actor, clock = _started()
    transport.deliver("vp", SteerAccelCommand(0.1, 0.5))
    _tick(entity, clock)
    assert entity.active_source == "vp"

    transport.deliver("vp", SteerAccelCommand(0.1, 0.5))
    transport.deliver("si", ControlCommand(0.0, 3.0, -3.0))
    _tick(entity, clock)
    assert entity.active_source == "si"

    # VisionPilot keeps talking, the CR52 falls silent: after its stale_s
    # VisionPilot drives again.
    for _ in range(7):
        transport.deliver("vp", SteerAccelCommand(0.1, 0.5))
        _tick(entity, clock)
    assert entity.active_source == "vp"
    assert _last_ackermann(actor).steer == pytest.approx(0.1)


def test_a_stale_source_is_held_by_default() -> None:
    entity, transport, actor, clock = _started()
    transport.deliver("si", ControlCommand(0.0, 3.0, -3.0))
    _tick(entity, clock)
    for _ in range(10):
        _tick(entity, clock)
    assert entity.active_source == "si"
    assert _last_ackermann(actor).speed == pytest.approx(3.0)
    actor.apply_control.assert_not_called()


def test_a_stale_source_brakes_when_asked_to() -> None:
    entity, transport, actor, clock = _started(on_stale="brake")
    transport.deliver("si", ControlCommand(0.0, 3.0, -3.0))
    _tick(entity, clock)
    for _ in range(10):
        _tick(entity, clock)
    assert _last_control(actor).brake == pytest.approx(1.0)


def test_a_trajectory_is_tracked_by_the_framework_follower() -> None:
    entity, transport, actor, clock = _started(actor=_actor(speed_mps=0.0))
    # Straight ahead along the ego's heading, in the local frame, at 5 m/s.
    points: List[TrajectoryCommandPoint] = [
        TrajectoryCommandPoint(
            200_000 * i, Pose.from_xyz_yaw(8.6 + 1.0 * i, 2.0, 0.0, 0.0), 5.0
        )
        for i in range(30)
    ]
    transport.deliver("planner", TrajectoryCommand(stamp_us=0, points=tuple(points)))
    _tick(entity, clock)
    assert entity.active_source == "planner"
    control = _last_control(actor)
    assert control.throttle > 0.0
    assert control.brake == pytest.approx(0.0)
    assert abs(control.steer) < 0.05
    actor.apply_ackermann_control.assert_not_called()


def test_the_plan_of_a_trajectory_is_built_once() -> None:
    entity, transport, _, clock = _started()
    command = TrajectoryCommand(
        stamp_us=0,
        points=tuple(
            TrajectoryCommandPoint(
                100_000 * i, Pose.from_xyz_yaw(8.6 + 0.5 * i, 2.0, 0.0, 0.0), 5.0
            )
            for i in range(10)
        ),
    )
    transport.deliver("planner", command)
    _tick(entity, clock)
    planned = entity._planned
    assert planned is not None and planned[0] is command
    _tick(entity, clock)
    assert entity._planned is planned


# ---------------------------------------------------------------------------
# Published cameras
# ---------------------------------------------------------------------------

_CAMERA = Ros2CameraConfig(
    logical_id="main_cam",
    topic="/carla/hero/main_cam/image",
    image_width=4,
    image_height=2,
    fov=60.0,
    fps=10.0,
    position_x=1.25,
    position_z=1.58,
)


def _image(timestamp_s: float) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=timestamp_s, width=4, height=2, raw_data=bytes(range(32))
    )


def test_no_camera_is_spawned_unless_one_is_configured() -> None:
    _, _, _, clock = _started()
    clock.world.spawn_actor.assert_not_called()


def test_a_configured_camera_is_mounted_on_the_ego() -> None:
    _, _, actor, clock = _started(cameras=(_CAMERA,))
    blueprint = clock.world.get_blueprint_library.return_value.find.return_value
    blueprint.set_attribute.assert_any_call("image_size_x", "4")
    blueprint.set_attribute.assert_any_call("fov", "60.0")
    blueprint.set_attribute.assert_any_call("sensor_tick", "0.1")
    (_, transform), kwargs = clock.world.spawn_actor.call_args
    assert kwargs["attach_to"] is actor
    assert transform.location.x == pytest.approx(1.25)
    assert transform.location.z == pytest.approx(1.58)


def test_each_frame_is_published_as_it_arrives() -> None:
    _, transport, _, clock = _started(cameras=(_CAMERA,))
    sensor = clock.world.spawn_actor.return_value
    (listener,) = sensor.listen.call_args.args
    listener(_image(1.25))
    listener(_image(1.35))
    assert [frame.stamp_us for frame in transport.images] == [1_250_000, 1_350_000]
    frame = transport.images[0]
    assert (frame.camera, frame.width, frame.height) == (0, 4, 2)
    assert frame.bgra == bytes(range(32))


def test_the_cameras_go_before_the_transport_closes() -> None:
    entity, transport, _, clock = _started(cameras=(_CAMERA,))
    sensor = clock.world.spawn_actor.return_value
    (listener,) = sensor.listen.call_args.args
    entity.on_scenario_end(clock.world)
    sensor.stop.assert_called_once()
    sensor.destroy.assert_called_once()
    assert transport.closed == 1
    # A frame still in flight after the end is dropped.
    listener(_image(2.0))
    assert transport.images == []


# ---------------------------------------------------------------------------
# Turn indicators
# ---------------------------------------------------------------------------

# CARLA's VehicleLightState bits.
_LOW_BEAM, _RIGHT_BLINKER, _LEFT_BLINKER = 0x2, 0x10, 0x20


def _lights(actor: MagicMock, initial: int) -> List[int]:
    """Have *actor* keep its light state as CARLA does; the states it was set to."""
    state = [initial]
    set_to: List[int] = []

    def _set(value: int) -> None:
        state[0] = int(value)
        set_to.append(int(value))

    actor.get_light_state.side_effect = lambda: state[0]
    actor.set_light_state.side_effect = _set
    return set_to


def test_autoware_s_turn_indicators_command_lights_the_ego() -> None:
    actor = _actor()
    set_to = _lights(actor, _LOW_BEAM)
    entity, transport, _, clock = _started(actor=actor)

    transport.deliver_turn_indicators(2)  # ENABLE_LEFT
    _tick(entity, clock)
    transport.deliver_turn_indicators(2)  # the same, again: nothing to set
    _tick(entity, clock)
    transport.deliver_turn_indicators(0)  # NO_COMMAND changes nothing
    _tick(entity, clock)
    transport.deliver_turn_indicators(1)  # DISABLE
    _tick(entity, clock)

    assert set_to == [_LOW_BEAM | _LEFT_BLINKER, _LOW_BEAM]


def test_the_state_reports_the_indicators_the_ego_shows() -> None:
    actor = _actor()
    _lights(actor, _RIGHT_BLINKER)
    entity, transport, _, clock = _started(actor=actor)
    _tick(entity, clock)
    assert transport.published[-1].turn_indicators == 3  # ENABLE_RIGHT


def test_an_unknown_turn_indicators_command_is_ignored() -> None:
    actor = _actor()
    set_to = _lights(actor, 0)
    entity, transport, _, clock = _started(actor=actor)
    transport.deliver_turn_indicators(9)
    _tick(entity, clock)
    assert set_to == []
