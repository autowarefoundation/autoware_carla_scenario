"""The follow-trajectory action against a simulated vehicle.

The world and the vehicle are fakes with just enough physics to close the loop:
a teleport puts the vehicle where it was told, a target velocity carries it
through the tick, and in ``FOLLOW`` mode its pedals and steering drive a
kinematic bicycle on the MKZ's powertrain.  So each test is about the action's
decisions -- where it puts the vehicle, when, and when it says it is done.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    CarlaWorldPose,
    FollowTrajectoryAction,
    ReferenceContext,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryTimeCondition,
    TrajectoryTiming,
    TrajectoryVertex,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions.follow_trajectory import HIDDEN_DEPTH_M
from autoware_carla_scenario.driver.control import ControlConfig, steer_angle
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity

from ._vehicle_physics import mkz_physics, mkz_powertrain

_DT = 0.05


def _at(time: Optional[float]) -> Optional[TrajectoryTimeCondition]:
    """A vertex's time, as the condition it departs on (``None``: no time)."""
    return None if time is None else TrajectoryTimeCondition(time)


class _Actor:
    """A vehicle (or walker) with the CARLA calls the action makes."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        speed: float = 0.0,
        type_id: str = "vehicle.lincoln.mkz",
    ) -> None:
        self.type_id = type_id
        self.attributes = {"role_name": "npc1"}
        self.bounding_box = SimpleNamespace(extent=SimpleNamespace(x=2.4, y=1.0, z=0.9))
        self.x, self.y, self.z, self.yaw = x, y, 0.0, yaw
        self.speed = speed
        self.vx = speed * math.cos(math.radians(yaw))
        self.vy = speed * math.sin(math.radians(yaw))
        self.physics = True
        self.control: Any = carla.VehicleControl()
        self.walker_controls: list[Any] = []
        self.teleports = 0
        self.powertrain = mkz_powertrain()
        self.config = ControlConfig()

    # -- what the action calls ------------------------------------------
    def get_transform(self) -> carla.Transform:
        return carla.Transform(
            carla.Location(x=self.x, y=self.y, z=self.z), carla.Rotation(yaw=self.yaw)
        )

    def get_velocity(self) -> carla.Vector3D:
        return carla.Vector3D(self.vx, self.vy, 0.0)

    def get_angular_velocity(self) -> carla.Vector3D:
        return carla.Vector3D(0.0, 0.0, 0.0)

    def get_control(self) -> Any:
        return self.control

    def get_physics_control(self) -> Any:
        return mkz_physics()

    def set_transform(self, transform: carla.Transform) -> None:
        self.x, self.y, self.z = (
            transform.location.x,
            transform.location.y,
            transform.location.z,
        )
        self.yaw = transform.rotation.yaw
        self.teleports += 1

    def set_target_velocity(self, velocity: carla.Vector3D) -> None:
        self.vx, self.vy = velocity.x, velocity.y
        self.speed = math.hypot(self.vx, self.vy)

    def set_target_angular_velocity(self, velocity: carla.Vector3D) -> None:
        pass

    def set_simulate_physics(self, enabled: bool) -> None:
        self.physics = enabled

    def apply_control(self, control: Any) -> None:
        if isinstance(control, carla.WalkerControl):
            self.walker_controls.append(control)
        else:
            self.control = control

    # -- the world's tick -------------------------------------------------
    def step(self, dt: float, *, driven: bool) -> None:
        if not self.physics:
            return
        if driven:
            # A kinematic bicycle on the MKZ's pedals: CARLA steers positive
            # right, which is a growing (clockwise) yaw.
            accel = self.powertrain.acceleration(
                self.speed, self.control.throttle, self.control.brake, 1
            )
            self.speed = max(0.0, self.speed + accel * dt)
            angle = steer_angle(self.control.steer, self.config)
            self.yaw += math.degrees(
                self.speed / self.config.wheelbase_m * math.tan(angle) * dt
            )
            self.vx = self.speed * math.cos(math.radians(self.yaw))
            self.vy = self.speed * math.sin(math.radians(self.yaw))
        self.x += self.vx * dt
        self.y += self.vy * dt


class _Entity:
    def __init__(self, actor: _Actor, *, use_autopilot: bool = True) -> None:
        self.actor = actor
        self.use_autopilot = use_autopilot
        self.released = 0

    def release_from_traffic(self, world: Any) -> None:
        self.released += 1


class _World:
    def get_map(self) -> Any:
        return SimpleNamespace(get_waypoint=lambda *a, **k: None)

    def get_snapshot(self) -> Any:
        return SimpleNamespace(timestamp=SimpleNamespace(delta_seconds=_DT))


@pytest.fixture
def entity() -> Iterator[_Entity]:
    registered = _Entity(_Actor())
    register_entity("npc1", registered)
    yield registered
    unregister_entity("npc1")


def _line(
    length: float = 20.0, duration: Optional[float] = 2.0, count: int = 5
) -> Trajectory:
    """A straight line along CARLA +x, timed at constant speed or untimed."""
    return Trajectory(
        "line",
        [
            TrajectoryVertex(
                CarlaWorldPose(length * i / (count - 1), 0.0, 0.0, yaw=0.0),
                _at(None if duration is None else duration * i / (count - 1)),
            )
            for i in range(count)
        ],
    )


def _run(
    action: FollowTrajectoryAction,
    actor: _Actor,
    seconds: float,
    *,
    start: float = 0.0,
    driven: bool = False,
) -> float:
    """Tick the action and the world for *seconds*; return the clock."""
    world = _World()
    elapsed = start
    for _ in range(int(round(seconds / _DT))):
        action.tick(world, elapsed)
        actor.step(_DT, driven=driven)
        elapsed += _DT
    return elapsed


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_a_timing_needs_vertex_times(self) -> None:
        with pytest.raises(ValueError, match="no vertex times"):
            FollowTrajectoryAction("npc1", _line(duration=None), TrajectoryTiming())

    def test_hiding_needs_a_timed_position_replay(self) -> None:
        with pytest.raises(ValueError, match="hidden_outside_trajectory"):
            FollowTrajectoryAction(
                "npc1", _line(), None, hidden_outside_trajectory=True
            )
        with pytest.raises(ValueError, match="hidden_outside_trajectory"):
            FollowTrajectoryAction(
                "npc1",
                _line(),
                TrajectoryTiming(),
                TrajectoryFollowingMode.FOLLOW,
                hidden_outside_trajectory=True,
            )

    def test_a_negative_offset_is_refused(self) -> None:
        with pytest.raises(ValueError, match="initial_distance_offset"):
            FollowTrajectoryAction("npc1", _line(), initial_distance_offset=-1.0)

    def test_it_reissues_every_tick(self) -> None:
        assert FollowTrajectoryAction("npc1", _line()).reissues_while_running


# ---------------------------------------------------------------------------
# POSITION mode
# ---------------------------------------------------------------------------


class TestPositionWithTiming:
    def test_the_vehicle_is_where_the_trajectory_says(self, entity: _Entity) -> None:
        action = FollowTrajectoryAction("npc1", _line(), TrajectoryTiming())
        _run(action, entity.actor, 1.0)
        # The last pose was set at t = 0.95 and carried one tick on at 10 m/s.
        assert entity.actor.x == pytest.approx(10.0)
        assert entity.actor.speed == pytest.approx(10.0)
        assert action.state is ActionState.RUNNING

    def test_it_completes_at_the_last_vertex_and_stops_there(
        self, entity: _Entity
    ) -> None:
        action = FollowTrajectoryAction("npc1", _line(), TrajectoryTiming())
        _run(action, entity.actor, 2.5)
        assert action.finished
        assert action.state is ActionState.COMPLETE
        assert entity.actor.x == pytest.approx(20.0)
        assert entity.actor.speed == 0.0
        assert entity.actor.control.brake == 1.0

    def test_relative_timing_starts_with_the_action(self, entity: _Entity) -> None:
        action = FollowTrajectoryAction(
            "npc1", _line(), TrajectoryTiming(ReferenceContext.RELATIVE)
        )
        _run(action, entity.actor, 0.5, start=30.0)
        assert entity.actor.x == pytest.approx(5.0)

    def test_absolute_timing_follows_the_scenario_clock(self, entity: _Entity) -> None:
        """Started late, an absolute replay is already partway along."""
        action = FollowTrajectoryAction(
            "npc1", _line(), TrajectoryTiming(ReferenceContext.ABSOLUTE)
        )
        _run(action, entity.actor, 0.05, start=1.0)
        assert entity.actor.x == pytest.approx(10.5)

    def test_scale_slows_the_replay(self, entity: _Entity) -> None:
        action = FollowTrajectoryAction("npc1", _line(), TrajectoryTiming(scale=2.0))
        _run(action, entity.actor, 1.0)
        assert entity.actor.x == pytest.approx(5.0)
        assert entity.actor.speed == pytest.approx(5.0)

    def test_an_initial_offset_starts_further_along(self, entity: _Entity) -> None:
        action = FollowTrajectoryAction(
            "npc1", _line(), TrajectoryTiming(), initial_distance_offset=10.0
        )
        _run(action, entity.actor, 0.05)
        assert entity.actor.x == pytest.approx(10.5)

    def test_the_vehicle_is_taken_from_the_traffic_once(self, entity: _Entity) -> None:
        action = FollowTrajectoryAction("npc1", _line(), TrajectoryTiming())
        _run(action, entity.actor, 1.0)
        assert entity.released == 1


class TestPositionWithoutTiming:
    def test_it_goes_at_the_speed_it_had(self, entity: _Entity) -> None:
        entity.actor.vx = entity.actor.speed = 4.0
        action = FollowTrajectoryAction("npc1", _line(duration=None))
        _run(action, entity.actor, 1.0)
        assert entity.actor.x == pytest.approx(4.0)
        assert not action.finished

    def test_it_completes_at_the_end_of_the_path(self, entity: _Entity) -> None:
        entity.actor.vx = entity.actor.speed = 20.0
        action = FollowTrajectoryAction("npc1", _line(duration=None))
        _run(action, entity.actor, 2.0)
        assert action.state is ActionState.COMPLETE
        assert entity.actor.x == pytest.approx(20.0)

    def test_a_closed_path_goes_round_and_round(self, entity: _Entity) -> None:
        entity.actor.vx = entity.actor.speed = 10.0
        square = Trajectory(
            "square",
            [
                TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0)),
                TrajectoryVertex(CarlaWorldPose(10.0, 0.0, 0.0)),
                TrajectoryVertex(CarlaWorldPose(10.0, 10.0, 0.0)),
                TrajectoryVertex(CarlaWorldPose(0.0, 10.0, 0.0)),
            ],
            closed=True,
        )
        action = FollowTrajectoryAction("npc1", square)
        _run(action, entity.actor, 6.0)
        assert not action.finished


class TestHidden:
    def test_it_is_out_of_the_world_until_its_first_vertex(
        self, entity: _Entity
    ) -> None:
        late = Trajectory(
            "late",
            [
                TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0), _at(1.0)),
                TrajectoryVertex(CarlaWorldPose(10.0, 0.0, 0.0), _at(2.0)),
            ],
        )
        action = FollowTrajectoryAction(
            "npc1",
            late,
            TrajectoryTiming(ReferenceContext.ABSOLUTE),
            hidden_outside_trajectory=True,
        )
        _run(action, entity.actor, 0.5)
        assert entity.actor.z == pytest.approx(-HIDDEN_DEPTH_M)
        assert entity.actor.physics is False

        _run(action, entity.actor, 0.6, start=0.5)
        assert entity.actor.physics is True
        assert entity.actor.z == pytest.approx(0.0)

        _run(action, entity.actor, 1.0, start=1.1)
        assert action.finished
        assert entity.actor.z == pytest.approx(-HIDDEN_DEPTH_M)
        assert entity.actor.physics is False


class TestLateEntity:
    def test_a_run_begins_on_the_first_tick_that_finds_the_entity(self) -> None:
        """Triggered before the entity exists, it still takes it over and starts."""
        action = FollowTrajectoryAction(
            "late1", _line(), TrajectoryTiming(ReferenceContext.RELATIVE)
        )
        world = _World()
        action.tick(world, 0.0)  # triggered: nobody answers to "late1" yet
        late = _Entity(_Actor())
        register_entity("late1", late)
        try:
            _run(action, late.actor, 0.5, start=_DT)
        finally:
            unregister_entity("late1")
        assert late.released == 1
        # The clock started when the entity was found, not at the trigger.
        assert late.actor.x == pytest.approx(5.0)


class TestWho:
    def test_an_externally_driven_ego_is_refused(self) -> None:
        driven = _Entity(_Actor(), use_autopilot=False)
        register_entity("ego", driven)
        try:
            action = FollowTrajectoryAction("ego", _line(), TrajectoryTiming())
            _run(action, driven.actor, 0.5)
        finally:
            unregister_entity("ego")
        assert driven.actor.teleports == 0
        assert driven.released == 0

    def test_a_walker_is_placed_by_its_middle_and_animated(self) -> None:
        walker = _Entity(_Actor(type_id="walker.pedestrian.0001"))
        register_entity("walker1", walker)
        try:
            action = FollowTrajectoryAction("walker1", _line(), TrajectoryTiming())
            _run(action, walker.actor, 0.5)
        finally:
            unregister_entity("walker1")
        assert walker.actor.z == pytest.approx(walker.actor.bounding_box.extent.z)
        assert walker.actor.walker_controls[-1].speed > 0.0


# ---------------------------------------------------------------------------
# FOLLOW mode
# ---------------------------------------------------------------------------


def _bend() -> Trajectory:
    """20 m straight, then a quarter circle of 20 m radius to the right, at 6 m/s."""
    points = [(float(x), 0.0) for x in range(0, 20, 2)]
    for step in range(1, 16):
        angle = math.pi / 2 * step / 15
        points.append((20.0 + 20.0 * math.sin(angle), 20.0 - 20.0 * math.cos(angle)))
    vertices = []
    elapsed = 0.0
    for index, (x, y) in enumerate(points):
        if index:
            px, py = points[index - 1]
            elapsed += math.hypot(x - px, y - py) / 6.0
        vertices.append(TrajectoryVertex(CarlaWorldPose(x, y, 0.0), _at(elapsed)))
    return Trajectory("bend", vertices)


class TestFollow:
    def test_a_controller_drives_it_round_the_bend(self, entity: _Entity) -> None:
        trajectory = _bend()
        action = FollowTrajectoryAction(
            "npc1", trajectory, TrajectoryTiming(), TrajectoryFollowingMode.FOLLOW
        )
        worst = 0.0
        world = _World()
        elapsed = 0.0
        while elapsed < 20.0 and not action.finished:
            action.tick(world, elapsed)
            entity.actor.step(_DT, driven=True)
            elapsed += _DT
            assert action._resolved is not None
            near = action._resolved.project(entity.actor.x, entity.actor.y)
            sample = action._resolved.at_distance(near)
            worst = max(
                worst, math.hypot(entity.actor.x - sample.x, entity.actor.y - sample.y)
            )
        assert action.finished
        last = trajectory.vertices[-1].time
        assert last is not None and elapsed < last + 3.0
        # Round the bend to the right (CARLA's yaw grows clockwise), and within
        # the arrival band of where it ends.
        assert entity.actor.yaw > 60.0
        assert math.hypot(entity.actor.x - 40.0, entity.actor.y - 20.0) < 1.5
        assert worst < 1.0
        assert entity.actor.control.brake == 1.0

    def test_untimed_it_holds_the_speed_it_had(self, entity: _Entity) -> None:
        entity.actor.vx = entity.actor.speed = 5.0
        action = FollowTrajectoryAction(
            "npc1",
            _line(length=200.0, duration=None),
            following_mode=TrajectoryFollowingMode.FOLLOW,
        )
        _run(action, entity.actor, 5.0, driven=True)
        assert entity.actor.speed == pytest.approx(5.0, abs=0.5)
        assert entity.actor.teleports == 0
