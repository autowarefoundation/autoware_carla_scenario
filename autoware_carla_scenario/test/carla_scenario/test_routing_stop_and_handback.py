"""A routing run that ends on its own terms or another action's.

- A ``driver`` route to a goal with ``stop_at_goal`` and no times is paced to
  its goal and stopped there (#1016): the action plans the path itself, on the
  fixture map, rather than end on its first tick.
- A ``follow`` or ``position`` run overridden by another action, or ended
  early by its ``until``, stops the entity it took from its driver and hands
  it back (#1017).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario import (
    Lanelet2Pose,
    ReferenceContext,
    RoutingAction,
    RoutingMode,
    SetSpeedAction,
    TrajectoryTiming,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions._trajectory_run import TrajectoryRun
from autoware_carla_scenario.conditions import BaseCondition, ScenarioResult
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.entity import registry
from autoware_carla_scenario.entity.registry import (
    register_entity,
    register_pedestrian_entity,
    unregister_entity,
)

from .test_routing_trajectory_action import _DT, _Actor, _Entity, _line, _route, _World

_DATA = Path(__file__).resolve().parents[3] / "data"

#: A straight road lanelet of the fixture map, over 250 m long.
_LANELET = 104


class _At(BaseCondition):
    """Holds from scenario time *t* on."""

    def __init__(self, t: float) -> None:
        super().__init__(label="at")
        self._t = t

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        if elapsed >= self._t - 1e-9:
            return ScenarioResult(True, "", elapsed)
        return None


# ---------------------------------------------------------------------------
# #1016: stop_at_goal on a goal route
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_map() -> Iterator[MapManager]:
    """The fixture map, loaded once for this module."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    yield manager
    MapManager._instance = saved


class _Car:
    """An autopilot vehicle on the map, which the test moves along its lane."""

    use_autopilot = True

    def __init__(self, x: float, y: float, z: float, yaw: float) -> None:
        self.location = SimpleNamespace(x=x, y=y, z=z)
        self.rotation = SimpleNamespace(yaw=yaw, pitch=0.0, roll=0.0)
        self.routes: list[Lanelet2Pose] = []
        self.speeds: list[float] = []
        self.finishes: list[list[Any]] = []
        self.actor = SimpleNamespace(
            id=7,
            get_transform=lambda: SimpleNamespace(
                location=self.location, rotation=self.rotation
            ),
            get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
            get_speed_limit=lambda: 50.0,
        )

    def route_to(
        self, world, goal, *, initial_pose=None, ground_projection=None, waypoints=()
    ):  # noqa: ANN001, ANN201
        self.routes.append(goal)

    def resume_traffic(self, world: Any) -> None:
        pass

    def set_desired_speed(self, world: Any, speed_kmh: float) -> None:
        self.speeds.append(speed_kmh)

    def finish_at(self, world: Any, points: list) -> bool:
        self.finishes.append(points)
        return True


@pytest.fixture
def car(
    loaded_map: MapManager, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[_Car, Any]]:
    """A car standing 10 m into the lanelet, and the lane's centreline."""
    from autoware_carla_scenario.coordinate.transform import to_carla_world

    MapManager._instance = loaded_map
    monkeypatch.setattr(registry, "_VEHICLES", {})
    monkeypatch.setattr(registry, "_PEDESTRIANS", {})
    start = to_carla_world(Lanelet2Pose(_LANELET, 10.0))
    vehicle = _Car(start.x, start.y, start.z, start.yaw)
    register_entity("npc1", vehicle)
    yield vehicle, to_carla_world


def test_a_goal_route_to_stop_at_needs_the_tick_loop(
    car: tuple[_Car, Any],
) -> None:
    goal = Lanelet2Pose(_LANELET, 60.0)
    assert RoutingAction("npc1", goal, stop_at_goal=True).needs_tick_loop
    # Without it, the route is handed to the driver and the run ends.
    assert not RoutingAction("npc1", goal).needs_tick_loop


@pytest.mark.parametrize("speed", [None, 10.0])
def test_a_goal_route_stops_its_vehicle_at_the_goal(
    car: tuple[_Car, Any], speed: Optional[float]
) -> None:
    """Planned on the map, with no pacer handed in: the run goes on until the
    vehicle arrives, which is told no speed there, and only then completes."""
    vehicle, to_carla_world = car
    action = RoutingAction(
        "npc1", Lanelet2Pose(_LANELET, 60.0), speed=speed, stop_at_goal=True
    )
    world = object()
    elapsed = 0.0
    for s in range(10, 46, 5):
        point = to_carla_world(Lanelet2Pose(_LANELET, float(s)))
        vehicle.location.x, vehicle.location.y = point.x, point.y
        action.tick(world, elapsed)
        elapsed += _DT
        assert action.state is not ActionState.COMPLETE, s
    assert vehicle.routes == [Lanelet2Pose(_LANELET, 60.0)]
    assert 0.0 not in vehicle.speeds  # still its driver's (or its given) speed
    at_goal = to_carla_world(Lanelet2Pose(_LANELET, 60.0))
    # The last metres driven by the backend itself, along the lanes to the
    # goal, as a written trajectory's are: stopped there, not braked on from
    # wherever arrival is noticed.
    assert len(vehicle.finishes) == 1
    end = vehicle.finishes[0][-1]
    assert abs(end.x - at_goal.x) < 0.5 and abs(end.y - at_goal.y) < 0.5
    vehicle.location.x, vehicle.location.y = at_goal.x, at_goal.y
    for _ in range(3):
        action.tick(world, elapsed)
        elapsed += _DT
    assert vehicle.speeds[-1] == 0.0
    assert action.state is ActionState.COMPLETE


def test_a_goal_route_not_to_stop_at_ends_at_once(car: tuple[_Car, Any]) -> None:
    """Without stop_at_goal, as before: the route is handed over and the run
    ends, with no path planned."""
    vehicle, _ = car
    action = RoutingAction("npc1", Lanelet2Pose(_LANELET, 60.0))
    for i in range(3):
        action.tick(object(), i * _DT)
    assert action._pacer is None
    assert action.state is ActionState.COMPLETE
    assert vehicle.speeds == []


# ---------------------------------------------------------------------------
# #1017: an overridden follow/position run hands its entity back
# ---------------------------------------------------------------------------


class _Handed(_Entity):
    """A vehicle that counts the hand-backs and the speeds it is told."""

    def __init__(self, actor: _Actor) -> None:
        super().__init__(actor)
        self.resumed = 0
        self.speeds: list[float] = []

    def resume_traffic(self, world: Any) -> None:
        self.resumed += 1

    def set_desired_speed(self, world: Any, kmh: float) -> None:
        self.speeds.append(kmh)


@pytest.fixture
def handed() -> Iterator[_Handed]:
    from autoware_carla_scenario.actions.base import reset_control_slots

    reset_control_slots()
    vehicle = _Handed(_Actor(speed=8.0))
    register_entity("npc1", vehicle)
    yield vehicle
    unregister_entity("npc1")
    reset_control_slots()


def _override(action: TrajectoryRun, actor: _Actor, *, driven: bool) -> SetSpeedAction:
    """Run *action* and, from 2 s on, a SetSpeedAction that overrides it."""
    brake = SetSpeedAction("npc1", 0.0, condition=_At(2.0), overrides=True)
    world, t = _World(), 0.0
    while t < 4.0 - 1e-9:
        action.tick(world, t)
        brake.tick(world, t)
        actor.step(_DT, driven=driven)
        t += _DT
    return brake


@pytest.mark.parametrize(
    ("mode", "timing"),
    [(RoutingMode.FOLLOW, None), (RoutingMode.POSITION, TrajectoryTiming())],
)
def test_an_overridden_run_hands_its_vehicle_back_stopped(
    handed: _Handed, mode: RoutingMode, timing: Optional[TrajectoryTiming]
) -> None:
    trajectory = _line(length=400.0, duration=None if timing is None else 50.0)
    action = _route("npc1", trajectory, timing, mode=mode)
    _override(action, handed.actor, driven=mode is RoutingMode.FOLLOW)
    assert handed.released == 1
    assert handed.resumed == 1
    assert action.state is ActionState.COMPLETE
    assert handed.actor.control.throttle == 0.0
    assert handed.actor.control.brake == 1.0
    assert handed.speeds == [0.0]  # the overriding action's, to the backend


def test_a_run_ending_on_its_own_keeps_its_vehicle(handed: _Handed) -> None:
    """Arrival is no override: the vehicle stays stopped where it ends."""
    action = _route("npc1", _line(length=10.0, duration=1.0), TrajectoryTiming())
    world, t = _World(), 0.0
    while t < 2.0:
        action.tick(world, t)
        t += _DT
    assert action.state is ActionState.COMPLETE
    assert handed.resumed == 0


@pytest.mark.parametrize(
    ("mode", "timing"),
    [(RoutingMode.FOLLOW, None), (RoutingMode.POSITION, TrajectoryTiming())],
)
def test_a_run_its_until_ends_early_hands_its_vehicle_back_stopped(
    handed: _Handed, mode: RoutingMode, timing: Optional[TrajectoryTiming]
) -> None:
    """OpenSCENARIO's stop transition, as an override is: not left released
    on the follower's last pedals."""
    trajectory = _line(length=400.0, duration=None if timing is None else 50.0)
    action = _route("npc1", trajectory, timing, mode=mode, until=_At(2.0))
    world, t = _World(), 0.0
    while t < 3.0 - 1e-9:
        action.tick(world, t)
        handed.actor.step(_DT, driven=mode is RoutingMode.FOLLOW)
        t += _DT
    assert action.state is ActionState.COMPLETE
    assert handed.released == 1
    assert handed.resumed == 1
    assert handed.actor.control.throttle == 0.0
    assert handed.actor.control.brake == 1.0


def test_a_run_its_until_ends_after_arrival_keeps_its_vehicle(
    handed: _Handed,
) -> None:
    """Held at the end until *until* holds, as on arrival: not handed back."""
    trajectory = _line(length=10.0, duration=1.0)
    action = _route("npc1", trajectory, TrajectoryTiming(), until=_At(2.0))
    world, t = _World(), 0.0
    while t < 3.0:
        action.tick(world, t)
        t += _DT
    assert action.state is ActionState.COMPLETE
    assert handed.resumed == 0


def test_an_overridden_hidden_vehicle_is_left_parked() -> None:
    """An actor hidden under the map is left there (#1013)."""
    from autoware_carla_scenario.actions.base import reset_control_slots

    reset_control_slots()
    vehicle = _Handed(_Actor())
    register_entity("npc1", vehicle)
    try:
        trajectory = _line(length=40.0, duration=2.0)
        action = TrajectoryRun(
            "npc1",
            trajectory,
            TrajectoryTiming(ReferenceContext.ABSOLUTE, offset=5.0),
            hidden_outside_trajectory=True,
        )
        # Its clock starts at 5 s on the scenario's: hidden before then.
        _override(action, vehicle.actor, driven=False)
    finally:
        unregister_entity("npc1")
        reset_control_slots()
    assert vehicle.actor.physics is False
    assert vehicle.resumed == 0


class _Walker:
    """A pedestrian the registry knows as one, released and never resumed."""

    def __init__(self, actor: _Actor) -> None:
        self.actor = actor
        self.use_autopilot = True
        self.released = 0

    def release_from_traffic(self, world: Any) -> None:
        self.released += 1


def test_an_overridden_walker_is_stood_still() -> None:
    from autoware_carla_scenario.actions.base import reset_control_slots

    reset_control_slots()
    walker = _Walker(_Actor(speed=1.4, type_id="walker.pedestrian.0001"))
    register_pedestrian_entity("npc1", walker)  # type: ignore[arg-type]
    try:
        action = _route(
            "npc1", _line(length=40.0, duration=None), mode=RoutingMode.FOLLOW
        )
        world, t = _World(), 0.0
        while t < 1.0:
            action.tick(world, t)
            t += _DT
        assert walker.actor.walker_controls[-1].speed > 0.0
        action._stop_overridden(world, "test")
    finally:
        unregister_entity("npc1")
        reset_control_slots()
    assert walker.released == 1
    assert walker.actor.walker_controls[-1].speed == 0.0
