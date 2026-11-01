"""A repeating (``once=False``) routing action starts every run afresh (#1057).

A run's progress -- where along its path the entity is, whether the backend
already drives its last metres, how far along it started -- belongs to that
run, not to the next one the action's condition triggers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import pytest

from autoware_carla_scenario import (
    Lanelet2Pose,
    ReferenceContext,
    RoutingAction,
    RoutingMode,
    TrajectoryTiming,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.conditions import BaseCondition, ScenarioResult
from autoware_carla_scenario.coordinate import CarlaWorldPose
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.entity import registry
from autoware_carla_scenario.entity.registry import register_entity
from autoware_carla_scenario.trajectory.model import Trajectory, TrajectoryVertex

from .test_routing_stop_and_handback import _LANELET, _Car, loaded_map  # noqa: F401
from .test_routing_trajectory_action import _DT, _Actor, _Entity, _line, _route, _World


class _Armed(BaseCondition):
    """Holds once each time the test arms it: one trigger per run."""

    def __init__(self) -> None:
        super().__init__(label="armed")
        self.armed = False

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        if self.armed:
            self.armed = False
            return ScenarioResult(True, "armed", elapsed)
        return None


class _Follower(_Car):
    """A car that takes a written trajectory's stops as its route."""

    def __init__(self, x: float, y: float, z: float, yaw: float) -> None:
        super().__init__(x, y, z, yaw)
        self.followed: list[list[Any]] = []

    def follow_route(self, world: Any, whole: Any) -> None:
        self.followed.append(list(whole))


@pytest.fixture
def lane(
    loaded_map: MapManager,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Any]:
    """The fixture map loaded and an empty registry; the lane's placement."""
    from autoware_carla_scenario.coordinate.transform import to_carla_world

    MapManager._instance = loaded_map
    monkeypatch.setattr(registry, "_VEHICLES", {})
    monkeypatch.setattr(registry, "_PEDESTRIANS", {})

    def at(s: float) -> Any:
        return to_carla_world(Lanelet2Pose(_LANELET, float(s)))

    yield at


def _place(car: _Car, point: Any) -> None:
    car.location.x, car.location.y = point.x, point.y


def test_every_run_to_a_goal_has_its_last_metres_finished(lane: Any) -> None:
    """``stop_at_goal`` hands the backend the last metres on each run, not on
    the first only."""
    start = lane(10.0)
    car = _Car(start.x, start.y, start.z, start.yaw)
    register_entity("npc1", car)
    armed = _Armed()
    action = RoutingAction(
        "npc1",
        Lanelet2Pose(_LANELET, 60.0),
        condition=armed,
        stop_at_goal=True,
        once=False,
    )
    world = object()
    elapsed = 0.0
    for run in (1, 2):
        _place(car, lane(10.0))
        armed.armed = True
        for s in [*range(10, 61, 5), 60, 60, 60]:
            _place(car, lane(s))
            action.tick(world, elapsed)
            elapsed += _DT
        assert action.state is ActionState.STANDBY, run
        assert len(car.finishes) == run
    assert len(car.routes) == 2


def test_every_driver_run_starts_from_the_offset_given(lane: Any) -> None:
    """A run that begins with its vehicle partway along raises the offset to
    there for that run only: the next, from the start, keeps every stop."""
    points = [lane(s) for s in (10, 20, 30, 40, 50, 60)]
    vertices = [
        TrajectoryVertex(CarlaWorldPose(p.x, p.y, p.z, yaw=p.yaw)) for p in points
    ]
    start = lane(30.0)
    car = _Follower(start.x, start.y, start.z, start.yaw)
    register_entity("npc1", car)
    armed = _Armed()
    action = RoutingAction(
        "npc1",
        trajectory=Trajectory("t", vertices),
        mode=RoutingMode.DRIVER,
        condition=armed,
        once=False,
    )
    world = _World()
    elapsed = 0.0
    stops: list[list[int]] = []
    for begin in (30, 10):
        _place(car, lane(begin))
        armed.armed = True
        action.tick(world, elapsed)
        elapsed += _DT
        stops.append(list(action._stop_vertices))
        for s in range(begin, 61, 2):
            _place(car, lane(s))
            action.tick(world, elapsed)
            elapsed += _DT
        for _ in range(3):
            action.tick(world, elapsed)
            elapsed += _DT
        assert action.state is ActionState.STANDBY, begin
    # The first run starts where the car stands, 20 m along; the second, at
    # the path's start, passes no stop by.
    assert stops == [[3, 4, 5], [1, 2, 3, 4, 5]]
    assert action._initial_distance_offset == 0.0


def test_every_position_run_records_its_arrivals_from_its_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each run finds the entity from its own start: the second notes no
    arrival at a waypoint it has not reached yet."""
    monkeypatch.setattr(registry, "_VEHICLES", {})
    monkeypatch.setattr(registry, "_PEDESTRIANS", {})
    actor = _Actor()
    register_entity("npc1", _Entity(actor))
    armed = _Armed()
    action = _route(
        "npc1",
        _line(length=200.0, duration=20.0, count=11),
        TrajectoryTiming(ReferenceContext.RELATIVE),
        condition=armed,
        once=False,
    )
    world = _World()
    elapsed = 0.0
    early: list[list[int]] = []
    for _run in (1, 2):
        armed.armed = True
        for _ in range(3):  # the trigger and two ticks of the run
            action.tick(world, elapsed)
            elapsed += _DT
        early.append([a.vertex for a in action.take_waypoint_arrivals()])
        while action.state is not ActionState.STANDBY:
            action.tick(world, elapsed)
            elapsed += _DT
        action.take_waypoint_arrivals()
    assert early == [[0], [0]]
