"""A routed pedestrian's walker controller is released however its run ends (#1012).

In ``driver`` mode a RoutingAction walks a pedestrian on CARLA's
``controller.ai.walker``.  Arriving released it; an override, an ``until``
condition and a second route did not, so the controller kept walking the
walker to the first route's target and ``entity.walking`` stayed true.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    CarlaWorldPose,
    PedestrianEntity,
    PedestrianEntityConfig,
    RoutingAction,
    SpawnTransform,
    WalkStraightAction,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.conditions import (
    AlwaysTrueCondition,
    ElapsedTimeCondition,
)
from autoware_carla_scenario.conditions.base import BaseCondition, ScenarioResult
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import OpenDrivePose
from autoware_carla_scenario.coordinate.transform import to_carla_world
from autoware_carla_scenario.entity import _walker_ai
from autoware_carla_scenario.entity.registry import clear_entities, register_entity
from autoware_carla_scenario.trajectory.model import (
    ResolvedTrajectory,
    Trajectory,
    TrajectoryVertex,
)

_DATA = Path(__file__).resolve().parents[3] / "data"
_TICK_S = 0.05
_WALKER_ID = 4077


class _Never(BaseCondition):
    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        return None


class _Controller:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def start(self) -> None:
        self.calls.append("start")

    def go_to_location(self, location: Any) -> None:
        self.calls.append(("go", round(location.x, 1)))

    def set_max_speed(self, speed: float) -> None:
        self.calls.append(("speed", speed))

    def stop(self) -> None:
        self.calls.append("stop")

    def destroy(self) -> None:
        self.calls.append("destroy")


class _AiWorld:
    """A world with a navigation mesh, which hands out one controller."""

    id = 4242

    def __init__(self) -> None:
        self.controller = _Controller()

    def set_pedestrians_cross_factor(self, factor: float) -> None:
        pass

    def get_random_location_from_navigation(self) -> Any:
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(find=lambda name: name)

    def spawn_actor(self, blueprint: Any, transform: Any, attach_to: Any) -> Any:
        return self.controller

    def get_snapshot(self) -> Any:
        return SimpleNamespace(timestamp=SimpleNamespace(delta_seconds=_TICK_S))

    def get_map(self) -> Any:
        return SimpleNamespace(get_waypoint=lambda *a, **k: None)


class _Walker:
    """A walker that stays where it is put and records the speeds it is sent."""

    id = _WALKER_ID
    type_id = "walker.pedestrian.0001"

    def __init__(self, world: _AiWorld, at: Any) -> None:
        self.world = world
        self.loc = carla.Location(at.x, at.y, at.z)
        self.controls: list[float] = []

    def get_world(self) -> _AiWorld:
        return self.world

    def get_location(self) -> Any:
        return self.loc

    def get_transform(self) -> Any:
        return carla.Transform(self.loc, carla.Rotation(yaw=0.0))

    def get_velocity(self) -> Any:
        return carla.Vector3D(0.0, 0.0, 0.0)

    def apply_control(self, control: Any) -> None:
        self.controls.append(round(control.speed, 2))


@pytest.fixture(scope="module")
def _map() -> Iterator[None]:
    MapManager.reset()
    mm = MapManager.get_instance()
    mm.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    yield
    mm._routing_graph = None
    mm._lanelet_map = None
    mm._road_network = None
    mm._geo_origin = None
    mm._mgrs_offset = None
    MapManager.reset()


def _points(*s_values: float) -> list[Any]:
    """Points on the walkway beside road 173, at each *s*."""
    return [
        to_carla_world(OpenDrivePose(road_id="173", lane_id=-1, s=s, t=-4.0))
        for s in s_values
    ]


#: Two walks from the same start: east, along the road, and back west.
def _east() -> list[Any]:
    return _points(60.0, 63.0, 66.0, 69.0, 72.0)


def _west() -> list[Any]:
    return _points(60.0, 57.0, 54.0, 51.0, 48.0)


def _trajectory(name: str, points: list[Any]) -> Trajectory:
    return Trajectory(
        name, [TrajectoryVertex(CarlaWorldPose(p.x, p.y, p.z, yaw=0.0)) for p in points]
    )


@pytest.fixture
def scene(_map: None) -> Iterator[SimpleNamespace]:
    for walker_id in list(_walker_ai._DRIVEN):
        _walker_ai.release(walker_id)
    clear_entities()
    world = _AiWorld()
    walker = _Walker(world, _east()[0])
    entity = PedestrianEntity(
        PedestrianEntityConfig(
            role_name="w1", spawn_location=SpawnTransform(carla.Transform())
        )
    )
    entity._walker = walker  # type: ignore[assignment]
    register_entity("w1", entity)
    yield SimpleNamespace(world=world, walker=walker, entity=entity)
    for walker_id in list(_walker_ai._DRIVEN):
        _walker_ai.release(walker_id)
    clear_entities()


def _run(action: Any, world: _AiWorld, ticks: int, t: float = 0.0) -> float:
    for _ in range(ticks):
        action.tick(world, t)
        t += _TICK_S
    return t


def _walking_route(
    world: _AiWorld, label: str = "east", **kwargs: Any
) -> RoutingAction:
    kwargs.setdefault("until", _Never(label="never"))
    route = RoutingAction(
        "w1", trajectory=_trajectory(label, _east()), label=label, **kwargs
    )
    return route


def test_a_route_overridden_by_a_walk_straight_releases_the_controller(
    scene: SimpleNamespace,
) -> None:
    route = _walking_route(scene.world)
    t = _run(route, scene.world, 3)
    assert scene.entity.walking and route.walking
    assert scene.world.controller.calls[0] == "start"
    stand = WalkStraightAction(
        "w1", 0.0, condition=AlwaysTrueCondition(), overrides=True, label="stand"
    )
    stand.tick(scene.world, t)
    assert route.state is ActionState.COMPLETE
    assert scene.world.controller.calls[-2:] == ["stop", "destroy"]
    assert not scene.entity.walking
    assert not route.walking
    assert _WALKER_ID not in _walker_ai._DRIVEN
    # The walker stood, and the walk straight's own control followed.
    assert scene.walker.controls[-2:] == [0.0, 0.0]
    # The run's end hands back nothing twice.
    calls = list(scene.world.controller.calls)
    route.close()
    assert scene.world.controller.calls == calls


def test_a_second_route_overriding_the_first_walks_to_its_own_targets(
    scene: SimpleNamespace,
) -> None:
    first = _walking_route(scene.world)
    t = _run(first, scene.world, 3)
    first_targets = [
        round(location.x, 1) for _, location in _walker_ai._DRIVEN[_WALKER_ID].targets
    ]
    second = RoutingAction(
        "w1",
        trajectory=_trajectory("west", _west()),
        condition=AlwaysTrueCondition(),
        overrides=True,
        label="west",
    )
    mark = len(scene.world.controller.calls)
    _run(second, scene.world, 3, t)
    assert first.state is ActionState.COMPLETE
    after = scene.world.controller.calls[mark:]
    assert after[:3] == ["stop", "destroy", "start"]
    second_targets = [
        round(location.x, 1) for _, location in _walker_ai._DRIVEN[_WALKER_ID].targets
    ]
    assert second_targets != first_targets
    goes = [call for call in after if isinstance(call, tuple) and call[0] == "go"]
    assert goes[0] == ("go", second_targets[0])
    assert second.walking and scene.entity.walking
    # West: the second route's targets are on the other side of the start.
    start_x = _west()[0].x
    end_x = _west()[-1].x
    assert (second_targets[0] - start_x) * (end_x - start_x) > 0.0


def test_a_route_ended_by_its_until_releases_the_controller(
    scene: SimpleNamespace,
) -> None:
    cut = _walking_route(scene.world, until=ElapsedTimeCondition(0.1, label="cut"))
    _run(cut, scene.world, 6)
    assert cut.state is ActionState.COMPLETE
    assert scene.world.controller.calls[0] == "start"
    assert scene.world.controller.calls[-2:] == ["stop", "destroy"]
    assert not scene.entity.walking
    assert not cut.walking
    assert _WALKER_ID not in _walker_ai._DRIVEN
    assert scene.walker.controls[-1] == 0.0


def test_a_run_closed_mid_walk_releases_the_controller(scene: SimpleNamespace) -> None:
    route = _walking_route(scene.world)
    _run(route, scene.world, 3)
    route.close()
    assert scene.world.controller.calls[-2:] == ["stop", "destroy"]
    assert not scene.entity.walking
    route.reset()
    assert not route.walking


def test_an_arrived_walk_is_released_once_as_before(scene: SimpleNamespace) -> None:
    route = _walking_route(scene.world, until=None)
    t = _run(route, scene.world, 3)
    end = _east()[-1]
    scene.walker.loc = carla.Location(end.x, end.y, end.z)
    _run(route, scene.world, 3, t)
    assert route.state is ActionState.COMPLETE
    assert route.finished
    calls = scene.world.controller.calls
    assert calls[-2:] == ["stop", "destroy"]
    assert calls.count("stop") == 1 and calls.count("destroy") == 1
    assert not scene.entity.walking
    controls = list(scene.walker.controls)
    route.close()
    assert scene.world.controller.calls == calls
    assert scene.walker.controls == controls


def _entity_with(walker: Any) -> PedestrianEntity:
    entity = PedestrianEntity(
        PedestrianEntityConfig(
            role_name="w2", spawn_location=SpawnTransform(carla.Transform())
        )
    )
    entity._walker = walker
    return entity


def _straight(length: float) -> ResolvedTrajectory:
    return ResolvedTrajectory([0.0, length], [0.0, 0.0], [0.0, 0.0], [None, None])


def test_stop_walk_is_safe_twice_and_on_a_destroyed_walker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.entity import pedestrian_entity as pe

    released: list[int] = []
    monkeypatch.setattr(pe, "_release_walker_ai", lambda w: released.append(w.id))
    controls: list[float] = []
    walker = SimpleNamespace(
        id=5, apply_control=lambda control: controls.append(control.speed)
    )
    entity = _entity_with(walker)
    entity.walk_along(_straight(20.0), 1.4)
    assert entity.walking
    entity.stop_walk()
    assert not entity.walking
    assert controls == [0.0]
    entity.stop_walk()
    assert controls == [0.0]  # no walk: nothing more sent
    # A walker gone from the world: its control is refused.
    gone = SimpleNamespace(id=6, apply_control=lambda control: _raise())
    entity = _entity_with(gone)
    entity.walk_along(_straight(20.0), 1.4)
    entity.stop_walk()
    assert not entity.walking
    # A destroyed entity: no walker at all.
    entity.walk_along(_straight(20.0), 1.4)
    entity._walker = None
    entity.stop_walk()
    entity.stop_walk()
    assert not entity.walking


def _raise() -> None:
    raise RuntimeError("destroyed")


def test_a_new_walk_releases_the_controller_of_the_one_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.entity import pedestrian_entity as pe

    released: list[int] = []
    monkeypatch.setattr(pe, "_release_walker_ai", lambda w: released.append(w.id))
    entity = _entity_with(SimpleNamespace(id=9, apply_control=lambda control: None))
    entity.walk_along(_straight(20.0), 1.4)
    entity.walk_along(_straight(10.0), 1.4)
    assert released == [9, 9]
    walk = entity.walk_path
    assert walk is not None and walk.length == pytest.approx(10.0)
