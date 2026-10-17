"""OffRoadFractionCondition against a fake world and map (no CARLA server)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import OffRoadFractionCondition
from autoware_carla_scenario.coordinate.map_manager import MapManager

_WORLD_IDS = itertools.count(1)


@dataclass
class _Vec:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class _Rotation:
    yaw: float = 0.0


@dataclass
class _Transform:
    location: _Vec
    rotation: _Rotation


@dataclass
class _Box:
    location: _Vec
    extent: _Vec


@dataclass
class _Actor:
    role_name: str
    transform: _Transform
    bounding_box: _Box
    id: int = 1
    is_alive: bool = True

    @property
    def attributes(self) -> dict[str, str]:
        return {"role_name": self.role_name}

    def get_transform(self) -> _Transform:
        return self.transform


@dataclass
class _Query:
    x: float
    y: float
    z: float
    project_to_road: bool
    lane_type: int


@dataclass
class _Map:
    """A map whose driving lanes are wherever *on_lane(x, y)* holds."""

    on_lane: Callable[[float, float], bool]
    queries: list[_Query] = field(default_factory=list)

    def get_waypoint(
        self, location: carla.Location, project_to_road: bool, lane_type: int
    ) -> Optional[object]:
        self.queries.append(
            _Query(location.x, location.y, location.z, project_to_road, lane_type)
        )
        return object() if self.on_lane(location.x, location.y) else None


@dataclass
class _World:
    actors: list[_Actor]
    carla_map: _Map
    id: int = field(default_factory=lambda: next(_WORLD_IDS))
    get_map_calls: int = 0

    def get_actors(self) -> list[_Actor]:
        return self.actors

    def get_actor(self, actor_id: int) -> Optional[_Actor]:
        return next((a for a in self.actors if a.id == actor_id), None)

    def get_map(self) -> _Map:
        self.get_map_calls += 1
        return self.carla_map


@pytest.fixture(autouse=True)
def _no_held_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """No MapManager holding a run's map: the condition takes the world's."""
    monkeypatch.setattr(MapManager, "_instance", None)


def _actor(
    role_name: str = "Ego",
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    box_offset: tuple[float, float] = (0.0, 0.0),
    actor_id: int = 1,
) -> _Actor:
    """An actor whose box is 4 m long and 2 m wide: grid points 1 m apart."""
    return _Actor(
        role_name,
        _Transform(_Vec(x, y, z), _Rotation(yaw)),
        _Box(_Vec(box_offset[0], box_offset[1], 0.0), _Vec(2.0, 1.0, 0.75)),
        id=actor_id,
    )


def _world(on_lane: Callable[[float, float], bool], *actors: _Actor) -> _World:
    return _World(list(actors), _Map(on_lane))


def _everywhere(x: float, y: float) -> bool:
    return True


def _points(world: _World) -> set[tuple[float, float]]:
    return {(round(q.x, 6), round(q.y, 6)) for q in world.carla_map.queries}


class TestConstruction:
    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01])
    def test_fraction_outside_zero_one_is_refused(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="fraction"):
            OffRoadFractionCondition(fraction, label="offroad")

    @pytest.mark.parametrize("fraction", [1 / 15, 0.3, 1.0])
    def test_fraction_in_zero_one_is_taken(self, fraction: float) -> None:
        condition = OffRoadFractionCondition(fraction, label="offroad")
        assert condition.get_details() == {"entity_name": "Ego", "fraction": fraction}


class TestCheck:
    def test_on_the_lanes_is_no_result(self) -> None:
        world = _world(_everywhere, _actor())
        condition = OffRoadFractionCondition(1 / 15, label="offroad")

        assert condition.check(world, 1.0) is None
        assert condition.off_road_fraction == 0.0
        # A 5 x 3 grid, the box's edges and corners included.
        assert _points(world) == {
            (float(x), float(y)) for x in (-2, -1, 0, 1, 2) for y in (-1, 0, 1)
        }

    def test_fails_at_the_threshold(self) -> None:
        # The lane ends 1.5 m ahead: the front edge (3 points) is off.
        world = _world(lambda x, y: x <= 1.5, _actor())
        condition = OffRoadFractionCondition(3 / 15, label="offroad")

        result = condition.check(world, 2.5)

        assert result is not None
        assert not result.passed
        assert result.elapsed_seconds == 2.5
        assert "20%" in result.message
        assert condition.off_road_fraction == pytest.approx(0.2)

    def test_just_below_the_threshold_is_no_result(self) -> None:
        world = _world(lambda x, y: x <= 1.5, _actor())
        condition = OffRoadFractionCondition(4 / 15, label="offroad")

        assert condition.check(world, 1.0) is None
        assert condition.off_road_fraction == pytest.approx(3 / 15)

    def test_yaw_turns_the_grid(self) -> None:
        # Facing +y (yaw 90): forward is +y, and right, CARLA being
        # left-handed, is -x.  With the lane ending 1.5 m ahead in y, the
        # front edge is off, as it is at yaw 0 with the lane ending in x.
        world = _world(lambda x, y: y <= 1.5, _actor(yaw=90.0))
        condition = OffRoadFractionCondition(1 / 15, label="offroad")

        assert condition.check(world, 1.0) is not None
        assert condition.off_road_fraction == pytest.approx(3 / 15)
        assert _points(world) == {
            (float(x), float(y)) for x in (-1, 0, 1) for y in (-2, -1, 0, 1, 2)
        }

    def test_yaw_90_right_is_minus_x(self) -> None:
        # The grid is symmetric about the box's centre, so the side the box
        # is offset to is what tells right from left: 0.5 m to the right at
        # yaw 90 is -x in CARLA's left-handed frame, so the box spans x in
        # [-1.5, 0.5], and a road ending at x = -1.25 leaves its right-hand
        # long side (5 points) off.
        world = _world(lambda x, y: x >= -1.25, _actor(yaw=90.0, box_offset=(0.0, 0.5)))
        condition = OffRoadFractionCondition(1 / 3, label="offroad")

        assert condition.check(world, 1.0) is not None
        assert condition.off_road_fraction == pytest.approx(5 / 15)
        assert {x for x, _ in _points(world)} == {-1.5, -0.5, 0.5}

    def test_box_location_offset_moves_the_grid(self) -> None:
        # The box's centre 1 m ahead of the transform (and 0.5 m right):
        # the grid spans x in [-1, 3], so a lane ending at 2.5 leaves the
        # front edge off where the unshifted box would be wholly on it.
        world = _world(lambda x, y: x <= 2.5, _actor(box_offset=(1.0, 0.5)))
        condition = OffRoadFractionCondition(1 / 15, label="offroad")

        assert condition.check(world, 1.0) is not None
        assert condition.off_road_fraction == pytest.approx(3 / 15)
        assert (3.0, 1.5) in _points(world)
        assert (-1.0, -0.5) in _points(world)

    def test_box_location_offset_follows_yaw(self) -> None:
        world = _world(_everywhere, _actor(yaw=90.0, box_offset=(1.0, 0.0)))
        condition = OffRoadFractionCondition(1 / 15, label="offroad")

        condition.check(world, 1.0)

        # The box's centre is 1 m ahead along +y.
        assert (0.0, 1.0) in _points(world)
        assert {y for _, y in _points(world)} == {-1.0, 0.0, 1.0, 2.0, 3.0}

    def test_missing_entity_is_no_result(self) -> None:
        world = _world(lambda x, y: False)
        condition = OffRoadFractionCondition(1 / 15, label="offroad")

        assert condition.check(world, 1.0) is None
        assert condition.off_road_fraction is None
        assert world.get_map_calls == 0

    def test_queries_at_the_entity_height_for_driving_lanes_only(self) -> None:
        world = _world(_everywhere, _actor(z=12.5))
        condition = OffRoadFractionCondition(1 / 15, label="offroad")

        condition.check(world, 1.0)

        queries = world.carla_map.queries
        assert len(queries) == 15
        assert {q.z for q in queries} == {12.5}
        assert {q.project_to_road for q in queries} == {False}
        assert {q.lane_type for q in queries} == {int(carla.LaneType.Driving)}

    def test_watches_the_named_entity(self) -> None:
        ego = _actor("Ego", actor_id=1)
        npc = _actor("npc1", x=100.0, actor_id=2)
        world = _world(lambda x, y: x < 50.0, ego, npc)
        condition = OffRoadFractionCondition(1.0, "npc1", label="npc offroad")

        assert condition.check(world, 1.0) is not None
        assert condition.get_details()["entity_name"] == "npc1"
        assert condition.get_details()["off_road_fraction"] == 1.0

    def test_map_is_fetched_once(self) -> None:
        world = _world(_everywhere, _actor())
        condition = OffRoadFractionCondition(1 / 15, label="offroad")

        condition.check(world, 1.0)
        condition.check(world, 2.0)

        assert world.get_map_calls == 1

    def test_takes_the_runs_map_when_one_is_held(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        held = _Map(lambda x, y: False)
        manager = MapManager()
        monkeypatch.setattr(manager, "_carla_map", held)
        monkeypatch.setattr(MapManager, "_instance", manager)
        world = _world(_everywhere, _actor())
        condition = OffRoadFractionCondition(1.0, label="offroad")

        assert condition.check(world, 1.0) is not None
        assert world.get_map_calls == 0
        assert len(held.queries) == 15


def test_angles_are_degrees() -> None:
    # A 45-degree yaw puts the front-centre point on the diagonal.
    world = _world(_everywhere, _actor(yaw=45.0))
    OffRoadFractionCondition(1 / 15, label="offroad").check(world, 1.0)

    front = 2.0 / math.sqrt(2.0)
    assert (round(front, 6), round(front, 6)) in _points(world)
