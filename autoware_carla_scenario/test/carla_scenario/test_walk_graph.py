"""The pedestrian walk graph: crosswalks that only overlap the walkways (#519).

A small synthetic map, drawn as Autoware and roadgen draw one::

        C (corner pavement, from the end of N, y 7..30 at x = 51)
        |
    ====N====================  walkway, y = 6, x -50..50, ``one_way: yes``
              | X |            crosswalk, x 9..11, reaching 0.3 m onto each
    ====S====================  walkway, y = -6, x -50..50, ``one_way: yes``

    F: a crosswalk far away, overlapping nothing.

Nothing shares a point but N and C, so Lanelet2's routing graph has no way
from S to N; the walk graph has, across X.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import autoware_lanelet2_extension_python.projection  # noqa: F401
import lanelet2.core
import lanelet2.routing
import lanelet2.traffic_rules
import pytest
import typesafe_carla.carla as carla
from lanelet2.core import AttributeMap, Lanelet, LaneletMap, LineString3d, Point3d

from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import Lanelet2Pose
from autoware_carla_scenario.coordinate.walk_graph import WalkGraph, walk_graph

NORTH, SOUTH, CROSSWALK, CORNER, FAR, ROAD = 1, 2, 3, 4, 5, 6

_ids = iter(range(1000, 1_000_000))


def _line(coords: list[tuple[float, float]]) -> Any:
    return LineString3d(
        next(_ids),
        [Point3d(next(_ids), x, y, 0.0) for x, y in coords],
        AttributeMap({"type": "line_thin", "subtype": "solid"}),
    )


def _lanelet(lanelet_id: int, left: Any, right: Any, subtype: str, one_way: str) -> Any:
    return Lanelet(
        lanelet_id,
        left,
        right,
        AttributeMap(
            {
                "type": "lanelet",
                "subtype": subtype,
                "location": "urban",
                "one_way": one_way,
            }
        ),
    )


def _map() -> Any:
    lanelet_map = LaneletMap()
    north_left = _line([(-50.0, 7.0), (0.0, 7.0), (50.0, 7.0)])
    north_right = _line([(-50.0, 5.0), (0.0, 5.0), (50.0, 5.0)])
    lanelet_map.add(_lanelet(NORTH, north_left, north_right, "walkway", "yes"))
    lanelet_map.add(
        _lanelet(
            SOUTH,
            _line([(-50.0, -5.0), (0.0, -5.0), (50.0, -5.0)]),
            _line([(-50.0, -7.0), (0.0, -7.0), (50.0, -7.0)]),
            "walkway",
            "yes",
        )
    )
    lanelet_map.add(
        _lanelet(
            ROAD,
            _line([(-50.0, 5.0), (50.0, 5.0)]),
            _line([(-50.0, -5.0), (50.0, -5.0)]),
            "road",
            "yes",
        )
    )
    # From the S walkway to the N one, 0.3 m onto each: no point shared.
    lanelet_map.add(
        _lanelet(
            CROSSWALK,
            _line([(9.0, -5.3), (9.0, 5.3)]),
            _line([(11.0, -5.3), (11.0, 5.3)]),
            "crosswalk",
            "no",
        )
    )
    # Round the corner from the end of N, on the very points it ends on.
    end_left, end_right = (
        north_left[len(north_left) - 1],
        north_right[len(north_right) - 1],
    )
    lanelet_map.add(
        Lanelet(
            CORNER,
            LineString3d(next(_ids), [end_left, Point3d(next(_ids), 50.0, 30.0, 0.0)]),
            LineString3d(next(_ids), [end_right, Point3d(next(_ids), 52.0, 30.0, 0.0)]),
            AttributeMap({"type": "lanelet", "subtype": "walkway", "one_way": "yes"}),
        )
    )
    lanelet_map.add(
        _lanelet(
            FAR,
            _line([(500.0, 0.0), (500.0, 10.0)]),
            _line([(502.0, 0.0), (502.0, 10.0)]),
            "crosswalk",
            "no",
        )
    )
    return lanelet_map


@pytest.fixture(scope="module")
def lanelet_map() -> Any:
    return _map()


@pytest.fixture(scope="module")
def graph(lanelet_map: Any) -> WalkGraph:
    return WalkGraph(lanelet_map)


def _runs(walk: list[Lanelet2Pose]) -> list[int]:
    """The lanelets of *walk*, each once per stretch on it."""
    runs: list[int] = []
    for pose in walk:
        if not runs or runs[-1] != pose.lanelet_id:
            runs.append(pose.lanelet_id)
    return runs


def _xy(lanelet_map: Any, pose: Lanelet2Pose) -> tuple[float, float]:
    centerline = lanelet2.geometry.to2D(
        lanelet_map.laneletLayer[pose.lanelet_id].centerline
    )
    point = lanelet2.geometry.interpolatedPointAtDistance(centerline, pose.s)
    return (float(point.x), float(point.y))


class TestTheMapAsDrawn:
    def test_lanelet2_has_no_way_across(self, lanelet_map: Any) -> None:
        """The reason the walk graph exists: the crosswalk joins nothing."""
        rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Pedestrian,
        )
        routing = lanelet2.routing.RoutingGraph(lanelet_map, rules)
        south = lanelet_map.laneletLayer[SOUTH]
        north = lanelet_map.laneletLayer[NORTH]
        assert routing.shortestPath(south, north) is None
        # and one_way: yes keeps a pedestrian off a walkway walked backwards.
        assert not rules.canPass(south.invert())


class TestTheWalkGraph:
    def test_it_crosses_at_the_crosswalk(self, graph: WalkGraph) -> None:
        walk = graph.walk(Lanelet2Pose(SOUTH, 5.0), Lanelet2Pose(NORTH, 95.0), 1.5)
        assert walk is not None
        assert _runs(walk) == [SOUTH, CROSSWALK, NORTH]
        # Along S to the crossing at x = 10 (s = 60), and from it along N.
        on_south = [p.s for p in walk if p.lanelet_id == SOUTH]
        on_north = [p.s for p in walk if p.lanelet_id == NORTH]
        assert on_south[-1] == pytest.approx(60.0)
        assert on_north[0] == pytest.approx(60.0)
        assert walk[0] == Lanelet2Pose(SOUTH, 5.0)
        assert walk[-1] == Lanelet2Pose(NORTH, 95.0)

    def test_a_one_way_walkway_is_walked_either_way(self, graph: WalkGraph) -> None:
        walk = graph.walk(Lanelet2Pose(NORTH, 90.0), Lanelet2Pose(NORTH, 10.0), 1.5)
        assert walk is not None
        assert _runs(walk) == [NORTH]
        stations = [p.s for p in walk]
        assert stations == sorted(stations, reverse=True)

    def test_back_across_the_way_it_came(self, graph: WalkGraph) -> None:
        walk = graph.walk(Lanelet2Pose(NORTH, 95.0), Lanelet2Pose(SOUTH, 5.0), 1.5)
        assert walk is not None
        assert _runs(walk) == [NORTH, CROSSWALK, SOUTH]

    def test_lanelets_sharing_an_end_are_joined(self, graph: WalkGraph) -> None:
        walk = graph.walk(Lanelet2Pose(SOUTH, 5.0), Lanelet2Pose(CORNER, 10.0), 1.5)
        assert walk is not None
        assert _runs(walk) == [SOUTH, CROSSWALK, NORTH, CORNER]

    def test_the_poses_are_a_step_apart(
        self, graph: WalkGraph, lanelet_map: Any
    ) -> None:
        walk = graph.walk(Lanelet2Pose(SOUTH, 5.0), Lanelet2Pose(NORTH, 95.0), 1.5)
        assert walk is not None
        points = [_xy(lanelet_map, p) for p in walk]
        steps = [math.dist(a, b) for a, b in zip(points, points[1:])]
        # A step along a lanelet, or onto the next one: from a walkway's
        # centreline to the crosswalk's end, 0.7 m off it.
        assert max(steps) <= 1.5 + 1e-6
        # The walk is the one along the pavements and over the crossing.
        assert sum(steps) == pytest.approx(55.0 + 0.7 + 10.6 + 0.7 + 35.0, abs=0.1)

    def test_nothing_joins_a_far_crosswalk(self, graph: WalkGraph) -> None:
        assert graph.walk(Lanelet2Pose(SOUTH, 5.0), Lanelet2Pose(FAR, 5.0), 1.5) is None

    def test_a_road_is_not_walked(self, graph: WalkGraph) -> None:
        assert not graph.walkable(ROAD)
        assert (
            graph.walk(Lanelet2Pose(SOUTH, 5.0), Lanelet2Pose(ROAD, 5.0), 1.5) is None
        )

    def test_it_is_built_once_per_map(self, lanelet_map: Any) -> None:
        assert walk_graph(lanelet_map) is walk_graph(lanelet_map)
        assert walk_graph(_map()) is not walk_graph(lanelet_map)


class _StandingWalker:
    def __init__(self, transform: Any) -> None:
        self._transform = transform

    def get_transform(self) -> Any:
        return self._transform


@pytest.fixture
def loaded(lanelet_map: Any) -> Iterator[None]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._lanelet_map = lanelet_map
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    yield
    MapManager._instance = saved


class TestAWalkerCrossesTheRoad:
    def test_route_to_crosses_at_the_crosswalk(self, loaded: None) -> None:
        from autoware_carla_scenario.coordinate.poses import CarlaWorldPose
        from autoware_carla_scenario.coordinate.transform import (
            on_lanelet,
            to_carla_world,
        )
        from autoware_carla_scenario.entity import (
            PedestrianEntity,
            PedestrianEntityConfig,
            SpawnTransform,
        )

        walker = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1", spawn_location=SpawnTransform(carla.Transform())
            )
        )
        walker._walker = _StandingWalker(  # type: ignore[assignment]
            to_carla_world(Lanelet2Pose(SOUTH, 5.0)).to_carla_transform()
        )
        walker.route_to(object(), Lanelet2Pose(NORTH, 95.0))  # type: ignore[arg-type]

        walk = walker._walk
        assert walk is not None
        points = [
            CarlaWorldPose(x=x, y=y, z=z) for x, y, z in zip(walk.xs, walk.ys, walk.zs)
        ]
        # Every point on a walkable lanelet: none cut across the road.
        walkable = (NORTH, SOUTH, CROSSWALK)
        assert all(any(on_lanelet(p, lid) for lid in walkable) for p in points)
        assert any(on_lanelet(p, CROSSWALK) for p in points)
