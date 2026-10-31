"""RouteGraph: a road ahead walked along the lanelets, as the closed loop's policies did."""

from __future__ import annotations

import numpy as np

from autoware_carla_scenario.t4_scene.route import RouteGraph, _resample


def _line(*points: tuple[float, float]) -> np.ndarray:
    return _resample(np.array(points, dtype=np.float64))


def _fork() -> RouteGraph:
    """Lanelet 1 (x 0-10) on to 2 (straight on to x 20) or 3 (which leaves
    2 at x 12 and bends away to y 6); 4 runs beside 2, a lane change away."""
    return RouteGraph(
        [1, 2, 3, 4],
        np.stack(
            [
                _line((0.0, 0.0), (10.0, 0.0)),
                _line((10.0, 0.0), (20.0, 0.0)),
                _line((10.0, 0.0), (12.0, 0.0), (16.0, 6.0), (20.0, 6.0)),
                _line((10.0, -3.5), (20.0, -3.5)),
            ]
        ),
        successors={1: (2, 3)},
        neighbours={2: (4,), 4: (2,)},
    )


def test_the_road_ahead_takes_the_branch_it_keeps_to() -> None:
    straight = np.array([[x, 0.0] for x in np.arange(1.0, 20.0, 2.0)])
    assert _fork().match(straight) == [1, 2]
    bending = np.array(
        [[1.0, 0.0], [5.0, 0.0], [9.0, 0.0], [11.0, 0.0], [14.0, 3.0], [18.0, 6.0]]
    )
    assert _fork().match(bending) == [1, 3]


def test_a_lane_change_steps_to_the_neighbour() -> None:
    changing = np.array(
        [[1.0, 0.0], [5.0, 0.0], [9.0, 0.0], [12.0, 0.0], [15.0, -3.5], [18.0, -3.5]]
    )
    assert _fork().match(changing) == [1, 2, 4]


def test_a_point_heading_the_other_way_is_on_no_lanelet() -> None:
    backwards = np.array([[x, 0.0] for x in np.arange(9.0, 0.0, -2.0)])
    assert _fork().match(backwards) == []


def test_a_road_ahead_on_no_lanelet_is_the_pin_alone() -> None:
    nowhere = np.array([[50.0, 50.0], [52.0, 50.0]])
    assert _fork().match(nowhere, start_id=1) == [1]
    assert _fork().match(nowhere) == []


def test_the_pin_wins_a_tie_for_the_first_lanelet() -> None:
    mouth = np.array([[10.5, 0.0], [11.0, 0.0]])
    assert _fork().match(mouth, start_id=3) == [3]
    assert _fork().match(mouth, start_id=2) == [2]


def test_the_nearest_route_lanelet_is_found_within_reach() -> None:
    graph = _fork()
    assert graph.nearest([1, 2, 3], np.array([15.0, 0.5])) == 1
    assert graph.nearest([1, 2], np.array([15.0, 20.0]), 3.0) is None


def _lanelet_map(gap: float):  # type: ignore[no-untyped-def]
    """Two road lanelets along x, the second's bounds starting *gap* metres
    after the first's end, as roadgen's junction nodes do."""
    import lanelet2
    from lanelet2.core import Lanelet, LineString3d, Point3d, getId

    def bound(y: float, x0: float, x1: float) -> LineString3d:
        return LineString3d(
            getId(), [Point3d(getId(), x0, y, 0.0), Point3d(getId(), x1, y, 0.0)]
        )

    lanelets = [
        Lanelet(getId(), bound(1.5, x0, x1), bound(-1.5, x0, x1))
        for x0, x1 in ((0.0, 10.0), (10.0 + gap, 20.0))
    ]
    for lanelet in lanelets:
        lanelet.attributes["subtype"] = "road"
        lanelet.attributes["one_way"] = "yes"
    return lanelet2.core.createMapFromLanelets(lanelets), [int(x.id) for x in lanelets]


def test_bound_ends_that_nearly_meet_are_welded_into_a_join() -> None:
    lanelet_map, (first, second) = _lanelet_map(0.1)
    assert RouteGraph.of(lanelet_map, owned=False).successors(first) == ()
    lanelet_map, (first, second) = _lanelet_map(0.1)
    graph = RouteGraph.of(lanelet_map)
    assert graph.successors(first) == (second,)
    road = np.array([[x, 0.0] for x in np.arange(1.0, 20.0, 2.0)])
    assert graph.match(road) == [first, second]


def test_an_owned_map_is_placed_at_its_local_tags() -> None:
    lanelet_map, (first, _) = _lanelet_map(0.0)
    for point in lanelet_map.pointLayer:
        point.attributes["local_x"] = str(point.x + 100.0)
        point.attributes["local_y"] = str(point.y)
    graph = RouteGraph.of(lanelet_map)
    road = np.array([[x, 0.0] for x in np.arange(101.0, 110.0, 2.0)])
    assert graph.match(road) == [first]
