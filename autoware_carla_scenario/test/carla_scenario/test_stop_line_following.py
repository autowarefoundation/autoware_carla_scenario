"""Finding a stop line some lanelets ahead of where a vehicle starts."""

from __future__ import annotations

import pytest

lanelet2 = pytest.importorskip("lanelet2")

from autoware_carla_scenario.utils.stop_line import (  # noqa: E402
    get_stop_line_linestrings_with_following,
)


def _road_with_stop_line_on(index: int, count: int = 3) -> tuple[object, list[int], int]:
    """*count* 10 m lanelets end to end, a traffic light's stop line at the
    start of lanelet *index*."""
    from lanelet2.core import (
        AttributeMap,
        Lanelet,
        LaneletMap,
        LineString3d,
        Point3d,
        TrafficLight,
        getId,
    )

    lanelet_map = LaneletMap()
    left = [Point3d(getId(), 10.0 * i, 1.75, 0.0) for i in range(count + 1)]
    right = [Point3d(getId(), 10.0 * i, -1.75, 0.0) for i in range(count + 1)]
    lanelets = [
        Lanelet(getId(), LineString3d(getId(), left[i : i + 2]), LineString3d(getId(), right[i : i + 2]))
        for i in range(count)
    ]
    stop_line = LineString3d(getId(), [left[index], right[index]], AttributeMap({"type": "stop_line"}))
    light = LineString3d(
        getId(),
        [Point3d(getId(), 10.0 * index + 5, 1.0, 5.0), Point3d(getId(), 10.0 * index + 5, -1.0, 5.0)],
        AttributeMap({"type": "traffic_light"}),
    )
    regulatory_element = TrafficLight(getId(), AttributeMap(), [light], stop_line)
    lanelets[index].addRegulatoryElement(regulatory_element)
    for lanelet in lanelets:
        lanelet_map.add(lanelet)
    return lanelet_map, [lanelet.id for lanelet in lanelets], stop_line.id


def test_one_step_finds_a_stop_line_on_the_next_lanelet() -> None:
    lanelet_map, ids, stop_line = _road_with_stop_line_on(1)
    found = get_stop_line_linestrings_with_following(lanelet_map, ids[0])
    assert [(owner, line.id) for owner, line in found] == [(ids[1], stop_line)]


def test_a_stop_line_two_lanelets_ahead_needs_depth_two() -> None:
    lanelet_map, ids, stop_line = _road_with_stop_line_on(2)
    assert get_stop_line_linestrings_with_following(lanelet_map, ids[0]) == []
    found = get_stop_line_linestrings_with_following(lanelet_map, ids[0], depth=2)
    assert [(owner, line.id) for owner, line in found] == [(ids[2], stop_line)]
