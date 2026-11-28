"""``stop_line_offset``'s ``through_junction: false`` keeps its walk back off
junction lanelets: a case whose approach is clear of junctions for less than
the offset is dropped, rather than started before another junction."""

from __future__ import annotations

from typing import Any

import pytest
from lanelet2.core import (
    AttributeMap,
    Lanelet,
    LaneletMap,
    LineString3d,
    Point3d,
    TrafficLight,
    getId,
)

from autoware_carla_scenario.sweeper.bindings import (
    StopLineApproachBinding,
    StopLineOffsetBinding,
    parse_binding,
)
from autoware_carla_scenario.sweeper.constraints import create_routing_graph


def _road() -> tuple[Any, Any, dict[str, Any]]:
    """One straight lane along x: a 30 m road, a 20 m junction lanelet (going
    straight), a 20 m approach, then the turn, whose light's stop line is
    0.5 m in.  The approach is clear of junctions for 20.5 m."""
    points: dict[float, tuple[Any, Any]] = {}

    def edge(x: float) -> tuple[Any, Any]:
        if x not in points:
            points[x] = (Point3d(getId(), x, 1.5, 0.0), Point3d(getId(), x, -1.5, 0.0))
        return points[x]

    def lanelet(x0: float, x1: float, turn: str | None = None) -> Any:
        attributes = {
            "type": "lanelet",
            "subtype": "road",
            "location": "urban",
            "one_way": "yes",
        }
        if turn is not None:
            attributes["turn_direction"] = turn
        (l0, r0), (l1, r1) = edge(x0), edge(x1)
        kind = AttributeMap({"type": "line_thin"})
        return Lanelet(
            getId(),
            LineString3d(getId(), [l0, l1], kind),
            LineString3d(getId(), [r0, r1], kind),
            AttributeMap(attributes),
        )

    lanes = {
        "road": lanelet(0.0, 30.0),
        "junction": lanelet(30.0, 50.0, "straight"),
        "approach": lanelet(50.0, 70.0),
        "turn": lanelet(70.0, 90.0, "right"),
    }
    stop_line = LineString3d(
        getId(),
        [Point3d(getId(), 70.5, 1.5, 0.0), Point3d(getId(), 70.5, -1.5, 0.0)],
        AttributeMap({"type": "stop_line"}),
    )
    light = LineString3d(
        getId(),
        [Point3d(getId(), 72.0, 3.0, 5.0), Point3d(getId(), 73.0, 3.0, 5.0)],
        AttributeMap({"type": "traffic_light"}),
    )
    lanes["turn"].addRegulatoryElement(
        TrafficLight(getId(), AttributeMap(), [light], stop_line)
    )
    lanelet_map = LaneletMap()
    for each in lanes.values():
        lanelet_map.add(each)
    return lanelet_map, create_routing_graph(lanelet_map), lanes


def test_by_default_the_walk_back_crosses_the_junction_before() -> None:
    lanelet_map, graph, lanes = _road()
    result = StopLineOffsetBinding("ego.spawn_s", 30.0).resolve(
        lanes["turn"].id, lanelet_map, graph
    )
    # 20.5 m of approach, then 9.5 m into the 20 m junction lanelet.
    assert result.lanelet_id_override == lanes["junction"].id
    assert result.value == pytest.approx(10.5, abs=0.01)


def test_kept_off_junctions_the_walk_back_is_refused_at_the_junction() -> None:
    lanelet_map, graph, lanes = _road()
    binding = StopLineOffsetBinding("ego.spawn_s", 30.0, through_junction=False)
    with pytest.raises(
        ValueError, match="clear of junctions for 20.50 m, short of the 30 m"
    ):
        binding.resolve(lanes["turn"].id, lanelet_map, graph)


def test_kept_off_junctions_an_offset_the_approach_holds_resolves() -> None:
    lanelet_map, graph, lanes = _road()
    result = StopLineOffsetBinding("ego.spawn_s", 15.0, through_junction=False).resolve(
        lanes["turn"].id, lanelet_map, graph
    )
    assert result.lanelet_id_override == lanes["approach"].id
    assert result.value == pytest.approx(5.5, abs=0.01)


def test_stop_line_approach_lands_where_stop_line_offset_does() -> None:
    lanelet_map, graph, lanes = _road()
    landed = StopLineApproachBinding("p.lanelet_id", 30.0).resolve(
        lanes["turn"].id, lanelet_map, graph
    )
    assert landed.value == lanes["junction"].id
    with pytest.raises(ValueError, match="through_junction: false"):
        StopLineApproachBinding("p.lanelet_id", 30.0, through_junction=False).resolve(
            lanes["turn"].id, lanelet_map, graph
        )


def test_through_junction_is_read_from_yaml() -> None:
    binding = parse_binding(
        "ego.spawn_s",
        {"type": "stop_line_offset", "offset": 50.0, "through_junction": False},
    )
    assert binding == StopLineOffsetBinding("ego.spawn_s", 50.0, through_junction=False)


@pytest.mark.parametrize("kind", ["stop_line_offset", "stop_line_approach"])
def test_through_junction_is_a_bool_not_its_text(kind: str) -> None:
    with pytest.raises(ValueError, match="through_junction must be true or false"):
        parse_binding("ego.spawn_s", {"type": kind, "through_junction": "false"})
