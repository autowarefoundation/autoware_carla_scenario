"""Crossings and crosswalks on the Nishi-Shinjuku map."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.sweeper import topology
from autoware_carla_scenario.sweeper.bindings import parse_binding
from autoware_carla_scenario.sweeper.constraints import (
    create_routing_graph,
    find_matching_lanelets,
    parse_constraint,
)
from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

_OSM = Path(__file__).resolve().parents[3] / "data" / "nishishinjuku.osm"


@pytest.fixture(scope="module")
def nishishinjuku() -> tuple[Any, Any]:
    lanelet_map = load_lanelet2_map(_OSM)
    return lanelet_map, create_routing_graph(lanelet_map)


class TestOnTheMap:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_a_crossing_vehicle_starts_on_a_lane_into_the_junction(
        self, nishishinjuku, side: str
    ) -> None:
        lanelet_map, graph = nishishinjuku
        (pick, *_) = find_matching_lanelets(
            [parse_constraint({"type": "has_crossing", "value": side})],
            lanelet_map,
            graph,
        )
        lanelet = parse_binding("k", {"type": "crossing", "side": side}).resolve(
            pick, lanelet_map, graph
        )
        s = parse_binding("k", {"type": "crossing_s", "side": side}).resolve(
            pick, lanelet_map, graph
        )
        start = lanelet_map.laneletLayer[lanelet.value]
        assert lanelet.value != pick
        # It leads into the junction lanelet that crosses the ego's path.
        assert any("turn_direction" in nxt.attributes for nxt in graph.following(start))
        import lanelet2.geometry

        assert isinstance(s.value, float)
        assert 0.0 <= s.value <= lanelet2.geometry.length2d(start)

    def test_a_pedestrian_starts_at_the_kerb_and_faces_across(
        self, nishishinjuku
    ) -> None:
        lanelet_map, graph = nishishinjuku
        (pick, *_) = find_matching_lanelets(
            [parse_constraint({"type": "has_crosswalk_ahead", "distance": 60.0})],
            lanelet_map,
            graph,
        )
        starts = {
            side: topology.crosswalk_start(
                lanelet_map.laneletLayer[pick], lanelet_map, graph, side
            )
            for side in ("left", "right")
        }
        left, right = starts["left"], starts["right"]
        crosswalk = lanelet_map.laneletLayer[left.lanelet_id]
        assert "crosswalk" == str(crosswalk.attributes["subtype"])
        # The same crosswalk, from opposite kerbs, walked opposite ways.
        assert left.lanelet_id == right.lanelet_id
        assert {left.heading, right.heading} == {0.0, math.pi}
        assert left.s != right.s

    def test_no_crosswalk_ahead_raises_so_the_case_is_dropped(
        self, nishishinjuku
    ) -> None:
        lanelet_map, graph = nishishinjuku
        everything = {ll.id for ll in lanelet_map.laneletLayer}
        with_crosswalk = set(
            find_matching_lanelets(
                [parse_constraint({"type": "has_crosswalk_ahead", "distance": 60.0})],
                lanelet_map,
                graph,
            )
        )
        lacking = min(everything - with_crosswalk)
        with pytest.raises(ValueError, match="no crosswalk"):
            topology.crosswalk_start(
                lanelet_map.laneletLayer[lacking], lanelet_map, graph, "left"
            )
