"""A point along the road from the pick: where another road user stands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.sweeper.bindings import (
    RouteOffsetBinding,
    RouteOffsetSBinding,
    parse_binding,
    route_offset_pose,
)
from autoware_carla_scenario.sweeper.constraints import create_routing_graph
from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

_OSM = Path(__file__).resolve().parents[3] / "data" / "nishishinjuku.osm"


@pytest.fixture(scope="module")
def nishishinjuku() -> tuple[Any, Any]:
    lanelet_map = load_lanelet2_map(_OSM)
    return lanelet_map, create_routing_graph(lanelet_map)


class TestRouteOffsetPose:
    def test_ahead_on_the_same_lane_stays_on_the_pick(self, nishishinjuku) -> None:
        lanelet_map, graph = nishishinjuku
        assert route_offset_pose(183, lanelet_map, graph, distance=60.0) == (183, 60.0)

    def test_behind_walks_back_onto_a_predecessor(self, nishishinjuku) -> None:
        lanelet_map, graph = nishishinjuku
        import lanelet2.geometry

        lanelet_id, s = route_offset_pose(183, lanelet_map, graph, distance=-40.0)
        # Walking the same predecessors back covers exactly the 40 m.
        current, behind = lanelet_map.laneletLayer[183], 0.0
        while current.id != lanelet_id:
            current = graph.previous(current)[0]
            behind += lanelet2.geometry.length2d(current)
        assert lanelet_id != 183
        assert behind - s == pytest.approx(40.0)

    @pytest.mark.parametrize(("side", "beside"), [("left", 182), ("right", 184)])
    def test_a_side_lane_is_the_one_a_vehicle_may_change_into(
        self, nishishinjuku, side: str, beside: int
    ) -> None:
        lanelet_map, graph = nishishinjuku
        lanelet_id, s = route_offset_pose(
            183, lanelet_map, graph, distance=60.0, side=side
        )
        assert lanelet_id == beside
        # Projected across, not copied: the lanes differ in length.
        assert s == pytest.approx(60.0, abs=0.5)

    def test_the_oncoming_lane_runs_the_other_way(self, nishishinjuku) -> None:
        lanelet_map, graph = nishishinjuku
        lanelet_id, _ = route_offset_pose(
            183, lanelet_map, graph, distance=-40.0, side="opposite"
        )
        ego_lane = route_offset_pose(183, lanelet_map, graph, distance=-40.0)[0]
        assert lanelet_id not in {ego_lane, 182, 183, 184}

    def test_running_off_the_road_raises_so_the_case_is_dropped(
        self, nishishinjuku
    ) -> None:
        lanelet_map, graph = nishishinjuku
        with pytest.raises(ValueError, match="runs off the road|has no"):
            route_offset_pose(183, lanelet_map, graph, distance=100_000.0)

    def test_an_unknown_side_is_refused(self, nishishinjuku) -> None:
        lanelet_map, graph = nishishinjuku
        with pytest.raises(ValueError, match="side"):
            route_offset_pose(183, lanelet_map, graph, distance=0.0, side="up")


class TestBindings:
    def test_the_pair_name_one_point(self, nishishinjuku) -> None:
        lanelet_map, graph = nishishinjuku
        lanelet = RouteOffsetBinding("k", distance=60.0, side="left").resolve(
            183, lanelet_map, graph
        )
        s = RouteOffsetSBinding("k", distance=60.0, side="left").resolve(
            183, lanelet_map, graph
        )
        assert lanelet.value == 182
        assert s.value == pytest.approx(60.0, abs=0.5)
        # Neither moves the searched slot: they describe someone else.
        assert lanelet.lanelet_id_override is None
        assert s.lanelet_id_override is None

    def test_they_are_reachable_from_the_yaml_form(self) -> None:
        assert parse_binding(
            "scenario.spawn_overrides.npc1.lanelet_id",
            {"type": "route_offset", "distance": 20.0, "side": "right"},
        ) == RouteOffsetBinding(
            "scenario.spawn_overrides.npc1.lanelet_id", distance=20.0, side="right"
        )
        assert isinstance(
            parse_binding(
                "scenario.spawn_overrides.npc1.s", {"type": "route_offset_s"}
            ),
            RouteOffsetSBinding,
        )

    def test_an_unknown_side_is_refused_when_it_is_written(self) -> None:
        with pytest.raises(ValueError, match="side"):
            RouteOffsetBinding("k", side="up")
