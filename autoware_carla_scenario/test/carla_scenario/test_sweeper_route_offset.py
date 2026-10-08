"""A point along the road from the pick: where another road user stands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.authoring.models import (
    BindingRef,
    MapRef,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.registry import MAP_TRAFFIC_SIDE_REF
from autoware_carla_scenario.editor.map_preview import resolve_map_refs
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

    def test_where_traffic_keeps_left_the_oncoming_lane_is_on_the_right(
        self, nishishinjuku
    ) -> None:
        # Tokyo: across the centre line is the driver's right.
        lanelet_map, graph = nishishinjuku
        lanelet_id, s = route_offset_pose(
            183,
            lanelet_map,
            graph,
            distance=60.0,
            side="opposite",
            traffic_side="left",
        )
        assert lanelet_id not in {182, 183, 184}
        side, alignment = _beside(lanelet_map, 183, 60.0, lanelet_id)
        assert side == "right"
        assert alignment < -0.9  # it runs the other way

    def test_inside_a_junction_the_oncoming_lane_goes_straight_through(
        self, nishishinjuku
    ) -> None:
        # 296 runs straight through a junction; across it, so does 298 the
        # other way.  A turning junction lanelet crosses, but a straight one is
        # the oncoming lane there.
        lanelet_map, graph = nishishinjuku
        lanelet_id, _ = route_offset_pose(
            296,
            lanelet_map,
            graph,
            distance=9.7,
            side="opposite",
            traffic_side="left",
        )
        assert lanelet_id == 298
        assert lanelet_map.laneletLayer[298].attributes["turn_direction"] == "straight"
        side, alignment = _beside(lanelet_map, 296, 9.7, 298)
        assert (side, alignment < -0.9) == ("right", True)

    def test_where_traffic_keeps_right_it_is_looked_for_on_the_left(
        self, nishishinjuku
    ) -> None:
        # The same lane read as right-hand traffic finds nothing across a
        # centre line on its left: there is none, the oncoming road is right.
        lanelet_map, graph = nishishinjuku
        with pytest.raises(ValueError, match="no opposite lane"):
            route_offset_pose(183, lanelet_map, graph, distance=60.0, side="opposite")

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

    def test_an_unknown_traffic_side_is_refused(self, nishishinjuku) -> None:
        lanelet_map, graph = nishishinjuku
        with pytest.raises(ValueError, match="traffic side"):
            route_offset_pose(
                183, lanelet_map, graph, distance=0.0, traffic_side="middle"
            )


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
        with pytest.raises(ValueError, match="traffic_side"):
            RouteOffsetBinding("k", traffic_side="middle")

    def test_the_traffic_side_reaches_the_walk(self, nishishinjuku) -> None:
        lanelet_map, graph = nishishinjuku
        binding = parse_binding(
            "k",
            {
                "type": "route_offset",
                "distance": 60.0,
                "side": "opposite",
                "traffic_side": "left",
            },
        )
        assert (
            binding.resolve(183, lanelet_map, graph).value
            == route_offset_pose(
                183,
                lanelet_map,
                graph,
                distance=60.0,
                side="opposite",
                traffic_side="left",
            )[0]
        )


class TestTrafficSideComesFromTheMap:
    def test_the_exported_binding_reads_the_map_group(self) -> None:
        swept = BindingRef(
            type="route_offset", params={"distance": 10.0, "side": "opposite"}
        ).to_sweep_dict()
        assert swept["traffic_side"] == MAP_TRAFFIC_SIDE_REF

    @pytest.mark.parametrize(
        ("group", "side"), [("nishishinjuku", "left"), ("town10hd_opt", "right")]
    )
    def test_each_map_group_resolves_it(self, group: str, side: str) -> None:
        conf = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "autoware_carla_scenario"
            / "examples"
            / "conf"
        )
        cfg = OmegaConf.merge(
            OmegaConf.load(conf / "config.yaml"),
            OmegaConf.load(conf / "map" / f"{group}.yaml"),
        )
        assert cfg.map.traffic_side == side

    def test_the_editor_fills_it_from_the_document(self) -> None:
        ref = {"type": "route_offset", "traffic_side": MAP_TRAFFIC_SIDE_REF}
        tokyo = ScenarioDocument(id="t", map=MapRef(group="nishishinjuku"))
        assert resolve_map_refs(ref, tokyo)["traffic_side"] == "left"
        named = ScenarioDocument(
            id="n", map=MapRef(group="town10hd_opt", traffic_side="left")
        )
        assert resolve_map_refs(ref, named)["traffic_side"] == "left"
        elsewhere = ScenarioDocument(id="e", map=MapRef(group="no_such_group"))
        assert resolve_map_refs(ref, elsewhere)["traffic_side"] == "right"


def _beside(
    lanelet_map: Any, ego_id: int, s: float, other_id: int
) -> tuple[str, float]:
    """Which side of the ego's lane *other_id* is at *s*, and how its heading aligns."""
    import lanelet2.geometry

    centre = lanelet2.geometry.to2D(lanelet_map.laneletLayer[ego_id].centerline)
    p = lanelet2.geometry.interpolatedPointAtDistance(centre, s)
    q = lanelet2.geometry.interpolatedPointAtDistance(centre, s + 1.0)
    hx, hy = q.x - p.x, q.y - p.y
    other = list(lanelet2.geometry.to2D(lanelet_map.laneletLayer[other_id].centerline))
    k = min(
        range(len(other) - 1),
        key=lambda i: (other[i].x - p.x) ** 2 + (other[i].y - p.y) ** 2,
    )
    side = "left" if hx * (other[k].y - p.y) - hy * (other[k].x - p.x) > 0 else "right"
    ox, oy = other[k + 1].x - other[k].x, other[k + 1].y - other[k].y
    return side, (hx * ox + hy * oy) / ((hx**2 + hy**2) ** 0.5 * (ox**2 + oy**2) ** 0.5)
