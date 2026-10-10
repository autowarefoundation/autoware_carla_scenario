"""The route search: a drive as a pattern of road, found on a Lanelet2 map.

Most tests run on the synthetic crossroads of :mod:`._route_maps` -- a
signalised approach from the west with a lane beside it, crosswalks on the W
and E legs, an E road that bends 45 degrees to the left, and a one-way road
elsewhere -- and a few on the nishishinjuku fixture map, to show a realistic
pattern is found on a real one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.route.model import (
    JunctionSegmentSpec,
    LaneSegmentSpec,
    RouteMatch,
    RouteSegmentMatch,
    anchor_problem,
    parse_anchor,
    parse_route_search,
)
from autoware_carla_scenario.route.search import find_route_matches, lane_shape
from autoware_carla_scenario.sweeper.expand import expand_sweep

from . import _route_maps as rm

_DATA = Path(__file__).resolve().parents[3] / "data"


@pytest.fixture(scope="module")
def road() -> tuple[Any, Any]:
    lanelet_map = rm.crossroads()
    return lanelet_map, rm.routing_graph(lanelet_map)


def _find(road: tuple[Any, Any], **search: Any) -> list[RouteMatch]:
    lanelet_map, graph = road
    return find_route_matches(parse_route_search(search), lanelet_map, graph)


def _lane(**kw: Any) -> dict[str, Any]:
    return {"kind": "lane", **kw}


def _junction(**kw: Any) -> dict[str, Any]:
    return {"kind": "junction", **kw}


# ---------------------------------------------------------------------------
# The search, read
# ---------------------------------------------------------------------------


class TestParsing:
    def test_a_full_search_is_read(self) -> None:
        spec = parse_route_search(
            {
                "segments": [
                    _lane(length={"min": 30, "max": 60}, lanes_left={"min": 1}),
                    _junction(turn="left", traffic_light=True, crosswalk_exit="no"),
                    _lane(shape="curved_left"),
                ],
                "ego_spawn_s": 5,
                "ego_goal_margin": 2,
                "max_matches": 3,
                "seed": 7,
            }
        )
        assert spec.junction_count == 1
        first, junction, _last = spec.segments
        assert isinstance(first, LaneSegmentSpec)
        assert isinstance(junction, JunctionSegmentSpec)
        assert first.length.min == 30 and first.length.max == 60
        assert first.lanes_left.min == 1 and first.lanes_left.max is None
        # YAML 1.1's bare yes/no arrive as booleans.
        assert junction.traffic_light == "yes"
        assert junction.crosswalk_exit == "no"
        assert (spec.ego_spawn_s, spec.ego_goal_margin, spec.seed) == (5.0, 2.0, 7)

    @pytest.mark.parametrize(
        ("search", "message"),
        [
            ({"segments": []}, "at least one segment"),
            ({"segments": [{"kind": "road"}]}, "'lane' or 'junction'"),
            ({"segments": [_lane(turn="left")]}, "does not take"),
            ({"segments": [_junction(length={"max": 3})]}, "does not take"),
            ({"segments": [_lane(length={"min": 9, "max": 3})]}, "above its max"),
            ({"segments": [_lane(shape="wiggly")]}, "shape must be one of"),
            ({"segments": [_junction(traffic_light="maybe")]}, "traffic_light"),
            ({"segments": [_lane(lanes_left={"min": 1.5})]}, "a count"),
            ({"segments": [_lane()], "max_matches": 0}, "max_matches"),
            ({"segments": [_lane()], "colour": 1}, "does not take"),
        ],
    )
    def test_a_malformed_search_says_what_is_wrong(
        self, search: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            parse_route_search(search)

    def test_anchors(self) -> None:
        assert parse_anchor("start") == ("start", -1, "")
        assert parse_anchor("junction:1:exit") == ("junction", 1, "exit")
        assert parse_anchor("segment:0:end") == ("segment", 0, "end")
        with pytest.raises(ValueError, match="not a route anchor"):
            parse_anchor("junction:1:start")
        assert anchor_problem("junction:1:entry", 3, 1) is not None
        assert anchor_problem("segment:2:end", 3, 1) is None
        assert anchor_problem(None, 0, 0) is None

    def test_shape_thresholds(self) -> None:
        import math

        assert lane_shape(math.radians(14.9)) == "straight"
        assert lane_shape(math.radians(-15.0)) == "straight"
        assert lane_shape(math.radians(15.1)) == "curved_left"
        assert lane_shape(math.radians(-40)) == "curved_right"


# ---------------------------------------------------------------------------
# Junctions
# ---------------------------------------------------------------------------


class TestJunctions:
    @pytest.mark.parametrize(
        ("turn", "junction", "exit_"),
        [
            ("left", rm.J[("W", "N")], rm.N_OUT),
            ("right", rm.J[("W", "S")], rm.S_OUT),
            ("straight", rm.J[("W", "E")], rm.E_OUT1),
        ],
    )
    def test_each_turn_is_found_from_the_signalised_approach(
        self, road: tuple[Any, Any], turn: str, junction: int, exit_: int
    ) -> None:
        matches = _find(
            road,
            segments=[
                _lane(length={"min": 30, "max": 60}),
                _junction(turn=turn, traffic_light="yes"),
                _lane(length={"max": 20}),
            ],
        )
        # Only the W approach is signalised.
        assert [m.lanelet_ids for m in matches] == [
            (rm.W_IN1, rm.W_IN2, junction, exit_)
        ]
        match = matches[0]
        lane, way, out = match.segments
        # The approach is grown back 60 m from the junction: 10 m into W_IN1.
        assert match.start_s == pytest.approx(40.0)
        assert (lane.start, lane.end) == pytest.approx((0.0, 60.0))
        assert way.kind == "junction" and way.turn == turn
        assert way.lanelet_ids == (junction,)
        assert out.length == pytest.approx(20.0)
        assert match.end_s == pytest.approx(20.0)
        assert match.length == pytest.approx(out.end)

    def test_unsignalised_left_turns_are_all_found(self, road: tuple[Any, Any]) -> None:
        matches = _find(
            road,
            segments=[
                _lane(length={"min": 30, "max": 60}),
                _junction(turn="left", traffic_light="no"),
                _lane(length={"max": 20}),
            ],
        )
        junctions = {m.segments[1].lanelet_ids[0] for m in matches}
        assert junctions == {rm.J[("E", "S")], rm.J[("N", "E")], rm.J[("S", "W")]}

    @pytest.mark.parametrize(
        ("crossing", "expected"),
        [
            # Every approach crosses the W straight-through way...
            ({"crossing_from_left": "yes", "crossing_from_right": "yes"}, True),
            # ...the E left turn among them, from the opposite approach.
            ({"crossing_from_opposite": "yes"}, True),
            ({"crossing_from_left": "no"}, False),
        ],
    )
    def test_crossing_traffic(
        self, road: tuple[Any, Any], crossing: dict[str, str], expected: bool
    ) -> None:
        matches = _find(
            road,
            segments=[
                _lane(length={"max": 50}),
                _junction(turn="straight", traffic_light="yes", **crossing),
            ],
        )
        assert bool(matches) is expected

    def test_crossing_is_conflict_not_presence(self, road: tuple[Any, Any]) -> None:
        # Turning right from W, nothing from the opposite approach crosses the
        # ego's way (the E road's own right turn goes elsewhere), though the
        # opposite approach has lanelets in the junction.
        found = _find(
            road,
            segments=[
                _lane(length={"max": 50}),
                _junction(
                    turn="right", traffic_light="yes", crossing_from_opposite="yes"
                ),
            ],
        )
        lanes = [m.segments[1].lanelet_ids for m in found]
        assert (rm.J[("W", "S")],) in lanes

    def test_crosswalks_on_the_legs(self, road: tuple[Any, Any]) -> None:
        def ways(**crosswalks: str) -> set[int]:
            found = _find(
                road,
                segments=[
                    _lane(length={"max": 50}),
                    _junction(traffic_light="yes", **crosswalks),
                    _lane(length={"max": 20}),
                ],
            )
            return {m.segments[1].lanelet_ids[0] for m in found}

        # The W leg's crosswalk is on every W way's entry; only going straight
        # leaves by the E leg, which has the other one.
        assert ways(crosswalk_entry="yes") == {
            rm.J[("W", "E")],
            rm.J[("W", "N")],
            rm.J[("W", "S")],
        }
        assert ways(crosswalk_exit="yes") == {rm.J[("W", "E")]}
        assert ways(crosswalk_exit="no") == {rm.J[("W", "N")], rm.J[("W", "S")]}

    def test_a_route_may_start_and_end_with_a_junction(
        self, road: tuple[Any, Any]
    ) -> None:
        matches = _find(road, segments=[_junction(turn="right")])
        assert len(matches) == 4
        assert all(len(m.lanelet_ids) == 1 for m in matches)
        assert all(m.start_s == 0.0 for m in matches)


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


class TestLanes:
    def test_lanes_beside_and_stop_lines(self, road: tuple[Any, Any]) -> None:
        matches = _find(
            road,
            segments=[
                _lane(
                    length={"max": 100},
                    lanes_right={"min": 1},
                    lanes_left={"max": 0},
                    stop_line="yes",
                    traffic_light_stop_line="yes",
                ),
                _junction(turn="straight"),
            ],
        )
        assert [m.lanelet_ids for m in matches] == [
            (rm.W_IN1, rm.W_IN2, rm.J[("W", "E")])
        ]
        assert matches[0].start_s == 0.0  # 100 m: the whole road

    def test_no_lane_beside(self, road: tuple[Any, Any]) -> None:
        matches = _find(road, segments=[_lane(lanes_right={"max": 0}, stop_line="yes")])
        assert matches == []

    def test_opposite_lane(self, road: tuple[Any, Any]) -> None:
        two_way = _find(road, segments=[_lane(length={"min": 50}, opposite_lane="yes")])
        one_way = _find(road, segments=[_lane(length={"min": 50}, opposite_lane="no")])
        assert rm.ONE_WAY1 not in {m.lanelet_ids[0] for m in two_way}
        assert [m.lanelet_ids for m in one_way] == [(rm.ONE_WAY1, rm.ONE_WAY2)]

    def test_a_lone_lane_segment_starts_where_its_road_does(
        self, road: tuple[Any, Any]
    ) -> None:
        matches = _find(road, segments=[_lane(length={"min": 50, "max": 80})])
        starts = {m.lanelet_ids[0] for m in matches}
        # Never part-way along a road: W_IN2, which W_IN1 precedes, starts none.
        assert rm.W_IN2 not in starts and rm.W_IN1 in starts
        one_way = next(m for m in matches if m.lanelet_ids[0] == rm.ONE_WAY1)
        assert one_way.length == pytest.approx(80.0)
        assert one_way.end_s == pytest.approx(20.0)

    def test_shape(self, road: tuple[Any, Any]) -> None:
        bend = _find(
            road,
            segments=[_junction(turn="straight"), _lane(shape="curved_left")],
        )
        assert [m.lanelet_ids for m in bend] == [
            (rm.J[("W", "E")], rm.E_OUT1, rm.E_OUT2)
        ]
        straight = _find(
            road,
            segments=[
                _junction(turn="straight"),
                _lane(shape="straight", length={"max": 50}),
            ],
        )
        assert (rm.J[("W", "E")], rm.E_OUT1) in [m.lanelet_ids for m in straight]

    def test_length_bounds_an_interior_lane_segment(
        self, road: tuple[Any, Any]
    ) -> None:
        # E_OUT1 + E_OUT2 is about 80.6 m and E_IN2 + E_IN1 about 80 m: an
        # interior segment covers whole lanelets, so a range that excludes both
        # finds no route through two junctions.
        assert (
            _find(
                road,
                segments=[
                    _junction(),
                    _lane(length={"max": 50}),
                    _junction(),
                ],
            )
            == []
        )


class TestNoMatchAndMany:
    def test_no_match(self, road: tuple[Any, Any]) -> None:
        assert (
            _find(
                road,
                segments=[
                    _junction(turn="left", traffic_light="yes", crosswalk_exit="yes")
                ],
            )
            == []
        )

    def test_matches_are_sorted_and_indexed(self, road: tuple[Any, Any]) -> None:
        matches = _find(road, segments=[_junction(turn="straight")])
        assert [m.lanelet_ids for m in matches] == sorted(
            m.lanelet_ids for m in matches
        )
        assert [m.index for m in matches] == list(range(len(matches)))

    def test_max_matches_and_seed(self, road: tuple[Any, Any]) -> None:
        every = _find(road, segments=[_junction()])
        first = _find(road, segments=[_junction()], max_matches=3)
        assert [m.lanelet_ids for m in first] == [m.lanelet_ids for m in every[:3]]
        shuffled = _find(road, segments=[_junction()], max_matches=5, seed=1)
        again = _find(road, segments=[_junction()], max_matches=5, seed=1)
        assert [m.lanelet_ids for m in shuffled] == [m.lanelet_ids for m in again]
        assert {m.lanelet_ids for m in shuffled} <= {m.lanelet_ids for m in every}


# ---------------------------------------------------------------------------
# A match, carried through Hydra
# ---------------------------------------------------------------------------


class TestTheMatchAsConfig:
    def test_round_trip(self, road: tuple[Any, Any]) -> None:
        (match,) = _find(
            road,
            segments=[
                _lane(length={"min": 30, "max": 60}),
                _junction(turn="left", traffic_light="yes"),
                _lane(length={"max": 20}),
            ],
        )
        config = match.to_config()
        assert config["junction_turns"] == ["left"]
        assert config["segment_kinds"] == ["lane", "junction", "lane"]
        back = RouteMatch.from_config(config)
        assert back is not None
        assert back.lanelet_ids == match.lanelet_ids
        assert [s.lanelet_ids for s in back.segments] == [
            s.lanelet_ids for s in match.segments
        ]
        assert back.anchor_s("junction:0:entry") == pytest.approx(60.0)
        assert back.anchor_s("junction:0:exit") == pytest.approx(
            match.segments[1].end, abs=1e-3
        )
        assert back.anchor_s("end") == pytest.approx(match.length, abs=1e-3)

    def test_empty_config_is_no_match(self) -> None:
        assert RouteMatch.from_config({}) is None
        assert RouteMatch.from_config({"lanelet_ids": []}) is None

    def test_inconsistent_config_is_refused(self) -> None:
        with pytest.raises(ValueError, match="add up"):
            RouteMatch.from_config(
                {
                    "lanelet_ids": [1, 2],
                    "segment_kinds": ["lane"],
                    "segment_ends": [10.0],
                    "segment_lanelet_counts": [1],
                    "junction_turns": [],
                }
            )

    def test_an_anchor_the_route_lacks(self) -> None:
        match = RouteMatch(
            (1,), 0.0, 10.0, (RouteSegmentMatch("lane", 0.0, 10.0, (1,)),)
        )
        with pytest.raises(ValueError, match="no junction 0"):
            match.anchor_s("junction:0:entry")


class TestExpansion:
    def test_one_case_per_match(self, road: tuple[Any, Any]) -> None:
        lanelet_map, _graph = road
        cases = expand_sweep(
            {
                "route": {
                    "segments": [
                        _lane(length={"min": 30, "max": 60}),
                        _junction(turn="left"),
                        _lane(length={"max": 20}),
                    ],
                    "ego_spawn_s": 5.0,
                    "ego_goal_margin": 2.0,
                }
            },
            lanelet_map,
            ["extra=1"],
        )
        assert len(cases) == 4
        first = cases[0]
        assert first[:4] == [
            f"ego.spawn_lanelet_id={rm.W_IN1}",
            "ego.spawn_s=45.0",
            f"ego.goal_lanelet_id={rm.N_OUT}",
            "ego.goal_s=18.0",
        ]
        assert (
            f"scenario.route.lanelet_ids=[{rm.W_IN1},{rm.W_IN2},602,{rm.N_OUT}]"
            in first
        )
        assert "scenario.route.segment_kinds=[lane,junction,lane]" in first
        assert "scenario.route.junction_turns=[left]" in first
        assert first[-1] == "extra=1"

    def test_no_goal_when_the_search_says_so(self, road: tuple[Any, Any]) -> None:
        lanelet_map, _graph = road
        (case, *_rest) = expand_sweep(
            {"route": {"segments": [_junction(turn="right")], "ego_goal": False}},
            lanelet_map,
        )
        assert not any(o.startswith("ego.goal") for o in case)

    def test_route_and_constraints_together_are_refused(
        self, road: tuple[Any, Any]
    ) -> None:
        with pytest.raises(ValueError, match="one of them"):
            expand_sweep(
                {
                    "route": {"segments": [_junction()]},
                    "constraints": {"ego.spawn_lanelet_id": [{"type": "is_junction"}]},
                },
                road[0],
            )

    def test_a_spawn_past_the_route_drops_the_case(self, road: tuple[Any, Any]) -> None:
        cases = expand_sweep(
            {"route": {"segments": [_junction(turn="right")], "ego_spawn_s": 500.0}},
            road[0],
        )
        assert cases == []

    def test_the_expanded_overrides_compose(
        self, road: tuple[Any, Any], tmp_path: Path
    ) -> None:
        """What the expansion writes is what an exported config declares."""
        from omegaconf import OmegaConf

        from autoware_carla_scenario.authoring.hydra_config import (
            ROUTE_CONFIG_DEFAULTS,
        )

        (case, *_rest) = expand_sweep(
            {
                "route": {
                    "segments": [_junction(turn="left"), _lane(length={"max": 20})]
                }
            },
            road[0],
        )
        config = OmegaConf.create(
            {
                "scenario": {"route": dict(ROUTE_CONFIG_DEFAULTS)},
                "ego": {
                    "spawn_lanelet_id": 0,
                    "spawn_s": 0.0,
                    "goal_lanelet_id": None,
                    "goal_s": 0.0,
                },
            }
        )
        OmegaConf.set_struct(config, True)
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parser = OverridesParser.create()
        for override in parser.parse_overrides(case):
            OmegaConf.update(
                config, override.key_or_group, override.value(), merge=False
            )
        route = OmegaConf.to_container(config.scenario.route)
        assert isinstance(route, dict)
        match = RouteMatch.from_config({str(k): v for k, v in route.items()})
        assert match is not None and match.junctions[0].turn == "left"


# ---------------------------------------------------------------------------
# A real map
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nishishinjuku() -> tuple[Any, Any]:
    from autoware_carla_scenario.sweeper.constraints import create_routing_graph
    from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

    lanelet_map = load_lanelet2_map(
        _DATA / "nishishinjuku.osm", _DATA / "nishishinjuku_carla.xodr"
    )
    return lanelet_map, create_routing_graph(lanelet_map)


class TestNishishinjuku:
    def test_a_signalised_right_turn_with_crossing_traffic(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        matches = _find(
            nishishinjuku,
            segments=[
                _lane(length={"min": 30, "max": 60}, opposite_lane="yes"),
                _junction(
                    turn="right",
                    traffic_light="yes",
                    crossing_from_opposite="yes",
                    crosswalk_exit="yes",
                ),
                _lane(length={"min": 20, "max": 40}),
            ],
            max_matches=10,
        )
        assert matches
        graph = nishishinjuku[1]
        layer = nishishinjuku[0].laneletLayer
        for match in matches:
            # Consecutive in the routing graph, no lane change.
            for a, b in zip(match.lanelet_ids, match.lanelet_ids[1:]):
                assert b in {ll.id for ll in graph.following(layer[a])}
            lane, way, out = match.segments
            assert 30.0 - 1e-6 <= lane.length <= 60.0 + 1e-6
            assert 20.0 - 1e-6 <= out.length <= 40.0 + 1e-6
            assert all(
                str(layer[i].attributes["turn_direction"]) == "right"
                for i in way.lanelet_ids
            )
            assert way.turn == "right"

    def test_opposite_lanes_are_found_on_a_left_hand_traffic_map(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        from autoware_carla_scenario.route.geometry import MapFeatures

        features = MapFeatures(*nishishinjuku)
        sides = {
            found[1]
            for ll in nishishinjuku[0].laneletLayer
            if features.is_road(ll) and not features.is_junction(ll)
            for found in [features.opposite(ll)]
            if found is not None
        }
        # Japan drives on the left: the opposite road is mostly on the right.
        assert "right" in sides


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_each_split_between_two_lane_segments_is_its_own_match() -> None:
    """Three lanelets, two lane segments: split after the first or the second."""
    builder = rm._Builder()
    for lanelet_id, (a, b) in enumerate(((0, 50), (50, 100), (100, 150)), start=1):
        builder.lanelet(lanelet_id, rm._line((a, -rm._HALF), (b, -rm._HALF)))
    graph = rm.routing_graph(builder.map)
    matches = find_route_matches(
        parse_route_search({"segments": [_lane(), _lane()]}), builder.map, graph
    )
    splits = [[s.lanelet_ids for s in m.segments] for m in matches]
    assert splits == [[(1,), (2, 3)], [(1, 2), (3,)]]
    assert [m.index for m in matches] == [0, 1]


class TestJunctionPaths:
    """A path drawn as two junction lanelets is classified by where it enters."""

    @pytest.fixture(scope="class")
    def split(self) -> tuple[Any, Any]:
        lanelet_map = rm.crossroads(split_path=True)
        return lanelet_map, rm.routing_graph(lanelet_map)

    def test_a_member_is_named_by_its_first_lanelet(
        self, split: tuple[Any, Any]
    ) -> None:
        from autoware_carla_scenario.route.geometry import MapFeatures

        lanelet_map, graph = split
        members = MapFeatures(lanelet_map, graph).junction_members(
            [lanelet_map.laneletLayer[rm.J[("W", "E")]]]
        )
        by_id = {int(m.lanelet.id): m for m in members}
        # Its second lanelet heads within 45 deg of the ego: by its own
        # heading it would read as the ego's approach and be dropped.
        assert rm.SPLIT_SECOND not in by_id
        first = by_id[rm.SPLIT_FIRST]
        assert (first.approach, first.turn, first.conflicts) == ("left", "left", True)

    def test_a_crossing_pose_is_measured_from_where_the_path_enters(
        self, split: tuple[Any, Any]
    ) -> None:
        from autoware_carla_scenario.route.frame import RouteFrame
        from autoware_carla_scenario.route.positions import resolve_route_pose
        from autoware_carla_scenario.trajectory.model import RouteCrossingPose

        (match,) = find_route_matches(
            parse_route_search(
                {
                    "segments": [
                        _lane(length={"max": 50}),
                        _junction(turn="straight", traffic_light="yes"),
                    ]
                }
            ),
            *split,
        )
        pose = resolve_route_pose(
            RouteCrossingPose(approach="left", turn="left", distance=1.0),
            RouteFrame(match, *split),
        )
        assert (pose.lanelet_id, pose.s) == (rm.SPLIT_FIRST, pytest.approx(1.0))  # type: ignore[union-attr]
