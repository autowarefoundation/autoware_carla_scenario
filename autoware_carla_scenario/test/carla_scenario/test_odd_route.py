"""A scenario's planned route checked against an ODD, on the Lanelet2 map alone.

Routed for real on the nishishinjuku fixture map, loaded once for the module
(the routing graph takes a couple of seconds to build).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

import lanelet2.geometry

from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import Lanelet2Pose
from autoware_carla_scenario.odd import (
    UNDECIDED,
    OddAttribute,
    OddDefinition,
    OddModule,
    PlannedRoute,
    RouteError,
    combine_route_coverage,
    default_odd,
    plan_route,
    plan_route_coverage,
    probes,
    register_odd,
)
from autoware_carla_scenario.odd import registry as odd_registry
from autoware_carla_scenario.odd.cli import main as odd_main
from autoware_carla_scenario.odd.openodd import _missing
from autoware_carla_scenario.odd.route import MISSING_BUCKET, UNDECIDED_BUCKET
from autoware_carla_scenario.scenario_runner import ScenarioRunner
from autoware_carla_scenario.sweeper.constraints import create_routing_graph
from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

DATA = Path(__file__).resolve().parents[3] / "data"
OSM_PATH = DATA / "nishishinjuku.osm"
XODR_PATH = DATA / "nishishinjuku_carla.xodr"

#: The left turn of intersection_passing/left_turn: 203 (60 km/h), the
#: junction lanelet 411 (50 km/h), then 207 (50 km/h).
LEFT_TURN = PlannedRoute(
    start=Lanelet2Pose(lanelet_id=203, s=25.0),
    goal=Lanelet2Pose(lanelet_id=207, s=10.0),
    name="left_turn",
)


@pytest.fixture(scope="module")
def nishishinjuku() -> tuple[Any, Any]:
    lanelet_map = load_lanelet2_map(OSM_PATH, XODR_PATH)
    return lanelet_map, create_routing_graph(lanelet_map)


def _cover(odd: Any, route: Any, nishishinjuku: tuple[Any, Any]) -> Any:
    lanelet_map, graph = nishishinjuku
    return plan_route_coverage(odd, route, lanelet_map=lanelet_map, routing_graph=graph)


def _length(nishishinjuku: tuple[Any, Any], lanelet_id: int) -> float:
    return float(lanelet2.geometry.length2d(nishishinjuku[0].laneletLayer[lanelet_id]))


def _map_odd(
    *modules: OddModule, extra: tuple[OddAttribute, ...] = ()
) -> OddDefinition:
    """The default ODD's map attributes, plus *extra*, under *modules*."""
    attributes = [a for a in default_odd().attributes if a.name.startswith("scenery.")]
    return OddDefinition("t", [*attributes, *extra], list(modules))


def _attribute(odd: OddDefinition, name: str) -> OddAttribute:
    return next(a for a in odd.attributes if a.name == name)


# ---------------------------------------------------------------------------
# Probes on a lanelet
# ---------------------------------------------------------------------------


def _fake_lanelet(**tags: str) -> Any:
    return SimpleNamespace(id=1, attributes=dict(tags))


class _Graph:
    def __init__(self, besides: int) -> None:
        self.count = besides

    def besides(self, lanelet: Any) -> list[Any]:
        return [lanelet] * self.count


class TestOnLanelet:
    def test_location_and_subtype_are_the_tags(self) -> None:
        lanelet = _fake_lanelet(location="nonurban", subtype="highway")
        assert probes.lanelet_location.on_lanelet(lanelet, None, None) == "nonurban"  # type: ignore[attr-defined]
        assert probes.lanelet_subtype.on_lanelet(lanelet, None, None) == "highway"  # type: ignore[attr-defined]

    def test_an_untagged_lanelet_reads_nothing(self) -> None:
        lanelet = _fake_lanelet()
        assert probes.lanelet_location.on_lanelet(lanelet, None, None) is None  # type: ignore[attr-defined]
        assert probes.lanelet_subtype.on_lanelet(lanelet, None, None) is None  # type: ignore[attr-defined]
        assert probes.lanelet_speed_limit_kph.on_lanelet(lanelet, None, None) is None  # type: ignore[attr-defined]

    def test_the_speed_limit_is_the_tag(self) -> None:
        lanelet = _fake_lanelet(speed_limit="40")
        assert probes.speed_limit_kph.on_lanelet(lanelet, None, None) == 40.0  # type: ignore[attr-defined]
        assert probes.lanelet_speed_limit_kph.on_lanelet(lanelet, None, None) == 40.0  # type: ignore[attr-defined]

    def test_without_the_tag_the_run_reads_carlas_limit(self) -> None:
        # speed_limit_kph falls back to CARLA at run time: the map cannot tell.
        on = probes.speed_limit_kph.on_lanelet  # type: ignore[attr-defined]
        assert on(_fake_lanelet(), None, None) is UNDECIDED

    def test_a_speed_limit_that_is_not_a_number_is_missing(self) -> None:
        lanelet = _fake_lanelet(speed_limit="fast")
        assert probes.speed_limit_kph.on_lanelet(lanelet, None, None) is None  # type: ignore[attr-defined]

    def test_a_junction_lanelet_has_a_turn_direction(self) -> None:
        on = probes.in_junction.on_lanelet  # type: ignore[attr-defined]
        assert on(_fake_lanelet(turn_direction="left"), None, None) is True
        assert on(_fake_lanelet(), None, None) is False

    def test_lane_count_is_the_lanelets_beside_and_nothing_in_a_junction(self) -> None:
        on = probes.lane_count.on_lanelet  # type: ignore[attr-defined]
        assert on(_fake_lanelet(), None, _Graph(3)) == 3
        assert on(_fake_lanelet(turn_direction="straight"), None, _Graph(3)) is None

    def test_on_the_fixture_map(self, nishishinjuku: tuple[Any, Any]) -> None:
        lanelet_map, graph = nishishinjuku
        straight, junction = (
            lanelet_map.laneletLayer[183],
            lanelet_map.laneletLayer[411],
        )

        def read(probe: Any, lanelet: Any) -> Any:
            return probe.on_lanelet(lanelet, lanelet_map, graph)

        assert read(probes.lanelet_location, straight) == "urban"
        assert read(probes.lanelet_subtype, straight) == "road"
        assert read(probes.speed_limit_kph, straight) == 60.0
        assert read(probes.in_junction, straight) is False
        assert read(probes.lane_count, straight) == 3  # 182, 183, 184
        assert read(probes.in_junction, junction) is True
        assert read(probes.lane_count, junction) is None

    def test_what_only_the_run_knows_has_no_lanelet_form(self) -> None:
        for probe in (
            probes.ego_speed_kph,
            probes.rain,
            probes.fog,
            probes.illumination,
            probes.traffic_density,
            probes.pedestrian_nearby,
        ):
            assert not hasattr(probe, "on_lanelet"), probe

    def test_a_concept_nothing_measures_is_missing_on_a_route_too(self) -> None:
        assert _missing.on_lanelet(_fake_lanelet(), None, None) is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class TestPlanRoute:
    def test_the_shortest_path_from_start_to_goal(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        planned = plan_route(LEFT_TURN, *nishishinjuku)
        assert [i for i, _ in planned] == [203, 411, 207]
        metres = dict(planned)
        assert metres[203] == pytest.approx(_length(nishishinjuku, 203) - 25.0)
        assert metres[411] == pytest.approx(_length(nishishinjuku, 411))
        assert metres[207] == pytest.approx(10.0)

    def test_through_the_waypoints_in_order(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        detour = PlannedRoute(
            start=LEFT_TURN.start,
            goal=LEFT_TURN.goal,
            via=(Lanelet2Pose(lanelet_id=415, s=0.0),),
        )
        ids = [i for i, _ in plan_route(detour, *nishishinjuku)]
        assert ids[:2] == [203, 415]  # straight on, not the left turn
        assert ids[-1] == 207
        assert len(ids) == len(set(ids))

    def test_a_waypoint_on_the_start_lanelet_adds_nothing(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        route = PlannedRoute(
            start=LEFT_TURN.start,
            goal=LEFT_TURN.goal,
            via=(Lanelet2Pose(lanelet_id=203, s=0.0), Lanelet2Pose(411, 0.0)),
        )
        assert [i for i, _ in plan_route(route, *nishishinjuku)] == [203, 411, 207]

    def test_a_lane_change_splits_the_length(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        lanelet_map, graph = nishishinjuku
        after = graph.following(lanelet_map.laneletLayer[184])[0].id
        route = PlannedRoute(
            start=Lanelet2Pose(lanelet_id=183, s=0.0),
            goal=Lanelet2Pose(lanelet_id=after, s=5.0),
        )
        planned = plan_route(route, lanelet_map, graph)
        assert [i for i, _ in planned] == [183, 184, after]
        metres = dict(planned)
        assert metres[183] == pytest.approx(_length(nishishinjuku, 183) / 2)
        assert metres[184] == pytest.approx(_length(nishishinjuku, 184) / 2)

    def test_no_goal(self, nishishinjuku: tuple[Any, Any]) -> None:
        with pytest.raises(RouteError, match="no goal"):
            plan_route(PlannedRoute(start=LEFT_TURN.start), *nishishinjuku)

    def test_a_lanelet_the_map_does_not_have(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        route = PlannedRoute(start=LEFT_TURN.start, goal=Lanelet2Pose(999999999, 0.0))
        with pytest.raises(RouteError, match="999999999"):
            plan_route(route, *nishishinjuku)

    def test_no_path(self, nishishinjuku: tuple[Any, Any]) -> None:
        # 411 turns into 207 and nowhere else: 203, before it, is behind it.
        route = PlannedRoute(start=Lanelet2Pose(411, 0.0), goal=Lanelet2Pose(203, 0.0))
        lanelet_map, graph = nishishinjuku
        if graph.shortestPath(
            lanelet_map.laneletLayer[411], lanelet_map.laneletLayer[203]
        ):
            pytest.skip("the fixture map routes round the block")
        with pytest.raises(RouteError, match="no route"):
            plan_route(route, lanelet_map, graph)

    def test_no_map_loaded(self) -> None:
        saved = MapManager._instance
        MapManager.reset()
        try:
            with pytest.raises(RouteError, match="no Lanelet2 map"):
                plan_route_coverage(default_odd(), LEFT_TURN)
        finally:
            MapManager._instance = saved

    def test_a_scenario_is_read_for_its_spawn_waypoints_and_goal(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        scenario = SimpleNamespace(
            ego_config=object(),
            _spawn_pose=LEFT_TURN.start,
            goal_pose=LEFT_TURN.goal,
            waypoint_poses=[],
        )
        coverage = _cover(default_odd(), scenario, nishishinjuku)
        assert coverage.lanelet_ids == [203, 411, 207]
        scenario._spawn_pose = None
        with pytest.raises(RouteError, match="spawn"):
            _cover(default_odd(), scenario, nishishinjuku)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_an_odd_without_modules_holds_everywhere(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        coverage = _cover(default_odd(), LEFT_TURN, nishishinjuku)
        assert coverage.inside_m == pytest.approx(coverage.length_m)
        assert not coverage.leaves_odd

    def test_a_lanelet_ruled_out_by_the_map_is_outside(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        odd = _map_odd()
        limit = _attribute(odd, "scenery.speed_limit")
        odd = _map_odd(OddModule("slow_roads", include_and=[limit.at_most(50)]))
        coverage = _cover(odd, LEFT_TURN, nishishinjuku)
        assert coverage.leaves_odd
        [outside] = coverage.outside()
        assert outside.lanelet_id == 203  # the 60 km/h approach
        assert outside.failing_modules == ["slow_roads"]
        assert coverage.outside_m == pytest.approx(outside.length_m)
        assert coverage.inside_m == pytest.approx(coverage.length_m - outside.length_m)

    def test_what_the_map_cannot_tell_leaves_it_undecided(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        rain = OddAttribute("environment.rain", probes.rain, values=["none", "heavy"])
        odd = _map_odd(extra=(rain,))
        location = _attribute(odd, "scenery.location")
        odd = _map_odd(
            OddModule(
                "dry_city",
                include_and=[location.is_in(["urban"]), rain.is_in(["none"])],
            ),
            extra=(rain,),
        )
        coverage = _cover(odd, LEFT_TURN, nishishinjuku)
        assert not coverage.leaves_odd
        assert coverage.undecided_m == pytest.approx(coverage.length_m)
        assert all(
            ll.values["environment.rain"] is UNDECIDED for ll in coverage.lanelets
        )
        assert coverage.describe()["lanelets"][0]["values"]["environment.rain"] == (
            "undecided"
        )

    def test_an_open_attribute_does_not_save_a_lanelet_the_map_rules_out(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        rain = OddAttribute("environment.rain", probes.rain, values=["none", "heavy"])
        odd = _map_odd(extra=(rain,))
        location = _attribute(odd, "scenery.location")
        odd = _map_odd(
            OddModule(
                "dry_countryside",
                include_and=[location.is_in(["nonurban"]), rain.is_in(["none"])],
            ),
            extra=(rain,),
        )
        coverage = _cover(odd, LEFT_TURN, nishishinjuku)
        assert [ll.verdict for ll in coverage.lanelets] == ["outside"] * 3

    def test_a_derived_attribute_follows_its_sources(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        odd = _map_odd()
        limit = _attribute(odd, "scenery.speed_limit")
        rain = OddAttribute("environment.rain", probes.rain, values=["none", "heavy"])

        class Derived:
            def __init__(
                self, rules: list[tuple[str, Any]], sources: list[Any]
            ) -> None:
                self.rules = rules
                self.sources = sources

            def from_values(self, values: dict[str, Any]) -> Any:
                for literal, condition in self.rules:
                    verdict = condition.evaluate(values, {})
                    if verdict is None:
                        return None
                    if verdict:
                        return literal
                return None

            def __call__(self, world: Any) -> Any:
                return None

        fast = OddAttribute(
            "road_class",
            Derived(
                [("fast", limit.at_least(60)), ("slow", limit.less_than(60))], [limit]
            ),
            values=["fast", "slow"],
        )
        wet = OddAttribute(
            "wet",
            Derived(
                [("wet", rain.is_in(["heavy"])), ("dry", rain.is_in(["none"]))], [rain]
            ),
            values=["wet", "dry"],
        )
        odd = OddDefinition(
            "t",
            [*odd.attributes, rain, fast, wet],
            [OddModule("no_fast_roads", exclude_or=[fast.equals("fast")])],
        )
        coverage = _cover(odd, LEFT_TURN, nishishinjuku)
        assert [ll.values["road_class"] for ll in coverage.lanelets] == [
            "fast",
            "slow",
            "slow",
        ]
        assert all(ll.values["wet"] is UNDECIDED for ll in coverage.lanelets)
        assert [ll.verdict for ll in coverage.lanelets] == [
            "outside",
            "inside",
            "inside",
        ]
        assert "road_class" in coverage.map_attributes
        assert "wet" not in coverage.map_attributes

    def test_a_goal_at_the_start_of_an_outside_lanelet_does_not_leave(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        odd = _map_odd()
        junction = _attribute(odd, "scenery.junction")
        odd = _map_odd(OddModule("no_junctions", include_and=[junction.equals(False)]))
        touching = PlannedRoute(start=LEFT_TURN.start, goal=Lanelet2Pose(411, 0.0))
        coverage = _cover(odd, touching, nishishinjuku)
        assert coverage.lanelets[-1].verdict == "outside"
        assert not coverage.leaves_odd
        driving = _cover(odd, LEFT_TURN, nishishinjuku)
        assert [ll.lanelet_id for ll in driving.outside()] == [411]

    def test_a_probe_that_raises_on_a_lanelet_reads_nothing(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        def broken(world: Any) -> Any:
            return None

        def boom(lanelet: Any, lanelet_map: Any, graph: Any) -> Any:
            raise RuntimeError("boom")

        broken.on_lanelet = boom  # type: ignore[attr-defined]
        x = OddAttribute("x", broken, values=[1, 2])
        odd = OddDefinition(
            "t", [x], [OddModule("needs_x", exclude_or=[x.is_unknown()])]
        )
        coverage = _cover(odd, [203], nishishinjuku)
        assert coverage.lanelets[0].values["x"] is None
        assert coverage.leaves_odd  # missing, and the ODD requires it
        assert coverage.expected_m["x"] == {
            MISSING_BUCKET: pytest.approx(_length(nishishinjuku, 203))
        }


# ---------------------------------------------------------------------------
# Expected coverage
# ---------------------------------------------------------------------------


class TestExpectedCoverage:
    def test_metres_per_bucket(self, nishishinjuku: tuple[Any, Any]) -> None:
        coverage = _cover(default_odd(), LEFT_TURN, nishishinjuku)
        metres = dict(plan_route(LEFT_TURN, *nishishinjuku))
        speed = coverage.expected_m["scenery.speed_limit"]
        assert speed == {
            "[50, 60)": pytest.approx(metres[411] + metres[207]),
            "[60, 80)": pytest.approx(metres[203]),
        }
        assert list(speed) == ["[50, 60)", "[60, 80)"]  # in bucket order
        assert coverage.expected_m["scenery.junction"] == {
            "false": pytest.approx(metres[203] + metres[207]),
            "true": pytest.approx(metres[411]),
        }
        # No lane count inside a junction, as at run time.
        assert coverage.expected_m["scenery.lane_count"][
            MISSING_BUCKET
        ] == pytest.approx(metres[411])
        # What only the run can tell is not expected in any bucket.
        assert coverage.expected_m["environment.rain"] == {
            UNDECIDED_BUCKET: pytest.approx(coverage.length_m)
        }

    def test_a_route_given_as_lanelets_is_taken_whole(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        coverage = _cover(default_odd(), [203, 411], nishishinjuku)
        assert coverage.length_m == pytest.approx(
            _length(nishishinjuku, 203) + _length(nishishinjuku, 411)
        )

    def test_a_batch_names_the_buckets_no_route_reaches(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        odd = _map_odd()
        location = _attribute(odd, "scenery.location")
        odd = _map_odd(
            OddModule("no_private", exclude_or=[location.is_in(["private"])])
        )
        routes = [
            _cover(odd, LEFT_TURN, nishishinjuku),
            _cover(odd, [183, 187], nishishinjuku),
        ]
        summary = combine_route_coverage(odd, routes)
        assert summary.expected_m["scenery.location"] == {
            "urban": pytest.approx(sum(r.length_m for r in routes))
        }
        # "private" is outside the ODD: not a target, so not unreached.
        assert summary.unreached["scenery.location"] == ["nonurban"]
        assert "[60, 80)" not in summary.unreached["scenery.speed_limit"]
        assert "[0, 30)" in summary.unreached["scenery.speed_limit"]
        assert summary.leaving == []
        json.dumps(summary.describe())

    def test_attributes_only_the_run_reads_are_never_unreached(
        self, nishishinjuku: tuple[Any, Any]
    ) -> None:
        summary = combine_route_coverage(
            default_odd(), [_cover(default_odd(), LEFT_TURN, nishishinjuku)]
        )
        assert "environment.rain" not in summary.unreached
        assert "dynamic.ego_speed" not in summary.unreached


# ---------------------------------------------------------------------------
# scenario-odd route
# ---------------------------------------------------------------------------


def _speed_odd() -> OddDefinition:
    odd = _map_odd()
    limit = _attribute(odd, "scenery.speed_limit")
    return _map_odd(OddModule("slow_roads", include_and=[limit.at_most(50)]))


class TestCli:
    @pytest.fixture(autouse=True)
    def _fixture_map(
        self, nishishinjuku: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> Iterator[None]:
        # The map is loaded once for the module, not once per command.
        import autoware_carla_scenario.sweeper.constraints as constraints
        import autoware_carla_scenario.sweeper.map_loader as map_loader

        monkeypatch.setattr(map_loader, "load_map", lambda paths: nishishinjuku[0])
        monkeypatch.setattr(
            constraints, "create_routing_graph", lambda m: nishishinjuku[1]
        )
        register_odd("route_test_slow_roads", _speed_odd)
        yield
        odd_registry._REGISTRY.pop("route_test_slow_roads", None)

    MAP = [f"map.lanelet2_path={OSM_PATH}", f"map.xodr_path={XODR_PATH}"]

    def test_routes_inside_the_odd(self, capsys: Any) -> None:
        assert odd_main(["route", "default", "lane_change/left", *self.MAP]) == 0
        out = capsys.readouterr().out
        assert "lane_change/left: lanelets 183 -> 187 -> 348 -> 141" in out
        assert "unreached scenery.location: nonurban, private" in out

    def test_a_route_leaving_the_odd(self, capsys: Any) -> None:
        argv = [
            "route",
            "route_test_slow_roads",
            "scenario=lane_change/left",
            *self.MAP,
        ]
        assert odd_main(argv) == 1
        assert "OUTSIDE lanelet 183" in capsys.readouterr().out

    def test_json(self, capsys: Any) -> None:
        argv = [
            "route",
            "default",
            "intersection_passing/left_turn",
            "--json",
            *self.MAP,
        ]
        assert odd_main(argv) == 0
        out = json.loads(capsys.readouterr().out)
        # The goal intersection_passing derives from its expected route.
        assert out["routes"][0]["lanelet_ids"] == [203, 411, 207]
        assert out["not_planned"] == {}

    def test_a_route_that_cannot_be_planned(self, capsys: Any) -> None:
        argv = [
            "route",
            "default",
            "pedestrian_dart_out/pedestrian_dart_out",
            *self.MAP,
        ]
        assert odd_main(argv) == 2
        assert "NOT PLANNED: no goal" in capsys.readouterr().out

    def test_leaving_outranks_not_planned(self, capsys: Any) -> None:
        argv = [
            "route",
            "route_test_slow_roads",
            "lane_change/left",
            "pedestrian_dart_out/pedestrian_dart_out",
            *self.MAP,
        ]
        assert odd_main(argv) == 1

    def test_an_odd_that_cannot_be_read(self, capsys: Any) -> None:
        assert odd_main(["route", "no_such_odd", "lane_change/left"]) == 2
        assert "no_such_odd" in capsys.readouterr().err

    def test_a_glob_matching_nothing(self, capsys: Any) -> None:
        assert odd_main(["route", "default", "no_such_scenario/*"]) == 2


# ---------------------------------------------------------------------------
# The runner's warning
# ---------------------------------------------------------------------------


class TestRunnerWarning:
    def _runner(self, odd: OddDefinition) -> Any:
        return SimpleNamespace(odd=odd)

    def _scenario(self) -> Any:
        return SimpleNamespace(
            ego_config=object(),
            _spawn_pose=LEFT_TURN.start,
            goal_pose=LEFT_TURN.goal,
            waypoint_poses=[],
        )

    def test_it_warns_when_the_route_leaves(
        self, nishishinjuku: tuple[Any, Any], caplog: Any
    ) -> None:
        saved = MapManager._instance
        MapManager.reset()
        manager = MapManager.get_instance()
        manager._lanelet_map, manager._routing_graph = nishishinjuku
        try:
            with caplog.at_level(logging.WARNING):
                ScenarioRunner._warn_if_route_leaves_odd(
                    self._runner(_speed_odd()), "s", self._scenario()
                )
        finally:
            MapManager._instance = saved
        assert "leaves ODD t" in caplog.text
        assert "203" in caplog.text

    def test_it_never_fails_the_run(self, caplog: Any) -> None:
        saved = MapManager._instance
        MapManager.reset()  # no map: the route cannot be planned
        try:
            with caplog.at_level(logging.WARNING):
                ScenarioRunner._warn_if_route_leaves_odd(
                    self._runner(_speed_odd()), "s", self._scenario()
                )
                ScenarioRunner._warn_if_route_leaves_odd(
                    SimpleNamespace(),  # type: ignore[arg-type]
                    "s",
                    self._scenario(),
                )
        finally:
            MapManager._instance = saved
        assert caplog.text == ""
