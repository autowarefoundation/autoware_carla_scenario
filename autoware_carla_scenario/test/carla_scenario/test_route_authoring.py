"""A logical scenario as a document: the route search, route vertices, progress.

What a transpiler writes -- the route search, ``route_vertices`` on a *Follow
Trajectory* card, ``route_progress`` triggers and waypoint conditions,
``appear_on_start`` -- is read, validated, round-tripped, compiled and built
into the framework's own runtime objects here; the editor shows and edits the
route search; and :class:`DeclarativeScenario` picks a route on the loaded map.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typesafe_carla.carla as carla
import yaml
from fastapi.testclient import TestClient

from autoware_carla_scenario import (
    EgoConfig,
    FollowTrajectoryAction,
    Lanelet2Pose,
    RouteCrossingPose,
    RouteLanePose,
    RouteProgressCondition,
    SpawnTransform,
    TurnAction,
)
from autoware_carla_scenario.authoring.builders import (
    instantiate_action,
    instantiate_condition,
)
from autoware_carla_scenario.authoring.compiler import BuildContext, compile_document
from autoware_carla_scenario.authoring.hydra_config import build_scenario_config
from autoware_carla_scenario.authoring.models import (
    JunctionSegment,
    LaneSegment,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.persistence import DraftStore, dump_yaml
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.declarative import (
    DeclarativeScenario,
    DeclarativeScenarioConfig,
)
from autoware_carla_scenario.editor.app import create_app
from autoware_carla_scenario.editor.service import EditorError, parse_route_yaml
from autoware_carla_scenario.route import (
    RouteMatch,
    clear_scenario_route,
    parse_route_search,
    scenario_route,
)
from autoware_carla_scenario.route.model import LaneSegmentSpec
from autoware_carla_scenario.route.search import find_route_matches

from . import _route_maps as rm

#: The document a transpiler writes for "the ego turns left at a signalised
#: junction; an oncoming car comes through it; a pedestrian waits at the
#: crosswalk on the way out" -- every new feature once.
LOGICAL = """
id: logical_left_turn
title: Left turn with an oncoming car
timeout_seconds: 60
route:
  segments:
    - kind: lane
      length: {min: 30, max: 60}
      lanes_left: {max: 0}
      opposite_lane: yes
      shape: straight
    - kind: junction
      turn: left
      traffic_light: yes
      crossing_from_opposite: yes
      crosswalk_exit: any
    - kind: lane
      length: {max: 40}
  ego_spawn_s: 5.0
  ego_goal: true
  ego_goal_margin: 2.0
  match_index: 0
  max_matches: 16
entities:
  - {id: ego, kind: ego, driven_by: autopilot}
  - id: oncoming
    kind: vehicle
    spawn: {hidden: true}
  - id: walker
    kind: pedestrian
    spawn: {hidden: true}
actions:
  - id: oncoming_drive
    type: follow_trajectory
    actor: oncoming
    trigger:
      type: route_progress
      params: {value: -40, anchor: "junction:0:entry"}
    params:
      path_source: route
      following_mode: follow
      time_domain: none
      speed_ms: 8.0
      appear_on_start: true
      route_vertices:
        - {kind: crossing, junction: 0, approach: opposite, turn: straight, distance: -30}
        - {kind: crossing, junction: 0, approach: opposite, turn: straight, distance: 0}
        - {kind: crossing, junction: 0, approach: opposite, turn: straight, distance: 25}
    advance_conditions:
      - vertex: 2
        condition:
          type: route_progress
          params: {value: -5, anchor: "junction:0:entry"}
  - id: walker_cross
    type: follow_trajectory
    actor: walker
    trigger:
      type: route_progress
      params: {value: 0, anchor: "junction:0:entry"}
    params:
      path_source: route
      following_mode: follow
      time_domain: none
      speed_ms: 1.3
      appear_on_start: true
      route_vertices: |
        crosswalk junction=0 leg=exit side=right along=-1.5
        crosswalk junction=0 leg=exit side=right along=6
  - id: parked
    type: follow_trajectory
    actor: ego
    phase: post_tick
    trigger:
      type: route_progress
      params: {value: 1000}
    params:
      path_source: route
      time_domain: none
      route_vertices:
        - {kind: lane, ds: 0}
        - {kind: opposite, ds: 10, lane: 1, anchor: "segment:0:start"}
        - {kind: roadside, ds: 20, side: left, kerb_distance: 1.0, anchor: start}
assertions:
  pass:
    - type: route_progress
      params: {value: -1.0, anchor: end}
  fail:
    - type: timeout
      params: {timeout_seconds: 60}
"""


def _approx_pose(
    lanelet_id: int, s: float, t: float = 0.0, heading: float = 0.0
) -> Any:
    """A Lanelet2 pose compared with its s approximately."""
    return Lanelet2Pose(lanelet_id, pytest.approx(s), t, heading)  # type: ignore[arg-type]


def _document(text: str = LOGICAL) -> ScenarioDocument:
    return ScenarioDocument.model_validate(yaml.safe_load(text))


def _raw(text: str = LOGICAL) -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(text))


def _messages(document: ScenarioDocument, severity: str = "error") -> list[str]:
    return [
        i.message for i in validate_document(document).issues if i.severity == severity
    ]


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


class TestTheDocument:
    def test_it_is_valid(self) -> None:
        report = validate_document(_document())
        assert report.ok, report.errors

    def test_it_round_trips(self) -> None:
        document = _document()
        again = ScenarioDocument.model_validate(
            yaml.safe_load(dump_yaml(document.to_yaml_dict()))
        )
        assert again.to_yaml_dict() == document.to_yaml_dict()
        assert again.route is not None
        lane, junction, _out = again.route.segments
        assert isinstance(lane, LaneSegment) and lane.opposite_lane == "yes"
        assert isinstance(junction, JunctionSegment)

    def test_a_document_without_a_route_says_nothing_of_one(self) -> None:
        assert "route" not in ScenarioDocument().to_yaml_dict()

    def test_a_typo_in_a_segment_is_refused(self) -> None:
        raw = _raw()
        raw["route"]["segments"][1]["lenght"] = {"max": 3}
        with pytest.raises(ValueError, match="lenght"):
            ScenarioDocument.model_validate(raw)

    def test_the_route_search_reads_back_as_the_search(self) -> None:
        document = _document()
        assert document.route is not None
        spec = parse_route_search(document.route.to_sweep_dict())
        assert spec.junction_count == 1
        lane = spec.segments[0]
        assert isinstance(lane, LaneSegmentSpec) and lane.lanes_left.max == 0
        assert (spec.ego_spawn_s, spec.ego_goal_margin, spec.max_matches) == (
            5.0,
            2.0,
            16,
        )


class TestValidation:
    def test_route_vertices_and_progress_need_a_route(self) -> None:
        raw = _raw()
        del raw["route"]
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("Route vertices are placed" in m for m in messages)
        assert any("Route progress is measured" in m for m in messages)

    def test_an_anchor_or_junction_the_pattern_lacks(self) -> None:
        raw = _raw()
        raw["actions"][0]["trigger"]["params"]["anchor"] = "junction:1:entry"
        raw["actions"][0]["params"]["route_vertices"][0]["junction"] = 2
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("has 1 junction(s)" in m for m in messages)
        assert any("Route vertex 1: junction 2" in m for m in messages)

    def test_a_malformed_vertex_is_reported_against_its_field(self) -> None:
        raw = _raw()
        raw["actions"][1]["params"]["route_vertices"] = "crosswalk leg=sideways\nlane"
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("Route vertices: vertex 1" in m and "leg" in m for m in messages)

    def test_a_route_search_the_search_refuses(self) -> None:
        raw = _raw()
        raw["route"]["segments"][0]["length"] = {"min": 90, "max": 60}
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("above its max" in m for m in messages)
        raw["route"]["segments"] = []
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("at least one segment" in m for m in messages)

    def test_a_route_and_a_constraint_search_together(self) -> None:
        raw = _raw()
        raw["entities"][1]["spawn"] = {
            "mode": "constraint_search",
            "lanelet_id": 5,
            "constraints": [{"type": "is_junction"}],
        }
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("route search already decides" in m for m in messages)

    def test_two_goals(self) -> None:
        raw = _raw()
        raw["entities"][0]["goal"] = {"lanelet_id": 4}
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("route.ego_goal to false" in m for m in messages)
        raw["route"]["ego_goal"] = False
        assert _messages(ScenarioDocument.model_validate(raw)) == []

    def test_an_autoware_ego_is_given_the_routes_goal(self) -> None:
        raw = _raw()
        raw["entities"][0]["driven_by"] = "autoware"
        assert _messages(ScenarioDocument.model_validate(raw)) == []
        raw["route"]["ego_goal"] = False
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("Autoware ego has no goal" in m for m in messages)

    def test_a_visible_npc_on_a_fixed_lanelet_is_one_maps(self) -> None:
        raw = _raw()
        raw["entities"][1]["spawn"] = {"lanelet_id": 5}
        warnings = _messages(ScenarioDocument.model_validate(raw), "warning")
        assert any("names a lanelet of one map" in m for m in warnings)
        assert any("not spawned out of the world" in m for m in warnings)

    def test_appear_on_start_and_hiding_outside_together(self) -> None:
        raw = _raw()
        raw["actions"][0]["params"]["hidden_outside_trajectory"] = True
        messages = _messages(ScenarioDocument.model_validate(raw))
        assert any("choose one" in m for m in messages)

    def test_a_hidden_entity_brought_in_by_appearing_is_not_stranded(self) -> None:
        warnings = _messages(_document(), "warning")
        assert not any("stays there for the run" in m for m in warnings)


# ---------------------------------------------------------------------------
# Compiled and built
# ---------------------------------------------------------------------------


class TestBuild:
    def test_the_cards_build_framework_objects(self) -> None:
        compiled = compile_document(_document())
        ctx = BuildContext(scenario=None)
        oncoming = instantiate_action(compiled.actions[0], ctx)
        assert isinstance(oncoming, FollowTrajectoryAction)
        assert oncoming._appear_on_start is True
        assert oncoming._following_mode.value == "follow"
        vertices = oncoming.trajectory.vertices
        assert vertices[0].position == RouteCrossingPose(
            junction=0, approach="opposite", distance=-30.0, turn="straight"
        )
        assert isinstance(vertices[1].advance, RouteProgressCondition)
        assert oncoming.trajectory.is_relative
        trigger = oncoming._condition
        assert isinstance(trigger, RouteProgressCondition)
        assert trigger.get_details()["anchor"] == "junction:0:entry"
        assert trigger.get_details()["entity_name"] == "Ego"

        parked = instantiate_action(compiled.actions[2], ctx)
        assert isinstance(parked, FollowTrajectoryAction)
        assert parked.trajectory.vertices[0].position == RouteLanePose()

    def test_the_assertion_builds(self) -> None:
        compiled = compile_document(_document())
        condition = instantiate_condition(
            compiled.pass_conditions[0], BuildContext(scenario=None)
        )
        assert isinstance(condition, RouteProgressCondition)
        assert condition.get_details()["value"] == -1.0

    def test_the_hydra_config(self) -> None:
        config = build_scenario_config(_document())
        assert config["sweep"] == {"route": _document().route.to_sweep_dict()}  # type: ignore[union-attr]
        assert config["scenario"]["route"]["lanelet_ids"] == []
        assert "constraints" not in config["sweep"]
        # The keys the expansion writes are declared, so struct mode takes them.
        written = RouteMatch((1,), 0.0, 1.0, ()).to_config()
        assert set(config["scenario"]["route"]) == set(written)
        assert "goal_lanelet_id" not in config["ego"]

    def test_the_package_config_class_takes_the_route_keys(self) -> None:
        config = build_scenario_config(_document())
        scenario = {k: v for k, v in config["scenario"].items() if k != "name"}
        DeclarativeScenarioConfig(name="x", **scenario)


# ---------------------------------------------------------------------------
# The run picks its route
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def road() -> tuple[Any, Any]:
    lanelet_map = rm.crossroads()
    return lanelet_map, rm.routing_graph(lanelet_map)


@pytest.fixture
def loaded(road: tuple[Any, Any]) -> Iterator[None]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._lanelet_map, manager._routing_graph = road
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    yield
    clear_scenario_route()
    MapManager._instance = saved


def _scenario(
    document: ScenarioDocument, route: dict[str, Any] | None = None
) -> DeclarativeScenario:
    return DeclarativeScenario(
        EgoConfig(
            spawn_location=SpawnTransform(
                carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
            )
        ),
        spawn_pose=Lanelet2Pose(lanelet_id=0, s=0.0),
        config=DeclarativeScenarioConfig(route=route or {}),
        document=document,
    )


class TestTheRunPicksItsRoute:
    def test_unexpanded_it_searches_the_map(self, loaded: None) -> None:
        scenario = _scenario(_document())
        scenario._prepare_route()
        match = scenario_route()
        assert match is not None
        # The one signalised left turn of the crossroads.
        assert match.lanelet_ids == (rm.W_IN1, rm.W_IN2, rm.J[("W", "N")], rm.N_OUT)
        assert scenario._spawn_pose == _approx_pose(rm.W_IN1, 45.0)
        assert scenario.goal_pose == _approx_pose(rm.N_OUT, 38.0)
        # An ego the TrafficManager drives is handed the route as waypoints,
        # and turned at no junction: the route it follows goes through it.
        assert [p.lanelet_id for p in scenario.waypoint_poses] == [
            rm.W_IN2,
            rm.J[("W", "N")],
        ]
        assert not any(isinstance(a, TurnAction) for a in scenario._pre_tick_actions)

    def test_without_a_goal_the_ego_is_turned_at_each_junction(
        self, loaded: None
    ) -> None:
        """No goal, no RoutingAction: the route has nothing to be handed with."""
        raw = _raw()
        raw["route"]["ego_goal"] = False
        scenario = _scenario(ScenarioDocument.model_validate(raw))
        scenario._prepare_route()
        assert scenario.goal_pose is None
        assert scenario.waypoint_poses == []
        turns = [a for a in scenario._pre_tick_actions if isinstance(a, TurnAction)]
        assert [t.label for t in turns] == ["route_turn_0"]
        assert isinstance(turns[0]._condition, RouteProgressCondition)

    def test_expanded_it_takes_the_match_it_was_given(
        self, loaded: None, road: tuple[Any, Any]
    ) -> None:
        document = _document()
        assert document.route is not None
        (match,) = find_route_matches(
            parse_route_search(document.route.to_sweep_dict()), *road
        )
        scenario = _scenario(document, match.to_config())
        scenario._prepare_route()
        given = scenario_route()
        assert given is not None and given.lanelet_ids == match.lanelet_ids
        # The spawn and the goal came through the ego's own keys.
        assert scenario._spawn_pose == Lanelet2Pose(0, 0.0)

    def test_an_autoware_ego_gets_the_route_as_waypoints(
        self, loaded: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            DeclarativeScenario, "ego_requires_goal", property(lambda self: True)
        )
        scenario = _scenario(_document())
        scenario._prepare_route()
        assert [p.lanelet_id for p in scenario.waypoint_poses] == [
            rm.W_IN2,
            rm.J[("W", "N")],
        ]
        assert not any(isinstance(a, TurnAction) for a in scenario._pre_tick_actions)

    def test_a_map_without_the_route(self, loaded: None) -> None:
        raw = _raw()
        raw["route"]["segments"][1]["crosswalk_exit"] = "yes"
        with pytest.raises(ValueError, match="no route of this map matches"):
            _scenario(ScenarioDocument.model_validate(raw))._prepare_route()

    def test_a_match_index_past_the_matches(self, loaded: None) -> None:
        raw = _raw()
        raw["route"]["match_index"] = 3
        with pytest.raises(ValueError, match="has 1 matching route"):
            _scenario(ScenarioDocument.model_validate(raw))._prepare_route()

    def test_a_document_without_a_route_clears_the_last_one(self, loaded: None) -> None:
        _scenario(_document())._prepare_route()
        assert scenario_route() is not None
        raw = _raw()
        del raw["route"]
        for action in raw["actions"]:
            action["params"]["path_source"] = "vertices"
            action["params"]["vertices"] = [[0, 0], [1, 0]]
            action.pop("trigger")
            action.pop("advance_conditions", None)
        raw["assertions"]["pass"] = [
            {"type": "elapsed_time", "params": {"duration_seconds": 5}}
        ]
        for entity in raw["entities"]:
            entity.setdefault("spawn", {})["lanelet_id"] = rm.W_IN1
        _scenario(ScenarioDocument.model_validate(raw))._prepare_route()
        assert scenario_route() is None


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(draft_dir=tmp_path / "drafts", export_dir=tmp_path / "packages")
    )


@pytest.fixture
def draft_id(client: TestClient) -> str:
    response = client.post(
        "/new", data={"kind": "cut_in", "title": "Cut in"}, follow_redirects=False
    )
    return response.headers["location"].rsplit("/", 1)[-1]


class TestTheEditor:
    def test_the_route_search_is_edited_as_yaml_and_shown_as_a_list(
        self, client: TestClient, draft_id: str, tmp_path: Path
    ) -> None:
        route = dump_yaml(_document().route.to_sweep_dict())  # type: ignore[union-attr]
        client.post(f"/draft/{draft_id}/scenario", data={"route_yaml": route})
        stored = DraftStore(tmp_path / "drafts").get(draft_id)
        assert stored is not None and stored.document.route is not None
        junction = stored.document.route.segments[1]
        assert isinstance(junction, JunctionSegment) and junction.turn == "left"
        body = client.get(f"/draft/{draft_id}/inspector/scenario").text
        assert "lane 30-60 m, lanes left &lt;= 0, straight, opposite lane: yes" in body
        assert "junction, turn left, traffic light: yes" in body
        assert 'name="route_yaml"' in body

    def test_an_empty_box_removes_it(
        self, client: TestClient, draft_id: str, tmp_path: Path
    ) -> None:
        client.post(
            f"/draft/{draft_id}/scenario",
            data={"route_yaml": "segments: [{kind: junction}]"},
        )
        client.post(f"/draft/{draft_id}/scenario", data={"route_yaml": "  "})
        stored = DraftStore(tmp_path / "drafts").get(draft_id)
        assert stored is not None and stored.document.route is None

    def test_a_bad_search_is_refused_with_the_reason(self) -> None:
        with pytest.raises(EditorError, match="Route search"):
            parse_route_yaml("segments: [{kind: roundabout}]")
        with pytest.raises(EditorError, match="not YAML"):
            parse_route_yaml("segments: [")

    def test_route_vertices_and_route_progress_are_offered(
        self, client: TestClient, draft_id: str
    ) -> None:
        body = client.get(f"/draft/{draft_id}").text
        assert "Route progress" in body or "route_progress" in body


def test_the_documented_example_is_a_valid_document() -> None:
    """docs/logical_scenarios.md's full example is the transpiler's reference."""
    docs = Path(__file__).resolve().parents[2] / "docs" / "logical_scenarios.md"
    text = docs.read_text(encoding="utf-8")
    block = text.split("```yaml\n", 1)[1].split("```", 1)[0]
    document = ScenarioDocument.model_validate(yaml.safe_load(block))
    report = validate_document(document)
    assert report.ok, report.errors
    assert not report.warnings, report.warnings
    compile_document(document)


def test_a_match_index_the_search_never_returns() -> None:
    raw = _raw()
    raw["route"]["max_matches"] = 2
    raw["route"]["match_index"] = 2
    messages = _messages(ScenarioDocument.model_validate(raw))
    assert any("can never be taken" in m for m in messages)


class _OpenRoad:
    """A CARLA map with no junction: every point of a route is kept."""

    def get_waypoint(self, location: Any) -> None:
        return None


class _OpenWorld:
    def get_map(self) -> _OpenRoad:
        return _OpenRoad()


class _RouteTM:
    def __init__(self) -> None:
        self.paths: list[tuple[Any, list[Any]]] = []

    def set_path(self, actor: Any, path: list[Any]) -> None:
        self.paths.append((actor, path))


class _RouteClient:
    def __init__(self, tm: _RouteTM) -> None:
        self._tm = tm

    def get_trafficmanager(self, port: int) -> _RouteTM:
        return self._tm


class TestABackendDrivenEgoFollowsTheRoute:
    """``BackendDriven.route_to`` hands the ego's traffic backend the whole route."""

    @staticmethod
    def _ego() -> tuple[Any, _RouteTM]:
        from autoware_carla_scenario.entity.ego import EgoVehicle

        ego = EgoVehicle()
        ego._vehicle = SimpleNamespace(id=1)  # type: ignore[assignment]
        tm = _RouteTM()
        ego.set_client(_RouteClient(tm))  # type: ignore[arg-type]
        return ego, tm

    def test_through_the_waypoints_to_the_goal(self, loaded: None) -> None:
        from autoware_carla_scenario.coordinate.transform import (
            to_carla_location,
            to_carla_world,
        )
        from autoware_carla_scenario.traffic.traffic_manager import ROUTE_POINT_STEP_M

        ego, tm = self._ego()
        start = Lanelet2Pose(lanelet_id=rm.W_IN1, s=45.0)
        goal = Lanelet2Pose(lanelet_id=rm.N_OUT, s=38.0)

        ego.route_to(
            _OpenWorld(),  # type: ignore[arg-type]
            goal,
            initial_pose=to_carla_world(start),
            waypoints=[
                Lanelet2Pose(rm.W_IN2, 0.0),
                Lanelet2Pose(rm.J[("W", "N")], 0.0),
            ],
        )

        ((_actor, path),) = tm.paths
        # A step ahead of where it stands: a point behind it would never be
        # reached, and the TrafficManager would wait on it.
        ahead = to_carla_location(Lanelet2Pose(rm.W_IN1, 45.0 + ROUTE_POINT_STEP_M))
        assert path[0].distance(ahead) < 1e-6
        assert path[-1].distance(to_carla_location(goal)) < 1e-6
        steps = [a.distance(b) for a, b in zip(path, path[1:])]
        assert max(steps) <= ROUTE_POINT_STEP_M + 1e-6
        # The left turn: it ends on the N road, never having gone east or south.
        assert abs(path[-1].x) < 3.5 and -path[-1].y > 10.0
        assert all(p.x < 12.0 and -p.y > -12.0 for p in path)

    def test_a_route_that_cannot_be_planned_sends_nothing(
        self, loaded: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario.coordinate.transform import to_carla_world

        ego, tm = self._ego()
        with caplog.at_level("WARNING"):
            ego.route_to(
                _OpenWorld(),  # type: ignore[arg-type]
                Lanelet2Pose(lanelet_id=rm.ONE_WAY1, s=0.0),
                initial_pose=to_carla_world(Lanelet2Pose(rm.W_IN1, 45.0)),
            )
        assert tm.paths == []
        assert "no route to follow" in caplog.text


class _StandingActor:
    """An NPC's actor where the route starts, read when no initial pose is given."""

    def __init__(self, transform: Any) -> None:
        self._transform = transform
        self.id = 2

    def get_transform(self) -> Any:
        return self._transform


class TestAnNpcVehicleFollowsTheRoute:
    """The same route_to, on an NPC the traffic backend drives."""

    @staticmethod
    def _npc() -> tuple[Any, _RouteTM]:
        from autoware_carla_scenario.coordinate.transform import to_carla_world
        from autoware_carla_scenario.entity import (
            SpawnPointIndex,
            VehicleEntity,
            VehicleEntityConfig,
        )

        npc = VehicleEntity(
            VehicleEntityConfig(role_name="npc1", spawn_location=SpawnPointIndex(0))
        )
        npc._vehicle = _StandingActor(  # type: ignore[assignment]
            to_carla_world(Lanelet2Pose(rm.W_IN1, 45.0)).to_carla_transform()
        )
        tm = _RouteTM()
        npc.set_client(_RouteClient(tm))  # type: ignore[arg-type]
        return npc, tm

    def test_from_where_its_actor_is_through_the_waypoints(self, loaded: None) -> None:
        from autoware_carla_scenario.coordinate.transform import (
            to_carla_location,
        )

        npc, tm = self._npc()
        goal = Lanelet2Pose(lanelet_id=rm.N_OUT, s=38.0)

        npc.route_to(
            _OpenWorld(),  # type: ignore[arg-type]
            goal,
            waypoints=[Lanelet2Pose(rm.J[("W", "N")], 0.0)],
        )

        from autoware_carla_scenario.traffic.traffic_manager import ROUTE_POINT_STEP_M

        ((actor, path),) = tm.paths
        assert actor is npc.actor
        # Measured from where its actor is, exactly, not from a centreline vertex.
        ahead = Lanelet2Pose(rm.W_IN1, 45.0 + ROUTE_POINT_STEP_M)
        assert path[0].distance(to_carla_location(ahead)) < 0.05
        assert path[-1].distance(to_carla_location(goal)) < 1e-6
        assert abs(path[-1].x) < 3.5 and -path[-1].y > 10.0

    def test_a_routing_action_on_the_npc_reaches_it(
        self, loaded: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What a document's Set Goal card on an NPC builds, executed."""
        from autoware_carla_scenario.actions.routing import RoutingAction
        from autoware_carla_scenario.entity import registry

        npc, tm = self._npc()
        monkeypatch.setattr(registry, "_VEHICLES", {})
        monkeypatch.setattr(registry, "_PEDESTRIANS", {})
        registry.register_entity("npc1", npc)

        RoutingAction(
            "npc1",
            Lanelet2Pose(lanelet_id=rm.N_OUT, s=38.0),
            waypoints=[Lanelet2Pose(rm.J[("W", "N")], 0.0)],
        ).execute(_OpenWorld())  # type: ignore[arg-type]

        assert len(tm.paths) == 1


class _StandingWalker:
    def __init__(self, transform: Any) -> None:
        self._transform = transform

    def get_transform(self) -> Any:
        return self._transform


class _WalkingWalker:
    """A walker actor that stands where it is put and records its controls."""

    def __init__(self, transform: Any) -> None:
        self._transform = transform
        self.controls: list[Any] = []

    def get_transform(self) -> Any:
        return self._transform

    def apply_control(self, control: Any) -> None:
        self.controls.append(control)


class TestAPedestrianWalksTheRoute:
    """PedestrianEntity.route_to lays the walk out; walk_on walks it."""

    @staticmethod
    def _walker(at: Lanelet2Pose) -> Any:
        from autoware_carla_scenario.coordinate.transform import to_carla_world
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
        walker._walker = _WalkingWalker(  # type: ignore[assignment]
            to_carla_world(at).to_carla_transform()
        )
        return walker

    def test_with_no_walkable_way_a_leg_is_straight(self, loaded: None) -> None:
        """The crosswalks of the crossroads join nothing: the leg is straight."""
        from autoware_carla_scenario.coordinate.transform import to_carla_world

        start = Lanelet2Pose(rm.CROSSWALK_W, 2.0)
        goal = Lanelet2Pose(lanelet_id=rm.CROSSWALK_E, s=10.0)
        walker = self._walker(start)
        walker.route_to(_OpenWorld(), goal)

        walk = walker._walk
        here, there = to_carla_world(start), to_carla_world(goal)
        assert (walk.xs[0], walk.ys[0]) == pytest.approx((here.x, here.y))
        assert (walk.xs[-1], walk.ys[-1]) == pytest.approx((there.x, there.y))
        assert walker.walking

    def test_through_a_waypoint(self, loaded: None) -> None:
        from autoware_carla_scenario.coordinate.transform import to_carla_world

        via = Lanelet2Pose(lanelet_id=rm.CROSSWALK_W, s=12.0)
        walker = self._walker(Lanelet2Pose(rm.CROSSWALK_W, 2.0))
        walker.route_to(
            _OpenWorld(), Lanelet2Pose(rm.CROSSWALK_E, 10.0), waypoints=[via]
        )

        at = to_carla_world(via)
        walk = walker._walk
        assert (
            min(math.hypot(x - at.x, y - at.y) for x, y in zip(walk.xs, walk.ys)) < 1e-6
        )

    def test_it_steers_along_the_walk_and_stops_on_arrival(self, loaded: None) -> None:
        walker = self._walker(Lanelet2Pose(rm.CROSSWALK_W, 2.0))
        walker.route_to(
            _OpenWorld(), Lanelet2Pose(rm.CROSSWALK_E, 10.0), walking_speed=1.2
        )
        walker.walk_on()
        (control,) = walker._walker.controls
        assert control.speed > 0.0 and walker.walking

        # Put it at the end of its walk: it stands still and is done.
        walk = walker._walk
        walker._walker._transform = carla.Transform(
            carla.Location(x=walk.xs[-1], y=walk.ys[-1], z=walk.zs[-1])
        )
        walker.walk_on()
        assert walker._walker.controls[-1].speed == 0.0
        assert not walker.walking

    def test_an_unspawned_walker_has_no_walk(self, loaded: None) -> None:
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
        walker.route_to(_OpenWorld(), Lanelet2Pose(rm.CROSSWALK_E, 10.0))
        assert not walker.walking


class TestTheRoutingActionWalks:
    def test_it_walks_until_the_walk_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.action_state import ActionState
        from autoware_carla_scenario.actions.routing import RoutingAction
        from autoware_carla_scenario.entity import registry
        from autoware_carla_scenario.entity.pedestrian_entity import PedestrianEntity

        calls: list[Any] = []

        class _Walker(PedestrianEntity):
            steps_left = 3

            @property
            def walking(self) -> bool:
                return self.steps_left > 0

            def route_to(self, world, goal, **kwargs):  # noqa: ANN001, ANN003
                calls.append(("route_to", kwargs["walking_speed"]))

            def walk_on(self) -> None:
                calls.append("walk_on")
                self.steps_left -= 1

        monkeypatch.setattr(registry, "_VEHICLES", {})
        monkeypatch.setattr(registry, "_PEDESTRIANS", {})
        registry.register_entity("walker1", _Walker.__new__(_Walker))

        action = RoutingAction("walker1", Lanelet2Pose(3, 7.0), walking_speed=0.9)
        assert action.needs_tick_loop
        action.tick(object(), 0.0)  # type: ignore[arg-type]
        assert calls == [("route_to", 0.9)]
        assert action.reissues_while_running

        for t in (0.05, 0.1, 0.15, 0.2, 0.25):
            action.tick(object(), t)  # type: ignore[arg-type]
        assert calls.count("walk_on") == 3
        assert action.state is ActionState.COMPLETE

    def test_a_walking_speed_must_be_positive(self) -> None:
        from autoware_carla_scenario.actions.routing import RoutingAction

        with pytest.raises(ValueError, match="walking_speed"):
            RoutingAction("walker1", Lanelet2Pose(3, 7.0), walking_speed=0.0)


class TestAnActionThatNeedsTheLoop:
    def test_registered_for_init_it_runs_on_the_loop(self) -> None:
        """A pedestrian routed from the start: walking needs the ticks."""
        from autoware_carla_scenario.actions import TurnAction
        from autoware_carla_scenario.scenario_base import BaseScenario
        from autoware_carla_scenario.traffic import TurnDirection

        class _Looping(TurnAction):
            @property
            def needs_tick_loop(self) -> bool:
                return True

        class _Scenario(BaseScenario):
            def setup(self) -> None: ...

            def is_done(self) -> bool:
                return False

        scenario = _Scenario.__new__(_Scenario)
        scenario._init_actions = []
        scenario._init_callbacks = []
        scenario._pre_tick_actions = []
        scenario._pre_tick_callbacks = []
        scenario._traffic_backend = None
        scenario._follows_tire_friction = True

        looping = _Looping("walker1", TurnDirection.LEFT)
        plain = TurnAction("walker1", TurnDirection.LEFT)
        scenario.register_init(looping)
        scenario.register_init(plain)
        assert scenario._pre_tick_actions == [looping]
        assert scenario._init_actions == [plain]
