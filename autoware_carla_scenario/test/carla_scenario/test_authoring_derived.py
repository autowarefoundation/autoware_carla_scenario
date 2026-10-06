"""Lanelets derived from the search, spawn offsets, and the lane-of condition."""

from __future__ import annotations

import math
from typing import Any

import pytest

from autoware_carla_scenario.authoring.hydra_config import build_scenario_config
from autoware_carla_scenario.authoring.models import (
    BindingRef,
    ConstraintNode,
    Entity,
    GoalSpec,
    LaneletChoice,
    ScenarioDocument,
    SpawnSpec,
)
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.validator import validate_document


def _derived(binding: str, **params: Any) -> dict[str, Any]:
    return LaneletChoice(
        mode="derived", binding=BindingRef(type=binding, params=params)
    ).model_dump()


def _with_ego_goal(document: ScenarioDocument, goal: GoalSpec) -> ScenarioDocument:
    ego = document.ego
    assert ego is not None
    ego.goal = goal
    return document


def _messages(document: ScenarioDocument) -> list[str]:
    return [issue.message for issue in validate_document(document).errors]


class TestValidation:
    def test_a_lanelet_derived_from_the_search_is_valid(self) -> None:
        document = _with_ego_goal(
            new_document(),
            GoalSpec(lanelet_id=141, **_derived("route_through", depth=1)),
        )
        assert _messages(document) == []

    def test_nothing_searched_leaves_nothing_to_derive_from(self) -> None:
        document = _with_ego_goal(
            new_document(), GoalSpec(lanelet_id=141, **_derived("matched"))
        )
        npc = document.entity("npc1")
        assert npc is not None
        npc.spawn.mode = "fixed"
        assert any("nothing is searched" in m for m in _messages(document))

    def test_an_offset_is_not_a_lanelet(self) -> None:
        document = _with_ego_goal(
            new_document(),
            GoalSpec(lanelet_id=141, **_derived("stop_line_offset", offset=10.0)),
        )
        assert any("not a lanelet" in m for m in _messages(document))

    def test_a_lanelet_takes_only_the_last_of_a_route(self) -> None:
        """A slot holds one id, so the route always gives its last lanelet."""
        binding = BindingRef(type="route_through", params={"depth": 2})
        assert binding.to_sweep_dict() == {
            "type": "route_through",
            "depth": 2,
            "last_only": True,
        }


class TestHydraConfig:
    def test_every_derived_lanelet_is_a_binding_on_its_own_key(self) -> None:
        document = _with_ego_goal(
            new_document(), GoalSpec(lanelet_id=141, **_derived("matched"))
        )
        ego = document.ego
        assert ego is not None
        ego.spawn = SpawnSpec(lanelet_id=183, **_derived("adjacent", side="right"))
        bindings = build_scenario_config(document)["sweep"]["bindings"]
        assert bindings["ego.goal_lanelet_id"] == {"type": "matched"}
        assert bindings["ego.spawn_lanelet_id"] == {"type": "adjacent", "side": "right"}

    def test_a_derived_parameter_has_its_key_declared(self) -> None:
        document = new_document()
        node = document.assertions.pass_conditions[0].children[0]
        node.searches["lanelet_id"] = LaneletChoice(**_derived("matched"))
        config = build_scenario_config(document)
        assert node.id in config["scenario"]["param_overrides"]
        key = f"scenario.param_overrides.{node.id}.lanelet_id"
        assert config["sweep"]["bindings"][key] == {"type": "matched"}


def test_an_equals_id_written_as_a_string_still_names_the_lanelet() -> None:
    """As the editor's text field hands it over, or a quoted YAML value."""
    from types import SimpleNamespace

    from autoware_carla_scenario.sweeper.constraints import parse_constraint

    node = ConstraintNode(type="equals", params={"value": "222"})
    constraint = parse_constraint(node.to_sweep_dict())
    assert constraint.evaluate(SimpleNamespace(id=222))
    assert not constraint.evaluate(SimpleNamespace(id=223))
    assert parse_constraint({"type": "equals", "value": "any"}).evaluate(
        SimpleNamespace(id=1)
    )


def test_a_pedestrian_defaults_to_a_walker_every_carla_has() -> None:
    assert Entity(id="p", kind="pedestrian").vehicle_type == "walker.pedestrian.0015"


def test_the_spawn_pose_carries_its_offsets() -> None:
    from autoware_carla_scenario.declarative import _spawn_pose

    entity = Entity(
        id="p",
        kind="pedestrian",
        spawn=SpawnSpec(lanelet_id=20, t=-3.0, heading=math.pi / 2),
    )
    entity.spawn.s.value = 40.0
    pose = _spawn_pose(entity)
    assert (pose.lanelet_id, pose.s, pose.t, pose.heading) == (
        20,
        40.0,
        -3.0,
        math.pi / 2,
    )


class TestSweeperBindings:
    def test_matched_is_the_pick(self) -> None:
        from autoware_carla_scenario.sweeper.bindings import parse_binding

        binding = parse_binding("x", {"type": "matched"})
        assert binding.resolve(183, lanelet_map=None).value == 183


class TestLaneOf:
    """Which lane the condition watches, relative to the position's."""

    @pytest.fixture(autouse=True)
    def _road_5_lane(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from autoware_carla_scenario.conditions.composition import (
            entity_lane_position,
        )
        from autoware_carla_scenario.coordinate import OpenDrivePose

        monkeypatch.setattr(
            entity_lane_position,
            "to_opendrive",
            lambda pose: (
                pose
                if isinstance(pose, OpenDrivePose)
                else OpenDrivePose(road_id="5", lane_id=-2, s=10.0)
            ),
        )

    @pytest.mark.parametrize(
        ("lane", "lane_id"),
        [("same", -2), ("left", -1), ("right", -3), ("any", None)],
    )
    def test_the_lane_follows_the_relation(self, lane: str, lane_id: Any) -> None:
        from autoware_carla_scenario.conditions import EntityLaneOfCondition
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        condition = EntityLaneOfCondition(
            "ego", Lanelet2Pose(lanelet_id=183, s=5.0), lane, label="x"
        )
        details = condition.get_details()
        assert (details["road_id"], details["lane_id"]) == ("5", lane_id)
        assert details["frame"] == "opendrive"

    def test_an_unknown_relation_is_refused(self) -> None:
        from autoware_carla_scenario.conditions import EntityLaneOfCondition
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        with pytest.raises(ValueError):
            EntityLaneOfCondition(
                "ego", Lanelet2Pose(lanelet_id=183, s=5.0), "behind", label="x"
            )

    def test_the_editor_offers_exactly_the_runtimes_relations(self) -> None:
        from autoware_carla_scenario.authoring.registry import get_condition_spec
        from autoware_carla_scenario.conditions.composition.entity_lane_position import (
            LANE_RELATIONS,
        )

        spec = get_condition_spec("entity_lane_of")
        assert spec is not None
        (lane,) = [f for f in spec.fields if f.name == "lane"]
        assert tuple(o.value for o in lane.options) == LANE_RELATIONS
