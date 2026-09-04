"""Scenario IR: parsing, layout separation, validation and compilation.

The load-bearing property under test is that ``ui`` is presentation only --
deleting or reshuffling it must never change what a scenario does.
"""

from __future__ import annotations

import pytest
import yaml

from autoware_carla_scenario.authoring.compiler import (
    EGO_ROLE,
    CompilationError,
    coerce_params,
    compile_document,
)
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    ConditionNode,
    ConstraintNode,
    Entity,
    ScenarioDocument,
    SpawnSpec,
)
from autoware_carla_scenario.authoring.persistence import (
    DraftStore,
    dump_document_yaml,
    load_document,
    save_document,
)
from autoware_carla_scenario.authoring.registry import FieldSpec, SelectOption
from autoware_carla_scenario.authoring.starter import blank_document, new_document
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.constants import EGO_ROLE_NAME


class TestStarterDocuments:
    def test_cut_in_example_is_valid(self) -> None:
        report = validate_document(new_document())
        assert report.ok, [f"{i.path}: {i.message}" for i in report.errors]

    def test_blank_document_is_valid(self) -> None:
        report = validate_document(blank_document())
        assert report.ok, [f"{i.path}: {i.message}" for i in report.errors]

    def test_cut_in_example_exercises_every_spawn_feature(self) -> None:
        """The starter is the tutorial; if it stops showing a feature, say so."""
        document = new_document()
        npc = document.entity("npc1")
        assert npc is not None
        assert npc.spawn.mode == "constraint_search"
        assert npc.spawn.constraints
        assert npc.spawn.s.mode == "derived"
        assert npc.spawn.s.binding is not None
        assert any(a.trigger is not None for a in document.actions)


class TestRoundTrip:
    def test_yaml_round_trip_is_lossless(self, tmp_path) -> None:
        document = new_document()
        path = save_document(document, tmp_path / "document.yaml")
        assert load_document(path).to_yaml_dict() == document.to_yaml_dict()

    def test_assertions_serialise_under_pass_and_fail_keys(self) -> None:
        raw = yaml.safe_load(dump_document_yaml(new_document()))
        assert set(raw["assertions"]) == {"pass", "fail"}

    def test_draft_wrapper_is_accepted_by_load_document(self, tmp_path) -> None:
        store = DraftStore(tmp_path)
        draft = store.create(new_document())
        assert load_document(store.path_for(draft.id)).id == "cut_in"

    def test_draft_ids_cannot_escape_the_store(self, tmp_path) -> None:
        store = DraftStore(tmp_path)
        with pytest.raises(ValueError):
            store.path_for("../../etc/passwd")


class TestLayoutIsPresentationOnly:
    def test_dropping_the_ui_block_does_not_change_compilation(self) -> None:
        document = new_document()
        with_layout = compile_document(document)

        stripped = ScenarioDocument.model_validate(document.to_yaml_dict())
        stripped.ui.actor_order = []
        stripped.ui.nodes = {}
        without_layout = compile_document(stripped)

        assert [a.node.id for a in with_layout.actions] == [
            a.node.id for a in without_layout.actions
        ]
        assert with_layout.roles == without_layout.roles

    def test_column_hints_only_reorder_the_lane(self) -> None:
        document = new_document()
        action = document.actions[0]
        document.ui.set_column(action.id, 7)
        assert document.actions_for("npc1")[0].id == action.id
        assert compile_document(document).actions[0].node.id == action.id

    def test_action_slots_leave_a_gap_where_a_step_is_empty(self) -> None:
        """A reaction has to be placeable after a cause on another track.

        The ego's own track is empty in between, so the gap has to survive into
        the rendered lane rather than being packed away.
        """
        document = new_document()
        action = document.actions[0]
        document.ui.set_column(action.id, 2)
        slots = document.action_slots("npc1")
        assert slots == [[], [], [action]]

    def test_a_step_holds_every_action_placed_in_it(self) -> None:
        """A step is a set, not a slot.

        Both are armed from the first tick and neither waits on the other, so
        they really do run alongside each other; spreading them across two
        columns would draw an order the runtime does not have.
        """
        document = new_document()
        first = document.actions[0]
        second = ActionNode(id="a_second", type="lane_change", actor="npc1")
        document.actions.append(second)
        document.ui.set_column(first.id, 1)
        document.ui.set_column(second.id, 1)
        slots = document.action_slots("npc1")
        assert [[a.id for a in slot] for slot in slots] == [[], [first.id, second.id]]

    def test_a_dependent_action_is_pushed_past_what_it_waits_on(self) -> None:
        """Nothing inside a step is ordered, so a dependency cannot share one."""
        document = new_document()
        cut_in = document.actions[0]
        reaction = ActionNode(
            id="a_reaction",
            type="lane_change",
            actor="ego",
            trigger=ConditionNode(
                type="action_state",
                params={"action": cut_in.id, "state": "completeState"},
            ),
        )
        document.actions.append(reaction)
        document.ui.set_column(cut_in.id, 0)
        document.ui.set_column(reaction.id, 0)

        document.enforce_dependency_order()
        assert document.ui.column_of(reaction.id) > document.ui.column_of(cut_in.id)

    def test_moving_a_dependency_carries_what_waits_on_it(self) -> None:
        document = new_document()
        cut_in = document.actions[0]
        reaction = ActionNode(
            id="a_reaction",
            type="lane_change",
            actor="ego",
            trigger=ConditionNode(
                type="action_state",
                params={"action": cut_in.id, "state": "completeState"},
            ),
        )
        document.actions.append(reaction)
        document.ui.set_column(cut_in.id, 4)
        document.ui.set_column(reaction.id, 1)

        document.enforce_dependency_order()
        assert document.ui.column_of(reaction.id) == 5

    def test_a_cycle_leaves_the_layout_alone(self) -> None:
        """Two actions waiting on each other can never fire, in any layout.

        Pushing them apart forever is not a repair, so the validator is left to
        report it.
        """
        document = new_document()
        first = document.actions[0]
        second = ActionNode(
            id="a_second",
            type="lane_change",
            actor="ego",
            trigger=ConditionNode(
                type="action_state",
                params={"action": first.id, "state": "completeState"},
            ),
        )
        first.trigger = ConditionNode(
            type="action_state",
            params={"action": second.id, "state": "completeState"},
        )
        document.actions.append(second)
        document.ui.set_column(first.id, 0)
        document.ui.set_column(second.id, 0)

        document.enforce_dependency_order()
        assert document.ui.column_of(first.id) == 0
        assert document.ui.column_of(second.id) == 0

    def test_sync_layout_drops_stale_entries(self) -> None:
        document = new_document()
        document.ui.actor_order.append("ghost")
        document.ui.set_column("ghost_action", 3)
        document.sync_layout()
        assert "ghost" not in document.ui.actor_order
        assert "ghost_action" not in document.ui.nodes


class TestValidation:
    def test_missing_ego_is_an_error(self) -> None:
        document = new_document()
        document.entities = [e for e in document.entities if e.kind != "ego"]
        report = validate_document(document)
        assert any("ego" in i.message for i in report.errors)

    def test_two_egos_is_an_error(self) -> None:
        document = new_document()
        document.entities.append(Entity(id="ego2", kind="ego"))
        assert any(
            "Only one ego" in i.message for i in validate_document(document).errors
        )

    def test_unknown_condition_type_is_an_error(self) -> None:
        document = new_document()
        document.assertions.pass_conditions = [ConditionNode(type="nope")]
        assert any(
            "Unknown condition type" in i.message
            for i in validate_document(document).errors
        )

    def test_condition_referencing_a_missing_entity_is_an_error(self) -> None:
        document = new_document()
        action = document.actions[0]
        assert action.trigger is not None
        action.trigger.children[0].params["target"] = "ghost"
        assert any(
            "unknown entity" in i.message for i in validate_document(document).errors
        )

    def test_composition_arity_is_enforced(self) -> None:
        document = new_document()
        document.assertions.pass_conditions = [ConditionNode(type="all", children=[])]
        assert any(
            "at least 2" in i.message for i in validate_document(document).errors
        )

    def test_no_pass_condition_is_an_error(self) -> None:
        document = new_document()
        document.assertions.pass_conditions = []
        assert any(
            "PASS condition" in i.message for i in validate_document(document).errors
        )

    def test_a_childless_composition_is_an_error(self) -> None:
        document = new_document()
        npc = document.entity("npc1")
        assert npc is not None
        npc.spawn.constraints = [ConstraintNode(type="not")]
        assert any(
            "at least one child constraint" in i.message
            for i in validate_document(document).errors
        )

    def test_an_overfull_wrapper_is_an_error(self) -> None:
        """``not`` takes one child, so a second one is rejected rather than dropped."""
        document = new_document()
        npc = document.entity("npc1")
        assert npc is not None
        npc.spawn.constraints = [
            ConstraintNode(
                type="not",
                constraints=[
                    ConstraintNode(type="is_junction"),
                    ConstraintNode(type="has_stop_line"),
                ],
            )
        ]
        assert any(
            "at most 1 child constraint" in i.message
            for i in validate_document(document).errors
        )

    def test_a_leaf_constraint_takes_no_children(self) -> None:
        document = new_document()
        npc = document.entity("npc1")
        assert npc is not None
        npc.spawn.constraints = [
            ConstraintNode(
                type="is_junction",
                constraints=[ConstraintNode(type="has_stop_line")],
            )
        ]
        assert any(
            "does not take child constraints" in i.message
            for i in validate_document(document).errors
        )

    def test_a_wrapper_serialises_its_child_the_way_the_sweeper_reads_it(self) -> None:
        """One child list in the IR; the sweeper's singular ``constraint`` key out."""
        node = ConstraintNode(
            type="not", constraints=[ConstraintNode(type="is_junction")]
        )
        assert node.to_sweep_dict() == {
            "type": "not",
            "constraint": {"type": "is_junction"},
        }

    def test_a_second_constraint_search_warns_about_the_sweeper_limit(self) -> None:
        """The sweeper enumerates one target key; the rest keep their defaults."""
        document = new_document()
        ego = document.ego
        assert ego is not None
        ego.spawn = SpawnSpec(
            mode="constraint_search",
            lanelet_id=42,
            constraints=[ConstraintNode(type="is_junction")],
        )
        report = validate_document(document)
        assert report.ok
        assert any("searches one entity" in i.message for i in report.warnings)


class TestCompilation:
    def test_roles_are_assigned_ego_first_then_numbered_npcs(self) -> None:
        compiled = compile_document(new_document())
        assert compiled.roles == {"ego": EGO_ROLE, "npc1": "npc1"}

    def test_ego_role_matches_the_framework_constant(self) -> None:
        assert EGO_ROLE == str(EGO_ROLE_NAME)

    def test_entity_references_are_resolved_to_role_names(self) -> None:
        compiled = compile_document(new_document())
        trigger = compiled.actions[0].trigger
        assert trigger is not None
        distance = trigger.children[0]
        assert distance.params["source"] == "npc1"
        assert distance.params["target"] == EGO_ROLE

    def test_invalid_documents_do_not_compile(self) -> None:
        document = new_document()
        document.assertions.pass_conditions = []
        with pytest.raises(CompilationError) as excinfo:
            compile_document(document)
        assert excinfo.value.issues

    def test_warnings_survive_compilation(self) -> None:
        document = new_document()
        document.assertions.fail_conditions = []
        assert compile_document(document).warnings


class TestParameterCoercion:
    """Form posts arrive as strings; constructors need real types."""

    def test_strings_become_declared_types(self) -> None:
        fields = (
            FieldSpec("count", "Count", "int", 0),
            FieldSpec("ratio", "Ratio", "number", 0.0),
            FieldSpec("flag", "Flag", "bool", False),
            FieldSpec("ids", "IDs", "int_list", []),
        )
        assert coerce_params(
            fields, {"count": "7", "ratio": "1.5", "flag": "on", "ids": ["1", "2"]}
        ) == {"count": 7, "ratio": 1.5, "flag": True, "ids": [1, 2]}

    def test_blank_values_fall_back_to_the_default(self) -> None:
        fields = (
            FieldSpec(
                "rule", "Rule", "select", "less_than", (SelectOption("less_than", "<"),)
            ),
        )
        assert coerce_params(fields, {"rule": "  "}) == {"rule": "less_than"}

    def test_unknown_keys_are_dropped(self) -> None:
        """A stale key must never reach a runtime constructor as a keyword."""
        fields = (FieldSpec("kept", "Kept", "text", ""),)
        assert coerce_params(fields, {"kept": "a", "stale": "b"}) == {"kept": "a"}

    def test_interpolations_pass_through_untouched(self) -> None:
        fields = (FieldSpec("values", "Values", "int_list_or_ref", []),)
        result = coerce_params(fields, {"values": "${map.no_3d_model_lanelet_ids}"})
        assert result["values"] == "${map.no_3d_model_lanelet_ids}"
