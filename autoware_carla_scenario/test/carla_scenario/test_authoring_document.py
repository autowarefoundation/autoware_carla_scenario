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
from autoware_carla_scenario.authoring.registry import (
    FieldSpec,
    SelectOption,
    action_specs,
)
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

    def test_the_cut_in_npc_does_not_start_in_the_lane_it_must_reach(self) -> None:
        """Without a sweep the fallback spawn is what runs.

        Starting NPC1 on the lanelet the PASS assertion names would satisfy the
        verdict before the lane change ran -- a reported successful cut-in that
        never happened.
        """
        document = new_document()
        target = document.assertions.pass_conditions[0].children[0].params["lanelet_id"]
        npc, ego = document.entity("npc1"), document.entity("ego")
        assert npc is not None and ego is not None
        assert npc.spawn.lanelet_id != target
        assert ego.spawn.lanelet_id == target

    def test_every_lane_numbers_its_steps_the_same_way(self) -> None:
        """A step number must mean the same column on every track.

        Actor tracks used to carry a leading spawn column the World and verdict
        lanes did not, so "step 3" pointed at different columns depending on
        which row you read.
        """
        document = new_document()
        widest = max(
            [len(document.action_slots(e.id)) for e in document.ordered_entities()]
            + [
                len(document.action_slots(None)),
                len(document.assertions.pass_conditions),
                len(document.assertions.fail_conditions),
            ]
        )
        # One trailing column for the "add" control, and nothing else.
        assert document.step_count() == widest + 1

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


class TestEgoGoal:
    """The goal is the ego's, and it is stored beside the ego's spawn."""

    @staticmethod
    def _with_goal(lanelet_id: int = 265, s: float = 12.5):
        from autoware_carla_scenario.authoring.models import GoalSpec

        document = new_document()
        ego = document.ego
        assert ego is not None
        ego.goal = GoalSpec(lanelet_id=lanelet_id, s=s)
        return document

    def test_an_ego_with_a_goal_is_valid(self) -> None:
        assert validate_document(self._with_goal()).ok

    def test_an_ego_needs_no_goal(self) -> None:
        # Only an ego that plans its own route reads one; the document does not
        # choose which stack drives, so a missing goal is not a finding.
        assert validate_document(new_document()).ok

    def test_a_goal_on_another_vehicle_is_an_error(self) -> None:
        from autoware_carla_scenario.authoring.models import GoalSpec

        document = new_document()
        npc = document.entity("npc1")
        assert npc is not None
        npc.goal = GoalSpec(lanelet_id=265)

        report = validate_document(document)

        assert not report.ok
        assert any("Only the ego" in issue.message for issue in report.errors)

    def test_a_goal_without_a_lanelet_is_an_error(self) -> None:
        report = validate_document(self._with_goal(lanelet_id=0))
        assert not report.ok
        assert any("positive lanelet ID" in issue.message for issue in report.errors)

    def test_an_init_set_goal_card_for_the_ego_warns(self) -> None:
        """Two goals delivered in one phase is one thing said twice."""
        from autoware_carla_scenario.authoring.models import ActionNode

        document = self._with_goal()
        document.actions.append(
            ActionNode(
                type="routing",
                title="Set Goal",
                actor="ego",
                phase="init",
                params={"goal_lanelet_id": 300, "goal_s": 0.0},
            )
        )

        report = validate_document(document)

        assert report.ok  # a warning, not an error: the run is well defined
        assert any("already delivered" in issue.message for issue in report.warnings)

    def test_the_same_card_on_the_tick_loop_is_fine(self) -> None:
        # Changing the destination mid-run is what the card is for.
        from autoware_carla_scenario.authoring.models import ActionNode

        document = self._with_goal()
        document.actions.append(
            ActionNode(
                type="routing",
                title="Set Goal",
                actor="ego",
                phase="pre_tick",
                params={"goal_lanelet_id": 300, "goal_s": 0.0},
            )
        )

        assert not any(
            "already delivered" in issue.message
            for issue in validate_document(document).warnings
        )

    def test_the_goal_survives_a_yaml_round_trip(self, tmp_path) -> None:
        path = save_document(self._with_goal(), tmp_path / "document.yaml")
        ego = load_document(path).ego

        assert ego is not None
        assert ego.goal is not None
        assert (ego.goal.lanelet_id, ego.goal.s) == (265, 12.5)


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


class TestEnvironmentTrack:
    """Which track an action is drawn in is the action type's business.

    A traffic light is set by the world, not by a car, and nothing in the build
    reads ``actor`` for one. Honouring the field anyway put the card in some
    vehicle's lane and claimed a relationship the run does not have.
    """

    @staticmethod
    def _document() -> ScenarioDocument:
        document = ScenarioDocument(
            title="t",
            id="t",
            entities=[
                Entity(id="ego", kind="ego", spawn=SpawnSpec(lanelet_id=1)),
            ],
            actions=[
                ActionNode(
                    id="swerve",
                    type="lane_change",
                    title="Swerve",
                    actor="ego",
                    params={"direction": "right"},
                ),
                ActionNode(
                    id="lights",
                    type="traffic_signal",
                    title="Lights",
                    actor="ego",  # a lie the editor used to let you tell
                    params={"state": "red", "target": "all"},
                ),
            ],
        )
        document.sync_layout()
        return document

    def test_an_environment_action_leaves_the_actor_track(self) -> None:
        document = self._document()
        assert [a.id for a in document.actions_for("ego")] == ["swerve"]
        assert [a.id for a in document.environment_actions()] == ["lights"]

    def test_a_stale_actor_is_dropped_rather_than_kept(self) -> None:
        """So the saved YAML stops naming a vehicle that does nothing."""
        document = self._document()
        lights = document.action("lights")
        assert lights is not None
        assert lights.actor is None

    def test_an_action_with_no_actor_yet_stays_reachable(self) -> None:
        """It is invalid, and a card nothing draws cannot be corrected."""
        document = self._document()
        document.actions.append(
            ActionNode(id="orphan", type="lane_change", title="?", actor=None)
        )
        document.sync_layout()
        assert "orphan" in [a.id for a in document.environment_actions()]

    def test_the_scope_is_what_actor_required_means(self) -> None:
        """One source of truth: the two cannot drift apart."""
        for spec in action_specs():
            assert spec.actor_required == (spec.scope == "actor")


class TestAddingToTheInitPhase:
    """A card added to the init cell stays in the init cell.

    Reported from the editor: adding a second action to init made it vanish.
    It had not vanished -- the add control's phase was dropped on the way in,
    so the card was created on the tick loop and drawn out on the step track,
    where it then warned about firing on the first tick.
    """

    @staticmethod
    def _document():
        from autoware_carla_scenario.authoring.models import Entity, ScenarioDocument

        document = ScenarioDocument(id="s", title="s")
        document.entities.append(Entity(id="npc1", kind="vehicle"))
        return document

    def test_the_requested_phase_is_honoured(self):
        from autoware_carla_scenario.editor.service import EditorService

        document = self._document()
        service = EditorService.__new__(EditorService)

        first = service.add_action(document, "lane_change", "npc1", "init")
        second = service.add_action(document, "lane_change", "npc1", "init")

        assert [first.phase, second.phase] == ["init", "init"]
        assert [a.id for a in document.init_actions("npc1")] == [first.id, second.id]

    def test_init_actions_take_no_step(self):
        from autoware_carla_scenario.editor.service import EditorService

        document = self._document()
        service = EditorService.__new__(EditorService)

        service.add_action(document, "lane_change", "npc1", "init")
        stepped = service.add_action(document, "lane_change", "npc1", "pre_tick")
        document.sync_layout()

        # The run's own first card is step 1, not step 2.
        assert document.ui.column_of(stepped.id) == 0
        assert document.action_slots("npc1") == [[stepped]]

    def test_an_init_action_is_not_warned_about_firing_immediately(self):
        from autoware_carla_scenario.authoring.validator import validate_document
        from autoware_carla_scenario.editor.service import EditorService

        document = self._document()
        service = EditorService.__new__(EditorService)
        service.add_action(document, "lane_change", "npc1", "init")
        service.add_action(document, "lane_change", "npc1", "init")
        document.sync_layout()

        report = validate_document(document)
        assert not [
            w for w in report.warnings if "fires on the first tick" in w.message
        ]


class TestInitTakesNoCondition:
    """The init phase has nothing to wait for, so it offers nowhere to wait.

    It is one step that begins at once and happens once: `run_init` evaluates a
    trigger a single time, at elapsed 0.0, so an action whose trigger was not
    already true would silently never run.
    """

    @staticmethod
    def _document_with_an_init_action():
        from autoware_carla_scenario.authoring.models import Entity, ScenarioDocument
        from autoware_carla_scenario.editor.service import EditorService

        document = ScenarioDocument(id="s", title="s")
        document.entities.append(Entity(id="npc1", kind="vehicle"))
        service = EditorService.__new__(EditorService)
        action = service.add_action(document, "lane_change", "npc1", "init")
        return document, service, action

    def test_a_condition_cannot_be_attached_to_one(self):
        import pytest

        from autoware_carla_scenario.editor.service import EditorError

        document, service, action = self._document_with_an_init_action()

        with pytest.raises(EditorError, match="initialization phase"):
            service.add_condition(document, f"trigger:{action.id}", "elapsed_time")

        assert action.trigger is None

    def test_an_action_that_waits_cannot_move_into_init(self):
        import pytest

        from autoware_carla_scenario.editor.service import EditorError

        document, service, action = self._document_with_an_init_action()
        action.phase = "pre_tick"
        service.add_condition(document, f"trigger:{action.id}", "elapsed_time")

        with pytest.raises(EditorError, match="Remove the trigger first"):
            service.update_action(document, action.id, {"phase": "init"})

        # Refused, not silently repaired by deleting what the author wrote.
        assert action.phase == "pre_tick"
        assert action.trigger is not None

    def test_a_hand_edited_document_is_rejected(self):
        from autoware_carla_scenario.authoring.models import ConditionNode
        from autoware_carla_scenario.authoring.validator import validate_document

        document, _, action = self._document_with_an_init_action()
        action.trigger = ConditionNode(type="elapsed_time", params={"duration": 1.0})

        report = validate_document(document)
        assert [i for i in report.errors if "initialization phase" in i.message]
