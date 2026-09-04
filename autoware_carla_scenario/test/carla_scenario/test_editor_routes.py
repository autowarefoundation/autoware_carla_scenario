"""Scenario Editor routes.

The editor keeps no client-side model: every edit is a form post that returns
re-rendered HTML. So the tests drive it the way a browser does -- post a form,
then assert on the stored document *and* on what came back -- which is also the
only way to catch a template that renders but shows the wrong thing.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import TypeVar

import pytest
import yaml
from fastapi.testclient import TestClient

from autoware_carla_scenario.authoring.models import (
    ActionNode,
    ConditionNode,
    Entity,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.persistence import Draft, DraftStore
from autoware_carla_scenario.editor.app import create_app
from autoware_carla_scenario.editor.service import EditorError, EditorService

T = TypeVar("T")


def _present(value: T | None, what: str) -> T:
    """Return *value*, failing the test when the editor dropped it."""
    assert value is not None, f"{what} is missing from the stored document"
    return value


def _draft(store: DraftStore, draft_id: str) -> Draft:
    """Return the stored draft, failing the test when it is gone."""
    return _present(store.get(draft_id), f"draft {draft_id}")


def _document(store: DraftStore, draft_id: str) -> ScenarioDocument:
    """Return the stored document for a draft."""
    return _draft(store, draft_id).document


def _entity(store: DraftStore, draft_id: str, entity_id: str) -> Entity:
    """Return a stored entity, failing the test when it is gone."""
    return _present(_document(store, draft_id).entity(entity_id), entity_id)


def _action(store: DraftStore, draft_id: str, action_id: str) -> ActionNode:
    """Return a stored action, failing the test when it is gone."""
    return _present(_document(store, draft_id).action(action_id), action_id)


def _condition(store: DraftStore, draft_id: str, node_id: str) -> ConditionNode:
    """Return a stored condition, failing the test when it is gone."""
    return _present(_document(store, draft_id).condition(node_id), node_id)


@pytest.fixture
def store(tmp_path: Path) -> DraftStore:
    return DraftStore(tmp_path / "drafts")


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
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[-1]


class TestPages:
    def test_index_lists_drafts(self, client: TestClient, draft_id: str) -> None:
        body = client.get("/").text
        assert draft_id in body
        assert "Cut in" in body

    def test_editor_page_renders_the_arrangement(
        self, client: TestClient, draft_id: str
    ) -> None:
        body = client.get(f"/draft/{draft_id}").text
        assert "Arrangement" in body
        assert "not elapsed time" in body
        # Actors, the triggered action, and both verdict lanes.
        for expected in ("Ego", "NPC1", "Cut in", "PASS", "FAIL"):
            assert expected in body, expected

    def test_the_step_ruler_is_as_long_as_the_busiest_track(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The ruler numbers every slot, including the trailing "add" one.

        The cut-in starter's longest track is NPC1: a spawn, one action and the
        slot its "+ action" control sits in.
        """
        body = client.get(f"/draft/{draft_id}").text
        steps = body.count('class="ed-ruler-step"')
        assert steps == 3, steps

    def test_a_condition_links_back_to_the_action_it_waits_on(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The canvas draws its causal link from the document's own reference.

        Without ``data-caused-by`` the reaction and its cause are two cards in
        neighbouring columns with nothing joining them, which reads as "these
        happen at the same time" rather than "this one causes that one".  The
        attribute must carry the referenced action id and nothing inferred from
        where the cards sit.
        """
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        cut_in = document["actions"][0]["id"]
        client.post(
            f"/draft/{draft_id}/action",
            data={"actor": "ego", "type_id": "lane_change"},
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        ego_action = document["actions"][-1]["id"]
        client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": f"trigger:{ego_action}", "type_id": "action_state"},
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        node = document["actions"][-1]["trigger"]["id"]
        client.post(
            f"/draft/{draft_id}/condition/{node}",
            data={"action": cut_in, "state": "completeState"},
        )

        body = client.get(f"/draft/{draft_id}").text
        assert f'data-caused-by="{cut_in}"' in body

    def test_a_condition_that_names_no_action_claims_no_cause(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Position must never be enough to draw a causal link.

        The starter's trigger is a distance/TTC pair -- true of the world, not
        produced by any action -- so it has to stay unlinked however the cards
        are arranged.
        """
        body = client.get(f"/draft/{draft_id}").text
        assert 'data-caused-by=""' in body

    def test_an_action_moves_into_an_empty_step(
        self, client: TestClient, draft_id: str
    ) -> None:
        """A lone card still moves, or a reaction could never follow its cause.

        The cut-in starter gives NPC1 a single action; packing a lane against
        its neighbours would pin it to the first step forever.
        """
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        action_id = document["actions"][0]["id"]
        client.post(f"/draft/{draft_id}/action/{action_id}/move", data={"delta": "2"})
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        assert document["ui"]["nodes"][action_id]["column_hint"] == 2

    def test_a_lanelet_field_gets_a_map_to_pick_from(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Nobody knows lanelet ids by heart, so the map is part of the control.

        The picker writes into the very input the form submits, so a picked id
        and a typed one are saved by the same path.
        """
        import yaml as _yaml

        document = _yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        node = document["assertions"]["pass"][0]["children"][0]["id"]
        body = client.get(f"/draft/{draft_id}/inspector/{node}").text

        assert 'id="pick-lanelet_id"' in body
        assert 'data-picks-into="pick-lanelet_id"' in body
        # The map is the only editor: a typed id would be a second way in to
        # keep in step with the picked one.
        assert "readonly" in body.split('id="pick-lanelet_id"')[1].split(">")[0]
        assert 'data-open-picker="picker-lanelet_id"' in body
        assert f'data-map-src="/draft/{draft_id}/map.osm"' in body
        # The current value is highlighted, or the map cannot show what is set.
        assert 'data-highlight="183"' in body

    def test_the_spawn_lanelet_uses_the_same_picker(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The spawn section is hand-written, so it can drift from the rest.

        It did: conditions stopped having a text box for a lanelet id while the
        entity inspector kept one. Both now render the same macro.
        """
        body = client.get(f"/draft/{draft_id}/inspector/ego").text
        assert 'id="pick-spawn_lanelet_id"' in body
        assert 'data-open-picker="picker-spawn_lanelet_id"' in body
        assert 'data-picks-into="pick-spawn_lanelet_id"' in body
        field = body.split('id="pick-spawn_lanelet_id"')[0].rsplit("<input", 1)[1]
        assert "ed-input-locked" in field

    def test_the_scenario_exclusion_list_is_picked_too(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Hand-written sections drift; this one is a set, not a single id."""
        body = client.get(f"/draft/{draft_id}/inspector/scenario").text
        assert 'id="pick-map_no_3d_model_lanelet_ids"' in body
        assert "data-picks-many" in body

    def test_a_picker_ships_the_layers_a_click_may_land_on(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Refusing a bad click needs the accepted layers on the client.

        A `bound` reports a linestring id, so accepting it would save a
        boundary as a lanelet.
        """
        body = client.get(f"/draft/{draft_id}/inspector/scenario").text
        layers = body.split('data-picks-layer="')[1].split('"')[0].split(",")
        assert "direction" in layers, "a click on a road lands on the arrow"
        assert "bound" not in layers

    def test_only_a_lanelet_field_gets_a_map(
        self, client: TestClient, draft_id: str
    ) -> None:
        """A plain number must not drag a wasm map into the inspector."""
        import yaml as _yaml

        document = _yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        # The cut-in trigger's TTC condition is seconds, not a lanelet.
        ttc = document["actions"][0]["trigger"]["children"][1]["id"]
        body = client.get(f"/draft/{draft_id}/inspector/{ttc}").text
        assert "ed-map-picker" not in body

    def test_a_dependent_action_cannot_be_moved_onto_its_dependency(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """Within a step nothing is ordered, so a dependency cannot share one.

        The move is repaired rather than refused: the card lands as close to
        where it was aimed as its dependencies allow.
        """
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        cut_in = document["actions"][0]["id"]
        client.post(
            f"/draft/{draft_id}/action", data={"type_id": "turn", "actor": "npc1"}
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        reaction = document["actions"][-1]["id"]

        client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": f"trigger:{reaction}", "type_id": "action_state"},
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        node = document["actions"][-1]["trigger"]["id"]
        client.post(
            f"/draft/{draft_id}/condition/{node}",
            data={"action": cut_in, "state": "completeState"},
        )

        stored = _document(store, draft_id)
        assert stored.ui.column_of(reaction) > stored.ui.column_of(cut_in)

        client.post(f"/draft/{draft_id}/action/{reaction}/move", data={"delta": "-1"})
        stored = _document(store, draft_id)
        assert stored.ui.column_of(reaction) > stored.ui.column_of(cut_in)

    def test_moving_a_dependency_carries_its_dependents(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        cut_in = document["actions"][0]["id"]
        client.post(
            f"/draft/{draft_id}/action", data={"type_id": "turn", "actor": "ego"}
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        reaction = document["actions"][-1]["id"]
        client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": f"trigger:{reaction}", "type_id": "action_state"},
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        node = document["actions"][-1]["trigger"]["id"]
        client.post(
            f"/draft/{draft_id}/condition/{node}",
            data={"action": cut_in, "state": "completeState"},
        )

        client.post(f"/draft/{draft_id}/action/{cut_in}/move", data={"delta": "2"})
        stored = _document(store, draft_id)
        assert stored.ui.column_of(cut_in) == 2
        assert stored.ui.column_of(reaction) == 3

    def test_a_condition_reads_as_subject_target_metric_value(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The reading the whole canvas is built around."""
        body = client.get(f"/draft/{draft_id}/canvas").text
        # Subject and target, the metric, and the value with its unit -- asserted
        # as the pieces the reading is made of rather than as one literal string,
        # so restyling the chip does not look like a regression.
        assert 'class="cond-subject"' in body
        assert "&#8594;" in body or "→" in body
        assert ">Distance<" in body
        assert ">TTC<" in body
        for value, unit in (("20.0", "m"), ("4.0", "s")):
            assert value in body
            assert '<span class="cond-unit">%s</span>' % unit in body

    def test_triggers_are_attached_to_actions_not_a_separate_lane(
        self, client: TestClient, draft_id: str
    ) -> None:
        body = client.get(f"/draft/{draft_id}/canvas").text
        assert 'data-links-to="node-' in body
        assert "Events" not in body

    def test_partials_render(self, client: TestClient, draft_id: str) -> None:
        assert client.get(f"/draft/{draft_id}/canvas").status_code == 200
        for object_id in ("scenario", "ego", "npc1"):
            response = client.get(f"/draft/{draft_id}/inspector/{object_id}")
            assert response.status_code == 200, object_id

    def test_the_ir_is_downloadable_as_yaml(
        self, client: TestClient, draft_id: str
    ) -> None:
        body = client.get(f"/draft/{draft_id}/yaml").text
        assert body.startswith("version: 1")
        assert "assertions:" in body

    def test_a_missing_draft_is_a_404_page_not_a_crash(
        self, client: TestClient
    ) -> None:
        response = client.get("/draft/does_not_exist")
        assert response.status_code == 404
        assert "Back to drafts" in response.text

    def test_the_page_styles_itself_without_a_cdn(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The editor is often run on a closed network; layout must not be remote."""
        body = client.get(f"/draft/{draft_id}").text
        assert '<link rel="stylesheet" href="/static/editor.css">' in body
        assert "cdn.tailwindcss.com" not in body
        assert client.get("/static/editor.css").status_code == 200

    def test_an_inspector_can_be_opened_without_javascript(
        self, client: TestClient, draft_id: str
    ) -> None:
        """?selected= renders server-side, so a deep link works and so do tests."""
        body = client.get(f"/draft/{draft_id}?selected=npc1").text
        assert "Candidate lanelets" in body
        assert "Constraint search" in body


class TestEntityEditing:
    def test_adding_an_entity_numbers_it_like_its_carla_role(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(f"/draft/{draft_id}/entity", data={"kind": "vehicle"})
        document = _document(store, draft_id)
        assert [e.id for e in document.entities] == ["ego", "npc1", "npc2"]

    def test_a_second_ego_is_refused(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        response = client.post(f"/draft/{draft_id}/entity", data={"kind": "ego"})
        assert "already has an ego" in response.text
        assert len(_document(store, draft_id).entities) == 2

    def test_updating_an_entity_persists(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/entity/npc1",
            data={
                "title": "Cut-in car",
                "vehicle_type": "vehicle.audi.tt",
                "initial_speed_kmh": "30",
                "spawn_mode": "fixed",
                "spawn_lanelet_id": "200",
                "spawn_s_mode": "fixed",
                "spawn_s": "12.5",
            },
        )
        entity = _entity(store, draft_id, "npc1")
        assert (entity.title, entity.vehicle_type, entity.initial_speed_kmh) == (
            "Cut-in car",
            "vehicle.audi.tt",
            30.0,
        )
        assert entity.spawn.mode == "fixed"
        assert (entity.spawn.lanelet_id, entity.spawn.s.value) == (200, 12.5)

    def test_a_derived_offset_stores_a_binding(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/entity/npc1",
            data={
                "spawn_mode": "constraint_search",
                "spawn_s_mode": "derived",
                "spawn_s": "10",
                "binding_type": "stop_line_offset",
                "binding_offset": "18",
            },
        )
        entity = _entity(store, draft_id, "npc1")
        assert entity.spawn.s.binding is not None
        assert entity.spawn.s.binding.type == "stop_line_offset"
        assert entity.spawn.s.binding.params == {"offset": 18.0}

    def test_a_bad_number_is_explained_and_not_stored(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        before = _entity(store, draft_id, "npc1")
        response = client.post(
            f"/draft/{draft_id}/entity/npc1", data={"initial_speed_kmh": "fast"}
        )
        assert "must be a number" in response.text
        after = _entity(store, draft_id, "npc1")
        assert after.initial_speed_kmh == before.initial_speed_kmh

    def test_deleting_an_entity_removes_conditions_that_named_it(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """A delete must not leave behind a validation error the user did not cause."""
        client.post(f"/draft/{draft_id}/entity/npc1/delete")
        document = _document(store, draft_id)
        assert document.entity("npc1") is None
        assert not [a for a in document.actions if a.actor == "npc1"]
        for root in document.condition_roots():
            for node in root.walk():
                assert "npc1" not in {str(v) for v in node.params.values()}


class TestActionEditing:
    def test_add_update_and_delete(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/action", data={"type_id": "turn", "actor": "ego"}
        )
        document = _document(store, draft_id)
        action = next(a for a in document.actions if a.type == "turn")

        client.post(
            f"/draft/{draft_id}/action/{action.id}",
            data={
                "title": "Turn right",
                "actor": "ego",
                "direction": "right",
                "search_distance": "120",
                "timing": "post_tick",
                "once": "on",
            },
        )
        updated = _action(store, draft_id, action.id)
        assert updated.title == "Turn right"
        assert updated.params == {"direction": "right", "search_distance": 120.0}
        assert updated.timing == "post_tick"

        client.post(f"/draft/{draft_id}/action/{action.id}/delete")
        assert _document(store, draft_id).action(action.id) is None

    def test_unchecking_once_is_stored(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        action = _document(store, draft_id).actions[0]
        client.post(f"/draft/{draft_id}/action/{action.id}", data={"direction": "left"})
        assert _action(store, draft_id, action.id).once is False

    def test_moving_an_action_changes_layout_only(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/action", data={"type_id": "turn", "actor": "npc1"}
        )
        document = _document(store, draft_id)
        before = [a.id for a in document.actions_for("npc1")]

        client.post(f"/draft/{draft_id}/action/{before[1]}/move", data={"delta": "-1"})
        after_document = _document(store, draft_id)
        # Nothing sequences these two, so they may share a step -- moving one
        # onto the other no longer pushes it aside.
        assert after_document.ui.column_of(before[0]) == after_document.ui.column_of(
            before[1]
        )
        assert len(after_document.action_slots("npc1")[0]) == 2
        # The semantic content is untouched: same actions, same triggers.
        assert {a.id for a in after_document.actions} == {
            a.id for a in document.actions
        }


class TestPredicateEditing:
    def test_a_second_trigger_condition_wraps_both_in_all(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        document = _document(store, draft_id)
        action = document.actions[0]
        assert action.trigger is not None
        assert action.trigger.type == "all"

        client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": f"trigger:{action.id}", "type_id": "speed"},
        )
        trigger = _present(_action(store, draft_id, action.id).trigger, "trigger")
        assert [c.type for c in trigger.children] == [
            "entity_distance",
            "ttc",
            "speed",
        ]

    def test_a_wrapper_refuses_a_second_child(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/condition", data={"slot": "fail", "type_id": "not"}
        )
        wrapper = _document(store, draft_id).assertions.fail_conditions[-1]
        client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": f"node:{wrapper.id}", "type_id": "collision"},
        )
        response = client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": f"node:{wrapper.id}", "type_id": "collision"},
        )
        assert "already has its" in response.text
        assert len(_condition(store, draft_id, wrapper.id).children) == 1

    def test_pass_and_fail_conditions_can_be_added_and_edited(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": "pass", "type_id": "elapsed_time"},
        )
        node = _document(store, draft_id).assertions.pass_conditions[-1]
        client.post(
            f"/draft/{draft_id}/condition/{node.id}",
            data={"rule": "greater_than", "duration_seconds": "12"},
        )
        updated = _condition(store, draft_id, node.id)
        assert updated.params == {"rule": "greater_than", "duration_seconds": 12.0}

    def test_deleting_the_only_trigger_condition_clears_the_trigger(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        action = _document(store, draft_id).actions[0]
        trigger = _present(action.trigger, "trigger")
        client.post(f"/draft/{draft_id}/condition/{trigger.id}/delete")
        assert _action(store, draft_id, action.id).trigger is None


class TestSpawnConstraints:
    def test_constraints_nest_and_delete(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        root = _entity(store, draft_id, "npc1").spawn.constraints[0]

        client.post(
            f"/draft/{draft_id}/constraint",
            data={
                "entity_id": "npc1",
                "type_id": "lanelet_length",
                "parent_id": root.id,
            },
        )
        added = _entity(store, draft_id, "npc1").spawn.constraints[0].constraints[-1]
        assert added.type == "lanelet_length"

        client.post(
            f"/draft/{draft_id}/constraint/{added.id}",
            data={"rule": "less_than", "value": "42.5", "selected": "npc1"},
        )
        updated = _entity(store, draft_id, "npc1").spawn.constraints[0].constraints[-1]
        assert updated.params == {"rule": "less_than", "value": 42.5}

        client.post(
            f"/draft/{draft_id}/constraint/{added.id}/delete", data={"selected": "npc1"}
        )
        remaining = _entity(store, draft_id, "npc1").spawn.constraints[0]
        assert added.id not in {n.id for n in remaining.walk()}

    def test_the_editor_uses_the_sweepers_own_syntax(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """No GUI-only constraint engine: the tree must parse with the sweeper."""
        from autoware_carla_scenario.sweeper.constraints import parse_constraint

        for node in _entity(store, draft_id, "npc1").spawn.constraints:
            assert parse_constraint(node.to_sweep_dict()) is not None

    def test_preview_without_a_loaded_map_still_describes_the_constraints(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Constraint editing must not wait on a Lanelet2 map."""
        from autoware_carla_scenario.editor import map_preview

        map_preview.clear_cache()
        response = client.post(
            f"/draft/{draft_id}/spawn-preview", data={"entity_id": "npc1"}
        )
        assert response.status_code == 200
        assert "Not evaluated yet" in response.text
        assert "Preview matches" in response.text

    def test_the_map_is_served_for_the_wasm_viewer(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The viewer parses the .osm in the browser, so the editor serves it."""
        response = client.get(f"/draft/{draft_id}/map.osm")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/xml")
        assert b"<osm" in response.content[:512]

    def test_a_scenario_without_a_map_file_cannot_serve_one(
        self, client: TestClient, draft_id: str
    ) -> None:
        client.post(f"/draft/{draft_id}/scenario", data={"map_lanelet2_path": ""})
        response = client.get(f"/draft/{draft_id}/map.osm")
        assert response.status_code == 404
        assert "Lanelet2" in response.text

    def test_the_preview_hands_the_viewer_what_it_needs(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The data attributes on the frame are the whole client-side contract."""
        from autoware_carla_scenario.editor import map_preview

        map_preview.clear_cache()
        body = client.post(
            f"/draft/{draft_id}/spawn-preview",
            data={"entity_id": "npc1", "load_map": "1"},
        ).text
        assert 'data-map-src="/draft/%s/map.osm"' % draft_id in body
        assert 'data-entity="npc1"' in body
        assert "data-highlight=" in body
        assert "hakuturu583.github.io/simple_lanelet2/viewer.js" in body

    def test_the_viewer_frame_is_hidden_until_the_module_loads(
        self, client: TestClient, draft_id: str
    ) -> None:
        """An empty box where a map should be is worse than no box."""
        from autoware_carla_scenario.editor import map_preview

        map_preview.clear_cache()
        body = client.post(
            f"/draft/{draft_id}/spawn-preview",
            data={"entity_id": "npc1", "load_map": "1"},
        ).text
        assert '<div class="ed-map-frame" hidden' in body
        # The viewer is the only renderer; nothing is drawn server-side to sit
        # underneath it and be mistaken for a second map.
        assert "data-map-fallback" not in body
        assert "<svg viewBox" not in body

    def test_a_self_hosted_viewer_can_be_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.editor import map_preview
        from autoware_carla_scenario.editor.app import MAP_VIEWER_ENV, create_app

        monkeypatch.setenv(MAP_VIEWER_ENV, "/vendor/viewer.js")
        map_preview.clear_cache()
        client = TestClient(create_app(draft_dir=tmp_path / "drafts"))
        draft_id = (
            client.post("/new", data={"kind": "cut_in"}, follow_redirects=False)
            .headers["location"]
            .rsplit("/", 1)[-1]
        )
        body = client.post(
            f"/draft/{draft_id}/spawn-preview",
            data={"entity_id": "npc1", "load_map": "1"},
        ).text
        assert 'data-map-viewer="/vendor/viewer.js"' in body

    def test_a_fixed_spawn_is_shown_on_the_map_too(
        self, client: TestClient, draft_id: str
    ) -> None:
        """A hand-typed lanelet ID is worth seeing, and clicking one is faster."""
        from autoware_carla_scenario.editor import map_preview

        map_preview.clear_cache()
        body = client.post(
            f"/draft/{draft_id}/spawn-preview",
            data={"entity_id": "ego", "load_map": "1"},
        ).text
        assert "ed-map-frame" in body
        assert 'data-entity="ego"' in body
        assert "183" in body  # the ego's fixed spawn lanelet
        # No match list: there are no constraints to match.
        assert "matched of" not in body
        # The viewer has one highlight colour, so a fixed spawn outlines the
        # pinned lanelet and nothing else.
        assert 'data-highlight="183"' in body

    def test_a_constraint_search_outlines_its_matches(
        self, client: TestClient, draft_id: str
    ) -> None:
        """What the single highlight channel means has to follow the mode.

        Under a search it is the matches; mixing the current spawn in would
        paint it the same colour and claim it is one of them.
        """
        from autoware_carla_scenario.editor import map_preview

        map_preview.clear_cache()
        body = client.post(
            f"/draft/{draft_id}/spawn-preview",
            data={"entity_id": "npc1", "load_map": "1"},
        ).text
        highlight = body.split('data-highlight="')[1].split('"')[0]
        matched = body.split("Matched IDs")[1]
        assert highlight, "a search with matches must outline them"
        assert all(f"{i}" in matched for i in highlight.split(",")[:5])

    def test_preview_reports_an_unloadable_map_without_failing(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        from autoware_carla_scenario.editor import map_preview

        map_preview.clear_cache()
        client.post(
            f"/draft/{draft_id}/scenario",
            data={"map_lanelet2_path": "", "map_xodr_path": ""},
        )
        response = client.post(
            f"/draft/{draft_id}/spawn-preview",
            data={"entity_id": "npc1", "load_map": "1"},
        )
        assert response.status_code == 200
        assert "no map files configured" in response.text


class TestScenarioMetadata:
    def test_updating_scenario_fields(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/scenario",
            data={
                "title": "Cut in v2",
                "scenario_id": "cut_in_v2",
                "description": "desc",
                "timeout_seconds": "45",
                "map_group": "nishishinjuku",
                "map_name": "NishishinjukuMap",
                "map_no_3d_model_lanelet_ids": "3, 4, 41",
            },
        )
        document = _document(store, draft_id)
        assert (document.id, document.title, document.timeout_seconds) == (
            "cut_in_v2",
            "Cut in v2",
            45.0,
        )
        assert document.map.no_3d_model_lanelet_ids == [3, 4, 41]

    def test_an_invalid_identifier_is_refused(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        response = client.post(
            f"/draft/{draft_id}/scenario", data={"scenario_id": "Not Valid"}
        )
        assert "Scenario id" in response.text
        assert _document(store, draft_id).id == "cut_in"


class TestValidateSaveExport:
    def test_validation_reports_a_clean_document(
        self, client: TestClient, draft_id: str
    ) -> None:
        assert "ready to export" in client.post(f"/draft/{draft_id}/validate").text

    def test_validation_reports_errors(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        draft = _draft(store, draft_id)
        draft.document.assertions.pass_conditions = []
        store.save(draft)
        response = client.post(f"/draft/{draft_id}/validate")
        assert "cannot be exported" in response.text

    def test_saving_a_draft_reports_where_it_went(
        self, client: TestClient, draft_id: str
    ) -> None:
        assert "Draft saved" in client.post(f"/draft/{draft_id}/save").text

    def test_deleting_a_draft_returns_to_the_list(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        response = client.post(f"/draft/{draft_id}/delete", follow_redirects=False)
        assert response.status_code == 303
        assert store.get(draft_id) is None

    def test_export_produces_a_downloadable_package(
        self, client: TestClient, tmp_path: Path, draft_id: str
    ) -> None:
        """The report comes back with a link, and the link serves the archive.

        The editor is used from other machines on the LAN, so an export that
        only wrote a directory on the host would put the package somewhere the
        person exporting cannot reach.
        """
        response = client.post(f"/draft/{draft_id}/export", data={"dev_mode": "on"})
        assert response.status_code == 200
        assert "Package exported" in response.text
        assert f"/draft/{draft_id}/package.zip" in response.text

        download = client.get(f"/draft/{draft_id}/package.zip")
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
        assert 'filename="cut_in.zip"' in download.headers["content-disposition"]

        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            names = archive.namelist()
        # One top-level directory, so unpacking does not spray the CWD.
        assert {n.split("/")[0] for n in names} == {"cut_in_scenario"}
        assert "cut_in_scenario/pyproject.toml" in names

    def test_the_package_tree_is_not_left_on_the_host(
        self, client: TestClient, tmp_path: Path, draft_id: str
    ) -> None:
        """Only the archive outlives the request; the build tree is temporary."""
        client.post(f"/draft/{draft_id}/export", data={"dev_mode": "on"})
        staged = sorted(p.name for p in (tmp_path / "packages").iterdir())
        assert staged == ["cut_in.zip"]

    def test_downloading_before_an_export_is_an_error_not_a_traceback(
        self, client: TestClient, draft_id: str
    ) -> None:
        """A stale or guessed link is a normal thing to click."""
        assert (
            "No exported package" in client.get(f"/draft/{draft_id}/package.zip").text
        )

    def test_a_failed_export_is_reported_as_a_failure(
        self, client: TestClient, store: DraftStore, tmp_path: Path, draft_id: str
    ) -> None:
        draft = _draft(store, draft_id)
        draft.document.assertions.pass_conditions = []
        store.save(draft)
        response = client.post(f"/draft/{draft_id}/export", data={"dev_mode": "on"})
        assert "Export failed" in response.text
        assert not (tmp_path / "packages" / "cut_in.zip").exists()


class TestServiceGuards:
    def test_unknown_drafts_raise(self, tmp_path: Path) -> None:
        service = EditorService(DraftStore(tmp_path))
        with pytest.raises(EditorError):
            service.require_draft("missing")

    def test_a_traversal_draft_id_raises(self, tmp_path: Path) -> None:
        service = EditorService(DraftStore(tmp_path))
        with pytest.raises(EditorError):
            service.require_draft("../secrets")
