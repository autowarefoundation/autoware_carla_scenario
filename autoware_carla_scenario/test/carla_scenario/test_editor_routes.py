"""Scenario Editor routes.

The editor keeps no client-side model: every edit is a form post that returns
re-rendered HTML. So the tests drive it the way a browser does -- post a form,
then assert on the stored document *and* on what came back -- which is also the
only way to catch a template that renders but shows the wrong thing.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any, TypeVar

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
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.editor import app as editor_app
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

    def test_the_map_route_refuses_a_file_outside_its_roots(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """The map path is a document field, and the editor binds 0.0.0.0.

        Handed straight to a FileResponse it made this route an arbitrary local
        file read for anyone who could reach the port.
        """
        draft = _draft(store, draft_id)
        for path in ("/etc/hostname", "../../../../etc/hosts", "pyproject.toml"):
            draft.document.map.lanelet2_path = path
            store.save(draft)
            assert client.get(f"/draft/{draft_id}/map.osm").status_code == 404, path

    def test_the_map_route_still_serves_a_map_inside_its_roots(
        self, client: TestClient, draft_id: str
    ) -> None:
        assert client.get(f"/draft/{draft_id}/map.osm").status_code == 200

    def test_deleting_an_action_drops_what_waited_on_it(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """A dangling reference would block compilation and export.

        The user would have to hunt down every dependent condition by hand for
        a delete they did not know was destructive.
        """
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

        client.post(f"/draft/{draft_id}/action/{cut_in}/delete")
        stored = _document(store, draft_id)
        survivor = stored.action(reaction)
        assert survivor is not None, "the dependent action survives"
        assert survivor.trigger is None
        assert validate_document(stored).ok

    def test_an_unconfigured_reference_says_so_on_its_card(
        self, client: TestClient, draft_id: str
    ) -> None:
        """A condition that needs an action must not read as world-scoped.

        `action_state` declares a reference; before one is chosen it used to
        render as "world", exactly like a collision or a timeout, so the card
        looked finished and gave no reason to open the inspector.
        """
        client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": "fail", "type_id": "action_state"},
        )
        body = client.get(f"/draft/{draft_id}").text
        assert "pick action" in body

        # A genuinely world-scoped condition keeps saying "world".
        assert 'class="cond-world">world' in body

    def test_choosing_the_action_gives_the_card_its_link(
        self, client: TestClient, draft_id: str
    ) -> None:
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        cut_in = document["actions"][0]["id"]
        client.post(
            f"/draft/{draft_id}/condition",
            data={"slot": "fail", "type_id": "action_state"},
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        node = document["assertions"]["fail"][-1]["id"]
        client.post(
            f"/draft/{draft_id}/condition/{node}",
            data={"action": cut_in, "state": "completeState"},
        )

        body = client.get(f"/draft/{draft_id}").text
        assert f'data-caused-by="{cut_in}"' in body
        assert "pick action" not in body

    def test_an_untriggered_action_drawn_late_is_warned_about(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """Its position says "later"; the runtime fires it on the first tick.

        An action with no trigger gets `AlwaysTrueCondition`, and everything is
        armed from tick one, so the step number is simply wrong about it.
        """
        client.post(
            f"/draft/{draft_id}/action", data={"type_id": "turn", "actor": "ego"}
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        action = document["actions"][-1]["id"]
        client.post(f"/draft/{draft_id}/action/{action}/move", data={"delta": "3"})

        report = validate_document(_document(store, draft_id))
        assert report.ok, "a position is presentation; it must not block anything"
        assert any("fires on the first tick" in w.message for w in report.warnings)

    def test_a_position_can_be_narrowed_to_a_stretch_of_the_lane(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """The runtime has always taken `s`/`t` rules; the editor threw them away.

        What this route owns is the round trip: a bound typed into the form has
        to reach the compiled parameters as a number.  Turning the pair into
        one-sided comparison rules is the generator's half, asserted against the
        spec in `test_authoring_registry` -- building the condition here would
        need a loaded map, which is neither what broke nor what this covers.
        """
        from autoware_carla_scenario.authoring.compiler import compile_document

        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        node = document["assertions"]["pass"][0]["children"][0]["id"]
        client.post(
            f"/draft/{draft_id}/condition/{node}",
            data={"entity": "npc1", "lanelet_id": "183", "s_min": "20", "s_max": "50"},
        )

        compiled = compile_document(_document(store, draft_id))
        params = compiled.pass_conditions[0].children[0].params

        assert (params["s_min"], params["s_max"]) == (20.0, 50.0)
        assert params["t_min"] is None and params["t_max"] is None

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

    def test_the_spawn_is_chosen_where_the_map_is(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Pinned or searched is a question about the map, so it is asked there.

        The column states which it is and offers Edit; the choice, and the
        search it leads to, live in the picker beside the map they are about.
        """
        body = client.get(f"/draft/{draft_id}/inspector/npc1").text
        picker = body.split('id="picker-spawn_lanelet_id"', 1)[1]
        column = body.split('id="picker-spawn_lanelet_id"', 1)[0]

        assert 'name="lanelet_mode_npc1_spawn"' not in column
        assert 'name="lanelet_mode_npc1_spawn"' in picker
        # The constraint tree, and each node's own parameters, come with it:
        # the inspector column is behind the map while the picker is open.
        assert "ed-constraint-fields" in picker
        assert "ed-constraint" not in column
        # Where along the lanelet is the same kind of question, so it is asked
        # in the same place -- the column only says what the answer is.
        assert 'name="spawn_s_mode"' not in column
        assert 'name="spawn_s_mode"' in picker
        assert 'name="spawn_s"' in picker

    def test_the_goal_is_picked_from_the_map_like_the_spawn(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """The ego's goal is a lanelet, so it is chosen the way every one is."""
        body = client.get(f"/draft/{draft_id}/inspector/ego").text

        assert 'id="pick-goal_lanelet_id"' in body
        assert 'data-open-picker="picker-goal_lanelet_id"' in body
        assert 'name="goal_s"' in body

    def test_only_the_ego_is_offered_a_goal(
        self, client: TestClient, draft_id: str
    ) -> None:
        body = client.get(f"/draft/{draft_id}/inspector/npc1").text
        assert "goal_lanelet_id" not in body

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
        assert "How this lanelet is chosen" in body
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
                "spawn_lanelet_id": "200",
                "spawn_s_mode": "fixed",
                "spawn_s": "12.5",
            },
        )
        # Fixed or searched comes from the picker's own route, not this form:
        # it is the question every lanelet field asks, and one place answers it.
        client.post(
            f"/draft/{draft_id}/lanelet-mode",
            data={"slot": "npc1.spawn", "mode": "fixed"},
        )
        entity = _entity(store, draft_id, "npc1")
        assert (entity.title, entity.vehicle_type, entity.initial_speed_kmh) == (
            "Cut-in car",
            "vehicle.audi.tt",
            30.0,
        )
        assert entity.spawn.mode == "fixed"
        assert (entity.spawn.lanelet_id, entity.spawn.s.value) == (200, 12.5)

    def test_the_ego_is_given_a_goal_beside_its_spawn(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/entity/ego",
            data={"goal_lanelet_id": "265", "goal_s": "12.5"},
        )
        ego = _entity(store, draft_id, "ego")
        assert ego.goal is not None
        assert (ego.goal.lanelet_id, ego.goal.s) == (265, 12.5)

    def test_an_empty_goal_lanelet_clears_the_goal(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        # An ego the TrafficManager drives may have no destination, and Clear
        # goal is how it goes back to having none.
        from autoware_carla_scenario.authoring.validator import validate_document

        client.post(f"/draft/{draft_id}/entity/ego", data={"goal_lanelet_id": ""})

        assert _entity(store, draft_id, "ego").goal is None
        assert validate_document(_document(store, draft_id)).ok

    def test_an_autoware_ego_left_without_a_goal_is_reported(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        from autoware_carla_scenario.authoring.validator import validate_document

        client.post(
            f"/draft/{draft_id}/entity/ego",
            data={"driven_by": "autoware", "goal_lanelet_id": ""},
        )

        entity = _entity(store, draft_id, "ego")
        assert (entity.driven_by, entity.goal) == ("autoware", None)
        report = validate_document(_document(store, draft_id))
        assert not report.ok
        assert any("no goal" in issue.message for issue in report.errors)

    def test_the_ego_says_which_stack_drives_it(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(f"/draft/{draft_id}/entity/ego", data={"driven_by": "autoware"})
        assert _entity(store, draft_id, "ego").driven_by == "autoware"

        # An unknown value is ignored rather than stored: the document only ever
        # holds a stack the export can name.
        client.post(f"/draft/{draft_id}/entity/ego", data={"driven_by": "__nope__"})
        assert _entity(store, draft_id, "ego").driven_by == "autoware"

    def test_a_vehicle_that_is_not_the_ego_is_not_given_a_driver(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(f"/draft/{draft_id}/entity/npc1", data={"driven_by": "autoware"})
        assert _entity(store, draft_id, "npc1").driven_by == "autopilot"

    def test_a_partial_form_leaves_the_goal_alone(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """Clicking a lanelet on the map posts the spawn alone; the goal stays."""
        client.post(
            f"/draft/{draft_id}/entity/ego",
            data={"goal_lanelet_id": "265", "goal_s": "12.5"},
        )
        client.post(f"/draft/{draft_id}/entity/ego", data={"spawn_lanelet_id": "200"})

        ego = _entity(store, draft_id, "ego")
        assert ego.spawn.lanelet_id == 200
        assert ego.goal is not None
        assert ego.goal.lanelet_id == 265

    def test_a_vehicle_that_is_not_the_ego_is_given_no_goal(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        # The inspector offers the control for the ego only; a form that names
        # one anyway must not store what validation would then reject.
        client.post(f"/draft/{draft_id}/entity/npc1", data={"goal_lanelet_id": "265"})
        assert _entity(store, draft_id, "npc1").goal is None

    def test_a_derived_offset_stores_a_binding(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/entity/npc1",
            data={
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
                "phase": "post_tick",
                "once": "on",
            },
        )
        updated = _action(store, draft_id, action.id)
        assert updated.title == "Turn right"
        assert updated.params == {"direction": "right", "search_distance": 120.0}
        assert updated.phase == "post_tick"

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


class TestLaneletSearches:
    """Every lanelet a document names is chosen the same way.

    The spawn could be searched for years before anything else could, because
    the panel that asks was written into the entity inspector rather than into
    the picker every lanelet field opens.
    """

    @staticmethod
    def _lane_condition(store: DraftStore, draft_id: str) -> ConditionNode:
        """The starter's PASS assertion names a lanelet; return that node."""
        return _document(store, draft_id).assertions.pass_conditions[0].children[0]

    def test_a_condition_lanelet_is_offered_the_same_choice_as_a_spawn(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        node = self._lane_condition(store, draft_id)
        body = client.get(f"/draft/{draft_id}/inspector/{node.id}").text
        picker = body.split('id="picker-lanelet_id"', 1)[1]
        column = body.split('id="picker-lanelet_id"', 1)[0]

        group = f"lanelet_mode_{node.id}_lanelet_id"
        assert f'name="{group}"' in picker
        assert f'name="{group}"' not in column
        assert f'"slot": "{node.id}.lanelet_id"' in picker

    def test_a_goal_is_offered_the_same_choice_as_a_spawn(
        self, client: TestClient, draft_id: str
    ) -> None:
        body = client.get(f"/draft/{draft_id}/inspector/ego").text
        picker = body.split('id="picker-goal_lanelet_id"', 1)[1]
        assert 'name="lanelet_mode_ego_goal"' in picker
        assert '"slot": "ego.goal"' in picker

    def test_a_set_of_lanelets_is_picked_by_hand(
        self, client: TestClient, draft_id: str
    ) -> None:
        """A search names one lanelet per run, so it cannot fill a set."""
        body = client.get(f"/draft/{draft_id}/inspector/scenario").text
        picker = body.split('id="picker-map_no_3d_model_lanelet_ids"', 1)[1]
        assert "How this lanelet is chosen" not in picker
        assert "data-picks-many" in picker

    def test_a_condition_lanelet_can_be_searched_for(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """The whole point: a constraint tree on something that is not a spawn."""
        node = self._lane_condition(store, draft_id)
        slot = f"{node.id}.lanelet_id"

        client.post(
            f"/draft/{draft_id}/lanelet-mode",
            data={"slot": slot, "mode": "constraint_search"},
        )
        client.post(
            f"/draft/{draft_id}/constraint",
            data={"slot": slot, "type_id": "is_junction"},
        )

        stored = _condition(store, draft_id, node.id).searches["lanelet_id"]
        assert stored.mode == "constraint_search"
        assert [c.type for c in stored.constraints] == ["is_junction"]

    def test_a_search_survives_a_flip_back_to_fixed(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """Changing one's mind twice must not cost the tree already written."""
        node = self._lane_condition(store, draft_id)
        slot = f"{node.id}.lanelet_id"
        client.post(
            f"/draft/{draft_id}/lanelet-mode",
            data={"slot": slot, "mode": "constraint_search"},
        )
        client.post(
            f"/draft/{draft_id}/constraint",
            data={"slot": slot, "type_id": "is_junction"},
        )
        client.post(
            f"/draft/{draft_id}/lanelet-mode", data={"slot": slot, "mode": "fixed"}
        )

        stored = _condition(store, draft_id, node.id).searches["lanelet_id"]
        assert stored.mode == "fixed"
        assert [c.type for c in stored.constraints] == ["is_junction"]

    def test_a_goal_can_be_searched_for_before_one_is_pinned(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """The search is *how* a goal is chosen, so it cannot need one first."""
        client.post(f"/draft/{draft_id}/entity/ego", data={"goal_lanelet_id": ""})
        assert _entity(store, draft_id, "ego").goal is None

        client.post(
            f"/draft/{draft_id}/lanelet-mode",
            data={"slot": "ego.goal", "mode": "constraint_search"},
        )
        goal = _present(_entity(store, draft_id, "ego").goal, "goal")
        assert goal.mode == "constraint_search"

    def test_a_searched_condition_lanelet_reaches_the_sweeper(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """A search is only real if the exported config can be swept on it."""
        from autoware_carla_scenario.authoring.hydra_config import (
            build_scenario_config,
        )

        node = self._lane_condition(store, draft_id)
        slot = f"{node.id}.lanelet_id"
        # The starter already searches NPC1's spawn, and the sweeper drives one
        # key per run; pin it so this search is the one that is driven.
        client.post(
            f"/draft/{draft_id}/lanelet-mode",
            data={"slot": "npc1.spawn", "mode": "fixed"},
        )
        client.post(
            f"/draft/{draft_id}/lanelet-mode",
            data={"slot": slot, "mode": "constraint_search"},
        )
        client.post(
            f"/draft/{draft_id}/constraint",
            data={"slot": slot, "type_id": "is_junction"},
        )

        config = build_scenario_config(_document(store, draft_id))
        key = f"scenario.param_overrides.{node.id}.lanelet_id"
        assert config["sweep"]["constraints"] == {key: [{"type": "is_junction"}]}
        # And the key it writes to is declared, or Hydra's struct mode refuses it.
        assert config["scenario"]["param_overrides"][node.id]["lanelet_id"] == 183

    def test_no_inspector_nests_a_form_inside_another(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        """The whole reason the pickers are rendered outside the forms.

        A search is made of forms -- one per constraint node, one to add one --
        and a nested `<form>` is dropped by the parser, so the controls inside
        it simply stop submitting. It fails silently, hence the test.
        """
        from html.parser import HTMLParser

        class _Nesting(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.depth = 0
                self.nested = 0

            def handle_starttag(self, tag: str, attrs: object) -> None:
                if tag == "form":
                    self.nested += 1 if self.depth else 0
                    self.depth += 1

            def handle_endtag(self, tag: str) -> None:
                if tag == "form":
                    self.depth = max(0, self.depth - 1)

        node = self._lane_condition(store, draft_id)
        # With a search open, which is when the picker holds the most forms.
        client.post(
            f"/draft/{draft_id}/lanelet-mode",
            data={"slot": f"{node.id}.lanelet_id", "mode": "constraint_search"},
        )
        client.post(
            f"/draft/{draft_id}/constraint",
            data={"slot": f"{node.id}.lanelet_id", "type_id": "lanelet_length"},
        )

        for target in ("scenario", "ego", "npc1", node.id):
            parser = _Nesting()
            parser.feed(client.get(f"/draft/{draft_id}/inspector/{target}").text)
            assert parser.nested == 0, f"{target} nests a form"

    def test_a_second_search_is_reported_rather_than_dropped(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        node = self._lane_condition(store, draft_id)
        client.post(
            f"/draft/{draft_id}/lanelet-mode",
            data={"slot": f"{node.id}.lanelet_id", "mode": "constraint_search"},
        )
        client.post(
            f"/draft/{draft_id}/constraint",
            data={"slot": f"{node.id}.lanelet_id", "type_id": "is_junction"},
        )
        report = validate_document(_document(store, draft_id))
        assert report.ok
        assert any("searches one lanelet" in i.message for i in report.warnings)


class TestSpawnConstraints:
    def test_constraints_nest_and_delete(
        self, client: TestClient, store: DraftStore, draft_id: str
    ) -> None:
        root = _entity(store, draft_id, "npc1").spawn.constraints[0]

        client.post(
            f"/draft/{draft_id}/constraint",
            data={
                "slot": "npc1.spawn",
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
            f"/draft/{draft_id}/lanelet-preview", data={"slot": "npc1.spawn"}
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

    def test_the_picker_hands_the_viewer_what_it_needs(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The data attributes on the frame are the whole client-side contract."""
        body = client.get(f"/draft/{draft_id}/inspector/npc1").text
        assert 'data-map-src="/draft/%s/map.osm"' % draft_id in body
        assert 'data-picks-into="pick-spawn_lanelet_id"' in body
        assert "data-highlight=" in body
        assert "hakuturu583.github.io/simple_lanelet2/viewer.js" in body

    def test_the_viewer_frame_is_hidden_until_the_module_loads(
        self, client: TestClient, draft_id: str
    ) -> None:
        """An empty box where a map should be is worse than no box."""
        body = client.get(f"/draft/{draft_id}/inspector/npc1").text
        assert '<div class="ed-map-frame ed-map-full" hidden' in body
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
        body = client.get(f"/draft/{draft_id}/inspector/npc1").text
        assert 'data-map-viewer="/vendor/viewer.js"' in body

    def test_a_fixed_spawn_outlines_the_lanelet_it_pins(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Nothing is evaluated for it: the pinned id is what the map outlines."""
        body = client.get(f"/draft/{draft_id}/inspector/ego").text
        picker = body.split('id="picker-spawn_lanelet_id"', 1)[1]

        assert 'data-highlight="183"' in picker  # the ego's fixed spawn lanelet
        # No match readout: there are no constraints to match.
        assert "picker-matches" not in picker

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
            f"/draft/{draft_id}/lanelet-preview",
            data={"slot": "npc1.spawn", "load_map": "1"},
        ).text
        highlight = body.split('data-picker-highlight="')[1].split('"')[0]
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
            f"/draft/{draft_id}/lanelet-preview",
            data={"slot": "npc1.spawn", "load_map": "1"},
        )
        assert response.status_code == 200
        assert "no map configured" in response.text


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


class TestEnvironmentTrack:
    """The canvas has a track for actions no vehicle performs."""

    def test_the_track_is_always_there(self, client: TestClient, draft_id: str) -> None:
        """Including when it is empty: it is where you go to add one."""
        body = client.get(f"/draft/{draft_id}").text
        assert "ed-lane-env" in body
        assert ">Environment<" in body

    def test_each_track_offers_only_its_own_actions(
        self, client: TestClient, draft_id: str
    ) -> None:
        """A car cannot set the lights, and the world cannot change lane."""
        body = client.get(f"/draft/{draft_id}").text
        forms = re.findall(r'<form[^>]*?/action"[\s\S]*?</form>', body)
        offers = {}
        for form in forms:
            actor = re.search(r'name="actor" value="([^"]*)"', form)
            assert actor is not None
            offers[actor.group(1)] = set(re.findall(r'<option value="([^"]+)"', form))
        assert offers["ego"] == {"lane_change", "routing", "turn"}
        assert offers[""] == {"traffic_signal"}

    def test_an_environment_action_needs_no_actor(
        self, client: TestClient, draft_id: str
    ) -> None:
        response = client.post(
            f"/draft/{draft_id}/action", data={"type_id": "traffic_signal", "actor": ""}
        )
        assert response.status_code == 200
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        assert document["actions"][-1]["actor"] is None

    def test_an_actor_cannot_be_pinned_on_one(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Nothing reads the field when the action is built, so storing it would
        only move the card into a lane whose vehicle does nothing."""
        client.post(
            f"/draft/{draft_id}/action", data={"type_id": "traffic_signal", "actor": ""}
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        lights = document["actions"][-1]["id"]

        client.post(
            f"/draft/{draft_id}/action/{lights}",
            data={"title": "Lights", "actor": "ego", "state": "red", "target": "all"},
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        assert document["actions"][-1]["actor"] is None

    def test_its_inspector_states_what_it_acts_on(
        self, client: TestClient, draft_id: str
    ) -> None:
        client.post(
            f"/draft/{draft_id}/action", data={"type_id": "traffic_signal", "actor": ""}
        )
        document = yaml.safe_load(client.get(f"/draft/{draft_id}/yaml").text)
        lights = document["actions"][-1]["id"]

        body = client.get(f"/draft/{draft_id}/inspector/{lights}").text
        assert "The environment, not a vehicle" in body
        assert 'name="actor"' not in body


class TestMapViewerReuse:
    """The spawn preview re-renders constantly; the map behind it does not.

    The frame carries the key `editor.js` parks it under, so the parsed wasm
    scene survives an edit instead of being fetched and parsed again.
    """

    def test_the_picker_frame_is_marked_for_reuse(
        self, client: TestClient, draft_id: str
    ) -> None:
        """An edit made in the picker re-renders it; the parsed map is kept."""
        body = client.get(f"/draft/{draft_id}/inspector/ego").text
        assert 'data-viewer-key="picker-spawn_lanelet_id"' in body

    def test_the_script_still_honours_that_key(self) -> None:
        """The attribute is only worth rendering if something reads it."""
        script = (Path(editor_app.__file__).parent / "static" / "editor.js").read_text()
        assert "viewerKey" in script
        assert "function reuseMap(" in script


class TestScenarioMap:
    """The places panel: every lanelet a scenario names, drawn once.

    The canvas says what happens and in what order; these tests are about the
    other half of the question -- where -- and about what an *abstract* scenario
    draws, which is one bound pattern rather than a set of matches.
    """

    @pytest.fixture(autouse=True)
    def _unloaded_map(self) -> None:
        """Start every test with no map parsed.

        The parse cache is process-global on purpose -- a map takes seconds and
        a session edits one -- so whether the panel binds a pattern depends on
        what ran *before* it. Without this, a test asserting on the id a
        document stores passed or failed on the order pytest happened to pick.
        """
        from autoware_carla_scenario.editor import map_preview

        map_preview.clear_cache()

    def test_the_page_opens_with_the_places_on_it(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The first thing on screen already says where the scenario happens."""
        body = client.get(f"/draft/{draft_id}").text
        panel = body.split('id="scenario-map"', 1)[1].split('id="editor-body"', 1)[0]

        assert 'data-viewer-key="scenario-map"' in panel
        # The ego's spawn and goal, the NPC's spawn, and the lanelet the PASS
        # condition watches -- every slot the document holds, not just spawns.
        for label in ("Ego", "NPC1", "Position (Lanelet2)"):
            assert label in panel
        assert 'data-lanelet="183"' in panel  # the ego's spawn
        assert 'data-lanelet="141"' in panel  # its goal

    def test_the_map_outlines_every_place_at_once(
        self, client: TestClient, draft_id: str
    ) -> None:
        highlight = (
            client.get(f"/draft/{draft_id}/map-view")
            .text.split('data-highlight="', 1)[1]
            .split('"', 1)[0]
        )
        assert set(highlight.split(",")) == {"183", "141", "184"}

    def test_clicking_a_place_opens_what_named_it(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The panel is read-only: a click routes to the object, not the id.

        Each place is one row carrying its lanelet and the request that selects
        what named it -- which is also what a click on the map reads, so the
        mapping is not sent a second time.
        """
        body = client.get(f"/draft/{draft_id}/map-view").text

        for lanelet, owner in ((183, "ego"), (184, "npc1")):
            row = body.split(f'data-focus-lanelet="{lanelet}"', 1)[1].split(">", 1)[0]
            assert f"/draft/{draft_id}/inspector/{owner}" in row
        # And no picking: a place is edited where it is written.
        assert "data-picks-into" not in body

    def test_an_unloaded_map_offers_to_bind_rather_than_binding(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Binding parses a city, so it is asked for rather than assumed."""
        body = client.get(f"/draft/{draft_id}/map-view").text

        assert "Bind a pattern" in body
        assert "searched, not bound yet" in body
        # The default the document carries is still drawn: that is what a run
        # without a sweep would use.
        assert 'data-lanelet="184"' in body

    def test_binding_draws_one_of_the_runs_a_sweep_would_perform(
        self, client: TestClient, draft_id: str
    ) -> None:
        body = client.get(f"/draft/{draft_id}/map-view?load_map=1").text

        assert "Pattern <b>1</b> of" in body
        assert "bound: match 1 of" in body
        # The bound lanelet is a match of the search, not the stored default.
        highlight = body.split('data-highlight="', 1)[1].split('"', 1)[0]
        assert "183" in highlight and "141" in highlight

    def test_stepping_the_pattern_binds_a_different_lanelet(
        self, client: TestClient, draft_id: str
    ) -> None:
        """Each step is one more of the runs the sweeper would enumerate."""
        client.get(f"/draft/{draft_id}/map-view?load_map=1")
        first = client.get(f"/draft/{draft_id}/map-view?pattern=0").text
        third = client.get(f"/draft/{draft_id}/map-view?pattern=2").text

        assert "Pattern <b>3</b> of" in third
        bound_first = first.split('data-highlight="', 1)[1].split('"', 1)[0]
        bound_third = third.split('data-highlight="', 1)[1].split('"', 1)[0]
        assert bound_first != bound_third

    def test_the_pattern_survives_an_edit_elsewhere(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The panel refreshes itself from a URL carrying what it is showing."""
        client.get(f"/draft/{draft_id}/map-view?load_map=1")
        body = client.get(f"/draft/{draft_id}/map-view?pattern=4").text

        assert f'hx-get="/draft/{draft_id}/map-view?pattern=4"' in body
        assert 'hx-trigger="scenario-changed from:body"' in body

    def test_a_pattern_past_the_last_wraps_to_the_first(
        self, client: TestClient, draft_id: str
    ) -> None:
        """The arrows can be held down without falling off the end."""
        client.get(f"/draft/{draft_id}/map-view?load_map=1")
        first = client.get(f"/draft/{draft_id}/map-view?pattern=0").text
        count = int(first.split("Pattern <b>1</b> of ", 1)[1].split("\n", 1)[0].strip())
        wrapped = client.get(f"/draft/{draft_id}/map-view?pattern={count}").text

        assert "Pattern <b>1</b> of" in wrapped

    def test_a_scenario_with_no_map_file_says_so(
        self, client: TestClient, draft_id: str
    ) -> None:
        """An empty box where a map should be is worse than no box."""
        client.post(f"/draft/{draft_id}/scenario", data={"map_lanelet2_path": ""})
        body = client.get(f"/draft/{draft_id}/map-view").text

        assert "No map" in body
        assert "ed-map-frame" not in body
        # The places are still listed: they are what the document says.
        assert "Ego" in body

    @staticmethod
    def _count_map_scans(monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Count how often a search is really walked against the map.

        The number is the point of the memo: matching visits every lanelet, and
        a real map is many times the size of this fixture's.
        """
        from autoware_carla_scenario.sweeper import constraints

        scans: list[int] = []
        real = constraints.find_matching_lanelets

        def counted(*args: Any, **kwargs: Any) -> Any:
            scans.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(constraints, "find_matching_lanelets", counted)
        return scans

    def test_an_edit_that_is_not_the_search_does_not_re_scan_the_map(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient, draft_id: str
    ) -> None:
        """The panel re-renders on every edit; the city is walked once."""
        scans = self._count_map_scans(monkeypatch)

        client.get(f"/draft/{draft_id}/map-view?load_map=1")
        assert len(scans) == 1

        client.post(f"/draft/{draft_id}/scenario", data={"title": "Renamed"})
        client.get(f"/draft/{draft_id}/map-view")
        assert len(scans) == 1, "an unrelated edit re-scanned the map"

    def test_the_picker_shares_that_answer(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient, draft_id: str
    ) -> None:
        """Choosing a spawn or a goal in the inspector asks the same question.

        The picker's match readout and the panel are two routes onto one
        search, so the answer is remembered for both -- and the memo is keyed on
        the constraint tree rather than on the slot, which is what makes that
        true even for two slots searching for the same thing.
        """
        scans = self._count_map_scans(monkeypatch)

        client.post(
            f"/draft/{draft_id}/lanelet-preview",
            data={"slot": "npc1.spawn", "load_map": "1"},
        )
        assert len(scans) == 1

        # Re-opening the picker, and the panel drawing the same search.
        client.post(f"/draft/{draft_id}/lanelet-preview", data={"slot": "npc1.spawn"})
        client.get(f"/draft/{draft_id}/map-view")
        assert len(scans) == 1, "the picker and the panel each walked the map"

    def test_editing_the_search_does_re_scan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: TestClient,
        store: DraftStore,
        draft_id: str,
    ) -> None:
        """The memo is keyed on the question, so a new question is asked."""
        scans = self._count_map_scans(monkeypatch)

        client.get(f"/draft/{draft_id}/map-view?load_map=1")
        length = _entity(store, draft_id, "npc1").spawn.constraints[0].constraints[1]
        client.post(
            f"/draft/{draft_id}/constraint/{length.id}",
            data={"rule": "greater_than_or_equal", "value": "30", "selected": "npc1"},
        )
        client.get(f"/draft/{draft_id}/map-view")

        assert len(scans) == 2

    def test_the_script_places_the_pins_it_is_handed(self) -> None:
        """The labels are the server's; only the coordinates are the viewer's."""
        script = (Path(editor_app.__file__).parent / "static" / "editor.js").read_text()
        assert "function layoutPins(" in script
        # Positions come from the viewer's own API rather than a second
        # projection of the .osm, which could disagree with the drawing.
        assert "focusOn(id)" in script
        assert "getView()" in script
