"""Waypoint conditions on a *Follow Trajectory* card: the IR, its checks, its build.

A card's ``advance_conditions`` put an ordinary condition tree on a vertex of
its trajectory (counted from 1, as the editor lists them).  These tests follow
one from the document -- and the editor that writes it -- through the
validator to the runtime action, whose vertex then carries the condition the
same builders make for a trigger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from autoware_carla_scenario import (
    AndCondition,
    ElapsedTimeCondition,
    EntityDistanceCondition,
    FollowTrajectoryAction,
    RelativeLanePose,
)
from autoware_carla_scenario.authoring.builders import instantiate_action
from autoware_carla_scenario.authoring.compiler import (
    BuildContext,
    CompilationError,
    compile_document,
)
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    Assertions,
    ConditionNode,
    Entity,
    ScenarioDocument,
    SpawnSpec,
    VertexCondition,
)
from autoware_carla_scenario.authoring.persistence import (
    DraftStore,
    load_document,
    save_document,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.editor.app import create_app
from autoware_carla_scenario.editor.service import EditorService

_ROWS = [
    [100.0, 200.0, None, 0.0],
    [110.0, 200.0, None, 1.0],
    [120.0, 200.0, None, 2.0],
]
_RELATIVE_ROWS = [
    [0.0, 0.0, 0, None, 0.0],
    [20.0, 0.0, 1, None, 2.0],
    [40.0, 0.0, 1, None, 4.0],
]


def _wait(seconds: float = 5.0) -> ConditionNode:
    return ConditionNode(type="elapsed_time", params={"duration_seconds": seconds})


def _document(
    gates: list[VertexCondition],
    params: dict[str, Any] | None = None,
    action_type: str = "follow_trajectory",
) -> ScenarioDocument:
    return ScenarioDocument(
        id="s",
        entities=[
            Entity(id="ego", kind="ego", spawn=SpawnSpec(lanelet_id=1)),
            Entity(id="npc1", kind="vehicle", spawn=SpawnSpec(lanelet_id=2)),
        ],
        actions=[
            ActionNode(
                id="a1",
                type=action_type,
                actor="npc1",
                params=(
                    {"path_source": "vertices", "vertices": _ROWS}
                    if params is None
                    else params
                ),
                advance_conditions=gates,
            )
        ],
        assertions=Assertions.model_validate({"pass": [_wait()]}),
    )


def _errors(document: ScenarioDocument) -> list[str]:
    return [issue.message for issue in validate_document(document).errors]


def _built(document: ScenarioDocument, action_id: str = "a1") -> FollowTrajectoryAction:
    compiled = compile_document(document)
    (card,) = [a for a in compiled.actions if a.node.id == action_id]
    action = instantiate_action(card, BuildContext(scenario=None))
    assert isinstance(action, FollowTrajectoryAction)
    return action


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


class TestTheDocument:
    def test_it_round_trips_through_yaml(self, tmp_path: Path) -> None:
        document = _document([VertexCondition(vertex=2, condition=_wait(3.0))])
        path = save_document(document, tmp_path / "s.yaml")
        assert "advance_conditions" in path.read_text()
        loaded = load_document(path)
        assert loaded == document
        (gate,) = loaded.actions[0].advance_conditions
        assert gate.vertex == 2
        assert gate.condition.params["duration_seconds"] == 3.0

    def test_an_older_document_without_any_still_loads(self) -> None:
        raw = _document([]).to_yaml_dict()
        del raw["actions"][0]["advance_conditions"]
        assert ScenarioDocument.model_validate(raw).actions[0].advance_conditions == []

    def test_vertices_are_counted_from_one(self) -> None:
        with pytest.raises(ValidationError):
            VertexCondition(vertex=0, condition=_wait())

    def test_the_condition_is_found_like_any_other(self) -> None:
        gate = VertexCondition(vertex=2, condition=_wait())
        document = _document([gate])
        assert document.condition(gate.condition.id) is gate.condition
        owner = document.vertex_condition_of(gate.condition.id)
        assert owner is not None and owner[1] is gate
        assert (
            document.vertex_condition_of(document.assertions.pass_conditions[0].id)
            is None
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_a_gate_on_an_inner_vertex_is_fine(self) -> None:
        assert _errors(_document([VertexCondition(vertex=2, condition=_wait())])) == []
        assert _errors(_document([VertexCondition(vertex=1, condition=_wait())])) == []

    def test_not_on_the_last_vertex(self) -> None:
        errors = _errors(_document([VertexCondition(vertex=3, condition=_wait())]))
        assert any("Vertex 3 is the last one" in e for e in errors)

    def test_not_past_the_last_vertex(self) -> None:
        errors = _errors(_document([VertexCondition(vertex=7, condition=_wait())]))
        assert any("There is no vertex 7: the trajectory has 3" in e for e in errors)

    def test_one_per_vertex(self) -> None:
        errors = _errors(
            _document(
                [
                    VertexCondition(vertex=2, condition=_wait()),
                    VertexCondition(vertex=2, condition=_wait(1.0)),
                ]
            )
        )
        assert any("two waypoint conditions" in e for e in errors)

    def test_not_on_a_lanelet_path(self) -> None:
        document = _document(
            [VertexCondition(vertex=2, condition=_wait())],
            {"path_source": "lanelets", "lanelet_ids": [1, 2], "speed_kmh": 30.0},
        )
        assert any(
            "lanelet path's vertices are generated" in e for e in _errors(document)
        )

    def test_the_condition_is_checked_as_a_trigger_is(self) -> None:
        unknown = _document(
            [VertexCondition(vertex=2, condition=ConditionNode(type="no_such_thing"))]
        )
        assert any(
            "Unknown condition type 'no_such_thing'" in e for e in _errors(unknown)
        )
        nobody = _document(
            [
                VertexCondition(
                    vertex=2,
                    condition=ConditionNode(
                        type="entity_distance",
                        params={"source": "npc1", "target": "ghost"},
                    ),
                )
            ]
        )
        assert any("ghost" in e for e in _errors(nobody))
        empty_all = _document(
            [VertexCondition(vertex=2, condition=ConditionNode(type="all"))]
        )
        assert any("needs at least" in e for e in _errors(empty_all))

    def test_an_action_without_vertices_takes_none(self) -> None:
        document = _document(
            [VertexCondition(vertex=1, condition=_wait())],
            {"target_speed_kmh": 20.0},
            action_type="set_speed",
        )
        assert any("takes no waypoint conditions" in e for e in _errors(document))

    def test_relative_vertices_are_counted_the_same(self) -> None:
        params = {"path_source": "relative_lane", "relative_vertices": _RELATIVE_ROWS}
        assert (
            _errors(_document([VertexCondition(vertex=2, condition=_wait())], params))
            == []
        )
        errors = _errors(
            _document([VertexCondition(vertex=3, condition=_wait())], params)
        )
        assert any("is the last one" in e for e in errors)

    def test_an_invalid_gate_blocks_compilation(self) -> None:
        with pytest.raises(CompilationError):
            compile_document(_document([VertexCondition(vertex=3, condition=_wait())]))


# ---------------------------------------------------------------------------
# What it builds
# ---------------------------------------------------------------------------


class TestBuild:
    def test_a_gate_becomes_the_vertex_advance_condition(self) -> None:
        action = _built(_document([VertexCondition(vertex=2, condition=_wait(7.0))]))
        vertices = action.trajectory.vertices
        assert vertices[0].advance is None and vertices[2].advance is None
        advance = vertices[1].advance
        assert isinstance(advance, ElapsedTimeCondition)
        assert advance.duration_seconds == 7.0
        assert action.trajectory.is_gated

    def test_compositions_and_entity_references_build_as_for_a_trigger(self) -> None:
        gate = VertexCondition(
            vertex=1,
            condition=ConditionNode(
                type="all",
                children=[
                    _wait(2.0),
                    ConditionNode(
                        type="entity_distance",
                        params={
                            "source": "npc1",
                            "target": "ego",
                            "rule": "greater_than",
                            "distance": 30.0,
                        },
                    ),
                ],
            ),
        )
        compiled = compile_document(_document([gate]))
        (card,) = [a for a in compiled.actions if a.node.id == "a1"]
        ((index, condition),) = card.advance_conditions
        assert index == 0  # the document's vertex 1
        # Entity ids are resolved to the roles the runtime knows.
        assert condition.children[1].params["target"] == "Ego"
        action = instantiate_action(card, BuildContext(scenario=None))
        assert isinstance(action, FollowTrajectoryAction)
        advance = action.trajectory.vertices[0].advance
        assert isinstance(advance, AndCondition)
        assert isinstance(advance._conditions[1], EntityDistanceCondition)

    def test_a_relative_card_carries_its_gates(self) -> None:
        params = {"path_source": "relative_lane", "relative_vertices": _RELATIVE_ROWS}
        action = _built(
            _document([VertexCondition(vertex=2, condition=_wait())], params)
        )
        vertex = action.trajectory.vertices[1]
        assert isinstance(vertex.position, RelativeLanePose)
        assert isinstance(vertex.advance, ElapsedTimeCondition)

    def test_without_gates_the_trajectory_is_ungated(self) -> None:
        assert not _built(_document([])).trajectory.is_gated


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------


def _stored(store: DraftStore, draft_id: str) -> ScenarioDocument:
    draft = store.get(draft_id)
    assert draft is not None
    return draft.document


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(draft_dir=tmp_path / "drafts", export_dir=tmp_path / "packages")
    )


def _card(client: TestClient, tmp_path: Path) -> tuple[str, str, DraftStore]:
    """A new draft with a vertex Follow Trajectory card: (draft, card, store)."""
    response = client.post(
        "/new", data={"kind": "cut_in", "title": "Cut in"}, follow_redirects=False
    )
    draft_id = response.headers["location"].rsplit("/", 1)[-1]
    store = DraftStore(tmp_path / "drafts")
    actor = next(e.id for e in _stored(store, draft_id).entities if e.kind != "ego")
    client.post(
        f"/draft/{draft_id}/action",
        data={"type_id": "follow_trajectory", "actor": actor},
    )
    card = next(
        a for a in _stored(store, draft_id).actions if a.type == "follow_trajectory"
    )
    client.post(
        f"/draft/{draft_id}/action/{card.id}",
        data={
            "path_source": "vertices",
            "vertices": "100, 200, , 0\n110, 200, , 1\n120, 200, , 2\n130, 200, , 3",
            "time_domain": "relative",
            "time_scale": "1",
            "time_offset": "0",
            "following_mode": "position",
            "initial_distance_offset": "0",
            "speed_kmh": "30",
            "lateral_offset_m": "0",
        },
    )
    return draft_id, card.id, store


class TestEditor:
    def test_a_waypoint_condition_is_added_edited_and_shown(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        draft_id, card_id, store = _card(client, tmp_path)
        inspector = client.get(f"/draft/{draft_id}/inspector/{card_id}").text
        assert "Waypoint conditions" in inspector
        assert f'value="advance:{card_id}"' in inspector

        page = client.post(
            f"/draft/{draft_id}/condition",
            data={
                "slot": f"advance:{card_id}",
                "vertex": "2",
                "type_id": "elapsed_time",
            },
        )
        assert page.status_code == 200
        card = _stored(store, draft_id).action(card_id)
        assert card is not None
        (gate,) = card.advance_conditions
        assert gate.vertex == 2 and gate.condition.type == "elapsed_time"

        # Its parameters are edited in the ordinary condition inspector.
        client.post(
            f"/draft/{draft_id}/condition/{gate.condition.id}",
            data={"rule": "greater_than_or_equal", "duration_seconds": "12"},
        )
        card = _stored(store, draft_id).action(card_id)
        assert card is not None
        assert card.advance_conditions[0].condition.params["duration_seconds"] == 12.0
        condition_page = client.get(
            f"/draft/{draft_id}/inspector/{gate.condition.id}"
        ).text
        assert "Waypoint condition: holds the entity" in condition_page
        assert "at vertex 2" in condition_page

        inspector = client.get(f"/draft/{draft_id}/inspector/{card_id}").text
        assert "leave vertex" in inspector
        assert gate.condition.id in inspector
        body = client.get(f"/draft/{draft_id}").text
        assert "1 waypoint condition" in body
        assert validate_document(_stored(store, draft_id)).ok

        # And it reaches the runtime.
        action = _built(_stored(store, draft_id), card_id)
        assert isinstance(action.trajectory.vertices[1].advance, ElapsedTimeCondition)

    def test_a_second_condition_on_the_same_vertex_is_anded(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        draft_id, card_id, store = _card(client, tmp_path)
        for type_id in ("elapsed_time", "always_true"):
            client.post(
                f"/draft/{draft_id}/condition",
                data={"slot": f"advance:{card_id}", "vertex": "2", "type_id": type_id},
            )
        card = _stored(store, draft_id).action(card_id)
        assert card is not None
        (gate,) = card.advance_conditions
        assert gate.condition.type == "all"
        assert [c.type for c in gate.condition.children] == [
            "elapsed_time",
            "always_true",
        ]

    def test_a_gate_moves_to_another_vertex_but_not_onto_a_taken_one(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        draft_id, card_id, store = _card(client, tmp_path)
        for vertex in ("1", "2"):
            client.post(
                f"/draft/{draft_id}/condition",
                data={
                    "slot": f"advance:{card_id}",
                    "vertex": vertex,
                    "type_id": "always_true",
                },
            )
        client.post(
            f"/draft/{draft_id}/action/{card_id}/advance/1", data={"vertex": "3"}
        )
        card = _stored(store, draft_id).action(card_id)
        assert card is not None
        assert [g.vertex for g in card.advance_conditions] == [1, 3]

        page = client.post(
            f"/draft/{draft_id}/action/{card_id}/advance/1", data={"vertex": "1"}
        )
        assert "already has a waypoint condition" in page.text
        card = _stored(store, draft_id).action(card_id)
        assert card is not None
        assert [g.vertex for g in card.advance_conditions] == [1, 3]

    def test_the_last_vertex_is_a_validation_error_in_the_editor(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        draft_id, card_id, store = _card(client, tmp_path)
        page = client.post(
            f"/draft/{draft_id}/condition",
            data={
                "slot": f"advance:{card_id}",
                "vertex": "4",
                "type_id": "always_true",
            },
        )
        assert "is the last one" in page.text
        assert not validate_document(_stored(store, draft_id)).ok

    def test_removing_a_gate(self, client: TestClient, tmp_path: Path) -> None:
        draft_id, card_id, store = _card(client, tmp_path)
        for vertex in ("1", "2"):
            client.post(
                f"/draft/{draft_id}/condition",
                data={
                    "slot": f"advance:{card_id}",
                    "vertex": vertex,
                    "type_id": "always_true",
                },
            )
        card = _stored(store, draft_id).action(card_id)
        assert card is not None
        root = card.advance_conditions[0].condition.id
        # Deleting the whole condition deletes the waypoint condition...
        client.post(f"/draft/{draft_id}/condition/{root}/delete")
        card = _stored(store, draft_id).action(card_id)
        assert card is not None
        assert [g.vertex for g in card.advance_conditions] == [2]
        # ...and so does the gate's own remove button.
        client.post(f"/draft/{draft_id}/action/{card_id}/advance/0/delete")
        card = _stored(store, draft_id).action(card_id)
        assert card is not None
        assert card.advance_conditions == []

    def test_an_action_without_vertices_refuses_one(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        draft_id, _, store = _card(client, tmp_path)
        other = next(
            a for a in _stored(store, draft_id).actions if a.type != "follow_trajectory"
        )
        page = client.post(
            f"/draft/{draft_id}/condition",
            data={
                "slot": f"advance:{other.id}",
                "vertex": "1",
                "type_id": "always_true",
            },
        )
        assert "takes no waypoint conditions" in page.text

    def test_deleting_an_entity_a_gate_names_drops_the_gate(
        self, tmp_path: Path
    ) -> None:
        gate = VertexCondition(
            vertex=2,
            condition=ConditionNode(
                type="entity_distance", params={"source": "npc1", "target": "npc2"}
            ),
        )
        document = _document([gate, VertexCondition(vertex=1, condition=_wait())])
        document.entities.append(Entity(id="npc2", spawn=SpawnSpec(lanelet_id=3)))
        assert _errors(document) == []
        EditorService(DraftStore(tmp_path / "drafts")).delete_entity(document, "npc2")
        card = document.action("a1")
        assert card is not None
        assert [g.vertex for g in card.advance_conditions] == [1]
        assert _errors(document) == []
