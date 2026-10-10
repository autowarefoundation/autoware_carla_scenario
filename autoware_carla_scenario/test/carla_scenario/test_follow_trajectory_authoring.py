"""Authoring a *Follow Trajectory* card: the text, the checks and what it builds.

The card keeps its path in the document -- map-frame vertices, a route of
lanelets, or vertices relative to an entity's lane -- so these tests follow a card from the editor's form, through the
validator, to the runtime action its builder returns.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario import (
    FollowTrajectoryAction,
    Lanelet2Pose,
    MapPose,
    ReferenceContext,
    RelativeLanePose,
    TrajectoryFollowingMode,
)
from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    Assertions,
    ConditionNode,
    Entity,
    ScenarioDocument,
    SpawnSpec,
)
from autoware_carla_scenario.authoring.persistence import DraftStore
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.editor.app import create_app
from autoware_carla_scenario.trajectory.authoring import (
    authored_timing,
    authored_trajectory,
    format_relative_vertices,
    format_vertices,
    parse_relative_vertices,
    parse_vertices,
    relative_trajectory_summary,
    trajectory_summary,
)

_ROWS = [[100.0, 200.0, 0.5, 0.0], [110.0, 200.0, None, 1.0], [120.0, 200.0, None, 2.0]]


def _document(
    params: dict[str, Any],
    *,
    hidden: bool = False,
    kind: Literal["vehicle", "pedestrian"] = "vehicle",
) -> ScenarioDocument:
    return ScenarioDocument(
        id="s",
        entities=[
            Entity(id="ego", kind="ego", spawn=SpawnSpec(lanelet_id=1)),
            Entity(id="npc1", kind=kind, spawn=SpawnSpec(lanelet_id=2, hidden=hidden)),
        ],
        actions=[
            ActionNode(id="a1", type="follow_trajectory", actor="npc1", params=params)
        ],
        assertions=Assertions.model_validate(
            {
                "pass": [
                    ConditionNode(type="elapsed_time", params={"duration_seconds": 5.0})
                ]
            }
        ),
    )


def _messages(document: ScenarioDocument, severity: str = "error") -> list[str]:
    report = validate_document(document)
    issues = report.errors if severity == "error" else report.warnings
    return [issue.message for issue in issues]


# ---------------------------------------------------------------------------
# Vertices as text
# ---------------------------------------------------------------------------


class TestVertexText:
    def test_one_vertex_per_line_with_optional_cells(self) -> None:
        rows = parse_vertices("100, 200, 0.5, 0\n# a comment\n\n110, 200, , 1\n120,200")
        assert rows == [
            (100.0, 200.0, 0.5, 0.0),
            (110.0, 200.0, None, 1.0),
            (120.0, 200.0, None, None),
        ]

    def test_rows_and_mappings_are_read_too(self) -> None:
        assert parse_vertices([[1, 2], {"x": 3, "y": 4, "time": 5}]) == [
            (1.0, 2.0, None, None),
            (3.0, 4.0, None, 5.0),
        ]

    def test_a_bad_line_is_named(self) -> None:
        with pytest.raises(ValueError, match="vertex 2"):
            parse_vertices("1, 2\n3, north")
        with pytest.raises(ValueError, match="vertex 1"):
            parse_vertices("1")

    def test_the_text_reads_back_the_same(self) -> None:
        text = format_vertices(_ROWS)
        assert text.splitlines()[1] == "110, 200, , 1"
        assert [list(row) for row in parse_vertices(text)] == _ROWS

    def test_map_coordinates_keep_their_precision(self) -> None:
        text = format_vertices([[81234.567, 50123.125]])
        assert parse_vertices(text) == [(81234.567, 50123.125, None, None)]

    def test_the_summary(self) -> None:
        assert trajectory_summary(_ROWS) == "3 vertices, 20.0 m, 2.0 s"
        assert trajectory_summary(None) == "no vertices"


# ---------------------------------------------------------------------------
# What the builder calls
# ---------------------------------------------------------------------------


class TestFactories:
    def test_vertices_become_map_poses(self) -> None:
        trajectory = authored_trajectory("vertices", vertices=_ROWS)
        first = trajectory.vertices[0]
        assert first.position == MapPose(100.0, 200.0, 0.5)
        assert first.time == 0.0
        assert trajectory.is_timed

    def test_a_lanelet_path_follows_the_centrelines_at_a_speed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "autoware_carla_scenario.coordinate.transform.lanelet_length",
            lambda lanelet_id: {1: 10.0, 2: 4.0}[lanelet_id],
        )
        trajectory = authored_trajectory(
            "lanelets", lanelet_ids=[1, 2], speed_kmh=36.0, lateral_offset_m=0.5
        )
        positions = [vertex.position for vertex in trajectory.vertices]
        assert positions[0] == Lanelet2Pose(1, 0.0, 0.5)
        assert positions[-1] == Lanelet2Pose(2, 4.0, 0.5)
        # Lanelet 2 starts where lanelet 1 ends: that point is taken once.
        assert Lanelet2Pose(2, 0.0, 0.5) not in positions
        # 14 m at 10 m/s.
        assert trajectory.vertices[-1].time == pytest.approx(1.4)

    def test_a_lanelet_path_without_a_speed_is_untimed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "autoware_carla_scenario.coordinate.transform.lanelet_length",
            lambda lanelet_id: 10.0,
        )
        assert not authored_trajectory(
            "lanelets", lanelet_ids=[1], speed_kmh=0
        ).is_timed

    def test_timing(self) -> None:
        assert authored_timing("none") is None
        timing = authored_timing("absolute", 2.0, 1.0)
        assert timing is not None
        assert timing.domain is ReferenceContext.ABSOLUTE
        assert (timing.scale, timing.offset) == (2.0, 1.0)


# ---------------------------------------------------------------------------
# The document checks
# ---------------------------------------------------------------------------


class TestValidation:
    def test_a_good_card_has_no_errors(self) -> None:
        assert (
            _messages(_document({"path_source": "vertices", "vertices": _ROWS})) == []
        )

    def test_too_few_vertices(self) -> None:
        document = _document({"path_source": "vertices", "vertices": _ROWS[:1]})
        assert any("at least two" in m for m in _messages(document))

    def test_a_time_reference_needs_times(self) -> None:
        document = _document({"path_source": "vertices", "vertices": [[0, 0], [1, 0]]})
        assert any("needs a time on every vertex" in m for m in _messages(document))
        untimed = _document(
            {
                "path_source": "vertices",
                "vertices": [[0, 0], [1, 0]],
                "time_domain": "none",
            }
        )
        assert _messages(untimed) == []

    def test_unparseable_vertices_are_a_field_error(self) -> None:
        document = _document({"path_source": "vertices", "vertices": "1, 2\n3, x"})
        assert any(m.startswith("Vertices: vertex 2") for m in _messages(document))

    def test_a_lanelet_path_needs_lanelets_and_a_speed(self) -> None:
        document = _document({"path_source": "lanelets", "speed_kmh": 0})
        messages = _messages(document)
        assert any("at least one lanelet" in m for m in messages)
        assert any("needs a speed" in m for m in messages)

    def test_hiding_needs_a_timed_position_replay(self) -> None:
        document = _document(
            {
                "path_source": "vertices",
                "vertices": _ROWS,
                "following_mode": "follow",
                "hidden_outside_trajectory": True,
            }
        )
        assert any("Position mode" in m for m in _messages(document))

    def test_a_pedestrian_may_follow_a_trajectory(self) -> None:
        document = _document(
            {"path_source": "vertices", "vertices": _ROWS}, kind="pedestrian"
        )
        assert _messages(document) == []

    def test_a_hidden_spawn_nothing_brings_in_is_warned_about(self) -> None:
        document = _document(
            {"path_source": "vertices", "vertices": _ROWS}, hidden=True
        )
        assert any("stays there" in m for m in _messages(document, "warning"))
        brought_in = _document(
            {
                "path_source": "vertices",
                "vertices": _ROWS,
                "hidden_outside_trajectory": True,
            },
            hidden=True,
        )
        assert not any("stays there" in m for m in _messages(brought_in, "warning"))

    def test_the_ego_never_spawns_hidden(self) -> None:
        document = ScenarioDocument(
            id="s",
            entities=[
                Entity(id="ego", kind="ego", spawn=SpawnSpec(lanelet_id=1, hidden=True))
            ],
        )
        assert any("ego cannot spawn" in m for m in _messages(document))


# ---------------------------------------------------------------------------
# What it builds
# ---------------------------------------------------------------------------


class TestBuild:
    def test_the_card_builds_the_runtime_action(self) -> None:
        from autoware_carla_scenario.actions.base import TickTiming
        from autoware_carla_scenario.authoring.builders import instantiate_action
        from autoware_carla_scenario.authoring.compiler import BuildContext

        document = _document(
            {
                "path_source": "vertices",
                "vertices": format_vertices(_ROWS),  # text, as a hand edit may leave it
                "time_domain": "absolute",
                "time_scale": 2.0,
                "following_mode": "follow",
                "initial_distance_offset": 3.0,
            }
        )
        compiled = compile_document(document)
        (card,) = [a for a in compiled.actions if a.node.id == "a1"]
        action = instantiate_action(card, BuildContext(scenario=None))
        assert isinstance(action, FollowTrajectoryAction)
        assert action.timing is TickTiming.PRE_TICK
        assert len(action.trajectory.vertices) == 3
        assert action.trajectory.vertices[0].position == MapPose(100.0, 200.0, 0.5)
        assert action._time_reference is not None
        assert action._time_reference.domain is ReferenceContext.ABSOLUTE
        assert action._time_reference.scale == 2.0
        assert action._following_mode is TrajectoryFollowingMode.FOLLOW
        assert math.isclose(action._initial_distance_offset, 3.0)


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


class TestEditor:
    def test_a_card_is_written_in_the_inspector(
        self, client: TestClient, tmp_path: Path
    ) -> None:
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
        document = _stored(store, draft_id)
        card = next(a for a in document.actions if a.type == "follow_trajectory")

        page = client.post(
            f"/draft/{draft_id}/action/{card.id}",
            data={
                "path_source": "vertices",
                "vertices": "100, 200, 0.5, 0\n110, 200, , 1\n120, 200, , 2",
                "time_domain": "absolute",
                "time_scale": "1",
                "time_offset": "0",
                "following_mode": "position",
                "initial_distance_offset": "0",
                "hidden_outside_trajectory": "on",
                "speed_kmh": "30",
                "lateral_offset_m": "0",
            },
        )
        assert page.status_code == 200
        stored = _stored(store, draft_id).action(card.id)
        assert stored is not None
        assert stored.params["vertices"] == _ROWS
        assert stored.params["hidden_outside_trajectory"] is True

        inspector = client.get(f"/draft/{draft_id}/inspector/{card.id}").text
        assert "3 vertices, 20.0 m, 2.0 s" in inspector
        assert "110, 200, , 1" in inspector

    def test_an_entity_is_set_to_spawn_hidden(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        response = client.post(
            "/new", data={"kind": "cut_in", "title": "Cut in"}, follow_redirects=False
        )
        draft_id = response.headers["location"].rsplit("/", 1)[-1]
        store = DraftStore(tmp_path / "drafts")
        actor = next(e.id for e in _stored(store, draft_id).entities if e.kind != "ego")

        client.post(
            f"/draft/{draft_id}/entity/{actor}",
            data={"spawn_hidden_shown": "1", "spawn_hidden": "on"},
        )
        entity = _stored(store, draft_id).entity(actor)
        assert entity is not None and entity.spawn.hidden

        client.post(
            f"/draft/{draft_id}/entity/{actor}", data={"spawn_hidden_shown": "1"}
        )
        entity = _stored(store, draft_id).entity(actor)
        assert entity is not None and not entity.spawn.hidden


class TestHiddenSpawn:
    def test_a_hidden_entity_is_spawned_under_the_map(self) -> None:
        import typesafe_carla.carla as carla

        from autoware_carla_scenario.actions.follow_trajectory import HIDDEN_DEPTH_M
        from autoware_carla_scenario.declarative import _hidden_if_asked

        at = carla.Transform(
            carla.Location(x=1.0, y=2.0, z=3.0), carla.Rotation(yaw=90.0)
        )
        hidden = Entity(id="npc1", spawn=SpawnSpec(lanelet_id=2, hidden=True))
        shown = Entity(id="npc2", spawn=SpawnSpec(lanelet_id=2))

        moved = _hidden_if_asked(hidden, at)
        assert (moved.location.x, moved.location.y) == (1.0, 2.0)
        assert moved.location.z == pytest.approx(3.0 - HIDDEN_DEPTH_M)
        assert moved.rotation.yaw == pytest.approx(90.0)
        assert _hidden_if_asked(shown, at) is at


# ---------------------------------------------------------------------------
# Lane-relative vertices
# ---------------------------------------------------------------------------

#: Pull out one lane to the left over 20 m and drive on, at 10 m/s.
_RELATIVE_ROWS = [
    [0.0, 0.0, 0, None, 0.0],
    [20.0, 0.0, 1, None, 2.0],
    [40.0, -0.5, 1, 0.1, 4.0],
]
_RELATIVE_TEXT = "0, 0, 0, , 0\n20, 0, 1, , 2\n40, -0.5, 1, 0.1, 4"


def _relative(params: dict[str, Any]) -> dict[str, Any]:
    return {"path_source": "relative_lane", **params}


class TestRelativeVertexText:
    def test_only_ds_is_required(self) -> None:
        rows = parse_relative_vertices("10\n# a comment\n\n20, 0.5\n30, , -1, , 3")
        assert rows == [
            (10.0, 0.0, 0, None, None),
            (20.0, 0.5, 0, None, None),
            (30.0, 0.0, -1, None, 3.0),
        ]

    def test_rows_and_mappings_are_read_too(self) -> None:
        assert parse_relative_vertices(
            [[1, 2, 1], {"ds": -3, "d_lane": -2, "time": 5}]
        ) == [(1.0, 2.0, 1, None, None), (-3.0, 0.0, -2, None, 5.0)]

    def test_d_lane_is_a_whole_number(self) -> None:
        with pytest.raises(ValueError, match="vertex 2: d_lane is a number of lanes"):
            parse_relative_vertices("1, 0, 1\n2, 0, 1.5")

    def test_a_bad_line_is_named(self) -> None:
        with pytest.raises(ValueError, match="vertex 1: ds is required"):
            parse_relative_vertices(", 1")
        with pytest.raises(ValueError, match="vertex 2: 'left' is not a number"):
            parse_relative_vertices("1\n2, 0, left")
        with pytest.raises(ValueError, match="got 6 values"):
            parse_relative_vertices("1, 2, 3, 4, 5, 6")

    def test_the_text_reads_back_the_same(self) -> None:
        text = format_relative_vertices(_RELATIVE_ROWS)
        assert text == _RELATIVE_TEXT
        assert [list(row) for row in parse_relative_vertices(text)] == _RELATIVE_ROWS

    def test_the_summary(self) -> None:
        assert (
            relative_trajectory_summary(_RELATIVE_ROWS)
            == "3 vertices, ds 0 to 40 m, lanes +0, +1, 4.0 s"
        )
        assert (
            relative_trajectory_summary([[5.0], [10.0]]) == "2 vertices, ds 5 to 10 m"
        )
        assert relative_trajectory_summary(None) == "no vertices"


class TestRelativeFactory:
    def test_rows_become_relative_lane_poses(self) -> None:
        trajectory = authored_trajectory(
            "relative_lane", relative_vertices=_RELATIVE_ROWS, reference_entity="Ego"
        )
        positions = [vertex.position for vertex in trajectory.vertices]
        assert positions[1] == RelativeLanePose(20.0, 0.0, 1, entity_ref="Ego")
        assert positions[2] == RelativeLanePose(40.0, -0.5, 1, 0.1, "Ego")
        assert trajectory.is_relative and trajectory.is_timed

    def test_no_reference_is_the_actor_itself(self) -> None:
        trajectory = authored_trajectory(
            "relative_lane", relative_vertices="0\n10", reference_entity=None
        )
        assert trajectory.vertices[0].position == RelativeLanePose(0.0)

    def test_too_few_vertices(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            authored_trajectory("relative_lane", relative_vertices="0")


class TestRelativeValidation:
    def test_a_good_card_has_no_errors(self) -> None:
        document = _document(
            _relative({"relative_vertices": _RELATIVE_ROWS, "reference_entity": "ego"})
        )
        assert _messages(document) == []

    def test_without_a_reference_it_is_the_actor(self) -> None:
        assert (
            _messages(_document(_relative({"relative_vertices": _RELATIVE_ROWS}))) == []
        )

    def test_an_unknown_reference_is_an_error(self) -> None:
        document = _document(
            _relative({"relative_vertices": _RELATIVE_ROWS, "reference_entity": "npc9"})
        )
        assert any("unknown entity 'npc9'" in m for m in _messages(document))

    def test_a_fractional_lane_change_is_a_field_error(self) -> None:
        document = _document(_relative({"relative_vertices": "0\n10, 0, 0.5"}))
        assert any(
            m.startswith("Relative vertices: vertex 2: d_lane")
            for m in _messages(document)
        )

    def test_too_few_and_mixed_times(self) -> None:
        assert any(
            "at least two" in m
            for m in _messages(
                _document(_relative({"relative_vertices": "0, 0, 0, , 0"}))
            )
        )
        assert any(
            "either every vertex has a time" in m
            for m in _messages(
                _document(_relative({"relative_vertices": "0, 0, 0, , 0\n10"}))
            )
        )

    def test_a_time_reference_needs_times(self) -> None:
        document = _document(_relative({"relative_vertices": "0\n10"}))
        assert any("needs a time on every vertex" in m for m in _messages(document))
        untimed = _document(
            _relative({"relative_vertices": "0\n10", "time_domain": "none"})
        )
        assert _messages(untimed) == []


class TestRelativeBuild:
    def test_the_card_builds_relative_vertices_against_a_role(self) -> None:
        from autoware_carla_scenario.authoring.builders import instantiate_action
        from autoware_carla_scenario.authoring.compiler import BuildContext

        document = _document(
            _relative(
                {
                    # Text, as a hand edit may leave it.
                    "relative_vertices": _RELATIVE_TEXT,
                    "reference_entity": "ego",
                }
            )
        )
        compiled = compile_document(document)
        (card,) = [a for a in compiled.actions if a.node.id == "a1"]
        # The document's entity id is resolved to the role the runtime knows.
        assert card.params["reference_entity"] == "Ego"
        action = instantiate_action(card, BuildContext(scenario=None))
        assert isinstance(action, FollowTrajectoryAction)
        positions = [vertex.position for vertex in action.trajectory.vertices]
        assert positions[1] == RelativeLanePose(20.0, 0.0, 1, entity_ref="Ego")
        assert action._time_reference is not None

    def test_without_a_reference_the_vertices_name_none(self) -> None:
        from autoware_carla_scenario.authoring.builders import instantiate_action
        from autoware_carla_scenario.authoring.compiler import BuildContext

        compiled = compile_document(
            _document(_relative({"relative_vertices": _RELATIVE_ROWS}))
        )
        (card,) = [a for a in compiled.actions if a.node.id == "a1"]
        action = instantiate_action(card, BuildContext(scenario=None))
        assert isinstance(action, FollowTrajectoryAction)
        assert all(
            isinstance(vertex.position, RelativeLanePose)
            and vertex.position.entity_ref is None
            for vertex in action.trajectory.vertices
        )


class TestRelativeEditor:
    def test_a_relative_card_is_written_in_the_inspector(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        response = client.post(
            "/new", data={"kind": "cut_in", "title": "Cut in"}, follow_redirects=False
        )
        draft_id = response.headers["location"].rsplit("/", 1)[-1]
        store = DraftStore(tmp_path / "drafts")
        document = _stored(store, draft_id)
        actor = next(e.id for e in document.entities if e.kind != "ego")
        ego = next(e.id for e in document.entities if e.kind == "ego")

        client.post(
            f"/draft/{draft_id}/action",
            data={"type_id": "follow_trajectory", "actor": actor},
        )
        card = next(
            a for a in _stored(store, draft_id).actions if a.type == "follow_trajectory"
        )
        page = client.post(
            f"/draft/{draft_id}/action/{card.id}",
            data={
                "path_source": "relative_lane",
                "reference_entity": ego,
                "relative_vertices": _RELATIVE_TEXT,
                "vertices": "",
                "time_domain": "relative",
                "time_scale": "1",
                "time_offset": "0",
                "following_mode": "position",
                "initial_distance_offset": "0",
                "speed_kmh": "30",
                "lateral_offset_m": "0",
            },
        )
        assert page.status_code == 200
        stored = _stored(store, draft_id).action(card.id)
        assert stored is not None
        assert stored.params["path_source"] == "relative_lane"
        assert stored.params["reference_entity"] == ego
        assert stored.params["relative_vertices"] == _RELATIVE_ROWS

        inspector = client.get(f"/draft/{draft_id}/inspector/{card.id}").text
        assert "3 vertices, ds 0 to 40 m, lanes +0, +1, 4.0 s" in inspector
        assert "40, -0.5, 1, 0.1, 4" in inspector
        assert "Relative to an entity&#39;s lane" in inspector
        assert validate_document(_stored(store, draft_id)).ok

    def test_a_bad_lane_count_is_refused_by_the_form(
        self, client: TestClient, tmp_path: Path
    ) -> None:
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
            data={"path_source": "relative_lane", "relative_vertices": "0\n5, 0, 0.5"},
        )
        stored = _stored(store, draft_id).action(card.id)
        assert stored is not None
        assert stored.params.get("relative_vertices") in (None, [])

    def test_deleting_the_reference_entity_clears_it(self, tmp_path: Path) -> None:
        from autoware_carla_scenario.editor.service import EditorService

        document = _document(
            _relative({"relative_vertices": _RELATIVE_ROWS, "reference_entity": "npc2"})
        )
        document.entities.append(Entity(id="npc2", spawn=SpawnSpec(lanelet_id=3)))
        assert _messages(document) == []
        EditorService(DraftStore(tmp_path / "drafts")).delete_entity(document, "npc2")
        card = document.action("a1")
        assert card is not None
        # The card stays, measured from its own actor now: no dangling id.
        assert card.params["reference_entity"] is None
        assert card.params["relative_vertices"] == _RELATIVE_ROWS
        assert _messages(document) == []
