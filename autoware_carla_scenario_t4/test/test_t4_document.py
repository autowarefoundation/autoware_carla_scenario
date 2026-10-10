"""A transcribed T4 scene, written down as a scenario document.

The scene is built here on the repository's fixture map (nishishinjuku):
lanelets 104-107 run side by side, heading about 100 degrees, so each road user
can be put on a known lanelet and the test can say where it must be placed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from autoware_carla_scenario.authoring.models import MapRef
from autoware_carla_scenario.authoring.persistence import DraftStore, load_document
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.trajectory import MapPose
from autoware_carla_scenario_t4 import T4Category, T4ObjectTrack, T4SceneTranscription
from autoware_carla_scenario_t4.cli import main
from autoware_carla_scenario_t4.document import (
    LaneletLocator,
    load_lanelet_map,
    transcription_to_document,
)

_DATA = Path(__file__).resolve().parents[2] / "data"
_OSM = _DATA / "nishishinjuku.osm"
_XODR = _DATA / "nishishinjuku_carla.xodr"

#: Where lanelets 104 and 106 start, and the way they run.
_L104 = (81483.95525, 50594.90635)
_L106 = (81490.19305, 50596.19295)
_HEADING = math.radians(100.3)


def _along(start: tuple[float, float], metres: float, yaw: float = _HEADING) -> MapPose:
    return MapPose(
        start[0] + metres * math.cos(_HEADING),
        start[1] + metres * math.sin(_HEADING),
        yaw,
    )


@pytest.fixture(scope="module")
def locator() -> LaneletLocator:
    return LaneletLocator(load_lanelet_map(_OSM, _XODR))


@pytest.fixture
def scene() -> T4SceneTranscription:
    frames = tuple(range(30))
    return T4SceneTranscription(
        scene_name="Scene 0001",
        area_map_id="nishishinjuku",
        ego_frames=frames,
        # 5 m/s up lanelet 104, from 10 m in.
        ego_poses=tuple(_along(_L104, 10.0 + 0.5 * f) for f in frames),
        ego_center_offset=1.4,
        objects=[
            # Seen from the start, ahead on the next lane but one.
            T4ObjectTrack(
                0,
                0,
                T4Category.VEHICLE,
                1.8,
                4.4,
                1.5,
                frames=frames,
                poses=tuple(_along(_L106, 30.0 + 0.8 * f) for f in frames),
            ),
            # Seen from frame 10: spawned out of the world.
            T4ObjectTrack(
                1,
                0,
                T4Category.VEHICLE,
                1.8,
                4.4,
                1.5,
                frames=tuple(range(10, 30)),
                poses=tuple(_along(_L104, 60.0 + 0.5 * f) for f in range(10, 30)),
            ),
            # Far from every lanelet: left out.
            T4ObjectTrack(
                2,
                4,
                T4Category.PEDESTRIAN,
                0.6,
                0.6,
                1.7,
                frames=frames,
                poses=tuple(MapPose(0.0, 0.0) for _ in frames),
            ),
        ],
    )


def _map() -> MapRef:
    return MapRef(lanelet2_path=str(_OSM), xodr_path=str(_XODR))


class TestPlacement:
    def test_a_pose_is_placed_on_its_lanelet(self, locator: LaneletLocator) -> None:
        placement = locator.locate(_along(_L106, 30.0))
        assert placement is not None
        assert placement.lanelet_id == 106
        assert placement.s == pytest.approx(30.0, abs=0.5)
        assert placement.t == pytest.approx(0.0, abs=0.3)
        assert placement.heading == pytest.approx(0.0, abs=0.05)

    def test_the_heading_is_relative_to_the_lane(self, locator: LaneletLocator) -> None:
        placement = locator.locate(_along(_L104, 20.0, yaw=_HEADING + 0.3))
        assert placement is not None and placement.heading == pytest.approx(
            0.3, abs=0.05
        )

    def test_nowhere_near_a_lanelet(self, locator: LaneletLocator) -> None:
        assert locator.locate(MapPose(0.0, 0.0)) is None


class TestDocument:
    def test_it_is_a_valid_document(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        document = transcription_to_document(scene, locator, _map())
        assert document.id == "scene_0001"
        report = validate_document(document)
        assert not report.errors, [i.message for i in report.errors]

    def test_the_ego_starts_where_the_recording_did(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        document = transcription_to_document(scene, locator, _map())
        ego = document.entity("ego")
        assert ego is not None and ego.spawn.lanelet_id == 104
        assert ego.spawn.s.value == pytest.approx(10.0, abs=0.5)
        # No goal in the scene: sent where the recording ended.
        assert ego.goal is not None and ego.goal.lanelet_id == 104
        assert ego.goal.s == pytest.approx(10.0 + 0.5 * 29, abs=0.5)

    def test_every_road_user_on_a_lanelet_is_replayed(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        document = transcription_to_document(scene, locator, _map())
        assert [e.id for e in document.entities] == ["ego", "vehicle_0", "vehicle_1"]
        cards = {a.actor: a for a in document.actions}
        card = cards["vehicle_0"]
        assert card.type == "follow_trajectory"
        assert card.params["time_domain"] == "absolute"
        # Every other frame, and the last.
        rows = card.params["vertices"]
        assert [row[3] for row in rows[:3]] == [0.0, 0.2, 0.4]
        assert rows[-1][3] == pytest.approx(2.9)

    def test_one_seen_late_spawns_out_of_the_world(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        document = transcription_to_document(scene, locator, _map())
        early, late = document.entity("vehicle_0"), document.entity("vehicle_1")
        assert early is not None and not early.spawn.hidden
        assert late is not None and late.spawn.hidden
        assert late.spawn.lanelet_id == 104

    def test_the_run_passes_when_the_recording_ends_and_fails_on_a_collision(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        document = transcription_to_document(scene, locator, _map())
        (condition,) = document.assertions.pass_conditions
        assert condition.type == "elapsed_time"
        assert condition.params["duration_seconds"] == pytest.approx(2.9)
        (failure,) = document.assertions.fail_conditions
        assert failure.type == "collision"
        assert not validate_document(document).warnings

    def test_follow_mode_hides_nobody(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        document = transcription_to_document(
            scene, locator, _map(), following_mode="follow"
        )
        assert not any(e.spawn.hidden for e in document.entities)
        assert not validate_document(document).errors

    def test_categories_and_the_ego_replay(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        document = transcription_to_document(
            scene, locator, _map(), categories=[T4Category.PEDESTRIAN], replay_ego=True
        )
        assert [e.id for e in document.entities] == ["ego"]
        assert [a.actor for a in document.actions] == ["ego"]

    def test_an_autoware_ego_is_not_replayed(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        with pytest.raises(ValueError, match="Autoware"):
            transcription_to_document(
                scene, locator, _map(), replay_ego=True, ego_driver="autoware"
            )

    def test_a_scene_on_another_map_is_refused(
        self, scene: T4SceneTranscription, locator: LaneletLocator
    ) -> None:
        elsewhere = T4SceneTranscription(
            scene_name="x",
            area_map_id="elsewhere",
            ego_frames=(0, 1),
            ego_poses=(MapPose(0.0, 0.0), MapPose(1.0, 0.0)),
        )
        with pytest.raises(ValueError, match="no lanelet"):
            transcription_to_document(elsewhere, locator, _map())


class TestCommand:
    def test_a_saved_transcription_becomes_a_document_and_a_draft(
        self,
        scene: T4SceneTranscription,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCENARIO_EDITOR_DRAFTS", str(tmp_path / "drafts"))
        source = scene.save(tmp_path / "scene.json")
        output = tmp_path / "scene.yaml"
        status = main(
            [
                str(source),
                "--lanelet2",
                str(_OSM),
                "--xodr",
                str(_XODR),
                "--output",
                str(output),
                "--to-editor",
            ]
        )
        assert status == 0
        document = load_document(output)
        assert len(document.actions) == 2
        drafts = DraftStore(tmp_path / "drafts").list()
        assert [d.document.id for d in drafts] == ["scene_0001"]

    def test_somewhere_to_write_is_required(self, tmp_path: Path) -> None:
        assert main([str(tmp_path), "--lanelet2", str(_OSM)]) == 2
