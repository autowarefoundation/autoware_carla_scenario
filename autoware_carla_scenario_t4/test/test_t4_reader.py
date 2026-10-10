"""Transcribing a T4 scene: the ego's drive and every road user's track.

The scene is written here in the T4 format itself -- ``meta.json``,
``scalars.npz`` and a zstd ``frames.pack`` bundle -- from road users whose
positions are known in the map frame, so each test can say where they must
come out.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
import zstandard

from autoware_carla_scenario_t4 import (
    T4Category,
    T4Detection,
    T4SceneTranscription,
    associate_tracks,
    read_t4_scene,
)

_FRAMES = 20
_HZ = 10.0
#: The ego drives North at 10 m/s, so the ego frame is turned 90 degrees.
_EGO_HEADING = math.pi / 2
_WHEEL_BASE = 2.8


def _ego(frame: int) -> tuple[float, float]:
    """The ego's rear axle at *frame*, in the map frame."""
    return 1000.0, 2000.0 + 10.0 * frame / _HZ


def _to_ego(frame: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
    """A map-frame position and velocity in *frame*'s ego frame."""
    ex, ey = _ego(frame)
    c, s = math.cos(-_EGO_HEADING), math.sin(-_EGO_HEADING)
    dx, dy = x - ex, y - ey
    return c * dx - s * dy, s * dx + c * dy, c * vx - s * vy, s * vx + c * vy


def _parked_car(frame: int):
    if frame >= _FRAMES:
        return None
    return (1010.0, 2005.0, 0.0, 0.0, 0.0, 0)  # x, y, yaw, vx, vy, label


def _walker(frame: int):
    """Crosses East at 1 m/s from frame 5, unseen in frames 10-12."""
    if frame < 5 or 10 <= frame <= 12:
        return None
    return (990.0 + (frame - 5) / _HZ, 2010.0, 0.0, 1.0, 0.0, 4)


def _clutter(frame: int):
    if frame not in (3, 4):
        return None
    return (1030.0, 2030.0, 0.0, 0.0, 0.0, 1)


def _write_bundle(path: Path, frames: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """A ``t4bundle`` v2 file: a fixed field, then the two variable GT fields."""
    fields = [
        {"name": "lanes_speed", "shape": [2, 1], "dtype": "float32"},
        {"name": "gt_boxes", "shape": [0, 9], "dtype": "float32", "variable": True},
        {"name": "gt_labels", "shape": [0], "dtype": "int64", "variable": True},
    ]
    magic = b"T4BUND\x00\x02"
    compressor = zstandard.ZstdCompressor()
    body = bytearray(magic)
    offsets, sizes = [], []
    for boxes, labels in frames:
        raw = (
            np.zeros((2, 1), np.float32).tobytes()
            + boxes.astype(np.float32).tobytes()
            + labels.astype(np.int64).tobytes()
        )
        blob = compressor.compress(raw)
        offsets.append(len(body))
        sizes.append(len(blob))
        body += blob
    header = json.dumps(
        {"format": "t4bundle", "version": 2, "n_frames": len(frames), "fields": fields}
    ).encode()
    index = (
        struct.pack("<I", len(header))
        + header
        + np.asarray(offsets, np.uint64).tobytes()
        + np.asarray(sizes, np.uint64).tobytes()
        + np.asarray([len(f[1]) for f in frames], np.uint32).tobytes()
        + np.asarray([len(f[1]) for f in frames], np.uint32).tobytes()
    )
    index_offset = len(body)
    body += index + struct.pack("<QQ", index_offset, len(index)) + magic
    path.write_bytes(bytes(body))


def _write_scene(root: Path, *, goal: bool = True, objects: bool = True) -> Path:
    derived = root / "scene_0" / "derived"
    derived.mkdir(parents=True)
    (derived / "meta.json").write_text(
        json.dumps(
            {
                "scene_name": "scene_0",
                "area_map_id": "map_42",
                "n_frames": _FRAMES,
                "annotation_source": "ground_truth",
            }
        )
    )
    trajectory = np.array(
        [
            [*_ego(frame), math.cos(_EGO_HEADING), math.sin(_EGO_HEADING)]
            for frame in range(_FRAMES)
        ]
    )
    trajectory[7] = np.nan  # a frame the localization dropped
    goal_row = (
        np.array([1000.0, 2100.0, math.cos(_EGO_HEADING), math.sin(_EGO_HEADING)])
        if goal
        else np.zeros(4)
    )
    np.savez(
        derived / "scalars.npz",
        trajectory=trajectory,
        shape=np.array([_WHEEL_BASE, 4.5, 1.8]),
        goal=goal_row,
    )
    if objects:
        frames = []
        for frame in range(_FRAMES):
            boxes, labels = [], []
            for source in (_parked_car, _walker, _clutter):
                found = source(frame)
                if found is None:
                    continue
                x, y, yaw, vx, vy, label = found
                lx, ly, lvx, lvy = _to_ego(frame, x, y, vx, vy)
                boxes.append([lx, ly, 0.5, 1.8, 4.4, 1.5, yaw - _EGO_HEADING, lvx, lvy])
                labels.append(label)
            frames.append(
                (np.asarray(boxes, np.float64).reshape(-1, 9), np.asarray(labels))
            )
        _write_bundle(derived / "frames.pack", frames)
    return root / "scene_0"


@pytest.fixture
def scene(tmp_path: Path) -> T4SceneTranscription:
    return read_t4_scene(_write_scene(tmp_path))


class TestEgo:
    def test_the_scene_is_named(self, scene: T4SceneTranscription) -> None:
        assert scene.scene_name == "scene_0"
        assert scene.area_map_id == "map_42"
        assert scene.annotation_source == "ground_truth"

    def test_a_frame_without_a_pose_is_left_out(
        self, scene: T4SceneTranscription
    ) -> None:
        assert 7 not in scene.ego_frames
        assert len(scene.ego_frames) == _FRAMES - 1

    def test_the_rear_axle_is_moved_to_the_vehicle_centre(
        self, scene: T4SceneTranscription
    ) -> None:
        """Half the wheel base ahead -- North, the way the ego faces."""
        assert scene.ego_center_offset == pytest.approx(_WHEEL_BASE / 2)
        start = scene.ego_start
        assert (start.x, start.y, start.yaw) == pytest.approx(
            (1000.0, 2000.0 + _WHEEL_BASE / 2, _EGO_HEADING)
        )

    def test_the_ego_trajectory_runs_on_the_scene_clock(
        self, scene: T4SceneTranscription
    ) -> None:
        trajectory = scene.ego_trajectory()
        assert trajectory.is_timed
        assert trajectory.vertices[1].time == pytest.approx(0.1)
        assert scene.duration == pytest.approx((_FRAMES - 1) / _HZ)

    def test_the_goal_is_read(self, scene: T4SceneTranscription) -> None:
        assert scene.goal is not None
        assert (scene.goal.x, scene.goal.y, scene.goal.yaw) == pytest.approx(
            (1000.0, 2100.0, _EGO_HEADING)
        )

    def test_an_all_zero_goal_is_no_goal(self, tmp_path: Path) -> None:
        assert read_t4_scene(_write_scene(tmp_path, goal=False)).goal is None


class TestObjects:
    def test_boxes_become_tracks_in_the_map_frame(
        self, scene: T4SceneTranscription
    ) -> None:
        (car,) = scene.objects_of(T4Category.VEHICLE)
        assert car.frames == tuple(f for f in range(_FRAMES) if f != 7)
        assert {(round(p.x, 3), round(p.y, 3)) for p in car.poses} == {(1010.0, 2005.0)}
        assert car.poses[0].yaw == pytest.approx(0.0, abs=1e-6)
        assert (car.width, car.length) == pytest.approx((1.8, 4.4))

    def test_a_short_gap_does_not_break_a_track(
        self, scene: T4SceneTranscription
    ) -> None:
        (walker,) = scene.objects_of(T4Category.PEDESTRIAN)
        assert walker.frames[0] == 5 and walker.frames[-1] == _FRAMES - 1
        assert 11 not in walker.frames
        assert walker.start_time == pytest.approx(0.5)

    def test_clutter_is_dropped(self, scene: T4SceneTranscription) -> None:
        assert len(scene.objects) == 2

    def test_a_track_is_a_timed_trajectory(self, scene: T4SceneTranscription) -> None:
        (walker,) = scene.objects_of(T4Category.PEDESTRIAN)
        trajectory = walker.trajectory()
        assert trajectory.vertices[0].time == pytest.approx(0.5)
        assert trajectory.vertices[0].position == walker.poses[0]

    def test_a_scene_without_a_bundle_has_only_the_ego(self, tmp_path: Path) -> None:
        assert read_t4_scene(_write_scene(tmp_path, objects=False)).objects == []


class TestAssociation:
    def _box(
        self, frame: int, x: float, label: int = 0, vx: float = 0.0
    ) -> T4Detection:
        return T4Detection(frame, label, x, 0.0, 0.0, vx, 0.0, 1.8, 4.4, 1.5)

    def test_a_moving_object_is_followed_by_its_velocity(self) -> None:
        """20 m/s moves 2 m a frame: beyond the gate, but not beyond the prediction."""
        boxes = [self._box(frame, 2.0 * frame, vx=20.0) for frame in range(6)]
        (track,) = associate_tracks(boxes, min_track_frames=1, match_distance_m=1.0)
        assert len(track.frames) == 6

    def test_different_kinds_of_road_user_are_not_linked(self) -> None:
        boxes = [self._box(0, 0.0, label=0), self._box(1, 0.1, label=4)]
        assert len(associate_tracks(boxes, min_track_frames=1)) == 2

    def test_a_long_gap_starts_a_new_track(self) -> None:
        boxes = [self._box(0, 0.0), self._box(10, 0.0)]
        assert len(associate_tracks(boxes, min_track_frames=1, max_gap_frames=5)) == 2

    def test_the_nearest_box_is_taken_first(self) -> None:
        boxes = [
            self._box(0, 0.0),
            self._box(0, 10.0),
            self._box(1, 9.5),
            self._box(1, 0.4),
        ]
        tracks = associate_tracks(boxes, min_track_frames=1)
        assert sorted(track.poses[-1].x for track in tracks) == [0.4, 9.5]


class TestPersistence:
    def test_a_saved_transcription_reads_back_the_same(
        self, scene: T4SceneTranscription, tmp_path: Path
    ) -> None:
        loaded = T4SceneTranscription.load(scene.save(tmp_path / "scene.json"))
        assert loaded == scene

    def test_a_file_of_another_kind_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "other.json"
        path.write_text(json.dumps({"format": "something_else"}))
        with pytest.raises(ValueError, match="transcription"):
            T4SceneTranscription.load(path)
