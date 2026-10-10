"""Transcribe a T4 driving scene into trajectories a scenario can replay.

A converted T4 scene (the format ``tier4/e2e-devkit`` reads) is a directory:

* ``derived/meta.json`` -- scene name, frame count, the area map it was
  recorded on;
* ``derived/scalars.npz`` -- ``trajectory`` ``[N, 4]``: the ego's rear-axle
  pose per frame as ``(x, y, cos, sin)`` in Autoware's ``map`` frame; ``shape``
  ``(wheel_base, length, width)``; ``goal`` ``(x, y, cos, sin)``, all zero when
  the scene has none;
* ``derived/frames.pack`` -- per-frame tensors, among them ``gt_boxes``
  ``[M, 9]`` = ``(x, y, z, width, length, height, yaw, vx, vy)`` and
  ``gt_labels`` ``[M]``, in that frame's ego (rear-axle) frame.

The ego trajectory and every object's track become a
:class:`~autoware_carla_scenario.trajectory.model.Trajectory` of
:class:`~autoware_carla_scenario.trajectory.model.MapPose` vertices, timed on the
scene's own clock (frame 0 at ``t = 0``, 10 Hz).  Nothing is converted to CARLA
coordinates here: that happens when the action runs, through the map the
scenario loaded -- which has to be the scene's area map
(:attr:`T4SceneTranscription.area_map_id`).

**Tracks are associated here.**  ``frames.pack`` stores boxes per frame with no
track identity, so consecutive frames' boxes are linked by
:func:`associate_tracks`: a constant-velocity prediction, a distance gate and a
greedy nearest-first match within the same object category.  Labels in a T4
scene may come from a tracker rather than a person (``meta.json``'s
``annotation_source`` says which), so a track can break where the recording
lost the object for longer than ``max_gap_frames``.

The transcription saves to a small JSON file (:meth:`T4SceneTranscription.save`)
so a scenario package can carry the scene it replays instead of the dataset.
"""

from __future__ import annotations

import enum
import json
import logging
import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .model import MapPose, Trajectory, TrajectoryVertex

logger = logging.getLogger(__name__)

__all__ = [
    "T4_FRAME_RATE_HZ",
    "T4Category",
    "T4Detection",
    "T4ObjectTrack",
    "T4SceneTranscription",
    "associate_tracks",
    "read_t4_scene",
]

#: Frames per second of a T4 scene.
T4_FRAME_RATE_HZ = 10.0

#: Version of the JSON :meth:`T4SceneTranscription.save` writes.
TRANSCRIPTION_FORMAT_VERSION = 1

_BUNDLE_MAGIC = b"T4BUND\x00\x02"


class T4Category(enum.Enum):
    """What kind of road user a T4 label is."""

    VEHICLE = "vehicle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"


#: T4 class id -> category (``t4_e2e_devkit.data.tracks``).
T4_LABEL_CATEGORIES: Dict[int, T4Category] = {
    0: T4Category.VEHICLE,
    1: T4Category.VEHICLE,
    2: T4Category.VEHICLE,
    3: T4Category.BICYCLE,
    4: T4Category.PEDESTRIAN,
}


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class T4ObjectTrack:
    """One road user of a T4 scene, followed through the frames it was seen in.

    Args:
        track_id: Identity within the transcription (assigned by association).
        label: The T4 class id seen most often on the track.
        category: What kind of road user that label is.
        width, length, height: The box size, median over the track (m).
        frames: The scene frames the object was seen in, increasing.
        poses: Its box centre in each of them, in the ``map`` frame.
    """

    track_id: int
    label: int
    category: T4Category
    width: float
    length: float
    height: float
    frames: Tuple[int, ...]
    poses: Tuple[MapPose, ...]

    @property
    def start_time(self) -> float:
        """Scene time of the first sighting (s)."""
        return self.frames[0] / T4_FRAME_RATE_HZ

    @property
    def end_time(self) -> float:
        """Scene time of the last sighting (s)."""
        return self.frames[-1] / T4_FRAME_RATE_HZ

    def trajectory(self, name: Optional[str] = None) -> Trajectory:
        """The track as a timed trajectory on the scene clock."""
        return Trajectory(
            name or f"t4_{self.category.value}_{self.track_id}",
            [
                TrajectoryVertex(pose, frame / T4_FRAME_RATE_HZ)
                for frame, pose in zip(self.frames, self.poses)
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of the track."""
        return {
            "track_id": self.track_id,
            "label": self.label,
            "category": self.category.value,
            "size": [self.width, self.length, self.height],
            "frames": list(self.frames),
            "poses": [[pose.x, pose.y, pose.yaw] for pose in self.poses],
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "T4ObjectTrack":
        """The inverse of :meth:`to_dict`."""
        width, length, height = (float(item) for item in value["size"])
        return cls(
            track_id=int(value["track_id"]),
            label=int(value["label"]),
            category=T4Category(value["category"]),
            width=width,
            length=length,
            height=height,
            frames=tuple(int(frame) for frame in value["frames"]),
            poses=tuple(
                MapPose(float(x), float(y), None if yaw is None else float(yaw))
                for x, y, yaw in value["poses"]
            ),
        )


@dataclass
class T4SceneTranscription:
    """Everything a scenario needs from a T4 scene to replay it.

    Args:
        scene_name: The scene's name (``meta.json``), or its directory's.
        area_map_id: The area map it was recorded on, when ``meta.json`` says.
            The scenario has to load that map: the poses are in its frame.
        ego_frames: The frames with a valid ego pose.
        ego_poses: The ego's **vehicle centre** at each, in the ``map`` frame
            (the recording's rear axle moved forward by
            :attr:`ego_center_offset`).
        ego_center_offset: How far ahead of the rear axle the centre is (m).
        ego_size: ``(wheel_base, length, width)`` when the scene states it.
        goal: The scene's destination (rear axle), when it has one.
        objects: Every associated object track.
        annotation_source: Who drew the boxes (``meta.json``), when known.
    """

    scene_name: str
    area_map_id: Optional[str]
    ego_frames: Tuple[int, ...]
    ego_poses: Tuple[MapPose, ...]
    ego_center_offset: float = 0.0
    ego_size: Optional[Tuple[float, float, float]] = None
    goal: Optional[MapPose] = None
    objects: List[T4ObjectTrack] = field(default_factory=list)
    annotation_source: Optional[str] = None

    @property
    def duration(self) -> float:
        """Scene time from the first ego frame to the last (s)."""
        return (self.ego_frames[-1] - self.ego_frames[0]) / T4_FRAME_RATE_HZ

    @property
    def ego_start(self) -> MapPose:
        """Where the ego starts (vehicle centre)."""
        return self.ego_poses[0]

    @property
    def ego_end(self) -> MapPose:
        """Where the ego was at the end of the recording (vehicle centre)."""
        return self.ego_poses[-1]

    def ego_trajectory(self, name: Optional[str] = None) -> Trajectory:
        """The ego's drive as a timed trajectory on the scene clock."""
        return Trajectory(
            name or f"{self.scene_name}_ego",
            [
                TrajectoryVertex(pose, frame / T4_FRAME_RATE_HZ)
                for frame, pose in zip(self.ego_frames, self.ego_poses)
            ],
        )

    def objects_of(self, *categories: T4Category) -> List[T4ObjectTrack]:
        """The object tracks of the given categories (all with none given)."""
        if not categories:
            return list(self.objects)
        return [track for track in self.objects if track.category in categories]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of the transcription."""
        return {
            "format": "t4_transcription",
            "version": TRANSCRIPTION_FORMAT_VERSION,
            "frame_rate_hz": T4_FRAME_RATE_HZ,
            "scene_name": self.scene_name,
            "area_map_id": self.area_map_id,
            "annotation_source": self.annotation_source,
            "ego": {
                "frames": list(self.ego_frames),
                "poses": [[pose.x, pose.y, pose.yaw] for pose in self.ego_poses],
                "center_offset": self.ego_center_offset,
                "size": None if self.ego_size is None else list(self.ego_size),
            },
            "goal": None
            if self.goal is None
            else [self.goal.x, self.goal.y, self.goal.yaw],
            "objects": [track.to_dict() for track in self.objects],
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "T4SceneTranscription":
        """The inverse of :meth:`to_dict`.

        Raises:
            ValueError: If *value* is not a transcription this version reads.
        """
        if value.get("format") != "t4_transcription":
            raise ValueError("not a T4 scene transcription")
        if value.get("version") != TRANSCRIPTION_FORMAT_VERSION:
            raise ValueError(
                f"unsupported T4 transcription version {value.get('version')!r}"
            )
        ego = value["ego"]
        goal = value.get("goal")
        size = ego.get("size")
        return cls(
            scene_name=str(value["scene_name"]),
            area_map_id=value.get("area_map_id"),
            annotation_source=value.get("annotation_source"),
            ego_frames=tuple(int(frame) for frame in ego["frames"]),
            ego_poses=tuple(
                MapPose(float(x), float(y), None if yaw is None else float(yaw))
                for x, y, yaw in ego["poses"]
            ),
            ego_center_offset=float(ego.get("center_offset", 0.0)),
            ego_size=None
            if size is None
            else (float(size[0]), float(size[1]), float(size[2])),
            goal=None
            if goal is None
            else MapPose(float(goal[0]), float(goal[1]), float(goal[2])),
            objects=[T4ObjectTrack.from_dict(track) for track in value["objects"]],
        )

    def save(self, path: Union[str, Path]) -> Path:
        """Write the transcription as JSON to *path*."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "T4SceneTranscription":
        """Read a transcription :meth:`save` wrote."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_t4_scene(
    scene_dir: Union[str, Path],
    *,
    max_gap_frames: int = 5,
    min_track_frames: int = 5,
    match_distance_m: float = 2.0,
    ego_center_offset: Optional[float] = None,
) -> T4SceneTranscription:
    """Read a converted T4 scene and transcribe its ego and objects.

    Args:
        scene_dir: The scene directory (the one holding ``derived/``).
        max_gap_frames: Most frames an object may go unseen and keep its track.
        min_track_frames: Tracks seen in fewer frames are dropped as clutter.
        match_distance_m: Association gate around an object's predicted
            position (m); widened by how far it could have moved meanwhile.
        ego_center_offset: Rear axle to vehicle centre (m).  ``None`` takes half
            the wheel base the scene states, or 0 without one.

    Raises:
        FileNotFoundError: If the scene has no ``derived/scalars.npz``.
        ValueError: If the files are not the T4 format this reads.
    """
    scene_dir = Path(scene_dir)
    derived = scene_dir / "derived"
    meta_path = derived / "meta.json"
    meta: Dict[str, Any] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    )
    with np.load(derived / "scalars.npz", allow_pickle=False) as scalars:
        trajectory = np.asarray(scalars["trajectory"], dtype=np.float64)
        shape = (
            np.asarray(scalars["shape"], dtype=np.float64).reshape(-1)
            if "shape" in scalars.files
            else np.zeros(0)
        )
        goal_row = (
            np.asarray(scalars["goal"], dtype=np.float64).reshape(-1)
            if "goal" in scalars.files
            else np.zeros(4)
        )
    if trajectory.ndim != 2 or trajectory.shape[1] < 4:
        raise ValueError(
            f"{scene_dir}: scalars trajectory must be [N, 4] (x, y, cos, sin), "
            f"got {trajectory.shape}"
        )

    ego_size = (
        (float(shape[0]), float(shape[1]), float(shape[2])) if shape.size >= 3 else None
    )
    if ego_center_offset is None:
        ego_center_offset = ego_size[0] / 2.0 if ego_size is not None else 0.0

    valid = np.isfinite(trajectory[:, :4]).all(axis=1) & (
        np.hypot(trajectory[:, 2], trajectory[:, 3]) > 1e-6
    )
    headings = np.arctan2(trajectory[:, 3], trajectory[:, 2])
    ego_frames: List[int] = []
    ego_poses: List[MapPose] = []
    for frame in np.flatnonzero(valid):
        x, y, yaw = trajectory[frame, 0], trajectory[frame, 1], headings[frame]
        ego_frames.append(int(frame))
        ego_poses.append(
            MapPose(
                float(x + ego_center_offset * math.cos(yaw)),
                float(y + ego_center_offset * math.sin(yaw)),
                float(yaw),
            )
        )
    if len(ego_frames) < 2:
        raise ValueError(f"{scene_dir}: fewer than two frames with a valid ego pose")

    goal = None
    if goal_row.size >= 4 and np.any(goal_row[:4]) and np.isfinite(goal_row[:4]).all():
        goal = MapPose(
            float(goal_row[0]),
            float(goal_row[1]),
            float(math.atan2(goal_row[3], goal_row[2])),
        )

    objects: List[T4ObjectTrack] = []
    pack = derived / "frames.pack"
    if pack.is_file():
        detections = _world_detections(_read_boxes(pack), trajectory, valid)
        objects = associate_tracks(
            detections,
            max_gap_frames=max_gap_frames,
            min_track_frames=min_track_frames,
            match_distance_m=match_distance_m,
        )
    else:
        logger.warning(
            "%s: no derived/frames.pack; transcribing the ego only", scene_dir
        )

    return T4SceneTranscription(
        scene_name=str(meta.get("scene_name") or scene_dir.name),
        area_map_id=None
        if meta.get("area_map_id") is None
        else str(meta["area_map_id"]),
        ego_frames=tuple(ego_frames),
        ego_poses=tuple(ego_poses),
        ego_center_offset=float(ego_center_offset),
        ego_size=ego_size,
        goal=goal,
        objects=objects,
        annotation_source=meta.get("annotation_source"),
    )


@dataclass(frozen=True)
class T4Detection:
    """One box in one frame, in the ``map`` frame."""

    frame: int
    label: int
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    width: float
    length: float
    height: float


def _read_boxes(path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``(gt_boxes [M, 9], gt_labels [M])`` of every frame of a T4 bundle.

    The bundle (format ``t4bundle`` version 2) is::

        magic | zstd(frame 0) | ... | index | <QQ index_offset, index_size> | magic

    and its index is ``<I header_size> | header JSON | <Q> offsets[n] |
    <Q> sizes[n] | <I> counts[n]`` per variable-length field, in declaration
    order.  A frame decompresses to its fields' bytes back to back.
    """
    import zstandard  # noqa: PLC0415 -- only a scene with objects needs it

    raw_file = path.read_bytes()
    trailer_size = len(_BUNDLE_MAGIC) + 16
    if (
        len(raw_file) < 2 * len(_BUNDLE_MAGIC) + 16
        or raw_file[: len(_BUNDLE_MAGIC)] != _BUNDLE_MAGIC
        or raw_file[-len(_BUNDLE_MAGIC) :] != _BUNDLE_MAGIC
    ):
        raise ValueError(f"{path}: not a T4 bundle")
    index_offset, index_size = struct.unpack(
        "<QQ", raw_file[-trailer_size : -len(_BUNDLE_MAGIC)]
    )
    if index_offset + index_size > len(raw_file) - trailer_size:
        raise ValueError(f"{path}: invalid bundle index bounds")
    index = raw_file[index_offset : index_offset + index_size]
    (header_size,) = struct.unpack("<I", index[:4])
    header = json.loads(index[4 : 4 + header_size])
    if header.get("format") != "t4bundle" or header.get("version") != 2:
        raise ValueError(
            f"{path}: unsupported bundle {header.get('format')}/{header.get('version')}"
        )
    n_frames = int(header["n_frames"])
    fields: List[Dict[str, Any]] = list(header["fields"])
    names = {field_["name"] for field_ in fields}
    if not {"gt_boxes", "gt_labels"} <= names:
        raise ValueError(f"{path}: the bundle has no gt_boxes/gt_labels")
    variable = [field_ for field_ in fields if field_.get("variable", False)]
    pos = 4 + header_size
    offsets = np.frombuffer(index, np.uint64, n_frames, pos)
    pos += 8 * n_frames
    sizes = np.frombuffer(index, np.uint64, n_frames, pos)
    pos += 8 * n_frames
    counts: Dict[str, np.ndarray] = {}
    for field_ in variable:
        counts[field_["name"]] = np.frombuffer(index, np.uint32, n_frames, pos)
        pos += 4 * n_frames

    decompressor = zstandard.ZstdDecompressor()
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    for frame in range(n_frames):
        start = int(offsets[frame])
        blob = decompressor.decompress(
            raw_file[start : start + int(sizes[frame])],
            max_output_size=1 << 31,
        )
        values: Dict[str, np.ndarray] = {}
        cursor = 0
        for field_ in fields:
            dtype = np.dtype(field_["dtype"])
            shape = list(field_["shape"])
            if field_.get("variable", False):
                shape[0] = int(counts[field_["name"]][frame])
            count = int(np.prod(shape)) if shape else 1
            if field_["name"] in ("gt_boxes", "gt_labels"):
                values[field_["name"]] = np.frombuffer(
                    blob, dtype, count, cursor
                ).reshape(shape)
            cursor += count * dtype.itemsize
        frames.append(
            (
                np.asarray(values["gt_boxes"], dtype=np.float64).reshape(-1, 9),
                np.asarray(values["gt_labels"], dtype=np.int64).reshape(-1),
            )
        )
    return frames


def _world_detections(
    frames: Sequence[Tuple[np.ndarray, np.ndarray]],
    trajectory: np.ndarray,
    valid: np.ndarray,
) -> List[T4Detection]:
    """Every frame's boxes moved from that frame's ego frame into the map frame."""
    detections: List[T4Detection] = []
    for frame, (boxes, labels) in enumerate(frames):
        if frame >= len(trajectory) or not valid[frame]:
            continue
        ego_x, ego_y = trajectory[frame, 0], trajectory[frame, 1]
        heading = math.atan2(trajectory[frame, 3], trajectory[frame, 2])
        c, s = math.cos(heading), math.sin(heading)
        for row, label in zip(boxes, labels):
            if not np.isfinite(row[:7]).all() or int(label) not in T4_LABEL_CATEGORIES:
                continue
            vx, vy = (
                (float(row[7]), float(row[8]))
                if np.isfinite(row[7:9]).all()
                else (0.0, 0.0)
            )
            detections.append(
                T4Detection(
                    frame=frame,
                    label=int(label),
                    x=float(ego_x + c * row[0] - s * row[1]),
                    y=float(ego_y + s * row[0] + c * row[1]),
                    yaw=float(_wrap(heading + row[6])),
                    vx=float(c * vx - s * vy),
                    vy=float(s * vx + c * vy),
                    width=float(row[3]),
                    length=float(row[4]),
                    height=float(row[5]),
                )
            )
    return detections


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


@dataclass
class _OpenTrack:
    detections: List[T4Detection]

    @property
    def last(self) -> T4Detection:
        return self.detections[-1]

    @property
    def category(self) -> T4Category:
        return T4_LABEL_CATEGORIES[self.last.label]

    def velocity(self) -> Tuple[float, float]:
        """The box's own velocity, or the track's last step when it has none."""
        last = self.last
        if (last.vx, last.vy) != (0.0, 0.0) or len(self.detections) < 2:
            return last.vx, last.vy
        before = self.detections[-2]
        dt = (last.frame - before.frame) / T4_FRAME_RATE_HZ
        return (last.x - before.x) / dt, (last.y - before.y) / dt

    def predict(self, frame: int) -> Tuple[float, float, float]:
        """``(x, y, gate widening)`` at *frame*, by constant velocity."""
        vx, vy = self.velocity()
        dt = (frame - self.last.frame) / T4_FRAME_RATE_HZ
        return (
            self.last.x + vx * dt,
            self.last.y + vy * dt,
            0.5 * math.hypot(vx, vy) * dt,
        )


def associate_tracks(
    detections: Iterable[T4Detection],
    *,
    max_gap_frames: int = 5,
    min_track_frames: int = 5,
    match_distance_m: float = 2.0,
) -> List[T4ObjectTrack]:
    """Link per-frame boxes into tracks.

    Frame by frame, every open track predicts where its object is now (constant
    velocity from the box's own velocity, or its last step); each box within
    ``match_distance_m`` -- plus half the distance the object could have
    covered -- of a prediction of the same category is a candidate, and the
    candidates are taken nearest first, one box per track.  A box no track
    takes opens a new one; a track unmatched for more than ``max_gap_frames``
    is closed.

    Args:
        detections: Boxes in the ``map`` frame, any order.
        max_gap_frames: Most frames a track may go unmatched.
        min_track_frames: Tracks with fewer boxes are dropped.
        match_distance_m: The base association gate (m).

    Returns:
        The tracks, numbered by when they began.
    """
    by_frame: Dict[int, List[T4Detection]] = {}
    for detection in detections:
        by_frame.setdefault(detection.frame, []).append(detection)

    open_tracks: List[_OpenTrack] = []
    closed: List[_OpenTrack] = []
    for frame in sorted(by_frame):
        still_open: List[_OpenTrack] = []
        for track in open_tracks:
            (
                still_open if frame - track.last.frame <= max_gap_frames + 1 else closed
            ).append(track)
        open_tracks = still_open

        boxes = by_frame[frame]
        candidates: List[Tuple[float, int, int]] = []
        for track_index, track in enumerate(open_tracks):
            px, py, widening = track.predict(frame)
            for box_index, box in enumerate(boxes):
                if T4_LABEL_CATEGORIES[box.label] is not track.category:
                    continue
                gap = math.hypot(box.x - px, box.y - py)
                if gap <= match_distance_m + widening:
                    candidates.append((gap, track_index, box_index))
        candidates.sort()
        taken_tracks: set[int] = set()
        taken_boxes: set[int] = set()
        for _, track_index, box_index in candidates:
            if track_index in taken_tracks or box_index in taken_boxes:
                continue
            open_tracks[track_index].detections.append(boxes[box_index])
            taken_tracks.add(track_index)
            taken_boxes.add(box_index)
        for box_index, box in enumerate(boxes):
            if box_index not in taken_boxes:
                open_tracks.append(_OpenTrack([box]))
    closed.extend(open_tracks)

    kept = [track for track in closed if len(track.detections) >= min_track_frames]
    kept.sort(key=lambda track: (track.detections[0].frame, track.detections[0].x))
    return [_finish(track_id, track) for track_id, track in enumerate(kept)]


def _finish(track_id: int, track: _OpenTrack) -> T4ObjectTrack:
    detections = track.detections
    label = Counter(box.label for box in detections).most_common(1)[0][0]
    return T4ObjectTrack(
        track_id=track_id,
        label=label,
        category=T4_LABEL_CATEGORIES[label],
        width=float(np.median([box.width for box in detections])),
        length=float(np.median([box.length for box in detections])),
        height=float(np.median([box.height for box in detections])),
        frames=tuple(box.frame for box in detections),
        poses=tuple(MapPose(box.x, box.y, box.yaw) for box in detections),
    )


def _wrap(angle: float) -> float:
    """*angle* wrapped into ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
