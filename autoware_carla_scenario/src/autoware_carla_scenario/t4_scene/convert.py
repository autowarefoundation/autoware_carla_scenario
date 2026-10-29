"""A run's trajectory log as a T4Scene (docs/t4_scene.md).

The trajectory log (``{stem}.trajectory.jsonl``, docs/trajectory_log.md) holds
every vehicle, walker and traffic light of the run, tick by tick, in CARLA's
world frame.  This module reads it back -- no CARLA, no run -- and writes it as
a T4Scene, the Protobuf exchange contract of tier4/e2e-devkit
(``t4_scene.proto``, contract 0.1.0): the ego's poses and motion, every other
road user near it as a box, and the lights' states on the lanelets they
govern, on one uniform time grid, in Autoware's ``map`` frame, with the
Lanelet2 map the run was on as a separate ``MapResource``.

What the log does not hold is left out, not invented: the ego's acceleration,
angular velocity, steering and turn indicator, a route and a goal; a vehicle's
class beyond ``vehicle`` (the log has its blueprint, not its ``base_type``).
A tick the log does not have (it skips a tick without road users) is a row
whose ego, objects and lights are unavailable, never an interpolated one.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
import orjson

from ..constants import EGO_ROLES, FIXED_DELTA_SECONDS
from ..rerun import TRAJECTORY_SUFFIX
from ._proto import t4_scene_pb2 as pb
from .map_resource import map_resource

__all__ = [
    "LOG_SUFFIX",
    "MAP_SUFFIX",
    "SCENE_SUFFIX",
    "T4SceneExport",
    "trajectory_log_to_t4_scene",
    "write_t4_scene",
]

logger = logging.getLogger(__name__)

#: The suffix of a trajectory log; the run's stem is the file name without it.
LOG_SUFFIX = TRAJECTORY_SUFFIX
#: The T4Scene written for ``{stem}.trajectory.jsonl``: ``{stem}.t4scene.pb``.
SCENE_SUFFIX = ".t4scene.pb"
#: Its map, the ``MapResource`` the scene's ``MapReference`` names.
MAP_SUFFIX = ".t4scene.map.pb"

#: The contract's object categories this module writes (``Annotations.labels``).
_VEHICLE = 1
_PEDESTRIAN = 7
#: The contract's effective lane states (``TrafficLights.lane_states``); any
#: other CARLA state (``Off``, ``Unknown``) is UNKNOWN, 0.
_LANE_STATES = {"red": 1, "yellow": 2, "green": 3}
#: How far (m) a light's stop-line waypoint may be from the stop line of a
#: lanelet it governs: ``TrafficSignalAction``'s ``STOP_LINE_MATCH_M``.
_STOP_LINE_MATCH_M = 2.0

_TICK_NS = round(FIXED_DELTA_SECONDS * 1e9)
_DTYPES = {
    "?": pb.NumericArray.BOOL,
    "i8": pb.NumericArray.INT64,
    "f8": pb.NumericArray.FLOAT64,
}
#: CARLA's left-handed frame to the map's right-handed one: ``y`` flips.
_FLIP_Y = np.array([1.0, -1.0, 1.0])
_NO_STATES = np.zeros((0, 10))


@dataclass(frozen=True)
class T4SceneExport:
    """A converted log: the scene, and the map resource it names (or ``None``)."""

    scene: pb.T4Scene
    map_resource: Optional[pb.MapResource]


class _Actor(NamedTuple):
    """An ``actor`` record: what the log says once about a vehicle or walker."""

    type_id: str
    role: str
    extent: tuple[float, float, float]
    center: tuple[float, float, float]


@dataclass
class _Log:
    """What the conversion uses of a trajectory log: its records, and the
    ticks on the scene's grid -- the others only carry the lights' changes."""

    world: dict[str, Any] = field(default_factory=dict)
    map: Optional[dict[str, Any]] = None
    lights: list[dict[str, Any]] = field(default_factory=list)
    actors: dict[int, _Actor] = field(default_factory=dict)
    #: Each grid frame the log has a tick on: its ``actors`` as an (N, 10)
    #: array, and every light's state then.
    rows: dict[int, tuple[npt.NDArray[np.float64], dict[int, str]]] = field(
        default_factory=dict
    )
    first_frame: int = -1
    last_frame: int = -1
    #: The frame an entity spawned hidden was put on its place, by actor id.
    spawned: dict[int, int] = field(default_factory=dict)
    #: The frame an entity was taken out, by actor id.
    despawned: dict[int, int] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None

    def states(self, frame: int) -> npt.NDArray[np.float64]:
        """The (N, 10) actor states of the tick on *frame*; none without one."""
        return self.rows[frame][0] if frame in self.rows else _NO_STATES

    def on_scene(self, actor_id: int, frame: int) -> bool:
        """Whether a road user is on the scene at *frame*, not parked out of
        sight before its ``SpawnAction`` or after its ``DespawnAction``."""
        spawned = self.spawned.get(actor_id)
        despawned = self.despawned.get(actor_id)
        return (spawned is None or frame >= spawned) and (
            despawned is None or frame < despawned
        )


def _read_log(path: Path, step_frames: int) -> _Log:
    log = _Log()
    lights: dict[int, str] = {}
    with path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{number}: not a JSON line ({error})"
                ) from None
            kind = record.get("type")
            if kind == "tick":
                frame = int(record["frame"])
                if log.first_frame < 0:
                    log.first_frame = frame
                log.last_frame = frame
                # Light states are deltas: fold every tick's in, grid or not.
                for light_id, state in (record.get("lights") or {}).items():
                    lights[int(light_id)] = str(state)
                if (frame - log.first_frame) % step_frames == 0:
                    states = np.asarray(record["actors"], dtype=np.float64).reshape(
                        -1, 10
                    )
                    log.rows[frame] = (states, dict(lights))
            elif kind == "actor":
                log.actors[int(record["id"])] = _Actor(
                    type_id=str(record["type_id"]),
                    role=str(record.get("role") or ""),
                    extent=_xyz(record["extent"]),
                    center=_xyz(record.get("center") or (0.0, 0.0, 0.0)),
                )
            elif kind == "world":
                log.world = record
            elif kind == "map":
                log.map = record
            elif kind == "traffic_light":
                log.lights.append(record)
            elif kind == "spawn" and record.get("id") is not None:
                log.spawned.setdefault(int(record["id"]), int(record["frame"]))
            elif kind == "despawn" and record.get("id") is not None:
                log.despawned.setdefault(int(record["id"]), int(record["frame"]))
            elif kind == "result":
                log.result = record
    if log.first_frame < 0:
        raise ValueError(f"{path} has no tick: there is nothing to convert")
    return log


def _xyz(values: Sequence[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class _MapFrame:
    """CARLA's world frame to Autoware's ``map`` frame (docs/trajectory_log.md,
    ``coordinate.transform.to_map_frame``): ``y`` flips and the map's offsets
    are added; ``roll`` keeps its sign, ``pitch`` and ``yaw`` flip."""

    offset: npt.NDArray[np.float64]

    @classmethod
    def of(cls, record: Optional[Mapping[str, Any]]) -> _MapFrame:
        mgrs = (record or {}).get("mgrs_offset") or (0.0, 0.0)
        z = float((record or {}).get("z_offset") or 0.0)
        return cls(np.array([float(mgrs[0]), float(mgrs[1]), z]))

    def positions(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """(N, 3) CARLA locations as map-frame positions."""
        return xyz * _FLIP_Y + self.offset

    @staticmethod
    def angles(rpy_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """(N, 3) CARLA rotations (roll, pitch, yaw; degrees) as map-frame
        Euler angles (radians)."""
        return np.radians(rpy_deg) * np.array([1.0, -1.0, -1.0])


def _rotations(angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(N, 3, 3) body-to-map rotations of (N, 3) roll/pitch/yaw (ZYX)."""
    cr, cp, cw = np.cos(angles).T
    sr, sp, sw = np.sin(angles).T
    return np.stack(
        [
            np.stack(
                [cw * cp, cw * sp * sr - sw * cr, cw * sp * cr + sw * sr], axis=-1
            ),
            np.stack(
                [sw * cp, sw * sp * sr + cw * cr, sw * sp * cr - cw * sr], axis=-1
            ),
            np.stack([-sp, cp * sr, cp * cr], axis=-1),
        ],
        axis=-2,
    )


def _quaternions(angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(N, 4) unit quaternions ``(x, y, z, w)`` of (N, 3) roll/pitch/yaw (ZYX)."""
    cr, cp, cw = np.cos(angles / 2).T
    sr, sp, sw = np.sin(angles / 2).T
    return np.stack(
        [
            sr * cp * cw - cr * sp * sw,
            cr * sp * cw + sr * cp * sw,
            cr * cp * sw - sr * sp * cw,
            cr * cp * cw + sr * sp * sw,
        ],
        axis=-1,
    )


def _array(
    values: npt.ArrayLike, dtype: str, mask: Optional[npt.ArrayLike] = None
) -> pb.NumericArray:
    """*values* as a ``NumericArray``: little-endian bytes of *dtype*, zeroed
    where *mask* marks them missing."""
    data = np.array(values, dtype=dtype if dtype == "?" else "<" + dtype)
    result = pb.NumericArray(dtype=_DTYPES[dtype], shape=data.shape)
    if mask is not None:
        missing = np.broadcast_to(np.asarray(mask, dtype=bool), data.shape)
        if missing.any():
            data[missing] = 0
            result.mask = missing.tobytes()
    result.data = data.tobytes()
    return result


def _register_in_map_frame(target: Any) -> None:
    """Register *target*'s axes (a ``T4Scene``'s or a ``MapReference``'s) as
    the world's: the scene's poses and the map's points are both in Autoware's
    ``map`` frame, which is the frame the two are registered in."""
    target.global_origin.CopyFrom(pb.Pose2D(cos_yaw=1.0))
    target.global_origin_3d.CopyFrom(
        pb.RigidTransform3D(
            rotation=pb.Matrix3x3(m00=1.0, m11=1.0, m22=1.0), translation=pb.Vector3()
        )
    )


def _is_ego(role: str) -> bool:
    return role.lower() in EGO_ROLES


class _StopLine(NamedTuple):
    """Where a lanelet a traffic light governs stops its traffic."""

    lanelet_id: int
    #: The stop line as (S, 2, 2) segments in the map frame.
    segments: npt.NDArray[np.float64]
    #: The lanelet's heading there (radians).
    heading: float


class _Map(NamedTuple):
    """A run's map, ready for its scenes: the resource, the source file's
    SHA-256, and the stop lines of the lanelets its lights govern."""

    resource: pb.MapResource
    digest: str
    stop_lines: tuple[_StopLine, ...]


@functools.lru_cache(maxsize=4)
def _read_map(
    lanelet2_path: Path,
    xodr_path: Optional[Path],
    projector_type: Optional[str],
    _mtime: int,
) -> _Map:
    """A map as the run read it, encoded once however many logs ran on it
    (``_mtime`` ties the cache to the file's contents)."""
    from ..sweeper.map_loader import load_lanelet2_map  # noqa: PLC0415

    lanelet_map = load_lanelet2_map(lanelet2_path, xodr_path, projector_type)
    digest = hashlib.sha256(lanelet2_path.read_bytes()).hexdigest()
    return _Map(map_resource(lanelet_map), digest, tuple(_stop_lines(lanelet_map)))


def _load_map(
    log: _Log,
    lanelet2_path: Optional[Path],
    xodr_path: Optional[Path],
    projector_type: Optional[str],
) -> Optional[_Map]:
    """The run's Lanelet2 map, read as the run read it (so it lands in the frame
    the log's ``mgrs_offset`` and ``z_offset`` convert the poses into);
    ``None`` when there is none to read."""
    record = log.map or {}
    lanelet2_record = record.get("lanelet2") or {}
    if lanelet2_path is None:
        if not lanelet2_record.get("path"):
            return None
        lanelet2_path = Path(lanelet2_record["path"])
        if not lanelet2_path.is_file():
            logger.warning(
                "The run's Lanelet2 map %s is not here; the scene is written without a "
                "map (pass lanelet2_path to give it one)",
                lanelet2_path,
            )
            return None
    if xodr_path is None and (record.get("opendrive") or {}).get("path"):
        candidate = Path(record["opendrive"]["path"])
        xodr_path = candidate if candidate.is_file() else None
    loaded = _read_map(
        lanelet2_path.resolve(),
        None if xodr_path is None else xodr_path.resolve(),
        projector_type if projector_type is not None else record.get("projector_type"),
        lanelet2_path.stat().st_mtime_ns,
    )
    if lanelet2_record.get("sha256") and lanelet2_record["sha256"] != loaded.digest:
        logger.warning(
            "%s is not the Lanelet2 map the run read (SHA-256 %s, the log says %s)",
            lanelet2_path,
            loaded.digest,
            lanelet2_record["sha256"],
        )
    return loaded


def _stop_lines(lanelet_map: Any) -> list[_StopLine]:
    """Every lanelet a traffic light governs, and its stop line: its traffic
    light regulatory element's, else the lanelet's end."""
    from ..utils.stop_line import regulatory_element_stop_lines  # noqa: PLC0415

    governed = []
    for lanelet in lanelet_map.laneletLayer:
        elements = [
            element
            for element in lanelet.regulatoryElements
            if element.attributes["subtype"] == "traffic_light"
        ]
        centerline = np.array([(p.x, p.y) for p in lanelet.centerline])
        if not elements or len(centerline) < 2:
            continue
        line = next(
            (
                found[0]
                for element in elements
                if (found := regulatory_element_stop_lines(element, set()))
            ),
            None,
        )
        ends: list[Any] = (
            [lanelet.leftBound[-1], lanelet.rightBound[-1]]
            if line is None or len(line) < 2
            else list(line)
        )
        points = np.array([(p.x, p.y) for p in ends])
        direction = centerline[-1] - centerline[-2]
        governed.append(
            _StopLine(
                lanelet.id,
                np.stack([points[:-1], points[1:]], axis=1),
                math.atan2(direction[1], direction[0]),
            )
        )
    return governed


def _distance_to(
    segments: npt.NDArray[np.float64], point: npt.NDArray[np.float64]
) -> float:
    """The distance from *point* to the nearest of (S, 2, 2) *segments*."""
    start, along = segments[:, 0], segments[:, 1] - segments[:, 0]
    t = ((point - start) * along).sum(axis=1) / np.maximum(
        (along**2).sum(axis=1), 1e-12
    )
    nearest = start + np.clip(t, 0.0, 1.0)[:, None] * along
    return float(np.hypot(*(nearest - point).T).min())


def _lane_lights(
    log: _Log, stop_lines: Sequence[_StopLine], frame: _MapFrame
) -> dict[int, int]:
    """The CARLA light (its id) governing each lanelet that has a traffic
    light: the nearest light with a stop-line waypoint on the lanelet's stop
    line, facing the lanelet's way."""
    stops = [
        (
            int(light["id"]),
            frame.positions(np.array(stop[:3], dtype=np.float64))[:2],
            -math.radians(float(stop[3])),
        )
        for light in log.lights
        for stop in light.get("stops") or ()
    ]
    governed: dict[int, int] = {}
    for line in stop_lines:
        nearest = min(
            (
                (_distance_to(line.segments, point), light_id)
                for light_id, point, yaw in stops
                if math.cos(yaw - line.heading) >= 0.5
            ),
            default=None,
        )
        if nearest is not None and nearest[0] <= _STOP_LINE_MATCH_M:
            governed[line.lanelet_id] = nearest[1]
    return governed


def _fill_metadata(scene: pb.T4Scene, log: _Log, stem: str) -> None:
    metadata = scene.scene_metadata
    metadata.scene_id = stem
    if log.world.get("episode") is not None:
        metadata.recording_id = str(log.world["episode"])
    metadata.tags.append("autoware_carla_scenario")
    if log.map is not None and log.map.get("name"):
        metadata.tags.append(f"map:{log.map['name']}")
    if log.result is not None:
        metadata.tags.append(
            "result:passed" if log.result.get("passed") else "result:failed"
        )


def _fill_ego(
    scene: pb.T4Scene,
    log: _Log,
    frames: Sequence[int],
    ego_id: int,
    ego: _Actor,
    frame: _MapFrame,
    rear_axle: float,
) -> list[Optional[npt.NDArray[np.float64]]]:
    """Write the ego's rows; return its origin (map-frame x, y) on each row,
    ``None`` on a row it is not on."""
    count = len(frames)
    poses = np.zeros((count, 3))
    poses_3d = np.tile(np.eye(4), (count, 1, 1))
    velocities = np.zeros((count, 2))
    speeds = np.zeros(count)
    origins: list[Optional[npt.NDArray[np.float64]]] = [None] * count
    for row, carla_frame in enumerate(frames):
        states = log.states(carla_frame)
        mine = states[states[:, 0] == ego_id]
        if not len(mine):
            continue
        state = mine[0]
        rotation = _rotations(_MapFrame.angles(state[None, 4:7]))[0]
        origin = frame.positions(state[1:4])
        base_link = origin + rotation[:, 0] * rear_axle
        velocity = state[7:10] * _FLIP_Y
        poses[row] = (
            base_link[0],
            base_link[1],
            math.atan2(rotation[1, 0], rotation[0, 0]),
        )
        poses_3d[row, :3, :3] = rotation
        poses_3d[row, :3, 3] = base_link
        velocities[row] = velocity[:2]
        speeds[row] = float(velocity @ rotation[:, 0])
        origins[row] = origin[:2]
    available = np.array([origin is not None for origin in origins])
    missing = ~available
    message = scene.ego
    message.poses.CopyFrom(_array(poses, "f8", missing[:, None]))
    message.poses_3d.CopyFrom(_array(poses_3d, "f8", missing[:, None, None]))
    message.velocities.CopyFrom(_array(velocities, "f8", missing[:, None]))
    message.longitudinal_speeds.CopyFrom(_array(speeds, "f8", missing))
    message.available.CopyFrom(_array(available, "?"))
    ex, ey, _ = ego.extent
    message.shape.length = 2.0 * ex
    message.shape.width = 2.0 * ey
    message.shape.base_link_to_front = ego.center[0] + ex - rear_axle
    message.shape.base_link_to_rear_axle = 0.0
    return origins


def _fill_annotations(
    scene: pb.T4Scene,
    log: _Log,
    frames: Sequence[int],
    ego_origins: Sequence[Optional[npt.NDArray[np.float64]]],
    frame: _MapFrame,
    max_distance_m: Optional[float],
) -> None:
    """Write every other road user's box on each row, a tick at a time."""
    others = {i: actor for i, actor in log.actors.items() if not _is_ego(actor.role)}
    rows: list[tuple[npt.NDArray[Any], ...]] = []
    for row, carla_frame in enumerate(frames):
        states = log.states(carla_frame)
        ids = states[:, 0].astype(np.int64)
        keep = np.array(
            [int(i) in others and log.on_scene(int(i), carla_frame) for i in ids],
            dtype=bool,
        )
        origins = frame.positions(states[:, 1:4])
        centre = ego_origins[row]
        if max_distance_m is not None and centre is not None:
            keep &= np.hypot(*(origins[:, :2] - centre).T) <= max_distance_m
        states, ids, origins = states[keep], ids[keep], origins[keep]
        actors = [others[int(i)] for i in ids]
        angles = _MapFrame.angles(states[:, 4:7])
        rotations = _rotations(angles)
        # The box's centre is given in CARLA's (left-handed) actor frame.
        centres = np.array([a.center for a in actors]).reshape(-1, 3) * _FLIP_Y
        extents = np.array([a.extent for a in actors]).reshape(-1, 3)
        rows.append(
            (
                origins + np.einsum("nij,nj->ni", rotations, centres),
                2.0 * extents[:, [1, 0, 2]],  # (width, length, height)
                np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]),
                states[:, 7:9] * _FLIP_Y[:2],
                np.array(
                    [
                        _PEDESTRIAN if a.type_id.startswith("walker.") else _VEHICLE
                        for a in actors
                    ],
                    dtype=np.int64,
                ),
                ids,
                _quaternions(angles),
            )
        )
    positions, dimensions, headings, velocities, labels, track_ids, orientations = (
        np.concatenate(column) for column in zip(*rows)
    )
    values = scene.annotations.values
    values.positions.CopyFrom(_array(positions, "f8"))
    values.dimensions.CopyFrom(_array(dimensions, "f8"))
    values.headings.CopyFrom(_array(headings, "f8"))
    values.velocities.CopyFrom(_array(velocities, "f8"))
    values.labels.CopyFrom(_array(labels, "i8"))
    values.orientations.CopyFrom(_array(orientations, "f8"))
    values.track_ids.values.extend(pb.OptionalString(value=str(i)) for i in track_ids)
    counts = [len(row[0]) for row in rows]
    scene.annotations.offsets.CopyFrom(
        _array(np.concatenate([[0], np.cumsum(counts)]), "i8")
    )
    scene.annotations.available.CopyFrom(_array([f in log.rows for f in frames], "?"))


def _fill_lights(
    scene: pb.T4Scene, log: _Log, frames: Sequence[int], lane_lights: Mapping[int, int]
) -> None:
    """Write the effective state of each governed lanelet on every row.

    Only effective lane states: the log has CARLA's lights, not the map's
    regulatory elements and their bulbs, so raw groups are not observed
    (``group_available``), rather than observed empty.
    """
    lanes = sorted(lane_lights)
    available = np.array([f in log.rows for f in frames])
    states = np.array(
        [
            [
                _LANE_STATES.get(log.rows[f][1].get(lane_lights[lane], "").lower(), 0)
                for lane in lanes
            ]
            for f in frames
            if f in log.rows
        ],
        dtype=np.int64,
    ).reshape(-1)
    lights = scene.traffic_lights
    lights.values.lane_ids.CopyFrom(_array(np.tile(lanes, int(available.sum())), "i8"))
    lights.values.lane_states.CopyFrom(_array(states, "i8"))
    # A CARLA light is one round lamp: no lane's state comes from an arrow.
    lights.values.lane_is_arrow.CopyFrom(_array(np.zeros(len(states)), "?"))
    lights.group_offsets.CopyFrom(_array(np.zeros(len(frames) + 1), "i8"))
    lights.lane_offsets.CopyFrom(
        _array(np.concatenate([[0], np.cumsum(available * len(lanes))]), "i8")
    )
    lights.available.CopyFrom(_array(available, "?"))
    lights.group_available.CopyFrom(_array(np.zeros(len(frames)), "?"))
    lights.lane_available.CopyFrom(_array(available, "?"))


def trajectory_log_to_t4_scene(
    log_path: Path | str,
    *,
    step_s: float = 0.1,
    current_index: int = 0,
    max_distance_m: Optional[float] = 100.0,
    rear_axle_offset_m: Optional[float] = None,
    with_map: bool = True,
    lanelet2_path: Optional[Path | str] = None,
    xodr_path: Optional[Path | str] = None,
    projector_type: Optional[str] = None,
) -> T4SceneExport:
    """Convert a trajectory log into a T4Scene and the map resource it names.

    Args:
        log_path: The run's ``{stem}.trajectory.jsonl``.
        step_s: The scene's time step: a multiple of the log's tick
            (``FIXED_DELTA_SECONDS``).  The default, 0.1 s, is the T4 datasets'.
        current_index: The scene's decision anchor, the row its history ends
            at; 0 makes the whole run its recorded future.  Negative counts
            from the end.
        max_distance_m: The road users within this distance of the ego are
            the scene's objects; ``None`` takes all of them.
        rear_axle_offset_m: Where the ego's ``base_link``, the rear axle, is
            from its actor origin along its x axis (negative: behind).  By
            default half its box behind, as the run itself takes it on CARLA
            UE5 (``driver.rear_axle_offset_m``).
        with_map: Write the run's Lanelet2 map as the scene's map.
        lanelet2_path: The Lanelet2 map to use instead of the one the log
            names (a map moved since the run).
        xodr_path: The OpenDRIVE file read for the map's origin, likewise.
        projector_type: The projection to read the map with, instead of the
            run's.

    Raises:
        ValueError: If the log has no tick or no ego, or *step_s* or
            *current_index* does not fit it.
    """
    path = Path(log_path)
    step_frames = round(step_s / FIXED_DELTA_SECONDS)
    if step_frames < 1 or not math.isclose(
        step_frames * FIXED_DELTA_SECONDS, step_s, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"step_s={step_s} is not a multiple of the log's tick ({FIXED_DELTA_SECONDS} s)"
        )
    log = _read_log(path, step_frames)
    ego_id = next((i for i, actor in log.actors.items() if _is_ego(actor.role)), None)
    if ego_id is None:
        raise ValueError(f"{path} has no ego (an actor whose role is 'ego' or 'hero')")
    ego = log.actors[ego_id]
    frames = list(range(log.first_frame, log.last_frame + 1, step_frames))
    if not -len(frames) <= current_index < len(frames):
        raise ValueError(
            f"current_index={current_index} is outside the scene's {len(frames)} rows"
        )

    scene = pb.T4Scene(current_index=current_index % len(frames))
    stem = (
        path.name[: -len(LOG_SUFFIX)] if path.name.endswith(LOG_SUFFIX) else path.stem
    )
    _fill_metadata(scene, log, stem)
    scene.timestamps_ns.CopyFrom(
        _array([(f - frames[0]) * _TICK_NS for f in frames], "i8")
    )
    scene.frame_indices.CopyFrom(_array(frames, "i8"))
    _register_in_map_frame(scene)

    frame = _MapFrame.of(log.map)
    rear_axle = (
        -0.5 * ego.extent[0] if rear_axle_offset_m is None else rear_axle_offset_m
    )
    ego_origins = _fill_ego(scene, log, frames, ego_id, ego, frame, rear_axle)
    _fill_annotations(scene, log, frames, ego_origins, frame, max_distance_m)

    loaded = None
    if with_map:
        loaded = _load_map(
            log,
            None if lanelet2_path is None else Path(lanelet2_path),
            None if xodr_path is None else Path(xodr_path),
            projector_type,
        )
    if loaded is None:
        return T4SceneExport(scene=scene, map_resource=None)
    reference = scene.map
    reference.sha256 = loaded.resource.sha256
    reference.scope = pb.MapReference.FULL
    reference.has_elevation = True
    reference.map_id = f"lanelet2-sha256:{loaded.digest}"
    _register_in_map_frame(reference)
    _fill_lights(scene, log, frames, _lane_lights(log, loaded.stop_lines, frame))
    return T4SceneExport(scene=scene, map_resource=loaded.resource)


def write_t4_scene(
    log_path: Path | str,
    output_dir: Optional[Path | str] = None,
    **options: Any,
) -> tuple[Path, Optional[Path]]:
    """Convert *log_path* and write ``{stem}.t4scene.pb`` and, with a map,
    ``{stem}.t4scene.map.pb`` beside it (or in *output_dir*).

    *options* are :func:`trajectory_log_to_t4_scene`'s.

    Returns:
        The scene's path and the map resource's (``None`` without a map).
    """
    path = Path(log_path)
    export = trajectory_log_to_t4_scene(path, **options)
    directory = path.parent if output_dir is None else Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = export.scene.scene_metadata.scene_id
    scene_path = directory / f"{stem}{SCENE_SUFFIX}"
    scene_path.write_bytes(export.scene.SerializeToString(deterministic=True))
    map_path: Optional[Path] = None
    if export.map_resource is not None:
        map_path = directory / f"{stem}{MAP_SUFFIX}"
        map_path.write_bytes(export.map_resource.SerializeToString(deterministic=True))
    return scene_path, map_path
