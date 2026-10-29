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
whose ego and objects are unavailable, never an interpolated one.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import orjson

from ..constants import FIXED_DELTA_SECONDS
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
LOG_SUFFIX = ".trajectory.jsonl"
#: The T4Scene written for ``{stem}.trajectory.jsonl``: ``{stem}.t4scene.pb``.
SCENE_SUFFIX = ".t4scene.pb"
#: Its map, the ``MapResource`` the scene's ``MapReference`` names.
MAP_SUFFIX = ".t4scene.map.pb"

#: The contract's object categories this module writes (``Annotations.labels``).
VEHICLE = 1
PEDESTRIAN = 7
#: The contract's effective lane states (``TrafficLights.lane_states``).
_LANE_STATES = {"red": 1, "yellow": 2, "green": 3}
#: How far (m) a light's stop-line waypoint may be from the stop line of a
#: lanelet it governs.
_STOP_LINE_TOLERANCE_M = 3.0

_TICK_NS = round(FIXED_DELTA_SECONDS * 1e9)


@dataclass(frozen=True)
class T4SceneExport:
    """A converted log: the scene, and the map resource it names (or ``None``)."""

    scene: pb.T4Scene
    map_resource: Optional[pb.MapResource]


@dataclass
class _Actor:
    """An ``actor`` record: what the log says once about a vehicle or walker."""

    id: int
    type_id: str
    role: str
    extent: tuple[float, float, float]
    center: tuple[float, float, float]


@dataclass
class _Log:
    """The records of a trajectory log the conversion uses."""

    world: dict[str, Any] = field(default_factory=dict)
    map: Optional[dict[str, Any]] = None
    lights: list[dict[str, Any]] = field(default_factory=list)
    actors: dict[int, _Actor] = field(default_factory=dict)
    ticks: list[dict[str, Any]] = field(default_factory=list)
    #: The frame an entity spawned hidden was put on its place, by actor id.
    spawned: dict[int, int] = field(default_factory=dict)
    #: The frame an entity was taken out, by actor id.
    despawned: dict[int, int] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None


def _read_log(path: Path) -> _Log:
    log = _Log()
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
            if kind == "world":
                log.world = record
            elif kind == "map":
                log.map = record
            elif kind == "traffic_light":
                log.lights.append(record)
            elif kind == "actor":
                log.actors[int(record["id"])] = _Actor(
                    id=int(record["id"]),
                    type_id=str(record["type_id"]),
                    role=str(record.get("role") or ""),
                    extent=_xyz(record["extent"]),
                    center=_xyz(record.get("center") or (0.0, 0.0, 0.0)),
                )
            elif kind == "tick":
                log.ticks.append(record)
            elif kind == "spawn" and record.get("id") is not None:
                log.spawned.setdefault(int(record["id"]), int(record["frame"]))
            elif kind == "despawn" and record.get("id") is not None:
                log.despawned.setdefault(int(record["id"]), int(record["frame"]))
            elif kind == "result":
                log.result = record
    if not log.ticks:
        raise ValueError(f"{path} has no tick: there is nothing to convert")
    return log


def _xyz(values: Sequence[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class _MapFrame:
    """CARLA's world frame to Autoware's ``map`` frame (docs/trajectory_log.md)."""

    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0

    @classmethod
    def of(cls, record: Optional[Mapping[str, Any]]) -> _MapFrame:
        if record is None:
            return cls()
        mgrs = record.get("mgrs_offset") or (0.0, 0.0)
        return cls(float(mgrs[0]), float(mgrs[1]), float(record.get("z_offset") or 0.0))

    def position(self, x: float, y: float, z: float) -> npt.NDArray[np.float64]:
        return np.array([x + self.offset_x, -y + self.offset_y, z + self.offset_z])

    @staticmethod
    def rotation(roll: float, pitch: float, yaw: float) -> npt.NDArray[np.float64]:
        """The body-to-map rotation of a CARLA rotation (degrees): ``roll``
        keeps its sign, ``pitch`` and ``yaw`` flip (``to_map_frame``)."""
        r, p, w = math.radians(roll), -math.radians(pitch), -math.radians(yaw)
        cr, sr, cp, sp, cw, sw = (
            math.cos(r),
            math.sin(r),
            math.cos(p),
            math.sin(p),
            math.cos(w),
            math.sin(w),
        )
        return np.array(
            [
                [cw * cp, cw * sp * sr - sw * cr, cw * sp * cr + sw * sr],
                [sw * cp, sw * sp * sr + cw * cr, sw * sp * cr - cw * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )


def _quaternion_xyzw(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A unit quaternion ``(x, y, z, w)`` of a proper rotation matrix."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            s / 4,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [
            s / 4,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [
            (m[0, 1] + m[1, 0]) / s,
            s / 4,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            s / 4,
            (m[1, 0] - m[0, 1]) / s,
        ]
    quaternion = np.array(q)
    return quaternion / np.linalg.norm(quaternion)


def _array(
    values: npt.ArrayLike, dtype: str, mask: Optional[npt.ArrayLike] = None
) -> pb.NumericArray:
    """*values* as a ``NumericArray``: C-order little-endian bytes of *dtype*,
    zeroed where *mask* marks them missing."""
    data = np.ascontiguousarray(
        np.asarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
    )
    codes = {
        "?": pb.NumericArray.BOOL,
        "i8": pb.NumericArray.INT64,
        "f8": pb.NumericArray.FLOAT64,
    }
    result = pb.NumericArray(dtype=codes[dtype], shape=data.shape)
    if mask is not None:
        missing = np.broadcast_to(np.asarray(mask, dtype=bool), data.shape)
        if missing.any():
            data = np.where(missing, np.zeros((), dtype=data.dtype), data)
            result.mask = np.ascontiguousarray(missing, dtype=np.uint8).tobytes()
    result.data = data.tobytes()
    return result


def _register_in_map_frame(target: Any) -> None:
    """Register *target*'s axes (a ``T4Scene``'s or a ``MapReference``'s) as
    the world's: the scene's poses and the map's points are both in Autoware's
    ``map`` frame, which is the frame the two are registered in."""
    target.global_origin.CopyFrom(pb.Pose2D(x=0.0, y=0.0, cos_yaw=1.0, sin_yaw=0.0))
    target.global_origin_3d.CopyFrom(
        pb.RigidTransform3D(
            rotation=pb.Matrix3x3(m00=1.0, m11=1.0, m22=1.0),
            translation=pb.Vector3(x=0.0, y=0.0, z=0.0),
        )
    )


def _is_ego(role: str) -> bool:
    return role.lower() in ("ego", "hero")


def _rows(
    ticks: Sequence[Mapping[str, Any]], step_frames: int
) -> tuple[list[int], dict[int, int]]:
    """The grid's CARLA frames, from the first tick to the last, and the tick
    (its index in *ticks*) on each frame the log has."""
    by_frame = {int(tick["frame"]): index for index, tick in enumerate(ticks)}
    first, last = int(ticks[0]["frame"]), int(ticks[-1]["frame"])
    return list(range(first, last + 1, step_frames)), by_frame


def _visible(actor_id: int, frame: int, log: _Log) -> bool:
    """Whether a road user is on the scene at *frame*, not parked out of sight
    before its ``SpawnAction`` or after its ``DespawnAction``."""
    spawned = log.spawned.get(actor_id)
    despawned = log.despawned.get(actor_id)
    return (spawned is None or frame >= spawned) and (
        despawned is None or frame < despawned
    )


def _load_map(
    log: _Log,
    lanelet2_path: Optional[Path],
    xodr_path: Optional[Path],
    projector_type: Optional[str],
) -> Optional[tuple[Any, Optional[str]]]:
    """The run's Lanelet2 map, read as the run read it, and its file's SHA-256;
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
    digest = hashlib.sha256(lanelet2_path.read_bytes()).hexdigest()
    if lanelet2_record.get("sha256") and lanelet2_record["sha256"] != digest:
        logger.warning(
            "%s is not the Lanelet2 map the run read (SHA-256 %s, the log says %s)",
            lanelet2_path,
            digest,
            lanelet2_record["sha256"],
        )
    # Read as MapManager reads it, so the map lands in the frame the log's
    # mgrs_offset and z_offset convert the poses into.
    import lanelet2.io  # noqa: PLC0415

    from ..coordinate.projection import load_lanelet2, map_origin  # noqa: PLC0415

    (lat, lon, _), _ = map_origin(lanelet2_path, xodr_path)
    lanelet_map, _, _ = load_lanelet2(
        lanelet2_path,
        lanelet2.io.Origin(lat, lon),
        projector_type if projector_type is not None else record.get("projector_type"),
    )
    return lanelet_map, digest


def _stop_lines(
    lanelet_map: Any,
) -> Iterator[tuple[int, npt.NDArray[np.float64], float]]:
    """Every lanelet a traffic light governs: its id, the middle of its stop
    line (the end of the lanelet without one) and its heading there."""
    for lanelet in lanelet_map.laneletLayer:
        governed = [
            element
            for element in lanelet.regulatoryElements
            if element.attributes["subtype"] == "traffic_light"
        ]
        if not governed:
            continue
        centerline = np.array([(p.x, p.y) for p in lanelet.centerline])
        if len(centerline) < 2:
            continue
        direction = centerline[-1] - centerline[-2]
        heading = math.atan2(direction[1], direction[0])
        stop = centerline[-1]
        for element in governed:
            lines = (
                element.parameters["ref_line"]
                if "ref_line" in element.parameters
                else []
            )
            if len(lines):
                points = np.array([(p.x, p.y) for p in lines[0]])
                stop = points.mean(axis=0)
                break
        yield lanelet.id, stop, heading


def _lane_lights(
    lights: Iterable[Mapping[str, Any]], lanelet_map: Any, frame: _MapFrame
) -> dict[int, int]:
    """The CARLA light (its id) governing each lanelet that has a traffic
    light: the light with a stop-line waypoint at the lanelet's stop line,
    facing the lanelet's way."""
    stops = [
        (int(light["id"]), frame.position(*stop[:3])[:2], -math.radians(float(stop[3])))
        for light in lights
        for stop in light.get("stops") or ()
    ]
    governed: dict[int, int] = {}
    if not stops:
        return governed
    for lanelet_id, stop_line, heading in _stop_lines(lanelet_map):
        best: Optional[tuple[float, int]] = None
        for light_id, point, yaw in stops:
            distance = float(np.hypot(*(point - stop_line)))
            if distance > _STOP_LINE_TOLERANCE_M or math.cos(yaw - heading) < 0.5:
                continue
            if best is None or distance < best[0]:
                best = (distance, light_id)
        if best is not None:
            governed[lanelet_id] = best[1]
    return governed


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
    log = _read_log(path)
    step_frames = round(step_s / FIXED_DELTA_SECONDS)
    if step_frames < 1 or not math.isclose(
        step_frames * FIXED_DELTA_SECONDS, step_s, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"step_s={step_s} is not a multiple of the log's tick ({FIXED_DELTA_SECONDS} s)"
        )
    egos = [actor for actor in log.actors.values() if _is_ego(actor.role)]
    if not egos:
        raise ValueError(f"{path} has no ego (an actor whose role is 'ego' or 'hero')")
    ego = egos[0]
    frame = _MapFrame.of(log.map)
    frames, by_frame = _rows(log.ticks, step_frames)
    count = len(frames)
    if not -count <= current_index < count:
        raise ValueError(
            f"current_index={current_index} is outside the scene's {count} rows"
        )
    anchor = current_index % count

    scene = pb.T4Scene(current_index=anchor)
    stem = (
        path.name[: -len(LOG_SUFFIX)] if path.name.endswith(LOG_SUFFIX) else path.stem
    )
    scene.scene_metadata.scene_id = stem
    if log.world.get("episode") is not None:
        scene.scene_metadata.recording_id = str(log.world["episode"])
    tags = ["autoware_carla_scenario"]
    if log.map is not None and log.map.get("name"):
        tags.append(f"map:{log.map['name']}")
    if log.result is not None:
        tags.append("result:passed" if log.result.get("passed") else "result:failed")
    scene.scene_metadata.tags.extend(tags)
    first = frames[0]
    scene.timestamps_ns.CopyFrom(_array([(f - first) * _TICK_NS for f in frames], "i8"))
    scene.frame_indices.CopyFrom(_array(frames, "i8"))
    _register_in_map_frame(scene)

    rear_axle = (
        -0.5 * ego.extent[0] if rear_axle_offset_m is None else rear_axle_offset_m
    )
    poses = np.zeros((count, 3))
    poses_3d = np.tile(np.eye(4), (count, 1, 1))
    velocities = np.zeros((count, 2))
    speeds = np.zeros(count)
    ego_available = np.zeros(count, dtype=bool)
    ego_xy: list[Optional[npt.NDArray[np.float64]]] = [None] * count

    positions: list[npt.NDArray[np.float64]] = []
    dimensions: list[tuple[float, float, float]] = []
    headings: list[float] = []
    object_velocities: list[tuple[float, float]] = []
    labels: list[int] = []
    track_ids: list[str] = []
    orientations: list[npt.NDArray[np.float64]] = []
    offsets = [0]
    objects_available = np.zeros(count, dtype=bool)

    lane_lights: dict[int, int] = {}
    lanelet_map = None
    resource: Optional[pb.MapResource] = None
    if with_map:
        loaded = _load_map(
            log,
            None if lanelet2_path is None else Path(lanelet2_path),
            None if xodr_path is None else Path(xodr_path),
            projector_type,
        )
        if loaded is not None:
            lanelet_map, digest = loaded
            resource = map_resource(lanelet_map)
            scene.map.sha256 = resource.sha256
            scene.map.scope = pb.MapReference.FULL
            scene.map.has_elevation = True
            _register_in_map_frame(scene.map)
            if digest is not None:
                scene.map.map_id = f"lanelet2-sha256:{digest}"
            lane_lights = _lane_lights(log.lights, lanelet_map, frame)

    light_states: dict[int, str] = {}
    lane_rows: list[list[tuple[int, int]]] = []
    lights_available = np.zeros(count, dtype=bool)
    # Light states are deltas: carry them over the ticks between rows too.
    tick_cursor = 0
    for row, carla_frame in enumerate(frames):
        while (
            tick_cursor < len(log.ticks)
            and int(log.ticks[tick_cursor]["frame"]) <= carla_frame
        ):
            for light_id, state in (log.ticks[tick_cursor].get("lights") or {}).items():
                light_states[int(light_id)] = str(state)
            tick_cursor += 1
        index = by_frame.get(carla_frame)
        if index is None:
            offsets.append(len(positions))
            lane_rows.append([])
            continue
        tick = log.ticks[index]
        states: dict[int, list[float]] = {
            int(state[0]): state for state in tick["actors"]
        }

        ego_state = states.get(ego.id)
        if ego_state is not None:
            rotation = frame.rotation(ego_state[4], ego_state[5], ego_state[6])
            origin = frame.position(ego_state[1], ego_state[2], ego_state[3])
            base_link = origin + rotation @ np.array([rear_axle, 0.0, 0.0])
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            velocity = np.array([ego_state[7], -ego_state[8], ego_state[9]])
            poses[row] = (base_link[0], base_link[1], yaw)
            poses_3d[row, :3, :3] = rotation
            poses_3d[row, :3, 3] = base_link
            velocities[row] = velocity[:2]
            speeds[row] = float(velocity @ rotation[:, 0])
            ego_available[row] = True
            ego_xy[row] = origin[:2]

        objects_available[row] = True
        for actor_id, state in states.items():
            actor = log.actors.get(actor_id)
            if actor is None or actor_id == ego.id or _is_ego(actor.role):
                continue
            if not _visible(actor_id, carla_frame, log):
                continue
            origin = frame.position(state[1], state[2], state[3])
            centre = ego_xy[row]
            if (
                max_distance_m is not None
                and centre is not None
                and float(np.hypot(*(origin[:2] - centre))) > max_distance_m
            ):
                continue
            rotation = frame.rotation(state[4], state[5], state[6])
            cx, cy, cz = actor.center
            positions.append(origin + rotation @ np.array([cx, -cy, cz]))
            ex, ey, ez = actor.extent
            dimensions.append((2.0 * ey, 2.0 * ex, 2.0 * ez))
            headings.append(math.atan2(rotation[1, 0], rotation[0, 0]))
            object_velocities.append((state[7], -state[8]))
            labels.append(
                PEDESTRIAN if actor.type_id.startswith("walker.") else VEHICLE
            )
            track_ids.append(str(actor_id))
            orientations.append(_quaternion_xyzw(rotation))
        offsets.append(len(positions))

        lights_available[row] = True
        lane_rows.append(
            [
                (
                    lanelet_id,
                    _LANE_STATES.get(light_states.get(light_id, "").lower(), 0),
                )
                for lanelet_id, light_id in sorted(lane_lights.items())
            ]
        )

    missing = ~ego_available
    ego_message = scene.ego
    ego_message.poses.CopyFrom(_array(poses, "f8", missing[:, None]))
    ego_message.poses_3d.CopyFrom(_array(poses_3d, "f8", missing[:, None, None]))
    ego_message.velocities.CopyFrom(_array(velocities, "f8", missing[:, None]))
    ego_message.longitudinal_speeds.CopyFrom(_array(speeds, "f8", missing))
    ego_message.available.CopyFrom(_array(ego_available, "?"))
    ex, ey, _ = ego.extent
    ego_message.shape.length = 2.0 * ex
    ego_message.shape.width = 2.0 * ey
    ego_message.shape.base_link_to_front = ego.center[0] + ex - rear_axle
    ego_message.shape.base_link_to_rear_axle = 0.0

    annotations = scene.annotations
    values = annotations.values
    values.positions.CopyFrom(_array(np.reshape(positions, (-1, 3)), "f8"))
    values.dimensions.CopyFrom(_array(np.reshape(dimensions, (-1, 3)), "f8"))
    values.headings.CopyFrom(_array(headings, "f8"))
    values.velocities.CopyFrom(_array(np.reshape(object_velocities, (-1, 2)), "f8"))
    values.labels.CopyFrom(_array(labels, "i8"))
    values.orientations.CopyFrom(_array(np.reshape(orientations, (-1, 4)), "f8"))
    values.track_ids.values.extend(
        pb.OptionalString(value=track) for track in track_ids
    )
    annotations.offsets.CopyFrom(_array(offsets, "i8"))
    annotations.available.CopyFrom(_array(objects_available, "?"))

    if lanelet_map is not None:
        lights = scene.traffic_lights
        lane_ids = [lanelet_id for row in lane_rows for lanelet_id, _ in row]
        lane_states = [state for row in lane_rows for _, state in row]
        lane_offsets = np.concatenate([[0], np.cumsum([len(row) for row in lane_rows])])
        # Effective lane states only: the log has CARLA's lights, not the map's
        # regulatory elements and their bulbs, so raw groups are not observed
        # (group_available), rather than observed empty.
        lights.values.lane_ids.CopyFrom(_array(lane_ids, "i8"))
        lights.values.lane_states.CopyFrom(_array(lane_states, "i8"))
        # A CARLA light is one round lamp: no lane's state comes from an arrow.
        lights.values.lane_is_arrow.CopyFrom(_array(np.zeros(len(lane_ids)), "?"))
        lights.group_offsets.CopyFrom(_array(np.zeros(count + 1), "i8"))
        lights.lane_offsets.CopyFrom(_array(lane_offsets, "i8"))
        lights.available.CopyFrom(_array(lights_available, "?"))
        lights.group_available.CopyFrom(_array(np.zeros(count), "?"))
        lights.lane_available.CopyFrom(_array(lights_available, "?"))
    return T4SceneExport(scene=scene, map_resource=resource)


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
