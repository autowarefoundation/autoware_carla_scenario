"""Record a scenario run -- its vehicles, pedestrians, traffic lights, the ego's
plan, its conditions and verdict -- as JSON lines.

The scenario runner steps the world itself, so it records right after each
``world.tick()`` from the snapshot that tick produced -- every tick, with no
second CARLA client following the world from outside (one that polls the server
while the runner loads a map can leave CARLA 0.10 answering every call with
``std::exception``).  Consumers (the carla-hub run report) replay the file.

One JSON object per line, in CARLA's (left-handed) frame, metres/degrees
(format version :data:`SCHEMA_VERSION`):

* ``{"type": "world", "schema_version": 2, "map": ..., "episode": ...}`` --
  once, when recording starts.
* ``{"type": "map", "name", "source", "lanelet2", "opendrive",
  "projector_info", "projector_type", "mgrs_offset", "z_offset"}`` -- once,
  next: the map files the run read, each with its path and SHA-256.
* ``{"type": "actor", "id", "type_id", "role", "extent": [x,y,z],
  "center": [x,y,z]}`` -- the first time a vehicle/walker is seen; ``extent`` is
  the bounding box's half size, ``center`` its offset from the actor origin.
* ``{"type": "traffic_light", "id", "opendrive_id", "location": [x,y,z],
  "boxes": [[x, y, z, ex, ey, ez, yaw], ...], "stops": [[x, y, z, yaw], ...]}``
  -- every traffic light, once: its light heads' boxes (world frame, half
  sizes) and the waypoints of its stop lines.
* ``{"type": "warmup", "ticks", "start_frame", "end_frame"}`` -- once, before
  the first tick: how many frames of the CARLA recording are the warm-up,
  which the file leaves out.
* ``{"type": "actor_gone", "id", "frame", "t"}`` -- before the first tick a
  recorded vehicle/walker is missing from.
* ``{"type": "tick", "frame", "t", "actors": [[id, x, y, z, roll, pitch, yaw,
  vx, vy, vz], ...], "control": {id: [throttle, steer, brake, gear]},
  "lights": {id: "Red"|"Yellow"|"Green"|"Off"|"Unknown"}}`` -- ``t`` is the
  scenario's simulated time (the clock pass/fail conditions report);
  ``control`` only for the ego; ``lights`` only the lights whose state changed
  since the last written tick (all of them in the first).  Ticks without any
  vehicle/walker are not written.
* ``{"type": "markers", "frame", "t", "actor", "polylines": [...]}`` -- after
  a tick on which the ego's debug markers (its plan) changed.
* ``{"type": "condition", "frame", "t", "role", "index", "label",
  "satisfied"}`` -- when a pass/fail condition is first checked, and whenever
  it changes.
* ``{"type": "action", "frame", "t", "phase", "index", "label", "action_type",
  "state"}`` -- an action's lifecycle state (:class:`ActionState`'s value) on
  the first tick of the loop, and whenever it changes.
* ``{"type": "adversity", "frame", "t", "label", "agent", "kind", "location",
  "probability", "is_probability", "weight", "avoidable", "victims",
  "response"}`` -- an adversity NADE sampled, on its tick (#397).
* ``{"type": "waypoint_arrival", "frame", "t", "label", "entity", "vertex",
  "kind", "mode", "target_t", "actual_t", "error_s"}`` -- a routed entity
  reached a waypoint with a time or a departure condition: when it was due
  (its time; the moment its condition first held) and when it arrived.
* ``{"type": "collision", "frame", "t", "actors": [role, role], "ids":
  [id, id], "impulse", "ego", "phase", "duration"}`` -- a vehicle other than
  the ego began (``phase: "start"``) or ended (``"end"``, with how long it
  lasted) a contact with a vehicle or a walker
  (:mod:`~autoware_carla_scenario.npc_collisions`).
* ``{"type": "walker_fallen", "frame", "t", "id", "role", "drop", "tilt"}``
  -- a walker was seen fallen over, once per walker.
* ``{"type": "spawn", "frame", "t", "label", "entity", "id"}`` -- a
  ``SpawnAction`` brought an entity spawned hidden into the scene.
* ``{"type": "despawn", "frame", "t", "label", "entity", "id", "reason"}``
  -- a ``DespawnAction`` (``reason: "action"``), or the run for a walker
  lying fallen (``"fallen"``), took an entity out of the world.
* ``{"type": "result", "passed", "message", "elapsed_seconds",
  "conditions"}`` -- last, the verdict.

:data:`SCHEMA_PATH` is the JSON Schema (2020-12) of a line, the format's
contract for readers in any language, and :data:`DRAFT7_SCHEMA_PATH` the same
in draft-07 (docs/trajectory_log.md).

Lines are encoded with orjson (stdlib ``json``'s output, faster; non-ASCII
text is written as UTF-8 rather than escaped) and the file is unbuffered,
so each line reaches the OS in one write and a killed run loses at most the
tick being written.
Recording is best effort: an error here is logged once and recording stops, it
never fails the scenario.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import IO, Any, Optional

import orjson
import typesafe_carla.carla as carla

from .constants import EGO_ROLES
from .adversity_event import AdversityEvent
from .coordinate.map_manager import MapManager, run_carla_map
from .utils.traffic_light import traffic_light_state_name
from .waypoint_arrival import WaypointArrival

logger = logging.getLogger(__name__)

#: The version of the format the recorder writes (``world.schema_version``).
SCHEMA_VERSION = 2
#: The JSON Schema (2020-12) every line of the file conforms to.
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "trajectory.schema.json"
#: The same schema in draft-07, for code generators that cannot read 2020-12
#: tuples (generated from SCHEMA_PATH by scripts/trajectory_schema_draft7.py).
DRAFT7_SCHEMA_PATH = Path(__file__).resolve().parent / "trajectory_recorder.schema.json"
#: Actor type prefixes whose motion is recorded.
RECORDED_KINDS = ("vehicle.", "walker.")


def _r(value: float) -> float:
    return round(value, 3)


def _light_record(tl: Any) -> dict[str, Any]:
    loc = tl.get_location()
    boxes: list[list[float]] = []
    stops: list[list[float]] = []
    try:
        for b in tl.get_light_boxes():
            boxes.append(
                [
                    _r(b.location.x),
                    _r(b.location.y),
                    _r(b.location.z),
                    _r(b.extent.x),
                    _r(b.extent.y),
                    _r(b.extent.z),
                    _r(b.rotation.yaw),
                ]
            )
        for wp in tl.get_stop_waypoints():
            t = wp.transform
            stops.append(
                [
                    _r(t.location.x),
                    _r(t.location.y),
                    _r(t.location.z),
                    _r(t.rotation.yaw),
                ]
            )
    except RuntimeError:
        pass
    try:
        opendrive_id = tl.get_opendrive_id()
    except (RuntimeError, AttributeError):
        opendrive_id = ""
    return {
        "type": "traffic_light",
        "id": tl.id,
        "opendrive_id": opendrive_id,
        "location": [_r(loc.x), _r(loc.y), _r(loc.z)],
        "boxes": boxes,
        "stops": stops,
    }


def _light_state(tl: Any) -> str:
    return traffic_light_state_name(tl.state)


#: (path, size, mtime) -> SHA-256: a sweep reads the same map case after case.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_ref(path: Optional[Path]) -> Optional[dict[str, Any]]:
    """*path* as the map record names a file: where, how big, its SHA-256."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha256()
        with resolved.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        _DIGESTS[key] = digest.hexdigest()
    return {"path": str(resolved), "sha256": _DIGESTS[key], "bytes": stat.st_size}


def _projector_info(lanelet2_path: Optional[Path]) -> Optional[dict[str, Any]]:
    """The Lanelet2 map's ``map_projector_info.yaml``, as read, or None."""
    if lanelet2_path is None:
        return None
    import yaml  # noqa: PLC0415 - read once a run, where it is needed

    from .coordinate.projection import PROJECTOR_INFO_FILENAME  # noqa: PLC0415

    info_path = Path(lanelet2_path).parent / PROJECTOR_INFO_FILENAME
    if not info_path.is_file():
        return None
    info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    return info if isinstance(info, dict) else None


def _map_offsets() -> tuple[Optional[list[float]], Optional[float]]:
    """The loaded Lanelet2 map's MGRS offset and elevation offset, if any."""
    mm = MapManager.get_instance()
    try:
        ox, oy = mm.mgrs_offset
        z_offset = mm.z_offset
    except RuntimeError:  # no Lanelet2 map loaded
        return None, None
    return [_r(ox), _r(oy)], _r(z_offset)


def _metadata(what: str, read: Any, default: Any = None) -> Any:
    """``read()``, or *default* with a warning when it fails: the map record's
    description of the map must not cost the run its trajectory log."""
    try:
        return read()
    except Exception:  # noqa: BLE001 - best effort, like the whole log
        logger.warning(
            "Trajectory recording: could not read %s; the map record leaves it null",
            what,
            exc_info=True,
        )
        return default


def _polylines(markers: Any) -> list[dict[str, Any]]:
    return [
        {
            "points": [[_r(x), _r(y), _r(z)] for x, y, z in polyline.points],
            "color": [int(c) for c in polyline.color],
            "thickness": _r(polyline.thickness),
            "point_size": _r(polyline.point_size),
        }
        for polyline in markers.polylines
    ]


class TrajectoryRecorder:
    """Writes a run to *path*: the world's motion, one :meth:`record` call per
    tick, and what the run made of it.

    Args:
        path: The ``.jsonl`` file to write.
        map_source: Where the map came from (the run's ``map.source``), for
            the ``map`` record.
        lanelet2_path: The Lanelet2 map the run read.
        opendrive_path: The OpenDRIVE the run read.
        projector_type: The projection the config overrode the Lanelet2
            map's with, if any.
    """

    def __init__(
        self,
        path: Path,
        map_source: Optional[str] = None,
        lanelet2_path: Optional[Path] = None,
        opendrive_path: Optional[Path] = None,
        projector_type: Optional[str] = None,
    ) -> None:
        self.path = path
        self._map_source = map_source
        self._lanelet2_path = lanelet2_path
        self._opendrive_path = opendrive_path
        self._projector_type = projector_type
        self._out: IO[bytes] | None = None
        self._known: dict[int, Any] = {}  # actor id -> actor, or None if not recorded
        self._egos: dict[int, Any] = {}
        self._lights: list[Any] = []
        self._shown: dict[int, str] = {}
        # The recorded actors of the last tick, to tell the ones gone.
        self._present: set[int] = set()
        # The tick last recorded (written or not): what later records refer to.
        self._frame = 0
        self._t = 0.0
        self._markers: Any = None
        self._conditions: dict[tuple[str, int], bool] = {}
        self._actions: dict[tuple[str, int], str] = {}

    @property
    def active(self) -> bool:
        return self._out is not None

    def start(self, world: carla.World) -> None:
        """Open the file and write the world and its traffic lights."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._out = self.path.open("wb", buffering=0)
            # The run's map, not a fresh get_map(): only its name is wanted,
            # and CARLA builds the whole road network for one.
            map_name = run_carla_map(world).name
            self._emit(
                {
                    "type": "world",
                    "schema_version": SCHEMA_VERSION,
                    "map": map_name,
                    "episode": world.id,
                }
            )
            self._emit_map(map_name)
            for tl in world.get_actors().filter("traffic.traffic_light"):
                self._lights.append(tl)
                self._emit(_light_record(tl))
        except Exception:
            self._fail("could not start")

    def record(self, world: carla.World, elapsed: float) -> None:
        """Write the tick the world was just stepped to, at scenario time *elapsed*."""
        if self._out is None:
            return
        try:
            self._record(world, elapsed)
        except Exception:
            self._fail("stopped")

    def record_warmup(self, ticks: int, start_frame: int, end_frame: int) -> None:
        """Write where the warm-up ended: it took *ticks* world ticks from the
        CARLA recorder's start at *start_frame* (``-1``: no recording) to the
        scenario clock's at *end_frame*."""
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "warmup",
                    "ticks": max(int(ticks), 0),
                    "start_frame": None if start_frame < 0 else int(start_frame),
                    "end_frame": int(end_frame),
                }
            ),
        )

    def record_markers(self, markers: Any) -> None:
        """Write the ego's debug markers on the tick last recorded, when they
        are not the ones written last (an entity hands out one object per
        plan; no markers at all are written once, as none)."""
        if self._out is None:
            return
        if markers is not None and not markers:
            markers = None
        if markers is self._markers:
            return
        self._markers = markers
        egos = list(self._egos)
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "markers",
                    "frame": self._frame,
                    "t": _r(self._t),
                    "actor": egos[0] if egos else None,
                    "polylines": [] if markers is None else _polylines(markers),
                }
            ),
        )

    def record_condition(
        self, role: str, index: int, label: str, satisfied: bool
    ) -> None:
        """Write a pass/fail condition's state on the tick last recorded, when
        it was not checked before or has changed since."""
        if self._out is None:
            return
        key = (role, index)
        satisfied = bool(satisfied)
        if key in self._conditions and self._conditions[key] == satisfied:
            return
        self._conditions[key] = satisfied
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "condition",
                    "frame": self._frame,
                    "t": _r(self._t),
                    "role": role,
                    "index": index,
                    "label": label,
                    "satisfied": bool(satisfied),
                }
            ),
        )

    def record_action(
        self, phase: str, index: int, label: str, action_type: str, state: str
    ) -> None:
        """Write an action's lifecycle state (an :class:`ActionState` value) on
        the tick last recorded, when it was not written before or has changed
        since."""
        if self._out is None:
            return
        key = (phase, index)
        if self._actions.get(key) == state:
            return
        self._actions[key] = state
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "action",
                    "frame": self._frame,
                    "t": _r(self._t),
                    "phase": phase,
                    "index": index,
                    "label": label,
                    "action_type": action_type,
                    "state": state,
                }
            ),
        )

    def record_adversity(self, label: str, event: AdversityEvent) -> None:
        """Write an adversity NADE sampled (:class:`AdversityEvent`) on the
        tick last recorded, from the action labelled *label*."""
        if self._out is None:
            return
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "adversity",
                    "frame": self._frame,
                    "t": _r(self._t),
                    "label": label,
                    # Every field of the event, so none can be left out.
                    **dataclasses.asdict(event),
                }
            ),
        )

    def record_waypoint_arrival(self, label: str, arrival: WaypointArrival) -> None:
        """Write a timed waypoint reached (:class:`WaypointArrival`) by the
        routing action labelled *label*, on the tick last recorded."""
        if self._out is None:
            return
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "waypoint_arrival",
                    "frame": self._frame,
                    "t": _r(self._t),
                    "label": label,
                    "entity": arrival.entity,
                    "vertex": int(arrival.vertex),
                    "kind": arrival.kind,
                    "mode": arrival.mode,
                    "target_t": _r(arrival.target_t),
                    "actual_t": _r(arrival.actual_t),
                    "error_s": _r(arrival.error_s),
                }
            ),
        )

    def record_collision(self, collision: Any) -> None:
        """Write a collision of a vehicle other than the ego
        (:class:`~autoware_carla_scenario.npc_collisions.NpcCollision`)."""
        if self._out is None:
            return
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "collision",
                    "frame": int(collision.frame),
                    "t": _r(collision.t),
                    "actors": [str(name) for name in collision.actors],
                    "ids": [int(i) for i in collision.ids],
                    "impulse": _r(collision.impulse),
                    "ego": bool(collision.ego),
                    "phase": str(collision.phase),
                    "duration": None
                    if collision.duration is None
                    else _r(collision.duration),
                }
            ),
        )

    def record_walker_fallen(self, fallen: Any) -> None:
        """Write a walker seen fallen over
        (:class:`~autoware_carla_scenario.npc_collisions.WalkerFallen`)."""
        if self._out is None:
            return
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "walker_fallen",
                    "frame": int(fallen.frame),
                    "t": _r(fallen.t),
                    "id": int(fallen.id),
                    "role": str(fallen.role),
                    "drop": _r(fallen.drop),
                    "tilt": _r(fallen.tilt),
                }
            ),
        )

    def record_spawn(self, label: str, entity: str, actor_id: int) -> None:
        """Write that the ``SpawnAction`` labelled *label* brought *entity*
        (its actor *actor_id*, ``-1`` if unknown) into the scene, on the tick
        last recorded."""
        if self._out is None:
            return
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "spawn",
                    "frame": self._frame,
                    "t": _r(self._t),
                    "label": label,
                    "entity": entity,
                    "id": None if actor_id < 0 else int(actor_id),
                }
            ),
        )

    def record_despawn(
        self, label: str, entity: str, actor_id: int, reason: str = "action"
    ) -> None:
        """Write that *label* removed *entity* (its actor *actor_id*, ``-1``
        if it had none), on the tick last recorded: a ``DespawnAction``
        (*reason* ``"action"``) or the run, a walker down (``"fallen"``)."""
        if self._out is None:
            return
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "despawn",
                    "frame": self._frame,
                    "t": _r(self._t),
                    "label": label,
                    "entity": entity,
                    "id": None if actor_id < 0 else int(actor_id),
                    "reason": reason,
                }
            ),
        )

    def record_result(self, result: Any) -> None:
        """Write the run's verdict (a :class:`ScenarioResult`), last."""
        self._guarded(
            "stopped",
            lambda: self._emit(
                {
                    "type": "result",
                    "passed": bool(result.passed),
                    "message": str(result.message),
                    "elapsed_seconds": _r(result.elapsed_seconds),
                    "conditions": [
                        {
                            "label": status.label,
                            "role": status.role,
                            "condition_type": status.condition_type,
                            "satisfied": bool(status.satisfied),
                            "message": status.message,
                        }
                        for status in result.condition_statuses
                    ],
                }
            ),
        )

    def close(self) -> None:
        """Close the file; a failing close (full disk, ...) is logged, never raised."""
        out, self._out = self._out, None
        if out is None:
            return
        try:
            out.close()
        except Exception:
            logger.warning(
                "Trajectory recording: closing %s failed", self.path, exc_info=True
            )

    def _guarded(self, what: str, write: Any) -> None:
        if self._out is None:
            return
        try:
            write()
        except Exception:
            self._fail(what)

    def _emit_map(self, map_name: str) -> None:
        """Write the map record.  What it describes the map with is read from
        files that may be missing or unreadable: a field that cannot be read
        is null, with a warning, and the log goes on."""
        offsets = _metadata("the map offsets", _map_offsets, (None, None))
        record = {
            "type": "map",
            "name": map_name.split("/")[-1],
            "source": self._map_source,
            "lanelet2": _metadata(
                "the Lanelet2 map's digest", lambda: _file_ref(self._lanelet2_path)
            ),
            "opendrive": _metadata(
                "the OpenDRIVE's digest", lambda: _file_ref(self._opendrive_path)
            ),
            "projector_info": _metadata(
                "map_projector_info.yaml",
                lambda: _projector_info(self._lanelet2_path),
            ),
            "projector_type": self._projector_type,
            "mgrs_offset": offsets[0],
            "z_offset": offsets[1],
        }
        try:
            self._emit(record)
        except orjson.JSONEncodeError:
            # A projector file holding what JSON has no type for (a YAML date).
            logger.warning(
                "Trajectory recording: map_projector_info.yaml of %s is not "
                "JSON; the map record leaves it out",
                self._lanelet2_path,
                exc_info=True,
            )
            record["projector_info"] = None
            self._emit(record)

    def _emit(self, record: dict[str, Any]) -> None:
        assert self._out is not None
        self._out.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    def _fail(self, what: str) -> None:
        logger.warning("Trajectory recording %s (%s)", what, self.path, exc_info=True)
        self.close()

    def _record(self, world: carla.World, elapsed: float) -> None:
        snapshot = world.get_snapshot()
        self._frame, self._t = int(snapshot.frame), elapsed
        new = [s.id for s in snapshot if s.id not in self._known]
        if new:
            for actor in world.get_actors(new):
                if not actor.type_id.startswith(RECORDED_KINDS):
                    self._known[actor.id] = None
                    continue
                role = actor.attributes.get("role_name", "")
                box = actor.bounding_box
                self._known[actor.id] = actor
                if role.lower() in EGO_ROLES:
                    self._egos[actor.id] = actor
                self._emit(
                    {
                        "type": "actor",
                        "id": actor.id,
                        "type_id": actor.type_id,
                        "role": role,
                        "extent": [
                            _r(box.extent.x),
                            _r(box.extent.y),
                            _r(box.extent.z),
                        ],
                        "center": [
                            _r(box.location.x),
                            _r(box.location.y),
                            _r(box.location.z),
                        ],
                    }
                )
            for actor_id in new:  # gone before it could be looked up
                self._known.setdefault(actor_id, None)

        rows = []
        for s in snapshot:
            if self._known.get(s.id) is None:
                continue
            tf, v = s.get_transform(), s.get_velocity()
            loc, rot = tf.location, tf.rotation
            rows.append(
                [
                    s.id,
                    _r(loc.x),
                    _r(loc.y),
                    _r(loc.z),
                    _r(rot.roll),
                    _r(rot.pitch),
                    _r(rot.yaw),
                    _r(v.x),
                    _r(v.y),
                    _r(v.z),
                ]
            )
        present = {row[0] for row in rows}
        for actor_id in sorted(self._present - present):
            self._emit(
                {
                    "type": "actor_gone",
                    "id": actor_id,
                    "frame": self._frame,
                    "t": _r(elapsed),
                }
            )
        self._present = present
        if not rows:
            return
        for actor_id in [i for i in self._egos if i not in present]:
            del self._egos[actor_id]  # destroyed: get_control() would return garbage
        control = {}
        for actor_id, actor in self._egos.items():
            try:
                c = actor.get_control()
            except RuntimeError:
                continue
            control[str(actor_id)] = [_r(c.throttle), _r(c.steer), _r(c.brake), c.gear]
        changed = {}
        for tl in self._lights:
            try:
                state = _light_state(tl)
            except RuntimeError:
                continue
            if self._shown.get(tl.id) != state:
                self._shown[tl.id] = changed[str(tl.id)] = state
        record: dict[str, Any] = {
            "type": "tick",
            "frame": snapshot.frame,
            "t": _r(elapsed),
            "actors": rows,
            "control": control,
        }
        if changed:
            record["lights"] = changed
        self._emit(record)


__all__ = [
    "DRAFT7_SCHEMA_PATH",
    "EGO_ROLES",
    "RECORDED_KINDS",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "TrajectoryRecorder",
]
