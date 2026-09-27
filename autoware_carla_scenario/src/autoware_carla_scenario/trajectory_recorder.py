"""Record a scenario's vehicles, pedestrians and traffic lights as JSON lines.

The scenario runner steps the world itself, so it records right after each
``world.tick()`` from the snapshot that tick produced -- every tick, with no
second CARLA client following the world from outside (one that polls the server
while the runner loads a map can leave CARLA 0.10 answering every call with
``std::exception``).  Consumers (the carla-hub run report) replay the file.

One JSON object per line, in CARLA's (left-handed) frame, metres/degrees:

* ``{"type": "world", "map": ..., "episode": ...}`` -- once, when recording
  starts.
* ``{"type": "actor", "id", "type_id", "role", "extent": [x,y,z],
  "center": [x,y,z]}`` -- the first time a vehicle/walker is seen; ``extent`` is
  the bounding box's half size, ``center`` its offset from the actor origin.
* ``{"type": "traffic_light", "id", "opendrive_id", "location": [x,y,z],
  "boxes": [[x, y, z, ex, ey, ez, yaw], ...], "stops": [[x, y, z, yaw], ...]}``
  -- every traffic light, once: its light heads' boxes (world frame, half
  sizes) and the waypoints of its stop lines.
* ``{"type": "tick", "frame", "t", "actors": [[id, x, y, z, roll, pitch, yaw,
  vx, vy, vz], ...], "control": {id: [throttle, steer, brake, gear]},
  "lights": {id: "Red"|"Yellow"|"Green"|"Off"|"Unknown"}}`` -- ``t`` is the
  scenario's simulated time (the clock pass/fail conditions report);
  ``control`` only for the ego; ``lights`` only the lights whose state changed
  since the last written tick (all of them in the first).  Ticks without any
  vehicle/walker are not written.

The file is line-buffered, so a killed run loses at most the tick being written.
Recording is best effort: an error here is logged once and recording stops, it
never fails the scenario.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO, Any

import carla

logger = logging.getLogger(__name__)

#: Actor type prefixes whose motion is recorded.
RECORDED_KINDS = ("vehicle.", "walker.")
#: ``role_name`` values whose control inputs are recorded.
EGO_ROLES = ("ego", "hero")


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
    return str(tl.state).rsplit(".", 1)[-1]


class TrajectoryRecorder:
    """Writes the world's motion to *path*, one :meth:`record` call per tick."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._out: IO[str] | None = None
        self._known: dict[int, Any] = {}  # actor id -> actor, or None if not recorded
        self._egos: dict[int, Any] = {}
        self._lights: list[Any] = []
        self._shown: dict[int, str] = {}

    @property
    def active(self) -> bool:
        return self._out is not None

    def start(self, world: carla.World) -> None:
        """Open the file and write the world and its traffic lights."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._out = self.path.open("w", buffering=1, encoding="utf-8")
            self._emit(
                {"type": "world", "map": world.get_map().name, "episode": world.id}
            )
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

    def _emit(self, record: dict[str, Any]) -> None:
        assert self._out is not None
        self._out.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _fail(self, what: str) -> None:
        logger.warning("Trajectory recording %s (%s)", what, self.path, exc_info=True)
        self.close()

    def _record(self, world: carla.World, elapsed: float) -> None:
        snapshot = world.get_snapshot()
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
        if not rows:
            return
        present = {row[0] for row in rows}
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


__all__ = ["EGO_ROLES", "RECORDED_KINDS", "TrajectoryRecorder"]
