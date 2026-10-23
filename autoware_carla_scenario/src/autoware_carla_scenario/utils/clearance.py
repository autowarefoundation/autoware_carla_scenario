"""What stands in a road user's way: walkers ahead of a vehicle, vehicles
ahead of a walker (#521).

The TrafficManager yields to walkers by itself; what the framework drives
does not: a pedestrian walked along its route (the route walker,
:func:`~autoware_carla_scenario.entity.pedestrian_entity.steer_walker`) and a
vehicle driven by the framework's own controller (``follow`` mode, a final
approach) went straight into whatever stood there.  :func:`first_in_way`
answers for them, from one world snapshot a frame shared by every caller --
the positions of all actors in one call -- and actor kinds and sizes read
once per actor.
"""

from __future__ import annotations

import math
from typing import Any, Optional

__all__ = ["Obstacle", "first_in_way"]

#: An actor this far (m) or more below the ground is out of the world.
_HIDDEN_BELOW_M = -100.0


class Obstacle:
    """An actor in the way: where, how big (radius, m) and how fast."""

    def __init__(
        self, actor_id: int, x: float, y: float, radius: float, speed: float
    ) -> None:
        self.id = actor_id
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.ahead = 0.0
        self.side = 0.0


class _Cache:
    """One frame's actor positions, and every actor's kind and size."""

    def __init__(self) -> None:
        self.frame = -1
        self.episode = -1
        self.states: list[tuple[int, float, float, float, float]] = []
        self.kinds: dict[int, tuple[str, float]] = {}


_CACHE = _Cache()


def _states(world: Any) -> list[tuple[int, float, float, float, float]]:
    """``(id, x, y, z, speed)`` of every actor this frame."""
    snapshot = world.get_snapshot()
    frame = int(snapshot.frame)
    episode = int(world.id) if hasattr(world, "id") else 0
    if episode != _CACHE.episode:
        _CACHE.kinds = {}
        _CACHE.episode = episode
    if frame != _CACHE.frame:
        states = []
        for seen in snapshot:
            location = seen.get_transform().location
            velocity = seen.get_velocity()
            states.append(
                (
                    int(seen.id),
                    location.x,
                    location.y,
                    location.z,
                    math.hypot(velocity.x, velocity.y),
                )
            )
        _CACHE.states = states
        _CACHE.frame = frame
    return _CACHE.states


def _kind(world: Any, actor_id: int) -> tuple[str, float]:
    """``("vehicle" | "walker" | "", radius)`` of *actor_id*, read once."""
    if actor_id not in _CACHE.kinds:
        kind, radius = "", 0.0
        actor = world.get_actor(actor_id)
        if actor is not None:
            type_id = str(actor.type_id)
            if type_id.startswith("vehicle."):
                kind = "vehicle"
            elif type_id.startswith("walker."):
                kind = "walker"
            if kind:
                extent = actor.bounding_box.extent
                radius = math.hypot(float(extent.x), float(extent.y))
        _CACHE.kinds[actor_id] = (kind, radius)
    return _CACHE.kinds[actor_id]


def first_in_way(
    world: Any,
    self_id: int,
    x: float,
    y: float,
    heading_deg: float,
    kind: str,
    distance: float,
    half_width: float,
) -> Optional[Obstacle]:
    """The nearest *kind* (``"vehicle"`` or ``"walker"``) actor ahead of
    ``(x, y)``, facing *heading_deg* (CARLA), within *distance* metres along
    and *half_width* across (its own size added), or ``None``."""
    heading = math.radians(heading_deg)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    best: Optional[Obstacle] = None
    for actor_id, ox, oy, oz, speed in _states(world):
        if actor_id == self_id or oz < _HIDDEN_BELOW_M:
            continue
        dx, dy = ox - x, oy - y
        if dx * dx + dy * dy > (distance + 6.0) ** 2:
            continue
        found_kind, radius = _kind(world, actor_id)
        if found_kind != kind:
            continue
        ahead = dx * cos_h + dy * sin_h
        side = -dx * sin_h + dy * cos_h
        if ahead + radius <= 0.0 or ahead - radius > distance:
            continue
        if abs(side) > half_width + radius:
            continue
        if best is None or ahead < best.ahead:
            found = Obstacle(actor_id, ox, oy, radius, speed)
            found.ahead = ahead
            found.side = side
            best = found
    return best
