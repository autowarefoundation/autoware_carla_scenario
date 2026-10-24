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

__all__ = ["Obstacle", "first_in_way", "occupied"]

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
        #: Each actor's heading (deg) this frame, by id.
        self.yaws: dict[int, float] = {}
        #: Each road user's half length and half width (m), by id.
        self.boxes: dict[int, tuple[float, float]] = {}


_CACHE = _Cache()


def _states(world: Any) -> list[tuple[int, float, float, float, float]]:
    """``(id, x, y, z, speed)`` of every actor this frame."""
    snapshot = world.get_snapshot()
    frame = int(snapshot.frame)
    episode = int(world.id) if hasattr(world, "id") else 0
    if episode != _CACHE.episode:
        _CACHE.kinds = {}
        _CACHE.boxes = {}
        _CACHE.episode = episode
        _CACHE.frame = -1  # a new episode's frames are read afresh
    if frame != _CACHE.frame:
        states = []
        yaws: dict[int, float] = {}
        for seen in snapshot:
            transform = seen.get_transform()
            location = transform.location
            yaws[int(seen.id)] = float(transform.rotation.yaw)
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
        _CACHE.yaws = yaws
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
                _CACHE.boxes[actor_id] = (float(extent.x), float(extent.y))
        _CACHE.kinds[actor_id] = (kind, radius)
    return _CACHE.kinds[actor_id]


def overlaps(
    world: Any,
    self_id: int,
    x: float,
    y: float,
    yaw_deg: float,
    half_length: float,
    half_width: float,
    margin: float = 0.3,
) -> Optional[int]:
    """The id of a vehicle or walker whose outline (a rectangle, its
    bounding box) comes within *margin* of the rectangle *half_length* by
    *half_width* at CARLA ``(x, y)`` facing *yaw_deg*, or ``None``: a road
    user standing where one of that size would be put.  One in the lane
    beside it does not count, as a circle round each would."""
    reach = math.hypot(half_length, half_width) + margin
    for actor_id, ox, oy, oz, _speed in _states(world):
        if actor_id == self_id or oz < _HIDDEN_BELOW_M:
            continue
        kind, size = _kind(world, actor_id)
        if not kind or math.hypot(ox - x, oy - y) > reach + size:
            continue
        other_length, other_width = _CACHE.boxes.get(actor_id, (size, size))
        if _rectangles_meet(
            (x, y, math.radians(yaw_deg), half_length + margin, half_width + margin),
            (ox, oy, math.radians(_CACHE.yaws.get(actor_id, 0.0)), other_length, other_width),
        ):
            return actor_id
    return None


def _rectangles_meet(
    a: tuple[float, float, float, float, float],
    b: tuple[float, float, float, float, float],
) -> bool:
    """Whether two rectangles ``(x, y, heading rad, half length, half width)``
    overlap (separating axes)."""
    ax, ay, ah, al, aw = a
    bx, by, bh, bl, bw = b
    dx, dy = bx - ax, by - ay
    for heading in (ah, ah + math.pi / 2.0, bh, bh + math.pi / 2.0):
        ux, uy = math.cos(heading), math.sin(heading)
        gap = abs(dx * ux + dy * uy)
        reach_a = abs(al * (math.cos(ah) * ux + math.sin(ah) * uy)) + abs(
            aw * (-math.sin(ah) * ux + math.cos(ah) * uy)
        )
        reach_b = abs(bl * (math.cos(bh) * ux + math.sin(bh) * uy)) + abs(
            bw * (-math.sin(bh) * ux + math.cos(bh) * uy)
        )
        if gap > reach_a + reach_b:
            return False
    return True


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
