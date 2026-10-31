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

An actor's bounding box is not centred on its origin: CARLA places it at
``bounding_box.location`` in the actor's own frame (a car's sits about a
metre forward of its origin, between the axles).  Every rectangle here is
centred on the box -- the origin turned by the actor's heading and moved by
that offset (:func:`box_centre`) -- not on the origin (#884).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "Obstacle",
    "actor_box_centre",
    "box_centre",
    "claim_spot",
    "first_in_way",
    "overlaps",
]

#: An actor this far (m) or more below the ground is out of the world.
_HIDDEN_BELOW_M = -100.0


class Obstacle:
    """An actor in the way: where (its bounding box's centre), how big
    (radius, m) and how fast."""

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
        #: Each road user's bounding box offset from its origin, forward and
        #: to the right (m, its own frame: CARLA's ``bounding_box.location``),
        #: by id.
        self.offsets: dict[int, tuple[float, float]] = {}
        #: Spots road users were put on this frame, which its snapshot does
        #: not show yet: ``(id, box centre x, y, yaw, half length, half
        #: width)``.
        self.claims: list[tuple[int, float, float, float, float, float]] = []
        self.claims_frame = -1


_CACHE = _Cache()


def _states(world: "carla.World") -> list[tuple[int, float, float, float, float]]:
    """``(id, x, y, z, speed)`` of every actor this frame."""
    snapshot = world.get_snapshot()
    frame = int(snapshot.frame)
    episode = int(world.id) if hasattr(world, "id") else 0
    if episode != _CACHE.episode:
        _CACHE.kinds = {}
        _CACHE.boxes = {}
        _CACHE.offsets = {}
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


def _kind(world: "carla.World", actor_id: int) -> tuple[str, float]:
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
                _CACHE.offsets[actor_id] = _offset_of(actor)
        _CACHE.kinds[actor_id] = (kind, radius)
    return _CACHE.kinds[actor_id]


def _offset_of(actor: "carla.Actor") -> tuple[float, float]:
    """*actor*'s bounding box offset from its origin, forward and to the
    right (m): ``(0, 0)`` where it has none to read."""
    # Not getattr() with a default, which Codon lacks (docs/typecheck.md).
    try:
        location = actor.bounding_box.location
        return float(location.x), float(location.y)
    except AttributeError:
        return 0.0, 0.0


def _turned(
    x: float, y: float, yaw_deg: float, offset: tuple[float, float]
) -> tuple[float, float]:
    """``(x, y)`` moved by *offset* -- forward and to the right -- of a body
    facing *yaw_deg* (CARLA's frame, left-handed: +y is right of +x)."""
    forward, right = offset
    if forward == 0.0 and right == 0.0:
        return x, y
    heading = math.radians(yaw_deg)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return (
        x + forward * cos_h - right * sin_h,
        y + forward * sin_h + right * cos_h,
    )


def _centre(actor_id: int, x: float, y: float) -> tuple[float, float]:
    """Where *actor_id*'s bounding box is centred this frame, its origin at
    ``(x, y)`` (its offset read by :func:`_kind`)."""
    return _turned(
        x,
        y,
        _CACHE.yaws.get(actor_id, 0.0),
        _CACHE.offsets.get(actor_id, (0.0, 0.0)),
    )


def box_centre(
    world: "carla.World", actor_id: int, x: float, y: float, yaw_deg: float
) -> tuple[float, float]:
    """Where road user *actor_id*'s bounding box would be centred with its
    origin put at CARLA ``(x, y)`` facing *yaw_deg*: the rectangle
    :func:`overlaps` and :func:`claim_spot` take for one about to be put
    there.  ``(x, y)`` itself for an actor not in the world, or one that is
    neither a vehicle nor a walker, and for no actor (a negative id): a prop
    not spawned yet, whose blueprint carries no bounding box to read (CARLA
    knows an actor's box only once it stands in the world)."""
    if actor_id < 0:
        return x, y
    _states(world)
    _kind(world, actor_id)
    return _turned(x, y, yaw_deg, _CACHE.offsets.get(actor_id, (0.0, 0.0)))


def actor_box_centre(
    actor: "carla.Actor", transform: "carla.Transform"
) -> tuple[float, float]:
    """Where *actor*'s bounding box is centred, *transform* being where the
    actor is: what a vehicle looks ahead from."""
    return _turned(
        float(transform.location.x),
        float(transform.location.y),
        float(transform.rotation.yaw),
        _offset_of(actor),
    )


def claim_spot(
    world: "carla.World",
    actor_id: int,
    x: float,
    y: float,
    yaw_deg: float,
    half_length: float,
    half_width: float,
) -> None:
    """Note that *actor_id* is being put, its bounding box centred at CARLA
    ``(x, y)`` (:func:`box_centre`), this frame: :func:`overlaps` counts it
    there before the world's next snapshot shows it (two entities due at one
    spot on the same tick do not both appear)."""
    frame = int(world.get_snapshot().frame)
    if frame != _CACHE.claims_frame:
        _CACHE.claims = []
        _CACHE.claims_frame = frame
    _CACHE.claims.append((actor_id, x, y, yaw_deg, half_length, half_width))


def overlaps(
    world: "carla.World",
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
    *half_width* centred at CARLA ``(x, y)`` facing *yaw_deg*, or ``None``:
    a road user standing where one of that size would be put (its box
    centred there: :func:`box_centre`).  One in the lane beside it does not
    count, as a circle round each would."""
    reach = math.hypot(half_length, half_width) + margin
    mine = (x, y, math.radians(yaw_deg), half_length + margin, half_width + margin)
    states = _states(world)
    if _CACHE.claims_frame == _CACHE.frame:
        for claimant, cx, cy, cyaw, cl, cw in _CACHE.claims:
            if claimant != self_id and _rectangles_meet(
                mine, (cx, cy, math.radians(cyaw), cl, cw)
            ):
                return claimant
    for actor_id, ox, oy, oz, _speed in states:
        if actor_id == self_id or oz < _HIDDEN_BELOW_M:
            continue
        kind, size = _kind(world, actor_id)
        if not kind:
            continue
        ox, oy = _centre(actor_id, ox, oy)
        if math.hypot(ox - x, oy - y) > reach + size:
            continue
        other_length, other_width = _CACHE.boxes.get(actor_id, (size, size))
        if _rectangles_meet(
            mine,
            (
                ox,
                oy,
                math.radians(_CACHE.yaws.get(actor_id, 0.0)),
                other_length,
                other_width,
            ),
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


def first_box_in_way(
    world: "carla.World",
    self_id: int,
    x: float,
    y: float,
    heading_deg: float,
    kind: str,
    distance: float,
    half_width: float,
) -> Optional[Obstacle]:
    """The nearest *kind* (``"vehicle"`` or ``"walker"``) actor whose outline
    (its bounding box, turned as it stands) meets the rectangle *distance*
    metres ahead of ``(x, y)`` facing *heading_deg* (CARLA), *half_width*
    across, or ``None``.  Unlike :func:`first_in_way`, which takes each actor
    as a circle round its box, a vehicle in the lane beside the rectangle is
    not in it (vehicle_56 waited for the ego passing in the next lane)."""
    heading = math.radians(heading_deg)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    half = distance / 2.0
    mine = (x + cos_h * half, y + sin_h * half, heading, half, half_width)
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
        ox, oy = _centre(actor_id, ox, oy)
        dx, dy = ox - x, oy - y
        length, width = _CACHE.boxes.get(actor_id, (radius, radius))
        yaw = math.radians(_CACHE.yaws.get(actor_id, 0.0))
        if not _rectangles_meet(mine, (ox, oy, yaw, length, width)):
            continue
        ahead = dx * cos_h + dy * sin_h
        if best is None or ahead < best.ahead:
            found = Obstacle(actor_id, ox, oy, radius, speed)
            found.ahead = ahead
            found.side = -dx * sin_h + dy * cos_h
            best = found
    return best


def first_in_way(
    world: "carla.World",
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
        ox, oy = _centre(actor_id, ox, oy)
        dx, dy = ox - x, oy - y
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


#: How far (m) a point a vehicle's corner would stand on a sidewalk at is
#: moved over, at most -- a lane -- and in what steps.
KERB_SHIFT_MAX_M = 3.5
KERB_SHIFT_STEP_M = 0.25


def off_the_kerb(
    carla_map: "carla.Map",
    points: list[tuple[float, float, float]],
    half_length: float,
    half_width: float,
    x0: float,
    y0: float,
) -> list[tuple[float, float, float]]:
    """*points* -- a vehicle's path from ``(x0, y0)``, its middle's -- each
    moved over (up to :data:`KERB_SHIFT_MAX_M`) where a corner of the
    vehicle, *half_length* by *half_width*, would stand on a sidewalk -- its
    goal too -- facing along the path there or still the way it came in: a
    long vehicle turns later than its path does, its front swinging in.  So
    it swings wide round a corner a recording made with no kerbs (SUMO's)
    has an 11 m bus cut, over a kerb it cannot climb (vehicle_41).  Moved
    the way the point before was, where either way would do."""
    out: list[tuple[float, float, float]] = []
    previous = (x0, y0)
    sign_before = 1.0
    for index, (x, y, z) in enumerate(points):
        if index + 1 < len(points):
            nx, ny = points[index + 1][0], points[index + 1][1]
        else:
            # The goal: facing the way it came in.
            nx, ny = 2.0 * x - previous[0], 2.0 * y - previous[1]
        headings = [
            _unit(nx - previous[0], ny - previous[1]),
            _unit(x - previous[0], y - previous[1]),
        ]
        along = [h for h in headings if h is not None]
        if not along:
            out.append((x, y, z))
            continue
        hx, hy = along[0]
        # Left of the way it goes, in CARLA's frame (y down): (hy, -hx).
        lx, ly = hy, -hx

        def clear(cx: float, cy: float) -> bool:
            return not any(
                _on_kerb(carla_map, cx, cy, z, ux, uy, half_length, half_width)
                for ux, uy in along
            )

        best = (x, y)
        if not clear(x, y):
            shift = KERB_SHIFT_STEP_M
            while shift <= KERB_SHIFT_MAX_M + 1e-9:
                found = False
                for sign in (sign_before, -sign_before):
                    cx, cy = x + lx * shift * sign, y + ly * shift * sign
                    if clear(cx, cy):
                        best, found, sign_before = (cx, cy), True, sign
                        break
                if found:
                    break
                shift += KERB_SHIFT_STEP_M
        out.append((best[0], best[1], z))
        previous = (x, y)
    return out


def _unit(dx: float, dy: float) -> Optional[tuple[float, float]]:
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return dx / norm, dy / norm


def _on_kerb(
    carla_map: "carla.Map",
    x: float,
    y: float,
    z: float,
    hx: float,
    hy: float,
    half_length: float,
    half_width: float,
) -> bool:
    """Whether a corner of the box at ``(x, y)`` facing ``(hx, hy)`` stands
    on a sidewalk."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    lx, ly = hy, -hx
    for along in (half_length, -half_length):
        for side in (half_width, -half_width):
            cx = x + hx * along + lx * side
            cy = y + hy * along + ly * side
            try:
                waypoint = carla_map.get_waypoint(
                    carla.Location(cx, cy, z),
                    project_to_road=False,
                    lane_type=carla.LaneType.Any,
                )
            except RuntimeError:
                continue
            if waypoint is not None and waypoint.lane_type == carla.LaneType.Sidewalk:
                return True
    return False
