"""Metrics: how the ego drove, read off the world per frame.

A **metric** reads a quantity off the world while a run drives, under a key the
framework fixes, as a measure (:mod:`autoware_carla_scenario.measures`) does.
The two read different things, and are kept apart for it:

* a **measure** reads what a scenario *sets up* -- the gap to a vehicle ahead,
  a pedestrian's speed.  It is a condition the ego drives in, so an ODD can map
  its taxonomy onto it (``measure:`` in a binding file), and a sweep can draw a
  scenario's parameters from it;
* a **metric** reads what the ego *does* with it -- whether it collides, leaves
  the road, drives against the traffic, how close it comes to other road users
  and to the edge of the lanes.  It is an outcome, so it has no place in an ODD:
  the ego's own driving would decide whether a run is inside the domain it is
  tested in, and an ODD that excluded, say, ``offroad`` would report a run that
  left the road as outside the ODD rather than as failed.  An ODD refuses to map
  onto a metric (:func:`~autoware_carla_scenario.odd.scenario_measure`).
  A run watches one instead, with
  :class:`~autoware_carla_scenario.conditions.MetricCondition`.

The built-in metrics (:data:`BUILT_IN_METRICS`) are the per-timestep metrics of
NVIDIA's AlpaSim (``src/eval/src/eval/scorers``,
https://github.com/NVlabs/alpasim), kept under AlpaSim's own names so results
read the same in both, and computed here from CARLA's actors and map instead of
AlpaSim's trajdata map and logged boxes.  A metric reads one frame; AlpaSim
folds the frames of a run into one value with :attr:`Metric.aggregation`
(``max`` for an event: it happened at some point).

An event (a collision, being off the road) reads ``1.0`` when it holds and
``0.0`` when it does not, as AlpaSim's booleans do.  Every metric reads
``None`` when there is no ego, and the map-based ones when the world has no
map.

Four are not AlpaSim's and read the run so far rather than a frame: how the
ego weaves from side to side (:data:`LATERAL_REVERSALS`,
:data:`LATERAL_WOBBLE_M`, :data:`STEERING_REVERSALS`) and how often its turn
indicators changed (:data:`TURN_INDICATOR_CHANGES`), from the ego's motion the
runner hands this module every tick (:func:`track_ego_motion`).

AlpaSim's metrics that need more than the world -- the logged ground-truth
trajectory (``progress``, ``dist_to_gt_trajectory``, ``dist_to_gt_location``,
``min_ade``), the driver's plan (``plan_deviation``, ``open_loop_collision``),
its safety monitor (``safety_monitor_triggered``) or a camera image
(``img_is_black``) -- are not metrics here; ``docs/metrics.md`` says why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from .constants import EGO_ROLE_NAME
from .entity_role import find_actor_in_world

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "BUILT_IN_METRICS",
    "COLLISION_ANY",
    "COLLISION_AT_FAULT",
    "COLLISION_FRONT",
    "COLLISION_LATERAL",
    "COLLISION_REAR",
    "IMPORTANCE_SAMPLING_WEIGHT",
    "LATERAL_REVERSALS",
    "LATERAL_WOBBLE_M",
    "MIN_DISTANCE_TO_LANE_BOUNDARY_M",
    "MIN_DISTANCE_TO_OBSTACLE_M",
    "Metric",
    "OFFROAD",
    "OFFROAD_OR_COLLISION",
    "OFFROAD_OR_COLLISION_AT_FAULT",
    "STEERING_REVERSALS",
    "TURN_INDICATOR_CHANGES",
    "WRONG_LANE",
    "is_metric",
    "read_metric",
    "reset_ego_motion",
    "set_importance_sampling_weight",
    "track_ego_motion",
]

#: The ego's box overlaps another road user's, whichever side it is on.
COLLISION_ANY = "collision_any"
#: ... and the other overlaps the ego's front edge.
COLLISION_FRONT = "collision_front"
#: ... the ego's rear edge, and not its front.
COLLISION_REAR = "collision_rear"
#: ... neither edge: a side swipe.
COLLISION_LATERAL = "collision_lateral"
#: A front or a lateral collision: one the ego is taken to have caused.
COLLISION_AT_FAULT = "collision_at_fault"
#: A corner of the ego's box is on no driving lane.
OFFROAD = "offroad"
#: Off the road, or in a collision.
OFFROAD_OR_COLLISION = "offroad_or_collision"
#: Off the road, or in a collision at fault.
OFFROAD_OR_COLLISION_AT_FAULT = "offroad_or_collision_at_fault"
#: The ego heads against the driving lane it is on.
WRONG_LANE = "wrong_lane"
#: Metres between the ego's box and the nearest other road user's.
MIN_DISTANCE_TO_OBSTACLE_M = "min_distance_to_obstacle_m"
#: Metres between the ego's box and the edge of the driving lanes around it.
MIN_DISTANCE_TO_LANE_BOUNDARY_M = "min_distance_to_lane_boundary_m"
#: The run's importance-sampling weight so far (NADE's), not AlpaSim's: the
#: likelihood ratio a batch of runs is weighted by.
IMPORTANCE_SAMPLING_WEIGHT = "importance_sampling_weight"
#: How many times so far the ego's drift across its lane turned back, by more
#: than :data:`_LATERAL_REVERSAL_GAP_M`: its weaving, not AlpaSim's.
LATERAL_REVERSALS = "lateral_reversals"
#: Metres: how far the ego strays across its lane about its course over the
#: last :data:`_WOBBLE_WINDOW_S` seconds (RMS), not AlpaSim's.
LATERAL_WOBBLE_M = "lateral_wobble_m"
#: How many times so far the steering applied to the ego turned back, by more
#: than :data:`_STEERING_REVERSAL_GAP`, not AlpaSim's.
STEERING_REVERSALS = "steering_reversals"
#: How many times so far the ego's turn indicators changed (off, left, right,
#: hazard), not AlpaSim's.
TURN_INDICATOR_CHANGES = "turn_indicator_changes"

#: How far the ego must turn from its lane's heading to be against it, in
#: degrees: AlpaSim's 2/3 pi.
_WRONG_LANE_HEADING_DEG = 120.0
#: How far from the ego's centre a lane beside its own still counts towards
#: the drivable area, in metres: AlpaSim's lane search radius.
_LANE_SEARCH_RADIUS_M = 6.0
#: How many lanes beside its own the drivable area is walked to, at most.
_MAX_SIDE_LANES = 8
#: How far the ego's drift across its lane must turn back to count as a
#: reversal, in metres: less is the noise of holding a lane.
_LATERAL_REVERSAL_GAP_M = 0.1
#: How far the steering must turn back to count as a reversal, in CARLA's
#: normalised steer (``1`` the vehicle's full lock): about 0.7 degrees at the
#: wheels of a car that locks at 70.
_STEERING_REVERSAL_GAP = 0.01
#: The seconds :data:`LATERAL_WOBBLE_M` looks back over: a weave of 0.25 Hz
#: or faster turns back within it, where a lane change is mostly the course.
_WOBBLE_WINDOW_S = 4.0

#: What the ego's box meets in a frame, as bits: a road user at its front
#: edge, at its rear edge, at its side only, and the edge of the road.
_FRONT = 1
_REAR = 2
_LATERAL = 4
_OFFROAD = 8
_ANY_COLLISION = _FRONT | _REAR | _LATERAL
_AT_FAULT = _FRONT | _LATERAL

#: A point on the ground, ``(x, y)`` in CARLA's world frame.
Point = tuple[float, float]
#: An oriented box on the ground: its four corners, front left, front right,
#: rear right, rear left.
Box = list[Point]


@dataclass(frozen=True)
class Metric:
    """How the ego drove, read off the world.

    Attributes:
        key: Its key, AlpaSim's metric name, e.g. :data:`COLLISION_ANY`.
        read: Reads it from the world; ``None`` when there is nothing to read.
        title: A short name, as the scenario editor lists it.
        unit: The unit it is read in; ``""`` for an event (``1`` or ``0``).
        text: A description.
        aggregation: How AlpaSim folds a run's frames into one value:
            ``"max"`` (an event that happened at some point), ``"min"`` (the
            closest the ego came), ``"mean"`` or ``"last"``.
    """

    key: str
    read: Callable[[carla.World], Optional[float]]
    title: str = ""
    unit: str = ""
    text: str = ""
    aggregation: str = "max"


# ---------------------------------------------------------------------------
# Boxes on the ground
# ---------------------------------------------------------------------------


def _box_of(actor: "carla.Actor", transform: "carla.Transform") -> Box:
    """*actor*'s bounding box on the ground, in the world frame, *transform*
    being where the actor is."""
    heading = math.radians(transform.rotation.yaw)
    yaw = heading
    extent_x, extent_y = 0.0, 0.0
    offset_x, offset_y = 0.0, 0.0
    # Not getattr() with a default, which Codon lacks (docs/typecheck.md).
    try:
        box = actor.bounding_box
        extent_x, extent_y = float(box.extent.x), float(box.extent.y)
        offset_x, offset_y = float(box.location.x), float(box.location.y)
        yaw += math.radians(box.rotation.yaw)
    except AttributeError:
        pass
    # CARLA's frame is left-handed: +y is to the right of +x.
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-forward[1], forward[0])
    centre = (
        transform.location.x
        + offset_x * math.cos(heading)
        - offset_y * math.sin(heading),
        transform.location.y
        + offset_x * math.sin(heading)
        + offset_y * math.cos(heading),
    )

    def corner(along: float, across: float) -> Point:
        return (
            centre[0] + along * forward[0] + across * right[0],
            centre[1] + along * forward[1] + across * right[1],
        )

    return [
        corner(extent_x, -extent_y),
        corner(extent_x, extent_y),
        corner(-extent_x, extent_y),
        corner(-extent_x, -extent_y),
    ]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _inside(point: Point, box: Box) -> bool:
    """Whether *point* is in the convex *box* or on its edge."""
    sign = 0.0
    for i in range(len(box)):
        side = _cross(box[i], box[(i + 1) % len(box)], point)
        if abs(side) <= 1e-12:
            continue
        if sign == 0.0:
            sign = side
        elif (side > 0.0) != (sign > 0.0):
            return False
    return True


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0
    if length2 > 0.0:
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Whether segment *ab* meets segment *cd*, touching included."""
    if (
        _cross(c, d, a) * _cross(c, d, b) < 0.0
        and _cross(a, b, c) * _cross(a, b, d) < 0.0
    ):
        return True
    return (
        _point_segment_distance(a, c, d) <= 1e-9
        or _point_segment_distance(b, c, d) <= 1e-9
        or _point_segment_distance(c, a, b) <= 1e-9
        or _point_segment_distance(d, a, b) <= 1e-9
    )


def _edges(box: Box) -> list[tuple[Point, Point]]:
    return [(box[i], box[(i + 1) % len(box)]) for i in range(len(box))]


def _segment_meets_box(a: Point, b: Point, box: Box) -> bool:
    if _inside(a, box) or _inside(b, box):
        return True
    for c, d in _edges(box):
        if _segments_cross(a, b, c, d):
            return True
    return False


def _boxes_meet(one: Box, other: Box) -> bool:
    for a, b in _edges(one):
        if _segment_meets_box(a, b, other):
            return True
    # One wholly inside the other.
    return _inside(other[0], one)


def _box_distance(one: Box, other: Box) -> float:
    """Metres between two boxes on the ground; ``0.0`` when they meet."""
    if _boxes_meet(one, other):
        return 0.0
    one_edges, other_edges = _edges(one), _edges(other)
    best = math.inf
    for p in one:
        for a, b in other_edges:
            best = min(best, _point_segment_distance(p, a, b))
    for p in other:
        for a, b in one_edges:
            best = min(best, _point_segment_distance(p, a, b))
    return best


def _centre(box: Box) -> Point:
    return (
        sum(p[0] for p in box) / len(box),
        sum(p[1] for p in box) / len(box),
    )


def _radius(box: Box, centre: Point) -> float:
    return math.hypot(box[0][0] - centre[0], box[0][1] - centre[1])


# ---------------------------------------------------------------------------
# The frame: the ego and the road users around it, read once
# ---------------------------------------------------------------------------


class _Frame:
    """What one frame of one world reads, each part once: the ego's box and
    the others', what its box meets, and the lane under its centre.

    Every box also has the circle around it (its centre and the distance to
    its corners): two boxes whose circles are apart are apart, which spares
    the exact tests for the road users nowhere near the ego.
    """

    has_ego: bool
    ego_yaw: float
    ego_box: Box
    ego_centre: Point
    ego_radius: float
    others: list[Box]
    others_centre: list[Point]
    others_radius: list[float]
    #: What the ego's box meets: _FRONT, _REAR and _LATERAL once the road
    #: users have been read (collisions_read), _OFFROAD once the road has
    #: (offroad_read).
    hits: int
    collisions_read: bool
    offroad_read: bool
    centre_read: bool
    centre_waypoint: Optional["carla.Waypoint"]

    def __init__(self, ego_yaw: float, ego_box: Box, others: list[Box]) -> None:
        self.has_ego = bool(ego_box)
        self.ego_yaw = ego_yaw
        self.ego_box = ego_box
        self.ego_centre = _centre(ego_box) if ego_box else (0.0, 0.0)
        self.ego_radius = _radius(ego_box, self.ego_centre) if ego_box else 0.0
        self.others = others
        self.others_centre = [_centre(box) for box in others]
        self.others_radius = [
            _radius(box, centre) for box, centre in zip(others, self.others_centre)
        ]
        self.hits = 0
        self.collisions_read = False
        self.offroad_read = False
        self.centre_read = False
        self.centre_waypoint = None

    def clearance(self, i: int) -> float:
        """How far apart the ego's circle and the *i*-th road user's are; at
        most the distance between their boxes, and ``0.0`` when they overlap."""
        centre = self.others_centre[i]
        apart = math.hypot(
            centre[0] - self.ego_centre[0], centre[1] - self.ego_centre[1]
        )
        return max(apart - self.ego_radius - self.others_radius[i], 0.0)


_NO_EGO = _Frame(0.0, [], [])


class _Cache:
    """The world read last, its last frame's number and what was read then,
    and its map.

    The world is held, not its ``id()``: a world freed and another made may
    get the same ``id()``, and would be read off the old one's frame and map.
    """

    world: Optional["carla.World"]
    at: Optional[int]
    frame: Optional[_Frame]
    carla_map: Optional["carla.Map"]
    driving: int

    def __init__(self) -> None:
        self.world = None
        self.at = None
        self.frame = None
        self.carla_map = None
        self.driving = 0


_CACHE = _Cache()


def _frame(world: "carla.World") -> _Frame:
    held = _CACHE.world
    if held is None or held is not world:
        _CACHE.world = world
        _CACHE.at, _CACHE.frame, _CACHE.carla_map = None, None, None
    at: Optional[int] = None
    try:
        at = int(world.get_snapshot().frame)
    except Exception:
        pass
    cached = _CACHE.frame
    if at is not None and at == _CACHE.at and cached is not None:
        return cached
    found = _read_frame(world)
    _CACHE.at, _CACHE.frame = at, found
    return found


def _read_frame(world: "carla.World") -> _Frame:
    try:
        actors = list(world.get_actors())
        # By id once found: a scan by role name reads every actor's
        # attributes, which CARLA builds anew for each read.
        ego = find_actor_in_world(world, EGO_ROLE_NAME)
    except Exception:
        ego = None
    if ego is None:
        return _NO_EGO
    others: list[Box] = []
    for actor in actors:
        kind = actor.type_id
        if actor.id == ego.id or not (
            kind.startswith("vehicle.") or kind.startswith("walker.")
        ):
            continue
        others.append(_box_of(actor, actor.get_transform()))
    transform = ego.get_transform()
    return _Frame(transform.rotation.yaw, _box_of(ego, transform), others)


def _map(world: "carla.World") -> "carla.Map":
    """The map of the world :func:`_frame` holds, fetched once: CARLA builds
    it anew on every call."""
    cached = _CACHE.carla_map
    if cached is not None:
        return cached
    import typesafe_carla.carla as carla  # noqa: PLC0415

    from .coordinate.map_manager import run_carla_map  # noqa: PLC0415

    found = run_carla_map(world)
    _CACHE.carla_map = found
    # typesafe_carla's lane type is an int (an IntEnum on the API side).
    _CACHE.driving = int(carla.LaneType.Driving)
    return found


def _driving_waypoint(world: "carla.World", point: Point) -> "Optional[carla.Waypoint]":
    """The driving lane under *point*, ``None`` when it is on none."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    carla_map = _map(world)
    return carla_map.get_waypoint(
        carla.Location(x=point[0], y=point[1], z=0.0),
        project_to_road=False,
        lane_type=_CACHE.driving,
    )


def _collision_bits(frame: _Frame) -> int:
    """AlpaSim's ``CollisionScorer``: each road user the ego's box meets meets
    its front edge, else its rear edge, else its side."""
    if not frame.collisions_read:
        box = frame.ego_box
        for i, other in enumerate(frame.others):
            if frame.clearance(i) > 0.0:
                continue
            if _segment_meets_box(box[0], box[1], other):
                frame.hits |= _FRONT
            elif _segment_meets_box(box[2], box[3], other):
                frame.hits |= _REAR
            elif _boxes_meet(box, other):
                frame.hits |= _LATERAL
        frame.collisions_read = True
    return frame.hits


def _offroad_bits(world: "carla.World", frame: _Frame) -> int:
    """AlpaSim's ``offroad``: the ego's box is not on the driving lanes.

    AlpaSim asks whether the union of the lanes around the ego covers its box;
    CARLA answers per point, so this asks it of the box's corners.  Raises
    without a map.
    """
    if not frame.offroad_read:
        for corner in frame.ego_box:
            if _driving_waypoint(world, corner) is None:
                frame.hits |= _OFFROAD
                break
        frame.offroad_read = True
    return frame.hits


def _centre_waypoint(world: "carla.World", frame: _Frame) -> "Optional[carla.Waypoint]":
    """The driving lane under the ego's centre."""
    if not frame.centre_read:
        frame.centre_waypoint = _driving_waypoint(world, frame.ego_centre)
        frame.centre_read = True
    return frame.centre_waypoint


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------


def _event(world: "carla.World", mask: int) -> Optional[float]:
    """``1.0`` when the ego's box meets any of *mask*, else ``0.0``."""
    frame = _frame(world)
    if not frame.has_ego:
        return None
    if mask & _ANY_COLLISION:
        _collision_bits(frame)
    if mask & _OFFROAD:
        _offroad_bits(world, frame)
    return 1.0 if frame.hits & mask else 0.0


def _collision_any(world: "carla.World") -> Optional[float]:
    return _event(world, _ANY_COLLISION)


def _collision_front(world: "carla.World") -> Optional[float]:
    return _event(world, _FRONT)


def _collision_rear(world: "carla.World") -> Optional[float]:
    return _event(world, _REAR)


def _collision_lateral(world: "carla.World") -> Optional[float]:
    return _event(world, _LATERAL)


def _collision_at_fault(world: "carla.World") -> Optional[float]:
    return _event(world, _AT_FAULT)


#: The running scenario's importance-sampling weight, which the runner
#: publishes every tick (:func:`set_importance_sampling_weight`); None
#: outside a run.  Published rather than read off the scenario, which this
#: module, imported by everything that reads a metric, cannot import.
_RUN: dict[str, Optional[float]] = {"weight": None}


def set_importance_sampling_weight(weight: Optional[float]) -> None:
    """Publish the running scenario's weight
    (:attr:`BaseScenario.importance_sampling_weight`), or None when the run
    ends."""
    _RUN["weight"] = weight


def _importance_sampling_weight(world: "carla.World") -> Optional[float]:
    """The running scenario's weight, or None outside a run."""
    return _RUN["weight"]


class _Reversals:
    """Counts how often a signal turns back by at least *gap*: the steering
    reversal count of driver-workload studies (Markkula and Engstrom, 2006),
    which a drift one way, however long, does not add to."""

    gap: float
    count: int
    #: +1 while the signal rises, -1 while it falls, 0 until it has moved
    #: *gap* either way.
    heading: int
    #: The furthest it has gone the way it heads; until it heads anywhere,
    #: the lowest and the highest it has been.
    extreme: float
    low: float
    high: float
    started: bool

    def __init__(self, gap: float) -> None:
        self.gap = gap
        self.count = 0
        self.heading = 0
        self.extreme = 0.0
        self.low = 0.0
        self.high = 0.0
        self.started = False

    def add(self, value: float) -> None:
        if not self.started:
            self.low = self.high = value
            self.started = True
        if self.heading == 0:
            self.low = min(self.low, value)
            self.high = max(self.high, value)
            if value - self.low >= self.gap:
                self.heading, self.extreme = 1, value
            elif self.high - value >= self.gap:
                self.heading, self.extreme = -1, value
        elif (value - self.extreme) * self.heading >= 0.0:
            self.extreme = value
        elif abs(self.extreme - value) >= self.gap:
            self.count += 1
            self.heading, self.extreme = -self.heading, value


class _Motion:
    """The ego's motion through the run so far (:func:`track_ego_motion`).

    ``lateral`` is how far the ego has moved across its lanes since the run
    began, in metres to the right: its velocity across the lane under it,
    summed over time.  The lane's heading is taken anew every frame, so a bend
    is not a drift; moving from lane to lane is, without the jump of a lane
    centre measured from.
    """

    ego_id: int
    at: Optional[float]
    lateral: float
    #: ``(elapsed, lateral)`` of the frames of the last
    #: :data:`_WOBBLE_WINDOW_S` seconds, oldest first.
    window: list[tuple[float, float]]
    lateral_reversals: _Reversals
    steering_reversals: _Reversals
    #: The indicators the ego showed last (``TurnIndicatorState.value``), -1
    #: before the first frame, and how many times they changed since.
    indicator: int
    indicator_changes: int

    def __init__(self) -> None:
        self.ego_id = -1
        self.at = None
        self.lateral = 0.0
        self.window = []
        self.lateral_reversals = _Reversals(_LATERAL_REVERSAL_GAP_M)
        self.steering_reversals = _Reversals(_STEERING_REVERSAL_GAP)
        self.indicator = -1
        self.indicator_changes = 0

    def wobble(self) -> Optional[float]:
        """The RMS of :attr:`lateral` over the window about the straight line
        that fits it best; None before it holds three frames."""
        n = len(self.window)
        if n < 3:
            return None
        mean_t = sum(t for t, _ in self.window) / n
        mean_y = sum(y for _, y in self.window) / n
        stt = sum((t - mean_t) ** 2 for t, _ in self.window)
        sty = sum((t - mean_t) * (y - mean_y) for t, y in self.window)
        slope = sty / stt if stt > 0.0 else 0.0
        residual = sum((y - mean_y - slope * (t - mean_t)) ** 2 for t, y in self.window)
        return math.sqrt(residual / n)


_MOTION: list[_Motion] = [_Motion()]


def reset_ego_motion() -> None:
    """Forget the ego's motion: the runner calls it as a run begins and
    ends."""
    _MOTION[0] = _Motion()


def track_ego_motion(world: "carla.World", elapsed: float) -> None:
    """Add the frame of *world* at *elapsed* simulated seconds to the ego's
    motion, which :data:`LATERAL_REVERSALS`, :data:`LATERAL_WOBBLE_M`,
    :data:`STEERING_REVERSALS` and :data:`TURN_INDICATOR_CHANGES` read.

    The runner calls it every tick; a frame at the time of the last one adds
    nothing.  Off the driving lanes the ego has no lane to drift across, and
    its drift holds until it is back on one.  Another ego starts the motion
    anew.
    """
    try:
        ego = find_actor_in_world(world, EGO_ROLE_NAME)
    except Exception:
        ego = None
    if ego is None:
        return
    motion = _MOTION[0]
    if motion.ego_id != ego.id:
        motion = _Motion()
        motion.ego_id = ego.id
        _MOTION[0] = motion
    last = motion.at
    if last is not None and elapsed <= last:
        return
    motion.at = elapsed
    try:
        motion.steering_reversals.add(float(ego.get_control().steer))
    except Exception:
        pass
    _track_indicator(motion, ego)
    if last is not None:
        frame = _frame(world)
        try:
            waypoint = _centre_waypoint(world, frame)
        except Exception:
            waypoint = None
        if waypoint is not None:
            velocity = ego.get_velocity()
            yaw = math.radians(waypoint.transform.rotation.yaw)
            across = -velocity.x * math.sin(yaw) + velocity.y * math.cos(yaw)
            motion.lateral += across * (elapsed - last)
    motion.lateral_reversals.add(motion.lateral)
    start = elapsed - _WOBBLE_WINDOW_S
    motion.window = [p for p in motion.window if p[0] >= start]
    motion.window.append((elapsed, motion.lateral))


def _track_indicator(motion: _Motion, ego: "carla.Actor") -> None:
    """Count a change of the indicators *ego* shows since the last frame."""
    # Imported here: it reads CARLA's light bits, and this module does not
    # import CARLA at import time.
    from .utils.turn_indicator import is_vehicle, turn_indicators_of  # noqa: PLC0415

    try:
        if not is_vehicle(ego):
            return
        shown = int(turn_indicators_of(ego).value)
    except Exception:
        return
    if motion.indicator >= 0 and shown != motion.indicator:
        motion.indicator_changes += 1
    motion.indicator = shown


def _tracked(world: "carla.World") -> Optional[_Motion]:
    """The ego's motion, None without an ego or before its first frame."""
    motion = _MOTION[0]
    if not _frame(world).has_ego or motion.at is None:
        return None
    return motion


def _lateral_reversals(world: "carla.World") -> Optional[float]:
    motion = _tracked(world)
    return None if motion is None else float(motion.lateral_reversals.count)


def _lateral_wobble_m(world: "carla.World") -> Optional[float]:
    motion = _tracked(world)
    return None if motion is None else motion.wobble()


def _steering_reversals(world: "carla.World") -> Optional[float]:
    motion = _tracked(world)
    return None if motion is None else float(motion.steering_reversals.count)


def _turn_indicator_changes(world: "carla.World") -> Optional[float]:
    motion = _tracked(world)
    if motion is None or motion.indicator < 0:
        return None
    return float(motion.indicator_changes)


def _offroad(world: "carla.World") -> Optional[float]:
    return _event(world, _OFFROAD)


def _offroad_or_collision(world: "carla.World") -> Optional[float]:
    return _event(world, _OFFROAD | _ANY_COLLISION)


def _offroad_or_collision_at_fault(world: "carla.World") -> Optional[float]:
    return _event(world, _OFFROAD | _AT_FAULT)


def _min_distance_to_obstacle_m(world: "carla.World") -> Optional[float]:
    """AlpaSim's ``MinDistanceToObstacleScorer``: box to box, ``None`` when
    the ego is alone."""
    frame = _frame(world)
    if not frame.has_ego or not frame.others:
        return None
    # Nearest circle first: once a circle is further than the best box yet,
    # so are the boxes of it and of every road user after it.
    order = sorted(range(len(frame.others)), key=frame.clearance)
    best = math.inf
    for i in order:
        if frame.clearance(i) >= best:
            break
        best = min(best, _box_distance(frame.ego_box, frame.others[i]))
    return best


def _wrong_lane(world: "carla.World") -> Optional[float]:
    """AlpaSim's ``wrong_lane``: the ego heads more than 120 degrees away from
    the driving lane under its centre.

    ``None`` when its centre is on no driving lane: off the road, there is no
    lane to be the wrong way along.
    """
    # Imported here: the kinematics package imports CARLA, and this module
    # does not at import time (the scenario editor reads it without CARLA).
    from .kinematics.angle import normalize_angle_deg  # noqa: PLC0415

    frame = _frame(world)
    if not frame.has_ego:
        return None
    waypoint = _centre_waypoint(world, frame)
    if waypoint is None:
        return None
    gap = abs(normalize_angle_deg(frame.ego_yaw - waypoint.transform.rotation.yaw))
    return 1.0 if gap > _WRONG_LANE_HEADING_DEG else 0.0


def _lateral(waypoint: "carla.Waypoint", point: Point) -> float:
    """How far *point* is to the right of *waypoint*'s centre line, in metres."""
    transform = waypoint.transform
    yaw = math.radians(transform.rotation.yaw)
    dx = point[0] - transform.location.x
    dy = point[1] - transform.location.y
    return -dx * math.sin(yaw) + dy * math.cos(yaw)


def _side_width(waypoint: "carla.Waypoint", right: bool, start: float) -> float:
    """How far the driving lanes reach beyond *waypoint*'s, to one side.

    Lanes beside it are walked while they are driving lanes whose near edge is
    within :data:`_LANE_SEARCH_RADIUS_M` of the ego's centre, *start* metres
    from it; their widths are summed.  A lane of the opposite direction counts,
    as it does in AlpaSim, whose lane union takes every lane around the ego.
    """
    reach = 0.0
    lane = waypoint
    for _ in range(_MAX_SIDE_LANES):
        beside = lane.get_right_lane() if right else lane.get_left_lane()
        if beside is None or int(beside.lane_type) != _CACHE.driving:
            break
        if start + reach > _LANE_SEARCH_RADIUS_M:
            break
        reach += float(beside.lane_width)
        # Across the centre line, CARLA's "right" of a lane of the opposite
        # direction is the ego's left: keep walking away from the ego.
        if beside.lane_id * lane.lane_id < 0:
            right = not right
        lane = beside
    return reach


def _min_distance_to_lane_boundary_m(world: "carla.World") -> Optional[float]:
    """AlpaSim's ``min_distance_to_lane_boundary_m``: how far the ego's box is
    from the edge of the driving lanes around it; ``0.0`` off the road.

    The lanes are taken as parallel strips beside the lane under the ego's
    centre, as wide as CARLA says they are there, which holds on a road and
    is an approximation through a junction or where a lane widens.
    """
    frame = _frame(world)
    if not frame.has_ego:
        return None
    if _offroad_bits(world, frame) & _OFFROAD:
        return 0.0
    waypoint = _centre_waypoint(world, frame)
    if waypoint is None:
        return 0.0
    half = float(waypoint.lane_width) / 2.0
    laterals = [_lateral(waypoint, corner) for corner in frame.ego_box]
    centre_lateral = _lateral(waypoint, frame.ego_centre)
    right_edge = half + _side_width(waypoint, True, half - centre_lateral)
    left_edge = half + _side_width(waypoint, False, half + centre_lateral)
    gap = min(right_edge - max(laterals), left_edge + min(laterals))
    return max(gap, 0.0)


#: The built-in metrics, by key: AlpaSim's, as CARLA reads them.
BUILT_IN_METRICS: dict[str, Metric] = {
    m.key: m
    for m in [
        Metric(
            COLLISION_ANY,
            _collision_any,
            title="Collision (any)",
            text="1 while the ego's bounding box overlaps another vehicle's or "
            "a pedestrian's, else 0 (AlpaSim collision_any)",
        ),
        Metric(
            COLLISION_FRONT,
            _collision_front,
            title="Collision (front)",
            text="1 while another road user's box overlaps the ego's front edge "
            "(AlpaSim collision_front)",
        ),
        Metric(
            COLLISION_REAR,
            _collision_rear,
            title="Collision (rear)",
            text="1 while another road user's box overlaps the ego's rear edge "
            "and not its front (AlpaSim collision_rear)",
        ),
        Metric(
            COLLISION_LATERAL,
            _collision_lateral,
            title="Collision (lateral)",
            text="1 while another road user's box overlaps the ego's and "
            "neither its front nor its rear edge (AlpaSim collision_lateral)",
        ),
        Metric(
            COLLISION_AT_FAULT,
            _collision_at_fault,
            title="Collision at fault",
            text="1 during a front or a lateral collision "
            "(AlpaSim collision_at_fault)",
        ),
        Metric(
            OFFROAD,
            _offroad,
            title="Off the road",
            text="1 while a corner of the ego's box is on no driving lane "
            "(AlpaSim offroad)",
        ),
        Metric(
            OFFROAD_OR_COLLISION,
            _offroad_or_collision,
            title="Off the road or collision",
            text="1 while off the road or in any collision "
            "(AlpaSim offroad_or_collision)",
        ),
        Metric(
            OFFROAD_OR_COLLISION_AT_FAULT,
            _offroad_or_collision_at_fault,
            title="Off the road or collision at fault",
            text="1 while off the road or in a collision at fault "
            "(AlpaSim offroad_or_collision_at_fault)",
        ),
        Metric(
            WRONG_LANE,
            _wrong_lane,
            title="Against the lane",
            text="1 while the ego heads more than 120 degrees away from the "
            "driving lane under it, else 0; unread off the road "
            "(AlpaSim wrong_lane)",
        ),
        Metric(
            MIN_DISTANCE_TO_OBSTACLE_M,
            _min_distance_to_obstacle_m,
            title="Distance to obstacle",
            unit="m",
            text="Box to box, to the nearest other vehicle or pedestrian; 0 "
            "when touching (AlpaSim min_distance_to_obstacle_m)",
            aggregation="min",
        ),
        Metric(
            MIN_DISTANCE_TO_LANE_BOUNDARY_M,
            _min_distance_to_lane_boundary_m,
            title="Distance to lane edge",
            unit="m",
            text="From the ego's box to the edge of the driving lanes around "
            "it; 0 off the road (AlpaSim min_distance_to_lane_boundary_m)",
            aggregation="min",
        ),
        Metric(
            IMPORTANCE_SAMPLING_WEIGHT,
            _importance_sampling_weight,
            title="Importance-sampling weight",
            unit="ratio",
            text="The run's likelihood ratio so far, the product of NADE's "
            "p/q and (1-p)/(1-q) (1 without NADE); a batch of runs weighted "
            "by its last value estimates the naturalistic rate",
            aggregation="last",
        ),
        Metric(
            LATERAL_REVERSALS,
            _lateral_reversals,
            title="Weaving (reversals)",
            unit="count",
            text="How many times so far the ego's drift across its lane turned "
            "back by more than 0.1 m; a lane change is a drift, not a reversal",
            aggregation="last",
        ),
        Metric(
            LATERAL_WOBBLE_M,
            _lateral_wobble_m,
            title="Weaving (amplitude)",
            unit="m",
            text="How far the ego strayed across its lane about its course "
            "over the last 4 s, RMS",
            aggregation="max",
        ),
        Metric(
            STEERING_REVERSALS,
            _steering_reversals,
            title="Steering reversals",
            unit="count",
            text="How many times so far the steering applied to the ego "
            "turned back by more than 0.01 of full lock",
            aggregation="last",
        ),
        Metric(
            TURN_INDICATOR_CHANGES,
            _turn_indicator_changes,
            title="Turn indicator changes",
            unit="count",
            text="How many times so far the ego's turn indicators changed: "
            "off, left, right or hazard",
            aggregation="last",
        ),
    ]
}


def is_metric(key: str) -> bool:
    """Whether *key* is a built-in metric's."""
    return key in BUILT_IN_METRICS


def read_metric(key: str, world: "carla.World") -> Optional[float]:
    """The built-in metric *key* in *world*.

    ``None`` when there is nothing to read, the key is not a metric's, or the
    metric raised.
    """
    if key not in BUILT_IN_METRICS:
        return None
    metric = BUILT_IN_METRICS[key]
    try:
        return metric.read(world)
    except Exception:
        return None
