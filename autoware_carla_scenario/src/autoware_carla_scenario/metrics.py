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

from .conditions.base import find_actor_in_list
from .constants import EGO_ROLE_NAME

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "COLLISION_ANY",
    "COLLISION_AT_FAULT",
    "COLLISION_FRONT",
    "COLLISION_LATERAL",
    "COLLISION_REAR",
    "BUILT_IN_METRICS",
    "LANE_SEARCH_RADIUS_M",
    "MIN_DISTANCE_TO_LANE_BOUNDARY_M",
    "MIN_DISTANCE_TO_OBSTACLE_M",
    "OFFROAD",
    "OFFROAD_OR_COLLISION",
    "OFFROAD_OR_COLLISION_AT_FAULT",
    "WRONG_LANE",
    "WRONG_LANE_HEADING_RAD",
    "Metric",
    "is_metric",
    "read_metric",
    "reset_metrics",
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

#: How far the ego must turn from its lane's heading to be against it, in
#: radians: AlpaSim's 2/3 pi (120 degrees).
WRONG_LANE_HEADING_RAD = math.pi * 2.0 / 3.0
#: How far from the ego's centre a lane beside its own still counts towards
#: the drivable area, in metres: AlpaSim's lane search radius.
LANE_SEARCH_RADIUS_M = 6.0


#: A point on the ground, ``(x, y)`` in CARLA's world frame.
Point = tuple[float, float]
#: An oriented box on the ground: its four corners, front left, front right,
#: rear right, rear left.
Box = list[Point]

#: How many lanes beside its own the drivable area is walked to, at most.
_MAX_SIDE_LANES = 8


# ---------------------------------------------------------------------------
# Boxes on the ground
# ---------------------------------------------------------------------------


def _box_of(actor: "carla.Actor") -> Box:
    """*actor*'s bounding box on the ground, in the world frame."""
    transform = actor.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    extent_x, extent_y = 0.0, 0.0
    offset_x, offset_y = 0.0, 0.0
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
    heading = math.radians(transform.rotation.yaw)
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


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Whether segment *ab* meets segment *cd*, touching included."""
    d1, d2 = _cross(c, d, a), _cross(c, d, b)
    d3, d4 = _cross(a, b, c), _cross(a, b, d)
    if ((d1 > 0.0) != (d2 > 0.0) and d1 != 0.0 and d2 != 0.0) and (
        (d3 > 0.0) != (d4 > 0.0) and d3 != 0.0 and d4 != 0.0
    ):
        return True
    return (
        _point_segment_distance(a, c, d) <= 1e-9
        or _point_segment_distance(b, c, d) <= 1e-9
        or _point_segment_distance(c, a, b) <= 1e-9
        or _point_segment_distance(d, a, b) <= 1e-9
    )


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0
    if length2 > 0.0:
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


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
    best = math.inf
    for p in one:
        for a, b in _edges(other):
            best = min(best, _point_segment_distance(p, a, b))
    for p in other:
        for a, b in _edges(one):
            best = min(best, _point_segment_distance(p, a, b))
    return best


# ---------------------------------------------------------------------------
# The frame: the ego and the road users around it, read once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """How the ego drove, read off the world.

    Attributes:
        key: Its key, AlpaSim's metric name, e.g. :data:`COLLISION_ANY`.
        read: Reads it from the world; ``None`` when there is nothing to read.
        unit: The unit it is read in; ``""`` for an event (``1`` or ``0``).
        text: A description.
        aggregation: How AlpaSim folds a run's frames into one value:
            ``"max"`` (an event that happened at some point), ``"min"`` (the
            closest the ego came), ``"mean"`` or ``"last"``.
    """

    key: str
    read: Callable[[carla.World], Optional[float]]
    unit: str = ""
    text: str = ""
    aggregation: str = "max"


class _Collisions:
    """How the ego's box meets the others' in one frame."""

    front: bool
    rear: bool
    lateral: bool

    def __init__(self) -> None:
        self.front = False
        self.rear = False
        self.lateral = False

    @property
    def any(self) -> bool:
        return self.front or self.rear or self.lateral

    @property
    def at_fault(self) -> bool:
        return self.front or self.lateral


class _Frame:
    """What one frame of one world reads: the ego, its box and the others'."""

    ego: Optional["carla.Actor"]
    ego_box: Box
    others: list[Box]

    def __init__(
        self, ego: Optional["carla.Actor"], ego_box: Box, others: list[Box]
    ) -> None:
        self.ego = ego
        self.ego_box = ego_box
        self.others = others


class _Cache:
    """The frame read last, by ``(id(world), frame)``, and each world's map."""

    at: Optional[tuple[int, int]]
    frame: Optional[_Frame]
    maps: dict[int, "carla.Map"]

    def __init__(self) -> None:
        self.at = None
        self.frame = None
        self.maps = {}


_CACHE = _Cache()


def reset_metrics() -> None:
    """Forget what was read: the last frame and the worlds' maps.

    :func:`~autoware_carla_scenario.measures.set_measured_scenario` calls it
    when a run starts and ends,
    so a run never reads the frame or the map of the one before it.
    """
    _CACHE.at = None
    _CACHE.frame = None
    _CACHE.maps = {}


def _frame(world: "carla.World") -> _Frame:
    at: Optional[tuple[int, int]] = None
    try:
        at = (id(world), int(world.get_snapshot().frame))
    except Exception:
        at = None
    if at is not None and _CACHE.at is not None and at == _CACHE.at:
        cached = _CACHE.frame
        if cached is not None:
            return cached
    found = _read_frame(world)
    _CACHE.at, _CACHE.frame = at, found
    return found


def _read_frame(world: "carla.World") -> _Frame:
    try:
        actors = list(world.get_actors())
        ego = find_actor_in_list(actors, EGO_ROLE_NAME)
    except Exception:
        return _Frame(None, [], [])
    if ego is None:
        return _Frame(None, [], [])
    others: list[Box] = []
    for actor in actors:
        kind = actor.type_id
        if actor.id == ego.id or not (
            kind.startswith("vehicle.") or kind.startswith("walker.")
        ):
            continue
        others.append(_box_of(actor))
    return _Frame(ego, _box_of(ego), others)


def _map(world: "carla.World") -> "carla.Map":
    key = id(world)
    if key not in _CACHE.maps:
        _CACHE.maps[key] = world.get_map()
    return _CACHE.maps[key]


def _driving_lane_type() -> int:
    import typesafe_carla.carla as carla  # noqa: PLC0415

    # typesafe_carla's lane type is an int (an IntEnum on the API side).
    return int(carla.LaneType.Driving)


def _driving_waypoint(world: "carla.World", point: Point) -> "Optional[carla.Waypoint]":
    """The driving lane under *point*, ``None`` when it is on none."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    return _map(world).get_waypoint(
        carla.Location(x=point[0], y=point[1], z=0.0),
        project_to_road=False,
        lane_type=_driving_lane_type(),
    )


def _centre(box: Box) -> Point:
    return (
        sum(p[0] for p in box) / len(box),
        sum(p[1] for p in box) / len(box),
    )


# ---------------------------------------------------------------------------
# Collisions and obstacles
# ---------------------------------------------------------------------------


def _collisions(world: "carla.World") -> Optional[_Collisions]:
    """AlpaSim's ``CollisionScorer``: each road user the ego's box meets
    meets its front edge, else its rear edge, else its side."""
    frame = _frame(world)
    if frame.ego is None:
        return None
    found = _Collisions()
    box = frame.ego_box
    front = (box[0], box[1])
    rear = (box[2], box[3])
    for other in frame.others:
        if not _boxes_meet(box, other):
            continue
        if _segment_meets_box(front[0], front[1], other):
            found.front = True
        elif _segment_meets_box(rear[0], rear[1], other):
            found.rear = True
        else:
            found.lateral = True
    return found


def _event(happened: bool) -> float:
    return 1.0 if happened else 0.0


def _collision_any(world: "carla.World") -> Optional[float]:
    found = _collisions(world)
    return None if found is None else _event(found.any)


def _collision_front(world: "carla.World") -> Optional[float]:
    found = _collisions(world)
    return None if found is None else _event(found.front)


def _collision_rear(world: "carla.World") -> Optional[float]:
    found = _collisions(world)
    return None if found is None else _event(found.rear)


def _collision_lateral(world: "carla.World") -> Optional[float]:
    found = _collisions(world)
    return None if found is None else _event(found.lateral)


def _collision_at_fault(world: "carla.World") -> Optional[float]:
    found = _collisions(world)
    return None if found is None else _event(found.at_fault)


def _min_distance_to_obstacle_m(world: "carla.World") -> Optional[float]:
    """AlpaSim's ``MinDistanceToObstacleScorer``: box to box, ``None`` when
    the ego is alone."""
    frame = _frame(world)
    if frame.ego is None or not frame.others:
        return None
    return min(_box_distance(frame.ego_box, other) for other in frame.others)


# ---------------------------------------------------------------------------
# The road
# ---------------------------------------------------------------------------


def _is_offroad(world: "carla.World") -> Optional[bool]:
    """AlpaSim's ``offroad``: the ego's box is not on the driving lanes.

    AlpaSim asks whether the union of the lanes around the ego covers its box;
    CARLA answers per point, so this asks it of the box's corners.
    """
    frame = _frame(world)
    if frame.ego is None:
        return None
    for corner in frame.ego_box:
        if _driving_waypoint(world, corner) is None:
            return True
    return False


def _offroad(world: "carla.World") -> Optional[float]:
    offroad = _is_offroad(world)
    return None if offroad is None else _event(offroad)


def _offroad_or_collision(world: "carla.World") -> Optional[float]:
    offroad = _is_offroad(world)
    found = _collisions(world)
    if offroad is None or found is None:
        return None
    return _event(offroad or found.any)


def _offroad_or_collision_at_fault(world: "carla.World") -> Optional[float]:
    offroad = _is_offroad(world)
    found = _collisions(world)
    if offroad is None or found is None:
        return None
    return _event(offroad or found.at_fault)


def _heading_gap(one_deg: float, other_deg: float) -> float:
    """The angle between two headings, in radians, from 0 to pi."""
    gap = math.radians(one_deg - other_deg) % (2.0 * math.pi)
    return min(gap, 2.0 * math.pi - gap)


def _wrong_lane(world: "carla.World") -> Optional[float]:
    """AlpaSim's ``wrong_lane``: the ego heads more than
    :data:`WRONG_LANE_HEADING_RAD` away from the driving lane under its centre.

    ``None`` when its centre is on no driving lane: off the road, there is no
    lane to be the wrong way along.
    """
    frame = _frame(world)
    if frame.ego is None:
        return None
    waypoint = _driving_waypoint(world, _centre(frame.ego_box))
    if waypoint is None:
        return None
    gap = _heading_gap(
        frame.ego.get_transform().rotation.yaw, waypoint.transform.rotation.yaw
    )
    return _event(gap > WRONG_LANE_HEADING_RAD)


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
    within :data:`LANE_SEARCH_RADIUS_M` of the ego's centre, *start* metres
    from it; their widths are summed.  A lane of the opposite direction counts,
    as it does in AlpaSim, whose lane union takes every lane around the ego.
    """
    driving = _driving_lane_type()
    reach = 0.0
    lane = waypoint
    for _ in range(_MAX_SIDE_LANES):
        beside = lane.get_right_lane() if right else lane.get_left_lane()
        if beside is None or int(beside.lane_type) != driving:
            break
        if start + reach > LANE_SEARCH_RADIUS_M:
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
    if frame.ego is None:
        return None
    offroad = _is_offroad(world)
    if offroad is None or offroad:
        return 0.0
    centre = _centre(frame.ego_box)
    waypoint = _driving_waypoint(world, centre)
    if waypoint is None:
        return 0.0
    half = float(waypoint.lane_width) / 2.0
    laterals = [_lateral(waypoint, corner) for corner in frame.ego_box]
    centre_lateral = _lateral(waypoint, centre)
    right_edge = half + _side_width(waypoint, True, half - centre_lateral)
    left_edge = half + _side_width(waypoint, False, half + centre_lateral)
    gap = min(right_edge - max(laterals), left_edge + min(laterals))
    return max(gap, 0.0)


def _by_key(metrics: list[Metric]) -> dict[str, Metric]:
    found: dict[str, Metric] = {}
    for metric in metrics:
        found[metric.key] = metric
    return found


#: The built-in metrics, by key: AlpaSim's, as CARLA reads them.
BUILT_IN_METRICS: dict[str, Metric] = _by_key(
    [
        Metric(
            COLLISION_ANY,
            _collision_any,
            "",
            "1 while the ego's bounding box overlaps another vehicle's or a "
            "pedestrian's, else 0 (AlpaSim collision_any)",
            aggregation="max",
        ),
        Metric(
            COLLISION_FRONT,
            _collision_front,
            "",
            "1 while another road user's box overlaps the ego's front edge "
            "(AlpaSim collision_front)",
            aggregation="max",
        ),
        Metric(
            COLLISION_REAR,
            _collision_rear,
            "",
            "1 while another road user's box overlaps the ego's rear edge and "
            "not its front (AlpaSim collision_rear)",
            aggregation="max",
        ),
        Metric(
            COLLISION_LATERAL,
            _collision_lateral,
            "",
            "1 while another road user's box overlaps the ego's and neither its "
            "front nor its rear edge (AlpaSim collision_lateral)",
            aggregation="max",
        ),
        Metric(
            COLLISION_AT_FAULT,
            _collision_at_fault,
            "",
            "1 during a front or a lateral collision (AlpaSim collision_at_fault)",
            aggregation="max",
        ),
        Metric(
            OFFROAD,
            _offroad,
            "",
            "1 while a corner of the ego's box is on no driving lane "
            "(AlpaSim offroad)",
            aggregation="max",
        ),
        Metric(
            OFFROAD_OR_COLLISION,
            _offroad_or_collision,
            "",
            "1 while off the road or in any collision (AlpaSim offroad_or_collision)",
            aggregation="max",
        ),
        Metric(
            OFFROAD_OR_COLLISION_AT_FAULT,
            _offroad_or_collision_at_fault,
            "",
            "1 while off the road or in a collision at fault "
            "(AlpaSim offroad_or_collision_at_fault)",
            aggregation="max",
        ),
        Metric(
            WRONG_LANE,
            _wrong_lane,
            "",
            "1 while the ego heads more than 120 degrees away from the driving "
            "lane under it, else 0; unread off the road (AlpaSim wrong_lane)",
            aggregation="max",
        ),
        Metric(
            MIN_DISTANCE_TO_OBSTACLE_M,
            _min_distance_to_obstacle_m,
            "m",
            "Box to box, to the nearest other vehicle or pedestrian; 0 when "
            "touching (AlpaSim min_distance_to_obstacle_m)",
            aggregation="min",
        ),
        Metric(
            MIN_DISTANCE_TO_LANE_BOUNDARY_M,
            _min_distance_to_lane_boundary_m,
            "m",
            "From the ego's box to the edge of the driving lanes around it; 0 off "
            "the road (AlpaSim min_distance_to_lane_boundary_m)",
            aggregation="min",
        ),
    ]
)


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
