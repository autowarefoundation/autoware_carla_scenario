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

Fourteen read the criticality of the frame with TU Munich's CommonRoad tools:
eleven CriMe measures (:data:`CRIME_TTC_S`, :data:`CRIME_HW_M`,
:data:`CRIME_BTN`, :data:`CRIME_TTR_S`, ...), the area the ego can still reach by CommonRoad-Reach
(:data:`REACH_DRIVABLE_AREA_M2`, :data:`REACH_DRIVABLE_AREA_SUM_M2`) and, by
that reachable set, the worst-case time it has left to react
(:data:`CRIME_WTTR_S`).  Those packages run in worker processes
of their own (:mod:`autoware_carla_scenario.commonroad`), on the frames a run
recorded, once it is over (:mod:`autoware_carla_scenario.metric_batch`):
:func:`read_metric` reads nothing of them off a live world.

AlpaSim's metrics that need more than the world -- the logged ground-truth
trajectory (``progress``, ``dist_to_gt_trajectory``, ``dist_to_gt_location``,
``min_ade``), the driver's plan (``plan_deviation``, ``open_loop_collision``),
its safety monitor (``safety_monitor_triggered``) or a camera image
(``img_is_black``) -- are not metrics here; ``docs/metrics.md`` says why.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from .constants import EGO_ROLE_NAME
from .entity_role import find_actor_in_world
from .failure import FailureCategory
from .utils.clearance import _HIDDEN_BELOW_M, _level_gap, _vertical_span

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "BUILT_IN_METRICS",
    "COLLISION_ANY",
    "COLLISION_AT_FAULT",
    "COLLISION_FRONT",
    "COLLISION_LATERAL",
    "COLLISION_REAR",
    "COMMONROAD_METRICS",
    "CRIME_A_LAT_REQ_MPS2",
    "CRIME_A_LONG_REQ_MPS2",
    "CRIME_BTN",
    "CRIME_DCE_M",
    "CRIME_HW_M",
    "CRIME_STN",
    "CRIME_THW_S",
    "CRIME_TTB_S",
    "CRIME_TTCE_S",
    "CRIME_TTC_S",
    "CRIME_TTR_S",
    "CRIME_WTTR_S",
    "IMPORTANCE_SAMPLING_WEIGHT",
    "LATERAL_REVERSALS",
    "LATERAL_WOBBLE_M",
    "MIN_DISTANCE_TO_LANE_BOUNDARY_M",
    "MIN_DISTANCE_TO_OBSTACLE_M",
    "Metric",
    "OFFROAD",
    "OFFROAD_OR_COLLISION",
    "OFFROAD_OR_COLLISION_AT_FAULT",
    "REACH_DRIVABLE_AREA_M2",
    "REACH_DRIVABLE_AREA_SUM_M2",
    "STEERING_REVERSALS",
    "TURN_INDICATOR_CHANGES",
    "WRONG_LANE",
    "is_metric",
    "read_metric",
    "ego_progress",
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
#: Seconds until the ego hits a road user ahead in its lane, both holding
#: their acceleration until they stop (CriMe's TTC); ``inf`` when they never
#: meet.
CRIME_TTC_S = "crime_ttc_s"
#: Seconds until the ego reaches where the road user ahead is now (CriMe's
#: THW, time headway).
CRIME_THW_S = "crime_thw_s"
#: Metres from the ego's front to the rear of the road user ahead, along the
#: lane (CriMe's HW, headway).
CRIME_HW_M = "crime_hw_m"
#: Seconds left before the ego must brake at full to avoid a collision
#: (CriMe's TTB, time-to-brake).
CRIME_TTB_S = "crime_ttb_s"
#: Seconds until the ego and a road user come closest, both going on as they
#: go (CriMe's TTCE, time-to-closest-encounter).
CRIME_TTCE_S = "crime_ttce_s"
#: Metres between the ego and a road user when they come closest (CriMe's DCE,
#: distance of closest encounter); ``0`` when they would collide.
CRIME_DCE_M = "crime_dce_m"
#: The longitudinal acceleration the ego needs not to hit the road user ahead,
#: m/s^2, negative to brake (CriMe's ALongReq): the least deceleration that
#: avoids it, counting a contact within the prediction horizon (``horizon_s``,
#: 3 s by default) after the ego has stopped; ``-inf`` when braking alone
#: cannot avoid a contact within it.
CRIME_A_LONG_REQ_MPS2 = "crime_a_long_req_mps2"
#: The lateral acceleration the ego needs to steer clear of a road user, m/s^2
#: (CriMe's ALatReq).
CRIME_A_LAT_REQ_MPS2 = "crime_a_lat_req_mps2"
#: The deceleration the ego needs not to hit the road user ahead over the most
#: it can brake (CriMe's BTN, brake threat number): above ``1`` braking alone
#: no longer avoids it; ``inf`` when no braking avoids a contact within the
#: prediction horizon (``horizon_s``, 3 s by default).
CRIME_BTN = "crime_btn"
#: The lateral acceleration the ego needs to steer clear of a road user over the
#: most it can steer (CriMe's STN, steer threat number): above ``1`` steering
#: alone no longer avoids it.
CRIME_STN = "crime_stn"
#: Seconds the ego can go on as it goes before no evasive manoeuvre -- braking,
#: kick-down, steering left or right -- avoids the collision any more (CriMe's
#: TTR, time-to-react, by simulating each); ``inf`` when it hits no one, ``0``
#: when none does even now.  Seconds a frame: evaluated every 0.5 s by default
#: (:attr:`~autoware_carla_scenario.commonroad.CommonRoadSettings.metric_periods_s`).
CRIME_TTR_S = "crime_ttr_s"
#: Seconds the ego can go on as it goes before no way it can go is free of
#: collisions to the end of the horizon (CriMe's WTTR, worst-case
#: time-to-react, by CommonRoad-Reach's reachable set); ``inf`` when it hits no
#: one, ``0`` when no way is left even now.
CRIME_WTTR_S = "crime_wttr_s"
#: Square metres of road the ego can still reach without a collision at the
#: end of the horizon (CommonRoad-Reach's drivable area); ``0`` when every
#: way it can go meets a road user: a collision it can no longer avoid.
REACH_DRIVABLE_AREA_M2 = "reach_drivable_area_m2"
#: The drivable area summed over the horizon's steps (CriMe's DA), m^2.
REACH_DRIVABLE_AREA_SUM_M2 = "reach_drivable_area_sum_m2"

#: The metrics the CommonRoad worker reads (:mod:`autoware_carla_scenario.commonroad`).
COMMONROAD_METRICS: tuple[str, ...] = (
    CRIME_TTC_S,
    CRIME_THW_S,
    CRIME_HW_M,
    CRIME_TTB_S,
    CRIME_TTCE_S,
    CRIME_DCE_M,
    CRIME_A_LONG_REQ_MPS2,
    CRIME_A_LAT_REQ_MPS2,
    CRIME_BTN,
    CRIME_STN,
    CRIME_TTR_S,
    CRIME_WTTR_S,
    REACH_DRIVABLE_AREA_M2,
    REACH_DRIVABLE_AREA_SUM_M2,
)

#: How far the ego must turn from its lane's heading to be against it, in
#: degrees: AlpaSim's 2/3 pi.
_WRONG_LANE_HEADING_DEG = 120.0
#: How far beyond a driving lane's edge a point still counts as on the lane,
#: in metres, when CARLA's point query finds no lane under it (#1997): that
#: query misses a band about 0.2 m wide inside a lane, next to the boundary
#: with the lane beside it, and the lane projected to is measured instead
#: (:func:`_driving_lane_at`).  A tenth of a metre takes up the rounding of
#: the lane's width; a point further out is off the road.
_LANE_EDGE_TOLERANCE_M = 0.1
#: How far before or beyond a lane's waypoint, along the lane, a point still
#: counts as across from it, in metres: CARLA's projection lands up to about
#: 0.3 m from the foot of the perpendicular, and the waypoint of the lane
#: beside it at the same ``s`` is not quite across from it on a bend.  A point
#: further along is measured against the lane followed to it.
_LANE_ALONG_TOLERANCE_M = 0.3
#: How far above or below the lane projected to a point may be and still be
#: on it, in metres: less than a level of road over another (a bridge, an
#: underpass), whose lane is not the one under the point.
_LANE_LEVEL_M = 2.5
#: The spacing of the points the map's junction lanes are sampled at to find
#: every lane under a point in a junction (:func:`_junction_lanes_near`), in
#: metres.
_JUNCTION_SAMPLE_M = 1.0
#: The side of the square cells the junction lanes' samples are filed in, in
#: metres: more than a lane's half width and a sample's reach, so a point's
#: cell and the eight around it hold every sample it can be on.
_JUNCTION_CELL_M = 8.0
#: How far across a junction lane its centre line may be from the ego's
#: centre for the ego to count as on it, in metres, at most its half width
#: and :data:`_LANE_EDGE_TOLERANCE_M`: a lane the ego's centre is on the edge
#: of is more likely one beside the ego's than the one it follows, and taking
#: it would hide the ego driving the wrong way along its own.
_JUNCTION_LATERAL_M = 1.0
#: How far from the ego's centre a lane beside its own still counts towards
#: the drivable area, in metres: AlpaSim's lane search radius.
_LANE_SEARCH_RADIUS_M = 6.0
#: How many lanes beside its own the drivable area is walked to, at most.
_MAX_SIDE_LANES = 8
#: How far the ego's drift across its lane must turn back to count as a
#: reversal, in metres: less is the noise of holding a lane.
_LATERAL_REVERSAL_GAP_M = 0.1
#: How far the steering must turn back to count as a reversal, in CARLA's
#: normalised steer (``1`` the vehicle's full lock): about 3.5 degrees at the
#: wheels of a car that locks at 70.  A driver policy's controller holding a
#: lane or a bend moves its steer by a few hundredths, closed loop; sawing
#: through a turn, by a tenth or more.
_STEERING_REVERSAL_GAP = 0.05
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
#: A point on a lane's centre line: ``(x, y, z, yaw, half_width)``, the yaw
#: in degrees along the lane's direction of travel, the half width in metres.
_LaneSample = tuple[float, float, float, float, float]


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
        boolean: A yes/no event: it reads ``1`` or ``0`` (and is judged so),
            and the run's trajectory log writes it as ``true`` or ``false``.
    """

    key: str
    read: Callable[[carla.World], Optional[float]]
    title: str = ""
    unit: str = ""
    text: str = ""
    aggregation: str = "max"
    boolean: bool = False


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
    """Whether *point* is in the convex *box* or on its edge.

    A box of no area (a road user whose extent read as zero: a point, or a
    segment) holds only the points on it (#1997): every side's cross product
    is zero for it, which would otherwise hold every point on the line.
    """
    sign = 0.0
    for i in range(len(box)):
        side = _cross(box[i], box[(i + 1) % len(box)], point)
        if abs(side) <= 1e-12:
            continue
        if sign == 0.0:
            sign = side
        elif (side > 0.0) != (sign > 0.0):
            return False
    if sign == 0.0:
        # Every side is in line with the point: the box is a point or a
        # segment (or the point is a corner of a box that is); on it or not.
        for i in range(len(box)):
            if _point_segment_distance(point, box[i], box[(i + 1) % len(box)]) <= 1e-9:
                return True
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

    The boxes are on the ground, and each other road user has how far it is
    above or below the ego's level of road (:attr:`others_gap`): where roads
    cross at different heights, a road user on the bridge over the ego is in
    no collision with it, and is as far from it as the height between them at
    least.  The lanes are asked for at the ego's height (:attr:`ego_z`), so
    that they are the ones it drives on (#1580).

    Every box also has the circle around it (its centre and the distance to
    its corners): two boxes whose circles are apart are apart, which spares
    the exact tests for the road users nowhere near the ego.
    """

    has_ego: bool
    ego_yaw: float
    #: The height of the ego's origin, which the map is asked at.
    ego_z: float
    ego_box: Box
    ego_centre: Point
    ego_radius: float
    others: list[Box]
    others_centre: list[Point]
    others_radius: list[float]
    #: Metres between the ego's box and each other's in height, ``0.0`` for
    #: one on the ego's level of road (``utils.clearance._level_gap``).
    others_gap: list[float]
    #: What the ego's box meets: _FRONT, _REAR and _LATERAL once the road
    #: users have been read (collisions_read), _OFFROAD once the road has
    #: (offroad_read).
    hits: int
    collisions_read: bool
    offroad_read: bool
    centre_read: bool
    centre_waypoint: Optional["carla.Waypoint"]

    def __init__(
        self,
        ego_yaw: float,
        ego_z: float,
        ego_box: Box,
        others: list[Box],
        others_gap: list[float],
    ) -> None:
        self.has_ego = bool(ego_box)
        self.ego_yaw = ego_yaw
        self.ego_z = ego_z
        self.ego_box = ego_box
        self.ego_centre = _centre(ego_box) if ego_box else (0.0, 0.0)
        self.ego_radius = _radius(ego_box, self.ego_centre) if ego_box else 0.0
        self.others = others
        self.others_centre = [_centre(box) for box in others]
        self.others_radius = [
            _radius(box, centre) for box, centre in zip(others, self.others_centre)
        ]
        self.others_gap = others_gap
        self.hits = 0
        self.collisions_read = False
        self.offroad_read = False
        self.centre_read = False
        self.centre_waypoint = None

    def clearance(self, i: int) -> float:
        """How far apart the ego's circle and the *i*-th road user's are, with
        the height between them (:attr:`others_gap`); at most the distance
        between their boxes, and ``0.0`` when they overlap."""
        centre = self.others_centre[i]
        apart = math.hypot(
            centre[0] - self.ego_centre[0], centre[1] - self.ego_centre[1]
        )
        planar = max(apart - self.ego_radius - self.others_radius[i], 0.0)
        return math.hypot(planar, self.others_gap[i])

    def distance(self, i: int) -> float:
        """Metres between the ego's box and the *i*-th road user's: on the
        ground, and across the height between them when it is on another
        level of road; ``0.0`` when they meet."""
        return math.hypot(
            _box_distance(self.ego_box, self.others[i]), self.others_gap[i]
        )


_NO_EGO = _Frame(0.0, 0.0, [], [], [])


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
    #: The map's driving lanes in junctions, sampled every
    #: :data:`_JUNCTION_SAMPLE_M` and filed by :data:`_JUNCTION_CELL_M` cell
    #: (:func:`_junction_lanes_near`), once junctions_read.
    junctions: dict[tuple[int, int], list[_LaneSample]]
    junctions_read: bool

    def __init__(self) -> None:
        self.world = None
        self.at = None
        self.frame = None
        self.carla_map = None
        self.driving = 0
        self.junctions = {}
        self.junctions_read = False


_CACHE = _Cache()


def _hold(world: "carla.World") -> None:
    """Make *world* the one the cache holds, dropping another's."""
    held = _CACHE.world
    if held is None or held is not world:
        _CACHE.world = world
        _CACHE.at, _CACHE.frame, _CACHE.carla_map = None, None, None
        _CACHE.junctions = {}
        _CACHE.junctions_read = False


def _frame(world: "carla.World") -> _Frame:
    _hold(world)
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
    transform = ego.get_transform()
    ego_span = _vertical_span(ego, transform)
    others: list[Box] = []
    others_gap: list[float] = []
    for actor in actors:
        kind = actor.type_id
        if actor.id == ego.id or not (
            kind.startswith("vehicle.") or kind.startswith("walker.")
        ):
            continue
        placed = actor.get_transform()
        # A road user parked under the map (hidden) is out of the scene.
        if placed.location.z < _HIDDEN_BELOW_M:
            continue
        others.append(_box_of(actor, placed))
        # One on another level of road -- on a bridge over the ego, in an
        # underpass beneath it -- may overlap the ego's box on the ground, but
        # is as far from it as the height between them.
        others_gap.append(_level_gap(ego_span, _vertical_span(actor, placed)))
    return _Frame(
        transform.rotation.yaw,
        float(transform.location.z),
        _box_of(ego, transform),
        others,
        others_gap,
    )


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


def _along(waypoint: "carla.Waypoint", point: Point) -> float:
    """How far *point* is ahead of *waypoint* along its lane, in metres."""
    transform = waypoint.transform
    yaw = math.radians(transform.rotation.yaw)
    dx = point[0] - transform.location.x
    dy = point[1] - transform.location.y
    return dx * math.cos(yaw) + dy * math.sin(yaw)


def _on_lane(waypoint: "carla.Waypoint", point: Point, z: float, driving: int) -> bool:
    """Whether *point*, at the height *z*, is on *waypoint*'s lane where the
    waypoint is: a driving lane, across from it (within
    :data:`_LANE_ALONG_TOLERANCE_M` along the lane), within its half width and
    :data:`_LANE_EDGE_TOLERANCE_M` across it, and on its level of road."""
    if int(waypoint.lane_type) != driving:
        return False
    if abs(float(waypoint.transform.location.z) - z) > _LANE_LEVEL_M:
        return False
    if abs(_along(waypoint, point)) > _LANE_ALONG_TOLERANCE_M:
        return False
    half = float(waypoint.lane_width) / 2.0
    return abs(_lateral(waypoint, point)) <= half + _LANE_EDGE_TOLERANCE_M


def _on_lane_or_beside(
    waypoint: "carla.Waypoint", point: Point, z: float, driving: int
) -> "Optional[carla.Waypoint]":
    """*waypoint* when *point* is on its lane (:func:`_on_lane`), else the
    lane to its left or its right when the point is on that, else ``None``."""
    if _on_lane(waypoint, point, z, driving):
        return waypoint
    left = waypoint.get_left_lane()
    if left is not None and _on_lane(left, point, z, driving):
        return left
    right = waypoint.get_right_lane()
    if right is not None and _on_lane(right, point, z, driving):
        return right
    return None


def _driving_lane_at(
    carla_map: "carla.Map", point: Point, z: float, driving: int
) -> "Optional[carla.Waypoint]":
    """The driving lane under *point*, at the height *z*, ``None`` when it is
    on none.

    CARLA's point query (``get_waypoint(project_to_road=False)``) is asked
    first.  It finds no lane in a band about 0.2 m wide inside a lane, next to
    its boundary with the lane beside it (#1997), so when it finds none the
    point is projected to the nearest driving lane, and is on the road when it
    is within that lane's half width, or the half width of a driving lane
    beside it, plus :data:`_LANE_EDGE_TOLERANCE_M`.  The projection lands a
    few tenths of a metre before or beyond the point along the lane, and past
    a lane's end when the point is beyond it, so the lane is followed (onto
    the lanes before or after it) to where it is across from the point before
    it is measured.  A point beyond the outer lane's edge -- on a shoulder, a
    sidewalk, off the road -- is further out than that, and stays off.
    """
    import typesafe_carla.carla as carla  # noqa: PLC0415

    location = carla.Location(x=point[0], y=point[1], z=z)
    found = carla_map.get_waypoint(location, project_to_road=False, lane_type=driving)
    if found is not None:
        return found
    near = carla_map.get_waypoint(location, project_to_road=True, lane_type=driving)
    if near is None:
        return None
    on = _on_lane_or_beside(near, point, z, driving)
    if on is not None:
        return on
    ahead = _along(near, point)
    if abs(ahead) <= _LANE_ALONG_TOLERANCE_M:
        return None
    moved = near.next(ahead) if ahead > 0.0 else near.previous(-ahead)
    for waypoint in moved:
        on = _on_lane_or_beside(waypoint, point, z, driving)
        if on is not None:
            return on
    return None


def _driving_waypoint(
    world: "carla.World", point: Point, z: float
) -> "Optional[carla.Waypoint]":
    """The driving lane under *point*, at the height *z* (the ego's: where
    roads cross at different heights, the lane is the one it is on), ``None``
    when it is on none (:func:`_driving_lane_at`)."""
    carla_map = _map(world)
    return _driving_lane_at(carla_map, point, z, _CACHE.driving)


def _collision_bits(frame: _Frame) -> int:
    """AlpaSim's ``CollisionScorer``: each road user the ego's box meets meets
    its front edge, else its rear edge, else its side.  One on another level
    of road (a bridge, an underpass) meets none."""
    if not frame.collisions_read:
        box = frame.ego_box
        for i, other in enumerate(frame.others):
            # Its height above or below the ego's level is in the clearance.
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
            if _driving_waypoint(world, corner, frame.ego_z) is None:
                frame.hits |= _OFFROAD
                break
        frame.offroad_read = True
    return frame.hits


def _centre_waypoint(world: "carla.World", frame: _Frame) -> "Optional[carla.Waypoint]":
    """The driving lane under the ego's centre."""
    if not frame.centre_read:
        frame.centre_waypoint = _driving_waypoint(world, frame.ego_centre, frame.ego_z)
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

    def __init__(self, gap: float) -> None:
        self.gap = gap
        self.count = 0
        self.heading = 0
        self.extreme = 0.0
        self.low = math.inf
        self.high = -math.inf

    def add(self, value: float) -> None:
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
    window: deque[tuple[float, float]]
    lateral_reversals: _Reversals
    steering_reversals: _Reversals
    #: The indicators the ego showed last (``TurnIndicatorState.value``), -1
    #: before the first frame, and how many times they changed since.
    indicator: int
    indicator_changes: int
    #: How far the ego has driven (m); whether that is :data:`_MOVED_M`, and
    #: since when it has stood still (:data:`_STILL_MPS`), ``None`` while it
    #: moves.
    travelled_m: float
    moved: bool
    still_since: Optional[float]

    def __init__(self) -> None:
        self.ego_id = -1
        self.at = None
        self.lateral = 0.0
        self.window = deque()
        self.lateral_reversals = _Reversals(_LATERAL_REVERSAL_GAP_M)
        self.steering_reversals = _Reversals(_STEERING_REVERSAL_GAP)
        self.indicator = -1
        self.indicator_changes = 0
        self.travelled_m = 0.0
        self.moved = False
        self.still_since = None

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
    nothing.  Off the driving lanes, and in a junction, the ego has no lane
    to drift across, and its drift holds until it is back on one.  Another ego starts the motion
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
    try:
        velocity = ego.get_velocity()
    except Exception:
        return
    _track_progress(
        motion,
        math.hypot(velocity.x, velocity.y),
        elapsed,
        0.0 if last is None else elapsed - last,
    )
    if last is not None:
        # The ego's centre alone, not the frame :func:`_frame` reads: that
        # reads every actor, and this runs every tick whether or not anything
        # reads a metric.
        _hold(world)
        try:
            transform = ego.get_transform()
            waypoint = _driving_waypoint(
                world, _centre(_box_of(ego, transform)), float(transform.location.z)
            )
        except Exception:
            waypoint = None
        # A junction's lanes overlap, and the one read may be another's, at
        # any angle to the ego's: its drift holds through the junction.
        if waypoint is not None and not waypoint.is_junction:
            yaw = math.radians(waypoint.transform.rotation.yaw)
            # A lane the other way (overtaking on it) is crossed the ego's way.
            if math.cos(yaw - math.radians(transform.rotation.yaw)) < 0.0:
                yaw += math.pi
            across = -velocity.x * math.sin(yaw) + velocity.y * math.cos(yaw)
            motion.lateral += across * (elapsed - last)
    motion.lateral_reversals.add(motion.lateral)
    start = elapsed - _WOBBLE_WINDOW_S
    while motion.window and motion.window[0][0] < start:
        motion.window.popleft()
    motion.window.append((elapsed, motion.lateral))


#: The ego has moved once it has driven this far (m): a creep counts, a
#: roll on the spot does not.
_MOVED_M = 1.0
#: It stands still while it goes slower than this (m/s).
_STILL_MPS = 0.3


def _track_progress(motion: _Motion, speed: float, elapsed: float, dt: float) -> None:
    """Note how far the ego, going at *speed* (m/s) for the *dt* s up to
    *elapsed*, has driven, and since when it has stood still."""
    motion.travelled_m += speed * dt
    if motion.travelled_m >= _MOVED_M:
        motion.moved = True
    if speed >= _STILL_MPS:
        motion.still_since = None
    elif motion.still_since is None:
        motion.still_since = elapsed


def ego_progress() -> Optional[tuple[bool, float]]:
    """Whether the ego has moved this run, and how long it had stood still
    at its last frame, in seconds; ``None`` before its first frame.

    What tells a timeout apart (:func:`~autoware_carla_scenario.failure.progress_detail`).
    """
    motion = _MOTION[0]
    at = motion.at
    if at is None:
        return None
    since = motion.still_since
    return motion.moved, 0.0 if since is None else at - since


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
    if motion.at is None:
        return None
    try:
        ego = find_actor_in_world(world, EGO_ROLE_NAME)
    except Exception:
        return None
    return motion if ego is not None and ego.id == motion.ego_id else None


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


# ---------------------------------------------------------------------------
# CommonRoad: CriMe's and CommonRoad-Reach's, read in a worker of their own
# ---------------------------------------------------------------------------


def _commonroad(key: str) -> Optional[float]:
    """Nothing: a CommonRoad metric is computed from a run's recorded frames
    once it is over (:mod:`autoware_carla_scenario.metric_batch`), not read
    off a live world."""
    return None


def _crime_ttc_s(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_TTC_S)


def _crime_thw_s(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_THW_S)


def _crime_hw_m(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_HW_M)


def _crime_ttb_s(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_TTB_S)


def _crime_ttce_s(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_TTCE_S)


def _crime_dce_m(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_DCE_M)


def _crime_a_long_req_mps2(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_A_LONG_REQ_MPS2)


def _crime_a_lat_req_mps2(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_A_LAT_REQ_MPS2)


def _crime_btn(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_BTN)


def _crime_stn(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_STN)


def _crime_ttr_s(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_TTR_S)


def _crime_wttr_s(world: "carla.World") -> Optional[float]:
    return _commonroad(CRIME_WTTR_S)


def _reach_drivable_area_m2(world: "carla.World") -> Optional[float]:
    return _commonroad(REACH_DRIVABLE_AREA_M2)


def _reach_drivable_area_sum_m2(world: "carla.World") -> Optional[float]:
    return _commonroad(REACH_DRIVABLE_AREA_SUM_M2)


def _offroad(world: "carla.World") -> Optional[float]:
    return _event(world, _OFFROAD)


def _offroad_or_collision(world: "carla.World") -> Optional[float]:
    return _event(world, _OFFROAD | _ANY_COLLISION)


def _offroad_or_collision_at_fault(world: "carla.World") -> Optional[float]:
    return _event(world, _OFFROAD | _AT_FAULT)


def _min_distance_to_obstacle_m(world: "carla.World") -> Optional[float]:
    """AlpaSim's ``MinDistanceToObstacleScorer``: box to box, ``None`` when
    the ego is alone; across the height between them too for a road user on
    another level of road (:meth:`_Frame.distance`)."""
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
        best = min(best, frame.distance(i))
    return best


def _cell(point: Point) -> tuple[int, int]:
    """The :data:`_JUNCTION_CELL_M` cell *point* is in."""
    return (
        int(math.floor(point[0] / _JUNCTION_CELL_M)),
        int(math.floor(point[1] / _JUNCTION_CELL_M)),
    )


def _junction_lanes_near(world: "carla.World", point: Point) -> list[_LaneSample]:
    """The samples of the map's junction lanes in *point*'s cell and the eight
    around it.

    Every driving lane in a junction is sampled every
    :data:`_JUNCTION_SAMPLE_M` (``Map.generate_waypoints``), once per map:
    ``Junction.get_waypoints`` and the map's topology leave out some of a
    junction's connecting lanes.
    """
    if not _CACHE.junctions_read:
        cells: dict[tuple[int, int], list[_LaneSample]] = {}
        driving = _CACHE.driving
        for waypoint in _map(world).generate_waypoints(_JUNCTION_SAMPLE_M):
            if not waypoint.is_junction or int(waypoint.lane_type) != driving:
                continue
            transform = waypoint.transform
            x, y = float(transform.location.x), float(transform.location.y)
            sample: _LaneSample = (
                x,
                y,
                float(transform.location.z),
                float(transform.rotation.yaw),
                float(waypoint.lane_width) / 2.0,
            )
            cells.setdefault(_cell((x, y)), []).append(sample)
        _CACHE.junctions = cells
        _CACHE.junctions_read = True
    column, row = _cell(point)
    near: list[_LaneSample] = []
    for i in range(column - 1, column + 2):
        for j in range(row - 1, row + 2):
            found = _CACHE.junctions.get((i, j))
            if found is not None:
                near.extend(found)
    return near


def _heads_along_junction_lane(
    world: "carla.World", point: Point, z: float, yaw: float
) -> bool:
    """Whether a driving lane in a junction under *point* (at the height *z*)
    runs within :data:`_WRONG_LANE_HEADING_DEG` of *yaw*.

    A junction's connecting lanes overlap each other and the ends of the roads
    around it, and the lane CARLA finds under a point may be another
    approach's, at any angle to the ego's (#1997): each sample of a junction
    lane (:func:`_junction_lanes_near`) the point is across from, within half
    a sample's spacing along it and, across it, within its half width and
    :data:`_LANE_EDGE_TOLERANCE_M` and within :data:`_JUNCTION_LATERAL_M`, is
    a lane under the point.
    """
    from .kinematics.angle import normalize_angle_deg  # noqa: PLC0415

    reach = _JUNCTION_SAMPLE_M / 2.0 + _LANE_EDGE_TOLERANCE_M
    for x, y, sample_z, sample_yaw, half in _junction_lanes_near(world, point):
        if abs(sample_z - z) > _LANE_LEVEL_M:
            continue
        across = min(half + _LANE_EDGE_TOLERANCE_M, _JUNCTION_LATERAL_M)
        dx, dy = point[0] - x, point[1] - y
        if dx * dx + dy * dy > (across + reach) * (across + reach):
            continue
        rad = math.radians(sample_yaw)
        if abs(dx * math.cos(rad) + dy * math.sin(rad)) > reach:
            continue
        if abs(-dx * math.sin(rad) + dy * math.cos(rad)) > across:
            continue
        if abs(normalize_angle_deg(yaw - sample_yaw)) <= _WRONG_LANE_HEADING_DEG:
            return True
    return False


def _wrong_lane(world: "carla.World") -> Optional[float]:
    """AlpaSim's ``wrong_lane``: the ego heads more than 120 degrees away from
    the driving lane under its centre.

    Where junction lanes are under its centre -- in a junction, whose
    connecting lanes overlap, and on the end of a road they reach over -- it
    is against the lane only when it heads more than 120 degrees away from
    the lane found and from every junction lane under its centre
    (:func:`_heads_along_junction_lane`): the lane CARLA finds there may be
    another approach's, and the ego on any of them is on its way.

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
    if gap <= _WRONG_LANE_HEADING_DEG:
        return 0.0
    if _heads_along_junction_lane(world, frame.ego_centre, frame.ego_z, frame.ego_yaw):
        return 0.0
    return 1.0


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


#: The built-in metrics, by key: AlpaSim's, as CARLA reads them, and
#: CriMe's and CommonRoad-Reach's (:data:`COMMONROAD_METRICS`).
BUILT_IN_METRICS: dict[str, Metric] = {
    m.key: m
    for m in [
        Metric(
            COLLISION_ANY,
            _collision_any,
            title="Collision (any)",
            boolean=True,
            text="1 while the ego's bounding box overlaps another vehicle's or "
            "a pedestrian's, else 0 (AlpaSim collision_any)",
        ),
        Metric(
            COLLISION_FRONT,
            _collision_front,
            title="Collision (front)",
            boolean=True,
            text="1 while another road user's box overlaps the ego's front edge "
            "(AlpaSim collision_front)",
        ),
        Metric(
            COLLISION_REAR,
            _collision_rear,
            title="Collision (rear)",
            boolean=True,
            text="1 while another road user's box overlaps the ego's rear edge "
            "and not its front (AlpaSim collision_rear)",
        ),
        Metric(
            COLLISION_LATERAL,
            _collision_lateral,
            title="Collision (lateral)",
            boolean=True,
            text="1 while another road user's box overlaps the ego's and "
            "neither its front nor its rear edge (AlpaSim collision_lateral)",
        ),
        Metric(
            COLLISION_AT_FAULT,
            _collision_at_fault,
            title="Collision at fault",
            boolean=True,
            text="1 during a front or a lateral collision (AlpaSim collision_at_fault)",
        ),
        Metric(
            OFFROAD,
            _offroad,
            title="Off the road",
            boolean=True,
            text="1 while a corner of the ego's box is on no driving lane "
            "(AlpaSim offroad)",
        ),
        Metric(
            OFFROAD_OR_COLLISION,
            _offroad_or_collision,
            title="Off the road or collision",
            boolean=True,
            text="1 while off the road or in any collision "
            "(AlpaSim offroad_or_collision)",
        ),
        Metric(
            OFFROAD_OR_COLLISION_AT_FAULT,
            _offroad_or_collision_at_fault,
            title="Off the road or collision at fault",
            boolean=True,
            text="1 while off the road or in a collision at fault "
            "(AlpaSim offroad_or_collision_at_fault)",
        ),
        Metric(
            WRONG_LANE,
            _wrong_lane,
            title="Against the lane",
            boolean=True,
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
            "turned back by more than 0.05 of full lock",
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
        Metric(
            CRIME_TTC_S,
            _crime_ttc_s,
            title="Time to collision (CriMe)",
            unit="s",
            text="Seconds until the ego hits a road user ahead in its lane, "
            "both holding their acceleration until they stop; inf when they "
            "never meet (CriMe TTC)",
            aggregation="min",
        ),
        Metric(
            CRIME_THW_S,
            _crime_thw_s,
            title="Time headway (CriMe)",
            unit="s",
            text="Seconds until the ego reaches where the road user ahead is "
            "now (CriMe THW)",
            aggregation="min",
        ),
        Metric(
            CRIME_HW_M,
            _crime_hw_m,
            title="Headway (CriMe)",
            unit="m",
            text="From the ego's front to the rear of the road user ahead, "
            "along the lane (CriMe HW)",
            aggregation="min",
        ),
        Metric(
            CRIME_TTB_S,
            _crime_ttb_s,
            title="Time to brake (CriMe)",
            unit="s",
            text="Seconds left before the ego must brake at full to avoid a "
            "collision (CriMe TTB)",
            aggregation="min",
        ),
        Metric(
            CRIME_TTCE_S,
            _crime_ttce_s,
            title="Time to closest encounter (CriMe)",
            unit="s",
            text="Seconds until the ego and a road user come closest, both "
            "going on as they go (CriMe TTCE)",
            aggregation="min",
        ),
        Metric(
            CRIME_DCE_M,
            _crime_dce_m,
            title="Distance of closest encounter (CriMe)",
            unit="m",
            text="Metres between the ego and a road user when they come "
            "closest; 0 when they would collide (CriMe DCE)",
            aggregation="min",
        ),
        Metric(
            CRIME_A_LONG_REQ_MPS2,
            _crime_a_long_req_mps2,
            title="Required deceleration (CriMe)",
            unit="m/s^2",
            text="The longitudinal acceleration the ego needs not to hit the "
            "road user ahead, negative to brake; -inf when braking alone "
            "cannot avoid a contact within the prediction horizon "
            "(horizon_s, 3 s by default) (CriMe ALongReq)",
            aggregation="min",
        ),
        Metric(
            CRIME_A_LAT_REQ_MPS2,
            _crime_a_lat_req_mps2,
            title="Required lateral acceleration (CriMe)",
            unit="m/s^2",
            text="The lateral acceleration the ego needs to steer clear of a "
            "road user (CriMe ALatReq)",
            aggregation="max",
        ),
        Metric(
            CRIME_BTN,
            _crime_btn,
            title="Brake threat number (CriMe)",
            unit="ratio",
            text="The deceleration the ego needs not to hit the road user "
            "ahead over the most it can brake; above 1 braking alone no "
            "longer avoids it, inf when no braking avoids a contact within "
            "the prediction horizon (horizon_s, 3 s by default) (CriMe BTN)",
            aggregation="max",
        ),
        Metric(
            CRIME_STN,
            _crime_stn,
            title="Steer threat number (CriMe)",
            unit="ratio",
            text="The lateral acceleration the ego needs to steer clear of a "
            "road user over the most it can steer; above 1 steering alone no "
            "longer avoids it (CriMe STN)",
            aggregation="max",
        ),
        Metric(
            CRIME_TTR_S,
            _crime_ttr_s,
            title="Time to react (CriMe)",
            unit="s",
            text="Seconds the ego can go on as it goes before no evasive "
            "manoeuvre (braking, kick-down, steering) avoids the collision; 0 "
            "when none does even now (CriMe TTR)",
            aggregation="min",
        ),
        Metric(
            CRIME_WTTR_S,
            _crime_wttr_s,
            title="Worst-case time to react (CriMe)",
            unit="s",
            text="Seconds the ego can go on as it goes before no way it can go "
            "is free of collisions to the end of the horizon; 0 when none is "
            "left (CriMe WTTR, by CommonRoad-Reach)",
            aggregation="min",
        ),
        Metric(
            REACH_DRIVABLE_AREA_M2,
            _reach_drivable_area_m2,
            title="Drivable area (CommonRoad-Reach)",
            unit="m^2",
            text="Road the ego can still reach without a collision at the end "
            "of the horizon; 0 when a collision can no longer be avoided "
            "(CommonRoad-Reach)",
            aggregation="min",
        ),
        Metric(
            REACH_DRIVABLE_AREA_SUM_M2,
            _reach_drivable_area_sum_m2,
            title="Drivable area over the horizon (CommonRoad-Reach)",
            unit="m^2",
            text="The ego's drivable area summed over the horizon's steps "
            "(CriMe DA, by CommonRoad-Reach)",
            aggregation="min",
        ),
    ]
}

#: What a run a metric fails is a failure of
#: (:mod:`autoware_carla_scenario.failure`), when a
#: :class:`~autoware_carla_scenario.conditions.MetricCondition` on it is
#: registered without saying.  A metric not here reads more than one thing
#: (off the road *or* a collision) or no failure (a weight), and says nothing.
_FAILURE_CATEGORIES: dict[str, FailureCategory] = {
    COLLISION_ANY: FailureCategory.COLLISION,
    COLLISION_FRONT: FailureCategory.COLLISION,
    COLLISION_REAR: FailureCategory.COLLISION,
    COLLISION_LATERAL: FailureCategory.COLLISION,
    COLLISION_AT_FAULT: FailureCategory.COLLISION,
    OFFROAD: FailureCategory.LANE_KEEPING,
    MIN_DISTANCE_TO_LANE_BOUNDARY_M: FailureCategory.LANE_KEEPING,
    WRONG_LANE: FailureCategory.TRAFFIC_RULE,
    MIN_DISTANCE_TO_OBSTACLE_M: FailureCategory.SAFETY_MARGIN,
    LATERAL_REVERSALS: FailureCategory.STABILITY,
    LATERAL_WOBBLE_M: FailureCategory.STABILITY,
    STEERING_REVERSALS: FailureCategory.STABILITY,
    TURN_INDICATOR_CHANGES: FailureCategory.STABILITY,
    CRIME_TTC_S: FailureCategory.SAFETY_MARGIN,
    CRIME_THW_S: FailureCategory.SAFETY_MARGIN,
    CRIME_HW_M: FailureCategory.SAFETY_MARGIN,
    CRIME_TTB_S: FailureCategory.SAFETY_MARGIN,
    CRIME_TTCE_S: FailureCategory.SAFETY_MARGIN,
    CRIME_DCE_M: FailureCategory.SAFETY_MARGIN,
    CRIME_A_LONG_REQ_MPS2: FailureCategory.SAFETY_MARGIN,
    CRIME_A_LAT_REQ_MPS2: FailureCategory.SAFETY_MARGIN,
    CRIME_BTN: FailureCategory.SAFETY_MARGIN,
    CRIME_STN: FailureCategory.SAFETY_MARGIN,
    CRIME_TTR_S: FailureCategory.SAFETY_MARGIN,
    CRIME_WTTR_S: FailureCategory.SAFETY_MARGIN,
    REACH_DRIVABLE_AREA_M2: FailureCategory.SAFETY_MARGIN,
    REACH_DRIVABLE_AREA_SUM_M2: FailureCategory.SAFETY_MARGIN,
}


def is_metric(key: str) -> bool:
    """Whether *key* is a built-in metric's."""
    return key in BUILT_IN_METRICS


def read_metric(key: str, world: "carla.World") -> Optional[float]:
    """The built-in metric *key* in *world*.

    ``None`` when there is nothing to read, the key is not a metric's, or the
    metric raised.  A CommonRoad metric reads nothing here: it is computed
    from a run's recorded frames once the run is over
    (:mod:`autoware_carla_scenario.metric_batch`).
    """
    if key not in BUILT_IN_METRICS:
        return None
    metric = BUILT_IN_METRICS[key]
    try:
        return metric.read(world)
    except Exception:
        return None
