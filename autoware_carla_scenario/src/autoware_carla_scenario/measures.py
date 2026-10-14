"""Scenario measures: what a run sets up, read off the world under fixed keys.

A scenario stages road users -- a vehicle ahead in the next lane, a pedestrian
running out -- and its parameters say how: how far ahead, how fast.  A
**measure** reads such a quantity off the world while the run drives, under a
key the framework fixes (:data:`VEHICLE_AHEAD_GAP_M`, ...), so every scenario
speaks the same vocabulary of what it sets up, whoever wrote it.

Every :class:`~autoware_carla_scenario.BaseScenario` has the built-in measures
(:data:`BUILT_IN_MEASURES`), which read the world generically.  A scenario
that knows better -- which actor *is* its cut-in vehicle -- replaces one with
:meth:`~autoware_carla_scenario.BaseScenario.register_measure`, or adds one of
its own.

Two things are keyed by measures, and neither knows the ODD:

* a scenario's ``controls`` (its config): which of its parameters sets which
  measure, and what it can stage;
* an ODD's mapping of its taxonomy onto them
  (:func:`~autoware_carla_scenario.odd.scenario_measure`, ``measure:`` in a
  binding file), which is how coverage is measured and how a sweep draws the
  scenario's parameters from the ODD.

The vehicle ahead, along the ego's lane
---------------------------------------
:data:`VEHICLE_AHEAD_GAP_M` is the free road between the ego and the vehicle
ahead, measured along the ego's lane on the Lanelet2 map:

* the **ego's lane** is the lanelet the ego is on (the one under it whose
  centreline runs closest to its heading), the lanelets following it -- every
  branch, up to :data:`AHEAD_RANGE_M` ahead -- and the lanelet before it; its
  arc length ``s`` runs along their centrelines, continued from one to the
  next;
* a vehicle **counts** when it is on one of those lanelets or on a lanelet
  beside one (its left or right neighbour in the routing graph, a lane change
  allowed or not), and its origin is further along the lane than the ego's;
* the **gap** is ``s`` of its rear bumper less ``s`` of the ego's front bumper,
  each bumper the middle of the actor's bounding box's end along its own
  heading (:class:`Bumpers`), projected onto the ego's lane: onto the lanelet
  the vehicle is on, or the one it is beside, and those around it.  A vehicle
  in the lane beside that overlaps the ego reads a negative gap.

So the gap does not depend on how the lanelets beside each other are cut or
how long they are, and on a curve it is the distance along the road.  Where
there is no lane to measure along -- no Lanelet2 map loaded, or the ego on no
lanelet that runs its way -- the measure falls back to the **geometric**
definition: a vehicle ahead of the ego's origin within a lane of its centre
line (:data:`LANE_HALF_WIDTH_M`), the gap bumper to bumper along the ego's
heading.  Which definition read the gap is recorded
(:func:`vehicle_ahead_definitions`).

:data:`VEHICLE_AHEAD_RELATIVE_SPEED_KPH` is read in the same frame: the
lead's velocity along the lane's tangent at its rear bumper, less the ego's
along the tangent at its front bumper.

These are version 2 of the vehicle-ahead measures (:attr:`Measure.version`);
version 1 was centre to centre along the ego's heading, a vehicle counted
within three half-lanes of the ego's centre line.  Coverage keeps the two
apart (docs/odd.md, "Measure versions").

How the ego drives what a run sets up -- a collision, leaving the road -- is
not a measure but a **metric** (:mod:`autoware_carla_scenario.metrics`): an
outcome, which no ODD maps onto and a condition watches instead.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import lanelet2.core
import lanelet2.geometry

from .conditions.base import find_actor_in_list
from .constants import EGO_ROLE_NAME
from .coordinate.map_manager import MapManager
from .metrics import is_metric

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from .scenario_base import BaseScenario

__all__ = [
    "AHEAD_RANGE_M",
    "ALONG_LANE",
    "BUILT_IN_MEASURES",
    "Bumpers",
    "CROSSING_PEDESTRIAN_GAP_M",
    "CROSSING_PEDESTRIAN_SPEED_MS",
    "EGO_LANE_HEADING_DEG",
    "GEOMETRIC",
    "LANE_HALF_WIDTH_M",
    "Measure",
    "ON_LANELET_M",
    "PEDESTRIAN_LATERAL_M",
    "PEDESTRIAN_MOVING_MS",
    "VEHICLE_AHEAD_GAP_M",
    "VEHICLE_AHEAD_RELATIVE_SPEED_KPH",
    "VEHICLE_AHEAD_VERSION",
    "bumpers_of",
    "bumpers_of_box",
    "check_measure_key",
    "in_ego_frame",
    "measure_version",
    "measured_scenario",
    "read_measure",
    "set_measured_scenario",
    "vehicle_ahead_definition",
    "vehicle_ahead_definitions",
    "vehicle_ahead_gap_between",
]

logger = logging.getLogger(__name__)

#: Metres of free road, along the ego's lane, to the nearest vehicle ahead in
#: its lane or a lane beside it: its rear bumper less the ego's front bumper.
VEHICLE_AHEAD_GAP_M = "vehicle_ahead_gap_m"
#: That vehicle's speed along the ego's lane less the ego's, in km/h.
VEHICLE_AHEAD_RELATIVE_SPEED_KPH = "vehicle_ahead_relative_speed_kph"
#: Metres to the nearest pedestrian ahead that is moving.
CROSSING_PEDESTRIAN_GAP_M = "crossing_pedestrian_gap_m"
#: That pedestrian's speed, in m/s.
CROSSING_PEDESTRIAN_SPEED_MS = "crossing_pedestrian_speed_ms"

#: The version of the vehicle-ahead measures' definition (the module
#: docstring): 2, bumper to bumper along the ego's lane on the map.
VEHICLE_AHEAD_VERSION = 2
#: The vehicle ahead was read along the ego's lane on the Lanelet2 map.
ALONG_LANE = "along_lane"
#: The vehicle ahead was read along the ego's heading: no lane to read along.
GEOMETRIC = "geometric"

#: How far ahead a road user counts as ahead of the ego, in metres.
AHEAD_RANGE_M = 100.0
#: Half a lane's width, in metres.  Without a lane to measure along (the
#: geometric definition), a vehicle within three times this of the ego's
#: centre line is in its lane or a lane beside it.
LANE_HALF_WIDTH_M = 1.75
#: How far outside a lanelet's bounds a vehicle's origin may be and still be
#: on it, in metres: what the boundary's float error needs, no more.
ON_LANELET_M = 0.2
#: The largest angle, in degrees, between the ego's heading and the
#: centreline of a lanelet under it for that lanelet to be the ego's.
EGO_LANE_HEADING_DEG = 60.0
#: How far to either side of the ego's centre line a pedestrian ahead counts.
PEDESTRIAN_LATERAL_M = 10.0
#: The speed above which a pedestrian is moving rather than standing, in m/s.
PEDESTRIAN_MOVING_MS = 0.5

_KMH_PER_MS = 3.6
#: Lanelets nearest a point that are asked whether they hold it.
_CANDIDATES = 8
#: Metres of lane looked along beyond a lanelet, either way, to project a
#: point near its end onto: more than a vehicle's length.
_REACH_MARGIN_M = 15.0


@dataclass(frozen=True)
class Measure:
    """A quantity a run sets up, read off the world.

    Attributes:
        key: Its key, e.g. :data:`VEHICLE_AHEAD_GAP_M`.
        read: Reads it from the world; ``None`` when there is nothing to read.
        unit: The unit it is read in.
        text: A description.
        version: The version of its definition: raised whenever what it
            reads changes meaning, so coverage measured with one is not
            merged with coverage measured with another (docs/odd.md,
            "Measure versions").
    """

    key: str
    read: Callable[[carla.World], Optional[float]]
    unit: str = ""
    text: str = ""
    version: int = 1


@dataclass(frozen=True)
class Bumpers:
    """Where a vehicle's bumpers are: metres from its origin along its heading.

    Read off its bounding box (:func:`bumpers_of`), or, before it is spawned,
    off its blueprint (:func:`~autoware_carla_scenario.placement.vehicle_bumpers`).
    Zero for both reads centre to centre.

    Attributes:
        front: From its origin forward to its front bumper.
        rear: From its origin back to its rear bumper (a distance, positive).
    """

    front: float = 0.0
    rear: float = 0.0


def bumpers_of_box(box: "carla.BoundingBox") -> Bumpers:
    """The bumpers of a vehicle whose bounding box is *box* (in its frame)."""
    centre = float(box.location.x)
    half = float(box.extent.x)
    return Bumpers(front=centre + half, rear=half - centre)


def bumpers_of(actor: "carla.Actor") -> Bumpers:
    """*actor*'s bumpers, from its bounding box; zero when it has none."""
    # Not getattr() with a default, which Codon lacks (docs/typecheck.md).
    try:
        box = actor.bounding_box
    except AttributeError:
        return Bumpers()
    return bumpers_of_box(box)


# ---------------------------------------------------------------------------
# The ego's frame (pedestrians, and the geometric fallback)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Around:
    """A pedestrian, seen from the ego: metres ahead and to the side, speed."""

    actor_id: int
    ahead: float
    lateral: float
    #: Its own speed, in m/s.
    speed: float


@dataclass(frozen=True)
class _Lead:
    """The vehicle ahead: the gap to it, and how fast it pulls away."""

    gap: float
    #: Its velocity along the frame less the ego's, in m/s.
    closing: float


class _Cache:
    """What was read for which world at which frame: (id(world), frame), and
    the pedestrians around the ego and the vehicle ahead then.

    The world is held too: a world freed and another made may get the same
    ``id()``, and its first frames would be read off the old one's cache."""

    world: Optional["carla.World"] = None
    at: Optional[tuple[int, int]] = None
    around: Optional[list[_Around]] = None
    lead_at: Optional[tuple[int, int]] = None
    lead: Optional[_Lead] = None


_CACHE = _Cache()
#: Pedestrian actor id -> how far ahead of the ego it was when it set off.
_SET_OFF: dict[int, float] = {}
#: Definition -> how many frames the vehicle ahead was read with it.
_DEFINITIONS: dict[str, int] = {}


def _frame_of(world: "carla.World") -> Optional[int]:
    try:
        return int(world.get_snapshot().frame)
    except Exception:
        return None


def _frame_key(world: "carla.World") -> Optional[tuple[int, int]]:
    held = _CACHE.world
    if held is None or held is not world:
        _CACHE.world = world
        _CACHE.at, _CACHE.around = None, None
        _CACHE.lead_at, _CACHE.lead = None, None
    frame = _frame_of(world)
    if frame is None:
        return None
    return (id(world), int(frame))


def _around(world: "carla.World") -> Optional[list[_Around]]:
    """Every pedestrian, in the ego's frame; once per frame."""
    at = _frame_key(world)
    if at is not None and _CACHE.at is not None and at == _CACHE.at:
        return _CACHE.around
    found: Optional[list[_Around]]
    try:
        actors = list(world.get_actors())
        ego = find_actor_in_list(actors, EGO_ROLE_NAME)
        found = None if ego is None else _pedestrians_around(ego, actors)
    except Exception:
        found = None
    _CACHE.at, _CACHE.around = at, found
    return found


def in_ego_frame(
    ego: "carla.Transform", location: "carla.Location"
) -> tuple[float, float]:
    """*location* seen from an ego at *ego*: ``(ahead, lateral)`` in metres.

    ``ahead`` is along the ego's heading, ``lateral`` across it (positive to
    the ego's right, CARLA's left-handed frame), both from the ego's origin.
    The pedestrian measures read in it, and the vehicle-ahead measures where
    there is no lane to read along.
    """
    yaw = math.radians(ego.rotation.yaw)
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    dx = location.x - ego.location.x
    dy = location.y - ego.location.y
    return dx * forward_x + dy * forward_y, -dx * forward_y + dy * forward_x


def _pedestrians_around(
    ego: "carla.Actor", actors: "list[carla.Actor]"
) -> list[_Around]:
    transform = ego.get_transform()
    found: list[_Around] = []
    for actor in actors:
        if actor.id == ego.id or not actor.type_id.startswith("walker."):
            continue
        ahead, lateral = in_ego_frame(transform, actor.get_location())
        velocity = actor.get_velocity()
        found.append(
            _Around(
                actor_id=actor.id,
                ahead=ahead,
                lateral=lateral,
                speed=math.hypot(velocity.x, velocity.y),
            )
        )
    return found


# ---------------------------------------------------------------------------
# A vehicle, as the vehicle-ahead measures see it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Vehicle:
    """Where a vehicle is (CARLA world, yaw in radians), its bumpers, its
    velocity."""

    x: float
    y: float
    yaw: float
    front: float
    rear: float
    vx: float = 0.0
    vy: float = 0.0

    def bumper(self, ahead: float) -> tuple[float, float]:
        """The point *ahead* metres along its heading from its origin."""
        return (
            self.x + ahead * math.cos(self.yaw),
            self.y + ahead * math.sin(self.yaw),
        )


def _placed(transform: "carla.Transform", bumpers: Bumpers) -> _Vehicle:
    return _Vehicle(
        x=float(transform.location.x),
        y=float(transform.location.y),
        yaw=math.radians(float(transform.rotation.yaw)),
        front=bumpers.front,
        rear=bumpers.rear,
    )


def _moving(actor: "carla.Actor") -> _Vehicle:
    transform = actor.get_transform()
    bumpers = bumpers_of(actor)
    velocity = actor.get_velocity()
    return _Vehicle(
        x=float(transform.location.x),
        y=float(transform.location.y),
        yaw=math.radians(float(transform.rotation.yaw)),
        front=bumpers.front,
        rear=bumpers.rear,
        vx=float(velocity.x),
        vy=float(velocity.y),
    )


# ---------------------------------------------------------------------------
# The ego's lane on the Lanelet2 map
# ---------------------------------------------------------------------------


class _Line:
    """A lanelet's centreline in CARLA world coordinates, with its arc length."""

    xs: list[float]
    ys: list[float]
    s: list[float]

    def __init__(self, xs: list[float], ys: list[float]) -> None:
        self.xs = xs
        self.ys = ys
        s = [0.0]
        for i in range(1, len(xs)):
            s.append(s[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
        self.s = s

    @property
    def length(self) -> float:
        return self.s[-1]

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """The foot of ``(x, y)`` on the line, clamped to it: its arc length,
        the squared distance to it, and the line's heading there (radians,
        CARLA's yaw)."""
        best_s = 0.0
        best_d2 = math.inf
        best_yaw = 0.0
        for i in range(len(self.xs) - 1):
            ax, ay = self.xs[i], self.ys[i]
            dx, dy = self.xs[i + 1] - ax, self.ys[i + 1] - ay
            seg2 = dx * dx + dy * dy
            if seg2 <= 0.0:
                continue
            t = min(max(((x - ax) * dx + (y - ay) * dy) / seg2, 0.0), 1.0)
            fx, fy = ax + t * dx, ay + t * dy
            d2 = (x - fx) * (x - fx) + (y - fy) * (y - fy)
            if d2 < best_d2:
                best_d2 = d2
                best_s = self.s[i] + t * math.sqrt(seg2)
                best_yaw = math.atan2(dy, dx)
        if best_d2 == math.inf and self.xs:  # a single point
            best_d2 = (x - self.xs[0]) ** 2 + (y - self.ys[0]) ** 2
        return best_s, best_d2, best_yaw


class _Lines:
    """Centrelines by lanelet id, for the map they were read off.

    The map itself is held, not its ``id()``: a map freed and another loaded
    may get the same ``id()``, and would be read with the old one's lines."""

    lanelet_map: Optional[lanelet2.core.LaneletMap] = None
    by_id: dict[int, _Line] = {}


_LINES = _Lines()


def _forget_other_maps(lanelet_map: lanelet2.core.LaneletMap) -> None:
    """Drop what was read off a map other than *lanelet_map*."""
    held = _LINES.lanelet_map
    if held is None or held is not lanelet_map:
        _LINES.lanelet_map = lanelet_map
        _LINES.by_id = {}
        _FRAME_MEMO.key, _FRAME_MEMO.frame = None, None


def _line(lanelet_id: int) -> _Line:
    """The centreline of *lanelet_id* in CARLA world coordinates."""
    mm = MapManager.get_instance()
    lanelet_map = mm.lanelet_map
    _forget_other_maps(lanelet_map)
    if lanelet_id in _LINES.by_id:
        return _LINES.by_id[lanelet_id]
    offset_x, offset_y = mm.mgrs_offset
    xs: list[float] = []
    ys: list[float] = []
    for point in lanelet_map.laneletLayer[lanelet_id].centerline:
        xs.append(float(point.x) - offset_x)
        ys.append(-(float(point.y) - offset_y))
    line = _Line(xs, ys)
    _LINES.by_id[lanelet_id] = line
    return line


def _lanelets_at(x: float, y: float) -> list[int]:
    """The lanelets whose bounds hold the CARLA world point ``(x, y)``."""
    mm = MapManager.get_instance()
    offset_x, offset_y = mm.mgrs_offset
    point = lanelet2.core.BasicPoint2d(x + offset_x, -y + offset_y)
    found: list[int] = []
    for distance, lanelet in lanelet2.geometry.findNearest(
        mm.lanelet_map.laneletLayer, point, _CANDIDATES
    ):
        if float(distance) <= ON_LANELET_M:
            found.append(int(lanelet.id))
    return found


def _angle_between(a: float, b: float) -> float:
    """The angle between two headings, in radians, in [0, pi]."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _ego_lanelet(ego: _Vehicle) -> Optional[int]:
    """The lanelet the ego is on: of those under it, the one whose centreline
    runs closest to its heading (within :data:`EGO_LANE_HEADING_DEG`)."""
    best: Optional[int] = None
    best_angle = math.radians(EGO_LANE_HEADING_DEG)
    best_d2 = math.inf
    for lanelet_id in _lanelets_at(ego.x, ego.y):
        _s, d2, yaw = _line(lanelet_id).project(ego.x, ego.y)
        angle = _angle_between(yaw, ego.yaw)
        if angle < best_angle - 1e-9 or (
            abs(angle - best_angle) <= 1e-9 and d2 < best_d2
        ):
            best, best_angle, best_d2 = lanelet_id, angle, d2
    return best


def _beside(lanelet_id: int) -> list[int]:
    """The lanelets beside *lanelet_id*, either side, a lane change allowed
    across the line between or not (the routing graph's neighbours)."""
    mm = MapManager.get_instance()
    graph = mm.routing_graph
    lanelet = mm.lanelet_map.laneletLayer[lanelet_id]
    found: list[int] = []
    left = graph.left(lanelet)
    if left is not None:
        found.append(int(left.id))
    right = graph.right(lanelet)
    if right is not None:
        found.append(int(right.id))
    adjacent_left = graph.adjacentLeft(lanelet)
    if adjacent_left is not None:
        found.append(int(adjacent_left.id))
    adjacent_right = graph.adjacentRight(lanelet)
    if adjacent_right is not None:
        found.append(int(adjacent_right.id))
    return found


class _LaneFrame:
    """The ego's lane: its lanelet, those following and the one before it,
    each with the ``s`` it starts at; and the lanelets beside them."""

    ego_lanelet: int
    #: Lanelet of the lane -> ``s`` at its start (the ego's lanelet's is 0).
    start: dict[int, float]
    #: Lanelet of the lane -> the one before it along the lane.
    parent: dict[int, int]
    #: Lanelet of the lane -> those after it (several where the lane branches).
    children: dict[int, list[int]]
    #: Lanelet beside the lane -> the lanelet of the lane it is beside.
    beside: dict[int, int]

    def __init__(self, ego_lanelet: int, reach: float) -> None:
        """The lane from *ego_lanelet* on, until *reach* metres past its start."""
        mm = MapManager.get_instance()
        graph = mm.routing_graph
        layer = mm.lanelet_map.laneletLayer
        self.ego_lanelet = ego_lanelet
        self.start = {ego_lanelet: 0.0}
        self.parent = {}
        self.children = {ego_lanelet: []}
        queue = [ego_lanelet]
        index = 0
        while index < len(queue):
            current = queue[index]
            index += 1
            end = self.start[current] + _line(current).length
            if end >= reach:
                continue
            for following in graph.following(layer[current]):
                next_id = int(following.id)
                if next_id in self.start:
                    continue
                self.start[next_id] = end
                self.parent[next_id] = current
                self.children[current].append(next_id)
                self.children[next_id] = []
                queue.append(next_id)
        # Back along the lane, so a point just behind the ego's lanelet has an
        # s too (one way back, as the ego came).
        back = ego_lanelet
        behind = 0.0
        while behind < _REACH_MARGIN_M:
            previous = graph.previous(layer[back])
            if len(previous) == 0:
                break
            previous_id = int(previous[0].id)
            if previous_id in self.start:
                break
            length = _line(previous_id).length
            self.start[previous_id] = self.start[back] - length
            self.parent[back] = previous_id
            self.children[previous_id] = [back]
            behind += length
            back = previous_id
        self.beside = {}
        for lanelet_id in list(self.start):
            for other in _beside(lanelet_id):
                if other not in self.start and other not in self.beside:
                    self.beside[other] = lanelet_id

    def reference(self, x: float, y: float) -> Optional[int]:
        """The lanelet of the lane a vehicle at ``(x, y)`` is measured along:
        the one it is on, or the one beside the lanelet it is on; ``None``
        when it is on neither."""
        on_lane: Optional[int] = None
        on_lane_d2 = math.inf
        beside: Optional[int] = None
        for lanelet_id in _lanelets_at(x, y):
            if lanelet_id in self.start:
                _s, d2, _yaw = _line(lanelet_id).project(x, y)
                if d2 < on_lane_d2:
                    on_lane, on_lane_d2 = lanelet_id, d2
            elif beside is None and lanelet_id in self.beside:
                beside = self.beside[lanelet_id]
        if on_lane is not None:
            return on_lane
        return beside

    def _around(self, lanelet_id: int) -> list[int]:
        """*lanelet_id* and the lanelets of the lane within
        :data:`_REACH_MARGIN_M` before and after it."""
        found = [lanelet_id]
        first = self.start[lanelet_id]
        last = first + _line(lanelet_id).length
        current = lanelet_id
        while current in self.parent:
            current = self.parent[current]
            found.append(current)
            if self.start[current] < first - _REACH_MARGIN_M:
                break
        queue = list(self.children[lanelet_id])
        index = 0
        while index < len(queue):
            current = queue[index]
            index += 1
            found.append(current)
            if self.start[current] < last + _REACH_MARGIN_M:
                queue.extend(self.children[current])
        return found

    def project(self, reference: int, x: float, y: float) -> tuple[float, float]:
        """``(s, heading)`` of the point ``(x, y)`` on the lane near
        *reference*: the foot on the nearest centreline of *reference* and
        the lanelets around it, and the lane's heading there."""
        best_s = 0.0
        best_yaw = 0.0
        best_d2 = math.inf
        for lanelet_id in self._around(reference):
            s, d2, yaw = _line(lanelet_id).project(x, y)
            if d2 < best_d2:
                best_s, best_yaw, best_d2 = self.start[lanelet_id] + s, yaw, d2
        return best_s, best_yaw


class _FrameMemo:
    """The last lane frame built, and the ego pose and map it was built for."""

    #: The ego's x, y and yaw; the map is :data:`_LINES`'s.
    key: Optional[tuple[float, float, float]] = None
    frame: Optional[_LaneFrame] = None


_FRAME_MEMO = _FrameMemo()


def _lane_frame(ego: _Vehicle) -> Optional[_LaneFrame]:
    """The ego's lane; ``None`` with no Lanelet2 map, or the ego on no lanelet
    that runs its way."""
    mm = MapManager.get_instance()
    try:
        lanelet_map = mm.lanelet_map
    except RuntimeError:
        return None
    _forget_other_maps(lanelet_map)
    key = (ego.x, ego.y, ego.yaw)
    if _FRAME_MEMO.key is not None and _FRAME_MEMO.key == key:
        return _FRAME_MEMO.frame
    frame: Optional[_LaneFrame] = None
    ego_lanelet = _ego_lanelet(ego)
    if ego_lanelet is not None:
        found: int = ego_lanelet
        s, _d2, _yaw = _line(found).project(ego.x, ego.y)
        frame = _LaneFrame(found, s + ego.front + AHEAD_RANGE_M + _REACH_MARGIN_M)
    _FRAME_MEMO.key, _FRAME_MEMO.frame = key, frame
    return frame


# ---------------------------------------------------------------------------
# The vehicle ahead
# ---------------------------------------------------------------------------


class _EgoOnLane:
    """The ego on its lane: ``s`` of its origin and of its front bumper, and
    the lane's heading at its front bumper."""

    origin: float
    front: float
    yaw: float

    def __init__(self, frame: _LaneFrame, ego: _Vehicle) -> None:
        self.origin, _yaw = frame.project(frame.ego_lanelet, ego.x, ego.y)
        front_x, front_y = ego.bumper(ego.front)
        self.front, self.yaw = frame.project(frame.ego_lanelet, front_x, front_y)


def _ahead_along_lane(
    frame: _LaneFrame, ego: _Vehicle, on_lane: _EgoOnLane, other: _Vehicle
) -> Optional[_Lead]:
    reference = frame.reference(other.x, other.y)
    if reference is None:
        return None
    origin, _yaw = frame.project(reference, other.x, other.y)
    if origin <= on_lane.origin:
        return None
    rear_x, rear_y = other.bumper(-other.rear)
    rear, yaw = frame.project(reference, rear_x, rear_y)
    gap = rear - on_lane.front
    if gap > AHEAD_RANGE_M:
        return None
    along = other.vx * math.cos(yaw) + other.vy * math.sin(yaw)
    ego_along = ego.vx * math.cos(on_lane.yaw) + ego.vy * math.sin(on_lane.yaw)
    return _Lead(gap=gap, closing=along - ego_along)


def _ahead_geometrically(ego: _Vehicle, other: _Vehicle) -> Optional[_Lead]:
    forward_x, forward_y = math.cos(ego.yaw), math.sin(ego.yaw)
    dx, dy = other.x - ego.x, other.y - ego.y
    ahead = dx * forward_x + dy * forward_y
    lateral = -dx * forward_y + dy * forward_x
    if ahead <= 0.0 or abs(lateral) > 3 * LANE_HALF_WIDTH_M:
        return None
    rear_x, rear_y = other.bumper(-other.rear)
    rear = (rear_x - ego.x) * forward_x + (rear_y - ego.y) * forward_y
    gap = rear - ego.front
    if gap > AHEAD_RANGE_M:
        return None
    closing = (other.vx - ego.vx) * forward_x + (other.vy - ego.vy) * forward_y
    return _Lead(gap=gap, closing=closing)


def _nearest_ahead(
    ego: _Vehicle, others: list[_Vehicle]
) -> tuple[Optional[_Lead], str]:
    """The nearest of *others* ahead of *ego*, and the definition it was read
    with."""
    frame = _lane_frame(ego)
    nearest: Optional[_Lead] = None
    if frame is None:
        for other in others:
            lead = _ahead_geometrically(ego, other)
            if lead is not None and (nearest is None or lead.gap < nearest.gap):
                nearest = lead
        return nearest, GEOMETRIC
    on_lane = _EgoOnLane(frame, ego)
    for other in others:
        lead = _ahead_along_lane(frame, ego, on_lane, other)
        if lead is not None and (nearest is None or lead.gap < nearest.gap):
            nearest = lead
    return nearest, ALONG_LANE


def vehicle_ahead_definition(ego: "carla.Transform") -> str:
    """The definition the vehicle-ahead measures read with for an ego at
    *ego*: :data:`ALONG_LANE`, or :data:`GEOMETRIC` without a lane to read
    along (no Lanelet2 map, or the ego on no lanelet that runs its way)."""
    if _lane_frame(_placed(ego, Bumpers())) is None:
        return GEOMETRIC
    return ALONG_LANE


def vehicle_ahead_gap_between(
    ego: "carla.Transform",
    ego_bumpers: Bumpers,
    other: "carla.Transform",
    other_bumpers: Bumpers,
) -> Optional[float]:
    """What :data:`VEHICLE_AHEAD_GAP_M` reads for a vehicle at *other*, seen
    from an ego at *ego*, alone on the road: bumper to bumper along the ego's
    lane (the module docstring), or along its heading without one.

    ``None`` when the measure would not count it as ahead (not on the ego's
    lane or a lane beside it, behind the ego, or further than
    :data:`AHEAD_RANGE_M`).  The built-in measure reads the nearest such
    vehicle with this same code, so a vehicle placed where this reads *v*,
    alone ahead of the ego, is measured at *v*
    (:func:`~autoware_carla_scenario.placement.place_at_vehicle_ahead_gap`).
    """
    lead, _definition = _nearest_ahead(
        _placed(ego, ego_bumpers), [_placed(other, other_bumpers)]
    )
    if lead is None:
        return None
    return lead.gap


def _vehicle_ahead(world: "carla.World") -> Optional[_Lead]:
    """The vehicle ahead of the ego; once per frame."""
    at = _frame_key(world)
    if at is not None and _CACHE.lead_at is not None and at == _CACHE.lead_at:
        return _CACHE.lead
    lead: Optional[_Lead] = None
    try:
        actors = list(world.get_actors())
        ego_actor = find_actor_in_list(actors, EGO_ROLE_NAME)
        if ego_actor is not None:
            ego = _moving(ego_actor)
            reach = AHEAD_RANGE_M + 2 * _REACH_MARGIN_M
            others: list[_Vehicle] = []
            for actor in actors:
                if actor.id == ego_actor.id or not actor.type_id.startswith("vehicle."):
                    continue
                location = actor.get_location()
                # Along the lane is never shorter than the straight line.
                if math.hypot(location.x - ego.x, location.y - ego.y) > reach:
                    continue
                others.append(_moving(actor))
            lead, definition = _nearest_ahead(ego, others)
            _record_definition(definition)
    except Exception:
        logger.debug("the vehicle ahead could not be read", exc_info=True)
        lead = None
    _CACHE.lead_at, _CACHE.lead = at, lead
    return lead


def _record_definition(definition: str) -> None:
    if definition == GEOMETRIC and definition not in _DEFINITIONS:
        logger.warning(
            "%s: no lane to measure along (no Lanelet2 map, or the ego on no "
            "lanelet that runs its way); the vehicle ahead is read along the "
            "ego's heading instead",
            VEHICLE_AHEAD_GAP_M,
        )
    _DEFINITIONS[definition] = _DEFINITIONS.get(definition, 0) + 1


def vehicle_ahead_definitions() -> dict[str, int]:
    """How many frames of the run the vehicle ahead was read with each
    definition (:data:`ALONG_LANE`, :data:`GEOMETRIC`)."""
    return dict(_DEFINITIONS)


def _crossing_pedestrian(world: "carla.World") -> Optional[_Around]:
    around = _around(world)
    if around is None:
        return None
    candidates = [
        a
        for a in around
        if 0.0 < a.ahead <= AHEAD_RANGE_M
        and abs(a.lateral) <= PEDESTRIAN_LATERAL_M
        and a.speed > PEDESTRIAN_MOVING_MS
    ]
    return min(candidates, key=lambda a: a.ahead, default=None)


# ---------------------------------------------------------------------------
# The built-in measures
# ---------------------------------------------------------------------------


def _vehicle_ahead_gap_m(world: "carla.World") -> Optional[float]:
    ahead = _vehicle_ahead(world)
    if ahead is None:
        return None
    return ahead.gap


def _vehicle_ahead_relative_speed_kph(world: "carla.World") -> Optional[float]:
    ahead = _vehicle_ahead(world)
    if ahead is None:
        return None
    return ahead.closing * _KMH_PER_MS


def _crossing_pedestrian_gap_m(world: "carla.World") -> Optional[float]:
    pedestrian = _crossing_pedestrian(world)
    if pedestrian is None:
        return None
    # The gap it set off at, held while it crosses: that is what a scenario
    # stages, not every gap the ego then closes through.
    return _SET_OFF.setdefault(pedestrian.actor_id, pedestrian.ahead)


def _crossing_pedestrian_speed_ms(world: "carla.World") -> Optional[float]:
    pedestrian = _crossing_pedestrian(world)
    if pedestrian is None:
        return None
    return pedestrian.speed


#: The measures every scenario has, by key.
BUILT_IN_MEASURES: dict[str, Measure] = {
    m.key: m
    for m in (
        Measure(
            VEHICLE_AHEAD_GAP_M,
            _vehicle_ahead_gap_m,
            "m",
            "Bumper to bumper, along the ego's lane on the Lanelet2 map, to "
            "the nearest vehicle ahead on its lane or a lane beside it (a "
            "vehicle about to cut in is measured from before it changes "
            "lanes); along the ego's heading where there is no lane",
            VEHICLE_AHEAD_VERSION,
        ),
        Measure(
            VEHICLE_AHEAD_RELATIVE_SPEED_KPH,
            _vehicle_ahead_relative_speed_kph,
            "km/h",
            "That vehicle's speed along the ego's lane less the ego's; "
            "negative when the ego closes in",
            VEHICLE_AHEAD_VERSION,
        ),
        Measure(
            CROSSING_PEDESTRIAN_GAP_M,
            _crossing_pedestrian_gap_m,
            "m",
            "How far ahead the nearest pedestrian ahead that is moving was "
            "when it set off, held while it crosses: one standing at the kerb "
            "is not measured until it sets off",
        ),
        Measure(
            CROSSING_PEDESTRIAN_SPEED_MS,
            _crossing_pedestrian_speed_ms,
            "m/s",
            "That pedestrian's speed",
        ),
    )
}


def measure_version(key: str) -> int:
    """The version of measure *key*'s definition: a built-in measure's
    :attr:`Measure.version`, and 1 for any other key.

    A scenario that replaces a built-in measure measures the same thing
    better, so the key's version is the built-in's whoever reads it.
    """
    if key in BUILT_IN_MEASURES:
        return BUILT_IN_MEASURES[key].version
    return 1


def check_measure_key(key: str, where: str) -> None:
    """Refuse *key* as a measure's: empty, or a metric's.

    A metric (:mod:`autoware_carla_scenario.metrics`) reads how the ego drove,
    an outcome, which nothing maps onto or sets as a measure.  *where* names
    the caller in the message, e.g. ``"register_measure"``.

    Raises:
        ValueError: If *key* is empty or a metric's.
    """
    if not key:
        raise ValueError(f"{where}(): key must not be empty")
    if is_metric(key):
        raise ValueError(
            f"{where}({key}): {key} is a metric, how the ego drove, not a "
            "condition it drives in: an ODD cannot map onto it, nor a scenario "
            "measure it. Watch it with MetricCondition instead (docs/metrics.md)"
        )


# ---------------------------------------------------------------------------
# The running scenario
# ---------------------------------------------------------------------------

_MEASURED: dict[str, Optional["BaseScenario"]] = {"scenario": None}


def set_measured_scenario(scenario: Optional["BaseScenario"]) -> None:
    """Make *scenario*'s measures the ones :func:`read_measure` reads.

    The runner sets the scenario it runs, and clears it when the run ends.
    """
    _MEASURED["scenario"] = scenario
    _CACHE.world = None
    _CACHE.at, _CACHE.around = None, None
    _CACHE.lead_at, _CACHE.lead = None, None
    _SET_OFF.clear()
    _DEFINITIONS.clear()


def measured_scenario() -> Optional["BaseScenario"]:
    """The scenario whose measures are read, if one is running."""
    return _MEASURED["scenario"]


def read_measure(key: str, world: "carla.World") -> Optional[float]:
    """The running scenario's measure *key*, or the built-in one outside a run.

    ``None`` when there is nothing to read, the key is not measured, or the
    measure raised.
    """
    scenario = _MEASURED["scenario"]
    if scenario is not None:
        return scenario.measure(key, world)
    if key not in BUILT_IN_MEASURES:
        return None
    measure = BUILT_IN_MEASURES[key]
    try:
        return measure.read(world)
    except Exception:
        return None
