"""What a route search asks of a Lanelet2 map: lanes, junctions, crosswalks.

Everything here works on a lanelet map and its routing graph passed in, in the
Lanelet2 map frame (right-handed, x East, y North), so it is tested on small
synthetic maps.  :class:`MapFeatures` caches the answers per lanelet: a search
asks the same lanelet the same question once per candidate route through it.

The definitions are the ones ``docs/logical_scenarios.md`` states:

* a **junction lanelet** carries a ``turn_direction`` tag;
* the **lanes beside** a lanelet are its routing-graph neighbours of the same
  direction, lane-changeable or not (``left()`` / ``adjacentLeft()``), counted
  outwards;
* the **opposite lane** abreast of a point is a lanelet running the other way
  (heading difference over :data:`OPPOSITE_MIN_TURN_DEG`) next to the
  outermost same-direction lane on either side, there: one sharing that
  lane's outer bound, else one within :data:`OPPOSITE_SEARCH_M` of it -- so a
  map of left-hand traffic (opposite lanes on the right) answers as one of
  right-hand traffic does.  Both lanes are followed along to the lanelet
  actually abreast of the point (:meth:`MapFeatures.abreast`), since the two
  directions of a road are often split into lanelets at different places; a
  lanelet's own opposite lane is the one abreast of its middle;
* the **members** of a junction are the paths through it that are reachable
  from the ego's way through the routing graph's ``conflicting`` relation,
  within :data:`JUNCTION_RADIUS_M`; a member is named by the first junction
  lanelet of its path (where it enters), and its **approach** is that
  lanelet's heading where it enters, against the ego's (see
  :func:`approach_of`);
* a **crosswalk** (a lanelet of subtype ``crosswalk``) is on the junction's
  **entry** leg when its centreline crosses the ego's path between
  :data:`CROSSWALK_WINDOW_M` before the junction and the middle of the ego's
  way through it, and on the **exit** leg from there to
  :data:`CROSSWALK_WINDOW_M` after it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

# autoware_lanelet2_extension_python must be imported before lanelet2.
from autoware_lanelet2_extension_python.projection import MGRSProjector as _  # noqa: F401
import lanelet2.core
import lanelet2.geometry

from ..trajectory.relative_lane import (
    _arc,
    _centreline,
    _end_heading,
    _heading_at,
    _length,
    _neighbour,
    _point_at,
    _start_heading,
    _straightest,
    _turn,
)

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet, ConstLineString3d, Lanelet, LaneletMap
    from lanelet2.routing import RoutingGraph

__all__ = [
    "APPROACH_OPPOSITE_MIN_DEG",
    "APPROACH_SAME_MAX_DEG",
    "CROSSWALK_WINDOW_M",
    "Crosswalk",
    "JUNCTION_RADIUS_M",
    "JunctionMember",
    "MapFeatures",
    "OPPOSITE_MIN_TURN_DEG",
    "OPPOSITE_SEARCH_M",
    "approach_of",
]

#: Entering within this many degrees of the ego's own heading is the ego's
#: approach; ignored.
APPROACH_SAME_MAX_DEG = 45.0
#: Entering at more than this many degrees from it is the opposite approach;
#: between the two is the left (heading clockwise of the ego's, i.e. coming
#: from its left) or the right.
APPROACH_OPPOSITE_MIN_DEG = 135.0
#: How far (m) from the ego's way through a junction another junction lanelet
#: may start and still be part of the same junction.
JUNCTION_RADIUS_M = 60.0
#: How far (m) before and after a junction a crosswalk still crosses its leg.
CROSSWALK_WINDOW_M = 10.0
#: How far (m) from the outermost lane's outer bound an unconnected
#: opposite-direction lanelet may be found.
OPPOSITE_SEARCH_M = 4.0
#: The least heading difference (deg) of a lanelet running the other way.
OPPOSITE_MIN_TURN_DEG = 150.0
#: The most same-direction lanes counted on one side.
_MAX_LANES = 12
#: The most lanelets :meth:`MapFeatures.abreast` walks along a lane.
_MAX_ABREAST_HOPS = 20
#: The most junction lanelets walked back to where another path enters.
_MAX_JUNCTION_HOPS = 4
#: Lanelet subtypes that are not for vehicles.
_NOT_ROADS = frozenset(
    {"crosswalk", "walkway", "stairs", "bicycle_lane", "road_shoulder", "parking"}
)


def approach_of(entry_heading: float, other_heading: float) -> str:
    """Where a lanelet entering a junction comes from, seen from the ego.

    ``same``, ``left``, ``right`` or ``opposite``, by the angle from the ego's
    heading where it enters to the other's (both radians, anticlockwise):
    within :data:`APPROACH_SAME_MAX_DEG` is the ego's own approach, beyond
    :data:`APPROACH_OPPOSITE_MIN_DEG` the opposite one.  Traffic from the
    ego's left drives towards its right, a heading clockwise of the ego's.
    """
    delta = math.degrees(
        (other_heading - entry_heading + math.pi) % (2.0 * math.pi) - math.pi
    )
    if abs(delta) <= APPROACH_SAME_MAX_DEG:
        return "same"
    if abs(delta) >= APPROACH_OPPOSITE_MIN_DEG:
        return "opposite"
    return "left" if delta < 0.0 else "right"


@dataclass(frozen=True)
class JunctionMember:
    """Another path through a junction the ego drives through.

    Attributes:
        lanelet: The path's first junction lanelet: where it enters.
        approach: ``left``, ``right`` or ``opposite`` (:func:`approach_of`),
            by that lanelet's heading.
        turn: The path's turn: ``straight``, or the one non-straight
            ``turn_direction`` of its lanelets (``mixed`` for both ways).
        conflicts: Whether any lanelet of the path conflicts with the ego's
            way through.
    """

    lanelet: ConstLanelet
    approach: str
    turn: str
    conflicts: bool


@dataclass(frozen=True)
class Crosswalk:
    """A crosswalk across one leg of a junction.

    Attributes:
        lanelet: The crosswalk lanelet.
        leg: ``entry`` or ``exit``.
        u: Where it crosses the ego's path, in metres from the junction entry
            (negative: before it).
        point: That crossing point ``(x, y)``.
        heading: The ego's path direction there (rad).
    """

    lanelet: ConstLanelet
    leg: str
    u: float
    point: tuple[float, float]
    heading: float


def _segment_intersection(
    p0: tuple[float, float],
    p1: tuple[float, float],
    q0: tuple[float, float],
    q1: tuple[float, float],
) -> Optional[tuple[float, float]]:
    """``(a, b)`` fractions where segments p and q cross, or ``None``."""
    rx, ry = p1[0] - p0[0], p1[1] - p0[1]
    sx, sy = q1[0] - q0[0], q1[1] - q0[1]
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return None
    qpx, qpy = q0[0] - p0[0], q0[1] - p0[1]
    a = (qpx * sy - qpy * sx) / denom
    b = (qpx * ry - qpy * rx) / denom
    if -1e-9 <= a <= 1.0 + 1e-9 and -1e-9 <= b <= 1.0 + 1e-9:
        return a, b
    return None


class MapFeatures:
    """The questions a route search asks of a map, answered once per lanelet."""

    map: LaneletMap
    graph: RoutingGraph
    _lengths: dict[int, float]
    _lanes: dict[tuple[int, str], int]
    _opposite: dict[int, Optional[tuple[ConstLanelet, str]]]
    _crosswalks: Optional[
        list[
            tuple[
                ConstLanelet,
                list[tuple[float, float]],
                tuple[float, float, float, float],
            ]
        ]
    ]
    _conflicting: dict[int, list[ConstLanelet]]

    def __init__(self, lanelet_map: LaneletMap, routing_graph: RoutingGraph) -> None:
        self.map = lanelet_map
        self.graph = routing_graph
        self._lengths = {}
        self._lanes = {}
        self._opposite = {}
        self._crosswalks = None
        self._conflicting = {}

    # -- lanelets ------------------------------------------------------------

    def lanelet(self, lanelet_id: int) -> Lanelet:
        """The lanelet with *lanelet_id*."""
        return self.map.laneletLayer[int(lanelet_id)]

    def length(self, lanelet: ConstLanelet) -> float:
        """Its centreline length (m)."""
        found = self._lengths.get(int(lanelet.id))
        if found is None:
            found = _length(lanelet)
            self._lengths[int(lanelet.id)] = found
        return found

    @staticmethod
    def is_junction(lanelet: ConstLanelet) -> bool:
        """Whether it is a junction lanelet (``turn_direction`` tagged)."""
        return "turn_direction" in lanelet.attributes

    @staticmethod
    def turn(lanelet: ConstLanelet) -> str:
        """Its ``turn_direction``, or ``""``."""
        if "turn_direction" not in lanelet.attributes:
            return ""
        return str(lanelet.attributes["turn_direction"])

    @staticmethod
    def is_road(lanelet: ConstLanelet) -> bool:
        """Whether a vehicle route may use it (not a crosswalk, walkway...)."""
        if "subtype" not in lanelet.attributes:
            return True
        return str(lanelet.attributes["subtype"]) not in _NOT_ROADS

    def following(self, lanelet: ConstLanelet) -> list[ConstLanelet]:
        """The lanelets following it, by id."""
        return sorted(self.graph.following(lanelet), key=lambda ll: int(ll.id))

    def previous(self, lanelet: ConstLanelet) -> list[ConstLanelet]:
        """The lanelets preceding it, by id."""
        return sorted(self.graph.previous(lanelet), key=lambda ll: int(ll.id))

    def straightest_previous(self, lanelet: ConstLanelet) -> Optional[ConstLanelet]:
        """The predecessor that turns least into it, or ``None``."""
        return _straightest(
            self.graph.previous(lanelet), _start_heading(lanelet), ahead=False
        )

    def straightest_following(self, lanelet: ConstLanelet) -> Optional[ConstLanelet]:
        """The successor that turns least out of it, or ``None``."""
        return _straightest(
            self.graph.following(lanelet), _end_heading(lanelet), ahead=True
        )

    # -- lanes beside --------------------------------------------------------

    def neighbour(self, lanelet: ConstLanelet, side: str) -> Optional[ConstLanelet]:
        """The same-direction lane on *side*, lane-changeable or not."""
        return _neighbour(self.graph, lanelet, side == "left")

    def lanes_beside(self, lanelet: ConstLanelet, side: str) -> int:
        """How many same-direction lanes lie on its *side*."""
        key = (int(lanelet.id), side)
        known = self._lanes.get(key)
        if known is not None:
            return known
        found, current, seen = 0, lanelet, {int(lanelet.id)}
        while found < _MAX_LANES:
            beside = self.neighbour(current, side)
            if beside is None or int(beside.id) in seen:
                break
            seen.add(int(beside.id))
            found += 1
            current = beside
        self._lanes[key] = found
        return found

    def outermost(self, lanelet: ConstLanelet, side: str) -> ConstLanelet:
        """The last same-direction lane on its *side* (itself if none)."""
        current: ConstLanelet = lanelet
        seen = {int(lanelet.id)}
        for _step in range(_MAX_LANES):
            beside = self.neighbour(current, side)
            if beside is None or int(beside.id) in seen:
                break
            seen.add(int(beside.id))
            current = beside
        return current

    # -- the other direction -------------------------------------------------

    def opposite(self, lanelet: ConstLanelet) -> Optional[tuple[ConstLanelet, str]]:
        """``(lanelet, side)`` of the lane running the other way beside it.

        Looked for abreast of *lanelet*'s middle (:meth:`opposite_abreast`);
        *side* is which side of *lanelet* the opposite road is on.  ``None``
        when it has none there (a one-way road).
        """
        key = int(lanelet.id)
        if key not in self._opposite:
            points = _centreline(lanelet)
            x, y = _point_at(points, self.length(lanelet) / 2.0)
            self._opposite[key] = self.opposite_abreast(lanelet, x, y)
        return self._opposite[key]

    def opposite_abreast(
        self, lanelet: ConstLanelet, x: float, y: float
    ) -> Optional[tuple[ConstLanelet, str]]:
        """``(lanelet, side)`` of the opposite-direction lane abreast of ``(x, y)``.

        ``(x, y)`` is a point on *lanelet*.  The outermost same-direction lane
        on each side is taken where it is abreast of the point, the lane
        beyond its outer bound looked for there, and the lanelet of that lane
        abreast of the point returned -- not whichever lanelet of it happens
        to touch the edge lane's middle, which on a road whose two directions
        are split into lanelets at different places is another one.
        """
        best: Optional[tuple[float, int, ConstLanelet, str]] = None
        for side in ("left", "right"):
            edge = self.abreast(self.outermost(lanelet, side), x, y)
            found = self._opposite_of_edge(edge, side, x, y)
            if found is None:
                continue
            hit: tuple[float, ConstLanelet] = found
            gap, beyond = hit
            other = self.abreast(beyond, x, y)
            key = (gap, int(other.id))
            if best is None or key < best[:2]:
                best = (gap, int(other.id), other, side)
        if best is None:
            return None
        return best[2], best[3]

    @staticmethod
    def gap_to(lanelet: ConstLanelet, x: float, y: float) -> float:
        """How far ``(x, y)`` is from *lanelet*'s centreline (m)."""
        points = _centreline(lanelet)
        best = math.inf
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            dx, dy = x1 - x0, y1 - y0
            squared = dx * dx + dy * dy
            f = 0.0 if squared < 1e-12 else ((x - x0) * dx + (y - y0) * dy) / squared
            f = min(max(f, 0.0), 1.0)
            best = min(best, math.hypot(x0 + f * dx - x, y0 + f * dy - y))
        return best

    def abreast(self, lanelet: ConstLanelet, x: float, y: float) -> ConstLanelet:
        """The lanelet of *lanelet*'s lane that ``(x, y)`` is abreast of.

        Walks the lane forwards and backwards through the routing graph --
        not into or out of a junction -- while that brings its centreline
        nearer the point.
        """
        current: ConstLanelet = lanelet
        best = self.gap_to(lanelet, x, y)
        junction = self.is_junction(lanelet)
        seen = {int(lanelet.id)}
        for _hop in range(_MAX_ABREAST_HOPS):
            nearer: Optional[ConstLanelet] = None
            for other in self.previous(current) + self.following(current):
                if int(other.id) in seen or self.is_junction(other) != junction:
                    continue
                gap = self.gap_to(other, x, y)
                if gap < best - 1e-6:
                    best, nearer = gap, other
            if nearer is None:
                break
            seen.add(int(nearer.id))
            current = nearer
        return current

    def _opposite_of_edge(
        self, edge: ConstLanelet, side: str, px: float, py: float
    ) -> Optional[tuple[float, ConstLanelet]]:
        """The lanelet running the other way beyond *edge*'s *side* bound.

        Looked for abreast of ``(px, py)``.
        """
        points = _centreline(edge)
        along, _t = _arc(edge, px, py)
        mx, my = _point_at(points, along)
        heading = _heading_at(points, along)
        bound = edge.leftBound if side == "left" else edge.rightBound
        candidates: list[tuple[float, ConstLanelet]] = []
        try:
            sharing = list(self.map.laneletLayer.findUsages(bound))
        except Exception:  # noqa: BLE001 -- a binding without the overload
            sharing = []
        for usage in sharing:
            other: ConstLanelet = usage
            if int(other.id) == int(edge.id) or not self.is_road(other):
                continue
            if self._runs_opposite(other, mx, my, heading):
                candidates.append((0.0, other))
        if not candidates:
            # Not connected through a shared bound: look just beyond it.
            bx, by = self.nearest_on(bound, mx, my)
            nx, ny = bx - mx, by - my
            norm = math.hypot(nx, ny) or 1.0
            probe = (bx + nx / norm * 1.0, by + ny / norm * 1.0)
            for distance, near in lanelet2.geometry.findNearest(
                self.map.laneletLayer, lanelet2.core.BasicPoint2d(*probe), 6
            ):
                nearby: ConstLanelet = near
                if float(distance) > OPPOSITE_SEARCH_M:
                    continue
                if int(nearby.id) == int(edge.id) or not self.is_road(nearby):
                    continue
                if self.is_junction(nearby) != self.is_junction(edge):
                    continue
                if self._runs_opposite(nearby, mx, my, heading):
                    candidates.append((float(distance), nearby))
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], int(item[1].id)))

    def _runs_opposite(
        self, other: ConstLanelet, x: float, y: float, heading: float
    ) -> bool:
        s, _t = _arc(other, x, y)
        other_heading = _heading_at(_centreline(other), s)
        return math.degrees(_turn(heading, other_heading)) >= OPPOSITE_MIN_TURN_DEG

    @staticmethod
    def nearest_on(
        linestring: ConstLineString3d, x: float, y: float
    ) -> tuple[float, float]:
        points = [(float(p.x), float(p.y)) for p in linestring]
        best = points[0]
        best_gap = math.inf
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            dx, dy = x1 - x0, y1 - y0
            squared = dx * dx + dy * dy
            f = 0.0 if squared < 1e-12 else ((x - x0) * dx + (y - y0) * dy) / squared
            f = min(max(f, 0.0), 1.0)
            px, py = x0 + f * dx, y0 + f * dy
            gap = math.hypot(px - x, py - y)
            if gap < best_gap:
                best, best_gap = (px, py), gap
        return best

    # -- regulatory elements -------------------------------------------------

    @staticmethod
    def has_stop_line(lanelet: ConstLanelet) -> bool:
        """Whether it owns a stop line (see ``has_stop_line``)."""
        from ..sweeper.constraints import HasStopLineConstraint  # noqa: PLC0415

        return HasStopLineConstraint().evaluate(lanelet)

    @staticmethod
    def has_traffic_light_stop_line(lanelet: ConstLanelet) -> bool:
        """Whether its stop line is a traffic light's."""
        from ..sweeper.constraints import (  # noqa: PLC0415
            HasTrafficLightStopLineConstraint,
        )

        return HasTrafficLightStopLineConstraint().evaluate(lanelet)

    @staticmethod
    def has_traffic_light(lanelet: ConstLanelet) -> bool:
        """Whether a traffic light regulates it."""
        for element in lanelet.regulatoryElements:
            attributes = element.attributes
            if (
                "subtype" in attributes
                and str(attributes["subtype"]) == "traffic_light"
            ):
                return True
            if "TrafficLight" in type(element).__name__:
                return True
        return False

    # -- junctions -----------------------------------------------------------

    def conflicting(self, lanelet: ConstLanelet) -> list[ConstLanelet]:
        """The lanelets the routing graph says conflict with it."""
        key = int(lanelet.id)
        if key not in self._conflicting:
            found: list[ConstLanelet] = []
            for item in self.graph.conflicting(lanelet):
                # A binding that wraps each in a ConstLaneletOrArea (an area
                # has no lanelet, and is dropped below); simple-lanelet2 hands
                # out the lanelets themselves, which have no `lanelet`.
                conflict = item
                if hasattr(item, "lanelet"):
                    inner = item.lanelet
                    if inner:
                        conflict = inner
                found.append(conflict)
            self._conflicting[key] = [ll for ll in found if hasattr(ll, "centerline")]
        return self._conflicting[key]

    def junction_members(self, way: Sequence[ConstLanelet]) -> list[JunctionMember]:
        """The other lanelets of the junction *way* (the ego's) drives through.

        By id; the ego's approach (``same``) left out.
        """
        own = {int(ll.id) for ll in way}
        centre_points = [p for ll in way for p in _centreline(ll)]
        cx = sum(p[0] for p in centre_points) / len(centre_points)
        cy = sum(p[1] for p in centre_points) / len(centre_points)
        entry_heading = _start_heading(way[0])
        direct = {int(o.id) for ll in way for o in self.conflicting(ll)}
        members: dict[int, ConstLanelet] = {}
        frontier = list(way)
        while frontier:
            current = frontier.pop()
            for other in self.conflicting(current):
                oid = int(other.id)
                if oid in own or oid in members or not self.is_junction(other):
                    continue
                sx, sy = _centreline(other)[0]
                if math.hypot(sx - cx, sy - cy) > JUNCTION_RADIUS_M:
                    continue
                members[oid] = other
                frontier.append(other)
        # A member part-way along another path through the junction is
        # classified, and placed, by where that path enters: its first
        # junction lanelet.
        paths: dict[tuple[int, str], JunctionMember] = {}
        for oid in sorted(members):
            path = self._path_into(members[oid])
            first = path[0]
            if int(first.id) in own:
                continue  # a branch off the ego's own way
            approach = approach_of(entry_heading, _start_heading(first))
            if approach == "same":
                continue
            turns = {self.turn(ll) for ll in path} - {"straight", ""}
            turn = (
                "straight"
                if not turns
                else (turns.pop() if len(turns) == 1 else "mixed")
            )
            key = (int(first.id), turn)
            conflicts = any(int(ll.id) in direct for ll in path)
            known = paths.get(key)
            if known is not None:
                conflicts = conflicts or known.conflicts
            paths[key] = JunctionMember(
                lanelet=first, approach=approach, turn=turn, conflicts=conflicts
            )
        return [paths[key] for key in sorted(paths)]

    def _path_into(self, lanelet: ConstLanelet) -> list[ConstLanelet]:
        """*lanelet* and the junction lanelets before it, from where they enter."""
        path: list[ConstLanelet] = [lanelet]
        for _hop in range(_MAX_JUNCTION_HOPS):
            before = [p for p in self.previous(path[0]) if self.is_junction(p)]
            if not before:
                break
            entering = _straightest(before, _start_heading(path[0]), ahead=False)
            if entering is None or any(int(entering.id) == int(p.id) for p in path):
                break
            enters: ConstLanelet = entering
            path.insert(0, enters)
        return path

    # -- crosswalks ----------------------------------------------------------

    def _all_crosswalks(
        self,
    ) -> list[
        tuple[
            ConstLanelet, list[tuple[float, float]], tuple[float, float, float, float]
        ]
    ]:
        """Every crosswalk of the map, its centreline, and its bounding box
        ``(min x, max x, min y, max y)``, which each junction tests first."""
        if self._crosswalks is None:
            crosswalks: list[
                tuple[
                    ConstLanelet,
                    list[tuple[float, float]],
                    tuple[float, float, float, float],
                ]
            ] = []
            for each in self.map.laneletLayer:
                ll: ConstLanelet = each
                if (
                    "subtype" in ll.attributes
                    and str(ll.attributes["subtype"]) == "crosswalk"
                ):
                    line = _centreline(ll)
                    xs = [p[0] for p in line]
                    ys = [p[1] for p in line]
                    box = (min(xs), max(xs), min(ys), max(ys))
                    crosswalks.append((ll, line, box))
            self._crosswalks = crosswalks
            return crosswalks
        return self._crosswalks

    def junction_crosswalks(
        self,
        way: Sequence[ConstLanelet],
        entry: Optional[ConstLanelet],
        exit_: Optional[ConstLanelet],
    ) -> list[Crosswalk]:
        """The crosswalks across the legs of the junction *way* drives through.

        *entry* is the lanelet the ego comes in on and *exit_* the one it
        leaves on (either may be ``None``); ordered by where they cross.
        """
        path: list[tuple[float, float]] = []
        offsets: list[float] = []
        if entry is not None:
            entering: ConstLanelet = entry
            points = _centreline(entering)
            length = self.length(entering)
            start = max(0.0, length - CROSSWALK_WINDOW_M)
            path.extend(_slice(points, start, length))
            offsets.append(start - length)
        else:
            offsets.append(0.0)
        for ll in way:
            points = _centreline(ll)
            path.extend(points if not path else points[1:])
        if exit_ is not None:
            leaving: ConstLanelet = exit_
            path.extend(_slice(_centreline(leaving), 0.0, CROSSWALK_WINDOW_M)[1:])
        if len(path) < 2:
            return []
        through = sum(self.length(ll) for ll in way)
        u0 = offsets[0]
        arc = [u0]
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            arc.append(arc[-1] + math.hypot(x1 - x0, y1 - y0))
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        lo_x, hi_x = min(xs) - 1.0, max(xs) + 1.0
        lo_y, hi_y = min(ys) - 1.0, max(ys) + 1.0
        found: dict[int, Crosswalk] = {}
        for crosswalk, line, (min_x, max_x, min_y, max_y) in self._all_crosswalks():
            if max_x < lo_x or min_x > hi_x or max_y < lo_y or min_y > hi_y:
                continue
            for i, (p0, p1) in enumerate(zip(path, path[1:])):
                for q0, q1 in zip(line, line[1:]):
                    hit = _segment_intersection(p0, p1, q0, q1)
                    if hit is None:
                        continue
                    u = arc[i] + hit[0] * (arc[i + 1] - arc[i])
                    if (
                        u < -CROSSWALK_WINDOW_M - 1e-6
                        or u > through + CROSSWALK_WINDOW_M
                    ):
                        continue
                    leg = "entry" if u <= through / 2.0 else "exit"
                    point = (
                        p0[0] + hit[0] * (p1[0] - p0[0]),
                        p0[1] + hit[0] * (p1[1] - p0[1]),
                    )
                    heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
                    cid = int(crosswalk.id)
                    if cid not in found:
                        found[cid] = Crosswalk(crosswalk, leg, u, point, heading)
        return sorted(found.values(), key=lambda c: (c.u, int(c.lanelet.id)))


def _slice(
    points: list[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    """The part of a polyline between arc lengths *start* and *end*."""
    out = [_point_at(points, start)]
    travelled = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        travelled += math.hypot(x1 - x0, y1 - y0)
        if start < travelled < end:
            out.append((x1, y1))
    out.append(_point_at(points, end))
    return out
