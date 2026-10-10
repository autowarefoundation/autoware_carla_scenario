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
* the **opposite lane** of a lanelet is a lanelet running the other way
  (heading difference over :data:`OPPOSITE_MIN_TURN_DEG`) next to the
  outermost same-direction lane on either side: one sharing that lane's outer
  bound, else one within :data:`OPPOSITE_SEARCH_M` of it -- so a map of
  left-hand traffic (opposite lanes on the right) answers as one of
  right-hand traffic does;
* the **members** of a junction are the junction lanelets reachable from the
  ego's through the routing graph's ``conflicting`` relation, within
  :data:`JUNCTION_RADIUS_M` of the ego's way through it; each one's
  **approach** is its heading where it enters, against the ego's (see
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
from typing import Any, Optional, Sequence

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
    """Another lanelet of a junction the ego drives through.

    Attributes:
        lanelet: The junction lanelet.
        approach: ``left``, ``right`` or ``opposite`` (:func:`approach_of`).
        turn: Its ``turn_direction``.
        conflicts: Whether it conflicts with the ego's way through.
    """

    lanelet: Any
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

    lanelet: Any
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

    def __init__(self, lanelet_map: Any, routing_graph: Any) -> None:
        self.map = lanelet_map
        self.graph = routing_graph
        self._lengths: dict[int, float] = {}
        self._lanes: dict[tuple[int, str], int] = {}
        self._opposite: dict[int, Optional[tuple[Any, str]]] = {}
        self._crosswalks: Optional[list[tuple[Any, list[tuple[float, float]]]]] = None
        self._conflicting: dict[int, list[Any]] = {}

    # -- lanelets ------------------------------------------------------------

    def lanelet(self, lanelet_id: int) -> Any:
        """The lanelet with *lanelet_id*."""
        return self.map.laneletLayer[int(lanelet_id)]

    def length(self, lanelet: Any) -> float:
        """Its centreline length (m)."""
        found = self._lengths.get(int(lanelet.id))
        if found is None:
            found = _length(lanelet)
            self._lengths[int(lanelet.id)] = found
        return found

    @staticmethod
    def is_junction(lanelet: Any) -> bool:
        """Whether it is a junction lanelet (``turn_direction`` tagged)."""
        return "turn_direction" in lanelet.attributes

    @staticmethod
    def turn(lanelet: Any) -> str:
        """Its ``turn_direction``, or ``""``."""
        if "turn_direction" not in lanelet.attributes:
            return ""
        return str(lanelet.attributes["turn_direction"])

    @staticmethod
    def is_road(lanelet: Any) -> bool:
        """Whether a vehicle route may use it (not a crosswalk, walkway...)."""
        if "subtype" not in lanelet.attributes:
            return True
        return str(lanelet.attributes["subtype"]) not in _NOT_ROADS

    def following(self, lanelet: Any) -> list[Any]:
        """The lanelets following it, by id."""
        return sorted(self.graph.following(lanelet), key=lambda ll: int(ll.id))

    def previous(self, lanelet: Any) -> list[Any]:
        """The lanelets preceding it, by id."""
        return sorted(self.graph.previous(lanelet), key=lambda ll: int(ll.id))

    def straightest_previous(self, lanelet: Any) -> Any:
        """The predecessor that turns least into it, or ``None``."""
        return _straightest(
            self.graph.previous(lanelet), _start_heading(lanelet), ahead=False
        )

    def straightest_following(self, lanelet: Any) -> Any:
        """The successor that turns least out of it, or ``None``."""
        return _straightest(
            self.graph.following(lanelet), _end_heading(lanelet), ahead=True
        )

    # -- lanes beside --------------------------------------------------------

    def neighbour(self, lanelet: Any, side: str) -> Any:
        """The same-direction lane on *side*, lane-changeable or not."""
        return _neighbour(self.graph, lanelet, side == "left")

    def lanes_beside(self, lanelet: Any, side: str) -> int:
        """How many same-direction lanes lie on its *side*."""
        key = (int(lanelet.id), side)
        found = self._lanes.get(key)
        if found is None:
            found, current, seen = 0, lanelet, {int(lanelet.id)}
            while found < _MAX_LANES:
                current = self.neighbour(current, side)
                if current is None or int(current.id) in seen:
                    break
                seen.add(int(current.id))
                found += 1
            self._lanes[key] = found
        return found

    def outermost(self, lanelet: Any, side: str) -> Any:
        """The last same-direction lane on its *side* (itself if none)."""
        current, seen = lanelet, {int(lanelet.id)}
        for _step in range(_MAX_LANES):
            beside = self.neighbour(current, side)
            if beside is None or int(beside.id) in seen:
                break
            seen.add(int(beside.id))
            current = beside
        return current

    # -- the other direction -------------------------------------------------

    def opposite(self, lanelet: Any) -> Optional[tuple[Any, str]]:
        """``(lanelet, side)`` of the lane running the other way beside it.

        *side* is which side of *lanelet* the opposite road is on.  ``None``
        when it has none (a one-way road).
        """
        key = int(lanelet.id)
        if key not in self._opposite:
            self._opposite[key] = self._find_opposite(lanelet)
        return self._opposite[key]

    def _find_opposite(self, lanelet: Any) -> Optional[tuple[Any, str]]:
        best: Optional[tuple[float, int, Any, str]] = None
        for side in ("left", "right"):
            edge = self.outermost(lanelet, side)
            found = self._opposite_of_edge(edge, side)
            if found is None:
                continue
            gap, other = found
            key = (gap, int(other.id))
            if best is None or key < best[:2]:
                best = (gap, int(other.id), other, side)
        return None if best is None else (best[2], best[3])

    def _opposite_of_edge(self, edge: Any, side: str) -> Optional[tuple[float, Any]]:
        """The lanelet running the other way beyond *edge*'s *side* bound."""
        points = _centreline(edge)
        mid_s = self.length(edge) / 2.0
        mx, my = _point_at(points, mid_s)
        heading = _heading_at(points, mid_s)
        bound = edge.leftBound if side == "left" else edge.rightBound
        candidates: list[tuple[float, Any]] = []
        try:
            sharing = list(self.map.laneletLayer.findUsages(bound))
        except Exception:  # noqa: BLE001 -- a binding without the overload
            sharing = []
        for other in sharing:
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
            for distance, other in lanelet2.geometry.findNearest(
                self.map.laneletLayer, lanelet2.core.BasicPoint2d(*probe), 6
            ):
                if float(distance) > OPPOSITE_SEARCH_M:
                    continue
                if int(other.id) == int(edge.id) or not self.is_road(other):
                    continue
                if self.is_junction(other) != self.is_junction(edge):
                    continue
                if self._runs_opposite(other, mx, my, heading):
                    candidates.append((float(distance), other))
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], int(item[1].id)))

    def _runs_opposite(self, other: Any, x: float, y: float, heading: float) -> bool:
        s, _t = _arc(other, x, y)
        other_heading = _heading_at(_centreline(other), s)
        return math.degrees(_turn(heading, other_heading)) >= OPPOSITE_MIN_TURN_DEG

    @staticmethod
    def nearest_on(linestring: Any, x: float, y: float) -> tuple[float, float]:
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
    def has_stop_line(lanelet: Any) -> bool:
        """Whether it owns a stop line (see ``has_stop_line``)."""
        from ..sweeper.constraints import HasStopLineConstraint  # noqa: PLC0415

        return HasStopLineConstraint().evaluate(lanelet)

    @staticmethod
    def has_traffic_light_stop_line(lanelet: Any) -> bool:
        """Whether its stop line is a traffic light's."""
        from ..sweeper.constraints import (  # noqa: PLC0415
            HasTrafficLightStopLineConstraint,
        )

        return HasTrafficLightStopLineConstraint().evaluate(lanelet)

    @staticmethod
    def has_traffic_light(lanelet: Any) -> bool:
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

    def conflicting(self, lanelet: Any) -> list[Any]:
        """The lanelets the routing graph says conflict with it."""
        key = int(lanelet.id)
        if key not in self._conflicting:
            found = []
            for item in self.graph.conflicting(lanelet):
                found.append(getattr(item, "lanelet", None) or item)
            self._conflicting[key] = [ll for ll in found if hasattr(ll, "centerline")]
        return self._conflicting[key]

    def junction_members(self, way: Sequence[Any]) -> list[JunctionMember]:
        """The other lanelets of the junction *way* (the ego's) drives through.

        By id; the ego's approach (``same``) left out.
        """
        own = {int(ll.id) for ll in way}
        centre_points = [p for ll in way for p in _centreline(ll)]
        cx = sum(p[0] for p in centre_points) / len(centre_points)
        cy = sum(p[1] for p in centre_points) / len(centre_points)
        entry_heading = _start_heading(way[0])
        direct = {int(o.id) for ll in way for o in self.conflicting(ll)}
        members: dict[int, Any] = {}
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
        found: list[JunctionMember] = []
        for oid in sorted(members):
            other = members[oid]
            approach = approach_of(entry_heading, _start_heading(other))
            if approach == "same":
                continue
            found.append(
                JunctionMember(
                    lanelet=other,
                    approach=approach,
                    turn=self.turn(other),
                    conflicts=oid in direct,
                )
            )
        return found

    # -- crosswalks ----------------------------------------------------------

    def _all_crosswalks(self) -> list[tuple[Any, list[tuple[float, float]]]]:
        if self._crosswalks is None:
            self._crosswalks = [
                (ll, _centreline(ll))
                for ll in self.map.laneletLayer
                if "subtype" in ll.attributes
                and str(ll.attributes["subtype"]) == "crosswalk"
            ]
        return self._crosswalks

    def junction_crosswalks(
        self, way: Sequence[Any], entry: Any, exit_: Any
    ) -> list[Crosswalk]:
        """The crosswalks across the legs of the junction *way* drives through.

        *entry* is the lanelet the ego comes in on and *exit_* the one it
        leaves on (either may be ``None``); ordered by where they cross.
        """
        path: list[tuple[float, float]] = []
        offsets: list[float] = []
        if entry is not None:
            points = _centreline(entry)
            length = self.length(entry)
            start = max(0.0, length - CROSSWALK_WINDOW_M)
            path.extend(_slice(points, start, length))
            offsets.append(start - length)
        else:
            offsets.append(0.0)
        for ll in way:
            points = _centreline(ll)
            path.extend(points if not path else points[1:])
        if exit_ is not None:
            path.extend(_slice(_centreline(exit_), 0.0, CROSSWALK_WINDOW_M)[1:])
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
        for crosswalk, line in self._all_crosswalks():
            cxs = [p[0] for p in line]
            cys = [p[1] for p in line]
            if max(cxs) < lo_x or min(cxs) > hi_x or max(cys) < lo_y or min(cys) > hi_y:
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
