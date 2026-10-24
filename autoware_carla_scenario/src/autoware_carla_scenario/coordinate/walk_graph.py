"""Where a pedestrian can walk on the Lanelet2 map, and the shortest way there.

Lanelet2's own routing graph joins two lanelets only where they share a
boundary or an end, and that is not how pedestrian lanelets are drawn.  A
crosswalk lies *across* the walkway it starts and ends on, reaching a little
way onto it, and shares no point with it: Autoware's maps are drawn that way,
and so is every map roadgen writes (its crossings run ``CROSSWALK_OVERLAP``
onto the pavement each side).  A walkway is also one lanelet the length of a
block, so a crosswalk meets it part way along, where no lanelet ends.  And a
walkway tagged ``one_way: yes`` -- as roadgen tags every lanelet -- is
one-way for a pedestrian in Lanelet2's rules.  The Lanelet2 graph of such a
map has no way across a road at all.

:class:`WalkGraph` is the graph a pedestrian walks instead.  Its nodes are
points on the centrelines (a lanelet ``s``) of the walkable lanelets
(:meth:`~autoware_carla_scenario.route.geometry.MapFeatures.is_walkable`): each
lanelet's two ends, and every point where the end of another walkable lanelet reaches
it -- inside its outline, or within :data:`JOIN_TOLERANCE_M` of it.  Along a
lanelet, consecutive nodes are joined both ways, whatever its ``one_way``
tag: a pedestrian walks a pavement either way.  An end and the point it
reaches are joined by the step between them.  So a crosswalk end on a
walkway is a way on and off it, and two walkways that share an end, as a
pavement and the corner pavement after it do, are joined at it too.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..route.geometry import MapFeatures
from .poses import Lanelet2Pose

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet, LaneletMap

__all__ = [
    "JOIN_TOLERANCE_M",
    "WalkGraph",
    "walk_graph",
]

#: How far outside another walkable lanelet the end of one may stop and still
#: be a way onto it: a gap a pedestrian steps across.
JOIN_TOLERANCE_M: float = 0.5

#: Two nodes on one lanelet closer than this along it are the same node.
_SAME_STATION_M = 1e-3

#: The side of a cell of the grid the ends are matched to outlines on.
_CELL_M = 25.0


class _Walkable:
    """One walkable lanelet: its centreline, with ``s`` at each vertex, and its
    outline (the left bound, then the right one backwards)."""

    lanelet_id: int
    xs: List[float]
    ys: List[float]
    arc: List[float]
    outline: List[Tuple[float, float]]
    box: Tuple[float, float, float, float]

    def __init__(self, lanelet: "ConstLanelet") -> None:
        self.lanelet_id = int(lanelet.id)
        self.xs = []
        self.ys = []
        self.arc = []
        for point in lanelet.centerline:
            x = float(point.x)
            y = float(point.y)
            if self.arc:
                self.arc.append(
                    self.arc[-1] + math.hypot(x - self.xs[-1], y - self.ys[-1])
                )
            else:
                self.arc.append(0.0)
            self.xs.append(x)
            self.ys.append(y)
        self.outline = []
        for point in lanelet.leftBound:
            self.outline.append((float(point.x), float(point.y)))
        right: List[Tuple[float, float]] = []
        for point in lanelet.rightBound:
            right.append((float(point.x), float(point.y)))
        right.reverse()
        self.outline.extend(right)
        if not self.outline:
            for i in range(len(self.xs)):
                self.outline.append((self.xs[i], self.ys[i]))
        min_x = min(p[0] for p in self.outline)
        min_y = min(p[1] for p in self.outline)
        max_x = max(p[0] for p in self.outline)
        max_y = max(p[1] for p in self.outline)
        self.box = (min_x, min_y, max_x, max_y)

    @property
    def length(self) -> float:
        return self.arc[-1] if self.arc else 0.0

    def end(self, at_start: bool) -> Tuple[float, float]:
        """The first point of the centreline, or its last."""
        i = 0 if at_start else len(self.xs) - 1
        return (self.xs[i], self.ys[i])

    def at(self, s: float) -> Tuple[float, float]:
        """The point ``s`` metres along the centreline, clamped to it."""
        last = len(self.xs) - 1
        for i in range(last):
            if s <= self.arc[i + 1] or i == last - 1:
                span = self.arc[i + 1] - self.arc[i]
                f = 0.0 if span <= 0.0 else min(max((s - self.arc[i]) / span, 0.0), 1.0)
                return (
                    self.xs[i] + f * (self.xs[i + 1] - self.xs[i]),
                    self.ys[i] + f * (self.ys[i + 1] - self.ys[i]),
                )
        return (self.xs[0], self.ys[0])

    def project(self, x: float, y: float) -> float:
        """``s`` of the point of the centreline nearest ``(x, y)``."""
        best_s = 0.0
        best = math.inf
        for i in range(len(self.xs) - 1):
            ax = self.xs[i]
            ay = self.ys[i]
            dx = self.xs[i + 1] - ax
            dy = self.ys[i + 1] - ay
            span = dx * dx + dy * dy
            f = 0.0 if span <= 0.0 else ((x - ax) * dx + (y - ay) * dy) / span
            f = min(max(f, 0.0), 1.0)
            d = math.hypot(ax + f * dx - x, ay + f * dy - y)
            if d < best:
                best = d
                best_s = self.arc[i] + f * (self.arc[i + 1] - self.arc[i])
        return best_s

    def outside_by(self, x: float, y: float) -> float:
        """How far ``(x, y)`` is outside the outline: 0 inside it."""
        inside = False
        nearest = math.inf
        count = len(self.outline)
        for i in range(count):
            ax, ay = self.outline[i]
            bx, by = self.outline[(i + 1) % count]
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
            dx = bx - ax
            dy = by - ay
            span = dx * dx + dy * dy
            f = 0.0 if span <= 0.0 else ((x - ax) * dx + (y - ay) * dy) / span
            f = min(max(f, 0.0), 1.0)
            nearest = min(nearest, math.hypot(ax + f * dx - x, ay + f * dy - y))
        return 0.0 if inside else nearest


def _cells(
    box: Tuple[float, float, float, float], margin: float
) -> List[Tuple[int, int]]:
    """The grid cells a box, grown by *margin*, covers."""
    x0 = int(math.floor((box[0] - margin) / _CELL_M))
    y0 = int(math.floor((box[1] - margin) / _CELL_M))
    x1 = int(math.floor((box[2] + margin) / _CELL_M))
    y1 = int(math.floor((box[3] + margin) / _CELL_M))
    out: List[Tuple[int, int]] = []
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            out.append((ix, iy))
    return out


class WalkGraph:
    """The walkable lanelets of a map, joined where a pedestrian can step from
    one onto another (see the module docstring), and searched for the
    shortest walk between two poses on them.

    Args:
        lanelet_map: The map to read the walkable lanelets of.
    """

    _walkables: Dict[int, _Walkable]
    # Node i is _node_s[i] metres along lanelet _node_lanelet[i].
    _node_lanelet: List[int]
    _node_s: List[float]
    # The nodes on each lanelet, in order of s once built.
    _stations: Dict[int, List[int]]
    _edges: List[List[Tuple[int, float]]]

    def __init__(self, lanelet_map: "LaneletMap") -> None:
        self._walkables = {}
        self._node_lanelet = []
        self._node_s = []
        self._stations = {}
        self._edges = []
        for lanelet in lanelet_map.laneletLayer:
            if MapFeatures.is_walkable(lanelet):
                walkable = _Walkable(lanelet)
                if len(walkable.xs) >= 2:
                    self._walkables[walkable.lanelet_id] = walkable
        grid: Dict[Tuple[int, int], List[int]] = {}
        for lanelet_id, walkable in self._walkables.items():
            for cell in _cells(walkable.box, JOIN_TOLERANCE_M):
                if cell not in grid:
                    grid[cell] = []
                grid[cell].append(lanelet_id)
        for lanelet_id, walkable in self._walkables.items():
            for at_start in (True, False):
                x, y = walkable.end(at_start)
                end = self._node(lanelet_id, 0.0 if at_start else walkable.length)
                cell = (int(math.floor(x / _CELL_M)), int(math.floor(y / _CELL_M)))
                if cell not in grid:
                    continue
                for other_id in grid[cell]:
                    if other_id == lanelet_id:
                        continue
                    other = self._walkables[other_id]
                    if other.outside_by(x, y) > JOIN_TOLERANCE_M:
                        continue
                    s = other.project(x, y)
                    ox, oy = other.at(s)
                    self._join(end, self._node(other_id, s), math.hypot(ox - x, oy - y))
        for lanelet_id in self._stations:
            nodes = self._stations[lanelet_id]
            nodes.sort(key=lambda node: self._node_s[node])
            for i in range(len(nodes) - 1):
                a = nodes[i]
                b = nodes[i + 1]
                self._join(a, b, self._node_s[b] - self._node_s[a])

    def _node(self, lanelet_id: int, s: float) -> int:
        """The node *s* metres along *lanelet_id*, made if there is none."""
        if lanelet_id not in self._stations:
            self._stations[lanelet_id] = []
        nodes = self._stations[lanelet_id]
        for node in nodes:
            if abs(self._node_s[node] - s) < _SAME_STATION_M:
                return node
        node = len(self._node_s)
        self._node_lanelet.append(lanelet_id)
        self._node_s.append(s)
        self._edges.append([])
        nodes.append(node)
        return node

    def _join(self, a: int, b: int, cost: float) -> None:
        self._edges[a].append((b, cost))
        self._edges[b].append((a, cost))

    def walkable(self, lanelet_id: int) -> bool:
        """Whether *lanelet_id* is one of the lanelets the graph walks."""
        return lanelet_id in self._walkables

    def walk(
        self, start: Lanelet2Pose, goal: Lanelet2Pose, step: float
    ) -> Optional[List[Lanelet2Pose]]:
        """The shortest walk from *start* to *goal*: a pose every *step* metres
        along it, *start* and *goal* included.  ``None`` if either is off the
        walkable lanelets, or nothing joins them."""
        if start.lanelet_id not in self._walkables:
            return None
        if goal.lanelet_id not in self._walkables:
            return None
        nodes = self._shortest(start, goal)
        if nodes is None:
            return None
        found: List[int] = nodes
        stops: List[Tuple[int, float]] = [(start.lanelet_id, float(start.s))]
        for node in found:
            stops.append((self._node_lanelet[node], self._node_s[node]))
        stops.append((goal.lanelet_id, float(goal.s)))
        poses: List[Lanelet2Pose] = []
        for i in range(len(stops) - 1):
            lanelet_id, s = stops[i]
            next_id, next_s = stops[i + 1]
            poses.append(Lanelet2Pose(lanelet_id=lanelet_id, s=s))
            if next_id != lanelet_id:
                continue  # a step from one lanelet onto the next
            gap = next_s - s
            count = int(math.ceil(abs(gap) / step))
            for k in range(1, count):
                poses.append(Lanelet2Pose(lanelet_id=lanelet_id, s=s + gap * k / count))
        poses.append(Lanelet2Pose(lanelet_id=goal.lanelet_id, s=float(goal.s)))
        return poses

    def _shortest(self, start: Lanelet2Pose, goal: Lanelet2Pose) -> Optional[List[int]]:
        """The nodes of the shortest walk from *start* to *goal*, in order --
        none when *start* walks straight along its lanelet to *goal* -- or
        ``None`` when there is no walk."""
        count = len(self._node_s)
        # Two nodes of the query's own: the start, and the goal.
        source = count
        target = count + 1
        dist: List[float] = [math.inf] * (count + 2)
        back: List[int] = [-1] * (count + 2)
        dist[source] = 0.0
        frontier: List[Tuple[float, int]] = [(0.0, source)]
        while frontier:
            travelled, node = heapq.heappop(frontier)
            if travelled > dist[node]:
                continue
            if node == target:
                break
            steps: List[Tuple[int, float]] = []
            if node == source:
                for on in self._stations[start.lanelet_id]:
                    steps.append((on, abs(self._node_s[on] - start.s)))
                if start.lanelet_id == goal.lanelet_id:
                    steps.append((target, abs(goal.s - start.s)))
            else:
                steps.extend(self._edges[node])
                if self._node_lanelet[node] == goal.lanelet_id:
                    steps.append((target, abs(goal.s - self._node_s[node])))
            for there, cost in steps:
                total = travelled + cost
                if total < dist[there]:
                    dist[there] = total
                    back[there] = node
                    heapq.heappush(frontier, (total, there))
        if math.isinf(dist[target]):
            return None
        path: List[int] = []
        node = back[target]
        while node != source:
            path.append(node)
            node = back[node]
        path.reverse()
        return path


_cached_map: Optional["LaneletMap"] = None
_cached_graph: Optional[WalkGraph] = None


def walk_graph(lanelet_map: "LaneletMap") -> WalkGraph:
    """The :class:`WalkGraph` of *lanelet_map*, built on first use and kept
    until another map is asked for."""
    global _cached_map, _cached_graph
    if _cached_graph is not None and _cached_map is lanelet_map:
        graph: WalkGraph = _cached_graph
        return graph
    built = WalkGraph(lanelet_map)
    _cached_map = lanelet_map
    _cached_graph = built
    return built
