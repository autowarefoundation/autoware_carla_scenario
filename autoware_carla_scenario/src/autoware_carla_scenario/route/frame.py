"""A route match laid on its map: route s to lanelet poses and back.

:class:`RouteFrame` is the route's *reference line*: the centrelines of its
lanelets end to end, from ``start_s`` on the first to ``end_s`` on the last,
with route s running along it.  It answers the two questions everything else
asks -- where is route s (:meth:`RouteFrame.locate`, :meth:`RouteFrame.point`)
and how far along is this point (:meth:`RouteFrame.project`).  Projecting a
point measures its progress *along the route*, whichever lane beside the route
it is in: a vehicle that changed onto a parallel lane is abreast of the same
route s.
"""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING, Optional

from ..trajectory.relative_lane import (
    _centreline,
    _heading_at,
    _length,
    _point_at,
    advance_along_lane,
)
from .geometry import MapFeatures
from .model import RouteMatch

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet, Lanelet, LaneletMap
    from lanelet2.routing import RoutingGraph

__all__ = ["RouteFrame"]

#: How far (m) either side of the last answer :meth:`RouteFrame.project` looks
#: when it is given one.
PROJECT_WINDOW_M = 40.0
#: Farther than this (m) from the route within the window, a projection
#: searches the whole route instead (a few lane widths).
PROJECT_FALLBACK_M = 12.0


class RouteFrame:
    """A :class:`~.model.RouteMatch` on the lanelet map it was found on.

    Raises:
        ValueError: If the match names a lanelet the map does not have.
    """

    match: RouteMatch
    map: LaneletMap
    graph: RoutingGraph
    features: MapFeatures
    lanelets: list[ConstLanelet]
    _offsets: list[float]
    _xs: list[float]
    _ys: list[float]
    _arc: list[float]

    def __init__(
        self, match: RouteMatch, lanelet_map: LaneletMap, routing_graph: RoutingGraph
    ) -> None:
        self.match = match
        self.map = lanelet_map
        self.graph = routing_graph
        self.features = MapFeatures(lanelet_map, routing_graph)
        self.lanelets = []
        for lanelet_id in match.lanelet_ids:
            try:
                # A Lanelet, held as the routing graph's ConstLanelets are.
                lanelet: ConstLanelet = lanelet_map.laneletLayer[int(lanelet_id)]
                self.lanelets.append(lanelet)
            except Exception as exc:  # noqa: BLE001 -- lanelet2 raises its own
                raise ValueError(
                    f"the route names lanelet {lanelet_id}, which this map does "
                    "not have"
                ) from exc
        # Route s of each lanelet's start (its own s = 0).
        self._offsets = []
        travelled = -match.start_s
        for lanelet in self.lanelets:
            self._offsets.append(travelled)
            travelled += _length(lanelet)
        # The reference line, sampled at the lanelets' own points.
        xs: list[float] = []
        ys: list[float] = []
        arc: list[float] = []
        for index, lanelet in enumerate(self.lanelets):
            points = _centreline(lanelet)
            along = 0.0
            for i, (x, y) in enumerate(points):
                if i:
                    along += math.hypot(x - points[i - 1][0], y - points[i - 1][1])
                s = self._offsets[index] + along
                if arc and s <= arc[-1] + 1e-9:
                    continue
                xs.append(x)
                ys.append(y)
                arc.append(s)
        self._xs, self._ys, self._arc = xs, ys, arc

    @property
    def length(self) -> float:
        """The route's length (m)."""
        return self.match.length

    # -- route s to the map ----------------------------------------------------

    def locate(self, s: float) -> tuple[int, float]:
        """``(lanelet_id, s)`` of route s *s*.

        Before the route's start the lane is followed backwards from it, past
        its end forwards, the straightest way (as a
        :class:`~autoware_carla_scenario.trajectory.RelativeLanePose` is).

        Raises:
            ValueError: If that runs off the map.
        """
        if s < 0.0:
            return advance_along_lane(
                self.map, self.graph, self.match.lanelet_ids[0], self.match.start_s, s
            )
        if s > self.length:
            return advance_along_lane(
                self.map,
                self.graph,
                self.match.lanelet_ids[-1],
                self.match.end_s,
                s - self.length,
            )
        index = max(0, bisect.bisect_right(self._offsets, s) - 1)
        lanelet = self.lanelets[index]
        return int(lanelet.id), min(
            max(s - self._offsets[index], 0.0), _length(lanelet)
        )

    def point(self, s: float) -> tuple[float, float, float]:
        """``(x, y, heading)`` of route s *s* on the reference line, map frame."""
        lanelet_id, along = self.locate(s)
        points = _centreline(self.map.laneletLayer[lanelet_id])
        x, y = _point_at(points, along)
        return x, y, _heading_at(points, along)

    def lanelet_at(self, s: float) -> Lanelet:
        """The lanelet route s *s* is on."""
        return self.map.laneletLayer[self.locate(s)[0]]

    # -- the map to route s ----------------------------------------------------

    def project(
        self, x: float, y: float, near: Optional[float] = None
    ) -> tuple[float, float]:
        """``(s, distance)``: route s of the point nearest ``(x, y)``, and how far.

        Clamped to the route.  *near* first restricts the search to
        :data:`PROJECT_WINDOW_M` either side of it, so a route that passes the
        same place twice answers with the pass that is meant; when the point
        is more than :data:`PROJECT_FALLBACK_M` from the route there -- the
        entity was teleported, or *near* is stale -- the whole route is
        searched and the nearer answer taken.
        """
        if near is not None:
            windowed = self._nearest(x, y, near)
            if windowed[0] <= PROJECT_FALLBACK_M:
                return self._clamped(windowed)
            whole = self._nearest(x, y, None)
            return self._clamped(min(windowed, whole))
        return self._clamped(self._nearest(x, y, None))

    def _clamped(self, found: tuple[float, float]) -> tuple[float, float]:
        gap, s = found
        if gap is math.inf:
            return 0.0, math.inf
        return min(max(s, 0.0), self.length), gap

    def _nearest(
        self, x: float, y: float, near: Optional[float]
    ) -> tuple[float, float]:
        """``(distance, s)`` of the nearest reference-line point, within the window."""
        best = (math.inf, 0.0)
        arc = self._arc
        # The segments that reach into the window: from the one ending at or
        # past its start to the last one starting at or before its end (arc
        # does not decrease).
        first = 0
        stop = len(arc) - 1
        if near is not None:
            first = max(bisect.bisect_left(arc, near - PROJECT_WINDOW_M) - 1, 0)
            stop = min(bisect.bisect_right(arc, near + PROJECT_WINDOW_M), stop)
        for i in range(first, stop):
            x0, y0 = self._xs[i], self._ys[i]
            dx, dy = self._xs[i + 1] - x0, self._ys[i + 1] - y0
            squared = dx * dx + dy * dy
            f = 0.0 if squared < 1e-12 else ((x - x0) * dx + (y - y0) * dy) / squared
            f = min(max(f, 0.0), 1.0)
            gap = math.hypot(x0 + f * dx - x, y0 + f * dy - y)
            if gap < best[0]:
                best = (gap, arc[i] + f * (arc[i + 1] - arc[i]))
        return best

    def height(self, s: float) -> float:
        """The reference line's elevation at route s *s* (map frame, m)."""
        lanelet_id, along = self.locate(min(max(s, 0.0), self.length))
        points = [
            (float(p.x), float(p.y), float(p.z))
            for p in self.map.laneletLayer[lanelet_id].centerline
        ]
        travelled = 0.0
        for (x0, y0, z0), (x1, y1, z1) in zip(points, points[1:]):
            span = math.hypot(x1 - x0, y1 - y0)
            if span > 0.0 and travelled + span >= along:
                return z0 + (z1 - z0) * max(0.0, (along - travelled) / span)
            travelled += span
        return points[-1][2]

    def junction_way(
        self, index: int
    ) -> tuple[list[ConstLanelet], Optional[ConstLanelet], Optional[ConstLanelet]]:
        """Junction *index*'s lanelets, and the lanelets it is entered from and left onto.

        The entry (exit) is the route's own lanelet before (after) it, or the
        straightest one the map has when the junction starts (ends) the route.

        Raises:
            ValueError: If the route has no such junction.
        """
        segment = self.match.junction(index)
        ids = list(self.match.lanelet_ids)
        first = ids.index(segment.lanelet_ids[0])
        last = first + len(segment.lanelet_ids) - 1
        way = self.lanelets[first : last + 1]
        features = self.features
        # if statements, not conditional expressions: Codon types none whose
        # branches are a lanelet and an Optional of one.
        entry: Optional[ConstLanelet] = None
        if first > 0:
            entry = self.lanelets[first - 1]
        else:
            entry = features.straightest_previous(way[0])
        exit_: Optional[ConstLanelet] = None
        if last + 1 < len(self.lanelets):
            exit_ = self.lanelets[last + 1]
        else:
            exit_ = features.straightest_following(way[-1])
        return way, entry, exit_


def ego_placement(
    frame: RouteFrame, spawn_s: float, goal: bool, goal_margin: float
) -> tuple[tuple[int, float], Optional[tuple[int, float]]]:
    """``((lanelet, s) of the ego's spawn, (lanelet, s) of its goal or None)``.

    The spawn is *spawn_s* metres along the route, the goal *goal_margin*
    before its end.

    Raises:
        ValueError: If the spawn is not on the route, or not before the goal.
    """
    if spawn_s > frame.length:
        raise ValueError(
            f"ego_spawn_s {spawn_s:g} m is past the end of the route "
            f"({frame.length:.1f} m)"
        )
    spawn = frame.locate(spawn_s)
    # Both returns hand back an Optional local: Codon types a tuple by its
    # members, and a bare None is no Optional[tuple[int, float]].
    if not goal:
        no_goal: Optional[tuple[int, float]] = None
        return spawn, no_goal
    goal_s = max(0.0, frame.length - goal_margin)
    if goal_s <= spawn_s:
        raise ValueError(
            f"the goal ({goal_s:.1f} m along the route) is not ahead of the "
            f"spawn ({spawn_s:g} m)"
        )
    goal_at: Optional[tuple[int, float]] = frame.locate(goal_s)
    return spawn, goal_at
