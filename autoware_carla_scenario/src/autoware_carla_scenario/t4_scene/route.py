"""The lanelets a road ahead runs along, as the closed loop's policies chose them.

A live T4Scene's route (:class:`~autoware_carla_scenario.t4_scene.live.LiveT4Scene`)
is the road ahead the policy is sent, walked along the map's lanelets.  The walk
here is the one OnePlanner's egodriver policy made of the same points
(``oneplanner.closed_loop.lanelet_map.LaneletMap.match_route``), so a T4Scene
driver plans on the route that policy planned on:

* the drivable lanelets (Autoware's ``ACCEPTABLE_LANE_SUBTYPES``), each
  Lanelet2 centreline resampled to 20 stations by arc length;
* the lane graph from a vehicle ``RoutingGraph`` by German rules -- successors
  straight on, and the same-direction neighbours a lane change away -- after
  bound ends within 0.3 m are welded into one point, as roadgen's junctions
  need for Lanelet2 to join them;
* a road-ahead point is on a lanelet within 2.5 m of its centreline, heading
  the same way.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["RouteGraph"]

#: Autoware's ``ACCEPTABLE_LANE_SUBTYPES``.
_LANE_SUBTYPES = frozenset(
    {
        "bicycle_lane",
        "crosswalk",
        "highway",
        "pedestrian_lane",
        "road",
        "road_shoulder",
        "walkway",
    }
)
#: Stations of a resampled centreline (T4's ``POINTS_PER_LANELET``).
_STATIONS = 20
#: Bound ends closer than this are one point (:func:`_weld_joins`).
_JOIN_TOLERANCE_M = 0.3
#: Road-ahead points in a row the lane graph cannot explain before the walk rejoins.
_REJOIN_AFTER_POINTS = 4
#: How far a road-ahead point may be from a lanelet's centreline and be on it.
_MAX_OFFSET_M = 2.5


class RouteGraph:
    """A Lanelet2 map's drivable lanelets and lane graph, for :meth:`match`."""

    def __init__(
        self,
        ids: Sequence[int],
        centres: npt.NDArray[np.float64],
        successors: Mapping[int, tuple[int, ...]],
        neighbours: Mapping[int, tuple[int, ...]],
    ) -> None:
        self.ids = list(ids)
        self._index = {lanelet_id: i for i, lanelet_id in enumerate(self.ids)}
        self._centres = np.asarray(centres, dtype=np.float64).reshape(-1, _STATIONS, 2)
        self._successors = successors
        self._neighbours = neighbours
        steps = np.diff(self._centres, axis=1)
        self._unit_steps = steps / np.maximum(
            np.linalg.norm(steps, axis=-1, keepdims=True), 1e-9
        )

    @classmethod
    def of(cls, lanelet_map: Any, owned: bool = True) -> "RouteGraph":
        """The graph of *lanelet_map*.  *owned*, for a map nothing else reads,
        moves its points: onto their ``local_x``/``local_y`` tags, as Autoware
        places them (a projection of their lat/lon lands within a micrometre,
        enough to break a tie the other way), and bound ends that nearly meet
        onto one."""
        import lanelet2
        from lanelet2.traffic_rules import Locations, Participants

        def xy(line: Any) -> npt.NDArray[np.float64]:
            return np.array([(point.x, point.y) for point in line], dtype=np.float64)

        if owned:
            _place_at_local_tags(lanelet_map)
        kept = [
            lanelet
            for lanelet in lanelet_map.laneletLayer
            if "subtype" in lanelet.attributes
            and str(lanelet.attributes["subtype"]) in _LANE_SUBTYPES
        ]
        centres = (
            np.stack([_resample(xy(lanelet.centerline)) for lanelet in kept])
            if kept
            else np.zeros((0, _STATIONS, 2))
        )
        # The lane graph last: welding moves a point by up to the tolerance.
        if owned:
            _weld_joins(lanelet_map)
        rules = lanelet2.traffic_rules.create(Locations.Germany, Participants.Vehicle)
        graph = lanelet2.routing.RoutingGraph(lanelet_map, rules)
        sides = (graph.left, graph.right, graph.adjacentLeft, graph.adjacentRight)
        successors = {
            int(lanelet.id): tuple(int(o.id) for o in graph.following(lanelet, False))
            for lanelet in kept
        }
        neighbours = {
            int(lanelet.id): tuple(
                int(o.id) for side in sides if (o := side(lanelet)) is not None
            )
            for lanelet in kept
        }
        return cls(
            [int(lanelet.id) for lanelet in kept], centres, successors, neighbours
        )

    def successors(self, lanelet_id: int) -> tuple[int, ...]:
        """The lanelets a vehicle on *lanelet_id* can drive straight into."""
        return self._successors.get(lanelet_id, ())

    def neighbours(self, lanelet_id: int) -> tuple[int, ...]:
        """Same-direction lanelets sharing a bound with *lanelet_id* -- a lane change away."""
        return self._neighbours.get(lanelet_id, ())

    def nearest(
        self,
        route_ids: Sequence[int],
        xy: npt.NDArray[np.float64],
        max_distance_m: float = math.inf,
    ) -> Optional[int]:
        """Index into *route_ids* of the lanelet whose centreline passes
        nearest *xy*; ``None`` when none passes within *max_distance_m*."""
        known = [
            (position, self._index[r])
            for position, r in enumerate(route_ids)
            if r in self._index
        ]
        if not known:
            return None
        distance, _, _ = _segment_distance(
            np.asarray(xy[:2], dtype=np.float64)[None],
            self._centres[[i for _, i in known]],
        )
        best = int(np.argmin(distance[0]))
        return known[best][0] if distance[0, best] < max_distance_m else None

    def match(
        self, route_xy: npt.NDArray[np.float64], start_id: Optional[int] = None
    ) -> list[int]:
        """The lanelets a polyline of ``(N, 2)`` map-frame points runs along, in order.

        Walked along the lane graph rather than matched point by point: at
        each point the walk stays on its lanelet or steps to a successor or a
        same-direction neighbour (a lane change), bridging one lanelet shorter
        than the points' spacing.  A junction's branches overlap at its mouth,
        so every choice -- the first lanelet, a branch, a rejoin -- goes to the
        lanelet whose graph explains the most of the points ahead
        (:meth:`_Walk.reach`).  A point nothing explains is passed over; after
        a run of them the walk rejoins there.  *start_id*, the lanelet the ego
        is on, wins a tie for the first lanelet.
        """
        route = np.asarray(route_xy, dtype=np.float64)[:, :2]
        pinned = [start_id] if start_id in self._index else []
        if len(route) < 2 or not self.ids:
            return pinned  # type: ignore[return-value]
        # Only lanelets the route can touch: a box around it, grown by the offset.
        low = route.min(axis=0) - _MAX_OFFSET_M
        high = route.max(axis=0) + _MAX_OFFSET_M
        near = np.flatnonzero(
            np.any(
                np.all((self._centres >= low) & (self._centres <= high), axis=-1),
                axis=-1,
            )
        )
        if not len(near):
            return pinned  # type: ignore[return-value]
        heading = _unit_tangents(route)
        distance, segment, _ = _segment_distance(route, self._centres[near])  # [R, L]
        step = self._unit_steps[near]
        rows = np.arange(len(near))
        walk = _Walk(self, route)
        for k, lanelet_index in enumerate(near):
            walk.know(self.ids[lanelet_index], distance[:, k])

        order: list[int] = []
        unexplained = 0
        for index in range(len(route)):
            aligned = step[rows, segment[index]] @ heading[index] > 0.5
            offset = np.where(aligned, distance[index], np.inf)
            ranked = np.argsort(offset)
            candidates = [
                self.ids[k] for k in near[ranked[offset[ranked] <= _MAX_OFFSET_M]]
            ]
            if not candidates:
                continue
            if not order:
                # Where the walk starts is decided like any branch: by how far
                # the route then keeps to it.  The pin wins a tie.
                choices = pinned + [c for c in candidates if c not in pinned]
                order.append(max(choices, key=lambda c: walk.reach(c, index)))  # type: ignore[arg-type]
                continue
            if order[-1] in candidates:
                unexplained = 0
                continue
            way_on = walk.step(order[-1], candidates, index)
            if way_on:
                # Only an immediate repeat is dropped: a route may come back to
                # a lanelet further on (a loop), and a bridge must stay.
                order.extend(lid for lid in way_on if lid != order[-1])
                unexplained = 0
                continue
            # Nothing in the graph explains a run of points: the route went
            # somewhere the map does not connect.  Pick it up again there.
            unexplained += 1
            if unexplained >= _REJOIN_AFTER_POINTS:
                order.append(max(candidates, key=lambda c: walk.reach(c, index)))
                unexplained = 0
        return order or pinned  # type: ignore[return-value]


class _Walk:
    """One :meth:`RouteGraph.match` call's view of its points, by index.

    Each lanelet's distance to every point is computed once per call, and how
    far the graph from a lanelet explains the points from one on is
    remembered: a walk asks the same of the same lanelets at every branch.
    """

    def __init__(self, graph: RouteGraph, route: npt.NDArray[np.float64]) -> None:
        self._graph = graph
        self._route = route
        self._off: dict[int, npt.NDArray[np.intp]] = {}
        self._reach: dict[tuple[int, int, int], int] = {}

    def know(self, lanelet_id: int, distance: npt.NDArray[np.float64]) -> None:
        """*lanelet_id*'s distance to every point, already computed."""
        self._off[lanelet_id] = np.flatnonzero(distance > _MAX_OFFSET_M)

    def kept(self, lanelet_id: int, start: int) -> int:
        """How many points from *start* on stay within reach of *lanelet_id*."""
        off = self._off.get(lanelet_id)
        if off is None:
            index = self._graph._index[lanelet_id]
            distance, _, _ = _segment_distance(
                self._route, self._graph._centres[[index]]
            )
            off = self._off[lanelet_id] = np.flatnonzero(distance[:, 0] > _MAX_OFFSET_M)
        after = off[np.searchsorted(off, start) :]
        return int(after[0]) - start if len(after) else len(self._route) - start

    def reach(self, lanelet_id: int, start: int, depth: int = 6) -> int:
        """How many points from *start* on *lanelet_id* and the graph beyond explain.

        A lanelet alone is the wrong measure: the ego's own may end a few
        metres on, and its successor then "keeps" the route longer though both
        are on it.  Following the graph a few lanelets deep scores the right
        branch by the whole route and the wrong one by where it leaves.
        """
        key = (lanelet_id, start, depth)
        if key not in self._reach:
            self._reach[key] = self._reach_uncached(lanelet_id, start, depth)
        return self._reach[key]

    def _reach_uncached(self, lanelet_id: int, start: int, depth: int) -> int:
        if lanelet_id not in self._graph._index:
            return 0
        kept = self.kept(lanelet_id, start)
        if kept >= len(self._route) - start or depth == 0 or kept == 0:
            return kept
        onward = self._graph.successors(lanelet_id) + self._graph.neighbours(lanelet_id)
        # Hand over a point early: the next lanelet starts where this one ends.
        rest = start + kept - 1
        beyond = max((self.reach(n, rest, depth - 1) for n in onward), default=0)
        # Never less than the lanelet alone: a branch with successors that
        # explain nothing must not lose to a dead end.
        return max(kept, kept - 1 + beyond)

    def step(self, current: int, candidates: list[int], start: int) -> list[int]:
        """The way on from *current* that the points from *start* keep to longest."""
        # Straight on, or a lane change: both are a step in Autoware's route.
        following = self._graph.successors(current) + self._graph.neighbours(current)
        steps = [[c] for c in candidates if c in following]
        if not steps:
            # A lanelet shorter than the points' spacing is skipped over, and a
            # lane change may land on the neighbour's successor.
            steps = [
                [bridge, c]
                for c in candidates
                for bridge in following
                if c in self._graph.successors(bridge)
                or c in self._graph.neighbours(bridge)
            ]
        if len(steps) <= 1:
            return steps[0] if steps else []
        return max(steps, key=lambda way: self.reach(way[-1], start))


def _resample(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """*points* at :data:`_STATIONS` stations of equal arc length; a degenerate
    polyline repeats its first point."""
    s = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    if len(points) < 2 or s[-1] <= 1e-9:
        return np.repeat(points[:1], _STATIONS, axis=0)
    at = np.linspace(0.0, s[-1], _STATIONS)
    return np.stack([np.interp(at, s, points[:, k]) for k in range(2)], axis=1)


def _segment_distance(
    points: npt.NDArray[np.float64], lines: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[Any], ...]:
    """Distance from each of ``[R, 2]`` *points* to each of ``[L, P, 2]``
    *lines*, the segment it falls on and the nearest point.  To the segments,
    not the stations: a long lanelet's stations can be ~7 m apart."""
    start, step = lines[:, :-1], np.diff(lines, axis=1)  # [L, S, 2]
    length2 = np.maximum((step**2).sum(-1), 1e-12)
    offset = points[:, None, None, :] - start[None]  # [R, L, S, 2]
    t = np.clip((offset * step[None]).sum(-1) / length2[None], 0.0, 1.0)
    gap = np.linalg.norm(offset - t[..., None] * step[None], axis=-1)  # [R, L, S]
    segment = gap.argmin(axis=-1)
    pick = segment[..., None]
    lanes = np.arange(lines.shape[0])[None, :]
    picked_t = np.take_along_axis(t, pick, axis=-1)[..., 0]
    nearest = start[lanes, segment] + picked_t[..., None] * step[lanes, segment]
    return np.take_along_axis(gap, pick, axis=-1)[..., 0], segment, nearest


def _unit_tangents(lines: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Unit tangents of ``[..., P, 2]`` polylines, by central differences."""
    tangent = np.gradient(lines, axis=-2)
    return tangent / np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)


def _place_at_local_tags(lanelet_map: Any) -> None:
    """Put each point tagged ``local_x``/``local_y`` there, and have the
    lanelets work their centrelines out again from the moved points."""
    for point in lanelet_map.pointLayer:
        if "local_x" in point.attributes and "local_y" in point.attributes:
            point.x = float(point.attributes["local_x"])
            point.y = float(point.attributes["local_y"])
    for lanelet in lanelet_map.laneletLayer:
        lanelet.resetCache()


def _weld_joins(lanelet_map: Any) -> None:
    """Make bound ends that coincide one point, as Lanelet2's ``follows`` needs.

    The routing graph joins lanelets whose bounds share their end points by
    identity; roadgen's maps give a junction's lanelets nodes of their own
    where they meet the road.  Bucketed by position, probing the adjacent
    buckets, so it stays linear in the map's size.
    """
    bounds = {
        bound.id
        for lanelet in lanelet_map.laneletLayer
        for bound in (lanelet.leftBound, lanelet.rightBound)
    }
    buckets: dict[tuple[int, int], list[Any]] = {}
    for line in lanelet_map.lineStringLayer:
        if line.id not in bounds:
            continue
        for end in (0, len(line) - 1):
            point = line[end]
            bx = round(point.x / _JOIN_TOLERANCE_M)
            by = round(point.y / _JOIN_TOLERANCE_M)
            near = (
                other
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for other in buckets.get((bx + dx, by + dy), ())
                if math.hypot(other.x - point.x, other.y - point.y) <= _JOIN_TOLERANCE_M
            )
            if (weld := next(near, None)) is not None and weld.id != line[0].id:
                line[end] = weld
            else:
                buckets.setdefault((bx, by), []).append(point)
