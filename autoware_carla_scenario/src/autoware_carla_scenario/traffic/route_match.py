"""Match a recorded track to the lanes it was driven on (#521).

A route's waypoints are points -- at the roadside, inside a junction, across
a lane change -- and the lane under each one on its own is a guess: the
nearest lanelet may be the opposite lane, a crossing branch of the junction,
or a lane the next point cannot be reached from.  :func:`match_track` reads
the whole track instead, sampled densely, and picks the sequence of lanelets
that is closest to it, runs the way it runs, and is drivable from one to the
next on the routing graph (following on, changing lane, or a short path):
a Viterbi search over a few candidate lanelets per sample.  What comes out
is one Lanelet2 pose where the track leaves each lanelet, and where it ends:
stops a vehicle's route through which keeps to the recorded lanes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    import lanelet2.core
    import lanelet2.routing

    from ..coordinate.poses import Lanelet2Pose

__all__ = ["TrackMatchError", "match_track"]

#: How far (m) from a sample a lanelet may be to be a candidate for it.
MATCH_RADIUS_M: float = 4.0
#: The most (deg) a candidate lane may run off the track's heading.
MATCH_HEADING_DEG: float = 60.0
#: Candidates kept per sample, the nearest first.
_CANDIDATES = 6
#: Lateral distance (m) that costs as much as one unit.
_LATERAL_SCALE_M = 1.0
#: Heading error (rad) that costs as much as one unit.
_HEADING_SCALE_RAD = 0.5
#: What a transition costs, by how it is driven.
_FOLLOW_COST = 0.0
_LANE_CHANGE_COST = 2.0
_PATH_COST = 1.0
#: A detour longer than this many times the straight line (plus the slack
#: below) is not how the track went.
_DETOUR_FACTOR = 3.0
_DETOUR_SLACK_M = 30.0


class TrackMatchError(ValueError):
    """No drivable sequence of lanes follows the track."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class _Candidate:
    """A lanelet a sample may be on: where along it, and at what cost."""

    lanelet: "lanelet2.core.ConstLanelet"
    s: float
    cost: float

    def __init__(
        self, lanelet: "lanelet2.core.ConstLanelet", s: float, cost: float
    ) -> None:
        self.lanelet = lanelet
        self.s = s
        self.cost = cost


def match_track(
    samples: Sequence[tuple[float, float, float]],
    lanelet_map: "lanelet2.core.LaneletMap",
    routing_graph: "lanelet2.routing.RoutingGraph",
    *,
    radius: float = MATCH_RADIUS_M,
) -> list["Lanelet2Pose"]:
    """The stops of the lanes *samples* run along: ``(x, y, heading)`` in the
    Lanelet2 map frame (m, rad), in driving order.  The first stop is where
    the track starts, on the lane it starts on; then one where it leaves each
    lanelet, and the last where it ends.

    A sample with no lane near it, or none running its way (a point off the
    road), is left out.

    Raises:
        TrackMatchError: if no sample is near a lane, or no drivable
            sequence of lanes joins them.
    """
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415

    layers: list[list[_Candidate]] = []
    kept: list[tuple[float, float, float]] = []
    for x, y, heading in samples:
        found = _candidates(lanelet_map, x, y, heading, radius)
        if found:
            layers.append(found)
            kept.append((x, y, heading))
    if not layers:
        raise TrackMatchError("no point of the track is on a lane running its way")
    lengths: dict[int, float] = {}
    paths: dict[tuple[int, int], Optional[float]] = {}
    cost = [c.cost for c in layers[0]]
    back: list[list[int]] = []
    for i_prev, (prev, layer) in enumerate(zip(layers, layers[1:])):
        row: list[float] = []
        links: list[int] = []
        for cand in layer:
            best = math.inf
            best_k = -1
            for k, before in enumerate(prev):
                if cost[k] == math.inf:
                    continue
                step = _transition(
                    before, cand, routing_graph, lengths, paths, kept, i_prev
                )
                total = cost[k] + step
                if total < best:
                    best, best_k = total, k
            row.append(best + cand.cost)
            links.append(best_k)
        cost = row
        back.append(links)
    end = min(range(len(cost)), key=lambda k: cost[k])
    if cost[end] == math.inf:
        raise TrackMatchError("no drivable sequence of lanes joins the track's points")
    chosen: list[_Candidate] = [layers[-1][end]]
    k = end
    for i in range(len(back) - 1, -1, -1):
        k = back[i][k]
        chosen.append(layers[i][k])
    chosen.reverse()
    first = chosen[0]
    stops: list[Lanelet2Pose] = [
        Lanelet2Pose(lanelet_id=int(first.lanelet.id), s=float(first.s))
    ]
    for i, cand in enumerate(chosen):
        last = i + 1 == len(chosen)
        if last or chosen[i + 1].lanelet.id != cand.lanelet.id:
            stops.append(Lanelet2Pose(lanelet_id=int(cand.lanelet.id), s=float(cand.s)))
    return stops


def _candidates(
    lanelet_map: "lanelet2.core.LaneletMap",
    x: float,
    y: float,
    heading: float,
    radius: float,
) -> list[_Candidate]:
    """The road lanelets within *radius* of ``(x, y)`` running its way."""
    import lanelet2.core  # noqa: PLC0415
    import lanelet2.geometry  # noqa: PLC0415

    from ..route.geometry import MapFeatures  # noqa: PLC0415

    point = lanelet2.core.BasicPoint2d(x, y)
    out: list[_Candidate] = []
    for _distance, lanelet in lanelet2.geometry.findWithin2d(
        lanelet_map.laneletLayer, point, radius
    ):
        if MapFeatures.is_walkable(lanelet) or not MapFeatures.is_road(lanelet):
            continue
        centre = lanelet2.geometry.to2D(lanelet.centerline)
        arc = lanelet2.geometry.toArcCoordinates(centre, point)
        length = lanelet2.geometry.length(centre)
        s = min(max(float(arc.length), 0.0), float(length))
        error = _angle(_heading_at(centre, s) - heading)
        if abs(error) > math.radians(MATCH_HEADING_DEG):
            continue
        cost = (float(arc.distance) / _LATERAL_SCALE_M) ** 2 + (
            error / _HEADING_SCALE_RAD
        ) ** 2
        out.append(_Candidate(lanelet, s, cost))
    out.sort(key=lambda c: c.cost)
    return out[:_CANDIDATES]


def _transition(
    before: _Candidate,
    after: _Candidate,
    graph: "lanelet2.routing.RoutingGraph",
    lengths: dict[int, float],
    paths: dict[tuple[int, int], Optional[float]],
    samples: Sequence[tuple[float, float, float]],
    index: int,
) -> float:
    """What going from *before* to *after* costs: ``inf`` if it cannot be driven."""
    a, b = before.lanelet, after.lanelet
    if a.id == b.id:
        return 0.0 if after.s >= before.s - 1.0 else 5.0
    if b.id in {ll.id for ll in graph.following(a)}:
        return _FOLLOW_COST
    for side in (graph.left(a), graph.right(a)):
        if side is not None and side.id == b.id:
            return _LANE_CHANGE_COST
    # Only what the two lanelets decide is cached -- the length of the
    # lanelets strictly between them, ``None`` with no path -- since *paths*
    # is shared by every pair of samples on them: where on *a* and *b* this
    # pair sits is added for this pair alone.
    key = (int(a.id), int(b.id))
    if key not in paths:
        path = graph.shortestPath(a, b)
        if path is None:
            paths[key] = None
        else:
            found: "lanelet2.routing.LaneletPath" = path
            lanelets = list(found)
            paths[key] = sum(_length(ll, lengths) for ll in lanelets[1:-1])
    middle = paths[key]
    if middle is None:
        return math.inf
    driven = middle + (_length(a, lengths) - before.s) + after.s
    x0, y0, _ = samples[index]
    x1, y1, _ = samples[index + 1]
    straight = math.hypot(x1 - x0, y1 - y0)
    if driven > _DETOUR_FACTOR * straight + _DETOUR_SLACK_M:
        return math.inf
    return _PATH_COST + max(0.0, driven - straight) / 10.0


def _length(lanelet: "lanelet2.core.ConstLanelet", lengths: dict[int, float]) -> float:
    import lanelet2.geometry  # noqa: PLC0415

    key = int(lanelet.id)
    if key not in lengths:
        lengths[key] = float(lanelet2.geometry.length2d(lanelet))
    return lengths[key]


def _heading_at(centre: "lanelet2.core.ConstLineString2d", s: float) -> float:
    """The heading (rad) of polyline *centre* at *s* metres along it."""
    points = [(p.x, p.y) for p in centre]
    travelled = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        step = math.hypot(x1 - x0, y1 - y0)
        if travelled + step >= s or (x1, y1) == points[-1]:
            return math.atan2(y1 - y0, x1 - x0)
        travelled += step
    return 0.0


def _angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
