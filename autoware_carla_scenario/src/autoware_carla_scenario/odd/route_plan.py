"""Plan a scenario's route on the Lanelet2 map before it runs.

The route-planning half of :mod:`~autoware_carla_scenario.odd.route`, which
re-exports everything here under the same names: where a route starts, the
poses it passes through and where it ends (:class:`PlannedRoute`), and the
lanelets it drives on, with the metres driven on each (:func:`plan_route`),
as the shortest path in the map's routing graph, lane changes allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet, LaneletMap
    from lanelet2.routing import LaneletPath, RoutingGraph

    from ..coordinate.poses import Lanelet2Pose
    from ..scenario_base import BaseScenario

__all__ = [
    "PlannedRoute",
    "RouteError",
    "plan_route",
    "route_lanelets",
    "route_poses",
]


class RouteError(ValueError):
    """A route that cannot be planned: no goal, an unknown lanelet, no path."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class PlannedRoute:
    """Where a route starts, the lanelets it passes through, and where it ends.

    Args:
        start: The ego's initial pose.
        goal: Where it is routed to.  ``None`` cannot be planned.
        via: Poses the route passes through, in order (their lanelets).
        name: What the report calls it.
    """

    start: "Lanelet2Pose"
    goal: Optional["Lanelet2Pose"] = None
    via: Sequence["Lanelet2Pose"] = ()
    name: str = ""

    @classmethod
    def of_scenario(cls, scenario: "BaseScenario", name: str = "") -> "PlannedRoute":
        """The route a scenario's ego will be given: spawn, waypoints, goal.

        Read before ``setup()``, so a goal or waypoints the scenario only
        derives there are not known yet.

        Raises:
            RouteError: if the scenario has no spawn pose.
        """
        # Read once; a duck-typed scenario may not have the attribute at all.
        try:
            spawn: Optional[Lanelet2Pose] = scenario._spawn_pose
        except AttributeError:
            spawn = None
        if spawn is None:
            raise RouteError(
                f"{type(scenario).__name__} names no spawn pose to start a route at"
            )
        return cls(
            start=spawn,
            goal=scenario.goal_pose,
            via=tuple(scenario.waypoint_poses),
            name=name or type(scenario).__name__,
        )


def _lanelet(lanelet_map: LaneletMap, lanelet_id: int, what: str) -> ConstLanelet:
    layer = lanelet_map.laneletLayer
    if not layer.exists(int(lanelet_id)):
        raise RouteError(f"the {what} names lanelet {lanelet_id}, not in the map")
    # A Lanelet, held as the routing graph's ConstLanelets are.
    lanelet: ConstLanelet = layer[int(lanelet_id)]
    return lanelet


def route_lanelets(
    route: PlannedRoute,
    lanelet_map: LaneletMap,
    routing_graph: RoutingGraph,
    *,
    bidirectional: bool = False,
) -> list[ConstLanelet]:
    """The lanelets of *route*, in order, from its start to its goal.

    Each leg (start to the first waypoint, ... to the goal) is the shortest
    path in *routing_graph*, lane changes allowed.  A leg that ends behind
    where it begins on the same lanelet goes round a loop back to it, so the
    lanelet is in the route twice.  A *bidirectional* graph -- a pedestrian's
    -- has every lanelet both ways round: a leg may end on a lanelet
    inverted, and the first may start on one.

    Raises:
        RouteError: if the route has no goal, names a lanelet the map does
            not have, or a leg has no path.
    """
    path: list[ConstLanelet] = []
    for leg in _legs(route, lanelet_map, routing_graph, bidirectional, False):
        # Each leg starts on the lanelet the last one ended on.
        path.extend(leg.lanelets[1:] if path else leg.lanelets)
    return path


def plan_route(
    route: PlannedRoute,
    lanelet_map: LaneletMap,
    routing_graph: RoutingGraph,
) -> list[tuple[int, float]]:
    """The lanelets of *route*, in order, with the metres driven on each.

    The lanelets are :func:`route_lanelets`'s, and the metres those
    :func:`route_poses` drives on each: a lane change splits the stretch the
    two lanelets run side by side at the point it crosses.

    Raises:
        RouteError: if the route has no goal, names a lanelet the map does
            not have, or a leg has no path.
    """
    planned: list[tuple[int, float]] = []
    for leg in _legs(route, lanelet_map, routing_graph, False, False):
        spans = _driven(leg, _lengths(leg.lanelets), routing_graph)
        for i, (lanelet, (begin, end)) in enumerate(zip(leg.lanelets, spans)):
            if i == 0 and planned:
                # Where the last leg ended: the same lanelet, driven on.
                lanelet_id, metres = planned[-1]
                planned[-1] = (lanelet_id, metres + (end - begin))
            else:
                planned.append((int(lanelet.id), end - begin))
    return planned


def route_poses(
    route: PlannedRoute,
    lanelet_map: LaneletMap,
    routing_graph: RoutingGraph,
    step: float,
    *,
    bidirectional: bool = False,
    straight_legs: bool = False,
) -> list[Lanelet2Pose]:
    """Poses every *step* metres along *route*, from its start to its goal.

    The centrelines of each leg's lanelets, sampled, each leg ending on its
    waypoint and the last on the goal.  A lane change is driven across the
    middle of the two lanelets side by side -- unless the leg begins past it,
    when it crosses where the leg begins, or ends before it, when it crosses
    abreast of where the leg ends: the poses leave the first lanelet there and
    join the second abreast of it, so they never run back.  With *straight_legs*
    a leg *routing_graph* has no path for is its end pose alone, for whatever
    follows the poses to go straight there; without, it is a
    :class:`RouteError`.

    Raises:
        RouteError: as :func:`route_lanelets`.
    """
    poses: list[Lanelet2Pose] = []
    for leg in _legs(route, lanelet_map, routing_graph, bidirectional, straight_legs):
        if not leg.lanelets:
            _add(poses, leg.end)
            continue
        lengths = _lengths(leg.lanelets)
        spans = _driven(leg, lengths, routing_graph)
        last = len(leg.lanelets) - 1
        for i, lanelet in enumerate(leg.lanelets):
            begin, end = spans[i]
            if i == 0:
                # Where the leg begins, even when it changes lanes right there.
                _add(poses, _pose_at(lanelet, lengths[i], begin))
            u = begin
            while u < end:
                _add(poses, _pose_at(lanelet, lengths[i], u))
                u += step
            if i == last:
                _add(poses, _pose_at(lanelet, lengths[i], end))
    return poses


@dataclass
class _Leg:
    """From *begin* to *end*, on *lanelets* -- none for a leg to go straight."""

    begin: Lanelet2Pose
    end: Lanelet2Pose
    lanelets: list[ConstLanelet]


def _legs(
    route: PlannedRoute,
    lanelet_map: LaneletMap,
    routing_graph: RoutingGraph,
    bidirectional: bool,
    straight_legs: bool,
) -> list[_Leg]:
    """The legs of *route*: start to the first waypoint, ... to the goal."""
    if route.goal is None:
        raise RouteError("no goal: the route cannot be planned")
    goal: Lanelet2Pose = route.goal
    stops: list[tuple[Lanelet2Pose, str]] = [(route.start, "start")]
    for i, pose in enumerate(route.via, 1):
        stops.append((pose, f"waypoint {i}"))
    stops.append((goal, "goal"))

    here = _lanelet(lanelet_map, route.start.lanelet_id, "start")
    # Which way round the lanelet the next leg starts on is travelled: set by
    # the leg that ended on it, and free on a bidirectional graph until then.
    free = bidirectional
    legs: list[_Leg] = []
    for (begin, _), (end, what) in zip(stops, stops[1:]):
        there = _lanelet(lanelet_map, end.lanelet_id, what)
        starts = [here, here.invert()] if free else [here]
        ends = [there, there.invert()] if bidirectional else [there]
        found: list[ConstLanelet] = []
        if there.id != here.id:
            found = _shortest_leg(routing_graph, starts, ends)
        else:
            found = _same_lanelet_leg(routing_graph, starts, ends, begin, end)
        if not found:
            if not straight_legs:
                raise RouteError(
                    f"no route from lanelet {here.id} back round to itself "
                    f"({what} is behind)"
                    if there.id == here.id
                    else f"no route from lanelet {here.id} to lanelet {there.id} "
                    f"({what})"
                )
            legs.append(_Leg(begin, end, found))
            here, free = there, bidirectional
            continue
        legs.append(_Leg(begin, end, found))
        here, free = found[-1], False
    return legs


def _same_lanelet_leg(
    routing_graph: RoutingGraph,
    starts: Sequence[ConstLanelet],
    ends: Sequence[ConstLanelet],
    begin: Lanelet2Pose,
    end: Lanelet2Pose,
) -> list[ConstLanelet]:
    """A leg from *begin* to *end* on the same lanelet, or none.

    The lanelet alone when *end* is not behind *begin* the way one of *starts*
    is travelled; otherwise round a loop back to it: from the lanelet, or a
    lane it changes to as any other leg may, along the shortest path from a
    lanelet that follows that one to one of *ends*.  Compared by length, as
    :func:`_shortest_leg` compares.
    """
    for start in starts:
        length = _lengths([start])[0]
        at = _along(start, length, _clamp(begin.s, length))
        if _along(start, length, _clamp(end.s, length)) >= at:
            return [start]
    best: list[ConstLanelet] = []
    best_length = 0.0
    for start in starts:
        # Leave the lanelet ahead, never by lane changes alone: changing out
        # and back is beside the begin, not round to behind it.
        for lanes in _lane_changes_from(routing_graph, start):
            for following in routing_graph.following(lanes[-1]):
                rest = _shortest_leg(routing_graph, [following], ends)
                if not rest:
                    continue
                lanelets = lanes + rest
                length = sum(_lengths(lanelets))
                if not best or length < best_length:
                    best, best_length = lanelets, length
    return best


def _lane_changes_from(
    routing_graph: RoutingGraph, lanelet: ConstLanelet
) -> list[list[ConstLanelet]]:
    """*lanelet* alone, then each run of lane changes from it, left and right:
    ``[lanelet, left]``, ``[lanelet, left, left of that]``, ..."""
    runs: list[list[ConstLanelet]] = [[lanelet]]
    for right in (False, True):
        lanes: list[ConstLanelet] = [lanelet]
        while True:
            beside = (
                routing_graph.right(lanes[-1])
                if right
                else routing_graph.left(lanes[-1])
            )
            if beside is None or any(ll.id == beside.id for ll in lanes):
                break
            lane: ConstLanelet = beside
            lanes = lanes + [lane]
            runs.append(lanes)
    return runs


def _shortest_leg(
    routing_graph: RoutingGraph,
    starts: Sequence[ConstLanelet],
    ends: Sequence[ConstLanelet],
) -> list[ConstLanelet]:
    """The shortest of the paths from any of *starts* to any of *ends*, or none.

    Compared by length, which is the cost Lanelet2 routes by by default.
    """
    best: list[ConstLanelet] = []
    best_length = 0.0
    for start in starts:
        for end in ends:
            leg: Optional[LaneletPath] = routing_graph.shortestPath(start, end)
            if leg is None:
                continue
            found: LaneletPath = leg
            lanelets = list(found)
            length = sum(_lengths(lanelets))
            if not best or length < best_length:
                best, best_length = lanelets, length
    return best


def _spans(leg: _Leg, lengths: Sequence[float]) -> list[tuple[float, float]]:
    """``(begin, end)`` metres along each lanelet of *leg*, the way it is
    travelled: the first from the leg's begin on, the last up to its end."""
    last = len(leg.lanelets) - 1
    spans: list[tuple[float, float]] = []
    for i, lanelet in enumerate(leg.lanelets):
        length = lengths[i]
        begin = _along(lanelet, length, _clamp(leg.begin.s, length)) if i == 0 else 0.0
        end = (
            _along(lanelet, length, _clamp(leg.end.s, length)) if i == last else length
        )
        spans.append((begin, max(begin, end)))
    return spans


def _driven(
    leg: _Leg, lengths: Sequence[float], routing_graph: RoutingGraph
) -> list[tuple[float, float]]:
    """``(begin, end)`` metres driven along each lanelet of *leg*, the way it
    is travelled: its :func:`_spans`, cut where a lane change crosses.

    A lanelet left by a lane change is left at its middle and the lanelet
    changed to joined at its own, though not behind the point abreast of
    where the first is entered (the leg's begin, or where a lane change
    joined it).  Entered past its middle, or left where the leg ends before
    it, it is left there instead and the lanelet changed to joined abreast of
    that point.  Either way no further on than the leg ends on the lanelet
    changed to: a leg that ends there before the join crosses abreast of its
    end.
    """
    spans = _spans(leg, lengths)
    changes = _lane_changes(leg.lanelets, routing_graph)
    last = len(leg.lanelets) - 1
    # How far each lanelet may be driven: to its span's end, and on one left
    # by a lane change no further than abreast of how far the next may be --
    # back through a run of lane changes to where the leg ends.
    limits = [end for _begin, end in spans]
    for i in range(last - 1, -1, -1):
        if changes[i][1]:
            back = _abreast(
                leg.lanelets[i + 1],
                lengths[i + 1],
                limits[i + 1],
                leg.lanelets[i],
                lengths[i],
            )
            limits[i] = min(limits[i], back)
    driven: list[tuple[float, float]] = []
    joined = 0.0
    for i, lanelet in enumerate(leg.lanelets):
        begin, end = spans[i]
        before, after = changes[i]
        if before:
            begin = min(max(begin, joined), end)
        if after:
            beside = leg.lanelets[i + 1]
            middle = lengths[i] / 2.0
            if begin <= middle <= limits[i]:
                # Middle to middle, but never behind where the lanelet is
                # entered.
                joined = max(
                    lengths[i + 1] / 2.0,
                    _abreast(lanelet, lengths[i], begin, beside, lengths[i + 1]),
                )
                end = middle
            else:
                end = max(begin, min(limits[i], middle))
                joined = _abreast(lanelet, lengths[i], end, beside, lengths[i + 1])
            joined = min(joined, limits[i + 1])
        driven.append((begin, end))
    return driven


def _abreast(
    lanelet: ConstLanelet,
    length: float,
    u: float,
    beside: ConstLanelet,
    beside_length: float,
) -> float:
    """Metres along *beside*, the way it is travelled, abreast of *u* metres
    along *lanelet*: where that centreline point projects onto its centreline."""
    # Imported here, as _lengths imports.
    from ..trajectory.relative_lane import (  # noqa: PLC0415
        _arc,
        _centreline,
        _point_at,
    )

    # Measured on the lanelets as mapped, not inverted: ``s`` is.
    mapped = lanelet.invert() if lanelet.inverted() else lanelet
    x, y = _point_at(_centreline(mapped), _along(lanelet, length, u))
    mapped_beside = beside.invert() if beside.inverted() else beside
    s, _t = _arc(mapped_beside, x, y)
    return _along(beside, beside_length, min(s, beside_length))


def _lane_changes(
    path: Sequence[ConstLanelet], routing_graph: RoutingGraph
) -> list[tuple[bool, bool]]:
    """For each lanelet of *path*: whether it is reached by a lane change, and
    whether it is left by one, rather than to and from a lanelet that follows."""
    steps = [
        path[i + 1].id not in {ll.id for ll in routing_graph.following(path[i])}
        for i in range(len(path) - 1)
    ]
    return [
        (i > 0 and steps[i - 1], i < len(steps) and steps[i]) for i in range(len(path))
    ]


def _lengths(lanelets: Sequence[ConstLanelet]) -> list[float]:
    """The centreline length of each of *lanelets* (m)."""
    # Imported here: trajectory.model, which relative_lane imports, imports
    # the conditions, which import traffic.base, which names PlannedRoute.
    from ..trajectory.relative_lane import _length  # noqa: PLC0415

    return [_length(lanelet) for lanelet in lanelets]


def _along(lanelet: ConstLanelet, length: float, value: float) -> float:
    """``s`` <-> metres the way *lanelet* is travelled: the same unless it is
    travelled backwards (inverted), as a pedestrian may walk it."""
    return length - value if lanelet.inverted() else value


def _pose_at(lanelet: ConstLanelet, length: float, u: float) -> Lanelet2Pose:
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415

    return Lanelet2Pose(lanelet_id=int(lanelet.id), s=_along(lanelet, length, u))


def _add(poses: list[Lanelet2Pose], pose: Lanelet2Pose) -> None:
    """Append *pose* unless it is the last one again (where legs meet)."""
    if not poses or poses[-1] != pose:
        poses.append(pose)


def _clamp(s: float, length: float) -> float:
    return min(max(float(s), 0.0), length)
