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
) -> list[ConstLanelet]:
    """The lanelets of *route*, in order, from its start to its goal.

    Each leg (start to the first waypoint, ... to the goal) is the shortest
    path in *routing_graph*, lane changes allowed.

    Raises:
        RouteError: if the route has no goal, names a lanelet the map does
            not have, or a leg has no path.
    """
    if route.goal is None:
        raise RouteError("no goal: the route cannot be planned")
    goal: Lanelet2Pose = route.goal

    stops: list[tuple[ConstLanelet, str]] = [
        (_lanelet(lanelet_map, route.start.lanelet_id, "start"), "start")
    ]
    for i, pose in enumerate(route.via, 1):
        what = f"waypoint {i}"
        stops.append((_lanelet(lanelet_map, pose.lanelet_id, what), what))
    stops.append((_lanelet(lanelet_map, goal.lanelet_id, "goal"), "goal"))

    path: list[ConstLanelet] = [stops[0][0]]
    for there, what in stops[1:]:
        if there.id == path[-1].id:
            continue  # already on it
        leg: Optional[LaneletPath] = routing_graph.shortestPath(path[-1], there)
        if leg is None:
            raise RouteError(
                f"no route from lanelet {path[-1].id} to lanelet {there.id} ({what})"
            )
        found: LaneletPath = leg
        path.extend(list(found)[1:])
    return path


def plan_route(
    route: PlannedRoute,
    lanelet_map: LaneletMap,
    routing_graph: RoutingGraph,
) -> list[tuple[int, float]]:
    """The lanelets of *route*, in order, with the metres driven on each.

    The lanelets are :func:`route_lanelets`'s.

    Raises:
        RouteError: if the route has no goal, names a lanelet the map does
            not have, or a leg has no path.
    """
    path = route_lanelets(route, lanelet_map, routing_graph)
    spans = _driven_spans(route, path)
    changes = _lane_changes(path, routing_graph)
    # A lane change: the vehicle drives along the two lanelets side by side,
    # so their length is split between them rather than counted twice.
    planned: list[tuple[int, float]] = []
    for i, lanelet in enumerate(path):
        begin, end = spans[i]
        sides = (1 if i > 0 and changes[i - 1] else 0) + (
            1 if i < len(changes) and changes[i] else 0
        )
        planned.append((int(lanelet.id), (end - begin) / (1 + sides)))
    return planned


def route_poses(
    route: PlannedRoute,
    lanelet_map: LaneletMap,
    routing_graph: RoutingGraph,
    step: float,
) -> list[Lanelet2Pose]:
    """Poses every *step* metres along *route*, from its start to its goal.

    The centrelines of :func:`route_lanelets`, sampled, ending on the goal.  A
    lane change is driven across the middle of the two lanelets side by side:
    the poses leave the first at half its length and join the second at half
    of its own, rather than running the length of both.

    Raises:
        RouteError: as :func:`route_lanelets`.
    """
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415

    path = route_lanelets(route, lanelet_map, routing_graph)
    spans = _driven_spans(route, path)
    changes = _lane_changes(path, routing_graph)
    last = len(path) - 1
    poses: list[Lanelet2Pose] = []
    for i, lanelet in enumerate(path):
        begin, end = spans[i]
        half = _length(lanelet) / 2.0
        if i > 0 and changes[i - 1]:
            begin = max(begin, half)
        if i < last and changes[i]:
            end = min(end, half)
        s = begin
        while s < end:
            poses.append(Lanelet2Pose(lanelet_id=int(lanelet.id), s=s))
            s += step
        if i == last:
            poses.append(Lanelet2Pose(lanelet_id=int(lanelet.id), s=end))
    return poses


def _driven_spans(
    route: PlannedRoute, path: Sequence[ConstLanelet]
) -> list[tuple[float, float]]:
    """``(begin, end)`` metres along each lanelet of *path*: the start lanelet
    from the start pose on, the goal lanelet up to the goal pose."""
    goal_s = route.goal.s if route.goal is not None else 0.0
    last = len(path) - 1
    spans: list[tuple[float, float]] = []
    for i, lanelet in enumerate(path):
        length = _length(lanelet)
        begin = _clamp(route.start.s, length) if i == 0 else 0.0
        end = _clamp(goal_s, length) if i == last else length
        spans.append((begin, max(begin, end)))
    return spans


def _lane_changes(
    path: Sequence[ConstLanelet], routing_graph: RoutingGraph
) -> list[bool]:
    """Whether the step from each lanelet of *path* to the next is a lane
    change rather than a lanelet that follows it (one fewer than *path*)."""
    changes: list[bool] = []
    for i in range(len(path) - 1):
        following = {ll.id for ll in routing_graph.following(path[i])}
        changes.append(path[i + 1].id not in following)
    return changes


def _length(lanelet: ConstLanelet) -> float:
    import lanelet2.geometry  # noqa: PLC0415

    return float(lanelet2.geometry.length2d(lanelet))


def _clamp(s: float, length: float) -> float:
    return min(max(float(s), 0.0), length)
