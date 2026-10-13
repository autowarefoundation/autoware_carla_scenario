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


def plan_route(
    route: PlannedRoute,
    lanelet_map: LaneletMap,
    routing_graph: RoutingGraph,
) -> list[tuple[int, float]]:
    """The lanelets of *route*, in order, with the metres driven on each.

    Each leg (start to the first waypoint, ... to the goal) is the shortest
    path in *routing_graph*, lane changes allowed.

    Raises:
        RouteError: if the route has no goal, names a lanelet the map does
            not have, or a leg has no path.
    """
    if route.goal is None:
        raise RouteError("no goal: the route cannot be planned")
    goal: Lanelet2Pose = route.goal
    import lanelet2.geometry  # noqa: PLC0415

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

    # Metres on each: the start lanelet from the start pose on, the goal
    # lanelet up to the goal pose.
    last = len(path) - 1
    driven: list[float] = []
    for i, lanelet in enumerate(path):
        length = float(lanelet2.geometry.length2d(lanelet))
        begin = _clamp(route.start.s, length) if i == 0 else 0.0
        end = _clamp(goal.s, length) if i == last else length
        driven.append(max(0.0, end - begin))

    # A lane change: the vehicle drives along the two lanelets side by side,
    # so their length is split between them rather than counted twice.
    lateral = [0] * len(path)
    for i in range(last):
        following = {ll.id for ll in routing_graph.following(path[i])}
        if path[i + 1].id not in following:
            lateral[i] += 1
            lateral[i + 1] += 1
    return [
        (int(lanelet.id), metres / (1 + sides))
        for lanelet, metres, sides in zip(path, driven, lateral)
    ]


def _clamp(s: float, length: float) -> float:
    return min(max(float(s), 0.0), length)
