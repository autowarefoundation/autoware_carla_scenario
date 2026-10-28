"""The route the running scenario is about.

A logical scenario's poses and its ``route_progress`` conditions are measured
along one route -- the match its route search found on the map the run is on.
The scenario names it here once (:func:`set_scenario_route`) and every action
and condition built from the document reads it back, the way signal controllers
are registered for the conditions that watch them.

The route is cleared with :func:`clear_scenario_route`, which a scenario calls
first so a route left behind by the previous run in the same process is never
read as this one's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from .model import RouteMatch

if TYPE_CHECKING:
    from lanelet2.core import LaneletMap

    from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
    from ..trajectory.model import (
        MapPose,
        RelativeLanePose,
        RouteCrossingPose,
        RouteCrosswalkPose,
        RouteLanePose,
        RouteOppositePose,
        RouteRoadsidePose,
    )

    from lanelet2.routing import RoutingGraph

    from .frame import RouteFrame

__all__ = [
    "clear_scenario_route",
    "lane_place",
    "scenario_route",
    "scenario_route_frame",
    "set_scenario_route",
]

_ROUTE: Optional[RouteMatch] = None
_FRAME: Optional["RouteFrame"] = None


def set_scenario_route(
    match: RouteMatch,
    lanelet_map: Optional[LaneletMap] = None,
    routing_graph: Optional[RoutingGraph] = None,
) -> None:
    """Make *match* the scenario's route.

    *lanelet_map* and *routing_graph* lay it on a given map; left out, it is
    laid on the loaded one (:class:`~autoware_carla_scenario.coordinate.MapManager`)
    when first asked for.
    """
    global _ROUTE, _FRAME
    _ROUTE = match
    _FRAME = None
    if lanelet_map is not None:
        from .frame import RouteFrame  # noqa: PLC0415

        if routing_graph is None:
            from ..sweeper.constraints import create_routing_graph  # noqa: PLC0415

            routing_graph = create_routing_graph(lanelet_map)
        _FRAME = RouteFrame(match, lanelet_map, routing_graph)


def clear_scenario_route() -> None:
    """Forget the scenario's route."""
    global _ROUTE, _FRAME
    _ROUTE = None
    _FRAME = None


def scenario_route() -> Optional[RouteMatch]:
    """The scenario's route, or ``None``."""
    return _ROUTE


def scenario_route_frame() -> "RouteFrame":
    """The scenario's route laid on its map.

    Raises:
        ValueError: If the scenario has no route.
    """
    global _FRAME
    if _ROUTE is None:
        raise ValueError(
            "this scenario has no route: a route pose or a route_progress "
            "condition needs the document's route search (or "
            "set_scenario_route) to have chosen one"
        )
    if _FRAME is None:
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from .frame import RouteFrame  # noqa: PLC0415

        manager = MapManager.get_instance()
        _FRAME = RouteFrame(_ROUTE, manager.lanelet_map, manager.routing_graph)
    return _FRAME


def lane_place(
    position: Union[
        CarlaWorldPose,
        Lanelet2Pose,
        OpenDrivePose,
        MapPose,
        RelativeLanePose,
        RouteLanePose,
        RouteOppositePose,
        RouteCrossingPose,
        RouteCrosswalkPose,
        RouteRoadsidePose,
    ],
) -> Optional[Lanelet2Pose]:
    """The lane place *position* is on: a Lanelet2 pose as it is, a route pose
    on a lane (lane, opposite, crossing; one counted from an anchor) as the
    scenario's route places it; ``None`` for any other position, or with no
    route.

    What a spawn shifts along, and the way a place without a heading of its
    own faces (along the lane).
    """
    # Imported here: trajectory.model imports this package (a cycle at the
    # top of the module).
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415

    if isinstance(position, Lanelet2Pose):
        # Named as the class: Codon does not narrow a union by isinstance()
        # (docs/typecheck.md).
        given: Lanelet2Pose = position
        return given
    if TYPE_CHECKING:
        return None  # the route's poses are not compiled here (docs/typecheck.md)
    from ..trajectory.model import (  # noqa: PLC0415
        RouteCrossingPose,
        RouteLanePose,
        RouteOppositePose,
    )
    from .positions import resolve_route_pose  # noqa: PLC0415

    if not isinstance(position, (RouteLanePose, RouteOppositePose, RouteCrossingPose)):
        return None
    try:
        # A pose counted from the ego (no anchor) has no ego position here.
        placed = resolve_route_pose(position, scenario_route_frame())
    except ValueError:
        return None
    return placed if isinstance(placed, Lanelet2Pose) else None
