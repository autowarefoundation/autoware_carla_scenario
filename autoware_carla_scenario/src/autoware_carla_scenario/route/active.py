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

from typing import TYPE_CHECKING, Any, Optional

from .model import RouteMatch

if TYPE_CHECKING:
    from .frame import RouteFrame

__all__ = [
    "clear_scenario_route",
    "scenario_route",
    "scenario_route_frame",
    "set_scenario_route",
]

_ROUTE: Optional[RouteMatch] = None
_FRAME: Optional["RouteFrame"] = None


def set_scenario_route(
    match: RouteMatch, lanelet_map: Any = None, routing_graph: Any = None
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
