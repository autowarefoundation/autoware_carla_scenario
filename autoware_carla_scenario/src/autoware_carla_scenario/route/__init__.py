"""Logical scenarios: a drive described as a pattern of road, found on any map.

* :mod:`.model` -- the route search and its matches as data (no map needed);
* :mod:`.search` -- finding the matches on a Lanelet2 map;
* :mod:`.frame` -- a match laid on its map: route s to lanelet poses and back;
* :mod:`.positions` -- placing route poses (``RouteLanePose`` and its kin);
* :mod:`.active` -- the route the running scenario is about.

Only the data model and the registry are imported here, so the editor -- which
has no simulator and no Lanelet2 -- can read a document's route search.  See
``docs/logical_scenarios.md``.
"""

from .active import (
    clear_scenario_route,
    scenario_route,
    scenario_route_frame,
    set_scenario_route,
)
from .model import (
    JunctionSegmentSpec,
    LaneSegmentSpec,
    Range,
    RouteMatch,
    RouteSearchSpec,
    RouteSegmentMatch,
    parse_anchor,
    parse_route_search,
)

__all__ = [
    "JunctionSegmentSpec",
    "LaneSegmentSpec",
    "Range",
    "RouteMatch",
    "RouteSearchSpec",
    "RouteSegmentMatch",
    "clear_scenario_route",
    "parse_anchor",
    "parse_route_search",
    "scenario_route",
    "scenario_route_frame",
    "set_scenario_route",
]
