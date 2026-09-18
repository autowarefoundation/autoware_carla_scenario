"""Compatibility home of the manoeuvre mixin and its vocabulary.

The mechanism moved twice and the names stayed: the manoeuvres now live on
:class:`~autoware_carla_scenario.traffic.driven.BackendDriven`, which
delegates them to whatever traffic backend drives the run, and the TrafficManager
bodies they used to contain are in
:mod:`autoware_carla_scenario.traffic.traffic_manager` -- one backend among
several rather than the only way to drive a vehicle.

This module re-exports both so that ``from ...entity.tm_driving import
LaneChangeDirection`` keeps working for any scenario package written against it.
Nothing in this package imports it any more.  New code should import
:class:`~autoware_carla_scenario.traffic.driven.BackendDriven` and the
vocabulary from :mod:`autoware_carla_scenario.traffic`.
"""

from __future__ import annotations

from ..traffic.base import (
    LaneChangeDirection,
    LaneChanging,
    TurnDirection,
    TurningAtJunctions,
)
from ..traffic.traffic_manager import (
    TURN_POST_JUNCTION_DISTANCE_M,
    TURN_SEARCH_DISTANCE_M,
    TURN_WAYPOINT_STEP_M,
    compute_turn_route,
)
from ..traffic.driven import BackendDriven

__all__ = [
    "BackendDriven",
    "LaneChangeDirection",
    "LaneChanging",
    "TrafficManagerDriven",
    "TurnDirection",
    "TurningAtJunctions",
    "TURN_POST_JUNCTION_DISTANCE_M",
    "TURN_SEARCH_DISTANCE_M",
    "TURN_WAYPOINT_STEP_M",
    "compute_turn_route",
]


class TrafficManagerDriven(BackendDriven):
    """A vehicle the TrafficManager drives, said the old way.

    Deprecated in favour of :class:`BackendDriven`, and identical to it: an
    entity that is given only a CARLA client *is* TrafficManager-driven, which
    is what this name always meant.  Kept because external scenario packages
    subclass it.
    """
