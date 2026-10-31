"""When a pedestrian on (or at the edge of) a road has a vehicle too close.

The one test both :class:`~.walk_straight.WalkStraightAction` and a routed
pedestrian stepped over a road stretch (:mod:`~..entity._walker_ai`) turn
back or wait by.  *Too close* is measured in the lane frame, as
:class:`~autoware_carla_scenario.conditions.TimeToCollisionCondition` and
:class:`~autoware_carla_scenario.conditions.EntityDistanceCondition` measure
with ``coordinate_system=LANE``: the separation along the road between the
vehicle's centre and the pedestrian's, and its time to collision (that
separation over the speed at which it closes).  Either at or under its
threshold is a threat; a threshold left ``None`` is not looked at.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..coordinate.lane_distance import lane_closing_speed, lane_separation
from ..coordinate.poses import CarlaWorldPose
from ..utils.clearance import _HIDDEN_BELOW_M

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "DEFAULT_TURN_BACK_WITHIN_M",
    "PASSING_CLEARANCE_M",
    "STEP_BACK_M",
    "road_threat",
]

#: How far onto the road a pedestrian may have walked and still turn back (m),
#: when the scenario does not say: about one stride and a half.
DEFAULT_TURN_BACK_WITHIN_M = 1.0

#: How far back from the kerb a pedestrian that turns back stands (m), at most
#: as far as where it set off.  Not on the kerb's edge itself: a CARLA 0.10
#: walker stopped with its feet on the edge tips over, slowly, into the road.
STEP_BACK_M = 0.5

#: The room (m) a vehicle the framework drives keeps to each side of it for a
#: walker: one passing closer is braked for (a bicycle passed 0.5 m off its
#: side toppled).
PASSING_CLEARANCE_M = 1.0

#: Closing speeds below this are "not closing", as in the TTC condition.
_CLOSING_SPEED_EPSILON = 1e-6


def road_threat(
    world: "carla.World",
    location: "carla.Location",
    walking_x: float,
    walking_y: float,
    distance_m: Optional[float],
    ttc_s: Optional[float],
    near_m: Optional[float] = None,
) -> Optional[str]:
    """What threatens a pedestrian at *location*, walking at
    ``(walking_x, walking_y)`` m/s -- a vehicle at most *distance_m* away along
    the road, or at most *ttc_s* from it -- or ``None``.  Every vehicle in the
    world counts but one parked under the map (#853); with *near_m*, only
    those within that straight-line distance (to spare the lane search for
    the far ones)."""
    if distance_m is None and ttc_s is None:
        return None
    target = CarlaWorldPose(x=location.x, y=location.y, z=location.z, yaw=0.0)
    for vehicle in world.get_actors().filter("vehicle.*"):
        at = vehicle.get_location()
        # Parked under the map until its spawn: its lane projection (2D, or
        # onto the road above) would put it on the road (#853).
        if at.z < _HIDDEN_BELOW_M:
            continue
        if near_m is not None:
            reach: float = near_m
            if math.hypot(at.x - location.x, at.y - location.y) > reach:
                continue
        source = CarlaWorldPose(x=at.x, y=at.y, z=at.z, yaw=0.0)
        separation = lane_separation(source, target)
        if separation is None:
            continue
        if distance_m is not None and separation <= distance_m:
            return f"vehicle {vehicle.id} {separation:.1f} m away along the road"
        if ttc_s is None or separation < _CLOSING_SPEED_EPSILON:
            continue
        threshold: float = ttc_s
        driving = vehicle.get_velocity()
        closing = lane_closing_speed(
            source, driving.x, driving.y, target, walking_x, walking_y
        )
        if closing is None or closing < _CLOSING_SPEED_EPSILON:
            continue
        ttc = separation / closing
        if ttc <= threshold:
            return f"vehicle {vehicle.id} {ttc:.1f} s away along the road"
    return None
