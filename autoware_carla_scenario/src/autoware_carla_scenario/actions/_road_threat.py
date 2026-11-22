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

from ..coordinate.lane_distance import _NEAR_ZERO, _along_s, _project, _reach
from ..coordinate.poses import CarlaWorldPose, OpenDrivePose
from ..utils.clearance import HIDDEN_BELOW_M

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
    the far ones).

    The separation and the closing speed are
    :func:`~..coordinate.lane_distance.lane_separation`'s and
    :func:`~..coordinate.lane_distance.lane_closing_speed`'s, worked out from
    their own steps so that what does not change between vehicles is done
    once a call (#2336): the pedestrian is projected onto the road once, and
    its speed along the road read once, not for every vehicle; each vehicle
    is projected once, and the chain between the two walked once, for both
    the separation and the closing speed.  Every step is deterministic, so
    the answers are the same.
    """
    if distance_m is None and ttc_s is None:
        return None
    target = CarlaWorldPose(x=location.x, y=location.y, z=location.z, yaw=0.0)
    # The pedestrian's projection and speed along the road, each worked out
    # the first time a vehicle needs it.
    target_projected = False
    target_od: Optional[OpenDrivePose] = None
    target_read = False
    target_along: Optional[float] = None
    # A bound on how fast any vehicle can close on it, for the TTC: a speed
    # along the road is ``speed * cos(angle)`` (``_along_s``), never more than
    # the speed in plan view, so the closing speed -- one along-road speed
    # minus the other, each signed -- is never more than the two plan-view
    # speeds summed (in floating point too: rounding is monotonic).
    walking_speed = math.hypot(walking_x, walking_y)
    for vehicle in world.get_actors().filter("vehicle.*"):
        at = vehicle.get_location()
        # Parked under the map until its spawn: its lane projection (2D, or
        # onto the road above) would put it on the road (#853).
        if at.z < HIDDEN_BELOW_M:
            continue
        if near_m is not None:
            reach: float = near_m
            if math.hypot(at.x - location.x, at.y - location.y) > reach:
                continue
        # Whether its time to collision can be within *ttc_s*: not when the
        # two plan-view speeds sum to less than what counts as closing at
        # all, below.  A NaN speed can.
        can_close = False
        driving_x = driving_y = 0.0
        if ttc_s is not None:
            driving = vehicle.get_velocity()
            driving_x, driving_y = driving.x, driving.y
            can_close = not (
                math.hypot(driving_x, driving_y) + walking_speed
                < _CLOSING_SPEED_EPSILON
            )
        if distance_m is None and not can_close:
            continue  # nothing it could threaten by
        if not target_projected:
            target_od = _project(target)
            target_projected = True
        if target_od is None:
            continue  # lane_separation: no measurement
        on_road: OpenDrivePose = target_od
        source = CarlaWorldPose(x=at.x, y=at.y, z=at.z, yaw=0.0)
        source_od = _project(source)
        if source_od is None:
            continue
        found = _reach(source_od, on_road)
        if found is None:
            continue
        separation = found.distance
        if distance_m is not None and separation <= distance_m:
            return f"vehicle {vehicle.id} {separation:.1f} m away along the road"
        if ttc_s is None or separation < _CLOSING_SPEED_EPSILON:
            continue
        if not can_close:
            continue
        threshold: float = ttc_s
        # lane_closing_speed, from the same projections and chain.
        if found.distance < _NEAR_ZERO:
            continue
        source_along = _along_s(source, driving_x, driving_y)
        if not target_read:
            target_along = _along_s(target, walking_x, walking_y)
            target_read = True
        if source_along is None or target_along is None:
            continue
        source_sign = 1.0 if found.forward else -1.0
        target_sign = 1.0 if found.target_forward else -1.0
        closing = source_sign * source_along - target_sign * target_along
        if closing < _CLOSING_SPEED_EPSILON:
            continue
        ttc = separation / closing
        if ttc <= threshold:
            return f"vehicle {vehicle.id} {ttc:.1f} s away along the road"
    return None
