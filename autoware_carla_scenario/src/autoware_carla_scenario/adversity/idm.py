"""The car-following model the triggers compare accelerations with.

TeraSim asks SUMO's own model (``traci.vehicle.getFollowSpeed``) for the
acceleration a vehicle would have with no leader (*free flow*) and behind its
leader (*car following*); its triggers fire on the gap between the two (a
vehicle held back by its leader by more than 1.5 m/s² may "ignore" it).  CARLA
has no such model to ask, so the port uses the Intelligent Driver Model with
SUMO's default passenger-car parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DEFAULT_IDM",
    "IdmParameters",
    "car_following_acceleration",
    "free_flow_acceleration",
]


@dataclass(frozen=True)
class IdmParameters:
    """Intelligent Driver Model parameters.

    The defaults are SUMO's passenger car: ``accel`` 2.6, ``decel`` 4.5,
    ``minGap`` 2.5, ``tau`` 1.0, ``emergencyDecel`` 9.0.

    Attributes:
        max_acceleration: Maximum acceleration ``a`` in m/s².
        comfortable_deceleration: Comfortable deceleration ``b`` in m/s².
        minimum_gap: Standstill gap ``s0`` in metres.
        time_headway: Desired time headway ``T`` in seconds.
        exponent: Free-road exponent ``delta``.
        emergency_deceleration: Hardest braking in m/s²; the model never asks
            for more.
    """

    max_acceleration: float = 2.6
    comfortable_deceleration: float = 4.5
    minimum_gap: float = 2.5
    time_headway: float = 1.0
    exponent: float = 4.0
    emergency_deceleration: float = 9.0


#: SUMO's passenger car.
DEFAULT_IDM = IdmParameters()


def _clip(acceleration: float, params: IdmParameters) -> float:
    return min(
        max(acceleration, -params.emergency_deceleration), params.max_acceleration
    )


def _stop(speed: float, params: IdmParameters) -> float:
    # A desired speed of zero (a closed road) means stop: as hard as allowed.
    return -params.emergency_deceleration if speed > 0.0 else 0.0


def free_flow_acceleration(
    speed: float, desired_speed: float, params: IdmParameters = DEFAULT_IDM
) -> float:
    """Acceleration with the road ahead empty, towards *desired_speed* (m/s)."""
    if desired_speed <= 0.0:
        return _stop(speed, params)
    free_road = (max(speed, 0.0) / desired_speed) ** params.exponent
    return _clip(params.max_acceleration * (1.0 - free_road), params)


def car_following_acceleration(
    speed: float,
    desired_speed: float,
    gap: float,
    leader_speed: float,
    params: IdmParameters = DEFAULT_IDM,
) -> float:
    """Acceleration behind a leader *gap* metres ahead (bumper to bumper).

    A gap of zero or less is a collision already: the model brakes as hard as
    it may.
    """
    if gap <= 0.0:
        return -params.emergency_deceleration
    if desired_speed <= 0.0:
        return _stop(speed, params)
    speed = max(speed, 0.0)
    desired_gap = params.minimum_gap + max(
        0.0,
        speed * params.time_headway
        + speed
        * (speed - leader_speed)
        / (2.0 * math.sqrt(params.max_acceleration * params.comfortable_deceleration)),
    )
    free_road = (speed / desired_speed) ** params.exponent
    return _clip(
        params.max_acceleration * (1.0 - free_road - (desired_gap / gap) ** 2),
        params,
    )
