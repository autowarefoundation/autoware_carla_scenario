"""Driving a CARLA vehicle along its SUMO vehicle's trajectory, with physics.

SUMO plans; CARLA moves the car.  Each tick the vehicle is steered towards a
point on SUMO's path ahead (pure pursuit, TeraSim's law -- see below) and its
throttle and brake are set by a PI speed controller with SUMO's own
acceleration as feedforward.  The speed it aims for is SUMO's, corrected by how
far the car is behind or ahead of its SUMO vehicle, so a car that falls behind
catches up instead of drifting.

The steering is TeraSim's (``terasim_service/utils/carla/ackermann_control.py``
in https://github.com/autowarefoundation/TeraSim, Apache-2.0).  TeraSim sends
its command through CARLA's Ackermann controller; on CARLA 0.10 that
controller's speed loop overshoots and oscillates by about ±1.5 m/s on a 5 m/s
step whatever its PID settings, so the speed is controlled here and sent as
throttle and brake, as an autonomy stack's actuation command is.

Plain functions of numbers, tested without either simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "LongitudinalCommand",
    "SpeedTuning",
    "SteeringTuning",
    "longitudinal",
    "lookahead_distance",
    "pursuit_curvature",
    "steer_command",
    "SteerCommand",
    "SteerResponse",
    "to_vehicle_frame",
]


@dataclass(frozen=True)
class SteeringTuning:
    """Pure pursuit's look-ahead.

    Shorter than TeraSim's 7-15 m: with the curvature loop below closing on the
    car's actual turning, a long look-ahead only cuts corners -- on CARLA's
    Town10 it put cars into the next lane on curves.
    """

    lookahead_min_m: float = 4.0
    lookahead_max_m: float = 10.0


@dataclass(frozen=True)
class SteerResponse:
    """How CARLA 0.10's steering input maps onto the road wheels, and its loop.

    Measured on ``vehicle.lincoln.mkz`` at 8 m/s: the wheel angle grows with the
    square of the input, about 51° × steer² (0.1 → 0.5°, 0.2 → 2.1°,
    0.3 → 4.6°), far from the nominal 70° × steer, and changes with speed.  The
    feedforward inverts that square law; a PI term on curvature error absorbs
    what it gets wrong.
    """

    gain_deg: float = 51.0
    kp: float = 3.0
    ki: float = 2.0
    integral_limit: float = 1.0


@dataclass(frozen=True)
class SteerCommand:
    steer: float
    integral: float


@dataclass(frozen=True)
class SpeedTuning:
    """The PI speed controller.

    Tuned on CARLA 0.10's ``vehicle.lincoln.mkz`` against a SUMO-like profile
    (2 m/s² to 8 m/s, hold, -3 m/s² to a stop): 0.12 m/s RMS speed error,
    0.33 m/s at worst.

    Attributes:
        kp: Throttle (or brake) per m/s of speed error.
        ki: Per m·s of accumulated speed error; what overcomes the throttle a
            heavy car needs before it moves at all.
        kff: Per m/s² of SUMO's own acceleration.
        integral_limit: Anti-windup bound on the accumulated error, m.
        k_position: m/s of extra target speed per metre the car is behind its
            SUMO vehicle (less when ahead).
        max_position_correction: Bound on that correction, m/s.
        stop_speed: Below this, with SUMO stopped, the car is held on the brake.
    """

    kp: float = 0.8
    ki: float = 0.4
    kff: float = 0.15
    integral_limit: float = 2.0
    k_position: float = 1.5
    max_position_correction: float = 6.0
    stop_speed: float = 0.3


@dataclass(frozen=True)
class LongitudinalCommand:
    throttle: float
    brake: float
    integral: float
    target_speed: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def to_vehicle_frame(
    origin_x: float, origin_y: float, yaw_deg: float, x: float, y: float
) -> tuple[float, float]:
    """(forward, right) of point (x, y) in a frame at the origin facing *yaw_deg*.

    CARLA's frame is left-handed: y points to the right of x seen from above.
    So a point to the right comes out positive, and so does the steer towards
    it -- a positive steer turns right in CARLA.
    """
    dx, dy = x - origin_x, y - origin_y
    yaw = math.radians(yaw_deg)
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(
        yaw
    ) * dy


def lookahead_distance(
    speed: float, tuning: SteeringTuning = SteeringTuning()
) -> float:
    """How far ahead on the path the steering aims, metres."""
    return _clamp(speed, tuning.lookahead_min_m, tuning.lookahead_max_m)


def pursuit_curvature(
    *,
    x: float,
    y: float,
    yaw_deg: float,
    lookahead_x: float,
    lookahead_y: float,
    rear_axle_offset: float = 0.0,
) -> float:
    """Curvature (1/m, CARLA's sign: positive turns right) towards the look-ahead point."""
    yaw = math.radians(yaw_deg)
    axle_x = x + math.cos(yaw) * rear_axle_offset
    axle_y = y + math.sin(yaw) * rear_axle_offset
    forward, right = to_vehicle_frame(axle_x, axle_y, yaw_deg, lookahead_x, lookahead_y)
    distance = max(math.hypot(forward, right), 0.1)
    return 2.0 * math.sin(math.atan2(right, forward)) / distance


def steer_command(
    *,
    curvature: float,
    actual_curvature: float,
    wheel_base: float,
    integral: float,
    dt: float = 0.05,
    response: SteerResponse = SteerResponse(),
) -> SteerCommand:
    """CARLA's normalised steer input for a desired curvature.

    Args:
        curvature: The curvature pure pursuit asks for, 1/m.
        actual_curvature: The car's, from its yaw rate over its speed, 1/m.
        integral: The accumulated curvature error carried from the last tick.
    """
    wheel_deg = math.degrees(math.atan(wheel_base * curvature))
    feedforward = math.copysign(
        math.sqrt(abs(wheel_deg) / response.gain_deg), wheel_deg
    )
    error = curvature - actual_curvature
    integral = _clamp(
        integral + response.ki * error * dt,
        -response.integral_limit,
        response.integral_limit,
    )
    return SteerCommand(
        _clamp(feedforward + response.kp * error + integral, -1.0, 1.0), integral
    )


def longitudinal(
    *,
    speed: float,
    sumo_speed: float,
    sumo_acceleration: float,
    longitudinal_error: float,
    integral: float,
    dt: float = 0.05,
    tuning: SpeedTuning = SpeedTuning(),
) -> LongitudinalCommand:
    """Throttle and brake that keep a car on its SUMO vehicle.

    Args:
        speed: The car's speed, m/s.
        sumo_speed: Its SUMO vehicle's speed, m/s.
        sumo_acceleration: Its SUMO vehicle's acceleration, m/s².
        longitudinal_error: How far ahead of the car its SUMO vehicle is, m
            (negative when the car is ahead).
        integral: The accumulated speed error carried from the last tick.
    """
    if sumo_speed < 0.05 and speed < tuning.stop_speed:
        # Stopped in SUMO (a red light, a queue): hold it there.
        return LongitudinalCommand(0.0, 1.0, 0.0, 0.0)
    correction = _clamp(
        tuning.k_position * longitudinal_error,
        -tuning.max_position_correction,
        tuning.max_position_correction,
    )
    target = max(0.0, sumo_speed + correction)
    error = target - max(0.0, speed)
    integral = _clamp(
        integral + error * dt, -tuning.integral_limit, tuning.integral_limit
    )
    u = tuning.kp * error + tuning.ki * integral + tuning.kff * sumo_acceleration
    if u >= 0.0:
        return LongitudinalCommand(min(1.0, u), 0.0, integral, target)
    return LongitudinalCommand(0.0, min(1.0, -u), integral, target)
