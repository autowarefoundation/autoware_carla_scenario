"""Kinematic prediction (TeraSim's ``trajectory_predictor`` and trajectory utils).

NADE predicts every candidate's modalities at constant acceleration, a few
seconds ahead; these are the pieces that do not need the map.
"""

from __future__ import annotations

import math

from ..kinematics.angle import normalize_angle_rad
from .geometry import PredictedState

__all__ = [
    "interpolate_trajectory",
    "lane_change_longitudinal_speed",
    "predict_distance_velocity",
]


def predict_distance_velocity(
    velocity: float,
    acceleration: float,
    times: list[float],
    max_velocity: float,
) -> tuple[list[float], list[float]]:
    """Distance travelled and speed at each of *times*, at constant acceleration.

    The speed is clipped to ``[0, max_velocity]`` at each time and the
    distance integrates it trapezoidally between consecutive times, starting
    at 0 at ``times[0]``: a braking agent stops instead of reversing.

    Returns:
        ``(distances, velocities)``, one entry per time.
    """
    velocities = [
        min(max(velocity + acceleration * t, 0.0), max_velocity) for t in times
    ]
    distances = [0.0] * len(times)
    for i in range(1, len(times)):
        distances[i] = distances[i - 1] + 0.5 * (velocities[i] + velocities[i - 1]) * (
            times[i] - times[i - 1]
        )
    return distances, velocities


def lane_change_longitudinal_speed(
    speed: float, lane_width: float, duration: float = 1.0
) -> float:
    """Longitudinal speed while crossing *lane_width* in *duration* seconds.

    The lateral speed ``lane_width / duration`` is taken out of *speed*; an
    agent too slow for it has no longitudinal speed left.
    """
    lateral = lane_width / duration
    return math.sqrt(max(speed * speed - lateral * lateral, 0.0))


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def interpolate_trajectory(
    trajectory: list[PredictedState], resolution: float
) -> list[PredictedState]:
    """Resample *trajectory* every *resolution* seconds from its first time.

    Linear in position, speed and time; the yaw turns the short way between
    samples and comes back wrapped into ``[-pi, pi)``.  The last sample is
    the last multiple of *resolution* that does not pass the end.
    """
    if len(trajectory) < 2:
        return list(trajectory)
    start = trajectory[0].t
    count = int((trajectory[-1].t - start) / resolution + 1e-9) + 1
    unwrapped = [trajectory[0].yaw]
    for i in range(1, len(trajectory)):
        unwrapped.append(
            unwrapped[-1]
            + normalize_angle_rad(trajectory[i].yaw - trajectory[i - 1].yaw)
        )
    resampled: list[PredictedState] = []
    segment = 1
    for n in range(count):
        t = start + n * resolution
        while segment < len(trajectory) - 1 and trajectory[segment].t < t:
            segment += 1
        before = trajectory[segment - 1]
        after = trajectory[segment]
        span = after.t - before.t
        f = 0.0 if span <= 0.0 else min(max((t - before.t) / span, 0.0), 1.0)
        resampled.append(
            PredictedState(
                _lerp(before.x, after.x, f),
                _lerp(before.y, after.y, f),
                normalize_angle_rad(
                    _lerp(unwrapped[segment - 1], unwrapped[segment], f)
                ),
                _lerp(before.speed, after.speed, f),
                t,
            )
        )
    return resampled
