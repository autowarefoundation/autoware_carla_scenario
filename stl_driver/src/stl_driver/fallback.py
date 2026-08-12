# SPDX-License-Identifier: Apache-2.0
"""What to drive when the specification cannot be satisfied.

Two situations, and they are different.

*No route, or no ego pose yet.*  Nothing is wrong; the session has just
started, or the runtime has not published a route.  Hold the current heading
and coast, which is what :class:`RouteFollowerPolicy` does in the same spot.

*A hard rule is violated by every plan considered.*  Something is wrong.
Brake, harder than the comfort rule permits -- the point of marking a rule
``hard`` is that comfort is no longer the priority.

The emergency plan deliberately bypasses the optimizer rather than being
expressed as a strongly-weighted rule.  A specification says what good driving
looks like; it should not have to also encode what to do when good driving is
impossible, and a planner that could always find *something* would obscure the
fact that it had failed.
"""

from __future__ import annotations

import numpy as np
from carla_driver_interface.geometry import Pose, Trajectory

from stl_driver.config import STLDriverConfig

__all__ = ["emergency_brake_plan", "hold_plan"]


def hold_plan(
    time_now_us: int,
    speed_mps: float,
    config: STLDriverConfig,
) -> Trajectory:
    """Continue straight ahead in the rig frame, easing off.

    Used before a route arrives.  Straight ahead is what "hold this heading"
    means to a controller that reads its target from the plan's own geometry,
    and a gentle deceleration is the honest response to not knowing where the
    road goes.
    """
    return _straight_line_plan(
        time_now_us,
        speed_mps,
        decel_mps2=config.limits.max_decel_mps2 * 0.4,
        config=config,
    )


def emergency_brake_plan(
    time_now_us: int,
    speed_mps: float,
    config: STLDriverConfig,
    centreline_xy: np.ndarray | None = None,
) -> Trajectory:
    """Stop as quickly as the vehicle can, along the route if we know it.

    Following the centreline rather than the current heading matters on a
    bend: braking straight ahead there would leave the lane while stopping.
    """
    return _straight_line_plan(
        time_now_us,
        speed_mps,
        decel_mps2=config.limits.emergency_decel_mps2,
        config=config,
        centreline_xy=centreline_xy,
    )


def _straight_line_plan(
    time_now_us: int,
    speed_mps: float,
    decel_mps2: float,
    config: STLDriverConfig,
    centreline_xy: np.ndarray | None = None,
) -> Trajectory:
    """Integrate a deceleration profile and lay it out as a trajectory."""
    horizon = config.horizon
    step_us = int(round(horizon.step_s * 1e6))

    speed = max(0.0, float(speed_mps))
    travelled = 0.0
    plan = Trajectory.empty()

    for index in range(1, horizon.n_steps + 1):
        speed = max(0.0, speed - decel_mps2 * horizon.step_s)
        travelled += speed * horizon.step_s
        x, y, heading = _point_at(travelled, centreline_xy)
        # Strictly increasing and strictly greater than `time_now_us`: the
        # driver service silently drops anything at or before it, and
        # `Trajectory.append` rejects a non-increasing stamp outright.
        plan.append(
            time_now_us + index * step_us,
            Pose.from_xyz_yaw(x, y, 0.0, heading),
        )
    return plan


def _point_at(distance_m: float, centreline_xy: np.ndarray | None) -> tuple[float, float, float]:
    """Position and heading at an arc length, along the route or straight on."""
    if centreline_xy is None or len(centreline_xy) < 2:
        return distance_m, 0.0, 0.0

    # The centreline is uniformly spaced, so the index is arithmetic.
    spacing = float(np.linalg.norm(centreline_xy[1] - centreline_xy[0]))
    if spacing < 1e-9:
        return distance_m, 0.0, 0.0

    position = np.clip(distance_m / spacing, 0.0, len(centreline_xy) - 1.0)
    lower = int(np.floor(position))
    upper = min(lower + 1, len(centreline_xy) - 1)
    weight = position - lower
    point = centreline_xy[lower] * (1.0 - weight) + centreline_xy[upper] * weight

    tangent = centreline_xy[min(upper, len(centreline_xy) - 1)] - centreline_xy[lower]
    heading = float(np.arctan2(tangent[1], tangent[0])) if np.linalg.norm(tangent) > 1e-9 else 0.0
    return float(point[0]), float(point[1]), heading
