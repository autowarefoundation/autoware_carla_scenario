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
from stl_driver.frenet import RouteTable

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
    route: RouteTable | None = None,
    station0_m: float = 0.0,
    offset0_m: float = 0.0,
) -> Trajectory:
    """Stop as quickly as the vehicle can, along the route if we know it.

    Following the route rather than the current heading matters on a bend:
    braking straight ahead there would leave the lane while stopping.

    The plan starts where the ego is on the route -- ``station0_m`` and
    ``offset0_m``, as :func:`~stl_driver.frenet.project_to_frenet` gives them
    -- not at the table's first point. A table built with ``behind_m`` starts
    behind the ego, and a plan measured from its first point would be shifted
    back by the ego's station: below a few metres per second, every pose
    behind the vehicle.

    The offset is held, not eased back to the centreline. A normal plan
    starts from the measured offset too, so the two agree where the plan
    begins; after that, braking must not also swerve -- moving across the
    lane under maximum braking spends grip the stop needs, and an emergency
    stop is not the moment to recentre. Held, the path runs parallel to the
    route, so its heading is the route's tangent at each station; at rest
    the station stops changing and the heading with it, which is the
    held-at-rest heading :func:`stl_driver.rollout.plan_xy` gives a normal
    plan.

    Args:
        route: the route table; ``None`` brakes straight ahead in the rig
            frame.
        station0_m: the ego's station on ``route``.
        offset0_m: the ego's signed lateral offset from ``route``.
    """
    travelled = _braking_distances(speed_mps, config.limits.emergency_decel_mps2, config)
    if route is None or len(route) < 2:
        return _lay_out_straight(time_now_us, travelled, config)

    positions, tangents = route.frenet_to_xy_many(
        station0_m + travelled, np.full_like(travelled, offset0_m)
    )
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    return _trajectory(time_now_us, positions, headings, config)


def _braking_distances(speed_mps: float, decel_mps2: float, config: STLDriverConfig) -> np.ndarray:
    """Distance covered by each plan step under a constant deceleration."""
    horizon = config.horizon
    speed = max(0.0, float(speed_mps))
    travelled = 0.0
    distances = np.empty(horizon.n_steps, dtype=np.float64)
    for index in range(horizon.n_steps):
        speed = max(0.0, speed - decel_mps2 * horizon.step_s)
        travelled += speed * horizon.step_s
        distances[index] = travelled
    return distances


def _straight_line_plan(
    time_now_us: int,
    speed_mps: float,
    decel_mps2: float,
    config: STLDriverConfig,
) -> Trajectory:
    """Integrate a deceleration profile and lay it out straight ahead."""
    return _lay_out_straight(time_now_us, _braking_distances(speed_mps, decel_mps2, config), config)


def _lay_out_straight(
    time_now_us: int, travelled: np.ndarray, config: STLDriverConfig
) -> Trajectory:
    """Distances along the rig's own heading: the rig is at the origin, facing +x."""
    positions = np.stack([travelled, np.zeros_like(travelled)], axis=1)
    return _trajectory(time_now_us, positions, np.zeros_like(travelled), config)


def _trajectory(
    time_now_us: int,
    positions: np.ndarray,
    headings: np.ndarray,
    config: STLDriverConfig,
) -> Trajectory:
    """Poses at one plan step apart, from one step after `time_now_us`."""
    step_us = int(round(config.horizon.step_s * 1e6))
    plan = Trajectory.empty()
    for index in range(len(positions)):
        # Strictly increasing and strictly greater than `time_now_us`: the
        # driver service silently drops anything at or before it, and
        # `Trajectory.append` rejects a non-increasing stamp outright.
        plan.append(
            time_now_us + (index + 1) * step_us,
            Pose.from_xyz_yaw(
                float(positions[index, 0]), float(positions[index, 1]), 0.0, float(headings[index])
            ),
        )
    return plan
