# SPDX-License-Identifier: Apache-2.0
"""What the world looks like this step, reduced to numbers the rules can read.

Three jobs: move other actors from the ``local`` frame into ours, predict
where they will be over the planning horizon, and work out which of them we
are following.  Plus the traffic light, which is a single distance.

All of it is numpy and none of it depends on the plan, so it happens once per
``drive()`` call rather than once per candidate.  The optimizer evaluates
around seventy plans per step; anything constant across them belongs here.

Coordinate frames
-----------------
The runtime publishes actors in the alpasim ``local`` frame while a policy
plans in ``rig``.  The ego pose ``anchor = session.latest_pose()`` maps rig to
local -- the driver service composes ``anchor @ plan_pose`` to publish a plan,
and the runtime's controller undoes it with
``pose_local_to_rig.inverse().transform_points(...)``.  Both call sites agree,
so ``anchor.inverse()`` is what brings an actor into the rig frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from carla_driver_interface.geometry import Pose, quat_to_yaw
from carla_driver_interface.grpc_api import CarlaRendererData, TrafficLightState

from stl_driver.config import STLDriverConfig
from stl_driver.frenet import RouteTable, project_to_frenet

__all__ = ["ActorForecast", "TrafficLight", "forecast_actors", "read_traffic_light"]

#: Light states that require us to stop at the line.
_STOPPING_STATES = frozenset(
    {
        TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
        TrafficLightState.TRAFFIC_LIGHT_STATE_YELLOW,
    }
)


@dataclass(frozen=True)
class ActorForecast:
    """Where every other actor is expected to be, over the horizon."""

    #: ``(T, A, 2)`` predicted disc centres in the rig frame.
    centres: np.ndarray
    #: ``(A,)`` disc radii.
    radii: np.ndarray
    #: ``(T,)`` station of the vehicle we are following; ``inf`` where none.
    lead_station_m: np.ndarray
    #: How many actors the forecast came from, for reporting.
    n_actors: int

    @property
    def n_discs(self) -> int:
        return int(self.radii.shape[0])


@dataclass(frozen=True)
class TrafficLight:
    """The light governing the ego, reduced to a station and a gate."""

    #: Station the plan must not pass, or ``inf`` when no light applies.
    stop_station_m: float
    #: ``True`` when a red or amber light governs us.
    active: bool


def read_traffic_light(
    renderer_data: CarlaRendererData | None,
    station0_m: float,
    config: STLDriverConfig,
) -> TrafficLight:
    """Where to stop for the light ahead, if any.

    ``renderer_data`` is ``None`` under an upstream alpasim runtime, or when
    the extension payload could not be parsed; a policy that assumed otherwise
    would crash the moment it left this project's runtime.
    """
    if renderer_data is None:
        return TrafficLight(stop_station_m=math.inf, active=False)

    state = renderer_data.ego_traffic_light
    distance = float(renderer_data.ego_traffic_light_distance_m)
    # The proto documents the distance as negative when no stop line applies to
    # us -- no light, or one whose line we have already crossed. A stopping
    # state alone is therefore not enough to act on: once past the line, the
    # thing to do is clear the intersection, not stop dead inside it.
    if state not in _STOPPING_STATES or distance < 0.0:
        return TrafficLight(stop_station_m=math.inf, active=False)

    # Never place the stop line behind where we already are. Inside the last
    # few metres the margin would otherwise ask the plan to travel backwards,
    # which no plan can do -- so `stop_on_red` would be violated by every
    # candidate, the fallback would engage, and because the vehicle then stops
    # inside the light's trigger volume it would stay there indefinitely.
    # Clamped, the rule reads "do not advance", which is both satisfiable and
    # what a driver stopped just short of the line actually does.
    stop_station = station0_m + max(0.0, distance - config.limits.stop_line_margin_m)
    return TrafficLight(stop_station_m=stop_station, active=True)


def forecast_actors(
    renderer_data: CarlaRendererData | None,
    anchor: Pose,
    table: RouteTable,
    station0_m: float,
    config: STLDriverConfig,
) -> ActorForecast:
    """Predict other actors over the horizon, in the rig frame.

    Prediction is constant velocity.  That is deliberately unambitious: a
    richer model would be guessing, and the specification already demands
    clearance from wherever the actors actually turn out to be, re-planned at
    10 Hz.  What matters more than the model's sophistication is that it is
    honest about a vehicle continuing to move.

    Args:
        renderer_data: the CARLA extension payload, or ``None``.
        anchor: ego pose mapping rig to local.
        table: the route, for deciding which actors are in our lane.
        station0_m: the ego's station now.
        config: limits and horizon.
    """
    horizon = config.horizon
    n_steps = horizon.n_steps
    empty = ActorForecast(
        centres=np.zeros((n_steps, 0, 2), dtype=np.float32),
        radii=np.zeros((0,), dtype=np.float32),
        lead_station_m=np.full((n_steps,), np.inf, dtype=np.float32),
        n_actors=0,
    )
    if renderer_data is None or not renderer_data.actors:
        return empty

    to_rig = anchor.inverse()
    rotation = to_rig.rotation_matrix

    centres_local = []
    velocities_local = []
    yaws_rig = []
    sizes = []
    for actor in renderer_data.actors:
        pose = actor.pose_local_to_aabb
        centres_local.append([pose.vec.x, pose.vec.y, pose.vec.z])
        velocity = actor.dynamic_state.linear_velocity
        velocities_local.append([velocity.x, velocity.y, velocity.z])
        quat_local = np.array([pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w])
        yaws_rig.append(_yaw_in_rig(to_rig, quat_local))
        sizes.append([actor.aabb.size_x, actor.aabb.size_y])

    centres = to_rig.transform_points(np.asarray(centres_local, dtype=np.float64))[:, :2]
    # Velocity is a free vector, so it rotates but does not translate. The
    # runtime resolves actor velocities in the local frame (not the actor's
    # own), which is what makes a plain rotation correct here.
    velocities = (np.asarray(velocities_local, dtype=np.float64) @ rotation.T)[:, :2]
    yaws = np.asarray(yaws_rig, dtype=np.float64)
    sizes_array = np.asarray(sizes, dtype=np.float64)

    centres_over_time, radii = _disc_forecast(centres, velocities, yaws, sizes_array, config)
    lead_station = _lead_station(centres_over_time, sizes_array, table, station0_m, config)

    return ActorForecast(
        centres=centres_over_time.astype(np.float32),
        radii=radii.astype(np.float32),
        lead_station_m=lead_station.astype(np.float32),
        n_actors=len(renderer_data.actors),
    )


def _yaw_in_rig(to_rig: Pose, quat_local_xyzw: np.ndarray) -> float:
    """Heading of an actor once brought into the rig frame."""
    from carla_driver_interface.geometry import quat_mul

    return float(quat_to_yaw(quat_mul(to_rig.quat_xyzw, quat_local_xyzw)))


def _disc_forecast(
    centres: np.ndarray,
    velocities: np.ndarray,
    yaws: np.ndarray,
    sizes: np.ndarray,
    config: STLDriverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Two discs per actor, advanced at constant velocity.

    A rectangle would be exact but not differentiable at its corners, and a
    single disc around a 4.5 m car either swallows the adjacent lane or leaves
    its ends uncovered.  Two discs at the quarter points are the usual
    compromise and are tight enough that the collision margin covers the rest.
    """
    horizon = config.horizon
    times = np.arange(1, horizon.n_steps + 1, dtype=np.float64) * horizon.step_s

    lengths = np.maximum(sizes[:, 0], 0.5)
    widths = np.maximum(sizes[:, 1], 0.5)
    quarter = lengths / 4.0
    radii = np.hypot(quarter, widths / 2.0)

    heading = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)
    # (A, 2, 2): each actor's two disc centres, fore and aft of its origin.
    disc_centres = np.stack(
        [centres + quarter[:, None] * heading, centres - quarter[:, None] * heading],
        axis=1,
    )
    disc_velocities = np.repeat(velocities[:, None, :], 2, axis=1)

    # (T, A, 2, 2) -> (T, A*2, 2)
    predicted = disc_centres[None] + disc_velocities[None] * times[:, None, None, None]
    predicted = predicted.reshape(len(times), -1, 2)

    return predicted, np.repeat(radii, 2)


def _lead_station(
    centres_over_time: np.ndarray,
    sizes: np.ndarray,
    table: RouteTable,
    station0_m: float,
    config: STLDriverConfig,
) -> np.ndarray:
    """Station of the nearest in-lane actor ahead, per timestep.

    "In lane" is decided in Frenet coordinates rather than by heading, so a
    vehicle on a parallel road does not become our lead just because it points
    the same way.
    """
    limits = config.limits
    n_steps, n_discs, _ = centres_over_time.shape
    if n_discs == 0:
        return np.full((n_steps,), np.inf)

    flat = centres_over_time.reshape(-1, 2)
    station, offset = project_to_frenet(table, flat)
    station = station.reshape(n_steps, n_discs)
    offset = offset.reshape(n_steps, n_discs)

    # A disc's own half-width still counts as intruding, so use the actor's
    # width rather than treating it as a point.
    half_widths = np.repeat(np.maximum(sizes[:, 1], 0.5) / 2.0, 2)
    in_lane = np.abs(offset) - half_widths[None, :] < limits.lead_capture_half_width_m
    ahead = station > station0_m + 0.5

    candidate = np.where(in_lane & ahead, station, np.inf)
    lead = candidate.min(axis=1)

    # Reading the *rear* of the lead vehicle: the discs are its centres, so
    # back off by the quarter length plus the radius the gap already accounts
    # for. Being a little conservative here is the right direction to err.
    lengths = np.repeat(np.maximum(sizes[:, 0], 0.5), 2)
    rear_offset = float(np.max(lengths) / 2.0) if len(lengths) else 0.0
    return np.where(np.isfinite(lead), lead - rear_offset, np.inf)
