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

__all__ = [
    "ActorForecast",
    "TrafficLight",
    "footprint_discs",
    "forecast_actors",
    "read_traffic_light",
]


def footprint_discs(
    length_m: float,
    width_m: float,
    max_inflation_m: float,
) -> tuple[np.ndarray, float]:
    """Cover a vehicle's box with discs, bounding how much wider it becomes.

    Returns offsets along the body's own longitudinal axis, measured from its
    centre, and the shared radius.

    Discs must reach the corners, so ``n`` of them spaced ``L/n`` apart need a
    radius of ``hypot(L/2n, W/2)`` -- always more than the true half-width.
    That surplus is the *lateral* inflation, and it is the number that decides
    whether a vehicle in an adjacent lane reads as a collision:

        two discs on a 4.5 x 2.0 m car -> radius 1.51 m, a 3.0 m-wide car

    which leaves 0.29 m of clearance to a vehicle one 3.5 m lane away, so an
    ordinary lane offset presents as an imminent collision.  Choosing ``n``
    from a tolerance instead makes the error a property one can state: four
    discs bring the same car to 2.3 m, and the cost is a handful more
    distances in a single vectorised operation.

    Longitudinal over-coverage is harmless by comparison -- being a little
    cautious about the space directly in front is what the rule is for.
    """
    length = max(float(length_m), 0.5)
    width = max(float(width_m), 0.5)
    half_width = width / 2.0

    # Smallest n whose radius stays within the tolerance of the half-width.
    limit = half_width + max(float(max_inflation_m), 1e-3)
    reach_sq = limit * limit - half_width * half_width
    n = 1 if reach_sq <= 0 else max(1, math.ceil(length / (2.0 * math.sqrt(reach_sq))))
    # Bounded so a malformed size cannot produce thousands of discs. The
    # ceiling is high enough that no road vehicle reaches it -- a 14 m
    # articulated bus needs eleven -- so in practice the tolerance always
    # holds rather than being quietly traded away here.
    n = min(n, 16)

    spacing = length / n
    offsets = (np.arange(n, dtype=np.float64) + 0.5) * spacing - length / 2.0
    radius = math.hypot(spacing / 2.0, half_width)
    return offsets, radius


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
    #: ``(T,)`` that vehicle's speed along the route; 0 where there is none.
    lead_speed_mps: np.ndarray
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

    # The margin never takes more than half the distance that is left.
    #
    # CARLA only reports a light once the ego is inside its trigger volume,
    # which on Town10HD means the first sighting can be four metres from the
    # line at four metres per second. A fixed four-metre margin then demands
    # the plan stop dead where it stands, which no plan can do, so the rule is
    # violated by every candidate and the emergency fallback engages on what
    # is actually a comfortably stoppable approach. Halving instead leaves
    # room to stop while still keeping the vehicle short of the line.
    margin = min(config.limits.stop_line_margin_m, 0.5 * distance)
    stop_station = station0_m + max(0.0, distance - margin)
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
        lead_speed_mps=np.zeros((n_steps,), dtype=np.float32),
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

    centres_over_time, radii = _disc_forecast(centres, velocities, yaws, sizes_array, table, config)
    shape = centres_over_time.shape[:2]
    stations, offsets = project_to_frenet(table, centres_over_time.reshape(-1, 2))
    stations = stations.reshape(shape)
    offsets = offsets.reshape(shape)

    along = _along_route_speeds(centres, velocities, table)
    lead_station, lead_speed = _lead_station(
        stations, offsets, radii, sizes_array, along, station0_m, config
    )
    centres_over_time = _drop_traffic_behind(centres_over_time, config)

    return ActorForecast(
        centres=centres_over_time.astype(np.float32),
        radii=radii.astype(np.float32),
        lead_station_m=lead_station.astype(np.float32),
        lead_speed_mps=lead_speed.astype(np.float32),
        n_actors=len(renderer_data.actors),
    )


def _drop_traffic_behind(
    centres_over_time: np.ndarray,
    config: STLDriverConfig,
) -> np.ndarray:
    """Push discs that lie behind the ego out of the collision check.

    A vehicle queuing up behind us at a red light comes well inside the
    collision margin, and `collision_free` -- a hard rule -- then reports an
    imminent collision with something we are driving *away* from.  In dense
    traffic that fired on a fifth of all steps, latching the emergency
    fallback while the vehicle sat still and correctly waited at a light.

    Our plan only ever moves forward, so it cannot cause a collision with
    something behind the rear bumper; avoiding that is the following driver's
    problem.  Discs that later move ahead of us -- a vehicle overtaking, say
    -- reappear at the timesteps where they do, because the test is applied
    per timestep rather than once per actor.

    Excluded discs are moved far away rather than removed, which keeps the
    array rectangular for the batched distance computation downstream.

    "Behind" is measured along the rig frame's own ``+x`` -- straight ahead --
    rather than by route station.  Station would be the natural choice, but
    the route is published from the ego forward, and projecting onto it clamps
    everything behind the start to station zero: every follower would look
    like it was exactly beside us, and none would ever be dropped.
    """
    limits = config.limits
    # The rear bumper, relative to the rig origin at the rear axle.
    rear_extent = limits.ego_rear_axle_to_centre_m - limits.ego_length_m / 2.0
    threshold = rear_extent - limits.collision_margin_m

    behind = centres_over_time[..., 0] < threshold
    if not behind.any():
        return centres_over_time

    moved = centres_over_time.copy()
    moved[behind] = 1.0e4
    return moved


def _along_route_speeds(
    centres: np.ndarray,
    velocities: np.ndarray,
    table: RouteTable,
) -> np.ndarray:
    """Each actor's speed resolved along the route, which is what RSS needs.

    How much following distance we require depends on how fast the vehicle
    ahead is already travelling, so its speed has to be measured in the same
    direction as the gap: along the road, not along whatever heading it
    happens to have.
    """
    stations, offsets = project_to_frenet(table, centres)
    _, tangents = table.frenet_to_xy_many(stations, offsets)
    return np.einsum("ak,ak->a", velocities, tangents)


def _yaw_in_rig(to_rig: Pose, quat_local_xyzw: np.ndarray) -> float:
    """Heading of an actor once brought into the rig frame."""
    from carla_driver_interface.geometry import quat_mul

    return float(quat_to_yaw(quat_mul(to_rig.quat_xyzw, quat_local_xyzw)))


def _disc_forecast(
    centres: np.ndarray,
    velocities: np.ndarray,
    yaws: np.ndarray,
    sizes: np.ndarray,
    table: RouteTable,
    config: STLDriverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Cover each actor with discs and predict them over the horizon.

    A rectangle would be exact but has corners the optimizer cannot
    differentiate through.  Discs are the usual substitute; how many of them
    is the interesting choice, and :func:`footprint_discs` explains why it is
    driven by the lateral inflation rather than fixed at two.

    All actors share one disc count so the result stays a rectangular array;
    it is taken from the largest actor present, which is the conservative
    direction -- a small car simply gets a few more, tighter discs than it
    needs.

    The prediction itself is either straight-line or along the road; see
    :func:`_predict_centres`.
    """
    horizon = config.horizon
    times = np.arange(1, horizon.n_steps + 1, dtype=np.float64) * horizon.step_s

    lengths = np.maximum(sizes[:, 0], 0.5)
    widths = np.maximum(sizes[:, 1], 0.5)

    tolerance = config.limits.footprint_max_inflation_m
    n_discs = max(
        len(footprint_discs(float(length), float(width), tolerance)[0])
        for length, width in zip(lengths, widths, strict=True)
    )

    # Per-actor offsets and radii at that shared disc count.
    spacing = lengths / n_discs
    offsets = (np.arange(n_discs, dtype=np.float64)[None, :] + 0.5) * spacing[:, None] - (
        lengths[:, None] / 2.0
    )
    radii = np.hypot(spacing / 2.0, widths / 2.0)

    # (T, A, 2) centres and (T, A, 2) unit headings.
    predicted_centres, headings = _predict_centres(centres, velocities, yaws, table, times, config)

    # (T, A, n_discs, 2) -> (T, A * n_discs, 2)
    predicted = (
        predicted_centres[:, :, None, :] + offsets[None, :, :, None] * headings[:, :, None, :]
    )
    predicted = predicted.reshape(len(times), -1, 2)

    return predicted, np.repeat(radii, n_discs)


def _predict_centres(
    centres: np.ndarray,
    velocities: np.ndarray,
    yaws: np.ndarray,
    table: RouteTable,
    times: np.ndarray,
    config: STLDriverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Where each actor's centre will be, and which way it will point.

    Two models, selectable because only measurement can say which drives
    better.

    **Straight line.**  Position plus velocity times time.  Simple, and wrong
    in a specific way that matters: it ignores the road.  On a bend it sends
    every vehicle off along the tangent, so a car we are following leaves our
    lane and stops being our lead, and a car in the oncoming lane is predicted
    into our path.  Both errors grow with the square of the horizon.

    **Along the road.**  Resolve the velocity into the route's own frame --
    how fast along, how fast across -- and advance those instead, mapping back
    through the route geometry.  A vehicle travelling down the road stays on
    the road however it curves, which is the case that matters most, since
    that is the vehicle we are following.  A vehicle crossing has a large
    lateral rate and small longitudinal one, so it sweeps across the route and
    off the far side rather than being carried along it.

    Neither knows anything about the *other* vehicle's lane; the route is
    ours.  That is the honest limit of both, and the reason the lateral rate
    may be decayed: a vehicle's lateral velocity relative to our road is the
    least persistent thing about it.
    """
    limits = config.limits
    n_times = len(times)

    heading_now = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)

    straight = centres[None] + velocities[None] * times[:, None, None]
    straight_headings = np.repeat(heading_now[None], n_times, axis=0)
    if not limits.frenet_prediction:
        return straight, straight_headings

    stations, offsets = project_to_frenet(table, centres)

    # Frenet coordinates only describe the space the route actually
    # parameterises. Projection clamps at both ends, so a vehicle *behind* the
    # ego -- and the route is published from the ego forward -- comes back at
    # station zero, which would teleport every follower onto our front bumper.
    # A vehicle far off to the side is no better described: the frame folds
    # where the road bends away from it.
    #
    # Outside that region the straight line is the more honest answer, so the
    # two models are chosen per actor rather than globally.
    describable = (
        (stations > 1e-3)
        & (stations < table.table_length_m - 1e-3)
        & (np.abs(offsets) <= limits.frenet_prediction_half_width_m)
    )
    if not describable.any():
        return straight, straight_headings
    _, tangents = table.frenet_to_xy_many(stations, offsets)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    along = np.einsum("ak,ak->a", velocities, tangents)
    across = np.einsum("ak,ak->a", velocities, normals)

    # A vehicle's lateral rate relative to *our* road is the least persistent
    # part of its motion -- it is mostly the road bending, or a lane change
    # that ends. Integrating it forever manufactures conflicts; decaying it
    # keeps the first seconds, which is where a genuine cut-in shows up.
    decay = limits.prediction_lateral_decay_s
    if decay > 0.0:
        lateral_travel = across[None, :] * decay * (1.0 - np.exp(-times[:, None] / decay))
    else:
        lateral_travel = across[None, :] * times[:, None]

    predicted_s = stations[None, :] + along[None, :] * times[:, None]
    predicted_d = offsets[None, :] + lateral_travel

    positions, road_tangents = table.frenet_to_xy_many(predicted_s.ravel(), predicted_d.ravel())
    predicted = positions.reshape(n_times, -1, 2)
    road_tangents = road_tangents.reshape(n_times, -1, 2)

    # Point each vehicle the way it is actually going: along the road, turned
    # by however much of its motion is across it. A stationary vehicle has no
    # direction of travel, so it keeps the heading it was observed with.
    road_normals = np.stack([-road_tangents[..., 1], road_tangents[..., 0]], axis=-1)
    speed = np.hypot(along, across)
    moving = speed > 0.1
    direction = road_tangents * along[None, :, None] + road_normals * across[None, :, None]
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = np.divide(direction, np.maximum(norm, 1e-9))
    headings = np.where(moving[None, :, None], direction, straight_headings)

    keep = describable[None, :, None]
    return np.where(keep, predicted, straight), np.where(keep, headings, straight_headings)


def _lead_station(
    stations: np.ndarray,
    offsets: np.ndarray,
    radii: np.ndarray,
    sizes: np.ndarray,
    along: np.ndarray,
    station0_m: float,
    config: STLDriverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Rear of the nearest in-lane actor ahead, and its speed, per timestep.

    "In lane" is decided in Frenet coordinates rather than by heading, so a
    vehicle on a parallel road does not become our lead just because it points
    the same way.  Its speed comes back too, because the RSS distance in
    :func:`~stl_driver.rollout._headway_slack` depends on how fast the thing
    ahead is already travelling, not only on how fast we are.
    """
    limits = config.limits
    n_steps, n_discs = stations.shape
    if n_discs == 0:
        return np.full((n_steps,), np.inf), np.zeros((n_steps,))

    # How many discs each actor contributed. Derived rather than assumed: the
    # count follows from the footprint tolerance and the largest vehicle
    # present, so hardcoding it would silently pair discs with the speed
    # belonging to some other actor.
    per_actor = n_discs // max(1, len(sizes))

    # The *centre* has to be in our lane, not merely an edge of it. Allowing an
    # edge to qualify widens the window into the neighbouring lane, and a car
    # passing there becomes the vehicle we believe we are following -- a few
    # metres ahead, which reads as a violation however far the real lead is.
    in_lane = np.abs(offsets) < limits.lead_capture_half_width_m

    # The rear edge of each disc, which is the part of the lead we would reach
    # first. Measured from the disc's own radius rather than from half the
    # length of the largest vehicle anywhere in the scene -- with one ambulance
    # on the map, that put every lead three metres further back than it was.
    rear = stations - radii[None, :]

    # Who we are following is decided *now*, and then that vehicle is followed.
    #
    # Choosing it afresh at each step of the horizon looks equivalent and is
    # not, because the only reference available to a plan-independent forecast
    # is where the ego is at t = 0. Four seconds out the ego has driven twenty
    # metres past that mark, so a car overtaking us from behind crosses it and
    # is promptly adopted as the vehicle we are following -- at a gap measured
    # from a position we left two seconds ago, and therefore negative.
    #
    # Measured on the road, that was not a corner case. The lead station would
    # read a steady 74.6 m for most of the horizon, dip to 0.4 m for three or
    # four steps as someone passed, and return; `safe_headway` reported -16 m
    # against a road that was completely clear, and the policy braked from
    # 7 m/s to a standstill over ten steps. It accounted for nearly every
    # violation of the rule -- 19.4% of a seven-hundred-step drive -- and for
    # the stops that had no visible cause.
    #
    # Nothing is given up by fixing the identity. A vehicle that cuts in ahead
    # is picked up by the next replan a tenth of a second later, and its
    # geometry is `collision_free`'s business in the meantime -- that rule
    # watches every actor's footprint, not just the one we follow.
    ahead_now = (stations[0] > station0_m + 0.5) & in_lane[0]
    if not ahead_now.any():
        return np.full((n_steps,), np.inf), np.zeros((n_steps,))

    followed = int(np.argmin(np.where(ahead_now, rear[0], np.inf))) // per_actor
    columns = slice(followed * per_actor, (followed + 1) * per_actor)

    # Its nearest edge over the horizon, and absent once it has left our lane
    # -- a vehicle turning off is no longer something we are following, and
    # its station along *our* route stops meaning anything once it has gone.
    still_ours = in_lane[:, columns].any(axis=1)
    lead = np.where(still_ours, rear[:, columns].min(axis=1), np.inf)
    lead_speed = np.where(still_ours, float(along[followed]), 0.0)
    return lead, lead_speed
