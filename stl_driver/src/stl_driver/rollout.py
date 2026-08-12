# SPDX-License-Identifier: Apache-2.0
"""Plan parameters in, STL signal out -- differentiably.

This is the bridge between the optimizer's six numbers and the twelve columns
the specification reads.  It has to be differentiable end to end, because the
whole point of STLCG++ is that a robustness value has a gradient with respect
to the trajectory that produced it.

Two rules shape the code:

**Nothing that clamps.**  A hard ``clamp`` or ``where`` on a plan-dependent
quantity has zero gradient on one side, which tells the optimizer a parameter
does not matter when in fact it matters a great deal.  Parameters are mapped
into their physical ranges with ``sigmoid``/``tanh``, the speed floor is a
``softplus``, and the one unavoidable clamp -- the station index -- is applied
to a *detached* tensor so the live path keeps its exact derivative.

**Nothing plan-independent computed here.**  The route table, the actor
forecast, the stop-line station: all of it is precomputed in numpy and arrives
frozen in :class:`RolloutInputs`.  The optimizer calls this function about
seventy times per policy step, so anything that does not depend on ``params``
must not be inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stl_driver.config import STLDriverConfig
from stl_driver.signals import N_COLUMNS, Col, saturate

__all__ = ["N_PARAMS", "RolloutInputs", "plan_xy", "rollout"]

#: Four longitudinal acceleration knots plus two lateral control offsets.
#: Small on purpose: the optimizer explores this space by quasi-random
#: sampling, and sampling density falls off fast with dimension.
N_PARAMS = 6

#: Times, in seconds, at which the acceleration profile is pinned. The value
#: at ``t = 0`` is *not* a free parameter -- it is the acceleration the vehicle
#: currently has, so that the plan starts from the vehicle's actual state
#: rather than stepping to a new one instantaneously.
#:
#: This is not a detail. A step change in acceleration is, by definition,
#: unbounded jerk: differencing a commanded 2 m/s^2 against a measured 0 over
#: one 0.1 s tick gives 20 m/s^3, five times the comfort bound. Left free, the
#: first knot makes `comfort` unsatisfiable by *any* plan that accelerates,
#: and the search spends its whole budget picking between violations.
_KNOT_TIMES_S = (0.0, 1.0, 2.0, 3.0, 4.0)

#: Sharpness of the softplus speed floor. Higher tracks `max(v, 0)` more
#: closely but softens the gradient over a narrower band; 4.0 puts the
#: transition within a few centimetres per second of zero.
_SPEED_FLOOR_BETA = 4.0


@dataclass(frozen=True)
class RolloutInputs:
    """Everything about the current step that does not depend on the plan.

    Built once per ``drive()`` call by :mod:`stl_driver.policy`, then shared
    across every candidate the optimizer evaluates.
    """

    # -- route -------------------------------------------------------------
    #: ``(M, 2)`` uniform centreline in the rig frame.
    route_xy: torch.Tensor
    #: ``(M, 2)`` unit tangents.
    route_tangent: torch.Tensor
    #: Station spacing of the table above.
    route_ds: float
    #: Arc length of the route as published; beyond it the table extrapolates.
    published_length_m: float

    # -- ego ---------------------------------------------------------------
    #: Station and signed offset of the ego at ``t = 0``.
    station0_m: float
    offset0_m: float
    speed0_mps: float
    accel0_mps2: float

    # -- environment -------------------------------------------------------
    #: ``(T, A, 2)`` predicted actor disc centres, rig frame; ``A`` may be 0.
    actor_centres: torch.Tensor
    #: ``(A,)`` disc radii matching the second axis above.
    actor_radii: torch.Tensor
    #: How many vehicles those discs came from, for reporting.
    n_actors: int
    #: ``(E,)`` ego footprint disc offsets along the heading, measured from the
    #: rig origin (the rear axle), and their shared radius.
    ego_disc_offsets: torch.Tensor
    ego_disc_radius: float
    #: ``(T,)`` station of the vehicle we are following; ``+inf`` when none.
    lead_station_m: torch.Tensor
    #: ``(T,)`` that vehicle's speed along the route; 0 when there is none.
    lead_speed_mps: torch.Tensor
    #: Station of the stop line, when a light governs us.
    stop_station_m: float
    #: ``+gate`` when a red or amber light applies, ``-gate`` otherwise.
    light_gate: float
    #: The posted limit to hold ourselves to.
    speed_limit_mps: float
    #: Distance `makes_progress` asks the plan to cover.
    progress_target_m: float

    @property
    def n_actor_discs(self) -> int:
        return int(self.actor_radii.shape[0])


def _map_params(
    params: torch.Tensor,
    accel0_mps2: float,
    config: STLDriverConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unconstrained parameters to physical acceleration knots and offsets.

    Smooth bijections rather than clamps, so that a candidate pinned against a
    limit still reports which way it wants to move.

    The returned knots are prefixed with the measured acceleration, which is
    what anchors the profile to the vehicle's current state; see
    :data:`_KNOT_TIMES_S`.
    """
    limits = config.limits
    accel_span = limits.max_accel_mps2 + limits.max_decel_mps2
    free = -limits.max_decel_mps2 + accel_span * torch.sigmoid(params[..., :4])

    measured = torch.full_like(free[..., :1], float(accel0_mps2))
    knots = torch.cat([measured, free], dim=-1)
    offsets = limits.lane_half_width_m * torch.tanh(params[..., 4:6])
    return knots, offsets


def _interpolate_knots(knots: torch.Tensor, times_s: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear acceleration profile through the knots.

    ``knots`` holds one value per entry of :data:`_KNOT_TIMES_S`, the first of
    which is the measured acceleration rather than a free parameter.
    """
    knot_times = torch.tensor(_KNOT_TIMES_S, dtype=times_s.dtype, device=times_s.device)

    # Built as a sum of masked segments rather than by indexing: only a few
    # knots, and this stays differentiable in the knot *values* with no
    # gather, which keeps it clean under vmap.
    profile = torch.zeros_like(times_s)
    for index in range(len(_KNOT_TIMES_S) - 1):
        start_t, end_t = knot_times[index], knot_times[index + 1]
        weight = ((times_s - start_t) / (end_t - start_t)).clamp(0.0, 1.0)
        segment = knots[..., index] + weight * (knots[..., index + 1] - knots[..., index])
        # Half-open intervals, so a time landing exactly on an interior knot
        # belongs to one segment only and is not counted twice.
        inside = ((times_s > start_t) & (times_s <= end_t)).to(times_s.dtype)
        if index == 0:
            inside = ((times_s >= start_t) & (times_s <= end_t)).to(times_s.dtype)
        profile = profile + inside * segment
    # Past the final knot, hold the last value.
    beyond = (times_s > knot_times[-1]).to(times_s.dtype)
    return profile + beyond * knots[..., -1]


def _lookup_centreline(
    station: torch.Tensor,
    inputs: RolloutInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centreline point and unit tangent at each station, differentiably.

    The subtle part is the index.  ``floor`` has zero derivative almost
    everywhere, so differentiating through it would silently zero
    ``d(position)/d(station)`` -- and a policy whose lateral gradient is zero
    simply never steers, which looks like a tuning problem rather than a bug.
    Detaching the index and keeping the fractional part live gives the exact
    piecewise-linear derivative instead.
    """
    table_size = inputs.route_xy.shape[0]
    index_real = station / inputs.route_ds
    lower = torch.floor(index_real.detach()).clamp(0, table_size - 2).long()
    weight = (index_real - lower).unsqueeze(-1)

    point = inputs.route_xy[lower] * (1.0 - weight) + inputs.route_xy[lower + 1] * weight
    tangent = (
        inputs.route_tangent[lower] * (1.0 - weight) + inputs.route_tangent[lower + 1] * weight
    )
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return point, tangent


def _kinematics(
    params: torch.Tensor,
    inputs: RolloutInputs,
    config: STLDriverConfig,
) -> dict[str, torch.Tensor]:
    """Integrate the plan: speed, station, offset and their derivatives."""
    horizon = config.horizon
    dt = horizon.step_s
    device = inputs.route_xy.device
    dtype = inputs.route_xy.dtype

    times = torch.arange(1, horizon.n_steps + 1, device=device, dtype=dtype) * dt
    knots, offsets = _map_params(params, inputs.accel0_mps2, config)

    commanded = _interpolate_knots(knots, times)

    # Speed, floored at zero: a stopped car stays stopped rather than
    # reversing.
    #
    # The floor is a plain clamp rather than a softplus, and that is a
    # correctness point, not a style one. `softplus(0)` is `log 2`, so a
    # softplus floor reports a stationary vehicle as creeping at
    # `log(2)/beta` -- 0.17 m/s at beta = 4. Differencing that against a
    # measured standstill yields a phantom 1.7 m/s^2 of acceleration and
    # 17 m/s^3 of jerk on the first step, which puts `comfort` beyond reach
    # of *any* plan that pulls away from rest. A clamp is exactly zero at
    # zero.
    #
    # The clamp costs gradient below zero, where the plan is asking to
    # reverse. That region is reachable but uninteresting -- the sampler
    # covers it, and no good plan lives there.
    raw_speed = inputs.speed0_mps + torch.cumsum(commanded, dim=-1) * dt
    speed = raw_speed.clamp_min(0.0)

    station = inputs.station0_m + torch.cumsum(speed, dim=-1) * dt

    # Acceleration and jerk as *commanded*, not as differenced from the
    # floored speed.
    #
    # Differencing the floored speed sounds more honest -- it is what the
    # vehicle would feel -- but the floor is a clamp, and a clamp puts a kink
    # in the speed exactly where a plan comes to rest. One 0.1 s step later
    # that kink is a step change in acceleration and a jerk of forty-odd
    # metres per second cubed, ten times the comfort bound. Measured, this was
    # the whole of the worst `comfort` violations: -43, -66. No vehicle feels
    # that on coming to a stop; it is the discretisation of our own clamp.
    #
    # The commanded profile is piecewise linear between knots and smooth by
    # construction, and it is what the controller is being asked to achieve.
    accel = commanded
    accel0 = torch.as_tensor(inputs.accel0_mps2, device=device, dtype=dtype).reshape(1)
    accel0 = accel0.expand(*accel.shape[:-1], 1) if accel.dim() > 1 else accel0
    jerk = torch.diff(accel, dim=-1, prepend=accel0) / dt

    # Lateral offset as a function of distance travelled, not of time, so the
    # path is decoupled from the speed profile: braking must not also swerve.
    reference = max(8.0, config.cruise_speed_mps * horizon.duration_s)
    progress = ((station - inputs.station0_m) / reference).clamp(0.0, 1.0)
    # Quadratic Bezier from the *measured* offset, which keeps the plan
    # continuous with where the vehicle actually is rather than snapping to
    # the centreline every step.
    offset = (
        (1.0 - progress) ** 2 * inputs.offset0_m
        + 2.0 * progress * (1.0 - progress) * offsets[..., 0:1]
        + progress**2 * offsets[..., 1:2]
    )

    return {
        "raw_speed": raw_speed,
        "speed": speed,
        "station": station,
        "accel": accel,
        "jerk": jerk,
        "offset": offset,
    }


def plan_xy(
    params: torch.Tensor,
    inputs: RolloutInputs,
    config: STLDriverConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The planned path in the rig frame.

    Returns ``(positions, headings)`` with shapes ``(T, 2)`` and ``(T,)``.
    Shared with :func:`rollout` so that what the rules judge and what the
    vehicle is asked to drive can never drift apart.
    """
    state = _kinematics(params, inputs, config)
    point, tangent = _lookup_centreline(state["station"], inputs)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    positions = point + state["offset"].unsqueeze(-1) * normal

    velocity = _central_difference(positions, config.horizon.step_s)
    heading = torch.atan2(velocity[..., 1], velocity[..., 0])
    return positions, heading


def _central_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    """Time derivative of a ``(..., T, K)`` path, one-sided at the ends."""
    forward = values[..., 1:, :] - values[..., :-1, :]
    first = forward[..., :1, :]
    last = forward[..., -1:, :]
    interior = (values[..., 2:, :] - values[..., :-2, :]) * 0.5
    return torch.cat([first, interior, last], dim=-2) / dt


def rollout(
    params: torch.Tensor,
    inputs: RolloutInputs,
    config: STLDriverConfig,
) -> torch.Tensor:
    """Turn plan parameters into the ``[T, D]`` signal the rules read.

    Pure: no state, no side effects, safe under ``vmap`` over a leading batch
    of ``params``.
    """
    limits = config.limits
    horizon = config.horizon
    cap = limits.saturation_m

    state = _kinematics(params, inputs, config)
    speed = state["speed"]
    station = state["station"]

    point, tangent = _lookup_centreline(station, inputs)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    positions = point + state["offset"].unsqueeze(-1) * normal

    velocity = _central_difference(positions, horizon.step_s)
    acceleration = _central_difference(velocity, horizon.step_s)
    # Lateral acceleration as (v x a) / |v| rather than the textbook
    # v^2 * curvature with its (vx^2+vy^2)^{3/2} denominator: algebraically the
    # same, but far better conditioned at low speed, where the cubed form
    # divides by an epsilon and reports a swerve that is not there.
    cross = velocity[..., 0] * acceleration[..., 1] - velocity[..., 1] * acceleration[..., 0]
    lateral_accel = cross / velocity.norm(dim=-1).clamp_min(0.1)

    clearance = _clearance(positions, tangent, inputs, config)
    headway = _headway_slack(station, speed, inputs, config)

    stop_slack = saturate(
        torch.as_tensor(inputs.stop_station_m, dtype=station.dtype, device=station.device)
        - station,
        cap,
    )
    route_slack = saturate(
        torch.as_tensor(inputs.published_length_m, dtype=station.dtype, device=station.device)
        - station,
        cap,
    )
    progress_slack = saturate(
        (station - inputs.station0_m) - inputs.progress_target_m,
        cap,
    )
    speed_headroom = inputs.speed_limit_mps - speed

    light_gate = torch.full_like(speed, float(inputs.light_gate))

    # Keyed by column so the assembly cannot silently transpose two of them,
    # then ordered by the enum's own values.
    by_column: dict[Col, torch.Tensor] = {
        # The physical speed, not the raw one: a plan that brakes to a stop
        # and holds has a raw speed that keeps falling past zero, and
        # `no_reverse` would report a parked vehicle as reversing. A
        # specification that punishes the right answer is worse than none.
        Col.SPEED: speed,
        Col.ACCEL: state["accel"],
        Col.JERK: state["jerk"],
        Col.LATERAL_OFFSET: state["offset"],
        Col.LATERAL_ACCEL: lateral_accel,
        Col.CLEARANCE: clearance,
        Col.HEADWAY_SLACK: headway,
        Col.STOP_SLACK: stop_slack,
        Col.LIGHT_GATE: light_gate,
        Col.ROUTE_SLACK: route_slack,
        Col.PROGRESS_SLACK: progress_slack,
        Col.SPEED_HEADROOM: speed_headroom,
    }
    if len(by_column) != N_COLUMNS:
        raise AssertionError(f"expected {N_COLUMNS} columns, built {len(by_column)}")
    return torch.stack([by_column[column] for column in Col], dim=-1)


def _clearance(
    positions: torch.Tensor,
    tangent: torch.Tensor,
    inputs: RolloutInputs,
    config: STLDriverConfig,
) -> torch.Tensor:
    """Distance from the ego footprint to the nearest predicted actor.

    The ego is two discs placed *ahead* of the plan point, because the rig
    origin is the rear axle rather than the vehicle centre; centring them on
    the origin would over-report forward clearance by most of a car length.

    Aggregation is the exact minimum, and the reasoning is worth recording
    because a soft minimum is the tempting choice here.

    A soft minimum would spread gradient across every nearby actor rather
    than only the closest, which sounds strictly better.  But both forms
    carry an error that scales with ``log(N)/beta``, and ``N`` is the traffic
    density: unnormalised, it under-reports clearance -- a phantom collision
    that appears as the road gets busy, freezing the policy in traffic;
    normalised, it over-reports by the same amount, which at forty discs is
    0.46 m of the 0.6 m collision margin silently spent.  Neither is
    acceptable for the one rule whose job is to keep the vehicle out of other
    vehicles.

    The exact minimum has no such error.  Its gradient reaches only the
    closest disc, which is the standard behaviour of a min-max problem and
    converges perfectly well as the plan moves and the argument of the
    minimum changes -- and the temporal smoothing inside ``Always`` still
    spreads gradient across timesteps.
    """
    limits = config.limits
    cap = limits.saturation_m

    if inputs.n_actor_discs == 0:
        # Exactly `cap`, which `saturate` can never reach, so "no actor" is
        # distinguishable from "an actor a very long way off".
        return torch.full(positions.shape[:-1], cap, dtype=positions.dtype, device=positions.device)

    offsets = inputs.ego_disc_offsets.to(device=positions.device, dtype=positions.dtype)
    # (T, E, 2): one ego disc centre per offset, laid out along the heading.
    # The offsets are measured from the rig origin, which is the rear axle --
    # `RolloutInputs` has already shifted them from body-centre, so they run
    # from behind the origin to well ahead of it.
    ego_centres = positions.unsqueeze(-2) + offsets.reshape(-1, 1) * tangent.unsqueeze(-2)

    # (T, E, A)
    separation = ego_centres.unsqueeze(-2) - inputs.actor_centres.unsqueeze(-3)
    distance = separation.norm(dim=-1)
    gap = distance - inputs.ego_disc_radius - inputs.actor_radii
    gap = gap - limits.collision_margin_m

    flattened = gap.reshape(*gap.shape[:-2], -1)
    return saturate(flattened.min(dim=-1).values, cap)


def _headway_slack(
    station: torch.Tensor,
    speed: torch.Tensor,
    inputs: RolloutInputs,
    config: STLDriverConfig,
) -> torch.Tensor:
    """Gap to the lead vehicle beyond the RSS safe longitudinal distance.

    The question RSS asks is: if the vehicle ahead braked as hard as it
    possibly could, starting now, could we still stop behind it -- given that
    we take some time to react, might still be accelerating while we do, and
    then brake at a rate we are willing to guarantee?

        d = v_r*rho + a_accel*rho^2/2 + (v_r + rho*a_accel)^2 / 2*b_min
                                                 - v_f^2 / 2*b_max

    The final term is what a fixed time headway leaves out, and it is the
    whole difference in character.  It credits the vehicle ahead for the
    distance its *own* braking will take, so the requirement depends on the
    two speeds rather than on ours alone.  Following at a matched speed asks
    for much less than `time * v + gap`, and a lead pulling away asks for
    almost nothing -- which is the behaviour a driver would recognise, and
    what stops the policy sitting far behind traffic for no reason.

    It is not less safe: with the lead stopped, ``v_f`` is zero, the term
    vanishes, and the requirement is the full stopping distance again.

    See Shalev-Shwartz, Shammah & Shashua, *On a Formal Model of Safe and
    Scalable Self-Driving Cars* (2017), and Hekmatnejad et al., *Encoding and
    Monitoring Responsibility Sensitive Safety Rules for Automated Vehicles in
    Signal Temporal Logic* (MEMOCODE 2019), for the same model written as STL.
    """
    limits = config.limits
    cap = limits.saturation_m

    lead = inputs.lead_station_m.to(station.dtype)
    absent = ~torch.isfinite(lead)
    # Substituted before the arithmetic, not after: `inf - station` would be
    # `inf`, and `saturate(inf)` is `nan`, which poisons the whole signal.
    lead = torch.where(absent, station.detach() + cap * 2.0, lead)
    lead_speed = inputs.lead_speed_mps.to(station.dtype).clamp_min(0.0)

    rho = limits.rss_response_s
    accel = limits.rss_response_accel_mps2
    brake = limits.rss_brake_mps2
    lead_brake = limits.rss_lead_brake_mps2

    required = (
        speed * rho
        + 0.5 * accel * rho * rho
        + (speed + rho * accel) ** 2 / (2.0 * brake)
        - lead_speed**2 / (2.0 * lead_brake)
    )
    # RSS can call for nothing at all when the lead is quicker than us; we
    # still decline to ride its bumper.
    required = required.clamp_min(limits.standstill_gap_m)

    slack = saturate((lead - station) - required, cap)
    # Saturate first, then substitute, so that "no lead vehicle" reads as
    # exactly `cap` -- the same value `_clearance` reports when there are no
    # actors.
    return torch.where(absent, torch.full_like(slack, cap), slack)
