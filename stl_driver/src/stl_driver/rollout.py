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

__all__ = ["N_PARAMS", "RolloutInputs", "knot_step_limit", "plan_xy", "rollout"]

#: Four longitudinal acceleration knots plus two lateral control offsets.
#: Small on purpose: the optimizer explores this space by quasi-random
#: sampling, and sampling density falls off fast with dimension.
N_PARAMS = 6

#: How many points the acceleration profile is pinned at, including ``t = 0``.
#: Four of them are free parameters; see `N_PARAMS`.
_N_KNOTS = 5


def knot_times_s(horizon) -> tuple[float, ...]:
    """Times at which the acceleration profile is pinned, spread over the horizon.

    The value at ``t = 0`` is *not* a free parameter -- it is the acceleration
    the vehicle currently has, so that the plan starts from the vehicle's
    actual state rather than stepping to a new one instantaneously.

    This is not a detail. A step change in acceleration is, by definition,
    unbounded jerk: differencing a commanded 2 m/s^2 against a measured 0 over
    one 0.1 s tick gives 20 m/s^3, five times the comfort bound. Left free, the
    first knot makes `comfort` unsatisfiable by *any* plan that accelerates,
    and the search spends its whole budget picking between violations.

    Spread over whatever the horizon is rather than fixed at one second apart,
    so that lengthening the horizon lengthens the plan rather than appending
    time the parameters cannot shape -- past the final knot the profile holds
    its last value, which for a four-knot plan on an eight-second horizon is
    four seconds of constant acceleration nobody chose.
    """
    spacing = horizon.duration_s / (_N_KNOTS - 1)
    return tuple(index * spacing for index in range(_N_KNOTS))

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
    #: ``(M,)`` signed route curvature, positive for a left turn.
    route_curvature: torch.Tensor
    #: Station spacing of the table above.
    route_ds: float
    #: Arc length of the route as published; beyond it the table extrapolates.
    #:
    #: A 0-d tensor, like every scalar below that reaches the arithmetic, and
    #: for one reason: ``torch.compile`` specialises on the *value* of a Python
    #: float it sees, so a quantity that differs every step generates a fresh
    #: graph every step. Measured, that took a 400-step drive from 57.9 s to
    #: 200.5 s, dynamo giving up at its recompilation limit with this field
    #: named as the cause. Tensors are traced as inputs, not baked in.
    published_length_m: torch.Tensor

    # -- ego ---------------------------------------------------------------
    #: Station and signed offset of the ego at ``t = 0``.
    station0_m: torch.Tensor
    offset0_m: torch.Tensor
    speed0_mps: torch.Tensor
    accel0_mps2: torch.Tensor

    # -- environment -------------------------------------------------------
    #: ``(T, A, 2)`` predicted actor disc centres, rig frame; ``A`` may be 0.
    actor_centres: torch.Tensor
    #: ``(A,)`` disc radii matching the second axis above.
    actor_radii: torch.Tensor
    #: How many vehicles those discs came from, for reporting.
    n_actors: int
    #: How many of the discs above are real rather than padding, for reporting.
    #: The tensors are a fixed width so that there is one graph shape to
    #: compile, which makes their own width uninformative -- and would make a
    #: scene busy enough to be truncated look exactly like any other. See
    #: `stl_driver.perception.pad_actor_discs`.
    n_forecast_discs: int
    #: ``(E,)`` ego footprint disc offsets along the heading, measured from the
    #: rig origin (the rear axle), and their shared radius.
    ego_disc_offsets: torch.Tensor
    ego_disc_radius: float
    #: ``(T,)`` station of the vehicle we are following; ``+inf`` when none.
    lead_station_m: torch.Tensor
    #: ``(T,)`` that vehicle's speed along the route; 0 when there is none.
    lead_speed_mps: torch.Tensor
    #: Station of the stop line, when a light governs us; ``+inf`` when none.
    stop_station_m: torch.Tensor
    #: ``+gate`` when a red or amber light applies, ``-gate`` otherwise.
    light_gate: torch.Tensor
    #: The posted limit to hold ourselves to.
    speed_limit_mps: torch.Tensor
    #: Distance `makes_progress` asks the plan to cover.
    progress_target_m: torch.Tensor

    @property
    def n_actor_discs(self) -> int:
        return int(self.actor_radii.shape[0])


def knot_step_limit(limits, horizon) -> float:
    """The largest acceleration change one knot interval may carry.

    The profile is piecewise linear between knots, so its slope -- the jerk --
    is a knot-to-knot change divided by the interval. Bounding the change by
    ``plan_jerk_limit() * interval`` bounds the jerk.

    The inset that keeps this off the comfort bound belongs to `Limits`, along
    with the matching one on acceleration, so that this and everything else
    anchoring a plan cannot drift apart -- see `Limits.plan_accel_envelope`.
    """
    spacing = horizon.duration_s / (_N_KNOTS - 1)
    return limits.plan_jerk_limit() * spacing


def _map_params(
    params: torch.Tensor,
    accel0_mps2: torch.Tensor,
    config: STLDriverConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unconstrained parameters to physical acceleration knots and offsets.

    Smooth bijections rather than clamps, so that a candidate pinned against a
    limit still reports which way it wants to move.

    The knots are parameterised as *changes* from the measured acceleration
    onwards, each bounded by what the jerk limit allows over one knot interval,
    rather than as absolute values.  The profile is piecewise linear between
    knots, so its slope -- the jerk -- is exactly a knot-to-knot change divided
    by the interval; bounding the change bounds the jerk by construction.

    Left as absolute values, the reachable set includes plans no vehicle should
    execute, and the search spends its budget deciding how badly to break the
    comfort bound instead of how best to drive.  It showed up on the road as
    the vehicle failing to pull away after a slowdown: from a measured
    -4 m/s^2, asking for +2 is a change of 6 across one second, half again the
    jerk limit, so *every* way of accelerating away scored as a violation.

    That does make `comfort`'s jerk term true by construction, in the way
    `no_reverse` is. The rule stays because it states the requirement rather
    than leaving it as an unwritten property of the parameterisation.
    """
    limits = config.limits
    step_limit = knot_step_limit(limits, config.horizon)

    # Start from the measured acceleration and walk, one bounded step per knot.
    measured = torch.zeros_like(params[..., :1]) + accel0_mps2
    deltas = step_limit * torch.tanh(params[..., :4])
    knots = torch.cumsum(torch.cat([measured, deltas], dim=-1), dim=-1)
    # The walk must still land inside what the vehicle can do -- just inside,
    # for the same reason the step is: clamping exactly onto the bound leaves
    # `comfort` reading exactly zero, satisfied with no margin at all, and
    # every accelerating plan pinned there.
    lowest, highest = limits.plan_accel_envelope()
    knots = knots.clamp(lowest, highest)

    offsets = limits.lane_half_width_m * torch.tanh(params[..., 4:6])
    return knots, offsets


def _interpolate_knots(
    knots: torch.Tensor, times_s: torch.Tensor, times: tuple[float, ...]
) -> torch.Tensor:
    """Piecewise-linear acceleration profile through the knots.

    ``knots`` holds one value per entry of :func:`knot_times_s`, the first of
    which is the measured acceleration rather than a free parameter.
    """
    knot_times = torch.tensor(times, dtype=times_s.dtype, device=times_s.device)

    # Built as a sum of masked segments rather than by indexing: only a few
    # knots, and this stays differentiable in the knot *values* with no
    # gather, which keeps it clean under vmap.
    profile = torch.zeros_like(times_s)
    for index in range(len(times) - 1):
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
    curvature = (
        inputs.route_curvature[lower] * (1.0 - weight[..., 0])
        + inputs.route_curvature[lower + 1] * weight[..., 0]
    )
    return point, tangent, curvature


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

    commanded = _interpolate_knots(knots, times, knot_times_s(horizon))

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
    accel0 = inputs.accel0_mps2.reshape(1)
    accel0 = accel0.expand(*accel.shape[:-1], 1) if accel.dim() > 1 else accel0
    jerk = torch.diff(accel, dim=-1, prepend=accel0) / dt

    # Lateral offset as a function of distance travelled, not of time, so the
    # path is decoupled from the speed profile: braking must not also swerve.
    reference = max(8.0, config.cruise_speed_mps * horizon.duration_s)
    raw_progress = (station - inputs.station0_m) / reference
    progress = raw_progress.clamp(0.0, 1.0)
    # Quadratic Bezier from the *measured* offset, which keeps the plan
    # continuous with where the vehicle actually is rather than snapping to
    # the centreline every step.
    offset = (
        (1.0 - progress) ** 2 * inputs.offset0_m
        + 2.0 * progress * (1.0 - progress) * offsets[..., 0:1]
        + progress**2 * offsets[..., 1:2]
    )

    # How sharply the offset profile itself bends, per metre of road.
    #
    # A quadratic Bezier has a constant second derivative in its parameter, so
    # this is one expression rather than a difference -- and being analytic it
    # is exact, where differencing the sampled path was not. Outside the
    # clamped span the offset is held constant, so it bends not at all.
    inside = ((raw_progress > 0.0) & (raw_progress < 1.0)).to(dtype)
    offset_ss = inside * (
        2.0 * (inputs.offset0_m - 2.0 * offsets[..., 0:1] + offsets[..., 1:2]) / reference**2
    )

    return {
        "raw_speed": raw_speed,
        "speed": speed,
        "station": station,
        "accel": accel,
        "jerk": jerk,
        "offset": offset,
        "offset_ss": offset_ss,
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
    point, tangent, _ = _lookup_centreline(state["station"], inputs)
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
    cap = limits.saturation_m

    state = _kinematics(params, inputs, config)
    speed = state["speed"]
    station = state["station"]

    point, tangent, curvature = _lookup_centreline(station, inputs)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    positions = point + state["offset"].unsqueeze(-1) * normal

    # Lateral acceleration in the road's own frame: the curvature the plan
    # actually follows, times speed squared.
    #
    # Two terms. Riding at offset `d` on a bend of curvature `k` tightens the
    # radius, because a positive offset is towards the inside of a left turn:
    # `k / (1 - k*d)`. And steering across the road adds the curvature of the
    # offset profile itself, `d''(s)`. Cross terms in `d'` are dropped -- the
    # offset moves about a metre over thirty, so `d'^2` is under a percent and
    # buys nothing for the arithmetic it costs.
    #
    # This replaces differencing the planned positions twice, which aliased
    # the polyline's vertices into the answer; see `frenet.build_route_table`.
    denominator = (1.0 - curvature * state["offset"]).clamp_min(0.1)
    lateral_accel = speed**2 * (curvature / denominator + state["offset_ss"])

    clearance = _clearance(positions, tangent, inputs, config)
    headway = _headway_slack(station, speed, inputs, config)

    stop_slack = _stop_slack(station, speed, inputs, config)
    route_slack = saturate(
        inputs.published_length_m - station,
        cap,
    )
    # What to ask for is the lesser of what free-flow travel would cover and
    # what the stop line leaves room for. Asking for the full target while a
    # red light is 15 m away is asking for the plan to run it, and reporting
    # the refusal as a liveness failure -- measured, 75 steps of a 400-step
    # drive once the light stopped excusing them wholesale.
    #
    # Reducing the demand rather than switching the rule off is what keeps the
    # vehicle moving up to the line. A rule made vacuous contributes a constant,
    # and a constant has no gradient towards anything: gated off entirely, the
    # vehicle stopped 18.7 m short of a line it could see and waited there.
    # This still rewards every metre gained, and stops rewarding them exactly
    # at the line.
    # Less the room the final stop itself takes: the plan has to be *able* to
    # stop at the line the whole way in, so the ground it can still cover is
    # the distance to the line less that. Measured from the vehicle's own
    # speed rather than the plan's, so every candidate is asked for the same
    # thing and none can lower the bar by slowing down.
    _, brake, ramp = _stopping_terms(config.limits)
    stopping = inputs.speed0_mps**2 / (2.0 * brake) + inputs.speed0_mps * ramp
    # Floored a little below zero rather than at it, so that a vehicle stopped
    # at the line is asked for slightly less than nothing and reads as
    # satisfied. At exactly zero it reads exactly zero -- neither satisfied nor
    # violated -- which is no way to report the most correct thing it does.
    floor = -config.limits.liveness_stop_gap_m
    reachable = (inputs.stop_station_m - inputs.station0_m - stopping).clamp_min(floor)
    demanded = torch.minimum(inputs.progress_target_m, reachable)
    progress_slack = saturate(
        (station - inputs.station0_m) - demanded,
        cap,
    )
    # Capped above, and only above: being further under the limit than the
    # tolerance is not more legal, and the objective was paying as though it
    # were.
    #
    # `speed_limit` is a *constraint*, but its contribution grew smoothly with
    # every metre per second given up, and `makes_progress` saturates. Swept on
    # an empty straight road under an 8 m/s limit, the two crossed at 6.5 m/s:
    # from 6.0 to 8.0 the progress term gained 0.33 and the speed term lost
    # 1.05, so the objective's happiest cruise was a metre and a half under the
    # limit and approaching the limit made every plan score worse. On the road
    # that showed up as decelerations with nothing within 50 m to explain them.
    #
    # The violation side is untouched: below zero this reads its true
    # magnitude, so exceeding the limit costs exactly what it did. A clamp on
    # the other side would have been simpler and is wrong for the reason
    # `RuleSet.objective` gives about clamps generally -- it takes the gradient
    # to exactly zero, and a rule with no gradient anywhere it is comfortably
    # satisfied is a rule that cannot start pushing back until it is too late.
    # `tanh` matches the identity in value and slope at zero and decays from
    # there.
    raw_headroom = inputs.speed_limit_mps - speed
    scale = limits.speed_headroom_scale_mps
    speed_headroom = torch.where(
        raw_headroom > 0.0,
        scale * torch.tanh(raw_headroom / scale),
        raw_headroom,
    )

    light_gate = torch.zeros_like(speed) + inputs.light_gate

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


def _stopping_terms(limits) -> tuple[float, float, float]:
    """``(envelope floor, braking rate, ramp coefficient)`` for a stop.

    Stopping takes ``v^2 / 2b`` plus ``v * b / 2j`` -- the textbook distance
    and what the jerk-limited ramp to full braking costs. At 5 m/s those are
    3.2 m and 2.5 m, so the ramp is not a correction but half the answer.

    Shared so that the rule asking whether there is room to stop and the rule
    asking how much ground is left to cover cannot disagree about what
    stopping costs.
    """
    lowest, _ = limits.plan_accel_envelope()
    brake = -lowest
    return lowest, brake, brake / (2.0 * limits.plan_jerk_limit())


def _stop_slack(
    station: torch.Tensor,
    speed: torch.Tensor,
    inputs: RolloutInputs,
    config: STLDriverConfig,
) -> torch.Tensor:
    """Room left to stop at the line, rather than distance left to it.

    `stop_on_red` used to read the distance itself, which asks the plan never
    to reach the line inside its horizon -- and a horizon is four seconds
    long. Approaching at 5 m/s, "do not cover the next 20 m" is what the rule
    demanded, so the vehicle slowed while the line was still far away and
    crept the rest, stopping well short of it. That is not what a red light
    asks for. It asks that you be able to stop at the line, and being able to
    is a bound on speed, not on position.

    So the requirement is the distance stopping actually takes, and the slack
    is what is left over. Told the same way as the RSS gap in
    `_headway_slack`, and jerk-limited for the same reason the plan is:
    braking cannot appear instantly, and the ramp is worth more than the
    arithmetic suggests. From ``b`` the plan may command and ``j`` the jerk
    it may use,

        d = v^2 / 2b   +   v * b / 2j

    where the first term is the textbook stopping distance and the second is
    what the ramp to full braking costs. At 5 m/s they are 3.2 m and 2.5 m,
    so leaving the ramp out would have understated the requirement by nearly
    half and put the line under the bumper.

    Approaching at any speed is then allowed until the room runs out, which
    is how the line is arrived at rather than crept up on -- and at rest the
    requirement is zero, so stopping anywhere short of the line satisfies it.
    """
    limits = config.limits
    _, brake, ramp = _stopping_terms(limits)
    required = speed**2 / (2.0 * brake) + speed * ramp
    return saturate((inputs.stop_station_m - station) - required, limits.saturation_m)


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
