# SPDX-License-Identifier: Apache-2.0
"""The differentiable model.

Two kinds of test here.  The analytic ones check that the kinematics mean what
they say.  The gradient ones check that the model is differentiable *in the
way the optimizer needs* -- which is a separate question, and the one that
actually bites: a rollout can produce perfectly sensible trajectories while
silently reporting a zero gradient, and the only symptom is a policy that
never steers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stl_driver.config import Horizon, STLDriverConfig
from stl_driver.rollout import N_PARAMS, plan_xy, rollout
from stl_driver.rules import RuleSet, build_rules
from stl_driver.signals import N_COLUMNS, Col
from stl_driver.stl_backend import smooth
from tests.conftest import curved_route, make_inputs, stationary_actor, straight_route


def hold_speed_params() -> torch.Tensor:
    """Parameters meaning "hold this acceleration, stay on the centreline".

    The profile is parameterised as changes from the measured acceleration, so
    "no change" is simply zero.
    """
    return torch.zeros(N_PARAMS, dtype=torch.float32)


def constant_accel_params(accel: float, accel0: float = 0.0) -> torch.Tensor:
    """Parameters that walk the profile to a requested acceleration."""
    from stl_driver.optimizer import _accel_params

    config = STLDriverConfig()
    return _accel_params(accel, config.limits, accel0, config.horizon)


# ---------------------------------------------------------------------------
# Shape and kinematics
# ---------------------------------------------------------------------------


def test_the_signal_has_the_shape_the_rules_expect(config):
    signal = rollout(hold_speed_params(), make_inputs(config), config)
    assert signal.shape == (config.horizon.n_steps, N_COLUMNS)
    assert torch.isfinite(signal).all()


def test_holding_speed_keeps_the_speed(config):
    inputs = make_inputs(config, speed0_mps=7.0)
    signal = rollout(hold_speed_params(), inputs, config)
    assert torch.allclose(
        signal[:, Col.SPEED], torch.full((config.horizon.n_steps,), 7.0), atol=1e-3
    )


def test_acceleration_ramps_in_from_the_measured_value(config):
    """The profile starts at the acceleration the vehicle has, not at the knot.

    So "constant acceleration parameters" describe a ramp from the measured
    acceleration to the requested one over the first knot interval, and only
    then a constant. That is what keeps the opening jerk finite -- see
    `_KNOT_TIMES_S`.
    """
    inputs = make_inputs(config, speed0_mps=2.0, accel0_mps2=0.0)
    signal = rollout(constant_accel_params(1.0), inputs, config)

    def expected_speed(t: float, a: float = 1.0, v0: float = 2.0) -> float:
        if t <= 1.0:  # linear ramp 0 -> a
            return v0 + a * t * t / 2.0
        return v0 + a / 2.0 + a * (t - 1.0)

    dt = config.horizon.step_s
    for index in (9, 19, 39):
        t = (index + 1) * dt
        assert signal[index, Col.SPEED].item() == pytest.approx(expected_speed(t), abs=0.06)


def test_the_opening_jerk_is_bounded(config):
    """The bug that made `comfort` unsatisfiable by every accelerating plan.

    With a free first knot, the commanded acceleration steps away from the
    measured one inside a single 0.1 s tick, and differencing gives ten times
    the knot value in jerk -- 20 m/s^3 for a 2 m/s^2 request, five times the
    comfort bound. Pinning the first knot to the measurement bounds it.
    """
    inputs = make_inputs(config, speed0_mps=0.0, accel0_mps2=0.0)
    signal = rollout(constant_accel_params(config.limits.max_accel_mps2 * 0.95), inputs, config)
    assert signal[:, Col.JERK].abs().max().item() < config.limits.max_jerk_mps3


def test_no_measurable_state_can_force_a_jerk_violation(config):
    """`comfort` must be reachable from wherever the vehicle actually is.

    The plan is anchored at the measured acceleration and the opening jerk is
    the difference between that and the first knot over one 0.1 s tick, so any
    disagreement about the envelope arrives in the jerk ten times over. The
    two disagreed by the inset -- the measurement was confined to the comfort
    bounds and the knot to 98% of them -- which made `comfort` unsatisfiable
    by *every* candidate on the 15 steps of a 400-step drive that followed a
    hard stop, all of them with the measured acceleration pinned at -4.0.

    Swept over the whole envelope and both extremes of the plan, because the
    failure only appeared at its edges.
    """
    from stl_driver.policy import EgoPolicy
    from stl_driver.rollout import knot_step_limit

    limits = config.limits
    policy = EgoPolicy(config=config)
    hardest = torch.tensor([3.0, 3.0, 3.0, 3.0, 0.0, 0.0])  # tanh saturates: full steps
    for measured in np.linspace(-limits.max_decel_mps2, limits.max_accel_mps2, 17):
        # What the policy would actually hand the rollout for this reading.
        # Reset first: the anchor is rate-limited between steps, and this is
        # sweeping states the vehicle might be *in*, not a sequence it passes
        # through.
        policy._anchor_accel_mps2 = None
        accel0 = policy._plausible_acceleration(float(measured))
        inputs = make_inputs(config, speed0_mps=4.0, accel0_mps2=accel0)
        for params in (hardest, -hardest):
            signal = rollout(params, inputs, config)
            assert (
                signal[:, Col.JERK].abs().max().item() <= limits.max_jerk_mps3
            ), f"measured {measured:.2f} -> accel0 {accel0:.3f}"

    # And the bound is the knot step itself, not something looser that happens
    # to fit: one knot interval of change is the most a plan can ask for.
    assert knot_step_limit(limits, config.horizon) < limits.max_jerk_mps3


def test_a_standstill_is_reported_as_a_standstill(config):
    """A softplus floor would report a stopped car creeping at log(2)/beta.

    That phantom speed becomes a phantom acceleration and a large phantom
    jerk the moment it is differenced against a measured standstill.
    """
    inputs = make_inputs(config, speed0_mps=0.0, accel0_mps2=0.0)
    signal = rollout(constant_accel_params(0.0), inputs, config)
    assert signal[:, Col.SPEED].abs().max().item() < 1e-5
    assert signal[:, Col.ACCEL].abs().max().item() < 1e-4
    assert signal[:, Col.JERK].abs().max().item() < 1e-3


def test_the_speed_floor_stops_the_vehicle_rather_than_reversing_it(config):
    """Braking hard from a crawl settles at rest: the station stops advancing.

    The physical claim is about the station, not about the speed column --
    see the next test for why those differ.
    """
    inputs = make_inputs(config, speed0_mps=0.5)
    signal = rollout(constant_accel_params(-4.0), inputs, config)
    station = signal[:, Col.PROGRESS_SLACK]  # monotone in station
    assert torch.all(torch.diff(station) >= -1e-4)
    # Having stopped, it stays stopped: the last second covers no ground.
    assert (station[-1] - station[-10]).abs().item() < 0.05


def test_braking_to_a_stop_is_not_reported_as_reversing(config):
    """The speed column is the physical speed, and this is why.

    A plan that brakes to a halt and holds there has a *raw* speed that keeps
    falling past zero for the rest of the horizon. Exposing that to the rules
    would make `no_reverse` -- a hard rule, so one that triggers the emergency
    fallback -- fire on a vehicle that is simply parked. A specification that
    punishes the correct answer is worse than no specification.
    """
    from stl_driver.rules import RuleSet, build_rules

    inputs = make_inputs(config, speed0_mps=0.5)
    signal = rollout(constant_accel_params(-4.0), inputs, config)

    assert signal[:, Col.SPEED].min().item() >= 0.0
    rules = RuleSet(build_rules(config))
    assert float(rules.per_rule(signal)["no_reverse"]) > 0.0


def test_no_lateral_input_stays_on_the_centreline_of_a_straight_route(config):
    inputs = make_inputs(config)
    signal = rollout(hold_speed_params(), inputs, config)
    assert torch.allclose(
        signal[:, Col.LATERAL_OFFSET], torch.zeros(config.horizon.n_steps), atol=1e-5
    )
    positions, heading = plan_xy(hold_speed_params(), inputs, config)
    assert torch.allclose(positions[:, 1], torch.zeros(config.horizon.n_steps), atol=1e-4)
    assert torch.allclose(heading, torch.zeros(config.horizon.n_steps), atol=1e-4)


def test_the_plan_starts_from_the_measured_lateral_offset(config):
    """C0 continuity with where the vehicle actually is.

    Snapping to the centreline at t=0 would ask the controller to teleport.
    """
    inputs = make_inputs(config, offset0_m=0.6)
    signal = rollout(hold_speed_params(), inputs, config)
    assert signal[0, Col.LATERAL_OFFSET].item() == pytest.approx(0.6, abs=0.05)


def test_a_curve_produces_lateral_acceleration(config):
    """And a straight road does not -- the low-speed conditioning check."""
    straight = rollout(hold_speed_params(), make_inputs(config), config)
    assert straight[:, Col.LATERAL_ACCEL].abs().max().item() < 0.05

    curved = rollout(
        hold_speed_params(),
        make_inputs(config, route=curved_route(radius_m=40.0), speed0_mps=8.0),
        config,
    )
    # v^2 / r = 64 / 40 = 1.6 m/s^2, give or take the discretisation.
    assert curved[:, Col.LATERAL_ACCEL].abs().max().item() == pytest.approx(1.6, rel=0.25)


def test_lateral_acceleration_stays_finite_at_a_standstill(config):
    """The reason for dividing by |v| rather than |v|^3.

    A stopped plan on a curve must not report an infinite swerve.
    """
    inputs = make_inputs(config, route=curved_route(radius_m=20.0), speed0_mps=0.0)
    signal = rollout(constant_accel_params(0.0), inputs, config)
    assert torch.isfinite(signal[:, Col.LATERAL_ACCEL]).all()
    assert signal[:, Col.LATERAL_ACCEL].abs().max().item() < 1.0


# ---------------------------------------------------------------------------
# Environment columns
# ---------------------------------------------------------------------------


def test_clearance_saturates_when_the_road_is_empty(config):
    signal = rollout(hold_speed_params(), make_inputs(config), config)
    assert torch.allclose(
        signal[:, Col.CLEARANCE],
        torch.full((config.horizon.n_steps,), config.limits.saturation_m),
    )


def test_clearance_falls_as_the_plan_approaches_an_obstacle(config):
    centres, radii = stationary_actor(config, ahead_m=25.0)
    inputs = make_inputs(config, speed0_mps=6.0, actor_centres=centres, actor_radii=radii)
    signal = rollout(hold_speed_params(), inputs, config)
    clearance = signal[:, Col.CLEARANCE]
    assert clearance[0] > clearance[-1]
    assert clearance.min().item() < 5.0


def test_clearance_accounts_for_the_rear_axle_origin(config):
    """The ego discs sit ahead of the plan point, not centred on it.

    Centring them would over-report forward clearance by most of a car
    length -- which is the difference between stopping and not.
    """
    centres, radii = stationary_actor(config, ahead_m=10.0, radius_m=1.0)
    inputs = make_inputs(config, speed0_mps=0.0, actor_centres=centres, actor_radii=radii)
    signal = rollout(constant_accel_params(0.0), inputs, config)

    limits = config.limits
    # Nearest ego disc is at +3.5 m, so the true gap is 10 - 3.5 - r_ego - r_a.
    ego_offsets, ego_radius = config.ego_footprint()
    expected = 10.0 - max(ego_offsets) - ego_radius - 1.0
    expected -= limits.collision_margin_m
    # Tolerance is `saturate`'s tanh compression, not the minimum itself.
    assert signal[0, Col.CLEARANCE].item() == pytest.approx(expected, rel=1e-3)


def test_clearance_is_exact_in_dense_traffic(config):
    """No error that scales with the number of vehicles on the road.

    A soft minimum, in either its normalised or unnormalised form, is wrong by
    up to log(N)/beta -- with twenty vehicles that is 0.46 m, most of the
    collision margin, spent silently and in a direction that depends on which
    form was chosen. The exact minimum simply does not have the problem.
    """
    n_actors = 20
    generator = torch.Generator().manual_seed(3)
    centres = torch.rand((n_actors, 2), generator=generator) * torch.tensor([60.0, 12.0])
    centres[:, 1] -= 6.0
    centres = centres.unsqueeze(0).expand(config.horizon.n_steps, n_actors, 2).contiguous()
    radii = torch.full((n_actors,), 1.4)

    inputs = make_inputs(config, speed0_mps=5.0, actor_centres=centres, actor_radii=radii)
    signal = rollout(hold_speed_params(), inputs, config)

    # Recompute the exact minimum the same way, without any smoothing.
    positions, _ = plan_xy(hold_speed_params(), inputs, config)
    limits = config.limits
    ego_offsets, ego_radius = config.ego_footprint()
    offsets = torch.tensor(ego_offsets)
    ego = positions.unsqueeze(-2) + offsets.reshape(-1, 1) * torch.tensor([1.0, 0.0])
    gaps = (
        (ego.unsqueeze(-2) - centres.unsqueeze(-3)).norm(dim=-1)
        - ego_radius
        - radii
        - limits.collision_margin_m
    )
    exact = gaps.reshape(gaps.shape[0], -1).min(dim=-1).values

    # `saturate` compresses by ~0.1% at these magnitudes; nothing else differs.
    assert torch.allclose(signal[:, Col.CLEARANCE], exact, rtol=2e-3, atol=1e-3)


def test_headway_saturates_with_no_lead_vehicle(config):
    signal = rollout(hold_speed_params(), make_inputs(config), config)
    assert torch.allclose(
        signal[:, Col.HEADWAY_SLACK],
        torch.full((config.horizon.n_steps,), config.limits.saturation_m),
    )
    assert torch.isfinite(signal).all()


def rss_distance(speed: float, lead_speed: float, limits) -> float:
    """The RSS safe longitudinal distance, written out independently."""
    rho = limits.rss_response_s
    required = (
        speed * rho
        + 0.5 * limits.rss_response_accel_mps2 * rho * rho
        + (speed + rho * limits.rss_response_accel_mps2) ** 2 / (2 * limits.rss_brake_mps2)
        - lead_speed**2 / (2 * limits.rss_lead_brake_mps2)
    )
    return max(required, limits.standstill_gap_m)


def test_headway_tightens_as_we_close_on_a_lead(config):
    inputs = make_inputs(config, speed0_mps=8.0, lead_station_m=30.0)
    signal = rollout(hold_speed_params(), inputs, config)
    slack = signal[:, Col.HEADWAY_SLACK]
    assert slack[0] > slack[-1]
    expected0 = 30.0 - rss_distance(8.0, 0.0, config.limits)
    assert slack[0].item() == pytest.approx(expected0, abs=1.0)


def test_a_lead_at_our_own_speed_needs_far_less_room_than_a_stopped_one(config):
    """The term a fixed time headway leaves out, and why it matters.

    RSS credits the vehicle ahead for the distance its own braking takes, so
    matching its speed is a much easier requirement than catching a stopped
    one. A `time * v + gap` rule cannot tell the two apart, and demands the
    stopped-car distance in both -- which is how the policy came to sit 2.6x
    further back than it needed to.
    """
    limits = config.limits
    stopped = rss_distance(8.0, 0.0, limits)
    matched = rss_distance(8.0, 8.0, limits)
    assert matched < stopped - 3.0

    moving = make_inputs(config, speed0_mps=8.0, lead_station_m=20.0, lead_speed_mps=8.0)
    halted = make_inputs(config, speed0_mps=8.0, lead_station_m=20.0, lead_speed_mps=0.0)
    assert (
        rollout(hold_speed_params(), moving, config)[0, Col.HEADWAY_SLACK]
        > rollout(hold_speed_params(), halted, config)[0, Col.HEADWAY_SLACK]
    )


def test_a_lead_pulling_away_still_leaves_a_standstill_gap(config):
    """RSS can ask for nothing at all; we decline to ride a bumper anyway."""
    limits = config.limits
    assert rss_distance(2.0, 12.0, limits) == pytest.approx(limits.standstill_gap_m)


def test_a_stopped_lead_still_demands_the_full_stopping_distance(config):
    """The relaxation must not cost anything where it matters."""
    limits = config.limits
    required = rss_distance(10.0, 0.0, limits)
    # Reaction, then braking from whatever speed that reaction left us at.
    assert required > 10.0**2 / (2 * limits.rss_brake_mps2)


def test_the_light_gate_carries_the_configured_magnitude(config):
    off = rollout(hold_speed_params(), make_inputs(config, light_active=False), config)
    on = rollout(hold_speed_params(), make_inputs(config, light_active=True), config)
    assert torch.allclose(
        off[:, Col.LIGHT_GATE],
        torch.full((config.horizon.n_steps,), -config.limits.gate_magnitude_m),
    )
    assert torch.allclose(
        on[:, Col.LIGHT_GATE],
        torch.full((config.horizon.n_steps,), config.limits.gate_magnitude_m),
    )


def test_route_slack_measures_distance_to_the_published_end(config):
    """Near zero -- where the rule actually bites -- saturation is transparent.

    `saturate` compresses globally, so a 60 m slack reads smaller than 60 m.
    That is harmless because the rule only asks about the sign, and it is
    checked here at the scale that matters.
    """
    inputs = make_inputs(config, route=straight_route(length_m=24.0), speed0_mps=6.0)
    signal = rollout(hold_speed_params(), inputs, config)
    # Four seconds at 6 m/s covers 24 m, so the slack runs down to about zero.
    assert signal[-1, Col.ROUTE_SLACK].item() == pytest.approx(0.0, abs=0.3)
    assert signal[-1, Col.ROUTE_SLACK] < signal[0, Col.ROUTE_SLACK]


def test_saturation_is_near_transparent_at_the_scales_the_rules_use(config):
    """The bound on the distortion `saturate` introduces.

    Rules compare against zero with margins of a few metres, so what matters
    is that the mapping is close to the identity there -- not that it is exact
    at 80 m, where the only requirement is that nothing runs away.
    """
    from stl_driver.signals import saturate

    cap = config.limits.saturation_m
    for value in (0.1, 0.5, 1.0, 3.0, 5.0):
        for sign in (1.0, -1.0):
            saturated = float(saturate(torch.tensor(sign * value), cap))
            assert saturated == pytest.approx(sign * value, rel=0.003)
    # And nothing ever escapes the cap, however large the input.
    assert float(saturate(torch.tensor(1e6), cap)) <= cap
    assert float(saturate(torch.tensor(-1e6), cap)) >= -cap


def test_speed_headroom_tracks_the_posted_limit(config):
    """Exact where the limit binds, saturating where it does not.

    Being further under a limit than the tolerance is not more legal, and an
    unsaturated column made the objective pay for it: swept on an empty road,
    the happiest cruise was a metre and a half under an 8 m/s limit and every
    plan approaching the limit scored worse. Above the limit the reading stays
    exact, so a violation costs what it always did.
    """
    scale = config.limits.speed_headroom_scale_mps

    def headroom(speed: float, limit: float) -> float:
        inputs = make_inputs(config, speed0_mps=speed, speed_limit_mps=limit)
        return rollout(hold_speed_params(), inputs, config)[0, Col.SPEED_HEADROOM].item()

    # Over the limit: exact, and unbounded.
    assert headroom(11.5, 10.0) == pytest.approx(-1.5, abs=0.05)
    # Just under it: still close to the truth, since this is where it binds.
    assert headroom(9.8, 10.0) == pytest.approx(0.2, abs=0.05)
    # Far under it: saturated, and never past the scale it saturates over.
    assert headroom(4.0, 10.0) <= scale + 1e-6
    assert headroom(4.0, 10.0) == pytest.approx(scale, abs=0.02)
    # But still moving, or the rule would have no gradient where it is
    # satisfied and could not start pushing back until too late.
    assert headroom(9.0, 10.0) < headroom(8.0, 10.0)


# ---------------------------------------------------------------------------
# Differentiability -- the tests that matter most
# ---------------------------------------------------------------------------


def test_rollout_passes_gradcheck(config):
    """Analytic gradients agree with numerical ones, in double precision."""
    small = STLDriverConfig(horizon=Horizon(n_steps=8, step_s=0.1))
    inputs = make_inputs(small, speed0_mps=5.0)
    inputs = _to_double(inputs)

    params = torch.zeros(N_PARAMS, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda p: rollout(p, inputs, small),
        (params,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_the_lateral_gradient_survives_the_station_lookup(config):
    """The `floor`-without-detach bug, caught the only way it can be.

    `torch.floor` has zero derivative almost everywhere. Differentiating
    through it zeroes d(position)/d(station), and the resulting policy plans
    perfectly reasonable-looking trajectories that never respond to lateral
    cost. Finite differences against autograd is what distinguishes that from
    a merely badly-tuned optimizer.
    """
    inputs = make_inputs(config, route=curved_route(radius_m=50.0), speed0_mps=7.0)
    rules = RuleSet(build_rules(config))

    def score(params: torch.Tensor) -> torch.Tensor:
        return rules.objective(rollout(params, inputs, config), **smooth(4.0))

    params = torch.tensor([0.1, -0.2, 0.05, 0.0, 0.3, -0.4], dtype=torch.float32)
    analytic = torch.func.grad(score)(params)

    step = 1e-3
    numeric = torch.zeros_like(params)
    for index in range(N_PARAMS):
        shifted_up = params.clone()
        shifted_up[index] += step
        shifted_down = params.clone()
        shifted_down[index] -= step
        numeric[index] = (score(shifted_up) - score(shifted_down)) / (2.0 * step)

    assert torch.allclose(analytic, numeric, atol=2e-2, rtol=5e-2)
    # And specifically: the lateral parameters are not dead.
    assert analytic[4:].abs().max().item() > 1e-3


def test_the_longitudinal_gradient_points_the_right_way(config):
    """Closing on an obstacle, the objective must prefer decelerating.

    The obstacle sits where the plan currently stops just short of it, which
    is the band where the clearance rule has gradient to give -- see
    `test_gradients_cannot_escape_an_already_colliding_plan` for what happens
    outside it.
    """
    centres, radii = stationary_actor(config, ahead_m=43.0)
    inputs = make_inputs(config, speed0_mps=9.0, actor_centres=centres, actor_radii=radii)
    rules = RuleSet(build_rules(config))

    def score(params: torch.Tensor) -> torch.Tensor:
        return rules.objective(rollout(params, inputs, config), **smooth(4.0))

    params = hold_speed_params()
    signal = rollout(params, inputs, config)
    # Precondition: the plan is near the obstacle but not yet through it.
    assert 0.0 < signal[:, Col.CLEARANCE].min().item() < 2.0

    gradient = torch.func.grad(score)(params)
    # Raising the accel knots means going faster; the objective must fall.
    assert gradient[:4].sum().item() < 0.0


def test_gradients_cannot_escape_an_already_colliding_plan(config):
    """Why the optimizer samples before it descends.

    Once a plan passes through an obstacle, it does so at any speed, so the
    minimum clearance over the horizon barely depends on the speed profile and
    the gradient goes quiet. Gradient descent alone would sit there. Escaping
    needs a *different* plan, which is what the sampling stage supplies -- so
    this is not a defect to fix but the reason the search has two stages.
    """
    centres, radii = stationary_actor(config, ahead_m=18.0)
    inputs = make_inputs(config, speed0_mps=9.0, actor_centres=centres, actor_radii=radii)
    rules = RuleSet(build_rules(config))

    formula = next(r for r in rules.rules if r.name == "collision_free").formula

    def clearance_pull(scenario_inputs) -> float:
        def score(p: torch.Tensor) -> torch.Tensor:
            return formula.robustness(rollout(p, scenario_inputs, config), **smooth(4.0))

        return torch.func.grad(score)(hold_speed_params())[:4].abs().sum().item()

    colliding = rollout(hold_speed_params(), inputs, config)
    assert colliding[:, Col.CLEARANCE].min().item() < 0.0

    # The same rule, in the band where the plan stops just short, pulls far
    # harder -- so the quiet gradient above is about the situation, not about
    # the rule being weak.
    near_centres, near_radii = stationary_actor(config, ahead_m=43.0)
    near_miss = make_inputs(
        config, speed0_mps=9.0, actor_centres=near_centres, actor_radii=near_radii
    )
    assert clearance_pull(near_miss) > 10.0 * clearance_pull(inputs)

    # Escaping such a state is the sampler's job, not the gradient's; that it
    # succeeds is asserted where it matters, in
    # `test_optimizer.py::test_it_stops_short_of_a_stationary_obstacle`.


def test_gradients_flow_from_every_rule_to_the_parameters(config):
    """No rule may be structurally invisible to the optimizer.

    Checked across several scenarios rather than one, because some rules are
    gated on states that exclude each other. `stop_on_red` and `keep_moving`
    turn on opposite states of the same traffic light, and demanding one
    scenario satisfy both would be demanding a contradiction.

    `speed_limit` needs its own for a different reason: the headroom column
    saturates once a plan is comfortably legal, so a vehicle 7 m/s under the
    limit is *meant* to feel no pull from it -- that saturation is what stopped
    the objective paying for every metre per second given up. The rule is
    reachable where it is relevant, which is near the limit, so that is where
    this asks.
    """
    centres, radii = stationary_actor(config, ahead_m=30.0, lateral_m=1.0)
    common = dict(
        route=curved_route(radius_m=60.0),
        speed0_mps=7.0,
        actor_centres=centres,
        actor_radii=radii,
        lead_station_m=45.0,
        published_length_m=70.0,
    )
    at_the_limit = dict(common, speed0_mps=config.limits.default_speed_limit_mps - 0.2)
    scenarios = {
        "red light": make_inputs(config, stop_station_m=50.0, light_active=True, **common),
        "green light": make_inputs(config, light_active=False, **common),
        "at the speed limit": make_inputs(config, light_active=False, **at_the_limit),
    }
    rules = RuleSet(build_rules(config))
    params = torch.tensor([0.2, -0.1, 0.1, 0.0, 0.2, -0.1], dtype=torch.float32)

    reachable: set[str] = set()
    for inputs in scenarios.values():
        for rule in rules.rules:

            def score(p: torch.Tensor, rule=rule, inputs=inputs) -> torch.Tensor:
                return rule.formula.robustness(rollout(p, inputs, config), **smooth(4.0))

            if torch.func.grad(score)(params).abs().max().item() > 1e-9:
                reachable.add(rule.name)

    dead = set(rules.names) - reachable
    assert not dead, f"rules with no gradient to the plan in any scenario: {sorted(dead)}"


def test_a_vacuous_rule_contributes_no_gradient(config):
    """The flip side: an inapplicable rule must not tug on the plan at all."""
    rules = RuleSet(build_rules(config))
    green = make_inputs(config, speed0_mps=7.0, light_active=False)
    stop_on_red = next(r for r in rules.rules if r.name == "stop_on_red").formula

    def score(p: torch.Tensor) -> torch.Tensor:
        return stop_on_red.robustness(rollout(p, green, config), **smooth(4.0))

    gradient = torch.func.grad(score)(torch.zeros(N_PARAMS))
    assert gradient.abs().max().item() < 1e-9


def test_vmap_matches_a_python_loop(config):
    """The optimizer's batching fast path, against the obvious slow one.

    Upstream hardcodes `signal.reshape([T, 1])` inside the temporal
    operators, so a manual batch dimension breaks where `vmap` does not. This
    is the canary that says which path is live on the pinned torch.
    """
    inputs = make_inputs(config, speed0_mps=6.0)
    rules = RuleSet(build_rules(config))
    batch = torch.randn(12, N_PARAMS, generator=torch.Generator().manual_seed(0))

    batched_signals = torch.vmap(lambda p: rollout(p, inputs, config))(batch)
    looped_signals = torch.stack([rollout(p, inputs, config) for p in batch])
    # Tolerance is float32 rounding on the jerk column, whose values run to
    # tens of m/s^3 -- the two paths reassociate the same arithmetic.
    assert torch.allclose(batched_signals, looped_signals, atol=1e-4, rtol=1e-4)

    batched_scores = torch.vmap(lambda s: rules.objective(s, **smooth(2.0)))(batched_signals)
    looped_scores = torch.stack([rules.objective(s, **smooth(2.0)) for s in looped_signals])
    assert torch.allclose(batched_scores, looped_scores, atol=1e-4)


def test_rollout_is_deterministic(config):
    inputs = make_inputs(config)
    params = torch.tensor([0.3, -0.2, 0.1, 0.4, 0.2, -0.3])
    first = rollout(params, inputs, config)
    second = rollout(params, inputs, config)
    assert torch.equal(first, second)


def _to_double(inputs):
    """Cast the tensor fields of RolloutInputs to float64 for gradcheck."""
    from dataclasses import replace

    return replace(
        inputs,
        route_xy=inputs.route_xy.to(torch.float64),
        route_tangent=inputs.route_tangent.to(torch.float64),
        route_curvature=inputs.route_curvature.to(torch.float64),
        actor_centres=inputs.actor_centres.to(torch.float64),
        actor_radii=inputs.actor_radii.to(torch.float64),
        lead_station_m=inputs.lead_station_m.to(torch.float64),
        lead_speed_mps=inputs.lead_speed_mps.to(torch.float64),
        ego_disc_offsets=inputs.ego_disc_offsets.to(torch.float64),
        published_length_m=inputs.published_length_m.to(torch.float64),
        station0_m=inputs.station0_m.to(torch.float64),
        offset0_m=inputs.offset0_m.to(torch.float64),
        speed0_mps=inputs.speed0_mps.to(torch.float64),
        accel0_mps2=inputs.accel0_mps2.to(torch.float64),
        stop_station_m=inputs.stop_station_m.to(torch.float64),
        light_gate=inputs.light_gate.to(torch.float64),
        speed_limit_mps=inputs.speed_limit_mps.to(torch.float64),
        progress_target_m=inputs.progress_target_m.to(torch.float64),
    )


def test_the_route_table_covers_any_plan_the_optimizer_can_produce(config):
    """Guards the "lookup never clamps" invariant the gradient depends on.

    A plan that ran off the end of the table would hit the index clamp, where
    the station gradient is zero.
    """
    inputs = make_inputs(config, speed0_mps=config.limits.default_speed_limit_mps)
    signal = rollout(constant_accel_params(config.limits.max_accel_mps2 * 0.999), inputs, config)
    travelled = (signal[:, Col.PROGRESS_SLACK] + inputs.progress_target_m).max().item()
    table_length = (inputs.route_xy.shape[0] - 1) * inputs.route_ds
    assert inputs.station0_m + travelled < table_length


def test_a_batch_of_random_parameters_never_produces_nans(config):
    """Sobol seeds land anywhere in the box; none may poison the objective."""
    centres, radii = stationary_actor(config, ahead_m=6.0)
    inputs = make_inputs(
        config,
        speed0_mps=0.0,
        actor_centres=centres,
        actor_radii=radii,
        lead_station_m=3.0,
        stop_station_m=1.0,
        light_active=True,
    )
    batch = torch.randn(64, N_PARAMS, generator=torch.Generator().manual_seed(7)) * 4.0
    signals = torch.vmap(lambda p: rollout(p, inputs, config))(batch)
    assert torch.isfinite(signals).all()


def test_the_plan_path_and_the_signal_agree(config):
    """`plan_xy` is what the vehicle drives; the signal is what the rules judge.

    They share `_kinematics` precisely so they cannot disagree, and this is
    what says so.
    """
    inputs = make_inputs(config, route=curved_route(radius_m=45.0), speed0_mps=6.0)
    params = torch.tensor([0.2, 0.1, -0.1, 0.0, 0.4, 0.2])
    positions, _ = plan_xy(params, inputs, config)
    signal = rollout(params, inputs, config)

    from stl_driver.frenet import build_route_table, project_to_frenet

    table = build_route_table(
        curved_route(radius_m=45.0),
        ds=config.route_table_ds_m,
        length_m=config.route_table_length_m(),
    )
    _, offset = project_to_frenet(table, positions.detach().numpy())
    assert np.allclose(offset, signal[:, Col.LATERAL_OFFSET].detach().numpy(), atol=0.05)


def test_coming_to_a_stop_does_not_manufacture_absurd_jerk(config):
    """The clamp is ours; the vehicle should not be charged for it.

    Differencing the floored speed puts a kink where a plan comes to rest, and
    one 0.1 s step later that kink is a step change in acceleration and a jerk
    of forty-odd metres per second cubed -- ten times the comfort bound, on a
    manoeuvre a driver would not notice. Measured before the fix, that was the
    whole of the worst `comfort` violations.
    """
    from stl_driver.rules import RuleSet, build_rules

    # Braking to a halt well inside the horizon, so the clamp definitely bites.
    inputs = make_inputs(config, speed0_mps=3.0)
    signal = rollout(constant_accel_params(-3.5), inputs, config)

    assert signal[:, Col.SPEED].min().item() >= 0.0  # it does stop
    assert signal[:, Col.JERK].abs().max().item() < config.limits.max_jerk_mps3 * 2.0

    rules = RuleSet(build_rules(config))
    assert float(rules.per_rule(signal)["comfort"]) > -5.0


def test_comfort_still_sees_a_harsh_deceleration(config):
    """The fix must not blind the rule to what it is for."""
    from stl_driver.rules import RuleSet, build_rules

    inputs = make_inputs(config, speed0_mps=12.0, accel0_mps2=0.0)
    signal = rollout(constant_accel_params(-3.99), inputs, config)
    rules = RuleSet(build_rules(config))
    # Commanded deceleration near the bound is still reported near the bound.
    assert signal[:, Col.ACCEL].min().item() < -3.0
    assert float(rules.per_rule(signal)["comfort"]) < 1.0


def test_a_constant_bend_gives_a_constant_lateral_acceleration(config):
    """Lateral acceleration must be the road's, not the table's sampling.

    Reading it back off the planned positions meant differencing a polyline
    twice, and a polyline keeps all its turning at the vertices: on a genuinely
    constant 25 m bend the trace oscillated between 1.80 and 2.20 against a
    true 1.96, and between 1.60 and 2.80 with a 2 m table. `comfort` is an
    `Always`, so the peak was the only part of that anyone read.
    """
    for radius_m, speed_mps in ((25.0, 7.0), (40.0, 12.0), (60.0, 4.0)):
        inputs = make_inputs(config, route=curved_route(radius_m=radius_m), speed0_mps=speed_mps)
        signal = rollout(torch.zeros(N_PARAMS), inputs, config)
        # Skip the first and last few metres of road. `np.gradient` is
        # one-sided at the table's ends, and past the published route the table
        # extrapolates straight, so the curvature ramps at both. How many
        # *steps* that is depends on the speed, which is why the margin is
        # expressed as a distance.
        margin = int(3.0 / (speed_mps * config.horizon.step_s)) + 1
        lateral = signal[margin:-margin, Col.LATERAL_ACCEL]
        truth = speed_mps**2 / radius_m
        assert torch.allclose(lateral, torch.full_like(lateral, truth), rtol=0.03)


def test_steering_across_the_road_shows_up_as_lateral_acceleration(config):
    """The offset profile's own bend counts, not just the road's."""
    straight = make_inputs(config, route=straight_route(), speed0_mps=8.0)
    held = rollout(torch.zeros(N_PARAMS), straight, config)
    assert held[:, Col.LATERAL_ACCEL].abs().max() < 1e-6

    swerve = torch.zeros(N_PARAMS)
    swerve[4] = 2.0
    swerve[5] = -2.0
    moved = rollout(swerve, straight, config)
    assert moved[:, Col.LATERAL_ACCEL].abs().max() > 0.1
