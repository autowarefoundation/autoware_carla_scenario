# SPDX-License-Identifier: Apache-2.0
"""Does the search actually produce good driving?

These are the tests that would catch a specification that is individually
correct rule by rule but collectively describes a car that never moves, or one
that drives through things.  They assert on behaviour -- speed profiles, where
the plan stops -- rather than on robustness numbers.
"""

from __future__ import annotations

import math

import pytest
import torch

from stl_driver.config import STLDriverConfig
from stl_driver.optimizer import RobustnessOptimizer
from stl_driver.rollout import N_PARAMS, rollout
from stl_driver.rules import RuleSet, build_rules
from stl_driver.signals import Col, signal_from_columns
from tests.conftest import curved_route, make_inputs, stationary_actor, straight_route

CONFIG = STLDriverConfig()


@pytest.fixture(scope="module")
def optimizer() -> RobustnessOptimizer:
    search = RobustnessOptimizer(RuleSet(build_rules(CONFIG)), CONFIG)
    # Pay torch's lazy initialisation before any test measures anything, which
    # takes more than one solve: the first in a process runs at a third of the
    # steady-state rate, so it exhausts its time budget after a single
    # refinement step and plans visibly worse. That is why `EgoPolicy` warms up
    # in `on_session_start` rather than on the first step the runtime waits for.
    for _ in range(3):
        search.solve(make_inputs(CONFIG, speed0_mps=5.0))
    return search


def constant(value: float) -> torch.Tensor:
    return torch.full((CONFIG.horizon.n_steps,), float(value))


def speeds(solution) -> torch.Tensor:
    return solution.signal[:, Col.SPEED]


def stations(solution, inputs) -> torch.Tensor:
    return solution.signal[:, Col.PROGRESS_SLACK] + inputs.progress_target_m + inputs.station0_m


# ---------------------------------------------------------------------------
# The anti-parking property
# ---------------------------------------------------------------------------


def test_on_an_empty_road_it_drives(optimizer):
    """The regression test for the whole liveness argument.

    Standing still satisfies every safety rule. If the liveness rules or the
    objective's saturation were wrong, this is what would fail -- and it would
    fail as a car that sits at the start line forever.
    """
    inputs = make_inputs(CONFIG, speed0_mps=0.0)
    solution = optimizer.solve(inputs)

    assert solution.robustness > 0.0
    assert not solution.needs_fallback
    # Four seconds from rest under a 2 m/s^2 bound cannot reach cruise, so the
    # test is that it accelerates hard rather than that it arrives.
    #
    # The final speed is deliberately a loose sanity bound rather than a tight
    # one. On an empty road the binding rule is `no_reverse` -- pinned near
    # zero by the standing start -- so nothing in the specification pushes on
    # the *final* speed, and the number it lands on tracks how many refinement
    # steps happened to fit the budget. The two assertions below are the ones
    # that carry the meaning: it accelerates near the limit, and it covers more
    # ground than `makes_progress` asked for.
    assert speeds(solution)[-1].item() > 3.0
    assert solution.signal[:, Col.ACCEL].max().item() > CONFIG.limits.max_accel_mps2 * 0.7
    # And covers more ground than `makes_progress` demanded.
    travelled = speeds(solution).sum().item() * CONFIG.horizon.step_s
    assert travelled > inputs.progress_target_m


def test_from_a_standstill_it_pulls_away_smoothly(optimizer):
    inputs = make_inputs(CONFIG, speed0_mps=0.0)
    solution = optimizer.solve(inputs)
    speed = speeds(solution)
    assert torch.all(torch.diff(speed) > -0.5)  # no lurching backwards
    assert solution.signal[:, Col.ACCEL].max().item() <= CONFIG.limits.max_accel_mps2 + 0.3


def test_at_cruise_it_holds_a_steady_speed(optimizer):
    inputs = make_inputs(CONFIG, speed0_mps=8.0)
    solution = optimizer.solve(inputs)
    speed = speeds(solution)
    assert speed.mean().item() == pytest.approx(8.0, abs=1.5)
    assert speed.std().item() < 1.5


def test_it_does_not_exceed_the_posted_limit(optimizer):
    inputs = make_inputs(CONFIG, speed0_mps=5.0, speed_limit_mps=6.0)
    solution = optimizer.solve(inputs)
    tolerance = CONFIG.limits.speed_tolerance_mps
    assert speeds(solution).max().item() < 6.0 + tolerance + 0.2
    assert float(solution.per_rule["speed_limit"]) > -0.1


# ---------------------------------------------------------------------------
# Reacting to the world
# ---------------------------------------------------------------------------


def test_it_stops_short_of_a_stationary_obstacle(optimizer):
    """The plan must not pass through the obstacle, and must slow for it."""
    centres, radii = stationary_actor(CONFIG, ahead_m=22.0)
    inputs = make_inputs(CONFIG, speed0_mps=7.0, actor_centres=centres, actor_radii=radii)
    solution = optimizer.solve(inputs)

    assert solution.signal[:, Col.CLEARANCE].min().item() > 0.0
    assert speeds(solution)[-1].item() < speeds(solution)[0].item()
    assert float(solution.per_rule["collision_free"]) > 0.0


def test_it_slows_for_a_slower_vehicle_ahead(optimizer):
    centres, radii = stationary_actor(CONFIG, ahead_m=30.0)
    inputs = make_inputs(
        CONFIG,
        speed0_mps=9.0,
        actor_centres=centres,
        actor_radii=radii,
        lead_station_m=28.0,
    )
    solution = optimizer.solve(inputs)
    assert speeds(solution)[-1].item() < 6.0
    assert float(solution.per_rule["safe_headway"]) > -0.5


def test_it_stops_before_a_red_light(optimizer):
    stop_station = 25.0
    inputs = make_inputs(CONFIG, speed0_mps=7.0, stop_station_m=stop_station, light_active=True)
    solution = optimizer.solve(inputs)

    final_station = stations(solution, inputs)[-1].item()
    assert final_station < stop_station
    assert float(solution.per_rule["stop_on_red"]) > 0.0
    # It need not be stationary by the end of this horizon -- the plan is
    # replanned at 10 Hz and only has to keep the line ahead of it -- but it
    # must clearly be slowing for the light.
    assert speeds(solution)[-1].item() < speeds(solution)[0].item() * 0.75


def test_it_drives_through_a_green_light(optimizer):
    """The same geometry with the light inactive must not cause a stop."""
    inputs = make_inputs(CONFIG, speed0_mps=7.0, stop_station_m=25.0, light_active=False)
    solution = optimizer.solve(inputs)
    assert speeds(solution).mean().item() > 5.0
    assert stations(solution, inputs)[-1].item() > 25.0


def test_it_slows_before_the_end_of_the_route(optimizer):
    inputs = make_inputs(
        CONFIG, route=straight_route(length_m=200.0), speed0_mps=8.0, published_length_m=18.0
    )
    solution = optimizer.solve(inputs)
    assert stations(solution, inputs)[-1].item() < 18.0 + 1.0
    assert speeds(solution)[-1].item() < speeds(solution)[0].item()


def test_it_stays_in_lane_on_a_curve(optimizer):
    inputs = make_inputs(CONFIG, route=curved_route(radius_m=45.0), speed0_mps=7.0)
    solution = optimizer.solve(inputs)
    offset = solution.signal[:, Col.LATERAL_OFFSET]
    assert offset.abs().max().item() < CONFIG.limits.lane_half_width_m
    assert float(solution.per_rule["lane_keeping"]) > 0.0


def test_it_respects_the_lateral_acceleration_bound_on_a_tight_curve(optimizer):
    inputs = make_inputs(CONFIG, route=curved_route(radius_m=25.0), speed0_mps=9.0)
    solution = optimizer.solve(inputs)
    # Slowing is the only way to hold the bound at this radius.
    assert speeds(solution)[-1].item() < 9.0


def test_it_recentres_from_a_lateral_offset(optimizer):
    inputs = make_inputs(CONFIG, speed0_mps=6.0, offset0_m=0.9)
    solution = optimizer.solve(inputs)
    offset = solution.signal[:, Col.LATERAL_OFFSET]
    assert offset[0].item() == pytest.approx(0.9, abs=0.1)  # starts where we are
    assert abs(offset[-1].item()) < 0.6  # and heads back


# ---------------------------------------------------------------------------
# When the specification cannot be met
# ---------------------------------------------------------------------------


def test_an_unavoidable_obstacle_triggers_the_fallback(optimizer):
    """Two metres ahead at 10 m/s: nothing satisfies `collision_free`."""
    centres, radii = stationary_actor(CONFIG, ahead_m=2.0)
    inputs = make_inputs(CONFIG, speed0_mps=10.0, actor_centres=centres, actor_radii=radii)
    solution = optimizer.solve(inputs)

    assert solution.robustness < 0.0
    assert solution.needs_fallback
    assert solution.worst_rule == "collision_free"


def test_even_when_failing_it_prefers_slowing_down(optimizer):
    """A hopeless situation must still produce the least-bad plan.

    The fallback overrides the trajectory, but the solution's own speed
    profile is what says the search understood the situation rather than
    giving up arbitrarily.
    """
    centres, radii = stationary_actor(CONFIG, ahead_m=6.0)
    inputs = make_inputs(CONFIG, speed0_mps=11.0, actor_centres=centres, actor_radii=radii)
    solution = optimizer.solve(inputs)
    assert speeds(solution)[-1].item() < speeds(solution)[0].item()


def test_a_soft_violation_alone_does_not_trigger_the_fallback(optimizer):
    """Being slightly out of lane is not a reason to slam on the brakes."""
    inputs = make_inputs(CONFIG, speed0_mps=6.0, offset0_m=1.6)
    solution = optimizer.solve(inputs)
    assert not solution.needs_fallback


# ---------------------------------------------------------------------------
# Search behaviour
# ---------------------------------------------------------------------------


def test_the_same_inputs_give_the_same_plan(optimizer):
    """Determinism: the runtime's rollouts are seeded and must stay repeatable.

    This holds by construction now rather than by luck. While the search
    stopped at a wall-clock bound, two solves of the same situation could
    complete different numbers of refinement steps and return different plans
    -- so this test was one busy machine away from failing, and the closed
    loop was worse than that: the same build on the same route came back
    96.0%, 91.0% and 93.0% satisfied over 340, 238 and 236 metres.
    """
    inputs = make_inputs(CONFIG, speed0_mps=6.0)
    first = optimizer.solve(inputs)
    second = optimizer.solve(inputs)
    assert torch.equal(first.params, second.params)
    assert first.robustness == second.robustness


def test_refinement_improves_on_the_best_sample(optimizer):
    """Otherwise the gradient stage is costing latency for nothing.

    Measured on the objective, because that is what selects the plan that gets
    driven -- see `RobustnessOptimizer.solve` for why a minimum over rules is
    the wrong thing to maximise once a rule can be gated off by an obstruction.
    """
    inputs = make_inputs(CONFIG, speed0_mps=4.0)
    rules = RuleSet(build_rules(CONFIG))

    candidates = optimizer._candidates(None, 0.0)
    with torch.no_grad():
        sampled = [float(rules.objective(rollout(p, inputs, CONFIG))) for p in candidates]
    solution = optimizer.solve(inputs)

    assert solution.n_refine_steps > 0
    with torch.no_grad():
        chosen = float(rules.objective(solution.signal))
    assert chosen >= max(sampled) - 1e-6


def test_the_search_never_returns_a_worse_plan_than_its_seed(optimizer):
    """Refinement keeps the best iterate, so it cannot make things worse."""
    for speed in (0.0, 3.0, 7.0, 12.0):
        inputs = make_inputs(CONFIG, speed0_mps=speed)
        solution = optimizer.solve(inputs)
        braking = rollout(optimizer._candidates(None, 0.0)[-1], inputs, CONFIG)
        rules = RuleSet(build_rules(CONFIG))
        assert solution.robustness >= float(rules.spec_robustness(braking)) - 1e-6


def test_a_warm_start_is_accepted_and_helps(optimizer):
    inputs = make_inputs(CONFIG, speed0_mps=6.0)
    rules = RuleSet(build_rules(CONFIG))
    cold = optimizer.solve(inputs, warm_start=None)
    warm = optimizer.solve(inputs, warm_start=cold.params)
    with torch.no_grad():
        assert float(rules.objective(warm.signal)) >= float(rules.objective(cold.signal)) - 1e-6


def test_the_hold_speed_candidate_holds_speed(optimizer):
    """It did not, and it failed where it was needed most.

    The parameters describe jerk-bounded changes from the vehicle's *measured*
    acceleration, and `_candidates` built the fixed plans as though that were
    always zero. From 6 m/s with a measured -2 m/s^2, the plan offered to the
    search as "hold speed" brought the vehicle to a standstill inside the
    horizon; at +1.5 it ran to 12 m/s against an 8 m/s limit. So on exactly
    the steps where the vehicle was already slowing -- the steps where
    something has to offer the alternative of not slowing -- no candidate did.
    """
    for accel0 in (-2.0, -1.0, 0.0, 1.5):
        inputs = make_inputs(CONFIG, speed0_mps=6.0, accel0_mps2=accel0)
        # The three fixed plans are the last three rows: the warm start's slot,
        # then hold, then the braking seed.
        hold = optimizer._candidates(None, accel0)[-2]
        with torch.no_grad():
            speed = rollout(hold, inputs, CONFIG)[:, Col.SPEED]
        # It cannot hold 6.0 exactly -- walking the acceleration back to zero
        # takes a knot interval, and that is honest. It must not run away with
        # the measured value, which is what leaving `accel0` at zero did.
        assert abs(float(speed[-1]) - 6.0) < 1.6, f"from {accel0} m/s^2"


def test_the_sampled_plans_are_lane_shaped(optimizer):
    """What the sampling stage spends its sixty-four plans on.

    Drawn as a box in logit space, the two lateral parameters gave 53% of
    plans ending against the corridor edge -- `tanh` saturates, so a uniform
    logit is not a uniform lane position -- and 47% whose Bezier control point
    lay the opposite side of the centreline from their end offset, which is an
    S through a 1.2 m corridor.
    """
    half = CONFIG.limits.lane_half_width_m
    control = half * torch.tanh(optimizer._seeds[:, 4])
    end = half * torch.tanh(optimizer._seeds[:, 5])

    # Centred by default: that is the lane position a vehicle holds unless it
    # has a reason not to.
    assert (end.abs() < 0.3).float().mean() > 0.4
    assert (end.abs() > 0.9 * half).float().mean() < 0.15

    # No sampled path swings away from where it is going, or past it.
    assert torch.all(control * end >= -1e-6)
    assert torch.all(control.abs() <= end.abs() + 1e-6)

    # The edge of the corridor is still reachable, or nothing could ever be
    # sampled that moves aside.
    assert end.abs().max() > 0.8 * half


def test_nothing_the_search_can_try_breaks_the_comfort_bounds(optimizer):
    """The guard against the two envelopes drifting apart again.

    A plan is held to `Limits.plan_accel_envelope`, which is inset from the
    comfort bounds the specification asserts. Anything that reaches for those
    bounds directly to build or anchor a plan re-opens the gap, and the gap
    arrives in the opening jerk multiplied by ten -- it made `comfort`
    unsatisfiable by every candidate after a hard stop. This asks the question
    end to end instead: over the whole range of states the vehicle can report,
    is there anything at all the search might try that the rule would reject?
    """
    limits = CONFIG.limits
    lowest, highest = limits.plan_accel_envelope()
    for accel0 in (lowest, 0.0, highest):
        inputs = make_inputs(CONFIG, speed0_mps=5.0, accel0_mps2=accel0)
        candidates = optimizer._candidates(torch.zeros(N_PARAMS), accel0)
        with torch.no_grad():
            for params in candidates:
                signal = rollout(params, inputs, CONFIG)
                accel = signal[:, Col.ACCEL]
                jerk = signal[:, Col.JERK]
                assert accel.max().item() <= limits.max_accel_mps2
                assert accel.min().item() >= -limits.max_decel_mps2
                assert jerk.abs().max().item() <= limits.max_jerk_mps3


def test_the_candidate_batch_is_the_same_width_with_or_without_a_warm_start(optimizer):
    """A width that varies is a graph that gets recompiled mid-drive.

    Costed at 3.4 s on the second step of a compiled session -- the first step
    with a previous plan to reuse -- against a 0.064 s median and a 0.100 s
    policy timestep.
    """
    cold = optimizer._candidates(None, 0.0)
    warm = optimizer._candidates(torch.zeros(N_PARAMS), 0.0)
    assert cold.shape == warm.shape == (CONFIG.optimizer.n_samples + 3, N_PARAMS)
    # The braking plan stays last either way: it is the seed that rescues a
    # step where every sampled plan is unsafe, and tests reach for it by index.
    assert torch.allclose(cold[-1], warm[-1])


def test_alternatives_are_reported_for_inspection(optimizer):
    inputs = make_inputs(CONFIG, speed0_mps=6.0)
    solution = optimizer.solve(inputs)
    assert len(solution.alternatives) == CONFIG.optimizer.n_reported_candidates
    assert all(a.shape == solution.params.shape for a in solution.alternatives)


def test_every_rule_is_reported(optimizer):
    inputs = make_inputs(CONFIG, speed0_mps=6.0)
    solution = optimizer.solve(inputs)
    rules = RuleSet(build_rules(CONFIG))
    assert set(solution.per_rule) == set(rules.names)
    assert all(math.isfinite(v) for v in solution.per_rule.values())


def test_the_reported_robustness_is_the_minimum_over_rules(optimizer):
    inputs = make_inputs(CONFIG, speed0_mps=6.0)
    solution = optimizer.solve(inputs)
    assert solution.robustness == pytest.approx(min(solution.per_rule.values()), abs=1e-4)


def test_the_search_is_a_fixed_amount_of_work_by_default(optimizer):
    """No wall-clock bound unless one is asked for, and this is why.

    Under a bound, how many steps the search completes depends on how busy the
    machine is, and once the ego acts differently at all the whole scenario
    diverges from there -- the same build on the same route came back 96.0%,
    91.0% and 93.0% satisfied over 340, 238 and 236 metres. A fixed amount of
    work makes the same situation give the same plan.
    """
    assert CONFIG.optimizer.time_budget_s is None
    solution = optimizer.solve(make_inputs(CONFIG, speed0_mps=6.0))
    assert solution.n_refine_steps == CONFIG.optimizer.n_refine_steps


def test_the_time_budget_is_honoured_when_one_is_set(optimizer):
    """Deployments that must hold a deadline can still ask for one."""
    from dataclasses import replace

    budget = 0.070
    config = replace(CONFIG, optimizer=replace(CONFIG.optimizer, time_budget_s=budget))
    bounded = RobustnessOptimizer(RuleSet(build_rules(config)), config)
    bounded.solve(make_inputs(config, speed0_mps=6.0))  # pay torch's start-up
    solution = bounded.solve(make_inputs(config, speed0_mps=6.0))
    # Generous: this asserts the cutoff works, not that the machine is fast.
    assert solution.solve_seconds < budget * 4.0


def test_a_tiny_budget_degrades_to_fewer_steps_rather_than_failing():
    """A slow machine must drop refinement, not miss the deadline."""
    from dataclasses import replace

    config = replace(CONFIG, optimizer=replace(CONFIG.optimizer, time_budget_s=1e-9))
    optimizer = RobustnessOptimizer(RuleSet(build_rules(config)), config)
    solution = optimizer.solve(make_inputs(config, speed0_mps=6.0))
    assert solution.n_refine_steps == 0
    assert solution.signal is not None  # still a usable plan


def test_the_batched_path_is_the_one_in_use(optimizer):
    """If this ever flips, the loop fallback is live and latency will show it."""
    solution = optimizer.solve(make_inputs(CONFIG, speed0_mps=6.0))
    assert solution.used_vmap


def test_a_distant_predicted_conflict_does_not_trigger_an_emergency_stop(optimizer):
    """Emergency braking is for what we cannot avoid, not for a forecast.

    Constant-velocity prediction of crossing traffic routinely says we will
    meet something three seconds out. That deserves slowing down -- which the
    optimizer does -- not seven metres per second squared of braking, which at
    10 Hz would then be re-decided a tenth of a second later.
    """
    centres, radii = stationary_actor(CONFIG, ahead_m=34.0)
    inputs = make_inputs(CONFIG, speed0_mps=9.0, actor_centres=centres, actor_radii=radii)
    solution = optimizer.solve(inputs)

    assert not solution.needs_fallback
    assert solution.imminent_rule is None
    # But it is unmistakably slowing for it.
    assert speeds(solution)[-1].item() < speeds(solution)[0].item()


def test_an_imminent_conflict_still_triggers_the_emergency_stop(optimizer):
    """The distinction has to cut both ways or the fallback is useless."""
    centres, radii = stationary_actor(CONFIG, ahead_m=2.0)
    inputs = make_inputs(CONFIG, speed0_mps=10.0, actor_centres=centres, actor_radii=radii)
    solution = optimizer.solve(inputs)

    assert solution.needs_fallback
    assert solution.imminent_rule == "collision_free"


def test_the_fallback_window_is_measured_in_the_opening_of_the_horizon(optimizer):
    """`imminent_hard_violation` judges only the first stretch of the plan."""
    from stl_driver.signals import signal_from_columns

    late = torch.full((CONFIG.horizon.n_steps,), 8.0)
    late[-5:] = -3.0  # a violation right at the end
    signal = signal_from_columns(clearance=late)
    assert optimizer.rules.imminent_hard_violation(signal, 12) is None
    assert optimizer.rules.any_hard_violated(optimizer.rules.per_rule(signal))

    early = torch.full((CONFIG.horizon.n_steps,), 8.0)
    early[3] = -3.0
    assert (
        optimizer.rules.imminent_hard_violation(signal_from_columns(clearance=early), 12)
        == "collision_free"
    )


def test_a_stationary_ego_does_not_emergency_brake_at_an_approaching_vehicle():
    """Braking is not a remedy for something our speed never caused.

    A vehicle moving towards a stopped ego violates `collision_free` exactly
    as badly under maximum braking. Standing on the brakes buys no safety and
    costs everything else -- no progress, and a vehicle stopped dead in
    traffic that did not expect it, which is how a stall becomes a rear-end.
    """
    optimizer = RobustnessOptimizer(RuleSet(build_rules(CONFIG)), CONFIG)
    optimizer.solve(make_inputs(CONFIG, speed0_mps=5.0))

    centres, radii = stationary_actor(CONFIG, ahead_m=3.0)
    stopped = make_inputs(CONFIG, speed0_mps=0.0, actor_centres=centres, actor_radii=radii)
    solution = optimizer.solve(stopped)

    # The violation is still reported...
    assert float(solution.per_rule["collision_free"]) < 0.0
    # ...but the emergency plan is not pretended to be a remedy for it.
    assert not solution.needs_fallback


def test_a_moving_ego_still_emergency_brakes_for_the_same_obstacle():
    """The distinction is about our speed, not about the obstacle."""
    optimizer = RobustnessOptimizer(RuleSet(build_rules(CONFIG)), CONFIG)
    optimizer.solve(make_inputs(CONFIG, speed0_mps=5.0))

    centres, radii = stationary_actor(CONFIG, ahead_m=3.0)
    moving = make_inputs(CONFIG, speed0_mps=9.0, actor_centres=centres, actor_radii=radii)
    solution = optimizer.solve(moving)

    assert solution.needs_fallback
    assert solution.imminent_rule == "collision_free"


# ---------------------------------------------------------------------------
# Which plan gets driven
# ---------------------------------------------------------------------------


def test_a_satisfying_plan_beats_a_higher_scoring_one_that_violates():
    """Satisfying the specification comes first; the objective only ranks."""
    from stl_driver.optimizer import _lexicographic_pick

    merit = torch.tensor([10.0, 1.0])
    robustness = torch.tensor([-0.5, 0.2])
    assert _lexicographic_pick(merit, robustness) == 1


def test_among_satisfying_plans_the_best_objective_wins():
    from stl_driver.optimizer import _lexicographic_pick

    merit = torch.tensor([1.0, 5.0, 3.0])
    robustness = torch.tensor([0.1, 0.2, 2.0])
    assert _lexicographic_pick(merit, robustness) == 1


def test_when_nothing_satisfies_the_objective_decides():
    """A third of city driving admits no satisfying plan at all."""
    from stl_driver.optimizer import _lexicographic_pick

    merit = torch.tensor([1.0, 7.0, 3.0])
    robustness = torch.tensor([-0.1, -2.0, -0.5])
    assert _lexicographic_pick(merit, robustness) == 1


def test_selection_does_not_reward_simply_stopping(optimizer):
    """The failure mode that ranking by robustness alone produces.

    Robustness is a minimum over rules, and a rule gated off by an obstruction
    contributes nothing to a minimum -- so where the liveness rules are
    excused, a halt violates nothing and outscores everything. Ranking on it
    makes stopping the answer to every hard question.
    """
    from stl_driver.optimizer import _lexicographic_pick

    # A halt that satisfies everything, and a plan that also satisfies
    # everything while actually going somewhere.
    stopped = signal_from_columns(speed=constant(0.0), clearance=constant(0.5))
    driving = signal_from_columns(speed=constant(7.0))
    with torch.no_grad():
        merit = torch.tensor(
            [float(optimizer.rules.objective(stopped)), float(optimizer.rules.objective(driving))]
        )
        robustness = torch.tensor(
            [
                float(optimizer.rules.spec_robustness(stopped)),
                float(optimizer.rules.spec_robustness(driving)),
            ]
        )
    assert _lexicographic_pick(merit, robustness) == 1


def test_the_compiled_paths_are_actually_used_when_present():
    """Compiling a function and then not calling it buys nothing.

    That is exactly what happened: `prepare` built a compiled `_refine_losses`
    and the refinement loop went on calling the eager one, so enabling
    `compile_search` sped up only the sampling stage -- which the profile says
    is the cheaper half. Refining a single seed costs more per step than
    scoring all sixty-six sampled candidates.

    Compilation itself takes about a minute, far too long for a unit test, so
    this substitutes a sentinel for the compiled artefact and checks it is
    reached.
    """
    optimizer = RobustnessOptimizer(RuleSet(build_rules(CONFIG)), CONFIG)
    inputs = make_inputs(CONFIG, speed0_mps=6.0)
    calls: list[str] = []

    def spy_refine(params, inp, temperature):
        calls.append("refine")
        return optimizer._refine_losses(params, inp, temperature)

    def spy_scores(signals):
        calls.append("scores")
        return optimizer._batched_scores_eager(signals)

    optimizer._compiled_refine = spy_refine
    optimizer._compiled_scores = spy_scores
    optimizer.solve(inputs)

    assert "refine" in calls, "the refinement loop ignored the compiled forward pass"
    assert "scores" in calls, "candidate scoring ignored the compiled path"


def test_the_batch_of_one_shortcut_computes_the_same_thing():
    """Bypassing `vmap` at a batch of one must be invisible in the answer.

    It is there because `vmap` at a batch of one is pure dispatch overhead --
    15.06 ms against 10.64 for the same arithmetic called directly -- and the
    refinement runs a single seed by default. A shortcut that changed the
    numbers would be a different search wearing the same name.
    """
    optimizer = RobustnessOptimizer(RuleSet(build_rules(CONFIG)), CONFIG)
    inputs = make_inputs(CONFIG, speed0_mps=7.0)
    params = torch.tensor([[0.4, -0.9, 0.2, 0.1, 0.6, -0.3]])

    shortcut_signals = optimizer._batched_rollout(params, inputs)
    batched_signals = torch.vmap(lambda p: rollout(p, inputs, CONFIG))(params)
    assert torch.allclose(shortcut_signals, batched_signals, atol=1e-5)

    shortcut_merit, shortcut_rob = optimizer._batched_scores_eager(shortcut_signals)
    batched = torch.vmap(optimizer.rules.merit_and_robustness)(batched_signals)
    assert torch.allclose(shortcut_merit, batched[0], atol=1e-4)
    assert torch.allclose(shortcut_rob, batched[1], atol=1e-4)

    losses, signals = optimizer._refine_losses(params.requires_grad_(True), inputs, 4.0)
    assert losses.shape == (1,)
    assert signals.shape[0] == 1
    assert torch.allclose(signals.detach(), batched_signals, atol=1e-5)
