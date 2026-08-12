# SPDX-License-Identifier: Apache-2.0
"""The specification, tested clause by clause.

Under exact min/max semantics a rule's robustness is a determinate number, not
merely a sign, so every test here asserts the value.  A sign-only test would
pass just as happily on a rule that reads the wrong column or applies the
wrong bound.

Each test builds a signal that satisfies everything comfortably and then
disturbs exactly one column, so a failure names its own cause.
"""

from __future__ import annotations

import math

import pytest
import torch

from stl_driver.config import STLDriverConfig
from stl_driver.rules import RuleSet, build_rules
from stl_driver.signals import signal_from_columns

CONFIG = STLDriverConfig()
T = CONFIG.horizon.n_steps
LIMITS = CONFIG.limits


@pytest.fixture(scope="module")
def rules() -> RuleSet:
    return RuleSet(build_rules(CONFIG))


def constant(value: float) -> torch.Tensor:
    return torch.full((T,), float(value))


def spike(baseline: float, value: float, at: int) -> torch.Tensor:
    values = constant(baseline)
    values[at] = value
    return values


def robustness(rules: RuleSet, name: str, signal: torch.Tensor) -> float:
    return float(rules.per_rule(signal)[name])


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


def test_speed_limit_reports_the_tightest_headroom(rules):
    signal = signal_from_columns(speed_headroom=spike(4.0, 1.5, at=11))
    # Bound is `headroom > -tolerance`, so robustness is headroom + tolerance.
    expected = 1.5 + LIMITS.speed_tolerance_mps
    assert robustness(rules, "speed_limit", signal) == pytest.approx(expected)


def test_speed_limit_is_violated_by_exceeding_the_tolerance(rules):
    signal = signal_from_columns(speed_headroom=spike(4.0, -2.0, at=6))
    assert robustness(rules, "speed_limit", signal) == pytest.approx(
        -2.0 + LIMITS.speed_tolerance_mps
    )
    assert robustness(rules, "speed_limit", signal) < 0.0


def test_no_reverse_catches_a_single_negative_speed(rules):
    signal = signal_from_columns(speed=spike(6.0, -0.4, at=23))
    assert robustness(rules, "no_reverse", signal) == pytest.approx(-0.4 + 0.05)


def test_stop_on_red_is_vacuous_when_no_light_governs_us(rules):
    """An inactive light contributes the gate magnitude, not the stop slack.

    This is the property that keeps an inapplicable rule from either dominating
    the objective or dragging the reported robustness around.
    """
    signal = signal_from_columns(
        light_gate=constant(-LIMITS.gate_magnitude_m),
        stop_slack=constant(-30.0),  # deep violation, but no light applies
    )
    assert robustness(rules, "stop_on_red", signal) == pytest.approx(LIMITS.gate_magnitude_m)


def test_stop_on_red_reports_the_stop_line_overshoot_when_active(rules):
    signal = signal_from_columns(
        light_gate=constant(LIMITS.gate_magnitude_m),
        stop_slack=spike(15.0, -2.5, at=30),
    )
    assert robustness(rules, "stop_on_red", signal) == pytest.approx(-2.5)


def test_stop_on_red_is_satisfied_when_the_plan_stops_short(rules):
    signal = signal_from_columns(
        light_gate=constant(LIMITS.gate_magnitude_m),
        stop_slack=constant(3.0),
    )
    assert robustness(rules, "stop_on_red", signal) == pytest.approx(3.0)


def test_the_gate_is_large_enough_for_a_realistic_overshoot(rules):
    """Guards the sizing argument in `Limits.gate_magnitude_m`.

    Anything a driver would recognise as running a light must be reported
    faithfully rather than clipped at the gate.
    """
    overshoot = 8.0
    signal = signal_from_columns(
        light_gate=constant(LIMITS.gate_magnitude_m),
        stop_slack=constant(-overshoot),
    )
    assert robustness(rules, "stop_on_red", signal) == pytest.approx(-overshoot)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_collision_free_reports_the_closest_approach(rules):
    signal = signal_from_columns(clearance=spike(20.0, 0.3, at=15))
    assert robustness(rules, "collision_free", signal) == pytest.approx(0.3)


def test_collision_free_is_violated_by_any_overlap(rules):
    signal = signal_from_columns(clearance=spike(20.0, -0.7, at=15))
    assert robustness(rules, "collision_free", signal) == pytest.approx(-0.7)


def test_safe_headway_reports_the_tightest_following_slack(rules):
    signal = signal_from_columns(headway_slack=spike(30.0, -1.25, at=9))
    assert robustness(rules, "safe_headway", signal) == pytest.approx(-1.25)


def test_stay_on_route_reports_the_remaining_distance(rules):
    signal = signal_from_columns(route_slack=spike(40.0, 2.0, at=T - 1))
    assert robustness(rules, "stay_on_route", signal) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Lane keeping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_lane_keeping_is_symmetric_and_zero_at_the_boundary(rules, sign):
    signal = signal_from_columns(lateral_offset=constant(sign * LIMITS.lane_half_width_m))
    assert robustness(rules, "lane_keeping", signal) == pytest.approx(0.0, abs=1e-6)


def test_lane_keeping_reports_the_worst_excursion(rules):
    signal = signal_from_columns(lateral_offset=spike(0.0, -1.0, at=20))
    assert robustness(rules, "lane_keeping", signal) == pytest.approx(
        LIMITS.lane_half_width_m - 1.0
    )


# ---------------------------------------------------------------------------
# Comfort
# ---------------------------------------------------------------------------


def test_comfort_reports_the_tightest_of_its_six_bounds(rules):
    """A jerk spike must be the binding term even with accel slack to spare."""
    signal = signal_from_columns(
        accel=constant(0.5),
        jerk=spike(0.0, 3.6, at=17),
        lateral_accel=constant(0.2),
    )
    assert robustness(rules, "comfort", signal) == pytest.approx(LIMITS.max_jerk_mps3 - 3.6)


def test_comfort_catches_harsh_braking(rules):
    signal = signal_from_columns(accel=spike(0.0, -5.0, at=4))
    assert robustness(rules, "comfort", signal) == pytest.approx(LIMITS.max_decel_mps2 - 5.0)


def test_comfort_catches_a_sharp_lateral_swerve(rules):
    signal = signal_from_columns(lateral_accel=spike(0.0, 3.0, at=8))
    assert robustness(rules, "comfort", signal) == pytest.approx(LIMITS.max_lat_accel_mps2 - 3.0)


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def test_makes_progress_takes_the_best_moment_in_the_horizon(rules):
    """`Eventually` is a max over time -- reaching the target once suffices."""
    signal = signal_from_columns(progress_slack=spike(-5.0, 2.0, at=T - 1))
    assert robustness(rules, "makes_progress", signal) == pytest.approx(2.0)


def test_makes_progress_fails_when_the_plan_never_covers_the_distance(rules):
    signal = signal_from_columns(progress_slack=constant(-7.5))
    assert robustness(rules, "makes_progress", signal) == pytest.approx(-7.5)


def test_keep_moving_ignores_a_stall_before_the_window_opens(rules):
    """Pins that intervals are timestep indices, not seconds.

    The window opens at 1.0 s, which is index 10 at a 0.1 s step. A stall at
    index 5 is inside the first second and must be tolerated -- pulling away
    from a standstill takes time.
    """
    stalled_early = signal_from_columns(speed=spike(6.0, 0.0, at=5))
    assert robustness(rules, "keep_moving", stalled_early) > 0.0


def test_keep_moving_catches_a_stall_inside_the_window(rules):
    stalled_late = signal_from_columns(speed=spike(6.0, 0.0, at=12))
    assert robustness(rules, "keep_moving", stalled_late) == pytest.approx(-LIMITS.creep_speed_mps)


def test_keep_moving_is_vacuous_at_a_red_light(rules):
    """Stopping at a red light must not read as a liveness failure."""
    signal = signal_from_columns(
        light_gate=constant(LIMITS.gate_magnitude_m),
        speed=constant(0.0),
    )
    assert robustness(rules, "keep_moving", signal) == pytest.approx(LIMITS.gate_magnitude_m)


def test_keep_moving_is_vacuous_behind_a_close_vehicle(rules):
    """Nor may queueing behind a stopped vehicle."""
    signal = signal_from_columns(
        headway_slack=constant(0.5),  # below standstill_gap_m
        speed=constant(0.0),
    )
    assert robustness(rules, "keep_moving", signal) > 0.0


# ---------------------------------------------------------------------------
# The specification as a whole
# ---------------------------------------------------------------------------


def test_spec_robustness_is_the_minimum_over_the_rules(rules):
    """What makes the per-rule breakdown a decomposition rather than an aside."""
    signal = signal_from_columns(
        clearance=spike(20.0, 0.4, at=3),
        headway_slack=spike(30.0, 1.1, at=7),
        lateral_offset=spike(0.0, 0.9, at=12),
    )
    per_rule = rules.per_rule(signal)
    assert float(rules.spec_robustness(signal)) == pytest.approx(
        min(float(value) for value in per_rule.values())
    )


def test_an_all_satisfying_signal_has_positive_robustness(rules):
    signal = signal_from_columns(speed=constant(6.0))
    assert float(rules.spec_robustness(signal)) > 0.0


def test_hard_violations_are_detected_for_the_fallback(rules):
    signal = signal_from_columns(clearance=spike(20.0, -1.0, at=15))
    per_rule = rules.per_rule(signal)
    assert rules.any_hard_violated(per_rule)
    assert rules.worst_violated(per_rule) == "collision_free"


def test_a_soft_violation_does_not_trigger_the_fallback(rules):
    signal = signal_from_columns(lateral_offset=spike(0.0, 1.4, at=15))
    per_rule = rules.per_rule(signal)
    assert not rules.any_hard_violated(per_rule)
    assert rules.worst_violated(per_rule) == "lane_keeping"


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def test_the_objective_shows_diminishing_returns_past_the_margin(rules):
    """Slack beyond the margin must all but stop earning reward.

    This is what lets the liveness rules win once safety is comfortably met --
    without it the search buys clearance it does not need instead of driving.
    The saturation is a ``tanh`` rather than a clamp, so the reward decays
    asymptotically instead of stopping dead; what matters is that the marginal
    value collapses.
    """

    def objective_at(clearance: float) -> float:
        return float(rules.objective(signal_from_columns(clearance=constant(clearance))))

    # Equal-width windows, so this compares marginal rates rather than spans.
    gain_below_margin = objective_at(1.8) - objective_at(0.2)
    gain_far_beyond = objective_at(6.6) - objective_at(5.0)
    assert gain_below_margin > 0.0
    assert gain_far_beyond < 0.05 * gain_below_margin  # over an order of magnitude


def test_the_objective_still_rewards_slack_below_the_margin(rules):
    tight = signal_from_columns(clearance=constant(0.2))
    comfortable = signal_from_columns(clearance=constant(1.8))
    assert float(rules.objective(comfortable)) > float(rules.objective(tight))


def test_a_deep_hard_violation_is_penalised_without_limit(rules):
    """Hard rules saturate only above; severity always registers.

    A soft rule would flatten out, but "twice as far into an obstacle" has to
    keep scoring worse or the optimizer has no reason to climb back out.

    Measured on a rule set holding `collision_free` alone. Clearance appears
    in two rules now -- it also tells the liveness rules whether something is
    obstructing us -- so varying it moves more than one term of the full
    objective, and the isolation is what makes the arithmetic checkable.
    """
    from stl_driver.ruleset import RuleSet

    collision_free = RuleSet(tuple(rule for rule in rules.rules if rule.name == "collision_free"))

    def objective_at(clearance: float) -> float:
        return float(collision_free.objective(signal_from_columns(clearance=constant(clearance))))

    assert objective_at(-1.0) - objective_at(-6.0) == pytest.approx(50.0, rel=1e-3)
    assert objective_at(-6.0) - objective_at(-11.0) == pytest.approx(50.0, rel=1e-3)


def test_a_deep_soft_violation_is_bounded_in_influence(rules):
    """And the converse: a preference cannot escalate without limit."""

    def objective_at(offset: float) -> float:
        return float(rules.objective(signal_from_columns(lateral_offset=constant(offset))))

    near = objective_at(2.0) - objective_at(4.0)
    far = objective_at(8.0) - objective_at(10.0)
    assert far < 0.05 * near


def test_the_objective_prefers_progress_once_safety_is_satisfied(rules):
    """The anti-parking property, stated at the objective level."""
    parked = signal_from_columns(speed=constant(0.0), progress_slack=constant(-16.0))
    driving = signal_from_columns(speed=constant(8.0), progress_slack=constant(0.5))
    assert float(rules.objective(driving)) > float(rules.objective(parked))


def test_the_objective_prefers_safety_over_progress_when_they_conflict(rules):
    """And the reverse: progress must never buy its way through a collision."""
    unsafe_but_moving = signal_from_columns(
        speed=constant(10.0), progress_slack=constant(4.0), clearance=constant(-1.0)
    )
    safe_but_stopped = signal_from_columns(
        speed=constant(0.0), progress_slack=constant(-16.0), clearance=constant(3.0)
    )
    assert float(rules.objective(safe_but_stopped)) > float(rules.objective(unsafe_but_moving))


def test_every_rule_has_a_description(rules):
    """`stl-driver explain` renders these; an empty one is a silent gap."""
    assert all(rule.description for rule in rules.rules)


def test_the_rule_set_covers_the_four_declared_families(rules):
    expected = {
        "speed_limit",
        "no_reverse",
        "stop_on_red",
        "collision_free",
        "safe_headway",
        "stay_on_route",
        "lane_keeping",
        "comfort",
        "makes_progress",
        "keep_moving",
    }
    assert set(rules.names) == expected


def test_rule_windows_follow_the_horizon_rather_than_hardcoded_indices():
    """`keep_moving` must mean "after one second" whatever the step size is.

    Intervals are indices, so a rule that hardcoded 10 would silently become a
    half-second window if `step_s` were halved.
    """
    from stl_driver.config import Horizon

    fine = STLDriverConfig(horizon=Horizon(n_steps=80, step_s=0.05))
    assert fine.horizon.index_at(1.0) == 20
    build_rules(fine)  # must construct without tripping the interval guard

    tiny = STLDriverConfig(horizon=Horizon(n_steps=4, step_s=0.1))
    build_rules(tiny)  # windows clamp into the horizon rather than overflowing


def test_a_rule_whose_interval_overflows_the_horizon_is_rejected():
    """The guard that stops an overflowing window reading as -1e9."""
    from stl_driver.ruleset import Rule, validate_rules
    from stl_driver.signals import Col, column
    from stl_driver.stl_backend import Always

    bad = Rule(
        "overflowing",
        Always(column("speed", Col.SPEED) > 0.0, interval=[0, T + 3]),
        weight=1.0,
        margin=1.0,
        description="deliberately malformed",
    )
    with pytest.raises(ValueError, match="does not fit"):
        validate_rules((bad,), T)


def test_margins_and_weights_are_positive_and_finite(rules):
    assert all(rule.weight > 0.0 for rule in rules.rules)
    assert all(rule.margin > 0.0 for rule in rules.rules)
    # Finiteness is the safety property: see `RuleSet.objective`.
    assert all(math.isfinite(rule.margin) for rule in rules.rules)


# ---------------------------------------------------------------------------
# Swapping the specification out
# ---------------------------------------------------------------------------


def test_a_different_specification_can_be_supplied_without_editing_anything():
    """`rules.py` is meant to be replaceable, so this is the contract for it.

    A caller hands `EgoPolicy` its own rule set; nothing in the package needs
    editing or monkey-patching for the vehicle to drive by different rules.
    """
    from stl_driver.policy import EgoPolicy
    from stl_driver.ruleset import Rule, RuleSet
    from stl_driver.signals import Col, column
    from stl_driver.stl_backend import Always

    crawl = RuleSet(
        (
            Rule(
                "never_exceed_walking_pace",
                Always(column("speed", Col.SPEED) < 1.4),
                weight=1.0,
                margin=1.0,
                hard=True,
                description="A deliberately different specification.",
            ),
        )
    )
    policy = EgoPolicy(rules=crawl)
    assert policy.rules.names == ("never_exceed_walking_pace",)

    signal = signal_from_columns(speed=constant(9.0))
    assert float(policy.rules.spec_robustness(signal)) < 0.0
    assert float(policy.rules.spec_robustness(signal_from_columns(speed=constant(0.5)))) > 0.0


def test_the_machinery_does_not_depend_on_this_particular_specification():
    """`ruleset` must stay generic, or swapping the spec means editing it too.

    Checked structurally rather than by grepping prose: what matters is that
    the machinery imports nothing that knows about driving, and defines no
    rules of its own.
    """
    import ast
    import pathlib as _pathlib

    from stl_driver import ruleset

    tree = ast.parse(_pathlib.Path(ruleset.__file__).read_text())
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    domain = {"stl_driver.signals", "stl_driver.config", "stl_driver.rules"}
    assert not (imported & domain), f"machinery imports {imported & domain}"

    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "build_rules" not in defined


def test_the_liveness_gate_excuses_an_obstruction_that_is_not_a_lead_vehicle(rules):
    """Crossing traffic and vehicles alongside stop a car too.

    Gating only on the lead vehicle missed most of what actually holds a car
    up in a city: measured over 500 steps, 90 of 165 reported `makes_progress`
    violations had no lead vehicle at all but something within two metres.
    """
    blocked = signal_from_columns(
        speed=constant(0.0), progress_slack=constant(-12.0), clearance=constant(0.5)
    )
    assert robustness(rules, "makes_progress", blocked) > 0.0
    assert robustness(rules, "keep_moving", blocked) > 0.0


def test_the_progress_gate_looks_at_the_whole_horizon_not_just_now(rules):
    """A road clear now and blocked in two seconds is not one to accelerate down.

    `makes_progress` is about distance covered *over* the horizon, so asking
    only about this instant is the wrong question -- and demanding progress
    there demands a plan the safety rules forbid.
    """
    clear_then_blocked = torch.full((T,), 40.0)
    clear_then_blocked[20:] = 0.5
    signal = signal_from_columns(
        speed=constant(0.0), progress_slack=constant(-12.0), clearance=clear_then_blocked
    )
    assert robustness(rules, "makes_progress", signal) > 0.0


def test_a_genuinely_clear_road_still_demands_progress(rules):
    """The gate must not become an excuse for never moving."""
    idle = signal_from_columns(speed=constant(0.0), progress_slack=constant(-12.0))
    assert robustness(rules, "makes_progress", idle) < 0.0
    assert robustness(rules, "keep_moving", idle) < 0.0
