# SPDX-License-Identifier: Apache-2.0
"""**The driving specification.** This is the file to edit.

Ten formulas in Signal Temporal Logic that say what the vehicle may and may
not do.  Everything else in the package -- the rollout, the search, the gRPC
plumbing -- exists so that these can be evaluated on a candidate plan and
maximised.  Change the behaviour here, not in the planner.

The file is deliberately nothing but the specification: :class:`Rule` and
:class:`RuleSet`, which say what a rule *is* and how a set of them is scored,
live in :mod:`stl_driver.ruleset` and do not need touching to change what the
car does.

Replacing it wholesale
----------------------
:func:`build_rules` is the only thing anyone outside this file calls, so an
alternative specification is a module that offers the same function::

    from stl_driver.policy import EgoPolicy
    from stl_driver.ruleset import RuleSet
    from my_rules import build_rules

    policy = EgoPolicy(rules=RuleSet(build_rules(config)))

``EgoPolicy`` takes the rule set directly, so nothing needs editing or
monkey-patching to drive by a different set of rules.

How a rule takes part
---------------------
``weight``
    Its pull in the objective the optimizer ascends.

``margin``
    How much slack is enough.  The objective saturates each rule's
    contribution around its margin, so a safety rule that already holds with
    room to spare stops competing with the liveness rules.  Without this, a
    plan would keep buying more clearance it does not need at the cost of ever
    getting anywhere.

``hard``
    Whether a violation may summon the emergency plan, and how the objective
    treats a violation.  See :meth:`RuleSet.objective`.

Every rule reads the ``[T, D]`` signal described in :mod:`stl_driver.signals`,
whose columns are all *slacks* -- quantities that should stay positive -- so
each rule compares against the constant ``0``.  Speed limits, stop-line
positions and route lengths change every step; folding them into the column
rather than the formula is what lets the formula tree be built once instead of
at 10 Hz.
"""

from __future__ import annotations

from stl_driver.config import STLDriverConfig
from stl_driver.ruleset import Rule, RuleSet, validate_rules
from stl_driver.signals import Col, column
from stl_driver.stl_backend import Always, And, Eventually, Implies

# Re-exported so that `from stl_driver.rules import RuleSet` keeps working and
# a replacement specification can import everything it needs from one place.
__all__ = ["Rule", "RuleSet", "build_rules"]


def build_rules(config: STLDriverConfig) -> tuple[Rule, ...]:
    """Construct the specification.

    Called once, when the policy is built.  Nothing scenario-dependent appears
    here -- speed limits, stop-line positions and route lengths reach the
    formulas through the signal's columns, not through these constants.
    """
    limits = config.limits
    horizon = config.horizon
    last = horizon.n_steps - 1

    speed = column("speed", Col.SPEED)
    accel = column("accel", Col.ACCEL)
    jerk = column("jerk", Col.JERK)
    offset = column("lateral_offset", Col.LATERAL_OFFSET)
    lateral_accel = column("lateral_accel", Col.LATERAL_ACCEL)
    clearance = column("clearance", Col.CLEARANCE)
    headway = column("headway_slack", Col.HEADWAY_SLACK)
    stop_slack = column("stop_slack", Col.STOP_SLACK)
    light = column("light_gate", Col.LIGHT_GATE)
    route = column("route_slack", Col.ROUTE_SLACK)
    progress = column("progress_slack", Col.PROGRESS_SLACK)
    headroom = column("speed_headroom", Col.SPEED_HEADROOM)

    # `keep_moving` only starts asking after a second, so that a plan is
    # allowed to pull away from a standstill rather than being required to
    # already be moving at t=0.
    moving_from = horizon.index_at(1.0)

    rules = (
        # -- legality ------------------------------------------------------
        Rule(
            "speed_limit",
            Always(headroom > -limits.speed_tolerance_mps),
            weight=2.0,
            margin=1.0,
            description="Never exceed the posted limit by more than the tolerance.",
        ),
        Rule(
            "no_reverse",
            # Currently guaranteed by the model -- `rollout` floors the speed
            # at zero -- so this reads as a tautology today. It is kept
            # because it states the requirement rather than leaving it as an
            # unwritten property of the kinematics: if the model ever gains a
            # reverse gear, or the floor is relaxed to recover gradient, this
            # is the line that notices.
            Always(speed > -0.05),
            weight=4.0,
            margin=0.5,
            hard=True,
            description="Never plan to travel backwards along the route.",
        ),
        Rule(
            "stop_on_red",
            # The gate is a signed constant, not a distance, so an inactive
            # light makes this vacuously true at exactly +gate_magnitude
            # instead of at whatever the stop slack happens to be.
            Always(Implies(light > 0.0, stop_slack > 0.0)),
            weight=6.0,
            margin=2.0,
            hard=True,
            description="While a red or amber light governs us, never cross its stop line.",
        ),
        # -- safety --------------------------------------------------------
        Rule(
            "collision_free",
            # The margin is already folded into the column, so the threshold
            # stays 0 like every other rule's.
            Always(clearance > 0.0),
            weight=10.0,
            margin=2.0,
            hard=True,
            description="Keep clearance from every predicted actor at all times.",
        ),
        Rule(
            "safe_headway",
            Always(headway > 0.0),
            weight=5.0,
            margin=3.0,
            description="Stay at least a time headway behind the vehicle ahead.",
        ),
        Rule(
            "stay_on_route",
            Always(route > 0.0),
            weight=4.0,
            margin=5.0,
            hard=True,
            description="Never plan past the end of the published route.",
        ),
        # -- lane keeping --------------------------------------------------
        Rule(
            "lane_keeping",
            # Judged from half a second in, not from now.
            #
            # The plan starts at the *measured* lateral offset, so over the
            # full horizon the worst value is whatever the vehicle inherited
            # and no choice of parameters can change it. An `Always` from t=0
            # is therefore not merely uninformative -- its gradient to the
            # lateral parameters is identically zero, and a car nudged out of
            # lane would have no reason to come back. Excluding the opening
            # moments asks the plan about the part of the future it controls.
            Always(
                (offset < limits.lane_half_width_m) & (offset > -limits.lane_half_width_m),
                interval=[horizon.index_at(0.5), last],
            ),
            weight=3.0,
            # The margin is nearly the full half-width, which is what makes
            # this rule *centre* the vehicle rather than merely keep it inside
            # the corridor. A small margin would saturate as soon as the plan
            # was legal, and a car sitting 0.9 m off centre in a 1.2 m corridor
            # would have no reason to come back.
            margin=limits.lane_half_width_m * 0.85,
            description="Stay within the corridor around the route centreline.",
        ),
        # -- comfort -------------------------------------------------------
        Rule(
            "comfort",
            Always(
                (accel < limits.max_accel_mps2)
                & (accel > -limits.max_decel_mps2)
                & (jerk < limits.max_jerk_mps3)
                & (jerk > -limits.max_jerk_mps3)
                & (lateral_accel < limits.max_lat_accel_mps2)
                & (lateral_accel > -limits.max_lat_accel_mps2)
            ),
            weight=1.0,
            margin=0.5,
            description="Keep acceleration, jerk and lateral acceleration within comfort bounds.",
        ),
        # -- liveness ------------------------------------------------------
        # Standing still satisfies every rule above. These are what stop the
        # optimizer from discovering that and parking forever. Stating them as
        # logic rather than as a progress bonus in the cost keeps the whole
        # specification in one language -- and means a stopped car reports
        # *which* requirement it is failing.
        Rule(
            "makes_progress",
            # Gated on the same condition as `keep_moving`, so that waiting at
            # a red light or queueing behind a stopped vehicle is not reported
            # as a failure to make progress. Without the gate, every correct
            # stop would drive the specification's robustness negative and the
            # reported value would stop being a useful health signal.
            #
            # Progress is monotone, so `Eventually` reduces to its value at the
            # end of the horizon: a linear reward with a clean gradient.
            Implies(
                And(light < 0.0, headway > limits.standstill_gap_m),
                Eventually(progress > 0.0, interval=[0, last]),
            ),
            weight=2.0,
            # Finite, and that matters: an unbounded progress reward can
            # outbid a hard safety violation outright. Surplus distance beyond
            # this buys nothing -- `speed_limit` is what caps the top end.
            margin=limits.progress_margin_m,
            description="Cover a reasonable distance over the planning horizon.",
        ),
        Rule(
            "keep_moving",
            # Redundant with `makes_progress` by design: that one is an
            # integral over the horizon, this one is instantaneous, and they
            # fail in different ways. A plan that creeps the whole time fails
            # progress; one that lurches and stops fails this.
            Always(
                Implies(
                    And(light < 0.0, headway > limits.standstill_gap_m),
                    speed > limits.creep_speed_mps,
                ),
                interval=[moving_from, last],
            ),
            weight=2.0,
            margin=1.0,
            description="With no red light and nothing close ahead, do not crawl.",
        ),
    )

    validate_rules(rules, horizon.n_steps)
    return rules
