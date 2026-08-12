# SPDX-License-Identifier: Apache-2.0
"""The driving specification, in Signal Temporal Logic.

This module is the point of the package.  Everything else -- the rollout, the
optimizer, the gRPC plumbing -- exists so that these formulas can be evaluated
on a candidate plan and maximised.  If you want to change how the car behaves,
change it here, not in the planner.

Each rule is an STL formula over the signal described in
:mod:`stl_driver.signals`, together with three numbers that say how it takes
part in the search:

``weight``
    Its pull in the optimizer's objective.

``margin``
    How much slack is enough.  The objective saturates each rule's
    contribution around its margin, so a safety rule that already holds with
    room to spare stops competing with the liveness rules.  Without this, a
    plan would keep buying more clearance it does not need at the cost of ever
    getting anywhere.

``hard``
    Whether a violation should abandon optimization for the fallback plan
    rather than emit a best-effort trajectory.  It also decides how the
    objective treats a violation: see :meth:`RuleSet.objective`.

Two evaluations, deliberately different
---------------------------------------
:meth:`RuleSet.spec_robustness` is the plain conjunction of every rule under
exact min/max semantics.  That is the specification -- what gets reported, and
what decides whether a plan is acceptable.

:meth:`RuleSet.objective` is a weighted, saturated sum.  That is what the
optimizer ascends.  They are not the same function on purpose: a min of mins
is flat in every rule except the currently-worst one, so a gradient step
improves exactly one rule at a time and the others are invisible.  Keeping the
scoring function separate from the specification is what makes the search
tractable without weakening what "satisfied" means.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from stl_driver.config import STLDriverConfig
from stl_driver.signals import Col, column
from stl_driver.stl_backend import (
    EXACT,
    Always,
    And,
    Eventually,
    Implies,
    STLFormula,
    assert_interval_fits,
    conjoin,
)

__all__ = ["Rule", "RuleSet", "build_rules"]


@dataclass(frozen=True)
class Rule:
    """One named clause of the specification."""

    name: str
    formula: STLFormula
    weight: float
    margin: float
    hard: bool = False
    #: One line, surfaced by ``stl-driver explain``.
    description: str = ""


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

    _validate(rules, horizon.n_steps)
    return rules


def _validate(rules: tuple[Rule, ...], n_steps: int) -> None:
    """Catch the mistakes that would otherwise surface as strange behaviour."""
    names = [rule.name for rule in rules]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"duplicate rule names: {sorted(duplicates)}")

    for rule in rules:
        for interval in _intervals(rule.formula):
            assert_interval_fits(interval, n_steps, what=f"rule {rule.name!r}")
        if rule.weight <= 0.0:
            raise ValueError(f"rule {rule.name!r} has non-positive weight {rule.weight}")
        if rule.margin <= 0.0:
            raise ValueError(f"rule {rule.name!r} has non-positive margin {rule.margin}")


def _intervals(formula: STLFormula) -> list[Any]:
    """Every temporal interval in a formula tree.

    Walks children by attribute rather than by type so it keeps working if a
    rule starts using an operator this module does not import.
    """
    found: list[Any] = []
    stack: list[Any] = [formula]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        interval = getattr(node, "interval", None)
        if interval is not None:
            found.append(interval)
        for attribute in ("subformula", "subformula1", "subformula2"):
            child = getattr(node, attribute, None)
            if child is not None:
                stack.append(child)
    return found


class RuleSet:
    """The specification, plus the two ways of evaluating it."""

    def __init__(self, rules: tuple[Rule, ...]) -> None:
        self.rules = rules
        self.spec = conjoin(rule.formula for rule in rules)
        self._weights = torch.tensor([rule.weight for rule in rules], dtype=torch.float32)
        self._margins = torch.tensor([rule.margin for rule in rules], dtype=torch.float32)
        self.hard_rules = tuple(rule.name for rule in rules if rule.hard)

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self.rules)

    def spec_robustness(self, signal: torch.Tensor) -> torch.Tensor:
        """The specification's robustness: exact, and the number we report.

        Equal to the minimum over the individual rules, which is what makes
        the per-rule breakdown in ``debug_scalars`` a genuine decomposition of
        this value rather than a parallel calculation.
        """
        return self.spec.robustness(signal, **EXACT)

    def per_rule(self, signal: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Each rule's robustness separately, for reporting and diagnosis."""
        evaluation = kwargs or EXACT
        return {rule.name: rule.formula.robustness(signal, **evaluation) for rule in self.rules}

    def objective(self, signal: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """What the optimizer ascends: a weighted, saturated sum.

        Saturating around each rule's margin is what lets the search make
        trade-offs.  A conjunction would report only the worst rule, so a
        gradient step could improve nothing else; an unsaturated sum would let
        a rule with lots of slack keep earning reward for slack nobody needs,
        which in practice means buying clearance instead of making progress.

        The saturation is ``margin * tanh(rho / margin)`` rather than a clamp
        so that the gradient never dies outright -- a plan far outside a bound
        still learns which way to move, which a clamp would deny it.

        Soft and hard rules saturate differently, and the asymmetry is the
        whole safety argument:

        * A **soft** rule saturates in *both* directions.  It expresses a
          preference, so its influence is bounded either way and no amount of
          preference can outbid a safety violation.
        * A **hard** rule saturates only above.  Rewards for surplus safety
          level off, but the penalty for violating one grows without limit, so
          a collision always dominates however much progress is on offer.

        Getting this wrong is not subtle in its consequences: with an
        unbounded progress reward, a plan that clips an obstacle while making
        good speed outscores one that stops safely.
        """
        total = torch.zeros((), dtype=torch.float32)
        for index, rule in enumerate(self.rules):
            robustness = rule.formula.robustness(signal, **kwargs)
            margin = self._margins[index]
            if math.isfinite(rule.margin):
                saturated = margin * torch.tanh(robustness / margin)
                if rule.hard:
                    # Keep violations linear and unbounded; saturate rewards.
                    saturated = torch.minimum(robustness, saturated)
                robustness = saturated
            total = total + self._weights[index] * robustness
        return total

    def worst_violated(self, per_rule: Mapping[str, torch.Tensor | float]) -> str | None:
        """Name of the most-violated rule, or ``None`` if all are satisfied."""
        violations = {name: float(value) for name, value in per_rule.items() if float(value) < 0.0}
        if not violations:
            return None
        return min(violations, key=lambda name: violations[name])

    def any_hard_violated(self, per_rule: Mapping[str, torch.Tensor | float]) -> bool:
        """Whether a rule marked ``hard`` is violated anywhere in the horizon."""
        return any(float(per_rule[name]) < 0.0 for name in self.hard_rules if name in per_rule)

    def imminent_hard_violation(self, signal: torch.Tensor, n_steps: int) -> str | None:
        """The first hard rule violated within the opening ``n_steps``, if any.

        This, not :meth:`any_hard_violated`, is what should summon the
        emergency plan.  A hard rule violated at the *end* of a four-second
        horizon means the best plan found this step runs into trouble three
        seconds out -- against a constant-velocity forecast of traffic that
        will itself brake, turn and yield in the meantime.  Treating that the
        same as a violation happening now means braking at seven metres per
        second squared for a car crossing an intersection thirty metres away,
        which is both wrong and, at 10 Hz, wrong again a tenth of a second
        later.

        Restricting the check to the opening of the horizon asks the question
        that actually justifies an emergency stop: is there something we can
        no longer plan our way out of?  Everything further out is left to the
        optimizer, which is already slowing for it, and to the next replan,
        which will know more.

        Evaluating on a truncated signal works because every hard rule is an
        unbounded ``Always``; a rule with a bounded interval would be padded
        past the end and read as catastrophically violated, so
        :func:`build_rules` keeps the hard ones unbounded.
        """
        window = signal[: max(2, min(n_steps, signal.shape[0]))]
        for name in self.hard_rules:
            rule = next(r for r in self.rules if r.name == name)
            if float(rule.formula.robustness(window, **EXACT)) < 0.0:
                return name
        return None
