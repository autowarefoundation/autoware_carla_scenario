# SPDX-License-Identifier: Apache-2.0
"""The machinery a specification is made of: what a rule *is*, and how a set of
them is evaluated.

Nothing here says anything about driving.  The rules themselves live in
:mod:`stl_driver.rules`, which is the file to edit or replace when you want
different behaviour; this module is what stays the same underneath.

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

from stl_driver.stl_backend import EXACT, STLFormula, assert_interval_fits, conjoin

__all__ = ["Rule", "RuleSet", "validate_rules"]


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


def validate_rules(rules: tuple[Rule, ...], n_steps: int) -> None:
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

    def to(self, device: str) -> RuleSet:
        """Move the weights and margins the objective needs onto a device.

        The formulas themselves allocate their working tensors on whatever
        device the signal arrives on, so only these two need moving.
        """
        self._weights = self._weights.to(device)
        self._margins = self._margins.to(device)
        return self

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
            total = total + self._weights[index] * self._saturate(robustness, index, rule)
        return total

    def _saturate(self, robustness: torch.Tensor, index: int, rule: Rule) -> torch.Tensor:
        """One rule's contribution to the objective; see :meth:`objective`."""
        if not math.isfinite(rule.margin):
            return robustness
        margin = self._margins[index]
        saturated = margin * torch.tanh(robustness / margin)
        if rule.hard:
            # Keep violations linear and unbounded; saturate rewards.
            saturated = torch.minimum(robustness, saturated)
        return saturated

    def merit_and_robustness(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Both exact scores, from one evaluation of the rules.

        :meth:`objective` and :meth:`spec_robustness` are two reductions of the
        same per-rule robustness values -- a weighted saturated sum and a
        minimum -- and every caller that wants one wants the other, since
        selection is lexicographic in the pair. Evaluating the formulas twice
        for that was the single largest cost in a policy step: at ten rules
        over a forty-step horizon each pass costs about as much as the rollout
        it is judging, twice over.

        Equal to calling the two separately, which
        :mod:`tests.test_ruleset` asserts directly -- the conjunction *is* the
        minimum over the rules, which is also what makes the per-rule breakdown
        in ``debug_scalars`` a genuine decomposition.
        """
        total = torch.zeros((), dtype=torch.float32)
        worst: torch.Tensor | None = None
        for index, rule in enumerate(self.rules):
            robustness = rule.formula.robustness(signal, **EXACT)
            worst = robustness if worst is None else torch.minimum(worst, robustness)
            total = total + self._weights[index] * self._saturate(robustness, index, rule)
        assert worst is not None  # a RuleSet is never empty; `conjoin` refuses one
        return total, worst

    def worst_violated(self, per_rule: Mapping[str, torch.Tensor | float]) -> str | None:
        """Name of the most-violated rule, or ``None`` if all are satisfied."""
        violations = {name: float(value) for name, value in per_rule.items() if float(value) < 0.0}
        if not violations:
            return None
        return min(violations, key=lambda name: violations[name])

    def any_hard_violated(self, per_rule: Mapping[str, torch.Tensor | float]) -> bool:
        """Whether a rule marked ``hard`` is violated anywhere in the horizon."""
        return any(float(per_rule[name]) < 0.0 for name in self.hard_rules if name in per_rule)

    def hard_robustness(self, signal: torch.Tensor, n_steps: int) -> float:
        """How well the hard rules hold over the opening ``n_steps``.

        The same quantity :meth:`imminent_hard_violation` tests the sign of,
        kept separately so two plans can be compared on it even when both are
        violating.
        """
        window = signal[: max(2, min(n_steps, signal.shape[0]))]
        return min(
            float(next(r for r in self.rules if r.name == name).formula.robustness(window, **EXACT))
            for name in self.hard_rules
        )

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
