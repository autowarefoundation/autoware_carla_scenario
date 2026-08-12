# SPDX-License-Identifier: Apache-2.0
"""Finding the plan that best satisfies the specification.

Two stages, and the split is not incidental.

**Sample first.**  A quasi-random sweep of the six-dimensional parameter box,
evaluated in one batch.  This is what makes the search robust to the shape of
the robustness landscape: once a plan already passes through an obstacle, its
clearance barely depends on the speed profile -- the plan goes through the
obstacle at any speed -- so the gradient there is close to flat and descent
alone would never escape.  Sampling does not care.

**Then descend.**  A handful of Adam steps on the single best sample, against
a smoothed robustness.  This is what makes the result precise rather than
merely the best of sixty-four guesses: it converts "brake somewhat" into
"brake exactly hard enough to stop at the line".

Sampling without descent plans coarsely; descent without sampling gets stuck.
The combination is worth more than either.

Throughout, the *scoring* function is the weighted objective while the
*acceptance* test is the plain specification -- see :mod:`stl_driver.ruleset`
for why those must not be the same function.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from stl_driver.config import STLDriverConfig
from stl_driver.rollout import N_PARAMS, RolloutInputs, rollout
from stl_driver.ruleset import RuleSet
from stl_driver.stl_backend import smooth

logger = logging.getLogger(__name__)

__all__ = ["PlanSolution", "RobustnessOptimizer"]


@dataclass
class PlanSolution:
    """The chosen plan, and enough about the search to explain it."""

    params: torch.Tensor
    signal: torch.Tensor
    robustness: float
    per_rule: dict[str, float]
    #: Best candidates from the sampling stage, for `sampled_trajectories`.
    alternatives: list[torch.Tensor] = field(default_factory=list)
    n_refine_steps: int = 0
    solve_seconds: float = 0.0
    #: Set when a hard rule is violated soon enough that no plan avoids it.
    needs_fallback: bool = False
    #: Which hard rule that was, if any.
    imminent_rule: str | None = None
    #: Name of the worst-violated rule, or None when all are satisfied.
    worst_rule: str | None = None
    #: Whether the batched fast path was used, for diagnosis.
    used_vmap: bool = True


class RobustnessOptimizer:
    """Searches plan space for the highest-robustness trajectory."""

    def __init__(self, rules: RuleSet, config: STLDriverConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        self.rules = rules.to(self.device)
        self._seeds = self._build_seeds().to(self.device)
        # Set on the first failure, then honoured for the rest of the session.
        # Batching relies on `vmap` seeing through upstream's hardcoded
        # `reshape([T, 1])`; it works on the pinned versions, but a policy that
        # crashed on a torch upgrade would be a poor trade for the speed.
        self._vmap_ok = True
        # Populated by `prepare()` when compilation is enabled; the eager
        # paths are used until then, and again if compilation fails.
        self._compiled_batch: Any = None
        self._compiled_refine: Any = None
        self._compiled_scores: Any = None

    def prepare(self, example: RolloutInputs) -> None:
        """Do the expensive one-off work before the runtime starts waiting.

        Compilation takes about a minute, so it belongs here -- called from
        `EgoPolicy.on_session_start`, which the driver service invokes before
        any observation arrives -- rather than on the first step, where it
        would look like a policy that missed its deadline by a factor of a
        thousand.

        Failure is not fatal.  ``torch.compile`` reaches into a specification
        built out of a third-party library's ``nn.Module`` tree, and if it
        cannot handle some corner of that, the eager path is still correct and
        merely slower.
        """
        settings = self.config.optimizer
        if not settings.compile_search:
            # Still worth one solve: the first in a process runs at a third of
            # the steady-state rate, and would otherwise spend its whole
            # budget on torch's lazy initialisation.
            self.solve(example)
            return

        try:
            self._compiled_batch = torch.compile(self._score_batch_eager, dynamic=True)
            self._compiled_refine = torch.compile(self._refine_losses, dynamic=True)
            self._compiled_scores = torch.compile(self._batched_scores_eager, dynamic=True)
            self.solve(example)
        except Exception:  # noqa: BLE001 - any compile failure falls back
            logger.warning("torch.compile failed; continuing eagerly", exc_info=True)
            self._compiled_batch = None
            self._compiled_refine = None
            self._compiled_scores = None

    # -- seeds -------------------------------------------------------------

    def _build_seeds(self) -> torch.Tensor:
        """Quasi-random parameter seeds, drawn once and reused every step.

        Sobol rather than uniform because it fills the box far more evenly at
        these sample counts, and drawn once rather than per step because the
        runtime's rollouts are seeded and a policy that re-randomised would
        make them irreproducible.
        """
        optimizer = self.config.optimizer
        engine = torch.quasirandom.SobolEngine(
            dimension=N_PARAMS, scramble=True, seed=optimizer.seed
        )
        unit = engine.draw(optimizer.n_samples)
        # Map (0, 1) onto a symmetric box in logit space. +-3 covers almost the
        # whole physical range once the sigmoid and tanh are applied, without
        # wasting samples deep in saturation where plans are indistinguishable.
        return (unit * 2.0 - 1.0) * 3.0

    def _candidates(self, warm_start: torch.Tensor | None) -> torch.Tensor:
        """The sampling stage's parameter batch.

        The Sobol sweep plus three plans worth having regardless: hold the
        current speed, brake comfortably to a stop, and whatever worked last
        step.  The braking seed matters most -- it is the one that rescues a
        step where every random plan happens to be unsafe.
        """
        limits = self.config.limits
        fixed = [
            _accel_params(0.0, limits),
            _accel_params(-limits.max_decel_mps2 * 0.9, limits),
        ]
        if warm_start is not None:
            fixed.append(warm_start.detach().clone())
        return torch.cat([self._seeds, torch.stack(fixed).to(self.device)], dim=0)

    # -- search ------------------------------------------------------------

    def solve(
        self,
        inputs: RolloutInputs,
        warm_start: torch.Tensor | None = None,
    ) -> PlanSolution:
        """Search for the plan that best satisfies the specification."""
        started = time.perf_counter()
        optimizer = self.config.optimizer

        candidates = self._candidates(warm_start)
        signals, scores, merit, robustness, used_vmap = self._score_batch(candidates, inputs)

        order = torch.argsort(scores, descending=True)
        alternatives = [
            candidates[int(index)].clone()
            for index in order[1 : 1 + optimizer.n_reported_candidates]
        ]

        # Refinement starts from the highest smoothed objective -- the
        # gradients it is about to follow are the smoothed ones, so that is the
        # consistent guide to which plan is worth improving.
        n_seeds = max(1, min(optimizer.n_refine_seeds, len(candidates)))
        seeds = candidates[order[:n_seeds]].clone()

        # The plan we drive is the highest *exact* objective seen anywhere.
        #
        # Not the highest robustness, which is the natural first guess and is
        # wrong: robustness is a minimum over rules, and a rule that has been
        # gated off contributes nothing to a minimum. Where an obstruction
        # excuses the liveness rules -- which is most of city driving -- coming
        # to a halt then violates nothing at all and scores best of anything on
        # offer. Selecting on it makes stopping the answer to every hard
        # question.
        #
        # The objective does not degenerate that way: liveness enters it with a
        # weight, so standing still costs something even when it is permitted.
        # Robustness keeps the job it is good at -- saying whether the
        # specification holds, and deciding the emergency plan below.
        best_sample = _lexicographic_pick(merit, robustness)
        best_params = candidates[best_sample].clone()
        best_signal = signals[best_sample]
        best_merit = float(merit[best_sample])
        best_satisfies = bool(robustness[best_sample] >= 0.0)

        # Carry on from last step's plan unless something is meaningfully
        # better. The warm start is the final entry of the candidate batch, so
        # it has already been rolled out and scored; this only changes which
        # of two near-equal plans is preferred.
        margin = optimizer.continuity_margin
        if warm_start is not None and margin > 0.0:
            carried = len(candidates) - 1
            carried_satisfies = bool(robustness[carried] >= 0.0)
            if carried_satisfies >= best_satisfies and float(merit[carried]) >= best_merit - margin:
                best_sample = carried
                best_satisfies = carried_satisfies
                best_params = candidates[carried].clone()
                best_signal = signals[carried]
                best_merit = float(merit[carried])
                # Descend from the carried plan as well, so continuity is a
                # starting point rather than merely a tie-break at the end.
                seeds = torch.cat([best_params.unsqueeze(0), seeds[:-1]], dim=0)

        refined_params, refined_signal, refined_merit, refined_ok, n_steps = self._refine(
            seeds, inputs, started
        )
        # Refinement has to clear the same bar: it starts from the carried plan
        # when that was preferred, so accepting any improvement at all would
        # let it drift a little every step and undo the continuity.
        bar = margin if warm_start is not None else 0.0
        if (refined_ok, refined_merit) > (best_satisfies, best_merit + bar):
            best_params, best_signal = refined_params, refined_signal

        per_rule = {name: float(value) for name, value in self.rules.per_rule(best_signal).items()}
        best_robustness = float(self.rules.spec_robustness(best_signal))

        # Only a violation we can no longer plan our way out of summons the
        # emergency plan; see `RuleSet.imminent_hard_violation`.
        window = max(2, self.config.horizon.index_at(optimizer.fallback_window_s))
        imminent = self.rules.imminent_hard_violation(best_signal, window)
        if imminent is not None and not self._braking_can_help(inputs):
            # Braking changes nothing, so it is not the answer.
            #
            # A conflict driven by another vehicle's motion rather than our own
            # -- something crossing towards a stationary ego -- is violated
            # just as badly by the hardest stop available. Emergency braking
            # then buys no safety and costs everything else: the vehicle sits
            # under maximum deceleration, makes no progress, and gets rear-
            # ended by traffic that did not expect it to stop.
            #
            # The violation is still reported. What changes is that we stop
            # pretending the emergency plan is a remedy for it.
            imminent = None

        return PlanSolution(
            params=best_params,
            signal=best_signal,
            robustness=best_robustness,
            per_rule=per_rule,
            alternatives=alternatives,
            n_refine_steps=n_steps,
            solve_seconds=time.perf_counter() - started,
            needs_fallback=imminent is not None,
            imminent_rule=imminent,
            worst_rule=self.rules.worst_violated(per_rule),
            used_vmap=used_vmap,
        )

    def _braking_can_help(self, inputs: RolloutInputs) -> bool:
        """Whether stopping harder is capable of changing anything.

        It is, whenever we are moving: the emergency plan decelerates harder
        than any plan the optimizer can express, so it always ends up short of
        where the chosen plan would.  That covers the obstacle two metres ahead
        at ten metres per second -- unavoidable, but braking is still right.

        It is not, when we are already stopped.  A conflict with a vehicle
        moving towards a stationary ego is violated exactly as badly under
        maximum braking, because our speed was never what caused it.  Braking
        there buys nothing and costs everything else: no progress, and a
        vehicle standing on the brakes in traffic that did not expect it to
        stop -- which is how a stall turns into being rear-ended.

        The violation is still reported either way.  What this decides is only
        whether the emergency plan is a *remedy* for it.
        """
        return float(inputs.speed0_mps) > self.config.limits.creep_speed_mps

    def _score_batch(
        self,
        candidates: torch.Tensor,
        inputs: RolloutInputs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Roll out every candidate and score it every way selection needs.

        Returns the signals, the smoothed objective (which guides the search)
        and the exact specification robustness (which decides the answer).

        No gradients are needed here, so the whole stage runs under
        ``inference_mode``: it is the largest tensor work in the step and
        recording a graph for it would be pure waste.
        """
        temperature = self.config.optimizer.temperature_start
        with torch.inference_mode():
            if self._vmap_ok:
                try:
                    batch = self._compiled_batch or self._score_batch_eager
                    signals, scores, merit, robustness = batch(candidates, inputs, temperature)
                    return (
                        signals.clone(),
                        scores.clone(),
                        merit.clone(),
                        robustness.clone(),
                        True,
                    )
                except (RuntimeError, TypeError, IndexError):
                    # Batching is an optimization, not a requirement. Fall back
                    # permanently rather than paying the failure every step.
                    self._vmap_ok = False

            signals = torch.stack([rollout(p, inputs, self.config) for p in candidates])
            scores = torch.stack([self.rules.objective(s, **smooth(temperature)) for s in signals])
            scored = [self.rules.merit_and_robustness(s) for s in signals]
            merit = torch.stack([m for m, _ in scored])
            robustness = torch.stack([r for _, r in scored])
            return signals.clone(), scores.clone(), merit.clone(), robustness.clone(), False

    def _score_batch_eager(
        self,
        candidates: torch.Tensor,
        inputs: RolloutInputs,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The batched sampling stage, as one function so it can be compiled."""
        signals = torch.vmap(lambda p: rollout(p, inputs, self.config))(candidates)
        scores = torch.vmap(lambda s: self.rules.objective(s, **smooth(temperature)))(signals)
        merit, robustness = torch.vmap(self.rules.merit_and_robustness)(signals)
        return signals, scores, merit, robustness

    def _refine(
        self,
        seeds: torch.Tensor,
        inputs: RolloutInputs,
        started: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool, int]:
        """Gradient-ascend the smoothed objective from several seeds at once.

        Descending from one starting point finds the nearest local optimum,
        and which one that is depends entirely on where the sampling stage
        happened to leave us.  Descending from several finds the best of
        several, and costs almost nothing extra: the seeds are independent, so
        one ``vmap`` covers them all in a single batched pass.  Measured, a
        step over eight seeds costs 13.6 ms against 10.9 ms for one on CPU,
        and on a GPU sixty-four cost the same as one.

        The temperature sharpens as it goes: soft early so every timestep
        contributes gradient and the step is well-informed, sharp late so the
        last steps optimize something close to the true minimum they will be
        judged by.

        Every iterate of every seed is scored by the *exact* specification,
        not by the smoothed objective being descended, and the best across all
        of them is kept.  Smoothing can prefer a plan the specification does
        not, and it is the specification that decides.

        What is *not* considered is the seeds themselves.  They came from the
        sampling stage and :meth:`solve` already holds the best of them, so
        returning one here would only launder an unrefined candidate into the
        refined slot -- where it clears the continuity margin, becomes the warm
        start, and is preferred again next step.  With one seed that was
        harmless; with several it was not, because the seed that wins on
        robustness is systematically the one that stops.  Robustness is a
        minimum over rules, and where the liveness rules are gated off by an
        obstruction, stopping violates nothing at all.
        """
        settings = self.config.optimizer
        params = seeds.detach().clone().requires_grad_(True)
        adam = torch.optim.Adam([params], lr=settings.learning_rate)

        # Nothing to report until a step has actually been taken. `solve`
        # compares against its own best, so this loses to everything.
        best_params = seeds[0].detach().clone()
        best_signal = torch.zeros(())
        best_merit = -math.inf
        best_ok = False

        completed = 0
        # Measured cost of the last iteration, used to decide whether the next
        # one fits. Checking only whether the budget has *already* been spent
        # would let a step that starts at 59 ms of a 60 ms budget run to 70 --
        # the bound has to predict the step it is about to allow, or it is not
        # a bound at all.
        step_cost = 0.0
        for step in range(settings.n_refine_steps):
            budget = settings.time_budget_s
            if budget is not None and (time.perf_counter() - started) + step_cost > budget:
                # Degrade by taking fewer steps rather than by missing the
                # policy deadline. The seeds are already usable plans.
                break
            step_started = time.perf_counter()

            denominator = max(1, settings.n_refine_steps - 1)
            temperature = settings.temperature_start + (
                settings.temperature_end - settings.temperature_start
            ) * (step / denominator)

            adam.zero_grad(set_to_none=True)
            refine = self._compiled_refine or self._refine_losses
            losses, signals = refine(params, inputs, temperature)

            # Score the iterates we just rolled out, before stepping away from
            # them -- the signals are already in hand, so this costs one
            # batched formula evaluation rather than another set of rollouts.
            #
            # Except on the first pass, where "before stepping" means the seeds
            # themselves, which are the sampling stage's business and not ours.
            if step > 0:
                with torch.no_grad():
                    candidates = signals.detach()
                    candidate_merit, candidate_rob = self._batched_scores(candidates)
                    candidate_ok = candidate_rob >= 0.0
                    pick = _lexicographic_pick(candidate_merit, candidate_rob)
                    if (bool(candidate_ok[pick]), float(candidate_merit[pick])) > (
                        best_ok,
                        best_merit,
                    ):
                        best_ok = bool(candidate_ok[pick])
                        best_merit = float(candidate_merit[pick])
                        best_params = params[pick].detach().clone()
                        best_signal = candidates[pick]

            # One backward for every seed: the losses are independent, so the
            # sum's gradient is each seed's own gradient in its own row.
            losses.sum().backward()
            adam.step()
            completed += 1
            # Slight over-estimate on purpose: better to skip a marginal step
            # than to blow the deadline on a momentarily busy machine.
            step_cost = max(step_cost, time.perf_counter() - step_started)

        # The final step moved the parameters somewhere never scored; the
        # whole loop would otherwise be one step of work thrown away.
        if completed:
            with torch.no_grad():
                final_signals = self._batched_rollout(params.detach(), inputs)
                final_merit, final_rob = self._batched_scores(final_signals)
                pick = _lexicographic_pick(final_merit, final_rob)
                if (bool(final_rob[pick] >= 0.0), float(final_merit[pick])) > (
                    best_ok,
                    best_merit,
                ):
                    best_ok = bool(final_rob[pick] >= 0.0)
                    best_merit = float(final_merit[pick])
                    best_params = params[pick].detach().clone()
                    best_signal = final_signals[pick]

        return best_params, best_signal, best_merit, best_ok, completed

    def _batched_rollout(self, params: torch.Tensor, inputs: RolloutInputs) -> torch.Tensor:
        if params.shape[0] == 1:
            return rollout(params[0], inputs, self.config).unsqueeze(0)
        return torch.vmap(lambda p: rollout(p, inputs, self.config))(params)

    def _batched_scores(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact objective and exact robustness for a batch, in one pass.

        Exact rather than smoothed -- the smoothing exists to make gradients
        informative, and has no business deciding anything.
        """
        scores = self._compiled_scores or self._batched_scores_eager
        return scores(signals)

    def _batched_scores_eager(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The body of :meth:`_batched_scores`, separated so it can be compiled."""
        if signals.shape[0] == 1:
            merit, robustness = self.rules.merit_and_robustness(signals[0])
            return merit.reshape(1), robustness.reshape(1)
        return torch.vmap(self.rules.merit_and_robustness)(signals)

    def _refine_losses(
        self,
        params: torch.Tensor,
        inputs: RolloutInputs,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One refinement step's forward pass over every seed at once.

        At a batch of one there is nothing to batch, and ``vmap`` is then pure
        dispatch overhead: measured, 15.06 ms through it against 10.64 ms
        calling the same arithmetic directly, and 3.41 against 1.93 for the
        exact scoring. Since the refinement runs a single seed by default, that
        overhead was most of what a tuning run spent its time on -- the policy
        is two thirds of a closed-loop run's wall clock, and the refinement is
        most of the policy.

        The alternative reading of the same measurement is that seeds are
        nearly free -- 12.41 ms per seed at one, 1.71 at eight, 0.47 at
        thirty-two, for about the same total. Both are worth having; this one
        is the change that does not alter what the search finds.
        """
        if params.shape[0] == 1:
            signal = rollout(params[0], inputs, self.config)
            loss = -self.rules.objective(signal, **smooth(temperature))
            return loss.reshape(1), signal.unsqueeze(0)
        signals = torch.vmap(lambda p: rollout(p, inputs, self.config))(params)
        losses = torch.vmap(lambda s: -self.rules.objective(s, **smooth(temperature)))(signals)
        return losses, signals


def _lexicographic_pick(merit: torch.Tensor, robustness: torch.Tensor) -> int:
    """Index of the plan to drive: satisfying first, then best objective.

    Two criteria, in order, because neither works alone.

    Ranking by robustness -- a minimum over rules -- is degenerate. A rule
    gated off by an obstruction contributes nothing to a minimum, and where an
    obstruction excuses the liveness rules, coming to a halt violates nothing
    at all and outscores everything on offer. Stopping becomes the answer to
    every hard question.

    Ranking by the objective alone fixes that but stops preferring plans that
    actually *satisfy* the specification: the weights will happily trade a
    comfort violation for progress. Measured, that drove 18% further with no
    emergency stops at all -- and halved the fraction of steps meeting the
    specification, with comfort violated on nearly twice as many.

    So: prefer a plan that satisfies the specification, and among those take
    the best objective. Where nothing satisfies it -- which is a third of city
    driving -- the objective decides, since there is no satisfying plan to
    prefer and the weights are the only guidance left.
    """
    satisfying = robustness >= 0.0
    if bool(satisfying.any()):
        masked = torch.where(satisfying, merit, torch.full_like(merit, -math.inf))
        return int(torch.argmax(masked))
    return int(torch.argmax(merit))


def _accel_params(accel_mps2: float, limits, accel0_mps2: float = 0.0) -> torch.Tensor:
    """Parameters that walk the acceleration profile to a requested value.

    The inverse of the mapping in :func:`~stl_driver.rollout._map_params`,
    which parameterises the profile as jerk-bounded *changes* from the measured
    acceleration rather than as absolute values. Reaching a distant target may
    take more than one knot, exactly as it would on the road.
    """
    from stl_driver.rollout import knot_step_limit

    step = knot_step_limit(limits)
    current = float(np.clip(accel0_mps2, -limits.max_decel_mps2, limits.max_accel_mps2))
    target = float(np.clip(accel_mps2, -limits.max_decel_mps2, limits.max_accel_mps2))

    logits = []
    for _ in range(4):
        delta = float(np.clip(target - current, -step, step))
        current += delta
        # atanh, kept off the asymptote where it is infinite.
        fraction = float(np.clip(delta / step, -0.999, 0.999))
        logits.append(0.5 * math.log((1.0 + fraction) / (1.0 - fraction)))
    return torch.tensor(logits + [0.0, 0.0], dtype=torch.float32)
