# SPDX-License-Identifier: Apache-2.0
"""Every tunable in one place.

Split into four frozen dataclasses so that the rule library, the rollout and
the optimizer can each be handed only what they need -- and so a test can vary
one limit without constructing a whole policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "Horizon",
    "Limits",
    "OptimizerConfig",
    "STLDriverConfig",
]


@dataclass(frozen=True)
class Horizon:
    """The planning grid.

    ``step_s`` matches the runtime's ``policy_timestep_s`` so that consecutive
    plans line up, and ``n_steps`` covers four seconds -- long enough that the
    controller's speed-dependent pure-pursuit lookahead (4-20 m) always falls
    inside the plan.
    """

    n_steps: int = 40
    step_s: float = 0.1

    @property
    def duration_s(self) -> float:
        return self.n_steps * self.step_s

    def index_at(self, seconds: float) -> int:
        """Timestep index for a time in seconds, clamped into the horizon.

        STL intervals are indices, not seconds.  Every rule that wants a
        wall-clock window goes through here rather than hardcoding a number,
        so that changing ``step_s`` cannot silently reinterpret a rule.
        """
        return max(0, min(self.n_steps - 1, int(round(seconds / self.step_s))))


@dataclass(frozen=True)
class Limits:
    """The physical and legal envelope the rules assert.

    These are the numbers that give the specification its meaning; a reader
    who wants to know what "comfortable" means should find the answer here.
    """

    # -- legality ----------------------------------------------------------
    #: Allowed overshoot of the posted limit before `speed_limit` is violated.
    speed_tolerance_mps: float = 0.5
    #: Fallback cap when the runtime reports no speed limit (0 means unknown).
    default_speed_limit_mps: float = 13.9
    #: Stop this far short of the reported traffic-light distance.
    stop_line_margin_m: float = 4.0

    # -- safety ------------------------------------------------------------
    #: Clearance kept from every predicted actor footprint.
    collision_margin_m: float = 0.6
    #: Time headway to the lead vehicle, on top of `standstill_gap_m`.
    headway_time_s: float = 1.4
    standstill_gap_m: float = 4.0
    #: Half-width of the corridor `lane_keeping` allows.
    lane_half_width_m: float = 1.2

    # -- comfort -----------------------------------------------------------
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 4.0
    max_jerk_mps3: float = 4.0
    max_lat_accel_mps2: float = 2.5
    #: Only the fallback plan may exceed `max_decel_mps2`.
    emergency_decel_mps2: float = 7.0

    # -- liveness ----------------------------------------------------------
    #: `makes_progress` asks for this fraction of what free-flow travel over
    #: the horizon would cover. Below 1 so that ordinary caution -- easing off
    #: for a curve, following a slightly slower vehicle -- does not read as a
    #: violation; it is a floor on effort, not a demand to speed.
    progress_fraction: float = 0.5
    #: Surplus distance beyond the progress target that stops earning reward.
    #: Finite on purpose -- an unbounded liveness reward would eventually
    #: outbid a hard safety rule, which is exactly the trade no specification
    #: should be willing to make.
    progress_margin_m: float = 5.0
    #: Below this the vehicle counts as stopped rather than crawling.
    creep_speed_mps: float = 1.0

    # -- geometry ----------------------------------------------------------
    #: Ego body dimensions, used to build its collision footprint.
    ego_length_m: float = 4.7
    ego_width_m: float = 2.0
    #: Distance from the rig origin to the centre of the body. The rig frame
    #: is anchored at the rear axle, not the vehicle centre, so a footprint
    #: centred on the origin would sit most of a car length too far back and
    #: over-report forward clearance.
    ego_rear_axle_to_centre_m: float = 1.4

    #: How much wider than the true body a footprint may be.
    #:
    #: Covering a box with discs always inflates it laterally, and the
    #: inflation is what decides whether a vehicle in the *next lane* reads as
    #: a collision. Two discs per car inflate a 2.0 m body to 3.0 m, which
    #: leaves 0.29 m to a vehicle one 3.5 m lane away -- so any slight lane
    #: offset registers as an imminent collision and the policy emergency
    #: brakes on an empty road. Capping the inflation instead, and using
    #: however many discs that takes, keeps the approximation honest where it
    #: matters. See `footprint_discs`.
    footprint_max_inflation_m: float = 0.15
    #: Lateral half-extent within which an actor counts as in our lane.
    lead_capture_half_width_m: float = 1.6

    # -- prediction --------------------------------------------------------
    #: Predict other vehicles along the road rather than in a straight line.
    #: See `_predict_centres` for what each model gets wrong.
    frenet_prediction: bool = True
    #: Time constant on an actor's lateral rate relative to our route. 0
    #: integrates it unchanged over the whole horizon.
    prediction_lateral_decay_s: float = 1.5
    #: How far off the route Frenet coordinates still describe a vehicle
    #: usefully. Beyond it -- or behind the route's start -- the straight
    #: line is the more honest model, and the two are mixed per actor.
    frenet_prediction_half_width_m: float = 8.0

    # -- numerics ----------------------------------------------------------
    #: Distance-like columns saturate here rather than running to infinity.
    #: An unbounded column would collapse the softmax weights inside `And` and
    #: dominate the weighted objective -- the sort of bug that shows up as
    #: "the policy ignores red lights whenever there is no vehicle ahead".
    saturation_m: float = 100.0
    #: Magnitude of the +-gate columns that drive the implications.
    #:
    #: `Implies(a, b)` is `max(-rho_a, rho_b)`, so this one number sets both
    #: the value of a vacuously-true rule (`+gate`) and the floor on an active
    #: violation (`-gate`). It must be large enough that realistic violations
    #: are reported faithfully -- a +-1 gate would clip anything worse than a
    #: 1 m overshoot and starve the optimizer of gradient -- and the per-rule
    #: `margin` in the objective is what stops the vacuous `+gate` from
    #: dominating. 20 m comfortably brackets both.
    gate_magnitude_m: float = 20.0

    def __post_init__(self) -> None:
        if self.gate_magnitude_m <= self.stop_line_margin_m:
            raise ValueError(
                "gate_magnitude_m must exceed stop_line_margin_m, or ordinary "
                "stop-line overshoots would be clipped"
            )


@dataclass(frozen=True)
class OptimizerConfig:
    """Search budget and shape."""

    #: Quasi-random seeds drawn once at construction and reused every step.
    n_samples: int = 64
    #: Adam steps applied to the single best seed.
    n_refine_steps: int = 8
    learning_rate: float = 0.3
    #: Inverse temperature schedule for the smoothed objective: start soft so
    #: every timestep contributes gradient, finish sharp so the last steps
    #: optimize something close to the true min.
    temperature_start: float = 2.0
    temperature_end: float = 8.0
    #: Wall-clock bound on the whole search, enforced by predicting whether the
    #: next refinement step fits rather than by noticing afterwards that it did
    #: not. The policy step is 100 ms; this leaves room for the plan-to-proto
    #: conversion, the gRPC round trip, and a machine slower than the one it
    #: was tuned on. Overrunning degrades into fewer refinement steps.
    time_budget_s: float = 0.070
    #: Fixed for reproducibility -- the runtime's rollouts are seeded, and a
    #: policy that re-randomised would make them irreproducible.
    seed: int = 0
    #: How far into the horizon a hard-rule violation must fall to count as
    #: imminent enough for the emergency plan. Beyond this the optimizer is
    #: already slowing for it and the next replan -- a tenth of a second
    #: later, with a fresh forecast -- gets another say.
    fallback_window_s: float = 1.2
    #: How much better a different plan must be, in robustness, before it is
    #: preferred over continuing last step's.
    #:
    #: Nothing in the specification distinguishes two plans that satisfy it
    #: equally well, so with no tie-breaker the search picks a different one
    #: every step -- measured: the parameter vector jumped clear across its
    #: own box on 48% of steps. The plans are individually fine, yet the
    #: commanded speed jitters by a third of a metre per second every
    #: 100 ms, which the tracking controller faithfully turns into throttle
    #: and brake. From inside the car that is the difference between driving
    #: and being driven.
    #:
    #: Continuity is a property of a *sequence* of plans, so it cannot be
    #: written as a rule about one plan; it belongs here, in how the search
    #: chooses, rather than in what the specification demands.
    continuity_margin: float = 0.3
    #: Candidates echoed back as `DriveResult.sampled_trajectories_in_rig`.
    n_reported_candidates: int = 4
    #: Threads torch may use. The tensors here are tiny; more threads costs
    #: more in synchronisation than it saves in arithmetic.
    torch_threads: int = 2


@dataclass(frozen=True)
class STLDriverConfig:
    """Everything :class:`~stl_driver.policy.EgoPolicy` needs."""

    #: Speed the policy aims for when nothing else binds. Capped by the posted
    #: limit when the runtime reports one.
    cruise_speed_mps: float = 8.0
    horizon: Horizon = field(default_factory=Horizon)
    limits: Limits = field(default_factory=Limits)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    #: Spacing of the resampled route table. Uniform on purpose: it turns the
    #: differentiable station lookup into index arithmetic.
    route_table_ds_m: float = 1.0
    #: Extra table length beyond what the fastest plan could reach, so the
    #: lookup never clamps and never loses its gradient at the end of a route.
    route_table_margin_m: float = 20.0

    def ego_footprint(self) -> tuple[list[float], float]:
        """The ego's collision discs, as offsets from the rig origin.

        `footprint_discs` measures offsets from the body centre; the rig frame
        is anchored at the rear axle, so they are shifted forward by
        `ego_rear_axle_to_centre_m`. Skipping that shift puts the whole
        footprint most of a car length too far back, which reads as generous
        clearance to whatever is directly ahead.
        """
        from stl_driver.perception import footprint_discs

        limits = self.limits
        offsets, radius = footprint_discs(
            limits.ego_length_m, limits.ego_width_m, limits.footprint_max_inflation_m
        )
        return [float(o) + limits.ego_rear_axle_to_centre_m for o in offsets], radius

    def target_speed_mps(self, speed_limit_mps: float) -> float:
        """The speed to aim for, given what the runtime reports as the limit."""
        limit = (
            float(speed_limit_mps) if speed_limit_mps > 0.0 else self.limits.default_speed_limit_mps
        )
        return min(self.cruise_speed_mps, limit)

    def route_table_length_m(self) -> float:
        """How far the resampled route must reach to cover any plan."""
        top_speed = max(
            self.cruise_speed_mps,
            self.limits.default_speed_limit_mps,
        )
        return top_speed * self.horizon.duration_s + self.route_table_margin_m

    def progress_target_m(self, target_speed_mps: float, current_speed_mps: float) -> float:
        """Distance `makes_progress` asks the plan to cover.

        Scaled by the speed we are *starting* from, not just the one we are
        aiming for.  A target of ``fraction * v_target * duration`` is simply
        unreachable from a standstill -- the acceleration limit caps a
        four-second pull-away well below it -- so the rule would be violated
        by every plan and the specification would be asking for the
        impossible.  Averaging the two speeds is the distance a smooth
        acceleration from one to the other would cover.
        """
        average = 0.5 * (max(0.0, current_speed_mps) + max(0.0, target_speed_mps))
        return self.limits.progress_fraction * average * self.horizon.duration_s

    def __post_init__(self) -> None:
        if self.cruise_speed_mps <= 0.0:
            raise ValueError(f"cruise_speed_mps must be positive, got {self.cruise_speed_mps}")
        if not math.isfinite(self.route_table_ds_m) or self.route_table_ds_m <= 0.0:
            raise ValueError(f"route_table_ds_m must be positive, got {self.route_table_ds_m}")
