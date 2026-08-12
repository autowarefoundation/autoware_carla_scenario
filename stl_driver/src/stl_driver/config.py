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
    #: Distance from the rig origin (rear axle) to the two footprint discs
    #: that approximate the ego. NOT centred on the origin: the rig frame is
    #: anchored at the rear axle, so a body-centred pair would under-report
    #: forward clearance by roughly half a vehicle length.
    ego_disc_offsets_m: tuple[float, ...] = (1.0, 3.5)
    ego_disc_radius_m: float = 1.1
    #: Lateral half-extent within which an actor counts as in our lane.
    lead_capture_half_width_m: float = 1.6

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
