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
    # -- following distance, after RSS ------------------------------------
    #: The longitudinal safe distance follows the Responsibility-Sensitive
    #: Safety model (Shalev-Shwartz, Shammah & Shashua 2017), which asks:
    #: if the vehicle ahead braked as hard as it can, right now, could we
    #: still stop behind it after a response delay?
    #:
    #:   d = v_r*rho + a_accel*rho^2/2 + (v_r + rho*a_accel)^2 / 2*b_min
    #:                                            - v_f^2 / 2*b_max
    #:
    #: The last term is the one a fixed time headway leaves out, and it is the
    #: whole difference: it credits the *lead* for the distance its own
    #: braking takes. Following a car at the same speed, `1.4 s + 4 m` demands
    #: 15.2 m at 8 m/s where RSS asks 10.4 m, and a lead pulling away is given
    #: a gap it does not need at all. Measured before the change: 2.6x the
    #: required gap kept while following, and `safe_headway` reported violated
    #: with the lead 29 m away.
    #:
    #: Response delay before our braking takes effect.
    rss_response_s: float = 0.5
    #: What we might still be doing during that delay.
    rss_response_accel_mps2: float = 2.0
    #: The braking we guarantee to apply afterwards.
    rss_brake_mps2: float = 4.0
    #: The hardest the vehicle ahead might brake. Larger is more conservative.
    rss_lead_brake_mps2: float = 8.0
    #: Never ride closer than this, whatever the arithmetic says.
    standstill_gap_m: float = 2.5
    #: Half-width of the corridor `lane_keeping` allows.
    lane_half_width_m: float = 1.2

    # -- comfort -----------------------------------------------------------
    #: What `comfort` asserts. These bounds belong to the *specification*: they
    #: say when a ride stops being comfortable. What a *plan* is allowed to
    #: command is a little tighter -- see `plan_bound_inset` and the two
    #: accessors below, which is what everything building or anchoring a plan
    #: must ask, rather than reaching for these.
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 4.0
    max_jerk_mps3: float = 4.0
    max_lat_accel_mps2: float = 2.5
    #: Only the fallback plan may exceed `max_decel_mps2`.
    emergency_decel_mps2: float = 7.0
    #: How far inside the comfort bounds a plan keeps itself.
    #:
    #: Landing exactly on a bound leaves the rule reading exactly zero, where
    #: float error alone decides whether it counts as satisfied, and every
    #: accelerating plan pinned there.
    #:
    #: The gap is small and the arithmetic around it is not, which is why the
    #: envelope is derived here rather than at each use. The plan is anchored
    #: at the measured acceleration and the opening jerk is the difference
    #: between that and the first knot over one 0.1 s tick -- so two callers
    #: insetting differently put ten times their disagreement into the jerk.
    #: Measured, when the ego's acceleration was confined to the comfort bounds
    #: and the first knot to these: 0.8 m/s^3 of opening jerk before the plan
    #: asked for anything, a fifth of the budget, which made `comfort`
    #: unsatisfiable by every candidate on the steps that followed a hard stop.
    #: The vehicle's only relief was to pull away more gently than it meant to.
    plan_bound_inset: float = 0.98

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
    #: Clearance below which something counts as obstructing us, for the
    #: purpose of excusing a lack of progress.
    #:
    #: The liveness rules were gated on the traffic light and the *lead
    #: vehicle* only. Measured over 500 steps, that missed most of what
    #: actually stops a car: of 165 steps where `makes_progress` was reported
    #: violated, 90 had no lead vehicle at all but something else within two
    #: metres -- crossing traffic, a vehicle alongside at a junction -- and 60
    #: were clear at that instant but blocked later in the same horizon. Only
    #: 15 were genuine.
    liveness_clearance_m: float = 2.0

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
    #: Lateral half-extent within which an actor's *centre* counts as being
    #: in our lane, and so as something to follow.
    #:
    #: Applied to the centre rather than to the near edge. Allowing an edge
    #: to qualify widens the window to +-2.6 m, which reaches into the
    #: neighbouring lane -- and a car passing in the next lane then becomes
    #: the vehicle we are following, a few metres ahead.
    lead_capture_half_width_m: float = 1.7

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
    #: How near a predicted disc must come before the rollout is told about it.
    #:
    #: Measured against the disc's *closest approach over the horizon*, which
    #: already includes the other vehicle's predicted motion, so this radius
    #: has only to cover how far the ego itself can get: at
    #: `default_speed_limit_mps` over a four-second horizon, 55.6 m, plus the
    #: front of the car (3.75 m from the rear axle) and `collision_margin_m`.
    #: Round that up and nothing the ego could touch is outside it.
    #:
    #: Worth having because the simulator reports every vehicle on the map and
    #: almost none of them are relevant: 200 steps in traffic of thirty gave 90
    #: discs per step, of which 19 came within 60 m and the 32nd-nearest never
    #: came within 61.3 m on any step. Gating there cuts the search from 59.9
    #: to 48.8 ms and the forecast's own cost with it. See
    #: `_drop_traffic_out_of_reach`.
    actor_reach_m: float = 60.0

    # -- numerics ----------------------------------------------------------
    #: How many actor footprint discs the rollout is given, always.
    #:
    #: The forecast produces as many as there are vehicles within
    #: `actor_reach_m`, which on a 400-step drive through thirty cars varied
    #: continuously and peaked at 29 -- and a width that varies is a graph
    #: shape that varies, recompiled mid-drive. Padding to a fixed width
    #: leaves one shape; the rest of this is headroom over that peak.
    #:
    #: Not free, and it was thought to be: the argument was that the time goes
    #: on dispatching operations rather than performing them, so a wider tensor
    #: costs the same. That held while the search ran eagerly and stopped
    #: holding when it was compiled, which is what dispatch overhead is for.
    #: Measured compiled, at 128 discs the search takes 48.8 ms against 41.8 at
    #: 48 -- so the width is worth keeping near what the road actually
    #: produces. Discs beyond it are dropped farthest-first, which is the right
    #: way round: `collision_free` is a minimum, so only the nearest can decide
    #: it.
    max_actor_discs: int = 48
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

    def plan_accel_envelope(self) -> tuple[float, float]:
        """Accelerations a plan may command, as ``(lowest, highest)``.

        The single answer to "how hard may we ask the vehicle to speed up or
        slow down". Anything generating a plan, anchoring one at the vehicle's
        measured state, or inverting the parameterisation asks here; only the
        specification in `rules.py` reads the comfort bounds directly, because
        only it is stating the requirement rather than staying inside it.
        """
        return (
            -self.max_decel_mps2 * self.plan_bound_inset,
            self.max_accel_mps2 * self.plan_bound_inset,
        )

    def plan_jerk_limit(self) -> float:
        """The largest rate of change of acceleration a plan may command."""
        return self.max_jerk_mps3 * self.plan_bound_inset

    def __post_init__(self) -> None:
        if self.gate_magnitude_m <= self.stop_line_margin_m:
            raise ValueError(
                "gate_magnitude_m must exceed stop_line_margin_m, or ordinary "
                "stop-line overshoots would be clipped"
            )
        if not 0.0 < self.plan_bound_inset <= 1.0:
            raise ValueError("plan_bound_inset must be a fraction of the comfort bounds")


@dataclass(frozen=True)
class OptimizerConfig:
    """Search budget and shape."""

    #: Quasi-random seeds drawn once at construction and reused every step.
    n_samples: int = 64
    #: Adam steps applied to each seed, all of them taken unless
    #: `time_budget_s` is set.
    #:
    #: Chosen to be what the search used to *reach* under a 70 ms wall-clock
    #: bound -- measured at 3.9 to 4.5 steps -- so that making the count fixed
    #: changes when the work happens rather than how much of it there is. The
    #: ceiling was 8 before, and raising this to it would roughly double the
    #: policy's latency past the 100 ms step it has to fit inside.
    n_refine_steps: int = 4
    #: How many starting points the gradient stage descends from at once.
    #:
    #: Descending from one finds the nearest local optimum, and which one that
    #: is depends on wherever the sampling stage happened to leave off. The
    #: seeds are independent, so a batched pass covers them all -- and the cost
    #: of that pass is per call, not per seed, because the tensors involved are
    #: tiny and the time goes on dispatching operations rather than on
    #: performing them. Measured: 12.41 ms per seed at one, 1.71 at eight, 0.47
    #: at thirty-two, for about the same total each time.
    #:
    #: So breadth is very nearly free, and at a fixed 70 ms budget it buys a
    #: great deal: one seed completes 3.42 refinement steps, eight complete
    #: 2.80 each -- 22.4 gradient steps in total against 3.4, with the median
    #: solve unchanged at 70 ms.
    #:
    #: This was 1 for a long time, and the reason it could not be raised is
    #: worth keeping. `_refine` used to return the highest *robustness* among
    #: its seeds, and robustness -- a minimum over rules -- is maximised by
    #: stopping whenever the safety rules bind. With one seed a braking plan
    #: had to beat a balanced starting point to be returned at all; with
    #: sixteen there was nearly always one that won. It became the warm start,
    #: the continuity margin preferred it next step, and the vehicle braked
    #: itself to a halt it never recovered from: 19.3 m travelled against
    #: 135.8 m, stopped for 87% of steps.
    #:
    #: Selection is lexicographic now -- satisfying first, then the objective,
    #: which unlike robustness does not reward standing still -- so that trap
    #: is gone. Confirmed in closed loop over three runs each on a pinned
    #: route: satisfied 95.2/95.5/95.8% at one seed against 96.2/97.8/98.2% at
    #: eight, with the emergency plan driven 7/7/7 times against 2/4/4. No
    #: collisions either way.
    n_refine_seeds: int = 8
    learning_rate: float = 0.3
    #: Inverse temperature schedule for the smoothed objective: start soft so
    #: every timestep contributes gradient, finish sharp so the last steps
    #: optimize something close to the true min.
    temperature_start: float = 2.0
    temperature_end: float = 8.0
    #: Wall-clock bound on the whole search, or ``None`` to always take every
    #: refinement step above.
    #:
    #: ``None`` by default, which is a deliberate choice about what the policy
    #: *is*: a fixed amount of work, so that the same situation produces the
    #: same plan. Under a wall-clock bound it does not -- how many steps the
    #: search completes depends on how busy the machine is, and once the ego
    #: acts differently at all the whole scenario diverges from there. Two runs
    #: of the same build on the same pinned route came back 96.0%, 91.0% and
    #: 93.0% satisfied over 340, 238 and 236 metres, which is a wider spread
    #: than most changes worth making, and made the closed loop nearly useless
    #: as something to measure against.
    #:
    #: Set it for a deployment that must hold a deadline more than it must be
    #: reproducible. It is enforced by predicting whether the next step fits
    #: rather than by noticing afterwards that it did not -- a bound that only
    #: checks whether it has already been exceeded lets a step starting at
    #: 59 ms of a 60 ms budget run to 70 -- and overrunning degrades into fewer
    #: refinement steps rather than into a missed deadline.
    time_budget_s: float | None = None
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
    #: Threads torch may use.
    #:
    #: This was 2, on the grounds that the tensors here are tiny and more
    #: threads would cost more in synchronisation than they saved in
    #: arithmetic. That stopped being true when the actor discs were padded to
    #: a fixed width and the search compiled: 67 candidates over 40 timesteps
    #: against 48 discs is no longer tiny. Measured, one configuration per
    #: process so that nothing recompiles for the wrong reason: 91.2 ms at one
    #: thread, 71.9 at two, 59.9 at four, 61.5 at eight.
    torch_threads: int = 4
    #: Compile the two hot paths with ``torch.compile`` before driving.
    #:
    #: On, and measured on the road rather than on a bench -- for two drives
    #: this was the wrong answer, and what made it right is worth keeping,
    #: because both faults were the same fault and neither is visible on a
    #: bench.
    #:
    #: `torch.compile` specialises on what it sees, so anything that varies
    #: step to step is a new graph compiled mid-drive, inside the policy's own
    #: deadline. It specialised first on the *values* of the Python floats in
    #: `RolloutInputs` -- the ego station, the route length, the speed -- and a
    #: 400-step drive went from 57.9 s to 200.5 s against dynamo's
    #: recompilation limit. Those are 0-d tensors now. It then specialised on
    #: two *shapes*: the actor footprint tensor, as wide as there are vehicles
    #: nearby, and the candidate batch, one row wider once a previous plan
    #: exists to warm start from. Both are fixed widths now -- see
    #: `Limits.max_actor_discs` and `RobustnessOptimizer._candidates`.
    #:
    #: A bench sees none of this: traffic density is what varies, and a bench
    #: pins it. Two runs each on a pinned route through thirty vehicles, per
    #: step: median solve 0.0749/0.0751 s eager against 0.0636/0.0658 s
    #: compiled, p95 0.0863/0.0865 against 0.0737/0.0750, worst step
    #: 0.1015/0.1008 against 0.0897/0.0827. Against a 0.100 s timestep that
    #: last column is the one that decides: compiled fits inside it on every
    #: step of both runs, and eager does not on either. Specification
    #: satisfaction was 85.5/86.2% against 87.2/83.5% -- a spread the closed
    #: loop cannot resolve, so latency is what this is bought for.
    #:
    #: The compile itself costs a few seconds, paid in `prepare` before the
    #: runtime is waiting. Failure there is caught and falls back to eager.
    compile_search: bool = True
    #: Where the search runs: ``"cpu"``, ``"cuda"``, or ``"auto"``.
    #:
    #: CPU is the default because it is faster at this size, not merely
    #: smaller: 64 candidates over a 40-step horizon is small enough that
    #: kernel launch latency dominates the arithmetic. It stops being the
    #: right answer if the search grows -- a few thousand candidates puts the
    #: batched STL masks into the tens of millions of elements -- so the
    #: choice is exposed rather than assumed. Install the matching torch with
    #: `uv sync --extra cuda`.
    #:
    #: Worth knowing before reaching for it: measured on this specification,
    #: 64x the samples and 25x the refinement steps with no time limit at all
    #: changed the fraction of steps satisfying the specification by nothing
    #: whatever, and rescued exactly zero violations. The violations are
    #: situations with no satisfying plan, not searches that ran out of time.
    device: str = "cpu"


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

    def resolved_device(self) -> str:
        """The device to run on, with ``"auto"`` decided here.

        Kept as a method rather than resolved at construction so that a config
        can be built, pickled and compared without asking about hardware.
        """
        import torch

        requested = self.optimizer.device
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' but torch reports no CUDA device; install the "
                "CUDA build with `uv sync --extra cuda`, or use device='auto'"
            )
        return requested

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
