# The specification

This is what the policy aims at.  It is the same set of rules that
`src/stl_driver/rules.py` builds and that `stl-driver explain` prints -- this
document explains *why* each one is phrased the way it is.

`src/stl_driver/rules.py` is nothing but the specification, so it can be
edited or replaced outright; `ruleset.py` holds what a rule *is* and how a set
of them is scored, and knows nothing about driving.  `EgoPolicy(rules=...)`
takes a rule set directly, so a different specification needs no edits here at
all.

## It aims; it does not guarantee

Nothing in this design proves the rules will hold.  Worth being plain about,
because "specification" invites the opposite assumption:

* The search **maximises** robustness over 64 sampled plans plus a few
  gradient steps, inside a 70 ms budget.  It is not a constraint solver and
  issues no feasibility certificate -- even where a satisfying plan exists, it
  may not be found.
* Some situations admit no satisfying plan at all.  Another vehicle driving at
  a stationary ego violates `collision_free` whatever we do.
* The rules are evaluated against a *forecast* of the other traffic.  Being
  satisfied means the plan is safe against what we predicted, which is not the
  same as being safe.

What the design does give is that the shortfall is **stated rather than
hidden**.  `spec_robustness` is reported every step and is negative exactly
when some rule fails; the per-rule breakdown names which.  Measured over 500
steps of Town10HD with 30 vehicles: every rule held on 65.8% of steps, and the
violations were mostly `safe_headway`, `comfort` and the two liveness rules.
`collision_free` went negative on 11% of steps against the forecast -- and the
rollout ended with zero actual collisions.

The one place a violation changes what is driven is a `hard` rule breached
within the next 1.2 s, when braking could still help; then the emergency plan
replaces the optimizer's.  That fired on 0.6% of steps.  Everywhere else the
best plan found is driven, violations and all.

Every rule is an STL formula over a signal of twelve columns, sampled at
0.1 s over a 4 s horizon.  Each column is a **slack**: a quantity that should
stay positive.  Speed limits, stop-line positions and route lengths change
every step, so they are folded into the column rather than written into the
formula, which leaves every rule comparing against the constant `0` and lets
the formula tree be built once instead of at 10 Hz.

Read `Always(clearance > 0)` as "there is always clearance to spare", not as
"the gap is always a positive number".

## Reading a rule

```python
Rule("collision_free", Always(clearance > 0.0), weight=10.0, margin=2.0, hard=True)
```

| field | meaning |
|---|---|
| `weight` | its pull in the objective the optimizer ascends |
| `margin` | how much slack is *enough*; the objective saturates around it |
| `hard` | whether a violation abandons the plan for an emergency stop |

`margin` is what lets the search trade off.  A safety rule satisfied with room
to spare stops competing with the liveness rules, so the vehicle does not buy
clearance it does not need at the cost of ever getting anywhere.

`hard` decides one more thing: a violation summons the emergency plan, but
only if that plan could actually help.  Braking harder always helps when we
are *moving* -- even for an obstacle too close to avoid.  It helps not at all
when we are stopped and something is moving towards us, because our speed was
never what caused the conflict.  Standing on the brakes there buys no safety
and costs everything else: no progress, and a vehicle stopped dead in traffic
that did not expect it, which is how a stall turns into being rear-ended.  The
violation is still reported either way; what the test decides is whether the
emergency plan is a *remedy* for it.

`hard` also changes how a violation is scored.  Soft rules saturate in *both*
directions -- they express preferences, so their influence is bounded either
way.  Hard rules saturate only above: surplus safety levels off, but the
penalty for violating one grows without limit.  That asymmetry is the safety
argument, and it is load-bearing: with an unbounded progress reward, a plan
that clips an obstacle at good speed outscores one that stops safely.

## Legality

### `speed_limit` — soft, weight 2, margin 1.0
```
□ (speed_headroom > -0.5)
```
Never exceed the posted limit by more than 0.5 m/s.  `speed_headroom` is
`limit - speed`; the limit comes from CARLA when available and falls back to
13.9 m/s (50 km/h) when it does not.

### `no_reverse` — hard, weight 4, margin 0.5
```
□ (speed > -0.05)
```
Currently guaranteed by the model — the rollout floors the speed at zero — so
it reads as a tautology today.  It is kept because it states the requirement
rather than leaving it as an unwritten property of the kinematics.  If the
model ever gains a reverse gear, or the floor is relaxed to recover gradient,
this is the line that notices.

### `stop_on_red` — hard, weight 6, margin 2.0
```
□ ((light_gate > 0) ⇒ (stop_slack > 0))
```
While a red or amber light governs us, never cross its stop line.

`light_gate` is a signed constant of ±20 m rather than a boolean, and the
magnitude is chosen, not arbitrary.  `Implies(a, b)` evaluates to
`max(-ρ_a, ρ_b)`, so one number controls two things: an inactive light makes
the rule vacuously true at exactly `+gate`, and an active one cannot read
below `-gate`.  Too small a gate -- ±1, say -- would clip any overshoot worse
than a metre and starve the optimizer of the gradient it needs to fix it.
Twenty metres brackets both concerns, and the objective's `margin` is what
keeps the vacuous `+gate` from dominating.

## Safety

### `collision_free` — hard, weight 10, margin 2.0
```
□ (clearance > 0)
```
Keep clearance from every predicted actor at all times.  Every vehicle,
including the ego, is covered by discs whose count comes from a stated bound
on how much wider they make it -- 0.15 m -- rather than being fixed at two.
Two discs turn a 2.0 m car into a 3.0 m one, which leaves almost nothing
between us and a vehicle one lane over, so an ordinary lane offset reads as an
imminent collision.

The ego's discs are placed relative to the *rear axle*, since that is where
the rig frame is anchored, not the vehicle centre.

Other vehicles are predicted **along the road**, not in a straight line:
their velocity is resolved into the route's own frame and advanced there, so a
car we are following stays in its lane however the road bends.  A straight
line leaves the lane tangentially within a second or two, which is exactly
when it matters.  Frenet coordinates only describe the space the route
parameterises, though, so a vehicle behind the route's start -- and the route
begins at the ego -- would come back at station zero, on our front bumper.
The two models are therefore chosen per vehicle.

The clearance is an exact minimum over disc pairs, not a soft one.  A soft
minimum would spread gradient more widely, but its error scales with
`log(N)/β` where `N` is the traffic density: unnormalised it manufactures
phantom collisions in traffic, normalised it silently spends 0.46 m of the
0.6 m collision margin at twenty vehicles.  Neither is acceptable for the one
rule whose job is to keep the vehicle out of other vehicles.

### `safe_headway` — soft, weight 5, margin 3.0
```
□ (headway_slack > 0)
```
Stay at least `1.4 s × speed + 4 m` behind the vehicle ahead.  The lead is
chosen in route coordinates rather than by heading, so a vehicle on a parallel
road does not become our lead just because it points the same way.

### `stay_on_route` — hard, weight 4, margin 5.0
```
□ (route_slack > 0)
```
Never plan past the end of the published route.  The runtime publishes 80 m of
route ahead, and four seconds at 20 m/s is exactly 80 m, so this binds on
ordinary steps rather than only at the true end of a journey.

## Lane keeping

### `lane_keeping` — soft, weight 3, margin 1.02
```
□[0.5s, 4.0s] ((lateral_offset < 1.2) ∧ (lateral_offset > -1.2))
```
Stay within 1.2 m of the route centreline.

Two details are deliberate.  The window starts half a second in, not at zero:
the plan begins at the *measured* lateral offset, so over the full horizon the
worst value is whatever the vehicle inherited and no choice of parameters can
change it.  An `Always` from `t = 0` is not merely uninformative -- its
gradient to the lateral parameters is identically zero, and a car nudged out
of lane would have no reason to come back.

And the margin is nearly the full half-width, which is what makes this rule
*centre* the vehicle rather than merely keep it legal.  A small margin would
saturate as soon as the plan was inside the corridor.

## Comfort

### `comfort` — soft, weight 1, margin 0.5
```
□ (|accel| within [-4, 2] ∧ |jerk| < 4 ∧ |lateral_accel| < 2.5)
```
All in SI units.  Only the emergency fallback exceeds these.

Lateral acceleration is computed as `(ẋÿ - ẏẍ) / max(‖ṗ‖, 0.1)` rather than
the textbook `v²κ`, whose `(ẋ²+ẏ²)^{3/2}` denominator divides by an epsilon at
low speed and reports a swerve that is not there.

## Liveness

Standing still satisfies every rule above.  These two are what stop the
optimizer from discovering that and parking forever.  Stating them as logic
rather than as a progress bonus in the cost keeps the whole specification in
one language -- and means a stopped vehicle reports *which* requirement it is
failing.

### `makes_progress` — soft, weight 2, margin 5.0
```
□ ((no red light ∧ nothing close ahead) ⇒ ◇[0, 4.0s] (progress_slack > 0))
```
Cover a reasonable distance over the horizon: half of what travelling at the
average of the current and target speeds would cover.

Scaling by the *current* speed matters.  A target of "half the cruise speed
for four seconds" is simply unreachable from a standstill -- the acceleration
limit caps a four-second pull-away well below it -- so the rule would be
violated by every plan and the specification would be asking for the
impossible.

The gate is the same one `keep_moving` uses, so that waiting at a red light is
not reported as a failure to make progress.  Without it, every correct stop
would drive the reported robustness negative and it would stop being a useful
health signal.

### `keep_moving` — soft, weight 2, margin 1.0
```
□[1.0s, 4.0s] ((no red light ∧ headway > 4 m) ⇒ (speed > 1.0))
```
Redundant with `makes_progress` by design: that one is an integral over the
horizon, this one is instantaneous, and they fail in different ways.  A plan
that creeps the whole time fails progress; one that lurches and stops fails
this.

The window opens at one second so that a plan is allowed to pull away from a
standstill rather than being required to already be moving.

## Two evaluations, deliberately different

`RuleSet.spec_robustness` is the plain conjunction of every rule under exact
min/max semantics.  That is the specification: what gets reported, and what
decides whether a plan is acceptable.  It equals the minimum over the
individual rules, which is what makes the per-rule breakdown in
`debug_scalars` a genuine decomposition rather than a parallel calculation.

`RuleSet.objective` is the weighted, saturated sum described above, evaluated
with a smooth approximation so it has useful gradients.  That is what the
optimizer ascends.

They are not the same function on purpose.  A min of mins is flat in every
rule except the currently-worst one, so a gradient step improves exactly one
rule at a time and the others are invisible.  Keeping the scoring function
separate from the specification is what makes the search tractable without
weakening what "satisfied" means.

## Intervals are indices

STLCG++ measures temporal intervals in timesteps, not seconds.  Every rule
that wants a wall-clock window goes through `Horizon.index_at`, so changing
the step size cannot silently reinterpret a rule.  A bounded interval must
also fit inside the horizon: upstream pads a robustness trace beyond its
bounds with `-1e9`, which would read as a catastrophic violation.
`build_rules` asserts this at construction.
