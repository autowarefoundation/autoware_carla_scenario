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
Rule("collision_free", Always(clearance > 0.0), weight=10.0, margin=0.5, hard=True)
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

### `stop_on_red` — hard, weight 6, margin 0.5
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

The margin is small for the same reason `collision_free`'s is.  At 2.0 the
rule paid the vehicle to stay away from the line: decomposed at a settled stop
7.2 m short of a line it had every right to reach, closing three of those
metres would have cost 7.9 of `stop_on_red` to gain 5.0 of `makes_progress`.
At 0.5 the same three metres cost 0.5.  The rule is hard, so below zero the
penalty stays linear and unbounded; only the reward for surplus is capped.

`stop_slack` is the room left before the *stop station*, less what stopping
from the plan's speed takes (`v²/2b + v·b/2j`, `Limits.stopping_distance_m`).
The stop station is the reported distance to the line less
`stop_line_margin_m`, 4 m.  Both are measured at the rig origin, the rear
axle, so the margin is what keeps the front bumper (3.75 m ahead of it) short
of the line.  Amber is treated as red.

The margin gives way only to what can no longer be stopped short of: it is
never more than the distance less `Limits.stop_room_m`, the most room this
rule asks for while the hardest stop a plan can make is under way.  That is
more than the stopping distance from a cruise -- the rule charges the ramp to
full braking at every instant, so a second into it, the ground covered plus
what is still asked for peaks 1.2 m above it at 4 m/s and 3.1 m above at
8 m/s -- and equal to it once braking at the bound.  So a light first seen,
or turning red, too close for the full margin still leaves a stop some plan
satisfies; only where the line itself is out of reach is the rule violated,
and the emergency fallback then is right.  A stopped vehicle keeps the whole
margin, and one already inside it is held where it stands.

A margin that gave way stays given way while the same line governs us.  Read
afresh each step it grows back once the brakes are on -- the room to stop
shrinks faster than the distance -- and every step it put the stop station at
the very edge of what could still be reached, so the first plan short of the
hardest allowed stop violated the rule: closed loop, a light turning red with
2 m of the margin left handed a step to the fallback that way.

This replaced `min(margin, 0.5 · distance)` of the live distance, which within
two margins of the line put the stop station halfway to it, so it slid
forward as the vehicle crept: a red light 30 m ahead stopped the rear axle at
28.1 m rather than 26 m.  Now it stops at 25.3 m -- short of 26 by less than
`liveness_stop_gap_m`, the room `makes_progress` lets a stopped vehicle leave
(#1147).

## Safety

### `collision_free` — hard, weight 10, margin 0.5
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
parameterises, though, so a vehicle behind the route table's start -- which
is at the ego, reaching back over it when the route is published from ahead of
it -- would come back at station zero, on our front bumper.
The two models are therefore chosen per vehicle.

The margin is half a metre, which saturates by about a metre and a half of
clearance -- as much air as any plan needs to be judged safe.  At 2.0 the
objective still paid about 9.4 per metre of clearance at the half-metre a car
in the next lane leaves, against 2.0 per metre of progress, so backing off
outscored getting on whenever traffic came past: on a 400-step drive, 44 steps
braked below -1 m/s² with every rule satisfied by simply holding speed.

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
Stay at least the RSS safe longitudinal distance behind the vehicle ahead
(Shalev-Shwartz et al., 2017): far enough that, if it braked as hard as it can
right now, we could still stop behind it after a response delay,

```
d = v·ρ + a·ρ²/2 + (v + ρ·a)² / 2·b_min − v_lead² / 2·b_max ,   d ≥ standstill_gap_m
```

with `ρ = rss_response_s` (0.5 s), `a = rss_response_accel_mps2` (2 m/s²),
`b_min = rss_brake_mps2` (4 m/s²), `b_max = rss_lead_brake_mps2` (8 m/s²) and
`standstill_gap_m` 2.5 m.  The last term credits the lead for its own braking,
so at 8 m/s a lead at the same speed asks for 10.4 m and a stopped one 14.4 m.
`headway_slack` is the bumper-to-bumper gap less `d`: from the ego's front
bumper (`ego_rear_axle_to_centre_m + ego_length_m / 2`, 3.75 m ahead of the
rear axle the rig frame is anchored at) to the lead's rear edge.

The rear edge is the lead's *box*, from its own length and width: the nearest
station any of its corners projects to, so a vehicle at an angle reaches back
by its corner.  The discs that cover it are `collision_free`'s and are not
used for this.  Their radius is mostly half the vehicle's width and their
count is shared with the largest vehicle in the scene, so the rear disc's
centre less its radius put a car's rear 0.6–0.8 m behind its bumper, and the
ego queued that much further back than `standstill_gap_m` (#1151).

The lead is chosen in route coordinates rather than by heading, so a vehicle on
a parallel road does not become our lead just because it points the same way;
one of an actor's disc centres has to be within `lead_capture_half_width_m`
of the route.
It is chosen once, at t = 0, and followed over the horizon, so a car
overtaking from behind is never adopted half way through it.  "Ahead" starts
at the middle of the ego's body rather than at its bumper, and it is the
front of the other vehicle's box -- the same box its rear is taken from --
that has to be past it: a vehicle cutting in level with our front half is
followed, its gap reads negative and the policy drops back to let it in.

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

Lateral acceleration is the curvature the plan follows times the speed
squared, read in the road's own frame rather than by differencing the planned
positions twice, which aliased the route polyline's vertices into the answer:

```
a_lat = v² · (κ / (1 − κ·d) + d″(s))
```

`κ` is the route's curvature at the plan's station, looked up from the route
table rather than differentiated; riding at offset `d` on a bend tightens the
radius to `κ / (1 − κ·d)` (the denominator floored at 0.1); and steering
across the road adds the offset profile's own curvature `d″(s)`.  Cross terms
in `d′` are dropped -- the offset moves about a metre over thirty, so `d′²` is
under a percent.

## Liveness

Standing still satisfies every rule above.  `makes_progress` is what stops
the optimizer from discovering that and parking forever.  Stating them as logic
rather than as a progress bonus in the cost keeps the whole specification in
one language -- and means a stopped vehicle reports *which* requirement it is
failing.

### `makes_progress` — soft, weight 2, margin 5.0
```
□ ((headway_slack > standstill_gap_m) ∧ (clearance > liveness_clearance_m))
    ⇒ ◇[0, 4.0s] (progress_slack > 0)
```
Cover a reasonable distance over the horizon: half of what travelling at the
average of the current and target speeds would cover.

Scaling by the *current* speed matters.  A target of "half the cruise speed
for four seconds" is simply unreachable from a standstill -- the acceleration
limit caps a four-second pull-away well below it -- so the rule would be
violated by every plan and the specification would be asking for the
impossible.

The rule is gated on *no traffic stopping us*, so that waiting for a
legitimate reason is not reported as a failure to make progress.  Two things
count: a vehicle close ahead (`headway_slack` no more than `standstill_gap_m`,
2.5 m, above the RSS distance) and anything else close enough to be worth
waiting for (`clearance` within `liveness_clearance_m`, 2 m).  Gating on the
lead vehicle alone ignored crossing traffic and vehicles alongside at
junctions, which is most of what actually stops a car in a city: of 165
reported violations over 500 steps, 90 had no lead vehicle at all but
something within two metres, and only 15 were genuine.

A red light is deliberately *not* in the gate.  What the rule demands is
capped by the room the stop line leaves -- the distance to the stop station
less the stopping distance from the current speed, floored at
`-liveness_stop_gap_m` -- so approaching a red light satisfies the rule
rather than being excused from it, and a rule that is satisfied still has a
gradient towards the line.

`makes_progress` wraps that condition in `Always`, so it reads "if nothing
obstructs us at any point in the horizon".  Asking only about the present
instant is the wrong question for a rule about distance covered *over* the
horizon: a road clear now and blocked in two seconds is not one to accelerate
down, and demanding progress there demands a plan the safety rules forbid.
There was a second liveness rule, `keep_moving`, asking pointwise for speed
above a crawl.  It was removed.  It is not a traffic rule -- no road code asks
a driver not to go slowly -- and it existed only to help `makes_progress` stop
the search discovering that standing still satisfies every safety rule.
Measured, it was not needed for that: on fixed situations the plans chosen
with and without it agreed to two decimal places, neither parked from any of
seven starting speeds, and a drive covered 192.3 m against 191.2 m.

What it did cost was a gate.  Any approach to a red light ends at rest, so a
rule demanding speed has to be excused near a stop line, and that excuse is
what made the vehicle stop 18.7 m short of a line it could see from 26 m and
wait there -- a rule made vacuous contributes a constant, and a constant has
no gradient towards the line.  `makes_progress` needs no such excuse, because
what it demands is already capped by the room the stop line leaves.

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
