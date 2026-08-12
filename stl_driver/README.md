# stl_driver

An autonomous-driving policy whose behaviour is **written down as Signal
Temporal Logic** and driven by maximising the robustness of that
specification.

The point is that the driving rules are not spread through a planner as
heuristics and thresholds.  They are ten formulas in
[`src/stl_driver/rules.py`](src/stl_driver/rules.py), which you can read,
test, and change:

```
speed_limit      □ (speed_headroom > -0.5)
no_reverse       □ (speed > -0.05)
stop_on_red      □ ((light_gate > 0) ⇒ (stop_slack > 0))
collision_free   □ (clearance > 0)
safe_headway     □ (headway_slack > 0)
stay_on_route    □ (route_slack > 0)
lane_keeping     □[0.5s,4s] (|lateral_offset| < 1.2)
comfort          □ (|accel| ≤ 2 ∧ |jerk| < 4 ∧ |lat_accel| < 2.5)
makes_progress   □ (clear ⇒ ◇[0,4s] (progress_slack > 0))
keep_moving      □[1s,4s] (clear ⇒ (speed > 1.0))
```

`uv run stl-driver explain` prints the live specification with its bounds.
[`docs/RULES.md`](docs/RULES.md) explains why each rule is phrased the way it
is.

That file is *only* the specification -- what a rule is and how a set of them
is scored lives in `ruleset.py`, which knows nothing about driving. To drive
by different rules, replace `build_rules` and hand the result over; nothing
else needs editing:

```python
EgoPolicy(rules=RuleSet(my_build_rules(config)))
```

**It aims at the rules; it does not guarantee them.** The search maximises
robustness under a 70 ms budget rather than solving a constraint system, some
situations admit no satisfying plan, and the rules are checked against a
*forecast* of the other traffic. What the design gives is that the shortfall
is reported rather than hidden: over 500 steps with 30 vehicles every rule
held on 65.8% of steps, and `spec_robustness` said so on the rest.
[`docs/RULES.md`](docs/RULES.md) has the breakdown.

Built on:

- [`carla_driver_interface`](https://github.com/hakuturu583/carla_driver_interface)
  — the alpasim-compatible egodriver gRPC server and CARLA closed-loop runtime
- [`stlcg++`](https://github.com/UW-CTRL/stlcg-plus-plus) — differentiable STL
  robustness in PyTorch

## How a step works

```
route + ego state + CARLA ground truth
            │
            ▼   (numpy, once per step)
      RolloutInputs ──────────────────────────┐
            │                                 │
   plan parameters (6-D) ──► differentiable ──► signal [40 × 12]
            ▲                  rollout              │
            │                                       ▼
            │                              STL specification
            │                                       │
            │                    ┌──────────────────┴──────────────────┐
            └── Adam ascends ─── objective                    spec robustness
                                (weighted, smooth)          (exact; decides)
```

The search has two stages, and the split is not incidental:

**Sample first.** 64 quasi-random plans plus three fixed ones (hold speed,
brake, last step's answer), rolled out and scored in a single batch. This is
what makes the search robust to the shape of the landscape: once a plan
already passes through an obstacle, its clearance barely depends on the speed
profile — the plan goes through at any speed — so the gradient there is nearly
flat and descent alone would never escape.

**Then descend.** A handful of Adam steps on the best sample, against a
smoothed robustness with a sharpening temperature. This is what turns "brake
somewhat" into "brake exactly hard enough to stop at the line".

If no plan satisfies the rules marked `hard`, the policy emits an emergency
stop instead of its best effort — driving a plan that violates a rule you said
you would never violate is what makes `hard` meaningless.

Median step: **73 ms** against the runtime's 100 ms budget, measured by
`tests/test_latency.py`. The search is bounded by predicting whether the next
refinement step fits, so a slow machine takes fewer steps rather than missing
the deadline.

## Why the behaviour is explainable

Every step reports each rule's robustness separately in
`DriveResult.debug_scalars`. When the vehicle does something surprising, the
answer is in the numbers, in the same language the behaviour was specified in:

```
[  0] v= 0.23 rob=  0.293 fb=0 light=0 | comfort=0.29 keep_moving=0.30
[  20] v= 0.55 rob= -0.894 fb=1 light=1 | stop_on_red=-0.89 no_reverse=0.05
[  60] v= 0.00 rob=  0.000 fb=0 light=1 | stop_on_red=0.00 lane_keeping=1.16
[ 160] v= 3.49 rob=  0.304 fb=0 light=0 | comfort=0.30 keep_moving=1.04
```

That is a real trace from CARLA: approaching a red light, braking hard enough
to trip the fallback, holding exactly at the stop line for ten seconds, then
pulling away when it turns green.

`scripts/record_drive.py` films a rollout from a chase camera with those
numbers drawn on every frame; `recordings/stl_drive.mp4` is 70 s of Town10HD
with 30 background vehicles.

```bash
docker compose up -d carla
uv run --extra carla python scripts/record_drive.py --steps 700 --traffic 30
```

Watching it is what surfaced three bugs that no unit test would have caught,
because each one produced plausible-looking driving:

- **The footprint was inflated into the next lane.** Approximating a car with
  two discs turns a 2.0 m body into a 3.0 m one, leaving 0.29 m to a vehicle
  one lane over -- so an ordinary lane offset read as an imminent collision.
  Discs are now chosen from a stated inflation tolerance instead.
- **Vehicles behind counted as collisions.** A car queuing up behind us at a
  red light sat inside the collision margin, and since the plan only ever
  moves forward, that is a collision we cannot cause.
- **A forecast three seconds out triggered emergency braking.** Constant
  velocity says crossing traffic will meet us; that deserves slowing down, not
  7 m/s^2, and certainly not re-deciding it 100 ms later. The emergency plan
  now answers a narrower question -- is there something we can no longer plan
  our way out of? -- over the opening 1.2 s.

Together those took emergency stops from 25.5% of steps to 4.5%, lane
invasions from 15 to 0, and route tracking error from 0.55 m to 0.21 m.

What remained was a vehicle that still stopped, at nothing, for seconds at a
time. Two further changes, each measured on its own over repeated runs against
a fresh simulator:

- **Other vehicles are predicted along the road, not in a straight line.**
  Their velocity is resolved into the route's frame and advanced there. A
  straight line ignores the road: on a bend it carries the car we are
  following out of our lane, and carries oncoming traffic into it.
- **The emergency plan is only summoned when braking could help.** It always
  can when we are moving. It cannot when we are stopped and something is
  moving towards us -- our speed was never the cause -- and braking there
  merely guarantees we are stationary when it arrives.

Measured over two runs each, 500 steps on Town10HD with 30 vehicles:

| | distance | collisions | stopped at nothing | emergency stops |
|---|---|---|---|---|
| neither | 128.5 m | 0 | 26.9% | 2.0% |
| prediction only | 130.7 m | 0 | 26.2% | 3.0% |
| braking test only | 125.0 m | 0 | 24.9% | 1.1% |
| **both** | **135.0 m** | **0** | **22.0%** | **1.1%** |

An earlier attempt at the same symptom is worth recording as a warning: two
plausible fixes, adopted on a single measurement, turned 0 collisions into
470. They were reverted. Nothing here is kept without repeated runs against a
freshly restarted simulator, because a single rollout in this environment says
almost nothing.

## Setup

```bash
uv sync --extra dev          # driver only
uv sync --extra dev --extra carla   # plus the CARLA client, to run the simulator half
```

Python 3.11 or 3.12, inherited from `carla_driver_interface`. The `carla`
extra is pinned to 0.9.16 rather than upstream's 0.9.15 floor because 0.9.15
publishes no wheel for either version.

## Running

Without a simulator, against the built-in fake world:

```bash
uv run stl-driver demo --steps 200
```

Against real CARLA:

```bash
docker compose up -d carla                       # carlasim/carla:0.9.16
uv run stl-driver serve --port 50051 &
uv run --extra carla carla-driver-interface run \
    --carla-host localhost --map Town10HD_Opt --traffic 30 \
    --driver localhost:50051 --steps 400 --metrics-json /tmp/rollout.json
docker compose down
```

Server and client must be the same CARLA version. This policy reads no camera
images, so if Vulkan fails inside the container, `-nullrhi` is a viable
fallback where it would not be for a vision-based policy.

## Tests

```bash
uv run pytest -m "not slow"   # rules, rollout, perception, policy  (~4 s)
uv run pytest                 # plus closed-loop rollouts and latency  (~3 min)
```

The tests worth knowing about:

- `test_stlcgpp_contract.py` pins the upstream STL semantics this depends on.
  STLCG++ is installed from an unreleased commit, so a bump fails here rather
  than as a policy that mysteriously drives into things.
- `test_rules.py` asserts exact robustness values per rule, not just signs — a
  sign-only test passes just as happily on a rule reading the wrong column.
- `test_rollout.py` checks the gradient by finite differences. A rollout can
  produce perfectly sensible trajectories while silently reporting a zero
  gradient, and the only symptom is a policy that never steers.
- `test_optimizer.py::test_on_an_empty_road_it_drives` is the anti-parking
  regression: standing still satisfies every safety rule, so if the liveness
  argument breaks, this is what catches it.
- `test_closed_loop.py` drives the whole loop over real gRPC, with subclasses
  of `FakeWorld` that inject a red light and a parked vehicle — the stock fake
  world reports neither, so those rules would otherwise never fire.

## A note on the dependency

Verifying against real CARLA surfaced two bugs in `carla_driver_interface`.
Both are fixed and merged in
[PR #3](https://github.com/hakuturu583/carla_driver_interface/pull/3); this
project pins the merge commit, which is the earliest that works — before it,
the policy deadlocks at the first red light it stops for:

1. **The traffic-light distance was an unsigned Euclidean distance**, so it
   started growing again once the ego crossed the line. A policy could not
   tell "1 m to go" from "1 m past" — and because a stopped vehicle never
   leaves the light's trigger volume, one that correctly stopped just past the
   line was told forever that a line lay ahead, and never moved again.
   Measured at +1.06 m when the line was 1.06 m behind.

2. **Teardown destroyed the background vehicles before standing the traffic
   manager down**, leaving it driving actors that no longer existed. The
   resulting C++ exception is thrown on the traffic manager's own thread,
   where no Python handler can reach it, so the process aborted with SIGABRT
   after a rollout that had already succeeded.

`stl_driver` also defends itself against the first independently: a stop line
inside the margin is clamped so the rule stays satisfiable, and CARLA's
finite-differenced acceleration is confined to the physical envelope before it
anchors the plan.
