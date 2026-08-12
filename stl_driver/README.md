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
