# Adversities

Each of TeraSim's phase-1 adversities, carried out by a road user placed so
that its own trigger fires, and NADE among background traffic: the CARLA
check of the [adversarial actions](../adversity.md) (#404).

Unlike the other examples these are **concrete** scenarios, not logical
ones: each case names its lanelets, on Town10HD_Opt.

```bash
uv run scenario scenario=adversity/cut_in map=town10hd_opt
```

## Scenario

The ego is driven by the TrafficManager, as in the other examples. Every
light is set green, except in `traffic_rule`, which sets them all red. The
road user that carries the adversity out is held at its speed until the
adversity takes it over.

The adversity's own trigger (`AdversityFeasibleCondition`, the action's
default, at `AdversityLocation.ANY`) decides when it fires. The case is placed
so that the trigger holds:

| Case | Who | Placed | The adversity | Fires |
|---|---|---|---|---|
| `brake_check` | NPC1 | 15 m ahead of the ego on lanelet 324, both at 30 km/h | `AdversarialBrakeAction`: brakes at 9 m/s² for 2 s, to a stop | about 1.3 s in |
| `ignore_leader` | NPC1 | 25 m behind a slow ego (8 km/h) on lanelet 324, at 25 km/h | `IgnoreLeaderAction`: speeds up at the free-flow rate for 2 s, as if the ego were not there | about 0.7 s in |
| `cut_in` | NPC1 | 6 m ahead in the lane on the ego's left (324 beside 446), at 30 km/h | `AdversarialCutInAction`: changes in front of the ego in 1 s | at once |
| `traffic_rule` | NPC1 | 10 m ahead of the ego on lanelet 719, every light red | `RunTrafficRuleAction`: slowing for its red light, drives on through it at the free-flow rate for 2 s | about 24 s in, at the light |
| `jaywalking` | pedestrian1 | at the kerb 35 m ahead of the ego, off any crosswalk | `JaywalkAction`: walks straight across the road | at once |
| `nade` | the background vehicles | 40 of the TrafficManager's, around the ego on lanelet 324 | `AdversarialTrafficAction` (NADE) over the five kinds | several times a run |

**When it passes.** A single adversity's run passes once the adversity has
been carried out and its road user handed back to traffic: its action
reaches `completeState` (`ActionStateCondition`). It fails if that has not
happened by `timeout_seconds`.

**NADE.**

- NADE samples far more often than TeraSim's numbers would, so that one 45 s
  run shows several adversities:
  - each kind's naturalistic probability is p = 0.05 a tick while its trigger
    holds;
  - each is drawn with q = 0.5;
  - the cooldown is 2 s;
  - an adversity counts when it challenges another NPC, not only the ego.
- The run passes after `nade_seconds`.
- The run's log records each sampled adversity as an `adversity` record, and
  the importance-sampling weight as the `importance_sampling_weight` metric
  (see [The weight](../adversity.md#the-weight)).

## Parameters

| Key | Default | What it sets |
|---|---|---|
| `scenario.kind` | `brake_check` | `brake_check`, `ignore_leader`, `cut_in`, `traffic_rule`, `jaywalking` or `nade` |
| `scenario.npc_lanelet_id` | 324 | The NPC's lanelet, or the pedestrian's kerb |
| `scenario.npc_ahead_m` | 15.0 | How far ahead of the ego's spawn `s` it starts (negative: behind) |
| `scenario.npc_offset_m` | -3.0 | The pedestrian's offset from its lanelet's centreline (negative: right) |
| `scenario.npc_speed_kmh` | 30.0 | The NPC's speed until the adversity takes it over |
| `scenario.ego_speed_kmh` | `null` | The ego's speed, held for the run; `null`: the TrafficManager's |
| `scenario.red_light` | `false` | Every light red |
| `scenario.timeout_seconds` | 30.0 | Fails when not carried out by then |
| `scenario.nade_seconds` | 45.0 | NADE: the run's length |
| `scenario.nade_probability` | 0.05 | NADE: each kind's naturalistic probability a tick |
| `scenario.nade_is_probability` | 0.5 | NADE: the importance-sampling probability |
| `scenario.nade_cooldown` | 2.0 | NADE: no adversity for this long after one (s) |
| `scenario.nade_challenge_npcs` | `true` | NADE: an adversity may challenge another NPC |
| `scenario.nade_seed` | 7 | NADE: the sampler's seed |

## Result

Each case was run on CARLA 0.10 with Town10HD_Opt through `scenario`. The
ego was driven by the TrafficManager and the run-time static check was on.
Every case PASSED: `brake_check` at 3.45 s, `ignore_leader` at 3.05 s,
`cut_in` at 1.20 s, `traffic_rule` at 26.40 s, `jaywalking` at 6.10 s, and
`nade` after its 45 s. The trajectory logs show the following:

| Case | What happened |
|---|---|
| `brake_check` | The NPC braked from 8.3 m/s to a stop in 1.25 s. It was held there to the command's end and handed back 2 s after firing. |
| `ignore_leader` | The NPC's path sped it up from 6.8 to 8.3 m/s for 2 s behind the 2.2 m/s ego. |
| `cut_in` | The NPC was in front of the ego 1 s after firing. |
| `traffic_rule` | The NPC, slowing for its red light at 8.2 m/s, drove 16 m on through the junction in 2 s. |
| `jaywalking` | The pedestrian walked straight across the ego's lane, at its speed times a factor drawn from the run's seed. |
| `nade` | Ten adversities in 45 s among 40 background vehicles: five cut-ins, three brake checks and two ignore-leaders. Each was sampled at q = 0.5, so the weight fell to about 4e-23. |

## Placed for the trigger

- **`ignore_leader`.** An NPC faster than its speed limit would slow down even
  as it ignores its leader. Its free-flow acceleration, towards the limit, is
  then negative too. This matches TeraSim, so the NPC drives below the limit.
- **`traffic_rule`.** The trigger fires only for an NPC that slows for its
  light rather than for a leader, so the NPC is placed ahead of the ego.
