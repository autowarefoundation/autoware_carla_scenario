# Cut-in

A vehicle in the next lane, 10 m ahead of the ego, changes into the ego's lane
2 s into the run. The ego must not hit it -- on every lane with a lane on that
side.

```bash
# Every case on the map, one run each
uv run scenario --multirun hydra/sweeper=lanelet_constraint \
    scenario=cut_in/left map=<map>
```

## Scenario

| | |
|---|---|
| **Ego** | Constraint search: a lane with a lane on the cut-in side, not in a junction, at least 60 m long, excluding the map's lanelets with no 3D model. 5 m along it, 30 km/h. |
| **NPC1** | *Beside the matched lanelet*, on the cut-in side: the lane next to the ego's. 15 m along it, 30 km/h. |
| **Initialization** | Every traffic light green. |
| **NPC1, after 2 s** | Change lane towards the ego. |
| **Pass** | 3 s have passed since NPC1 reached the ego's OpenDRIVE lane (*lane of a position*: the matched lanelet at the ego's spawn, the same lane), with no collision. |
| **Fail** | The ego collides with anything -- NPC1, another actor, a static object (`ego_collided`); the ego's centre is on none of the lanes ahead of it -- the matched lanelet and the lanelets that follow it, up to 300 m from its start -- for 0.2 s: it changed lanes, turned off, or left the road (`ego_left_its_lane`); or 20 s pass. |

## Variants

=== "From the left"

    ![Cut-in from the left in the Scenario Editor](media/cut_in_left.png)

    `scenario=cut_in/left` -- a lane on the ego's left; NPC1 changes right.

    <video controls muted playsinline width="100%" src="../media/cut_in_left.mp4"></video>

    *Town10HD_Opt, the first case: the search matched lanelet 324, where the ego starts; NPC1 starts beside it on lanelet 446. Result: PASSED.*

=== "From the right"

    ![Cut-in from the right in the Scenario Editor](media/cut_in_right.png)

    `scenario=cut_in/right` -- a lane on the ego's right; NPC1 changes left.

    <video controls muted playsinline width="100%" src="../media/cut_in_right.mp4"></video>

    *Town10HD_Opt, the first case: the search matched lanelet 446, where the ego starts; NPC1 starts beside it on lanelet 324. Result: PASSED.*

## Parameters

| Parameter | Value |
|---|---|
| NPC1 ahead of the ego | 10 m along its lanelet (at s = 15 m): `npc_ahead_m` |
| or, measured | `npc_gap_m`: the gap `vehicle_ahead_gap_m` reads at the start (centre to centre along the ego's heading); NPC1's `s` is solved for so it does. Unset by default; an ODD-sampled sweep sets it |
| NPC1 and ego initial speed | 30 km/h |
| Cut-in after | 2 s |
| Held in the ego's lane for | 3 s |
| Lanelet at least | 60 m |
| Timeout | 20 s |
| Fail on any collision of the ego | on: `fail_on_any_collision` (off: only a collision with NPC1, `hit_cut_in_vehicle`) |
| Fail on leaving the lanes ahead | on: `fail_on_leaving_lane` |
| The lanes ahead reach | 300 m from the start of the matched lanelet: `lanes_ahead_m` |
| Off them for | 0.2 s: `leaving_lane_seconds` (0: the first tick off) |

## Staying on the lanes ahead

The ego is to keep its lane while NPC1 cuts in: braking, or slowing, is the
answer the scenario is after, and a run that swerved into the next lane or off
the road has not passed it, whatever it did not hit. The lanes ahead are the
lanelet the ego spawns on and the ones that follow it in the routing graph
(`lanelets_ahead`: `following` without lane changes, every branch, up to
`lanes_ahead_m`); in none of them is in the lane beside, on a lanelet that
turns off them, or in no lanelet at all.

It is not a condition of its own but a composition of the framework's,
`left_lanes_ahead` in `examples/cut_in.py`, which any scenario can build the
same way:

```python
on_lanes = [
    EntityLanePositionCondition(
        EGO_ROLE_NAME, Lanelet2Pose(lanelet_id=i, s=0.0), label=f"ego_on_lanelet_{i}"
    )
    for i in lanelets_ahead(spawn.lanelet_id, 300.0)
]
self.register_fail_condition(
    PersistentCondition(                     # off them for 0.2 s, not one tick at a seam
        NotCondition(                        # on none of them -- off the road included
            OrCondition(on_lanes),           # on one of them
            holds_when_pending=True,
        ),
        0.2,
        label="ego_left_its_lane",
    )
)
```

`holds_when_pending=True` is what makes it a logical NOT. A condition that has
not fired returns nothing, and a plain `NotCondition` only flips the verdict of
a child that has, so it could not say "on none of these". The position tested
is the vehicle's centre (its CARLA location), so a wheel over the line is not
leaving the lane; a seam between two lanelets is covered by the next one, and
the 0.2 s wait absorbs a tick that falls between them.

A scenario document says the same with the editor's cards -- Persistent, NOT
(with *Holds while the child does not*), ANY and one Position (Lanelet2) per
lanelet:

```yaml
- type: persistent
  params: {duration: 0.2}
  children:
    - type: not
      params: {holds_when_pending: true}
      children:
        - type: any
          children:
            - {type: entity_lane_position, params: {entity: ego, lanelet_id: 79016}}
            - {type: entity_lane_position, params: {entity: ego, lanelet_id: 79017}}
```

"Off the road" alone is the same with every lanelet of the road under the ANY.

## Placing NPC1 by the measured gap

`npc_ahead_m` is a raw setting: NPC1's `s` on its lanelet less the ego's on
its own. Lanelets side by side are not the same length on a curve -- the
outer one is longer -- so the gap the `vehicle_ahead_gap_m` measure then
reads is not `npc_ahead_m`, and differs from place to place.

`npc_gap_m` is that gap itself. When it is set, the scenario places NPC1
with `place_at_vehicle_ahead_gap`: from the ego's spawn pose, it finds the
`s` on NPC1's lanelet at which the measure's own geometry reads `npc_gap_m`,
so the run starts at exactly that gap wherever it is placed. A gap the
lanelet cannot reach (too short, or curving out of the lanes the measure
counts) fails the setup, saying what gaps it does reach.

The scenario's controls draw the gap this way (`target: measure`,
[Controls](../odd.md#controls)), so a sweep's `vehicle_ahead_gap` draws are
the gaps the runs start at:

```bash
uv run scenario scenario=cut_in/left map=<map> scenario.npc_gap_m=15
```

The relative speed needs no solving: `npc_initial_speed_kmh` less the ego's
initial speed is what the `vehicle_ahead_relative_speed_kph` measure reads at
the start, up to the cosine of the small heading difference on a curve.
