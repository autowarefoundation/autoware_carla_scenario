# Example Scenarios

The examples shipped in `autoware_carla_scenario.examples` are **logical**
scenarios: none of them names an intersection, a lane or a map. Each says what
the lanelet it starts on has to *be* -- the approach to a signalised junction, a
lane with a lane on its left, a road's right-hand edge -- and the
[lanelet-constraint sweeper](../usage.md) expands it, on whatever map it is run
on, into one concrete scenario per lanelet that qualifies. Every other place the
scenario names follows from the lanelet the sweep picked.

| Scenario | What it checks | Variants |
|---|---|---|
| [Traffic Light Compliance](traffic_light_compliance.md) | Waits at a red light, goes on green | -- |
| [Temporary Stop](temporary_stop.md) | Stops at a stop sign's line, then goes | -- |
| [Intersection Passing](intersection_passing.md) | Turns or goes straight through a junction | left, right, right from a standstill, straight |
| [Lane Change](lane_change.md) | Changes lane -- or refuses where there is none | left, right, refused left, refused right |
| [Pedestrian Dart-out](pedestrian_dart_out.md) | Does not hit a pedestrian who runs out | -- |
| [Cut-in](cut_in.md) | Does not hit a vehicle cutting in | from the left, from the right |

## How each page reads

**In the Scenario Editor.** The screenshot is the example authored in the
[Scenario Editor](../scenario_editor.md), open on Town10HD_Opt. The *Places* panel binds one of
the lanelets the search matches -- the same one the video below it runs -- and
draws every place that follows from it: an NPC on the lane beside the ego, the
lanelet a condition watches, the ego's goal.

A lanelet in the editor is chosen one of three ways:

- **Fixed** -- this lanelet.
- **Constraint search** -- the lanelets matching a constraint tree; the sweeper
  runs the scenario once per match. One lanelet per scenario is searched for.
- **From the search** -- worked out from the lanelet the search picked, case by
  case, by a *binding*:

  | Binding | The lanelet it gives |
  |---|---|
  | The matched lanelet | The pick itself |
  | Beside the matched lanelet | The lane on its left or right a vehicle may change into |
  | Along the route from the matched lanelet | The lanelet that many steps on |
  | Lanelet before the stop line | The lanelet a spawn that far before the pick's stop line stands on |

  A spawn's *position along* its lanelet can be derived the same way --
  *Before stop line* puts it a fixed distance short of the stop line.

**Scenario.** What happens, in the order the editor lays it out: who starts
where, what is done and when, and what makes the run pass or fail.

**Parameters.** The values the example runs with. Each can be overridden from
the command line like any Hydra key -- `scenario.timeout_seconds=30`,
`ego.initial_speed_kmh=10` -- and the sweep's constraints are what decide
where it runs.

**Result.** One case, run on CARLA 0.10 with
[SUMO driving the traffic](../traffic_backends.md): CARLA's chase camera on the
left, `sumo-gui` replaying the same run on the right.

## Running one

```bash
# Every case on a map, one after another
uv run scenario --multirun hydra/sweeper=lanelet_constraint \
    scenario=cut_in/left map=town10hd_opt traffic=sumo

# Or list the cases, to run them anywhere
uv run scenario-expand scenario=cut_in/left map=town10hd_opt
```
