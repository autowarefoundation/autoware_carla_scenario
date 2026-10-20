# Adversarial Actions and NADE

The framework ports [TeraSim](https://github.com/mcity/TeraSim)'s adversarial
behaviour generation (#385). There are three ways to use it:

- **An adversarial action**: one named NPC or pedestrian does one adversarial
  thing when its trigger holds. For example, it brakes hard in front of a
  follower, or cuts in.
- **NADE**: an environment action that picks, every tick, which agent near
  the ego carries out which adversity. It samples them more often than they
  happen naturally, and keeps the importance-sampling weight that makes the
  run's statistics unbiased.
- **Scheduled static adversities**: a stalled vehicle, or the aftermath of a
  collision, from one time to another.

## The layers

| Layer | Module | What it does |
|---|---|---|
| Observation | `adversity.observation` | Builds a `Scene` of the agents in the Lanelet2 map frame. It answers TeraSim's traci questions: the leader and follower within 40 m, the lanes beside, the next signal, a pedestrian's crosswalk and nearest road. |
| Triggers | `adversity.triggers` | One pure function per kind. Each takes what the agent sees and returns an `AdversarialCommand`, or `None`. `adversity_command` applies the location gate first, then the kind's trigger. |
| Location | `adversity.location` | `AdversityLocation`: `ANY`, `INTERSECTION`, `HIGHWAY`, `ROUNDABOUT`. These are read from the map by the ODD's own probes, so an adversity fires where coverage records that location. |
| Commands | `adversity.command` | An acceleration, a lane change or a path, planned along the lanes into a timed path (`plan_command`, `predict_command`). |
| Execution | `actions._adversarial_execution` | Drives the path: it takes the vehicle from its traffic backend, follows the path, then hands the vehicle back. |
| NADE | `adversity.nade`, `adversity.avoidance` | Candidates, prediction, challenge, avoidability, importance sampling and the victims' response. |
| Actions | `actions.adversarial`, `actions.adversarial_traffic`, `actions.stalled_object` | What a scenario registers. |

## The kinds

The numbers are TeraSim's defaults; each can be overridden in
`TriggerThresholds`. The accelerations come from the Intelligent Driver Model
(`adversity.idm`), which stands in for SUMO's car-following model.

| Kind | Action | TeraSim | Fires when | Does |
|---|---|---|---|---|
| `BRAKE_CHECK` | `AdversarialBrakeAction` | Follower | A follower is close behind, and neither is stopped. The agent is not speeding up, and the follower is not slowing. | Brakes at the emergency deceleration (9 m/s²) for 2 s. |
| `IGNORE_LEADER` | `IgnoreLeaderAction` | Leader | Its leader holds it back by more than 1.5 m/s² (free flow against car following), and the speed difference is at least 2 m/s. | Accelerates at the free-flow rate for 2 s. |
| `CUT_IN` | `AdversarialCutInAction` | LaneChange | A lane beside is blocked only by the vehicle behind there (closer than SUMO's secure gap). | Changes into that lane in 1 s. |
| `LANE_CHANGE_ABORT` | `LaneChangeAbortAction` | LanechangeAbort | Blocked as for `CUT_IN`. The agent is moving, and its lanelet lasts at least 1 s more at its speed. | Drifts half a lane toward that side and back in 1 s. |
| `HEAD_ON` | `HeadOnAction` | Headon | As for `IGNORE_LEADER`, but the agent's lanelet and its leader's start more than 120° apart. | Uses `IGNORE_LEADER`'s command. |
| `TRAFFIC_RULE` | `RunTrafficRuleAction` | TrafficRule | It is stopping for a red or yellow light rather than for its leader. | Drives on through at the free-flow rate for 2 s. |
| `MERGE` | `AdversarialMergeAction` | Merge | It is beside a merging lane with traffic on it, and not already changing lanes (turned 5° or more off its lane). | One lane over: speeds up as if the merging vehicle ahead were not there, when that beats giving way by more than 1.5 m/s², for 2 s. Two lanes over: changes into the lane between in 1 s, when nobody there is within 2 s of headway. |
| `JAYWALKING` | `JaywalkAction` | Jaywalking | A pedestrian is off any crosswalk, with a road within 10 m. | Walks straight across at its speed (1.4 m/s if it stands) × U(0.5, 2.0), for up to 5 s. |
| `RUNNING_RED_LIGHT` | `PedestrianRunRedLightAction` | RunningRedLight | A pedestrian is waiting (at most 0.2 m/s) within 3 m of a crosswalk whose light is red. | Crosses it end to end at 2.8 m/s. |
| `STOP_CROSSING` | `StopCrossingAction` | StopCrossing | A pedestrian is on a crosswalk. | Stands still for N(2 s, 0.5 s), then walks a crosswalk's length on along its heading at 2.8 m/s. |

The pedestrian kinds (`PEDESTRIAN_KINDS`) are only ever a pedestrian's, and
the other kinds only a vehicle's.

### When an action fires

Each adversarial action takes the entity and a required `location`. Its
default trigger is an `AdversityFeasibleCondition` of its kind at that
location, which holds while the kind's trigger would fire. When the action
fires, the command the condition found is planned on the map and driven
(`following_mode`: `FOLLOW` by default, or `POSITION`, as TeraSim's
`moveToXY` does). At the end of the path the vehicle is handed back to its
traffic backend.

A `condition` given instead decides when the action fires. On that tick
the adversity's own trigger is still asked; when it does not hold, nothing is
done and a warning is logged. To require both, give the `AndCondition` of
the two.

```python
from autoware_carla_scenario import (
    AdversarialCutInAction,
    AdversityLocation,
    ElapsedTimeCondition,
)

# Cuts in on a highway whenever the gap allows it.
self.register_pre_tick(AdversarialCutInAction("npc1", AdversityLocation.HIGHWAY))
# Asked once, 3 s in: cuts in if it can then, else logs a warning.
self.register_pre_tick(
    AdversarialCutInAction(
        "npc2", AdversityLocation.ANY, ElapsedTimeCondition(3.0, label="later")
    )
)
```

### What is a merging lane

TeraSim recognises a ramp by its SUMO edge name, which contains `"Ramp"`.
Lanelet2 has no such name, so `MapFeatures.merges` defines a merging lane as
a lane that ends into the lane on its left. It either ends beside that lane,
or shares what follows with it. This holds at the lanelet itself, or along
its straightest successors within 100 m (`MERGE_WINDOW_M`).

### A crosswalk's light

`RUNNING_RED_LIGHT` reads the Lanelet2 traffic-light regulatory element that
the crosswalk lanelet carries. That is the pedestrian signal in an Autoware
map. A crosswalk without one has no red light to run.

## NADE

`AdversarialTrafficAction` runs TeraSim's NADE loop on every tick of the run:

1. **Candidates**: the `max_controlled` (5) vehicles nearest the ego within
   `control_radius` (50 m), and every pedestrian there.
2. **Adversity**: each candidate's adversity is the first entry of
   `adversities` whose trigger fires. That gives `{normal: 1 - p,
   adversarial: p}`.
3. **Prediction**: `horizon` (5 s) every `resolution` (0.5 s). The ego is
   predicted normal: it holds its acceleration along its lane, or changes
   lane when its turn indicator says so.
4. **Challenge**: the candidate's adversarial future conflicts with the
   ego's normal future (TeraSim's AV-centred NADE). With `challenge_npcs`, it
   is also checked against the other candidates' futures.
5. **Avoidability**: if no victim can brake out of the conflict (emergency
   brake, with a 0.5 m buffer), `p` is multiplied by 3e-4 and the magnitude
   by 40.
6. **Importance sampling**: `q = clip(p × magnitude (20), 0,
   max_is_probability (0.05))`, or `fixed_is_probability` when that is set.
   At most one adversity is sampled per tick, the nearest challenger first,
   and none for `cooldown` (5.5 s) after one.
7. **Response**: the sampled adversity's NPC victims avoid it with
   probability q = 0.2 (p = 0.99) by braking at the emergency deceleration
   for 3 s, or else accept it and hold their speed for 2 s. That response
   carries its own weight. An ego victim is never driven.

```python
from autoware_carla_scenario import (
    AdversarialTrafficAction,
    AdversityKind,
    AdversityLocation,
    NadeAdversity,
)

self.register_pre_tick(
    AdversarialTrafficAction(
        [
            NadeAdversity(AdversityKind.CUT_IN, AdversityLocation.HIGHWAY, 0.002),
            NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, 0.001),
        ],
        seed=7,
    )
)
```

It drives the NPCs and pedestrians a scenario registers. A traffic backend's
background vehicles have no entity to drive them through yet (#417).

### The weight

Each tick of a run multiplies the run's likelihood ratio by one factor:

- `p / q` when an adversity is sampled;
- `(1 - p) / (1 - q)` for each candidate whose adversity challenged a victim
  but was not drawn;
- the response's own factor, when one is drawn.

A statistic weighted by this ratio is the naturalistic one. The action's
`weight` is the ratio. `BaseScenario.importance_sampling_weight` is the
product of every action's weight, which is 1.0 without NADE. You can read it:

- as the built-in metric `importance_sampling_weight` (aggregation `last`;
  see [Metrics](metrics.md));
- in the trajectory log, where each sampled adversity is an `adversity`
  record with its `p`, its `q`, the weight at that point, the victims and the
  response (see [Trajectory Log](trajectory_log.md)).

To count which kinds were sampled, register a coverage item on
`AdversarialTrafficAction.sampled_kind` (see [Coverage](coverage.md)).
Weighted aggregation over a batch belongs in openodd_py (#403).

## Stalled objects and collisions

TeraSim's `StalledObjectAdversity` and `CollisionAdversity` are scheduled,
not sampled. `StalledObjectAction` stands a vehicle still at a pose from
`start_time` to `end_time` (scenario seconds; `None` means it is never
removed). Outside that window the vehicle is out of the world.

A vehicle cannot be spawned once a run has started, so the scenario spawns it
in `setup`, preferably out of the world (`spawn.hidden`). The action is a
`FollowTrajectoryAction` of a held pose, with `hidden_outside_trajectory`.
`collision_objects` creates two of them for a collision's aftermath.

```python
from autoware_carla_scenario import Lanelet2Pose, StalledObjectAction, collision_objects

self.register_pre_tick(
    StalledObjectAction("stalled1", Lanelet2Pose(1043, 30.0), start_time=5.0)
)
for crashed in collision_objects(
    "npc3", "npc4", Lanelet2Pose(2001, 12.0), Lanelet2Pose(2001, 17.0, heading=0.4),
    start_time=0.0, end_time=60.0,
):
    self.register_pre_tick(crashed)
```

## Construction zones

`ConstructionZoneAction` lays out TeraSim's `ConstructionAdversity` along a
lane (`adversity.construction`): from `start_s` to `end_s` metres along it,
past its lanelet's end on along the straightest successors. These are the zones:

| Zone | Where | Objects |
|---|---|---|
| warning | `warning_zone_length` (100 m) before the start | signs on the shoulder, every `warning_zone_spacing` (30 m) |
| taper_in | `taper_length_in` (60 m) from the start | cones, from 0.3 m inside the lane's edge to the work zone's offset |
| buffer | `buffer_zone_length` (10 m) | cone, cone, barrier |
| work | to `taper_length_out` (30 m) before the end | `construction_type`: cone, barrier, or mixed |
| taper_out | to the end | cones, back to the edge |
| termination | `termination_zone_length` (30 m) after the end | signs |

`spacing` (20 m) is the distance between objects, scaled by 0.7 in a taper
and 0.8 in the buffer. With `use_dynamic_spacing` and `speed_limit_mph`,
MUTCD's spacings apply instead. A taper can be `linear` or `curved` (an
s-curve). The zone closes in from `side`, `"right"` by default; TeraSim instead
picked the side from the SUMO lane's index.

From `start_time` to `end_time` (`None`: never removed) the objects stand in
the world as CARLA props. Unlike a vehicle, a prop may appear while a run is
under way. When the zone appears, two things happen:

- `clear_traffic` takes the backend's background vehicles in the zone off the
  road;
- `close_lane` tells the backend the lane is closed. SUMO then disallows
  every vehicle class on it, as TeraSim's `full_lane` mode does. The
  TrafficManager cannot close a lane, and says so; a `TrafficSinkAction` on
  the zone keeps its traffic out.

TeraSim's `full_lane` mode places no object. Here the objects are always
placed: a closure Autoware cannot see is one it drives into. At `end_time`, or
when the run ends, the props are destroyed and the lane is reopened.

```python
from autoware_carla_scenario import ConstructionSettings, ConstructionZoneAction

self.register_pre_tick(
    ConstructionZoneAction(
        1043, start_s=120.0, end_s=260.0,
        settings=ConstructionSettings(construction_type="mixed", side="right"),
        start_time=0.0,
    )
)
```

## Coordinate conventions

- **Triggers and NADE**: everything is in the Lanelet2 map frame: x East,
  y North, yaw in radians counter-clockwise from +x. Gaps run bumper to
  bumper along the lanes. A lane's `s` is measured along its centreline, and
  `t` is positive to the left.
- **Execution**: planned paths are `MapPose` vertices in that frame. The
  trajectory follower places them in CARLA's left-handed world (y South, yaw
  in degrees clockwise) through the map's offset, as for any trajectory.
- **Signals**: a light's state is keyed by its Lanelet2 traffic-light
  regulatory element's id, and read off CARLA by `adversity.snapshot`.

## Differences from TeraSim

| TeraSim (SUMO) | Here (Lanelet2 and CARLA) |
|---|---|
| SUMO's car-following model | The Intelligent Driver Model, with TeraSim's gains as defaults |
| `getLeader`, `getLaneChangeStatePretty` | `Scene`: lane chains of the routing graph, with SUMO's secure gap for "blocked" |
| `convert2D` along an edge | The lanelet's centreline and its normal |
| A ramp by its edge name `"Ramp"` | `MapFeatures.merges` (above) |
| A walking area controlled by a TLS link | The crosswalk lanelet's own traffic light |
| A person's SUMO `maxSpeed` | `pedestrian_max_speed`, 2.8 m/s (the fastest jaywalker) |
| StopCrossing counts steps at dt = 0.1 | The wait is in seconds |
| `is_lane_changing`: heading 5° or more off the SUMO lane's angle | The same, off the Lanelet2 lanelet's direction (`LanePosition.heading`) |
| Background vehicles are adversarial | Only registered entities are driven (#417) |
| Construction zones: SUMO vtypes for cones, barriers and signs; full_lane only closes the lane | CARLA props (`static.prop.*`), always placed, plus the lane closed to the backend (#402) |
