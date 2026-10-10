# Trajectories and Recorded-Scene Replay

`FollowTrajectoryAction` moves a vehicle or a pedestrian along a given
trajectory. It is OpenSCENARIO's `FollowTrajectoryAction`, and it is what lets a
recorded drive be **written down as a scenario**: a recording is transcribed
into one trajectory per road user, and each one is replayed around the ego.
Reading a dataset is left to tools outside the framework (see
[Replaying a recorded scene](#replaying-a-recorded-scene)), so the framework
carries no dataset format.

## The action

```python
from autoware_carla_scenario import (
    FollowTrajectoryAction,
    Lanelet2Pose,
    MapPose,
    ReferenceContext,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryTiming,
    TrajectoryVertex,
)

path = Trajectory(
    "npc1_path",
    [
        TrajectoryVertex(Lanelet2Pose(lanelet_id=120, s=0.0), time=0.0),
        TrajectoryVertex(Lanelet2Pose(lanelet_id=120, s=30.0), time=4.0),
        TrajectoryVertex(MapPose(x=81234.5, y=50123.0), time=7.0),
    ],
)
self.register_pre_tick(
    FollowTrajectoryAction(
        "npc1",
        path,
        time_reference=TrajectoryTiming(ReferenceContext.RELATIVE),
        following_mode=TrajectoryFollowingMode.POSITION,
        label="npc1_follows_path",
    )
)
```

It has the same four parts as in OpenSCENARIO:

| OpenSCENARIO | Here | Notes |
|---|---|---|
| `Trajectory` (`name`, `closed`, `shape`) | `Trajectory(name, vertices, closed=False)` | Only the `Polyline` shape. A recording is a sequence of samples; a clothoid or NURBS fitted to one would add an approximation between the data and the vehicle. |
| `Vertex` (`time`, `Position`) | `TrajectoryVertex(position, time=None, advance=None)` | Either every vertex has a time or none has. `advance` is an extension: see [Waypoint conditions](#waypoint-conditions). |
| `TimeReference` | `time_reference=None` or `TrajectoryTiming(domain, scale, offset)` | `None` is `<None/>`: the vertex times are ignored and the entity keeps the speed it had when the action started. With a timing, a vertex at time `τ` is reached at `τ * scale + offset` seconds after the scenario (`ABSOLUTE`) or the action (`RELATIVE`) started. |
| `TrajectoryFollowingMode` | `following_mode=POSITION` or `FOLLOW` | See below. |
| `initialDistanceOffset` | `initial_distance_offset` | Start that many metres along. With a timing, the clock starts at the time the trajectory reaches that point. |

A vertex's position may be written in any frame the framework addresses:

- `Lanelet2Pose`, `OpenDrivePose` and `CarlaWorldPose` -- as everywhere else;
- `MapPose(x, y, yaw=None, z=None)` -- an absolute pose in **Autoware's `map`
  frame** (the Lanelet2 file's projected frame), which is the frame a recording
  states its poses in. A missing `yaw` is taken from the direction of the path;
  a missing `z` puts the entity on the road surface under it;
- `RelativeLanePose(ds=0.0, offset=0.0, d_lane=0, yaw=None, entity_ref=None)`
  -- a pose **relative to an entity, in lane coordinates** (OpenSCENARIO's
  `RelativeLanePosition`); see [below](#lane-relative-vertices).

Positions are converted to CARLA world coordinates when the action first runs,
through the loaded map. Vertices of different kinds may be mixed in one
trajectory.

### Lane-relative vertices

A `RelativeLanePose` writes the shape of a manoeuvre -- pull out, overtake, cut
in ahead of the ego -- once, to be played from wherever it begins:

```python
from autoware_carla_scenario import EGO_ROLE_NAME, RelativeLanePose

overtake = Trajectory(
    "overtake",
    [
        TrajectoryVertex(RelativeLanePose(), time=0.0),  # where npc1 is
        TrajectoryVertex(RelativeLanePose(ds=20.0, d_lane=1), time=2.0),
        TrajectoryVertex(RelativeLanePose(ds=60.0, d_lane=1), time=5.0),
        # 10 m ahead of the ego, in its lane: a cut-in.
        TrajectoryVertex(RelativeLanePose(ds=10.0, entity_ref=EGO_ROLE_NAME), time=7.0),
    ],
)
self.register_pre_tick(
    FollowTrajectoryAction("npc1", overtake, TrajectoryTiming(), label="npc1_overtakes")
)
```

| Field | OpenSCENARIO | Meaning |
|---|---|---|
| `entity_ref` | `entityRef` | `role_name` of the reference entity. `None` (default) is the entity the action moves. |
| `ds` | `ds` | Metres along the reference entity's lane, from where it is; negative goes back. |
| `d_lane` | `dLane` | Lanes across: `+1` one lane to the left, `-1` one to the right, `0` the reference's own lane. |
| `offset` | `offset` | Metres from the **target lane's centreline**, positive to the left -- the sign of `Lanelet2Pose.t`. The reference entity's own lateral position does not carry over. |
| `yaw` | `Orientation` (relative) | Radians from the target lane's direction, positive to the left (anticlockwise) -- the sense of `Lanelet2Pose.heading`. `None` faces along the path, as for `MapPose`. |

How a vertex is placed, on the Lanelet2 map and its routing graph:

1. **The reference's lane.** The lanelet the reference entity is on, and the
   exact distance along its centreline. Where lanelets overlap (a junction) or
   meet (a seam), the one whose direction is nearest the entity's heading
   wins, then the one nearest its elevation, then the lowest id.
2. **`ds` along the lane.** Past a lanelet's end the distance carries on into
   the lanelet that follows it in the routing graph; a negative `ds` past its
   start into the one before it. Where several follow (a fork) or precede (a
   merge), the one that **turns least** is taken, then the lowest id -- the
   "straight on" a route most often takes, and deterministic. A `ds` that runs
   off a lanelet nothing follows is an error naming that lanelet.
3. **`d_lane` across.** At the point `ds` reached, one step per lane through
   the routing graph's left or right neighbour -- lane-changeable or not (a
   solid line still has a lane beyond it), but only lanes of the same
   direction of travel. The point keeps abreast: its distance along the
   neighbour is where the point projects onto it, so the outside of a bend is
   not shortchanged. A missing neighbour is an error naming the lanelet and the
   side. Lanelet2 only knows a neighbour that shares the lanelet's whole side,
   so where the two lanes are split into lanelets at different points (one
   20 m lanelet beside two 10 m ones) there is no neighbour to step to, and
   the vertex raises the same error.
4. **`offset` and `yaw`** are applied on that lanelet, which gives a
   `Lanelet2Pose`, converted to CARLA coordinates like any other.

**When.** A trajectory with relative vertices is placed **when the action
starts** -- the first tick of a run, against where each reference entity is at
that tick -- and is then fixed: the vertices do not move with the reference
entity afterwards, so the rest of the action (timing, both modes, hiding) works
on it exactly as on absolute vertices. A repeating action (`once=False`) places
it again at the start of every run. A reference entity that is not in the world
yet puts the start off until it is -- for as long as it takes: the action
neither ends nor fails meanwhile, and warns once per run; a lane the map does
not have raises a
`ValueError` naming the trajectory and the vertex. These lane checks can only
be made then: they depend on where the reference entity is.

### Following modes

- **`POSITION`** places the entity on the trajectory every tick, with the
  velocity the trajectory implies (so speed conditions and the recorder see the
  recorded speed). It is a kinematic replay: it reproduces a recording exactly,
  whatever the vehicle could actually do.
- **`FOLLOW`** drives the vehicle there with the pure-pursuit and speed
  controller of the [external driver interface](driver_interface.md)
  (`TrajectoryFollower`, on the vehicle's own powertrain model). The motion is
  the vehicle's own, so it can lag or cut a corner. A pedestrian is walked
  towards the path.

### Who drives, and when it ends

For the length of the run the action is the entity's only driver. A vehicle is
taken back from the [traffic backend](traffic_backends.md) first
(`TrafficBackend.release`); an ego driven by an external stack (Autoware, a
driver policy) is refused, because two authorities on one vehicle is not a
scenario anyone wrote.

The action stays `runningState` until the entity reaches the end of the
trajectory: the last vertex's time with a timing, the end of the path without
one (never, for a closed trajectory), within `ARRIVAL_TOLERANCE_M` (1 m) of it
in `FOLLOW` mode. Then a vehicle is braked to a stop and a walker stood still
where the trajectory ended. Pass `until=` to end it on a condition of your own.

`hidden_outside_trajectory=True` (an extension, `POSITION` mode with a timing
only) keeps the entity out of the world -- parked under the map with its physics
off -- before its first vertex's time and after its last. That is what a road
user that enters a recording late or leaves it early needs, since an entity
cannot be spawned once a run has started.

## Waypoint conditions

What moves an entity on from one vertex to the next is a **condition**. For an
ordinary vertex it is the implicit one a trajectory always had: the clock (a
timed trajectory leaves the vertex at its time) or the entity's speed (an
untimed one simply goes on). `TrajectoryVertex(..., advance=condition)` adds a
gate to a vertex: the entity is **held at that vertex until the condition
holds**, and then goes on, on the clock or at its speed again. A trajectory
without gates (`Trajectory.is_gated` is `False`) moves exactly as it always
did.

Any `BaseCondition` can gate a vertex -- one the framework has now, one it
gains later, or your own: an elapsed time, the distance to another entity, a
speed, a traffic signal, an `ActionStateCondition` on another action, and
`AndCondition` / `OrCondition` / `StickyCondition` / `PersistentCondition`
compositions of them. The action asks it only what every condition answers,
`check(world, elapsed)`, and as for an action's trigger any result other than
`None` counts as satisfied. `elapsed` is the **scenario's elapsed time**, as
for triggers, so `ElapsedTimeCondition(30.0, label=...)` reads "not before the
scenario is 30 s old".

```python
from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    ComparisonRule,
    ElapsedTimeCondition,
    EntityDistanceCondition,
    FollowTrajectoryAction,
    MapPose,
    Trajectory,
    TrajectoryTiming,
    TrajectoryVertex,
)

crossing = Trajectory(
    "walker_crossing",
    [
        TrajectoryVertex(MapPose(81200.0, 50100.0), time=0.0),
        # At the kerb: wait until the ego is within 25 m, then step out.
        TrajectoryVertex(
            MapPose(81205.0, 50100.0),
            time=4.0,
            advance=EntityDistanceCondition(
                "walker1", EGO_ROLE_NAME, 25.0, ComparisonRule.LESS_THAN,
                label="ego_close",
            ),
        ),
        TrajectoryVertex(MapPose(81205.0, 50110.0), time=11.0),
    ],
)
self.register_pre_tick(
    FollowTrajectoryAction("walker1", crossing, TrajectoryTiming(), label="cross")
)

# A trajectory built elsewhere (a recording, a lanelet route) is gated by index:
held = recorded.gated({12: ElapsedTimeCondition(20.0, label="not_before_20s")})
```

**When a gate is checked.** From the tick the entity **reaches** the vertex --
not before -- then on every tick while it is held. Once the condition holds the
gate is passed for the rest of that run; it is not asked again. If it already
holds when the entity arrives, the entity does not stop at all. A gate on a
vertex before the point a run starts from (`initial_distance_offset`) is not
checked. A repeating action (`once=False`) re-arms every gate at the start of
each run, and a **closed** trajectory checks its gates again on every lap.
Whatever state the condition object keeps is its own: a `StickyCondition` that
latched stays latched across laps and runs.

**While held**, by following mode:

| Mode | Timing | While held | After it opens |
|---|---|---|---|
| `POSITION` | a `TrajectoryTiming` | The trajectory's **clock is paused** at the vertex's time: the entity stands on the vertex with zero velocity. | The clock runs again from the vertex's time: every later vertex is reached the hold later, and each segment keeps its recorded duration (and `scale`). The end -- and so the action's completion, and `hidden_outside_trajectory`'s disappearance -- comes the hold later too. |
| `POSITION` | none | The distance stops at the vertex, at zero velocity. | It goes on at the action's speed. |
| `FOLLOW` (vehicle) | either | A gate not yet passed is a **stop line**: the controller's plan ends on it, slowed so that it comes to rest there braking at 2 m/s². Within `ARRIVAL_TOLERANCE_M` (1 m) along the path the vehicle has reached it, and while held it stands on the brake. | The plan runs on to the next gate or the end. With a timing the clock does not run past a gate not passed, and every stamp after it is shifted by the time held. |
| `FOLLOW` (walker) | either | Walked onto the gate, slowing as it gets close; within 0.3 m it has reached it and stands still. | Walked on, at the speed or the shifted schedule. |

In `POSITION` mode with a timing, the vertex is reached when the trajectory's
clock reaches its time, so a pause is exact in every time domain: an
`ABSOLUTE` replay begun after a gate's time is held **at** that gate (the
pause is honoured, not skipped), `RELATIVE` and `ABSOLUTE` shift the rest of
the schedule alike, `initial_distance_offset` starts the clock where it would,
and `hidden_outside_trajectory` keeps a held entity in the world even when
the gate's time is the last vertex's. On the tick a gate that held the entity
opens, the entity is at the vertex, moving off.

In `FOLLOW` mode a vehicle cannot stop within a tick, so it has to plan for a
gate it may have to wait at: it slows down for every gate not yet passed, and
one whose condition holds on arrival is driven through at the speed the
braking left it, not at full speed.

`FollowTrajectoryAction.held_vertex` says which vertex (0-based) the entity is
being held at, or `None`.

The last vertex of an **open** trajectory cannot carry a gate -- there is no
next vertex to go to (`Trajectory` raises `ValueError`); to end the action on a
condition, pass `until=`. On a closed trajectory the last vertex leads back to
the first, so it can.

## In the Scenario Editor

**Follow Trajectory** (category *Vehicle / Motion*) is a card like any other,
for a vehicle, a pedestrian or a TrafficManager ego. Its path comes from one of
three sources:

- **Vertices (map frame)** -- written in the card, one vertex per line:
  `x, y[, yaw][, time]` (metres, radians, seconds; `81234.5, 50123.0, , 1.5`
  leaves the yaw to the path). The line above the text says how many vertices,
  how long and how many seconds it adds up to. This is what a recording
  transcribes to, so a replayed road user's card can be edited like any other.
- **Along lanelets** -- the lanelets picked on the map, followed along their
  centrelines at a lateral offset and timed at a constant speed.
- **Relative to an entity's lane** -- [lane-relative vertices](#lane-relative-vertices),
  one per line: `ds[, offset][, d_lane][, yaw][, time]` (metres, metres,
  lanes, radians, seconds; only `ds` is required, `d_lane` is a whole number),
  measured from the entity chosen in **Relative to** -- or, left empty, from
  the card's own actor. The document stores them as rows
  `[ds, offset, d_lane, yaw, time]` and the reference as an entity id, which
  the compiler turns into that entity's role name. Deleting the reference
  entity clears the field, so the card is then measured from its own actor.

**Waypoint conditions** are under the card's fields in the inspector: *+
waypoint condition* takes a vertex number -- counted **from 1**, as the lines
of the vertex list (map-frame or relative) -- and a condition type, and adds
an ordinary condition tree there, drawn and edited exactly as a trigger is
(click it for its fields, add children to an *All* / *Any*). A second
condition added to the same vertex is combined with the first in an *All*.
The vertex number of each can be changed in place, and the bin removes it.
The card on the canvas says how many it has. In the document they are a list
on the action, each a vertex and a condition node:

```yaml
actions:
- id: a_walker
  type: follow_trajectory
  actor: walker1
  params:
    path_source: vertices
    vertices: [[81200.0, 50100.0, null, 0.0], [81205.0, 50100.0, null, 4.0],
               [81205.0, 50110.0, null, 11.0]]
  advance_conditions:
  - vertex: 2          # the second line of the vertex list
    condition:
      type: entity_distance
      params: {source: walker1, target: ego, rule: less_than, distance: 25.0}
```

The compiler builds each condition with the same builders as a trigger, and
the validator checks them as it checks a trigger, plus where they sit: a
vertex the card has, not its last one, one condition per vertex, and only for
vertices written in the card -- a lanelet path's vertices are generated every
2 m, nothing anyone could count, so it takes none. A condition that names an
entity or action you delete goes with it, as a trigger does.

The other fields are the action's: the time reference (from the action start,
from the scenario start, or none), its scale and offset, the following mode,
how far along to start, and whether the entity is out of the world outside the
trajectory's time. The validator checks what the fields only mean together: a
time reference needs vertex times (or a speed, for lanelets), and keeping the
entity out of the world needs the Position mode and a time reference. For
relative vertices it checks the numbers, the timing and that the reference
entity exists; whether the lanes they name exist is only known when the action
starts.

An entity's **Spawn out of the world** option (non-ego entities) spawns it under
the map with its physics off, for a card that brings it in at its first vertex:
a road user a recording picks up late, placed where it first appears, could
otherwise collide at spawn with whatever stood there when the run began.

## Replaying a recorded scene

A transcribed recording becomes an ordinary scenario document -- the one the
editor edits and the declarative runtime runs -- with nothing beyond the parts
above:

- the ego spawns where the recording's ego started and is sent where it went;
- every road user is an entity spawned on the lanelet it was first seen on,
  with one Follow Trajectory card: map-frame vertices timed on the scenario
  clock (`time_domain: absolute`, `time_offset` moving the recording's first
  frame to `t = 0`), in Position mode and hidden outside its trajectory;
- a road user first seen after the start spawns **out of the world**
  (`spawn.hidden`) and appears at its first vertex.

Open it in the editor to add the conditions the test is about.

T4 scenes are transcribed this way by `scenario-import-t4` of the separate
`scene_to_scenario_transpiler` package, which depends on the framework, never
the other way round.

!!! warning
    Map-frame vertices are Autoware's `map` frame, so a document replaying a
    recording has to run on **the map the scene was recorded on**, loaded with
    the same projector.
