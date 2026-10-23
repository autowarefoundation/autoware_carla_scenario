# Routes, Trajectories and Recorded-Scene Replay

`RoutingAction` sends a vehicle or a pedestrian through **waypoints** to a
goal. Each waypoint may carry *when*: an **arrival time** or a **departure
condition**. Who moves the entity is its `mode`:

| `mode` | Who drives | Between waypoints | Timing |
|---|---|---|---|
| `RoutingMode.DRIVER` (default) | The entity's own driver: the TrafficManager or SUMO for a vehicle, the pedestrian route walker, Autoware for an Autoware ego | Lanelet2 routes (walkable lanelets for a pedestrian) | Met by speed commands to the driver ([below](#driver-mode-and-timing)); an Autoware ego takes the route only |
| `RoutingMode.FOLLOW` | The action's own controller (OpenSCENARIO's trajectory following, `FOLLOW`) | Straight legs | Kept by the controller on the trajectory's clock |
| `RoutingMode.POSITION` | Nobody: the entity is placed every tick (`POSITION`) | Straight legs | Exact (a kinematic replay) |

It is what lets a recorded drive be **written down as a scenario**: a recording
is transcribed into one trajectory per road user, and each one is replayed
around the ego (`follow` or `position`), or handed to the road user's own driver
to keep its timing (`driver`).

!!! warning "Breaking change (19.0.0): `FollowTrajectoryAction` is gone"
    `FollowTrajectoryAction` and the editor's *Follow Trajectory* card
    (`follow_trajectory`) were removed; `RoutingAction` and the *Set Goal*
    card (`routing`) do everything they did. In Python write
    `RoutingAction(entity, trajectory=..., time_reference=..., mode=RoutingMode.FOLLOW)`
    (or `POSITION`) for `FollowTrajectoryAction(entity, trajectory,
    time_reference, TrajectoryFollowingMode.FOLLOW)`. In a document change
    `type: follow_trajectory` to `type: routing` and `following_mode` to
    `mode` (`follow` / `position` as before, or `driver`), and state
    `time_domain` (the card's default is now `none`). A document that still
    has a `follow_trajectory` card is refused with a message saying so; there
    is no automatic conversion.
Reading a dataset is left to tools outside the framework (see
[Replaying a recorded scene](#replaying-a-recorded-scene)), so the framework
carries no dataset format.

## The action

A goal and Lanelet2 waypoints are what an ego is given
(`RoutingAction("ego", Lanelet2Pose(42, 10.0), waypoints=[Lanelet2Pose(17, 0.0)])`);
a `trajectory` writes the waypoints out, the last one the goal:

```python
from autoware_carla_scenario import (
    Lanelet2Pose,
    MapPose,
    ReferenceContext,
    RoutingAction,
    RoutingMode,
    Trajectory,
    TrajectoryTimeCondition as At,
    TrajectoryTiming,
    TrajectoryVertex,
)

path = Trajectory(
    "npc1_path",
    [
        TrajectoryVertex(Lanelet2Pose(lanelet_id=120, s=0.0), At(0.0)),
        TrajectoryVertex(Lanelet2Pose(lanelet_id=120, s=30.0), At(4.0)),
        TrajectoryVertex(MapPose(x=81234.5, y=50123.0), At(7.0)),
    ],
)
self.register_pre_tick(
    RoutingAction(
        "npc1",
        trajectory=path,
        time_reference=TrajectoryTiming(ReferenceContext.RELATIVE),
        mode=RoutingMode.POSITION,
        label="npc1_follows_path",
    )
)
```

A trajectory has the same four parts as OpenSCENARIO's trajectory following:

| OpenSCENARIO | Here | Notes |
|---|---|---|
| `Trajectory` (`name`, `closed`, `shape`) | `Trajectory(name, vertices, closed=False)` | Only the `Polyline` shape. A recording is a sequence of samples; a clothoid or NURBS fitted to one would add an approximation between the data and the vehicle. |
| `Vertex` (`time`, `Position`) | `TrajectoryVertex(position, advance=None)` | A vertex's time is its departure condition, `advance=TrajectoryTimeCondition(t)` (read back as `vertex.time`); any other condition can take its place -- see [Waypoint conditions](#waypoint-conditions). Either every vertex has a time or none has, a vertex with another condition counting as having one. |
| `TimeReference` | `time_reference=None` or `TrajectoryTiming(domain, scale, offset)` | `None` is `<None/>`: the vertex times are ignored and the entity keeps the speed it had when the action started. With a timing, a vertex at time `τ` is reached at `τ * scale + offset` seconds after the scenario (`ABSOLUTE`) or the action (`RELATIVE`) started. |
| `TrajectoryFollowingMode` | `mode=RoutingMode.POSITION` or `FOLLOW` (or `DRIVER`) | See below. |
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
        TrajectoryVertex(RelativeLanePose(), At(0.0)),  # where npc1 is
        TrajectoryVertex(RelativeLanePose(ds=20.0, d_lane=1), At(2.0)),
        TrajectoryVertex(RelativeLanePose(ds=60.0, d_lane=1), At(5.0)),
        # 10 m ahead of the ego, in its lane: a cut-in.
        TrajectoryVertex(RelativeLanePose(ds=10.0, entity_ref=EGO_ROLE_NAME), At(7.0)),
    ],
)
self.register_pre_tick(
    RoutingAction(
        "npc1",
        trajectory=overtake,
        time_reference=TrajectoryTiming(),
        mode=RoutingMode.FOLLOW,
        label="npc1_overtakes",
    )
)
```

| Field | OpenSCENARIO | Meaning |
|---|---|---|
| `entity_ref` | `entityRef` | `role_name` of the reference entity, a `str` or an `EntityRole` (stored as its string). `None` (default) is the entity the action moves. |
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

### Modes

- **`DRIVER`** hands the route to the entity's own driver -- see
  [Driver mode and timing](#driver-mode-and-timing).
- **`POSITION`** places the entity on the trajectory every tick, with the
  velocity the trajectory implies (so speed conditions and the recorder see the
  recorded speed). It is a kinematic replay: it reproduces a recording exactly,
  whatever the vehicle could actually do.
- **`FOLLOW`** drives the vehicle there with the pure-pursuit and speed
  controller of the [external driver interface](driver_interface.md)
  (`TrajectoryFollower`, on the vehicle's own powertrain model), by default with
  its `time_indexed` longitudinal controller, which keeps the vehicle where the
  trajectory's clock has it. The motion is
  the vehicle's own, so it can lag or cut a corner. A pedestrian is walked
  towards the path.

### Driver mode and timing

In `driver` mode the route starts where the entity is -- at the first waypoint
when it appears there (`appear_on_start`) -- and passes every waypoint, each
placed on the lanelet under it (a walkable one for a pedestrian). A vehicle
the traffic backend drives is handed it as a path
(`BackendDriven.route_to`), a pedestrian walks it (`PedestrianEntity.route_to`),
an Autoware ego is given it as a mission.

**Timing.** When waypoints carry times (with a time reference) or departure
conditions, the action keeps running until the entity reaches the goal and,
every tick, sets the driver's speed to meet them (`TrafficManager`
`set_desired_speed`, SUMO `setSpeed`, the walker's step speed):

- the next waypoint with a time: the distance left to it **along the planned
  path** over the time left, clamped to `[0, limit]` (the lane's speed limit,
  2 m/s for a walker) and smoothed. A late entity goes at the limit to catch
  up. The driver still stops for lights and yields; the speed is a target;
- the next waypoint with a departure condition: a rising `route_progress`
  condition (what a logical scenario times its road users by) is
  **predicted** -- its entity's progress rate says when it will hold -- and
  the entity is paced to arrive then. Any other condition, or a measured
  entity that is not moving forward, is approached at cruise speed (`speed`,
  else the limit) braking to stop 2 m short of the waypoint, where it **holds**
  until the condition does;
- with nothing due ahead: `speed` if given, else the limit.

!!! warning "A hold can deadlock"
    An NPC holding before a waypoint until the ego has progressed, while it
    stands in the ego's way, waits for ever: the ego cannot pass it. Prediction
    slows the NPC down long before it would hold, so a `route_progress`
    condition on a moving ego seldom holds it; a condition that cannot be
    predicted does. Place waypoints that wait for the ego out of its path.

**An Autoware ego** takes no speed: it is given the route only, and the action
warns once that its waypoints' times and conditions cannot be met.

**Arrivals.** In every mode, each waypoint with a time or a condition is
logged when reached -- when it was due (its time; the moment its condition
first held) against when the entity arrived -- as a `waypoint_arrival`
record of the [trajectory log](trajectory_log.md), so a run can be scored on
its timing.

**Keeping to the recorded lanes.** A vehicle's written-out waypoints are not
routed one by one onto the nearest lane: the whole track between them is
sampled every 2 m and matched to the lanes it runs along -- road lanelets
within 4 m running within 60 degrees of its heading, joined on the routing
graph by following on, a lane change or a short path, the closest sequence
chosen (`traffic.route_match`) -- and the route goes through where it leaves
each one. A track that cannot be matched falls back to its waypoints one by
one, leaving out those no lane route reaches, with a warning. The
TrafficManager is given the route with its own lane changes off, and a route
ending in a junction is carried on along the lane after it until it is clear
of the junction, so it is never handed an empty path.

**When it ends.** A route given as a goal (and Lanelet2 waypoints) with
nothing to time ends as soon as the vehicle has it (as an ego's initial goal
always has); a route written out as waypoints, timed or not, runs until the
vehicle is within 1 m of its end along the planned path (a timed one then sets
its driver's speed back to the cruise speed); a pedestrian's ends when it
arrives.

### Who drives in follow and position modes, and when it ends

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
`stop_at_end=False` (an extension) leaves a vehicle moving as it arrived instead
-- in `POSITION` mode with the last segment's velocity -- for an action that
hands it straight back to traffic, as the adversarial actions do
(`resume_traffic`).

`hidden_outside_trajectory=True` (an extension, `POSITION` mode with a timing
only) keeps the entity out of the world -- parked under the map with its physics
off -- before its first vertex's time and after its last. That is what a road
user that enters a recording late or leaves it early needs, since an entity
cannot be spawned once a run has started.

`appear_on_start=True` (an extension, either mode) is the other way in: when a
run of the action starts, the entity -- spawned hidden -- is put on the first
vertex facing along the path, physics on, moving at `speed` (standing if
unset), and then follows the trajectory. Triggered by a `RouteProgressCondition`
it enters when the ego has come as far as it had when the road user was first
seen; it cannot be combined with `hidden_outside_trajectory`.

Vertices may also be placed against a logical scenario's *route*
(`RouteLanePose`, `RouteOppositePose`, `RouteCrossingPose`,
`RouteCrosswalkPose`, `RouteRoadsidePose`; `path_source: route` in a
document) -- see [Logical Scenarios from Routes](logical_scenarios.md).

### When a road user leaves: `DespawnAction`

A route's end is where the road user's recording ends, but its driver does not
know that: a vehicle the TrafficManager drives carries on wherever its traffic
takes it. `DespawnAction(entity, condition)` (card **Despawn**, `despawn`)
takes a vehicle or a pedestrian -- never the ego -- out of the world when its
condition holds: the traffic backend lets go of it (and stops protecting it),
the collision log stops listening to it, its actor is destroyed and the entity
is unregistered. An action still naming it afterwards logs that once and does
nothing (a routing run on it ends); a condition reads it as absent. The run
writes a `despawn` record to the [trajectory log](trajectory_log.md).

```yaml
- id: npc1_leaves
  type: despawn
  actor: npc1
  trigger:
    type: action_state
    params: {action: npc1_drive, state: completeState}
```

A `driver` route written out as waypoints runs until the entity reaches its
last one, timed or not, so its completion is the moment it arrives.

A road user the ego has left behind for good can leave too: one standing at
least 30 m behind the ego along the road reads, in a trigger,

```yaml
type: any
children:
- {type: action_state, params: {action: npc1_drive, state: completeState}}
- type: all
  children:
  - {type: action_state, params: {action: npc1_drive, state: runningState}}
  - {type: standstill, params: {entity: npc1, duration: 5.0}}
  - type: entity_distance
    params: {source: ego, target: npc1, measure: lane, signed: true,
             rule: less_than_or_equal, distance: -30.0}
```

`signed` (lane frame only) measures how far the target is ahead of the
subject along the road, the way the subject faces, negative behind it
(`coordinate.lane_distance.lane_offset`), so it reads the same whether the ego
is moving or stopped.

A pedestrian's written-out waypoints a few metres apart (no more than 10 m
from one to the next: a recorded walk) are walked as written, from where it
starts; farther apart they are routed between on the walkable lanelets.

## Waypoint conditions

**Each vertex is departed on one condition**, its `advance`: when it holds,
the entity leaves the vertex for the next one. A vertex with none is passed
through on arrival.

- **Time is one kind of condition.** `TrajectoryTimeCondition(t)` departs the
  vertex when the trajectory's clock reaches `t` -- the scenario clock mapped
  through the action's `TrajectoryTiming` (domain, `scale`, `offset`) and its
  `initial_distance_offset`. That is what a vertex time has always meant, and
  `vertex.time` reads it back. Without a time reference times are ignored, as
  OpenSCENARIO's `<None/>` says, and such a vertex is passed through.
- **Any other condition** -- one the framework has now, one it gains later, or
  your own: an elapsed time, the distance to another entity, a speed, a
  traffic signal, an `ActionStateCondition` on another action, and
  `AndCondition` / `OrCondition` / `StickyCondition` / `PersistentCondition`
  compositions of them. The action asks it only what every condition answers,
  `check(world, elapsed)`, and as for a trigger anything but `None` counts as
  satisfied. `elapsed` is the **scenario's elapsed time**, so
  `ElapsedTimeCondition(30.0, label=...)` reads "not before the scenario is
  30 s old". (A `TrajectoryTimeCondition` *inside* a composition has no
  trajectory clock to read and compares the scenario's elapsed time; the
  editor refuses it there.)

A vertex cannot carry two: a time and another condition are combined, if
that is what is meant, by giving the vertex an `AndCondition` of an
`ElapsedTimeCondition` and the other.

**Between two vertices.** After vertex *i* is departed at scenario time `T`:

- if vertex *i + 1* has a time whose scenario time is not before `T`, the
  entity arrives there exactly then, moving linearly in between. When *i* was
  departed on its own time this is exactly the timed interpolation a
  recording is replayed with (a zero-duration segment is a jump), so **a
  trajectory of times alone moves as it always did**;
- otherwise -- no time on *i + 1*, or one already past because the entity
  waited -- it goes at the **action's speed**: `speed=` (m/s) when given, else
  the speed the entity had when the run started. At (near) zero that segment
  is never finished, and the action warns when the run starts.

So after a late departure, a next vertex whose time is at or just after the
departure is reached at once -- a jump in `POSITION`; in `FOLLOW` the vehicle
cannot keep up and drives as fast as its controller lets it. That is the rule
"arrive at its time if that is not before the departure" doing what it says.

**Waiting.** In `POSITION` mode a condition is checked from the tick the
entity **reaches** its vertex -- not before -- then on every tick while it
waits there, standing
still with zero velocity. Once it holds the vertex is departed for the rest of
that run; it is not asked again. If it already holds on arrival the entity
does not stop. A condition that never holds keeps the entity there until
`until=` ends the run. On the **last vertex** of an open path a condition
means "end the run when it holds" (a time: when the clock reaches it, as
ever); arriving at a last vertex with none ends the run. On the **first
vertex** it is the departure from the start (a time: wait at the start until
then). Vertices before the point a run starts from (`initial_distance_offset`)
are not checked. A repeating action (`once=False`) re-arms every condition at
the start of each run, and a **closed** path -- which cannot carry times --
checks its conditions again on every lap in `POSITION` mode. Whatever state a
condition object keeps is its own: a latched `StickyCondition` stays latched.

```python
from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    ComparisonRule,
    ElapsedTimeCondition,
    EntityDistanceCondition,
    MapPose,
    RoutingAction,
    RoutingMode,
    Trajectory,
    TrajectoryTimeCondition as At,
    TrajectoryTiming,
    TrajectoryVertex,
)

crossing = Trajectory(
    "walker_crossing",
    [
        TrajectoryVertex(MapPose(81200.0, 50100.0), At(0.0)),
        # At the kerb: wait until the ego is within 25 m, then step out.
        TrajectoryVertex(
            MapPose(81205.0, 50100.0),
            EntityDistanceCondition(
                "walker1", EGO_ROLE_NAME, 25.0, ComparisonRule.LESS_THAN,
                label="ego_close",
            ),
        ),
        # Across by 11 s -- or, if the ego came late, at 1.4 m/s.
        TrajectoryVertex(MapPose(81205.0, 50110.0), At(11.0)),
    ],
)
self.register_pre_tick(
    RoutingAction(
        "walker1",
        trajectory=crossing,
        time_reference=TrajectoryTiming(),
        mode=RoutingMode.FOLLOW,
        speed=1.4,
        label="cross",
    )
)

# A trajectory built elsewhere (a recording, a lanelet route): `gated` replaces
# what the vertices named depart on, a time included.
held = recorded.gated({12: ElapsedTimeCondition(20.0, label="not_before_20s")})
```

**By mode.**

- `DRIVER`: see [Driver mode and timing](#driver-mode-and-timing) -- the
  entity is paced to arrive when the condition holds, or held 2 m short of
  the waypoint until it does.

- `POSITION` places the entity where that timeline has it every tick; waiting,
  it stands on the vertex itself.
- `FOLLOW` (vehicle): a vertex whose condition has not held yet is a **stop
  line** -- the controller's plan ends on it, slowed to come to rest there at
  2 m/s². A car cannot stop within a tick, so here the condition is checked
  **before** arrival, from as far off as stopping there takes (braking
  distance, a second's travel and the arrival band); one that holds then is
  passed, letting the vehicle through at speed and costing the schedule
  nothing -- even if it would no longer hold by the time the vehicle gets
  there, and a `PersistentCondition` starts counting from that distance. The
  last vertex of an open path is the exception: it is checked only on
  arrival, since departing it ends the run.
  Otherwise it is looked at every tick; within `ARRIVAL_TOLERANCE_M` (1 m)
  along the path the vehicle has reached the vertex and waits on the brake.
  The plan after it is stamped from the timeline above.
- `FOLLOW` (walker): looked at early in the same way, walked onto the vertex,
  slowing as it gets close; within 0.3 m it has reached it and stands.
- On a **closed** path `FOLLOW` finds where the entity is by projecting it
  near where it was, and that projection does not wrap from the end of the
  path to its start -- so in `FOLLOW` mode a closed path's conditions are
  reliable on the first lap only. `POSITION` has no such limit.

`hidden_outside_trajectory` keeps the entity out of the world until the first
vertex is departed on its time, and after the run ends -- for a trajectory of
times alone, exactly as before. `RoutingAction.held_vertex` names the
vertex (0-based) the entity waits at for a condition other than a time, or
`None`.

!!! warning "Breaking change"
    `TrajectoryVertex` no longer takes `time=` (or a time as its second
    argument), and the editor's vertex rows no longer have a time column:
    write `TrajectoryVertex(position, TrajectoryTimeCondition(t))`, and in a
    document a `trajectory_time` waypoint condition on the vertex. The old
    forms raise an error that says so.

## In the Scenario Editor

**Set Goal** (category *Vehicle / Motion*) is a card like any other, for the
ego, a vehicle or a pedestrian. **Driven by** is its mode (Driver, Follow,
Position). Its route comes from one of five sources:

- **Goal lanelet, via lanelets** (the default) -- a goal lanelet and `s`
  along it, and the lanelets to pass on the way: what an ego is sent to. It
  takes no waypoint conditions and no time reference.
- **Vertices (map frame)** -- written in the card, one vertex per line:
  `x, y[, yaw]` (metres, radians; `81234.5, 50123.0` leaves the yaw to the
  path). The line above the text says how many vertices and how long. Their
  times are waypoint conditions (below). This is what a recording
  transcribes to, so a replayed road user's card can be edited like any other.
- **Along lanelets** -- the lanelets picked on the map, followed along their
  centrelines at a lateral offset and timed at a constant speed.
- **Along the scenario's route** -- route vertices, see
  [Logical Scenarios from Routes](logical_scenarios.md).
- **Relative to an entity's lane** -- [lane-relative vertices](#lane-relative-vertices),
  one per line: `ds[, offset][, d_lane][, yaw]` (metres, metres, lanes,
  radians; only `ds` is required, `d_lane` is a whole number),
  measured from the entity chosen in **Relative to** -- or, left empty, from
  the card's own actor. The document stores them as rows
  `[ds, offset, d_lane, yaw]` and the reference as an entity id, which
  the compiler turns into that entity's role name. Deleting the reference
  entity clears the field, so the card is then measured from its own actor.

**Waypoint conditions** -- what each vertex departs on, its time included --
are under the card's fields in the inspector: *+ waypoint condition* takes a
vertex number -- counted **from 1**: the Nth vertex of the list (map-frame or
relative), blank and comment lines not counted -- and a condition type, and
adds an ordinary condition tree there, drawn and edited exactly as a trigger
is (click it for its fields, add children to an *All* / *Any*). A vertex's
time is the type **Trajectory time** (`trajectory_time`, one field: `time`
on the trajectory's clock), read "depart vertex N at". A second condition
added to the same vertex is combined with the first in an *All* (a Trajectory
time cannot be combined: it is only ever a vertex's whole condition). The
vertex number of each can be changed in place, and the bin removes it; a
recording's long list of times folds away. The card on the canvas says how
many it has, and **Speed where no time paces** sets the action's `speed`. In
the document they are a list on the action, each a vertex and a condition
node:

```yaml
actions:
- id: a_walker
  type: routing
  actor: walker1
  params:
    path_source: vertices
    mode: follow
    vertices: [[81200.0, 50100.0, null], [81205.0, 50100.0, null],
               [81205.0, 50110.0, null]]
    time_domain: relative
    speed_ms: 1.4
  advance_conditions:
  - vertex: 1
    condition: {type: trajectory_time, params: {time: 0.0}}
  - vertex: 2          # the second vertex
    condition:
      type: entity_distance
      params: {source: walker1, target: ego, rule: less_than, distance: 25.0}
  - vertex: 3
    condition: {type: trajectory_time, params: {time: 11.0}}
```

The compiler builds each condition with the same builders as a trigger, and
the validator checks them as it checks a trigger, plus where they sit and the
timing rules: a vertex the card has, one condition per vertex, every vertex
departing on something once any has a time, times that do not decrease, a
time reference only with times, a Trajectory time only as a vertex's whole
condition, no condition waiting for its own action to leave the running state
(the action runs for as long as the entity waits), and only for vertices
written in the card -- a lanelet path's vertices are generated every 2 m,
nothing anyone could count, so it takes none (its speed times it). A
condition that names an entity or action you delete goes with it, as a
trigger does. The own-action rule is deliberately conservative: any
reference to the card's own action but "is running" is refused, even under a
*Not* where it could not deadlock. A row with a time cell, as documents had before, is refused
with a message that gives the `trajectory_time` condition to write instead.

The other fields are the action's: the time reference (from the action start,
from the scenario start, or none -- the default), its scale and offset, the mode,
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
  with one Set Goal card: map-frame vertices, each departed at its
  recorded time (a `trajectory_time` waypoint condition) on the scenario
  clock (`time_domain: absolute`, `time_offset` moving the recording's first
  frame to `t = 0`), in Position mode and hidden outside its trajectory;
- a road user first seen after the start spawns **out of the world**
  (`spawn.hidden`) and appears at its first vertex.

Open it in the editor to add the conditions the test is about -- a road
user can wait for the ego by replacing one of its vertex times with another
waypoint condition.

!!! note
    Transcribers written for the earlier format (a time column in the vertex
    rows) have to write `trajectory_time` waypoint conditions instead; a
    document with a time column is refused with a message saying so.

T4 scenes are transcribed this way by `scenario-import-t4` of the separate
`scene_to_scenario_transpiler` package, which depends on the framework, never
the other way round.

!!! warning
    Map-frame vertices are Autoware's `map` frame, so a document replaying a
    recording has to run on **the map the scene was recorded on**, loaded with
    the same projector.
