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
| `Vertex` (`time`, `Position`) | `TrajectoryVertex(position, time=None)` | Either every vertex has a time or none has. |
| `TimeReference` | `time_reference=None` or `TrajectoryTiming(domain, scale, offset)` | `None` is `<None/>`: the vertex times are ignored and the entity keeps the speed it had when the action started. With a timing, a vertex at time `τ` is reached at `τ * scale + offset` seconds after the scenario (`ABSOLUTE`) or the action (`RELATIVE`) started. |
| `TrajectoryFollowingMode` | `following_mode=POSITION` or `FOLLOW` | See below. |
| `initialDistanceOffset` | `initial_distance_offset` | Start that many metres along. With a timing, the clock starts at the time the trajectory reaches that point. |

A vertex's position may be written in any frame the framework addresses:

- `Lanelet2Pose`, `OpenDrivePose` and `CarlaWorldPose` -- as everywhere else;
- `MapPose(x, y, yaw=None, z=None)` -- an absolute pose in **Autoware's `map`
  frame** (the Lanelet2 file's projected frame), which is the frame a recording
  states its poses in. A missing `yaw` is taken from the direction of the path;
  a missing `z` puts the entity on the road surface under it.

Positions are converted to CARLA world coordinates when the action first runs,
through the loaded map.

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

## In the Scenario Editor

**Follow Trajectory** (category *Vehicle / Motion*) is a card like any other,
for a vehicle, a pedestrian or a TrafficManager ego. Its path comes from one of
two sources:

- **Vertices (map frame)** -- written in the card, one vertex per line:
  `x, y[, yaw][, time]` (metres, radians, seconds; `81234.5, 50123.0, , 1.5`
  leaves the yaw to the path). The line above the text says how many vertices,
  how long and how many seconds it adds up to. This is what a recording
  transcribes to, so a replayed road user's card can be edited like any other.
- **Along lanelets** -- the lanelets picked on the map, followed along their
  centrelines at a lateral offset and timed at a constant speed.

The other fields are the action's: the time reference (from the action start,
from the scenario start, or none), its scale and offset, the following mode,
how far along to start, and whether the entity is out of the world outside the
trajectory's time. The validator checks what the fields only mean together: a
time reference needs vertex times (or a speed, for lanelets), and keeping the
entity out of the world needs the Position mode and a time reference.

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
