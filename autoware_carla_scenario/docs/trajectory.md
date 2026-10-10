# Trajectories and T4 Scene Replay

`FollowTrajectoryAction` moves a vehicle or a pedestrian along a given
trajectory. It is OpenSCENARIO's `FollowTrajectoryAction`, and it is what lets a
recorded drive be **written down as a scenario**: a T4 scene (the format
TIER IV's private `tier4/e2e-devkit` reads) is transcribed into one trajectory
per road user, and each one is replayed around the ego. The T4 reader is a
package of its own, `autoware_carla_scenario_t4`, so the framework carries no
dataset format.

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

## Writing a T4 scene down as a scenario

The `autoware_carla_scenario_t4` workspace package reads a converted T4 scene and
writes it down as a scenario document -- the one the editor edits and the
declarative runtime runs:

```bash
uv run t4-scenario /data/t4/prd_jt/2025-12-09/scene_0001 \
    --lanelet2 maps/my_area/lanelet2_map.osm --xodr maps/my_area/map.xodr \
    --map-group my_area --map-name MyArea \
    --to-editor --transcription scene_0001.json
```

- `--to-editor` saves it as a draft (`scenario-editor` lists it); `--output`
  writes the document YAML instead or as well.
- The ego is placed where the recording's ego started and sent to the scene's
  goal (or where the recording ended); `--ego-driver autoware` makes Autoware
  drive it, `--replay-ego` replays the recorded drive on a TrafficManager ego.
- Every road user on a lanelet becomes an entity with a Follow Trajectory card:
  timed from the scenario start, in Position mode, out of the world before its
  first sighting and after its last (`--following-mode follow` drives the
  vehicles with the controller instead). `--categories` picks which,
  `--vehicle-type`/`--pedestrian-type` their blueprints; cyclists are replayed
  as walkers, since CARLA 0.10 has no bicycle.
- The run passes when the recording's duration has elapsed and fails on an ego
  collision. Add the conditions the test is about in the editor.
- A saved transcription (`--transcription`, a `.json`) can be given instead of
  the scene directory, so the dataset is read once.

The same is available from Python: `read_t4_scene`, `T4SceneTranscription` and
`transcription_to_document` (`autoware_carla_scenario_t4.document`).

### What is read

| File | What is used |
|---|---|
| `derived/meta.json` | `scene_name`, `area_map_id`, `annotation_source` |
| `derived/scalars.npz` | `trajectory` `[N, 4]` (the ego's rear axle, `x, y, cos, sin` in the `map` frame), `shape` (`wheel_base, length, width`), `goal` |
| `derived/frames.pack` | `gt_boxes` `[M, 9]` = `x, y, z, width, length, height, yaw, vx, vy` and `gt_labels`, per frame, in that frame's ego frame |

- The **ego** becomes a timed trajectory of its vehicle centre (the rear axle
  moved forward by half the wheel base), on the scene clock: frame 0 at
  `t = 0`, 10 Hz. Frames without a valid pose are left out.
- Every **box** is moved into the `map` frame through the ego pose of its frame,
  and the boxes are **associated into tracks**: `frames.pack` stores boxes per
  frame with no track identity, so each open track predicts its object's
  position at constant velocity and takes the nearest box of the same category
  within `match_distance_m` (widened by how far the object could have moved).
  A track unseen for more than `max_gap_frames` is closed; tracks shorter than
  `min_track_frames` are dropped as clutter.
- T4 labels map onto `T4Category.VEHICLE` (0, 1, 2), `BICYCLE` (3) and
  `PEDESTRIAN` (4).
- Entities spawn on a lanelet, so each first pose is placed on the Lanelet2 map:
  among the lanelets nearest it, the one it is inside and whose direction
  matches its heading. A road user near no lanelet is left out (with a warning).

!!! note
    Labels in a T4 scene may come from a tracker rather than a person
    (`annotation_source == "ground_truth"` marks the human-labelled ones). A
    track can break where the recording lost the object for longer than
    `max_gap_frames`.

!!! warning
    The poses are read as Autoware's `map` frame, so the document has to run on
    **the scene's area map** (`area_map_id`) loaded with the same projector.
