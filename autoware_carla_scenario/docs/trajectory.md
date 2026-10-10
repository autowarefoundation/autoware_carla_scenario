# Trajectories and T4 Scene Replay

`FollowTrajectoryAction` moves a vehicle or a pedestrian along a given
trajectory. It is OpenSCENARIO's `FollowTrajectoryAction`, and it is what lets a
recorded drive be **written down as a scenario**: a T4 scene (the format
TIER IV's `tier4/e2e-devkit` reads) is transcribed
into one trajectory per road user, and each one is replayed around the ego.

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

## Transcribing a T4 scene

```python
from autoware_carla_scenario import read_t4_scene

scene = read_t4_scene("/data/t4/prd_jt/2025-12-09/scene_0001")
print(scene.scene_name, scene.area_map_id, scene.duration)
print(len(scene.objects), "road users")
scene.save("scene_0001.json")  # a small JSON a scenario package can carry
```

`read_t4_scene` reads three files of the scene directory:

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

!!! note
    Labels in a T4 scene may come from a tracker rather than a person
    (`annotation_source == "ground_truth"` marks the human-labelled ones). A
    track can break where the recording lost the object for longer than
    `max_gap_frames`.

The poses stay in the `map` frame of the scene's area map, so **the scenario has
to load that map** (`scene.area_map_id`): its Lanelet2 projector is what places
them in CARLA.

## Replaying it

`T4ReplayScenario` is a whole scenario: the ego starts where the recording's ego
did, it is sent to the scene's goal (or to where the recording ended), every
road user is replayed around it, and the run passes when the recording's
duration has elapsed. Subclass it to add the conditions the test is about:

```python
from autoware_carla_scenario import CollisionCondition, T4ReplayScenario


class ReplayWithoutCollision(T4ReplayScenario):
    def setup(self) -> None:
        super().setup()
        self.register_fail_condition(CollisionCondition(label="no_collision"))


scenario = ReplayWithoutCollision("scene_0001.json")
```

With `replay_ego=True` the ego is put on the recorded drive too, which is only
for an ego nothing else drives (a TrafficManager ego).

To replay the road users in a scenario of your own, call `replay_t4_objects`
from its `setup()`:

```python
from autoware_carla_scenario import T4Category, T4SceneTranscription, replay_t4_objects

scene = T4SceneTranscription.load("scene_0001.json")
actions = replay_t4_objects(
    self,
    scene,
    categories=[T4Category.VEHICLE, T4Category.PEDESTRIAN],
    blueprint_for=lambda track: "vehicle.lincoln.mkz" if track.length < 5.5 else "vehicle.byd.j6gen2",
)
```

Each track becomes an entity named `t4_<category>_<track_id>` and a pre-tick
`FollowTrajectoryAction` timed on the scene clock (`ReferenceContext.ABSOLUTE`,
so frame 0 is the run's first tick). Vehicles are replayed as
`vehicle.tesla.model3` and pedestrians as `walker.pedestrian.0001` by default;
cyclists are replayed as walkers, since CARLA 0.10 has no bicycle. In `POSITION`
mode a track the recording picks up late is spawned out of the world and waits
there until its first sighting (`hide_when_absent`).
