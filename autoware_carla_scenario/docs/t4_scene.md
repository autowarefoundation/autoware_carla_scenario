# T4 Scene

A run's [trajectory log](trajectory_log.md) can be written as a **T4Scene**:
the Protobuf exchange contract of [tier4/e2e-devkit](https://github.com/tier4/e2e-devkit)
(`t4_scene.proto`, contract 0.1.0), which the devkit decodes into its
`T4Scene` -- the scene its data lists, scorers and visualisations work on. It
is not the [T4 dataset](t4_dataset.md), the nuScenes-style directory of
sensor data `t4.enabled` writes during a run: a T4Scene is written from the
run's log alone, holds no sensor data, and needs neither CARLA nor any extra.

## As each run ends

With `t4scene.enabled=true` the runner writes each run's scene itself, right
after it closes the run's trajectory log, beside it in the run's output
directory -- a batch's cases and a SLURM task's alike:

```bash
uv run scenario scenario=cut_in/cut_in t4scene.enabled=true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `t4scene.enabled` | bool | `false` | Write each run as `{ScenarioName}.t4scene.pb` and `{ScenarioName}.t4scene.map.pb` |
| `t4scene.step_s` | float | `0.1` | The scenes' time step, a multiple of the 0.05 s tick (`--step`); another value is refused before anything runs |
| `t4scene.max_distance_m` | float or null | `100.0` | The road users within this distance of the ego are a scene's objects; `null` takes all of them (`--max-distance`, `--all-objects`) |

A batch takes the first case's settings, as it does `t4`'s. The conversion
runs on the runner's thread once CARLA is done with the case (the map is read
once however many cases ran on it). Like the log itself it is best effort: a
log that cannot be converted is logged as a warning and the run's verdict
stands.

## To a driver policy, every step

With `driver.t4_scene=true` (`ego.entity=carla_driver`), every drive also
carries the run so far as a T4Scene, in `CarlaRendererData.t4_scene`
(carla-driver-interface 2.4 and later), for a policy that plans from scenes --
`carla_driver_interface`'s `T4SceneDriver`, which hands it to a devkit driver
(`t4_e2e_devkit.driver.AbstractT4Driver`):

```bash
uv run scenario scenario=cut_in/cut_in ego.entity=carla_driver \
    driver.map_dir=/tmp/maps driver.t4_scene=true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `driver.t4_scene` | bool | `false` | Send each drive's scene; needs `driver.map_dir` and `driver.send_renderer_data` |
| `driver.t4_scene_history_s` | float | `3.0` | How far back a scene reaches, the current row aside |
| `driver.t4_scene_wheel_base_m` | float or null | `null` | The ego's wheel base in the scenes' ego shape; `null` leaves it out |
| `driver.t4_scene_goal` | `scenario` or `road_ahead` | `scenario` | The scenes' goal: where the scenario routes the ego (`route_to`; none without one), or the far end of the road ahead the policy is sent -- where a policy reading the egodriver observations has its goal |
| `driver.t4_scene_map` | `run` or `map_set` | `run` | The scenes' map: the run's Lanelet2 map, or the map set's (`map_dir`: roadgen's from the world's OpenDRIVE, or `lanelet2_path`), the one a policy reading the egodriver observations is given. The route and the lights are placed on it |

Each scene is what the log converts to (below), built as the run goes from the
same records (`t4_scene.live`): history only, the current row last -- the
drive's own tick -- and the rows before it 0.1 s apart, counted back from it,
its states unrounded (the log keeps millimetres). The rows are anchored on the
drive, not on the run's first tick, so a `driver.policy_timestep_s` that is not
a multiple of 0.1 s (0.05, 0.15 s, ...) still shows the policy its own tick,
with the current sweep on it at its true time; with one that is (the default),
the rows are the ones the log converts to. Every row's time is still
nanoseconds since the run's first tick, so two scenes' rows need not fall on
the same times. A run younger than `t4_scene_history_s` sends the rows it has.
On top of what the log holds, a scene carries what the policy is told and
a log does not: the ego's velocity and acceleration as the egodriver
observations carry them (of `base_link`, the acceleration an estimate), body
angular velocity (on a row the policy was not told, the log's), front tyres'
angle and odometry times; the first LiDAR's
current sweep on the current row (`[N, 5]` in the rig frame, intensity on the
0-255 scale, the fifth column zero); the route on every row -- the road ahead
the policy is sent (`driver.route_horizon_m`) walked along the map's lanelets
as OnePlanner's egodriver policy walked it (drivable lanelets' centrelines,
within 2.5 m and heading the same way; successors and lane changes from a
vehicle routing graph, junction bound ends within 0.3 m welded; the points and
the ego's pose in single precision, as the egodriver observations carry them),
after the few the ego has just passed -- and the goal (`t4_scene_goal`). A policy
reading scenes is thus shown what one reading the egodriver observations is. Its map is written once into the map set as the `t4`
format (`<map_dir>/<map_id>/t4_map.pb`, a `MapResource`), which the scene
references. Every other observation is sent as before: a policy that does not
read scenes is unchanged.

## After a run

```bash
# Every log under a run's output directory, written beside each log
uv run scenario-t4scene outputs/2026-10-29/10-00-00
# One log, somewhere else, at 20 Hz, with every road user however far
uv run scenario-t4scene outputs/.../CutInScenario.trajectory.jsonl -o scenes/ --step 0.05 --all-objects
```

```python
from autoware_carla_scenario.t4_scene import trajectory_log_to_t4_scene, write_t4_scene

export = trajectory_log_to_t4_scene("CutInScenario.trajectory.jsonl")
export.scene          # t4_scene_pb2.T4Scene
export.map_resource   # t4_scene_pb2.MapResource, or None without a map
write_t4_scene("CutInScenario.trajectory.jsonl")  # -> the two paths
```

## Files

| File | What it holds |
|---|---|
| `{stem}.t4scene.pb` | One serialised `T4Scene` message: the whole run |
| `{stem}.t4scene.map.pb` | One serialised `MapResource`: the Lanelet2 map the run was on, which the scene's `map` names by SHA-256 |

The contract carries no map bytes in a scene, so the map is a file of its
own; a scene without one (`--no-map`, or a log whose map is not on this
machine any more) has no `map` and no `traffic_lights`.

## What goes where

| T4Scene | From the log |
|---|---|
| `scene_metadata` | `scene_id`: the log's stem; `recording_id`: the world's `episode`; `tags`: `autoware_carla_scenario`, `map:{name}`, and `result:passed`/`result:failed` when the run reached a verdict |
| `timestamps_ns`, `frame_indices` | One row every `--step` seconds (default 0.1 s, a multiple of the 0.05 s tick) from the first tick to the last: nanoseconds since the first tick, and the CARLA frame of each row |
| `current_index` | `--current-index` (default 0: the whole run is the recorded future) |
| `ego` | The actor whose role is `ego` or `hero`: `poses` and `poses_3d` at its `base_link`, `velocities` and `longitudinal_speeds` of its `base_link` (below), `angular_velocities` (body axes, rad/s, from the ticks' `angular_velocity`), `available`, `turn_indicators` (the log's `turn_indicators` records: disabled 1, left 2, right 3, hazard 4; unknown before the first), and its `shape` |
| `annotations` | Every other vehicle and walker within `--max-distance` (default 100 m) of the ego, as a box (the `boxes` group): its box's centre, size `(width, length, height)`, heading and quaternion, velocity, `labels` (`vehicle` 1, `pedestrian` 7) and `track_ids` (the CARLA actor id) |
| `traffic_lights` | One group per traffic light regulatory element of the map, its one element the round lamp of the CARLA light it shows: colour (red 1, yellow 2, green 3, else unknown 0), shape circle, status on (off for CARLA's `Off`, else unknown), confidence 1. A consumer derives each lanelet's state from the map, as for any source |
| `map` | The `MapResource`'s SHA-256, `scope` full, with elevation, `map_id` `lanelet2-sha256:{the .osm file's SHA-256}` |

What the log does not hold is left out, never made up:

- **A row the log has no tick for** -- a tick without road users is not
  logged -- has its ego, objects and lights unavailable; rows are never
  interpolated.
- **The ego** has no accelerations or steering, and its
  `shape` no `wheel_base`; nor angular velocity in a log without the ticks'
  `angular_velocity` (one written before it was recorded); nor turn indicators in a log without
  `turn_indicators` records (one written before they were recorded).
- **Other road users' turn indicators** are in the log but not in the scene:
  the contract has them for the ego only.
- **A vehicle's class** is `vehicle`: the log has its blueprint, not the
  `base_type` that says car, truck or bus.
- **A light's bulbs** beyond its one round lamp: CARLA's light has one state.
- **No route and no goal.**
- **An entity spawned hidden** is on the scene from its `spawn` record to its
  `despawn` record, not while it is parked out of sight.

## Frames

- The scene's axes are Autoware's `map` frame, the Lanelet2 map's, converted
  from CARLA's as the [trajectory log](trajectory_log.md#frames-units-and-time)
  says, with the `mgrs_offset` and `z_offset` of its `map` record. Both the
  scene and the map are registered at the identity (`global_origin`,
  `global_origin_3d`), so the devkit draws and scores one against the other.
- The ego's `base_link` is its **rear axle**, as in T4 data: where the run
  took it to be, which the log's `rear_axle` record says (where the ego put
  it: the driver policy's rig origin, the ROS 2 ego's odometry or the
  Autoware ego's `base_link`; else from the wheels), or `--rear-axle-offset` metres along its
  x axis; in a log without that record, half its box behind its origin, where
  the run takes it on CARLA UE5 (no wheel positions).
  `shape.base_link_to_front` is from there to its box's front, and
  `base_link_to_rear_axle` is 0.
- The ego's velocity is its `base_link`'s too. The log records its actor
  origin's, so the rigid-body lever arm is added: `v = v_origin + w x r`,
  `w` the tick's `angular_velocity` and `r` the rotated
  `(rear_axle_offset, 0, 0)`, in the map frame -- as the driver observations
  do (`ego_observation`). The forward speed is the same either way; a turning
  ego's sideways velocity is not (`yaw_rate * rear_axle_offset` of slip at
  the origin). A log written before ticks carried `angular_velocity` keeps
  the origin's velocity.
- A light shows a traffic-light regulatory element when it switches an
  OpenDRIVE signal (its `opendrive_id`) of the controller named after the
  element (`Controller_TL_{id}`, the Lanelet2-to-OpenDRIVE convention, as
  `TrafficSignalAction` maps them); else, for a map without such controllers
  (roadgen's, carla-ue5-maps'), when one of its stop-line waypoints is on a
  lanelet the element governs or within 50 m of lanelets before one -- the
  first such light, in the log's order. That is how
  `carla_driver_interface.hdmap` finds a light on a Lanelet2 map: CARLA puts
  its stop waypoints where its own junction starts, often metres before the
  map's stop line.

`scenario-t4scene` reads a map once however many of its logs ran on it.

## The map

The map is the run's own `.osm` (its path is in the log's `map` record;
`--lanelet2` and `--xodr` point at it when it has moved), read with the
projection the run read it with, and written in the contract's one profile,
`LANELET2_OSM` in `LOCAL_METERS`: OSM XML whose nodes carry their exact map
coordinates as `local_x`/`local_y`/`ele`, with auxiliary `lat`/`lon` that are
not a geolocation. Its SHA-256 is over the encoding, the projection and those
bytes. This is the devkit's own encoding (`t4_e2e_devkit.maps.io.osm_bytes`),
reimplemented on Lanelet2 so that writing a scene needs neither the devkit nor
its torch; the devkit's `MapResourceStore.register` verifies it.

## Reading it with the devkit

```python
from t4_e2e_devkit.serialization.protobuf import MapResourceStore, from_proto, t4_scene_pb2
from t4_e2e_devkit.viz import figure_to_rgb, plot_bev_frame

store = MapResourceStore()
store.register(t4_scene_pb2.MapResource.FromString(open("Run.t4scene.map.pb", "rb").read()))
scene = from_proto(
    t4_scene_pb2.T4Scene.FromString(open("Run.t4scene.pb", "rb").read()),
    map_resources=store,
)
figure, _ = plot_bev_frame(scene, {"ground_truth": None})  # the BEV at current_index
```

`dataclasses.replace(scene, current_index=i)` moves the anchor, so a video of
the run is one `plot_bev_frame` per row (the devkit's `FFmpegVideoWriter`
encodes them).

## The contract

`proto/t4_e2e_devkit/t4_scene.proto` is the devkit's `proto/t4_scene.proto`,
copied byte for byte; its Python modules are generated into
`t4_scene/_proto/` by `scripts/compile_protos.py` ([proto/README.md](https://github.com/autowarefoundation/autoware_carla_scenario/blob/master/autoware_carla_scenario/proto/README.md)).
Its descriptor keeps the devkit's name, `t4_scene.proto`, and package,
`t4.devkit.experimental.v1`, so the devkit's bindings and these load into one
process. The contract is experimental on the devkit's side: when it moves,
copy the file again and regenerate.
