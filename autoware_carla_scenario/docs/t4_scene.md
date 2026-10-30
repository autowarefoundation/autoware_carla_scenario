# T4 Scene

A run's [trajectory log](trajectory_log.md) can be written as a **T4Scene**:
the Protobuf exchange contract of [tier4/e2e-devkit](https://github.com/tier4/e2e-devkit)
(`t4_scene.proto`, contract 0.1.0), which the devkit decodes into its
`T4Scene` -- the scene its data lists, scorers and visualisations work on. It
is not the [T4 dataset](t4_dataset.md), the nuScenes-style directory of
sensor data `t4.enabled` writes during a run: a T4Scene is written after the
run, from its log alone, holds no sensor data, and needs neither CARLA nor any
extra.

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
| `ego` | The actor whose role is `ego` or `hero`: `poses` and `poses_3d` at its `base_link`, `velocities`, `longitudinal_speeds`, `available`, `turn_indicators` (the log's `turn_indicators` records: disabled 1, left 2, right 3, hazard 4; unknown before the first), and its `shape` |
| `annotations` | Every other vehicle and walker within `--max-distance` (default 100 m) of the ego: its box's centre, size `(width, length, height)`, heading and quaternion, velocity, `labels` (`vehicle` 1, `pedestrian` 7) and `track_ids` (the CARLA actor id) |
| `traffic_lights` | Effective lane states: each lanelet a light governs, with its light's state (red 1, yellow 2, green 3, else unknown 0) |
| `map` | The `MapResource`'s SHA-256, `scope` full, with elevation, `map_id` `lanelet2-sha256:{the .osm file's SHA-256}` |

What the log does not hold is left out, never made up:

- **A row the log has no tick for** -- a tick without road users is not
  logged -- has its ego, objects and lights unavailable; rows are never
  interpolated.
- **The ego** has no accelerations, angular velocity or steering, and its
  `shape` no `wheel_base`; nor turn indicators in a log without
  `turn_indicators` records (one written before they were recorded).
- **Other road users' turn indicators** are in the log but not in the scene:
  the contract has them for the ego only.
- **A vehicle's class** is `vehicle`: the log has its blueprint, not the
  `base_type` that says car, truck or bus.
- **Raw signal groups** are not observed (`group_available` false): the log
  has CARLA's lights, not the map's regulatory elements and their bulbs.
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
- A light governs a lanelet as `TrafficSignalAction` maps them: when it
  switches an OpenDRIVE signal (its `opendrive_id`) of the controller named
  after the lanelet's traffic-light regulatory element
  (`Controller_TL_{id}`, the Lanelet2-to-OpenDRIVE convention); else, for a
  map without such controllers (roadgen's, say), when one of its stop-line
  waypoints is within 2 m of the lanelet's stop line (its traffic-light
  regulatory element's, else its end) and faces the lanelet's way -- the
  nearest such light, when several are.

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
