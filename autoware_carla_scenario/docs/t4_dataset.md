# T4 Dataset

With `t4.enabled=true` each run is also written as a **T4 dataset**, TIER IV's
nuScenes-style format, which [t4-devkit](https://github.com/tier4/t4-devkit)
reads (`t4_devkit.T4Devkit`) and checks (`t4sanity`). The recorder puts sensors
of its own on the ego -- the LiDARs and cameras `t4.lidars` and `t4.cameras`
list, whatever the ego's entity is (Autoware, a driver policy, the autopilot)
-- and makes a key frame of every sweep: the ego's pose, a sweep of every
LiDAR, a frame of every camera, and the box of every vehicle and pedestrian
near the ego, which CARLA knows exactly.

The dataset is written by `autoware-carla-scenario-t4`, a Rust writer with a
PyO3 binding (`autoware_carla_scenario_t4/` in this repository), installed
with the `t4` extra. The runner only reads the world and the sensors and
hands them over; the writer keeps the tables, counts each box's LiDAR points,
and encodes and writes the sensor files on threads of its own while the run
goes on.

```bash
pip install 'autoware-carla-scenario[t4]'
uv run scenario scenario=cut_in/cut_in t4.enabled=true
uv run scenario scenario=cut_in/cut_in t4.enabled=true t4.frequency_hz=20 \
  't4.cameras=[{channel: CAM_FRONT, image_width: 1920, image_height: 1080}]'
```

A run asking for a dataset without the extra installed is refused before
anything starts (exit code 2), saying how to install it. Recording is best
effort, like the [trajectory log](trajectory_log.md): an error is logged and
the dataset stops, it never fails the scenario.

## Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `t4.enabled` | bool | `false` | Write each run as `{ScenarioName}.t4/` |
| `t4.frequency_hz` | float | `10.0` | Key frames a second: 20 (every tick), 10, 5, 4, 2 or 1 |
| `t4.max_distance_m` | float or null | `100.0` | Annotate the vehicles and pedestrians within this distance of the ego; `null` annotates all of them |
| `t4.jpeg_quality` | int | `90` | The camera frames' JPEG quality, 1-100 |
| `t4.lidars` | list | `LIDAR_TOP`, 2.2 m up | Each a `channel` and the settings of a `CarlaLidarSensorConfig` (`channels`, `range_m`, `points_per_second`, the field of view, the weather noise, the mount) |
| `t4.cameras` | list | `CAM_FRONT`, 1280x720, 90° | Each a `channel` and the settings of a `CarlaCameraSensorConfig` (resolution, `fov`, the mount, the blueprint's post-processing) |

A mount is relative to the vehicle actor in CARLA's convention (x forward, y
right, z up, degrees), as for the [driver policy's sensors](driver_interface.md).
A camera takes a frame with every key frame: its `fps` is not used. A dataset
needs at least one LiDAR or camera; without a LiDAR the first camera's frames
are the key frames.

A batch takes the first case's `t4` settings, as it does its traffic backend
and its ODD.

## Layout

```text
{ScenarioName}.t4/            the dataset (its id is the directory's name)
  0/                          its one version
    annotation/*.json         the tables
    data/LIDAR_TOP/00000.pcd.bin
    data/CAM_FRONT/00000.jpg
    map/lanelet2_map.osm      the Lanelet2 map the run was on, when it had one
    map/map_projector_info.yaml
    status.json
```

| Table | What it holds |
|---|---|
| `scene`, `log` | One: the run's file stem, the scenario class, the ego's blueprint, the map |
| `sample` | One per key frame, linked frame to frame |
| `ego_pose` | The ego's pose at each key frame, with its `twist` and `acceleration` in its own frame |
| `sensor`, `calibrated_sensor` | Each LiDAR and camera, and its mount on the ego; a camera's intrinsics |
| `sample_data` | Each sweep (`.pcd.bin`) and camera frame (`.jpg`) |
| `instance`, `sample_annotation` | Each road user, and its box at each key frame, with its velocity, acceleration and `num_lidar_pts` |
| `category` | `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `pedestrian` (from the blueprint's `base_type`), with a null `index` |
| `visibility` | One level, `unavailable`: the boxes are ground truth, their occlusion is not measured |
| `vehicle_state` | The ego's pedals and steering (in %), its gear and speed, at each key frame |
| `attribute`, `map` | No attributes; the map record names `map/lanelet2_map.osm` |

A sweep is T4's default layout: `float32` `(x, y, z, intensity, ring = -1)`
per point, in the LiDAR's own frame. Timestamps are Unix microseconds: the
wall-clock time the run's clock started, plus the simulated time since, so
key frames are exactly `1 / frequency_hz` apart.

## Frames

- The **map** frame is Autoware's: the Lanelet2 map's (CARLA's world frame
  with `y` flipped, plus the map's MGRS and elevation offsets; see
  [Trajectory Log](trajectory_log.md#frames-units-and-time)). Without a
  Lanelet2 map loaded it is CARLA's with `y` flipped.
- The **ego** frame (`base_link`) is at its rear axle, x forward, y left,
  z up, as for Autoware and the driver policies.
- A **LiDAR**'s frame is CARLA's sensor frame with `y` flipped (x forward,
  y left, z up).
- A **camera**'s frame is the optical one (x right, y down, z forward); its
  intrinsics are the pinhole of its `fov` and resolution, with no distortion.
- A **box** is its centre and orientation in the map frame, and its size
  `(width, length, height)`; velocities and accelerations are in the map
  frame too.

## Reading it

```python
from t4_devkit import T4Devkit
from t4_devkit.dataclass import LidarPointCloud

t4 = T4Devkit("outputs/2026-10-22/10-00-00/CutInScenario.t4")
sample = t4.sample[0]
path, boxes, _ = t4.get_sample_data(sample.data["LIDAR_TOP"])  # boxes in the LiDAR's frame
points = LidarPointCloud.from_file(path).points                 # (4, N): x, y, z, intensity
```

`t4sanity` passes every check of severity ERROR. It warns of two things a
simulated run does not have: the ROS bag (`input_bag/`, STR005) and the point
cloud map (`map/pointcloud_map.pcd`, STR009). The tests read every dataset they
write back with t4-devkit v0.8.0 and run its sanity checker on it
(`test_t4_dataset.py`). t4-devkit supports Python 3.10-3.12, and its sanity
checker needs `colorama`, which it does not declare.

## The writer

`autoware_carla_scenario_t4.T4Writer` writes any scene, not only a CARLA run's:

```python
import numpy as np
from autoware_carla_scenario_t4 import T4Writer

w = T4Writer("MyScene.t4/0", scene_name="MyScene")
w.add_sensor("LIDAR_TOP", "lidar", (0.0, 0.0, 2.0), (1.0, 0.0, 0.0, 0.0))
w.begin_frame(1_700_000_000_000_000, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
w.add_annotation(7, "car", (10.0, 0.0, 0.75), (1.8, 4.0, 1.5), (1.0, 0.0, 0.0, 0.0))
w.add_lidar("LIDAR_TOP", np.zeros((100, 4), dtype=np.float32))
w.end_frame()
print(w.finish())
```

Quaternions are `(w, x, y, z)`. Tokens are derived from `token_seed` (the
scene name by default), so the same frames written twice are the same
dataset. Its sources are `autoware_carla_scenario_t4/src/` (`cargo test` runs
the writer's own tests).
