# T4 Dataset

With `t4.enabled=true` each run is also written as a **T4 dataset**, TIER IV's
nuScenes-style format, which [t4-devkit](https://github.com/tier4/t4-devkit)
reads (`t4_devkit.T4Devkit`) and checks (`t4sanity`). Its sensors are the
driver policy's -- the LiDARs and cameras `driver.lidars` and `driver.cameras`
list ([External Driver Interface](driver_interface.md)) -- and a key frame is
a policy step (`driver.policy_timestep_s`): the ego's pose, the LiDARs' sweeps
as one `LIDAR_CONCAT` cloud in the ego's frame, a frame of every camera, and
the box of every vehicle and pedestrian near the ego, which CARLA knows
exactly.

```bash
pip install 'autoware-carla-scenario[t4]'
uv run scenario scenario=cut_in/cut_in ego.entity=carla_driver t4.enabled=true \
  'driver.lidars=[{logical_id: lidar_top, position_z: 2.2}]'
```

Where the data comes from:

- **An ego driven by a driver policy** (`ego.entity=carla_driver`) feeds the
  dataset what it sends the policy: the sweeps, already in the rig frame, and
  the camera frames, already JPEG (`driver.image_quality`). The dataset renders
  and encodes nothing of its own.
- **Any other ego** (Autoware, the autopilot, ROS 2) gets the same sensors put
  on it by the recorder, at the same rate; the recorder's writer encodes their
  frames.

The dataset is written by `autoware-carla-scenario-t4`, a Rust writer with a
PyO3 binding ([autoware_carla_scenario_t4](https://github.com/hakuturu583/autoware_carla_scenario_t4)), installed
with the `t4` extra. A run asking for a dataset without it is refused before
anything starts (exit code 2), saying how to install it. Recording is best
effort, like the [trajectory log](trajectory_log.md): an error is logged and
the dataset stops, it never fails the scenario.

## Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `t4.enabled` | bool | `false` | Write each run as `{ScenarioName}.t4/` |
| `t4.max_distance_m` | float or null | `100.0` | Annotate the vehicles and pedestrians within this distance of the ego; `null` annotates all of them. An actor parked out of the world (a hidden spawn, a trajectory entity before its start) is never annotated; revealed, it is annotated from the frame it appears |
| `t4.workers` | int | `4` | Threads writing the files, counting the boxes' points and encoding the recorder's own camera frames |
| `driver.lidars`, `driver.cameras` | list | | The sensors and their mounts |
| `driver.policy_timestep_s` | float | `0.1` | Seconds from one key frame to the next |
| `driver.image_quality` | int | `90` | The JPEG quality of the camera frames |
| `driver.rear_axle_offset_m` | float or null | `null` | Where the ego's `base_link` is (null: from the vehicle's wheels), unless the ego puts it elsewhere: a ROS 2 ego's `rear_axle_offset_m`, an Autoware ego's `base_link_offset_m` |

A dataset needs at least one LiDAR or camera; without a LiDAR the first
camera's frames are the key frames. A batch takes the first case's settings,
as it does its traffic backend and its ODD.

## Performance

The dataset is written on every key frame of every run, so the run's own
thread does as little as it can:

- the CARLA states -- the ego's and every road user's -- go to the writer as
  one array, and the writer turns them into poses and boxes in the map frame;
- a sweep is copied once, into a buffer the writer reuses from frame to
  frame; nothing of it is touched on the run's thread;
- the writer's threads write the sweeps, count each box's points (on a grid
  over the boxes, so a point is tested against the few boxes around it),
  write the JPEG frames and encode the recorder's own camera frames, while the
  run goes on; at most a bounded number of files wait for them.

A key frame with two 60,000-point sweeps, a JPEG frame and 50 road users
costs the run's thread about 0.3 ms.

## Layout

```text
{ScenarioName}.t4/            the dataset (its id is the directory's name)
  0/                          its one version
    annotation/*.json         the tables
    data/LIDAR_CONCAT/00000.pcd.bin
    data/CAMERA_FRONT_WIDE_120FOV/00000.jpg
    map/lanelet2_map.osm      the Lanelet2 map the run was on, when it had one
    map/map_projector_info.yaml
    status.json
```

| Table | What it holds |
|---|---|
| `scene`, `log` | One: the run's file stem, the scenario class, the ego's blueprint, the map |
| `sample` | One per key frame, linked frame to frame |
| `ego_pose` | The ego's pose at each key frame, with its `twist` and `acceleration` in its own frame |
| `sensor`, `calibrated_sensor` | `LIDAR_CONCAT` at the ego's origin, and each camera (its logical id upper-cased) with its mount and pinhole (or its lens, see [Frames](#frames)) |
| `sample_data` | Each sweep (`.pcd.bin`) and camera frame (`.jpg`) |
| `instance`, `sample_annotation` | Each road user, and its box at each key frame, with its velocity and `num_lidar_pts` |
| `category` | `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `pedestrian` (from the blueprint's `base_type`), with a null `index` |
| `visibility` | One level, `unavailable`: the boxes are ground truth, their occlusion is not measured |
| `vehicle_state` | The ego's pedals and steering (in %), its gear and speed, at each key frame |
| `attribute`, `map` | No attributes; the map record names `map/lanelet2_map.osm` |

A sweep is T4's default layout: `float32` `(x, y, z, intensity, ring = -1)`
per point, every LiDAR's points of the key frame together, in the ego's frame.
Timestamps are Unix microseconds: the wall-clock time the run's clock
started, plus the simulated time since, so key frames are exactly one policy
step apart.

## Frames

- The **map** frame is Autoware's: the Lanelet2 map's (CARLA's world frame
  with `y` flipped, plus the map's MGRS and elevation offsets; see
  [Trajectory Log](trajectory_log.md#frames-units-and-time)). Without a
  Lanelet2 map loaded it is CARLA's with `y` flipped.
- The **ego** frame (`base_link`) is at its rear axle, x forward, y left,
  z up: the driver policy's rig frame. `LIDAR_CONCAT`'s points are in it.
- A **camera**'s frame is the optical one (x right, y down, z forward); its
  intrinsics are the pinhole of its `fov` and resolution, with no distortion.
  A camera with a `lens` of its own ([F-theta](driver_interface.md#a-camera-with-a-lens-of-its-own-drivercameraslens))
  is written as OpenCV's rational model fitted to the lens: its focal lengths
  and principal point, and `camera_distortion`
  `[k1, k2, p1, p2, k3, k4, k5, k6]`, T4 having no F-theta model. The fit is
  logged; for Cosmos3's front wide camera it is within 0.05 px out to the
  image's corners. t4-devkit v0.8.0's own projection (`view_points`) applies
  the distortion before dividing by depth, so project with OpenCV
  (`cv2.projectPoints`).
- A **box** is its centre and orientation in the map frame, and its size
  `(width, length, height)`; velocities are in the map frame too.

## Reading it

```python
from t4_devkit import T4Devkit
from t4_devkit.dataclass import LidarPointCloud

t4 = T4Devkit("outputs/2026-10-22/10-00-00/CutInScenario.t4")
sample = t4.sample[0]
path, boxes, _ = t4.get_sample_data(sample.data["LIDAR_CONCAT"])  # boxes in base_link
points = LidarPointCloud.from_file(path).points                    # (4, N): x, y, z, intensity
```

`t4sanity` passes every check of severity ERROR. It warns of two things a
simulated run does not have: the ROS bag (`input_bag/`, STR005) and the point
cloud map (`map/pointcloud_map.pcd`, STR009). The tests read every dataset they
write back with t4-devkit v0.8.0, its latest release, and run its sanity
checker on it (`test_t4_dataset.py`). t4-devkit supports Python 3.10-3.12, and
its sanity checker needs `colorama`, which it does not declare.

## The writer

`autoware_carla_scenario_t4.T4Writer` writes any scene, not only a CARLA run's:

```python
import numpy as np
from autoware_carla_scenario_t4 import T4Writer

w = T4Writer("MyScene.t4/0", scene_name="MyScene")
w.add_sensor("LIDAR_CONCAT", "lidar", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
w.begin_frame(1_700_000_000_000_000, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
w.add_annotation(7, "car", (10.0, 0.0, 0.75), (1.8, 4.0, 1.5), (1.0, 0.0, 0.0, 0.0))
w.add_lidar("LIDAR_CONCAT", np.zeros((100, 4), dtype=np.float32))
w.end_frame()
print(w.finish())
```

Poses are right-handed and quaternions `(w, x, y, z)`; `begin_frame_carla`,
`register_actor` and `add_carla_boxes` take CARLA's states instead. Tokens are
derived from `token_seed` (the scene name by default), so the same frames
written twice are the same dataset. Its sources, its own tests and its
releases are [hakuturu583/autoware_carla_scenario_t4](https://github.com/hakuturu583/autoware_carla_scenario_t4).
