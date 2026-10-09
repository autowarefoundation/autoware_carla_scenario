# External Driver Interface

The ego vehicle in a scenario is normally driven by CARLA's TrafficManager. This page
describes the alternative: handing the ego to an **external driving policy** that plans
over gRPC, so a scenario becomes a test of that policy rather than of TrafficManager.

The wire contract is alpasim's `egodriver.EgodriverService`. Its policy side lives in
this workspace as **`autoware-carla-egodriver`**, a light package (grpcio, protobuf,
numpy, Pillow; Python 3.10+) a policy depends on without pulling in the scenario
framework. It replaces the driver half of
[`carla_driver_interface`](https://github.com/hakuturu583/carla_driver_interface), with
the same names, so policies written against that package port by changing imports.

## Architecture

The scenario framework plays the **runtime** role: it owns the world, ticks the
simulation, renders observations, and asks the policy what to do. The policy is a
separate process serving `egodriver.EgodriverService`.

```mermaid
flowchart LR
    subgraph scenario["Scenario process (Python 3.10-3.14)"]
        SR["ScenarioRunner<br/>owns the world and the tick loop"]
        CDE["CarlaDriverEntity"]
        CAM["CarlaCameraSensor(s)"]
        TF["TrajectoryFollower<br/>pure pursuit + yaw-rate trim + speed PI<br/>through the powertrain model"]
        SR -->|on_tick| CDE
        CAM -->|frames| CDE
        CDE --> TF
        TF -->|VehicleControl| SR
    end

    subgraph policy["Policy process"]
        P["egodriver.EgodriverService<br/>e.g. autoware-carla-egodriver serve"]
    end

    CDE -->|"observations + drive()"| P
    P -->|"planned trajectory"| CDE
```

Each simulation tick, `CarlaDriverEntity` tracks the most recent plan and applies a
control. Every `driver.policy_timestep_s` it also submits fresh observations — camera
frames, the ego's pose and dynamic state at every tick since the previous step (the
egomotion history the upstream runtime sends), and a route window rolled forward from
the ego's current position — and asks the policy to re-plan. Between policy steps the cached
plan keeps being tracked, which is how a policy slower than the simulation stays usable.

## Running a scenario against a policy

The scenario can serve the policy itself, in its own process, so one command runs
everything. Name the policy in `driver.policy` (a reference policy, or
`package.module:Class` for your own `BaseDriver`):

```bash
uv run scenario ego.entity=carla_driver driver.policy=route_follower
```

The policy is built once per run and served on a free local port; `driver.address`
is not used. Every scenario of a batch opens its own session on it, so a model is
loaded once. Over the wire nothing changes: the scenario still dials it over gRPC,
exactly as it dials a policy in another process.

To run the policy as a process of its own (another environment, a GPU host, an
alpasim driver), leave `driver.policy` unset and start the policy first. It is a
server, and the scenario is the client:

```bash
# In the policy's own environment
uv run autoware-carla-egodriver serve --policy route_follower --port 50051
```

Then run any scenario with the `carla_driver` ego entity:

```bash
uv run scenario ego.entity=carla_driver driver.address=localhost:50051
```

`ScenarioRunner` logs `Autopilot skipped for ego (id=…) — external control expected`
when the policy has taken over. Pass and fail conditions, recording, video rendering,
and sweeps all behave exactly as they do for a TrafficManager-driven ego.

## Configuration

The `driver` config group (`conf/driver/default.yaml`) holds the settings:

```yaml
driver:
  address: localhost:50051
  policy: null               # e.g. route_follower: served by the run itself
  timeout_s: 60.0
  policy_timestep_s: 0.1     # must be a multiple of the 0.05 s simulation step
  image_quality: 90
  warmup_s: 0.0              # run-up onto the spawn pose; see "Warming up a policy"
  route_horizon_m: 80.0
  route_resolution_m: 2.0
  rear_axle_offset_m: null   # null derives it from the wheel physics
  send_ground_truth: false
  send_renderer_data: true       # CARLA ground truth: lights, traffic, speed limit
  send_actor_ground_truth: true
  traffic_light_sight_distance_m: 60.0
  actor_horizon_m: 150.0
  random_seed: 0
  cameras:
    - logical_id: camera_front_wide_120fov
      image_width: 960
      image_height: 604
      fov: 120.0
      position_x: 1.5
      position_z: 1.6
  control:
    lookahead_gain_s: 0.6
    wheelbase_m: 2.8
    max_steer_angle_deg: 56.0   # CARLA 0.10: angle = 56 deg * steer**2
    steer_exponent: 2.0
    yaw_rate_ki: 3.0            # integral trim on the measured yaw rate
    speed_kp: 1.0               # plan tracked in time: plan acceleration + speed PI, in m/s²,
                                # turned into pedals by the ego's Chaos powertrain model
```

### Warming up a policy

A policy that reads a history -- past LiDAR maps, the ego's past poses -- has
none at the first step and plans as if the ego had stood still: an ego spawned
moving brakes on those first plans. `driver.warmup_s` gives it a run-up instead:
the `warmup_s` before the scenario's first frame are played by rule, so that
everything arrives at its first-frame pose at its initial speed.

- Every vehicle with an initial speed, the ego among them, is put back along its
  lane by its initial speed times `warmup_s`, keeping its offset from the lane
  centre and its heading relative to the lane, and driven onto its pose. One with
  no initial speed stands on its pose.
- Every pedestrian with an initial speed is moved in a straight line along its
  heading; one with none stands where it starts.
- The traffic lights are frozen, so the run-up spends none of their phases.

The policy gets every observation and plans all the way, but does not drive; the
scenario's clock starts, and the policy takes over, on arrival. Set `warmup_s` to
the history the policy reads (3 s for ResWorld's ego past).

`ego.entity` selects the entity and accepts three values:

| Value | Behaviour |
| --- | --- |
| `autopilot` (default) | CARLA's TrafficManager drives the ego. |
| `autoware` | Nothing drives the ego; the actor is left for an external stack. |
| `carla_driver` | An external policy drives the ego over the contract described here. |

A policy is usually built for one rig, so a preset can choose the vehicle and cameras
along with the rest of the group. `driver=vision_pilot` is
[VisionPilot](https://github.com/autowarefoundation/vision_pilot)'s: a Lincoln MKZ with
one 1920×1280, 50° front camera at 10 Hz (its own CARLA rig), and it hands the ego to the
policy (`ego.entity: carla_driver`):

```bash
uv run vision-pilot-driver --model-dir <weights> --port 50051   # in vision_pilot
uv run scenario driver=vision_pilot
```

A preset extends `driver/default` (`defaults: [default, _self_]`) and may set `ego.*`,
since the `driver` group is composed after `ego`.

`logical_id` is the name the policy looks a camera up by, so it must match what the
policy expects. `autoware-carla-egodriver`'s reference policies use
`camera_front_wide_120fov`.

## Using it from Python

Scenarios that build their ego themselves can attach the entity directly:

```python
from autoware_carla_scenario import CarlaDriverEntity, DriverClientConfig

scenario.ego_entity = CarlaDriverEntity(
    DriverClientConfig(address="localhost:50051", policy_timestep_s=0.1)
)
```

`ScenarioRunner` calls `BaseScenario.create_ego()`, which returns `ego_entity` when one
is set and otherwise instantiates `ego_type`. Assigning `ego_entity` is the supported
way to supply an entity that needs constructor arguments.

To test against a fake policy, pass a `BaseEgoDriverClient` implementation:

```python
CarlaDriverEntity(config, client=MyFakeDriverClient(config))
```

## CARLA ground truth

The alpasim contract has no field for a traffic light and none for other vehicles.
Both ride inside `DriveRequest.renderer_data`, an upstream-sanctioned `bytes` extension
point, as a serialized `carla_driver.v0.CarlaRendererData`:

| Field | Contents |
| --- | --- |
| `ego_traffic_light` | State of the light governing the ego lane |
| `ego_traffic_light_distance_m` | Distance to its stop line **along the ego's heading**; negative once crossed or when no line applies |
| `speed_limit_mps` | Posted limit for the ego lane, 0 when unknown |
| `actors[]` | Other vehicles: pose, bounding box, and velocity, all in the local frame |
| `weather`, `map_name`, `frame_id` | Scene context |
| `lidar[]` | One sweep per `driver.lidars` entry, packed `[N, 4]` float32 (x, y, z, intensity) in the rig frame |
| `map_id`, `traffic_lights[]` | With `driver.map_dir`: which map file set describes the world, and every light with its state and stop points |

!!! warning "Turning this off is not a no-op"
    A policy reads the payload defensively — a missing one means "no light applies" and
    "no other vehicles" rather than an error. So `send_renderer_data: false` does not
    fail loudly. It silently disables every rule that depends on the world outside the
    ego. For a specification-driven policy such as
    [stl_driver](https://github.com/hakuturu583/stl_driver), that is `stop_on_red`,
    `collision_free` and `safe_headway` — the car keeps driving and the rules simply
    stop applying.

### Finding the light that governs us

CARLA's `is_at_traffic_light()` answers a different question: whether the ego is inside
the light's *trigger volume*, which is about a metre thick along the road. A policy
asking CARLA therefore learns of a red light at the moment it arrives at it, when
stopping from any ordinary speed is already impossible.

So the lane graph is walked forward instead — up to `traffic_light_sight_distance_m` —
and any light whose stop line lies on one of those lanes governs us. Inside a junction
nothing governs us: having crossed the line, the thing to do is clear the box.

The point reported is not the stop waypoint itself but the **mouth of the junction it
governs**, reached by walking forward from the waypoint. Upstream measured stop
waypoints a median of 5.5 m short of their junction on `Town10HD_Opt`; a policy told to
stop there hesitates most of a car-and-a-half before the line a driver aims at.

This logic is ported from `carla_driver_interface`'s reference runtime
(`runtime/carla_world.py` at `af1dcd3`) so that a policy tuned against that runtime sees
the same numbers here.

### LiDAR

The contract has no LiDAR submission RPC, so sweeps ride in `renderer_data` too. Each
`driver.lidars` entry mounts a `sensor.lidar.ray_cast` (mount in CARLA's convention, like
a camera) spinning one full revolution per simulation tick, and each policy step sends
that tick's sweep, already converted into the rig frame. A sweep that does not arrive
within a second is left out rather than replaced by an older one.

A policy does not see that transport: the servicer records each sweep before `drive` as
a `LidarFrame` in `session.frame_history`, beside the camera frames, and announces it
through `on_frame` -- the one path every sensor takes. `frame_history_length` applies to
both, and `session.latest_frame("lidar_top").as_array()` unpacks the `[N, 4]` points as
`latest_frame("camera_front").as_array()` decodes an image.

```yaml
driver:
  lidars:
    - {logical_id: lidar_top, channels: 64, range_m: 100.0, position_z: 2.0}
```

### The map, as files

alpasim's services read a scene's map from its artifact, never off the wire; this is the
CARLA counterpart. With `driver.map_dir` set (and the `map` extra installed), the world's
OpenDRIVE is converted by [roadgen](https://pypi.org/project/roadgen/) at scenario start
and written to `<map_dir>/<map_id>/`, in the formats `driver.map_formats` names
(Lanelet2 by default), beside the OpenDRIVE source, roadgen's IR and its traces. `map_id`
is `<map name>-<first 12 hex digits of the OpenDRIVE's SHA-256>`, so a set is written
once per map and a changed map gets a new id.

Each step then carries only what changes: every light's state, named by its OpenDRIVE
signal id, with its stop points on OpenDRIVE lanes. A policy that sets `map_dir` (its
own copy of the directory) gets the set opened as `ctx.map`, and `ctx.stop_lines()`
resolves the lights through roadgen's traces into its format's own elements -- for
Lanelet2, the approach lanelet, the traffic-light regulatory element and the light's
line strings:

```python
class MyPolicy(BaseDriver):
    map_dir = "/shared/maps"   # the runtime's driver.map_dir, as this process sees it

    def drive(self, ctx):
        for stop in ctx.stop_lines():
            stop.state, stop.lane_ids, stop.position_local
```

### Diagnostics coming back

`DriveResponse.debug_info.unstructured_debug_info` is decoded as
`carla_driver.v0.CarlaDriveDebugInfo` when it parses as one, surfacing the policy's name,
its inference time, and whatever scalars it chose to publish. They are logged every ten
policy steps. Anything that does not parse is left as raw bytes — the field is
deliberately unstructured, and a policy is free to use its own encoding.

## Coordinate frames

Two conversions sit between CARLA and the contract, both handled in
`autoware_carla_scenario.driver.observation`:

* **Handedness.** CARLA's world is left-handed (x=East, y=South, z=Up, yaw clockwise);
  the contract is right-handed (x forward, y left, z up). Positions flip `y`, yaw is
  negated, and angular velocity — a pseudovector — picks up an extra sign flip.
* **Rig origin.** alpasim puts the rig origin at the rear-axle centre projected to the
  ground, while a CARLA actor's origin is at the vehicle centre. The offset is derived
  from the vehicle's wheel physics and logged at session start.

!!! warning "Rear-axle offset on CARLA 0.9.x"
    CARLA 0.9.x reports wheel positions in world coordinates and centimetres, so the
    derived offset can land far outside a plausible range. When that happens the value
    falls back to half the bounding-box length and a warning is logged. Set
    `driver.rear_axle_offset_m` (e.g. `-1.4`) to pin it explicitly.

Observations are submitted with the ego pose in the **local** frame and velocities in
the **rig** frame; the route is submitted in the rig frame. The plan comes back in the
local frame and is converted to the rig frame before the controller tracks it.

## Early termination

A policy can set `terminate_session` on its response. The entity records it, and
`ScenarioRunner` ends the tick loop — but only after evaluating the scenario's own pass
and fail conditions, so a condition firing on the same tick still decides the outcome.
On its own, an early stop is reported as a failure with the message
`Ego entity requested session termination`, because the scenario never satisfied its
pass condition.

## Writing a policy

A policy subclasses `BaseDriver` and returns a plan in the rig frame (x forward, y left,
origin on the ground below the rear axle); the servicer handles sessions, frame
retention, ego history and the rig/local conversion:

```python
from autoware_carla_egodriver.driver import BaseDriver, DriveContext, DriveResult
from autoware_carla_egodriver.server import run_server


class MyPolicy(BaseDriver):
    name = "my_policy"

    def drive(self, ctx: DriveContext) -> DriveResult:
        ...


run_server(MyPolicy(), port=50051)
```

Without CARLA, `autoware_carla_egodriver.testing.FakeLoop` drives a policy server over
real gRPC on a straight road, rendering each declared pinhole camera, which is what a
policy's CI runs (`autoware-carla-egodriver demo --driver localhost:50051` from the
command line). The scenario framework is then the CARLA-backed runtime for the same
server.

## Protobuf definitions

The protobuf definitions are **vendored**, not installed, in `autoware_carla_egodriver`.
alpasim's published `alpasim-grpc` requires Python ≥ 3.11, while this workspace supports
3.10 onwards -- Autoware's own environment is 3.10 -- so depending on it would drop 3.10
support. Instead the `.proto` files are copied verbatim from two upstreams (both
Apache-2.0) and compiled locally:

| Proto | Source | Carries |
| --- | --- | --- |
| `alpasim_grpc/v0/*` | `NVlabs/alpasim@6870924` | The `egodriver` service and its messages |
| `carla_driver/v0/*` | `hakuturu583/carla_driver_interface@af1dcd3` | The CARLA extension payloads |

Field numbers, package names, and import paths are preserved exactly, which is what
keeps the messages wire compatible.

Regenerate the committed modules after updating the vendored protos:

```bash
uv run python autoware_carla_scenario/scripts/compile_protos.py
```

`test_proto_generated.py` fails if the committed output drifts from the `.proto` files.
See `autoware_carla_egodriver/proto/README.md` for the full provenance.

## Limitations

* Only RGB cameras and ray-cast LiDARs are streamed.
* CARLA has no recorded drive, so with `send_ground_truth: true` the reference sent
  through `submit_recording_ground_truth` is the route itself: its waypoints, headed
  along the path and spaced `policy_timestep_s` apart. This is the *recorded*
  trajectory channel, unrelated to the CARLA ground truth in `renderer_data`.
* The route is a lane-following walk of CARLA's road graph, taking the first
  continuation at each fork. It is a rolling window, re-sent on every policy step so the
  horizon stays ahead of the vehicle, but it is not a global plan: the ego will not turn
  toward a goal. Scenarios needing a specific route should submit their own waypoints
  through the client.
* Actor reporting covers vehicles only — pedestrians are not included.
* Actor `dynamic_state` carries linear velocity; angular velocity and acceleration are
  left zero, matching the reference runtime.
