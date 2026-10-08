# autoware-carla-egodriver

Both ends of NVlabs/alpasim's `egodriver.EgodriverService` gRPC contract, light enough
for a driving policy to depend on:

- **policy side**: implement `BaseDriver.drive()` and serve it with `run_server()` (or
  `autoware-carla-egodriver serve --policy route_follower`); session bookkeeping, frame
  retention, ego history and the rig/local conversion are handled for you;
- **a CARLA-free runtime** for tests: `testing.FakeLoop` drives any policy server over
  real gRPC on a straight road, rendering each declared pinhole camera
  (`autoware-carla-egodriver demo --driver localhost:50051`).

The CARLA runtime is the scenario framework next door: `autoware_carla_scenario` with
`ego.entity=carla_driver` drives the same policy server inside a scenario.

```python
from autoware_carla_egodriver.driver import BaseDriver, DriveContext, DriveResult
from autoware_carla_egodriver.geometry import Pose, Trajectory
from autoware_carla_egodriver.server import run_server


class MyPolicy(BaseDriver):
    name = "my_policy"

    def drive(self, ctx: DriveContext) -> DriveResult:
        frame = ctx.session.latest_frame("camera_front")  # .as_array() -> RGB
        plan = Trajectory.empty()
        for i in range(1, 41):  # 4 s ahead, in the rig frame (x forward, y left)
            plan.append(ctx.time_now_us + i * 100_000, Pose.from_xyz_yaw(0.5 * i, 0.0, 0.0, 0.0))
        return DriveResult(trajectory_in_rig=plan)


run_server(MyPolicy(), port=50051)
```

Beyond alpasim's observations, a policy run by the scenario framework can read LiDAR
sweeps (`ctx.lidar_points()`, rig frame) and, with `map_dir` set to its copy of the
runtime's `driver.map_dir`, the world's map as files (`ctx.map`) and every traffic light
resolved into that map's own elements (`ctx.stop_lines()`, `autoware_carla_egodriver.hdmap`).

Dependencies: grpcio, protobuf 4.x, numpy, Pillow; Python 3.10–3.14. The wire contract
is vendored from alpasim and carla_driver_interface (`proto/README.md`), with field
numbers and service names unchanged, so the package also interoperates with an
upstream alpasim runtime.

This package replaces [carla_driver_interface](https://github.com/hakuturu583/carla_driver_interface)'s
driver half; its module and class names follow that package's, so policies port by
changing imports (`carla_driver_interface.driver.base` → `autoware_carla_egodriver.driver`,
`.geometry` → `autoware_carla_egodriver.geometry`, `.grpc_api` →
`autoware_carla_egodriver.protocol`, `.driver.server` → `autoware_carla_egodriver.server`).
