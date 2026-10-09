# Quick start: test a driving policy

From an empty directory to a scenario result, with no ROS 2 and no Autoware.
The framework downloads CARLA, launches it, serves the driving policy in its own
process and runs the scenario: after the setup it is one command.

You need Linux x86_64, [uv](https://docs.astral.sh/uv/), `git`, an NVIDIA GPU
that runs CARLA UE5, and about 30 GB of free disk for the simulator.

## 1. Add the framework to your project

The framework is a Python library. Depend on it from your project's
`pyproject.toml`, pinned to a git revision:

```toml
[project]
name = "my-policy-tests"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["autoware-carla-scenario"]

[tool.uv.sources]
autoware-carla-scenario = { git = "https://github.com/autowarefoundation/autoware_carla_scenario", subdirectory = "autoware_carla_scenario", rev = "v3.5.0" }
```

```bash
uv sync
```

This also installs `autoware-carla-egodriver`, the policy side of the gRPC contract,
and the CARLA client (prebuilt; nothing is compiled).

## 2. Download CARLA

```bash
uv run scenario-setup
```

This downloads CARLA's nightly Linux build (about 16 GB, with a progress bar) and
unpacks it, while it downloads, into `~/.autoware_carla_scenario/bin/carla`. That is
where the scenario runner looks for CARLA when `CARLA_EXECUTABLE` is not set. Run
it again at any time: an install that is still the current nightly is kept, a
newer one replaces it.

| Option | Meaning |
| --- | --- |
| `--force` | Download again even if the install is current |
| `--url URL` | Install another CARLA Linux package (`.tar.gz`), e.g. a release |
| `--dir DIR` | Unpack somewhere else; then point `CARLA_EXECUTABLE` at the launcher it prints |

`AUTOWARE_CARLA_SCENARIO_HOME` moves `~/.autoware_carla_scenario` as a whole.

## 3. Run a scenario against a policy

```bash
uv run scenario scenario=cut_in/left map=town10hd_opt \
  ego.spawn_lanelet_id=324 scenario.npc_lanelet_id=446 \
  ego.entity=carla_driver driver.policy=route_follower
```

The single command does everything:

1. It launches CARLA (`CarlaUnreal.sh`, on `server.port`). The first boot takes a
   minute or two. A CARLA already listening on that port is used instead.
2. It fetches the Town10HD_Opt Lanelet2 map into `~/autoware_data/maps`, the first
   time only.
3. It serves `route_follower`, a reference policy, inside the scenario's own process.
4. It runs the scenario, prints the pass/fail result, writes it under `outputs/`,
   and stops CARLA again.

`ego.entity=carla_driver` hands the ego to a policy over alpasim's
`egodriver.EgodriverService` contract. `driver.policy` names the policy the
scenario serves itself.

The example scenarios name lanelets of the NishishinjukuMap by default, so on
Town10HD_Opt a run gives its own. The two above are one of the cases the
scenario's sweep matches. `scenario-expand` lists every case without running
anything, and a multirun runs them all:

```bash
uv run scenario-expand scenario=cut_in/left map=town10hd_opt
uv run scenario --multirun scenario=cut_in/left map=town10hd_opt hydra/sweeper=lanelet_constraint \
  ego.entity=carla_driver driver.policy=route_follower
```

## 4. Test your own policy

A policy implements `BaseDriver.drive()`. It gets the session's observations and
returns a trajectory in the rig frame (x forward, y left):

```python
# src/my_policy_tests/__init__.py
from autoware_carla_egodriver.driver import BaseDriver, DriveContext, DriveResult
from autoware_carla_egodriver.geometry import Pose, Trajectory


class MyPolicy(BaseDriver):
    name = "my_policy"

    def drive(self, ctx: DriveContext) -> DriveResult:
        plan = Trajectory.empty()
        for i in range(1, 41):  # 4 s ahead at 5 m/s
            plan.append(ctx.time_now_us + i * 100_000, Pose.from_xyz_yaw(0.5 * i, 0.0, 0.0, 0.0))
        return DriveResult(trajectory_in_rig=plan)
```

So that `uv sync` installs the module, give the project a build backend:

```toml
[build-system]
requires = ["uv_build>=0.9.7,<0.10.0"]
build-backend = "uv_build"
```

Then name the policy by its import path:

```bash
uv run scenario scenario=cut_in/left map=town10hd_opt \
  ego.spawn_lanelet_id=324 scenario.npc_lanelet_id=446 \
  ego.entity=carla_driver driver.policy=my_policy_tests:MyPolicy
```

See [External Driver Interface](driver_interface.md) for what a policy receives
(cameras, LiDAR, the map, traffic lights). Other scenarios are listed in
[Example Scenarios](scenarios/index.md).

## Running the policy as a separate process

A policy can also live in a process of its own, for example an alpasim driver or a
model in another environment. Leave `driver.policy` unset and point
`driver.address` at it:

```bash
# Terminal 1 -- the policy
uv run autoware-carla-egodriver serve --policy my_policy_tests:MyPolicy --port 50051

# Terminal 2 -- the scenario
uv run scenario scenario=cut_in/left map=town10hd_opt \
  ego.spawn_lanelet_id=324 scenario.npc_lanelet_id=446 \
  ego.entity=carla_driver driver.address=localhost:50051
```

Both modes use the same gRPC contract and the same policy class. The only
difference is which process serves the policy.
