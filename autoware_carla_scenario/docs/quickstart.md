# Quick start: test a driving policy

From an empty directory to a scenario result, with no ROS 2 and no Autoware.
The framework downloads CARLA, launches it, serves the driving policy in its own
process and runs the scenario: after the setup it is one command.

You need Linux x86_64, [uv](https://docs.astral.sh/uv/), `git`, an NVIDIA GPU
that runs CARLA UE5, and about 30 GB of free disk for the simulator. (The
framework also runs on Linux aarch64, but CARLA's server does not: there, run
CARLA on an x86_64 host and add `server.host=<address>` to the commands below;
see [installation](installation.md#operating-system).)

## 1. Add the framework to your project

The framework is a Python library on
[PyPI](https://pypi.org/project/autoware-carla-scenario/). Depend on it from your
project's `pyproject.toml`:

```toml
[project]
name = "my-policy-tests"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["autoware-carla-scenario>=5"]
```

```bash
uv sync
```

This also installs [`carla-driver-interface`](https://github.com/hakuturu583/carla_driver_interface)
1.x, the policy side of the gRPC contract (versioned on its own; a policy depending on
any 1.x shares the environment),
and the CARLA client, typesafe_carla. Its wheel carries a prebuilt CPython package.
Where that package does not match the installed toolchain, the first run that
imports the client builds it once instead (15-50 min, ~14 GB of memory, needs `cc`)
into `~/.cache/typesafe_carla`. `uv run typesafe-codon pycarla` does that ahead of
time, and says "nothing to build" when the prebuilt package applies.

## 2. Download CARLA

```bash
uv run scenario-setup
```

This downloads CARLA's nightly Linux build (about 16 GB, with a progress bar) and
unpacks it, while it downloads, into `~/.autoware_carla_scenario/bin/carla`. That is
where the scenario runner looks for CARLA when `CARLA_EXECUTABLE` is not set. Run
it again at any time: an install that is still the current nightly is kept, a
newer one replaces it. If the connection drops partway, the download picks up
where it stopped (with an HTTP range request) instead of starting over; it gives
up after several reconnections in a row that deliver nothing, and stops (the next
run fetches it whole) if the nightly was replaced in the meantime.

| Option | Meaning |
| --- | --- |
| `--force` | Download again even if the install is current |
| `--url URL` | Install another CARLA Linux package (`.tar.gz`), e.g. a release |
| `--dir DIR` | Unpack into `DIR` (new or empty) instead; then point `CARLA_EXECUTABLE` at the launcher it prints |

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
scenario's sweep matches. `scenario-expand` lists every case, each as the
overrides to add to the command above, without running anything:

```bash
uv run scenario-expand scenario=cut_in/left map=town10hd_opt
```

## 4. Test your own policy

A policy implements `BaseDriver.drive()`. It gets the session's observations and
returns a trajectory in the rig frame (x forward, y left):

```python
# src/my_policy_tests/__init__.py
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.geometry import Pose, Trajectory


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
uv run carla-driver-interface serve --policy my_policy_tests:MyPolicy --port 50051

# Terminal 2 -- the scenario
uv run scenario scenario=cut_in/left map=town10hd_opt \
  ego.spawn_lanelet_id=324 scenario.npc_lanelet_id=446 \
  ego.entity=carla_driver driver.address=localhost:50051
```

Both modes use the same gRPC contract and the same policy class. The only
difference is which process serves the policy.
