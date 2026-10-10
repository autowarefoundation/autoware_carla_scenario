# Autoware as the Ego

`ego.entity=autoware` hands the ego to Autoware. The framework spawns the ego
at the scenario's start, serves the scenario's mission (initial pose, goal,
waypoints) on the `AutowareBridge` gRPC server, and waits for Autoware to
report it is localized, routed, engaged and driving before the scenario clock
starts. Autoware's side of the bridge is the `scenario_bridge` node of
`autoware_carla_interface`
([autowarefoundation/autoware_universe#13319](https://github.com/autowarefoundation/autoware_universe/pull/13319)).

Who starts Autoware is `autoware.launcher.type`:

| Value | Who starts Autoware |
| --- | --- |
| `none` (default) | Someone else: launched by hand, or by Autoware's own `with_scenario` launch, for one scenario |
| `docker` | The framework, from a local Autoware workspace, in that workspace's dev container |
| `command` | The framework, by running `autoware.launcher.command` on this host |

With a launcher, every scenario gets **a fresh Autoware**: started once the
scenario's ego exists and its mission is being served, and removed (every
process, and with `docker` the container) when the scenario ends. Nothing
carries over from one scenario to the next -- not the localization, not the
route, not the operation mode, not a planner's memory of the last run -- which
is what makes a batch or a sweep of Autoware scenarios meaningful.

## A local workspace in its dev container

The workspace is the one you already build in: a clone of
[`autowarefoundation/autoware`](https://github.com/autowarefoundation/autoware)
with Autoware's sources in `src/`, checked out on a branch whose
`autoware_carla_interface` has `scenario_bridge`.

```bash
uv run scenario 'scenario=intersection_passing/*' map=town10hd_opt \
  ego.entity=autoware \
  autoware.launcher.type=docker \
  autoware.launcher.workspace=$HOME/autoware \
  'autoware.launcher.launch=[autoware_launch,e2e_simulator.launch.xml,"map_path:=/home/aw/autoware_map/Town10HD_Opt","simulator_type:=carla","bridge_address:={bridge_address}"]'
```

What happens:

1. **Before anything else** (before CARLA starts), the workspace is checked:
    - The image and the workspace's mount point are read from
      `.devcontainer/<autoware.launcher.devcontainer>/devcontainer.json` and the
      Compose service it names -- the same image and the same path
      (`/home/aw/autoware`) VS Code uses. colcon writes absolute paths into
      `install/`, so a workspace built in VS Code runs here, and the reverse.
    - The workspace is built (`colcon build` in that image) when
      `autoware.launcher.build` says so: `auto` builds it only when
      `install/setup.bash` is missing, `always` before every batch, `never`
      leaves building to you. The output goes to `autoware-build.log`.
    - A workspace whose `autoware_carla_interface` has no `scenario_bridge` --
      one built from a branch without scenario support -- is refused here,
      not after the first scenario has timed out waiting for it.
    - Containers left behind by a run of yours that was killed (they carry the
      `autoware_carla_scenario.stack=uid-<your uid>` label) are removed.
      Another user's are left alone, but two of your own batches on one host
      are not supported -- the shared host network (CARLA's ports, the
      bridge's, the ROS graph) would not allow them anyway.
2. **For each scenario**, once the ego is spawned and the mission served:
    `docker run --rm --network host` of the dev container image, the workspace
    mounted where it was built, `source install/setup.bash && exec ros2 launch
    <autoware.launcher.launch>`. `{bridge_address}` in the launch arguments
    becomes the bridge's address, which the launch has to hand to
    `scenario_bridge`. The stack's output goes to
    `autoware-<n>-<scenario>.log` in the run's output directory.
3. **While the stack starts** (until `scenario_bridge` first reaches the
    bridge), the world ticks at about real time and the wait is measured in
    wall-clock seconds, `autoware.boot_timeout_s`. Once it has reached the
    bridge, localizing, routing and engaging is measured in ticks,
    `autoware.ready_timeout_ticks`, as before.
4. **When the scenario ends**, however it ends, the stack is stopped (`SIGINT`
    to `ros2 launch`, through `docker run --init`) and its container removed
    once it has stopped or `stop_timeout_s` has passed, before the next
    scenario starts. A scenario that is retried (`server.cooldown_max_retries`)
    starts another fresh stack, with nothing of the failed attempt's carried
    over.

Every scenario of a batch uses the same launcher: set `autoware.launcher` on
the command line, not in a scenario's config (a batch whose scenarios disagree
on it is refused).

A stack that exits during a scenario ends it: the result's message says
`The Autoware stack exited with code <n>`.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `workspace` | `~/autoware` | The `autowarefoundation/autoware` clone |
| `devcontainer` | `universe-devel-cuda` | The `.devcontainer/` variant whose image and mount are used |
| `image` | `null` | An image to use instead of the dev container's |
| `workspace_mount` | `null` | Where to mount the workspace instead of where the dev container does |
| `build` | `auto` | `auto` / `always` / `never` |
| `colcon_args` | `[--symlink-install, --cmake-args, -DCMAKE_BUILD_TYPE=Release]` | Arguments to `colcon build` |
| `launch` | (required) | `ros2 launch` package, file and `name:=value` arguments |
| `mounts` | `{~/autoware_data: /home/aw/autoware_data}` | Further mounts; a missing host path is skipped |
| `env` | `{}` | Variables set in the container, over the dev container's own |
| `ros_domain_ids` | `[]` | `ROS_DOMAIN_ID` per scenario, cycling; empty keeps the dev container's |
| `gpus` | `all` | Docker's `--gpus`; `null` for none |
| `user` | `null` | Docker's `--user`; by default the image maps `aw` to your `HOST_UID`/`HOST_GID` |
| `docker_args` | `[]` | Further `docker run` arguments |
| `stop_timeout_s` | `20.0` | How long stopping waits after each signal |

`ros_domain_ids` moves each scenario's stack to another ROS domain, so it does
not discover what is left of the last one before those participants time out:
`'autoware.launcher.ros_domain_ids=[10,11]'`.

## Sweeps

A sweep (`--multirun hydra/sweeper=lanelet_constraint`) runs each case in a
job of its own, and each job starts and removes its own Autoware the same way:

```bash
uv run scenario --multirun hydra/sweeper=lanelet_constraint \
  scenario=intersection_passing/right_turn_sweep map=town10hd_opt \
  ego.entity=autoware autoware.launcher.type=docker \
  autoware.launcher.workspace=$HOME/autoware \
  'autoware.launcher.launch=[...]' \
  +sweep.job_timeout_seconds=900
```

`sweep.job_timeout_seconds` (120 s by default; `+` adds it to a sweep YAML
that does not set it) is a whole job's budget --
starting Autoware, localizing, routing, engaging and the scenario itself --
so raise it: launching Autoware alone can take most of two minutes. A job the
sweeper kills at that limit leaves its container behind, and the next job's
preparation removes it before its own stack starts.

## Autoware on the host

`autoware.launcher.type=command` runs a command of your own instead, as a
process group stopped with `SIGINT`, then `SIGTERM`, then `SIGKILL`. It is
sent `SIGTERM` if the process that started it dies -- a sweeper job killed at
its timeout -- so it does not outlive the job. `{bridge_address}`, `{index}` and `{name}`
are filled in, and the address is in `AUTOWARE_BRIDGE_ADDRESS` too:

```bash
uv run scenario ego.entity=autoware autoware.launcher.type=command \
  'autoware.launcher.command=[bash,-c,"source ~/autoware/install/setup.bash && exec ros2 launch autoware_launch e2e_simulator.launch.xml bridge_address:={bridge_address} ..."]'
```

## From Python

```python
from pathlib import Path

from autoware_carla_scenario import ScenarioQueue
from autoware_carla_scenario.autoware_stack import (
    DockerAutowareConfig,
    DockerAutowareLauncher,
)

launcher = DockerAutowareLauncher(
    DockerAutowareConfig(
        workspace=Path("~/autoware"),
        launch=(
            "autoware_launch",
            "e2e_simulator.launch.xml",
            "bridge_address:={bridge_address}",
        ),
    )
)
queue = ScenarioQueue(map_name="Town10HD_Opt", autoware_launcher=launcher)
# Each scenario's AutowareEgoEntity is given the same launcher:
#   AutowareEgoEntity(config, bridge=..., launcher=launcher)
```

The queue prepares the launcher when it starts and closes it when it stops;
`AutowareEgoEntity` starts and stops a stack per scenario. Any object with
`prepare` / `start` / `poll` / `stop` / `close`
(`autoware_carla_scenario.autoware_stack.AutowareLauncher`) can stand in for
the two launchers that ship.
