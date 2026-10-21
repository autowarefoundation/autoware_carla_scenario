# Usage Guide

This guide explains how to use the `autoware-carla-scenario` package to run autonomous driving scenario tests in CARLA simulator.

## CLI Commands Overview

The package provides these CLI commands:

| Command | Framework | Purpose |
|---------|-----------|---------|
| [`scenario`](#scenario-scenario-runner) | Hydra | Run autonomous driving scenario tests in CARLA |
| `scenario-new` | argparse | Scaffold a standalone scenario package |
| [`scenario-check`](typecheck.md) | Hydra compose | Compile scenarios with Codon without running them (the static check the runner makes before a run) |
| [`scenario-slurm`](slurm.md#following-collecting-resubmitting) | argparse | Follow, collect, resubmit or cancel a batch run as SLURM array jobs (`launcher=slurm`) |
| [`scenario-odd`](odd.md#checking-an-odd) | argparse | Check an ODD and show its buckets; `scenario-odd route` checks scenarios' planned routes against it on the Lanelet2 map |
| [`detect-no-3d-model`](#detect-no-3d-model-3d-model-detection) | argparse | Detect lanelets without a matching 3D ground model in CARLA |
| [`viewer`](#viewer-scenario-result-viewer) | FastAPI + Uvicorn | Web UI for browsing and monitoring scenario test results |
| [`scenario-editor`](scenario_editor.md) | FastAPI + Uvicorn | Web UI for authoring scenarios and exporting reproducible packages |

`viewer` and `scenario-editor` are separate applications on separate ports, and
neither starts the other.

---

## `scenario` - Scenario Runner

Runs autonomous driving scenario tests in CARLA. Uses [Hydra](https://hydra.cc/) for configuration management. Requires a running CARLA server.

### How It Works

1. Parses the `scenario=...` argument (supports glob patterns for batch execution)
2. Composes a Hydra config from map, server, ego, entity, and scenario configs
3. Compiles the scenario with Codon and refuses it if it does not type-check
   ([Static Check](typecheck.md); `typecheck=off` skips it)
4. Connects to the CARLA server and loads the specified map (OpenDRIVE or built-in)
5. Spawns the ego vehicle at the configured Lanelet2 position (converted to CARLA coordinates)
6. Enables synchronous mode (fixed 20 Hz tick rate) and starts the native CARLA recorder
7. Runs a warm-up phase (5 ticks for physics stabilization)
8. Enables autopilot and applies initial speeds
9. Executes the main tick loop, checking pass/fail conditions each tick
10. Writes the results as JSON, and the run's trajectory log as it goes
11. Reloads the CARLA world for clean state between scenarios

### Basic Usage

```bash
# Run a single scenario
uv run scenario scenario=intersection_passing/straight

# Run a left-turn variant
uv run scenario scenario=intersection_passing/left_turn

# Run traffic light compliance scenario
uv run scenario scenario=traffic_light_compliance/traffic_light_compliance

# Run a lane change scenario
uv run scenario scenario=lane_change/left

# Run a temporary stop scenario
uv run scenario scenario=temporary_stop/temporary_stop
```

### Batch Execution with Glob Patterns

When the `scenario=` value contains glob metacharacters (`*`, `?`, `[`), all matching scenario configs are executed sequentially in a single CARLA session:

```bash
# Run all intersection-passing variants
uv run scenario scenario='intersection_passing/*'

# Run all scenarios matching a partial pattern
uv run scenario scenario='intersection_passing/left_*'

# Run all lane-change scenarios
uv run scenario scenario='lane_change/*'

# Run everything
uv run scenario scenario='*/*'
```

**Batch constraints:**

- All scenarios in a batch **must share the same map** (CARLA loads one map per session)
- Results are written to `outputs/YYYY-MM-DD/HH-MM-SS/batch_results.json`
- A summary table is printed at the end with per-scenario pass/fail status
- A case that cannot run at all -- its run raises on every attempt
  (`server.cooldown_max_retries`), e.g. a spawn point with no ground under it --
  does not end the batch: it is reported as `[ERROR]` with the error, its
  `batch_results.json` entry is `"passed": false` with `"error": true`, and the
  batch goes on with the next case. Ctrl+C still stops the batch
- A scenario with a `sweep:` section runs once, as configured; add `--multirun`
  to run every case of it (see [Running every sweep case in a batch](#running-every-sweep-case-in-a-batch))

#### Running every sweep case in a batch

With `--multirun`, a batch expands each scenario's `sweep:` -- lanelet
constraints, a route search, the [ODD sampler](odd.md#sampling-scenarios-from-the-odd) --
into its cases, the same ones `scenario-expand` prints, and runs them all in
the one queue, against one CARLA session (and, with `autoware.launcher`, a
fresh Autoware per case):

```bash
# Every lanelet case of both cut-in scenarios
uv run scenario --multirun 'scenario=cut_in/*' map=town10hd_opt

# 10 cases per scenario drawn from the ODD
uv run scenario --multirun 'scenario=cut_in/*' map=town10hd_opt \
  +sweep.odd_sample.count=10 +sweep.odd_sample.strategy=coverage
```

Scenarios that draw from one ODD go on from each other's draws rather than
each draw the same cases: with `strategy: coverage` a later scenario aims at
what the earlier ones left uncovered (see
[Sampling scenarios from the ODD](odd.md#sampling-scenarios-from-the-odd)).

A scenario with more than one case is listed as `name#k` in the summary, and
`batch_results.json` gives each case its `overrides`. `hydra/sweeper=...` may
be given but is not needed: the batch is the sweeper.

On a SLURM cluster, `launcher=slurm` runs a batch's cases as array jobs, several
at once, each on a CARLA of its own, with the same outputs (off unless
selected; see [Running a Batch on SLURM](slurm.md)):

```bash
uv run scenario --multirun 'scenario=cut_in/*' +sweep.odd_sample.count=24 \
  launcher=slurm launcher.array_parallelism=2
```

### Selecting a Published HD Map

A map group can name a git repository instead of local files, and the map is
fetched into `~/autoware_data/maps` on first use -- the same directory
Autoware's own setup downloads maps into, so a machine that has run it clones
nothing:

```bash
# Town10HD_Opt, as published for CARLA 0.10, pinned to a commit. The built-in
# scenario configs name Nishishinjuku lanelets, so place the ego on Town10's.
uv run scenario scenario=lane_change/left map=town10hd_opt \
  ego.spawn_lanelet_id=324 ego.spawn_s=10.0 ego.goal_lanelet_id=5650

# Any repository, any revision
uv run scenario map=town10hd_opt \
  map.source='git+https://huggingface.co/datasets/OWNER/NAME@REF#maps/T'

# A map directory on this machine, read in place
uv run scenario map=town10hd_opt map.directory=/path/to/autoware_maps/Town10HD_Opt
```

See [HD Maps from a Git Repository](maps.md).

### Lanelet Constraint Sweeping

The package ships a Hydra `Sweeper` plugin
(`hydra/sweeper=lanelet_constraint`) that expands a single scenario
config into one job per lanelet that matches a set of constraints,
optionally deriving per-job parameters via "bindings". Each scenario
YAML may declare its own `sweep:` section (see
`traffic_light_compliance.yaml` for an example).

```bash
# Run a Hydra multirun over every lanelet whose `ego.spawn_lanelet_id`
# satisfies the scenario's `sweep.constraints`.
uv run scenario --multirun scenario=traffic_light_compliance/traffic_light_compliance \
  hydra/sweeper=lanelet_constraint

# Resume a sweep from job number N (1-indexed; 0 disables).
uv run scenario --multirun scenario=traffic_light_compliance/traffic_light_compliance \
  hydra/sweeper=lanelet_constraint sweep.resume_from=10
```

Sweeps write results under `multirun/YYYY-MM-DD/HH-MM-SS/<job-index>/`,
not `outputs/`. The result viewer scans both directories.

To get the concrete scenarios without running them (for example, to fan them
out over a cluster), use `scenario-expand`. It takes the same arguments as
`scenario` and prints JSON with one list of Hydra overrides per concrete
scenario. These are the same cases, in the same order, that the sweeper runs:

```bash
uv run scenario-expand scenario=traffic_light_compliance/traffic_light_compliance
# {"scenario": "...", "cases": [["ego.spawn_lanelet_id=5", "ego.spawn_s=18.62..."], ...]}
```

A scenario without a `sweep:` section is already concrete, so it expands to a
single empty case. Only the Lanelet2 map is needed, not a CARLA server.

A *logical* scenario -- a document with a route search, whose `sweep:` holds
`route:` instead of `constraints:` -- expands to one case per route of the map
that matches its pattern, each with the ego's spawn and goal and the match
itself (`scenario.route.*`); see
[Logical Scenarios from Routes](logical_scenarios.md).

### Resume from a Specific Scenario

When running large batches, you can skip already-completed scenarios.
`--resume-from N` starts at the *N*th scenario (1-indexed, as for a sweep);
in a `--multirun` batch it counts the expanded cases:

```bash
# Resume from the 3rd scenario
uv run scenario scenario='intersection_passing/*' --resume-from 3
```

### Overriding Configuration Parameters

Any nested Hydra parameter can be overridden from the command line:

```bash
# Select a different map
uv run scenario scenario=intersection_passing/straight map=nishishinjuku

# Connect to a remote CARLA server
uv run scenario scenario=intersection_passing/straight server.host=192.168.1.100 server.port=3000

# Change ego vehicle type and speed
uv run scenario scenario=intersection_passing/straight \
  ego.vehicle_type=vehicle.tesla.model3 \
  ego.initial_speed_kmh=30.0

# Override scenario-specific parameters
uv run scenario scenario=intersection_passing/straight scenario.timeout_seconds=15.0

# Change Traffic Manager port
uv run scenario scenario=intersection_passing/straight traffic_manager.port=8200
```

### Hydra Configuration System

Configuration files are located in `src/autoware_carla_scenario/examples/conf/`:

```
conf/
├── config.yaml              # Root config (defines defaults and top-level overrides)
├── server/
│   └── localhost.yaml       # Server connection settings
├── ego/
│   └── default.yaml         # Ego vehicle settings
├── entity/
│   └── default.yaml         # Entity spawn and ground projection settings
├── traffic_manager/
│   └── default.yaml         # CARLA Traffic Manager settings
├── map/
│   └── nishishinjuku.yaml   # Map file paths and excluded lanelet IDs
└── scenario/
    ├── intersection_passing/
    │   ├── straight.yaml
    │   ├── left_turn.yaml
    │   ├── right_turn.yaml
    │   └── straight_3npc.yaml
    ├── lane_change/
    │   ├── left.yaml
    │   └── right.yaml
    ├── lane_change_fail/
    │   ├── left.yaml
    │   └── right.yaml
    ├── temporary_stop/
    │   └── temporary_stop.yaml
    └── traffic_light_compliance/
        └── traffic_light_compliance.yaml
```

#### Root Configuration (`config.yaml`)

Defines the default config group composition:

| Default Group | Default Value | Description |
|---------------|---------------|-------------|
| `server` | `localhost` | CARLA server connection |
| `ego` | `default` | Ego vehicle configuration |
| `entity` | `default` | Entity spawn and ground projection |
| `traffic_manager` | `default` | CARLA Traffic Manager |
| `scenario` | `intersection_passing/left_turn` | Scenario to run |
| `map` | `nishishinjuku` | Map to load |

#### Server Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server.host` | str | `localhost` | CARLA server hostname |
| `server.port` | int | `2000` | CARLA server port |
| `server.cooldown_seconds` | float | `2.0` | Wait time between consecutive scenario runs |
| `server.cooldown_max_retries` | int | `0` | Maximum retries on failure |

#### Ego Vehicle Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ego.vehicle_type` | str | `vehicle.mini.cooper` | CARLA blueprint ID for ego vehicle |
| `ego.initial_speed_kmh` | float | `0.0` | Initial speed of ego vehicle (km/h) |
| `ego.spawn_lanelet_id` | int | `242` | Lanelet ID for spawn position |
| `ego.spawn_s` | float | `25.0` | Longitudinal offset along lanelet centerline (meters) |

#### Entity Configuration

Controls ground projection (ray casting) and spawn retry behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity.ground_projection_ray_distance_upper` | float | `5.0` | Ray search range above estimated z (meters) |
| `entity.ground_projection_ray_distance_lower` | float | `10.0` | Ray search range below estimated z (meters) |
| `entity.spawn_retry_max_count` | int | `10` | Maximum retry attempts for spawn |
| `entity.spawn_retry_t_step` | float | `0.1` | Lateral offset shift per retry (meters) |
| `entity.spawn_retry_z_step` | float | `0.5` | Vertical shift per retry (meters) |

#### Traffic Manager Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traffic_manager.port` | int | `8100` | CARLA Traffic Manager RPC port |

#### Seeing a run

A run renders no video (since 15.0.0). What it leaves to look at is its
trajectory log, `{ScenarioName}.trajectory.jsonl`: every vehicle and pedestrian
tick by tick, the traffic lights, the map it ran on, the ego's plan (its debug
markers), each condition's state and the verdict ([Trajectory Log](trajectory_log.md)).
To see it, make it a [Rerun](https://rerun.io) recording, `{ScenarioName}.rrd`
beside the log, and open that in the Rerun viewer. The conversion is the
`autoware-carla-scenario-rerun` package, installed with the `rerun` extra:

```bash
# The converter (rerun), and the Rerun viewer its recordings are made for (viewer):
pip install 'autoware-carla-scenario[rerun,viewer]'
# A recording of every run, made as each run ends:
uv run scenario scenario=cut_in/cut_in rerun.enabled=true
uv run scenario 'scenario=cut_in/*' --multirun rerun.enabled=true
rerun outputs/2026-10-18/08-48-12/CutInScenario-3.rrd     # the Rerun viewer
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rerun.enabled` | bool | `false` | Convert each run's trajectory log into `{ScenarioName}.rrd` as soon as the run ends |
| `rerun.strict` | bool | `false` | Fail the conversion of a log with a record type or a field the converter does not know, rather than skip it with a warning |
| `rerun.lidar` | bool | `false` | Put the LiDAR sweeps the ego's driver policy gets into the recording too ([LiDAR sweeps](#lidar-sweeps-in-the-recording)) |
| `rerun.lidar_max_points` | int | `65536` | At most this many points of a sweep (an even subsample); `0` keeps them all |

With `rerun.enabled=true`:

- A run asking for recordings without the extra installed is refused before
  anything starts (exit code 2), saying how to install it.
- Each log is converted on a background thread as soon as its run ends, while
  the next case runs (the converter releases the GIL; a log takes milliseconds).
  A run that raised is converted too: its log is what shows what happened.
- At the end the run waits for the last conversions and prints how they went:
  `Rerun recordings: N/M written`, and a line per failure naming its case. A
  failed conversion does not change any verdict or the exit code.
- Each case's `{ScenarioName}_result.json` names its recording
  (`rerun_recording`), and so does its entry in a batch's
  `batch_results.json`, with `rerun_error` when the conversion failed. The
  viewer's scenario page shows it in the embedded Rerun viewer, without
  converting the log again ([Seeing a run in the viewer](#seeing-a-run-in-the-viewer)).
- A Hydra sweep (`--multirun` without a glob) runs each case in a process of
  its own, and each converts its own log. So does each task of a batch run on
  SLURM (`launcher=slurm`, [Running a Batch on SLURM](slurm.md)): it converts
  its own cases, and `scenario-slurm collect` reports their recordings as a
  local batch does.

### LiDAR sweeps in the recording

The trajectory log holds no point clouds. With `rerun.lidar=true` (and
`rerun.enabled=true`) the sweeps each driver LiDAR sends the ego's policy
(`driver.lidars`) are written during the run, in the world frame of the
recording, into `{ScenarioName}.lidar.rrd`; once the log is converted they are
merged into `{ScenarioName}.rrd` (`rerun rrd merge`) and that file is removed.
Each LiDAR is `world/lidar/<logical id>`, points coloured by height above the
ego, in the recording's 3D view; a sweep is drawn where the ego's pose put it
when the policy got it. It needs the Rerun SDK of the converter's version:

```bash
pip install 'autoware-carla-scenario[rerun-lidar]'
uv run scenario scenario=cut_in/cut_in driver=... rerun.enabled=true rerun.lidar=true
```

Points are most of a recording: about 0.8 MB a sweep at the default 65,536.
A merge that fails keeps `{ScenarioName}.lidar.rrd` beside the recording
(logged; not a failed conversion). The two cannot be opened together instead:
the viewer registers a file by its recording id, and the second replaces the
first.

A run made without it (or by a version before it) can be converted afterwards,
the same way:

```bash
# Every run of a session (a sweep's directory works the same), or one log:
python -m autoware_carla_scenario.rerun outputs/2026-10-18/08-48-12
python -m autoware_carla_scenario.rerun outputs/2026-10-18/08-48-12/CutInScenario-3.trajectory.jsonl
```

or from Python, `autoware_carla_scenario.rerun.convert_outputs(path)`. The
converter reads nothing but the log and the map files the log names, so it
runs anywhere, without CARLA. The CARLA recording (`{ScenarioName}.log`) is
still written: CARLA's own replayer (`client.replay_file()`) plays it back on a
server running the same map.

`video.*` settings (`video.enabled`, `video.width`, `video.height`) are
accepted for old command lines and configs, and do nothing: a run that sets
one warns that it is ignored and says the above.

#### Map Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `map.name` | str | **required** | CARLA map name (e.g., `NishishinjukuMap`) |
| `map.source` | str \| null | `null` | Git repository holding the HD map (see [maps.md](maps.md)) |
| `map.directory` | str \| null | `null` | Local map directory, read in place. Wins over `map.source` |
| `map.xodr_path` | str \| null | `null` | Path to custom OpenDRIVE file. If `null`, loads built-in CARLA map by name |
| `map.lanelet2_path` | str \| null | `null` | Path to Lanelet2 .osm file (required for lanelet-based spawn) |
| `map.no_3d_model_lanelet_ids` | list[int] | `[]` | Lanelet IDs to exclude from sweeps (no ground geometry) |

Map paths can use environment variables with Hydra's OmegaConf resolver:

```yaml
xodr_path: ${oc.env:NISHISHINJUKU_XODR_PATH,path/to/default.xodr}
```

### Supported Scenarios

#### Intersection Passing

Tests ego vehicle traversal through an intersection. Verifies that the ego passes through the expected OpenDRIVE road sequence.

| Config | Turn Direction | Description |
|--------|---------------|-------------|
| `intersection_passing/straight` | none | Drive straight through intersection |
| `intersection_passing/left_turn` | left | Left turn at intersection |
| `intersection_passing/right_turn` | right | Right turn at intersection |
| `intersection_passing/straight_3npc` | none | Straight with 3 NPC vehicles |

**Scenario-specific parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario.expected_route_lanelet_ids` | list[int] | varies | Lanelet IDs the ego must traverse |
| `scenario.timeout_seconds` | float | `5.0` | Maximum scenario duration |
| `scenario.min_speed_kmh` | float \| null | null | Minimum ego speed (fail if below). `null` disables check |
| `scenario.speed_check_delay_seconds` | float | `0.3` | Grace period before speed check starts |
| `scenario.turn_direction` | str \| null | null | `"left"`, `"right"`, or `null` for straight |
| `scenario.npc_vehicles` | list | `[]` | NPC vehicle configurations (see below) |

**NPC vehicle configuration:**

```yaml
npc_vehicles:
  - spawn_lanelet_id: 100
    spawn_s: 5.0
    vehicle_type: vehicle.mini.cooper
    initial_speed_kmh: 10.0
```

**Behavior:**
- Sets all traffic lights to GREEN
- Registers a TurnAction for left/right turns (applies steering via CARLA autopilot route)
- Pass condition: ego traverses all expected OpenDRIVE road IDs (checked via StickyCondition + AndCondition)
- Fail conditions: timeout, ego destroyed, speed below threshold

#### Lane Change

Tests ego vehicle lane change maneuvers. Can test both successful lane changes and expected failures.

| Config | Direction | Expected Result |
|--------|-----------|-----------------|
| `lane_change/left` | left | success |
| `lane_change/right` | right | success |
| `lane_change_fail/left` | left | fail (no adjacent lane) |
| `lane_change_fail/right` | right | fail (no adjacent lane) |

**Scenario-specific parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario.direction` | str | `"left"` | Lane change direction: `"left"` or `"right"` |
| `scenario.expect` | str | `"success"` | Expected outcome: `"success"` or `"fail"` |
| `scenario.timeout_seconds` | float | `10.0` | Maximum scenario duration |

**Behavior:**
- Sets all traffic lights to GREEN
- Immediately registers a LaneChangeAction
- Derives the target lane from the current spawn position and direction
- Pass condition: `expect=success` requires both `road_id` and `lane_id` to change; `expect=fail` passes when timeout is reached without a lane change
- Checks via EntityLanePositionCondition

#### Traffic Light Compliance

Tests that the ego vehicle correctly stops at a red light and proceeds after green.

| Config |
|--------|
| `traffic_light_compliance/traffic_light_compliance` |

**Scenario-specific parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario.light_switch_delay_seconds` | float | `5.0` | Delay before switching red to green |
| `scenario.timeout_seconds` | float | `8.0` | Maximum scenario duration |
| `scenario.stop_zone_min_m` | float | `0.0` | Near end of the stop zone: metres before the stop line (`stop_line_distance_m`) |
| `scenario.stop_zone_max_m` | float | `3.0` | Far end of the stop zone; standing still farther back until the green fails the run |
| `scenario.stop_line_tolerance_m` | float | `0.0` | How far the front bumper may pass the stop line on red before the run fails for running the red; that far past counts as at the line for the stop zone |
| `scenario.standstill_seconds` | float | `0.5` | How long the ego has to stand still in the stop zone, before the green, for it to count as stopped there (latched once it has) |
| `scenario.standstill_speed_mps` | float | `0.1` | Speed (m/s) at or below which the ego stands still |
| `scenario.departure_speed_kmh` | float | `10.0` | Speed (km/h) the ego must reach after green to pass; `null` uses `moving_speed_kmh` |
| `scenario.moving_speed_kmh` | float | `1.0` | The departure speed when `departure_speed_kmh` is `null`; no longer judges the red phase |
| `scenario.merging_time_seconds` | float | `0.5` | Deprecated and unused; kept so configs that set it still load |

**Behavior:**
- Sets all traffic lights to RED initially
- Registers a TrafficSignalAction to switch to GREEN after `light_switch_delay_seconds`
- Fails as soon as the ego's front bumper crosses the stop line while the light is red
- At the green, fails unless the ego has stood still with its front bumper in the stop zone (`stop_zone_min_m`-`stop_zone_max_m` before the stop line), saying whether it stopped short of the zone, stopped in it too late, or had not stopped
- Pass conditions: ego stood still in the stop zone before the green AND reaches `departure_speed_kmh` during green

#### Temporary Stop

Tests that the ego vehicle stops at a designated stop line for a minimum duration, then restarts.

| Config |
|--------|
| `temporary_stop/temporary_stop` |

**Scenario-specific parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario.s_margin` | float | `5.0` | Arc-length margin around stop position (meters) |
| `scenario.speed_threshold` | float | `0.1` | Max speed considered as "stopped" (m/s) |
| `scenario.stop_duration` | float | `0.3` | Minimum consecutive seconds to remain stopped |
| `scenario.restart_speed_kmh` | float | `3.0` | Speed threshold for restart detection (km/h) |
| `scenario.timeout_seconds` | float | `5.0` | Maximum scenario duration |

**Behavior:**
- Sets all traffic lights to GREEN
- Registers a TemporaryStopCondition at the configured stop position
- Pass condition: ego stays within `s_margin` of stop line at speed below `speed_threshold` for at least `stop_duration` seconds, then accelerates above `restart_speed_kmh`

### Scenario Evaluation

Each scenario registers **pass conditions** and **fail conditions**. The tick loop evaluates them every frame:

- **First pass condition satisfied** → scenario passes
- **First fail condition triggered** → scenario fails
- Common fail conditions: `TimeoutCondition` (exceeded time limit), `EntityExistenceCondition` (ego destroyed)

### ODD and Coverage

Every run is measured against an ODD, named by the `odd` key (`default` unless
overridden). Its attributes are sampled on every tick and written, with the
scenario's own cover items, to `{ScenarioName}_coverage.json`.
`scenario-coverage outputs/` merges those files into a report. See
[ODD](odd.md) and [Coverage](coverage.md).

```bash
uv run scenario scenario=intersection_passing/left_turn odd=path/to/urban.yaml
uv run scenario-coverage outputs/ --markdown coverage.md
```

Before running, `scenario-odd route` plans each scenario's ego route on the
Lanelet2 map (no CARLA server) and reports which parts of it are outside the
ODD, which are undecided until run time, the metres it is expected to cover
in each bucket, and the buckets no route reaches
([Checking a planned route](odd.md#checking-a-planned-route)):

```bash
uv run scenario-odd route path/to/urban.yaml 'intersection_passing/*' lane_change/left
```

### Output Files

Each scenario run generates output in `outputs/YYYY-MM-DD/HH-MM-SS/`:

| File | Description |
|------|-------------|
| `{ScenarioName}.log` | CARLA native recording (replay format) |
| `{ScenarioName}_result.json` | Machine-readable result with condition statuses |
| `{ScenarioName}_coverage.json` | ODD and scenario coverage hits of the run (merge with `scenario-coverage`, see [Coverage](coverage.md)) |
| `{ScenarioName}_odd.tar.gz` | The run's ODD as the files it was read from (an ODD read from OpenODD YAML; openodd's transmission archive, named by its revision; see [Versions of an ODD](odd.md#versions-of-an-odd)) |
| `{ScenarioName}_requirements.lock` | What the run ran with: every installed distribution pinned, in the requirements format (`pip install -r`), headed by the framework, the scenario's package, Python and the platform. A distribution from git is pinned to its commit; an editable one (a checkout) names its directory, commit and whether it had uncommitted changes |
| `{ScenarioName}_environment.json` | The same, structured; `_result.json` carries its summary (`environment`: the framework, the scenario's package, key dependencies, and the lock's id -- the SHA-256 of what it pins -- so runs on one environment can be found by it) |
| `{ScenarioName}.trajectory.jsonl` | The run, tick by tick, and the map, plan, conditions and verdict ([Trajectory Log](trajectory_log.md)); make it a Rerun recording to see the run ([Seeing a run](#seeing-a-run)) |
| `{ScenarioName}.rrd` | The Rerun recording of the trajectory log, with `rerun.enabled=true` ([Seeing a run](#seeing-a-run)) |
| `batch_results.json` | Batch summary (glob/batch mode only) |

**Result JSON format:**

```json
{
  "scenario": "intersection_passing/straight",
  "passed": true,
  "message": "All pass conditions satisfied",
  "elapsed_seconds": 3.42,
  "condition_statuses": [
    {"label": "Road 5 visited", "satisfied": true, "message": "..."},
    {"label": "Road 12 visited", "satisfied": true, "message": "..."}
  ]
}
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All scenarios passed |
| 1 | One or more scenarios failed or errored, or configuration/runtime error |

---

## Driving the Ego with an External Policy

By default CARLA's TrafficManager drives the ego. To hand it to an external driving
policy over the `egodriver` gRPC contract instead, start the policy server and select
the `carla_driver` entity:

```bash
# 1. Start the policy (in its own environment)
uv run carla-driver-interface serve --policy route_follower --port 50051

# 2. Run any scenario against it
uv run scenario ego.entity=carla_driver driver.address=localhost:50051
```

`ego.entity` accepts:

| Value | Behaviour |
| --- | --- |
| `autopilot` (default) | CARLA's TrafficManager drives the ego |
| `autoware` | Autoware drives the ego, over the `AutowareBridge`; with `autoware.launcher` the framework starts a fresh Autoware per scenario ([Autoware as the Ego](autoware.md)) |
| `carla_driver` | An external policy drives the ego |
| `ros2` | ROS 2 nodes drive the ego over DDS, with no bridge node ([ROS 2 over DDS](ros2.md)) |

Common overrides:

```bash
# Query the policy every simulation tick instead of every other one
uv run scenario ego.entity=carla_driver driver.policy_timestep_s=0.05

# Pin the rig origin offset when the derived value looks wrong (CARLA 0.9.x)
uv run scenario ego.entity=carla_driver driver.rear_axle_offset_m=-1.4

# Send a smaller, cheaper camera frame
uv run scenario ego.entity=carla_driver \
    driver.cameras.0.image_width=480 driver.cameras.0.image_height=302
```

See [External Driver Interface](driver_interface.md) for the full reference.

### Rendering sensors with carla_canvas

[carla_canvas](https://github.com/hakuturu583/carla_canvas) renders cameras and
LiDARs behind the CARLA API instead of the CARLA server: a configuration file
names a renderer service (3D Gaussian splatting, a LiDAR world model such as
[LidarVerse](https://github.com/hakuturu583/lidarverse), ...) and the sensor
types it renders, and typesafe_carla's sensor plugin hands those sensors'
measurements over from the service. `canvas.config` points a run at such a
file; nothing else about the run changes -- the scenario, the driver and the
measurements they read are the same:

```bash
pip install 'carla-canvas[client]>=0.2'
uv run scenario scenario=cut_in/cut_in ego.entity=carla_driver \
    canvas.config=path/to/lidarverse.yaml
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `canvas.config` | str | `null` | A carla_canvas configuration: its sensors are rendered by its renderer; `null` leaves every sensor to CARLA |

The run sets `CARLA_CANVAS_CONFIG` and `TYPESAFE_CARLA_SENSOR_PLUGIN` before
any sensor listens, which is when typesafe_carla loads the plugin, once for
the process: the cases of a batch name one file (a batch naming two is
refused). The renderer is the configuration's: started by the plugin when
nothing answers at `renderer.endpoint` and the file gives a
`renderer.command`, or run by hand (`carla-canvas serve --config ...`).
carla_canvas is not an extra of this package: it needs protobuf 7, and the
vendored alpasim modules are regenerated on protobuf 4. Under SLURM the file is
the nodes' path.

---

## `detect-no-3d-model` - 3D Model Detection

Detects lanelets that do not have a matching 3D ground model in CARLA. Useful for identifying areas where vehicle ground-projection ray casting will fail. Requires a running CARLA server with the target map loaded.

### How It Works

1. Reads the map configuration YAML to resolve `xodr_path` and `lanelet2_path`
2. Connects to the CARLA server
3. For each lanelet's center point, casts a vertical ray (upward and downward from estimated z) to detect ground geometry
4. Reports lanelets where the ray does not hit any 3D surface
5. Writes the detected IDs to the YAML configuration's `no_3d_model_lanelet_ids` field

### Usage

```bash
# Basic detection using map config YAML
uv run detect-no-3d-model conf/map/nishishinjuku.yaml

# Connect to a remote CARLA server
uv run detect-no-3d-model conf/map/nishishinjuku.yaml --host 192.168.1.100 --port 3000

# Increase ray search range for hilly terrain
uv run detect-no-3d-model conf/map/nishishinjuku.yaml --ray-upper 10.0 --ray-lower 10.0

# Longer timeout for slow connections
uv run detect-no-3d-model conf/map/nishishinjuku.yaml --timeout 30.0

# Debug logging
uv run detect-no-3d-model conf/map/nishishinjuku.yaml -v
```

### Command-Line Options

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `yaml_path` | - | Yes | - | Path to map configuration YAML (must contain `map.name`, `xodr_path`, `lanelet2_path`) |
| `--host` | - | No | `localhost` | CARLA server hostname |
| `--port` | - | No | `2000` | CARLA server port |
| `--ray-upper` | - | No | `5.0` | Ray search range above z estimate (meters) |
| `--ray-lower` | - | No | `5.0` | Ray search range below z estimate (meters) |
| `--timeout` | - | No | `10.0` | CARLA client timeout (seconds) |
| `--verbose` | `-v` | No | false | Enable DEBUG logging |

### Input YAML Requirements

The YAML file must contain at minimum:

```yaml
name: NishishinjukuMap        # CARLA map name
xodr_path: path/to/map.xodr  # Path to OpenDRIVE file
lanelet2_path: path/to/map.osm  # Path to Lanelet2 file
```

### Integration with Scenario Runner

The detected lanelet IDs are stored in the map config's `no_3d_model_lanelet_ids` field. The scenario runner's sweep feature uses this list to exclude problematic lanelets from automatic spawn point selection.

---

## `viewer` - Scenario Result Viewer

Web UI for browsing and monitoring scenario test results. Starts a FastAPI server with Uvicorn.

### How It Works

1. Scans the base path directory for scenario output files (`outputs/YYYY-MM-DD/HH-MM-SS/`)
2. Serves a web interface for browsing sessions and viewing individual scenario results, each with its run in an embedded Rerun viewer ([Seeing a run in the viewer](#seeing-a-run-in-the-viewer))
3. Provides an API for triggering scenario runs and monitoring progress in real time

### Usage

```bash
# Start with default settings (0.0.0.0:9000, current directory)
uv run viewer

# Custom host and port
VIEWER_HOST=127.0.0.1 VIEWER_PORT=8080 uv run viewer

# Specify results directory
VIEWER_BASE_PATH=/path/to/scenario/outputs uv run viewer
```

Then open `http://localhost:9000` (or the configured host/port) in a browser.

### Seeing a run in the viewer

A scenario's page shows its run in Rerun's web viewer, embedded in the page:
the case's trajectory log made into a Rerun recording ([Seeing a run](#seeing-a-run)).
A recording already beside the log (`X.rrd` beside `X.trajectory.jsonl`) is
shown as it is; otherwise the viewer makes it the first time the page is
opened (in a worker thread, some tens of milliseconds for a case) and keeps it
beside the log, or, where the session's directory cannot be written, in its
cache directory. Making one needs the converter in the viewer's environment:

```bash
pip install 'autoware-carla-scenario[rerun]'    # or: pip install autoware-carla-scenario-rerun
```

Without it the page says so, and still shows a recording already made.
The page's *Download .rrd* link is the recording itself, for the native
viewer (`rerun X.rrd`).

The web viewer is the `@rerun-io/web-viewer` package of the version the
converter writes recordings with (`RERUN_VERSION`, 0.38.1 for converter 0.1;
a recording opens only in a viewer of its minor version). By default the page
loads it from the unpkg CDN, pinned to that version. For a machine without
the network, keep a copy that the viewer serves itself:

```bash
uv run viewer --fetch-rerun-web-viewer                 # from the npm registry, checked against its integrity hash
uv run viewer --fetch-rerun-web-viewer --from web-viewer-0.38.1.tgz   # from the package's tarball, copied in
```

The copy goes to `~/.cache/autoware_carla_scenario/viewer/rerun-web-viewer/<version>/`
(`VIEWER_CACHE_DIR` moves the cache) and is used from then on.
`VIEWER_RERUN_WEB_VIEWER` overrides where the viewer comes from: an unpacked
copy of the package (a directory), a mirror of it (a URL), or `none` for no
embedded viewer (the page then names the recording only).

### Results on the ODD

A session's page links its results on the ODD
(`/session/{type}/{date}/{time}/odd`): openodd's coverage matrix
(`openodd.CoverageView`) of the cases' coverage files (`{stem}_coverage.json`),
along two ODD attributes picked above it (by default the two the cases were
drawn over that tell them apart most). Each cell holds the runs that spent
time in it, coloured by what they did -- green passed, red failed, yellow
flaky (both), white not sampled -- and a click on one drills into it: the
matrix of the next two attributes inside it, and below, the cases in it, each
linking to its page (and its run in the Rerun viewer) with its verdict, the
buckets it was drawn in and a few metrics of how the ego drove, read off its
trajectory log: the least gap from its box to another road user's
(`min gap`), its hardest braking (over half a second), when it first stopped,
how far it drove, its top and final speed, and the elapsed time. openodd's
plot of the drawn cases (`openodd.CasePlot`) is shown too.

### Comparing runs

`/compare` (the navigation bar's *Compare*, or *Compare with past runs* on a
session's page) compares two or more sessions of the same scenarios
(`?s=batch/2026-10-18/16-09-52&s=...`), the oldest the baseline:

- **Cases** are matched by what they ran -- their sweep overrides (less
  `++odd_case`, which names a case's position), else the ODD values they
  were drawn at -- not by their index, so a case keeps its match when a
  sweep's order or size changes. Each is marked by how its verdict changed
  between the first session that ran it and the last: `pass → fail` (a
  regression, listed first and in red), `fail → pass`, flaky, the same, or
  run by one session only. Beside it, the change of each metric (bold when
  more than 5%), each run's verdict linking to its page, and *side by side*:
  the first and last runs in two Rerun viewers on one page
  (`/compare/runs?run=type/date/time/index&run=...`).
- **ODD cells**: each session's passes and runs in each cell of the two
  attributes picked (the buckets the cases were drawn in), red where the last
  session passes fewer of them, green where more; and openodd's matrix of
  each session beside the others.
- **Pass rates** of each session.
### Running through SLURM

The *Run Scenarios* panel's **Launcher** runs the batch here, case after case
(`local`, the default), or submits it as a SLURM array job (`SLURM`,
[Running a Batch on SLURM](slurm.md)) with the `launcher.*` keys the panel
sets: the partition (from `sinfo`), `array_parallelism`, `cases_per_task`,
`time`, `mem`, `gres`, `start_stagger_seconds`, extra environment
variables (`launcher.env`), and for a viewer whose host sees the nodes' file
system at another path (an sshfs or NFS mount) `remote_cwd`, `python` and the
`setup` lines a job runs first ([A submit host with other
paths](slurm.md#a-submit-host-with-other-paths)); the extra overrides can set
any other. The
viewer runs `scenario [--multirun] scenario=... launcher=slurm ... --no-wait`
(with a sweeper, every case of the sweep) in its base path, and opens the
batch's page once it is submitted. Where `sbatch` is not on the viewer's
`PATH` the option is disabled, and says so.

A SLURM batch's page (`outputs/<date>/<time>/` with a `slurm/` directory,
whoever submitted it) follows it through `squeue` and the records its tasks
write: each case passed, failed, errored, running, pending or missing (with
why: its task's state and error), each submission's job and tasks (state,
host, cases done, peak memory), and the cases as they land, before the batch
has a summary. Its buttons run `scenario-slurm` on the batch: *Resubmit
missing* (the cases with no outcome, or that errored), *Resubmit failed too*
(`--failed`), with the array parallelism and cases per task to use, *Collect*
(the summary and `batch_results.json`) and *Cancel jobs*.

`VIEWER_SCENARIO_COMMAND` replaces `uv run scenario` as the command a SLURM
batch is submitted with (`.venv/bin/scenario`, say, where `uv run` must not
sync the environment).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIEWER_BASE_PATH` | Current working directory | Base path for scanning scenario result files |
| `VIEWER_HOST` | `0.0.0.0` | Server bind address |
| `VIEWER_PORT` | `9000` | Server listen port |
| `VIEWER_CACHE_DIR` | `~/.cache/autoware_carla_scenario/viewer` | The web viewer's copy, and recordings of sessions that cannot be written |
| `VIEWER_RERUN_WEB_VIEWER` | the copy in the cache, else the CDN | Where the Rerun web viewer is loaded from: a directory, a URL, or `none` |
| `VIEWER_RERUN_MAP_DIR` | `~/autoware_data/maps`, if it exists | Where the converter looks a map up by name when the log's own map file is not on this machine |
| `VIEWER_SCENARIO_COMMAND` | `uv run scenario` | The command a SLURM batch is submitted with |

### Web UI Pages

| Route | Description |
|-------|-------------|
| `/` | Session list — displays all test sessions grouped by date/time |
| `/session/{type}/{date}/{time}` | Session detail — shows all scenarios in a session with pass/fail status |
| `/session/{type}/{date}/{time}/{index}` | Scenario detail — the run in the Rerun viewer, condition tree, per-condition status |
| `/recording/{type}/{date}/{time}/{index}.rrd` | The case's Rerun recording (made first if need be) |
| `/session/{type}/{date}/{time}/odd` | The session's results on the ODD (`?x=`, `?y=` the attributes, `?at=attribute=bucket` the cell drilled into) |
| `/compare` | Sessions compared (`?s=type/date/time`, two or more; `?x=`, `?y=` the cells' attributes) |
| `/compare/runs` | Runs side by side in Rerun viewers (`?run=type/date/time/index`, up to four) |

### REST API Endpoints

The viewer also provides API endpoints for programmatic use and the web UI's interactive features:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/refresh` | Clear result cache and return updated session table HTML |
| `GET` | `/api/scenarios` | List all available scenario config names |
| `POST` | `/api/run/preview` | Preview the CLI command that would be executed |
| `POST` | `/api/run` | Start a scenario execution in the background |
| `GET` | `/api/run/status` | Get current run status (non-streaming) |
| `GET` | `/api/run/progress` | SSE (Server-Sent Events) stream for real-time progress |
| `GET` | `/api/launchers` | The launchers a run can use here (`slurm` only where `sbatch` is, with the partitions) |
| `GET` | `/api/slurm/submission` | The batch the viewer last submitted through SLURM, and its page once submitted |
| `GET` | `/api/slurm/{date}/{time}/status` | A SLURM batch's cases, jobs and tasks |
| `POST` | `/api/slurm/{date}/{time}/{resubmit,collect,cancel}` | `scenario-slurm` on the batch (`resubmit` takes `failed`, `array_parallelism`, `cases_per_task`) |
| `POST` | `/api/recording/{type}/{date}/{time}/{index}` | Make the case's Rerun recording if there is none; its URL and path (503 with the install hint without the converter) |

**Run request body format:**

```json
{
  "scenario": "*/*",
  "extra_overrides": ["server.host=192.168.1.100"],
  "sweeper": "",
  "launcher": "local"
}
```

With `"launcher": "slurm"`, `"slurm"` holds the `launcher.*` keys:
`{"partition": "main", "array_parallelism": 2, "cases_per_task": null,
"time": "02:00:00", "mem": "12G", "gres": null, "start_stagger_seconds": 30,
"remote_cwd": "workspace/OnePlanner", "python": ".venv/bin/python",
"setup": "source ~/env.sh", "env": {"NAME": "value"}}` (`setup`: a shell line
per line, without `${...}` or backslashes; `env`: `NAME=value` lines, or a
mapping).

**Progress SSE event format:**

```json
{
  "current": 2,
  "total": 5,
  "scenario_name": "intersection_passing/straight",
  "status": "running"
}
```

Where `status` is one of: `"running"`, `"idle"`, or `"done"`.

---

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `CARLA_EXECUTABLE` | scenario runner | Path to CARLA's launcher (`CarlaUnreal.sh`); defaults to the one `scenario-setup` installed |
| `AUTOWARE_CARLA_SCENARIO_HOME` | `scenario-setup`, scenario runner | Moves `~/.autoware_carla_scenario` (CARLA lives in `bin/carla`) |
| `NISHISHINJUKU_XODR_PATH` | map config | Override default OpenDRIVE file path for nishishinjuku |
| `NISHISHINJUKU_LANELET2_PATH` | map config | Override default Lanelet2 file path for nishishinjuku |
| `VIEWER_BASE_PATH` | viewer | Base path for scenario results |
| `VIEWER_HOST` | viewer | Viewer server bind address |
| `VIEWER_PORT` | viewer | Viewer server listen port |
| `VIEWER_CACHE_DIR` | viewer | The viewer's cache (the Rerun web viewer's copy, recordings it could not keep beside their logs) |
| `VIEWER_RERUN_WEB_VIEWER` | viewer | Where the Rerun web viewer is loaded from (a directory, a URL, or `none`) |
| `VIEWER_RERUN_MAP_DIR` | viewer | Where the Rerun converter looks maps up by name |
| `VIEWER_SCENARIO_COMMAND` | viewer | The command a SLURM batch is submitted with (default `uv run scenario`) |
| `SCENARIO_EDITOR_DRAFTS` | scenario editor | Directory holding scenario drafts |
| `SCENARIO_EDITOR_EXPORT_DIR` | scenario editor | Staging area for exported package archives |
| `SCENARIO_EDITOR_MAP_ROOTS` | scenario editor | Directories a Lanelet2 map may be served from |
| `SCENARIO_EDITOR_HOST` | scenario editor | Editor server bind address |
| `SCENARIO_EDITOR_PORT` | scenario editor | Editor server listen port |
| `SCENARIO_EDITOR_MAP_VIEWER` | scenario editor | URL of a self-hosted `simple_lanelet2` map viewer |
| `SCENARIO_EXPORT_FRAMEWORK_VERSION` | package export | Pin exported packages to a published framework release |
| `SCENARIO_EXPORT_FRAMEWORK_REPOSITORY` | package export | Repository URL recorded for a commit pin |

---

## Common Use Cases

### Use Case 1: Run a Single Scenario

```bash
# Start CARLA server first, then:
uv run scenario scenario=intersection_passing/straight
```

### Use Case 2: Run All Scenarios for a Map

```bash
uv run scenario scenario='*/*' map=nishishinjuku
```

### Use Case 3: Test on a Remote CARLA Server

```bash
uv run scenario scenario=intersection_passing/left_turn \
  server.host=192.168.1.100 server.port=2000
```

### Use Case 4: Detect Missing 3D Models Before Running Scenarios

```bash
# 1. Detect lanelets without ground geometry
uv run detect-no-3d-model conf/map/nishishinjuku.yaml

# 2. The detected IDs are written to no_3d_model_lanelet_ids in the YAML
# 3. Now run scenarios — sweep will skip these lanelets
uv run scenario scenario='intersection_passing/*' map=nishishinjuku
```

### Use Case 5: Browse Results in the Web UI

```bash
# 1. Run scenarios
uv run scenario scenario='*/*'

# 2. Launch the viewer pointing to the outputs directory
VIEWER_BASE_PATH=outputs uv run viewer

# 3. Open http://localhost:9000 in a browser
```

### Use Case 6: Resume a Failed Batch

```bash
# Original batch run (5 scenarios, stopped at the 4th, e.g. by Ctrl+C)
uv run scenario scenario='intersection_passing/*'

# Resume from the 4th scenario
uv run scenario scenario='intersection_passing/*' --resume-from 4
```

---

## Troubleshooting

### "Connection refused" or timeout when connecting to CARLA

**Cause**: CARLA server is not running, or host/port is incorrect.

**Solution**:
- Verify the CARLA server is started and accessible
- Check `server.host` and `server.port` overrides match the server
- Increase `--timeout` for `detect-no-3d-model`

### "batch scenarios must share the same map"

**Cause**: Glob pattern matched scenarios configured for different maps.

**Solution**:
- Use `map=<name>` override to force all scenarios to use the same map
- Or narrow the glob pattern to match scenarios for one map only

### Ego vehicle falls through the ground

**Cause**: The spawn lanelet has no corresponding 3D ground geometry in CARLA.

**Solution**:
- Run `detect-no-3d-model` to identify problematic lanelets
- Add them to `no_3d_model_lanelet_ids` in the map config
- Increase `entity.ground_projection_ray_distance_lower` for deeper ray casting
- Increase `entity.spawn_retry_max_count` for more spawn attempts

### Scenario times out unexpectedly

**Cause**: `scenario.timeout_seconds` is too short, or the ego vehicle is stuck.

**Solution**:
- Increase `scenario.timeout_seconds`
- Check `ego.initial_speed_kmh` — a value of `0.0` requires autopilot to start moving
- Enable verbose logging and check condition statuses in the result JSON

### "Unknown scenario name" error

**Cause**: The `scenario.name` field in the YAML does not match any registered scenario class.

**Solution**:
- Supported scenario names: `intersection_passing`, `lane_change`, `traffic_light_compliance`, `temporary_stop`
- Check the `name` field in your scenario YAML config

---

## Next Steps

- [API Reference](api.md) — Detailed API documentation for programmatic usage
- [Development Guide](development.md) — Contributing and development setup
