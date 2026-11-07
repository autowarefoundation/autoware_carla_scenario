# Usage Guide

This guide explains how to use the `autoware-carla-scenario` package to run autonomous driving scenario tests in CARLA simulator.

## CLI Commands Overview

The package provides these CLI commands:

| Command | Framework | Purpose |
|---------|-----------|---------|
| [`scenario`](#scenario-scenario-runner) | Hydra | Run autonomous driving scenario tests in CARLA |
| [`scenario-expand`](logical_scenarios.md) | Hydra compose | Print the concrete scenarios of a scenario as JSON, without running anything |
| `scenario-new` | argparse | Scaffold a standalone scenario package |
| [`scenario-setup`](installation.md) | argparse | Download a CARLA server (the nightly build by default) for the runner to launch |
| [`scenario-check`](typecheck.md) | Hydra compose | Compile scenarios with Codon without running them (the static check the runner makes before a run) |
| [`scenario-slurm`](slurm.md#following-collecting-resubmitting) | argparse | Follow, collect, resubmit or cancel a batch run as SLURM array jobs (`launcher=slurm`) |
| [`scenario-gate`](gate.md#judging-scenario-gate) | argparse | Judge logical scenarios' gates (runs, success rate, coverage, trajectory consistency) from their runs' gate records, or compare a baseline's runs with a candidate's for regressions |
| [`scenario-odd`](odd.md#checking-an-odd) | argparse | Check an ODD and show its buckets; `scenario-odd route` checks scenarios' planned routes against it on the Lanelet2 map |
| [`scenario-coverage`](coverage.md) | argparse | Merge the coverage files of scenario runs into an ODD/scenario coverage report |
| [`detect-no-3d-model`](#detect-no-3d-model-3d-model-detection) | argparse | Detect lanelets without a matching 3D ground model in CARLA |
| [`viewer`](#viewer-scenario-result-viewer) | FastAPI + Uvicorn | Web UI for browsing and monitoring scenario test results |
| [`scenario-editor`](scenario_editor.md) | FastAPI + Uvicorn | Web UI for authoring scenarios and exporting reproducible packages |
| [`scenario-hils`](hils.md) | argparse | Stream a run's ego to devices under test (CAN, GNSS, LiDAR, cameras); needs the `hils` extra |
| [`scenario-t4scene`](t4_scene.md) | argparse | Write runs' trajectory logs as T4Scenes for tier4/e2e-devkit |

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

# Run every lane-change scenario, however deep below lane_change/
uv run scenario scenario='lane_change/**'
```

The pattern is matched as `scenario/{pattern}.yaml` under each config
directory: `*` stays within one path component, and a `**` component crosses
them. A trailing `**` means everything below it and is read as `**/*`, so
`lane_change/**` and `**` select every config below `lane_change/` and
`scenario/`. A `**` inside a component (`lane_change/**left`) is refused with
an error naming the pattern, and so is a pattern that could reach outside
`scenario/`: one starting with `/` (`/lane_change/*`) or with a `..` component
(`../*`). The pattern means the same on every supported
Python version, and the viewer's Run form reads it the same way.

**Batch constraints:**

- All scenarios in a batch **must share the same map** (CARLA loads one map per session):
  the same `map.name`, `map.source`, `map.directory`, `map.lanelet2_path`,
  `map.xodr_path`, `map.projector_type` and `map.overwrite_xodr`, as
  configured. A batch whose cases differ in any of them is refused, naming the
  key and the cases (`map.no_3d_model_lanelet_ids` may differ: only a sweep's
  constraints read it)
- A config value a case would refuse -- an unknown `server.world_reset`, a
  `scenario.timeout_seconds` that is not positive, a `server.port` out of
  range, a NaN or negative `server.cooldown_seconds`, a
  `server.max_tick_rate_hz` or `server.fog_density_scale` that is not a
  finite number above 0, a `server.cooldown_max_retries` or
  `server.reload_every` that is not a whole number, a seed or count that is
  not one ([One Seed for the Whole Run](#one-seed-for-the-whole-run)), an
  `environment.*` value that is not a number, a `t4scene.step_s` that is not a multiple of the tick, ... -- is
  refused for every case before any scenario is built or the map is resolved,
  locally as under `launcher=slurm`: the batch prints
  `Error: <case>: <key> ...`, naming the case refused, and exits 1, a
  configuration error as a single run's is (`HYDRA_FULL_ERROR=1` shows the
  traceback). With `--multirun` each scenario's config is checked before its
  sweep reads the map, as the lanelet sweeper checks it, and reported as
  `Error: <scenario>: <key> ...`
- Results are written to `outputs/YYYY-MM-DD/HH-MM-SS/batch_results.json`
  (a batch started in the same second as another gets `HH-MM-SS-1`,
  then `HH-MM-SS-2`, ...: every batch has a directory of its own). The file
  is first written once the batch's pre-run checks have passed, just before
  its first case runs, with every case listed as `"passed": null,
  "pending": true` (as a SLURM batch's cases without an outcome are), and
  rewritten as each case ends, the cases not run yet still pending. So a
  batch that dies at any point from there on -- in its first case too --
  still lists every case, and
  [`scenario-gate`](gate.md#cases-that-raised-no-record-so-the-set-is-refused)
  refuses to judge it from the cases that finished alone; the viewer lists it
  as a batch with its cases pending. A batch refused before any case runs
  (a backend or an Autoware workspace that cannot run, CARLA not coming up)
  writes no summary. The summary written when the batch ends lists no
  pending case
- A summary table is printed at the end with per-scenario pass/fail status
- A case that cannot run at all -- its run raises on every attempt
  (`server.cooldown_max_retries`), e.g. a spawn point with no ground under it --
  does not end the batch: it is reported as `[ERROR]` with the error, its
  `batch_results.json` entry is `"passed": false` with `"error": true`, and the
  batch goes on with the next case. Ctrl+C still stops the batch
- A retry (`server.cooldown_max_retries`) runs on a reset world
  (`server.world_reset`): the world is reset right before every attempt that
  follows a run -- the next case's, or a retry's -- so the weather or frozen
  traffic lights the failed attempt set do not carry into it
- A reset that raises is the failure of the attempt it comes before: that
  attempt is retried, or the case is reported as `[ERROR]`, and the batch
  goes on. The case that ran before the reset keeps its verdict, and is not
  run again (#1941). Between cases the world is reset first, and
  `server.cooldown_seconds` is waited after it
- A CARLA server the batch started that dies mid-batch is restarted before the
  next attempt (a retry, or the next case), with the map loaded again, so the
  cases after the crash run rather than spend their attempts on a server that
  is gone. A restart that fails is that attempt's failure. A server the batch
  reused (already running when it started) is never restarted
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

A scenario whose sweep expands to no case (its constraints, route or ODD
sample match nothing on the map) is refused as the lanelet sweeper refuses
it: the batch prints `Error: <scenario>: the sweep expands to no case ...` for
each such scenario of the glob and exits 1 before anything runs, even when the
other scenarios have cases. A scenario without a `sweep:` is one case, never
refused. An exported scenario on a map of its own composed on another map
(`map=<id>_map` left out) is refused the same way, before its sweep reads any
map:
`Error: <scenario>: scenario '<id>' runs on the map its document names, but ...`
([Running an authored scenario](scenario_editor.md#running-an-authored-scenario)).
`launcher=slurm` expands the batch the same way, so it submits nothing.

On a SLURM cluster, `launcher=slurm` runs a batch's cases as array jobs, several
at once, each on a CARLA of its own, with the same outputs (off unless
selected; see [Running a Batch on SLURM](slurm.md)):

```bash
uv run scenario --multirun 'scenario=cut_in/*' +sweep.odd_sample.count=24 \
  launcher=slurm launcher.array_parallelism=2
```

#### Resetting the world between cases

Between two cases of a batch (and between two cases of a SLURM task) the world
is reset, so that nothing of one case is in the next. `server.world_reset`
says how:

| `server.world_reset` | What happens | Cost per case (CARLA UE5, Town10HD_Opt) |
|----------------------|--------------|-----------------------------------------|
| `reload` (default)   | The map is loaded again: a new episode | ~5 s |
| `soft`               | The map stays loaded; the world is put back as it was right after the load | well under a second |

```bash
uv run scenario --multirun 'scenario=cut_in/*' map=town10hd_opt server.world_reset=soft
# Soft resets, with a full reload on every 10th reset anyway
uv run scenario --multirun 'scenario=cut_in/*' map=town10hd_opt \
  server.world_reset=soft server.reload_every=10
```

`soft` is opt-in: it is not yet shown that every case gives the same result
after a soft reset as after a reload (#282). Compare a sweep run both ways
before relying on it, and keep `reload` for results that have to be
order-independent until that is shown for your scenarios.

**What a soft reset does.** Right after the map is loaded (at the start, and
after every reload) the runner reads a *baseline* off the world: its actors,
the episode settings, the weather, every traffic light's green, yellow and red
times and whether it is frozen, and the spectator's transform. A soft reset
then, in this order:

1. stops the CARLA recorder (the run stops its own already; this catches one
   a failed run left going);
2. destroys every actor the baseline does not list -- sensors and controllers
   first, then walkers, vehicles and anything else (props, an ego an external
   stack spawned) -- and ticks the world. Traffic lights, traffic signs and the
   spectator are the map's and are never destroyed;
3. gives every traffic light its baseline timing back, resets all of them to
   the start of their cycle (`World.reset_all_traffic_lights()`) and freezes or
   unfreezes each as it was;
4. puts the weather and the spectator back, and ticks the world;
5. checks that no actor outside the baseline is left and that none of the
   map's is gone;
6. applies the baseline episode settings (asynchronous, as a loaded world is;
   the next run makes it synchronous again).

What the run itself cleans up is cleaned up as before, whatever the reset: the
ego, the conditions' sensors, the props, the T4 and Rerun sensors, the
recorder, and the traffic backend (closed after every run, which is what resets
the TrafficManager).

**When it loads the map instead.** The map is loaded, as with `reload`, when

- there is no baseline (the map was not loaded by this runner yet, or reading
  the baseline failed);
- the server runs another episode than the one the baseline was read in:
  another client loaded a world -- another map, or the same one -- on the
  server;
- any step above raises, an actor survives its destroy, or an actor of the map
  is gone;
- the world has a traffic light the baseline does not list (the server listed
  it only after the baseline was read): its timing and frozen state are not
  known, so whatever a case did to it could not be put back;
- `server.reload_every` is N > 0 and N - 1 soft resets have been done since
  the last load (every N-th reset is a reload; `1` is the same as `reload`).

A reload reads a new baseline, so the resets after it are soft again.

**What a soft reset cannot guarantee.** These differ from a reload, and a
scenario whose result depends on them can give a different result after a soft
reset:

- **The episode goes on.** The frame number and the episode's elapsed time keep
  counting, and actor ids keep growing (a fresh episode starts both again).
  The runner measures a case on its own clock from its first tick, so its
  conditions and timeouts do not see this, but anything that reads raw frame
  numbers, `elapsed_seconds` or actor ids does.
- **Traffic light phase.** The lights are reset to the start of their cycle,
  which is where a load leaves them too, but how far into it they are on the
  case's first tick may differ by the ticks the reset itself takes.
- **Vehicle and physics state.** The previous case's vehicles are destroyed and
  new ones spawned, but state the server keeps outside an actor (physics
  scene, pooled resources) is not reset. In the trials behind #282 an ego's
  gear differed on every tick from a fresh-world run of the same case.
- **What has no getter in the CARLA API.** Map layers unloaded with
  `World.unload_map_layer()`, environment objects toggled with
  `enable_environment_objects()`, street lights switched through the
  `LightManager`, the pedestrians' cross factor and seed, and debug drawings
  still within their `life_time`. The framework changes none of these but the
  pedestrians' cross factor, which it puts back itself after starting each
  walker; a scenario that changes the others needs `reload`.
- **The TrafficManager.** It lives outside the episode, so a reload does not
  reset it either: the traffic backend's `close()` after each run is what does.

`server.world_reset` applies to the cases one batch runs in one queue
(`--multirun` with a glob or a `sweep:`, and the cases of one SLURM task). A
Hydra sweeper such as `hydra/sweeper=lanelet_constraint` runs each case as a
job of its own, which loads the map when it starts, so there is nothing to
reset between its cases.

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
optionally deriving per-job parameters via "bindings" (picks that end as
the same case once their bindings are applied give one job). Each scenario
YAML may declare its own `sweep:` section (see
`traffic_light_compliance.yaml` for an example).

`sweep.constraints` sweeps **one** parameter: it holds a single target key
(such as `ego.spawn_lanelet_id`) with its list of constraints, and every other
parameter a case needs is derived from the matched lanelet by
`sweep.bindings`. A `sweep.constraints` with a second key is refused with an
error when the sweep is expanded, rather than searched together with the first.
So is a section of the wrong shape: `sweep.constraints` that is not a mapping
of its key to a list of constraints, `sweep.bindings` that is not a mapping of
keys to bindings (`{type: ...}`), and `sweep.exclude` that is not a mapping of
keys to lists. A constraint or binding with an unknown `type`, a key it does
not take (`has_adjacent` takes `value`, not `side`), a missing key it needs
or a value it refuses (a `route_through` `depth` that is not a whole number of
at least 1) is refused the same way, naming its type and the key. Each is the
one line `Error: <scenario>: ...` that the sweeper, `scenario-expand` and a
batch print for an invalid sweep.

A `stop_line_offset` binding places a spawn `offset` metres before the pick's
stop line. When that does not fit on the pick, it walks back through
predecessor lanelets, and moves **its own spawn's** lanelet with it: the
lanelet key beside the `s` it writes (`ego.spawn_s` moves
`ego.spawn_lanelet_id`, `ego.goal_s` moves `ego.goal_lanelet_id`,
`scenario.spawn_overrides.<entity>.s` moves
`scenario.spawn_overrides.<entity>.lanelet_id`). So an NPC placed 300 m before
the stop line walks back on its own while the ego stays on the pick. A
walk-back of any other key has no lanelet to move and is refused. A
`stop_line_approach` binding with the same offset names the same lanelet, and
two bindings that set one key to different values on a pick (a
`stop_line_approach` with another offset than its spawn's `stop_line_offset`,
say) are refused with an error rather than one silently winning:

```yaml
sweep:
  constraints:
    ego.spawn_lanelet_id:
      - type: has_traffic_light_stop_line
  bindings:
    ego.spawn_s: {type: stop_line_offset, offset: 2}
    scenario.spawn_overrides.npc1.s: {type: stop_line_offset, offset: 300}
    scenario.spawn_overrides.npc1.lanelet_id: {type: stop_line_approach, offset: 300}
```

The `offset` of `stop_line_offset` and `stop_line_approach` is a finite number
of metres, at least 0; a string (`"15"`), `.nan`, a bool or a negative offset
is refused with an error naming the binding. Since a walk-back of the swept
key's own spawn (`ego.spawn_s` when the sweep picks `ego.spawn_lanelet_id`)
replaces the pick, several picks that walk back to the same predecessor can
end as the same case. Each distinct case runs once, as the first of those
picks gives it, and the log names the picks that were merged.

A constraint sees the pick, not where a walk-back puts the spawn. Use
`sweep.exclude` to drop a case by the values it ends up with once its bindings
are applied. Each key lists the values a case may not hold there. For example,
this keeps a spawn walked back from a junction off the lanelets CARLA does not
draw:

```yaml
sweep:
  exclude:
    ego.spawn_lanelet_id: ${map.no_3d_model_lanelet_ids}
```

A `route_through` binding gives the pick and the `depth` lanelets after it.
Where the graph forks, it follows the lowest id. With `through_junction: true`
it keeps going while the route's last lanelet is still in a junction, so a turn
whose junction is several lanelets long ends on the lane it comes out on:

```yaml
sweep:
  bindings:
    scenario.expected_route_lanelet_ids:
      type: route_through
      depth: 1
      through_junction: true
```

```bash
# Run a Hydra multirun over every lanelet whose `ego.spawn_lanelet_id`
# satisfies the scenario's `sweep.constraints`.
uv run scenario --multirun scenario=traffic_light_compliance/traffic_light_compliance \
  hydra/sweeper=lanelet_constraint

# Resume a sweep from job number N (1-indexed, at least 1).
uv run scenario --multirun scenario=traffic_light_compliance/traffic_light_compliance \
  hydra/sweeper=lanelet_constraint --resume-from 10
```

The resume point comes from `--resume-from N` on the command line or from
`sweep.resume_from` in the config. The command line wins: with
`sweep.resume_from: 5` in the config, `--resume-from 10` resumes at job 10.
(`--resume-from` reaches the sweeper as the `SWEEP_RESUME_FROM` environment
variable, which you may also set yourself; it wins over the config the same
way, so unset a `SWEEP_RESUME_FROM` left over in your shell.) `--resume-from` takes a whole number of at least 1: `0`, a negative
value or one that is not an integer is a usage error, `Error: --resume-from
...` and exit 2, before anything runs. `sweep.resume_from` is a whole number
too; `0` or `null` (or leaving it out) means it is not set, and every job runs
unless `--resume-from` says otherwise. A `sweep.resume_from` that is a bool, a
float (`2.7`, or even `3.0`), a string or negative, and a `SWEEP_RESUME_FROM`
that is not a whole number, is refused with one line, `Error:
sweep.resume_from must be a whole number of at least 1 ...` (or `Error:
SWEEP_RESUME_FROM ...`), exit 2, before the map is loaded -- even when
`--resume-from` overrides it -- rather than truncated, read as 1 or as "run
every job", or raised as a traceback.

Sweeps write results under `multirun/YYYY-MM-DD/HH-MM-SS/<job-index>/`,
not `outputs/`. The result viewer scans both directories. A sweep started in
the same second as another gets `HH-MM-SS-1`, then `HH-MM-SS-2`, ...: every
sweep has a directory of its own, unless `hydra.sweep.dir=` names one.

A failed, crashed or timed-out case does not stop the sweep: every case runs,
the sweeper logs `Sweep complete: P/N succeeded, F failed (T timed out).`, and
the process then exits 1 when any case failed (see [Exit Codes](#exit-codes)).

The sweeper runs each case in a forked process of its own. With
`server.cooldown_max_retries` it runs a case again only when it could not run:
its run raised (CARLA went away, a spawn failed), timed out
(`sweep.job_timeout_seconds`) or crashed. A case whose scenario ran to a
FAILED verdict is final and is not retried, as in a batch. The sweeper owns
these retries: the case's own queue does not retry it as well, so a case runs
at most `1 + server.cooldown_max_retries` times, as in a batch.
A retry runs in the case's job directory, and before it starts the sweeper
removes what the failed attempt wrote there (its result, coverage file,
trajectory log, recordings ...), as a batch and a SLURM worker do: the retry
takes the same file stem, so the viewer and the coverage merge read the
retry's outputs alone.

A case run again into a job directory an earlier sweep wrote in
(`hydra.sweep.dir=` naming it again, a `--resume-from` rerun) replaces that
run: before its first attempt the sweeper removes the run outputs the earlier
sweep left there -- every file and directory named after a run's stem, its
result, gate record, coverage file, trajectory log, recordings, T4 dataset
... -- so the case takes the same stem again and `scenario-gate`, the
coverage merge, the viewer and the sweep's verdict read the new run alone,
not both. Hydra's job log and `.hydra/` directory, the case's
`case_errored.json` (replaced as the case starts), the logs of the services
a run starts there (`backend_<name>.log`, `autoware-*.log`) and any other
file that is not a run's output are kept. A sweep into a new directory has
nothing to remove. Only a job directory that no other case of the sweep maps
to is cleaned: one several cases share (a `hydra.sweep.subdir=` that does not
tell the jobs apart) holds the runs of the cases a `--resume-from` skips,
and is left as it is -- so an earlier run of a case run again there stays
beside the new one, and `scenario-gate` and the coverage merge read both.
(A retry still removes every output its failed attempt left, the services'
logs among them.)

A case given up -- it raised, timed out or crashed on every attempt, or was
refused -- leaves `case_errored.json` in its job directory (its overrides,
last exit code, attempts and why): it wrote no result, so no gate record, and
[`scenario-gate`](gate.md#cases-that-raised-no-record-so-the-set-is-refused)
refuses the sweep's runs for it. A case of Hydra's basic sweeper (a
`--multirun` without `hydra/sweeper=lanelet_constraint`) whose run raised
leaves one too. The marker is also written before each attempt, saying the
case did not finish, and removed once it ends with a verdict: a case whose
sweep was killed with it (a segfault, the OOM killer, `SIGKILL`, Ctrl-C)
leaves it as well. A case whose scenario ran and failed leaves none, nor does
one a retry passed; one run again in the same directory replaces the marker
an earlier run left as it starts.

Before it launches the first case, the lanelet sweeper writes the sweep's
plan, `sweep_plan.json`, into the sweep directory (`hydra.sweep.dir`): each
case's job directory and overrides. `scenario-gate` counts a planned case
that left no gate record, no marker and no result as one that never ran, so
a sweep killed in case 8 of 20 is not judged from cases 1-7 alone. A sweep
run again into the same directory (with `--resume-from` too) merges its plan
into the one there, keeping the cases it skips. The viewer lists the job
directories and ignores the plan. Hydra's basic sweeper creates each job
directory only as it reaches the case and writes no plan: a basic sweep
killed mid-run leaves no trace of the cases it never reached
([Gate](gate.md#cases-that-raised-no-record-so-the-set-is-refused)).

A case that timed out or crashed had no chance to stop the CARLA server it
launched; that server's watchdog stops it within about a second, giving it
10 s before `SIGKILL`. So before the next attempt -- a retry or the next case
-- the sweeper waits until the CARLA processes the case launched have exited
(at most 15 s, then it kills them), on top of `server.cooldown_seconds`: the
next case must neither reuse the dying server nor start one while it still
holds the port, even if the case was killed while its server was still
loading. A case that ended normally stopped its server itself, so nothing is
waited for; nor is a server the case did not launch (one you started
yourself, which it reused), which no case stops.

A sweep whose config sets a value its runs would refuse (a `t4scene.step_s`
that is not a multiple of the tick, a negative `server.cooldown_max_retries`,
an unknown `server.world_reset` or `traffic.backend`, a
`scenario.timeout_seconds` that is not positive, a
`sweep.job_timeout_seconds` that is not a whole number of seconds, a
`sweep.resume_from` that is not a whole number, ...) exits
2 before the map is fetched or any case runs, naming the key, rather than have every case refuse
it in turn. So does one asking for an extra (`rerun`, `t4`) that is not
installed. A value only a case's own overrides set (its ODD draw, a
binding) is refused by that case, which ends the sweep with 2 as a refused
case does; outside a lanelet sweep the same value is a configuration error,
exit 1.

A sweep with no case to run exits 1 before anything runs, as a batch with no
case left does: its constraints, route or ODD sample match nothing on the map
(`the sweep expands to no case`), or `--resume-from` is past its last case.
A `--multirun` batch, the viewer and `scenario-expand` refuse such a sweep
with the same message, naming the scenario. They refuse an invalid sweep (a
`sweep.case_index` that is not a whole number >= 0 or is past its cases, a constraint that does not parse, a
Lanelet2 map that is missing, unreadable or cannot be resolved, ...) the same
way, `Error: <scenario>: <why>`; a batch or `scenario-expand` of a glob
reports every scenario refused, not just the first, and runs or lists nothing.
The sweeper (`hydra/sweeper=lanelet_constraint`, a single scenario's
`--multirun`) refuses an invalid sweep the same way, before any case runs:
one line on stderr, `Error: <scenario>: <why>`, exit 1, no traceback. These
are a config with no `sweep` section, `sweep.constraints` naming more than
one key, what the expansion refuses (a `sweep.case_index` that is not a whole number >= 0 or is past its cases, a
constraint or binding that does not parse, a `sweep.odd_sample` setting or
`weights` file it refuses, ...), and a Lanelet2 map it cannot load, whose
`<why>` is `LaneletConstraintSweeper: <why>`, naming the map: a
`map.directory` that does not exist, a `map.source` that cannot be fetched, a
map cache whose root is not a directory (`AUTOWARE_DATA_DIR` naming a file),
or a `map.lanelet2_path` that is missing or that Lanelet2 cannot read. With
`HYDRA_FULL_ERROR=1` each is raised instead, traceback and all. Invalid
`controls` are refused in `scenario-expand`'s format, before the map is
loaded (so before a `map.source` is cloned): one line on stderr per problem,
`<scenario>: <path>: <why>` (see [Checking controls](odd.md#checking-controls)),
exit 1; unlike `scenario-expand`, the sweeper raises them instead with
`HYDRA_FULL_ERROR=1`. Every other error
the sweeper meets keeps Hydra's traceback.

Ctrl-C or `SIGTERM` stops the sweep and the case it is running. The case is
sent `SIGTERM`, so it stops its CARLA server and Autoware stack as on Ctrl-C,
and `SIGKILL` if it is still running 30 s later (a second Ctrl-C does not
wait). The sweep then exits 130 (Ctrl-C) or 143 (`SIGTERM`) and names the case
to resume from.

To get the concrete scenarios without running them (for example, to fan them
out over a cluster), use `scenario-expand`. It takes the same arguments as
`scenario` and prints JSON with one list of Hydra overrides per concrete
scenario. These are the same cases, in the same order, that the sweeper runs:

```bash
uv run scenario-expand scenario=traffic_light_compliance/traffic_light_compliance
# {"scenario": "...", "cases": [["ego.spawn_lanelet_id=5", "ego.spawn_s=18.62..."], ...]}
```

A scenario without a `sweep:` section is already concrete, so it expands to a
single empty case. Only the Lanelet2 map is needed, not a CARLA server. A
sweep that expands to no case is refused rather than listed as `"cases": []`:
`Error: <scenario>: the sweep expands to no case ...` on stderr, nothing on
stdout, exit 1. So is an invalid sweep -- a `sweep.case_index` that is not a whole number >= 0 or is past its cases,
a constraint or binding that does not parse, constraints for more than one
key, a Lanelet2 map that is missing, unreadable or cannot be resolved (a
`map.directory` that does not exist, a `map.source` that cannot be fetched or
whose map cache root is not a directory), ...: `Error: <scenario>: <why>` (`HYDRA_FULL_ERROR=1` shows its traceback
instead). As for `scenario`, the last `scenario=` wins when it is
given more than once.

As with `scenario`, `scenario=` may be a glob. Each scenario it matches is
expanded with the same overrides and printed as its own JSON object, one per
line (JSON Lines), in the order a batch runs them (sorted by name). Each object
is the one that scenario prints on its own, `"scenario"` naming it:

```bash
uv run scenario-expand 'scenario=intersection_passing/*'
# {"scenario": "intersection_passing/left_turn", "cases": [...]}
# {"scenario": "intersection_passing/right_turn", "cases": [...]}
# ...
```

Every matched scenario is expanded before anything is printed. If one of them
fails (its `controls:` are invalid, its sweep is invalid or expands to no
case, or an override does not compose), each
failure goes to stderr headed by that scenario's name, nothing goes to stdout,
and the exit status is non-zero.

A *logical* scenario -- a document with a route search, whose `sweep:` holds
`route:` instead of `constraints:` -- expands to one case per route of the map
that matches its pattern, each with the ego's spawn and goal and the match
itself (`scenario.route.*`); see
[Logical Scenarios from Routes](logical_scenarios.md).

### Resume from a Specific Scenario

When running large batches, you can skip already-completed scenarios.
`--resume-from N` starts at the *N*th scenario (1-indexed, as for a sweep);
in a `--multirun` batch it counts the expanded cases. *N* is a whole number of
at least 1: `--resume-from 0`, `--resume-from -3` or `--resume-from abc` is a
usage error, `Error: --resume-from ...` and exit 2, rather than a run of every
scenario:

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

`server.port` and `traffic_manager.port` (and `traffic.options.port`) must be
TCP ports, integers within 1-65535; any other value (`0`, `70000`, `true`)
is refused before anything starts.

### One Seed for the Whole Run

`seed` (top level, `0` by default) is the one number every random draw of a run
follows, so the same seed runs the same scenario again:

```bash
uv run scenario scenario=intersection_passing/straight seed=42
```

A seed -- `seed`, `sweep.odd_sample.seed`, and each component's own below
(`background_traffic.seed`, `driver.random_seed`,
`driver.lidars[].noise_seed`) -- is a whole number from 0 to 2147483647
(`2**31 - 1`, `autoware_carla_scenario.seed.MAX_SEED`): `42` or `42.0`, not
`1.9`, a boolean, a string that is no integer, `-1` or `2147483648`, as a
route search's counts are ([Logical Scenarios](logical_scenarios.md)). The
upper bound is what the simulators take: CARLA's pedestrian seed is a 32-bit
unsigned integer and SUMO's `--seed` a 32-bit signed one, so a larger seed
used to fail every case once CARLA had started (#1560). The background
traffic source draws from one past `background_traffic.seed`, which wraps to
0 past 2147483647;
`sweep.odd_sample.count` is a whole number of at least 1. Any other value is
refused before anything starts, naming the key, on every path: a single run,
a batch (local or SLURM), a lanelet sweep and `scenario-expand` (which reads
only `seed` and `sweep.odd_sample`). The driver's and its LiDARs' seeds are
refused when their config is built, with an `ego.entity=carla_driver` run's
other checks; being typed fields, they take no string at all, not even
`"7"`.

The other whole-number settings a run reads follow the same rule:
`background_traffic.source.initial_vehicles` and `.max_vehicles` (not
negative; `max_vehicles` may be null), `server.cooldown_max_retries` and
`server.reload_every` (not negative). `2.0` is 2, but `2.5` or `true` is
refused, naming the key, rather than run as 2 or 1.

It is the seed of:

| What draws | Its own seed, which wins when set |
| --- | --- |
| The cases a sweep draws from the ODD (`sweep.odd_sample`), and the grip and visibility drawn for each | `sweep.odd_sample.seed` |
| The scenario: `BaseScenario.random_seed` | -- |
| The traffic backend: CARLA's TrafficManager (`set_random_device_seed`, each vehicle's parameter draws), SUMO (`--seed`, `netconvert --seed`, its own draws) | -- (the scenario's) |
| CARLA's pedestrian navigation (`World.set_pedestrians_seed`) | -- (the scenario's) |
| Background traffic | `background_traffic.seed` |
| The adversarial actions, `AdversarialTrafficAction`, `AdversityFeasibleCondition`, `TrafficSourceAction` | their `seed` argument (`Seed` in the editor) |
| The smart pedestrian's motion generator (`SmartPedestrianControllerAction`) | its `seed` argument |
| The driver policy's session (`DriveSessionRequest.random_seed`) | `driver.random_seed` |
| A LiDAR's weather noise | `driver.lidars[].noise_seed` |

Every one of those seeds is `null` (`None` in Python) by default, and a null seed
is the run's. In Python the run's seed lives in `autoware_carla_scenario.seed`:
the runner sets it from the scenario's `random_seed` before `setup()` (and
`scenario` sets it from the config before it builds the scenario), and a
component built with `seed=None` takes it when it is built. A scenario that
draws for itself seeds its own generator from `self.random_seed`.

A route search's `seed` (`sweep.route.seed`) is not one of them: unset, the
matches are not shuffled at all. A config with no `seed` key (a scenario
package's own `config.yaml` written before it existed) leaves the scenario the
`random_seed` it was built with.

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
├── driver/                  # External driver policy (ego.entity: carla_driver)
├── ros2/                    # Ego driven by ROS 2 nodes (ego.entity: ros2)
├── hils/                    # Ego driven by ECUs under test (ego.entity: hils)
├── traffic_manager/
│   └── default.yaml         # CARLA Traffic Manager settings
├── traffic/                 # Traffic backend: traffic_manager, sumo, none
├── launcher/                # Who runs a batch's cases: local, slurm
├── hydra/sweeper/
│   └── lanelet_constraint.yaml
├── map/
│   ├── nishishinjuku.yaml   # Map file paths and excluded lanelet IDs
│   └── town10hd_opt.yaml
└── scenario/
    ├── adversity/           # brake_check, cut_in, ignore_leader, jaywalking, nade, traffic_rule
    ├── cut_in/              # left, right
    ├── intersection_passing/
    │   ├── straight.yaml
    │   ├── left_turn.yaml
    │   ├── right_turn.yaml
    │   ├── right_turn_sweep.yaml
    │   └── straight_3npc.yaml
    ├── lane_change/
    │   ├── left.yaml
    │   └── right.yaml
    ├── lane_change_fail/
    │   ├── left.yaml
    │   └── right.yaml
    ├── pedestrian_dart_out/
    │   └── pedestrian_dart_out.yaml
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
| `driver` | `default` | Driver policy, used when `ego.entity: carla_driver` ([Driver Interface](driver_interface.md)) |
| `ros2` | `default` | Ego driven by ROS 2 nodes, used when `ego.entity: ros2` ([ROS 2](ros2.md)) |
| `hils` | `default` | Ego driven by ECUs under test, used when `ego.entity: hils` ([HILS](hils.md)) |
| `traffic_manager` | `default` | CARLA Traffic Manager |
| `traffic` | `traffic_manager` | Traffic backend: `traffic_manager`, `sumo` or `none` ([Traffic Backends](traffic_backends.md)) |
| `scenario` | `intersection_passing/left_turn` | Scenario to run |
| `map` | `nishishinjuku` | Map to load |
| `launcher` | `null` | Who runs a batch's cases: unset runs them here, one after another; `launcher=slurm` runs them as a SLURM array job ([SLURM](slurm.md)) |

#### Server Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server.host` | str | `localhost` | CARLA server hostname |
| `server.port` | int | `2000` | CARLA server port |
| `server.cooldown_seconds` | float | `2.0` | Wait time (seconds) between consecutive scenario runs; a finite number, `0` or more (`0`: none) |
| `server.cooldown_max_retries` | int | `0` | Maximum retries of a run that raised (not of one that ran and failed); a whole number, `0` or more (`2.0` is 2; `1.9` or `true` is refused) |
| `server.max_tick_rate_hz` | float | `null` | Upper bound (Hz) on how fast the run ticks the world, or `null` for as fast as the server allows; a finite number above `0` (`20` is real time at the 0.05 s step, which `ego.entity=ros2` wants: [ROS 2](ros2.md#timing)) |
| `server.fog_density_scale` | float | `null` | `carla.Weather.FogDensityScale` for a server the run starts and for the LiDARs' fog, or `null` to keep the server's own `0.001`; a finite number above `0` ([Driver Interface](driver_interface.md)) |
| `server.world_reset` | str | `reload` | How the world is reset between a batch's cases: `reload` or `soft` (see [Resetting the world between cases](#resetting-the-world-between-cases)) |
| `server.reload_every` | int | `0` | With `world_reset: soft`, load the map on every N-th reset anyway; `0` (or `null`): only when a soft reset cannot be done; a whole number, `0` or more |

#### Ego Vehicle Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ego.vehicle_type` | str | `vehicle.mini.cooper` | CARLA blueprint ID for ego vehicle |
| `ego.initial_speed_kmh` | float | `0.0` | Initial speed of ego vehicle (km/h); finite, `0` or more |
| `ego.spawn_lanelet_id` | int | `242` | Lanelet ID for spawn position |
| `ego.spawn_s` | float | `25.0` | Longitudinal offset along lanelet centerline (meters) |
| `ego.spawn_t` | float | `0.0` | Lateral offset from the lanelet centerline, positive to its left (meters) |
| `ego.spawn_heading` | float | `0.0` | Yaw relative to the lanelet's direction of travel, positive anticlockwise (radians) |
| `ego.goal_s` | float | `0.0` | Longitudinal offset along the goal lanelet (`ego.goal_lanelet_id`) centerline (meters) |

The speed must be finite and not negative, and the poses (`ego.spawn_s`,
`ego.spawn_t`, `ego.spawn_heading`, `ego.goal_s`) finite: a negative or NaN
speed would leave the ego still with no error, an infinite one break its
physics, and a NaN pose fail far from the key behind it. A run or batch
refuses such a value before anything starts, naming the key (`Error:
ego.initial_speed_kmh must be finite and at least 0, got -20`).

#### Entity Configuration

Controls ground projection (ray casting) and spawn retry behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity.ground_projection_ray_distance_upper` | float | `5.0` | Ray search range above estimated z (meters) |
| `entity.ground_projection_ray_distance_lower` | float | `10.0` | Ray search range below estimated z (meters) |
| `entity.spawn_retry_max_count` | int | `10` | Maximum retry attempts for spawn |
| `entity.spawn_retry_t_step` | float | `0.1` | Lateral offset shift per retry, tried to both sides (meters); finite, `0` or more (`0`: no lateral shift) |
| `entity.spawn_retry_z_step` | float | `0.5` | Vertical shift per retry, tried up and down (meters); finite, `0` or more (`0`: no vertical shift) |

The two ray distances must be finite and non-negative, and their sum (the
ray's length) positive. A ray that cannot hit anything would leave every pose
at its estimate (and a negative range means nothing), so a run or batch refuses such a value before anything starts,
naming the key (`Error: entity.ground_projection_ray_distance_lower must be
non-negative and finite, got -1.0`).

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
uv run scenario scenario=cut_in/left rerun.enabled=true
uv run scenario 'scenario=cut_in/*' --multirun rerun.enabled=true
rerun outputs/2026-10-18/08-48-12/CutInScenario-3.rrd     # the Rerun viewer
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rerun.enabled` | bool | `false` | Convert each run's trajectory log into `{ScenarioName}.rrd` as soon as the run ends |
| `rerun.strict` | bool | `false` | Fail the conversion of a log with a record type or a field the converter does not know, rather than skip it with a warning |
| `rerun.draw_radius_m` | float | `null` | Write only what comes this near the ego's trajectory, in metres: the map, the traffic lights and other actors (`autoware-carla-scenario-rerun` 0.3 or newer). `null` writes all of it |
| `rerun.lidar` | bool | `false` | Put the LiDAR sweeps the ego's driver policy gets into the recording too ([Sensors](#sensors-in-the-recording)) |
| `rerun.lidar_max_points` | int | `65536` | At most this many points of a sweep (an even subsample); `0` keeps them all |
| `rerun.cameras` | bool | `false` | Put the camera frames the ego's driver policy gets into the recording too ([Sensors](#sensors-in-the-recording)) |

With `rerun.enabled=true`:

- A run asking for recordings without the extra installed is refused before
  anything starts (exit code 2), saying how to install it.
- Each log is converted on a background thread as soon as its run ends, while
  the next case runs (the converter releases the GIL; a log takes milliseconds).
  A run that raised is converted too: its log is what shows what happened.
- At the end the run waits for the last conversions and prints how they went:
  `Rerun recordings: N/M written`, and a line per failure naming its case. A
  retried case counts once, by the conversion of its last attempt. A
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

#### A T4 dataset of a run

With `t4.enabled=true` each run is also written as a T4 dataset,
`{ScenarioName}.t4/`, which t4-devkit reads: the sweeps and frames of the
driver policy's sensors (`driver.lidars`, `driver.cameras`) at each policy
step, and the box of every vehicle and pedestrian from CARLA's ground truth
(the `t4` extra; [T4 Dataset](t4_dataset.md)).

#### A T4Scene of a run

With `t4scene.enabled=true` each run's trajectory log is also written as a
T4Scene, `{ScenarioName}.t4scene.pb` with its map, `{ScenarioName}.t4scene.map.pb`,
the scene tier4/e2e-devkit decodes, scores and draws ([T4 Scene](t4_scene.md));
`scenario-t4scene <output dir>` does the same after a run.

#### The switches

`rerun.enabled`, `rerun.strict`, `rerun.lidar`, `rerun.cameras`,
`t4.enabled`, `t4scene.enabled` and `background_traffic.enabled` are each
`true` or `false`, or one of their spellings as text, in any case and with
blanks around it (`"false"`, `"no"`, `"off"`, `"0"`, and `"true"`, `"yes"`,
`"on"`, `"1"`), or the integer `0` or `1`, as the SLURM launcher's flags are
([Running a Batch on SLURM](slurm.md)). So `"false"`, what
`${oc.env:RERUN,false}` or a quoted `rerun.enabled='false'` gives, is false.
Anything else (`ture`, `2`, `0.5`, null) is refused before anything starts,
in one line naming the key:

```text
rerun.enabled must be true or false, got 'ture'
```

A sweep's `sweep.odd_sample.tire_friction` and `sweep.odd_sample.visibility`
([Tire friction follows the weather](odd.md#tire-friction-follows-the-weather))
and a route search's `ego_goal` ([Logical Scenarios](logical_scenarios.md))
are read the same way, except that a missing or null one is true.
`autoware.launcher.rosdep` ([Autoware](autoware.md)) is read the same way,
and so is `metrics.heavy` ([Metrics](metrics.md)), except that a null one
leaves the decision to `$AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS`; a batch is
refused for a bad one in any of its cases before the first case runs.

### Metrics in the recording

A run judged on a `MetricCondition` writes the metrics it computed once it was
over into its trajectory log ([Metrics](metrics.md#how-metrics-are-computed)),
and its recording carries them:

- each metric as a scalar, `metrics/<key>` (`metrics/crime_ttc_s`), on the
  recording's `frame` and `sim_time` timelines -- add a time series view of
  `/metrics` in the viewer to plot them (the converter's layout has none yet);
- each metric condition in the **Conditions** view, satisfied over the frames
  its metric was NG.

The values are written with the Rerun SDK and merged into the recording once
the log is converted, as the sensors are (`rerun rrd merge`); the `rerun`
extra installs the SDK. Converting a log again puts them in again.

### Sensors in the recording

The trajectory log holds no sensor data. With `rerun.enabled=true` and:

- `rerun.lidar=true`, the sweeps each driver LiDAR sends the ego's policy
  (`driver.lidars`) are recorded as `world/lidar/<logical id>`: points in the
  recording's world frame, coloured by height above the ego, drawn where the
  ego's pose put them when the policy got them;
- `rerun.cameras=true`, the frames each driver camera sends it
  (`driver.cameras`) are recorded as `world/ego/pose/cameras/<logical id>`:
  the camera's mount on the ego and a pinhole of its field of view, so the 3D
  view draws its frustum with the frame in it, and the JPEG frames the policy
  got (a 2D view of the entity shows them).

They are written during the run into `{ScenarioName}.sensors.rrd`; once the log
is converted they are merged into `{ScenarioName}.rrd` (`rerun rrd merge`) and
that file is removed. It needs the Rerun SDK of the converter's version:

```bash
pip install 'autoware-carla-scenario[rerun-lidar]'
uv run scenario scenario=cut_in/left driver=... rerun.enabled=true \
    rerun.lidar=true rerun.cameras=true
```

Points are most of a recording: about 0.8 MB a sweep at the default 65,536;
a camera frame is its JPEG (`driver.image_quality`). A merge that fails keeps
`{ScenarioName}.sensors.rrd` beside the recording (logged; not a failed
conversion); converting the run again (below) merges it. The two cannot be
opened together instead: the viewer registers a file by its recording id, and
the second replaces the first.

A run made without it (or by a version before it) can be converted afterwards,
the same way:

```bash
# Every run of a session (a sweep's directory works the same), or one log:
python -m autoware_carla_scenario.rerun outputs/2026-10-18/08-48-12
python -m autoware_carla_scenario.rerun outputs/2026-10-18/08-48-12/CutInScenario-3.trajectory.jsonl
```

or from Python, `autoware_carla_scenario.rerun.convert_outputs(path)`. It also
converts a run again (with `--strict`, or a newer converter), and keeps the
sensors the run recorded:

- a `{ScenarioName}.sensors.rrd` left beside the log (a merge that failed) is
  merged into the new recording and removed, as during a run;
- the sensors in a `{ScenarioName}.rrd` it replaces (`world/lidar/**`,
  `world/ego/pose/cameras/**`) are carried over into the new one. That needs
  the Rerun SDK's CLI (`rerun-lidar` extra); without it, or if it fails, the
  old recording is left as it was and reported as `[SKIPPED]` (exit status 1).
  `--overwrite` (`overwrite=True`) replaces it anyway, without its sensors.

A log that fails to convert leaves the recording it would have replaced as it
was. The converter reads nothing but the log and the map files the log names,
so it runs anywhere, without CARLA. The CARLA recording (`{ScenarioName}.log`) is
still written: CARLA's own replayer (`client.replay_file()`) plays it back on a
server running the same map.

The command takes several paths and converts each on its own: a path that does
not exist, or cannot be converted (a directory it cannot read or write to), is
reported as `[FAILED] <path>: ...` and the others are still converted. It exits
0 when every recording was written, 1 when any conversion failed or was
skipped, and 2 when the converter is not installed. `--jobs N` converts N logs
at a time (all cores by default); an N below 1 is refused, with exit status 2,
before any path is converted -- `--jobs 0` included, which earlier versions
passed on to the converter.

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
| `map.projector_type` | str \| null | `null` | Projection the Lanelet2 map is read with (`mgrs`, `utm`, `local_xy`, `local_cartesian` or `transverse_mercator`). If `null`, taken from the map's `map_projector_info.yaml` |
| `map.overwrite_xodr` | bool | `false` | Install `map.xodr_path` into the CARLA asset named by `<MAP_NAME>_PATH` before loading the map, replacing CARLA's own road network (`map=nishishinjuku` turns it on) |
| `map.no_3d_model_lanelet_ids` | list[int] | `[]` | Lanelet IDs to exclude from sweeps (no ground geometry) |

Map paths can use environment variables with Hydra's OmegaConf resolver:

```yaml
xodr_path: ${oc.env:NISHISHINJUKU_XODR_PATH,path/to/default.xodr}
```

### Supported Scenarios

In the parameter tables below, **Default** is the value a key takes when the
scenario's YAML omits it (the dataclass default in `examples/configs.py`), and
**Shipped** is the value the shipped config (`examples/conf/scenario/...`) sets,
or — when it ships the default. Every shipped config lists every key of its
config class, so a `scenario.<key>=` override reaches any of them without `+`.

A run or batch refuses a value out of its key's bounds before anything starts,
naming the key (`Error: scenario.walk_speed_ms must be finite and positive, not
0.0`): the config class checks it when it is built (#1815, #1816). Every
`timeout_seconds`, and the thresholds, distances and durations a check
compares against -- `min_speed_kmh`, `speed_check_delay_seconds`,
`turn_indicator_grace_seconds`, `light_switch_delay_seconds`,
`moving_speed_kmh`, `departure_speed_kmh`, `timeout_after_green_seconds`,
`standstill_seconds`, `s_margin`, `stop_duration`, `restart_speed_kmh`,
`trigger_distance_m`, `walk_speed_ms`, `cut_in_after_seconds` and
`hold_seconds` -- must be finite and positive: at 0 or below the check would
never fire, or always hold, and the run pass or fail on nothing. Speeds and
margins that may be 0 -- `npc_initial_speed_kmh`, `npc_speed_kmh`,
`ego_speed_kmh`, `speed_threshold`, `standstill_speed_mps`,
`stop_line_tolerance_m`, `pass_beyond_m`, `turn_back_within_m`,
`turn_back_ttc_s` and `turn_back_distance_m` -- must be finite and not
negative, and `stop_zone_min_m` must not be past `stop_zone_max_m`. A
`null` still turns an optional check off.

Each of these numbers must be a number: an int (Hydra reads
`scenario.timeout_seconds=10` as one) or a float. A boolean, which would pass
as 1 or 0, a quoted string (`'10'`) or a `null` where no check is optional is
refused, naming the key (`Error: scenario.timeout_seconds must be a number,
got None`, #1818); `max_steering_reversals` and `max_turn_indicator_changes`
must be whole numbers. The scenario's word settings are checked too: `cut_in`'s `side`
must be `left` or `right`, and `intersection_passing`'s `turn_direction`
`left`, `right`, `straight` or `null` (straight through, with no turn
action). The check builds the scenario's config class with the keys it
defines; a key it does not define is left to the build, which refuses it.

#### Intersection Passing

Tests ego vehicle traversal through an intersection. Verifies that the ego passes through the expected OpenDRIVE road sequence.

| Config | `turn_direction` | `timeout_seconds` | `min_speed_kmh` | Description |
|--------|------------------|-------------------|-----------------|-------------|
| `intersection_passing/straight` | `straight` | `15.0` | `3.0` | Drive straight through intersection |
| `intersection_passing/left_turn` | `left` | `20.0` | — | Left turn at intersection |
| `intersection_passing/right_turn` | `right` | `20.0` | — | Right turn at intersection |
| `intersection_passing/right_turn_sweep` | `right` | `60.0` | — | Right turn from a standstill; its sweep takes right turns behind a traffic-light stop line |
| `intersection_passing/straight_3npc` | — (`null`) | `10.0` | — | Straight with 3 NPC vehicles, no `TurnAction` |

**Scenario-specific parameters:**

| Parameter | Type | Default | Shipped | Description |
|-----------|------|---------|---------|-------------|
| `scenario.expected_route_lanelet_ids` | list[int] | `[460, 265]` | per config | Lanelet IDs the ego must traverse; a junction's lanelets are judged by the roads before and after it |
| `scenario.timeout_seconds` | float | `5.0` | per config (above) | Maximum scenario duration |
| `scenario.min_speed_kmh` | float \| null | null | per config (above) | Minimum ego speed (fail if below; positive). `null` disables check |
| `scenario.speed_check_delay_seconds` | float | `0.3` | — | Grace period before speed check starts |
| `scenario.turn_direction` | str \| null | null | per config (above) | `"left"`, `"right"` or `"straight"` registers a `TurnAction` for that direction; `null` registers none |
| `scenario.turn_indicator_before_m` | float \| null | null | — | From this many metres before the junction (the stop line at the start of the route's first lanelet) until it is on that lanelet, the ego must show the indicator of `turn_direction` (`left` or `right`); it fails once it has not for `turn_indicator_grace_seconds`. `null` disables the check |
| `scenario.turn_indicator_grace_seconds` | float | `0.5` | — | How long the indicator may be wrong before the run fails |
| `scenario.max_steering_reversals` | int \| null | null | — | Fail once the steering has turned back this many times since the start (the `steering_reversals` metric, [metrics.md](metrics.md#weaving)). `null` disables the check |
| `scenario.max_turn_indicator_changes` | int \| null | null | — | Fail once the turn indicators have changed this many times since the start, counted apart from the steering (the `turn_indicator_changes` metric). `null` disables the check |
| `scenario.npc_vehicles` | list | `[]` | 3 vehicles (`straight_3npc`), else — | NPC vehicle configurations (see below) |

**NPC vehicle configuration:**

```yaml
npc_vehicles:
  - spawn_lanelet_id: 100
    spawn_s: 5.0
    vehicle_type: vehicle.mini.cooper
    initial_speed_kmh: 10.0
```

An NPC's `spawn_s` must be finite and its `initial_speed_kmh` finite and not
negative; a run or batch refuses another value before anything starts
(`Error: scenario.npc_vehicles[0].initial_speed_kmh must be finite and at
least 0, got nan`).

**Behavior:**
- Sets all traffic lights to GREEN
- Registers a TurnAction when `turn_direction` is set -- `left`, `right` or `straight` (sets the ego's path through the junction via the TrafficManager); none when it is `null`
- Pass condition: ego traverses all expected OpenDRIVE road IDs (checked via StickyCondition + AndCondition)
- Fail conditions: timeout, ego destroyed, speed below threshold

#### Lane Change

Tests ego vehicle lane change maneuvers. Can test both successful lane changes and expected failures.

| Config | Direction | Expected Result | `timeout_seconds` |
|--------|-----------|-----------------|-------------------|
| `lane_change/left` | left | success | `10.0` |
| `lane_change/right` | right | success | `10.0` |
| `lane_change_fail/left` | left | fail (no adjacent lane) | `1.5` |
| `lane_change_fail/right` | right | fail (no adjacent lane) | `1.5` |

**Scenario-specific parameters:**

| Parameter | Type | Default | Shipped | Description |
|-----------|------|---------|---------|-------------|
| `scenario.direction` | str | `"left"` | per config (above) | Lane change direction: `"left"` or `"right"` |
| `scenario.expect` | str | `"success"` | per config (above) | Expected outcome: `"success"` or `"fail"` |
| `scenario.timeout_seconds` | float | `10.0` | per config (above) | Maximum scenario duration |

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

| Parameter | Type | Default | Shipped | Description |
|-----------|------|---------|---------|-------------|
| `scenario.light_switch_delay_seconds` | float | `3.0` | `5.0` | Delay before switching red to green |
| `scenario.timeout_seconds` | float | `5.0` | `8.0` | Maximum scenario duration; ignored when `timeout_after_green_seconds` is set |
| `scenario.timeout_after_green_seconds` | float \| null | null | — | The timeout counted from the green instead: `light_switch_delay_seconds` plus this, in place of `timeout_seconds`, overrides included. `null` uses `timeout_seconds` |
| `scenario.stop_line_distance_m` | float \| null | null | — | Moves the ego along its lane so that its front bumper starts this many metres before the stop line ahead. `null` leaves the spawn pose as it is |
| `scenario.stop_zone_min_m` | float | `0.0` | `0.0` | Near end of the stop zone: metres before the stop line, as the `stop_line_distance_m` measure reads it |
| `scenario.stop_zone_max_m` | float | `3.0` | `3.0` | Far end of the stop zone; standing still farther back until the green fails the run |
| `scenario.stop_line_tolerance_m` | float | `0.0` | `0.0` | How far the front bumper may pass the stop line on red before the run fails for running the red; that far past counts as at the line for the stop zone |
| `scenario.standstill_seconds` | float | `0.5` | `0.5` | How long the ego has to stand still in the stop zone, before the green, for it to count as stopped there (latched once it has) |
| `scenario.standstill_speed_mps` | float | `0.1` | `0.1` | Speed (m/s) at or below which the ego stands still |
| `scenario.departure_speed_kmh` | float \| null | null | `10.0` | Speed (km/h) the ego must reach after green to pass; `null` uses `moving_speed_kmh` |
| `scenario.moving_speed_kmh` | float | `1.0` | — | The departure speed when `departure_speed_kmh` is `null`; no longer judges the red phase |
| `scenario.merging_time_seconds` | float | `0.5` | — | Deprecated and unused; kept so configs that set it still load |

**Behavior:**
- Sets all traffic lights to RED initially
- Registers a TrafficSignalAction to switch to GREEN after `light_switch_delay_seconds`
- Fails as soon as the ego's front bumper crosses the stop line while the light is red
- At the green, fails unless the ego has stood still with its front bumper in the stop zone (`stop_zone_min_m`-`stop_zone_max_m` before the stop line), saying whether it stopped short of the zone, stopped in it too late, or had not stopped
- Pass conditions: ego stood still in the stop zone before the green AND reaches `departure_speed_kmh` during green

#### Temporary Stop

Tests that the ego vehicle stops at a stop line for a minimum duration, then restarts. The stop lines are read off the map: the spawn lanelet's own, or else those of the first step of its successors, up to three steps on, that has any. Run unexpanded, the ego spawns on lanelet 285 of the Nishi-Shinjuku map; the config's `sweep:` expands one case per stop line that no traffic light governs (see [Temporary Stop](scenarios/temporary_stop.md)).

| Config |
|--------|
| `temporary_stop/temporary_stop` |

**Scenario-specific parameters:**

| Parameter | Type | Default | Shipped | Description |
|-----------|------|---------|---------|-------------|
| `scenario.s_margin` | float | `5.0` | `5.0` | Arc-length margin around stop position (meters) |
| `scenario.speed_threshold` | float | `0.1` | `0.1` | Max speed considered as "stopped" (m/s) |
| `scenario.stop_duration` | float | `1.0` | `0.3` | Minimum consecutive seconds to remain stopped |
| `scenario.restart_speed_kmh` | float | `3.0` | `3.0` | Speed threshold for restart detection (km/h) |
| `scenario.timeout_seconds` | float | `30.0` | `12.0` | Maximum scenario duration |

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

Each scenario run generates output in `outputs/YYYY-MM-DD/HH-MM-SS/` (a run
started in the same second as another run or a batch gets `HH-MM-SS-1`, then
`HH-MM-SS-2`, ...: every run has a directory of its own, unless
`hydra.run.dir=` names one):

| File | Description |
|------|-------------|
| `{ScenarioName}.log` | CARLA native recording (replay format) |
| `{ScenarioName}_result.json` | Machine-readable result with condition statuses |
| `{ScenarioName}_coverage.json` | ODD and scenario coverage hits of the run (merge with `scenario-coverage`, see [Coverage](coverage.md)) |
| `{ScenarioName}_gate.json` | A logical scenario's gate record: its case, outcome and the ego's drive in the route's frame, which `scenario-gate` judges the scenario's gate from (a document with a gate only, see [Gates](gate.md)) |
| `{ScenarioName}_odd.tar.gz` | The run's ODD as the files it was read from (an ODD read from OpenODD YAML; openodd's transmission archive, named by its revision; see [Versions of an ODD](odd.md#versions-of-an-odd)) |
| `{ScenarioName}_requirements.lock` | What the run ran with: every installed distribution pinned, in the requirements format (`pip install -r`), headed by the framework, the scenario's package, Python and the platform. A distribution from git is pinned to its commit; an editable one (a checkout) names its directory, commit and whether it had uncommitted changes |
| `{ScenarioName}_environment.json` | The same, structured; `_result.json` carries its summary (`environment`: the framework, the scenario's package, key dependencies, and the lock's id -- the SHA-256 of what it pins -- so runs on one environment can be found by it) |
| `{ScenarioName}.trajectory.jsonl` | The run, tick by tick, and the map, plan, conditions and verdict ([Trajectory Log](trajectory_log.md)); make it a Rerun recording to see the run ([Seeing a run](#seeing-a-run)), or a T4Scene for tier4/e2e-devkit with `scenario-t4scene` ([T4 Scene](t4_scene.md)) |
| `{ScenarioName}.rrd` | The Rerun recording of the trajectory log, with `rerun.enabled=true` ([Seeing a run](#seeing-a-run)) |
| `{ScenarioName}.sensors.rrd` | The driver's sensors recorded during the run (`rerun.lidar`, `rerun.cameras`); merged into `{ScenarioName}.rrd` and removed, so only left behind by a run or merge that did not finish ([Sensors in the recording](#sensors-in-the-recording)) |
| `{ScenarioName}.metrics.rrd` | The run's metrics as scalars, written for a converter that does not draw them and merged into `{ScenarioName}.rrd` and removed, so only left behind by a run killed during the merge ([Metrics in the recording](#metrics-in-the-recording)) |
| `{ScenarioName}.t4scene.pb`, `{ScenarioName}.t4scene.map.pb` | The trajectory log as a T4Scene and the map it names, with `t4scene.enabled=true` ([T4 Scene](t4_scene.md)) |
| `{ScenarioName}.t4/` | The run as a T4 dataset (a directory, its version `0` inside), with `t4.enabled=true` ([T4 Dataset](t4_dataset.md)) |
| `{ScenarioName}.sumo/` | The SUMO backend's log, generated demand and, with `fcd_output`, its floating car data (`fcd.xml`) and clock (`clock.csv`), with `traffic=sumo` ([Traffic Backends](traffic_backends.md)) |
| `batch_results.json` | Batch summary (glob/batch mode only) |

**Result JSON format:**

`{ScenarioName}_result.json` holds the result (`passed`, `message`,
`elapsed_seconds`, `condition_statuses`), then `rerun_recording` (the
`.rrd` made of the run, with `rerun.enabled=true`) and `failure` (on a failed
run, below) when they apply, then `environment` (the summary of
`_environment.json`) and `traffic` (what drove the traffic: the backend's
name, its settings and seed, and with SUMO how many vehicles it simulated at
each level when the run ended). Either of the last two is left out when it
could not be read. The scenario's name is not in it: the file is named after
it, and `batch_results.json` adds it to each of its entries.

On a pass, `message` is the message of the pass condition that fired -- here
an `AndCondition` of two `StickyCondition`s, which joins its children's
messages with ` AND ` -- or `Scenario completed successfully` when
`is_done()` ended the run. When a fail condition ends the run, it is that
condition's message.

`condition_statuses` has one entry per registered condition, pass conditions
first, each labelled `pass[i](label)` or `fail[i](label)`: whether it was
satisfied when the run ended, its message (`not triggered` for a fail
condition that was not, `not yet satisfied` for a pass condition), its class,
its role, and its `details` (an `AndCondition`'s are its children's statuses,
cut short here). A run of `intersection_passing/straight`:

```json
{
  "passed": true,
  "message": "Entity 'Ego' is on road '5' (any lane) (s=12.40, t=-1.75) at 1.10s AND Entity 'Ego' is on road '12' (any lane) (s=3.05, t=-1.80) at 3.42s",
  "elapsed_seconds": 3.42,
  "condition_statuses": [
    {
      "label": "pass[0](ego_on_road_5_sticky AND ego_on_road_12_sticky)",
      "satisfied": true,
      "message": "Entity 'Ego' is on road '5' (any lane) (s=12.40, t=-1.75) at 1.10s AND Entity 'Ego' is on road '12' (any lane) (s=3.05, t=-1.80) at 3.42s",
      "condition_type": "AndCondition",
      "role": "pass",
      "details": {"operator": "AND", "children": ["..."]}
    },
    {
      "label": "fail[0](speed_check_delay AND ego_min_speed)",
      "satisfied": false,
      "message": "not triggered",
      "condition_type": "AndCondition",
      "role": "fail",
      "details": {"operator": "AND", "children": ["..."]}
    },
    {
      "label": "fail[1](scenario_timeout)",
      "satisfied": false,
      "message": "not triggered",
      "condition_type": "TimeoutCondition",
      "role": "fail",
      "details": {"timeout_seconds": 15.0}
    },
    {
      "label": "fail[2](ego_existence)",
      "satisfied": false,
      "message": "not triggered",
      "condition_type": "EntityExistenceCondition",
      "role": "fail",
      "details": {"entity_name": "Ego"}
    }
  ],
  "environment": {
    "id": "731d9cc2a8447ef6f9230be81346e8e8297d9e85eb4570bdb68ce91964472fdf",
    "python": "3.12.3",
    "platform": "Linux-6.8.0-60-generic-x86_64-with-glibc2.39",
    "framework": {
      "name": "autoware-carla-scenario",
      "version": "23.1.24",
      "source": {"kind": "index"}
    },
    "scenario": {
      "name": "autoware-carla-scenario",
      "version": "23.1.24",
      "source": {"kind": "index"}
    },
    "packages": {
      "openodd-py": "0.18.0",
      "typesafe-carla": "0.8.1",
      "simple-lanelet2": "1.2.3",
      "hydra-core": "1.3.7",
      "numpy": "2.5.3"
    }
  },
  "traffic": {
    "backend": "traffic_manager",
    "port": 8100,
    "random_seed": 0,
    "follow_visibility": false,
    "visibility_m": null
  }
}
```

`environment.scenario` names the distribution the scenario's class comes
from, as `framework` does: the framework itself for a built-in scenario like
this one, a scenario package's own otherwise, `null` when no installed
distribution is found for it. A `source` other than `index` says where the
distribution came from (`vcs` with its URL and commit, `editable` with its
directory and commit, ...).

**Why a run failed:**

A failed run's result has a `failure`: the fail condition that ended it, and a
*category* that names what went wrong the same way in every scenario, where
the condition's label is the scenario author's. A batch, a sweep or two
versions of a driver can then be compared by what went wrong
(`batch_results.json` carries it for every case, and the summary prints it).

```json
"failure": {
  "condition": "fail[1](scenario_timeout)",
  "category": "progress",
  "at": 25.0,
  "message": "Timeout after 25.00s (limit: 25.0s)",
  "detail": "stalled",
  "also_triggered": []
}
```

| Category | What went wrong |
|---|---|
| `collision` | The ego hit something |
| `traffic_rule` | It broke a traffic rule: ran a red light, drove against the traffic |
| `signalling` | It did not signal what it did (no turn indicator) |
| `lane_keeping` | It left its lane or the road |
| `stability` | It did not drive steadily: sawed at the steering, weaved, flicked its indicators |
| `safety_margin` | It came too close: time to collision, headway, distance |
| `progress` | It did not get where the scenario asked in time |
| `driver` | The driver ended the run itself |
| `infrastructure` | The run could not go on: the ego vanished, or the run raised |
| `other` | None of these, or a condition that does not say |

A scenario gives a fail condition its category when it registers it:

```python
from autoware_carla_scenario.failure import FailureCategory

self.register_fail_condition(
    PersistentCondition(AndCondition([...]), 0.5, label="no_left_indicator"),
    category=FailureCategory.SIGNALLING,
)
```

Without one, the condition says (`BaseCondition.failure_category()`): a
collision condition is `collision`, a timeout `progress`, a time-to-collision
or headway condition `safety_margin`, the ego's existence `infrastructure`,
and a `MetricCondition` what its metric reads (the
collision metrics `collision`, `offroad` `lane_keeping`, `wrong_lane`
`traffic_rule`, the weaving ones `stability`, the CommonRoad ones
`safety_margin`); it is judged once the run is over, and fails it at the time
its metric's NG time ran out ([Metrics](metrics.md#judging-a-run-on-a-metric)). A sticky or persistent
condition says what it wraps. A combination of conditions (AND, OR, NOT) says
nothing, and the run's failure is `other`: give it a category.

> **A persistent condition never fires around a failure-only condition.**
> `PersistentCondition` counts only the ticks its child *passes*, and resets on
> a failure. `CollisionCondition`, `OffRoadFractionCondition`,
> `EntityExistenceCondition` and `TimeoutCondition` never pass: they report a
> failure (`passed=False`) or nothing (their classes set
> `REPORTS_ONLY_FAILURE`). So `PersistentCondition(OffRoadFractionCondition(...), 0.5)`
> -- "off the road for 0.5 s" -- never fires, and never fails the run.
> Building one logs a warning, and the scenario editor's validator warns on
> the Persistent card. What to write instead depends on the condition:
>
> - `CollisionCondition` and `TimeoutCondition` keep their failure once they
>   have fired (a collision is cached, a timeout stays past its limit), so
>   there is nothing to wait out: register them on their own.
>   `PersistentCondition(NotCondition(child), d)` would only fail the run `d`
>   later, not say that anything held for `d`.
> - `OffRoadFractionCondition` and `EntityExistenceCondition` check again on
>   every tick, so a plain `NotCondition(child)` turns their failure into a
>   pass the persistent condition counts, and it fires once the failure has
>   held for the duration (a NOT says no category: give it one).
> - To say a state has *not* held for a while, wrap the condition for that
>   state in `NotCondition(..., holds_when_pending=True)`, as the
>   [cut-in example](scenarios/cut_in.md#staying-on-the-lanes-ahead) does.
>
> Only a direct child is recognised: a failure-only condition under a sticky,
> AND or OR condition inside the persistent one draws no warning.

`detail` tells a `progress` failure apart by how the ego ended the run:
`never_moved` (it never drove 1 m), `stalled` (it had stood
still for 3 s or more: waiting for something that never cleared) or `moving`
(it was driving, but not where the scenario's pass condition asked). The other
fail conditions that held when the run ended are in `also_triggered`. The
driver ending the run is `driver`, and a case that raised on every attempt is
`infrastructure`.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All scenarios passed |
| 1 | One or more scenarios failed or errored, or configuration/runtime error (a batch refusing a config value a case would refuse prints `Error: <case>: <key> ...` before anything is built) |
| 2 | Refused before anything ran: the scenario does not type-check, an extra it asks for (`rerun`, `t4`) is not installed, a `--resume-from` that is not a whole number of at least 1, or (in a lanelet sweep) a config value its runs would refuse, such as a `t4scene.step_s` that is not a multiple of the tick or a `sweep.resume_from` that is not a whole number |
| 130, 143 | Stopped by Ctrl-C (130) or `SIGTERM` (143), after the run stopped what it started (its CARLA server, its Autoware stack); a sweep first stops the case it is running |

The same codes hold however the cases run: a single run, a glob batch, a
`--multirun` (each case of the scenario's sweep in one batch,
`hydra/sweeper=lanelet_constraint`, or Hydra's own sweeper over
comma-separated values) and `launcher=slurm`, which waits for the array and
collects it. A sweep runs every case before it exits, so 1 means at least
one case failed; the summary names how many. A refusal ends a sweep at once
with 2, since every case after it would be refused too. A run with no case
to run (nothing expanded, or `--resume-from` past the last case) exits 1:
exit 0 always means something ran and passed. In a lanelet sweep, a case
whose result JSON is unreadable or has no boolean `passed` (a missing key,
`"false"`, `1`) has no verdict and counts as failed. The exceptions:

- Under Hydra's own sweeper (`scenario --multirun key=a,b`, no
  `hydra/sweeper=`), a case that raises an exception ends the sweep at once
  with 1, as Hydra does; a case that runs and fails does not.
- `launcher=slurm` with `--no-wait` exits 0 once the batch is submitted; its
  verdict is what `scenario-slurm wait` or `scenario-slurm collect` exit with.
  Interrupted while it waits, it cancels the array and exits with the verdict
  of what it collects (`launcher.cancel_on_interrupt`, the default), or exits 130
  and leaves the array running.

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
| `hils` | ECUs under test drive the ego, through the resident HILS bridge `scenario-hils` (`hils.host`, `hils.port`; [HILS Bridge](hils.md)) |

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
uv run scenario scenario=cut_in/left ego.entity=carla_driver \
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

## `scenario-hils` - Streaming the Ego to ECUs (HILS)

A resident process on the HILS PC that follows a run's ego in CARLA, sends ECUs
under test their sensors' and buses' traffic (CAN, GNSS, LiDAR packets, camera
frames) through hilspy, initializes and finalizes them around each scenario,
and hands their commands back to a run with `ego.entity=hils`; configured by
one YAML file:

```bash
uvx --from 'autoware-carla-scenario[hils]' scenario-hils --example > hils.yaml
uvx --from 'autoware-carla-scenario[hils]' scenario-hils hils.yaml
```

See [HILS Bridge](hils.md).

---

## `detect-no-3d-model` - 3D Model Detection

Detects lanelets that do not have a matching 3D ground model in CARLA. Useful for identifying areas where vehicle ground-projection ray casting will fail. Requires a running CARLA server with the target map loaded.

### How It Works

1. Reads the map configuration YAML and resolves its map files the way `uv run scenario` does (see [Input YAML Requirements](#input-yaml-requirements))
2. Connects to the CARLA server and loads the map -- installing the OpenDRIVE first when `map.overwrite_xodr` is set, by name otherwise -- and, for a map that carries no OpenDRIVE, reads it back out of the loaded world into the map cache, as a run does
3. For each lanelet's center point, casts a vertical ray (upward and downward from estimated z) to detect ground geometry
4. Reports lanelets where the ray does not hit any 3D surface
5. Writes the detected IDs to the YAML configuration's `no_3d_model_lanelet_ids` field

The IDs go into `no_3d_model_lanelet_ids` in the YAML's `map:` section when it has one, and at the top level otherwise (the same place the map settings are read from). An existing value is replaced whatever its style (a block list, a flow list such as `[1, 2]`, an empty value, `null`); a missing key is added after the section's last entry. The file is edited as text, so comments, `${...}` interpolations and line endings are kept, and it is replaced atomically (a symlinked YAML is written through to its target). The edited text is parsed again first: if anything other than that one value would change -- or the section cannot be edited as text, e.g. `map: {name: X}` -- the tool exits with an error, prints the detected IDs, and leaves the file as it was.

### Usage

Run it from the repository root: the example map configs live under
`autoware_carla_scenario/src/autoware_carla_scenario/examples/conf/map/`, and `nishishinjuku.yaml` names its files relative to the root
(`data/nishishinjuku.osm`).

```bash
CONF=autoware_carla_scenario/src/autoware_carla_scenario/examples/conf/map

# Basic detection using map config YAML
uv run detect-no-3d-model $CONF/nishishinjuku.yaml

# A map named only by map.source (its OpenDRIVE is read back from CARLA)
uv run detect-no-3d-model $CONF/town10hd_opt.yaml

# Connect to a remote CARLA server
uv run detect-no-3d-model $CONF/nishishinjuku.yaml --host 192.168.1.100 --port 3000

# Match a run that overrides entity.ground_projection_ray_distance_* (hilly terrain)
uv run detect-no-3d-model $CONF/nishishinjuku.yaml --ray-upper 10.0 --ray-lower 20.0

# Longer timeout for slow connections
uv run detect-no-3d-model $CONF/nishishinjuku.yaml --timeout 30.0

# Debug logging
uv run detect-no-3d-model $CONF/nishishinjuku.yaml -v
```

### Command-Line Options

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `yaml_path` | - | Yes | - | Path to map configuration YAML (map settings at the top level or under `map:`; see below) |
| `--host` | - | No | `localhost` | CARLA server hostname |
| `--port` | - | No | `2000` | CARLA server port, 1-65535 |
| `--ray-upper` | - | No | `5.0` | Ray search range above z estimate (meters); the runner's `entity.ground_projection_ray_distance_upper` |
| `--ray-lower` | - | No | `10.0` | Ray search range below z estimate (meters); the runner's `entity.ground_projection_ray_distance_lower` |
| `--timeout` | - | No | `10.0` | CARLA client timeout (seconds); positive and finite |
| `--verbose` | `-v` | No | false | Enable DEBUG logging |

The ray defaults are the runner's shipped `entity.*` values, so the tool flags
only lanelets a run's own ground projection would miss. If your runs override
`entity.ground_projection_ray_distance_upper`/`_lower`, pass the same values
here, or the tool may flag lanelets those runs project onto (or miss ones they
cannot).

`--ray-upper` and `--ray-lower` must be finite and non-negative, with a
positive sum, as the runner's keys must be: a ray of no length hits nothing, so
every lanelet would be flagged and written to the YAML. The tool refuses such
values, a `--timeout` that is not positive and finite, and a `--port` outside
1-65535, with a usage error
(exit 2) before it connects to CARLA or reads the YAML.

### Input YAML Requirements

The map settings are resolved exactly as the scenario runner resolves the `map` config group: `lanelet2_path` and `xodr_path` (`~` and `${oc.env:...}` work; a relative path is taken from the working directory), `directory`, `source`, `projector_type` and `overwrite_xodr` (see [Map Configuration](#map-configuration)). A map given by files needs at minimum:

```yaml
name: NishishinjukuMap          # CARLA map name
xodr_path: path/to/map.xodr     # Path to OpenDRIVE file
lanelet2_path: path/to/map.osm  # Path to Lanelet2 file
```

A map given by `source` or `directory` needs only that key (`name` defaults to the map directory's name); when it ships no OpenDRIVE, the tool reads the road network back out of the loaded CARLA world and caches it, as a run does. With `overwrite_xodr: true` the OpenDRIVE is installed into the CARLA asset named by the `<MAP_NAME>_PATH` environment variable before the map is loaded; otherwise CARLA's own road network is kept.

These keys may also sit under a `map:` section (as in the example `nishishinjuku.yaml`, a `# @package _global_` config); `no_3d_model_lanelet_ids` is then written there too.

A config that cannot be read, is not valid YAML, or does not name a usable map is reported as a single `detect-no-3d-model: error: ...` line with exit status 1, before CARLA is contacted.

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
# Start with default settings (127.0.0.1:9000, current directory)
uv run viewer

# Custom host and port
uv run viewer --host 127.0.0.1 --port 8080

# Scan another project: the directory containing outputs/ and multirun/
uv run viewer --base-path /path/to/project

# All options
uv run viewer --help
```

Then open `http://localhost:9000` (or the configured host/port) in a browser.

| Option | Environment variable | Default |
|--------|----------------------|---------|
| `--host` | `VIEWER_HOST` | `127.0.0.1` |
| `--port` | `VIEWER_PORT` | `9000` |
| `--base-path` | `VIEWER_BASE_PATH` | the working directory |
| `--fetch-rerun-web-viewer [--from TGZ] [--version V]` | — | fetch the Rerun web viewer and exit ([below](#seeing-a-run-in-the-viewer)) |

Each option defaults to its environment variable, so a flag on the command line
wins over the environment. An unknown option is an error (exit 2) and serves
nothing, as is a port (`--port` or `VIEWER_PORT`) that is not an integer within
1-65535: 0, which would have the system pick a port the viewer does not report,
is refused too.

Ctrl-C or `SIGTERM` stops the viewer and the run it started from the Run form.
An open browser tab keeps the server waiting for it for at most 5 s. The
running `uv run scenario` job, which runs in a process group of its own, then
gets `SIGTERM`, and `SIGKILL` if any of that group is left 10 s later. That job
is recorded as failed, and the jobs after it do not start.

!!! warning "The viewer listens on this machine only by default"
    The viewer has no authentication, and its API starts scenario runs
    (`POST /api/run`) and SLURM submissions (`/api/slurm/...`). It therefore
    binds `127.0.0.1` unless told otherwise. It used to bind `0.0.0.0`, every
    interface; to serve other machines as before, pass `--host 0.0.0.0` (or set
    `VIEWER_HOST=0.0.0.0`) on a network you trust. The viewer prints a warning
    when it listens on anything but a loopback address.

    Binding loopback keeps other machines out, not a web page open in your own
    browser, which can send requests to `127.0.0.1` too. So the viewer also
    refuses them:

    * A `POST`, `PUT`, `PATCH` or `DELETE` from another site is refused with
      403: one whose `Sec-Fetch-Site` header is `cross-site` or `same-site`,
      or whose `Origin` header is not the viewer's own scheme, host and port
      (`Origin: null` included). A request with neither header, as curl or a
      script sends it, is served.
    * One `GET` can make something: a case's recording
      (`/recording/<session>/<index>.rrd`) is converted from its trajectory
      log, and written, when there is none yet. A request from another site
      (the same `Sec-Fetch-Site` and `Origin` rules) is then refused with 403
      and converts nothing. A recording already made is served to any page
      (a Rerun viewer hosted elsewhere may load its URL); the
      viewer's pages make it first (`POST /api/recording/...`), and the
      *Download .rrd* link, a typed URL (`Sec-Fetch-Site: none`) and curl
      are served as before. Every other `GET` only reads files on disk or
      queries SLURM. Like the editor's (see the scenario editor's guide),
      this rests on the browser sending `Sec-Fetch-Site`, which current
      browsers do for loopback addresses but not for plain HTTP on another
      address, and older browsers never send.
    * A JSON route (`/api/run`, `/api/run/preview`,
      `/api/slurm/.../resubmit`) refuses a body that is not sent as
      `Content-Type: application/json` with 415, so a `text/plain` body, which
      a page may send without asking the server first, is never read as JSON.
    * Against DNS rebinding, a request whose `Host` header is not
      `localhost`, a loopback address (`127.0.0.1`, `::1`) or the `--host` the
      viewer is bound to, on any port, is refused with 400. Bound to every
      interface (`0.0.0.0` or `::`), the viewer was asked to answer whatever
      name it is reached under, and checks no `Host`; the `Origin` check still
      applies.
      A page served from a rebound name then passes both checks, so a
      wildcard bind gives up the rebinding protection, even inside a container
      published only on `127.0.0.1`. Behind a reverse proxy or a remote port
      forwarder that serves it under another name or over HTTPS, the viewer
      refuses requests unless it is bound to every interface; an SSH `-L`
      tunnel to a loopback port keeps working.

### Seeing a run in the viewer

A scenario's page shows its run in Rerun's web viewer, embedded in the page:
the case's trajectory log made into a Rerun recording ([Seeing a run](#seeing-a-run)).
A recording already beside the log (`X.rrd` beside `X.trajectory.jsonl`) is
shown as it is; otherwise the viewer makes it the first time the page is
opened (in a worker thread, some tens of milliseconds for a case) and keeps it
beside the log, or, where the session's directory cannot be written, in its
cache directory. A recording beside the log that is older than the log (after a
copy that does not keep file times, `cp -r`, `scp`, `rsync -r`) is made again
into the cache directory, never over the one beside the log, which may hold the
run's sensors. Making one needs the converter in the viewer's environment:

```bash
pip install 'autoware-carla-scenario[rerun]'    # or: pip install autoware-carla-scenario-rerun
```

Without it the page says so, and still shows a recording already made.
The page's *Download .rrd* link is the recording itself, for the native
viewer (`rerun X.rrd`).

The web viewer is the `@rerun-io/web-viewer` package of the version the
converter writes recordings with (`RERUN_VERSION`, 0.38.1 for converter 0.2;
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
plot of the drawn cases (`openodd.CasePlot`) is shown too. A SLURM batch is
shown before it is collected, from what its tasks have recorded: a case
without an outcome yet is listed as PENDING, and a task that crashed recorded
none, so its case stays PENDING until the batch is collected. A case whose
recorded outcome holds no verdict, or a result that cannot be reported (no
string `message`, no number `elapsed_seconds`, `condition_statuses` or a
`failure` that cannot be read), shows as an error, as `scenario-slurm
collect` reports it.

### Comparing runs

`/compare` (the navigation bar's *Compare*, or *Compare with past runs* on a
session's page) compares two or more sessions of the same scenarios
(`?s=batch/2026-10-18/16-09-52&s=...`), the oldest the baseline:

- **Cases** are matched by what they ran -- their sweep overrides (less
  `++odd_case`, which names a case's position), else the ODD values they
  were drawn at -- not by their index, so a case keeps its match when a
  sweep's order or size changes. Cases of one session that share a key (a
  sweep that draws the same ODD values twice, as one over categorical
  buckets can) are all kept: each run is listed, and a session with more
  than one run of a case shows how many of them passed (`1/2 passed`).
  Each case is marked by how its verdict changed, a session's verdict being
  `pass` when every one of its runs passed, `fail` when none did (a fail and
  an error are both not passing), and mixed otherwise:
  - flaky (amber) when a session's verdict is mixed, whichever session it
    is, even when that session alone ran the case; or when the first and
    last sessions that ran it agree but a session between them does not;
  - else `pass → fail` (a regression, listed first and in red) when every
    run of the first session that ran it passed and no run of the last did,
    and `fail → pass` (fixed) the other way;
  - else the same, or run by one session only (new, or gone).

  Beside it, the change of each metric (the mean of the last session's runs
  less the first's; bold when more than 5%), each run's verdict linking to
  its page, and *side by side*: the first run of the first session and the
  last run of the last in two Rerun viewers on one page
  (`/compare/runs?run=type/date/time/index&run=...`).
- **ODD cells**: each session's passes and runs in each cell of the two
  attributes picked (the buckets the cases were drawn in), red where the last
  session passes fewer of them, green where more; and openodd's matrix of
  each session beside the others. Every run is counted, a repeated case's
  each run, so a session's cells add up to its pass rate's runs (but for the
  cases drawn in no bucket of the attributes picked).
- **Pass rates** of each session.

A SLURM batch can be compared before it is collected, from what its tasks
have recorded: a case without an outcome yet is PENDING and is left out of
the comparison, as if that session had not run it (no regression, and not in
its pass rate), and each session's summary says how many of its cases are
pending (`1/1 passed · 1 pending (not collected)`). A task that crashed
recorded no outcome, so its case stays pending until the batch is collected.
A case whose recorded outcome holds no verdict, or a result that cannot be
reported, is compared as an error.

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
batch's page once it is submitted. The command is run with no input and in
a process group of its own; one that has not ended after 30 minutes (it runs
the batch's checks before `sbatch`) is killed with its whole group, and the
submission is marked failed (exit status 124) with why, so the next submit is
taken. Stopped during `sbatch`, it leaves the batch's submission record in
place, as a timed-out `sbatch` does ([Running a Batch on
SLURM](slurm.md#how-it-works)): if `squeue --me` lists the job, it runs and the
batch's page collects it. Where `sbatch` is not on the viewer's `PATH` the option is disabled,
and says so.

A SLURM batch's page (`outputs/<date>/<time>/` with a `slurm/` directory,
whoever submitted it) follows it through `squeue` and the records its tasks
write: each case passed, failed, errored, running, pending or missing (with
why: its task's state and error), each submission's job and tasks (state,
host, cases done, peak memory), and the cases as they land, before the batch
has a summary. A submission with no job id names no job: while its `sbatch`
is in flight it reads *being submitted* and its cases pending; once that
`sbatch` timed out it reads *sbatch timed out*, with when, and its cases
unknown, since the job may exist all the same (check `squeue --me`); a record
its submitter left behind more than 600 s ago reads *no job*, and its cases
missing. Its buttons run `scenario-slurm` on the batch: *Resubmit
missing* (the cases with no outcome, including a result with no boolean
`passed` or one that cannot be reported -- no string `message`, no number
`elapsed_seconds`, unreadable `condition_statuses` or `failure` -- which
shows as failed, or that errored), *Resubmit failed too*
(`--failed`), with the array parallelism and cases per task to use, *Collect*
(the summary and `batch_results.json`) and *Cancel jobs*. Each shows the
command's output, marked failed when the command failed: *Collect* is marked
done when it collected the batch, even with cases that failed (exit 1 with a
new `slurm/collected.json`), and failed when it could not collect it (an
`Error:` such as an unreadable manifest or an unwritable batch directory, a
traceback, or a `scenario-slurm` that could not be started or did not end in
five minutes).

`VIEWER_SCENARIO_COMMAND` replaces `uv run scenario` as the command a SLURM
batch is submitted with (`.venv/bin/scenario`, say, where `uv run` must not
sync the environment).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIEWER_BASE_PATH` | Current working directory | Base path for scanning scenario result files (`--base-path` wins) |
| `VIEWER_HOST` | `127.0.0.1` | Server bind address (`--host` wins) |
| `VIEWER_PORT` | `9000` | Server listen port (`--port` wins) |
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
| `POST` | `/api/run` | Start a scenario execution in the background (409 `{"status": "busy"}` while a local run is still in progress) |
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
per line, without `${...}` or a backslash continuing it; `env`: `NAME=value`
lines, or a mapping, each value exported as written, so without `${...}` either
-- an OmegaConf interpolation in the override the viewer passes on).

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
| `VIEWER_HOST` | viewer | Viewer server bind address (default `127.0.0.1`; `--host` wins) |
| `VIEWER_PORT` | viewer | Viewer server listen port (`--port` wins) |
| `VIEWER_CACHE_DIR` | viewer | The viewer's cache (the Rerun web viewer's copy, recordings it could not keep beside their logs) |
| `VIEWER_RERUN_WEB_VIEWER` | viewer | Where the Rerun web viewer is loaded from (a directory, a URL, or `none`) |
| `VIEWER_RERUN_MAP_DIR` | viewer | Where the Rerun converter looks maps up by name |
| `VIEWER_SCENARIO_COMMAND` | viewer | The command a SLURM batch is submitted with (default `uv run scenario`) |
| `SCENARIO_EDITOR_DRAFTS` | scenario editor | Directory holding scenario drafts |
| `SCENARIO_EDITOR_EXPORT_DIR` | scenario editor | Staging area for exported package archives |
| `SCENARIO_EDITOR_MAP_ROOTS` | scenario editor | Directories a Lanelet2 map may be served from |
| `SCENARIO_EDITOR_HOST` | scenario editor | Editor server bind address (default `127.0.0.1`; `--host` wins) |
| `SCENARIO_EDITOR_PORT` | scenario editor | Editor server listen port (`--port` wins) |
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
# 1. Detect lanelets without ground geometry (from the repository root)
uv run detect-no-3d-model autoware_carla_scenario/src/autoware_carla_scenario/examples/conf/map/nishishinjuku.yaml

# 2. The detected IDs are written to no_3d_model_lanelet_ids in the YAML
# 3. Now run scenarios — sweep will skip these lanelets
uv run scenario scenario='intersection_passing/*' map=nishishinjuku
```

### Use Case 5: Browse Results in the Web UI

```bash
# 1. Run scenarios
uv run scenario scenario='*/*'

# 2. Launch the viewer from the project directory: it scans outputs/ and
#    multirun/ there (from elsewhere: --base-path <dir containing outputs/>)
uv run viewer

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

**Cause**: Glob pattern matched scenarios configured for different maps --
another `map.name`, or the same name with another `map.source`,
`map.directory`, `map.lanelet2_path`, `map.xodr_path`, `map.projector_type` or
`map.overwrite_xodr`. The message names the key that differs and the cases
setting each value.

**Solution**:
- Use `map=<name>` override to force all scenarios to use the same map, or set
  the differing key on the command line (e.g. `map.lanelet2_path=...`)
- Or narrow the glob pattern to match scenarios for one map only

### Ego vehicle falls through the ground

**Cause**: The spawn lanelet has no corresponding 3D ground geometry in CARLA.

**Solution**:
- Run `detect-no-3d-model` to identify problematic lanelets
- Add them to `no_3d_model_lanelet_ids` in the map config
- Increase `entity.ground_projection_ray_distance_lower` for deeper ray casting
  (and pass the same `--ray-lower` to `detect-no-3d-model`)
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
