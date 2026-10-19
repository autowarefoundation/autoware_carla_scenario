# Running a Batch on SLURM

A batch (`scenario 'scenario=<glob>'`, and with `--multirun` every case of each
scenario's sweep) runs its cases one after another, in one process, on one
CARLA. On a machine or cluster with [SLURM](https://slurm.schedmd.com/), the
same batch can run its cases as the tasks of a SLURM array job instead, several
at once, each on a CARLA of its own:

```bash
uv run scenario --multirun 'scenario=cut_in/*' +sweep.odd_sample.count=24 \
  launcher=slurm launcher.array_parallelism=2 launcher.gres=shard:1 launcher.mem=12G
```

**It is off unless a run selects it.** The `launcher` config group is not part
of the default config: a run that does not name it composes the same config as
before, and the batch runs here, exactly as it always has. `launcher=local`
says the same explicitly.

The batch writes what a local batch writes, where a local batch writes it
(`outputs/<date>/<time>/`), under the same names: every case's
`{Scenario}_result.json`, `{Scenario}.trajectory.jsonl`, coverage, ODD archive,
environment and CARLA recording, and `batch_results.json` with the summary
table at the end. A case's outputs do not depend on which task ran it or when.

## How it works

On the machine you run `scenario` on (a SLURM submit host):

1. The batch is expanded, labelled, built and statically checked exactly as a
   local batch is: the ODD sampling, the case labels, the [static
   check](typecheck.md) (once a config shape), the map fetched. A config error
   is refused here, before anything is submitted.
2. A manifest is written into the batch's output directory
   (`slurm/manifest.json`): every case's scenario and overrides, its resolved
   config, and the file stem it is to be named by.
3. `sbatch --array=0-(N-1)%K` submits the cases, `launcher.cases_per_task` to a
   task, at most `launcher.array_parallelism` tasks at once.
4. By default `scenario` follows the array (`squeue`, and the outcome each task
   records) until it ends, then prints the summary and writes
   `batch_results.json`, as a local batch does, and exits 0 when every case
   passed. Ctrl-C cancels the batch's jobs, waits (up to 60 s) for their tasks
   to leave the queue, and collects. With `--no-wait`
   (`launcher.wait=false`) it submits and exits, and a small job that runs once
   the array has ended (`--dependency=afterany`) collects the batch; should
   that job not be submitted, `scenario` says so, with the array's job id, and
   exits non-zero (`scenario-slurm wait` collects the batch instead).

Each task, on its node:

1. takes a slot of ports no other task on the node holds and nothing listens on
   ([Ports](#ports));
2. composes each of its cases' configs from the manifest's scenario and
   overrides, and refuses to run one that no longer resolves to the config it
   was submitted with (a job that does not run in the submitting environment);
3. builds them (not checked again: the submitting process did), starts a
   CARLA on its ports and runs its cases one after another through the same
   queue a local batch uses, writing their files into the batch's directory
   under the names a local batch gives them;
4. records each case's outcome as soon as it has run
   (`slurm/results/<case>.json`), and its own progress in
   `slurm/tasks/<submission>-<task>.json` (host, ports, state, peak resident
   memory);
5. stops its CARLA however it ends: at the end, on `scancel`, on the signal SLURM
   sends `launcher.signal_grace_seconds` before `launcher.time` runs out, and
   -- through the job script -- when the worker itself was killed.

## Following, collecting, resubmitting

`scenario-slurm` takes the batch's output directory:

```bash
scenario-slurm status   outputs/2026-10-18/12-00-00   # cases, jobs, tasks
scenario-slurm wait     outputs/2026-10-18/12-00-00   # follow, then collect
scenario-slurm collect  outputs/2026-10-18/12-00-00   # summary, batch_results.json
scenario-slurm resubmit outputs/2026-10-18/12-00-00   # the cases with no result
scenario-slurm cancel   outputs/2026-10-18/12-00-00   # scancel the batch's jobs
```

A case that has no outcome -- its task was cancelled, ran out of time, was
killed for memory, or its node failed -- is reported as `ERROR` with a message
saying so, and `collect` lists them. `resubmit` submits again the cases with no
outcome and those that raised on every attempt (`ERROR`); `--failed` adds the
ones that ran and failed, `--cases a,b` names them (labels or indices; a case
named twice runs once), and `--array-parallelism`/`--cases-per-task` change
how they are spread. A case run again replaces its earlier files, so
`resubmit` refuses while a job of the batch is queued or running, or while
`squeue` cannot tell whether one is (`--force` submits all the same).
`collect` can be run any number of times.

`sacct` is not needed: the batch is followed through `squeue` and the files the
tasks write.

The result viewer does the same from its pages: its run panel submits a
batch with `launcher=slurm`, and a SLURM batch's page shows its cases, jobs
and tasks as they run and resubmits, collects or cancels it
([Running through SLURM](usage.md#running-through-slurm)).

## Configuration

`launcher=slurm` selects `conf/launcher/slurm.yaml`; every key can be set on the
command line (`launcher.mem=12G`), or in a config.

| Key | Default | |
|---|---|---|
| `partition`, `account`, `qos` | `null` | `sbatch --partition/--account/--qos` |
| `time` | `null` | `sbatch --time`; the task is sent SIGTERM `signal_grace_seconds` before it runs out |
| `mem`, `cpus_per_task`, `gres` | `null` | `sbatch --mem/--cpus-per-task/--gres` (`gpu:1`, or `shard:1` to share a GPU) |
| `array_parallelism` | `null` | at most this many tasks at once (`--array=...%K`); `null`: no cap |
| `cases_per_task` | `null` | cases a task runs on its CARLA; `null`: the cases spread evenly over `array_parallelism` tasks (one case a task without a cap) |
| `job_name` | `null` | `acs-<date>-<time>` of the output directory |
| `extra_sbatch_args` | `[]` | further `sbatch` arguments |
| `export` | `ALL` | `sbatch --export` |
| `env_passthrough` | see below | variables recorded at submission and set again in the job |
| `setup` | `[]` | shell lines the job runs before the worker (`module load`, `source .venv/bin/activate`) |
| `env` | `{}` | variables set in the job, last |
| `python` | `null` | the interpreter the job runs; `null`: the submitting one |
| `headless` | `true` | unset `DISPLAY` in the job, so CARLA renders off-screen |
| `start_stagger_seconds` | `0` | no task of the batch starts sooner than this after another did: the first wave, the tasks started as slots free up, a resubmit's ([Shared state](#limits)) |
| `wait` | `true` | follow and collect; `false` (or `--no-wait`): submit and exit |
| `poll_seconds` | `15` | how often `wait` asks `squeue` |
| `cancel_on_interrupt` | `true` | Ctrl-C while waiting cancels the batch's jobs |
| `collect_job` | `true` | with `wait: false`, a job that collects once the array ends |
| `collect_mem` | `2G` | its `--mem` |
| `signal_grace_seconds` | `120` | `--signal=B:TERM@N` |
| `port_base`, `port_slots` | `20000`, `1000` | where the tasks' port slots are ([Ports](#ports)); `port_slots` at least 1 |
| `port_lock_dir` | `null` | where the slots' locks are; `null`: `/dev/shm/acs-slurm-ports-<uid>`, else `/tmp` ([Ports](#ports)) |

`cases_per_task` trades a task's start -- CARLA takes about 50 s to start, and
a driver policy about as long to load -- against how evenly the cases spread
and how many a lost task takes with it.

### The job's environment

A task runs with the Python the batch was submitted from (`sys.executable`,
or `launcher.python`) and its environment:

1. `sbatch --export=ALL` (`launcher.export`) hands the job the environment of
   the shell `sbatch` runs from;
2. the job script then sets again, as they were when the batch was submitted,
   the variables `launcher.env_passthrough` matches: `PATH`,
   `LD_LIBRARY_PATH`, `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV`,
   `CONDA_PREFIX`, `CARLA_*` (`CARLA_EXECUTABLE`), `AUTOWARE_*`,
   `ONEPLANNER_*`, `TYPESAFE_CARLA_*`, `GLIBC_TUNABLES`, `MALLOC_*`,
   `CUDA_HOME`, `CUDA_PATH`, `CUDAToolkit_ROOT` -- so `scenario-slurm resubmit`
   from another shell runs the cases as they were submitted;
3. `launcher.setup` runs;
4. `launcher.env` is set.

A driver policy's variables belong on `env_passthrough` (a pattern:
`launcher.env_passthrough=[...]` replaces the list) or in `launcher.env`. No
token or credential is on the default list, and a name with `TOKEN`,
`SECRET`, `KEY`, `PASSWORD` or `CREDENTIAL` in it (in any case) is not
recorded through a pattern (`AUTOWARE_*`) either -- only when
`env_passthrough` names it exactly, or `launcher.env` sets it: the manifest
and the job scripts holding the values are files in the batch's output
directory (both the user's alone, mode `0600` and `0700`). With
`export: NONE`, the job has only what steps 2 to 4 give it.

A config that reads the environment (`${oc.env:ONEPLANNER_MAP_DIR}`) is
resolved in the task too; a case whose config resolves differently there than
where it was submitted is refused, naming the keys that differ.

### Ports

Each task's CARLA needs ports no other CARLA on the node uses, and a user may
run one on the defaults (2000, TrafficManager 8100). A task takes a slot of 10
ports from `port_base`:

| Port | |
|---|---|
| `base+0` | CARLA RPC (`server.port`) |
| `base+1` | CARLA streaming (`-carla-streaming-port`) |
| `base+2` | CARLA secondary (`-carla-secondary-port`) |
| `base+3` | TrafficManager (`traffic_manager.port`) |
| `base+4` | the Autoware bridge the run hosts (`autoware.address`) |
| `base+5` | the driver bridge the run hosts (`driver.bridge.address`) |

The slot a task tries first is its array job id plus its task id, so the tasks
of an array start on different slots. It takes one only when no other task
holds it (an exclusive lock,
`<lock dir>/autoware_carla_scenario-port-<base>.lock`, which goes when the
task's process does) and none of its ports is bound; else the next. The lock
directory must be one every job on the node sees, which `/tmp` is not where
SLURM gives each job a private one (`job_container/tmpfs`): it is
`/dev/shm/acs-slurm-ports-<uid>` (or `/tmp` where that cannot be made), or
`launcher.port_lock_dir`. The default range, 20000-29999, is below Linux's
ephemeral ports.

`driver.address` is not changed: a policy the run serves itself
(`driver.policy`) listens on a port the OS picks, and one it does not is a
server someone else runs, which every task dials.

## Limits

- **Memory.** A task runs a CARLA (about 6 GB resident) and a scenario process
  (about 6 GB with a TensorRT driver policy loaded). Without cgroups, SLURM
  uses `--mem` only to schedule, so set it, and `array_parallelism`, to what
  the node has.
- **Shared state.** The tasks share the map caches (`~/autoware_data/maps`), the
  map files a driver policy reads (`driver.map_dir`) and the batch's output
  directory, so these must be on a file system every node sees. Map files,
  and the OpenDRIVE a run captures from the loaded world on a map's first run,
  are written atomically, so tasks that fill a cache at once do not break it.
  A driver policy's own caches are its own: one that two processes cannot
  load at once (OnePlanner's TensorRT engines, whose identity file is written
  under one temporary name, so two loads at the same moment fail one case
  with `filesystem error: cannot rename`) needs `start_stagger_seconds`,
  enough for one task to load it before the next starts (30 s for
  OnePlanner, which loads it once a task).
- **ROS 2 and Autoware egos** (`ego.entity=ros2`, `ego.entity=autoware` with
  a launcher) run their stacks on DDS, whose domain the tasks would share; give
  each its own `ROS_DOMAIN_ID` (the Autoware launcher's `ros_domain_ids`), or
  run such batches locally.
- **File names.** A case is named by its position in the batch, which is the
  name a local batch gives it whenever every earlier case opened its trajectory
  log. A case that raises before it opens it (while its ego is created or the
  world is set to synchronous mode) leaves its name to the next case in a local
  batch; under SLURM every case keeps its own.
- **`build_scenario_fn`.** A batch run from code with a `build_scenario_fn` is
  run by the tasks only when that function can be imported by name.
