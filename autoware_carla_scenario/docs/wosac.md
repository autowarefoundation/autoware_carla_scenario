# WOSAC Rollouts

A run's [trajectory log](trajectory_log.md) can be written as a rollout of the
**Waymo Open Sim Agents Challenge** (WOSAC): the Protobuf messages of the
Waymo Open Dataset's `sim_agents_submission.proto`, in which a simulation is
a `JointScene` -- every simulated agent's box centre and heading, at 10 Hz --
and a scenario's simulations, run from the same initial conditions, are its
`ScenarioRollouts`. The runs of one scenario (its seeds) are its rollouts, so
`scenario-wosac` works in two steps: `export` writes each run as one rollout,
and `pack` merges them by scenario into a `SimAgentsChallengeSubmission`.
Neither needs CARLA, a map or any extra.

The messages are the Waymo Open Dataset's own: what the challenge's metrics
(`waymo_open_dataset.wdl_limited.sim_agents_metrics`, the meta realism score)
and its submission server read.

## After a run

```bash
# Every log under the runs' directory, each written beside its log
uv run scenario-wosac export outputs/2026-11-13/
# Every rollout found there, merged by scenario, as WOSAC's shards and archive
uv run scenario-wosac pack outputs/2026-11-13/ -o submission/ --rollouts 32 \
    --metadata submission.yaml --tar
```

```python
from autoware_carla_scenario.wosac import (
    build_submission, check_runs, merge_rollouts, read_rollouts, read_run, rollout_files,
    trajectory_log_to_rollout, write_rollout, write_submission,
)

rollout = trajectory_log_to_rollout("CutInScenario.trajectory.jsonl")
rollout.rollouts   # sim_agents_submission_pb2.ScenarioRollouts, one JointScene
rollout.agents     # who each object_id is
write_rollout("CutInScenario.trajectory.jsonl")  # -> the .wosac.pb and .wosac.json

files = rollout_files([Path("outputs/2026-11-13")])
check_runs((read_run(f), str(f)) for f in files)  # rollouts of their scenarios?
scenarios = merge_rollouts(((read_rollouts(f), str(f)) for f in files), num_rollouts=32)
write_submission(build_submission(scenarios, {"unique_method_name": "..."}), "submission/")
```

### `export`

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | beside each log | Where to write; two logs of one stem are refused there |
| `--scenario-id` | the stem without its run counter | The rollouts' `scenario_id`: what groups runs into one scenario's rollouts (below) |
| `--current-index` | `10` | The current row, counted at 10 Hz from the log's first tick: WOSAC's `current_time_index` (1 s of history) |
| `--steps` | `80` | The rows after the current one: WOSAC's 8 s |
| `--all-steps` | | Every row up to the log's end instead |
| `--pad` | | Hold every agent's last pose on the rows after the log's end, rather than refuse a log too short for `--steps` |
| `--no-ego` | | Leave the ego out of the agents |
| `--map-frame` | | Positions in Autoware's `map` frame, its MGRS and elevation offsets added (below) |
| `--with-boxes` | | Also write the optional `width`, `length`, `height`, `valid` and `object_type` |
| `--object-ids` | | A YAML or JSON mapping of role names to object ids, over the ids derived from them |

A log too short for the rows asked for is refused, naming how many it has: a
run ends with its verdict, which can be well before 9 s. A log whose last
line is cut off is read without it, as for a [T4 Scene](t4_scene.md); a
log that fails does not stop the others, and the command exits 1. An
option out of range (`--current-index` below 0, `--steps` below 1, an empty
`--scenario-id`) is refused before any log is read, naming the option, and so
is a path that does not exist, as a directory with no log under it is (exit
2). So is an `--object-ids` file with an id that is not an int32 >= 0, or
with two keys of one role (`ego` and `hero`). Ctrl-C prints
`scenario-wosac: interrupted` and exits 130.

### `pack`

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | (required) | Where to write the shards; one that holds a submission already is refused |
| `--rollouts` | any | How many rollouts every scenario must have: the challenge takes exactly 32 |
| `--common-agents` | | Keep, in every rollout of a scenario, only the agents all of them have (the others are logged), rather than refuse the scenario |
| `--position-tolerance` | `0.5` | How far apart, in metres, an agent may start the future of two rollouts of a scenario ([below](#are-they-rollouts)) |
| `--heading-tolerance` | `0.1` | How differently headed, in radians, it may start it |
| `--metadata` | | A YAML or JSON mapping of the submission's other fields, by their names in the proto |
| `--shards` | `1` | How many files to split the scenarios into |
| `--tar` | | Also write `submission.tar.gz` of the shards: what the challenge's server takes |

It takes `*.wosac.pb` files, or the directories to find them under, and
prints how many rollouts each scenario got, then the files it wrote. A
scenario is refused, and nothing is written, when its runs are not rollouts
of it ([below](#are-they-rollouts)), when its rollouts simulate different
agents (without `--common-agents`), when two of its trajectories differ in
length, or when it has not `--rollouts` rollouts. `--rollouts` or
`--shards` below 1 is refused before anything is read, naming the option.

The metadata are the submission's fields the challenge asks for:

```yaml
account_name: you@example.com           # the account registered at waymo.com/open
unique_method_name: carla-autoware
authors: [Your Name]
affiliation: Your Lab
description: Autoware and CARLA's road users, closed loop at 20 Hz
method_link: https://example.com/method
uses_lidar_data: false
uses_camera_data: false
uses_public_model_pretraining: false
num_model_parameters: "200K"            # an integer and a multiplier suffix (K, M, B, T)
acknowledge_complies_with_closed_loop_requirement: true
```

`submission_type` is `SIM_AGENTS_SUBMISSION` unless the file says otherwise.
A field the message lacks, or a value of the wrong type, is refused.

## Scenarios and rollouts

WOSAC scores a scenario's rollouts as a distribution: 32 simulations from one
initial state, against what was logged. Here, the runs of one scenario --
one per seed -- are the rollouts, and a rollout's
`scenario_id` says which scenario it is one of. By default it is the log's
stem without the run counter the runner adds to a stem taken in its
directory already, so the runs of `CutInScenario` -- one per job directory
of a multirun, or `CutInScenario`, `CutInScenario-1`, ... in one -- are
rollouts of `CutInScenario`. Runs of one scenario class with different
parameters are different scenarios: give each its own `--scenario-id`.

To run a scenario's rollouts, run it once per seed -- the seed is what
differs between them, and every other setting is the scenario:

```bash
# Hydra's own sweeper: one job per seed, each in its own directory
uv run scenario --multirun scenario=cut_in/right 'seed=range(0,32)'
# Every log under the sweep's directory, then the rollouts packed
uv run scenario-wosac export multirun/
uv run scenario-wosac pack multirun/ -o submission/ --rollouts 32
```

What draws at random before the future starts has to draw the same in every
rollout, so it cannot follow the run's seed. A sweep drawing from the ODD
draws its cases from the run's seed unless `sweep.odd_sample.seed` is set,
and a scenario that places road users at random -- a
`TrafficSourceAction`'s `initial_vehicles`, `background_traffic`, SUMO's
ambient traffic -- places them anew with every seed: such a scenario's
initial state is not one, and its runs are not rollouts of it. Give the
road users their places, or pin the seed of what places them
(`background_traffic.seed`, an action's `seed`).

### Are they rollouts?

Whether runs are rollouts of one scenario cannot be told from what they did:
a future that differs may be chance, or a parameter that differs. It is told
from what they were asked to run. The `scenario` command gives each scenario
it builds the digest of its config, `scenario_hash`
(`autoware_carla_scenario.scenario_hash`): the SHA-256 of the scenario's
class and of the sections of the config that say what is simulated --
`scenario`, `map`, `ego`, `entity`, `environment`, `controls`,
`scenario_rules`, `background_traffic`, `traffic`, `odd` and `odd_case` --
and of `server.fog_density_scale` when it is set (it scales how thick
`environment.fog_density` renders, to the cameras and the LiDARs' fog alike;
left `null`, the server's own scale, it is not digested), without `seed`, and
without the settings that say where a run connects (hosts, addresses, ports,
ROS domains: what a SLURM task moves to its own slot), and without where the
map's files sit on this machine: the `map` section's paths (`xodr_path`,
`lanelet2_path`, `directory`, any `*_path` or `*_dir`) are left out, so the
same scenario run with the map at an absolute path, in another checkout or
through `NISHISHINJUKU_XODR_PATH` has the same hash. The map is told by
`map.name`, and by `map.source` and `map.pack`, which name a revision; what
the files held is in each run's trajectory log, whose
[`map`](trajectory_log.md#records) record has each map file's SHA-256.
Likewise, of `traffic.options`, where the traffic backend caches its SUMO
networks on this machine (`cache_dir`), the SUMO executable it runs
(`binary`) and whether it records SUMO's floating car data (`fcd_output`)
are left out, so a SLURM node with a `cache_dir` of its own runs rollouts of
the same scenario (#2620); `net_path` and `route_path` stay in, since the
network and the demand they name are the scenario. `odd` is digested as
written: an OpenODD file's path counts as written (its content is not read),
so give every rollout one spelling of it, a registered name or one path. A
number is digested by its value: a whole float as the integer it equals, so
`scenario.timeout_seconds=15` and a YAML's `15.0` are one scenario (#2481).
What drives the ego (`driver`, `autoware`, `ros2`) is the system
under test, not the scenario, and how a run is recorded does not change what
is simulated: neither is in it. Every top-level section of the `scenario`
command's config is listed as hashed or not (`SCENARIO_SECTIONS`,
`OTHER_SECTIONS`; the settings of the latter that are hashed all the same are
`SCENARIO_SETTINGS`), and a test keeps it so, so a new section cannot be left
out of the hash unseen. The runner writes the run's seed and its
hash into the trajectory log (its [`run`](trajectory_log.md#records)
record), and `export` into the `.wosac.json`, with each agent's pose on the
current row.

`pack` (`check_runs()`) then checks each scenario's runs from their
`.wosac.json`:

- **One scenario**: every run of a `scenario_id` has one `scenario_hash`.
  Runs of different parameters under one id are refused: give them their own
  `--scenario-id`.
- **A seed of its own**: two runs of one seed are one rollout run twice,
  and are refused.
- **One grid in one frame**: every run has the first run's `frame`,
  `offset`, `step_s` and `current_index`. A run exported with another
  `--map-frame` or `--current-index` is refused, naming both files and the
  field: its rollout would be scored on other rows, or kilometres from the
  others.
- **One initial state**: each agent starts its future -- the current row --
  where it does in the first run, within `--position-tolerance` and
  `--heading-tolerance`. The history before it is simulated too, so a stack
  that is not deterministic drifts a little; a road user placed at random
  does not start where it did.

A run whose log has no `run` record (written before it was recorded), or a
rollout without its `.wosac.json` (a merged file), is let through with a
warning: nothing says what it ran. A `.wosac.json` that `pack` cannot read
-- not a JSON object, a `seed` or an agent's `object_id` that is no whole
number, an `offset` that is not three numbers, an agent's `current` that is
not four -- is refused, naming the file, before anything is written.

Each agent's `object_id` has to be the same in every rollout, though CARLA
gives its actor a new id in every run. It is derived from what does stay the
same, the agent's role name, which the scenario gives the entity
(`npc1`, `background3`, ...): the first 4 bytes of the SHA-256 of its
canonical role name in UTF-8 (`ego` and `hero`, in any case, are `Ego`),
read as a big-endian integer with the top bit cleared
(`int(sha256(name.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF`),
`object_id_of()`. A scenario replaying a Waymo scenario gives its entities
the tracks' ids with `--object-ids` (`object_ids=`).

`.wosac.pb` files merge as Protobuf merges messages: two of one
`scenario_id`, concatenated, are one `ScenarioRollouts` holding both
rollouts (`cat a.wosac.pb b.wosac.pb`). `pack` does the same, and checks the
result.

## Files

| File | What it holds |
|---|---|
| `{stem}.wosac.pb` | One serialised `ScenarioRollouts`: the run's `scenario_id`, and one `JointScene` |
| `{stem}.wosac.json` | What the message has no field for: the `scenario_id`, the grid (`step_s`, `current_index`, `steps`, `padded_steps`, `current_frame`, `current_t`), the `frame` (`local` or `map`) and its `offset`, the run's verdict (`result`: `passed`, `message`, or null), its `seed` and `scenario_hash` (null for a log without a `run` record), and every agent: `object_id`, `role`, `actor_id`, `type_id`, `object_type`, `ego`, `valid_steps`, and `current` (its box centre and heading on the current row, `[x, y, z, heading]`) |
| `submission.binproto-00000-of-00001` | One serialised `SimAgentsChallengeSubmission` per shard: the metadata, and a contiguous share of the scenarios, sorted by `scenario_id` (`read_submission()` reads them back as one: every shard's scenarios, and the metadata once, the last shard's) |
| `submission.tar.gz` | The shards, with `--tar` |

Each `.wosac.pb`, `.wosac.json` and shard is written in one rename (a
scratch file beside it, renamed over it), so a run killed while it writes
leaves no truncated message, which would often still parse. `export` writes
the `.wosac.json` first and the `.wosac.pb` last, removing an earlier
export's `.wosac.pb` before: a rollout on disk always has its own sidecar,
never none (which `pack` would take for a merged file) nor an earlier
export's.

The run's own files are named after its stem (`run_outputs.WOSAC_SUFFIX`),
so a case run again under that stem removes the last attempt's rollout
rather than leave it to be packed.

## What goes where

| WOSAC | From the log |
|---|---|
| `ScenarioRollouts.scenario_id` | `--scenario-id`, else the stem without its run counter |
| `JointScene.simulated_trajectories` | One per agent: the vehicles and walkers on the scene at the current row, as WOSAC simulates the objects valid at its current step, sorted by object id by `pack` |
| `object_id` | `--object-ids`, else derived from the role name (above) |
| `center_x`, `center_y`, `center_z` | The centre of the agent's box (its origin plus the `center` of its `actor` record, turned with it) on each of the `--steps` rows after the current one, in the frame below |
| `heading` | Its yaw in that frame: radians, counterclockwise from the x axis (east) |
| `length`, `width`, `height` | With `--with-boxes`: its box (`2 * extent`), the same on every row |
| `valid` | With `--with-boxes`: whether the log has the agent on the row |
| `object_type` | With `--with-boxes`: `TYPE_PEDESTRIAN` for a walker, `TYPE_CYCLIST` for one of CARLA's bicycles, `TYPE_VEHICLE` for any other vehicle (a motorcycle too, as in the Waymo Open Dataset), else `TYPE_OTHER` |

What the log does not hold is held, never made up:

- **An agent gone** -- despawned, destroyed, parked out of sight, or on a
  tick the log skipped (a tick without road users is not logged) -- keeps
  its last pose on the rows it is missing from, and `valid` is false there.
  WOSAC assumes every agent valid for the whole rollout.
- **The rows after the log's end** (`--pad`) hold every agent's last pose,
  and are counted in `padded_steps`.
- **A road user without a role name** has no id that holds across runs: it
  is left out, with a warning naming its actor id. Two road users of one
  role, or two roles of one object id, are refused.
- **A road user that comes later** -- one on the scene only after the
  current row, or a hidden entity before its `SpawnAction` -- is not an
  agent, as WOSAC simulates only the objects valid at its current step.
- **The history** (the rows up to the current one) is not in a rollout:
  WOSAC's rollouts hold only the simulated future.

## Frames

- The axes are Autoware's `map` frame, the Lanelet2 map's (x east, y north,
  z up; right-handed), converted from CARLA's left-handed one as the
  [trajectory log](trajectory_log.md#frames-units-and-time) says. That is
  Waymo's convention: its `ObjectState.heading` is counterclockwise from the
  x axis, "right hand system about the Z axis" (`scenario.proto`), a
  submission's heading "counterclockwise from East"
  (`sim_agents_submission.proto`), and its metrics turn a box by the standard
  yaw rotation and take a road edge's counterclockwise winding as its inside.
  A left turn in CARLA (yaw decreasing) is a heading increasing here.
- By default the origin is CARLA's world origin -- the `map` frame without
  the map's MGRS and elevation offsets -- since the message holds single
  precision, which keeps millimetres there but not at an MGRS offset tens of
  kilometres out. `--map-frame` adds the offsets of the log's `map` record,
  so the positions are the Lanelet2 map's coordinates. Either way the
  `.wosac.json` says which (`frame`) and what was added (`offset`).
- The rows are WOSAC's 10 Hz: every second 20 Hz tick, from the log's first.

## Scoring

The submission is what the challenge's server scores, and what the
`waymo-open-dataset` package's sim agents metrics score locally (the meta
realism score: kinematic, interactive and map-based likelihoods). Both score
rollouts against a logged Waymo `Scenario` -- its tracks' history and sizes,
and its map -- matched by `scenario_id` and object id. This package writes
the rollouts, not that `Scenario`: scoring needs one whose ids are the
rollouts' (a run replaying a Waymo scenario with `--object-ids`, or a
`Scenario` written from a reference run), which is not written here yet.

## The contract

`proto/waymo_open_dataset/` holds `sim_agents_submission.proto` and the protos
it imports, copied byte for byte from waymo-research/waymo-open-dataset
(`src/waymo_open_dataset/`, at the commit
[proto/README.md](https://github.com/autowarefoundation/autoware_carla_scenario/blob/master/autoware_carla_scenario/proto/README.md)
names); their Python modules are generated into `wosac/_proto/` by
`scripts/compile_protos.py`. Every descriptor keeps its upstream name
(`waymo_open_dataset/protos/sim_agents_submission.proto`) and package
(`waymo.open_dataset`): the `waymo-open-dataset` package's bindings read
these files and these read theirs, and the two load into one process when the
package's protos are these same files (a different revision of one is a
duplicate file in Protobuf's descriptor pool).
