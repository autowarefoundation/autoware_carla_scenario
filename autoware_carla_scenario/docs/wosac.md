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
    build_submission, merge_rollouts, read_rollouts, rollout_files,
    trajectory_log_to_rollout, write_rollout, write_submission,
)

rollout = trajectory_log_to_rollout("CutInScenario.trajectory.jsonl")
rollout.rollouts   # sim_agents_submission_pb2.ScenarioRollouts, one JointScene
rollout.agents     # who each object_id is
write_rollout("CutInScenario.trajectory.jsonl")  # -> the .wosac.pb and .wosac.json

files = rollout_files([Path("outputs/2026-11-13")])
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
log that fails does not stop the others, and the command exits 1.

### `pack`

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | (required) | Where to write the shards; one that holds a submission already is refused |
| `--rollouts` | any | How many rollouts every scenario must have: the challenge takes exactly 32 |
| `--common-agents` | | Keep, in every rollout of a scenario, only the agents all of them have (the others are logged), rather than refuse the scenario |
| `--metadata` | | A YAML or JSON mapping of the submission's other fields, by their names in the proto |
| `--shards` | `1` | How many files to split the scenarios into |
| `--tar` | | Also write `submission.tar.gz` of the shards: what the challenge's server takes |

It takes `*.wosac.pb` files, or the directories to find them under, and
prints how many rollouts each scenario got, then the files it wrote. A
scenario is refused, and nothing is written, when its rollouts simulate
different agents (without `--common-agents`), when two of its trajectories
differ in length, or when it has not `--rollouts` rollouts.

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
num_model_parameters: "0"
acknowledge_complies_with_closed_loop_requirement: true
```

`submission_type` is `SIM_AGENTS_SUBMISSION` unless the file says otherwise.
A field the message lacks, or a value of the wrong type, is refused.

## Scenarios and rollouts

WOSAC scores a scenario's rollouts as a distribution: 32 simulations from one
initial state, against what was logged. Here, the runs of one scenario --
its seeds, or its repetitions -- are the rollouts, and a rollout's
`scenario_id` says which scenario it is one of. By default it is the log's
stem without the run counter the runner adds to a stem taken in its
directory already, so the runs of `CutInScenario` -- one per job directory
of a multirun, or `CutInScenario`, `CutInScenario-1`, ... in one -- are
rollouts of `CutInScenario`. Runs of one scenario class with different
parameters are different scenarios: give each its own `--scenario-id`.

Each agent's `object_id` has to be the same in every rollout, though CARLA
gives its actor a new id in every run. It is derived from what does stay the
same, the agent's role name, which the scenario gives the entity
(`npc1`, `background3`, ...): the first 31 bits of the SHA-256 of its
canonical role name (`ego` and `hero` are `Ego`), `object_id_of()`. A
scenario replaying a Waymo scenario gives its entities the tracks' ids with
`--object-ids` (`object_ids=`).

`.wosac.pb` files merge as Protobuf merges messages: two of one
`scenario_id`, concatenated, are one `ScenarioRollouts` holding both
rollouts (`cat a.wosac.pb b.wosac.pb`). `pack` does the same, and checks the
result.

## Files

| File | What it holds |
|---|---|
| `{stem}.wosac.pb` | One serialised `ScenarioRollouts`: the run's `scenario_id`, and one `JointScene` |
| `{stem}.wosac.json` | What the message has no field for: the `scenario_id`, the grid (`step_s`, `current_index`, `steps`, `padded_steps`, `current_frame`, `current_t`), the `frame` (`local` or `map`) and its `offset`, the run's verdict (`result`: `passed`, `message`, or null), and every agent: `object_id`, `role`, `actor_id`, `type_id`, `object_type`, `ego`, `valid_steps` |
| `submission.binproto-00000-of-00001` | One serialised `SimAgentsChallengeSubmission` per shard: the metadata, and a contiguous share of the scenarios, sorted by `scenario_id` (`read_submission()` reads them back as one) |
| `submission.tar.gz` | The shards, with `--tar` |

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
