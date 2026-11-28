# Migrating

## `scenario` exits 2 for more of what it refuses before a run

`scenario` now exits 2, which its `--help` and the exit-code table give for a
refusal before anything ran, where it exited 1 (#2612):

- a `scenario=` glob that matches no config, or is not a valid pattern (it now
  names the closest configs too, as `scenario-check` and `scenario-expand`
  do);
- `--no-wait` without `launcher=slurm` (in a glob's batch, before its sweeps
  are expanded);
- carla_canvas's client not installed, or a `canvas.config` file that is not
  there, which was a crash with a crash report (and a case the lanelet
  sweeper retried).

A script or CI step that reads exit 1 as "refused" for these reads 2 now.
Also refused, and new: a single run whose scenario's config sets
`launcher.type` (it ran here, ignoring it; exit 1: put `launcher=slurm` on the
command line or name the scenario with a glob), and a later job of Hydra's
own sweeper that names another `canvas.config` than the one the process
already applied, or none (exit 2; it ran with the first one's renderer: run
them as separate runs). A batch whose cases name two `canvas.config` files
prints an `Error:` line and exits 1 rather than crashing.

## NADE and route sweeps draw new values; unread sweep settings are refused

NADE's three random streams are now seeded past every seed a component
seeded with the run's seed itself takes (a `TrafficSourceAction`, the SUMO
backend), and an `AdversityFeasibleCondition`'s stream past them
([Adversity](adversity.md), `adversity.nade.stream_seed` and
`trigger_stream_seed`), so NADE no longer samples what those components draw
(#2607). A run still repeats from its seed, but NADE's draws and its triggers'
draws for every seed differ from those of earlier releases. So do the values
a route sweep's distributions draw: a route case's seed no longer holds its
`scenario.route.index` and `scenario.route.case_count`, so that it keeps its
values when another match appears or is excluded
([Usage](usage.md), "Drawing scalar parameters from distributions").

A sweep that expands now refuses a setting nothing in it reads, which it
used to ignore: `sweep.samples` above 1 without a distribution,
`sweep.bindings` without a lanelet constraint, and `sweep.exclude` without a
lanelet constraint or a route search. Drop the setting. `ConstructionSettings`
refuses a `merge_signs` that is not a whole number (`2.5`, `True`).

## `rerun_metrics` is gone

The module wrote a run's metrics into its Rerun recording for a converter that
did not draw `metric` records, but every converter the `rerun` extra installs
draws them itself, so it wrote nothing (#2485). Code importing it changes as
follows:

- `merge_metrics(...)`: drop the call; `rerun.convert_outputs()` (or the
  converter a run hands its log to) already puts the metrics in the recording,
  as `/metrics/<key>` ([Metrics in the recording](usage.md#metrics-in-the-recording)).
- `METRICS_SUFFIX`: import it from `autoware_carla_scenario.run_outputs`.
- `metric_records(trajectory)`: read the log's lines whose `type` is
  `metric` ([Trajectory Log](trajectory_log.md)).
- `METRICS_ENTITY` and `converter_draws_metrics()`: no replacement.

## `launcher.time` is text, never a number

`launcher.time` written as a number is now refused, as the config is read
([SLURM](slurm.md)). YAML reads an unquoted `12:00:00` as the base-60 number
43200 (and `120:00` as 7200), which `sbatch --time` took as minutes: a job
meant for 12 hours asked for 30 days. Such a number cannot be told from one
meant as minutes, so `time: 90` (and `time: 1.5`) is refused with it (#2483).
Quote the value in a config:

```yaml
time: "12:00:00"   # was: time: 12:00:00
time: "90"         # was: time: 90 (minutes)
```

On the command line `launcher.time=12:00:00` is text already; minutes are
quoted for Hydra: `launcher.time="'90'"`, not `launcher.time=90`. The viewer's
run panel quotes the field itself. Batches already submitted resubmit as
before: their manifest holds the time as text.

The same release caps `launcher.signal_grace_seconds` at 65535, what
`sbatch --signal` takes, and runs `launcher.setup` in the directory the batch
was submitted from and with `set +u` in every job, not only with
`launcher.remote_cwd`.

## Scenario hashes change once

`scenario_hash` ([WOSAC Rollouts](wosac.md#are-they-rollouts)) now digests a number by
its value -- a whole float as the integer it equals, so `15` and `15.0` are
one -- and leaves the `map` section's paths (`xodr_path`, `lanelet2_path`,
`directory`, any `*_path` or `*_dir`) out, as it leaves out endpoints
(#2481). Runs of one scenario no longer differ by a CLI override's spelling
(`scenario.timeout_seconds=15` against the YAML's `15.0`) or by where the map
sits on each machine. Nearly every scenario's hash changes once, as its config
has whole floats (`spawn_s: 25.0`): the trajectory logs and `.wosac.json`
files written before keep the old hash, so `scenario-wosac pack` refuses to
pack a rollout exported from an earlier log with one from a new log as one
scenario. Re-run, or export and pack the old runs and the new runs apart.
Two runs that differed only in a map path or a number's spelling, refused as
different scenarios before, are now packed together.

Scenario globs (`scenario='reach_goal/*'`, the viewer's Run list) no longer
select hidden files or dangling symlinks (#2481): an editor's
`.#default.yaml` lock link or a macOS `._default.yaml` used to be selected,
and the batch failed to compose it. A scenario config whose own name, or a
directory on its path below `scenario/`, starts with `.` is no longer found
by a glob; rename it.

## `debug_markers.plan_markers` is gone

The helper was exported but nothing in the framework called it: the
driver-backed entities draw their plan with `ego_plan.ego_plan_markers()`,
which also leaves out a point that is not a finite number (#2477). Code
importing `plan_markers` builds the same markers from the two helpers that
remain:

```python
from autoware_carla_scenario.debug_markers import (
    DebugMarkers,
    plan_polyline,
    world_point,
)

markers = DebugMarkers()
markers.add_polyline(plan_polyline([world_point(pose) for pose in plan.poses]))
```

## `visibility.at_quantile` is gone

The helper was exported but nothing read it: the weather rules' quantities
take their quantiles from their own distributions
([Weather rules](odd.md#weather-rules)). Code importing `at_quantile` writes
its own; it was `low + q * (high - low)`, with `q` the quantile clamped to
`[0, 1]` (#2251).

The same release refuses, as the config is read, what used to fail later
(#2251): a blank weather-rules path (`environment.rules`, its `default` or a
`vehicles` entry) is refused naming the key, rather than read as the
directory the run started in; and `PerceptionFaultInjectionAction` refuses
the ego under any of its role names (`ego` or `hero`, in any case), not only
`Ego`.

## The driver's steering angles default to null

`DriverControlSpec.max_steer_angle_deg` and `yaw_rate_trim_limit_deg` now
default to `None` instead of `56.0` and `25.0`, and the spec has their radian
twins, `max_steer_angle_rad` and `yaw_rate_trim_limit_rad`, also `None`
([Driver interface](driver_interface.md)). Null is the controller's default,
so a run steers as before; but the degrees used to win silently over a
`driver.control.max_steer_angle_rad` override, and one angle set both ways is
now refused. Code reading the spec's `max_steer_angle_deg` for a number reads
the effective angle from the controller's config instead:
`ControlConfig.from_mapping(dataclasses.asdict(spec)).max_steer_angle_rad`. A
config group of your own that sets `max_steer_angle_deg` keeps working, unless
it also sets `max_steer_angle_rad`.

The same release refuses as the config is read what used to fail later or
pass silently (#2241): a null `driver.control` (or `ros2.control`) setting is
now its default rather than an error; `driver.policy` and `driver.map_dir`
must be a non-empty string or null; `driver.map_formats` must be a list of
known format names, with `t4` only beside `driver.t4_scene`; and a HILS
device's `url` must be non-blank text, its `codec` neither empty nor a bare
`NAME:` ([HILS](hils.md)).

## A T4 dataset's LiDAR intensity is on the 0-255 scale

A run's T4 dataset (`t4.enabled`) wrote its sweeps' intensity on CARLA's
`[0, 1]` scale, which T4's readers take for a nearly black cloud. It is now
on T4's 0-255 scale, as Autoware's point clouds and the T4Scenes' sweeps are
([T4 Dataset](t4_dataset.md)): CARLA's value times 255. Code reading an
earlier dataset's `.pcd.bin` files that scaled the intensity itself drops
that; datasets written before stay on `[0, 1]` (#2237).

## An adversity trigger's draws have a stream of their own

`AdversityFeasibleCondition`, the default trigger of every adversarial action,
now seeds its draws (a jaywalker's speed, a stopped crosser's wait) from its
seed and a key of its entity, kind and label
([Adversity](adversity.md), `adversity.nade.trigger_stream_seed`), instead of
`random.Random(seed)`. Two actions following the run's seed no longer draw the
same numbers, nor NADE's (#2226). A run still repeats from its seed, but the
draws for any given seed, whether the run's or an explicit `seed=`, differ
from those of earlier releases: a scenario that pinned a jaywalker's path or a
crosser's wait to a seed needs its expectation updated. Renaming an action's
label or entity also changes its draws.

## `debug_markers.DRIVEN_PATH_COLOR` is gone

The colour was exported for a viewer to draw the driven path in, but nothing
read it: the Rerun converter colours the path itself. Code importing
`DRIVEN_PATH_COLOR` defines its own colour instead; it was `(255, 128, 0)`.

## To 23.0.0: metrics are judged once the run is over

`MetricCondition` is now a fail condition judged after the run, on metrics
computed from the frames the run recorded ([Metrics](metrics.md#judging-a-run-on-a-metric)),
instead of a condition checked every tick. A run fails when the metric met the
rule for more than `max_ng_seconds` (new, default `0`: at the first frame that
does) in all.

### What to change

- `PersistentCondition(MetricCondition(m, ...), seconds)` becomes
  `MetricCondition(m, ..., max_ng_seconds=seconds)`. Note the meaning moves
  from "held for `seconds` in a row" to "held for `seconds` in all".
- A `MetricCondition` wrapped in `NotCondition`, `AndCondition`,
  `OrCondition` or `StickyCondition`, or registered with
  `register_pass_condition`, is refused: register it on its own with
  `register_fail_condition`.
- A metric condition no longer ends a run early: a run that collides runs on
  to its other conditions or its timeout, and is failed on the metric
  afterwards, at the time the metric went NG. Give a scenario that should stop
  at a collision a `CollisionCondition` as well.

## SUMO's outputs are the run's own

The SUMO backend (`traffic=sumo`) writes its log, its generated demand and,
with `fcd_output`, `fcd.xml` and `clock.csv` to `<output>/{stem}.sumo/`,
named by the run's file stem like its `{stem}_result.json`, instead of
`<output>/sumo/`, which every run of a batch overwrote
([Traffic Backends](traffic_backends.md)). A tool reading
`<output>/sumo/fcd.xml` reads `<output>/{stem}.sumo/fcd.xml`: the stem is the
result file's name without `_result.json`. A backend prepared without a run
(`TrafficContext` with no `file_stem`) still writes to `<output>/sumo/`.

## The Speed card's Longitudinal and Lateral follow the entity

The editor's **Speed** card (`type_id: speed`) now measures its Longitudinal
and Lateral components along its own entity's heading, as the Acceleration
card always has: longitudinal is the speed along the way the entity faces
(negative while reversing), lateral the speed to its left. They used to be
the world's x and y, so "longitudinal speed above 5" held only while the
entity drove east ([#1089](https://github.com/autowarefoundation/autoware_carla_scenario/issues/1089)).

A saved document needs no edit, but one that picks Longitudinal or Lateral on
a Speed card now means the entity's own frame. One written to test a world
direction meant it only by accident; to keep it, write that condition in
Python, where `SpeedCondition` still defaults to
`SpeedCoordinateSystem.WORLD`. Magnitude is unchanged.

## The viewer and the editor bind 127.0.0.1 by default

`viewer` and `scenario-editor` now take command-line options (`--help`,
`--host`, `--port`, and the directories they serve; see
[usage](usage.md#viewer-scenario-result-viewer) and
[the editor's configuration](scenario_editor.md#configuration)). They used to
ignore every argument and bind `0.0.0.0`; both now bind `127.0.0.1`, because
neither has authentication and the viewer's API starts scenario runs and SLURM
submissions.

To serve other machines as before, say so: `uv run viewer --host 0.0.0.0`, or
`VIEWER_HOST=0.0.0.0` / `SCENARIO_EDITOR_HOST=0.0.0.0` in the environment,
which still works (a flag wins over it). Either prints a warning on stderr.

## To 15.0.0: no video, the trajectory log instead

15.0.0 removes the video a run used to render (`{Scenario}.mp4`). After each
run, the runner replayed the CARLA recording on the same server with an RGB
camera and encoded the frames with ffmpeg. That took about as long as the run,
and the next case waited for it. It also needed ffmpeg on every machine that
ran scenarios. A run is now seen through its trajectory log, which every run
already writes, from a Rerun recording made out of it.

### What is gone

| Removed | Instead |
|---|---|
| `{Scenario}.mp4` after each run | `{Scenario}.rrd`, a Rerun recording of `{Scenario}.trajectory.jsonl` made as the run ends (`rerun.enabled=true`, the `rerun` extra), opened in the Rerun viewer ([Seeing a run](usage.md#seeing-a-run)) |
| `video.width`, `video.height` (and `video.enabled`, which never shipped in a release) | Nothing to set. A run that sets one warns that it is ignored |
| `CameraRecorder`, `autoware_carla_scenario.camera_recorder`, `check_video_size`, `DEFAULT_IMAGE_WIDTH`/`HEIGHT` | — |
| `ScenarioRunner(video_width=, video_height=)`, `ScenarioQueue(video_width=, video_height=)`, `run_scenario_with_queue(video_size=)` | Drop the arguments |
| `ScenarioRunner._render_video_from_recording`, `DebugMarkerLog`, `DrivenPathDrawer`, `draw_debug_markers` | The trajectory log's `markers` records (the ego's plan) and its ticks (the path it drove) |
| `SpectatorCameraConfig`, `BaseScenario._spectator_camera_config` (where the video's camera went) | — : `BaseScenario.follow_with_spectator()` still moves CARLA's spectator, and takes the same arguments |
| The viewer's video player and its `/video/...` route | The scenario page shows the run in an embedded Rerun web viewer (`/recording/...rrd`): the recording `rerun.enabled=true` made as the run ended, else one the viewer makes on demand |
| The `ffmpeg-python` dependency, and ffmpeg itself | — |
| `with-ffmpeg` of the `pack-scenario-image` action | — |

The CARLA recording (`{Scenario}.log`) is still written: CARLA's replayer
(`client.replay_file()`) plays it back on a server running the same map. The
ego's debug markers are still collected (`EgoVehicle.debug_markers()`), and
are written into the trajectory log.

### What to change

- **Command lines and configs** that set `video.*`: drop it. It is accepted, so
  an old script still runs, and warned about.
- **Code** that passes `video_width`/`video_height`/`video_size`, or imports
  `CameraRecorder` or the drawing helpers: remove that. These are gone, and
  passing them is a `TypeError` or an `ImportError`.
- **Seeing runs**: install the converter and the viewer
  (`pip install 'autoware-carla-scenario[rerun,viewer]'`) and run with
  `rerun.enabled=true`: each run's log is converted into `X.rrd` beside it as
  the run ends, the result (`rerun_recording`) names it, and the viewer's
  scenario page shows it. Open it with `rerun X.rrd`. Runs made without it convert
  afterwards with `python -m autoware_carla_scenario.rerun outputs/...` (a
  session, a sweep, or one `X.trajectory.jsonl`). The log carries the map it
  was recorded on (paths and SHA-256 of the Lanelet2 and OpenDRIVE files), the
  warm-up, the ego's plan, every condition's state and the verdict
  ([Trajectory Log](trajectory_log.md)).
- **Container images** built with `with-ffmpeg: true`: drop the input.
