# Migrating

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
