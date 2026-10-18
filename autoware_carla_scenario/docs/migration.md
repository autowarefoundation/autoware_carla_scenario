# Migrating

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
| The viewer's video player and its `/video/...` route | The scenario page names the run's trajectory log, its Rerun recording, and the command to open it |
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
  the run ends, and the result (`rerun_recording`) and the viewer's scenario
  page name it. Open it with `rerun X.rrd`. Runs made without it convert
  afterwards with `python -m autoware_carla_scenario.rerun outputs/...` (a
  session, a sweep, or one `X.trajectory.jsonl`). The log carries the map it
  was recorded on (paths and SHA-256 of the Lanelet2 and OpenDRIVE files), the
  warm-up, the ego's plan, every condition's state and the verdict
  ([Trajectory Log](trajectory_log.md)).
- **Container images** built with `with-ffmpeg: true`: drop the input.
