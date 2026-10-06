# Scenario videos

The clips under `docs/scenarios/media/` are recorded with these scripts: each
example scenario runs on CARLA with SUMO driving the traffic, and the CARLA
chase camera is laid beside a `sumo-gui` replay of the same run.

```bash
# From the repository root, with a CARLA 0.10 server on localhost:2000
uv sync --extra sumo
uv run python autoware_carla_scenario/tools/scenario_video/render.py \
    autoware_carla_scenario/tools/scenario_video/cases.txt /tmp/scenario_videos
cp /tmp/scenario_videos/*.mp4 autoware_carla_scenario/docs/scenarios/media/
```

`--only <name> ...` records some of them; `--min-free-gb` (6 by default) holds
each run back until that much memory is available. Runs never overlap.

| Script | What it does |
|---|---|
| `render.py` | Expands each case's scenario on Town10HD_Opt, runs the case with `traffic=sumo`, and drives the others |
| `chase.py` | Films the ego every 0.1 s of simulation time, and logs every CARLA light's state with each frame |
| `sumo_replay.py` | Replays the run in `sumo-gui`: vehicles and persons from SUMO's FCD, and each signal at the state SUMO recorded (`SaveTLSStates`), one screenshot per CARLA frame |
| `compose.py` | Lays the two side by side, with the title, the time and the result, into an mp4 (ffmpeg) |
| `check_run.py` | Compares SUMO's signals with CARLA's lights link by link, through the same roadgen traces the backend matches them with, and reports how fast each person walked |

`cases.txt` lists each clip as `name|scenario|ego spawn lanelet|title|extra
overrides`. The spawn lanelet picks one of the cases the sweep expands to, and
is the one the caption on the scenario's docs page names.

Two cases are recorded with an override, said in their captions:

- `traffic_light_compliance`: `scenario.moving_speed_kmh=15`, so the run ends
  once the ego is visibly pulling away on green rather than at the first
  1 km/h.
- `lane_change_fail_*`: `scenario.timeout_seconds=8`, since the refusal passes
  on its timeout and 1.5 s shows nothing.

The SUMO side is a replay, not the run's own window: `sumo-gui` would otherwise
run the network's fixed-time programs, so the signals are set from what SUMO
recorded, which with `traffic_light_authority: carla` is CARLA's.
