# commonroad_metrics

The CommonRoad worker of `autoware_carla_scenario`: it reads how critical a
frame of a run is with TU Munich's CommonRoad tools --
[CriMe](https://github.com/CommonRoad/CommonRoad-CriMe)'s criticality measures
and [CommonRoad-Reach](https://github.com/CommonRoad/commonroad-reach)'s
reachable set -- for the framework's CommonRoad metrics (`crime_ttc_s`,
`reach_drivable_area_m2`, ...; `autoware_carla_scenario/docs/metrics.md`,
"CommonRoad metrics").

It is a project of its own, with its own lock, rather than an extra of the
framework: commonroad-io pins protobuf 3.20, where the framework's gRPC code
needs 4 or later, and CommonRoad-Reach builds wheels for CPython 3.10 and 3.11
only, where the framework runs on 3.10 through 3.14. Once a run is over, the
framework starts several of these as processes and shares the run's recorded
frames out among them, speaking to each in JSON lines.

## Install

```bash
cd commonroad_metrics
uv sync             # Python 3.10 or 3.11; uv fetches one if need be
uv run pytest
```

In a checkout the framework finds it by itself (`uv run --project
commonroad_metrics commonroad-metrics-worker`). Elsewhere, put
`commonroad-metrics-worker` on the `PATH`, or name the command in
`AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER`:

```bash
export AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER="/opt/commonroad_metrics/.venv/bin/commonroad-metrics-worker"
```

## What it evaluates

The framework sends the ego and the road users around it, in the CommonRoad
frame of the map (the OpenDRIVE inertial frame roadgen writes the scenario in).
The worker makes a CommonRoad scenario of them on the map's lanelets within
`radius_m` of the ego (the framework sends 100 m), every road user a dynamic obstacle whose trajectory runs on
at its velocity and acceleration (a braking one stops) over the horizon, and
evaluates the measures asked for on the ego at the first step:

| Key | CriMe / Reach | Unit |
|---|---|---|
| `crime_ttc_s` | TTC | s |
| `crime_thw_s` | THW | s |
| `crime_hw_m` | HW | m |
| `crime_ttb_s` | TTB | s |
| `crime_ttce_s` | TTCE | s |
| `crime_dce_m` | DCE | m |
| `crime_a_long_req_mps2` | ALongReq | m/s^2 |
| `crime_a_lat_req_mps2` | ALatReq | m/s^2 |
| `reach_drivable_area_m2` | the drivable area at the end of the horizon | m^2 |
| `reach_drivable_area_sum_m2` | the drivable area summed over the horizon (CriMe's DA) | m^2 |

The frame is moved to the ego and turned so that the lane under it runs along
+x before CriMe reads it: CriMe projects a road user's speed onto the lane with
the cosine of the lane's own heading, which is its speed along the lane only
where the lane runs along +x. Distances, times and areas do not change with the
turn.

CriMe's measures share each frame's scenario rather than each copying it,
and the ego's reference path and a lanelet's smoothed bounds are built once a
frame (`EvaluatorSettings.share_frame`; a test checks the values match CriMe's
own copying run). On a frame of ten road users the eight measures take about
95 ms together, against 520 ms as CriMe runs them. The drivable area is
CommonRoad-Reach's C++ computation: about 155 ms over a 3 s horizon, 45 ms
over 2 s (`horizon_steps`); its grid size changes neither the time nor the
area.

CriMe's TTS and TTR (time-to-steer, time-to-react) are left out: they simulate
the ego's manoeuvres and take seconds a frame.

## Protocol

One JSON request a line on the standard input, one answer a line on the
output; logs go to the standard error.

```json
{"op": "init", "map": "/path/map.commonroad.xml", "settings": {"dt": 0.1, "horizon_steps": 30, "radius_m": 150.0}}
{"ok": true, "measures": ["crime_ttc_s", "..."]}

{"op": "evaluate", "measures": ["crime_ttc_s"],
 "ego": {"id": 1, "kind": "car", "x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 15.0, "vy": 0.0, "length": 4.5, "width": 1.8},
 "others": [{"id": 2, "kind": "car", "x": 30.0, "y": 0.0, "yaw": 0.0, "vx": 5.0, "vy": 0.0}]}
{"ok": true, "values": {"crime_ttc_s": 2.55}, "errors": {}}

{"op": "close"}
{"ok": true}
```

A road user is `id` (positive), `kind` (`car`, `truck`, `bus`, `motorcycle`,
`bicycle`, `pedestrian`), `x`, `y`, `yaw`, `vx`, `vy`, `ax`, `ay`, `yaw_rate`,
`length` and `width`. A value is a number, `Infinity` (no conflict: TTC never
comes) or `null` (nothing to measure from); a measure that failed is `null`
with its error. While the ego is on no lanelet every measure is `null`.
`settings` takes `reach_settings` too: overrides of CommonRoad-Reach's
configuration by section and key (`{"planning": {"size_grid": 0.3}}`).
