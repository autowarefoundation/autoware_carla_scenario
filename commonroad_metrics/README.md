# commonroad_metrics

The CommonRoad worker of `autoware_carla_scenario`: it reads how critical a
frame of a run is with TU Munich's CommonRoad tools --
[CriMe](https://github.com/CommonRoad/CommonRoad-CriMe)'s criticality measures
and [CommonRoad-Reach](https://pypi.org/project/commonroad-reach/)'s
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
| `crime_btn` | BTN (brake threat number) | ratio |
| `crime_stn` | STN (steer threat number) | ratio |
| `crime_ttr_s` | TTR (time-to-react) | s |
| `crime_wttr_s` | WTTR (worst-case time-to-react), by Reach | s |
| `reach_drivable_area_m2` | the drivable area at the end of the horizon | m^2 |
| `reach_drivable_area_sum_m2` | the drivable area summed over the horizon (CriMe's DA) | m^2 |

The frame is moved to the ego and turned so that the lane under it runs along
+x before CriMe reads it: CriMe projects a road user's speed onto the lane with
the cosine of the lane's own heading, which is its speed along the lane only
where the lane runs along +x. Distances, times and areas do not change with the
turn.

The ego's lane is the lanelet under it heading most its way, and CriMe's
"ahead" is along that lanelet. When even that one runs against the ego
(`cos(heading - yaw) < 0`: the ego overtaking through the oncoming lane, or
driving the wrong way), a car coming at it head-on read as behind it -- TTC,
HW and THW `inf`, ALongReq and BTN `0` -- and Reach read the ego as reversing.
The worker reverses every lanelet of the frame then -- bounds, left and right,
predecessors and successors, neighbours -- so the ego's lane runs its way and
the frame reads as its mirror on a lane running the ego's way: with the ego
at 15 m/s in the oncoming lane and a car 60 m ahead coming at 10 m/s, TTC
2.22 s, ALongReq -10.29 m/s^2, BTN 0.89. A frame whose ego goes its lane's
way is not changed.

CriMe 0.4 reads the other road user's velocity and acceleration along the lane
as their magnitude, dropping their sign: a braking lead reads as speeding up
(TTC `inf`, ALongReq `0`) and a car coming the other way in the ego's lane as
going its way. The worker's TTC and ALongReq are CriMe's with the other's
components projected onto the lane, signed; BTN, which is ALongReq over the
most a car can brake, and ALatReq and STN, which take TTC, use them. The
ego's velocity is read signed too. A road user braking to a stop stops and stays there, as
the worker predicts it, rather than reversing as CriMe's constant acceleration
carries it on: TTC is the first contact with each holding its acceleration
until it stops. ALongReq is the least constant deceleration that keeps the ego
off the road user ahead in that model (found by bisection): CriMe's
`a - v_rel^2 / (2 gap)` where the gap closes on a road user going the ego's
way that does not stop first, `0` for a gap opening for good (where CriMe's
formula reads `-v_rel^2 / (2 gap)`), the deceleration that stops the ego behind
where a braking lead stops, for a car coming the other way the deceleration that
keeps it from reaching the ego while it moves or within the horizon after it
has stopped, and `-inf` (BTN `inf`) when no braking avoids the collision. A
contact counts while the ego is moving, and within the horizon
(`horizon_steps` x dt, 3 s by default) after it has stopped:
a road user that would reach a standing ego only later (a pedestrian crossing
a little towards it, a car rolling back) needs no more than the ego stopping
short of it. TTC has no horizon, as CriMe's. A road user behind the ego gives
no ALongReq.

CriMe reads a negative bumper-to-bumper headway -- a road user whose box
overlaps the ego's, a collision -- as a road user behind: HW, THW and TTC
`inf`, ALongReq and BTN `0`, the least critical reading for the most critical
frames. The worker reads a road user overlapping the ego with its centre ahead
of the ego's (or level with it) as touching: HW and THW `0`, and TTC, ALongReq
and BTN as for a gap closed to `0` -- while the ego closes on it TTC `0`,
ALongReq `-inf`, BTN `inf`, and ALatReq and STN, which take TTC, `inf`. `0`
rather than the negative gap, which TTC's contact time would read as a gap
still to close once the other pulls away. The boxes must touch: CriMe counts
a road user sharing any lanelet with the ego as in its lane, so one beside the
ego (straddling a lane line, or the ego changing lanes) overlaps it along the
lane without a contact, and reads HW `inf` as before, as does a road user
whose centre is behind the ego's.

A lanelet is kept when its centre line comes within `radius_m` of the ego
anywhere along it, not only at a vertex (a straight lanelet is just its two end
points), and always when the ego is on it. The reachable set is that of the
ego's own box: CommonRoad-Reach derives its discs and the radius it inflates
the other road users by from its default vehicle (a BMW 320i) as its
configuration is built, and the worker derives them again from the ego's
length and width. A reversing ego is given to Reach going backwards.

CriMe's measures share each frame's scenario rather than each copying it,
and the ego's reference path and a lanelet's smoothed bounds are built once a
frame (`EvaluatorSettings.share_frame`; a test checks the values match CriMe's
own copying run). On a frame of ten road users the eight measures from TTC to
ALatReq take about 95 ms together, against 520 ms as CriMe runs them; BTN and
STN, which compute ALongReq and ALatReq again, about 170 ms more. The drivable area is
CommonRoad-Reach's C++ computation: about 155 ms over a 3 s horizon, 45 ms
over 2 s (`horizon_steps`). Its grid (`reachable_set.size_grid`, 0.2 m by
default) changes both: a coarser grid is faster and over-approximates the
area -- on the test lane, behind a slower lead, 791 m^2 at 0.2 m, 833 m^2 at
0.5 m and 889 m^2 at 1 m -- and can move WTTR by a step.

WTTR is not CriMe's class, which loads CommonRoad-Reach's scenario from a
file, but its algorithm on the frame: the ego goes on along its predicted
trajectory until it would hit a road user (TTC\*); before that step, the latest
one from which the reachable set still reaches the end of the horizon is found
by bisection, each try a reachable set from the ego's predicted state there.
WTTR is the time to the step after it: `Infinity` when the ego hits no one, `0`
when no way is left even now. Unlike CriMe's TTC\*, the road's boundary is no
obstacle: the ego's prediction goes straight on and would leave every bend. A
frame with a collision ahead takes about 110 ms (a few reachable sets, each
over what is left of the horizon); one without, next to nothing.

TTR is CriMe's: the latest time one of the ego's evasive manoeuvres --
braking, kick-down, steering left or right, each simulated -- still avoids the
collision, at or under WTTR. It takes about 2 s a frame with a collision
ahead (next to nothing without one), which is why the framework evaluates it
every 0.5 s rather than every 0.1 s. CriMe reads `-Infinity` when no manoeuvre
is left even now; the worker reads `0`, as for WTTR. TTB is CriMe's too, the
same search with braking at full alone, and reads `0` likewise when braking
no longer avoids the collision even now (CriMe's `-Infinity`: no step does, or
the ego is in contact already).

A measure of the road user ahead in the ego's lane reads `null` when the
frame has no other road user at all. With road users around but none ahead in
the ego's lane -- beside it, behind it, in another lane -- CriMe reads that
none is: TTC, THW and HW `Infinity`, ALongReq, ALatReq, BTN and STN `0`. TTCE
and DCE read the closest road user, in the ego's lane or not. TTB and TTR
search the ego's own predicted path (CriMe's TTC\*, which counts the road's
boundary as an obstacle) and WTTR its reachable set: `Infinity` when it hits
nothing, with road users around or not.

## Protocol

One JSON request a line on the standard input, one answer a line on the
output; logs go to the standard error. So does anything else written to the
output: the worker answers on a duplicate of descriptor 1 and points
descriptor 1 at the standard error, so that a `print()` or a write from C++
code cannot break into an answer.

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

Every request line gets one answer, in order. One that cannot be served -- a
line that is not UTF-8 or not JSON, an unknown `op`, a field missing or of the
wrong type, settings `init` refuses -- is answered
`{"ok": false, "error": "..."}`, and the worker reads on. A blank line is no
request and gets no answer.

A road user with a NaN or an infinity among its numbers is not such a request
but a reading gone bad in one frame: `evaluate` answers it
`{"ok": true, "values": {...}, "errors": {...}}` with every measure asked
`null` and failed with the message naming the field (`"vx must be finite, got
nan"`; an unknown measure still fails with `"unknown measure"`). The framework
then reads nothing from those measures in that frame alone, as from any measure
that fails now and then, and the next frame is evaluated as ever.

The worker exits after answering `close`, and at the end of its input too.
The framework does not wait for the answer to `close`: it sends the request,
closes the worker's input, and kills a worker that has not exited within five
seconds (one deep in an evaluation).

The framework fails a run whose worker does not know a measure it asks for
(the `init` answer lists those it knows), answers `"unknown measure"` for one,
or leaves one out of an answer (neither a value nor an error); so it does a
run on which a measure fails in every frame. It waits for an answer
`request_timeout_s` (120 s by default) for an `evaluate`, and
`start_timeout_s` (1800 s) for the `init`; a worker that has not answered by
then is killed and the run fails.

The worker runs in a session and process group of its own, so the kill reaches
all of it: the worker started by `uv run` too, not only `uv`. The terminal's
Ctrl-C does not reach the worker then; the framework closes it instead, when a
run's evaluation is interrupted or fails, and when the framework exits. An
interrupted evaluation stops at once rather than finishing its remaining
frames. A framework killed outright (`SIGKILL`) closes the worker's input, and a
worker exits at its end, so a worker must not ignore the end of its input.

A road user is `id` (a positive integer, unique among the frame's road users;
one of the `others` with the ego's id is the ego, and left out), `kind` (`car`, `truck`, `bus`, `motorcycle`,
`bicycle`, `pedestrian`), `x`, `y`, `yaw`, `vx`, `vy`, `ax`, `ay`, `yaw_rate`,
`length` and `width`, every number finite (a frame with one that is not
fails its measures, above). An `id` that is not a positive integer (`2.0`,
`0`, `true`) or two of the `others` sharing one is out of protocol:
`{"ok": false, "error": "ValueError: road users share the id 2: ids must be unique"}`.
A value is a number, `Infinity` (no conflict: TTC never
comes), `-Infinity` (ALongReq when no braking avoids a contact within the
horizon; BTN is `Infinity` then) or `null` (nothing to measure from: no other
road user in the frame, above); a measure that failed is `null`
with its error. TTB and TTR are never `-Infinity`: `0` when no manoeuvre is
left even now. While the ego is on no lanelet every measure is `null`.
`settings` takes `dt` and `radius_m`, finite and positive, and `horizon_steps`,
an integer of at least 1 (`init` refuses anything else), and `reach_settings`:
overrides of CommonRoad-Reach's configuration by section and key (`{"reachable_set": {"size_grid": 0.3}}`). A
section or key the configuration does not have -- `planning.size_grid` for
`reachable_set.size_grid`, a misspelling -- is refused: `init` answers
`{"ok": false, "error": "ValueError: reach_settings['planning']['size_grid']: ..."}`
rather than every frame's Reach measures failing. So is a key the worker sets
itself for every reachable set, which an override would break or which would
overwrite it: `planning.dt` (the settings' `dt`, the step of the road users'
trajectories -- Reach stepping by another would run past their end, and they
would drop out of its collision check), `planning.step_start` and
`planning.steps_computation` (from `horizon_steps` and where each set
starts), and what Reach derives from the frame (`planning.p_lon_initial`,
`p_lat_initial`, `v_lon_initial`, `v_lat_initial`, `o_initial`, `route`,
`reference_path`, `lanelet_network`, `list_ids_lanelets`, `CLCS`).
