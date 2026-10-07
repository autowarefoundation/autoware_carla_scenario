# Standalone Binaries (experimental)

`scenario-build` compiles a scenario config into a native executable that
runs it against a CARLA server, with nothing else installed: no Python, no
Hydra, no Lanelet2, no pyxodr.

```text
$ uv run scenario-build scenario=lane_change/left
[ok] lane_change/left: dist/lane_change_left/bin/lane_change_left (50s)

$ dist/lane_change_left/bin/lane_change_left --host carla.local
```

It prints `PASSED: ...` or `FAILED: ...` and the path of the result JSON, as
`uv run scenario` does.

It builds on the [static check](typecheck.md): the same modules of the
scenario's package are collected and rewritten for Codon, and the config is
composed exactly as `uv run scenario` composes it. What differs is what they
are compiled against. The check compiles against a *model* of the framework,
whose functions only have types; a build compiles against a *runtime*
(`autoware_carla_scenario/standalone/codon/`), whose functions run.

!!! warning "Experimental"
    The runtime implements part of the framework (see
    [What runs](#what-runs)), and nothing here has been run against a CARLA
    server in CI yet. Treat a binary's verdict as a second opinion next to
    `uv run scenario`, not a replacement for it.

## The bundle

```text
dist/lane_change_left/
  bin/lane_change_left     the binary (RPATH $ORIGIN/../lib)
  lib/                     libtypesafe_carla_ffi.so and Codon's runtime
  share/map.acsmap         the map, baked at build time
  share/map.xodr           the OpenDRIVE, installed when the map asks for it
```

The bundle can be copied anywhere: the binary finds `lib/` and `share/` from
its own location. It needs glibc, libstdc++ and zlib from the system, as any
C++ program does, and Linux x86_64 (the only platform typesafe_carla ships
for).

```text
$ lane_change_left --help
  --host HOST            CARLA server host (built with: localhost)
  --port PORT            CARLA server port (built with: 2000)
  --tm-port PORT         TrafficManager port (built with: 8100)
  --timeout SECONDS      default timeout of the run (built with: 10.0)
  --output-dir DIR       where the result JSON and the replay log go
  --client-timeout SEC   seconds to wait for each CARLA call (default: 60)
  --map-data PATH        the baked map (default: <bundle>/share/map.acsmap)
```

Everything else the config decides -- the scenario's own values, the ego,
its spawn and goal, the map -- is compiled in. The exit status is 0 when the
scenario passed, 1 when it failed, 2 on a usage error and 3 when it could not
run (no server, a map it cannot load). A map with `map.overwrite_xodr` is
installed the way the Python runner installs it, through the same
`<MAP_NAME>_PATH` environment variable.

## How it works

### The baked map

The Python runner computes with Lanelet2 and pyxodr. A binary has neither:
`scenario-build` loads the map with the same `MapManager` and writes what the
coordinate functions read (`standalone/bake.py`):

- each lanelet's centerline and outline (the polygon Lanelet2's `inside` and
  `findNearest` use);
- each OpenDRIVE road's reference line and elevation, as pyxodr samples them;
- the MGRS offset between the two frames.

The runtime's `to_carla_world`, `to_lanelet2`, `on_lanelet`,
`project_onto_lanelet`, ... follow `coordinate/transform.py` step by step on
that geometry, and `test_standalone.py` holds them to the Python package on
the fixture map: of 605 sampled conversions, 604 agree to 1e-9 m (most to the
bit). The one that does not is a point projecting exactly onto a centerline
vertex, whose heading Lanelet2 takes from the neighbouring segment because it
sums the arc length to the vertex one ulp differently. Lookups the Python
runner makes on the CARLA map (`get_waypoint`, the spawn points, the ground
projection, the z offset) a binary makes on the same server.

Loading the baked map takes the binary about 0.2 s, against about 4 s for
Lanelet2 and pyxodr in the Python runner.

### What Codon needs beyond the check

A module that compiles can still run wrongly. The build fixes what this
project's scenarios ran into:

| What | In Codon 0.19 | What the build does |
|---|---|---|
| `config or MyConfig()` | Compiles to a Union it cannot read back: fails at run time ("invalid union getter"), whether or not `config` is `None` | Rewrites `a or b` to `_acs_or(a, lambda: b)` (same short-circuit), line for line (`standalone/transform.py`) |
| `f"{x:.2f}"`, `str(3.0)` | Formats floats under `std::locale("en_US.UTF-8")`: raises on a machine without that locale; `str(3.0)` is `"3"` | The runtime replaces float/int formatting and `"..." % args` program-wide with an exact, locale-free implementation that prints what CPython prints (`_fmt.codon`) |
| A `ClassVar` of the base class (`STABILIZE_TICKS`) | Not inherited by the subclass | The runner reads the subclass's own, else the base class's |
| Class hierarchies two levels deep | Miscompiled | The runtime keeps every class one level below its base (the model does already) |

And two that are not Codon's:

- `libtypesafe_carla_ffi.so` crashes (SIGSEGV) as it loads when `$HOME` is
  unset, as under `env -i`, some systemd units and minimal containers. The
  binary sets `HOME=/` first when it is missing.
- `codon build` writes its own library directories, on the build machine,
  ahead of the RPATH it is given. The build rewrites the RPATH to
  `$ORIGIN/../lib` alone, so a bundle never loads a Codon runtime from
  wherever the toolchain happened to be installed.

## What runs

A scenario that uses something the runtime does not implement yet does not
build: the error points at the line that uses it, as a static check error
does.

```text
[FAILED] cut_in/left: ... cut_in.py:147:13: error: CollisionCondition is not supported by the standalone runtime yet
```

| | Supported | Not yet |
|---|---|---|
| Ego | `ego.entity=autopilot` | `autoware`, `carla_driver` |
| Traffic | `traffic=traffic_manager` | `traffic=sumo`, background traffic |
| Conditions | `AlwaysTrue`, `And`, `Or`, `Not`, `Sticky`, `Persistent`, `ElapsedTime`, `Timeout`, `EntityExistence`, `Speed`, `EntityLanePosition` (on a lanelet, or a road and lane), `EntityLaneOf`, `beside_lane_id` | `Collision`, `Acceleration`, `RelativeSpeed`, `Standstill`, `EntityDistance`, `EntityPositionDistance`, `TemporaryStop`, `TimeHeadway`, `TimeToCollision`, `Waypoint`, `TrafficSignal`, `TrafficSignalController`; `anywhere_on_road(..., rules=...)` |
| Actions | `LaneChange`, `Turn`, `Routing` (no-op for the autopilot ego, as in Python), `TrafficSignal` with `TrafficLightTarget.ALL` | `TrafficSignal` by Lanelet2 id, `SetSpeed`, `Environment`, `WalkStraight`, `TrafficSignalController`, `TrafficSource`/`TrafficSink` |
| Entities | `VehicleEntity` | `PedestrianEntity` |
| Coordinates | `to_carla_world`, `to_opendrive`, `to_lanelet2`, `to_carla_location`, `snap_to_carla_road`, `on_lanelet`, `project_onto_lanelet`, `lanelet_length`, `find_nearest_traffic_light` | `project_onto_road`, stop lines, the Lanelet2 -> OpenDRIVE signal mapping |
| Runner | the full setup / warm-up / init / tick loop, pass and fail conditions, the CARLA replay log, `<Scenario>_result.json` | the trajectory recording, the video render, starting a CARLA server, a batch of scenarios in one process |
| Registration | `register_scenario()` | `register_scenario_builder()` (the Scenario Editor's exports) |

The result JSON is the Python runner's, less each condition's `details`, and
a condition's `condition_type` is the runtime class it was built as (a
scenario's own condition subclass reports `BaseCondition`).

### The example configs

Built with `uv run scenario-build` on this repository's examples:

| Config | Builds | First unsupported use |
|---|---|---|
| `lane_change/left`, `lane_change/right` | yes | |
| `lane_change_fail/left`, `lane_change_fail/right` | yes | |
| `intersection_passing/straight`, `left_turn`, `right_turn`, `right_turn_sweep`, `straight_3npc` | yes | |
| `cut_in/left`, `cut_in/right` | no | `CollisionCondition` (`cut_in.py:147`) |
| `pedestrian_dart_out` | no | `PedestrianEntity` (`pedestrian_dart_out.py:125`) |
| `temporary_stop` | no | `get_stop_line_poses_with_following()` (`temporary_stop.py:99`) |
| `traffic_light_compliance` | no | `StandstillCondition` (`traffic_light_compliance.py:116`) |

9 of the 14 configs build; each of the other five stops at the first thing
the runtime lacks. Each build compiles in about a minute (4 in parallel: the
14 in 3.5 minutes).

## Building

```bash
uv run scenario-build scenario=lane_change/left                 # dist/lane_change_left/
uv run scenario-build scenario='lane_change/*' --out dist      # one bundle per config
uv run scenario-build scenario=lane_change/left scenario.timeout_seconds=20
uv run scenario-build --jobs 2                                  # every config, 2 at a time
```

A build takes about a minute (the Codon compile) and needs the same Codon
toolchain as the static check, plus `g++`, which Codon links with. The baked
map is cached under `~/.cache/autoware_carla_scenario/standalone/`.
