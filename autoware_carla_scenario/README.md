# Autoware CARLA Scenario

A CARLA scenario testing framework for [Autoware](https://www.autoware.org/). Drives an Autoware ego vehicle through configurable scenarios in the [CARLA simulator](https://carla.org/), evaluates pass/fail conditions tick-by-tick, and produces JSON results plus replayed video.

It is typically run against OpenDRIVE maps produced by the `convert` CLI of [`autoware_lanelet2_to_opendrive`](https://github.com/hakuturu583/autoware_lanelet2_to_opendrive), but does not depend on it: the Lanelet2 and OpenDRIVE maps are loaded as two independent coordinate systems, related only through CARLA world coordinates.

## Features

- Automated scenario execution against CARLA UE5 (`0.10.0` and `ue5-dev`), through the statically typed [typesafe_carla](https://github.com/hakuturu583/typesafe_carla) client.
- [Hydra](https://hydra.cc/)-composed configurations for map, server, ego, entities, and scenario.
- Glob-pattern batch execution of multiple scenarios in a single CARLA session.
- Condition system: timing, collisions, traffic signals, speed/standstill, lane/area position, waypoint crossing, plus logical (`And`/`Or`/`Not`), latching (`Sticky`), and persistent combinators.
- Action system: turns, lane changes, traffic-light state, on-demand camera attachment.
- Pluggable traffic backends: CARLA's TrafficManager by default, `traffic=none` for an empty road, or a traffic simulator of your own through an entry point.
- Coordinate transforms between Lanelet2, OpenDRIVE, and CARLA world frames.
- Hydra `lanelet_constraint` sweeper plugin for parametric map-driven sweeps (resolvable without CARLA).
- FastAPI + Uvicorn web viewer for browsing results, replaying videos, and triggering runs.
- Two-pass video recording (CARLA native log → replayed RGB camera → ffmpeg H.264).
- pytest integration via `CarlaScenarioFixture` (auto-skips when `CARLA_EXECUTABLE` is unset).

## Installation

Python 3.10 to 3.12 (`>=3.10,<3.13`, the interpreters CI tests), Linux x86_64. Install via the workspace root:

```bash
# From the repository root
uv sync --dev
```

The CARLA client is [typesafe_carla](https://github.com/hakuturu583/typesafe_carla) (`typesafe-carla` on PyPI), a plain dependency — no extra to request — imported as `import typesafe_carla.carla as carla`. Its PyPI wheel carries the CPython package prebuilt, so the sync above compiles nothing ([installation](docs/installation.md) has the cases that build it instead). The official `carla` wheels, and CARLA 0.9.16 (UE4), are no longer supported. CARLA's simulator binary itself must be installed separately — see the [CARLA installation guide](https://carla.readthedocs.io/) and the per-package [installation docs](docs/installation.md).

## Quick usage

The package provides three CLI entry points:

| Command | Framework | Purpose |
|---------|-----------|---------|
| `scenario` | Hydra | Run autonomous-driving scenario tests in CARLA |
| `detect-no-3d-model` | argparse | Detect lanelets without a matching 3D ground model in CARLA |
| `viewer` | FastAPI + Uvicorn | Web UI for browsing and monitoring scenario results |

### Run a scenario

Requires a running CARLA server (with `CARLA_EXECUTABLE` exported or a server already listening on the configured host/port).

```bash
# Single scenario
uv run scenario scenario=intersection_passing/straight

# Batch run via glob pattern (sequential, single CARLA session)
uv run scenario scenario='intersection_passing/*'

# Other built-in scenarios
uv run scenario scenario=lane_change/left
uv run scenario scenario=temporary_stop/temporary_stop
uv run scenario scenario=traffic_light_compliance/traffic_light_compliance
```

Results (JSON + optional video) are written under the configured output directory and can be inspected interactively via:

```bash
uv run viewer
```

### Detect missing 3D models

For diagnosing lanelets that do not project onto a CARLA ground mesh:

```bash
uv run detect-no-3d-model --map /path/to/map.xodr
```

## Documentation

Full guides live under [`docs/`](docs/) and are served by MkDocs:

- [`docs/installation.md`](docs/installation.md) — system requirements, CARLA setup, `uv` install steps.
- [`docs/usage.md`](docs/usage.md) — every CLI command, configuration overrides, glob batch execution, viewer.
- [`docs/architecture.md`](docs/architecture.md) — module layout, scenario lifecycle, condition/action system.
- [`docs/api.md`](docs/api.md) — generated API reference.
- [`docs/development.md`](docs/development.md) — contributing, testing, pre-commit, release flow.
