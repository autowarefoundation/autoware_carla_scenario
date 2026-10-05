# Installation

This guide will help you install the `autoware-carla-scenario` package.

## System Requirements

### Operating System

- **Linux x86_64** (Ubuntu 22.04 is the reference; CI runs on
  `ubuntu-latest`). Every dependency resolves to a wheel, the CARLA client's
  CPython package included (see [CARLA client](#carla-client)), so nothing
  is compiled on install.

### Python Version

- **Python 3.10 – 3.12** — `pyproject.toml` declares
  `requires-python = ">=3.10,<3.13"`: the interpreters CI tests. No
  dependency caps it any more; it is raised together with CI's interpreter
  matrix. Check your version with `python --version`.

### CARLA Simulator

The framework targets CARLA UE5 (`0.10.0` and `ue5-dev`). CARLA 0.9.x (UE4)
is not supported. Follow the
[CARLA installation guide](https://carla.readthedocs.io/) to set up the
simulator binary itself.

### CARLA client

The Python client is
[typesafe_carla](https://github.com/hakuturu583/typesafe_carla), a plain
dependency of the package (`typesafe-carla>=0.2.0,<0.3`, from PyPI, Linux
x86_64 only), imported as `import typesafe_carla.carla as carla`. It also
provides the Codon library the static check compiles scenarios against, and
pulls in `typesafe-carla-toolchain`, the pinned Codon compiler. The official
`carla` wheels are not used, and no extra has to be requested.

The released wheel on PyPI carries its CPython package prebuilt
(`typesafe_carla/carla/_prebuilt`), one build that serves every Python 3.10+,
so `uv sync` (or `pip install`) is the whole installation: nothing is
compiled, no C compiler is needed, and nothing is fetched beyond the install
itself.

Only where no matching prebuilt package exists -- typesafe_carla installed
from a source checkout or an sdist, edited Codon sources, or a
`typesafe-carla-toolchain` release other than the one the wheel was built
with -- does the first `import typesafe_carla.carla` build it instead. That
build takes 15 to 30 minutes and about 8 GB of RAM, needs a C compiler
(`cc`), and lands in `~/.cache/typesafe_carla/pycarla`.

```bash
uv run typesafe-codon pycarla
```

runs it ahead of time, and reports that there is nothing to build when the
prebuilt package applies. Set `TYPESAFE_CARLA_PYCARLA_DIR` to put the build
elsewhere (a directory set this way is used in preference to the prebuilt
package), and `TYPESAFE_CARLA_PYCARLA_BUILD=0` to make a missing build an
`ImportError` instead of starting one.

### Package Manager

- **uv** (version 0.9.7+) — modern Python package manager, used as both
  the build backend and the workflow runner.

## Installing uv

If you don't have `uv` installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more options, see the [official uv documentation](https://docs.astral.sh/uv/).

## Installing the Package

The framework does not depend on
[`autoware_lanelet2_to_opendrive`](https://github.com/hakuturu583/autoware_lanelet2_to_opendrive),
the converter that typically produces its OpenDRIVE maps: a Lanelet2 map and an
OpenDRIVE map are loaded as two independent coordinate systems. Install the
converter separately when you need to generate a map.

### For Developers (Editable Installation)

1. Clone the repository:

    ```bash
    git clone https://github.com/hakuturu583/autoware_carla_scenario.git
    cd autoware_carla_scenario
    ```

2. Sync dependencies from the lock file:

    ```bash
    uv sync
    ```

    The CARLA client arrives with its CPython package prebuilt; nothing more
    to build (see [CARLA client](#carla-client)).

!!! note
    The Lanelet2 binding comes from
    [`simple-lanelet2`](https://github.com/hakuturu583/simple_lanelet2)
    as a prebuilt wheel, so `uv sync` needs no apt packages and no C++
    toolchain on any host, and the CARLA client's prebuilt CPython package
    means no `cc` either.

## Environment Configuration

Configure your environment by exporting the variables consumed by the
package, or by loading a `.env` file (the package depends on
`python-dotenv`). The most common variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `CARLA_EXECUTABLE` | when `ScenarioQueue` launches its own server | Path to `CarlaUE5.sh`. Pytest tests skip when this is unset. |
| `NISHISHINJUKU_MAP_PATH` | when overwriting the built-in `.xodr` | Path inside the CARLA install where the original `.xodr` lives. |
| `NISHISHINJUKU_XODR_PATH` | optional | Override the default OpenDRIVE file resolved in `conf/map/nishishinjuku.yaml`. |
| `NISHISHINJUKU_LANELET2_PATH` | optional | Override the default Lanelet2 `.osm` path. |
| `VIEWER_BASE_PATH`, `VIEWER_HOST`, `VIEWER_PORT` | optional | `viewer` web-app overrides. |

See the [Usage Guide](usage.md) and [Architecture](architecture.md) for
the full list.

## Verifying Installation

```bash
python -c "import autoware_carla_scenario; print('Installation successful!')"
```

This works without CARLA — running an actual scenario additionally
requires the built CARLA client, a live CARLA server and a valid
`CARLA_EXECUTABLE`.

## Dependencies

The package's runtime dependencies (declared in `pyproject.toml`):

- `typesafe-carla>=0.2.0,<0.3` (Linux x86_64) — the CARLA client and its Codon
  library; pulls in `typesafe-carla-toolchain`
- `pyxodr>=0.1.0` — OpenDRIVE parser used by `MapManager` / `to_opendrive`
- `opencv-python-headless>=4.8` — frame processing for the camera recorder (headless: the package makes no GUI calls)
- `numpy>=1.21`
- `pytest>=9.0.1`
- `python-dotenv>=1.2.2`
- `simple-lanelet2>=1.1.2` — the Lanelet2 binding, plus the Autoware
  regulatory-element extensions, as a single wheel
- `tqdm>=4.67.1`
- `hydra-core>=1.3.2`, `omegaconf>=2.3.0`
- `fastapi>=0.115.0`, `uvicorn[standard]>=0.34.0`, `jinja2>=3.1.0` —
  result viewer
- `pyyaml>=6.0`, `pydantic>=2.0.0`
- `ffmpeg-python>=0.2.0` — H.264 encoder driver for video recording

## Troubleshooting

### Import Errors

Ensure that:

1. You're using a supported Python (see [Python Version](#python-version)).
2. The package was installed in your active environment (`uv sync` from
   the repository root, or a fresh `uv venv` followed by `uv sync`).
3. If you see `ImportError: ... lanelet2 ...`, a `lanelet2` distribution
   from PyPI may be shadowing the one `simple-lanelet2` provides. Reset
   the environment with `rm -rf .venv && uv sync --dev`.

### CARLA Connection Issues

If you cannot connect to the CARLA simulator:

1. Ensure the CARLA server is running.
2. Check that your overrides match: `server.host`, `server.port`,
   `traffic_manager.port`.
3. Verify network connectivity to the CARLA server.

## Next Steps

Once installed, see the [Usage Guide](usage.md) to run scenarios.
