# Development Guide

This guide is for developers who want to contribute to the `autoware-carla-scenario` project.

## Development Environment Setup

### Prerequisites

- Python 3.10
- uv (version 0.9.7+)
- Git
- CARLA Simulator 0.10.0+

### Setting Up Your Environment

1. **Fork and clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/autoware_carla_scenario.git
cd autoware_carla_scenario
```

2. **Create a virtual environment and install dependencies**:
```bash
uv venv
source .venv/bin/activate
uv sync
```

3. **Install pre-commit hooks**:
```bash
pre-commit install
```

## Project Structure

```
autoware_carla_scenario/
├── src/
│   ├── autoware_carla_scenario/      # Main package
│   │   ├── actions/                  # Trigger-driven side effects (turn, lane change, traffic signal, attach camera)
│   │   ├── conditions/               # Pass/fail condition evaluators
│   │   │   └── composition/          # Composed conditions (lane position, area, speed, waypoint, ...)
│   │   ├── coordinate/               # Lanelet2 / OpenDRIVE / CARLA-world transforms, MapManager, snap
│   │   ├── entity/                   # VehicleEntity, EgoVehicle, the externally driven egos, spawn helpers
│   │   ├── examples/                 # Built-in scenarios + Hydra config tree (conf/)
│   │   ├── kinematics/               # Frame-tagged Vector3, velocity, acceleration
│   │   ├── sensor/                   # CameraSensorBase + CARLA RGB camera
│   │   ├── sweeper/                  # Lanelet-constraint sweeper logic
│   │   ├── tools/                    # detect-no-3d-model CLI
│   │   ├── ui/                       # Result viewer (FastAPI + scanner + runner + sweep resolver)
│   │   ├── utils/                    # Stop-line / traffic-light helpers
│   │   ├── constants.py, entity_role.py
│   │   ├── pytest_fixtures.py        # CarlaScenarioFixture
│   │   ├── scenario_base.py          # BaseScenario, EgoConfig
│   │   ├── scenario_queue.py         # ScenarioQueue (batch / retry / cooldown)
│   │   ├── scenario_runner.py        # Single-scenario execution
│   │   └── server.py                 # CarlaServerManager
│   └── hydra_plugins/
│       └── autoware_scenario_sweeper/ # Hydra-discoverable sweeper plugin shim
├── test/                              # Tests (use `-m "not integration"` to skip CARLA-dependent tests)
├── docs/                              # Documentation source (this site)
├── pyproject.toml                     # Project configuration
└── mkdocs.yml                         # Documentation config
```

## Running Tests

```bash
# Run unit tests only (skip the `integration` marker)
uv run pytest -v -m "not integration" autoware_carla_scenario/test/

# Run integration tests too (requires a running CARLA server, or one to launch: `scenario-setup` or CARLA_EXECUTABLE)
uv run pytest -v autoware_carla_scenario/test/
```

The workspace `pyproject.toml` defines these custom pytest markers:

- `integration` — tests that need a live CARLA server; they skip themselves
  without one.
- `slow` — tests that export a package and build a wheelhouse over the
  network. CI runs them in jobs of their own (`-m slow`).
- `codon` — tests that compile with Codon, some 1,240 of them. A pull
  request runs them in one leg per architecture, split into two shards
  (`AUTOWARE_CARLA_SCENARIO_TEST_SHARD=K/N` runs the K-th of N shards of a
  selection; the conftest hashes each test's node id into one).
- `codon_selftest` — the `codon` tests of the checker itself, on code of
  their own. A pull request runs them only when it changes something they
  read (the checker, the locked dependencies, the tests, CI; the list is in
  `.github/actions/codon-selftests/action.yml`); the nightly run and every
  other event run them. The compile checks of the package's own code (the
  library check, the built-in scenarios, the template) run on every pull
  request.

Local verification is the same command CI runs — no container involved:

```bash
# Full pytest suite (matches CI's `test` job)
uv run pytest -n auto

# Pre-commit (matches CI's `lint-and-format` job)
uv run pre-commit run --all-files
```

## Code Style and Quality

This project follows the same coding standards as the parent repository:

- **Type Hints**: All functions and methods must include type annotations
- **Docstrings**: Use Google-style docstrings
- **Code Formatting**: Enforced by Ruff formatter via pre-commit hooks
- **Linting**: Ruff linter with auto-fix enabled
- **Type checking**: `mypy --ignore-missing-imports` over both `src/` and `test/`

The full pre-commit configuration lives in
[`.pre-commit-config.yaml`](https://github.com/hakuturu583/autoware_carla_scenario/blob/master/.pre-commit-config.yaml)
at the repository root and runs the following hooks:

- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`,
  `check-added-large-files`, `check-merge-conflict`, `debug-statements`,
  `mixed-line-ending`
- `ruff` (with `--fix`) and `ruff-format` (`astral-sh/ruff-pre-commit`)
- `mypy` (local hook over `src/` and `test/`, run when a commit touches
  Python or the project configuration)

pre-commit runs no tests: the fast suite runs some 1,240 Codon compiles,
which took minutes on every commit. Run `uv run pytest` yourself; CI
runs it on every pull request.

## Building Documentation Locally

```bash
# Serve documentation locally
uv run mkdocs serve --config-file autoware_carla_scenario/mkdocs.yml

# Build documentation
uv run mkdocs build --config-file autoware_carla_scenario/mkdocs.yml
```

Visit `http://127.0.0.1:8000` to view the documentation.

## Contributing

### Contribution Guidelines

1. **Create an issue** first to discuss your proposed changes
2. **Fork the repository** and create a feature branch
3. **Write tests** for your changes
4. **Ensure all tests pass** and code is properly formatted
5. **Submit a pull request** with a clear description

### Commit Message Convention

Follow conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks
