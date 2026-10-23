# Scenario Package Template

A minimal, standalone **scenario package** for the
[`autoware-carla-scenario`](../../autoware_carla_scenario) framework. Copy this
directory, rename it, and you have your own scenarios living **outside** the
framework — with their own Python classes, their own config dataclasses, and
their own Hydra YAML configs.

This template exists because scenarios and their config `dataclass`es used to
only be definable *inside* `autoware_carla_scenario`. They no longer have to be.

## Layout

```
scenario_package_template/
├── pyproject.toml                     # declares the entry point (see below)
└── src/my_scenario_package/
    ├── __init__.py                    # register() — the entry point target
    ├── reach_goal.py                  # the scenario class (BaseScenario)
    ├── configs.py                     # the scenario's config dataclass
    └── conf/scenario/reach_goal/
        └── default.yaml               # a concrete scenario (YAML)
```

`tests/test_typecheck.py` (next to `src/`) runs the static check on the scenario.

## How it plugs in

Two framework primitives do all the work, both imported from the public API:

```python
from autoware_carla_scenario import register_conf_dir, register_scenario
```

- `register_scenario(name, ScenarioClass, ConfigDataclass)` — makes
  `scenario.name: <name>` build your class.
- `register_conf_dir(path)` — adds your `conf/` to Hydra's search path so
  `scenario=reach_goal/default` resolves.

Both are called from `register()` in `__init__.py`. That function is advertised
as an **entry point** in `pyproject.toml`:

```toml
[project.entry-points."autoware_carla_scenario.scenarios"]
my_scenario_package = "my_scenario_package:register"
```

When the package is installed, the `scenario` CLI discovers this entry point at
start-up (via `load_scenario_plugins()`) and calls `register()` for you. No edit
to `autoware_carla_scenario` is required.

## Try it

From the repository root (with the framework already installed in your env):

```bash
# Install this package into the same environment.
uv pip install -e examples/scenario_package_template

# Run the scenario — it is now discoverable by name.
uv run scenario scenario=reach_goal/default map=nishishinjuku

# Batch/glob and CLI overrides work exactly like the built-ins:
uv run scenario scenario='reach_goal/*' map=nishishinjuku
uv run scenario scenario=reach_goal/default scenario.timeout_seconds=20
```

## Check it

The static check compiles each scenario with Codon and reports a type error,
a misspelled config key or a bad value at its line, before CARLA starts (see
the framework's *Static Check* documentation):

```bash
# Only this package's scenarios: about 15 s per config.
uv run scenario-check scenario='reach_goal/*'

# One config, with the same overrides as the runner takes.
uv run scenario-check scenario=reach_goal/default scenario.timeout_seconds=20
```

`scenario-check` without arguments (or `scenario='*'`) checks every installed
scenario, the built-in ones too, which takes minutes. The exit status is 0 when
every config passed, 1 when one failed the check, and 2 when it could not run:
no Codon compiler, no config matching `scenario=`, or an override of a key
the config does not have (the message suggests the closest key).

`tests/test_typecheck.py` runs the same check under pytest (`uv run pytest`).

## Making it a workspace member (optional)

To develop the package alongside the framework in this monorepo, add it to the
root `pyproject.toml` workspace:

```toml
[tool.uv.workspace]
members = [
    "autoware_carla_scenario",
    "examples/scenario_package_template",
]
```

then run `uv sync`. This is optional — the package works as a plain editable
install too.
