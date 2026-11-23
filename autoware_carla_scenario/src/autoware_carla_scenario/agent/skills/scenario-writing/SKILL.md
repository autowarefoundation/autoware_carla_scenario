---
name: scenario-writing
description: This skill should be used when the user asks to "write a scenario", "create a CARLA scenario", "add a test scenario", "write a concrete scenario", "create a logical scenario", "add constraints", "run multirun", "spawn NPC", "add condition", "add action", mentions scenario writing for autoware_carla_scenario, or discusses CARLA test scenario creation workflow. Provides comprehensive guidance for the full scenario lifecycle from concrete to logical scenarios.
---

# CARLA Scenario Writing Guide

## Purpose

Guide the interactive creation of CARLA test scenarios for `autoware_carla_scenario`. Cover the full lifecycle: concrete scenario authoring, the static check, parameter tuning, logical scenario conversion (sweep constraints and bindings), and multirun execution.

The user-facing walkthrough this skill follows is the framework's Tutorial (`docs/tutorial.md`); when this skill and the code disagree, the code (and that tutorial) wins.

## Where Things Are

These skills work in the framework's own checkout and in a scenario package's repository alike. Paths in them mean:

- **`<framework>/`** — the installed `autoware_carla_scenario` package, the framework's source to read (find it with `uv run python -c "import autoware_carla_scenario, pathlib; print(pathlib.Path(autoware_carla_scenario.__file__).parent)"`). In the framework's own checkout it is `autoware_carla_scenario/src/autoware_carla_scenario/`.
- **`docs/<page>.md`** — a page of the framework's documentation: https://autowarefoundation.github.io/autoware_carla_scenario/ (the page of that name), or `autoware_carla_scenario/docs/<page>.md` in the framework's checkout.
- Paths of a scenario package (`src/<package>/...`) are relative to its repository.

Editing the framework itself (a built-in scenario, a new Condition or constraint) needs the framework's checkout.

## Where the Scenario Lives

Decide this first, with the user:

| | Scenario package (default for the user's own scenarios) | Built-in scenario (shipped with the framework) |
|---|---|---|
| Create | `uv run scenario-new <package> --scenario <name>` (`--output-dir DIR`) | Edit the framework under `<framework>/examples/` |
| Scenario class | `src/<package>/<name>.py` | `examples/<name>.py` |
| Config dataclass | `src/<package>/configs.py` | `examples/configs.py` |
| Concrete YAML | `src/<package>/conf/scenario/<name>/<variant>.yaml` | `examples/conf/scenario/<name>/<variant>.yaml` |
| Registration | `register()` in `src/<package>/__init__.py`, generated with its entry point | `_register_builtin_scenario` in `examples/run.py`, and the lists in `references/code-generation.md` ("Registering a built-in scenario") |
| Install | `uv pip install -e <package>` | — |

`scenario-new` refuses names that would clash (a built-in scenario or config group, a module the package imports, `configs`, `conf`, ...); the name must match `^[a-z][a-z0-9_]*$`. Import everything from the top-level `autoware_carla_scenario` package, not from `examples`.

## Scenario Writing Workflow

Follow these stages in order. Commit after completing the concrete scenario stage.

### Stage 1: Concrete Scenario

Create a Python class inheriting `BaseScenario` with Lanelet2 spawn coordinates. See `references/scenario-patterns.md` for the patterns and `references/code-generation.md` for the rules every generated file follows.

**Critical rules:**
- **Always specify positions in the Lanelet2 coordinate system** (`Lanelet2Pose(lanelet_id=..., s=...)`, `ego.spawn_lanelet_id` / `ego.spawn_s` in YAML). OpenDRIVE road/lane IDs are generated and change with the map conversion.
- **Use `self._setup_ego_spawn()` for the ego and `snap_to_carla_road(pose, world, ground_projection=self._ground_projection)` for NPCs.** Without ground projection, spawns clip into terrain or roadside objects.
- **Declare every attribute assigned on `self` at class level** (`_config: MyScenarioConfig`): the static check (Codon) refuses one it has not seen declared.
- **Register a `TimeoutCondition` fail condition.** Scenarios run as Python; termination is condition-driven and `is_done()` returns `False`.
- **The scenario passes as soon as *any* pass condition fires.** To require several things, combine them in one `AndCondition`, wrapping each state that only has to be reached once in a `StickyCondition`.
- **Prefer composing existing Conditions/Actions** over creating new ones. They leave structured logs (condition tree, labels) for debugging.

**Scenario class pattern:**
```python
class MyScenario(BaseScenario):
    _config: MyScenarioConfig  # every attribute set on self, declared here

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: MyScenarioConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection)
        self._config = config or MyScenarioConfig()

    def setup(self) -> None:
        self._setup_ego_spawn()           # Lanelet2 -> OpenDRIVE -> CARLA, snapped to the road
        # register_init(...): actions run once as the scenario starts (e.g. traffic lights)
        # register_pre_tick(...) / register_post_tick(...): actions, each gated by its condition
        # register_pass_condition(...)
        # register_fail_condition(...)  -- always a TimeoutCondition

    def is_done(self) -> bool:
        return False  # Driven by conditions
```

Then statically check it before CARLA is involved:

```bash
uv run scenario-check scenario='<name>/*'
```

It compiles the scenario with Codon and reports type errors, misspelled config keys and bad values at their line (`docs/typecheck.md`). A built-in module is also checked with `uv run scenario-check --library autoware_carla_scenario.examples.<name>`.

### Stage 2: Debug Concrete Scenario

Use the `debug-concrete-scenario` skill (with the `scenario-debugging` skill for domain knowledge) to execute, analyze, and iterate on the scenario until it passes. Once the user confirms the behavior matches intent, proceed to Stage 3.

### Stage 3: Review and Define Logical Scenario Parameters

Use the `scenario-review` skill to analyze the parameter design before defining logical parameters:

1. Identify candidate parameters from the concrete scenario
2. Analyze orthogonality between parameter pairs
3. Classify each candidate as the swept parameter, a binding, Python-computed, or a constant
4. Derive NPC positions that depend on the ego from the pick (a binding, or Python), never sweep them independently
5. Verify sweeper compatibility (the constraint and binding types exist)
6. Present a review summary for user confirmation

After the review is confirmed, write parameters as dataclass fields in `configs.py` and the corresponding YAML in `conf/scenario/`. Concrete and logical parameters share the same YAML file (or the sweep gets its own variant, e.g. `sweep.yaml`).

### Stage 4: Discover Constraints/Bindings and Add Sweep Config

**Do not rely on a hardcoded list.** `references/constraints-reference.md` says where the sweeper's constraint and binding types are registered in the source, and the rules a `sweep:` follows (one swept key, bindings, `exclude`), with examples.

**When existing Constraints/Bindings are insufficient:** propose the new type to the user with rationale and wait for approval, then implement it as `references/code-generation.md` describes.

Then edit the `sweep:` section in the scenario YAML. Avoid hardcoding Lanelet IDs in logical scenario constraints — this breaks cross-map portability. Preview the expansion without CARLA (`uv run scenario-expand scenario=<name>/<variant>`) before running it.

### Stage 5: Run Logical Scenario

```bash
# Logical scenario (parameter sweep)
uv run scenario --multirun scenario=<name>/<variant> hydra/sweeper=lanelet_constraint

# Concrete scenario (single run, no sweeper)
uv run scenario scenario=<name>/<variant>

# Every variant of a scenario as a batch
uv run scenario scenario='<name>/*'

# A different map (the maps are conf/map/*.yaml: nishishinjuku, town10hd_opt)
uv run scenario scenario=<name>/<variant> map=town10hd_opt
```

**Warning:** `--multirun` without `hydra/sweeper=lanelet_constraint` runs Hydra's basic sweeper, which ignores the scenario's `sweep:` and runs the scenario as configured (it warns).

Results go to `outputs/` (single runs, batches) and `multirun/` (sweeps); browse them with `uv run viewer` (port 9000).

## Discovering Available Conditions and Actions

**Do not rely on a hardcoded list.** Dynamically discover available types:

1. Conditions derive from `BaseCondition` (`conditions/base.py`): `conditions/` and `conditions/composition/` (entity-based ones such as `SpeedCondition`, `StandstillCondition`, `EntityLanePositionCondition`)
2. Actions derive from `BaseAction` (`actions/base.py`): `actions/`
3. Read constructor signatures (most take a keyword-only `label`)
4. Check the top-level `autoware_carla_scenario/__init__.py` `__all__` for the public API

**When existing Conditions/Actions are insufficient:** propose the new Condition/Action to the user with rationale and wait for approval, then implement it as `references/code-generation.md` describes.

## Additional Resources

### Reference Files

- **`references/constraints-reference.md`** — Sweeper constraint and binding types, rules, and YAML examples
- **`references/scenario-patterns.md`** — Scenario class patterns and common condition compositions
- **`references/code-generation.md`** — How to generate the scenario class, dataclass config, YAML and registration, and the rules every generated file follows

### Key Source Locations (under `<framework>/`)

- Scenario base class: `scenario_base.py`
- Conditions: `conditions/`; actions: `actions/`
- Coordinate transforms and the map: `coordinate/` (`MapManager`, `snap_to_carla_road`, `to_opendrive`)
- Sweeper: `sweeper/`
- Built-in scenarios: `examples/` (the patterns to copy); their Hydra configs: `examples/conf/`
- Scenario package generator: `scaffold/`; template: `examples/scenario_package_template/` in the framework's repository
