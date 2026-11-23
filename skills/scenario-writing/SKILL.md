---
name: scenario-writing
description: This skill should be used when the user asks to "write a scenario", "create a CARLA scenario", "add a test scenario", "write a concrete scenario", "create a logical scenario", "add constraints", "run multirun", "spawn NPC", "add condition", "add action", mentions scenario writing for autoware_carla_scenario, or discusses CARLA test scenario creation workflow. Provides comprehensive guidance for the full scenario lifecycle from concrete to logical scenarios.
---

# CARLA Scenario Writing Guide

## Purpose

Guide the interactive creation of CARLA test scenarios for `autoware_carla_scenario`. Cover the full lifecycle: concrete scenario authoring, the static check, parameter tuning, logical scenario conversion (sweep constraints and bindings), and multirun execution.

The user-facing walkthrough this skill follows is `autoware_carla_scenario/docs/tutorial.md`; when this skill and the code disagree, the code (and that tutorial) wins.

## Where the Scenario Lives

Decide this first, with the user:

| | Scenario package (default for the user's own scenarios) | Built-in scenario (shipped with the framework) |
|---|---|---|
| Create | `uv run scenario-new <package> --scenario <name>` (`--output-dir DIR`) | Edit the framework under `autoware_carla_scenario/src/autoware_carla_scenario/examples/` |
| Scenario class | `src/<package>/<name>.py` | `examples/<name>.py` |
| Config dataclass | `src/<package>/configs.py` | `examples/configs.py` |
| Concrete YAML | `src/<package>/conf/scenario/<name>/<variant>.yaml` | `examples/conf/scenario/<name>/<variant>.yaml` |
| Registration | `register()` in `src/<package>/__init__.py` (`register_scenario`, `register_conf_dir`), found through the `autoware_carla_scenario.scenarios` entry point | `_register_builtin_scenario("<name>", MyScenario, MyScenarioConfig)` in `examples/run.py`, plus the name in `scaffold.names.BUILTIN_SCENARIO_NAMES` (sorted) |
| Static check of the module | The package's `tests/test_typecheck.py` (generated) | List the module in `typecheck/library.py`'s `CHECKED` (sorted), or `EXCLUDED` as `NOT_YET_CHECKED` |
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

It compiles the scenario with Codon and reports type errors, misspelled config keys and bad values at their line (`autoware_carla_scenario/docs/typecheck.md`). A built-in module is also checked with `uv run scenario-check --library autoware_carla_scenario.examples.<name>`.

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

**Do not rely on a hardcoded list.** Read the sweeper source (`autoware_carla_scenario/src/autoware_carla_scenario/sweeper/`):

1. **Constraints** (`constraints.py`): leaf types are the `_LEAF_REGISTRY` mapping (`type:` string → class); the combinators `and`, `or`, `not`, `previous_of`, `following_of` are handled in `parse_constraint()`. Each class's docstring shows its YAML and its dataclass fields are its keys.
2. **Bindings** (`bindings.py`): the `_BINDING_REGISTRY` mapping; `parse_binding()` takes the dataclass fields (except `target_key`) as keys.

See `references/constraints-reference.md` for the YAML structure, the rules (one swept key, `exclude`, walk-backs) and examples.

**When existing Constraints/Bindings are insufficient:**
1. Propose the new type to the user with rationale, and wait for approval
2. A constraint subclasses `Constraint` and implements `evaluate(self, lanelet) -> bool` (a frozen dataclass, as the others are); add it to `_LEAF_REGISTRY` (a combinator to `parse_constraint()`)
3. A binding is a dataclass with `target_key` and `resolve(self, lanelet_id, lanelet_map, routing_graph=None) -> BindingResult`; add it to `_BINDING_REGISTRY`
4. Export it from `sweeper/__init__.py`, and follow `autoware_carla_scenario/docs/typecheck.md` for the static check (a public API change needs the same change in `typecheck/codon/`)

Then edit the `sweep:` section in the scenario YAML. Avoid hardcoding Lanelet IDs in logical scenario constraints — this breaks cross-map portability. Map-specific exclusions (like `no_3d_model_lanelet_ids`) are in `conf/map/<map_name>.yaml` and referenced as `${map.no_3d_model_lanelet_ids}`.

**Tip:** Insert `type: equals, value: any` as a placeholder. After a multirun, replace it with concrete lanelet IDs from failing runs to narrow down issues.

Preview the expansion without CARLA before running it:

```bash
uv run scenario-expand scenario=<name>/<variant>
```

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

**When existing Conditions/Actions are insufficient:**
1. Propose the new Condition/Action to the user with rationale, and wait for approval
2. A condition inherits `BaseCondition` and implements `check(self, world, elapsed) -> Optional[ScenarioResult]` (and `get_details()` for its log)
3. An action inherits `BaseAction` and implements `execute(self, world) -> None`; `timing` (`TickTiming.PRE_TICK` / `POST_TICK`), `condition` and `once` come from the base constructor
4. Export it from the subpackage's and the top-level `__init__.py`, and update the typed model (`typecheck/codon/`) and `typecheck/library.py` as `autoware_carla_scenario/docs/typecheck.md` describes

## Additional Resources

### Reference Files

- **`references/constraints-reference.md`** — Sweeper constraint and binding types, rules, and YAML examples
- **`references/scenario-patterns.md`** — Scenario class patterns and common condition compositions
- **`references/code-generation.md`** — How to generate the scenario class, dataclass config, YAML and registration, and the rules every generated file follows

### Key Source Locations (under `autoware_carla_scenario/src/autoware_carla_scenario/`)

- Scenario base class: `scenario_base.py`
- Conditions: `conditions/`; actions: `actions/`
- Coordinate transforms and the map: `coordinate/` (`MapManager`, `snap_to_carla_road`, `to_opendrive`)
- Sweeper: `sweeper/`
- Built-in scenarios: `examples/` (the patterns to copy); their Hydra configs: `examples/conf/`
- Scenario package generator: `scaffold/`; template: `examples/scenario_package_template/` at the repository root
