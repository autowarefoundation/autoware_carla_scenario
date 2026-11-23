# Scenario Code Generation

How to generate CARLA test scenario code for `autoware_carla_scenario` once the scenario design is confirmed with the user.

**What it covers:**
1. Python scenario classes inheriting from `BaseScenario`
2. Hydra-compatible dataclass configs
3. YAML configuration files
4. Registration: a scenario package's `register()`, or a built-in's `_register_builtin_scenario` in `examples/run.py`
5. New Condition/Action/Constraint/Binding classes, when explicitly approved
6. Sweep constraints and bindings for logical scenarios

**Code Generation Process:**

1. **Decide where the scenario lives** (the `scenario-writing` skill, "Where the Scenario Lives"). For a scenario package, start from `uv run scenario-new <package> --scenario <name>` and edit what it generates rather than writing the files from scratch.

2. **Read existing patterns first.** Before writing any code:
   - At least one built-in scenario close to the new one (e.g. `examples/temporary_stop.py`, `examples/cut_in.py` for NPCs) for the class pattern
   - `examples/configs.py` for dataclass patterns (and the shared configs: `EgoVehicleConfig`, `MapConfig`, ...)
   - For a built-in: `examples/run.py`'s `_register_builtin_scenario` calls; for a package: its generated `__init__.py`
   - The top-level `autoware_carla_scenario/__init__.py` `__all__` to know the public API

3. **Discover available Conditions/Actions/Constraints/Bindings.** Search the source (under `autoware_carla_scenario/src/autoware_carla_scenario/`):
   - `grep -rn "(BaseCondition)\|(CompositionCondition)" conditions/`
   - `grep -rn "(BaseAction)" actions/`
   - `sweeper/constraints.py`: `_LEAF_REGISTRY` and the combinators in `parse_constraint()`
   - `sweeper/bindings.py`: `_BINDING_REGISTRY`
   - Read constructor signatures of relevant classes

4. **Generate code following established patterns exactly:**
   - `from __future__ import annotations`, then a module docstring and class docstrings
   - Import from the top-level package (`from autoware_carla_scenario import ...`), never from `examples`; the CARLA API as `import typesafe_carla.carla as carla`
   - PEP 604 unions (`X | None`) in annotations
   - `logger = logging.getLogger(__name__)`
   - Every attribute assigned on `self` declared at class level (`_config: MyScenarioConfig`)

5. **Validate generated code:**
   - `uv run scenario-check scenario='<name>/*'` compiles the scenario with Codon and reports type errors, misspelled config keys and bad values
   - For a built-in module, also `uv run scenario-check --library autoware_carla_scenario.examples.<name>`
   - Check that dataclass field types match the YAML value types
   - For a package, its generated `tests/test_typecheck.py` runs the same check under pytest

**Critical Rules:**
- Positions MUST be `Lanelet2Pose(lanelet_id=..., s=...)` (`ego.spawn_lanelet_id` / `ego.spawn_s` in YAML), never OpenDRIVE coordinates
- `_setup_ego_spawn()` for the ego, `snap_to_carla_road(pose, world, ground_projection=self._ground_projection)` for NPCs
- Accept a `GroundProjectionConfig` and pass it to `BaseScenario.__init__`
- Always register a `TimeoutCondition` as a fail condition
- `is_done()` returns `False` — termination is condition-driven
- Initial world state (traffic lights, ...) through `register_init`, not by executing actions inside `setup()`
- YAML starts with `# @package _global_`
- The dataclass `name` field matches the registered scenario name
- The scenario name matches `^[a-z][a-z0-9_]*$` and names no built-in scenario or config group (`scenario-new` refuses those)

**Registering a built-in scenario** (only when it ships with the framework):
- `_register_builtin_scenario("<name>", MyScenario, MyScenarioConfig)` in `examples/run.py`, after the others
- The name in `autoware_carla_scenario.scaffold.names.BUILTIN_SCENARIO_NAMES` (sorted; `test_scaffold.py` checks the list against `run.py`)
- The module in `typecheck/library.py`'s `CHECKED` (sorted), or `EXCLUDED` as `NOT_YET_CHECKED` if it does not compile yet (`test_typecheck_library.py` fails until it is listed)
- A page under `docs/scenarios/` and in `mkdocs.yml`'s nav, as the other built-ins have

**New Condition/Action/Constraint/Binding Implementation:**

When implementing a new Condition:
- Inherit from `BaseCondition` (or `CompositionCondition` for an entity-based one)
- Implement `check(self, world, elapsed) -> Optional[ScenarioResult]`: `None` when not satisfied, a `ScenarioResult` when satisfied
- Implement `get_details()` for structured logging; take a keyword-only `label`

When implementing a new Action:
- Inherit from `BaseAction`
- Implement `execute(self, world) -> None`
- Pass `condition`, `timing` (`TickTiming.PRE_TICK` / `POST_TICK`) and `once` (one-shot or recurring) to the base constructor

When implementing a new Constraint:
- Subclass `Constraint` in `sweeper/constraints.py` as a frozen dataclass (its fields are its YAML keys) and implement `evaluate(self, lanelet) -> bool`
- Add the `type:` string to `_LEAF_REGISTRY` (a combinator goes in `parse_constraint()`)
- Export it from `sweeper/__init__.py`

When implementing a new Binding:
- A dataclass with `target_key` in `sweeper/bindings.py` implementing `resolve(self, lanelet_id, lanelet_map, routing_graph=None) -> BindingResult`
- Add the `type:` string to `_BINDING_REGISTRY`
- Export it from `sweeper/__init__.py`

For any of these: export public classes from the top-level `__init__.py`, mirror public API changes in the typed model (`typecheck/codon/`; `test_typecheck_model.py` fails until it matches), list a new module in `typecheck/library.py`, and add tests under `autoware_carla_scenario/test/`.

**Output Format:**
For each file generated, show:
1. The complete file path
2. The full file content
3. For existing files (`configs.py`, `run.py`, `__init__.py`), show only the additions needed
