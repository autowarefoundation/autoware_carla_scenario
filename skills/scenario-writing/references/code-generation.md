# Scenario Code Generation

How to generate CARLA test scenario code for the `autoware_carla_scenario` package once the scenario design is confirmed with the user.

**What it covers:**
1. Generate Python scenario classes inheriting from `BaseScenario`
2. Create Hydra-compatible dataclass configs
3. Write YAML configuration files with proper structure
4. Register new scenarios in `run.py` and `configs.py`
5. Implement new Condition/Action/Constraint/Binding classes when explicitly approved
6. Add sweep constraints and bindings for logical scenarios

**Code Generation Process:**

1. **Read existing patterns first.** Before writing any code:
   - Read `autoware_carla_scenario/src/autoware_carla_scenario/examples/configs.py` for dataclass patterns
   - Read at least one existing scenario (e.g., `examples/intersection_passing.py`) for the class pattern
   - Read `examples/run.py` for the registration pattern
   - Read the `__init__.py` to know available exports

2. **Discover available Conditions/Actions/Constraints/Bindings.** Search source files:
   - `grep -r "class.*BaseCondition" autoware_carla_scenario/src/`
   - `grep -r "class.*BaseAction" autoware_carla_scenario/src/`
   - `grep -r "class.*BaseConstraint" autoware_carla_scenario/src/autoware_carla_scenario/sweeper/constraints.py`
   - `grep -r "class.*BaseBinding" autoware_carla_scenario/src/autoware_carla_scenario/sweeper/bindings.py`
   - Read the `build_constraint()` and `build_binding()` factory functions for type string mappings
   - Read constructor signatures of relevant classes

3. **Generate code following established patterns exactly:**
   - Match import style (from autoware_carla_scenario import ...)
   - Match docstring format (module docstring + class docstring)
   - Match type annotation style (PEP 604 unions with `|`)
   - Use `from __future__ import annotations`
   - Always include `logger = logging.getLogger(__name__)`

4. **Validate generated code:**
   - Ensure all imports resolve to existing modules
   - Verify Condition/Action classes exist before using them
   - Check dataclass field types match YAML value types
   - Confirm `ScenarioRunConfig.scenario` union includes the new config type

**Critical Rules:**
- Spawn positions MUST use `Lanelet2Pose(lanelet_id=..., s=...)`, never OpenDRIVE coordinates
- Always use `_setup_ego_spawn()` for ego, `snap_to_carla_road()` for NPCs
- Always include `GroundProjectionConfig` for coordinate conversion
- Always register a `TimeoutCondition` as fail-safe
- `is_done()` always returns `False` — termination is condition-driven
- YAML must start with `# @package _global_` for Hydra resolution
- Dataclass `name` field must match the scenario name used in `build_scenario()`

**New Condition/Action/Constraint/Binding Implementation:**
When implementing a new Condition:
- Inherit from `BaseCondition`
- Implement `check(self, world, elapsed) -> Optional[ScenarioResult]`
- Return `None` when not satisfied, `ScenarioResult` when satisfied
- Implement `get_details()` for structured logging
- Export from the appropriate `__init__.py`

When implementing a new Action:
- Inherit from `BaseAction`
- Implement `execute(self, world) -> None`
- Set appropriate `TickTiming` (PRE_TICK or POST_TICK)
- Use `once=True` for one-shot actions, `once=False` for recurring

When implementing a new Constraint:
- Inherit from `BaseConstraint`
- Implement the constraint check interface
- Register the `type:` string in `build_constraint()` factory function in `sweeper/constraints.py`
- Export from the appropriate `__init__.py`

When implementing a new Binding:
- Inherit from `BaseBinding`
- Implement the resolve interface
- Register the `type:` string in `build_binding()` factory function in `sweeper/bindings.py`
- Export from the appropriate `__init__.py`

**Output Format:**
For each file generated, show:
1. The complete file path
2. The full file content
3. For existing files (configs.py, run.py), show only the additions needed
