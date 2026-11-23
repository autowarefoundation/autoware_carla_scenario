---
name: write-scenario
description: Interactively create a CARLA test scenario for autoware_carla_scenario step by step, from a natural-language description to a concrete scenario, its review, and a logical scenario with sweep constraints. Use when the user asks to write or create a new scenario end to end.
---

Guide the user through creating a CARLA test scenario for `autoware_carla_scenario`. Use the `scenario-writing` skill for domain knowledge (and its `references/`).

## Input

The user describes the scenario they want in natural language, in the request that invoked this skill. If they gave no description, ask for one.

## Interactive Workflow

Work through these stages sequentially, confirming with the user at each stage before proceeding.

### Stage 1: Understand the Scenario Concept

Parse the user's natural language description and clarify:
1. What behavior is being tested? (e.g., intersection passing, traffic light compliance, lane change)
2. What is the expected outcome? (pass condition)
3. What constitutes failure? (fail conditions)
4. Are NPC vehicles needed? Where relative to ego?
5. Where should the scenario live: a scenario package of its own (the default for the user's scenarios), or a built-in shipped with the framework? For a package, its name.

Ask the user focused clarifying questions. Do not assume — confirm the scenario concept before writing code.

### Stage 2: Discover Available Building Blocks

Before writing code, dynamically discover existing Conditions and Actions:
1. Search for the conditions (`BaseCondition` / `CompositionCondition` subclasses) in `autoware_carla_scenario/src/autoware_carla_scenario/conditions/`
2. Search for the actions (`BaseAction` subclasses) in `autoware_carla_scenario/src/autoware_carla_scenario/actions/`
3. Read constructor signatures of relevant classes
4. Read the built-in scenario closest to the user's (`examples/*.py`)

Present the user with a list of applicable existing Conditions/Actions for their scenario.

If existing building blocks are insufficient:
1. Explain what is missing and why
2. Propose the new Condition/Action with its interface
3. Wait for user approval before implementing

### Stage 3: Write Concrete Scenario

Follow `scenario-writing`'s `references/code-generation.md`.

**Scenario package** — generate it, install it, then fill it in:
```bash
uv run scenario-new <package> --scenario <name>
uv pip install -e <package>
```
1. **Python scenario class** (`<package>/src/<package>/<name>.py`) — inheriting `BaseScenario`
2. **Dataclass config** (`<package>/src/<package>/configs.py`)
3. **YAML config** (`<package>/src/<package>/conf/scenario/<name>/<variant>.yaml`) — with Lanelet2 spawn coordinates
4. **Registration** — already wired by `scenario-new` (`register()` in `__init__.py`, the `autoware_carla_scenario.scenarios` entry point)

**Built-in scenario** — under `autoware_carla_scenario/src/autoware_carla_scenario/`:
1. **Python scenario class** (`examples/<name>.py`)
2. **Dataclass config** (add to `examples/configs.py`)
3. **YAML config** (`examples/conf/scenario/<name>/<variant>.yaml`)
4. **Registration** — as `references/code-generation.md` lists under "Registering a built-in scenario"

**Critical: spawn positions must use Lanelet2 coordinates (lanelet_id + s), never OpenDRIVE coordinates.**

Show the user each file before writing. Confirm the overall design, then write all files.

Then run the static check, and fix what it reports before going on:
```bash
uv run scenario-check scenario='<name>/*'
```

### Stage 4: Debug Concrete Scenario

Follow the `debug-concrete-scenario` skill for `<name>/<variant>` to execute the scenario, analyze results, and iterate until it passes.

That skill autonomously runs the execute → analyze → fix loop and reports back. Once the user confirms the behavior matches their intent, proceed to Stage 5.

### Stage 5: Review and Define Logical Scenario Parameters

Once the concrete scenario passes, use the `scenario-review` skill to analyze the parameter design:

1. **Identify candidates:** List all hardcoded values that could vary across test runs
2. **Analyze orthogonality:** Check each parameter pair for independence using the orthogonality checklist
3. **Classify each candidate:** Determine whether each value should be a parameter, binding, Python-computed, or constant
4. **Validate NPC dependencies:** Ensure NPC positions that depend on ego are computed in Python, not swept independently
5. **Verify sweeper compatibility:** Confirm constraint/binding types exist for the proposed parameter design
6. **Present review summary:** Show the classification and orthogonality check results, then confirm the parameter set with the user

Update the YAML config to include concrete parameter values (these serve as the default concrete scenario).

### Stage 6: Discover Available Constraints/Bindings and Add Sweep Config

Before writing sweep constraints, dynamically discover existing Constraint and Binding types:
Follow `scenario-writing`'s `references/constraints-reference.md`: where the types are registered in the source, how to read their keys, and the rules (one swept key, bindings, `exclude`).

Present the user with applicable existing Constraints/Bindings for their scenario.

If existing Constraints/Bindings are insufficient:
1. Explain what is missing and why existing types cannot express the desired filter
2. Propose the new Constraint/Binding with its interface and `type:` string
3. Wait for user approval before implementing
4. Implement it as `scenario-writing`'s `references/code-generation.md` describes

Then edit the `sweep:` section of the YAML:
1. Help the user compose constraints that work across maps
2. Add bindings for derived parameters
3. Avoid hardcoding specific Lanelet IDs in constraints
4. Keep a single swept key: derive every other parameter (NPC lanelets, spawn `s`, the route) with bindings
5. Preview the expansion without CARLA: `uv run scenario-expand scenario=<name>/<variant>`

Suggest committing after this stage.

### Stage 7: Guide Multirun Execution

Provide the exact command for logical scenario execution:
```bash
uv run scenario --multirun scenario=<name>/<variant> hydra/sweeper=lanelet_constraint
```

Remind the user:
- Both `hydra/sweeper=lanelet_constraint` AND `--multirun` are required
- Without the sweeper name, `--multirun` runs Hydra's basic sweeper, which ignores `sweep:` and runs the concrete scenario (it warns)
- Use `map=<name>` to switch maps (the maps are `conf/map/*.yaml`: `nishishinjuku`, `town10hd_opt`)
- Results land under `multirun/`; browse them with `uv run viewer`
