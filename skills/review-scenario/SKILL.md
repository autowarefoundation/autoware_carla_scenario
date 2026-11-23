---
name: review-scenario
description: Review a CARLA scenario's parameter design for orthogonality and sweeper compatibility. Use when the user names an existing scenario (or its YAML config) and asks for its parameters to be reviewed.
---

Review the parameter design of an existing CARLA test scenario. Use the `scenario-review` skill for domain knowledge.

## Input

The user specifies a scenario (its name or the path to its YAML config) in the request that invoked this skill.

## Review Workflow

### Step 1: Read the Scenario

1. Find and read the scenario's YAML config: `conf/scenario/<name>/` in its scenario package (`src/<package>/conf/`), or for a built-in in `autoware_carla_scenario/src/autoware_carla_scenario/examples/conf/scenario/`
2. Read the corresponding Python scenario class (the module registered under that name: the package's `register()`, or `examples/run.py`'s `_register_builtin_scenario` calls)
3. Read its dataclass config (`configs.py` beside it)

If no specific scenario is given, ask the user which scenario to review.

### Step 2: Identify All Hardcoded Values

List all values in the YAML and Python code that could potentially be parameters:
- Lanelet IDs, s-values, speeds, distances, timeouts, thresholds
- NPC spawn positions and their relationship to ego
- Traffic light configurations

### Step 3: Analyze Orthogonality

Apply the orthogonality analysis from the `scenario-review` skill:
1. Check each parameter pair for independence
2. Identify derived values that should be bindings or computed in Python
3. Flag any anti-patterns (coupled lanelets, hardcoded s-values, excessive parameters)

### Step 4: Discover Available Constraint/Binding Types

Before recommending sweep configuration, dynamically discover existing types:
1. Constraints: `_LEAF_REGISTRY` and the combinators in `parse_constraint()` (`autoware_carla_scenario/src/autoware_carla_scenario/sweeper/constraints.py`)
2. Bindings: `_BINDING_REGISTRY` (`sweeper/bindings.py`)
3. Verify that recommended constraint/binding types actually exist, with the keys their dataclass fields name
4. Check what the current sweep expands to: `uv run scenario-expand scenario=<name>/<variant>`

### Step 5: Review Existing Sweep Config (if present)

If the scenario already has a `sweep:` section:
1. Verify constraints are correct and use existing constraint types, under a single swept key
2. Verify bindings are appropriate: every key a case depends on moves with the pick (a fixed `spawn_s`, NPC lanelet or route that only fits the original case is a bug), and `exclude` covers where walk-backs land
3. Check for missing constraints or bindings
4. Identify any anti-patterns in the current configuration

### Step 6: Present Recommendations

Present the review summary using the format from the `scenario-review` skill:
- Parameters vs. bindings vs. constants vs. Python-computed
- Orthogonality check results
- Specific recommendations for improvements
- Any anti-patterns found

Confirm the recommendations with the user before suggesting changes.

### Step 7: Apply Changes (if requested)

If the user agrees with the recommendations:
1. Update the YAML config with revised parameter/binding/constraint structure
2. Update the Python scenario class if an NPC position must be derived from the ego's (prefer a binding into a config field; compute in `setup()` only when no binding expresses it)
3. Update the dataclass config if parameters are added or removed
