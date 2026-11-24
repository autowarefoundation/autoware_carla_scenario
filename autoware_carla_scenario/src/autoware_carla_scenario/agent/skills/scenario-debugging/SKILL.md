---
name: scenario-debugging
description: This skill should be used when the user asks to "debug a scenario", "run a scenario", "fix a scenario", "why does my scenario fail", "scenario timeout", "spawn failure", "analyze scenario output", or discusses troubleshooting CARLA test scenario execution. Provides guidance on log analysis, common failure patterns, and iterative debugging.
---

# CARLA Scenario Debugging Guide

> Paths: `<framework>/` (the installed framework's source) and `docs/<page>.md` (its documentation) as in the `scenario-writing` skill's "Where Things Are".

## Purpose

Guide the autonomous debugging of CARLA concrete scenarios: execute, analyze logs, diagnose failures, fix issues, and iterate until the scenario passes.

## Execution Command

```bash
# Static check first: compiles the scenario with Codon, no CARLA needed
uv run scenario-check scenario=<name>/<variant>

# Run a concrete scenario (needs a reachable CARLA server)
uv run scenario scenario=<name>/<variant>

# With a specific map (conf/map/*.yaml: nishishinjuku, town10hd_opt)
uv run scenario scenario=<name>/<variant> map=town10hd_opt

# What a sweep expands to, without CARLA
uv run scenario-expand scenario=<name>/<variant>
```

## Where the Results Are

Each run writes `outputs/YYYY-MM-DD/HH-MM-SS/` (sweeps: `multirun/`). The full list is `docs/usage.md`, "Output Files"; the ones to read first:

- `{ScenarioName}_result.json` — `passed`, `message`, and the status of every condition (the condition tree)
- `{ScenarioName}.trajectory.jsonl` — the run tick by tick, with the map, plan, conditions and verdict (`docs/trajectory_log.md`); convertible to a Rerun recording to watch the run (`docs/usage.md`, "Seeing a run")
- `batch_results.json` — the summary of a batch (`scenario='<name>/*'`)

`uv run viewer` (http://localhost:9000) browses the same results: sessions, each case's condition tree, and a re-run button.

## Log Analysis

### Key Output Sections

When analyzing scenario output, look for these sections in order:

1. **Initialization**: Module imports, config loading, map loading
2. **Spawn**: Ego and NPC spawn results (Lanelet2 → OpenDRIVE → CARLA coordinate chain)
3. **Tick loop**: Per-tick condition checks, action executions
4. **Result**: Final pass/fail with the triggering condition name and details

### What to Extract

- **Which condition triggered**: The condition `label` in the result tells you exactly what ended the scenario
- **Timing**: Compare elapsed time vs. timeout to understand if the scenario had enough time
- **Condition values**: Many conditions log observed vs. expected values — use these to calibrate parameters

## Common Failure Patterns

### Spawn Failures

**Symptoms**: Error during spawn phase, vehicle not found, collision on spawn

**Causes and fixes**:
- Lanelet ID does not exist on the map → Verify the lanelet ID exists in the map the run uses (`map=`)
- Lanelet has no 3D model in CARLA → Avoid the map's `no_3d_model_lanelet_ids` (`conf/map/<map>.yaml`)
- Spawn position clips into terrain → Adjust `s` value or enable ground projection
- Another entity already occupies the spawn point → Adjust spawn timing or position

### Timeout

**Symptoms**: `TimeoutCondition` triggers as the fail condition

**Causes and fixes**:
- Timeout too short for ego to reach the target → Increase `timeout_seconds`
- Ego is stuck (not moving) → Check initial speed, route validity, traffic light state
- Pass condition threshold too strict → Relax the pass condition parameters

### Wrong Condition Triggers

**Symptoms**: A fail condition fires before the pass condition, or pass fires prematurely

**Causes and fixes**:
- Grace period too short → Increase grace period so the condition doesn't fire during warmup
- Speed threshold too high → Lower minimum speed if ego naturally slows in curves
- Condition ordering issue → Check if conditions interact (e.g., speed check fires before ego accelerates)

### Static Check Failures

**Symptoms**: `scenario-check` (or the check the runner makes before a run, `typecheck=auto`) reports an error at a line

**Causes and fixes**:
- An attribute assigned on `self` that is not declared at class level → declare it (`_config: MyScenarioConfig`)
- A misspelled config key or a value of the wrong type in the YAML → fix the YAML or the dataclass field
- A value that may be `None` used where it may not be → check for `None` first
- Code Codon cannot compile → see `docs/typecheck.md`; do not switch the check off to get past it

### Import / Runtime Errors

**Symptoms**: Python traceback before the scenario starts

**Causes and fixes**:
- Missing import → Check that the class exists in the top-level `autoware_carla_scenario/__init__.py` exports
- Unknown scenario name → The scenario is registered (`register()` of an installed package, or `_register_builtin_scenario` for a built-in) and `scenario.name` matches it; a package must be installed (`uv pip install -e <package>`)
- Wrong constructor arguments → Read the class signature and match parameters
- Config mismatch → Ensure dataclass fields match YAML keys and types

### Coordinate Issues

**Symptoms**: Vehicle spawns in wrong location, NPC not visible, condition never triggers

**Causes and fixes**:
- Using OpenDRIVE coordinates instead of Lanelet2 → Convert to `Lanelet2Pose`
- Wrong `s` value → Check lanelet length and adjust arc-length parameter
- Missing ground projection → Add `GroundProjectionConfig` to prevent terrain clipping

## Debugging Strategy

### Iteration Discipline

1. **One change at a time**: Fix one issue per iteration so you can attribute the result
2. **Read before fixing**: Always read the relevant source file before modifying it
3. **Check the log diff**: Compare output between iterations to verify your fix had the expected effect
4. **Know when to stop**: If the same issue persists after 3 attempts, escalate to the user — the problem may require domain knowledge you don't have

### Parameter Tuning Order

When a scenario fails due to timing/threshold issues, adjust in this order:
1. `timeout_seconds` — Give the scenario enough time first
2. Spawn positions (`lanelet_id`, `s`) — Ensure vehicles are where they should be
3. Speed parameters (`initial_speed_kmh`, min/max thresholds) — Match expected driving behavior
4. Grace periods and delays — Account for warmup and transition time
5. Condition thresholds — Last resort, as these define the test criteria

## Key Source Locations

- The scenario under debug: its scenario package (`src/<package>/`), or a built-in in `<framework>/examples/`
- Hydra configs: the package's `conf/`, and the framework's `<framework>/examples/conf/` (`map/`, `ego/`, `server/`, ...)
- Conditions: `<framework>/conditions/`
- Actions: `<framework>/actions/`
- Coordinate transforms: `<framework>/coordinate/`
