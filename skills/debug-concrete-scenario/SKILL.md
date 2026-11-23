---
name: debug-concrete-scenario
description: Execute a CARLA concrete scenario, analyze its results, and iterate until it passes. Use when the user names a scenario/variant to run and debug.
---

Debug and iterate on a CARLA concrete scenario until it passes. Use the `scenario-debugging` skill for domain knowledge on log analysis and common failure patterns.

## Input

The scenario to run (`<scenario>/<variant>`, optionally followed by extra `uv run scenario` options) comes from the request that invoked this skill.

If no scenario is given, ask the user which scenario to run.

## Workflow

### Step 1: Static Check, then Execute

First compile the scenario with Codon, which needs no CARLA and reports type errors, misspelled config keys and bad values at their line:
```bash
uv run scenario-check scenario=<name>/<variant>
```

Fix what it reports, then run the concrete scenario and capture full output:
```bash
uv run scenario scenario=<name>/<variant>
```

If the user provided additional options (e.g., `map=town10hd_opt`, `scenario.timeout_seconds=20`), append them to the command. A run needs a CARLA server (`server.host` / `server.port`); if none is reachable, say so and ask the user rather than debugging the scenario.

The run writes `outputs/YYYY-MM-DD/HH-MM-SS/`: `{ScenarioName}_result.json` (pass/fail, message, condition statuses) and `{ScenarioName}.trajectory.jsonl` (the run tick by tick, with the conditions and the verdict) are the files to read.

### Step 2: Analyze Results

Examine the output for:
- **Pass/fail status**: Which condition triggered the result?
- **Error messages**: Spawn failures, import errors, runtime exceptions, missing modules
- **Timing**: Did it timeout? Did conditions trigger too early or too late?
- **Warnings**: Messages that indicate potential issues (e.g., ground projection failures, coordinate conversion warnings)
- **Condition logs**: Which conditions were checked, in what order, and what values they observed (the condition tree in `_result.json`)

### Step 3: Diagnose and Fix

If the scenario fails or behaves unexpectedly:
1. Identify the root cause from the log output
2. Read the relevant source files (scenario class, config, YAML) to understand the issue
3. Apply the appropriate fix:
   - **Parameter issues**: Adjust timeouts, speeds, spawn positions in YAML or config
   - **Code errors**: Fix imports, logic errors, incorrect condition/action usage in the scenario class; re-run `scenario-check` after each code change
   - **Spawn failures**: Adjust Lanelet2 coordinates or ground projection settings
   - **Condition timing**: Adjust grace periods, thresholds, or condition ordering

### Step 4: Iterate

Repeat Step 1 → Step 2 → Step 3 until the scenario passes as intended.

If you are stuck after 3 iterations on the same issue, stop and ask the user for guidance rather than continuing to loop.

### Step 5: Report

Once the scenario passes (or if you hit an unresolvable issue), present to the user:
1. **Result**: Pass/fail and which condition triggered
2. **Adjustments made**: What was changed and why (list each change)
3. **Concerns**: Any remaining trade-offs or fragile parameters
4. Where to look: the run's output directory, and `uv run viewer` (http://localhost:9000) for the condition tree; the trajectory log can be turned into a Rerun recording to watch the run (`autoware_carla_scenario/docs/usage.md`, "Seeing a run")
5. Ask the user to confirm the behavior matches their intent
