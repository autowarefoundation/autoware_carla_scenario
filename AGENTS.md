# AGENTS.md

Project guidance for coding agents lives in [CLAUDE.md](CLAUDE.md): read it first.
The skills below come from `skills/`; `scripts/scenario_agent.py` installs them.

<!-- DEBUG-CONCRETE-SCENARIO-DISCOVERABILITY-START -->
## debug-concrete-scenario Discoverability

Execute a CARLA concrete scenario, analyze its results, and iterate until it passes. Use when the user names a scenario/variant to run and debug.

Use `$debug-concrete-scenario` when a prompt explicitly asks for this skill.
<!-- DEBUG-CONCRETE-SCENARIO-DISCOVERABILITY-END -->

<!-- REVIEW-SCENARIO-DISCOVERABILITY-START -->
## review-scenario Discoverability

Review a CARLA scenario's parameter design for orthogonality and sweeper compatibility. Use when the user names an existing scenario (or its YAML config) and asks for its parameters to be reviewed.

Use `$review-scenario` when a prompt explicitly asks for this skill.
<!-- REVIEW-SCENARIO-DISCOVERABILITY-END -->

<!-- SCENARIO-DEBUGGING-DISCOVERABILITY-START -->
## scenario-debugging Discoverability

This skill should be used when the user asks to "debug a scenario", "run a scenario", "fix a scenario", "why does my scenario fail", "scenario timeout", "spawn failure", "analyze scenario output", or discusses troubleshooting CARLA test scenario execution. Provides guidance on log analysis, common failure patterns, and iterative debugging.

Use `$scenario-debugging` when a prompt explicitly asks for this skill.
<!-- SCENARIO-DEBUGGING-DISCOVERABILITY-END -->

<!-- SCENARIO-REVIEW-DISCOVERABILITY-START -->
## scenario-review Discoverability

This skill should be used when reviewing CARLA scenario parameters for orthogonality, analyzing parameter dependencies, deciding which values should be parameters vs. computed, validating the parameter design of a logical scenario, or checking sweep compatibility. Use when "review parameters", "check orthogonality", "parameter design", "review scenario parameters", "parameter independence", or discussing parameter selection for sweeper constraints.

Use `$scenario-review` when a prompt explicitly asks for this skill.
<!-- SCENARIO-REVIEW-DISCOVERABILITY-END -->

<!-- SCENARIO-WRITING-DISCOVERABILITY-START -->
## scenario-writing Discoverability

This skill should be used when the user asks to "write a scenario", "create a CARLA scenario", "add a test scenario", "write a concrete scenario", "create a logical scenario", "add constraints", "run multirun", "spawn NPC", "add condition", "add action", mentions scenario writing for autoware_carla_scenario, or discusses CARLA test scenario creation workflow. Provides comprehensive guidance for the full scenario lifecycle from concrete to logical scenarios.

Use `$scenario-writing` when a prompt explicitly asks for this skill.
<!-- SCENARIO-WRITING-DISCOVERABILITY-END -->

<!-- WRITE-SCENARIO-DISCOVERABILITY-START -->
## write-scenario Discoverability

Interactively create a CARLA test scenario for autoware_carla_scenario step by step, from a natural-language description to a concrete scenario, its review, and a logical scenario with sweep constraints. Use when the user asks to write or create a new scenario end to end.

Use `$write-scenario` when a prompt explicitly asks for this skill.
<!-- WRITE-SCENARIO-DISCOVERABILITY-END -->
