# Writing Scenarios with a Coding Agent

The repository ships agent skills for writing, reviewing and debugging
scenarios. A **coding agent** — [Claude Code](https://code.claude.com/docs/en/setup)
or [Codex](https://github.com/openai/codex) — can load them and walk you
through a scenario, from a plain-text description to a logical scenario with
its sweep. This page is the procedure, from setup to editing the skills
themselves.

## What Is Shipped

| Path | What it is |
|---|---|
| `skills/` | One directory per skill (`SKILL.md`, plus `references/` for `scenario-writing`), written for any agent |
| `scripts/scenario_agent.py` | Installs every skill into one agent, then starts that agent in the repository |
| `agent` dependency group (root `pyproject.toml`) | [agent-skill-installer](https://github.com/omry/agent-skill-installer), which the script uses for the install |

| Skill | Use it to |
|---|---|
| `write-scenario` | Create a new scenario end to end (the default the script invokes) |
| `debug-concrete-scenario` | Run a concrete scenario, read the result, fix it until it passes |
| `review-scenario` | Review a scenario's parameter design before adding a sweep |
| `scenario-writing`, `scenario-debugging`, `scenario-review` | The domain knowledge the three above build on; the agent loads them when a task needs them |

## 1. Prerequisites

1. **The repository's environment**, as in [Installation](installation.md):
   `uv sync --dev` at the repository root.
2. **A coding agent on `PATH`, signed in**: `claude` (Claude Code) or `codex`
   (Codex). Start it once by hand to log in; the script does not sign you in.
3. **A CARLA server**, for the steps that run a scenario (debugging and the
   multirun). `uv run scenario-setup` downloads one; see
   [Installation](installation.md). Writing, the static check and the sweep
   preview need no CARLA.
4. **A Git checkout**: the skills are installed into the repository root, which
   the installer finds through `.git`.

## 2. Start the Agent

From anywhere in the checkout:

```bash
# Claude Code
uv run --group agent scripts/scenario_agent.py claude "I want to test that the ego stops at a stop sign, waits 3 seconds, then goes"

# Codex
uv run --group agent scripts/scenario_agent.py codex "I want to test that the ego stops at a stop sign, waits 3 seconds, then goes"
```

`--group agent` adds agent-skill-installer to the environment for the run.

The script then:

1. checks that the skill to invoke exists in `skills/`, and that the agent's
   command is on `PATH` (otherwise it stops with exit code 2, before
   installing anything);
2. installs every skill for that agent, as symlinks into the repository's
   `.claude/skills/` (Claude Code) or `.codex/skills/` (Codex) — both ignored by
   git — and rewrites each skill's discoverability block at the end of
   `CLAUDE.md` / `AGENTS.md`, identically to what is committed;
3. starts the agent in the repository root, with a first prompt that invokes
   the skill on your request: `/write-scenario <request>` in Claude Code,
   `$write-scenario <request>` in Codex.

The agent's session replaces the script: from there on you are talking to the
agent.

### Options

| Command | Effect |
|---|---|
| `scenario_agent.py claude "<request>"` | Install, then start Claude Code on `/write-scenario <request>` |
| `scenario_agent.py codex --skill review-scenario lane_change` | Install, then start Codex on `$review-scenario lane_change` |
| `scenario_agent.py claude` | Install, then start Claude Code with no prompt; pick a skill in the session |
| `scenario_agent.py codex --install-only` | Install only; start the agent yourself later |

The options may come before or after the request. `--help` lists them.

Once installed, the skills stay: starting `claude` or `codex` in the repository
root by hand finds them too.

## 3. Work Through the Scenario

In the session, a skill is invoked by name followed by what you want —
`/write-scenario ...` in Claude Code, `$write-scenario ...` in Codex. The
script's first prompt does that for you. `write-scenario` then goes through
these stages, asking you to confirm at each one:

| Stage | What the agent does | What you do |
|---|---|---|
| 1. Understand | Restates the behaviour under test, the pass and fail conditions, the NPCs, and asks where the scenario lives: a scenario package of its own (the default, made with `scenario-new`) or a built-in scenario of the framework | Answer its questions; name the package |
| 2. Discover | Searches the source for the Conditions, Actions and the built-in scenario closest to yours | Approve any new Condition/Action it proposes |
| 3. Write | Writes the scenario class, its config dataclass and the concrete YAML (and the registration, for a built-in), then runs the static check (`scenario-check`) | Review the files it shows before it writes them |
| 4. Debug | Runs the concrete scenario in CARLA (the `debug-concrete-scenario` skill), reads the result and fixes it, up to 3 tries on one issue before asking you | Keep a CARLA server reachable; confirm the behaviour is what you meant |
| 5. Review | Reviews the parameters (the `scenario-review` skill): the one swept lanelet, what is derived from it with bindings, what stays constant | Confirm the parameter design |
| 6. Sweep | Adds `sweep.constraints` / `bindings` to the YAML and previews the cases with `scenario-expand` | Commit, as it suggests |
| 7. Multirun | Gives the multirun command (`--multirun` with `hydra/sweeper=lanelet_constraint`) | Run it; browse the results with `uv run viewer` |

You can also call the other skills directly at any time, e.g.
`/debug-concrete-scenario <name>/<variant>` for a scenario that fails, or
`/review-scenario <name>` before adding a sweep.

The skills discover the framework's API from the source each time, so they do
not need updating for every new Condition or constraint. What they describe
follows the [Tutorial](tutorial.md) and [Usage](usage.md); where a skill and the
code disagree, the code wins.

## 4. Edit or Add a Skill

The skills are plain Markdown under `skills/<name>/`. The installs are symlinks,
so an edit is live in the next agent session with no reinstall.

- **Changing a skill's text**: edit `skills/<name>/SKILL.md` (or its
  `references/`). Keep it agent-neutral: no names only one agent understands
  (`$ARGUMENTS`, `allowed-tools`, `AskUserQuestion`); `test_agent_skills.py`
  refuses them.
- **Changing a skill's `description`** (its frontmatter), **adding or renaming
  a skill**: the discoverability blocks in `CLAUDE.md` and `AGENTS.md` come
  from the descriptions, so regenerate them and commit the result:

  ```bash
  uv run --group agent scripts/scenario_agent.py claude --install-only
  uv run --group agent scripts/scenario_agent.py codex --install-only
  ```

  `test_agent_skills.py` installs into a temporary repository and fails while
  the committed blocks differ from what an install writes.
- A new skill is a directory `skills/<name>/` with a `SKILL.md` whose
  frontmatter has `name: <name>` (the directory's name) and a `description`
  saying when to use it.

Run the skill tests with:

```bash
uv run pytest -o addopts= autoware_carla_scenario/test/carla_scenario/test_agent_skills.py
```

## 5. Remove the Installed Skills

The installs are only the ignored directories `.claude/skills/` and
`.codex/skills/` (with the installer's manifests beside each skill); delete
them to remove the skills from this checkout. Do not run the installer's own
`uninstall`: it also removes the discoverability blocks from the committed
`CLAUDE.md` and `AGENTS.md`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| ``scenario_agent: `claude` is not on PATH`` (exit 2) | Install the agent (the message links to it) or fix `PATH`; or use `--install-only` and start the agent another way |
| `scenario_agent: unknown skill ...` (exit 2) | The `--skill` name is not a directory of `skills/`; the message lists the names |
| `installing the skills failed` | The installer's own error is printed above it. The usual one is a skill directory in `.claude/skills/` or `.codex/skills/` that the installer did not create (no manifest beside it): remove that directory and run the script again |
| The agent does not list the skills | Start it from the repository root (the script does), and start a new session after installing; check that `.claude/skills/<name>` or `.codex/skills/<name>` exists |
| `git status` shows `CLAUDE.md` / `AGENTS.md` changed after a run | A skill's `description` changed without regenerating the blocks: commit the change (section 4) |
| A debugging step cannot run the scenario | No CARLA server is reachable: start one (`scenario-setup`, `server.host` / `server.port`), then ask the agent to continue |
