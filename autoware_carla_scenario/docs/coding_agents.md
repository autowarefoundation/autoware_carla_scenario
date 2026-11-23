# Writing Scenarios with a Coding Agent

The framework ships agent skills for writing, reviewing and debugging
scenarios. A **coding agent** — [Claude Code](https://code.claude.com/docs/en/setup)
or [Codex](https://github.com/openai/codex) — can load them and walk you
through a scenario, from a plain-text description to a logical scenario with
its sweep. `scenario-agent` installs the skills and starts the agent. It works
in this framework's checkout and in your own scenario package's repository
alike. This page is the procedure, from setup to editing the skills.

## What Is Shipped

| What | Where |
|---|---|
| The skills: one directory per skill (`SKILL.md`, plus `references/` for `scenario-writing`), written for any agent | In the package: `autoware_carla_scenario/agent/skills/` (in this checkout, `autoware_carla_scenario/src/autoware_carla_scenario/agent/skills/`) |
| `scenario-agent`: installs every skill into one agent, then starts that agent | Console script of the package |
| [agent-skill-installer](https://github.com/omry/agent-skill-installer), which `scenario-agent` installs with | The `agent` extra |

| Skill | Use it to |
|---|---|
| `write-scenario` | Create a new scenario end to end (the default `scenario-agent` invokes) |
| `debug-concrete-scenario` | Run a concrete scenario, read the result, fix it until it passes |
| `review-scenario` | Review a scenario's parameter design before adding a sweep |
| `scenario-writing`, `scenario-debugging`, `scenario-review` | The domain knowledge the three above build on; the agent loads them when a task needs them |

## 1. Prerequisites

1. **The framework with the `agent` extra**:
    - in this framework's checkout, `uv sync --dev` at the repository root (the
      dev group includes it);
    - in a scenario package's project, add the extra to its dependency on the
      framework: `uv add 'autoware-carla-scenario[agent]'` (or
      `uv pip install 'autoware-carla-scenario[agent]'` in its environment).
2. **A coding agent on `PATH`, signed in**: `claude` (Claude Code) or `codex`
   (Codex). Start it once by hand to log in; `scenario-agent` does not sign you
   in.
3. **A Git repository** to work in: the skills are installed into the root of
   the repository holding the working directory, found through `.git`. A
   scenario package from [`scenario-new`](tutorial.md#packaging-scenarios-in-a-separate-package)
   is not one until you `git init` it.
4. **A CARLA server**, for the steps that run a scenario (debugging and the
   multirun). `uv run scenario-setup` downloads one; see
   [Installation](installation.md). Writing, the static check and the sweep
   preview need no CARLA.

## 2. Start the Agent

From anywhere in the repository:

```bash
# Claude Code
uv run scenario-agent claude "I want to test that the ego stops at a stop sign, waits 3 seconds, then goes"

# Codex
uv run scenario-agent codex "I want to test that the ego stops at a stop sign, waits 3 seconds, then goes"
```

`scenario-agent` then:

1. checks that the skill to invoke exists, that the working directory is in a
   Git repository, that the agent's command is on `PATH`, and that none of the
   skill directories it would write already holds a skill the installer did
   not create (otherwise it stops with exit code 2, before installing
   anything);
2. installs every skill for that agent into the repository's
   `.claude/skills/` (Claude Code) or `.codex/skills/` (Codex), and writes each
   skill's discoverability block at the end of the repository's `CLAUDE.md` /
   `AGENTS.md` (creating the file if there is none). Each skill is a bundled
   skill of agent-skill-installer, read from the installed package and copied;
   the install records it as owned by `autoware-carla-scenario` and its
   version, so running `scenario-agent` again replaces it;
3. starts the agent in the repository root, with a first prompt that invokes
   the skill on your request: `/write-scenario <request>` in Claude Code,
   `$write-scenario <request>` in Codex.

The agent's session replaces `scenario-agent`: from there on you are talking to
the agent.

### Options

| Command | Effect |
|---|---|
| `scenario-agent claude "<request>"` | Install, then start Claude Code on `/write-scenario <request>` |
| `scenario-agent codex --skill review-scenario lane_change` | Install, then start Codex on `$review-scenario lane_change` |
| `scenario-agent claude` | Install, then start Claude Code with no prompt; pick a skill in the session |
| `scenario-agent codex --install-only` | Install only; start the agent yourself later |

The options may come before or after the request. `--help` lists them.

Once installed, the skills stay: starting `claude` or `codex` in the repository
root by hand finds them too.

### In a scenario package's repository

The installs and the discoverability blocks land in your repository. Keep the
installs out of version control, as this repository does:

```gitignore
.claude/skills/
.codex/
```

The blocks in `CLAUDE.md` / `AGENTS.md` are yours to commit or not; each run
writes them the same way for the same framework version. The copied skills
are those of the installed framework: after upgrading it, run
`scenario-agent <agent> --install-only` again to update them.

## 3. Work Through the Scenario

In the session, a skill is invoked by name followed by what you want —
`/write-scenario ...` in Claude Code, `$write-scenario ...` in Codex.
`scenario-agent`'s first prompt does that for you. `write-scenario` then goes
through these stages, asking you to confirm at each one:

| Stage | What the agent does | What you do |
|---|---|---|
| 1. Understand | Restates the behaviour under test, the pass and fail conditions, the NPCs, and asks where the scenario lives: a scenario package of its own (the default, made with `scenario-new`) or a built-in scenario of the framework (in its checkout) | Answer its questions; name the package |
| 2. Discover | Searches the framework's source for the Conditions, Actions and the built-in scenario closest to yours | Approve any new Condition/Action it proposes |
| 3. Write | Writes the scenario class, its config dataclass and the concrete YAML (and the registration, for a built-in), then runs the static check (`scenario-check`) | Review the files it shows before it writes them |
| 4. Debug | Runs the concrete scenario in CARLA (the `debug-concrete-scenario` skill), reads the result and fixes it, up to 3 tries on one issue before asking you | Keep a CARLA server reachable; confirm the behaviour is what you meant |
| 5. Review | Reviews the parameters (the `scenario-review` skill): the one swept lanelet, what is derived from it with bindings, what stays constant | Confirm the parameter design |
| 6. Sweep | Adds `sweep.constraints` / `bindings` to the YAML and previews the cases with `scenario-expand` | Commit, as it suggests |
| 7. Multirun | Gives the multirun command (`--multirun` with `hydra/sweeper=lanelet_constraint`) | Run it; browse the results with `uv run viewer` |

You can also call the other skills directly at any time, e.g.
`/debug-concrete-scenario <name>/<variant>` for a scenario that fails, or
`/review-scenario <name>` before adding a sweep.

The skills discover the framework's API from its source each time — the
installed package, wherever it is — so they do not need updating for every new
Condition or constraint. What they describe follows the [Tutorial](tutorial.md)
and [Usage](usage.md); where a skill and the code disagree, the code wins.

## 4. Edit or Add a Skill

The skills are plain Markdown under the package's `agent/skills/<name>/`;
change them in this framework's checkout
(`autoware_carla_scenario/src/autoware_carla_scenario/agent/skills/`). The
installs are copies: after an edit, run `scenario-agent <agent> --install-only`
(for each agent you use) and start a new session to see it.

- **Changing a skill's text**: edit its `SKILL.md` (or its `references/`). Keep
  it agent-neutral: no names only one agent understands (`$ARGUMENTS`,
  `allowed-tools`, `AskUserQuestion`); `test_agent_skills.py` refuses them.
  Refer to the framework's source as `<framework>/` and to its documentation as
  `docs/<page>.md` (`scenario-writing`'s "Where Things Are"), so the skill
  works outside this checkout too.
- **Changing a skill's `description`** (its frontmatter), **adding or renaming
  a skill**: the discoverability blocks in this repository's `CLAUDE.md` and
  `AGENTS.md` come from the descriptions, so regenerate them and commit the
  result:

  ```bash
  uv run scenario-agent claude --install-only
  uv run scenario-agent codex --install-only
  ```

  `test_agent_skills.py` installs into a temporary repository and fails while
  the committed blocks differ from what an install writes.
- A new skill is a directory `agent/skills/<name>/` with a `SKILL.md` whose
  frontmatter has `name: <name>` (the directory's name) and a `description`
  saying when to use it. Everything under the package directory goes into the
  wheel, so it ships with the next release.

Run the skill tests with:

```bash
uv run pytest -o addopts= autoware_carla_scenario/test/carla_scenario/test_agent_skills.py
```

## 5. Remove the Installed Skills

The installs are only the directories `.claude/skills/` and `.codex/skills/`
(with the installer's manifests beside each skill); delete them to remove the
skills from a repository. In this repository, do not run the installer's own
`uninstall`: it also removes the discoverability blocks from the committed
`CLAUDE.md` and `AGENTS.md`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| ``scenario-agent: `claude` is not on PATH`` (exit 2) | Install the agent (the message links to it) or fix `PATH`; or use `--install-only` and start the agent another way |
| `scenario-agent: run it inside a Git repository ...` (exit 2) | Run it from inside a Git checkout; `git init` a new scenario package first |
| `scenario-agent: agent-skill-installer is not installed ...` (exit 2) | Install the `agent` extra (section 1) |
| `scenario-agent: unknown skill ...` (exit 2) | The `--skill` name is not one of the shipped skills; the message lists them |
| `scenario-agent: installing the skills failed: refusing to replace unowned skill directories ...` (exit 2) | The listed directories in `.claude/skills/` or `.codex/skills/` hold skills the installer did not create (no install manifest in them), e.g. written by hand. Nothing was installed and `CLAUDE.md` / `AGENTS.md` are untouched: move or remove every listed directory and run `scenario-agent` again |
| `scenario-agent: installing the skills failed: ...` (exit 2), another message | The installer's error follows; fix what it names and run `scenario-agent` again |
| The agent does not list the skills | Start it from the repository root (`scenario-agent` does), and start a new session after installing; check that `.claude/skills/<name>` or `.codex/skills/<name>` exists |
| `git status` shows this repository's `CLAUDE.md` / `AGENTS.md` changed after a run | A skill's `description` changed without regenerating the blocks: commit the change (section 4) |
| A debugging step cannot run the scenario | No CARLA server is reachable: start one (`scenario-setup`, `server.host` / `server.port`), then ask the agent to continue |
