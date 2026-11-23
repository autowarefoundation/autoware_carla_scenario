"""Install the repository's agent skills into a coding agent and start it.

The scenario-writing skills live in ``skills/`` at the repository root, one
directory per skill (``SKILL.md`` and its references). This script installs
them for one coding agent with agent-skill-installer -- as editable symlinks
under the agent's own skills directory (``.claude/skills/``, ``.codex/skills/``)
-- and then starts that agent in the repository, optionally on a first prompt
that invokes one of the skills::

    uv run --group agent scripts/scenario_agent.py claude "stop at a stop sign"
    uv run --group agent scripts/scenario_agent.py codex --skill review-scenario lane_change

The installer also keeps a discoverability block per skill in the agent's
instruction file (``CLAUDE.md``, ``AGENTS.md``). Those blocks are committed, and
the install rewrites them identically, so a run leaves the tracked files as
they were; the installed skills themselves are ignored by git.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "skills"
DEFAULT_SKILL = "write-scenario"


@dataclass(frozen=True)
class Agent:
    """A coding agent the skills can be installed into and started."""

    # Both agent-skill-installer's ``--agent`` value and the command that
    # starts the agent's interactive session.
    name: str
    # How a prompt invokes a skill: ``/name`` in Claude Code, ``$name`` in Codex.
    skill_sigil: str
    # Where to get the agent when its command is missing.
    install_hint: str


AGENTS: dict[str, Agent] = {
    agent.name: agent
    for agent in (
        Agent("claude", "/", "https://code.claude.com/docs/en/setup"),
        Agent("codex", "$", "https://github.com/openai/codex"),
    )
}


def available_skills() -> list[str]:
    """Return the names of the skills in ``SKILLS_DIR``, sorted."""
    return sorted(p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md"))


def install_command(
    agents: Sequence[Agent], skills_dir: Path = SKILLS_DIR
) -> list[str]:
    """Return the agent-skill-installer command installing every skill.

    The skills go into the repository holding ``skills_dir``, for every agent
    in ``agents`` at once.
    """
    return [
        sys.executable,
        "-m",
        "agent_skill_installer",
        "--no-ui",
        "install",
        "--skill-path",
        str(skills_dir),
        "--all-src-skills",
        "--editable",
        "--agent",
        ",".join(agent.name for agent in agents),
        "--scope",
        "dir",
        "--target-dir",
        str(skills_dir.parent),
        "--repo",
    ]


def first_prompt(agent: Agent, skill: str, request: Sequence[str]) -> str | None:
    """Return the prompt that invokes ``skill`` on ``request``, if there is one.

    With no request, the agent starts without a prompt: the skills are
    installed and the user picks one from the session.
    """
    text = " ".join(request).strip()
    if not text:
        return None
    return f"{agent.skill_sigil}{skill} {text}"


def launch_command(agent: Agent, prompt: str | None) -> list[str]:
    """Return the command that starts ``agent``'s interactive session."""
    return [agent.name] if prompt is None else [agent.name, prompt]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install the scenario-writing skills into a coding agent and start it."
        ),
    )
    parser.add_argument("agent", choices=sorted(AGENTS), help="The agent to start.")
    parser.add_argument(
        "request",
        nargs="*",
        help=(
            "What to ask the skill, e.g. a scenario description for "
            "write-scenario. Without it the agent starts with no prompt."
        ),
    )
    parser.add_argument(
        "--skill",
        default=DEFAULT_SKILL,
        help=f"The skill the request invokes (default: {DEFAULT_SKILL}).",
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Install the skills and exit without starting the agent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Intermixed, so options may come between the agent and the request
    # (`codex --skill review-scenario lane_change`).
    args = build_parser().parse_intermixed_args(argv)
    agent = AGENTS[args.agent]

    skills = available_skills()
    if args.skill not in skills:
        print(
            f"scenario_agent: unknown skill {args.skill!r}; "
            f"choose from: {', '.join(skills)}",
            file=sys.stderr,
        )
        return 2

    # Looked up before installing, so a missing agent fails fast.
    executable = shutil.which(agent.name)
    if executable is None and not args.install_only:
        print(
            f"scenario_agent: `{agent.name}` is not on PATH; "
            f"install it first ({agent.install_hint}).",
            file=sys.stderr,
        )
        return 2

    installed = subprocess.run(install_command([agent]), check=False)
    if installed.returncode != 0:
        print("scenario_agent: installing the skills failed.", file=sys.stderr)
        return installed.returncode
    if executable is None or args.install_only:
        return 0

    command = launch_command(agent, first_prompt(agent, args.skill, args.request))
    os.chdir(REPO_ROOT)
    # Hand the terminal over to the agent: its session replaces this process.
    os.execv(executable, command)


if __name__ == "__main__":
    sys.exit(main())
