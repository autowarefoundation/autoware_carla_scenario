"""``scenario-agent``: install the scenario skills into a coding agent and start it.

The skills ship with the package (``autoware_carla_scenario/agent/skills/``).
This command installs them for one coding agent with agent-skill-installer
(the ``agent`` extra) into the Git repository holding the working directory
-- this framework's checkout, or a scenario package's repository -- under the
agent's own skills directory (``.claude/skills/``, ``.codex/skills/``), and
then starts that agent in the repository, optionally on a first prompt that
invokes one of the skills::

    scenario-agent claude "stop at a stop sign"
    scenario-agent codex --skill review-scenario lane_change

Where the skills sit inside that repository (this framework's checkout, with
the package installed editable) they are installed as symlinks, so edits to
them are live; otherwise they are copied, and running the command again
updates the copies.

The installer also keeps a discoverability block per skill in the agent's
instruction file (``CLAUDE.md``, ``AGENTS.md``) of the repository.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
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


def find_repository_root(start: Path) -> Path | None:
    """Return the Git (or Sapling) repository root holding ``start``, if any."""
    for directory in (start, *start.parents):
        if (directory / ".git").exists() or (directory / ".sl").exists():
            return directory
    return None


def install_command(
    agents: Sequence[Agent], repository: Path, skills_dir: Path = SKILLS_DIR
) -> list[str]:
    """Return the agent-skill-installer command installing every skill.

    The skills go into ``repository`` for every agent in ``agents`` at once:
    as symlinks when ``skills_dir`` is inside ``repository``, else as copies.
    """
    editable = skills_dir.resolve().is_relative_to(repository.resolve())
    return [
        sys.executable,
        "-m",
        "agent_skill_installer",
        "--no-ui",
        "install",
        "--skill-path",
        str(skills_dir),
        "--all-src-skills",
        "--editable" if editable else "--copy",
        "--agent",
        ",".join(agent.name for agent in agents),
        "--scope",
        "dir",
        "--target-dir",
        str(repository),
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
        prog="scenario-agent",
        description=(
            "Install the scenario-writing skills into a coding agent, in the Git "
            "repository holding the working directory, and start the agent there."
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


def _error(message: str) -> int:
    print(f"scenario-agent: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    # Intermixed, so options may come between the agent and the request
    # (`codex --skill review-scenario lane_change`).
    args = build_parser().parse_intermixed_args(argv)
    agent = AGENTS[args.agent]

    skills = available_skills()
    if args.skill not in skills:
        return _error(f"unknown skill {args.skill!r}; choose from: {', '.join(skills)}")

    repository = find_repository_root(Path.cwd())
    if repository is None:
        return _error(
            "run it inside a Git repository (this framework's checkout or a "
            "scenario package's): the skills are installed into its root"
        )

    # Looked up before installing, so a missing agent fails fast.
    executable = shutil.which(agent.name)
    if executable is None and not args.install_only:
        return _error(
            f"`{agent.name}` is not on PATH; install it first ({agent.install_hint})."
        )

    if importlib.util.find_spec("agent_skill_installer") is None:
        return _error(
            "agent-skill-installer is not installed: install the `agent` extra "
            "(`uv pip install 'autoware-carla-scenario[agent]'`)"
        )

    installed = subprocess.run(install_command([agent], repository), check=False)
    if installed.returncode != 0:
        print("scenario-agent: installing the skills failed.", file=sys.stderr)
        return installed.returncode
    if executable is None or args.install_only:
        return 0

    command = launch_command(agent, first_prompt(agent, args.skill, args.request))
    os.chdir(repository)
    # Hand the terminal over to the agent: its session replaces this process.
    os.execv(executable, command)


if __name__ == "__main__":
    sys.exit(main())
