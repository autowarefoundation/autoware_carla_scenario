"""``scenario-agent``: install the scenario skills into a coding agent and start it.

The skills ship inside this package (``autoware_carla_scenario/agent/skills/``),
each a bundled skill of agent-skill-installer (the ``agent`` extra): a
:class:`agent_skill_installer.SkillProject` per skill, read from the installed
package with :mod:`importlib.resources`. This command installs them for one
coding agent into the Git repository holding the working directory -- this
framework's checkout, or a scenario package's repository -- under the agent's
own skills directory (``.claude/skills/``, ``.codex/skills/``), and then starts
that agent in the repository, optionally on a first prompt that invokes one of
the skills::

    scenario-agent claude "stop at a stop sign"
    scenario-agent codex --skill review-scenario lane_change

The installs are copies, recorded as owned by this package and its version;
running the command again replaces them with the installed package's skills.
The installer also keeps a discoverability block per skill in the agent's
instruction file (``CLAUDE.md``, ``AGENTS.md``) of the repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from agent_skill_installer import SkillProject

#: The distribution the skills ship in, and the package they are read from.
DISTRIBUTION = "autoware-carla-scenario"
IMPORT_NAME = "autoware_carla_scenario"
#: Where the skills sit inside the package (``bundled_skill_path``'s parent).
SKILLS_PATH = "agent/skills"
SKILLS_DIR = Path(__file__).resolve().parent / "skills"
DEFAULT_SKILL = "write-scenario"


@dataclass(frozen=True)
class Agent:
    """A coding agent the skills can be installed into and started."""

    # Both agent-skill-installer's agent name and the command that starts the
    # agent's interactive session.
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
    """Return the names of the bundled skills, sorted."""
    return sorted(p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md"))


def skill_description(name: str) -> str:
    """Return the ``description`` of a bundled skill's ``SKILL.md`` frontmatter."""
    text = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    header = text.removeprefix("---\n").partition("\n---\n")[0]
    return str(yaml.safe_load(header)["description"])


def skill_projects() -> list[SkillProject]:
    """Return one agent-skill-installer project per bundled skill."""
    from agent_skill_installer import SkillProject

    version = metadata.version(DISTRIBUTION)
    return [
        SkillProject(
            package_name=DISTRIBUTION,
            import_name=IMPORT_NAME,
            version=version,
            skill_name=name,
            description=skill_description(name),
            bundled_skill_path=f"{SKILLS_PATH}/{name}",
            # A skill installed from its directory before it was bundled
            # (`--skill-path`) is owned under its own name: take it over.
            manifest_package_aliases=frozenset({name}),
        )
        for name in available_skills()
    ]


def unowned_skill_directories(
    agents: Sequence[Agent],
    repository: Path,
    projects: Sequence[SkillProject] | None = None,
) -> list[tuple[Path, str | None]]:
    """Return the skill directories the installer would refuse to replace.

    One per skill and agent whose directory in ``repository`` exists without
    this package's install manifest, as the installer itself reports it
    (``Installer.inspect_installations``: status ``unowned``), with the
    installer's reason when the directory holds a manifest it cannot accept
    (another package's, or an invalid one); ``None`` when it holds none.
    """
    from agent_skill_installer import Installer

    names = {agent.name for agent in agents}
    unowned = []
    for project in skill_projects() if projects is None else projects:
        for status in Installer(project).inspect_installations(repo=repository):
            if (
                status.agent in names
                and status.scope == "dir"
                and status.repo_target
                and status.status == "unowned"
                and status.skill_dir is not None
            ):
                unowned.append((status.skill_dir, status.error))
    return unowned


def install(agents: Sequence[Agent], repository: Path) -> list[str]:
    """Install every bundled skill for ``agents`` into ``repository``.

    Returns one line per installed skill and agent. Raises
    ``agent_skill_installer.InstallerError`` when the installer refuses -- and,
    before installing anything, when any target skill directory is one the
    installer did not create, naming all of them, so a refusal never leaves
    some skills installed and others not.
    """
    from agent_skill_installer import Installer, InstallerError

    names = [agent.name for agent in agents]
    projects = skill_projects()
    unowned = unowned_skill_directories(agents, repository, projects)
    if unowned:
        raise InstallerError(
            "refusing to replace unowned skill directories (without an install "
            f"manifest of {DISTRIBUTION}), nothing was installed: "
            + ", ".join(
                str(path) if reason is None else f"{path} ({reason})"
                for path, reason in unowned
            )
        )
    lines = []
    for project in projects:
        for result in Installer(project).install(
            names, "dir", repo_target=True, repo=repository
        ):
            lines.append(f"{project.skill_name} -> {result.agent}: {result.skill_dir}")
    return lines


def find_repository_root(start: Path) -> Path | None:
    """Return the Git (or Sapling) repository root holding ``start``, if any."""
    for directory in (start, *start.parents):
        if (directory / ".git").exists() or (directory / ".sl").exists():
            return directory
    return None


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

    try:
        from agent_skill_installer import InstallerError
    except ImportError:
        return _error(
            "agent-skill-installer is not installed: install the `agent` extra "
            "(`uv pip install 'autoware-carla-scenario[agent]'`)"
        )
    try:
        for line in install([agent], repository):
            print(f"scenario-agent: installed {line}")
    except InstallerError as error:
        return _error(f"installing the skills failed: {error}")
    if executable is None or args.install_only:
        return 0

    command = launch_command(agent, first_prompt(agent, args.skill, args.request))
    os.chdir(repository)
    # Hand the terminal over to the agent: its session replaces this process.
    os.execv(executable, command)


if __name__ == "__main__":
    sys.exit(main())
