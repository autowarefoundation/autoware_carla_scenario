"""The agent skills in ``skills/`` and ``scripts/scenario_agent.py``.

The skills are installed into Claude Code and Codex by agent-skill-installer,
which also writes a discoverability block per skill into ``CLAUDE.md`` and
``AGENTS.md``. Those blocks are committed, so they must be exactly what an
install writes: otherwise every run of the script would leave a diff.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILLS_DIR = _REPO_ROOT / "skills"
_SCRIPT = _REPO_ROOT / "scripts" / "scenario_agent.py"

# Names only Claude Code understands; a skill shared with Codex must not need
# them (a Claude Code command's argument placeholder and tool permissions, and
# its question tool).
_CLAUDE_ONLY = ("$ARGUMENTS", "allowed-tools", "argument-hint", "AskUserQuestion")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scenario_agent", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the module of the classes it decorates.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scenario_agent = _load_script()


def _skill_dirs() -> list[Path]:
    return sorted(p.parent for p in _SKILLS_DIR.glob("*/SKILL.md"))


def _frontmatter(skill_md: Path) -> dict[str, object]:
    text = skill_md.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{skill_md} has no YAML frontmatter"
    header, _, _ = text[len("---\n") :].partition("\n---\n")
    data = yaml.safe_load(header)
    assert isinstance(data, dict), f"{skill_md}'s frontmatter is not a mapping"
    return data


def test_the_workflow_skills_are_there() -> None:
    names = {p.name for p in _skill_dirs()}
    assert {
        "write-scenario",
        "review-scenario",
        "debug-concrete-scenario",
        "scenario-writing",
        "scenario-review",
        "scenario-debugging",
    } <= names


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda p: p.name)
def test_skill_frontmatter_names_its_directory(skill_dir: Path) -> None:
    data = _frontmatter(skill_dir / "SKILL.md")
    assert data.get("name") == skill_dir.name
    description = data.get("description")
    assert isinstance(description, str) and description.strip()


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda p: p.name)
def test_skill_is_agent_neutral(skill_dir: Path) -> None:
    for path in sorted(skill_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        found = [name for name in _CLAUDE_ONLY if name in text]
        assert not found, f"{path.relative_to(_REPO_ROOT)} uses {found}"


def test_committed_discoverability_blocks_match_an_install(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    shutil.copytree(_SKILLS_DIR, repo / "skills")
    for name in ("CLAUDE.md", "AGENTS.md"):
        shutil.copy2(_REPO_ROOT / name, repo / name)
    # The installer's `--repo` asserts a Git repository root.
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    for agent in scenario_agent.AGENTS.values():
        command = scenario_agent.install_command(agent, repo)
        subprocess.run(command, check=True, capture_output=True)

    for name in ("CLAUDE.md", "AGENTS.md"):
        assert (repo / name).read_text(encoding="utf-8") == (
            _REPO_ROOT / name
        ).read_text(encoding="utf-8"), (
            f"{name}'s discoverability blocks are stale: re-run "
            "`uv run --group agent scripts/scenario_agent.py <agent> "
            "--install-only` for claude and codex and commit the result"
        )
    assert (repo / ".claude" / "skills" / "write-scenario" / "SKILL.md").is_file()
    assert (repo / ".codex" / "skills" / "write-scenario" / "SKILL.md").is_file()


def test_available_skills_lists_the_skill_directories() -> None:
    assert scenario_agent.available_skills() == [p.name for p in _skill_dirs()]


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        ("claude", "/write-scenario stop at a stop sign"),
        ("codex", "$write-scenario stop at a stop sign"),
    ],
)
def test_first_prompt_invokes_the_skill(agent: str, expected: str) -> None:
    prompt = scenario_agent.first_prompt(
        scenario_agent.AGENTS[agent], "write-scenario", ["stop at", "a stop sign"]
    )
    assert prompt == expected
    assert scenario_agent.launch_command(scenario_agent.AGENTS[agent], prompt) == [
        agent,
        expected,
    ]


def test_no_request_starts_the_agent_without_a_prompt() -> None:
    agent = scenario_agent.AGENTS["claude"]
    assert scenario_agent.first_prompt(agent, "write-scenario", []) is None
    assert scenario_agent.launch_command(agent, None) == ["claude"]


def test_install_command_installs_every_skill_into_the_repository() -> None:
    command = scenario_agent.install_command(
        scenario_agent.AGENTS["codex"], Path("/repo")
    )
    assert command[:3] == [sys.executable, "-m", "agent_skill_installer"]
    assert "--all-src-skills" in command
    assert command[command.index("--skill-path") + 1] == str(Path("/repo/skills"))
    assert command[command.index("--agent") + 1] == "codex"
    assert command[command.index("--scope") + 1] == "dir"


def test_unknown_skill_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert scenario_agent.main(["claude", "--skill", "no-such-skill", "x"]) == 2
    assert "unknown skill" in capsys.readouterr().err


def test_missing_agent_command_is_reported_before_installing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(scenario_agent.shutil, "which", lambda _: None)

    def no_install(*args: object, **kwargs: object) -> None:
        raise AssertionError("installed although the agent is missing")

    monkeypatch.setattr(scenario_agent.subprocess, "run", no_install)
    assert scenario_agent.main(["codex", "something"]) == 2
    assert "`codex` is not on PATH" in capsys.readouterr().err


def test_install_only_does_not_start_the_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    def no_exec(*args: object) -> None:
        raise AssertionError("started the agent with --install-only")

    monkeypatch.setattr(scenario_agent.subprocess, "run", fake_run)
    monkeypatch.setattr(scenario_agent.os, "execv", no_exec)
    assert scenario_agent.main(["claude", "--install-only"]) == 0
    assert commands == [scenario_agent.install_command(scenario_agent.AGENTS["claude"])]
