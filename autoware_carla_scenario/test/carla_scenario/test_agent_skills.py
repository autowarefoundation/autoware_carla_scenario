"""The agent skills shipped in ``autoware_carla_scenario/agent/skills/`` and
``scenario-agent`` (:mod:`autoware_carla_scenario.agent.cli`).

The skills are installed into Claude Code and Codex by agent-skill-installer,
which also writes a discoverability block per skill into the repository's
``CLAUDE.md`` and ``AGENTS.md``. This repository commits those blocks, so they
must be exactly what an install writes: otherwise every run would leave a diff.
"""

from __future__ import annotations

import shutil
import subprocess
from importlib import resources
from pathlib import Path

import pytest
import yaml

from autoware_carla_scenario.agent import cli

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILLS_DIR = cli.SKILLS_DIR

# Names only Claude Code understands; a skill shared with Codex must not need
# them (a Claude Code command's argument placeholder and tool permissions, and
# its question tool).
_CLAUDE_ONLY = ("$ARGUMENTS", "allowed-tools", "argument-hint", "AskUserQuestion")


def _skill_dirs() -> list[Path]:
    return [_SKILLS_DIR / name for name in cli.available_skills()]


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
        assert not found, f"{path.relative_to(_SKILLS_DIR)} uses {found}"


def test_committed_discoverability_blocks_match_an_install(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("CLAUDE.md", "AGENTS.md"):
        shutil.copy2(_REPO_ROOT / name, repo / name)
    # The installer's `--repo` asserts a Git repository root.
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    cli.install(list(cli.AGENTS.values()), repo)

    for name in ("CLAUDE.md", "AGENTS.md"):
        assert (repo / name).read_text(encoding="utf-8") == (
            _REPO_ROOT / name
        ).read_text(encoding="utf-8"), (
            f"{name}'s discoverability blocks are stale: run "
            "`uv run scenario-agent <agent> --install-only` for claude and codex "
            "and commit the result"
        )
    assert (repo / ".claude" / "skills" / "write-scenario" / "SKILL.md").is_file()
    assert (repo / ".codex" / "skills" / "write-scenario" / "SKILL.md").is_file()


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        ("claude", "/write-scenario stop at a stop sign"),
        ("codex", "$write-scenario stop at a stop sign"),
    ],
)
def test_first_prompt_invokes_the_skill(agent: str, expected: str) -> None:
    prompt = cli.first_prompt(
        cli.AGENTS[agent], "write-scenario", ["stop at", "a stop sign"]
    )
    assert prompt == expected
    assert cli.launch_command(cli.AGENTS[agent], prompt) == [agent, expected]


def test_no_request_starts_the_agent_without_a_prompt() -> None:
    agent = cli.AGENTS["claude"]
    assert cli.first_prompt(agent, "write-scenario", []) is None
    assert cli.launch_command(agent, None) == ["claude"]


def test_each_skill_is_a_bundled_skill_project_of_the_package() -> None:
    projects = cli.skill_projects()
    assert [project.skill_name for project in projects] == cli.available_skills()
    for project in projects:
        assert project.package_name == "autoware-carla-scenario"
        assert (
            project.description
            == _frontmatter(_SKILLS_DIR / project.skill_name / "SKILL.md")[
                "description"
            ]
        )
        # The installer reads the skill from the installed package.
        bundled = resources.files(project.import_name).joinpath(
            project.required_bundled_skill_path()
        )
        assert bundled.joinpath("SKILL.md").is_file()


def test_installs_are_copies_owned_by_the_package(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    lines = cli.install([cli.AGENTS["claude"]], tmp_path)
    assert len(lines) == len(cli.available_skills())
    installed = tmp_path / ".claude" / "skills" / "scenario-writing"
    assert not installed.is_symlink()
    assert (installed / "references" / "code-generation.md").is_file()
    manifest = yaml.safe_load(
        (installed / ".scenario-writing-install.json").read_text(encoding="utf-8")
    )
    assert manifest["package"] == "autoware-carla-scenario"


def test_find_repository_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert cli.find_repository_root(nested) == tmp_path


def test_unknown_skill_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["claude", "--skill", "no-such-skill", "x"]) == 2
    assert "unknown skill" in capsys.readouterr().err


def test_outside_a_repository_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "find_repository_root", lambda _: None)
    assert cli.main(["claude", "--install-only"]) == 2
    assert "inside a Git repository" in capsys.readouterr().err


def test_missing_agent_command_is_reported_before_installing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)

    def no_install(*args: object, **kwargs: object) -> None:
        raise AssertionError("installed although the agent is missing")

    monkeypatch.setattr(cli, "install", no_install)
    assert cli.main(["codex", "something"]) == 2
    assert "`codex` is not on PATH" in capsys.readouterr().err


def test_install_only_does_not_start_the_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_install(agents: list[cli.Agent], repository: Path) -> list[str]:
        calls.append(([agent.name for agent in agents], repository))
        return []

    def no_exec(*args: object) -> None:
        raise AssertionError("started the agent with --install-only")

    monkeypatch.setattr(cli, "find_repository_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "install", fake_install)
    monkeypatch.setattr(cli.os, "execv", no_exec)
    assert cli.main(["claude", "--install-only"]) == 0
    assert calls == [(["claude"], tmp_path)]
