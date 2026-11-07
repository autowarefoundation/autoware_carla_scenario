"""``docs/api.md``'s "CLI entry points" table lists every console script (#1776).

The table says it lists what ``pyproject.toml`` defines; it had fallen behind
by half the commands.  Each ``[project.scripts]`` entry must have a row with
its command and its ``module:function`` exactly as ``pyproject.toml`` has
them, and the table must not list a command that is not defined there.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# tomllib landed in 3.11, and 3.10 is still the floor of the supported range.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_API_DOC = _PACKAGE_ROOT / "docs" / "api.md"
_ROW = re.compile(r"^\| `(?P<command>[^`]+)` \| `(?P<target>[^`]+)` \|")


@pytest.fixture(scope="module")
def scripts() -> dict[str, str]:
    """Return the package's console scripts."""
    data = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text())
    return data["project"]["scripts"]


@pytest.fixture(scope="module")
def table() -> dict[str, str]:
    """Return the "CLI entry points" table of docs/api.md, command -> target."""
    text = _API_DOC.read_text()
    section = text.split("## CLI entry points", 1)[1].split("\n## ", 1)[0]
    rows: dict[str, str] = {}
    in_table = False
    for line in section.splitlines():
        if line.startswith("| Command | Module |"):
            in_table = True
            continue
        if in_table and not line.startswith("|"):
            break
        match = _ROW.match(line)
        if in_table and match:
            rows[match["command"]] = match["target"]
    assert rows, "docs/api.md has no CLI entry points table"
    return rows


def test_every_script_is_listed(scripts: dict[str, str], table: dict[str, str]) -> None:
    missing = sorted(set(scripts) - set(table))
    assert not missing, f"docs/api.md's CLI entry points table lacks {missing}"


def test_every_row_is_a_script(scripts: dict[str, str], table: dict[str, str]) -> None:
    extra = sorted(set(table) - set(scripts))
    assert not extra, f"docs/api.md lists commands pyproject.toml lacks: {extra}"


def test_every_row_names_the_scripts_target(
    scripts: dict[str, str], table: dict[str, str]
) -> None:
    wrong = {
        command: (table[command], target)
        for command, target in scripts.items()
        if command in table and table[command] != target
    }
    assert not wrong, f"docs/api.md (row, pyproject.toml) disagree: {wrong}"


def test_rows_follow_pyproject_order(
    scripts: dict[str, str], table: dict[str, str]
) -> None:
    assert list(table) == list(scripts)


def test_usage_overview_lists_every_script(scripts: dict[str, str]) -> None:
    """docs/usage.md's "CLI Commands Overview" table, which says it lists the
    commands the package provides, names every console script (#1934)."""
    text = (_PACKAGE_ROOT / "docs" / "usage.md").read_text()
    section = text.split("## CLI Commands Overview", 1)[1].split("\n## ", 1)[0]
    listed = set(re.findall(r"^\| \[?`([^`]+)`", section, re.M))
    missing = [command for command in scripts if command not in listed]
    assert not missing, f"docs/usage.md's CLI Commands Overview lacks {missing}"
    extra = sorted(listed - set(scripts) - {"Command"})
    assert not extra, f"docs/usage.md lists commands pyproject.toml lacks: {extra}"
