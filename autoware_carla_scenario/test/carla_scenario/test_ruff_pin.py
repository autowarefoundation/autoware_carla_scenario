"""``uv run ruff`` is the ruff CI runs (#2055).

CI lints with the pre-commit hook's ruff, pinned by its ``rev``. The dev
dependency group pins the same version, so ``uv run ruff`` does not fall
through to whatever ruff the host has -- a newer one reformats files and
reports errors CI never sees.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parents[3]
_RUFF_HOOKS = "https://github.com/astral-sh/ruff-pre-commit"


def _pre_commit_ruff() -> str:
    config = yaml.safe_load((_ROOT / ".pre-commit-config.yaml").read_text())
    (repo,) = [repo for repo in config["repos"] if repo["repo"] == _RUFF_HOOKS]
    return str(repo["rev"]).removeprefix("v")


def _dev_group_ruff() -> str:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    pins = [
        match.group(1)
        for requirement in pyproject["dependency-groups"]["dev"]
        if (match := re.fullmatch(r"ruff==([^;\s]+)", requirement.strip()))
    ]
    assert len(pins) == 1, f"the dev group must pin ruff exactly once: {pins}"
    return pins[0]


def _locked_ruff() -> str:
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    (ruff,) = [pkg for pkg in lock["package"] if pkg["name"] == "ruff"]
    return str(ruff["version"])


def test_dev_group_pins_the_pre_commit_ruff() -> None:
    assert _dev_group_ruff() == _pre_commit_ruff()


def test_lock_holds_the_pinned_ruff() -> None:
    assert _locked_ruff() == _dev_group_ruff()
