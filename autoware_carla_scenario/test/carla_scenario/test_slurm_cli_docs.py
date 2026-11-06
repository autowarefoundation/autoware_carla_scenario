"""Every ``scenario-slurm`` option has help text and is in docs/slurm.md (#1768)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from autoware_carla_scenario.slurm import cli

DOCS = Path(__file__).resolve().parents[2] / "docs" / "slurm.md"


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    parser = cli._parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("scenario-slurm has no subcommands")


def _arguments() -> list[tuple[str, argparse.Action]]:
    return [
        (name, action)
        for name, sub in _subcommands().items()
        for action in sub._actions
        if not isinstance(action, argparse._HelpAction)
    ]


def test_subcommands_are_found() -> None:
    assert {"status", "wait", "collect", "resubmit", "cancel"} <= set(_subcommands())


@pytest.mark.parametrize(
    ("command", "action"),
    _arguments(),
    ids=lambda v: v if isinstance(v, str) else "/".join(v.option_strings or [v.dest]),
)
def test_every_argument_has_help(command: str, action: argparse.Action) -> None:
    assert action.help and action.help.strip(), (command, action.dest)


@pytest.mark.parametrize("command", sorted(_subcommands()))
def test_every_subcommand_and_option_is_documented(command: str) -> None:
    text = DOCS.read_text(encoding="utf-8")
    assert f"scenario-slurm {command}" in text
    for action in _subcommands()[command]._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        for option in action.option_strings:
            # `--cases a,b` or `--cases`, not the start of `--cases-per-task`.
            pattern = rf"`{re.escape(option)}[` ]"
            assert re.search(pattern, text), f"{command} {option} is not in {DOCS}"
