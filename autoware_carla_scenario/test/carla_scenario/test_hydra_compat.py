"""hydra_compat lets ``@hydra.main`` build its parser on Python 3.14 (#692).

On Python 3.14 argparse refuses Hydra 1.3's non-string ``--shell-completion``
help; importing ``autoware_carla_scenario.hydra_compat`` installs a parser
that accepts it and renders a plain string instead.  On Python 3.10 to 3.13,
where argparse checks no help, the import changes nothing.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from hydra._internal.utils import get_args_parser as hydra_get_args_parser

from autoware_carla_scenario import hydra_compat

#: Whether this Python's argparse checks help strings (3.14+).
CHECKS_HELP = hasattr(argparse.ArgumentParser, "_check_help")

hydra_main_module = importlib.import_module("hydra.main")


@pytest.mark.skipif(not CHECKS_HELP, reason="argparse checks help from 3.14")
def test_hydra_parser_fails_unpatched() -> None:
    """The bug this works around: Hydra's own parser does not build."""
    with pytest.raises(ValueError, match="badly formed help string"):
        hydra_get_args_parser()


@pytest.mark.skipif(not CHECKS_HELP, reason="argparse checks help from 3.14")
def test_patched_parser_builds_and_renders_help() -> None:
    assert hydra_main_module.get_args_parser is hydra_compat.get_args_parser

    parser = hydra_compat.get_args_parser()

    (completion,) = [
        a for a in parser._actions if "--shell-completion" in a.option_strings
    ]
    assert completion.help == hydra_compat.SHELL_COMPLETION_HELP
    assert all(a.help is None or isinstance(a.help, str) for a in parser._actions)
    assert hydra_compat.SHELL_COMPLETION_HELP in parser.format_help()
    assert parser.parse_args(["--multirun", "a=1"]).overrides == ["a=1"]
    # argparse's check is restored, and still refuses a bad string.
    assert getattr(argparse.ArgumentParser, "_check_help") is hydra_compat._check_help
    with pytest.raises(ValueError, match="badly formed help string"):
        argparse.ArgumentParser().add_argument("--x", help="%(nope)s")


@pytest.mark.skipif(CHECKS_HELP, reason="argparse checks no help before 3.14")
def test_noop_without_help_check() -> None:
    assert hydra_compat._check_help is None
    assert hydra_main_module.get_args_parser is hydra_get_args_parser
    hydra_get_args_parser()  # builds, as it always has here


def test_hydra_main_app_help(tmp_path: Path) -> None:
    """A ``@hydra.main`` app that imports hydra_compat answers ``--hydra-help``
    and runs, on every supported Python."""
    app = tmp_path / "app.py"
    app.write_text(
        textwrap.dedent(
            """
            import hydra

            import autoware_carla_scenario.hydra_compat  # noqa: F401


            @hydra.main(version_base=None, config_path=None, config_name=None)
            def task(cfg):
                print(f"value={cfg.value}")


            if __name__ == "__main__":
                task()
            """
        )
    )
    run_dir = f"hydra.run.dir={tmp_path / 'out'}"

    ran = subprocess.run(
        [sys.executable, str(app), "+value=7", run_dir],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 0, ran.stderr
    assert "value=7" in ran.stdout

    helped = subprocess.run(
        [sys.executable, str(app), "--hydra-help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert helped.returncode == 0, helped.stderr
    assert "--shell-completion" in helped.stdout
