"""hydra_compat lets ``@hydra.main`` build its parser on Python 3.14 (#692).

On Python 3.14 argparse refuses Hydra 1.3's non-string ``--shell-completion``
help; importing ``autoware_carla_scenario.hydra_compat`` has Hydra build its
parser with an ``ArgumentParser`` that skips that check for non-string help,
without touching ``argparse.ArgumentParser`` itself.  On Python 3.10 to 3.13,
where argparse checks no help, the import changes nothing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import hydra._internal.utils as hydra_utils
import pytest

from autoware_carla_scenario import hydra_compat

CHECKS_HELP = hydra_compat.CHECKS_HELP


def test_hydra_parser_fails_unpatched() -> None:
    """The bug this works around: Hydra's parser does not build without it,
    run in a fresh interpreter that never imported hydra_compat."""
    ran = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hydra._internal.utils import get_args_parser; get_args_parser()",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if CHECKS_HELP:
        assert ran.returncode != 0
        assert "badly formed help string" in ran.stderr
    else:
        assert ran.returncode == 0, ran.stderr


@pytest.mark.skipif(not CHECKS_HELP, reason="argparse checks help from 3.14")
def test_patched_parser_builds() -> None:
    parser = hydra_utils.get_args_parser()

    assert isinstance(parser, hydra_compat.LenientArgumentParser)
    (completion,) = [
        a for a in parser._actions if "--shell-completion" in a.option_strings
    ]
    # Hydra's own lazy help, rendered by Hydra with str(): the full
    # completion instructions, as on Python 3.13.
    assert "Install or Uninstall shell completion" in f"{completion.help}"
    assert parser.parse_args(["--multirun", "a=1"]).overrides == ["a=1"]
    # get_args(), shell completion's way in, builds the same parser.
    assert hydra_utils.get_args(["-m", "b=2"]).overrides == ["b=2"]


@pytest.mark.skipif(not CHECKS_HELP, reason="argparse checks help from 3.14")
def test_global_argparse_untouched() -> None:
    """Other parsers, in any thread, keep argparse's full help check."""
    with pytest.raises(ValueError, match="badly formed help string"):
        argparse.ArgumentParser().add_argument("--x", help="%(nope)s")
    # Hydra's stand-in still checks string help.
    with pytest.raises(ValueError, match="badly formed help string"):
        hydra_compat.LenientArgumentParser().add_argument("--x", help="%(nope)s")
    # Every other name in Hydra's view of argparse is argparse's own.
    shim = getattr(hydra_utils, "argparse")
    assert shim.Namespace is argparse.Namespace
    assert shim.ArgumentParser is hydra_compat.LenientArgumentParser


@pytest.mark.skipif(CHECKS_HELP, reason="argparse checks no help before 3.14")
def test_noop_without_help_check() -> None:
    assert getattr(hydra_utils, "argparse") is argparse
    hydra_utils.get_args_parser()  # builds, as it always has here


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
    # The completion instructions themselves, not just a one-line summary.
    assert "Install or Uninstall shell completion" in helped.stdout
    assert "eval" in helped.stdout
