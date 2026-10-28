"""Make Hydra 1.3's command-line parser build on Python 3.14 (#692).

Hydra 1.3 gives ``--shell-completion`` a help object that renders only
through ``repr``. From Python 3.14, argparse checks each help string as the
argument is added (``ArgumentParser._check_help``) and refuses that object,
so every ``@hydra.main`` command fails before it parses anything.

Where argparse has that check, importing this module replaces the parser
``hydra.main`` builds with one made the same way, except that the check is
skipped for help that is not a string and that help is then replaced by a
plain string, so ``--help`` renders too. On Python 3.10 to 3.13 importing it
changes nothing.

It must be imported before a ``@hydra.main`` function is called: the
``scenario`` entry point (``examples/run.py``) and the lanelet sweeper import
it. It can go once Hydra ships a fix.
"""

from __future__ import annotations

import argparse
import importlib

from hydra._internal.utils import get_args_parser as _hydra_get_args_parser

#: What ``--shell-completion`` says in ``--help`` instead of Hydra's object.
SHELL_COMPLETION_HELP = "Install or Uninstall shell completion"

#: argparse's help check (Python 3.14+), or ``None`` where there is none.
_check_help = getattr(argparse.ArgumentParser, "_check_help", None)


def get_args_parser() -> argparse.ArgumentParser:
    """Hydra's command-line parser, built so Python 3.14 accepts it."""
    check_help = _check_help
    assert check_help is not None  # only installed where argparse checks help

    def check_string_help(
        parser: argparse.ArgumentParser, action: argparse.Action
    ) -> None:
        if isinstance(action.help, str):
            check_help(parser, action)

    setattr(argparse.ArgumentParser, "_check_help", check_string_help)
    try:
        parser = _hydra_get_args_parser()
    finally:
        setattr(argparse.ArgumentParser, "_check_help", check_help)
    for action in parser._actions:
        if action.help is not None and not isinstance(action.help, str):
            action.help = SHELL_COMPLETION_HELP
    return parser


if _check_help is not None:
    # ``hydra.main`` names the decorator, so fetch the module that holds it by
    # name: the decorator looks ``get_args_parser`` up there when it is called.
    setattr(importlib.import_module("hydra.main"), "get_args_parser", get_args_parser)
