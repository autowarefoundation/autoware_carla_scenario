"""Make Hydra 1.3's command-line parser build on Python 3.14 (#692).

Hydra 1.3 gives ``--shell-completion`` a help object that renders only
through ``str``/``repr``. From Python 3.14, argparse checks each help string
as the argument is added (``ArgumentParser._check_help``) and refuses that
object, so every ``@hydra.main`` command fails before it parses anything.

Where argparse has that check, importing this module points the name
``argparse`` in ``hydra._internal.utils`` (the module whose
``get_args_parser()`` builds Hydra's parser) at a stand-in for argparse whose
``ArgumentParser`` skips the check for help that is not a string. Every other
name resolves to argparse's own. Hydra's parser is then built by Hydra's own
code, with its own arguments, for both of its callers: ``@hydra.main`` and
``get_args()``, which shell completion uses. ``--hydra-help`` still renders
Hydra's completion instructions, which it formats with ``str()``. Nothing
outside Hydra changes: ``argparse.ArgumentParser`` itself is not touched, so
another thread building a parser of its own still gets the full check.

On Python 3.10 to 3.13 importing it changes nothing.

It must be imported before a ``@hydra.main`` function is called: the
``scenario`` entry point (``examples/run.py``) and the lanelet sweeper import
it. It can go once Hydra ships a fix.
"""

from __future__ import annotations

import argparse
import types
from typing import Any

import hydra._internal.utils as _hydra_utils

#: Whether this Python's argparse checks help strings (3.14+).
CHECKS_HELP = hasattr(argparse.ArgumentParser, "_check_help")


class LenientArgumentParser(argparse.ArgumentParser):
    """argparse's parser, except that help other than a string is not checked."""

    def _check_help(self, action: argparse.Action) -> None:
        if isinstance(action.help, str):
            getattr(super(), "_check_help")(action)


class _ArgparseForHydra(types.ModuleType):
    """argparse as ``hydra._internal.utils`` sees it: its own module, with
    :class:`LenientArgumentParser` as ``ArgumentParser``."""

    ArgumentParser = LenientArgumentParser

    def __getattr__(self, name: str) -> Any:
        return getattr(argparse, name)


if CHECKS_HELP and not isinstance(getattr(_hydra_utils, "argparse"), _ArgparseForHydra):
    setattr(_hydra_utils, "argparse", _ArgparseForHydra("argparse"))
