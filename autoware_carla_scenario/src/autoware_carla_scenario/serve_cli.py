"""The bind options the web applications (``viewer``, ``scenario-editor``) share.

Both serve an HTTP API with no authentication, so both bind the loopback
interface unless told otherwise, and both say so on stderr when they are told
to listen on anything else.  An option's default is its environment variable,
so a flag on the command line wins over the environment, which wins over the
built-in default.

``port`` is the ``type=`` of every TCP port option in the package (these two,
``scenario-hils`` and ``detect-no-3d-model``): a port outside 1-65535, or one
that is not an integer, is a usage error (exit 2) rather than a traceback from
whatever opens the socket (#1774).  ``check_port`` is the range check on its
own, for a port that does not come from the command line (the editor's
"fetch OpenDRIVE" form, #1781).

What does not need ``argparse`` -- the default host, the range check, which
hosts are loopback ones and the warning -- is the checked :mod:`.serve_bind`'s,
re-exported here as the same functions.
"""

from __future__ import annotations

import argparse
import os

from .serve_bind import DEFAULT_HOST, check_port, is_loopback, warn_if_exposed

__all__ = [
    "DEFAULT_HOST",
    "add_bind_arguments",
    "check_port",
    "is_loopback",
    "port",
    "warn_if_exposed",
]


def port(text: str) -> int:
    """A ``--port`` value: a TCP port number, within 1-65535.

    For ``argparse``'s ``type=``, so a bad value is a usage error (exit 2).
    0 is refused too: it would have the system pick a port, which the web
    applications would neither report nor be reachable at the URL their docs
    give, and which no CARLA server listens on.
    """
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    try:
        return check_port(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_bind_arguments(
    parser: argparse.ArgumentParser,
    *,
    host_env: str,
    port_env: str,
    default_port: int,
) -> None:
    """Add ``--host`` and ``--port``, defaulting to *host_env* and *port_env*."""
    parser.add_argument(
        "--host",
        default=os.environ.get(host_env) or DEFAULT_HOST,
        help=f"address to bind (env {host_env}; default {DEFAULT_HOST}, this "
        "machine only). 0.0.0.0 serves every interface, with no authentication",
    )
    parser.add_argument(
        "--port",
        type=port,
        # A string default goes through `type`, so a malformed or out-of-range
        # environment value is an argparse error (exit 2) rather than a
        # traceback.
        default=os.environ.get(port_env) or str(default_port),
        help=f"port to listen on, 1-65535 (env {port_env}; default {default_port})",
    )
