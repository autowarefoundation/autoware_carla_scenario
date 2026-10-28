"""The bind options the web applications (``viewer``, ``scenario-editor``) share.

Both serve an HTTP API with no authentication, so both bind the loopback
interface unless told otherwise, and both say so on stderr when they are told
to listen on anything else.  An option's default is its environment variable,
so a flag on the command line wins over the environment, which wins over the
built-in default.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys

__all__ = ["DEFAULT_HOST", "add_bind_arguments", "is_loopback", "warn_if_exposed"]

#: The address both applications bind unless ``--host`` or the environment
#: names another: reachable from this machine only.
DEFAULT_HOST = "127.0.0.1"


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
        type=int,
        # A string default goes through `type`, so a malformed environment
        # value is an argparse error (exit 2) rather than a traceback.
        default=os.environ.get(port_env) or str(default_port),
        help=f"port to listen on (env {port_env}; default {default_port})",
    )


def is_loopback(host: str) -> bool:
    """Whether *host* is reachable from this machine only.

    An IPv4 or IPv6 loopback address (``127.0.0.0/8``, ``::1``, bracketed or
    not, or IPv4-mapped ``::ffff:127.0.0.1``) or ``localhost``.  Any other
    host name may resolve to anything, so it is not; nor is ``::``, every
    IPv6 interface.
    """
    if host.lower().rstrip(".") == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False  # A host name: it may resolve to anything.
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def warn_if_exposed(prog: str, host: str, port: int, what: str) -> None:
    """Warn on stderr that *prog* is about to serve beyond this machine."""
    if is_loopback(host):
        return
    # An IPv6 address is bracketed, so its port reads apart from it.
    shown = f"[{host}]" if ":" in host and not host.startswith("[") else host
    print(  # noqa: T201
        f"{prog}: WARNING: listening on {shown}:{port} with no authentication; "
        f"anyone who can reach this port can {what}. Pass --host 127.0.0.1 to "
        "serve this machine only.",
        file=sys.stderr,
    )
