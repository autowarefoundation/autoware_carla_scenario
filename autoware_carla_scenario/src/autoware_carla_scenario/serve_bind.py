"""Where the web applications (``viewer``, ``scenario-editor``) listen: the
default host, the TCP port range, which hosts are loopback ones, and the
warning when they listen beyond this machine.

Both serve an HTTP API with no authentication, so both bind
:data:`DEFAULT_HOST`, the loopback interface, unless told otherwise, and
both say so on stderr (:func:`warn_if_exposed`) when they are told to listen
on a host :func:`is_loopback` does not take.  :func:`check_port` is the
range check of every TCP port the package reads, from the command line
(``serve_cli.port``) or not (the editor's "fetch OpenDRIVE" form, #1781).

:mod:`.serve_cli`, which adds the options to an ``argparse`` parser,
re-exports these.
"""

from __future__ import annotations

import ipaddress
import sys

__all__ = [
    "DEFAULT_HOST",
    "check_port",
    "is_loopback",
    "warn_if_exposed",
]

#: The address both applications bind unless ``--host`` or the environment
#: names another: reachable from this machine only.
DEFAULT_HOST = "127.0.0.1"


def check_port(number: int) -> int:
    """Return *number* if it is a TCP port number, within 1-65535.

    Raises:
        ValueError: If it is not, worded to follow the option or field's
            name ("must be within 1-65535, got 70000").
    """
    if not 1 <= number <= 65535:
        raise ValueError(f"must be within 1-65535, got {number}")
    return number


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
    if isinstance(address, ipaddress.IPv6Address):
        # The member, for Codon, which reads no attribute of one member off
        # the union ip_address() gives (docs/typecheck.md).
        v6: ipaddress.IPv6Address = address
        if v6.ipv4_mapped:
            return v6.ipv4_mapped.is_loopback
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
