"""What the web applications' request guard decides: hosts, origins, refusals.

:mod:`.web_guard`'s ``RequestGuard`` refuses a request under a ``Host`` the
application is not served as, and a state change another site sends (#1844);
its module docstring says what is refused, and why.  The decisions are here,
on header values alone: which ``Host`` names an application answers to
(:class:`HostAllowlist`), how a ``Host`` header and an ``Origin`` read
(:func:`_split_host`, :func:`_parse_origin`), and what a request is refused
with (:func:`foreign_site`, :func:`refusal`).  :mod:`.web_guard` reads the
headers off the ASGI scope and answers; kept apart from it (fastapi, ASGI
callables) so the library check compiles these (#2364).  :mod:`.web_guard`
re-exports every name.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from .serve_bind import DEFAULT_HOST, is_loopback

__all__ = [
    "STATE_CHANGING_METHODS",
    "HostAllowlist",
    "foreign_site",
    "is_wildcard",
    "refusal",
]

#: The methods that change state, and so must come from the application's
#: own pages.
STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: The ``Sec-Fetch-Site`` values that name another site.  ``same-site`` is a
#: sibling origin (another port or subdomain), which is not this application
#: either.
_FOREIGN_FETCH_SITES = frozenset({"cross-site", "same-site"})

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _normalize_host(name: str) -> str:
    """*name* lower-cased, without IPv6 brackets or a trailing dot."""
    return name.strip().strip("[]").rstrip(".").lower()


def _is_zero_ipv4(name: str) -> bool:
    """Whether *name* is an IPv4 literal ``inet_aton`` reads as ``0.0.0.0``.

    One to four dot-separated parts, each of nothing but zeros (``0``,
    ``0.0``, ``00.0.0.0``): an empty part is no literal, and the resolver
    would look a name like that up instead.
    """
    parts = name.split(".")
    if len(parts) > 4:
        return False
    for part in parts:
        if not part or any(c != "0" for c in part):
            return False
    return True


def _is_zero_ipv6(name: str) -> bool:
    """Whether *name* is an IPv6 literal of the unspecified address (``::``).

    A valid literal (``ipaddress`` reads it) with no digit but zero: every
    group, and an embedded IPv4 address, is zero.  No zone (``::%eth0``),
    which names one interface.
    """
    if any(c not in "0:." for c in name):
        return False
    try:
        ipaddress.IPv6Address(name)
    except ValueError:
        return False
    return True


def is_wildcard(host: str) -> bool:
    """Whether *host* is an address that binds every interface (``0.0.0.0``, ``::``).

    Every numeric spelling of the unspecified address counts -- ``0``,
    ``0.0``, ``::0``, ``0:0:0:0:0:0:0:0`` bind every interface as well -- and
    an empty host, which uvicorn binds as ``0.0.0.0``.  Nothing else does: a
    string that is no IP literal (``0.0.0.0.0``, ``:::``) is looked up as a
    host name, which may resolve to one interface, so its ``Host`` is checked.
    So is one with a trailing dot (``0.0.0.0.``), an absolute host name to the
    resolver: unlike :func:`_normalize_host`, the dot is kept here.
    """
    name = host.strip().strip("[]").lower()
    if not name:
        return True
    if ":" in name:
        return _is_zero_ipv6(name)
    return _is_zero_ipv4(name)


class HostAllowlist:
    """The ``Host`` names an application answers to.

    The loopback names, and the host it is bound to; none checked when it is
    bound to every interface.  Mutable through :meth:`bind`, for the viewer,
    whose application exists before its command line is read.
    """

    bound_host: str

    def __init__(self, bound_host: str = DEFAULT_HOST) -> None:
        """Allow the loopback names and *bound_host*."""
        self.bound_host = bound_host

    def bind(self, bound_host: str) -> None:
        """Allow *bound_host* (in place of the host bound before)."""
        self.bound_host = bound_host

    def allows(self, name: str) -> bool:
        """Whether a request whose ``Host`` names *name* (no port) is answered."""
        if is_wildcard(self.bound_host):
            return True
        normalized = _normalize_host(name)
        if not normalized:
            return False
        return is_loopback(normalized) or normalized == _normalize_host(self.bound_host)


def _split_host(header: str) -> tuple[str, int | None] | None:
    """A ``Host`` header's name and port; ``None`` when it is malformed."""
    header = header.strip()
    no_port: int | None = None
    if header.startswith("["):
        end = header.find("]")
        if end < 0:
            return None
        name, rest = header[1:end], header[end + 1 :]
        if not rest:
            return name, no_port
        if not rest.startswith(":"):
            return None
        port_text = rest[1:]
    elif header.count(":") == 1:
        colon = header.find(":")
        name, port_text = header[:colon], header[colon + 1 :]
    elif ":" in header:
        return None  # An IPv6 address must be bracketed.
    else:
        return header, no_port
    if not (port_text.isascii() and port_text.isdigit()):
        return None
    port: int | None = int(port_text)
    return name, port


def _origin(scheme: str, host: str, port: int | None) -> tuple[str, str, int | None]:
    """An origin as a comparable tuple, the scheme's default port filled in.

    A port of 0 counts as none, as ``port or default`` had it.
    """
    scheme = scheme.lower()
    filled: int | None = port
    if port is None or port == 0:
        filled = None
        if scheme in _DEFAULT_PORTS:
            filled = _DEFAULT_PORTS[scheme]
    return scheme, _normalize_host(host), filled


def _parse_origin(text: str) -> tuple[str, str, int | None] | None:
    """An ``Origin`` header's origin; ``None`` for ``null`` or a malformed one."""
    try:
        parts = urlsplit(text.strip())
        port = parts.port
    except ValueError:
        return None
    hostname = parts.hostname
    if not parts.scheme or hostname is None or not hostname:
        return None
    return _origin(parts.scheme, hostname, port)


def foreign_site(
    headers: dict[str, str], scheme: str, host: tuple[str, int | None] | None
) -> tuple[int, str] | None:
    """The 403 a request from another site is refused with; ``None`` otherwise.

    *headers* are the request's, by lower-cased name; *scheme* is its own
    (``http`` when the server names none); *host* is its own ``Host``
    (:func:`_split_host`): without one, no ``Origin`` is the request's own.
    """
    fetch_site = headers["sec-fetch-site"] if "sec-fetch-site" in headers else ""
    if fetch_site.strip().lower() in _FOREIGN_FETCH_SITES:
        return 403, "Cross-site request refused"
    if "origin" in headers:
        if host is None:
            return 403, "Cross-origin request refused"
        name, port = host
        own = _origin(scheme, name, port)
        parsed = _parse_origin(headers["origin"])
        if parsed is None or parsed != own:
            return 403, "Cross-origin request refused"
    return None


def refusal(
    headers: dict[str, str], method: str, scheme: str, hosts: HostAllowlist
) -> tuple[int, str] | None:
    """The status and message a request is refused with; ``None`` to serve it.

    *headers* are the request's, by lower-cased name, *method* and *scheme*
    its own; *hosts* the ``Host`` names the application answers to.  A
    foreign ``Host`` is refused with 400, a state change from another site
    with 403 (:func:`foreign_site`).
    """
    host = _split_host(headers["host"] if "host" in headers else "")
    if host is None or not hosts.allows(host[0]):
        return 400, "Invalid Host header"
    if method.upper() not in STATE_CHANGING_METHODS:
        return None
    return foreign_site(headers, scheme, host)
