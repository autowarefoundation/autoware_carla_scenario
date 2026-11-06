"""The request guard the web applications (``viewer``, ``scenario-editor``) share.

Both bind the loopback interface (:mod:`.serve_cli`), but a loopback binding
only keeps other *machines* out: a web page open in the user's own browser can
still send requests to ``127.0.0.1``.  :class:`RequestGuard`, an ASGI
middleware installed in both applications, refuses the two ways such a page
gets in (#1844):

* **Cross-site requests.** A POST, PUT, PATCH or DELETE is refused with 403
  when the browser says it comes from another site (``Sec-Fetch-Site:
  cross-site`` or ``same-site``), or when it carries an ``Origin`` that is not
  the request's own scheme, host and port (``Origin: null`` included).  A
  request with neither header -- curl, a script -- is not a browser's
  cross-site request, and is let through.
* **DNS rebinding.** A request whose ``Host`` is not a name the application is
  served under is refused with 400: a page on ``attacker.example`` that
  rebinds its name to ``127.0.0.1`` still sends ``Host: attacker.example``.
  The names allowed are the loopback ones (``localhost``, ``127.0.0.1``,
  ``::1``, any loopback address) and the ``--host`` the application is bound
  to, on any port.  An application bound to every interface (``0.0.0.0`` or
  ``::``) was asked to answer whatever name it is reached under, so it checks
  no ``Host``; the ``Origin`` check still applies.

Only the ``Host`` check applies to a GET: reading is not a state change, and
the rebinding it stops is what would let a foreign page *read* a response.
A GET route that does more than read -- one that makes the server fetch or
clone a git repository, parse a map, or write a file -- declares the
:func:`refuse_cross_site` dependency, which refuses a cross-site request to it
with 403 by the same rules (#1850).  ``Sec-Fetch-Site: none`` (a URL typed in
the address bar, a bookmark) is served, as is a request with neither header.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from urllib.parse import urlsplit

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from .serve_cli import DEFAULT_HOST, is_loopback

__all__ = [
    "HostAllowlist",
    "RequestGuard",
    "cross_site_refusal",
    "is_wildcard",
    "refuse_cross_site",
]

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

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


def is_wildcard(host: str) -> bool:
    """Whether *host* is an address that binds every interface (``0.0.0.0``, ``::``)."""
    return _normalize_host(host) in ("0.0.0.0", "::", "")  # noqa: S104


class HostAllowlist:
    """The ``Host`` names an application answers to.

    The loopback names, and the host it is bound to; none checked when it is
    bound to every interface.  Mutable through :meth:`bind`, for the viewer,
    whose application exists before its command line is read.
    """

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
    if header.startswith("["):
        end = header.find("]")
        if end < 0:
            return None
        name, rest = header[1:end], header[end + 1 :]
        if not rest:
            return name, None
        if not rest.startswith(":"):
            return None
        port_text = rest[1:]
    elif header.count(":") == 1:
        name, port_text = header.split(":")
    elif ":" in header:
        return None  # An IPv6 address must be bracketed.
    else:
        return header, None
    if not (port_text.isascii() and port_text.isdigit()):
        return None
    return name, int(port_text)


def _origin(scheme: str, host: str, port: int | None) -> tuple[str, str, int | None]:
    """An origin as a comparable tuple, the scheme's default port filled in."""
    scheme = scheme.lower()
    return scheme, _normalize_host(host), port or _DEFAULT_PORTS.get(scheme)


def _parse_origin(text: str) -> tuple[str, str, int | None] | None:
    """An ``Origin`` header's origin; ``None`` for ``null`` or a malformed one."""
    try:
        parts = urlsplit(text.strip())
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    return _origin(parts.scheme, parts.hostname, port)


def _headers(scope: Scope) -> dict[str, str]:
    """*scope*'s headers by lower-cased name, the first of each name kept."""
    headers: dict[str, str] = {}
    for key, value in scope.get("headers", ()):
        headers.setdefault(key.decode("latin-1").lower(), value.decode("latin-1"))
    return headers


def _foreign_site(
    scope: Scope, headers: dict[str, str], host: tuple[str, int | None] | None
) -> tuple[int, str] | None:
    """The 403 a request from another site is refused with; ``None`` otherwise.

    *host* is the request's own ``Host`` (:func:`_split_host`); without one,
    no ``Origin`` is the request's own.
    """
    if headers.get("sec-fetch-site", "").strip().lower() in _FOREIGN_FETCH_SITES:
        return 403, "Cross-site request refused"
    origin = headers.get("origin")
    if origin is not None:
        own = _origin(scope.get("scheme", "http"), *host) if host else None
        if own is None or _parse_origin(origin) != own:
            return 403, "Cross-origin request refused"
    return None


def cross_site_refusal(scope: Scope) -> tuple[int, str] | None:
    """The 403 *scope* is refused with when another site sent it, whatever its method.

    The rules :class:`RequestGuard` applies to a state change: ``Sec-Fetch-Site``
    ``cross-site`` or ``same-site``, or an ``Origin`` that is not the request's
    own scheme, host and port.  ``None`` to serve it.
    """
    headers = _headers(scope)
    return _foreign_site(scope, headers, _split_host(headers.get("host", "")))


def refuse_cross_site(request: Request) -> None:
    """A route dependency refusing a request another site sends, GET included.

    For a GET route that does more than read: one that makes the server reach
    the network (a git fetch or clone), parse a map, or write a file, which a
    foreign page could otherwise make it do (#1850).  The application's own
    pages send ``Sec-Fetch-Site: same-origin``; a typed URL or a bookmark
    sends ``none``; curl sends neither.  All three are served.

    Raises:
        HTTPException: 403, the request comes from another site.
    """
    refusal = cross_site_refusal(request.scope)
    if refusal is not None:
        status, message = refusal
        raise HTTPException(status_code=status, detail=message)


class RequestGuard:
    """ASGI middleware refusing cross-site state changes and foreign ``Host`` names.

    See the module's docstring for what is refused, and why.
    """

    def __init__(self, app: ASGIApp, *, hosts: HostAllowlist) -> None:
        """Guard *app*, answering the ``Host`` names *hosts* allows."""
        self.app = app
        self.hosts = hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Answer a refused request here; hand any other to the application."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        refusal = self.refusal(scope)
        if refusal is not None:
            status, message = refusal
            await PlainTextResponse(message, status_code=status)(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def refusal(self, scope: Scope) -> tuple[int, str] | None:
        """The status and message *scope* is refused with; ``None`` to serve it."""
        headers = _headers(scope)
        host = _split_host(headers.get("host", ""))
        if host is None or not self.hosts.allows(host[0]):
            return 400, "Invalid Host header"
        if scope.get("method", "GET").upper() not in STATE_CHANGING_METHODS:
            return None
        return _foreign_site(scope, headers, host)
