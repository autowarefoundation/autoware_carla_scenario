"""The request guard the web applications (``viewer``, ``scenario-editor``) share.

Both bind the loopback interface (:mod:`.serve_bind`), but a loopback binding
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

**Framing.** Every response the guard passes on -- a page, a partial, a
static file, a 403, 404 or other error the application answers, and the
guard's own refusals -- carries ``X-Frame-Options: DENY`` and
``Content-Security-Policy: frame-ancestors 'none'`` (:data:`FRAME_HEADERS`),
so no page, the applications' own included, can load either one in a frame
and trick a click on Run, Rerun or Export (clickjacking, #2053).  Neither
application frames its own pages: the rerun viewer and the Lanelet2 map
viewer are wasm modules drawn on a canvas of the page itself.  The one
response without them is the plain-text 500 of an unhandled exception, which
Starlette's outermost ``ServerErrorMiddleware`` answers: it has nothing to
click.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from fastapi import HTTPException
from fastapi.requests import Request
from fastapi.responses import PlainTextResponse

from . import web_origin

# The decisions on header values, which the library check compiles from their
# own module; every name of it is re-exported here, where it has always been
# imported from (the same objects), and so are serve_bind's.
from .serve_bind import DEFAULT_HOST as DEFAULT_HOST
from .serve_bind import is_loopback as is_loopback
from .web_origin import _DEFAULT_PORTS as _DEFAULT_PORTS
from .web_origin import _FOREIGN_FETCH_SITES as _FOREIGN_FETCH_SITES
from .web_origin import STATE_CHANGING_METHODS as STATE_CHANGING_METHODS
from .web_origin import HostAllowlist as HostAllowlist
from .web_origin import _normalize_host as _normalize_host
from .web_origin import _origin as _origin
from .web_origin import _parse_origin as _parse_origin
from .web_origin import _split_host as _split_host
from .web_origin import is_wildcard as is_wildcard

__all__ = [
    "FRAME_HEADERS",
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

#: The headers every response carries so that no page can frame it (#2053):
#: ``X-Frame-Options`` for browsers that predate CSP's ``frame-ancestors``.
FRAME_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
)


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
    no ``Origin`` is the request's own.  The decision is the checked
    :func:`.web_origin.foreign_site`'s.
    """
    return web_origin.foreign_site(headers, scope.get("scheme", "http"), host)


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


def _unframeable(send: Send) -> Send:
    """*send*, adding :data:`FRAME_HEADERS` to the response's start.

    An ``X-Frame-Options`` the application set itself is replaced.  A
    ``Content-Security-Policy`` it set is kept: a browser enforces every
    policy a response carries, so the ``frame-ancestors`` one adds to it.
    """

    async def _send(message: Message) -> None:
        if message.get("type") == "http.response.start":
            kept = [
                (name, value)
                for name, value in message.get("headers", ())
                if bytes(name).lower() != b"x-frame-options"
            ]
            message["headers"] = [*kept, *FRAME_HEADERS]
        await send(message)

    return _send


class RequestGuard:
    """ASGI middleware refusing cross-site state changes and foreign ``Host`` names.

    It also adds :data:`FRAME_HEADERS` to every HTTP response it lets out.
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
        send = _unframeable(send)
        refusal = self.refusal(scope)
        if refusal is not None:
            status, message = refusal
            await PlainTextResponse(message, status_code=status)(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def refusal(self, scope: Scope) -> tuple[int, str] | None:
        """The status and message *scope* is refused with; ``None`` to serve it."""
        return web_origin.refusal(
            _headers(scope),
            scope.get("method", "GET"),
            scope.get("scheme", "http"),
            self.hosts,
        )
