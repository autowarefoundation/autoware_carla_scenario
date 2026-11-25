"""The request guard's decisions, moved into the checked
:mod:`autoware_carla_scenario.web_origin` (#2364).

``web_guard`` decided on the ASGI scope; the decisions now take header values
alone, rewritten for Codon (``port or default`` mixed an ``int`` with
``None``).  These tests hold them to the code they replace over many inputs,
and check that ``web_guard`` still hands out the same objects.
"""

from __future__ import annotations

import itertools
import random
from typing import Any
from urllib.parse import urlsplit

import pytest

from autoware_carla_scenario import serve_bind, web_guard, web_origin
from autoware_carla_scenario.serve_bind import DEFAULT_HOST, is_loopback

# ---------------------------------------------------------------------------
# The code web_guard ran before #2364, verbatim but for the names.  Only
# ``is_loopback`` is the live one: the old web_guard imported it from
# serve_bind too, and #2364 did not change it.
# ---------------------------------------------------------------------------

_OLD_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}
_OLD_STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_OLD_FOREIGN = frozenset({"cross-site", "same-site"})


def _old_normalize_host(name: str) -> str:
    return name.strip().strip("[]").rstrip(".").lower()


def _old_is_wildcard(host: str) -> bool:
    return _old_normalize_host(host) in ("0.0.0.0", "::", "")  # noqa: S104


def _old_allows(bound_host: str, name: str) -> bool:
    if _old_is_wildcard(bound_host):
        return True
    normalized = _old_normalize_host(name)
    if not normalized:
        return False
    return is_loopback(normalized) or normalized == _old_normalize_host(bound_host)


def _old_split_host(header: str) -> tuple[str, int | None] | None:
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
        return None
    else:
        return header, None
    if not (port_text.isascii() and port_text.isdigit()):
        return None
    return name, int(port_text)


def _old_origin(
    scheme: str, host: str, port: int | None
) -> tuple[str, str, int | None]:
    scheme = scheme.lower()
    return scheme, _old_normalize_host(host), port or _OLD_DEFAULT_PORTS.get(scheme)


def _old_parse_origin(text: str) -> tuple[str, str, int | None] | None:
    try:
        parts = urlsplit(text.strip())
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    return _old_origin(parts.scheme, parts.hostname, port)


def _old_foreign_site(
    scope: dict[str, Any], headers: dict[str, str], host: tuple[str, int | None] | None
) -> tuple[int, str] | None:
    if headers.get("sec-fetch-site", "").strip().lower() in _OLD_FOREIGN:
        return 403, "Cross-site request refused"
    origin = headers.get("origin")
    if origin is not None:
        own = _old_origin(scope.get("scheme", "http"), *host) if host else None
        if own is None or _old_parse_origin(origin) != own:
            return 403, "Cross-origin request refused"
    return None


def _old_headers(scope: dict[str, Any]) -> dict[str, str]:
    """*scope*'s headers by lower-cased name, the first of each name kept."""
    headers: dict[str, str] = {}
    for key, value in scope.get("headers", ()):
        headers.setdefault(key.decode("latin-1").lower(), value.decode("latin-1"))
    return headers


def _old_refusal(scope: dict[str, Any], bound_host: str) -> tuple[int, str] | None:
    headers = _old_headers(scope)
    host = _old_split_host(headers.get("host", ""))
    if host is None or not _old_allows(bound_host, host[0]):
        return 400, "Invalid Host header"
    if scope.get("method", "GET").upper() not in _OLD_STATE_CHANGING:
        return None
    return _old_foreign_site(scope, headers, host)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

_NAMES = [
    "",
    " ",
    "localhost",
    "LOCALHOST.",
    "127.0.0.1",
    "127.1.2.3",
    "::1",
    "[::1]",
    "::",
    "0.0.0.0",
    "::ffff:127.0.0.1",
    "192.168.1.5",
    "example.com",
    "Attacker.Example.",
    "[fe80::1]",
    "a:b",
]
_PORTS = ["", ":", ":0", ":80", ":443", ":8080", ":65536", ":x", ":٣", ": 1", ":-1"]
_SCHEMES = ["http", "HTTPS", "ws", "wss", "ftp", ""]


def _random_host_header(rng: random.Random) -> str:
    """A ``Host`` header, well formed or not."""
    if rng.random() < 0.15:
        alphabet = "[]:.0123456789abcdefABCDEF lohst"
        return "".join(rng.choice(alphabet) for _ in range(rng.randrange(12)))
    name = rng.choice(_NAMES)
    if ":" in name and not name.startswith("[") and rng.random() < 0.7:
        name = f"[{name}]"
    return f"{rng.choice(['', ' '])}{name}{rng.choice(_PORTS)}"


def _random_origin(rng: random.Random) -> str:
    """An ``Origin`` header, well formed or not."""
    roll = rng.random()
    if roll < 0.1:
        return rng.choice(["null", "", " ", "http://", "http://[::1", "://x"])
    if roll < 0.2:
        alphabet = "htps:/[]0123456789.lochst"
        return "".join(rng.choice(alphabet) for _ in range(rng.randrange(20)))
    name = rng.choice(_NAMES).strip()
    if ":" in name and not name.startswith("["):
        name = f"[{name}]"
    return f"{rng.choice(_SCHEMES)}://{name}{rng.choice(_PORTS)}"


def _random_scope(rng: random.Random) -> dict[str, Any]:
    headers: list[tuple[bytes, bytes]] = []
    if rng.random() < 0.9:
        headers.append((b"host", _random_host_header(rng).encode("utf-8")))
    if rng.random() < 0.2:
        # A second Host, in another case: the first one is the one read.
        name = rng.choice([b"host", b"Host", b"HOST"])
        headers.append((name, _random_host_header(rng).encode("utf-8")))
    if rng.random() < 0.5:
        headers.append((b"Origin", _random_origin(rng).encode("utf-8")))
    if rng.random() < 0.4:
        site = rng.choice(["cross-site", " Same-Site ", "same-origin", "none", ""])
        headers.append((b"sec-fetch-site", site.encode("latin-1")))
    rng.shuffle(headers)
    scope: dict[str, Any] = {"type": "http", "headers": headers}
    if rng.random() < 0.9:
        scope["method"] = rng.choice(["GET", "post", "PUT", "PATCH", "delete", "HEAD"])
    if rng.random() < 0.9:
        scope["scheme"] = rng.choice(["http", "https", "HTTP", "ws"])
    return scope


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_web_guard_re_exports_the_same_objects() -> None:
    for name in (
        "HostAllowlist",
        "is_wildcard",
        "STATE_CHANGING_METHODS",
        "_FOREIGN_FETCH_SITES",
        "_DEFAULT_PORTS",
        "_normalize_host",
        "_split_host",
        "_origin",
        "_parse_origin",
    ):
        assert getattr(web_guard, name) is getattr(web_origin, name), name
    assert web_guard.is_loopback is serve_bind.is_loopback
    assert web_guard.DEFAULT_HOST is serve_bind.DEFAULT_HOST
    assert web_origin.HostAllowlist().bound_host == DEFAULT_HOST


def test_the_constants_are_unchanged() -> None:
    assert web_origin.STATE_CHANGING_METHODS == _OLD_STATE_CHANGING
    assert web_origin._FOREIGN_FETCH_SITES == _OLD_FOREIGN
    assert web_origin._DEFAULT_PORTS == _OLD_DEFAULT_PORTS


@pytest.mark.parametrize("seed", range(4))
def test_a_host_header_splits_as_before(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(3000):
        header = _random_host_header(rng)
        assert web_origin._split_host(header) == _old_split_host(header), header


def test_a_host_is_allowed_as_before() -> None:
    names = _NAMES + [n.strip("[]") for n in _NAMES]
    for bound, name in itertools.product(names, names):
        assert web_origin.is_wildcard(bound) == _old_is_wildcard(bound)
        hosts = web_origin.HostAllowlist(bound)
        assert hosts.allows(name) == _old_allows(bound, name), (bound, name)
        hosts.bind(name)
        assert hosts.allows(bound) == _old_allows(name, bound), (name, bound)


@pytest.mark.parametrize("port", [None, 0, 1, 80, 443, 8080])
@pytest.mark.parametrize("scheme", _SCHEMES)
def test_an_origin_fills_the_default_port_as_before(
    scheme: str, port: int | None
) -> None:
    for host in _NAMES:
        assert web_origin._origin(scheme, host, port) == _old_origin(scheme, host, port)


@pytest.mark.parametrize("seed", range(4))
def test_an_origin_header_parses_as_before(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(3000):
        text = _random_origin(rng)
        assert web_origin._parse_origin(text) == _old_parse_origin(text), text


@pytest.mark.parametrize("seed", range(6))
def test_a_request_is_refused_as_before(seed: int) -> None:
    rng = random.Random(seed)
    refused = served = 0
    for _ in range(3000):
        scope = _random_scope(rng)
        bound = rng.choice(["127.0.0.1", "0.0.0.0", "192.168.1.5", "example.com"])
        guard = web_guard.RequestGuard(None, hosts=web_guard.HostAllowlist(bound))  # type: ignore[arg-type]
        expected = _old_refusal(scope, bound)
        assert guard.refusal(scope) == expected, scope
        headers = _old_headers(scope)
        host = _old_split_host(headers.get("host", ""))
        assert web_guard.cross_site_refusal(scope) == _old_foreign_site(
            scope, headers, host
        ), scope
        refused += expected is not None
        served += expected is None
    # The inputs reach both outcomes.
    assert refused and served


@pytest.mark.parametrize(
    "bound", ["0", "0.0", "0.0.0", "::0", "0::", "0:0:0:0:0:0:0:0", "[::0]", "00.0.0.0"]
)
def test_every_spelling_of_the_unspecified_address_is_a_wildcard(bound: str) -> None:
    """``--host 0`` binds every interface as ``0.0.0.0`` does: it must check no
    Host either, not refuse every other machine while warning it is exposed
    (#2497)."""
    assert web_origin.is_wildcard(bound)
    assert web_origin.HostAllowlist(bound).allows("192.168.1.5")


@pytest.mark.parametrize(
    "bound",
    [
        "127.0.0.1",
        "::1",
        "10.0.0.1",
        "0.0.0.1",
        "localhost",
        # No IP literal: looked up as a host name, which may resolve to one
        # interface (a search domain, /etc/hosts).
        "0.0.0.0.0",
        ".",
        "..",
        ".0",
        "0..0",
        ":",
        ":::",
        "0:0",
        "00000::",
        "::%eth0",
        "0.",
        "0.0.0.0.",
    ],
)
def test_an_address_with_anything_but_zeros_is_no_wildcard(bound: str) -> None:
    assert not web_origin.is_wildcard(bound)
    assert not web_origin.HostAllowlist(bound).allows("192.168.1.5")
