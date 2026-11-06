"""The viewer and the editor refuse cross-site state changes (#1844).

Binding loopback keeps other machines out, not a page open in the user's own
browser: ``web_guard.RequestGuard`` refuses a state-changing request another
site sends (its ``Origin`` or ``Sec-Fetch-Site``) and a request under a
``Host`` the application is not served as (DNS rebinding), and the viewer's
JSON routes refuse a body that is not sent as ``application/json`` (the
``text/plain`` a page may send cross-origin without a preflight).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.authoring.persistence import DraftStore
from autoware_carla_scenario.editor.app import create_app
from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import runner
from autoware_carla_scenario.editor import scenario_map
from autoware_carla_scenario.editor import service as service_module
from autoware_carla_scenario.maps import cache as cache_module
from autoware_carla_scenario.web_guard import (
    HostAllowlist,
    RequestGuard,
    cross_site_refusal,
)

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "autoware_carla_scenario" / "docs"

#: The viewer's own origin under TestClient.
LOOPBACK = "http://127.0.0.1"
FOREIGN = "https://attacker.example"

#: The viewer's state-changing routes; none needs anything on disk to be
#: refused, since the guard answers before the route runs.
VIEWER_POSTS = [
    "/api/refresh",
    "/api/run",
    "/api/run/preview",
    "/api/recording/single/2026-10-19/12-00-00/0",
    "/api/slurm/2026-10-19/12-00-00/resubmit",
    "/api/slurm/2026-10-19/12-00-00/cancel",
]


@pytest.fixture
def started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """What ``runner.start_run`` was asked to run (nothing is run)."""
    seen: list[Any] = []

    def fake_start_run(overrides_list: list[list[str]], **_: Any) -> bool:
        seen.append(overrides_list)
        return True

    monkeypatch.setattr(runner, "start_run", fake_start_run)
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    return seen


@pytest.fixture
def viewer() -> TestClient:
    return TestClient(app_module.app, base_url=LOOPBACK)


# ── Content type ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["/api/run", "/api/run/preview"])
def test_a_text_plain_body_is_refused_with_415(
    started: list[Any], viewer: TestClient, path: str
) -> None:
    response = viewer.post(
        path,
        content='{"scenario": "a"}',
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 415
    assert started == []


@pytest.mark.parametrize("path", ["/api/run", "/api/run/preview"])
def test_a_body_with_no_content_type_is_refused_with_415(
    started: list[Any], viewer: TestClient, path: str
) -> None:
    response = viewer.post(path, content=b'{"scenario": "a"}')
    assert response.status_code == 415
    assert started == []


def test_json_with_a_charset_is_accepted(viewer: TestClient) -> None:
    response = viewer.post(
        "/api/run/preview",
        content='{"scenario": "a"}',
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    assert "scenario=a" in response.json()["command"]


# ── Cross-site requests ──────────────────────────────────────────────────


@pytest.mark.parametrize("path", VIEWER_POSTS)
@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": FOREIGN},
        {"Origin": "null"},
        # Another port on this machine is another origin.
        {"Origin": "http://127.0.0.1:8080"},
        {"Origin": "https://127.0.0.1"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Origin": LOOPBACK, "Sec-Fetch-Site": "cross-site"},
    ],
    ids=lambda h: ",".join(f"{k}={v}" for k, v in h.items()),
)
def test_a_cross_site_request_is_refused_with_403(
    started: list[Any], viewer: TestClient, path: str, headers: dict[str, str]
) -> None:
    response = viewer.post(path, json={"scenario": "a"}, headers=headers)
    assert response.status_code == 403
    assert started == []


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": LOOPBACK},
        {"Origin": LOOPBACK + ":80", "Sec-Fetch-Site": "same-origin"},
        {"Sec-Fetch-Site": "none"},
    ],
    ids=lambda h: ",".join(f"{k}={v}" for k, v in h.items()) or "no-origin",
)
def test_a_same_origin_json_request_still_runs(
    started: list[Any], viewer: TestClient, headers: dict[str, str]
) -> None:
    preview = viewer.post("/api/run/preview", json={"scenario": "a"}, headers=headers)
    assert preview.status_code == 200
    assert "scenario=a" in preview.json()["command"]
    run = viewer.post("/api/run", json={"scenario": "a"}, headers=headers)
    assert run.status_code == 200, run.text
    assert run.json() == {"status": "started"}
    assert started == [[["scenario=a"]]]


def test_a_get_with_a_foreign_origin_is_answered(viewer: TestClient) -> None:
    """Reading is not a state change; the Host check is what guards it."""
    response = viewer.get("/api/run/status", headers={"Origin": FOREIGN})
    assert response.status_code == 200


# ── Host check ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["get", "post"])
def test_a_foreign_host_is_refused_with_400(viewer: TestClient, method: str) -> None:
    response = viewer.request(
        method.upper(), "/api/run/status", headers={"Host": "evil.example"}
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.0.0.1:9000", "localhost:9000", "[::1]:9000", "[::1]"]
)
def test_a_loopback_host_is_answered(viewer: TestClient, host: str) -> None:
    response = viewer.get("/api/run/status", headers={"Host": host})
    assert response.status_code == 200


def test_the_viewer_answers_the_host_it_is_bound_to(
    viewer: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module.HOSTS, "bound_host", "192.168.1.5")
    assert viewer.get("/", headers={"Host": "192.168.1.5:9000"}).status_code == 200
    status = viewer.get("/api/run/status", headers={"Host": "evil.example"})
    assert status.status_code == 400


@pytest.mark.parametrize(
    ("bound", "name", "allowed"),
    [
        ("127.0.0.1", "localhost", True),
        ("127.0.0.1", "127.0.0.1", True),
        ("127.0.0.1", "::1", True),
        ("127.0.0.1", "LOCALHOST.", True),
        ("127.0.0.1", "evil.example", False),
        ("127.0.0.1", "testserver", False),
        ("127.0.0.1", "", False),
        ("myhost", "myhost", True),
        ("myhost", "MyHost", True),
        ("myhost", "other", False),
        ("::1", "::1", True),
        # Every interface: reached under any name, as asked.
        ("0.0.0.0", "evil.example", True),
        ("::", "anything", True),
    ],
)
def test_the_host_allowlist(bound: str, name: str, allowed: bool) -> None:
    assert HostAllowlist(bound).allows(name) is allowed


async def _unreached(scope: Any, receive: Any, send: Any) -> None:
    """The application behind a guard whose refusal() alone is tested."""
    raise AssertionError("not called")


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ([(b"host", b"evil.example")], (400, "Invalid Host header")),
        ([], (400, "Invalid Host header")),
        ([(b"host", b"127.0.0.1:x")], (400, "Invalid Host header")),
        ([(b"host", b"::1")], (400, "Invalid Host header")),
        ([(b"host", b"127.0.0.1:9000"), (b"origin", b"http://127.0.0.1:9000")], None),
        ([(b"host", b"[::1]:9000"), (b"origin", b"http://[::1]:9000")], None),
        (
            [(b"host", b"localhost:9000"), (b"origin", b"http://127.0.0.1:9000")],
            (403, "Cross-origin request refused"),
        ),
    ],
)
def test_the_guard_s_refusals(
    headers: list[tuple[bytes, bytes]], expected: tuple[int, str] | None
) -> None:
    guard = RequestGuard(_unreached, hosts=HostAllowlist())
    scope = {"type": "http", "method": "POST", "scheme": "http", "headers": headers}
    assert guard.refusal(scope) == expected


def test_bound_to_every_interface_the_origin_is_still_checked() -> None:
    guard = RequestGuard(_unreached, hosts=HostAllowlist("0.0.0.0"))
    base = {"type": "http", "method": "POST", "scheme": "http"}
    own = [(b"host", b"box.lan:9000"), (b"origin", b"http://box.lan:9000")]
    assert guard.refusal({**base, "headers": own}) is None
    foreign = [(b"host", b"box.lan:9000"), (b"origin", FOREIGN.encode())]
    assert guard.refusal({**base, "headers": foreign}) == (
        403,
        "Cross-origin request refused",
    )


# ── Editor ───────────────────────────────────────────────────────────────


@pytest.fixture
def editor(tmp_path: Path, at_repo_root: Path) -> TestClient:
    return TestClient(
        create_app(draft_dir=tmp_path / "drafts", export_dir=tmp_path / "packages"),
        base_url=LOOPBACK,
    )


@pytest.mark.parametrize(
    "headers",
    [{"Origin": FOREIGN}, {"Sec-Fetch-Site": "cross-site"}],
    ids=["origin", "sec-fetch-site"],
)
def test_an_editor_post_from_another_site_creates_no_draft(
    editor: TestClient, tmp_path: Path, headers: dict[str, str]
) -> None:
    response = editor.post(
        "/new",
        data={"kind": "cut_in", "title": "Cut in"},
        headers=headers,
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert DraftStore(tmp_path / "drafts").list() == []


def test_a_same_origin_editor_form_post_still_redirects(
    editor: TestClient, tmp_path: Path
) -> None:
    response = editor.post(
        "/new",
        data={"kind": "cut_in", "title": "Cut in"},
        headers={"Origin": LOOPBACK, "Sec-Fetch-Site": "same-origin"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert len(DraftStore(tmp_path / "drafts").list()) == 1


def test_the_editor_refuses_a_foreign_host(editor: TestClient) -> None:
    assert editor.get("/", headers={"Host": "evil.example"}).status_code == 400


def test_the_editor_answers_the_host_it_is_bound_to(
    tmp_path: Path, at_repo_root: Path
) -> None:
    client = TestClient(
        create_app(draft_dir=tmp_path / "drafts", host="box.lan"),
        base_url="http://box.lan:9100",
    )
    assert client.get("/").status_code == 200


# ── GET routes that do more than read (#1850) ────────────────────────────

#: Headers a page on another site sends (or may send).
FOREIGN_HEADERS = [
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
    {"Origin": FOREIGN},
    {"Origin": "null"},
    {"Origin": LOOPBACK, "Sec-Fetch-Site": "cross-site"},
]

#: Headers the editor's own pages, a typed URL or bookmark, and curl send.
OWN_HEADERS = [
    {},
    {"Sec-Fetch-Site": "same-origin"},
    {"Sec-Fetch-Site": "none"},
    {"Origin": LOOPBACK, "Sec-Fetch-Site": "same-origin"},
]


def _ids(headers: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in headers.items()) or "no-headers"


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every git command the map cache tries to run (none is run)."""
    ran: list[Any] = []

    def _record(args: Any, **_: Any) -> Any:
        ran.append(args)
        raise AssertionError("git must not run")

    monkeypatch.setattr(cache_module.subprocess, "run", _record)
    return ran


@pytest.fixture
def listed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The repositories the map library was asked to list (none is read)."""
    seen: list[str] = []

    def _list_maps(uri: str, *, refresh: bool = False) -> list[Any]:
        seen.append(uri)
        return []

    monkeypatch.setattr(service_module, "list_maps", _list_maps)
    return seen


REPO = "git+https://github.com/example/maps@main"


@pytest.mark.parametrize("headers", FOREIGN_HEADERS, ids=_ids)
def test_another_site_cannot_make_the_map_library_run_git(
    editor: TestClient, no_git: list[Any], headers: dict[str, str]
) -> None:
    response = editor.get("/maps", params={"repo": REPO}, headers=headers)
    assert response.status_code == 403
    assert no_git == []


@pytest.mark.parametrize("headers", OWN_HEADERS, ids=_ids)
def test_the_map_library_still_lists_a_repository_for_this_site(
    editor: TestClient, listed: list[str], headers: dict[str, str]
) -> None:
    response = editor.get("/maps", params={"repo": REPO}, headers=headers)
    assert response.status_code == 200
    assert listed == [REPO]


def _draft(editor: TestClient) -> str:
    response = editor.post(
        "/new", data={"kind": "cut_in", "title": "Cut in"}, follow_redirects=False
    )
    return str(response.headers["location"]).rsplit("/", 1)[-1]


@pytest.fixture
def map_views(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """The ``load_map`` of every map view built (each built as it was)."""
    built: list[bool] = []
    real = scenario_map.build_map_view

    def _build(document: Any, *, pattern: int = 0, load_map: bool = False) -> Any:
        built.append(load_map)
        return real(document, pattern=pattern, load_map=False)

    monkeypatch.setattr(scenario_map, "build_map_view", _build)
    return built


@pytest.mark.parametrize("headers", FOREIGN_HEADERS, ids=_ids)
def test_another_site_cannot_make_the_map_panel_load_a_map(
    editor: TestClient, map_views: list[bool], headers: dict[str, str]
) -> None:
    draft_id = _draft(editor)
    response = editor.get(f"/draft/{draft_id}/map-view?load_map=1", headers=headers)
    assert response.status_code == 403
    assert map_views == []


@pytest.mark.parametrize("headers", OWN_HEADERS, ids=_ids)
def test_the_map_panel_still_loads_the_map_for_this_site(
    editor: TestClient, map_views: list[bool], headers: dict[str, str]
) -> None:
    draft_id = _draft(editor)
    response = editor.get(f"/draft/{draft_id}/map-view?load_map=1", headers=headers)
    assert response.status_code == 200
    assert map_views == [True]


def test_a_page_that_only_reads_is_served_to_another_site(editor: TestClient) -> None:
    """The guard covers the routes that fetch or parse, not every GET."""
    draft_id = _draft(editor)
    for path in ("/", f"/draft/{draft_id}", f"/draft/{draft_id}/yaml"):
        response = editor.get(path, headers={"Sec-Fetch-Site": "cross-site"})
        assert response.status_code == 200, path


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ([(b"host", b"127.0.0.1:9000")], None),
        ([(b"host", b"127.0.0.1:9000"), (b"sec-fetch-site", b"none")], None),
        ([(b"host", b"127.0.0.1:9000"), (b"sec-fetch-site", b"same-origin")], None),
        (
            [(b"host", b"127.0.0.1:9000"), (b"sec-fetch-site", b"cross-site")],
            (403, "Cross-site request refused"),
        ),
        (
            [(b"host", b"127.0.0.1:9000"), (b"origin", b"http://127.0.0.1:8080")],
            (403, "Cross-origin request refused"),
        ),
        # No Host to compare with: no Origin is this request's own.
        (
            [(b"origin", b"http://127.0.0.1:9000")],
            (403, "Cross-origin request refused"),
        ),
    ],
)
def test_the_cross_site_refusal_of_a_get(
    headers: list[tuple[bytes, bytes]], expected: tuple[int, str] | None
) -> None:
    scope = {"type": "http", "method": "GET", "scheme": "http", "headers": headers}
    assert cross_site_refusal(scope) == expected


# ── Docs ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("page", ["usage.md", "scenario_editor.md"])
def test_the_loopback_warning_describes_the_guard(page: str) -> None:
    text = (DOCS / page).read_text(encoding="utf-8")
    assert re.search(r"Origin", text) and "Sec-Fetch-Site" in text
    assert "DNS rebinding" in text
    assert "Sec-Fetch-Site: none" in text


def test_the_editor_docs_name_the_guarded_get_routes() -> None:
    text = (DOCS / "scenario_editor.md").read_text(encoding="utf-8")
    assert "/maps?repo=" in text and "map-view" in text


def test_the_viewer_docs_name_the_guarded_get_route() -> None:
    text = (DOCS / "usage.md").read_text(encoding="utf-8")
    assert ".rrd`) is converted" in text
