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
from autoware_carla_scenario.web_guard import HostAllowlist, RequestGuard

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


# ── Docs ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("page", ["usage.md", "scenario_editor.md"])
def test_the_loopback_warning_describes_the_guard(page: str) -> None:
    text = (DOCS / page).read_text(encoding="utf-8")
    assert re.search(r"Origin", text) and "Sec-Fetch-Site" in text
    assert "DNS rebinding" in text
