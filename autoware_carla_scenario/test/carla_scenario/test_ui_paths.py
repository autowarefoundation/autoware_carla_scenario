"""The viewer reads no file outside its base, whatever the URL names (#677).

Every route builds the session's directory through ``cases.session_dir``,
which resolves the path and checks it stays under ``multirun/`` or
``outputs/``; a name that does not is a 404.  ``%2E%2E`` segments reach the
app as ``..`` (the test client does not normalise them away).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import cases, scanner

SECRET = "TOP-SECRET-LOG-CONTENT"


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A viewer base inside a directory holding files it must not show:
    ``outputs/../..`` and ``multirun/../..`` are *tmp_path*/secret_dir."""
    outside = tmp_path / "secret_dir"
    outside.mkdir()
    (outside / "run.log").write_text(SECRET + "\n")
    (outside / "leak_result.json").write_text(
        json.dumps({"scenario": SECRET, "passed": True, "message": SECRET})
    )
    job = outside / "0"
    job.mkdir()
    (job / "run.log").write_text(SECRET + "\n")
    (job / "leak_result.json").write_text(
        json.dumps({"scenario": SECRET, "passed": True, "message": SECRET})
    )
    base = outside / "viewer_base"
    (base / "outputs").mkdir(parents=True)
    (base / "multirun").mkdir()
    monkeypatch.setattr(app_module, "BASE_PATH", base)
    scanner.clear_cache()
    yield base
    scanner.clear_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


_TRAVERSAL = [
    "/session/single/%2E%2E/%2E%2E/0",
    "/session/single/%2E%2E/%2E%2E",
    "/session/batch/%2E%2E/%2E%2E/0",
    "/session/batch/%2E%2E/%2E%2E",
    "/session/multirun/%2E%2E/%2E%2E/0",
    "/session/multirun/%2E%2E/%2E%2E",
    "/session/single/%2E/%2E%2E/0",
    "/session/single/%2E%2E/%2E%2E/odd",
    "/session/unknown/2026-01-01/00-00-00",
    "/session/unknown/2026-01-01/00-00-00/0",
    "/recording/single/%2E%2E/%2E%2E/0.rrd",
    "/api/slurm/%2E%2E/%2E%2E/status",
]


@pytest.mark.parametrize("url", _TRAVERSAL)
def test_a_name_outside_the_base_is_a_404(
    base: Path, client: TestClient, url: str
) -> None:
    response = client.get(url)
    assert response.status_code == 404, response.text[:200]
    assert SECRET not in response.text


_LONG = "x" * 5000


@pytest.mark.parametrize(
    "url",
    [
        f"/session/single/{_LONG}/{_LONG}",
        f"/session/single/2026-01-01/{_LONG}/0",
        f"/session/multirun/2026-01-01/{_LONG}/odd",
        f"/session/batch/{_LONG}/00-00-00",
        f"/api/slurm/{_LONG}/{_LONG}/status",
    ],
)
def test_a_name_too_long_for_a_file_is_a_404(
    base: Path, client: TestClient, url: str
) -> None:
    # Not a 500 from stat()'s ENAMETOOLONG.
    assert client.get(url).status_code == 404


@pytest.mark.parametrize(
    "url",
    [
        f"/compare?s=single/{_LONG}/{_LONG}&s=batch/{_LONG}/{_LONG}",
        f"/compare/runs?run=single/{_LONG}/{_LONG}/0",
        # "²".isdigit() is true, but it is no index.
        "/compare/runs?run=single/2026-01-01/00-00-00/%C2%B2",
    ],
)
def test_a_comparison_ignores_a_ref_it_cannot_read(
    base: Path, client: TestClient, url: str
) -> None:
    assert client.get(url).status_code == 200


def test_making_a_recording_outside_the_base_is_a_404(
    base: Path, client: TestClient
) -> None:
    response = client.post("/api/recording/single/%2E%2E/%2E%2E/0")
    assert response.status_code == 404
    assert SECRET not in response.text


def _symlinked_session(base: Path, root: str) -> str:
    """``<root>/2026-01-01/escape``, a symlink to the secret directory (or,
    under multirun/, to a directory whose job 0 is secret)."""
    date = base / root / "2026-01-01"
    date.mkdir()
    (date / "escape").symlink_to(base.parent, target_is_directory=True)
    return "2026-01-01/escape"


@pytest.mark.parametrize(
    ("session_type", "root"),
    [("single", "outputs"), ("batch", "outputs"), ("multirun", "multirun")],
)
def test_a_session_symlinked_out_of_the_base_is_a_404(
    base: Path, client: TestClient, session_type: str, root: str
) -> None:
    ref = _symlinked_session(base, root)
    for url in (f"/session/{session_type}/{ref}", f"/session/{session_type}/{ref}/0"):
        response = client.get(url)
        assert response.status_code == 404, url
        assert SECRET not in response.text
    # Nor is it listed.
    assert SECRET not in client.get("/").text


def test_a_multirun_job_symlinked_out_of_the_base_is_not_read(base: Path) -> None:
    session = base / "multirun" / "2026-01-01" / "00-00-00"
    session.mkdir(parents=True)
    (session / "0").symlink_to(base.parent / "0", target_is_directory=True)
    assert cases.job_dir(base, "2026-01-01", "00-00-00", 0) is None
    assert scanner.load_session(base, "multirun", "2026-01-01", "00-00-00") == []
    assert scanner.load_scenario(base, "multirun", "2026-01-01", "00-00-00", 0) is None
    assert cases.case_files(base, "multirun", "2026-01-01", "00-00-00", 0) is None


@pytest.mark.parametrize("session_type", ["single", "batch", "multirun"])
def test_the_scanner_refuses_what_session_dir_refuses(
    base: Path, session_type: str
) -> None:
    assert cases.session_dir(base, session_type, "..", "..") is None
    assert scanner.load_session(base, session_type, "..", "..") == []
    assert scanner.load_scenario(base, session_type, "..", "..", 0) is None


def test_contained_keeps_paths_under_the_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    assert cases.contained(root, "a") == root / "a"
    assert cases.contained(root, "missing", "b") == root / "missing" / "b"
    assert cases.contained(root, "..") is None
    assert cases.contained(root, "a", "..", "..", "x") is None
    assert cases.contained(root, "/etc") is None


def test_a_session_under_the_base_is_still_shown(
    base: Path, client: TestClient
) -> None:
    run = base / "outputs" / "2026-01-01" / "00-00-00"
    run.mkdir(parents=True)
    (run / "run.log").write_text("THE-RUN-LOG\n")
    (run / "Cut_result.json").write_text(
        json.dumps({"scenario": "cut_in", "passed": True, "message": ""})
    )
    assert client.get("/session/single/2026-01-01/00-00-00").status_code == 200
    page = client.get("/session/single/2026-01-01/00-00-00/0")
    assert page.status_code == 200
    assert "THE-RUN-LOG" in page.text
    assert "2026-01-01" in client.get("/").text
