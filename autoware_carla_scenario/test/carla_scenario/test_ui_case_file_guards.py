"""Case files and sessions the viewer does not read (#2239).

* A case file that is a symbolic link is neither read nor served: it may name
  a file outside the viewer's base.
* A job directory is named by its index alone (``01`` is no job).
* A session that does not exist has no page (404).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import case_paths, recordings, results, scanner

DATE = "2026-11-20"


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    scanner.clear_cache()
    yield tmp_path
    scanner.clear_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app, base_url="http://127.0.0.1")


def _job(base: Path, time: str, index: int, overrides: list[str], passed: bool) -> Path:
    job = base / "multirun" / DATE / time / str(index)
    (job / ".hydra").mkdir(parents=True)
    (job / ".hydra" / "overrides.yaml").write_text(
        "".join(f"- {o}\n" for o in overrides)
    )
    (job / "X_result.json").write_text(
        json.dumps({"passed": passed, "elapsed_seconds": 1})
    )
    return job


# ── Case files that are symbolic links ──────────────────────────────────────


def _trajectory(path: Path) -> None:
    path.write_text(
        json.dumps({"type": "actor", "id": 1, "role": "ego", "extent": [2, 1]}) + "\n"
    )


def test_a_symlinked_recording_is_not_served(
    base: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = base / "secret.rrd"
    secret.write_bytes(b"secret")
    job = _job(base, "10-00-00", 0, ["scenario=a/b"], True)
    _trajectory(job / "X.trajectory.jsonl")
    (job / "X.rrd").symlink_to(secret)
    # The link is no recording: one is made in the cache instead.
    monkeypatch.setattr(recordings, "_converter", lambda: None)
    response = client.get(f"/recording/multirun/{DATE}/10-00-00/0.rrd")
    assert response.status_code == 503
    assert b"secret" not in response.content


def test_a_symlinked_trajectory_is_no_case_log(base: Path, client: TestClient) -> None:
    secret = base / "secret.jsonl"
    _trajectory(secret)
    job = _job(base, "10-00-00", 0, ["scenario=a/b"], True)
    (job / "X.trajectory.jsonl").symlink_to(secret)
    assert client.get(f"/recording/multirun/{DATE}/10-00-00/0.rrd").status_code == 404
    page = client.get(f"/session/multirun/{DATE}/10-00-00/0")
    assert page.status_code == 200
    assert str(secret) not in page.text


def test_symlinked_case_files_are_not_read(base: Path) -> None:
    outside = base / "elsewhere"
    outside.mkdir()
    (outside / "result.json").write_text(json.dumps({"passed": True}))
    (outside / "coverage.json").write_text(json.dumps({"case": {"values": {"v": 1}}}))
    job = base / "multirun" / DATE / "10-00-00" / "0"
    job.mkdir(parents=True)
    (job / "X_result.json").symlink_to(outside / "result.json")
    assert case_paths.result_path(job) is None
    assert case_paths._stems_in(job) == []
    _trajectory(job / "X.trajectory.jsonl")
    (job / "X_coverage.json").symlink_to(outside / "coverage.json")
    [case] = results.load_results(base, "multirun", DATE, "10-00-00", metrics=False)
    assert case.files is not None and case.files.stem == "X"
    assert case.passed is None and case.errored
    assert case.coverage is None


def test_own_file_refuses_a_link_and_a_missing_file(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.write_text("x")
    (tmp_path / "link").symlink_to(real)
    assert case_paths.own_file(real)
    assert not case_paths.own_file(tmp_path / "link")
    assert not case_paths.own_file(tmp_path / "missing")
    assert not case_paths.own_file(tmp_path)


# ── The job-directory rule ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "index"),
    [
        ("0", 0),
        ("7", 7),
        ("10", 10),
        ("01", None),
        ("00", None),
        ("", None),
        ("²", None),
        ("1a", None),
        ("-1", None),
        ("9" * 19, None),
    ],
)
def test_a_job_directory_is_named_by_its_index(name: str, index: int | None) -> None:
    assert case_paths.job_index(name) == index


def test_a_zero_padded_directory_is_no_job(base: Path, client: TestClient) -> None:
    _job(base, "10-00-00", 0, ["scenario=a/b"], True)
    padded = base / "multirun" / DATE / "10-00-00" / "01"
    padded.mkdir()
    (padded / "X_result.json").write_text(json.dumps({"passed": False}))
    [summary] = scanner.scan_sessions(base)
    assert summary.total_count == 1
    items = scanner.load_session(base, "multirun", DATE, "10-00-00")
    assert [it.index for it in items] == [0]
    found = results.load_results(base, "multirun", DATE, "10-00-00", metrics=False)
    assert [c.index for c in found] == [0]
    assert client.get(f"/session/multirun/{DATE}/10-00-00/1").status_code == 200


# ── Sessions that do not exist ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        f"/session/multirun/{DATE}/09-00-00",
        f"/session/multirun/{DATE}/09-00-00/0",
        f"/session/multirun/{DATE}/09-00-00/odd",
        f"/session/single/{DATE}/09-00-00",
        "/session/batch/2026-01-01/09-00-00",
    ],
)
def test_a_session_that_does_not_exist_is_a_404(
    base: Path, client: TestClient, path: str
) -> None:
    _job(base, "10-00-00", 0, ["scenario=a/b"], True)
    assert client.get(path).status_code == 404


def test_a_session_that_exists_has_its_page(base: Path, client: TestClient) -> None:
    _job(base, "10-00-00", 0, ["scenario=a/b"], True)
    assert client.get(f"/session/multirun/{DATE}/10-00-00").status_code == 200
    assert client.get(f"/session/multirun/{DATE}/10-00-00/0").status_code == 200
