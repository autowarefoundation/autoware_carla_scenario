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
from autoware_carla_scenario.ui import case_paths, cases, recordings, results, scanner

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
    # No result of its own to read: pending, as the session pages count it
    # (#2543), not an error.
    assert case.passed is None and case.pending and not case.errored
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
    # No job 1 (``01`` is none): no such case (#2344).
    assert client.get(f"/session/multirun/{DATE}/10-00-00/1").status_code == 404


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


# ── Session metadata that is a symbolic link (#2543) ────────────────────────


@pytest.fixture
def secrets(tmp_path: Path) -> Path:
    """Files outside the viewer's base (``tmp_path/base``) that a session's
    metadata could link to: overrides, a batch summary, a SLURM manifest and
    a case outcome, each naming SECRET."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "overrides.yaml").write_text("- SECRET_OVERRIDE=1\n")
    (outside / "multirun.yaml").write_text(
        "hydra:\n  overrides:\n    task:\n    - SECRET_TASK=1\n"
    )
    (outside / "batch_results.json").write_text(
        json.dumps(
            [
                {
                    "scenario": "SECRET_SCENARIO",
                    "passed": True,
                    "message": "SECRET_MESSAGE",
                    "overrides": ["SECRET_OVERRIDE=1"],
                }
            ]
        )
    )
    (outside / "manifest.json").write_text(
        json.dumps({"version": 2, "cases": [{"index": 0, "label": "SECRET_SCENARIO"}]})
    )
    (outside / "0.json").write_text(
        json.dumps({"result": {"passed": True, "message": "SECRET_MESSAGE"}})
    )
    return outside


@pytest.fixture
def linked_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "base"
    root.mkdir()
    monkeypatch.setattr(app_module, "BASE_PATH", root)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    scanner.clear_cache()
    yield root
    scanner.clear_cache()


def _single(root: Path, time: str) -> Path:
    run = root / "outputs" / DATE / time
    run.mkdir(parents=True)
    (run / "Y_result.json").write_text(json.dumps({"passed": True, "message": "ok"}))
    return run


def test_a_symlinked_overrides_file_is_not_read(
    linked_base: Path, secrets: Path, client: TestClient
) -> None:
    job = linked_base / "multirun" / DATE / "10-00-00" / "0"
    (job / ".hydra").mkdir(parents=True)
    (job / ".hydra" / "overrides.yaml").symlink_to(secrets / "overrides.yaml")
    (job / "X_result.json").write_text(json.dumps({"passed": True}))
    assert cases._hydra_overrides(job) == []
    for path in (
        f"/session/multirun/{DATE}/10-00-00",
        f"/session/multirun/{DATE}/10-00-00/0",
    ):
        page = client.get(path)
        assert page.status_code == 200 and "SECRET" not in page.text


def test_a_symlinked_hydra_directory_is_not_read(
    linked_base: Path, secrets: Path, client: TestClient
) -> None:
    run = _single(linked_base, "12-00-00")
    hydra = secrets / "hydra"
    hydra.mkdir()
    # A real file in a linked directory: the link is the directory.
    (hydra / "overrides.yaml").write_text("- SECRET_OVERRIDE=1\n")
    (run / ".hydra").symlink_to(hydra)
    assert cases._hydra_overrides(run) == []
    page = client.get(f"/session/single/{DATE}/12-00-00/0")
    assert page.status_code == 200 and "SECRET" not in page.text


def test_a_symlinked_multirun_file_is_not_read(
    linked_base: Path, secrets: Path
) -> None:
    session = linked_base / "multirun" / DATE / "10-00-00"
    session.mkdir(parents=True)
    (session / "multirun.yaml").symlink_to(secrets / "multirun.yaml")
    assert cases._multirun_task(session) is None
    (session / "multirun.yaml").unlink()
    (session / "multirun.yaml").write_text((secrets / "multirun.yaml").read_text())
    assert cases._multirun_task(session) == ["SECRET_TASK=1"]


def test_a_symlinked_batch_summary_is_not_read(
    linked_base: Path, secrets: Path, client: TestClient
) -> None:
    batch = linked_base / "outputs" / DATE / "11-00-00"
    batch.mkdir(parents=True)
    (batch / "batch_results.json").symlink_to(secrets / "batch_results.json")
    assert cases.batch_entries(batch) is None
    assert not cases.is_batch(batch)
    assert "SECRET" not in client.get("/").text
    for path in (
        f"/session/batch/{DATE}/11-00-00",
        f"/session/batch/{DATE}/11-00-00/0",
    ):
        response = client.get(path)
        assert response.status_code == 404 and "SECRET" not in response.text


@pytest.mark.parametrize("linked", ["manifest", "slurm", "outcome", "results"])
def test_symlinked_slurm_records_are_not_read(
    linked_base: Path, secrets: Path, client: TestClient, linked: str
) -> None:
    batch = linked_base / "outputs" / DATE / "11-00-00"
    batch.mkdir(parents=True)
    slurm = secrets / "slurm"
    (slurm / "results").mkdir(parents=True)
    (slurm / "manifest.json").write_text((secrets / "manifest.json").read_text())
    (slurm / "results" / "0.json").write_text((secrets / "0.json").read_text())
    if linked == "slurm":
        (batch / "slurm").symlink_to(slurm)
    else:
        (batch / "slurm" / "results").mkdir(parents=True)
        manifest = batch / "slurm" / "manifest.json"
        if linked == "manifest":
            manifest.symlink_to(slurm / "manifest.json")
        else:
            manifest.write_text(json.dumps({"cases": [{"index": 0, "label": "a"}]}))
        if linked == "outcome":
            (batch / "slurm" / "results" / "0.json").symlink_to(
                slurm / "results" / "0.json"
            )
        elif linked == "results":
            (batch / "slurm" / "results").rmdir()
            (batch / "slurm" / "results").symlink_to(slurm / "results")
    found = cases.batch_entries(batch)
    if linked in ("manifest", "slurm"):
        # No manifest of its own: no batch.
        assert found is None
    else:
        # Its own manifest, but the outcome is not read: pending.
        assert found is not None
        [entry] = found[0]
        assert entry["pending"] is True and "SECRET" not in json.dumps(entry)
    assert "SECRET" not in client.get("/").text
    assert "SECRET" not in client.get(f"/session/batch/{DATE}/11-00-00").text
    if linked in ("manifest", "slurm"):
        # Nor is the batch's SLURM panel, or an action on it, a batch's.
        status = client.get(f"/api/slurm/{DATE}/11-00-00/status")
        assert status.status_code == 404 and "SECRET" not in status.text
        action = client.post(f"/api/slurm/{DATE}/11-00-00/collect")
        assert action.status_code == 404


# ── single and batch URLs name their own sessions (#2543) ───────────────────


def _batch(root: Path, time: str) -> Path:
    batch = root / "outputs" / DATE / time
    batch.mkdir(parents=True)
    (batch / "CutIn_result.json").write_text(
        json.dumps({"passed": True, "message": "ok", "elapsed_seconds": 1.0})
    )
    (batch / "batch_results.json").write_text(
        json.dumps(
            [
                {
                    "scenario": "CutIn",
                    "passed": True,
                    "message": "ok",
                    "elapsed_seconds": 1.0,
                    "overrides": [],
                }
            ]
        )
    )
    return batch


@pytest.mark.parametrize(
    "path",
    [
        f"/session/single/{DATE}/10-00-00",
        f"/session/single/{DATE}/10-00-00/0",
        f"/session/single/{DATE}/10-00-00/odd",
        f"/recording/single/{DATE}/10-00-00/0.rrd",
        f"/session/batch/{DATE}/11-00-00",
        f"/session/batch/{DATE}/11-00-00/0",
        f"/session/batch/{DATE}/11-00-00/odd",
    ],
)
def test_a_url_of_the_other_outputs_type_is_a_404(
    base: Path, client: TestClient, path: str
) -> None:
    _batch(base, "10-00-00")
    _single(base, "11-00-00")
    assert client.get(path).status_code == 404


def test_each_outputs_session_has_its_own_types_page(
    base: Path, client: TestClient
) -> None:
    batch = _batch(base, "10-00-00")
    single = _single(base, "11-00-00")
    assert cases.session_type_matches(batch, "batch")
    assert not cases.session_type_matches(batch, "single")
    assert cases.session_type_matches(single, "single")
    assert not cases.session_type_matches(single, "batch")
    assert cases.case_files(base, "single", DATE, "10-00-00", 0) is None
    assert cases.case_files(base, "single", DATE, "11-00-00", 0) is not None
    for path in (
        f"/session/batch/{DATE}/10-00-00",
        f"/session/batch/{DATE}/10-00-00/0",
        f"/session/single/{DATE}/11-00-00",
        f"/session/single/{DATE}/11-00-00/0",
    ):
        assert client.get(path).status_code == 200, path


def test_a_compare_ref_of_the_other_outputs_type_is_skipped(
    base: Path, client: TestClient
) -> None:
    _batch(base, "10-00-00")
    _single(base, "11-00-00")
    assert app_module._parse_ref(f"batch/{DATE}/10-00-00") is not None
    assert app_module._parse_ref(f"single/{DATE}/11-00-00") is not None
    assert app_module._parse_ref(f"single/{DATE}/10-00-00") is None
    assert app_module._parse_ref(f"batch/{DATE}/11-00-00") is None
    page = client.get(
        "/compare",
        params=[("s", f"single/{DATE}/10-00-00"), ("s", f"batch/{DATE}/11-00-00")],
    )
    assert page.status_code == 200
    assert f"single/{DATE}/10-00-00 (baseline)" not in page.text
