"""The viewer backend's audit (#2239).

* A run, preview or resubmit body that is not JSON is a 400, and a run body
  that names no scenario too: a truncated request never runs ``*/*``.
* Multirun and single sessions match their cases by the sweep's overrides,
  not the session's command line, so two sessions that differ only there are
  compared case by case.
* A case file that is a symbolic link is neither read nor served.
* A job directory is named by its index alone (``01`` is no job).
* A session that does not exist has no page (404).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import (
    case_paths,
    recordings,
    results,
    runner,
    scanner,
    slurm_launch,
)

DATE = "2026-11-20"
_JSON = {"Content-Type": "application/json"}


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


# ── Bodies that are not JSON ────────────────────────────────────────────────


@pytest.fixture
def started(base: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[list[str]]]:
    runs: list[list[list[str]]] = []

    def fake_start_run(overrides_list: list[list[str]], **_: Any) -> bool:
        runs.append(overrides_list)
        return True

    monkeypatch.setattr(runner, "start_run", fake_start_run)
    monkeypatch.setattr(
        app_module, "_resolve_scenario_names", lambda pattern: [pattern]
    )
    return runs


_GARBLED = [
    b'{"scenario": "intersection_passing/left_turn", "extra_overrides": [',
    b'{"scenario": "a/b"',
    b"not json",
    b'{"scenario": "\xff"}',
    b"[" * 100_000,
]


@pytest.mark.parametrize("body", _GARBLED)
def test_a_run_body_that_is_not_json_is_a_400(
    started: list[Any], client: TestClient, body: bytes
) -> None:
    response = client.post("/api/run", content=body, headers=_JSON)
    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "message": "the request body is not valid JSON",
    }
    assert started == []


@pytest.mark.parametrize(
    ("body", "headers"),
    [(b"", _JSON), (b"", {}), (b"  \n", _JSON), (b"{}", _JSON)],
    ids=["empty-json", "empty-untyped", "blank", "no-scenario"],
)
def test_a_run_body_that_names_no_scenario_is_a_400(
    started: list[Any], client: TestClient, body: bytes, headers: dict[str, str]
) -> None:
    response = client.post("/api/run", content=body, headers=headers)
    assert response.status_code == 400
    assert response.json()["status"] == "error"
    assert "scenario is required" in response.json()["message"]
    assert started == []


def test_every_scenario_runs_when_asked_for_by_name(
    started: list[Any], client: TestClient
) -> None:
    # What the Run panel sends: the scenario is always named.
    response = client.post(
        "/api/run",
        json={
            "scenario": "*/*",
            "extra_overrides": [],
            "sweeper": "",
            "launcher": "local",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    assert started == [[["scenario=*/*"]]]


@pytest.mark.parametrize("body", _GARBLED)
def test_a_preview_body_that_is_not_json_is_a_400(
    base: Path, client: TestClient, body: bytes
) -> None:
    response = client.post("/api/run/preview", content=body, headers=_JSON)
    assert response.status_code == 400
    assert response.json() == {"command": "# the request body is not valid JSON"}


def test_an_empty_preview_body_keeps_the_defaults(
    base: Path, client: TestClient
) -> None:
    response = client.post("/api/run/preview", content=b"", headers=_JSON)
    assert response.status_code == 200
    assert "'scenario=*/*'" in response.json()["command"]


def _slurm_batch(base: Path) -> Path:
    batch = base / "outputs" / DATE / "12-00-00"
    (batch / "slurm" / "results").mkdir(parents=True)
    (batch / "slurm" / "submissions").mkdir()
    (batch / "slurm" / "manifest.json").write_text(
        json.dumps({"cases": [{"index": 0, "label": "a"}]})
    )
    return batch


@pytest.fixture
def resubmitted(base: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    _slurm_batch(base)
    ran: list[list[str]] = []
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))

    def run_cli(args: list[str]) -> tuple[int, str]:
        ran.append(args)
        return 0, ""

    monkeypatch.setattr(slurm_launch, "run_cli", run_cli)
    return ran


def test_a_garbled_resubmit_body_is_a_400(
    resubmitted: list[list[str]], client: TestClient
) -> None:
    response = client.post(
        f"/api/slurm/{DATE}/12-00-00/resubmit",
        content=b'{"failed": true, "array_parallelism": 2',
        headers=_JSON,
    )
    assert response.status_code == 400
    assert response.json() == {
        "ok": False,
        "output": "the request body is not valid JSON",
    }
    # Not resubmitted without its flags.
    assert resubmitted == []


def test_an_empty_resubmit_body_resubmits_with_no_flags(
    resubmitted: list[list[str]], client: TestClient
) -> None:
    response = client.post(
        f"/api/slurm/{DATE}/12-00-00/resubmit", content=b"", headers=_JSON
    )
    assert response.status_code == 200
    assert len(resubmitted) == 1
    assert "--failed" not in resubmitted[0]


# ── Cases matched by the sweep's overrides ──────────────────────────────────


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


def _changes(base: Path, *times: str) -> list[tuple[str, str]]:
    sessions = [
        (time, results.load_results(base, "multirun", DATE, time, metrics=False))
        for time in times
    ]
    return sorted((m.key, m.change()) for m in results.compare(sessions).cases)


def test_sessions_differing_in_a_session_override_match_case_by_case(
    base: Path,
) -> None:
    # As the Run panel groups them: its extra override is given to every job.
    _job(base, "10-00-00", 0, ["scenario=a/b", "ego.speed=10"], True)
    _job(base, "10-00-00", 1, ["scenario=a/b", "ego.speed=20"], True)
    _job(
        base,
        "11-00-00",
        0,
        ["scenario=a/b", "ego.speed=10", "autoware.planner=new"],
        False,
    )
    _job(
        base,
        "11-00-00",
        1,
        ["scenario=a/b", "ego.speed=20", "autoware.planner=new"],
        True,
    )
    assert _changes(base, "10-00-00", "11-00-00") == [
        ("a/b|ego.speed=10", "regression"),
        ("a/b|ego.speed=20", "same"),
    ]


def test_a_cli_multiruns_command_line_is_read_from_multirun_yaml(
    base: Path,
) -> None:
    # A one-job session: its job's overrides are all shared, but Hydra's
    # record says which of them the sweep set.
    for time, planner, passed in (
        ("10-00-00", "old", True),
        ("11-00-00", "new", False),
    ):
        _job(
            base,
            time,
            0,
            ["scenario=a/b", "ego.speed=10", f"autoware.planner={planner}"],
            passed,
        )
        (base / "multirun" / DATE / time / "multirun.yaml").write_text(
            "hydra:\n  overrides:\n    task:\n"
            "    - scenario=a/b\n    - ego.speed=10\n"
            f"    - autoware.planner={planner}\n"
        )
    # ego.speed=10 is a sweep of one value, given as a fixed override: both
    # are the command line, and the case is matched by its scenario.
    assert _changes(base, "10-00-00", "11-00-00") == [("a/b", "regression")]
    (base / "multirun" / DATE / "11-00-00" / "multirun.yaml").write_text(
        "hydra:\n  overrides:\n    task:\n"
        "    - scenario=a/b\n    - ego.speed=10,20\n    - autoware.planner=new\n"
    )
    (base / "multirun" / DATE / "10-00-00" / "multirun.yaml").write_text(
        "hydra:\n  overrides:\n    task:\n"
        "    - scenario=a/b\n    - ego.speed=10,20\n    - autoware.planner=old\n"
    )
    assert _changes(base, "10-00-00", "11-00-00") == [
        ("a/b|ego.speed=10", "regression")
    ]


def test_an_unreadable_multirun_yaml_falls_back_to_the_shared_overrides(
    base: Path,
) -> None:
    _job(base, "10-00-00", 0, ["scenario=a/b", "x=1", "y=1"], True)
    _job(base, "10-00-00", 1, ["scenario=a/b", "x=2", "y=1"], True)
    (base / "multirun" / DATE / "10-00-00" / "multirun.yaml").write_text("[: not yaml")
    found = results.load_results(base, "multirun", DATE, "10-00-00", metrics=False)
    assert [c.overrides for c in found] == [["x=1"], ["x=2"]]


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
