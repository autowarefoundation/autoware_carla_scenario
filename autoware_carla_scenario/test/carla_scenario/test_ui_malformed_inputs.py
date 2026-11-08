"""Malformed files and requests are answered with 4xx, not 500 (#1964).

* A SLURM manifest that is not what the launcher writes (a case without an
  index, one that is not an object, an index that is not a number) is read as
  no manifest, so it cannot break the session list; a stem that is not a
  single safe path component is not used to name a case's files.
* A trajectory log or coverage file holding a number too large for an int or
  a float is read as the docstrings promise: the record is skipped.
* A case index too long to be a case is a 404 (or skipped in a comparison,
  as any other ref that names no case), not ENAMETOOLONG or int()'s limit.
* A run request that cannot run (a refused pattern, no match, a sweep that
  fails to expand, a SLURM option value that is not valid) is a 400 with the
  same JSON body as before.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import cases, results, runner, scanner, slurm_launch
from autoware_carla_scenario.ui import sweep_resolver

DATE = "2026-11-07"
TIME = "12-00-00"


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


# ── A malformed SLURM manifest ──────────────────────────────────────────────


def _slurm_batch(base: Path, manifest: Any) -> Path:
    batch = base / "outputs" / DATE / TIME
    (batch / "slurm" / "results").mkdir(parents=True)
    (batch / "slurm" / "submissions").mkdir()
    (batch / "slurm" / "manifest.json").write_text(json.dumps(manifest))
    return batch


@pytest.mark.parametrize(
    "case_list",
    [
        [{"label": "a"}],  # no index (KeyError)
        ["a"],  # not an object (AttributeError)
        [{"index": "x", "label": "a"}],  # not a number (ValueError)
        [{"index": 0.5, "label": "a"}],
        [{"index": True, "label": "a"}],
        [{"index": -1, "label": "a"}],
        [{"index": 0, "label": "a"}, {"index": 0, "label": "b"}],
    ],
)
def test_a_malformed_manifest_does_not_break_the_listing(
    base: Path, client: TestClient, case_list: list[Any]
) -> None:
    _slurm_batch(base, {"cases": case_list})
    assert client.get("/").status_code == 200
    assert client.get(f"/session/batch/{DATE}/{TIME}").status_code == 200
    assert client.get(f"/session/batch/{DATE}/{TIME}/0").status_code == 200
    assert client.get(f"/compare?s=batch/{DATE}/{TIME}").status_code == 200


def test_a_good_manifest_still_lists_its_cases(base: Path, client: TestClient) -> None:
    batch = _slurm_batch(
        base,
        {
            "cases": [
                {"index": 1, "label": "b", "case_overrides": "x=1"},
                {"index": 0, "label": "a", "case_overrides": ["x=0"]},
            ]
        },
    )
    found = cases.batch_entries(batch)
    assert found is not None
    entries, final = found
    assert not final
    assert [e["scenario"] for e in entries] == ["a", "b"]
    # Overrides that are not a list are none, not a list of characters.
    assert [e["overrides"] for e in entries] == [["x=0"], []]


@pytest.mark.parametrize("stem", ["../../x", "a/b", "..", 7])
def test_a_manifest_stem_that_is_not_a_safe_component_is_not_used(
    base: Path, stem: Any
) -> None:
    batch = _slurm_batch(base, {"cases": [{"index": 0, "label": "a", "stem": stem}]})
    files = cases.batch_case_files(batch)
    assert len(files) == 1
    for found in files:
        assert found is None or found.directory == batch and "/" not in found.stem
        assert found is None or found.stem not in (".", "..")


def test_a_safe_manifest_stem_is_used(base: Path) -> None:
    batch = _slurm_batch(
        base, {"cases": [{"index": 0, "label": "a", "stem": "Case-1"}]}
    )
    assert cases.batch_case_files(batch) == [cases.CaseFiles(batch, "Case-1")]


# ── Numbers too large in a trajectory log or a coverage file ────────────────


def _log(path: Path, *records: dict[str, Any] | str) -> Path:
    path.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in records) + "\n"
    )
    return path


def _tick(t: Any) -> str:
    return (
        '{"type": "tick", "frame": 1, "t": ' + str(t) + ', "actors": '
        "[[1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]]}"
    )


def test_an_actor_id_too_large_for_an_int_is_skipped(tmp_path: Path) -> None:
    good = {"type": "actor", "id": 1, "role": "ego", "extent": [2.0, 1.0, 0.7]}
    log = _log(
        tmp_path / "A.trajectory.jsonl",
        '{"type": "actor", "id": 1e400, "role": "ego"}',
        good,
        _tick(0.0),
        _tick(0.05),
    )
    plain = _log(tmp_path / "B.trajectory.jsonl", good, _tick(0.0), _tick(0.05))
    assert results.case_metrics(log) == results.case_metrics(plain)


def test_a_time_too_large_for_a_float_is_skipped(tmp_path: Path) -> None:
    good = {"type": "actor", "id": 1, "role": "ego", "extent": [2.0, 1.0, 0.7]}
    log = _log(
        tmp_path / "A.trajectory.jsonl",
        good,
        _tick(0.0),
        _tick("9" * 400),
        _tick(0.05),
    )
    plain = _log(tmp_path / "B.trajectory.jsonl", good, _tick(0.0), _tick(0.05))
    assert results.case_metrics(log) == results.case_metrics(plain)


@pytest.mark.parametrize(
    "seconds", [10**400, "x", [1]], ids=["too-large", "text", "list"]
)
def test_coverage_seconds_that_are_no_float_count_as_none(seconds: Any) -> None:
    coverage = {
        "odd": {"cells": {"attributes": ["a"], "rows": [[["b"], 1, seconds, 0.0]]}}
    }
    found = [
        results.CaseResult(
            index=0,
            label="s#1",
            passed=True,
            errored=False,
            elapsed_seconds=1.0,
            message="",
            overrides=[],
            files=None,
            coverage=coverage,
        )
    ]
    # The row counts (its ticks are), its seconds as none.
    cells = results._odd_cells(coverage)
    assert cells is not None
    assert results._in_cells(cells, [("a", "b")], 0.0)
    assert not results._in_cells(cells, [("a", "b")], 0.5)
    # openodd's view refuses a document with a number too long for a float
    # (#2012), so the case is in no cell, as the matrix leaves it out.
    viewable = not isinstance(seconds, int)
    assert results.cases_in_cell(found, [("a", "b")]) == (found if viewable else [])
    assert results.cases_in_cell(found, [("a", "b")], min_seconds=0.5) == []


def test_a_batch_whose_log_overflows_is_still_compared(
    base: Path, client: TestClient
) -> None:
    batch = base / "outputs" / DATE / TIME
    batch.mkdir(parents=True)
    result = {"passed": True, "message": "", "elapsed_seconds": 1.0}
    (batch / "batch_results.json").write_text(
        json.dumps([{"scenario": "s", "overrides": [], **result}])
    )
    (batch / "Case_result.json").write_text(json.dumps(result))
    _log(
        batch / "Case.trajectory.jsonl",
        '{"type": "actor", "id": 1e400, "role": "ego"}',
        _tick("9" * 400),
    )
    assert client.get(f"/compare?s=batch/{DATE}/{TIME}").status_code == 200
    assert client.get(f"/session/batch/{DATE}/{TIME}/odd").status_code == 200


# ── Case indices too long to be a case ──────────────────────────────────────


def _multirun(base: Path) -> Path:
    job = base / "multirun" / DATE / TIME / "0"
    job.mkdir(parents=True)
    (job / "Case_result.json").write_text(json.dumps({"passed": True}))
    return job


@pytest.mark.parametrize("digits", [19, 300, 5000])
def test_a_case_index_too_long_is_a_404(
    base: Path, client: TestClient, digits: int
) -> None:
    _multirun(base)
    index = "9" * digits
    session = f"multirun/{DATE}/{TIME}"
    assert client.get(f"/session/{session}/{index}").status_code == 404
    assert client.post(f"/api/recording/{session}/{index}").status_code == 404
    assert client.get(f"/recording/{session}/{index}.rrd").status_code == 404
    # The session and its case still are.
    assert client.get(f"/session/{session}/0").status_code == 200


@pytest.mark.parametrize("digits", [19, 300, 5000])
def test_a_comparison_skips_a_case_index_too_long(
    base: Path, client: TestClient, digits: int
) -> None:
    _multirun(base)
    response = client.get(
        f"/compare/runs?run=multirun/{DATE}/{TIME}/{'9' * digits}"
        f"&run=multirun/{DATE}/{TIME}/0"
    )
    assert response.status_code == 200


def test_job_dir_refuses_an_index_too_large(base: Path) -> None:
    _multirun(base)
    assert cases.job_dir(base, DATE, TIME, 0) is not None
    assert cases.job_dir(base, DATE, TIME, 10**300) is None
    assert scanner.load_scenario(base, "multirun", DATE, TIME, 10**5000) is None


@pytest.mark.parametrize("value", ["9" * 5000, 10**30], ids=["text", "number"])
def test_a_resubmit_count_too_long_is_a_400(
    base: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    _slurm_batch(base, {"cases": [{"index": 0, "label": "a"}]})
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))
    ran: list[Any] = []
    monkeypatch.setattr(slurm_launch, "run_cli", lambda args: ran.append(args))
    response = client.post(
        f"/api/slurm/{DATE}/{TIME}/resubmit",
        content=json.dumps({"array_parallelism": value}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["ok"] is False
    assert ran == []


# ── Run requests refused with 400 ───────────────────────────────────────────


@pytest.fixture
def started(base: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_start_run(overrides_list: list[list[str]], **_: Any) -> bool:
        seen["jobs"] = overrides_list
        return True

    monkeypatch.setattr(runner, "start_run", fake_start_run)
    return seen


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("../*", "'..' cannot be a path component"),
        ("a/**b", "'**' can only be an entire path component"),
        ("no_such_scenario_dir/*", "No scenarios match 'no_such_scenario_dir/*'"),
    ],
)
def test_a_pattern_that_cannot_run_is_a_400(
    started: dict[str, Any], client: TestClient, scenario: str, message: str
) -> None:
    response = client.post("/api/run", json={"scenario": scenario})
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert message in data["message"]
    assert started == {}


def test_a_sweep_that_fails_to_expand_is_a_400(
    started: dict[str, Any], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolve(name: str, extra: Any = None, **_: Any) -> list[list[str]]:
        raise sweep_resolver.SweepResolutionError(f"{name}: the map is missing")

    monkeypatch.setattr(sweep_resolver, "resolve_sweep", resolve)
    response = client.post(
        "/api/run", json={"scenario": "a/b", "extra_overrides": [], "sweeper": "x"}
    )
    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "message": "Sweep resolution failed: a/b: the map is missing",
    }
    assert started == {}


@pytest.fixture
def submitted(base: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    commands: list[list[str]] = []
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))
    monkeypatch.setattr(
        slurm_launch, "start_submit", lambda command, base: commands.append(command)
    )
    monkeypatch.delenv("VIEWER_SCENARIO_COMMAND", raising=False)
    return commands


def test_an_invalid_slurm_value_is_a_400(
    submitted: list[list[str]], client: TestClient
) -> None:
    body = {"scenario": "x", "launcher": "slurm", "slurm": {"python": "$(evil)"}}
    response = client.post("/api/run", json=body)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error" and "not a path" in data["message"]
    assert submitted == []
    preview = client.post("/api/run/preview", json=body)
    assert preview.status_code == 400
    assert preview.json()["command"].startswith("# ")
    assert "not a path" in preview.json()["command"]


def test_a_valid_slurm_request_is_still_answered_200(
    submitted: list[list[str]], client: TestClient
) -> None:
    body = {"scenario": "x", "launcher": "slurm", "slurm": {"partition": "main"}}
    response = client.post("/api/run", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "submitting"}
    assert client.post("/api/run/preview", json=body).status_code == 200


def test_a_submission_while_another_is_made_is_a_409(
    base: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))

    def busy(command: list[str], base: Path) -> None:
        raise RuntimeError("A batch is still being submitted")

    monkeypatch.setattr(slurm_launch, "start_submit", busy)
    response = client.post(
        "/api/run", json={"scenario": "x", "launcher": "slurm", "slurm": {}}
    )
    assert response.status_code == 409
    assert response.json() == {
        "status": "error",
        "message": "A batch is still being submitted",
    }
