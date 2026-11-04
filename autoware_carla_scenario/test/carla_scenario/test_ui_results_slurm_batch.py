"""The ODD and compare views read a SLURM batch as the session page does (#1433).

``results.load_results`` reads a batch's entries through
``cases.batch_entries``: before ``scenario-slurm collect`` (no
``batch_results.json`` yet) the cases the tasks recorded so far, the others
pending; after a resubmit (a summary older than a submission or an outcome)
the newer outcomes, not the stale summary's.

The files' times are set explicitly, so the order of the writes does not rest
on the filesystem's timestamp resolution.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import results, scanner

DATE = "2026-10-19"
TIME = "12-00-00"
LOCAL = "13-00-00"
T0 = 1_800_000_000


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    scanner.clear_cache()
    yield
    scanner.clear_cache()


def _at(path: Path, seconds: int) -> None:
    os.utime(path, ns=(seconds * 10**9, seconds * 10**9))


def _write(path: Path, data: Any, seconds: int) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
    _at(path, seconds)
    _at(path.parent, seconds)


def _result(passed: bool, message: str) -> dict[str, Any]:
    return {
        "passed": passed,
        "message": message,
        "elapsed_seconds": 1.0,
        "condition_statuses": [],
    }


def _record(batch: Path, index: int, passed: bool, message: str, at: int) -> None:
    """What a task records for case *index* (``slurm/results/<i>.json``)."""
    _write(
        batch / "slurm" / "results" / f"{index}.json",
        {"index": index, "stem": f"Case{index}", "result": _result(passed, message)},
        at,
    )


def _slurm_batch(base: Path, time: str = TIME) -> Path:
    """A SLURM batch of two cases, submitted, nothing recorded or collected."""
    batch = base / "outputs" / DATE / time
    root = batch / "slurm"
    (root / "submissions").mkdir(parents=True)
    (root / "results").mkdir()
    manifest = {
        "version": 1,
        "batch_dir": str(batch),
        "scenario_names": ["s"],
        "expand": True,
        "launcher": {"type": "slurm"},
        "cases": [
            {
                "index": i,
                "label": f"s#{i}",
                "scenario": "s",
                "overrides": [f"x={i}"],
                "case_overrides": [f"x={i}"],
                "stem": f"Case{i}",
            }
            for i in range(2)
        ],
    }
    _write(root / "manifest.json", manifest, T0)
    _write(root / "submissions" / "0.json", {"index": 0, "cases": [0, 1]}, T0)
    _at(root / "results", T0)
    return batch


def _collect(batch: Path, verdicts: list[tuple[bool, str]], at: int) -> None:
    """What ``scenario-slurm collect`` leaves: the summary, then
    ``collected.json``."""
    _write(
        batch / "batch_results.json",
        [
            {"scenario": f"s#{i}", "overrides": [f"x={i}"], **_result(p, m)}
            for i, (p, m) in enumerate(verdicts)
        ],
        at,
    )
    _write(
        batch / "slurm" / "collected.json",
        {
            "cases": len(verdicts),
            "passed": sum(p for p, _ in verdicts),
            "errored": [],
            "missing": [],
        },
        at,
    )
    _at(batch, at)


def _verdicts(base: Path, time: str = TIME) -> list[tuple[bool | None, str, bool]]:
    return [
        (r.passed, r.outcome, r.pending)
        for r in results.load_results(base, "batch", DATE, time, metrics=False)
    ]


def test_an_uncollected_batch_shows_what_its_tasks_recorded(tmp_path: Path) -> None:
    batch = _slurm_batch(tmp_path)
    assert _verdicts(tmp_path) == [(None, "none", True), (None, "none", True)]
    _record(batch, 0, True, "ok", T0 + 10)
    found = results.load_results(tmp_path, "batch", DATE, TIME, metrics=False)
    assert [(r.passed, r.outcome, r.pending) for r in found] == [
        (True, "pass", False),
        (None, "none", True),
    ]
    assert [r.label for r in found] == ["s#0", "s#1"]
    assert [r.overrides for r in found] == [["x=0"], ["x=1"]]
    assert found[0].message == "ok"
    assert "No result yet" in found[1].message
    # The case files follow the manifest's stems, as the session page's do.
    assert [r.files.stem if r.files else None for r in found] == ["Case0", "Case1"]


def test_a_resubmitted_batch_shows_the_newer_outcomes(tmp_path: Path) -> None:
    batch = _slurm_batch(tmp_path)
    _record(batch, 0, False, "collision", T0 + 10)
    _record(batch, 1, True, "ok", T0 + 10)
    _collect(batch, [(False, "collision"), (True, "ok")], T0 + 20)
    assert _verdicts(tmp_path) == [(False, "fail", False), (True, "pass", False)]
    # "Resubmit failed": case 0 runs again and passes; no collect yet.
    _write(
        batch / "slurm" / "submissions" / "1.json", {"index": 1, "cases": [0]}, T0 + 30
    )
    _record(batch, 0, True, "ok now", T0 + 40)
    found = results.load_results(tmp_path, "batch", DATE, TIME, metrics=False)
    assert [(r.passed, r.message) for r in found] == [(True, "ok now"), (True, "ok")]
    # As the session page has it.
    items = scanner.load_session(tmp_path, "batch", DATE, TIME)
    assert [it.passed for it in items] == [r.passed for r in found]


def test_a_local_batch_reads_its_summary(tmp_path: Path) -> None:
    batch = tmp_path / "outputs" / DATE / LOCAL
    batch.mkdir(parents=True)
    _write(
        batch / "batch_results.json",
        [
            {"scenario": "s#0", "overrides": ["x=0"], **_result(True, "ok")},
            {"scenario": "s#1", "overrides": ["x=1"], "error": True, "message": "x"},
        ],
        T0,
    )
    found = results.load_results(tmp_path, "batch", DATE, LOCAL, metrics=False)
    assert [(r.label, r.outcome, r.pending) for r in found] == [
        ("s#0", "pass", False),
        ("s#1", "error", False),
    ]


def test_a_pending_case_is_left_out_of_a_comparison(tmp_path: Path) -> None:
    """A case without an outcome yet is no regression and no fail in the
    pass rate: it is compared as if its session had not run it."""
    baseline = tmp_path / "outputs" / DATE / LOCAL
    baseline.mkdir(parents=True)
    _write(
        baseline / "batch_results.json",
        [
            {"scenario": f"s#{i}", "overrides": [f"x={i}"], **_result(True, "ok")}
            for i in range(2)
        ],
        T0,
    )
    batch = _slurm_batch(tmp_path, "14-00-00")
    _record(batch, 0, True, "ok", T0 + 10)
    loaded = [
        (
            f"batch/{DATE}/{time}",
            results.load_results(tmp_path, "batch", DATE, time, metrics=False),
        )
        for time in (LOCAL, "14-00-00")
    ]
    comparison = results.compare(loaded)
    assert comparison.totals == [(2, 2), (1, 1)]
    changes = {case.key: case.change() for case in comparison.cases}
    assert changes == {"s|x=0": "same", "s|x=1": "gone"}


def test_the_compare_page_counts_a_session_s_pending_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncollected batch is not shown as complete: its pass rate is of
    the cases with an outcome, and its pending cases are counted beside it."""
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    baseline = tmp_path / "outputs" / DATE / LOCAL
    baseline.mkdir(parents=True)
    _write(
        baseline / "batch_results.json",
        [
            {"scenario": f"s#{i}", "overrides": [f"x={i}"], **_result(True, "ok")}
            for i in range(2)
        ],
        T0,
    )
    batch = _slurm_batch(tmp_path, "14-00-00")
    _record(batch, 0, True, "ok", T0 + 10)
    found = results.load_results(tmp_path, "batch", DATE, "14-00-00")
    assert results.pending_count(found) == 1
    page = TestClient(app_module.app).get(
        f"/compare?s=batch/{DATE}/{LOCAL}&s=batch/{DATE}/14-00-00"
    )
    assert page.status_code == 200
    summary = page.text.split('id="compare-summary"', 1)[1]
    assert summary.count("data-pending=") == 1
    assert 'data-pending="1"' in summary
    assert "1/1 passed" in summary and "1 pending (not collected)" in summary
    # Not green, as a complete session's 100% is.
    baseline_card, partial_card = summary.split('font-mono">')[1:3]
    assert "2/2 passed" in baseline_card and "text-green-600" in baseline_card
    assert "1/1 passed" in partial_card and "text-green-600" not in partial_card


def test_the_odd_page_labels_a_pending_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    batch = _slurm_batch(tmp_path)
    _record(batch, 0, True, "ok", T0 + 10)
    page = TestClient(app_module.app).get(f"/session/batch/{DATE}/{TIME}/odd")
    assert page.status_code == 200
    assert 'data-outcome="pass"' in page.text
    assert 'data-outcome="none"' in page.text
    assert "PENDING" in page.text
