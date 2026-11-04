"""A SLURM batch resubmitted and collected again is not served stale (#1353).

The viewer keeps a final batch's session and case pages, but only while the
batch's files stay the same (``cases.batch_stamp``): a new collect shows on a
reload, and a submission newer than the summary makes the batch not final, its
cases read from what the tasks record (``slurm/results``).

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

from autoware_carla_scenario.ui import cases, scanner

DATE = "2026-10-19"
TIME = "12-00-00"
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


def _collect(batch: Path, passed: bool, message: str, seconds: int) -> None:
    """What ``scenario-slurm collect`` leaves: the summary, then
    ``collected.json``."""
    _write(
        batch / "batch_results.json",
        [{"scenario": "s", "overrides": ["x=0"], **_result(passed, message)}],
        seconds,
    )
    _write(
        batch / "slurm" / "collected.json",
        {"cases": 1, "passed": int(passed), "errored": [], "missing": []},
        seconds,
    )
    _at(batch, seconds)


def _batch(base: Path) -> Path:
    """A SLURM batch of one case, collected with the case failed."""
    batch = base / "outputs" / DATE / TIME
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
                "index": 0,
                "label": "s",
                "scenario": "s",
                "overrides": ["x=0"],
                "case_overrides": ["x=0"],
                "stem": "Case",
            }
        ],
    }
    _write(root / "manifest.json", manifest, T0)
    _write(root / "submissions" / "0.json", {"index": 0, "cases": [0]}, T0)
    _at(root / "results", T0)
    _collect(batch, False, "no outcome", T0 + 10)
    return batch


def _outcome(base: Path) -> tuple[bool | None, str]:
    (item,) = scanner.load_session(base, "batch", DATE, TIME)
    return item.passed, item.message


def _case(base: Path) -> tuple[bool | None, str]:
    view = scanner.load_scenario(base, "batch", DATE, TIME, 0)
    assert view is not None
    return view.passed, view.message


def test_a_new_collect_shows_on_reload_without_clearing_the_cache(
    tmp_path: Path,
) -> None:
    """The issue's probe."""
    batch = _batch(tmp_path)
    assert _outcome(tmp_path) == (False, "no outcome")
    assert _case(tmp_path) == (False, "no outcome")
    # The resubmitted case passes, and the array's collect job runs.
    _write(
        batch / "slurm" / "results" / "0.json",
        {"index": 0, "stem": "Case", "result": _result(True, "ok")},
        T0 + 20,
    )
    _collect(batch, True, "ok", T0 + 30)
    assert _outcome(tmp_path) == (True, "ok")
    assert _case(tmp_path) == (True, "ok")
    assert cases.batch_entries(batch) is not None
    assert cases.batch_entries(batch)[1]  # type: ignore[index]


def test_a_resubmission_shows_the_live_results_until_it_is_collected(
    tmp_path: Path,
) -> None:
    batch = _batch(tmp_path)
    assert _outcome(tmp_path) == (False, "no outcome")
    root = batch / "slurm"
    # "Resubmit failed": a submission newer than the summary.
    _write(root / "submissions" / "1.json", {"index": 1, "cases": [0]}, T0 + 20)
    found = cases.batch_entries(batch)
    assert found is not None
    entries, final = found
    assert not final
    assert entries[0]["pending"] and entries[0]["passed"] is None
    passed, message = _outcome(tmp_path)
    assert passed is None and "No result yet" in message
    # The case finishes: its outcome shows before the collect.
    _write(
        root / "results" / "0.json",
        {"index": 0, "stem": "Case", "result": _result(True, "ok")},
        T0 + 30,
    )
    assert _outcome(tmp_path) == (True, "ok")
    assert _case(tmp_path) == (True, "ok")
    # The collect makes it final again.
    _collect(batch, True, "ok", T0 + 40)
    found = cases.batch_entries(batch)
    assert found is not None and found[1]
    assert _outcome(tmp_path) == (True, "ok")


def test_a_result_newer_than_the_summary_makes_it_stale(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    _write(
        batch / "slurm" / "results" / "0.json",
        {"index": 0, "stem": "Case", "result": _result(True, "ok")},
        T0 + 20,
    )
    found = cases.batch_entries(batch)
    assert found is not None
    assert not found[1]
    assert found[0][0]["passed"] is True


def test_a_local_batch_s_summary_is_final(tmp_path: Path) -> None:
    batch = tmp_path / "outputs" / DATE / TIME
    batch.mkdir(parents=True)
    _write(
        batch / "batch_results.json",
        [{"scenario": "s", **_result(True, "ok")}],
        T0,
    )
    found = cases.batch_entries(batch)
    assert found is not None and found[1]


def test_an_unchanged_batch_is_served_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _batch(tmp_path)
    calls: list[Path] = []
    real = scanner.batch_entries

    def counting(directory: Path) -> Any:
        calls.append(directory)
        return real(directory)

    monkeypatch.setattr(scanner, "batch_entries", counting)
    for _ in range(3):
        assert _outcome(tmp_path) == (False, "no outcome")
    assert len(calls) == 1
    for _ in range(3):
        assert _case(tmp_path) == (False, "no outcome")
    assert len(calls) == 2
