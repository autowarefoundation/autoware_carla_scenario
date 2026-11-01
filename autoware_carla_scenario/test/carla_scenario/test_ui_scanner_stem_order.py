"""The scanner and the sweeper read the run the rest of the viewer shows (#1039).

A directory that holds runs ``X`` and ``X-1`` (a rerun into the same output
directory) has its case picked in ``stem_order`` (``X`` first) by
``cases.case_files`` and ``results.load_results``. The session list, the
detail page and the sweeper's verdict must read ``X``'s result too, not
``X-1``'s, which a plain sort puts first (``-`` sorts before ``_``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import _read_verdict
from autoware_carla_scenario.ui import cases, results, scanner
from autoware_carla_scenario.ui.case_paths import result_path

DATE = "2026-11-01"
TIME = "10-00-00"


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    scanner.clear_cache()
    yield
    scanner.clear_cache()


def _write_runs(directory: Path, runs: dict[str, bool]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stem, passed in runs.items():
        (directory / f"{stem}_result.json").write_text(
            json.dumps({"passed": passed, "message": stem, "elapsed_seconds": 1.0})
        )
        (directory / f"{stem}.trajectory.jsonl").write_text("")


RUNS = {"CutIn": True, "CutIn-1": False}


def test_result_path_takes_the_first_stem_in_stem_order(tmp_path: Path) -> None:
    _write_runs(tmp_path, {"X-10": False, "X-2": False, "X-1": True})
    assert result_path(tmp_path) == tmp_path / "X-1_result.json"
    _write_runs(tmp_path, {"X": True})
    assert result_path(tmp_path) == tmp_path / "X_result.json"


def test_result_path_of_a_directory_without_a_result(tmp_path: Path) -> None:
    (tmp_path / "X.trajectory.jsonl").write_text("")
    assert result_path(tmp_path) is None
    assert result_path(tmp_path / "missing") is None


def test_a_single_run_reads_the_case_the_viewer_shows(tmp_path: Path) -> None:
    _write_runs(tmp_path / "outputs" / DATE / TIME, RUNS)

    (summary,) = scanner.scan_sessions(tmp_path)
    (item,) = scanner.load_session(tmp_path, "single", DATE, TIME)
    view = scanner.load_scenario(tmp_path, "single", DATE, TIME, 0)
    (result,) = results.load_results(tmp_path, "single", DATE, TIME, metrics=False)
    files = cases.case_files(tmp_path, "single", DATE, TIME, 0)

    assert files is not None and files.stem == "CutIn"
    assert (summary.passed_count, summary.scenario_name) == (1, "CutIn")
    assert (item.passed, item.message, item.scenario_name) == (True, "CutIn", "CutIn")
    assert view is not None
    assert (view.passed, view.message) == (True, "CutIn")
    assert Path(view.trajectory_path or "").name == "CutIn.trajectory.jsonl"
    assert (result.passed, result.message) == (True, "CutIn")


def test_a_multirun_job_reads_the_case_the_viewer_shows(tmp_path: Path) -> None:
    _write_runs(tmp_path / "multirun" / DATE / TIME / "0", RUNS)

    (summary,) = scanner.scan_sessions(tmp_path)
    (item,) = scanner.load_session(tmp_path, "multirun", DATE, TIME)
    view = scanner.load_scenario(tmp_path, "multirun", DATE, TIME, 0)
    (result,) = results.load_results(tmp_path, "multirun", DATE, TIME, metrics=False)

    assert (summary.passed_count, summary.scenario_name) == (1, "CutIn")
    assert (item.passed, item.message) == (True, "CutIn")
    assert view is not None
    assert (view.passed, view.message) == (True, "CutIn")
    assert (result.passed, result.message) == (True, "CutIn")


def test_the_sweeper_reads_the_verdict_of_the_first_stem(tmp_path: Path) -> None:
    _write_runs(tmp_path, RUNS)
    assert _read_verdict(tmp_path) is True

    _write_runs(tmp_path, {"CutIn": False, "CutIn-1": True})
    assert _read_verdict(tmp_path) is False
