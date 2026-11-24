"""The scanner reads the run the rest of the viewer shows, and the sweeper the
last run (#1039).

A directory that holds runs ``X`` and ``X-1`` (a rerun into the same output
directory) has its case picked in ``stem_order`` by ``cases.case_files`` and
``results.load_results``: the last run, ``X-1``, whose verdict the rerun's
exit status and the gate read (#2416). The session list and the detail page
must read ``X-1``'s result too. The sweeper, which retries a case in the same
job directory, needs the verdict of the attempt it just ran: the last stem in
``stem_order`` too (``X-10`` after ``X-2``, which a plain sort reverses).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import _read_verdict
from autoware_carla_scenario.ui import cases, results, scanner
from autoware_carla_scenario.ui.case_paths import last_result_path, result_path

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


def test_result_path_takes_the_last_stem_in_stem_order(tmp_path: Path) -> None:
    _write_runs(tmp_path, {"X-1": True, "X-2": False})
    assert result_path(tmp_path) == tmp_path / "X-2_result.json"
    _write_runs(tmp_path, {"X-10": False, "X": True})
    assert result_path(tmp_path) == tmp_path / "X-10_result.json"


def test_result_path_of_a_last_run_without_a_result(tmp_path: Path) -> None:
    # The rerun crashed before it wrote its result: no result, not the
    # earlier run's PASS (#2416).
    _write_runs(tmp_path, {"X": True})
    (tmp_path / "X-1.trajectory.jsonl").write_text("")
    assert result_path(tmp_path) is None


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

    assert files is not None and files.stem == "CutIn-1"
    assert (summary.passed_count, summary.scenario_name) == (0, "CutIn-1")
    assert (item.passed, item.message, item.scenario_name) == (
        False,
        "CutIn-1",
        "CutIn-1",
    )
    assert view is not None
    assert (view.passed, view.message) == (False, "CutIn-1")
    assert Path(view.trajectory_path or "").name == "CutIn-1.trajectory.jsonl"
    assert (result.passed, result.message) == (False, "CutIn-1")


def test_a_multirun_job_reads_the_case_the_viewer_shows(tmp_path: Path) -> None:
    _write_runs(tmp_path / "multirun" / DATE / TIME / "0", RUNS)

    (summary,) = scanner.scan_sessions(tmp_path)
    (item,) = scanner.load_session(tmp_path, "multirun", DATE, TIME)
    view = scanner.load_scenario(tmp_path, "multirun", DATE, TIME, 0)
    (result,) = results.load_results(tmp_path, "multirun", DATE, TIME, metrics=False)

    assert (summary.passed_count, summary.scenario_name) == (0, "CutIn-1")
    assert (item.passed, item.message) == (False, "CutIn-1")
    assert view is not None
    assert (view.passed, view.message) == (False, "CutIn-1")
    assert (result.passed, result.message) == (False, "CutIn-1")


def test_last_result_path_takes_the_last_stem_in_stem_order(tmp_path: Path) -> None:
    assert last_result_path(tmp_path) is None
    _write_runs(tmp_path, {"X": True, "X-1": False})
    assert last_result_path(tmp_path) == tmp_path / "X-1_result.json"
    _write_runs(tmp_path, {"X-2": False, "X-10": True})
    assert last_result_path(tmp_path) == tmp_path / "X-10_result.json"


def test_last_result_path_is_none_when_the_last_run_left_no_result(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path, {"X": True})
    (tmp_path / "X-1.trajectory.jsonl").write_text("")
    assert last_result_path(tmp_path) is None


def test_the_sweeper_reads_the_verdict_of_the_last_attempt(tmp_path: Path) -> None:
    # A retried case: the first attempt left a result, the retry X-1 another.
    _write_runs(tmp_path, {"CutIn": True, "CutIn-1": False})
    assert _read_verdict(tmp_path) is False

    _write_runs(tmp_path, {"CutIn": False, "CutIn-1": True})
    assert _read_verdict(tmp_path) is True

    # A third attempt: a plain sort still puts CutIn-1 first.
    _write_runs(tmp_path, {"CutIn-1": True, "CutIn-2": False})
    assert _read_verdict(tmp_path) is False


def test_the_sweeper_reads_no_stale_verdict(tmp_path: Path) -> None:
    # The retry wrote its trajectory but no result: no verdict, not the
    # first attempt's.
    _write_runs(tmp_path, {"CutIn": True})
    (tmp_path / "CutIn-1.trajectory.jsonl").write_text("")
    assert _read_verdict(tmp_path) is None
