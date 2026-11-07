"""A multirun or single session still running is not served stale (#1964).

The viewer kept a multirun or single session's pages with nothing to
invalidate them: a job opened before its result existed stayed "No result
file found", a job added later was missing, and a single run opened mid-run
stayed empty, until the Update button.  They are stamped now, so a reload
shows what the run has written since; a finished session is still kept.

The times of the files and directories are set explicitly, so a change does
not rest on the filesystem's timestamp resolution.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.ui import scanner

DATE = "2026-11-07"
TIME = "12-00-00"
T0 = 1_800_000_000


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    scanner.clear_cache()
    yield
    scanner.clear_cache()


def _at(path: Path, seconds: int) -> None:
    os.utime(path, ns=(seconds * 10**9, seconds * 10**9))


def _result(passed: bool, message: str) -> dict[str, Any]:
    return {
        "scenario": "Case",
        "passed": passed,
        "message": message,
        "elapsed_seconds": 1.0,
        "condition_statuses": [],
    }


def _job(session: Path, index: int, seconds: int) -> Path:
    """A multirun job as Hydra starts it: its overrides, no result yet."""
    job = session / str(index)
    (job / ".hydra").mkdir(parents=True)
    (job / ".hydra" / "overrides.yaml").write_text(
        f"- scenario=case\n- x={index}\n", encoding="utf-8"
    )
    _at(job / ".hydra", seconds)
    _at(job, seconds)
    _at(session, seconds)
    return job


def _write_result(directory: Path, data: Any, seconds: int) -> None:
    path = directory / "Case_result.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    _at(path, seconds)
    _at(directory, seconds)


def _messages(base: Path, session_type: str) -> list[tuple[int, str]]:
    return [
        (item.index, item.message)
        for item in scanner.load_session(base, session_type, DATE, TIME)
    ]


def test_a_multirun_job_opened_before_its_result_shows_it_on_reload(
    tmp_path: Path,
) -> None:
    """The issue's probe: the result written and a second job added."""
    session = tmp_path / "multirun" / DATE / TIME
    job = _job(session, 0, T0)
    assert _messages(tmp_path, "multirun") == [(0, "No result file found")]
    view = scanner.load_scenario(tmp_path, "multirun", DATE, TIME, 0)
    assert view is not None and view.passed is None

    _write_result(job, _result(True, "done"), T0 + 10)
    _job(session, 1, T0 + 20)
    assert _messages(tmp_path, "multirun") == [
        (0, "done"),
        (1, "No result file found"),
    ]
    view = scanner.load_scenario(tmp_path, "multirun", DATE, TIME, 0)
    assert view is not None and view.passed is True and view.message == "done"


def test_a_result_written_in_place_shows_on_reload(tmp_path: Path) -> None:
    """A result read half written (no directory's time changes as it is
    finished) is not kept."""
    session = tmp_path / "multirun" / DATE / TIME
    job = _job(session, 0, T0)
    path = job / "Case_result.json"
    path.write_text('{"passed": tr', encoding="utf-8")
    _at(job, T0 + 10)
    assert _messages(tmp_path, "multirun") == [(0, "Failed to read result JSON")]
    path.write_text(json.dumps(_result(False, "late")), encoding="utf-8")
    _at(job, T0 + 10)  # as if in the same tick as its creation
    assert _messages(tmp_path, "multirun") == [(0, "late")]


def test_a_single_run_opened_mid_run_shows_its_result_on_reload(
    tmp_path: Path,
) -> None:
    run = tmp_path / "outputs" / DATE / TIME
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "overrides.yaml").write_text("- scenario=case\n")
    _at(run, T0)
    assert scanner.load_session(tmp_path, "single", DATE, TIME) == []
    assert scanner.load_scenario(tmp_path, "single", DATE, TIME, 0) is not None

    _write_result(run, _result(True, "single done"), T0 + 10)
    assert _messages(tmp_path, "single") == [(0, "single done")]
    view = scanner.load_scenario(tmp_path, "single", DATE, TIME, 0)
    assert view is not None and view.message == "single done"


def test_a_finished_multirun_session_is_still_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "multirun" / DATE / TIME
    for index in range(3):
        _write_result(_job(session, index, T0), _result(True, "ok"), T0 + 1)
    first = scanner.load_session(tmp_path, "multirun", DATE, TIME)
    case = scanner.load_scenario(tmp_path, "multirun", DATE, TIME, 1)

    def unexpected(*_: Any) -> None:
        raise AssertionError("read again although nothing changed")

    monkeypatch.setattr(scanner, "_read_result_json", unexpected)
    assert scanner.load_session(tmp_path, "multirun", DATE, TIME) is first
    assert scanner.load_scenario(tmp_path, "multirun", DATE, TIME, 1) is case
