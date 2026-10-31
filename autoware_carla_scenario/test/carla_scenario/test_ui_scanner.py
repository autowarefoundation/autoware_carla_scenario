"""The viewer's session scan survives a malformed result file (#785).

One result JSON that is not an object, not UTF-8, or names no scenario, or a
``batch_results.json`` with entries that are not objects, must not break the
scan of every other session (``scan_sessions`` feeds the session list).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import _read_verdict
from autoware_carla_scenario.ui import results, scanner

DATE = "2026-01-01"
BAD = "00-00-00"
GOOD = "00-00-01"


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    scanner.clear_cache()
    yield
    scanner.clear_cache()


def _session(base: Path, time: str, name: str, data: bytes) -> Path:
    directory = base / "outputs" / DATE / time
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(data)
    return directory


def _with_good_session(base: Path) -> None:
    _session(
        base,
        GOOD,
        "Good_result.json",
        json.dumps({"passed": True, "scenario": "Good"}).encode(),
    )


def _listed(base: Path) -> dict[str, str]:
    return {s.time: s.scenario_name for s in scanner.scan_sessions(base)}


@pytest.mark.parametrize(
    "data",
    [b"[1, 2]", b"\xff\xfe{", b'"text"', b"null"],
    ids=["list", "not-utf8", "string", "null"],
)
def test_a_single_result_that_is_no_json_object_lists_as_failed(
    tmp_path: Path, data: bytes
) -> None:
    _with_good_session(tmp_path)
    _session(tmp_path, BAD, "X_result.json", data)

    sessions = {s.time: s for s in scanner.scan_sessions(tmp_path)}

    assert sessions[GOOD].scenario_name == "Good"
    assert sessions[GOOD].passed_count == 1
    assert sessions[BAD].scenario_name == "X"
    assert sessions[BAD].passed_count == 0
    assert scanner.load_session(tmp_path, "single", DATE, BAD) == []
    assert scanner.load_scenario(tmp_path, "single", DATE, BAD, 0) is None


@pytest.mark.parametrize("scenario", [None, "", 5, ["a"]])
def test_a_result_naming_no_scenario_is_named_after_its_file(
    tmp_path: Path, scenario: object
) -> None:
    _with_good_session(tmp_path)
    _session(
        tmp_path,
        BAD,
        "X_result.json",
        json.dumps({"passed": True, "scenario": scenario}).encode(),
    )

    assert _listed(tmp_path) == {GOOD: "Good", BAD: "X"}
    (item,) = scanner.load_session(tmp_path, "single", DATE, BAD)
    assert item.scenario_name == "X"
    assert item.passed is True


def test_result_fields_of_the_wrong_type_are_left_out(tmp_path: Path) -> None:
    _session(
        tmp_path,
        BAD,
        "X_result.json",
        json.dumps(
            {
                "passed": "yes",
                "elapsed_seconds": "slow",
                "message": 3,
                "condition_statuses": [1, {"label": None, "details": [2]}],
            }
        ).encode(),
    )

    (item,) = scanner.load_session(tmp_path, "single", DATE, BAD)
    assert (item.passed, item.elapsed_seconds, item.message) == (None, None, "3")
    view = scanner.load_scenario(tmp_path, "single", DATE, BAD, 0)
    assert view is not None
    assert [c.label for c in view.condition_statuses] == [""]


def test_a_batch_summary_of_no_objects_is_skipped(tmp_path: Path) -> None:
    _with_good_session(tmp_path)
    _session(tmp_path, BAD, "batch_results.json", b"[1]")

    assert _listed(tmp_path) == {GOOD: "Good"}
    assert scanner.load_session(tmp_path, "batch", DATE, BAD) == []
    assert scanner.load_scenario(tmp_path, "batch", DATE, BAD, 0) is None


def test_batch_entries_that_are_not_objects_keep_the_others_position(
    tmp_path: Path,
) -> None:
    _with_good_session(tmp_path)
    _session(
        tmp_path,
        BAD,
        "batch_results.json",
        json.dumps(
            [
                {"scenario": None, "passed": True},
                7,
                {"scenario": "a", "passed": False, "elapsed_seconds": 1.5},
            ]
        ).encode(),
    )

    sessions = {s.time: s for s in scanner.scan_sessions(tmp_path)}
    assert sessions[GOOD].scenario_name == "Good"
    assert sessions[BAD].scenario_name == "a, unknown"
    assert (sessions[BAD].passed_count, sessions[BAD].total_count) == (1, 2)

    items = scanner.load_session(tmp_path, "batch", DATE, BAD)
    assert [(i.index, i.scenario_name, i.passed) for i in items] == [
        (0, "unknown", True),
        (2, "a", False),
    ]
    view = scanner.load_scenario(tmp_path, "batch", DATE, BAD, 2)
    assert view is not None
    assert view.elapsed_seconds == 1.5


def test_results_reads_malformed_batch_entries(tmp_path: Path) -> None:
    _session(
        tmp_path,
        BAD,
        "batch_results.json",
        json.dumps(
            [
                {
                    "scenario": None,
                    "passed": "no",
                    "elapsed_seconds": "x",
                    "overrides": 3,
                },
                [1],
                {"scenario": "a", "passed": True, "elapsed_seconds": 2},
            ]
        ).encode(),
    )

    cases = results.load_results(tmp_path, "batch", DATE, BAD, metrics=False)

    assert [(c.index, c.label, c.passed, c.elapsed_seconds) for c in cases] == [
        (0, "unknown", None, None),
        (2, "a", True, 2.0),
    ]
    assert cases[0].overrides == []
    assert cases[1].metrics == {"elapsed_s": 2.0}


def test_results_reads_a_single_result_of_wrong_types(tmp_path: Path) -> None:
    _session(
        tmp_path,
        BAD,
        "X_result.json",
        json.dumps({"passed": 1, "elapsed_seconds": "x"}).encode(),
    )

    (case,) = results.load_results(tmp_path, "single", DATE, BAD, metrics=False)

    assert (case.passed, case.elapsed_seconds, case.metrics) == (None, None, {})


def test_the_sweeper_reads_no_verdict_off_a_result_not_utf8(tmp_path: Path) -> None:
    (tmp_path / "X_result.json").write_bytes(b"\xff\xfe{")

    assert _read_verdict(tmp_path) is None
