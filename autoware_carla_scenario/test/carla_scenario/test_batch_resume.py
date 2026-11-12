"""How ``batch_resume`` reads and checks a resumed batch's summary, and reads
its kept entries back as results (#1957).

Pinned against summaries a hand, another version or a crash may have left:
values of the wrong type, keys missing or added, numbers JSON allows Python
to read in more than one way.  The messages, the results read back and the
summary written from them (``batch_report``) are compared whole.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.batch_report import (
    print_summary,
    write_batch_result_json,
)
from autoware_carla_scenario.batch_resume import (
    BATCH_RESULTS,
    KeptResult,
    ResumedBatch,
    ResumeIntoError,
    read_resumed_batch,
)
from autoware_carla_scenario.failure import FailureCategory


def _entry(scenario: str, stem: str, **members: Any) -> dict[str, Any]:
    return {"scenario": scenario, "overrides": None, "stem": stem, **members}


def _write(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BATCH_RESULTS).write_text(text, encoding="utf-8")
    return directory


def _read(
    directory: Path,
    labels: list[str],
    overrides: list[list[str]] | None = None,
) -> ResumedBatch:
    return read_resumed_batch(directory, labels, overrides)


def _refusal(
    directory: Path,
    labels: list[str],
    overrides: list[list[str]] | None = None,
) -> str:
    with pytest.raises(ResumeIntoError) as info:
        _read(directory, labels, overrides)
    return str(info.value)


def _kept(tmp_path: Path, text: str, count: int) -> list[KeptResult]:
    """The results of every entry of the summary *text*, of *count* cases
    named ``s0``, ``s1``, ..."""
    directory = _write(tmp_path / "run", text)
    resumed = _read(directory, [f"s{i}" for i in range(count)])
    return [KeptResult(entry) for entry in resumed.entries]


# -- read_resumed_batch: what it refuses, and how it says so ----------------


def test_no_such_directory(tmp_path: Path) -> None:
    assert _refusal(tmp_path / "nowhere", ["a"]) == (
        f"{tmp_path / 'nowhere'}: no such directory"
    )


def test_a_slurm_batch_s_directory(tmp_path: Path) -> None:
    (tmp_path / "slurm").mkdir()
    (tmp_path / "slurm" / "manifest.json").write_text("{}")
    _write(tmp_path, "[]")
    assert _refusal(tmp_path, []) == (
        f"{tmp_path}: a SLURM batch's directory; --resume-into resumes a "
        "local batch (resubmit a SLURM batch's cases with scenario-slurm)"
    )


def test_no_summary(tmp_path: Path) -> None:
    assert _refusal(tmp_path, ["a"]) == (
        f"{tmp_path}: holds no batch_results.json, so it is not the directory "
        "of a local batch"
    )


@pytest.mark.parametrize(
    ("text", "why"),
    [
        (
            "[{",
            "Expecting property name enclosed in double quotes: line 1 column 3 (char 2)",
        ),
        ("", "Expecting value: line 1 column 1 (char 0)"),
        ("[] []", "Extra data: line 1 column 4 (char 3)"),
    ],
)
def test_a_summary_that_is_no_json(tmp_path: Path, text: str, why: str) -> None:
    _write(tmp_path, text)
    assert _refusal(tmp_path, []) == (
        f"{tmp_path / BATCH_RESULTS}: the batch summary cannot be read ({why})"
    )


def test_a_summary_that_is_no_utf8(tmp_path: Path) -> None:
    (tmp_path / BATCH_RESULTS).write_bytes(b'["\xff"]')
    message = _refusal(tmp_path, ["a"])
    assert message.startswith(
        f"{tmp_path / BATCH_RESULTS}: the batch summary cannot be read ("
        "'utf-8' codec can't decode byte 0xff"
    )


def test_a_summary_nested_past_the_recursion_limit(tmp_path: Path) -> None:
    _write(tmp_path, "[" * 100_000 + "]" * 100_000)
    assert _refusal(tmp_path, ["a"]).startswith(
        f"{tmp_path / BATCH_RESULTS}: the batch summary cannot be read ("
    )


@pytest.mark.parametrize(
    "text", ['{"a": 1}', "null", "3", '"x"', '[{"a": 1}, 2]', "[[]]", "[null]"]
)
def test_a_summary_that_is_no_array_of_objects(tmp_path: Path, text: str) -> None:
    _write(tmp_path, text)
    assert _refusal(tmp_path, ["a", "b"]) == (
        f"{tmp_path / BATCH_RESULTS}: not a batch summary (not a JSON array "
        "of objects)"
    )


def test_an_empty_summary_of_an_empty_batch_is_read(tmp_path: Path) -> None:
    _write(tmp_path, "[]")
    resumed = _read(tmp_path, [])
    assert resumed.directory == tmp_path
    assert resumed.stems == []
    assert resumed.first_pending() is None


def test_a_summary_of_more_or_fewer_cases(tmp_path: Path) -> None:
    _write(tmp_path, json.dumps([_entry("a", "A"), _entry("b", "B")]))
    for labels in (["a"], ["a", "b", "c"]):
        assert _refusal(tmp_path, labels) == (
            f"{tmp_path / BATCH_RESULTS}: lists 2 case(s), this batch has "
            f"{len(labels)}; resume with the scenarios and overrides of the "
            "batch it ran"
        )


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        ("b", "'b'"),
        ("it's", '"it\'s"'),
        (None, "None"),
        (5, "5"),
        (1.0, "1.0"),
        (True, "True"),
        (["a"], "['a']"),
        ({"x": [1, None]}, "{'x': [1, None]}"),
        ("é\n", "'é\\n'"),
    ],
)
def test_a_case_of_another_scenario(tmp_path: Path, value: Any, shown: str) -> None:
    entries = [_entry("a", "A"), _entry("x", "B")]
    entries[1]["scenario"] = value
    _write(tmp_path, json.dumps(entries))
    assert _refusal(tmp_path, ["a", "a"]) == (
        f"{tmp_path / BATCH_RESULTS}: case 2 is {shown} there and 'a' here; "
        "resume with the scenarios and overrides of the batch it ran"
    )


def test_a_case_without_a_scenario(tmp_path: Path) -> None:
    entry = _entry("a", "A")
    del entry["scenario"]
    _write(tmp_path, json.dumps([entry]))
    assert _refusal(tmp_path, ["a"]) == (
        f"{tmp_path / BATCH_RESULTS}: case 1 is None there and 'a' here; "
        "resume with the scenarios and overrides of the batch it ran"
    )


@pytest.mark.parametrize(
    ("recorded", "expected", "shown_recorded", "shown_expected"),
    [
        (["x=1"], None, "['x=1']", "None"),
        ([], None, "[]", "None"),
        (None, ["x=1"], "None", "['x=1']"),
        (["x=2"], ["x=1"], "['x=2']", "['x=1']"),
        (["x=1", "y"], ["x=1"], "['x=1', 'y']", "['x=1']"),
        ([1], ["1"], "[1]", "['1']"),
        ("x=1", ["x=1"], "'x=1'", "['x=1']"),
        ({"x": "1"}, ["x=1"], "{'x': '1'}", "['x=1']"),
        ([["x=1"]], ["x=1"], "[['x=1']]", "['x=1']"),
        (False, None, "False", "None"),
        (0, None, "0", "None"),
    ],
)
def test_a_case_of_another_sweep_case(
    tmp_path: Path,
    recorded: Any,
    expected: list[str] | None,
    shown_recorded: str,
    shown_expected: str,
) -> None:
    entry = _entry("a", "A", overrides=recorded)
    _write(tmp_path, json.dumps([entry]))
    overrides = None if expected is None else [expected]
    assert _refusal(tmp_path, ["a"], overrides) == (
        f"{tmp_path / BATCH_RESULTS}: case 1 (a) ran with the sweep case "
        f"{shown_recorded} there and {shown_expected} here; resume with the "
        "scenarios, overrides and --multirun of the batch it ran"
    )


def test_a_case_without_overrides_is_one_without_a_sweep_case(
    tmp_path: Path,
) -> None:
    entry = _entry("a", "A")
    del entry["overrides"]
    _write(tmp_path, json.dumps([entry]))
    assert _read(tmp_path, ["a"]).stems == ["A"]
    assert _refusal(tmp_path, ["a"], [[]]) == (
        f"{tmp_path / BATCH_RESULTS}: case 1 (a) ran with the sweep case "
        "None there and [] here; resume with the scenarios, overrides and "
        "--multirun of the batch it ran"
    )


def test_the_sweep_cases_match(tmp_path: Path) -> None:
    entries = [
        _entry("a", "A", overrides=["x=1", "y=é"]),
        _entry("a", "A-1", overrides=[]),
    ]
    _write(tmp_path, json.dumps(entries))
    assert _read(tmp_path, ["a", "a"], [["x=1", "y=é"], []]).stems == ["A", "A-1"]


def test_every_case_is_matched_before_any_stem_is_read(tmp_path: Path) -> None:
    entries = [_entry("a", "A"), _entry("b", "B")]
    del entries[0]["stem"]
    entries[1]["scenario"] = "c"
    _write(tmp_path, json.dumps(entries))
    assert "case 2 is 'c' there and 'b' here" in _refusal(tmp_path, ["a", "b"])


def test_a_case_without_a_stem(tmp_path: Path) -> None:
    entries = [_entry("a", "A"), _entry("b", "B")]
    del entries[1]["stem"]
    _write(tmp_path, json.dumps(entries))
    assert _refusal(tmp_path, ["a", "b"]) == (
        f"{tmp_path / BATCH_RESULTS}: names no file stem for case 2 (b) -- it "
        "was written before a batch recorded its cases' stems -- so the "
        "outputs the cases run again left there cannot be told from the "
        "others', and nothing is removed; resume it into a new directory "
        "(without --resume-into)"
    )


@pytest.mark.parametrize(
    ("stem", "shown"),
    [
        (None, "None"),
        ("", "''"),
        (".", "'.'"),
        ("..", "'..'"),
        ("a/b", "'a/b'"),
        ("../A", "'../A'"),
        ("a\\b", "'a\\\\b'"),
        ("a\0b", "'a\\x00b'"),
        ("A", "'A'"),
        (5, "5"),
        (1.5, "1.5"),
        (True, "True"),
        (["B"], "['B']"),
        ({"B": 1}, "{'B': 1}"),
    ],
)
def test_a_stem_that_is_no_file_name_of_its_own(
    tmp_path: Path, stem: Any, shown: str
) -> None:
    entries = [_entry("a", "A"), _entry("b", "B")]
    entries[1]["stem"] = stem
    _write(tmp_path, json.dumps(entries))
    assert _refusal(tmp_path, ["a", "b"]) == (
        f"{tmp_path / BATCH_RESULTS}: case 2 (b) names the file stem {shown}, "
        "which is not a file name of its own in the directory"
    )


def test_stems_are_read_as_written(tmp_path: Path) -> None:
    entries = [
        _entry("a", "A b"),
        _entry("b", "...x"),
        _entry("c", "é-1"),
        _entry("d", " "),
    ]
    _write(tmp_path, json.dumps(entries))
    resumed = _read(tmp_path, ["a", "b", "c", "d"])
    assert resumed.stems == ["A b", "...x", "é-1", " "]


# -- ResumedBatch: where it resumes, and what it refuses to keep ------------


@pytest.mark.parametrize(
    ("pending", "first"),
    [
        ([None, None, None], None),
        ([None, True, True], 1),
        ([True, None, None], 0),
        ([1, "true", False], None),
        ([[True], {"a": True}, 1.0], None),
        ([False, None, True], 2),
    ],
)
def test_the_first_pending_case(
    tmp_path: Path, pending: list[Any], first: int | None
) -> None:
    entries = []
    for i, value in enumerate(pending):
        entry = _entry(f"s{i}", f"S{i}")
        if value is not None:
            entry["pending"] = value
        entries.append(entry)
    _write(tmp_path, json.dumps(entries))
    resumed = _read(tmp_path, ["s0", "s1", "s2"])
    assert resumed.first_pending() == first


def test_kept_cases_must_have_results(tmp_path: Path) -> None:
    entries = [
        _entry("s0", "S0"),
        _entry("s1", "S1", pending=1),
        _entry("s2", "S2", pending=True),
        _entry("s3", "S3", pending=True),
    ]
    _write(tmp_path, json.dumps(entries))
    resumed = _read(tmp_path, ["s0", "s1", "s2", "s3"])
    labels = ["L0", "L1", "L2", "L3"]
    for skip in (0, 1, 2):
        resumed.check_kept(skip, labels)
    for skip in (3, 4, 9):
        with pytest.raises(ResumeIntoError) as info:
            resumed.check_kept(skip, labels)
        assert str(info.value) == (
            f"{tmp_path}: case 3 (L2) has no result there yet; resume from "
            "case 3 or earlier (--resume-from 3)"
        )


@pytest.mark.parametrize(
    ("stem", "name", "accepted"),
    [
        ("Cut", "Cut", True),
        ("Cut-1", "Cut", True),
        ("Cut-12", "Cut", True),
        ("Cut-0", "Cut", False),
        ("Cut-01", "Cut", False),
        ("Cut-", "Cut", False),
        ("CutIn", "Cut", False),
        ("cut", "Cut", False),
        ("A.B-2", "A.B", True),
        ("AxB", "A.B", False),
        ("A+B", "A+B", True),
        ("AAB", "A+B", False),
    ],
)
def test_a_case_s_stem_names_its_class(
    tmp_path: Path, stem: str, name: str, accepted: bool
) -> None:
    entries = [_entry("s0", "S0"), _entry("s1", stem)]
    _write(tmp_path, json.dumps(entries))
    resumed = _read(tmp_path, ["s0", "s1"])
    if accepted:
        resumed.check_stems(1, [name])
        return
    with pytest.raises(ResumeIntoError) as info:
        resumed.check_stems(1, [name])
    assert str(info.value) == (
        f"{tmp_path}: case 2 wrote its outputs as {stem!r} there, which is "
        f"not a run of {name}; resume with the scenarios and overrides of the "
        "batch it ran"
    )


def test_stems_are_checked_from_the_resume_point(tmp_path: Path) -> None:
    entries = [_entry("s0", "Other"), _entry("s1", "Cut-1")]
    _write(tmp_path, json.dumps(entries))
    resumed = _read(tmp_path, ["s0", "s1"])
    resumed.check_stems(1, ["Cut"])
    resumed.check_stems(2, [])
    with pytest.raises(IndexError):
        resumed.check_stems(1, ["Cut", "Cut"])


# -- KeptResult: an entry read back as a result ------------------------------


def test_a_kept_entry_read_as_a_result(tmp_path: Path) -> None:
    entry = _entry(
        "s0",
        "S0",
        passed=False,
        message="line 1\nline 2",
        elapsed_seconds=12.25,
        condition_statuses=[
            {"label": "Goal", "satisfied": True, "message": "reached"},
            {"label": "No collision", "satisfied": False, "message": "hit"},
        ],
        failure={
            "condition": "No collision",
            "category": "collision",
            "at": 3.5,
            "message": "hit",
            "detail": "npc_1",
            "also_triggered": ["Timeout"],
        },
        rerun_recording="S0.rrd",
        rerun_error="conversion failed",
        error=True,
    )
    (result,) = _kept(tmp_path, json.dumps([entry]), 1)
    assert result.passed is False
    assert result.message == "line 1\nline 2"
    assert result.elapsed_seconds == 12.25
    assert [(s.label, s.satisfied, s.message) for s in result.condition_statuses] == [
        ("Goal", True, "reached"),
        ("No collision", False, "hit"),
    ]
    failure = result.failure
    assert failure is not None
    assert failure.condition == "No collision"
    assert failure.category is FailureCategory.COLLISION
    assert failure.at == 3.5
    assert failure.message == "hit"
    assert failure.detail == "npc_1"
    assert list(failure.also_triggered) == ["Timeout"]
    assert result.rerun_recording == "S0.rrd"
    assert result.rerun_error == "conversion failed"
    assert result.errored is True


def test_a_kept_entry_of_values_of_other_types(tmp_path: Path) -> None:
    entry = _entry(
        "s0",
        "S0",
        passed=1,
        message=["no"],
        elapsed_seconds=True,
        condition_statuses=[
            {"label": 5, "satisfied": 1, "message": None},
            "not a status",
            {"label": ["x"], "satisfied": True},
            {},
            [],
            {"label": {"a": 1.0}, "message": False, "satisfied": "true"},
        ],
        failure=["collision"],
        rerun_recording=5,
        rerun_error="",
        error=1,
    )
    (result,) = _kept(tmp_path, json.dumps([entry]), 1)
    assert result.passed is False
    assert result.message == ""
    assert result.elapsed_seconds == 0.0
    assert [(s.label, s.satisfied, s.message) for s in result.condition_statuses] == [
        ("5", False, "None"),
        ("['x']", True, ""),
        ("", False, ""),
        ("{'a': 1.0}", False, "False"),
    ]
    assert result.failure is None
    assert result.rerun_recording is None
    assert result.rerun_error is None
    assert result.errored is False


def test_an_entry_with_nothing_of_a_result(tmp_path: Path) -> None:
    (result,) = _kept(tmp_path, json.dumps([_entry("s0", "S0")]), 1)
    assert result.passed is False
    assert result.message == ""
    assert result.elapsed_seconds == 0.0
    assert result.condition_statuses == []
    assert result.failure is None
    assert result.rerun_recording is None
    assert result.rerun_error is None
    assert result.errored is False
    assert result.to_dict() == {}


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("3", 3.0),
        ("-2", -2.0),
        ("1e3", 1000.0),
        ("1E400", math.inf),
        ("-1e400", -math.inf),
        ("Infinity", math.inf),
        ("-0.0", -0.0),
        ("0.1000000000000000055511151231257827", 0.1),
        ("12345678901234567890", 12345678901234567890.0),
        ('"3"', 0.0),
        ("null", 0.0),
        ("false", 0.0),
        ("[1]", 0.0),
    ],
)
def test_elapsed_seconds_of_any_number(
    tmp_path: Path, text: str, seconds: float
) -> None:
    summary = (
        '[{"scenario": "s0", "overrides": null, "stem": "S0", '
        f'"elapsed_seconds": {text}}}]'
    )
    (result,) = _kept(tmp_path, summary, 1)
    assert result.elapsed_seconds == seconds
    assert math.copysign(1.0, result.elapsed_seconds) == math.copysign(1.0, seconds)


def test_elapsed_seconds_nan(tmp_path: Path) -> None:
    summary = (
        '[{"scenario": "s0", "overrides": null, "stem": "S0", "elapsed_seconds": NaN}]'
    )
    (result,) = _kept(tmp_path, summary, 1)
    assert math.isnan(result.elapsed_seconds)


def test_elapsed_seconds_past_a_float_raises_when_read(tmp_path: Path) -> None:
    """An integer no float holds is read, and raises only when its seconds
    are asked for, as Python's ``float()`` does."""
    summary = (
        '[{"scenario": "s0", "overrides": null, "stem": "S0", '
        f'"elapsed_seconds": {"9" * 400}}}]'
    )
    (result,) = _kept(tmp_path, summary, 1)
    assert result.passed is False
    with pytest.raises(OverflowError):
        result.elapsed_seconds  # noqa: B018


@pytest.mark.parametrize(
    "failure",
    [
        {"category": "collision", "at": "soon"},
        {"category": "collision", "also_triggered": 5},
        {"category": "collision", "at": [1]},
    ],
)
def test_a_failure_this_version_cannot_read_raises_when_read(
    tmp_path: Path, failure: dict[str, Any]
) -> None:
    entry = _entry("s0", "S0", passed=False, failure=failure)
    (result,) = _kept(tmp_path, json.dumps([entry]), 1)
    assert result.passed is False
    with pytest.raises((TypeError, ValueError)):
        result.failure  # noqa: B018


def test_a_failure_of_an_unknown_category(tmp_path: Path) -> None:
    entry = _entry("s0", "S0", failure={"category": "meteor", "at": None})
    (result,) = _kept(tmp_path, json.dumps([entry]), 1)
    failure = result.failure
    assert failure is not None
    assert failure.category is FailureCategory.OTHER
    assert failure.at == 0.0
    assert failure.condition == ""
    assert failure.detail is None


def test_to_dict_gives_the_entry_back_without_the_batch_s_own_keys(
    tmp_path: Path,
) -> None:
    entry = {
        "z_first": 1,
        "scenario": "s0",
        "passed": True,
        "overrides": None,
        "custom": {"nested": [1, 2.5, None, "é", {"k": False}]},
        "stem": "S0",
        "error": False,
        "pending": False,
        "message": "ok",
        "elapsed_seconds": 1,
        "": "empty key",
    }
    (result,) = _kept(tmp_path, json.dumps([entry]), 1)
    data = result.to_dict()
    assert list(data) == [
        "z_first",
        "passed",
        "custom",
        "pending",
        "message",
        "elapsed_seconds",
        "",
    ]
    assert data == {
        "z_first": 1,
        "passed": True,
        "custom": {"nested": [1, 2.5, None, "é", {"k": False}]},
        "pending": False,
        "message": "ok",
        "elapsed_seconds": 1,
        "": "empty key",
    }
    assert type(data["elapsed_seconds"]) is int
    assert type(data["custom"]["nested"][1]) is float


def test_the_last_of_a_repeated_key_is_kept(tmp_path: Path) -> None:
    summary = (
        '[{"scenario": "s0", "overrides": null, "stem": "S0", '
        '"message": "first", "passed": false, "message": "last"}]'
    )
    (result,) = _kept(tmp_path, summary, 1)
    assert result.message == "last"
    assert list(result.to_dict().items()) == [("message", "last"), ("passed", False)]


# -- The summary written from kept results (batch_report) -------------------

#: A summary of kept entries of every kind: written again, each keeps its
#: members (in their order, the batch's own keys rewritten), and numbers are
#: written as Python reads them.
_ODD_SUMMARY = r"""[
  {"message": "first", "scenario": "s0", "passed": true, "overrides": ["x=1"],
   "elapsed_seconds": 1E400, "stem": "S0", "error": true, "extra": [1, 2],
   "rerun_recording": "S0.rrd"},
  {"scenario": "s1", "overrides": ["x=2"], "stem": "S1", "passed": false,
   "message": "multi\nline é 😀 \u0000 \"q\"", "elapsed_seconds": 2.50,
   "failure": {"category": "timeout", "at": 1e2, "detail": null},
   "condition_statuses": [{"label": "Goal", "satisfied": false, "message": "x"}],
   "nan": NaN, "neg": -Infinity, "negzero": -0.0, "big": 123456789012345678901234567890,
   "float": 0.1000000000000000055511151231257827, "exp": 1.5e-7,
   "deep": {"a": {"b": [[], {}, [{}]]}}, "empty": {}, "list": [],
   "rerun_error": "conv", "rerun_recording": "S1.rrd", "pending": false},
  {"scenario": "s2", "overrides": ["x=3"], "stem": "S2", "passed": "yes",
   "error": false, "rerun_error": "", "elapsed_seconds": "slow",
   "condition_statuses": [{"label": 7, "satisfied": 1, "message": [1]}], "k": 1, "k": 2}
]"""

_ODD_WRITTEN = r"""[
  {
    "scenario": "s0",
    "overrides": [
      "x=1"
    ],
    "stem": "S0",
    "message": "first",
    "passed": true,
    "elapsed_seconds": Infinity,
    "extra": [
      1,
      2
    ],
    "rerun_recording": "S0.rrd",
    "error": true
  },
  {
    "scenario": "s1",
    "overrides": [
      "x=2"
    ],
    "stem": "S1",
    "passed": false,
    "message": "multi\nline é 😀 \u0000 \"q\"",
    "elapsed_seconds": 2.5,
    "failure": {
      "category": "timeout",
      "at": 100.0,
      "detail": null
    },
    "condition_statuses": [
      {
        "label": "Goal",
        "satisfied": false,
        "message": "x"
      }
    ],
    "nan": NaN,
    "neg": -Infinity,
    "negzero": -0.0,
    "big": 123456789012345678901234567890,
    "float": 0.1,
    "exp": 1.5e-07,
    "deep": {
      "a": {
        "b": [
          [],
          {},
          [
            {}
          ]
        ]
      }
    },
    "empty": {},
    "list": [],
    "rerun_error": "kept: S1",
    "rerun_recording": "S1.rrd",
    "pending": false
  },
  {
    "scenario": "s2",
    "overrides": [
      "x=3"
    ],
    "stem": "S2",
    "passed": "yes",
    "rerun_error": "",
    "elapsed_seconds": "slow",
    "condition_statuses": [
      {
        "label": 7,
        "satisfied": 1,
        "message": [
          1
        ]
      }
    ],
    "k": 2
  },
  {
    "scenario": "s3",
    "overrides": [
      "x=4"
    ],
    "stem": "S3",
    "passed": null,
    "message": "No result yet (the batch has not run this case)",
    "elapsed_seconds": null,
    "pending": true
  }
]"""


def _odd_kept(tmp_path: Path) -> tuple[list[str], list[list[str]], list[KeptResult]]:
    labels = ["s0", "s1", "s2", "s3"]
    overrides = [["x=1"], ["x=2"], ["x=3"], ["x=4"]]
    summary = json.loads(_ODD_SUMMARY)
    summary.append(_entry("s3", "S3", overrides=["x=4"], pending=True))
    text = (
        _ODD_SUMMARY[: _ODD_SUMMARY.rindex("]")]
        + ",\n"
        + json.dumps(summary[3])
        + "\n]"
    )
    directory = _write(tmp_path / "run", text)
    resumed = read_resumed_batch(directory, labels, overrides)
    assert resumed.first_pending() == 3
    assert resumed.stems == ["S0", "S1", "S2", "S3"]
    return labels, overrides, [KeptResult(e) for e in resumed.entries[:3]]


def test_kept_entries_are_written_again_as_they_were(tmp_path: Path) -> None:
    labels, overrides, kept = _odd_kept(tmp_path)
    errored = [i for i, result in enumerate(kept) if result.errored]
    assert errored == [0]
    path = write_batch_result_json(
        labels,
        kept,
        tmp_path / "out",
        overrides,
        errored=errored,
        rerun_errors={1: "kept: S1"},
        pending=True,
        stems=["S0", "S1", "S2", "S3"],
    )
    assert path.read_text(encoding="utf-8") == _ODD_WRITTEN


_ODD_PRINTED = """
============================================================
Batch Scenario Results
============================================================
  [ERROR] s0 (infs)
    Rerun: S0.rrd
  [FAIL] s1 (2.5s)
         [other]
         multi
         line é 😀 \x00 "q"
    [NG] Goal : x
    Rerun: S1.rrd
  [FAIL] s2 (0.0s)
    [NG] 7 : [1]
------------------------------------------------------------
Result JSON: {out}
Rerun recordings: 1/2 written (open one with: rerun <file.rrd>)
  [FAILED] s1: kept: S1
------------------------------------------------------------
1/3 scenarios passed, 1 errored
============================================================
"""


def test_kept_entries_are_listed_as_they_were(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels, overrides, kept = _odd_kept(tmp_path)
    out_dir = tmp_path / "out"
    passed = print_summary(
        labels[:3],
        kept,
        out_dir,
        overrides[:3],
        errored=[0],
        rerun_errors={1: "kept: S1"},
        stems=["S0", "S1", "S2"],
    )
    assert passed is False
    printed = capsys.readouterr().out
    assert printed == _ODD_PRINTED.format(out=out_dir / BATCH_RESULTS)


def test_a_kept_failure_this_version_cannot_read_is_listed_without_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _entry(
        "s0", "S0", passed=False, message="m", failure={"at": "soon"}, more=[1]
    )
    (result,) = _kept(tmp_path, json.dumps([entry]), 1)
    print_summary(["s0"], [result], tmp_path / "out", stems=["S0"])
    printed = capsys.readouterr().out
    assert "  [FAIL] s0 (0.0s)\n         m\n" in printed
    assert json.loads((tmp_path / "out" / BATCH_RESULTS).read_text()) == [
        {
            "scenario": "s0",
            "stem": "S0",
            "passed": False,
            "message": "m",
            "failure": {"at": "soon"},
            "more": [1],
        }
    ]


def test_a_kept_string_no_utf8_holds_fails_the_summary(tmp_path: Path) -> None:
    """A lone surrogate (an escape JSON allows) is read, and the summary
    written from it cannot be encoded."""
    summary = (
        '[{"scenario": "s0", "overrides": null, "stem": "S0", '
        '"note": "\\ud800", "message": "\\ud800"}]'
    )
    (result,) = _kept(tmp_path, summary, 1)
    assert result.message == "\ud800"
    assert result.to_dict() == {"note": "\ud800", "message": "\ud800"}
    with pytest.raises(UnicodeEncodeError):
        write_batch_result_json(["s0"], [result], tmp_path / "out")
