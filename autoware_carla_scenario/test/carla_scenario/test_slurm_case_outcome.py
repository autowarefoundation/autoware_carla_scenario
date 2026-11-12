"""How a SLURM batch's case outcome (``slurm/results/<i>.json``) is read.

Pinned against the outcomes a hand, another version, a crash or a killed task
may have left: each key of the record (``index``, ``label``, ``stem``,
``result``, ``errored``, ``submission``, ``task``, ``job_id``, ``host``,
``wall_seconds``, ``finished``, ``rerun_error``) and each key of its
``result`` (``ScenarioResult.to_dict()``: ``passed``, ``message``,
``elapsed_seconds``, ``condition_statuses``, ``rerun_recording``,
``failure``), missing or of every JSON type -- NaN, infinities, a number no
float holds, an integer no float holds exactly, non-ASCII text -- and a key
named twice.  Each is compared with what the readers of a batch made of the
outcome when it was read as a ``dict`` (the oracles below): the verdict, the
result ``collect`` reports (each of its properties, the error it raises),
what ``collect`` prints and writes byte for byte, what ``status``, ``wait``
(``submit.follow``), ``unfinished_cases``, the gate, the viewer's batch
status and its cases read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.batch_report import print_summary
from autoware_carla_scenario.failure import Failure, FailureCategory
from autoware_carla_scenario.slurm import cli, submit
from autoware_carla_scenario.slurm.collect import (
    RecordedResult,
    collect,
    missing_cases,
    unfinished_cases,
)
from autoware_carla_scenario.slurm.manifest import (
    MANIFEST_VERSION,
    BatchLayout,
    recorded_verdict,
    write_json,
)
from autoware_carla_scenario.slurm.settings import SlurmSettings

# -- The oracles: how the outcome was read as a dict -----------------------------


def _old_real_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        float(value)
    except OverflowError:
        return False
    return True


def _old_readable_status(status: Any) -> bool:
    return (
        isinstance(status, dict)
        and isinstance(status.get("label"), str)
        and isinstance(status.get("satisfied"), bool)
        and isinstance(status.get("message"), str)
    )


def _old_readable_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if not isinstance(result.get("message"), str):
        return False
    if not _old_real_number(result.get("elapsed_seconds")):
        return False
    statuses = result.get("condition_statuses")
    if statuses is not None and not (
        isinstance(statuses, list) and all(_old_readable_status(s) for s in statuses)
    ):
        return False
    failure = result.get("failure")
    if failure is None:
        return True
    if not isinstance(failure, dict):
        return False
    try:
        Failure.from_dict(failure)
    except (TypeError, ValueError, AttributeError, OverflowError):
        return False
    return True


def _old_verdict(record: Any) -> bool | None:
    if not isinstance(record, dict):
        return None
    result = record.get("result")
    if not isinstance(result, dict):
        return None
    passed = result.get("passed")
    if not isinstance(passed, bool):
        return None
    return passed if _old_readable_result(result) else None


class _OldStatus:
    def __init__(self, label: str, satisfied: bool, message: str) -> None:
        self.label = label
        self.satisfied = satisfied
        self.message = message


class _OldResult:
    """slurm.collect.RecordedResult as it read a dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    @property
    def passed(self) -> bool:
        return self.data.get("passed") is True

    @property
    def message(self) -> str:
        return str(self.data["message"])

    @property
    def elapsed_seconds(self) -> float:
        return float(self.data["elapsed_seconds"])

    @property
    def condition_statuses(self) -> list[_OldStatus]:
        return [
            _OldStatus(str(s["label"]), bool(s["satisfied"]), str(s["message"]))
            for s in self.data.get("condition_statuses") or []
        ]

    @property
    def rerun_recording(self) -> str | None:
        recording = self.data.get("rerun_recording")
        return None if recording is None else str(recording)

    @property
    def failure(self) -> Failure | None:
        failure = self.data.get("failure")
        return None if failure is None else Failure.from_dict(failure)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


def _old_error_result(message: str) -> _OldResult:
    failure = Failure("", FailureCategory.INFRASTRUCTURE, 0.0, message)
    return _OldResult(
        {
            "passed": False,
            "message": message,
            "elapsed_seconds": 0.0,
            "condition_statuses": [],
            "failure": failure.to_dict(),
        }
    )


# -- The corpus -------------------------------------------------------------------

#: A member left out of its object.
_ABSENT = "<absent>"

#: A value of every JSON type, as its text in the file.
_VALUES: list[str] = [
    _ABSENT,
    "null",
    "true",
    "false",
    "0",
    "1",
    "-1",
    "2",
    "1.5",
    "-0.0",
    "NaN",
    "Infinity",
    "-Infinity",
    "1e400",
    "123456789012345678901234567890",
    "1" + "0" * 400,
    '""',
    '"text"',
    '"\\u00fc ber \\u2603"',
    '"true"',
    '"r.rrd"',
    "[]",
    "[1]",
    '["a", "b"]',
    "{}",
    '{"k": "v"}',
    '{"label": "l", "satisfied": true, "message": "m"}',
]

_STATUS = '{"label": "reach", "satisfied": true, "message": "m"}'

#: Lists of condition statuses: their fields of every type.
_STATUS_LISTS: list[str] = [
    "[" + _STATUS + "]",
    '[{"label": "a", "satisfied": false, "message": "x", "details": '
    '{"d": [1, NaN, "\\u00fc"]}}, ' + _STATUS + "]",
    *[
        "["
        + json.dumps({"label": "l", "satisfied": True, "message": "m"})[:-1]
        + f', "{key}": {value}}}]'
        for key in ("label", "satisfied", "message")
        for value in ("null", "1", '"s"', "true", "[]", "{}")
    ],
    '[{"label": "l", "satisfied": true}]',
    '[{"satisfied": true, "message": "m"}]',
    "[1]",
    '["s"]',
    "[[]]",
    "[null]",
]

_FAILURE = (
    '{"condition": "fail[0](c)", "category": "safety_margin", "at": 1.5, '
    '"message": "too close", "detail": null, "also_triggered": ["x"]}'
)

#: Failures: their fields of every type.
_FAILURES: list[str] = [
    _FAILURE,
    "{}",
    '{"category": "nope"}',
    '{"category": 3}',
    '{"at": "x"}',
    '{"at": "2.5"}',
    '{"at": null}',
    '{"at": 1e400}',
    '{"at": ' + "1" + "0" * 400 + "}",
    '{"at": true}',
    '{"detail": 5}',
    '{"detail": "d\\u00fc"}',
    '{"also_triggered": 3}',
    '{"also_triggered": "ab"}',
    '{"also_triggered": [1, null]}',
    '{"condition": ["c"], "message": {"m": 1}}',
]


def _object(members: dict[str, str]) -> str:
    """The text of the object of *members* (each a value's JSON text)."""
    return (
        "{"
        + ", ".join(
            f"{json.dumps(key)}: {value}"
            for key, value in members.items()
            if value != _ABSENT
        )
        + "}"
    )


def _result(verdict: bool, **values: str) -> str:
    passed = verdict
    members = {
        "passed": "true" if passed else "false",
        "message": '"ok"' if passed else '"bad\\nworse \\u00fc"',
        "elapsed_seconds": "1.5",
        "condition_statuses": "[" + _STATUS + "]",
        "rerun_recording": '"r.rrd"',
        "failure": "null" if passed else _FAILURE,
    }
    members.update(values)
    return _object(members)


def _record(**values: str) -> str:
    members = {
        "index": "1",
        "label": '"s#2"',
        "stem": '"stem1"',
        "result": _result(True),
        "errored": "false",
        "submission": "0",
        "task": "0",
        "job_id": '"70"',
        "host": '"node\\u00fc"',
        "wall_seconds": "2.5",
        "finished": '"2026-11-12T00:00:00"',
    }
    members.update(values)
    return _object(members)


def _corpus() -> list[tuple[str, str]]:
    """(name, the text of case 1's outcome)."""
    corpus: list[tuple[str, str]] = []
    for passed in (True, False):
        for key in (
            "passed",
            "message",
            "elapsed_seconds",
            "condition_statuses",
            "rerun_recording",
            "failure",
        ):
            for value in _VALUES:
                corpus.append(
                    (
                        f"result-{passed}-{key}-{value}",
                        _record(result=_result(passed, **{key: value})),
                    )
                )
        for statuses in _STATUS_LISTS:
            corpus.append(
                (
                    f"statuses-{passed}-{statuses}",
                    _record(result=_result(passed, condition_statuses=statuses)),
                )
            )
        for failure in _FAILURES:
            corpus.append(
                (
                    f"failure-{passed}-{failure}",
                    _record(result=_result(passed, failure=failure)),
                )
            )
    for key in (
        "index",
        "label",
        "stem",
        "result",
        "errored",
        "submission",
        "task",
        "job_id",
        "host",
        "wall_seconds",
        "finished",
        "rerun_error",
    ):
        for value in _VALUES:
            corpus.append((f"record-{key}-{value}", _record(**{key: value})))
    for errored in ("true", '"x"', "1"):
        for rerun_error in ('"no space"', "[1]"):
            corpus.append(
                (
                    f"failed-{errored}-{rerun_error}",
                    _record(
                        result=_result(False), errored=errored, rerun_error=rerun_error
                    ),
                )
            )
    # Keys named twice: the first place and the last value.
    corpus += [
        (
            "twice-errored",
            '{"errored": true, "result": ' + _result(True) + ', "errored": false}',
        ),
        (
            "twice-result",
            '{"result": '
            + _result(True)
            + ', "errored": false, "result": '
            + _result(False)
            + "}",
        ),
        (
            "twice-in-result",
            '{"result": {"passed": false, "message": "a", "elapsed_seconds": 1, '
            '"passed": true, "message": "b"}}',
        ),
        (
            "twice-rerun-error",
            '{"rerun_error": "a", "result": ' + _result(True) + ', "rerun_error": ""}',
        ),
        ("only-result", '{"result": ' + _result(True) + "}"),
        ("empty-object", "{}"),
        (
            "extra-keys",
            '{"z": [1, {"y": NaN}], "result": '
            + _result(True, extra='{"\\u00fc": [Infinity, -0.0]}')
            + ', "a": null}',
        ),
        (
            "nested-details",
            _record(
                result=_result(
                    True,
                    condition_statuses='[{"label": "l", "satisfied": true, '
                    '"message": "m", "details": {"deep": [[[{"x": [1e-300, 5e-324]}]]]}}]',
                )
            ),
        ),
    ]
    # Files that hold no outcome: the case has none.
    corpus += [
        ("not-json", "{not json"),
        ("truncated", _record()[:-7]),
        ("empty", ""),
        ("array", "[" + _record() + "]"),
        ("string", '"x"'),
        ("number", "1"),
        ("null", "null"),
        ("trailing", _record() + " x"),
        ("integer-too-long", '{"result": {"passed": ' + "1" * 5000 + "}}"),
    ]
    return corpus


_CORPUS = _corpus()
_BYTES: dict[str, bytes] = {
    "bom": b"\xef\xbb\xbf" + _record().encode(),
    "not-utf-8": _record().encode().replace(b"s#2", b"s#\xff"),
    "latin-1": _record(label='"s\\u00fc"').replace("\\u00fc", "ü").encode("latin-1"),
}


def _old_record(content: bytes) -> dict[str, Any] | None:
    """The outcome as it was read: a dict, or None."""
    try:
        data = json.loads(content.decode("utf-8"))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# -- The batch ------------------------------------------------------------------


def _passing_record(index: int) -> str:
    return _record(index=str(index), label=f'"s#{index + 1}"', stem=f'"stem{index}"')


def _batch(tmp_path: Path, content: bytes) -> BatchLayout:
    """A batch of three cases, whose cases 0 and 2 passed and whose case 1
    left *content* as its outcome."""
    layout = BatchLayout(tmp_path / "batch")
    write_json(
        layout.manifest,
        {
            "version": MANIFEST_VERSION,
            "cases": [
                {"index": i, "label": f"s#{i + 1}", "stem": f"stem{i}"}
                for i in range(3)
            ],
            "launcher": {},
        },
    )
    layout.result(0).parent.mkdir(parents=True, exist_ok=True)
    layout.result(0).write_text(_passing_record(0), encoding="utf-8")
    layout.result(1).write_bytes(content)
    layout.result(2).write_text(_passing_record(2), encoding="utf-8")
    return layout


def _all_contents() -> list[tuple[str, bytes]]:
    return [(name, text.encode("utf-8")) for name, text in _CORPUS] + list(
        _BYTES.items()
    )


_CONTENTS = _all_contents()
_IDS = [name for name, _ in _CONTENTS]


def _outcome(call: Any) -> Any:
    """What *call* returns, or the type and text of what it raises."""
    try:
        return ("returns", call())
    except Exception as exc:  # noqa: BLE001
        return ("raises", type(exc).__name__, str(exc))


def _same(actual: Any, expected: Any) -> None:
    """Equal, NaN equal to NaN and -0.0 told from 0.0: compared as JSON text
    (repr for what JSON cannot write)."""

    def text(value: Any) -> str:
        try:
            return json.dumps(value, indent=1, ensure_ascii=False)
        except (TypeError, ValueError):
            return repr(value)

    assert text(actual) == text(expected)


def _recorded_result(layout: BatchLayout, case: int) -> Any:
    """The result collect reports for *case*'s outcome, as RecordedResult
    reads the record read_result gives."""
    record: Any = layout.read_result(case)
    # A record read as a dict, as it was before CaseOutcome.
    recorded_result: Any = RecordedResult
    if isinstance(record, dict):
        return recorded_result(dict(record["result"]))
    return recorded_result(record.result())


def _properties(result: Any) -> list[Any]:
    def statuses() -> list[tuple[str, bool, str]]:
        return [(s.label, s.satisfied, s.message) for s in result.condition_statuses]

    def failure() -> Any:
        failure = result.failure
        return None if failure is None else repr(failure)

    return [
        _outcome(lambda: result.passed),
        _outcome(lambda: result.message),
        _outcome(lambda: result.elapsed_seconds),
        _outcome(statuses),
        _outcome(lambda: result.rerun_recording),
        _outcome(failure),
        _outcome(lambda: json.dumps(result.to_dict(), indent=2, ensure_ascii=False)),
    ]


# -- The verdict and the reported result ------------------------------------------


def test_the_corpus_covers_every_kind_of_outcome() -> None:
    records = [_old_record(content) for _, content in _CONTENTS]
    verdicts = [_old_verdict(record) for record in records]
    assert verdicts.count(True) > 50
    assert verdicts.count(False) > 50
    assert verdicts.count(None) > 200
    assert records.count(None) == 12


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_verdict_of_each_outcome(tmp_path: Path, name: str, content: bytes) -> None:
    layout = _batch(tmp_path, content)
    old = _old_record(content)
    record = layout.read_result(1)
    assert (record is None) is (old is None)
    assert recorded_verdict(record) is _old_verdict(old)


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_reported_result_of_each_outcome(
    tmp_path: Path, name: str, content: bytes
) -> None:
    """RecordedResult's properties, of a result of any content: what each
    gives or raises."""
    layout = _batch(tmp_path, content)
    old = _old_record(content)
    if old is None or not isinstance(old.get("result"), dict):
        return
    _same(
        _properties(_recorded_result(layout, 1)), _properties(_OldResult(old["result"]))
    )


# -- What collect prints and writes -----------------------------------------------


def _no_verdict_message(layout: BatchLayout, case: int) -> str:
    return (
        f"No verdict: {layout.result(case)} holds no result whose `passed` "
        "is true or false, with a string `message`, a number "
        "`elapsed_seconds`, and `condition_statuses` and a `failure` that "
        "can be read. "
        f"Run the case again with: scenario-slurm resubmit {layout.batch_dir}"
    )


def _expected_collect(
    layout: BatchLayout, capsys: pytest.CaptureFixture[str]
) -> tuple[Any, ...]:
    """What collect printed and wrote of the batch, as the outcomes read as
    dicts made it: print_summary of the results the oracles read."""
    labels = ["s#1", "s#2", "s#3"]
    results: list[Any] = []
    errored: list[int] = []
    missing: list[int] = []
    unreadable: list[int] = []
    rerun_errors: dict[int, str] = {}
    target = str(layout.batch_dir)
    for case in range(3):
        record = _old_record(layout.result(case).read_bytes())
        if record is None:
            missing.append(case)
            results.append(
                _old_error_result(
                    "No result: the case did not finish under SLURM (it was "
                    f"never submitted). Run it again with: scenario-slurm "
                    f"resubmit {target}"
                )
            )
            continue
        if record.get("rerun_error"):
            rerun_errors[case] = str(record["rerun_error"])
        if _old_verdict(record) is None:
            unreadable.append(case)
            results.append(_old_error_result(_no_verdict_message(layout, case)))
            continue
        results.append(_OldResult(dict(record["result"])))
        if record.get("errored"):
            errored.append(case)
    all_passed = print_summary(
        labels,
        results,
        output_dir=Path(layout.batch_dir),
        overrides=None,
        errored=sorted({*errored, *missing, *unreadable}),
        rerun_errors=rerun_errors,
        stems=["stem0", "stem1", "stem2"],
    )
    if unreadable:
        print(  # noqa: T201
            f"{len(unreadable)} case(s) have a result with no verdict: "
            + ", ".join(labels[i] for i in unreadable)
            + f"\nRun them again with: scenario-slurm resubmit {target}"
        )
    if missing:
        print(  # noqa: T201
            f"{len(missing)} case(s) have no result: "
            + ", ".join(labels[i] for i in missing)
            + f"\nRun them again with: scenario-slurm resubmit {target}"
        )
    collected = json.dumps(
        {
            "cases": 3,
            "passed": sum(1 for r in results if r.passed),
            "errored": [labels[i] for i in sorted({*errored, *unreadable})],
            "missing": [labels[i] for i in missing],
        },
        indent=2,
        ensure_ascii=False,
    )
    captured = capsys.readouterr()
    return (
        0 if all_passed else 1,
        captured.out,
        captured.err,
        (layout.batch_dir / "batch_results.json").read_bytes(),
        collected + "\n",
    )


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_what_collect_prints_and_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, content: bytes
) -> None:
    layout = _batch(tmp_path, content)
    capsys.readouterr()
    code = collect(layout)
    captured = capsys.readouterr()
    actual = (
        code,
        captured.out,
        captured.err,
        (layout.batch_dir / "batch_results.json").read_bytes(),
        (layout.root / "collected.json").read_text(encoding="utf-8"),
    )
    assert actual == _expected_collect(layout, capsys)


def test_collect_writes_the_values_as_json_holds_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The summary's entry of a result is the merge ``{"scenario": ...,
    "stem": ..., **result}``, laid out by json.dumps: NaN, infinities, an
    integer no float holds, non-ASCII text, a condition's details."""
    content = _record(
        result=_result(
            True,
            elapsed_seconds="NaN",
            condition_statuses='[{"label": "l\\u00fc", "satisfied": true, '
            '"message": "m", "details": {"x": [Infinity, -0.0, '
            "123456789012345678901234567890, 1e400]}}]",
            extra='{"k": [], "j": {}}',
        )
    )
    layout = _batch(tmp_path, content.encode())
    assert collect(layout) == 0
    capsys.readouterr()
    entries = (layout.batch_dir / "batch_results.json").read_text(encoding="utf-8")
    records = [_passing_record(0), content, _passing_record(2)]
    expected = [
        {
            "scenario": f"s#{i + 1}",
            "stem": f"stem{i}",
            **json.loads(record)["result"],
        }
        for i, record in enumerate(records)
    ]
    assert entries == json.dumps(expected, indent=2, ensure_ascii=False)
    assert '"elapsed_seconds": NaN' in entries
    assert '"lü"' in entries
    assert "123456789012345678901234567890" in entries
    assert "Infinity,\n" in entries and "-0.0,\n" in entries


# -- What the batch's other readers make of it ------------------------------------


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_status_counts_each_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, content: bytes
) -> None:
    layout = _batch(tmp_path, content)
    old = _old_record(content)
    assert cli._status(layout) == 0
    done = 2 + (old is not None)
    passed = 2 + (old is not None and _old_verdict(old) is True)
    errored = int(old is not None and bool(old.get("errored")))
    assert capsys.readouterr().out == (
        f"Batch {layout.batch_dir}: 3 case(s)\n"
        f"  {done} with a result ({passed} passed, {errored} errored), "
        f"{3 - done} without\n"
    )


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_wait_counts_each_outcome(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    content: bytes,
) -> None:
    """``submit.follow``: a case is done when its outcome names the
    submission followed (``submission == 0``, of any type)."""
    layout = _batch(tmp_path, content)
    write_json(
        layout.submission(0), {"index": 0, "job_id": "70", "chunks": [[0, 1, 2]]}
    )
    monkeypatch.setattr(submit, "squeue_states", lambda job: {})
    old = _old_record(content)
    submit.follow(layout, 0, SlurmSettings(poll_seconds=0.1))
    first = capsys.readouterr().out.splitlines()[0]
    counted = old is not None and old.get("submission") == 0
    passed = 2 + (counted and _old_verdict(old) is True)
    assert first == (
        f"[slurm] job 70: {2 + counted}/3 case(s) done ({passed} passed); "
        "0 task(s) running, 0 pending"
    )


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_cases_to_run_again(tmp_path: Path, name: str, content: bytes) -> None:
    layout = _batch(tmp_path, content)
    old = _old_record(content)
    verdict = _old_verdict(old)
    again = old is None or verdict is None or bool(old.get("errored"))
    assert unfinished_cases(layout) == ([1] if again else [])
    assert unfinished_cases(layout, failed=True) == (
        [1] if again or not verdict else []
    )
    assert missing_cases(layout) == ([1] if old is None else [])


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_gate_s_unrecorded_cases(tmp_path: Path, name: str, content: bytes) -> None:
    from autoware_carla_scenario.gate.evaluate import _slurm_unrecorded

    layout = _batch(tmp_path, content)
    old = _old_record(content)
    recorded = (
        old is not None and _old_verdict(old) is not None and not old.get("errored")
    )
    assert _slurm_unrecorded(layout.batch_dir) == ([] if recorded else ["s#2"], 3)


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_viewer_s_batch_status(tmp_path: Path, name: str, content: bytes) -> None:
    from autoware_carla_scenario.ui import slurm_launch

    layout = _batch(tmp_path, content)
    old = _old_record(content)
    status = slurm_launch.batch_status(layout.batch_dir)
    assert status is not None
    case = status["cases"][1]
    if old is None:
        assert (case["state"], case["message"]) == ("missing", "never submitted")
        return
    recorded = old.get("result")
    result = recorded if isinstance(recorded, dict) else {}
    errored = bool(old.get("errored"))
    passed = not errored and _old_verdict(old) is True
    _same(
        case,
        {
            "index": 1,
            "label": "s#2",
            "stem": "stem1",
            "overrides": [],
            "state": "errored" if errored else "passed" if passed else "failed",
            "message": result.get("message", ""),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "host": old.get("host"),
            "job_id": old.get("job_id"),
        },
    )


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_viewer_s_cases(tmp_path: Path, name: str, content: bytes) -> None:
    """ui.cases reads the outcome itself, and its verdict as a batch's
    other readers do."""
    from autoware_carla_scenario.ui.cases import _slurm_entries

    layout = _batch(tmp_path, content)
    old = _old_record(content)
    entries = _slurm_entries(layout.batch_dir)
    assert entries is not None
    entry: dict[str, Any] = {"scenario": "s#2", "overrides": []}
    if isinstance(old, dict) and isinstance(old.get("result"), dict):
        entry.update(old["result"])
        if old.get("errored"):
            entry["error"] = True
        if _old_verdict(old) is None:
            entry.update(
                passed=False,
                error=True,
                elapsed_seconds=None,
                message=f"No verdict: {layout.result(1)} holds no result that "
                "can be reported",
            )
    else:
        entry.update(
            passed=None,
            message="No result yet (running under SLURM)",
            elapsed_seconds=None,
            pending=True,
        )
    _same(entries[1], entry)


def test_nan_and_negative_zero_are_compared_as_json_writes_them() -> None:
    _same([math.nan, -0.0], [math.nan, -0.0])
    with pytest.raises(AssertionError):
        _same([-0.0], [0.0])
