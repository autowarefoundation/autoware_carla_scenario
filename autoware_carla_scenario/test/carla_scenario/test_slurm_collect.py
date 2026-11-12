"""What ``slurm.collect`` says of a SLURM batch's cases that have no outcome.

Pinned against the records a hand, another version, a crash or a killed
submitter may have left: a submission's ``job_id``, ``submitted_at``,
``sbatch_timed_out`` and ``sbatch_timed_out_at``, and a task's ``state`` and
``error``, of every JSON type, missing or unreadable.  A missing case's
message, the advice it gives (#1801, #1803), the submission that advice is
about, the error raised where the records cannot be read, and how often the
clock is read, are compared whole; so are what ``collect`` prints and writes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.slurm import manifest
from autoware_carla_scenario.slurm.collect import (
    _missing_message,
    collect,
    missing_cases,
    unfinished_cases,
)
from autoware_carla_scenario.slurm.manifest import (
    MANIFEST_VERSION,
    BatchLayout,
    write_json,
)

#: What the fake clock reads.
_NOW = 1_800_000_000.0

_TARGET = "/b"
_WAIT = "scenario-slurm wait /b"
_RESUBMIT = "Run it again with: scenario-slurm resubmit /b"
_RUNNING = (
    "If `squeue --me` still lists the task, it is still queued or running: "
    "wait for it with: scenario-slurm wait /b; once it has ended, run the case "
    "again with: scenario-slurm resubmit /b"
)
_TIMED_OUT = (
    "If `squeue --me` lists the job, collect the batch once it ends: "
    "scenario-slurm collect /b; if not, run the case again with: "
    "scenario-slurm resubmit /b (a resubmit is refused until 600 s after the "
    "record was made, unless with --force)"
)
_IN_FLIGHT = (
    "Its submission is being submitted; try again once sbatch returns: "
    "scenario-slurm wait /b"
)
_NOTE_IN_FLIGHT = "no job id yet (sbatch in flight)"
_NOTE_STALE = (
    "no job id: its submitter stopped before sbatch returned, more than 600 s "
    "ago (a stale record)"
)
_NOTE_TIMED_OUT = (
    "no job id: sbatch timed out; the job may exist all the same: check "
    "`squeue --me`"
)


class _Clock:
    """``time.time()`` for slurm.manifest: always :data:`_NOW`, counting how
    often it is read."""

    def __init__(self) -> None:
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return _NOW


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(manifest, "time", SimpleNamespace(time=fake))
    return fake


def _layout(tmp_path: Path, cases: int = 1) -> BatchLayout:
    layout = BatchLayout(tmp_path / "batch")
    write_json(
        layout.manifest,
        {
            "version": MANIFEST_VERSION,
            "cases": [{"index": i, "label": f"s#{i + 1}"} for i in range(cases)],
            "launcher": {},
        },
    )
    return layout


def _set_task(layout: BatchLayout, status: Any, submission: int = 0) -> None:
    """Task 0 of *submission*'s status: none when *status* is ``_ABSENT``."""
    path = layout.task(submission, 0)
    if status is _ABSENT:
        path.unlink(missing_ok=True)
    else:
        write_json(path, status)


def _message(layout: BatchLayout, clock: _Clock, case: int = 0) -> tuple[Any, ...]:
    """(message, advice, about, clock reads), or (exception type, its text,
    clock reads) where it raises."""
    clock.reads = 0
    try:
        message, advice, about = _missing_message(layout, case, _TARGET)
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__, str(exc), clock.reads)
    return (message, advice, about, clock.reads)


def _no_result(detail: str, then: str) -> str:
    return f"No result: the case did not finish under SLURM ({detail}). {then}"


_ABSENT = object()


# -- Each piece of advice, whole ---------------------------------------------


@pytest.mark.parametrize(
    ("submission", "status", "expected"),
    [
        # A job, its task queued (no status yet) or unfinished: wait.
        (
            {"job_id": "77"},
            _ABSENT,
            (
                _no_result("task 0 of job 77 never started", _RUNNING),
                "running",
                0,
                0,
            ),
        ),
        *[
            (
                {"job_id": "77"},
                {"state": state},
                (_no_result(f"task 0 of job 77 {state}", _RUNNING), "running", 0, 0),
            )
            for state in ("starting", "waiting", "building", "running")
        ],
        # A status that is not an object reads as none: never started.
        (
            {"job_id": "77"},
            ["running"],
            (
                _no_result("task 0 of job 77 never started", _RUNNING),
                "running",
                0,
                0,
            ),
        ),
        # Its error, after the state, whatever the state.
        (
            {"job_id": 77},
            {"state": "running", "error": "out of memory"},
            (
                _no_result("task 0 of job 77 running: out of memory", _RUNNING),
                "running",
                0,
                0,
            ),
        ),
        # A job whose task finished: run it again.
        (
            {"job_id": "77"},
            {"state": "failed", "error": "boom"},
            (
                _no_result("task 0 of job 77 failed: boom", _RESUBMIT),
                "resubmit",
                None,
                0,
            ),
        ),
        (
            {"job_id": "77"},
            {"error": "boom"},
            (_no_result("task 0 of job 77 ?: boom", _RESUBMIT), "resubmit", None, 0),
        ),
        # States of the wrong type are no unfinished state.
        (
            {"job_id": "77"},
            {"state": 5},
            (_no_result("task 0 of job 77 5", _RESUBMIT), "resubmit", None, 0),
        ),
        (
            {"job_id": "77"},
            {"state": None, "error": 0},
            (_no_result("task 0 of job 77 None", _RESUBMIT), "resubmit", None, 0),
        ),
        (
            {"job_id": "77"},
            {"state": "RUNNING"},
            (_no_result("task 0 of job 77 RUNNING", _RESUBMIT), "resubmit", None, 0),
        ),
        # ... but one no set can hold raises.
        (
            {"job_id": "77"},
            {"state": ["running"]},
            ("TypeError", "unhashable type: 'list'", 0),
        ),
        (
            {"job_id": "77"},
            {"state": {"a": 1}},
            ("TypeError", "unhashable type: 'dict'", 0),
        ),
        # No job, its sbatch in flight: try again once it returns.
        (
            {"job_id": None, "submitted_at": _NOW},
            _ABSENT,
            (
                _no_result(
                    f"task 0 of submission 0 never started; {_NOTE_IN_FLIGHT}",
                    _IN_FLIGHT,
                ),
                "in_flight",
                0,
                2,
            ),
        ),
        # ... in flight for SBATCH_IN_FLIGHT_SECONDS, not one more.
        (
            {"job_id": "", "submitted_at": _NOW - 599.5},
            {"state": "running"},
            (
                _no_result(
                    f"task 0 of submission 0 running; {_NOTE_IN_FLIGHT}", _IN_FLIGHT
                ),
                "in_flight",
                0,
                2,
            ),
        ),
        (
            {"submitted_at": _NOW - 600},
            {"state": "running", "error": "x"},
            (
                _no_result(
                    f"task 0 of submission 0 running; {_NOTE_STALE}: x", _RESUBMIT
                ),
                "resubmit",
                None,
                2,
            ),
        ),
        # No job, its sbatch timed out, the record young: collect or resubmit
        # later.
        (
            {
                "job_id": None,
                "submitted_at": _NOW - 10,
                "sbatch_timed_out": True,
                "sbatch_timed_out_at": "2026-11-06T12:00:00",
            },
            _ABSENT,
            (
                _no_result(
                    "task 0 of submission 0 never started; no job id: sbatch "
                    "timed out at 2026-11-06T12:00:00; the job may exist all the "
                    "same: check `squeue --me`",
                    _TIMED_OUT,
                ),
                "timed_out",
                0,
                1,
            ),
        ),
        # ... the record old: run it again.
        (
            {"job_id": None, "submitted_at": _NOW - 660, "sbatch_timed_out": True},
            {"state": "failed"},
            (
                _no_result(
                    f"task 0 of submission 0 failed; {_NOTE_TIMED_OUT}", _RESUBMIT
                ),
                "resubmit",
                None,
                2,
            ),
        ),
        # ... no time it was submitted: run it again, the clock unread.
        (
            {"job_id": None, "sbatch_timed_out": True, "sbatch_timed_out_at": ""},
            _ABSENT,
            (
                _no_result(
                    f"task 0 of submission 0 never started; {_NOTE_TIMED_OUT}",
                    _RESUBMIT,
                ),
                "resubmit",
                None,
                0,
            ),
        ),
        # Only `true` is a timeout.
        (
            {"job_id": None, "submitted_at": _NOW, "sbatch_timed_out": "true"},
            _ABSENT,
            (
                _no_result(
                    f"task 0 of submission 0 never started; {_NOTE_IN_FLIGHT}",
                    _IN_FLIGHT,
                ),
                "in_flight",
                0,
                2,
            ),
        ),
        # A time no float holds raises once the clock is read.
        (
            {"job_id": None, "submitted_at": 10**400},
            _ABSENT,
            ("OverflowError", "int too large to convert to float", 1),
        ),
        # A record made infinitely far ahead is in flight for ever.
        (
            {"job_id": None, "submitted_at": math.inf},
            _ABSENT,
            (
                _no_result(
                    f"task 0 of submission 0 never started; {_NOTE_IN_FLIGHT}",
                    _IN_FLIGHT,
                ),
                "in_flight",
                0,
                2,
            ),
        ),
        (
            {"job_id": False, "submitted_at": math.nan, "sbatch_timed_out": True},
            _ABSENT,
            (
                _no_result(
                    f"task 0 of submission 0 never started; {_NOTE_TIMED_OUT}",
                    _RESUBMIT,
                ),
                "resubmit",
                None,
                2,
            ),
        ),
        # A job id of any truthy type is a job.
        (
            {"job_id": [1], "submitted_at": _NOW, "sbatch_timed_out": True},
            {"state": "done"},
            (_no_result("task 0 of job [1] done", _RESUBMIT), "resubmit", None, 0),
        ),
    ],
)
def test_a_missing_case_s_message_and_advice(
    tmp_path: Path,
    clock: _Clock,
    submission: dict[str, Any],
    status: Any,
    expected: tuple[Any, ...],
) -> None:
    layout = _layout(tmp_path)
    write_json(layout.submission(0), {"index": 0, "chunks": [[0]], **submission})
    _set_task(layout, status)
    assert _message(layout, clock) == expected


def test_a_case_never_submitted_is_to_run_again(tmp_path: Path, clock: _Clock) -> None:
    layout = _layout(tmp_path, cases=2)
    assert _message(layout, clock) == (
        _no_result("it was never submitted", _RESUBMIT),
        "resubmit",
        None,
        0,
    )
    # Neither by a submission that does not give it, nor by one that cannot
    # be read (#1356), nor by one whose chunks are none.
    write_json(layout.submission(0), {"index": 0, "job_id": "1", "chunks": [[1]]})
    write_json(layout.submission(1), {"index": 1, "job_id": "2", "chunks": None})
    layout.submission(2).write_text("not json", encoding="utf-8")
    write_json(layout.submission(3), [{"chunks": [[0]]}])
    assert _message(layout, clock)[:3] == (
        _no_result("it was never submitted", _RESUBMIT),
        "resubmit",
        None,
    )


def test_a_missing_case_names_its_task_and_its_last_submission(
    tmp_path: Path, clock: _Clock
) -> None:
    """The last submission that gave the case is the one the message is
    about; ``wait`` is told which unless it is the batch's latest."""
    layout = _layout(tmp_path, cases=3)
    write_json(
        layout.submission(0), {"index": 0, "job_id": "70", "chunks": [[0], [1, 2]]}
    )
    write_json(layout.submission(1), {"index": 1, "job_id": "71", "chunks": [[2]]})
    write_json(layout.task(0, 1), {"state": "running"})
    write_json(layout.task(1, 0), {"state": "waiting"})
    older = (
        "If `squeue --me` still lists the task, it is still queued or running: "
        "wait for it with: scenario-slurm wait --submission 0 /b; once it has "
        "ended, run the case again with: scenario-slurm resubmit /b"
    )
    assert _message(layout, clock, 1) == (
        _no_result("task 1 of job 70 running", older),
        "running",
        0,
        0,
    )
    assert _message(layout, clock, 2) == (
        _no_result("task 0 of job 71 waiting", _RUNNING),
        "running",
        1,
        0,
    )
    # An in-flight older submission: wait is told which, too.
    write_json(
        layout.submission(0),
        {"index": 0, "job_id": None, "submitted_at": _NOW, "chunks": [[0], [1]]},
    )
    assert _message(layout, clock, 1) == (
        _no_result(
            f"task 1 of submission 0 running; {_NOTE_IN_FLIGHT}",
            "Its submission is being submitted; try again once sbatch returns: "
            "scenario-slurm wait --submission 0 /b",
        ),
        "in_flight",
        0,
        2,
    )


# -- Every combination, against the decision table -----------------------------

_UNFINISHED = ("starting", "waiting", "building", "running")

_JOBS: list[Any] = [
    _ABSENT,
    None,
    "",
    0,
    False,
    [],
    {},
    0.0,
    "77",
    77,
    True,
    [1],
    {"a": 1},
    1.5,
    "0",
]
_STATES: list[Any] = [
    _ABSENT,
    *_UNFINISHED,
    "done",
    "failed",
    "",
    None,
    5,
    True,
    1.5,
    [1],
    {"a": 1},
    "RUNNING",
    "running ",
]
_ERRORS: list[Any] = [_ABSENT, None, "", 0, False, [], "boom", 7, [1], {"k": "v"}]
_SUBMITTED_AT: list[Any] = [
    _ABSENT,
    None,
    _NOW,
    _NOW - 599.5,
    _NOW - 600,
    _NOW - 660,
    _NOW + 5,
    "x",
    True,
    False,
    0,
    math.inf,
    -math.inf,
    math.nan,
    10**400,
    int(_NOW),
]
_TIMED_OUT_FLAGS: list[Any] = [_ABSENT, True, False, "true", 1, None]
_TIMED_OUT_AT: list[Any] = [_ABSENT, "", "2026-11-06T12:00:00", 0, 5, None]


def _statuses() -> list[Any]:
    """Statuses: none, one that is not an object, and each state with each
    error."""
    statuses: list[Any] = [_ABSENT, ["running"], "running"]
    for state in _STATES:
        for error in _ERRORS:
            status: dict[str, Any] = {"task": 0}
            if state is not _ABSENT:
                status["state"] = state
            if error is not _ABSENT:
                status["error"] = error
            statuses.append(status)
    return statuses


def _submission(job: Any, at: Any, flag: Any, flag_at: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"index": 0, "chunks": [[0]]}
    for key, value in (
        ("job_id", job),
        ("submitted_at", at),
        ("sbatch_timed_out", flag),
        ("sbatch_timed_out_at", flag_at),
    ):
        if value is not _ABSENT:
            record[key] = value
    return record


def _expected(submission: dict[str, Any], status: Any) -> tuple[Any, ...]:
    """What the decision table of #1801 and #1803 says of a missing case the
    batch's latest submission (0) gave its task 0."""
    status = status if isinstance(status, dict) else None
    job = submission.get("job_id")
    state = "never started" if status is None else status.get("state", "?")
    if job:
        detail = f"task 0 of job {job} {state}"
        reads = 0
    else:
        started = submission.get("submitted_at")
        timed_out = submission.get("sbatch_timed_out") is True
        if isinstance(started, (int, float)):
            if isinstance(started, int) and abs(started) > 2**1024:
                return ("OverflowError", "int too large to convert to float", 1)
            in_flight = _NOW - float(started) < 600
            reads = 1 if timed_out and in_flight else 2
        else:
            in_flight = False
            reads = 0
        if timed_out:
            at = submission.get("sbatch_timed_out_at")
            note = (
                "no job id: sbatch timed out"
                + (f" at {at}" if at else "")
                + "; the job may exist all the same: check `squeue --me`"
            )
        elif in_flight:
            note = _NOTE_IN_FLIGHT
        else:
            note = _NOTE_STALE
        detail = f"task 0 of submission 0 {state}; {note}"
    if status is not None and status.get("error"):
        detail += f": {status['error']}"
    if (
        job
        and status is not None
        and not isinstance(state, (str, int, float, type(None)))
    ):
        return ("TypeError", f"unhashable type: '{type(state).__name__}'", 0)
    if job and (status is None or state in _UNFINISHED):
        return (_no_result(detail, _RUNNING), "running", 0, reads)
    if not job and timed_out and in_flight:
        return (_no_result(detail, _TIMED_OUT), "timed_out", 0, reads)
    if not job and in_flight:
        return (_no_result(detail, _IN_FLIGHT), "in_flight", 0, reads)
    return (_no_result(detail, _RESUBMIT), "resubmit", None, reads)


def test_every_job_and_status_against_the_table(tmp_path: Path, clock: _Clock) -> None:
    layout = _layout(tmp_path)
    sbatches = [
        (_ABSENT, _ABSENT, _ABSENT),
        (_NOW, _ABSENT, _ABSENT),
        (_NOW - 660, _ABSENT, _ABSENT),
        (_NOW, True, "2026-11-06T12:00:00"),
        (_NOW - 660, True, _ABSENT),
    ]
    checked = 0
    for job in _JOBS:
        for at, flag, flag_at in sbatches:
            submission = _submission(job, at, flag, flag_at)
            write_json(layout.submission(0), submission)
            for status in _statuses():
                _set_task(layout, status)
                got = _message(layout, clock)
                assert got == _expected(submission, status), (submission, status)
                checked += 1
    assert checked == len(_JOBS) * 5 * (3 + len(_STATES) * len(_ERRORS))


def test_every_job_and_sbatch_against_the_table(tmp_path: Path, clock: _Clock) -> None:
    layout = _layout(tmp_path)
    statuses: list[Any] = [
        _ABSENT,
        {"state": "running"},
        {"state": "done", "error": "boom"},
        {"state": [1]},
    ]
    for job in _JOBS:
        for at in _SUBMITTED_AT:
            for flag in _TIMED_OUT_FLAGS:
                for flag_at in _TIMED_OUT_AT if flag is True else [_ABSENT]:
                    submission = _submission(job, at, flag, flag_at)
                    write_json(layout.submission(0), submission)
                    for status in statuses:
                        _set_task(layout, status)
                        got = _message(layout, clock)
                        assert got == _expected(submission, status), (
                            submission,
                            status,
                        )


# -- What collect prints and writes ---------------------------------------------


def _mixed_batch(tmp_path: Path) -> BatchLayout:
    """A batch of eight cases: passed, failed, errored, without a verdict,
    and four without an outcome, each with other advice."""
    layout = BatchLayout(tmp_path / "batch")
    write_json(
        layout.manifest,
        {
            "version": MANIFEST_VERSION,
            "cases": [
                {
                    "index": i,
                    "label": f"s#{i + 1}",
                    "stem": f"stem{i}",
                    "case_overrides": [f"x={i}"],
                }
                for i in range(8)
            ],
            "launcher": {},
            "expand": True,
        },
    )
    write_json(
        layout.submission(0),
        {"index": 0, "job_id": "70", "chunks": [[0, 1, 2, 3], [4], [5]]},
    )
    write_json(layout.task(0, 1), {"state": "running"})
    write_json(layout.task(0, 2), {"state": "failed", "error": "killed"})
    write_json(
        layout.submission(1),
        {"index": 1, "job_id": None, "submitted_at": _NOW, "chunks": [[6]]},
    )
    failure = {
        "condition": "c",
        "category": "safety_margin",
        "at": 1.5,
        "message": "too close",
    }

    def result(passed: bool, message: str, **more: Any) -> dict[str, Any]:
        return {
            "passed": passed,
            "message": message,
            "elapsed_seconds": 2.25,
            "condition_statuses": [
                {"label": "reach", "satisfied": passed, "message": "m"}
            ],
            "failure": None if passed else failure,
            **more,
        }

    write_json(layout.result(0), {"result": result(True, "ok")})
    write_json(
        layout.result(1),
        {"result": result(False, "bad\nworse"), "rerun_error": "no space"},
    )
    write_json(layout.result(2), {"result": result(False, "raised"), "errored": True})
    write_json(layout.result(3), {"result": {"passed": "false"}})
    return layout


def test_collect_prints_and_writes_a_mixed_batch(
    tmp_path: Path, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _mixed_batch(tmp_path)
    target = str(layout.batch_dir)
    assert collect(layout) == 1
    out = capsys.readouterr().out.replace(target, "<B>")
    assert out == _MIXED_OUT
    results = json.loads(
        (layout.batch_dir / "batch_results.json").read_text().replace(target, "<B>")
    )
    assert results == _MIXED_RESULTS
    collected = (layout.root / "collected.json").read_text()
    assert collected == (
        "{\n"
        '  "cases": 8,\n'
        '  "passed": 1,\n'
        '  "errored": [\n'
        '    "s#3",\n'
        '    "s#4"\n'
        "  ],\n"
        '  "missing": [\n'
        '    "s#5",\n'
        '    "s#6",\n'
        '    "s#7",\n'
        '    "s#8"\n'
        "  ]\n"
        "}\n"
    )
    assert missing_cases(layout) == [4, 5, 6, 7]
    assert unfinished_cases(layout) == [2, 3, 4, 5, 6, 7]
    assert unfinished_cases(layout, failed=True) == [1, 2, 3, 4, 5, 6, 7]
    # Quietly: the summary, without the hints.
    assert collect(layout, quiet_hint=True) == 1
    quiet = capsys.readouterr().out.replace(target, "<B>")
    assert quiet == _MIXED_OUT[: _MIXED_OUT.index("1 case(s) have a result")]


@pytest.mark.parametrize(
    ("submissions", "hint"),
    [
        # Of each advice, the first of running, in flight, timed out wins.
        (
            [
                {"job_id": None, "submitted_at": _NOW, "sbatch_timed_out": True},
                {"job_id": None, "submitted_at": _NOW},
                {"job_id": "72", "_state": "running"},
            ],
            "Some may still be queued or running: if `squeue --me` lists their "
            "tasks, wait for them with: scenario-slurm wait /b; once they have "
            "ended, run them again with: scenario-slurm resubmit /b",
        ),
        (
            [
                {"job_id": None, "submitted_at": _NOW, "sbatch_timed_out": True},
                {"job_id": None, "submitted_at": _NOW},
                {"job_id": None, "submitted_at": _NOW},
            ],
            "A submission of this batch is being submitted; try again once "
            "sbatch returns: scenario-slurm wait --submission 1 /b and "
            "scenario-slurm wait /b",
        ),
        (
            [
                {"job_id": None, "submitted_at": _NOW, "sbatch_timed_out": True},
                {"job_id": "71"},
                {"job_id": None},
            ],
            "An sbatch of this batch timed out: if `squeue --me` lists its job, "
            "collect the batch once it ends: scenario-slurm collect /b; if not, "
            "run them again with: scenario-slurm resubmit /b (a resubmit is "
            "refused until 600 s after the record was made, unless with --force)",
        ),
        (
            [
                {"job_id": None, "submitted_at": _NOW - 660},
                {"job_id": "71"},
                {"job_id": None},
            ],
            "Run them again with: scenario-slurm resubmit /b",
        ),
    ],
)
def test_collect_hints_the_first_advice(
    tmp_path: Path,
    clock: _Clock,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    submissions: list[dict[str, Any]],
    hint: str,
) -> None:
    from autoware_carla_scenario.slurm import collect as collect_module

    monkeypatch.setattr(collect_module, "resubmit_target", lambda *_a: _TARGET)
    layout = _layout(tmp_path, cases=3)
    for index, submission in enumerate(submissions):
        record = {"index": index, "chunks": [[index]], **submission}
        state = record.pop("_state", "done")
        write_json(layout.submission(index), record)
        write_json(layout.task(index, 0), {"state": state})
    assert collect(layout) == 1
    out = capsys.readouterr().out
    assert out.endswith(f"3 case(s) have no result: s#1, s#2, s#3\n{hint}\n")


def _infrastructure_error(index: int, message: str) -> dict[str, Any]:
    """The entry of case *index*, which has no outcome to report."""
    return {
        "scenario": f"s#{index + 1}",
        "overrides": [f"x={index}"],
        "stem": f"stem{index}",
        "passed": False,
        "message": message,
        "elapsed_seconds": 0.0,
        "condition_statuses": [],
        "failure": {
            "condition": "",
            "category": "infrastructure",
            "at": 0.0,
            "message": message,
            "detail": None,
            "also_triggered": [],
        },
        "error": True,
    }


#: What collect printed and wrote of _mixed_batch() (the batch's directory
#: written <B>), before the advice was split from the records it reads.
_MIXED_OUT = (
    "\n"
    "============================================================\n"
    "Batch Scenario Results\n"
    "============================================================\n"
    "  [PASS] s#1 (2.2s)\n"
    "    [OK] reach : m\n"
    "  [FAIL] s#2 (2.2s)\n"
    "         [safety_margin]\n"
    "         bad\n"
    "         worse\n"
    "    [NG] reach : m\n"
    "  [ERROR] s#3 (2.2s)\n"
    "         [safety_margin]\n"
    "         raised\n"
    "    [NG] reach : m\n"
    "  [ERROR] s#4 (0.0s)\n"
    "         [infrastructure]\n"
    "         No verdict: <B>/slurm/results/3.json holds no result whose `passed` is true or false, with a string `message`, a number `elapsed_seconds`, and `condition_statuses` and a `failure` that can be read. Run the case again with: scenario-slurm resubmit <B>\n"
    "  [ERROR] s#5 (0.0s)\n"
    "         [infrastructure]\n"
    "         No result: the case did not finish under SLURM (task 1 of job 70 running). If `squeue --me` still lists the task, it is still queued or running: wait for it with: scenario-slurm wait --submission 0 <B>; once it has ended, run the case again with: scenario-slurm resubmit <B>\n"
    "  [ERROR] s#6 (0.0s)\n"
    "         [infrastructure]\n"
    "         No result: the case did not finish under SLURM (task 2 of job 70 failed: killed). Run it again with: scenario-slurm resubmit <B>\n"
    "  [ERROR] s#7 (0.0s)\n"
    "         [infrastructure]\n"
    "         No result: the case did not finish under SLURM (task 0 of submission 1 never started; no job id yet (sbatch in flight)). Its submission is being submitted; try again once sbatch returns: scenario-slurm wait <B>\n"
    "  [ERROR] s#8 (0.0s)\n"
    "         [infrastructure]\n"
    "         No result: the case did not finish under SLURM (it was never submitted). Run it again with: scenario-slurm resubmit <B>\n"
    "------------------------------------------------------------\n"
    "Result JSON: <B>/batch_results.json\n"
    "------------------------------------------------------------\n"
    "1/8 scenarios passed, 6 errored\n"
    "============================================================\n"
    "1 case(s) have a result with no verdict: s#4\n"
    "Run them again with: scenario-slurm resubmit <B>\n"
    "4 case(s) have no result: s#5, s#6, s#7, s#8\n"
    "Some may still be queued or running: if `squeue --me` lists their tasks, wait for them with: scenario-slurm wait --submission 0 <B>; once they have ended, run them again with: scenario-slurm resubmit <B>\n"
)

_MIXED_RESULTS: list[dict[str, Any]] = [
    {
        "scenario": "s#1",
        "overrides": ["x=0"],
        "stem": "stem0",
        "passed": True,
        "message": "ok",
        "elapsed_seconds": 2.25,
        "condition_statuses": [{"label": "reach", "satisfied": True, "message": "m"}],
        "failure": None,
    },
    {
        "scenario": "s#2",
        "overrides": ["x=1"],
        "stem": "stem1",
        "passed": False,
        "message": "bad\nworse",
        "elapsed_seconds": 2.25,
        "condition_statuses": [{"label": "reach", "satisfied": False, "message": "m"}],
        "failure": {
            "condition": "c",
            "category": "safety_margin",
            "at": 1.5,
            "message": "too close",
        },
        "rerun_error": "no space",
    },
    {
        "scenario": "s#3",
        "overrides": ["x=2"],
        "stem": "stem2",
        "passed": False,
        "message": "raised",
        "elapsed_seconds": 2.25,
        "condition_statuses": [{"label": "reach", "satisfied": False, "message": "m"}],
        "failure": {
            "condition": "c",
            "category": "safety_margin",
            "at": 1.5,
            "message": "too close",
        },
        "error": True,
    },
    _infrastructure_error(
        3,
        "No verdict: <B>/slurm/results/3.json holds no result whose `passed` is true or false, with a string `message`, a number `elapsed_seconds`, and `condition_statuses` and a `failure` that can be read. Run the case again with: scenario-slurm resubmit <B>",
    ),
    _infrastructure_error(
        4,
        "No result: the case did not finish under SLURM (task 1 of job 70 running). If `squeue --me` still lists the task, it is still queued or running: wait for it with: scenario-slurm wait --submission 0 <B>; once it has ended, run the case again with: scenario-slurm resubmit <B>",
    ),
    _infrastructure_error(
        5,
        "No result: the case did not finish under SLURM (task 2 of job 70 failed: killed). Run it again with: scenario-slurm resubmit <B>",
    ),
    _infrastructure_error(
        6,
        "No result: the case did not finish under SLURM (task 0 of submission 1 never started; no job id yet (sbatch in flight)). Its submission is being submitted; try again once sbatch returns: scenario-slurm wait <B>",
    ),
    _infrastructure_error(
        7,
        "No result: the case did not finish under SLURM (it was never submitted). Run it again with: scenario-slurm resubmit <B>",
    ),
]
