"""ui.slurm_case_state's case_state, and ui.slurm_launch's _case_states that
calls it, against the function it was split out of.

``_original_case_states`` is ``ui.slurm_launch._case_states`` as it was before
the split (#839), copied verbatim; over many batches of records, submissions,
``squeue`` answers and task statuses, the new one must give the same entries,
their keys in the same order.  Only for submissions with a job id: a case of
one with none is no longer ``missing`` with "job None" (#1791), which the
tests at the end cover.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Any

import pytest

from autoware_carla_scenario.slurm.manifest import (
    SBATCH_IN_FLIGHT_SECONDS,
    CaseOutcome,
    SubmissionRecord,
    TaskStatus,
    readable_result,
)
from autoware_carla_scenario.slurm.records import JsonMembers
from autoware_carla_scenario.slurm.submit import ACTIVE_STATES
from autoware_carla_scenario.ui import slurm_launch
from autoware_carla_scenario.ui.slurm_case_state import case_state, no_job_case_state


def _original_case_states(
    layout: Any,
    manifest: dict[str, Any],
    squeue: dict[int, dict[int, str] | None],
) -> list[dict[str, Any]]:
    """Each case: its record's outcome, or where its task is."""
    # The last submission (and task) each case was given to.
    given: dict[int, tuple[int, int, dict[str, Any]]] = {}
    for index in layout.submissions():
        sub = layout.read_submission(index)
        for task, chunk in enumerate(sub.get("chunks") or []):
            for case in chunk:
                given[int(case)] = (index, task, sub)
    out = []
    for case in manifest.get("cases") or []:
        index = int(case["index"])
        record = layout.read_result(index)
        entry: dict[str, Any] = {
            "index": index,
            "label": case.get("label"),
            "stem": case.get("stem"),
            "overrides": case.get("case_overrides") or [],
        }
        if record is not None:
            # A result that is no object reads as an empty one, and only a
            # boolean ``true`` as a pass (#1718), in a result that can be
            # reported (#1726).
            result = record.get("result")
            if not isinstance(result, dict):
                result = {}
            entry.update(
                state="errored"
                if record.get("errored")
                else (
                    "passed"
                    if result.get("passed") is True
                    and readable_result(JsonMembers.from_value(result))
                    else "failed"
                ),
                message=result.get("message", ""),
                elapsed_seconds=result.get("elapsed_seconds"),
                host=record.get("host"),
                job_id=record.get("job_id"),
            )
        elif index not in given:
            entry.update(state="missing", message="never submitted")
        else:
            sub_index, task, sub = given[index]
            states = squeue.get(sub_index)
            queued = None if states is None else states.get(task)
            status = layout.task_dict(sub_index, task) or {}
            if queued in ACTIVE_STATES:
                running = queued == "RUNNING" and status.get("current") == index
                entry.update(
                    state="running" if running else "pending",
                    message=f"task {task} of job {sub.get('job_id')}: {queued.lower()}",
                )
            elif states is None:
                entry.update(state="unknown", message="squeue could not be asked")
            else:
                why = status.get("state", "never started")
                if status.get("error"):
                    why += f": {status['error']}"
                entry.update(
                    state="missing",
                    message=f"no result: task {task} of job {sub.get('job_id')} {why}",
                )
        out.append(entry)
    return out


class _Layout:
    """What _case_states reads of a BatchLayout, held in memory."""

    def __init__(
        self,
        submissions: dict[int, dict[str, Any]],
        results: dict[int, dict[str, Any]],
        tasks: dict[tuple[int, int], Any],
    ) -> None:
        self._submissions = submissions
        self._results = results
        self._tasks = tasks

    def submissions(self) -> list[int]:
        return sorted(self._submissions)

    def read_submission(self, index: int) -> dict[str, Any]:
        return self._submissions[index]

    def submission_record(self, index: int) -> SubmissionRecord | None:
        # As BatchLayout reads the record the dict would be written as.
        return SubmissionRecord(JsonMembers.from_value(self._submissions[index]))

    def read_result(self, index: int) -> CaseOutcome | None:
        # As BatchLayout reads the record the dict would be written as.
        record = self._results.get(index)
        return None if record is None else CaseOutcome(JsonMembers.from_value(record))

    def task_dict(self, submission: int, task: int) -> dict[str, Any] | None:
        """The task's status as it was read, a dict, for the original: None
        for one that is no object, as BatchLayout reads it since #1630."""
        status = self._tasks.get((submission, task))
        return status if isinstance(status, dict) else None

    def read_task(self, submission: int, task: int) -> TaskStatus | None:
        # As BatchLayout reads the status the dict would be written as.
        status = self.task_dict(submission, task)
        return None if status is None else TaskStatus(JsonMembers.from_value(status))


_SQUEUE_STATES = [*sorted(ACTIVE_STATES), "COMPLETED", "FAILED", "TIMEOUT", "running"]


def _record(rng: random.Random) -> dict[str, Any]:
    record: dict[str, Any] = {}
    errored = rng.choice([None, True, False, 1, 0, "", "yes", "missing"])
    if errored != "missing":
        record["errored"] = errored
    kind = rng.random()
    if kind < 0.15:
        pass  # no result
    elif kind < 0.25:
        record["result"] = rng.choice([None, {}, "passed", [True], 3, True])
    else:
        result: dict[str, Any] = {}
        passed = rng.choice(
            [True, False, None, 1, 0, "", "x", "true", "false", 1.0, "missing"]
        )
        if passed != "missing":
            result["passed"] = passed
        if rng.random() < 0.7:
            result["message"] = rng.choice(["", "ok", "collided", None, 3])
        if rng.random() < 0.7:
            result["elapsed_seconds"] = rng.choice([0.0, 12.5, None])
        if rng.random() < 0.3:
            result["condition_statuses"] = rng.choice(
                [
                    [],
                    None,
                    "x",
                    [{"label": "a", "satisfied": True, "message": "ok"}],
                    [{"label": "a", "satisfied": "yes", "message": "ok"}],
                ]
            )
        record["result"] = result
    if rng.random() < 0.7:
        record["host"] = rng.choice(["node1", None])
    if rng.random() < 0.7:
        record["job_id"] = rng.choice(["4711", 4711, None])
    return record


def _status(rng: random.Random, cases: list[int]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    if rng.random() < 0.6:
        current: Any = rng.choice([*cases, None, -1])
        if current is not None and rng.random() < 0.1:
            current = float(current)
        status["current"] = current
    if rng.random() < 0.7:
        status["state"] = rng.choice(["starting", "running", "done", "failed", ""])
    if rng.random() < 0.4:
        status["error"] = rng.choice(["", None, 0, "boom", "CARLA died", 7])
    return status


def _batch(rng: random.Random) -> tuple[_Layout, dict[str, Any], dict[int, Any]]:
    count = rng.randint(0, 12)
    cases = list(range(count))
    manifest_cases = []
    for index in cases:
        case: dict[str, Any] = {"index": index if rng.random() < 0.8 else str(index)}
        if rng.random() < 0.8:
            case["label"] = f"case {index}"
        if rng.random() < 0.8:
            case["stem"] = f"stem_{index}"
        if rng.random() < 0.5:
            case["case_overrides"] = rng.choice([None, [], [f"x={index}"]])
        manifest_cases.append(case)
    manifest: dict[str, Any] = {"cases": manifest_cases}
    if rng.random() < 0.05:
        manifest = rng.choice([{}, {"cases": None}])
    submissions: dict[int, dict[str, Any]] = {}
    tasks: dict[tuple[int, int], dict[str, Any]] = {}
    squeue: dict[int, Any] = {}
    for sub_index in range(rng.randint(0, 3)):
        pool = [c for c in cases if rng.random() < 0.7]
        rng.shuffle(pool)
        chunks: list[list[Any]] = []
        while pool:
            size = rng.randint(1, 3)
            chunk: list[Any] = pool[:size]
            pool = pool[size:]
            chunks.append([c if rng.random() < 0.8 else str(c) for c in chunk])
        sub: dict[str, Any] = {"chunks": chunks if rng.random() < 0.95 else None}
        # A submission with no job id differs from the original on purpose
        # (#1791): see the no-job tests below.
        sub["job_id"] = rng.choice(["1000", 1000])
        submissions[sub_index] = sub
        answer = rng.random()
        if answer < 0.2:
            squeue[sub_index] = None
        elif answer < 0.3:
            pass  # not asked: squeue.get() gives None
        else:
            squeue[sub_index] = {
                task: rng.choice(_SQUEUE_STATES)
                for task in range(len(chunks))
                if rng.random() < 0.7
            }
        for task in range(len(chunks)):
            if rng.random() < 0.8:
                tasks[(sub_index, task)] = _status(rng, cases)
    results = {index: _record(rng) for index in cases if rng.random() < 0.4}
    return _Layout(submissions, results, tasks), manifest, squeue


@pytest.mark.parametrize("seed", range(40))
def test_case_states_match_the_original(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        layout, manifest, squeue = _batch(rng)
        want = _original_case_states(layout, manifest, squeue)
        got = slurm_launch._case_states(layout, manifest, squeue)  # type: ignore[arg-type]
        assert [list(entry.items()) for entry in got] == [
            list(entry.items()) for entry in want
        ]


@pytest.mark.parametrize("status", [[1], "x", 5])
@pytest.mark.parametrize("queued", [None, "PENDING", "RUNNING", "COMPLETED"])
@pytest.mark.parametrize("asked", [True, False])
def test_a_task_file_holding_no_object_reads_as_no_status(
    status: Any, queued: str | None, asked: bool
) -> None:
    """BatchLayout.read_task reads a task file that holds JSON that is no
    object as no status (#1630): the case is where a task with no status
    is."""
    states = ({} if queued is None else {0: queued}) if asked else None
    layout = _Layout({0: {"chunks": [[0]], "job_id": "9"}}, {}, {(0, 0): status})
    none = _Layout({0: {"chunks": [[0]], "job_id": "9"}}, {}, {})
    manifest = {"cases": [{"index": 0}]}
    got = slurm_launch._case_states(layout, manifest, {0: states})  # type: ignore[arg-type]
    assert got == _original_case_states(layout, manifest, {0: states})
    assert got == slurm_launch._case_states(none, manifest, {0: states})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "result",
    [
        {"passed": "false"},
        {"passed": "true"},
        {"passed": 1},
        {"passed": None},
        {},
        "passed",
        [True],
        None,
        "missing",
        {"passed": True},
        {"passed": True, "message": "ok", "elapsed_seconds": "soon"},
        {
            "passed": True,
            "message": "ok",
            "elapsed_seconds": 1.0,
            "condition_statuses": [3],
        },
    ],
    ids=[
        "str-false",
        "str-true",
        "int",
        "null",
        "no-passed",
        "str-result",
        "list-result",
        "null-result",
        "no-result",
        "passed-only",
        "str-elapsed",
        "int-status",
    ],
)
def test_a_result_with_no_verdict_is_failed_not_passed(result: Any) -> None:
    """#1718: only a boolean ``passed`` in a ``result`` object is a verdict,
    and only in one that can be reported (#1726); any other record is a case
    that failed, and none raises."""
    record: dict[str, Any] = {"errored": False, "host": "node1"}
    if result != "missing":
        record["result"] = result
    layout = _Layout({0: {"chunks": [[0]], "job_id": "9"}}, {0: record}, {})
    manifest = {"cases": [{"index": 0}]}
    (entry,) = slurm_launch._case_states(layout, manifest, {0: {}})  # type: ignore[arg-type]
    assert entry["state"] == "failed"
    assert entry["host"] == "node1"
    passed = {
        "errored": False,
        "result": {"passed": True, "message": "", "elapsed_seconds": 1.0},
    }
    layout = _Layout({0: {"chunks": [[0]], "job_id": "9"}}, {0: passed}, {})
    (entry,) = slurm_launch._case_states(layout, manifest, {0: {}})  # type: ignore[arg-type]
    assert entry["state"] == "passed"


def test_the_typed_active_states_are_squeue_s() -> None:
    assert set(slurm_launch._ACTIVE_STATES) == set(ACTIVE_STATES)
    assert len(slurm_launch._ACTIVE_STATES) == len(ACTIVE_STATES)


_ACTIVE = tuple(sorted(ACTIVE_STATES))


@pytest.mark.parametrize(
    ("recorded", "errored", "passed", "given", "queued", "squeue_known", "at_case"),
    list(
        itertools.product(
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [None, "RUNNING", "PENDING", "COMPLETED"],
            [True, False],
            [True, False],
        )
    ),
)
def test_every_combination_of_inputs(
    recorded: bool,
    errored: bool,
    passed: bool,
    given: bool,
    queued: str | None,
    squeue_known: bool,
    at_case: bool,
) -> None:
    for task_error in (None, "boom"):
        state, message = case_state(
            recorded,
            errored,
            passed,
            given,
            queued,
            squeue_known,
            at_case,
            2,
            "77",
            "done",
            task_error,
            _ACTIVE,
        )
        if recorded:
            want_state = "errored" if errored else ("passed" if passed else "failed")
            assert (state, message) == (want_state, "")
        elif not given:
            assert (state, message) == ("missing", "never submitted")
        elif queued in ACTIVE_STATES:
            assert queued is not None
            running = queued == "RUNNING" and at_case
            assert state == ("running" if running else "pending")
            assert message == f"task 2 of job 77: {queued.lower()}"
        elif not squeue_known:
            assert (state, message) == ("unknown", "squeue could not be asked")
        else:
            why = "done" if task_error is None else "done: boom"
            assert (state, message) == ("missing", f"no result: task 2 of job 77 {why}")


@pytest.mark.parametrize(
    ("timed_out", "in_flight", "at", "want_state", "want_message"),
    [
        (
            True,
            True,
            "2026-11-06T12:00:00+09:00",
            "unknown",
            "no job id: the sbatch of submission 3 timed out at "
            "2026-11-06T12:00:00+09:00; the job may exist all the same: check "
            "`squeue --me`",
        ),
        (
            True,
            False,
            "",
            "unknown",
            "no job id: the sbatch of submission 3 timed out; the job may exist "
            "all the same: check `squeue --me`",
        ),
        (
            False,
            True,
            "",
            "pending",
            "no job id yet: submission 3 is being submitted (sbatch in flight)",
        ),
        (
            False,
            False,
            "",
            "missing",
            "no result: submission 3 has no job id (its submitter stopped "
            "before sbatch returned)",
        ),
    ],
    ids=["timed-out", "timed-out-no-time", "in-flight", "stale"],
)
def test_no_job_case_state(
    timed_out: bool, in_flight: bool, at: str, want_state: str, want_message: str
) -> None:
    """#1791: a case of a submission with no job id names no job, and is
    missing only when the record is stale."""
    state, message = no_job_case_state(3, timed_out, in_flight, at)
    assert (state, message) == (want_state, want_message)
    assert "None" not in message


@pytest.mark.parametrize("job", [None, "missing", ""], ids=["null", "absent", "empty"])
@pytest.mark.parametrize("kind", ["in-flight", "timed-out", "stale"])
def test_a_case_of_a_submission_with_no_job_id(job: Any, kind: str) -> None:
    """#1791: _case_states asks no squeue answer and reads no task status for
    a case whose submission has no job id; it says why there is none."""
    sub: dict[str, Any] = {
        "chunks": [[0, 1]],
        "submitted_at": time.time()
        - (SBATCH_IN_FLIGHT_SECONDS + 1 if kind == "stale" else 0),
    }
    if job != "missing":
        sub["job_id"] = job
    if kind == "timed-out":
        sub["sbatch_timed_out"] = True
        sub["sbatch_timed_out_at"] = "2026-11-06T12:00:00+09:00"
    passed = {
        "errored": False,
        "result": {"passed": True, "message": "ok", "elapsed_seconds": 1.0},
    }
    layout = _Layout({0: sub}, {1: passed}, {(0, 0): {"state": "done"}})
    manifest = {"cases": [{"index": 0}, {"index": 1}, {"index": 2}]}
    entries = slurm_launch._case_states(layout, manifest, {0: {}})  # type: ignore[arg-type]
    states = [entry["state"] for entry in entries]
    want = {"in-flight": "pending", "timed-out": "unknown", "stale": "missing"}[kind]
    assert states == [want, "passed", "missing"]
    assert "None" not in entries[0]["message"]
    assert "job None" not in str(entries)
