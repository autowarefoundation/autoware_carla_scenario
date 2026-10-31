"""ui.slurm_case_state's case_state, and ui.slurm_launch's _case_states that
calls it, against the function it was split out of.

``_original_case_states`` is ``ui.slurm_launch._case_states`` as it was before
the split (#839), copied verbatim; over many batches of records, submissions,
``squeue`` answers and task statuses, the new one must give the same entries,
their keys in the same order.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

import pytest

from autoware_carla_scenario.slurm.submit import ACTIVE_STATES
from autoware_carla_scenario.ui import slurm_launch
from autoware_carla_scenario.ui.slurm_case_state import case_state


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
            result = record.get("result") or {}
            entry.update(
                state="errored"
                if record.get("errored")
                else ("passed" if result.get("passed") else "failed"),
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
            status = layout.read_task(sub_index, task) or {}
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
        tasks: dict[tuple[int, int], dict[str, Any]],
    ) -> None:
        self._submissions = submissions
        self._results = results
        self._tasks = tasks

    def submissions(self) -> list[int]:
        return sorted(self._submissions)

    def read_submission(self, index: int) -> dict[str, Any]:
        return self._submissions[index]

    def read_result(self, index: int) -> dict[str, Any] | None:
        return self._results.get(index)

    def read_task(self, submission: int, task: int) -> dict[str, Any] | None:
        return self._tasks.get((submission, task))


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
        record["result"] = rng.choice([None, {}])
    else:
        result: dict[str, Any] = {}
        passed = rng.choice([True, False, None, 1, 0, "", "x", "missing"])
        if passed != "missing":
            result["passed"] = passed
        if rng.random() < 0.7:
            result["message"] = rng.choice(["", "ok", "collided", None, 3])
        if rng.random() < 0.7:
            result["elapsed_seconds"] = rng.choice([0.0, 12.5, None])
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
        job = rng.choice(["1000", 1000, None, "missing"])
        if job != "missing":
            sub["job_id"] = job
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
