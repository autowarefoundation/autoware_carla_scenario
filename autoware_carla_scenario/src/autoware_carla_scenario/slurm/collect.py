"""Collecting a SLURM batch: its summary and ``batch_results.json``.

What a local batch prints and writes when its last case has run, made from the
outcome each task wrote for each of its cases (``slurm/results/<i>.json``), in
the batch's order -- the same table, the same JSON, whichever task ran a case
and when.  A case with no outcome is reported as one: ``ERROR``, with a message
saying it has none and what to do about it (#1803): wait for its task where
that may still be queued or running (``scenario-slurm wait``), try again once
sbatch returns where its submission is in flight, and run it again
(``scenario-slurm resubmit``) otherwise -- saying, where its sbatch timed out
less than ``SBATCH_IN_FLIGHT_SECONDS`` ago, that a resubmit is refused until
then -- never advising a resubmit that ``resubmit`` would refuse.
So is a case whose outcome holds no verdict -- no ``result`` object, a
``passed`` that is not a boolean (#1718), or a result that cannot be reported
(``readable_result()``: its ``message``, ``elapsed_seconds``,
``condition_statuses`` or ``failure`` missing or of the wrong type, #1726): it did not pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..batch_report import print_summary
from ..failure import Failure, FailureCategory
from .manifest import (
    SBATCH_IN_FLIGHT_SECONDS,
    BatchLayout,
    no_job_note,
    recorded_verdict,
    resubmit_target,
    sbatch_in_flight,
    sbatch_timed_out,
    write_json,
)
from .records import JsonMembers

__all__ = [
    "RecordedResult",
    "collect",
    "missing_cases",
    "recorded_verdict",
    "unfinished_cases",
]


@dataclass
class _Status:
    label: str
    satisfied: bool
    message: str


def _value_json(members: JsonMembers, key: str) -> str:
    """The JSON text of the value of the member *key* of *members*, ``null``
    when there is none: what ``dict.get(key)`` gave of the object."""
    return members.value_text(key) if members.has(key) else "null"


class RecordedResult:
    """A case's :class:`~autoware_carla_scenario.conditions.ScenarioResult`, as
    a task recorded it (``ScenarioResult.to_dict()``): the members of its
    ``result`` (:meth:`.manifest.CaseOutcome.result`).

    Each member is read from its JSON text when it is asked for, so a value
    of the wrong type raises there, as reading it from the record read as a
    ``dict`` did; :meth:`to_dict` gives back values equal to the record's, so
    the batch's summary (``batch_report``) is written as it was."""

    members: JsonMembers

    def __init__(self, members: JsonMembers) -> None:
        self.members = members

    @property
    def passed(self) -> bool:
        # Only a boolean is a verdict: ``"false"`` or ``1`` is no pass (#1718).
        passed = json.loads(_value_json(self.members, "passed"))
        return isinstance(passed, bool) and bool(passed)

    @property
    def message(self) -> str:
        return str(self.members.value("message"))

    @property
    def elapsed_seconds(self) -> float:
        return float(self.members.value("elapsed_seconds"))

    @property
    def condition_statuses(self) -> list[_Status]:
        statuses = json.loads(_value_json(self.members, "condition_statuses"))
        read: list[_Status] = []
        if not bool(statuses):
            return read
        for s in statuses:
            read.append(
                _Status(str(s["label"]), bool(s["satisfied"]), str(s["message"]))
            )
        return read

    @property
    def rerun_recording(self) -> str | None:
        recording_json = _value_json(self.members, "rerun_recording")
        if recording_json == "null":
            return None
        return str(json.loads(recording_json))

    @property
    def failure(self) -> Failure | None:
        failure_json = _value_json(self.members, "failure")
        if failure_json == "null":
            return None
        if TYPE_CHECKING:
            # Failure.from_dict() reads a mapping of any values, which Codon
            # cannot express (typecheck/library.py's UNCALLED).
            return None
        else:
            return Failure.from_dict(json.loads(failure_json))

    def to_dict(self) -> dict[str, Any]:
        return {key: self.members.value(key) for key in self.members.keys()}


def _last_submission_of(
    layout: BatchLayout, case: int
) -> tuple[int, int, dict[str, Any]] | None:
    """(submission, task, the submission's record) that was last given *case*,
    if any.  A record that cannot be read gave no case (#1356)."""
    for index in reversed(layout.submissions()):
        submission = layout.submission_record(index)
        if submission is None:
            continue
        for task, chunk in enumerate(submission.get("chunks") or []):
            if case in chunk:
                return index, task, submission
    return None


#: The states a task's status records before the task has finished with its
#: cases (``slurm/worker.py``): it may still be running them -- or have been
#: killed (out of time, out of memory) before it could say otherwise.
_UNFINISHED_TASK_STATES = frozenset({"starting", "waiting", "building", "running"})

#: What a missing case's message advises (#1803), in the order of
#: ``cli._busy``'s reasons a resubmit is refused: its task may still be queued
#: or running; its submission's sbatch has not returned; its sbatch timed out
#: less than SBATCH_IN_FLIGHT_SECONDS ago; else run it again.
_RUNNING = "running"
_IN_FLIGHT = "in_flight"
_TIMED_OUT = "timed_out"
_RESUBMIT = "resubmit"
_ADVICE_ORDER = (_RUNNING, _IN_FLIGHT, _TIMED_OUT, _RESUBMIT)

_REFUSED_UNTIL = (
    f"a resubmit is refused until {SBATCH_IN_FLIGHT_SECONDS:.0f} s after the "
    "record was made, unless with --force"
)


def _wait_command(layout: BatchLayout, index: int, target: str) -> str:
    """``scenario-slurm wait`` on submission *index*: it follows the latest
    one unless told which."""
    submissions = layout.submissions()
    if submissions and submissions[-1] == index:
        return f"scenario-slurm wait {target}"
    return f"scenario-slurm wait --submission {index} {target}"


def _missing_message(
    layout: BatchLayout, case: int, target: str
) -> tuple[str, str, int | None]:
    """The message of *case*, which has no outcome, which advice it gives
    (``_RUNNING``, ``_IN_FLIGHT``, ``_TIMED_OUT`` or ``_RESUBMIT``), and the
    submission that advice is about (None for ``_RESUBMIT``): what
    :func:`_advice` says of what the batch's records say of it.

    The records are read here, in the order and only as far as before the
    advice was split from them: the clock is read as often, and a state no
    set holds or a ``submitted_at`` no float holds raises in the same cases
    (each condition below short-circuits as the advice's did)."""
    where = _last_submission_of(layout, case)
    if where is None:
        return _advice(
            layout,
            target,
            submitted=False,
            index=0,
            task=0,
            job="",
            has_job=False,
            state="",
            has_status=False,
            unfinished=False,
            error=None,
            note="",
            timed_out=False,
            in_flight=False,
        )
    index, task, submission = where
    status = layout.read_task(index, task)
    state = "never started" if status is None else status.get("state", "?")
    job = submission.get("job_id")
    # Never "job None" (#1796): why the submission has no job id.
    note = "" if job else no_job_note(submission)
    error = f"{status['error']}" if status is not None and status.get("error") else None
    # Queued (no status yet) or running -- or killed before its status said
    # so; a state no set holds raises, where a job's task has a status.
    unfinished = bool(job) and status is not None and state in _UNFINISHED_TASK_STATES
    timed_out = (
        not job and sbatch_timed_out(submission) and sbatch_in_flight(submission)
    )
    in_flight = not job and not timed_out and sbatch_in_flight(submission)
    return _advice(
        layout,
        target,
        submitted=True,
        index=index,
        task=task,
        job=f"{job}",
        has_job=bool(job),
        state=f"{state}",
        has_status=status is not None,
        unfinished=unfinished,
        error=error,
        note=note,
        timed_out=timed_out,
        in_flight=in_flight,
    )


def _advice(
    layout: BatchLayout,
    target: str,
    *,
    submitted: bool,
    index: int,
    task: int,
    job: str,
    has_job: bool,
    state: str,
    has_status: bool,
    unfinished: bool,
    error: str | None,
    note: str,
    timed_out: bool,
    in_flight: bool,
) -> tuple[str, str, int | None]:
    """The message of a case with no outcome, which advice it gives
    (``_RUNNING``, ``_IN_FLIGHT``, ``_TIMED_OUT`` or ``_RESUBMIT``), and the
    submission that advice is about (None for ``_RESUBMIT``), from what the
    batch's records say of it (:func:`_missing_message` reads them): whether
    any submission gave it (*submitted*), and if so the last one (*index*), its
    *task*, the submission's *job* id as text and whether it has one
    (*has_job*), the task's *state* as text and whether it has a status
    (*has_status*), whether that state is one of an unfinished task
    (*unfinished*), its *error* (None when it has none), why the submission
    has no job id (*note*), whether its sbatch timed out less than
    SBATCH_IN_FLIGHT_SECONDS ago (*timed_out*), and, if not, whether it is
    in flight (*in_flight*).

    The advice is chosen from what the batch's records say, asking SLURM
    nothing, as ``wait`` words it (#1801): a resubmit is advised only where
    ``resubmit`` takes it, or with the reason it is refused for now."""
    resubmit = f"Run it again with: scenario-slurm resubmit {target}"
    advice = _RESUBMIT
    about: int | None = None
    if not submitted:
        detail = "it was never submitted"
        then = resubmit
    else:
        if has_job:
            detail = f"task {task} of job {job} {state}"
        else:
            # Never "job None" (#1796): say why the submission has no job id.
            detail = f"task {task} of submission {index} {state}; {note}"
        if error is not None:
            detail += f": {error}"
        if has_job and (not has_status or unfinished):
            # Queued (no status yet) or running -- or killed before its status
            # said so: only squeue can tell, and resubmit refuses while it is.
            advice, about = _RUNNING, index
            then = (
                "If `squeue --me` still lists the task, it is still queued or "
                f"running: wait for it with: {_wait_command(layout, index, target)}; "
                "once it has ended, run the case again with: "
                f"scenario-slurm resubmit {target}"
            )
        elif not has_job and timed_out:
            advice, about = _TIMED_OUT, index
            then = (
                "If `squeue --me` lists the job, collect the batch once it "
                f"ends: scenario-slurm collect {target}; if not, run the case "
                f"again with: scenario-slurm resubmit {target} ({_REFUSED_UNTIL})"
            )
        elif not has_job and in_flight:
            advice, about = _IN_FLIGHT, index
            then = (
                "Its submission is being submitted; try again once sbatch "
                f"returns: {_wait_command(layout, index, target)}"
            )
        else:
            then = resubmit
    return (
        f"No result: the case did not finish under SLURM ({detail}). {then}",
        advice,
        about,
    )


def _missing_hint(layout: BatchLayout, target: str, advice: dict[str, set[int]]) -> str:
    """What the hint on the cases with no outcome advises, from the advice
    their messages give and the submissions it is about: a resubmit refuses
    the whole batch while any of them is busy (``cli._busy``), so the first
    of _ADVICE_ORDER wins.  ``wait`` follows the latest submission unless
    told which, so it is told which of the others it is about."""
    first = next(kind for kind in _ADVICE_ORDER if kind in advice)
    waits = " and ".join(
        _wait_command(layout, index, target) for index in sorted(advice[first])
    )
    if first == _RUNNING:
        return (
            "Some may still be queued or running: if `squeue --me` lists their "
            f"tasks, wait for them with: {waits}; once "
            f"they have ended, run them again with: scenario-slurm resubmit {target}"
        )
    if first == _IN_FLIGHT:
        return (
            "A submission of this batch is being submitted; try again once "
            f"sbatch returns: {waits}"
        )
    if first == _TIMED_OUT:
        return (
            "An sbatch of this batch timed out: if `squeue --me` lists its "
            f"job, collect the batch once it ends: scenario-slurm collect {target}; "
            f"if not, run them again with: scenario-slurm resubmit {target} "
            f"({_REFUSED_UNTIL})"
        )
    return f"Run them again with: scenario-slurm resubmit {target}"


def _failure_json(failure: Failure) -> str:
    """The JSON text of *failure*'s ``to_dict()``."""
    members = JsonMembers()
    members.set_text("condition", json.dumps(failure.condition))
    members.set_text("category", json.dumps(failure.category.name.lower()))
    members.set_text("at", json.dumps(failure.at))
    members.set_text("message", json.dumps(failure.message))
    detail = failure.detail
    members.set_text("detail", "null" if detail is None else json.dumps(detail))
    members.set_text("also_triggered", json.dumps(list(failure.also_triggered)))
    return members.object_text()


def _error_result(message: str) -> RecordedResult:
    """The outcome reported for a case that has none to report: ``ERROR``."""
    failure = Failure("", FailureCategory.INFRASTRUCTURE, 0.0, message)
    members = JsonMembers()
    members.set_text("passed", "false")
    members.set_text("message", json.dumps(message))
    members.set_text("elapsed_seconds", "0.0")
    members.set_text("condition_statuses", "[]")
    members.set_text("failure", _failure_json(failure))
    return RecordedResult(members)


def _no_verdict_message(layout: BatchLayout, case: int, target: str) -> str:
    return (
        f"No verdict: {layout.result(case)} holds no result whose `passed` "
        "is true or false, with a string `message`, a number "
        "`elapsed_seconds`, and `condition_statuses` and a `failure` that "
        "can be read. "
        f"Run the case again with: scenario-slurm resubmit {target}"
    )


def missing_cases(layout: BatchLayout) -> list[int]:
    """The cases of the batch no task has recorded an outcome for."""
    manifest = layout.read_manifest()
    return [
        case["index"]
        for case in manifest["cases"]
        if layout.read_result(int(case["index"])) is None
    ]


def unfinished_cases(layout: BatchLayout, *, failed: bool = False) -> list[int]:
    """The cases to run again: without an outcome, with one that holds no
    verdict, or one that raised on every attempt (``ERROR``); with *failed*,
    also those that ran and failed."""
    manifest = layout.read_manifest()
    chosen: list[int] = []
    for case in manifest["cases"]:
        index = int(case["index"])
        record = layout.read_result(index)
        verdict = recorded_verdict(record)
        if (
            record is None
            or verdict is None
            or record.get("errored")
            or (failed and not verdict)
        ):
            chosen.append(index)
    return chosen


def collect(layout: BatchLayout, *, quiet_hint: bool = False) -> int:
    """Print the batch's summary and write ``batch_results.json``, as a local
    batch does; 0 when every case passed, 1 otherwise.

    Raises:
        OSError: ``batch_results.json`` or ``collected.json`` cannot be
            written (an unwritable batch directory); the summary is printed
            first.
    """
    manifest = layout.read_manifest()
    target = resubmit_target(layout, manifest)
    labels: list[str] = []
    results: list[RecordedResult] = []
    errored: list[int] = []
    missing: list[int] = []
    # What the missing cases' messages advise, and the submissions it is
    # about (#1803).
    advice: dict[str, set[int]] = {}
    # The cases whose outcome holds no verdict or cannot be reported (#1718,
    # #1726): ERROR, as if missing.
    unreadable: list[int] = []
    # The cases whose Rerun recording the task failed to make (rerun.enabled).
    rerun_errors: dict[int, str] = {}
    for position, case in enumerate(manifest["cases"]):
        index = int(case["index"])
        labels.append(str(case["label"]))
        record = layout.read_result(index)
        if record is None:
            missing.append(position)
            message, kind, about = _missing_message(layout, index, target)
            about_kind = advice.setdefault(kind, set())
            if about is not None:
                about_kind.add(about)
            results.append(_error_result(message))
            continue
        # Reported whatever the result holds: the task recorded why (#1726).
        rerun_error = record.rerun_error
        if rerun_error is not None:
            rerun_errors[position] = rerun_error
        result = record.result()
        if result is None or recorded_verdict(record) is None:
            unreadable.append(position)
            results.append(_error_result(_no_verdict_message(layout, index, target)))
            continue
        results.append(RecordedResult(result))
        if record.errored:
            errored.append(position)
    overrides = (
        [list(case["case_overrides"]) for case in manifest["cases"]]
        if manifest.get("expand")
        else None
    )
    # Each case's file stem, as a local batch's summary names it (#1957); a
    # manifest that names none for a case gives the summary no stems.
    named = [str(case.get("stem") or "") for case in manifest["cases"]]
    stems = named if all(named) else None
    all_passed = print_summary(
        labels,
        results,
        output_dir=Path(layout.batch_dir),
        overrides=overrides,
        errored=sorted({*errored, *missing, *unreadable}),
        rerun_errors=rerun_errors,
        stems=stems,
    )
    if unreadable and not quiet_hint:
        print(  # noqa: T201
            f"{len(unreadable)} case(s) have a result with no verdict: "
            + ", ".join(labels[i] for i in unreadable)
            + f"\nRun them again with: scenario-slurm resubmit {target}"
        )
    if missing and not quiet_hint:
        print(  # noqa: T201
            f"{len(missing)} case(s) have no result: "
            + ", ".join(labels[i] for i in missing)
            + "\n"
            + _missing_hint(layout, target, advice)
        )
    # Written last: the summary and the hints are printed even when the batch
    # directory cannot be written (#1740).
    write_json(
        layout.root / "collected.json",
        {
            "cases": len(labels),
            "passed": sum(1 for r in results if r.passed),
            "errored": [labels[i] for i in sorted({*errored, *unreadable})],
            "missing": [labels[i] for i in missing],
        },
    )
    return 0 if all_passed else 1
