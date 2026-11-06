"""Collecting a SLURM batch: its summary and ``batch_results.json``.

What a local batch prints and writes when its last case has run, made from the
outcome each task wrote for each of its cases (``slurm/results/<i>.json``), in
the batch's order -- the same table, the same JSON, whichever task ran a case
and when.  A case with no outcome is reported as one: ``ERROR``, with a message
saying it has none and how to run it again (``scenario-slurm resubmit``).
So is a case whose outcome holds no verdict -- no ``result`` object, a
``passed`` that is not a boolean (#1718), or a result that cannot be reported
(``readable_result()``: its ``message``, ``elapsed_seconds``,
``condition_statuses`` or ``failure`` missing or of the wrong type, #1726): it did not pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..batch_report import print_summary
from ..failure import Failure, FailureCategory
from .manifest import BatchLayout, recorded_verdict, resubmit_target, write_json

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
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordedResult:
    """A case's :class:`~autoware_carla_scenario.conditions.ScenarioResult`, as
    a task recorded it (``ScenarioResult.to_dict()``)."""

    data: dict[str, Any]

    @property
    def passed(self) -> bool:
        # Only a boolean is a verdict: ``"false"`` or ``1`` is no pass (#1718).
        return self.data.get("passed") is True

    @property
    def message(self) -> str:
        return str(self.data["message"])

    @property
    def elapsed_seconds(self) -> float:
        return float(self.data["elapsed_seconds"])

    @property
    def condition_statuses(self) -> list[_Status]:
        return [
            _Status(str(s["label"]), bool(s["satisfied"]), str(s["message"]), s)
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


def _missing_message(layout: BatchLayout, case: int, target: str) -> str:
    where = _last_submission_of(layout, case)
    if where is None:
        detail = "it was never submitted"
    else:
        index, task, submission = where
        status = layout.read_task(index, task)
        state = "never started" if status is None else status.get("state", "?")
        detail = f"task {task} of job {submission.get('job_id')} {state}"
        if status is not None and status.get("error"):
            detail += f": {status['error']}"
    return (
        f"No result: the case did not finish under SLURM ({detail}). "
        f"Run it again with: scenario-slurm resubmit {target}"
    )


def _error_result(message: str) -> RecordedResult:
    """The outcome reported for a case that has none to report: ``ERROR``."""
    failure = Failure("", FailureCategory.INFRASTRUCTURE, 0.0, message)
    return RecordedResult(
        {
            "passed": False,
            "message": message,
            "elapsed_seconds": 0.0,
            "condition_statuses": [],
            "failure": failure.to_dict(),
        }
    )


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
            results.append(_error_result(_missing_message(layout, index, target)))
            continue
        # Reported whatever the result holds: the task recorded why (#1726).
        if record.get("rerun_error"):
            rerun_errors[position] = str(record["rerun_error"])
        if recorded_verdict(record) is None:
            unreadable.append(position)
            results.append(_error_result(_no_verdict_message(layout, index, target)))
            continue
        results.append(RecordedResult(dict(record["result"])))
        if record.get("errored"):
            errored.append(position)
    overrides = (
        [list(case["case_overrides"]) for case in manifest["cases"]]
        if manifest.get("expand")
        else None
    )
    all_passed = print_summary(
        labels,
        results,
        output_dir=Path(layout.batch_dir),
        overrides=overrides,
        errored=sorted({*errored, *missing, *unreadable}),
        rerun_errors=rerun_errors,
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
            + f"\nRun them again with: scenario-slurm resubmit {target}"
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
