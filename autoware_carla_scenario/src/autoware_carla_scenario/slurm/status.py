"""What ``scenario-slurm status`` prints of a SLURM batch, what ``cancel``
cancels of it, whether it is busy, so that ``resubmit`` refuses it, and
which of its cases ``resubmit --cases`` names.

Read from the batch's files (its manifest, submission records, task statuses
and case outcomes, :mod:`.manifest`'s typed records) and from ``squeue``
(:mod:`.queue`).  ``scenario-slurm`` (:mod:`.cli`, which parses the command
line with argparse) runs them; split out of it so the library check
compiles them.
"""

from __future__ import annotations

import logging
import sys
import time

from .manifest import (
    SBATCH_IN_FLIGHT_SECONDS,
    BatchLayout,
    Manifest,
    SubmissionRecord,
    TaskStatus,
    no_job_note,
    recorded_verdict,
    sbatch_in_flight,
    sbatch_timed_out,
)
from .queue import ACTIVE_STATES, cancel_jobs, squeue_states
from .records import JsonMembers

__all__ = [
    "active_jobs",
    "busy_reason",
    "cancel_batch",
    "named_cases",
    "print_status",
]

# Logged under scenario-slurm's name, as it was before this was split out of it.
logger = logging.getLogger("autoware_carla_scenario.slurm.cli")


def _any_active(states: dict[int, str]) -> bool:
    """Whether any of *states* is one of ``ACTIVE_STATES``."""
    for state in states.values():
        if state in ACTIVE_STATES:
            return True
    return False


def active_jobs(
    layout: BatchLayout, below: int | None = None
) -> tuple[list[str], list[str], list[int], list[int], list[int]]:
    """The batch's array jobs SLURM still lists, those squeue could not be
    asked about (which may be running too), the submissions whose sbatch has
    not returned yet (``manifest.sbatch_in_flight``), those whose record
    cannot be read yet is younger than ``SBATCH_IN_FLIGHT_SECONDS``, and those
    whose sbatch timed out, younger than that too (still in flight, but
    sbatch has returned, #1791): of every submission, or of those before
    *below*.

    A record that cannot be read (not JSON, or not a JSON object) may be one
    another submitter is still writing; past ``SBATCH_IN_FLIGHT_SECONDS``
    after the file was last modified, it is ignored with a warning, so that
    a corrupt record never refuses every resubmit (the viewer cannot pass
    ``--force``).
    """
    active: list[str] = []
    unknown: list[str] = []
    in_flight: list[int] = []
    unreadable: list[int] = []
    timed_out: list[int] = []
    for index in layout.submissions():
        if below is not None and index >= below:
            continue
        path = layout.submission(index)
        # Annotated: Codon 0.19 refuses the call's value for the local without.
        submission: SubmissionRecord | None = None
        try:
            submission = layout.read_submission(index)
        except FileNotFoundError:
            continue  # removed since: its sbatch failed
        except (OSError, ValueError):
            submission = None
        if submission is None:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < SBATCH_IN_FLIGHT_SECONDS:
                unreadable.append(index)
            else:
                logger.warning(
                    "Ignoring %s: it cannot be read as a submission record, "
                    "and was last modified %.0f s ago",
                    path,
                    age,
                )
            continue
        if not submission.has_job:
            if sbatch_in_flight(submission):
                if sbatch_timed_out(submission):
                    timed_out.append(index)
                else:
                    in_flight.append(index)
            continue
        job = submission.job_text()
        states = squeue_states(job)
        if states is None:
            unknown.append(job)
        elif _any_active(states):
            active.append(job)
    return active, unknown, in_flight, unreadable, timed_out


def busy_reason(layout: BatchLayout, below: int | None = None) -> str | None:
    """Why the batch cannot be resubmitted now, if it cannot: a job of it is
    queued or running, ``squeue`` cannot tell, another submission of it is
    in flight (another ``resubmit``, the viewer's, #1352), or one's sbatch
    timed out a short while ago (its job may exist, #1791)."""
    active, unknown, in_flight, unreadable, timed_out = active_jobs(layout, below)
    if active:
        return (
            f"job(s) {', '.join(active)} of this batch are still queued or "
            "running; wait for them (scenario-slurm wait), cancel them "
            "(scenario-slurm cancel), or pass --force"
        )
    if in_flight:
        return (
            f"submission(s) {', '.join([str(i) for i in in_flight])} of this batch "
            "are being submitted (sbatch has not returned yet), by another "
            "resubmit or the viewer; try again once it has, or pass --force. "
            "A record left by a submitter that died is ignored "
            f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s after it was made"
        )
    if timed_out:
        return (
            f"the sbatch of submission(s) {', '.join([str(i) for i in timed_out])} "
            "of this batch timed out, and SLURM may have taken the job all the "
            "same: check `squeue --me` (scancel the job if it is there and you "
            "want its cases run again), then pass --force, or try again once "
            f"the record is ignored, {SBATCH_IN_FLIGHT_SECONDS:.0f} s after it "
            "was made"
        )
    if unreadable:
        files = ", ".join([str(layout.submission(i)) for i in unreadable])
        return (
            f"the submission record(s) {files} of this batch are unreadable "
            "(another submitter may still be writing them); try again shortly, "
            "or pass --force. An unreadable record is ignored "
            f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s after it was last modified"
        )
    if unknown:
        # A task still running would have its outputs removed under it.
        return (
            f"squeue could not tell whether job(s) {', '.join(unknown)} of "
            "this batch are still queued or running; check with squeue, or pass "
            "--force if they are not"
        )
    return None


def _queued_text(states: dict[int, str] | None) -> str:
    """What ``squeue`` says of a submission's tasks, as ``status`` shows it."""
    if states is None:
        return "squeue unavailable"
    listed = ", ".join([f"{t}:{s}" for t, s in sorted(states.items())])
    return listed if listed else "ended"


def print_status(layout: BatchLayout) -> int:
    """Print the batch's cases, its submissions' jobs (as ``squeue`` lists
    them) and their tasks (as their statuses say): ``scenario-slurm
    status``.  Its exit status, 0."""
    manifest = layout.read_manifest()
    cases = manifest.cases()
    done = passed = errored = 0
    for case in cases:
        record = layout.read_result(case.index())
        if record is None:
            continue
        done += 1
        verdict = recorded_verdict(record)
        if verdict is not None and verdict:
            passed += 1
        if record.errored:
            errored += 1
    print(f"Batch {layout.batch_dir}: {len(cases)} case(s)")  # noqa: T201
    print(  # noqa: T201
        f"  {done} with a result ({passed} passed, {errored} errored), "
        f"{len(cases) - done} without"
    )
    for index in layout.submissions():
        submission = layout.submission_record(index)
        if submission is None:
            path = layout.submission(index)
            if path.exists():  # else removed since: its sbatch failed
                print(  # noqa: T201
                    f"  submission {index}: unreadable record {path} "
                    "(empty, not JSON, or not a JSON object)"
                )
            continue
        task_count = submission.size_of("chunks")
        size = f"{task_count} task(s), {submission.size_of('cases')} case(s)"
        if submission.has_job:
            job = submission.job_text()
            queued = _queued_text(squeue_states(job))
            print(f"  submission {index}: job {job}, {size}; in the queue: {queued}")  # noqa: T201
        else:
            # No job id to ask squeue about: never "job None ... ended" (#1791).
            print(f"  submission {index}: {size}; {no_job_note(submission)}")  # noqa: T201
        for task, chunk in enumerate(submission.chunk_list()):
            read = layout.read_task(index, task)
            # A task with no status (none yet, or none to read) reads as an
            # empty one.
            status = TaskStatus(JsonMembers()) if read is None else read
            # In the order the line was laid out: what raises for a value of
            # the wrong type raises first.
            state = status.state_text("not started")
            host = status.joined(" on ", "host")
            port = status.port_text()
            done_count = status.size_of("done")
            chunk_size = len(chunk)
            rss = status.true_text("peak_rss_mb")
            rss_text = "" if rss is None else f", peak RSS {rss} MB"
            error = status.joined(" -- ", "error")
            print(  # noqa: T201
                f"    task {task}: {state}{host}{port}"
                f", {done_count}/{chunk_size} case(s){rss_text}{error}"
            )
    return 0


def cancel_batch(layout: BatchLayout) -> int:
    """``scancel`` the batch's jobs (each submission's array job and collect
    job), saying which: ``scenario-slurm cancel``.  Its exit status: 1 when
    scancel failed."""
    jobs: list[str] = []
    for index in layout.submissions():
        submission = layout.submission_record(index)
        if submission is None:
            path = layout.submission(index)
            if path.exists():  # else removed since: its sbatch failed
                print(  # noqa: T201
                    f"Skipping {path}: it cannot be read as a submission record; "
                    "if it named a job, cancel that with scancel",
                    file=sys.stderr,
                )
            continue
        for key in ("job_id", "collect_job_id"):
            if submission.is_true(key):
                jobs.append(submission.text(key))
    if not jobs:
        print("The batch has no job to cancel.")  # noqa: T201
        return 0
    print(f"Cancelling job(s) {', '.join(jobs)}")  # noqa: T201
    return 0 if cancel_jobs(jobs) else 1


def named_cases(manifest: Manifest, names: str) -> list[int] | None:
    """The cases of the batch *names* names (``resubmit --cases``: labels or
    indices, comma-separated), sorted, each once (a case named twice, by its
    label and its index say, runs once); ``None``, once it has said which,
    when an entry names no case of the batch.

    An entry is a case's label (``str(case["label"])``, the last of two
    cases labelled alike giving its index) or else its index, ASCII digits
    below the number of cases.  Raises what reading the manifest's cases
    raises (:meth:`.manifest.Manifest.cases`, ``int(case["index"])``)."""
    labels: dict[str, int] = {}
    for case in manifest.cases():
        # The label before the index, as a dict display reads them.
        label = case.label()
        labels[label] = case.index()
    picked: set[int] = set()
    for entry in names.split(","):
        item = entry.strip()
        if item in labels:
            picked.add(labels[item])
        elif (
            item.isascii() and item.isdigit() and 0 <= int(item) < len(manifest.cases())
        ):
            picked.add(int(item))
        else:
            print(f"Error: no case {item!r} in the batch", file=sys.stderr)  # noqa: T201
            return None
    return sorted(picked)
