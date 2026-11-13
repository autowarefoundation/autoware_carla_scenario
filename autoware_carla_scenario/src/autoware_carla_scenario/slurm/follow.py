"""Following a SLURM submission until it ends, and what a submitter says of
it once it is submitted (docs/slurm.md).

:func:`follow` is ``scenario-slurm wait``, and what a submission that waits
(``launcher.wait``) ends in: it polls ``squeue`` (:mod:`.queue`) and counts
the cases whose outcomes name the submission, until SLURM lists none of its
tasks, then collects the batch (:mod:`.collect`).  :func:`after_submit` is
what :mod:`.submit` does once a submission's array is submitted (by
``scenario ... launcher=slurm`` or ``scenario-slurm resubmit``): follow it,
or say how to.  Both read the batch's files through :mod:`.manifest`'s typed
records; split out of :mod:`.submit` (which imports OmegaConf), which
re-exports them, so the library check compiles them.
"""

from __future__ import annotations

import shlex
import sys
import time

from .manifest import (
    SBATCH_IN_FLIGHT_SECONDS,
    BatchLayout,
    recorded_verdict,
    sbatch_in_flight,
    sbatch_timed_out,
)
from .queue import (
    CANCEL_WAIT_SECONDS,
    _any_active,
    _wait_until_gone,
    cancel_jobs,
    squeue_states,
)
from .scripts import CLI_MODULE
from .settings import SlurmSettings

__all__ = ["after_submit", "follow"]


def after_submit(layout: BatchLayout, index: int, settings: SlurmSettings) -> int:
    """Follow submission *index* to its end and collect (``wait``), or say
    how to; the exit status."""
    submission = layout.submission_record(index)
    if submission is None:
        print(  # noqa: T201
            f"Error: the submission record {layout.submission(index)} cannot be "
            f"read; see the batch with: scenario-slurm status {layout.batch_dir}",
            file=sys.stderr,
        )
        return 1
    if settings.wait:
        return follow(layout, index, settings)
    if submission.is_true("collect_error"):
        return 1  # submit_cases said what failed, and how to collect
    cli = f"{shlex.quote(sys.executable)} -m {CLI_MODULE}"
    target = shlex.quote(str(layout.batch_dir))
    job = submission.text("job_id")
    print(f"Submitted; not waiting (--no-wait). Job {job}.")  # noqa: T201
    if submission.is_true("collect_job_id"):
        collect_job = submission.text("collect_job_id")
        print(  # noqa: T201
            f"Collect job {collect_job} writes "
            f"{layout.batch_dir / 'batch_results.json'} once the array ends."
        )
    print(f"  follow:   scenario-slurm wait {target}")  # noqa: T201
    print(f"  status:   scenario-slurm status {target}")  # noqa: T201
    print(f"  collect:  scenario-slurm collect {target}")  # noqa: T201
    print(f"  (or {cli} <command> {target})")  # noqa: T201
    return 0


def _count(states: dict[int, str] | None, state: str) -> int:
    """How many of *states* (none when squeue could not tell) are *state*."""
    count = 0
    if states is not None:
        for listed in states.values():
            if listed == state:
                count += 1
    return count


def follow(layout: BatchLayout, index: int, settings: SlurmSettings) -> int:
    """Follow submission *index* until SLURM lists none of its tasks, then
    collect the batch; the exit status (0 when every case passed).

    A submission with no job yet (its sbatch has not returned, timed out, or
    its submitter died before it did) is neither followed nor collected
    (#1417, #1791): ``squeue -j None`` would read as a job that ended."""
    from .collect import collect  # noqa: PLC0415

    submission = layout.submission_record(index)
    if submission is None:
        print(  # noqa: T201
            f"Error: the submission record {layout.submission(index)} is missing "
            "or cannot be read; collect the batch with: "
            f"scenario-slurm collect {layout.batch_dir}",
            file=sys.stderr,
        )
        return 1
    if not submission.has_job:
        if sbatch_timed_out(submission):
            at = submission.timed_out_at_text()
            # Only while the record still counts as in flight is a resubmit
            # refused (#1796); after that a plain resubmit is taken.
            refused = (
                f" (refused until {SBATCH_IN_FLIGHT_SECONDS:.0f} s after the "
                "record was made, unless with --force)"
                if sbatch_in_flight(submission)
                else ""
            )
            print(  # noqa: T201
                f"Error: submission {index} of this batch has no job id to "
                "follow: its sbatch timed out"
                + ("" if at is None else f" at {at}")
                + ", and SLURM may have taken the job all the same. If "
                "`squeue --me` lists it, collect the batch once it ends: "
                f"scenario-slurm collect {layout.batch_dir}; if not, run its "
                f"cases again: scenario-slurm resubmit {layout.batch_dir}" + refused,
                file=sys.stderr,
            )
        elif sbatch_in_flight(submission):
            print(  # noqa: T201
                f"Error: submission {index} of this batch is being submitted "
                "(sbatch has not returned yet), by another resubmit or the "
                "viewer; try again once it has: "
                f"scenario-slurm wait {layout.batch_dir}",
                file=sys.stderr,
            )
        else:
            print(  # noqa: T201
                f"Error: submission {index} of this batch has no job: its "
                "submitter stopped before sbatch returned, more than "
                f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s ago, so the record is stale. "
                f"See the batch with: scenario-slurm status {layout.batch_dir}",
                file=sys.stderr,
            )
        return 1
    job = submission.text("job_id")
    cases = submission.chunk_cases()
    last = ""
    # Annotated: Codon 0.19 refuses None for the local without.
    unknown_since: float | None = None
    try:
        while True:
            states = squeue_states(job)
            done = passed = 0
            for case in cases:
                record = layout.read_result(case)
                if record is not None and record.names_submission(index):
                    done += 1
                    verdict = recorded_verdict(record)
                    if verdict is not None and verdict:
                        passed += 1
            running = _count(states, "RUNNING")
            pending = _count(states, "PENDING")
            line = (
                f"[slurm] job {job}: {done}/{len(cases)} case(s) done "
                f"({passed} passed); {running} task(s) running, {pending} pending"
            )
            if line != last:
                print(line, flush=True)  # noqa: T201
                last = line
            if states is not None and not _any_active(states):
                break
            if states is None:
                # squeue failed: give it a few polls before giving up on it.
                # (``unknown_since or time.monotonic()``: a start at 0.0 is
                # read again, as ``or`` read it.)
                if unknown_since is None or unknown_since == 0.0:
                    unknown_since = time.monotonic()
                if time.monotonic() - unknown_since > 10 * settings.poll_seconds:
                    print(  # noqa: T201
                        "[slurm] squeue keeps failing; stopped following. Collect "
                        f"later with: scenario-slurm collect {layout.batch_dir}",
                        flush=True,
                    )
                    return 1
            else:
                unknown_since = None
            time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        if settings.cancel_on_interrupt:
            cancel_jobs([job])
            print(  # noqa: T201
                f"\n[slurm] interrupted: cancelled job {job}. Run the cases that "
                f"did not finish with: scenario-slurm resubmit {layout.batch_dir}",
                flush=True,
            )
            _wait_until_gone(job, CANCEL_WAIT_SECONDS, settings.poll_seconds)
        else:
            print(  # noqa: T201
                f"\n[slurm] stopped following job {job}, which goes on. Follow it "
                f"again with: scenario-slurm wait {layout.batch_dir}",
                flush=True,
            )
            return 130
    return collect(layout)
