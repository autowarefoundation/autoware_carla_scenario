"""``scenario-slurm``: follow, collect, resubmit or cancel a SLURM batch.

A batch submitted with ``scenario ... launcher=slurm`` is named by its output
directory (docs/slurm.md)::

    scenario-slurm status   outputs/2026-10-18/12-00-00
    scenario-slurm wait     outputs/2026-10-18/12-00-00   # follow, then collect
    scenario-slurm collect  outputs/2026-10-18/12-00-00   # summary, batch_results.json
    scenario-slurm resubmit outputs/2026-10-18/12-00-00   # the cases with no result
    scenario-slurm cancel   outputs/2026-10-18/12-00-00

Light: no CARLA, no Hydra.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .collect import collect, unfinished_cases
from .manifest import SBATCH_IN_FLIGHT_SECONDS, BatchLayout, sbatch_in_flight
from .settings import SlurmSettings
from .submit import (
    ACTIVE_STATES,
    after_submit,
    cancel_jobs,
    squeue_states,
    submit_cases,
)

__all__ = ["main"]

logger = logging.getLogger(__name__)


def _layout(path: str) -> BatchLayout:
    return BatchLayout(Path(path).expanduser().resolve())


def _active_jobs(
    layout: BatchLayout, below: int | None = None
) -> tuple[list[str], list[str], list[int], list[int]]:
    """The batch's array jobs SLURM still lists, those squeue could not be
    asked about (which may be running too), the submissions whose sbatch has
    not returned yet (``manifest.sbatch_in_flight``), and those whose record
    cannot be read yet is younger than ``SBATCH_IN_FLIGHT_SECONDS``: of every
    submission, or of those before *below*.

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
    for index in layout.submissions():
        if below is not None and index >= below:
            continue
        path = layout.submission(index)
        try:
            submission = layout.read_submission(index)
        except FileNotFoundError:
            continue  # removed since: its sbatch failed
        except (OSError, ValueError):
            submission = None
        if not isinstance(submission, dict):
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
        job = submission.get("job_id")
        if not job:
            if sbatch_in_flight(submission):
                in_flight.append(index)
            continue
        states = squeue_states(str(job))
        if states is None:
            unknown.append(str(job))
        elif any(state in ACTIVE_STATES for state in states.values()):
            active.append(str(job))
    return active, unknown, in_flight, unreadable


def _busy(layout: BatchLayout, below: int | None = None) -> str | None:
    """Why the batch cannot be resubmitted now, if it cannot: a job of it is
    queued or running, ``squeue`` cannot tell, or another submission of it is
    in flight (another ``resubmit``, the viewer's, #1352)."""
    active, unknown, in_flight, unreadable = _active_jobs(layout, below)
    if active:
        return (
            f"job(s) {', '.join(active)} of this batch are still queued or "
            "running; wait for them (scenario-slurm wait), cancel them "
            "(scenario-slurm cancel), or pass --force"
        )
    if in_flight:
        return (
            f"submission(s) {', '.join(str(i) for i in in_flight)} of this batch "
            "are being submitted (sbatch has not returned yet), by another "
            "resubmit or the viewer; try again once it has, or pass --force. "
            "A record left by a submitter that died is ignored "
            f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s after it was made"
        )
    if unreadable:
        files = ", ".join(str(layout.submission(i)) for i in unreadable)
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


def _refuse_if_busy(layout: BatchLayout) -> Callable[[int], None]:
    """submit_cases' *before_sbatch* for a resubmit: with the new submission's
    index reserved, refuse should one made before it be in flight.  Of two
    resubmits at once, the one with the later index sees the other's record
    and refuses; so at most one of them runs the cases."""

    def check(index: int) -> None:
        reason = _busy(layout, below=index)
        if reason is not None:
            raise RuntimeError(reason)

    return check


def _status(layout: BatchLayout) -> int:
    manifest = layout.read_manifest()
    cases = manifest["cases"]
    done = passed = errored = 0
    for case in cases:
        record = layout.read_result(int(case["index"]))
        if record is None:
            continue
        done += 1
        passed += bool(record["result"]["passed"])
        errored += bool(record.get("errored"))
    print(f"Batch {layout.batch_dir}: {len(cases)} case(s)")  # noqa: T201
    print(  # noqa: T201
        f"  {done} with a result ({passed} passed, {errored} errored), "
        f"{len(cases) - done} without"
    )
    for index in layout.submissions():
        submission = layout.read_submission(index)
        job = submission.get("job_id")
        states = squeue_states(str(job)) if job else {}
        queued = (
            "squeue unavailable"
            if states is None
            else ", ".join(f"{t}:{s}" for t, s in sorted(states.items())) or "ended"
        )
        print(  # noqa: T201
            f"  submission {index}: job {job}, {len(submission['chunks'])} task(s), "
            f"{len(submission['cases'])} case(s); in the queue: {queued}"
        )
        for task, chunk in enumerate(submission["chunks"]):
            status = layout.read_task(index, task) or {}
            ports = status.get("ports") or {}
            rss = status.get("peak_rss_mb")
            print(  # noqa: T201
                f"    task {task}: {status.get('state', 'not started')}"
                f"{' on ' + status['host'] if status.get('host') else ''}"
                f"{' port ' + str(ports['rpc']) if ports.get('rpc') else ''}"
                f", {len(status.get('done') or [])}/{len(chunk)} case(s)"
                f"{f', peak RSS {rss} MB' if rss else ''}"
                f"{' -- ' + status['error'] if status.get('error') else ''}"
            )
    return 0


def _resubmit(args: argparse.Namespace) -> int:
    layout = _layout(args.batch_dir)
    manifest = layout.read_manifest()
    reason = None if args.force else _busy(layout)
    if reason is not None:
        print(f"Error: {reason}", file=sys.stderr)  # noqa: T201
        return 1
    if args.cases:
        labels = {str(c["label"]): int(c["index"]) for c in manifest["cases"]}
        # A case named twice (by its label and its index, say) runs once.
        picked: dict[int, None] = {}
        for item in args.cases.split(","):
            item = item.strip()
            if item in labels:
                picked[labels[item]] = None
            elif item.isdigit() and 0 <= int(item) < len(manifest["cases"]):
                picked[int(item)] = None
            else:
                print(f"Error: no case {item!r} in the batch", file=sys.stderr)  # noqa: T201
                return 1
        chosen = sorted(picked)
    else:
        chosen = unfinished_cases(layout, failed=args.failed)
    if not chosen:
        print("Nothing to resubmit: every case has a result.")  # noqa: T201
        return collect(layout) if args.wait else 0
    settings = SlurmSettings.from_mapping(manifest["launcher"]).replace(
        array_parallelism=args.array_parallelism,
        cases_per_task=args.cases_per_task,
        wait=args.wait,
    )
    labels_of = [manifest["cases"][i]["label"] for i in chosen]
    print(f"Resubmitting {len(chosen)} case(s): {', '.join(labels_of)}")  # noqa: T201
    if args.dry_run:
        return 0
    index = submit_cases(
        layout,
        manifest,
        settings,
        sorted(chosen),
        before_sbatch=None if args.force else _refuse_if_busy(layout),
    )
    return after_submit(layout, index, settings)


def _wait(args: argparse.Namespace) -> int:
    from .submit import follow  # noqa: PLC0415

    layout = _layout(args.batch_dir)
    manifest = layout.read_manifest()
    submissions = layout.submissions()
    if not submissions:
        print("Error: the batch was never submitted", file=sys.stderr)  # noqa: T201
        return 1
    index = submissions[-1] if args.submission is None else args.submission
    settings = SlurmSettings.from_mapping(manifest["launcher"])
    if args.poll_seconds is not None:
        settings = settings.replace(poll_seconds=args.poll_seconds)
    return follow(layout, index, settings)


def _cancel(args: argparse.Namespace) -> int:
    layout = _layout(args.batch_dir)
    jobs: list[str] = []
    for index in layout.submissions():
        submission = layout.read_submission(index)
        for key in ("job_id", "collect_job_id"):
            if submission.get(key):
                jobs.append(str(submission[key]))
    if not jobs:
        print("The batch has no job to cancel.")  # noqa: T201
        return 0
    print(f"Cancelling job(s) {', '.join(jobs)}")  # noqa: T201
    return 0 if cancel_jobs(jobs) else 1


def main(argv: list[str] | None = None) -> int:
    """The ``scenario-slurm`` command."""
    parser = argparse.ArgumentParser(
        prog="scenario-slurm",
        description="Follow, collect, resubmit or cancel a batch submitted with "
        "launcher=slurm (docs/slurm.md).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="the batch's cases, jobs and tasks")
    status.add_argument("batch_dir")

    wait = commands.add_parser(
        "wait", help="follow a submission until it ends, then collect"
    )
    wait.add_argument("batch_dir")
    wait.add_argument("--submission", type=int, default=None)
    wait.add_argument("--poll-seconds", type=float, default=None)

    collect_cmd = commands.add_parser(
        "collect", help="print the summary and write batch_results.json"
    )
    collect_cmd.add_argument("batch_dir")

    resubmit = commands.add_parser(
        "resubmit",
        help="run again the cases with no result, or that errored",
    )
    resubmit.add_argument("batch_dir")
    resubmit.add_argument(
        "--failed",
        action="store_true",
        help="also the cases that ran and failed",
    )
    resubmit.add_argument(
        "--cases", default=None, help="these cases instead (labels or indices, a,b)"
    )
    resubmit.add_argument("--array-parallelism", type=int, default=None)
    resubmit.add_argument("--cases-per-task", type=int, default=None)
    group = resubmit.add_mutually_exclusive_group()
    group.add_argument("--wait", dest="wait", action="store_true", default=None)
    group.add_argument("--no-wait", dest="wait", action="store_false")
    resubmit.add_argument(
        "--force", action="store_true", help="even while a job of the batch runs"
    )
    resubmit.add_argument("--dry-run", action="store_true")

    cancel = commands.add_parser("cancel", help="scancel the batch's jobs")
    cancel.add_argument("batch_dir")

    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            return _status(_layout(args.batch_dir))
        if args.command == "wait":
            return _wait(args)
        if args.command == "collect":
            return collect(_layout(args.batch_dir))
        if args.command == "resubmit":
            return _resubmit(args)
        return _cancel(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    sys.exit(main())
