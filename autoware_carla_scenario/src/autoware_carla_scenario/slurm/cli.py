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
import sys
from collections.abc import Callable
from pathlib import Path

from .collect import collect, unfinished_cases
from .manifest import BatchLayout
from .settings import SlurmSettings
from .status import (
    active_jobs,
    busy_reason,
    cancel_batch,
    named_cases,
    print_status,
)
from .submit import after_submit, submit_cases

__all__ = ["main"]


def _layout(path: str) -> BatchLayout:
    return BatchLayout(Path(path).expanduser().resolve())


# What status prints, whether the batch is busy, and what cancel cancels are
# the checked slurm.status's.  Bound here, not called through a function of
# this module, so that a batch's records are read as deep in the stack as
# they were read here (Python 3.10 counts the calls on the stack with the
# levels of nesting json.loads() reads).
_active_jobs = active_jobs
_busy = busy_reason
_status = print_status


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


def _resubmit(args: argparse.Namespace) -> int:
    layout = _layout(args.batch_dir)
    manifest = layout.read_manifest()
    reason = None if args.force else _busy(layout)
    if reason is not None:
        print(f"Error: {reason}", file=sys.stderr)  # noqa: T201
        return 1
    if args.cases:
        # The checked slurm.status's: an unknown entry is said there.
        named = named_cases(manifest, args.cases)
        if named is None:
            return 1
        chosen = named
    else:
        chosen = unfinished_cases(layout, failed=args.failed)
    if not chosen:
        print("Nothing to resubmit: every case has a verdict.")  # noqa: T201
        # A dry run never collects: it writes nothing (#1768).
        return collect(layout) if args.wait and not args.dry_run else 0
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
    """``scenario-slurm cancel`` (slurm.status's cancel_batch)."""
    return cancel_batch(_layout(args.batch_dir))


def _parser() -> argparse.ArgumentParser:
    """The ``scenario-slurm`` argument parser; every option is in docs/slurm.md."""
    parser = argparse.ArgumentParser(
        prog="scenario-slurm",
        description="Follow, collect, resubmit or cancel a batch submitted with "
        "launcher=slurm (docs/slurm.md).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    batch_help = "the batch's output directory (outputs/<date>/<time>)"

    status = commands.add_parser("status", help="the batch's cases, jobs and tasks")
    status.add_argument("batch_dir", help=batch_help)

    wait = commands.add_parser(
        "wait", help="follow a submission until it ends, then collect"
    )
    wait.add_argument("batch_dir", help=batch_help)
    wait.add_argument(
        "--submission",
        type=int,
        default=None,
        help="the submission to follow, by its index k "
        "(slurm/submissions/<k>.json; 0 is the first); default: the latest",
    )
    wait.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="seconds between two polls of squeue (below 0.1 is taken as 0.1, "
        "a negative number is refused); default: the batch's "
        "launcher.poll_seconds",
    )

    collect_cmd = commands.add_parser(
        "collect", help="print the summary and write batch_results.json"
    )
    collect_cmd.add_argument("batch_dir", help=batch_help)

    resubmit = commands.add_parser(
        "resubmit",
        help="run again the cases with no result, or that errored",
    )
    resubmit.add_argument("batch_dir", help=batch_help)
    resubmit.add_argument(
        "--failed",
        action="store_true",
        help="also the cases that ran and failed",
    )
    resubmit.add_argument(
        "--cases", default=None, help="these cases instead (labels or indices, a,b)"
    )
    resubmit.add_argument(
        "--array-parallelism",
        type=int,
        default=None,
        help="at most this many tasks at once (at least 1); "
        "default: the batch's launcher.array_parallelism",
    )
    resubmit.add_argument(
        "--cases-per-task",
        type=int,
        default=None,
        help="cases a task runs (at least 1); "
        "default: the batch's launcher.cases_per_task",
    )
    group = resubmit.add_mutually_exclusive_group()
    group.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=None,
        help="follow the new submission until it ends, then collect "
        "(collect at once when there is nothing to resubmit, unless with "
        "--dry-run); default: the batch's launcher.wait (with nothing to "
        "resubmit, it collects only with an explicit --wait)",
    )
    group.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="submit and exit; default: the batch's launcher.wait (with "
        "nothing to resubmit, it collects only with an explicit --wait)",
    )
    resubmit.add_argument(
        "--force",
        action="store_true",
        help="submit even while a job of the batch is queued or running, "
        "squeue cannot tell, or another submission of it is in flight or "
        "its sbatch timed out less than 600 s after its record was made",
    )
    resubmit.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cases it would resubmit, submitting and collecting "
        "nothing (even with --wait), and exit 0; it still exits 1 where a "
        "resubmit would refuse (a busy batch without --force, an unknown "
        "--cases entry, an out-of-range option) or on an error reading the batch",
    )

    cancel = commands.add_parser("cancel", help="scancel the batch's jobs")
    cancel.add_argument("batch_dir", help=batch_help)
    return parser


def main(argv: list[str] | None = None) -> int:
    """The ``scenario-slurm`` command."""
    parser = _parser()
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
        # Not through _cancel(): the records are read as deep as before.
        return cancel_batch(_layout(args.batch_dir))
    # OSError covers FileNotFoundError, and an unreadable manifest or an
    # unwritable batch directory (shared storage, a read-only mount): one
    # line, not a traceback (#1740).
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    sys.exit(main())
