"""SLURM's command-line tools, as the submitting side and the viewer run them:
``sbatch``, ``squeue``, ``scancel``, ``scontrol`` and ``sinfo`` (docs/slurm.md).

What :mod:`.submit` and ``ui.slurm_launch`` ask SLURM, split out of them so
that the library check compiles it (#1634); both re-export these names.  The
command lines and what is read of the tools' output are :mod:`.scripts`' and
:mod:`.settings`'.

Only the standard library here: the ``scenario-slurm`` commands import this
without CARLA.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence

from .scripts import squeue_task_states
from .settings import DEFAULT_MAX_ARRAY_SIZE, parse_max_array_size, require_sbatch

__all__ = [
    "ACTIVE_STATES",
    "CANCEL_WAIT_SECONDS",
    "SBATCH_TIMEOUT_SECONDS",
    "SCANCEL_TIMEOUT_SECONDS",
    "SbatchTimedOut",
    "availability",
    "cancel_jobs",
    "max_array_size",
    "partitions",
    "squeue_states",
]

logger = logging.getLogger(__name__)

#: The states of a task SLURM has not finished with.
ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "RUNNING",
        "COMPLETING",
        "SUSPENDED",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
    }
)

#: After a Ctrl-C's ``scancel``, how long ``submit.follow`` waits for the tasks
#: to leave the queue (stopping their CARLAs, recording their state) before it
#: collects.
CANCEL_WAIT_SECONDS = 60.0

#: How long ``sbatch`` may take before it is given up on (#1786): a submit
#: host whose controller does not answer would otherwise block
#: ``launcher=slurm`` and ``scenario-slurm resubmit`` for good.  Well under
#: ``manifest.SBATCH_IN_FLIGHT_SECONDS`` (600 s), so a record is not taken for
#: a dead submitter's while its sbatch still runs.
SBATCH_TIMEOUT_SECONDS = 120.0

#: How long ``scancel`` may take before ``cancel_jobs`` gives up (#1786).
SCANCEL_TIMEOUT_SECONDS = 60.0


class SbatchTimedOut(RuntimeError):
    """``sbatch`` did not return within :data:`SBATCH_TIMEOUT_SECONDS` and was
    stopped (#1786).  Unlike a refusal, the controller may have taken the job
    all the same, so the submission record its tasks read is kept."""

    # Spelled out: Codon 0.19 cannot raise an exception subclass that leaves
    # its constructor to the built-in base (docs/typecheck.md).
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _chunks(indices: list[int], size: int) -> list[list[int]]:
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def _run_sbatch(command: list[str]) -> str:
    """Run *command* (an ``sbatch --parsable``); the job id it printed.

    Raises:
        SbatchTimedOut: sbatch did not end within
            :data:`SBATCH_TIMEOUT_SECONDS` (#1786); a :class:`RuntimeError`.
        RuntimeError: sbatch failed or printed no job id.
    """
    command = [require_sbatch(), *command[1:]]
    logger.info("Submitting: %s", shlex.join(command))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=SBATCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        # run() killed sbatch.  Whether the controller took the job before
        # that cannot be told from here: say where to look.
        raise SbatchTimedOut(
            f"sbatch timed out after {SBATCH_TIMEOUT_SECONDS} s; the SLURM "
            "controller may not be answering, and may have taken the job all "
            "the same: its submission record is kept (no job id), so its tasks "
            "can run. If `squeue --me` lists the job, collect the batch once it "
            "ends (scenario-slurm collect), or scancel it; resubmits are "
            "refused while the record is in flight, and the record is ignored "
            "600 s after it was made"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"sbatch failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    # --parsable: "<job id>" or "<job id>;<cluster>".  Nothing printed is no
    # job id, not an IndexError past callers catching RuntimeError (#1630).
    lines = completed.stdout.strip().splitlines()
    job = lines[-1].split(";")[0].strip() if lines else ""
    if not job:
        raise RuntimeError(f"sbatch printed no job id: {completed.stdout!r}")
    return job


def max_array_size() -> int:
    """The cluster's ``MaxArraySize`` (``scontrol show config``): an array
    holds at most this many tasks.  SLURM's default, 1001, when ``scontrol``
    cannot be asked or names none (#1347)."""
    try:
        completed = subprocess.run(
            ["scontrol", "show", "config"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("scontrol show config failed (%s): MaxArraySize 1001", exc)
        return DEFAULT_MAX_ARRAY_SIZE
    limit: int | None = None
    if completed.returncode == 0:
        limit = parse_max_array_size(completed.stdout)
    if limit is None:
        logger.debug(
            "scontrol show config (exit %d) named no MaxArraySize: 1001",
            completed.returncode,
        )
        return DEFAULT_MAX_ARRAY_SIZE
    return limit


def _array_refused(message: str) -> bool:
    """Whether sbatch's *message* refuses an array's size (SLURM's "Invalid
    job array specification", or one naming MaxArraySize)."""
    lowered = message.lower()
    return "invalid job array" in lowered or "maxarraysize" in lowered


def squeue_states(job: str) -> dict[int, str] | None:
    """The state of each task of array *job* that SLURM still lists.

    ``{}`` once the job is gone from the queue (or never was); ``None`` when
    ``squeue`` cannot be asked (not on PATH, or it failed for another reason).
    """
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-r", "-j", str(job), "-o", "%i %T"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        if "Invalid job id" in completed.stderr:
            return {}
        return None
    return squeue_task_states(job, completed.stdout)


def _any_active(states: dict[int, str]) -> bool:
    """Whether any of *states* is one of :data:`ACTIVE_STATES`."""
    # A function of its own: Codon 0.19 crashes (exit status -11) on this
    # generator over an Optional local tested for None in the caller (#1634).
    return any(state in ACTIVE_STATES for state in states.values())


def _wait_until_gone(job: str, timeout: float, poll_seconds: float) -> bool:
    """Wait, at most *timeout* s, until SLURM lists no active task of *job*:
    collecting sooner would find the cancelled tasks' results half written.
    Whether they left; a second Ctrl-C stops waiting."""
    deadline = time.monotonic() + timeout
    try:
        while True:
            states = squeue_states(job)
            if states is not None and not _any_active(states):
                return True
            if time.monotonic() >= deadline:
                break
            time.sleep(min(poll_seconds, 1.0))
    except KeyboardInterrupt:
        pass
    print(  # noqa: T201
        f"[slurm] job {job} is still in the queue; collecting what there is",
        flush=True,
    )
    return False


def cancel_jobs(jobs: Sequence[str]) -> bool:
    """``scancel`` *jobs* (this user's own); whether scancel succeeded."""
    names = [str(job) for job in jobs if job]
    if not names:
        return True
    try:
        completed = subprocess.run(
            ["scancel", *names],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=SCANCEL_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        print(f"[slurm] could not run scancel: {exc}", file=sys.stderr)  # noqa: T201
        return False
    except subprocess.TimeoutExpired:
        print(  # noqa: T201
            f"[slurm] scancel {' '.join(names)} timed out after "
            f"{SCANCEL_TIMEOUT_SECONDS} s",
            file=sys.stderr,
        )
        return False
    if completed.returncode != 0:
        print(  # noqa: T201
            f"[slurm] scancel {' '.join(names)} failed: {completed.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


def availability() -> tuple[bool, str]:
    """Whether batches can be submitted from here, and if not, why."""
    missing = [tool for tool in ("sbatch", "squeue") if shutil.which(tool) is None]
    if missing:
        return False, (
            f"SLURM's {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
            "not on this viewer's PATH: run the viewer on a SLURM submit (login) "
            "host to launch batches through SLURM (docs/slurm.md)."
        )
    return True, ""


def partitions() -> list[str]:
    """The partitions ``sinfo`` lists (the default one first), if it can."""
    try:
        done = subprocess.run(
            ["sinfo", "-h", "-o", "%P"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    names = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    default = [n.rstrip("*") for n in names if n.endswith("*")]
    every: set[str] = {n.rstrip("*") for n in names}
    defaults: set[str] = set(default)
    others: set[str] = every - defaults
    return default + sorted(others)
