"""Scenario execution bridge using subprocess + progress tracking.

Runs ``uv run scenario`` in a subprocess to avoid importing CARLA
directly into the viewer process, providing SIGSEGV isolation.

Sweep parameter expansion is handled by :mod:`.sweep_resolver` before
this module is called, so each entry in *overrides_list* is a concrete
single-scenario run.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from .models import RunProgress, Status

logger = logging.getLogger(__name__)

RAW_OUTPUT_LOG = "raw_output.log"
"""Filename used to persist subprocess stdout/stderr alongside result JSON."""

STOP_TIMEOUT_SECONDS = 10.0
"""How long a stopped job's process group has after ``SIGTERM``, before ``SIGKILL``."""

# Global state for the current run.
_lock = threading.Lock()
_progress: RunProgress | None = None
_running = False
# Set by stop_run(): the worker starts no further job (#1351).
_stopping = False
# The job running now, started in a process group of its own, which
# stop_run() terminates.
_job: subprocess.Popen[str] | None = None
# The thread running the current run's jobs, once it has started.
_worker: threading.Thread | None = None


def get_progress() -> RunProgress | None:
    """Return the current run progress, or ``None`` if no run is active."""
    with _lock:
        return _progress


def get_status() -> tuple[bool, RunProgress | None]:
    """Return ``(is_running, progress)`` under a single lock acquisition."""
    with _lock:
        return _running, _progress


def _set_progress(p: RunProgress) -> None:
    with _lock:
        global _progress  # noqa: PLW0603
        _progress = p


def is_running() -> bool:
    """Return ``True`` if a scenario run is currently in progress."""
    with _lock:
        return _running


def start_run(
    overrides_list: list[list[str]],
    base_path: Path | None = None,
    extra_overrides: list[str] | None = None,
    group_as_multirun: bool = False,
) -> bool:
    """Start scenario execution in a background thread.

    The run counts as running from the moment this returns, before the
    worker thread is scheduled, and its progress replaces the previous
    run's: a second call refuses to start a run while one is in progress,
    and the progress stream does not end on the last run's ``done`` (#664).

    Args:
        overrides_list: List of override lists. Each inner list is passed
            as overrides to a single ``uv run scenario`` invocation.
        base_path: Working directory for the subprocess.
        extra_overrides: Additional Hydra overrides appended to every run.
        group_as_multirun: When ``True``, all jobs share a single
            ``multirun/{date}/{time}/`` directory with numbered
            subdirectories (0, 1, 2, …) instead of each writing to
            its own ``outputs/`` directory.

    Returns:
        ``True`` if the run was started, ``False`` if it was refused
        because another run is still in progress.
    """
    if not _claim_run(len(overrides_list)):
        logger.warning("A scenario run is already in progress")
        return False

    # `extra_overrides or []`, in a form Codon can type.
    extras: list[str] = []
    if extra_overrides:
        extras = extra_overrides
    thread = threading.Thread(
        target=_run_worker,
        args=(
            overrides_list,
            base_path,
            extras,
            group_as_multirun,
        ),
        daemon=True,
    )
    # The run's worker from before it starts, so no other run's thread can
    # take its place; stop_run() joins it only once it is alive.
    with _lock:
        global _worker  # noqa: PLW0603
        _worker = thread
    # A thread that does not start (``RuntimeError: can't start new
    # thread``) never runs the worker, which would end the run: end it here.
    started = False
    try:
        thread.start()
        started = True
    finally:
        if not started:
            _end_run()
    return True


def stop_run(timeout: float = STOP_TIMEOUT_SECONDS) -> None:
    """Stop the run in progress, if there is one (#1351).

    No further job starts: the worker checks between jobs.  The job running
    now, a process group of its own, gets ``SIGTERM``, and ``SIGKILL`` if any
    of the group is still there *timeout* seconds later.  The worker records
    that job as failed and ends the run, its progress ``done``.  This returns
    once the group is gone and the worker has ended, or the bounded waits
    have run out.
    """
    with _lock:
        global _stopping  # noqa: PLW0603
        if not _running:
            return
        _stopping = True
        job = _job
        worker = _worker
    if job is not None:
        logger.info("Stopping the running scenario (pid %d)", job.pid)
        _terminate_group(job.pid, timeout)
    if worker is not None and worker.is_alive():
        worker.join(timeout)


def _stop_requested() -> bool:
    with _lock:
        return _stopping


def _signal_group(pgid: int, sig: int) -> bool:
    """Send *sig* to the process group *pgid*; ``False`` if none of it is left."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _terminate_group(pgid: int, timeout: float) -> None:
    """``SIGTERM`` to the group *pgid*, then ``SIGKILL`` after *timeout* seconds.

    The group is gone once each of its processes has exited and been
    reaped (the job's own by the worker that waits for it).
    """
    if not _signal_group(pgid, int(signal.SIGTERM)):
        return
    deadline = time.monotonic() + timeout
    while _signal_group(pgid, 0):
        if time.monotonic() >= deadline:
            logger.warning(
                "The stopped scenario (pid %d) still runs %gs after SIGTERM: SIGKILL",
                pgid,
                timeout,
            )
            _signal_group(pgid, int(signal.SIGKILL))
            return
        time.sleep(0.1)


def _claim_run(total: int) -> bool:
    """Mark a run of *total* jobs as running, unless one already is.

    The check and the set happen under one lock acquisition, so two callers
    can never both claim a run.  The progress is reset with them, so that
    nobody sees the previous run's final ``done`` as this run's.
    """
    with _lock:
        global _running  # noqa: PLW0603
        global _progress  # noqa: PLW0603
        global _stopping  # noqa: PLW0603
        global _worker  # noqa: PLW0603
        if _running:
            return False
        _running = True
        _stopping = False
        _worker = None
        _progress = RunProgress(
            current=0, total=total, scenario_name="", status="running"
        )
        return True


def _end_run() -> None:
    """Mark the run as over, and its progress as ``done`` if it is not yet.

    The progress stream ends only on ``done``: a run that stops before it
    publishes one (an exception in the worker, or a worker that never
    started) must not leave the stream following it forever.
    """
    with _lock:
        global _running  # noqa: PLW0603
        global _progress  # noqa: PLW0603
        global _stopping  # noqa: PLW0603
        _running = False
        _stopping = False
        p = _progress
        if p is None:
            _progress = RunProgress(current=0, total=0, scenario_name="", status="done")
        elif p.status != "done":
            _progress = RunProgress(
                current=p.current, total=p.total, scenario_name="", status="done"
            )


def _run_worker(
    overrides_list: list[list[str]],
    base_path: Path | None,
    extra_overrides: list[str],
    group_as_multirun: bool,
) -> None:
    """Background worker that executes scenarios sequentially.

    :func:`start_run` has marked the run as running; the worker ends it
    (:func:`_end_run`) however it stops.
    """
    total = len(overrides_list)
    cwd = str(base_path) if base_path else None

    try:
        # When grouping as multirun, create a shared timestamped directory.
        multirun_dir: str | None = None
        if group_as_multirun and total > 0:
            now = datetime.now()  # noqa: DTZ005
            multirun_dir = _multirun_dir(now)
            logger.info("Grouping %d jobs under %s", total, multirun_dir)

        stopped = False
        for i, overrides in enumerate(overrides_list):
            if _stop_requested():
                stopped = True
                break
            scenario_name = _extract_scenario_name(overrides)
            _set_progress(
                RunProgress(
                    current=i + 1,
                    total=total,
                    scenario_name=scenario_name,
                    status="running",
                )
            )

            # `["uv", "run", "scenario", *overrides, *extra_overrides]`:
            # extend() takes any iterable, as unpacking does.
            cmd = ["uv", "run", "scenario"]
            cmd.extend(overrides)
            cmd.extend(extra_overrides)
            # Direct each job's output to multirun/{date}/{time}/{index}/
            if multirun_dir:
                cmd.append(f"hydra.run.dir={multirun_dir}/{i}")
            logger.info("Running [%d/%d]: %s", i + 1, total, " ".join(cmd))

            status: Status = "failed"
            launched = True
            try:
                result = _run_job(cmd, cwd)
                if result is None:
                    launched = False
                else:
                    status = "passed" if result.returncode == 0 else "failed"
                    if result.returncode != 0:
                        logger.warning(
                            "Scenario %s %s (exit code %d): %s",
                            scenario_name,
                            "was stopped" if _stop_requested() else "failed",
                            result.returncode,
                            result.stderr[-500:] if result.stderr else "",
                        )
                    # Save raw terminal output to the job directory.
                    if multirun_dir:
                        base = Path(cwd) if cwd else Path.cwd()
                        _save_raw_output(result, base / multirun_dir / str(i))
            except Exception as exc:
                # Whatever stops one job from being launched or collected (an
                # OSError, or anything unexpected) fails that job, not the
                # run: the next jobs still run (#1073).
                status = "failed"
                logger.error("Failed to run scenario %s: %s", scenario_name, exc)
            if not launched:
                # Stopped before this job started: it is not run.
                stopped = True
                break

            _set_progress(
                RunProgress(
                    current=i + 1,
                    total=total,
                    scenario_name=scenario_name,
                    status=status,
                )
            )

        if stopped:
            # _end_run() publishes the `done`, after the last job that ran.
            logger.info("The run was stopped: the jobs after it do not run")
            return

        # Signal completion
        _set_progress(
            RunProgress(
                current=total,
                total=total,
                scenario_name="",
                status="done",
            )
        )
    finally:
        _end_run()


def _run_job(
    cmd: list[str], cwd: str | None
) -> subprocess.CompletedProcess[str] | None:
    """Run *cmd* as ``subprocess.run(cmd, cwd=cwd, capture_output=True)`` does.

    The job is started in a session, so a process group, of its own, which
    :func:`stop_run` can terminate as a whole (#1351); it is ``None``, and
    nothing is started, once the run is being stopped.
    """
    job = _launch(cmd, cwd)
    if job is None:
        return None
    collected = False
    try:
        stdout, stderr = job.communicate()
        returncode = job.wait()
        collected = True
    finally:
        with _lock:
            global _job  # noqa: PLW0603
            _job = None
        if not collected:
            # As subprocess.run() kills the job it fails to collect.
            _signal_group(job.pid, int(signal.SIGKILL))
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def _launch(cmd: list[str], cwd: str | None) -> subprocess.Popen[str] | None:
    """Start *cmd*, unless the run is being stopped, as the run's job.

    Under the lock that :func:`stop_run` takes, so a job is either never
    started or one that stop_run() sees.
    """
    with _lock:
        global _job  # noqa: PLW0603
        if _stopping:
            return None
        job: subprocess.Popen[str] = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # A job may print bytes that are not UTF-8: replace them
            # rather than fail to read the output (#1073).
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        _job = job
    return job


def _multirun_dir(now: datetime) -> str:
    """``multirun/{date}/{time}`` of *now*: ``f"multirun/{now:%Y-%m-%d}/{now:%H-%M-%S}"``.

    Spelled out from the fields, as Codon formats no datetime; the same string
    for every year from 1000 on (strftime's ``%Y`` pads years before 1000
    only on some Pythons).
    """
    date = f"{now.year:04d}-{now.month:02d}-{now.day:02d}"
    time = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    return f"multirun/{date}/{time}"


def _save_raw_output(
    result: subprocess.CompletedProcess[str],
    job_dir: Path,
) -> None:
    """Save subprocess stdout/stderr to ``raw_output.log`` in *job_dir*."""
    parts: list[str] = []
    for stream in (result.stdout, result.stderr):
        if stream:
            parts.append(stream)
    if not parts:
        return
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / RAW_OUTPUT_LOG).write_text("\n\n".join(parts), encoding="utf-8")
    except OSError:
        logger.warning("Failed to save raw output to %s", job_dir)


def _extract_scenario_name(overrides: list[str]) -> str:
    """Extract a human-readable scenario name from override arguments."""
    for ov in overrides:
        if ov.startswith("scenario="):
            return ov[len("scenario=") :]
    return "scenario"


def build_command(
    scenario: str,
    extra_overrides: list[str] | None = None,
    sweeper: str = "",
) -> tuple[list[str], str]:
    """Build the ``uv run scenario`` command for preview purposes.

    Returns:
        A 2-tuple of ``(command_tokens, note)``.  *note* is a
        human-readable string (e.g. sweep info) or empty.
    """
    cmd = ["uv", "run", "scenario"]
    cmd.append(f"scenario={scenario}")
    if extra_overrides:
        cmd.extend(extra_overrides)
    note = f"sweep: {sweeper} (resolved to N individual jobs)" if sweeper else ""
    return cmd, note
