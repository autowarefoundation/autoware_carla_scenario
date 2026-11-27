"""Scenario execution bridge using subprocess + progress tracking.

Runs ``uv run scenario`` in a subprocess to avoid importing CARLA
directly into the viewer process, providing SIGSEGV isolation.

Sweep parameter expansion is handled by :mod:`.sweep_resolver` before
this module is called, so each entry in *overrides_list* is a concrete
single-scenario run.

A job's stdout and stderr are written to a file as the job runs, never held
in memory: ``raw_output.log`` in its job directory when the run is grouped as
a multirun, a temporary file (removed once the job has ended) otherwise, since
the directory Hydra gives an ungrouped job is not known here.  The last few
KB of it are what a failed job's warning logs.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO

from ..run_dirs import reserve_timestamped_dir
from ..utils.process_group import signal_group
from .case_paths import _override_label
from .models import RunProgress, Status

logger = logging.getLogger(__name__)

RAW_OUTPUT_LOG = "raw_output.log"
"""Filename used to persist subprocess stdout/stderr alongside result JSON."""

STOP_TIMEOUT_SECONDS = 10.0
"""How long a stopped job's process group has after ``SIGTERM``, before ``SIGKILL``."""

_RAW_LOG_MAX_BYTES = 256 * 1024  # 256 KB cap per log

_WARNING_TAIL_BYTES = 4 * 1024
"""How much of a failed job's output (its end) its warning logs."""


def _tail(log_path: Path, max_bytes: int) -> str:
    """The text of *log_path*, or its last *max_bytes* bytes (less the
    partial line they start in, when a newline follows it) after a note that
    it was truncated; bytes that are not UTF-8 are replaced.

    Raises:
        OSError: The file cannot be read.
    """
    size = log_path.stat().st_size
    if size <= max_bytes:
        return log_path.read_text(encoding="utf-8", errors="replace")
    with log_path.open(encoding="utf-8", errors="replace") as f:
        f.seek(size - max_bytes)
        tail = f.read()
    # Less the partial line it starts in, unless that is all of it: the end
    # of one line longer than max_bytes is shown rather than nothing.
    newline = tail.find("\n")
    if 0 <= newline < len(tail) - 1:
        tail = tail[newline + 1 :]
    return f"... (truncated, showing last {len(tail)} chars) ...\n{tail}"


def _read_raw_log(job_dir: Path) -> str:
    """Read raw terminal log from a job directory.

    Tries ``raw_output.log`` first (subprocess stdout/stderr captured by
    the runner), then falls back to ``run.log`` (Python logging output).
    Returns an empty string when neither file exists.  Large files are
    truncated to the last ``_RAW_LOG_MAX_BYTES`` bytes.  A symbolic link is
    not read: it may name any file on the machine (one in an unpacked results
    archive), which the viewer must not show.
    """
    for name in (RAW_OUTPUT_LOG, "run.log"):
        log_path = job_dir / name
        try:
            if log_path.is_symlink():
                continue
            return _tail(log_path, _RAW_LOG_MAX_BYTES)
        except OSError:
            continue
    return ""


# Global state for the current run.
_lock = threading.Lock()
_progress: RunProgress | None = None
_running = False
# Set by stop_run(): the worker starts no further job (#1351).
_stopping = False
# The job running now, started in a process group of its own, which
# stop_run() terminates.
_job: subprocess.Popen[bytes] | None = None
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
    """Send *sig* to the process group *pgid*; ``False`` if none of it is left
    (:func:`signal_group`), or it may not be signalled."""
    try:
        return signal_group(pgid, sig)
    except PermissionError:
        return False


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


def _failed_message(failed: int, total: int) -> str:
    """What a run's ``done`` says of its *failed* jobs of *total*: ``"k of N
    failed"``, or nothing when none failed (#2440)."""
    return f"{failed} of {total} failed" if failed > 0 else ""


def _end_run(failed: int = 0) -> None:
    """Mark the run as over, and its progress as ``done`` if it is not yet,
    with the *failed* jobs it counted (#2440).

    The progress stream ends only on ``done``: a run that stops before it
    publishes one (an exception in the worker, a stopped run, or a worker that
    never started) must not leave the stream following it forever.
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
                current=p.current,
                total=p.total,
                scenario_name="",
                status="done",
                message=_failed_message(failed, p.total),
                failed=failed,
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
    # The jobs that failed, which the run's `done` reports (#2440); a job a
    # stop ended is not one of them.
    failed = 0

    try:
        # When grouping as multirun, create a shared timestamped directory,
        # this run's alone (#1359).
        multirun_dir: str | None = None
        if group_as_multirun and total > 0:
            now = datetime.now()  # noqa: DTZ005
            base_dir = Path(cwd) if cwd else Path.cwd()
            try:
                multirun_dir = _reserve_multirun_dir(base_dir, now)
            except OSError as exc:
                # No job can run without it: the run fails as a whole, and
                # says why, not "Completed!" with nothing run (#2183).
                logger.error(
                    "Could not create the run's multirun directory under %s: %s",
                    base_dir,
                    exc,
                )
                _set_progress(
                    RunProgress(
                        current=0,
                        total=total,
                        scenario_name="",
                        status="done",
                        message="The run failed: could not create its multirun "
                        f"directory under {base_dir}: {exc}",
                    )
                )
                return
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
            # Where the job's output goes: its job directory's
            # raw_output.log when grouped, else a temporary file.
            log: Path | None = None
            temporary = False
            try:
                if multirun_dir:
                    base = Path(cwd) if cwd else Path.cwd()
                    job_dir = base / multirun_dir / str(i)
                    job_dir.mkdir(parents=True, exist_ok=True)
                    log = job_dir / RAW_OUTPUT_LOG
                else:
                    log = _temporary_log()
                    temporary = True
                returncode = _run_job(cmd, cwd, log)
                if returncode is None:
                    launched = False
                else:
                    status = "passed" if returncode == 0 else "failed"
                    if returncode != 0:
                        logger.warning(
                            "Scenario %s %s (exit code %d): %s",
                            scenario_name,
                            "was stopped" if _stop_requested() else "failed",
                            returncode,
                            _output_tail(log),
                        )
            except Exception as exc:
                # Whatever stops one job from being launched or collected (an
                # OSError, or anything unexpected) fails that job, not the
                # run: the next jobs still run (#1073).
                status = "failed"
                logger.error("Failed to run scenario %s: %s", scenario_name, exc)
            finally:
                if log is not None:
                    _settle_log(log, temporary)
            if not launched:
                # Stopped before this job started: it is not run.
                stopped = True
                break
            if status == "failed" and not _stop_requested():
                failed += 1

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

        # Signal completion, with how many jobs failed ("k of N failed").
        _set_progress(
            RunProgress(
                current=total,
                total=total,
                scenario_name="",
                status="done",
                message=_failed_message(failed, total),
                failed=failed,
            )
        )
    finally:
        _end_run(failed)


def _run_job(cmd: list[str], cwd: str | None, log: Path) -> int | None:
    """Run *cmd* in *cwd*, its stdout and stderr written to *log* as it runs
    (truncated first), and return its exit code.

    The output goes to the file, not through a pipe, so a job that prints a
    great deal is never held in memory.  The job is started in a session, so
    a process group, of its own, which :func:`stop_run` can terminate as a
    whole (#1351); it is ``None``, and nothing is started, once the run is
    being stopped.
    """
    with log.open("wb") as output:
        job = _launch(cmd, cwd, output)
        if job is None:
            return None
        collected = False
        try:
            returncode = job.wait()
            collected = True
        finally:
            with _lock:
                global _job  # noqa: PLW0603
                _job = None
            if not collected:
                # As subprocess.run() kills the job it fails to collect.
                _signal_group(job.pid, int(signal.SIGKILL))
    return returncode


def _launch(
    cmd: list[str], cwd: str | None, output: IO[bytes]
) -> subprocess.Popen[bytes] | None:
    """Start *cmd*, unless the run is being stopped, as the run's job, its
    stdout and stderr written to *output* (one file, in the order written).

    Under the lock that :func:`stop_run` takes, so a job is either never
    started or one that stop_run() sees.
    """
    with _lock:
        global _job  # noqa: PLW0603
        if _stopping:
            return None
        job: subprocess.Popen[bytes] = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _job = job
    return job


def _temporary_log() -> Path:
    """A new, empty temporary file for an ungrouped job's output."""
    fd, name = tempfile.mkstemp(prefix="scenario-job-", suffix=".log")
    os.close(fd)
    return Path(name)


def _output_tail(log: Path) -> str:
    """The end of a job's output in *log* (what its warning logs), its bytes
    that are not UTF-8 replaced (#1073); empty when it cannot be read."""
    try:
        return _tail(log, _WARNING_TAIL_BYTES)
    except OSError:
        return ""


def _settle_log(log: Path, temporary: bool) -> None:
    """Remove a job's output file once the job is over: a temporary one
    always, a job directory's ``raw_output.log`` when the job wrote nothing
    to it (so the scenario page shows ``run.log`` instead).

    The job directory, made for the log before the job started, is removed
    too when nothing else is in it -- a job that was never launched (the run
    was stopped first, or ``uv`` could not be started) and wrote nothing --
    so that the viewer does not list it as a case that left no result.  One
    that Hydra wrote to (``.hydra/``) is not empty, and is kept."""
    try:
        if temporary or log.stat().st_size == 0:
            log.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to tidy the job output file %s", log)
        return
    if not temporary:
        try:
            log.parent.rmdir()  # only while empty
        except OSError:
            pass


def _reserve_multirun_dir(base: Path, now: datetime) -> str:
    """Create a ``multirun/{date}/{time}`` of *now* under *base* that is this
    run's alone, and return it relative to *base*.

    Created with ``mkdir``, which fails when it exists, so a run started in
    the same second as another (a CLI sweep, another viewer's run) never
    shares it (#1359): the first gets ``{time}``, the
    next ``{time}-1``, then ``{time}-2``, and so on, as a batch's output
    directory does: the shared
    :func:`~autoware_carla_scenario.run_dirs.reserve_timestamped_dir`, which
    ``examples.run`` reserves its directories with too.  A symbolic link that
    resolves to nothing is raised on, not numbered past, as it always was here.
    """
    return reserve_timestamped_dir(base, "multirun", now, dangling_links_taken=False)


def _extract_scenario_name(overrides: list[str]) -> str:
    """The job's scenario, as the result views name it: its ``scenario=``
    override (``case_paths._override_label``), else ``scenario``."""
    label = _override_label(overrides)
    return label if label is not None else "scenario"


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
