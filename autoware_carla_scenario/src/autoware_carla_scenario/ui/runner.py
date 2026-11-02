"""Scenario execution bridge using subprocess + progress tracking.

Runs ``uv run scenario`` in a subprocess to avoid importing CARLA
directly into the viewer process, providing SIGSEGV isolation.

Sweep parameter expansion is handled by :mod:`.sweep_resolver` before
this module is called, so each entry in *overrides_list* is a concrete
single-scenario run.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from .models import RunProgress, Status

logger = logging.getLogger(__name__)

RAW_OUTPUT_LOG = "raw_output.log"
"""Filename used to persist subprocess stdout/stderr alongside result JSON."""

# Global state for the current run.
_lock = threading.Lock()
_progress: RunProgress | None = None
_running = False


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


def _claim_run(total: int) -> bool:
    """Mark a run of *total* jobs as running, unless one already is.

    The check and the set happen under one lock acquisition, so two callers
    can never both claim a run.  The progress is reset with them, so that
    nobody sees the previous run's final ``done`` as this run's.
    """
    with _lock:
        global _running  # noqa: PLW0603
        global _progress  # noqa: PLW0603
        if _running:
            return False
        _running = True
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
        _running = False
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

        for i, overrides in enumerate(overrides_list):
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

            try:
                result = subprocess.run(  # noqa: S603
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    # A job may print bytes that are not UTF-8: replace them
                    # rather than fail to read the output (#1073).
                    encoding="utf-8",
                    errors="replace",
                )
                status: Status = "passed" if result.returncode == 0 else "failed"
                if result.returncode != 0:
                    logger.warning(
                        "Scenario %s failed (exit code %d): %s",
                        scenario_name,
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

            _set_progress(
                RunProgress(
                    current=i + 1,
                    total=total,
                    scenario_name=scenario_name,
                    status=status,
                )
            )

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
