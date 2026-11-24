"""The numbers of a lanelet sweep's case: its timeout and its exit status.

:class:`~autoware_carla_scenario.sweeper.lanelet_constraint_sweeper.LaneletConstraintSweeper`
gives each case a hard timeout, ``sweep.job_timeout_seconds`` in whole
seconds (:func:`whole_job_seconds`), and reads the exit status of a case off
its scenario's verdict (:func:`verdict_exit_code`).  This module is the
numeric half of that: the sweeper reads the setting as YAML gave it, and the
result JSON, and words what it refuses; nothing here needs Hydra, so the
library check compiles it (docs/typecheck.md).  So are why a job given up
was given up, in words (:func:`marker_reason`), the case
``--resume-from N`` starts at (:func:`resume_from_env`) and the job
directories no other job of the sweep shares (:func:`owned_dirs`).  The
files a job's directory is cleared of and given are this module's too: an
earlier run's or a failed attempt's outputs (:func:`_remove_earlier_outputs`,
:func:`_remove_run_stems`), and the marker of a job that did not finish or
was given up (:func:`_write_unfinished_marker`, :func:`_write_marker`,
:func:`_remove_marker`).
"""

from __future__ import annotations

import fnmatch
import logging
import math
import signal
from pathlib import Path
from typing import Optional

from ..gate.errored import (
    remove_case_errored,
    write_case_errored,
    write_case_unfinished,
)
from ..run_outputs import remove_run_outputs, run_output_paths, run_output_stems

__all__ = [
    "DEFAULT_JOB_TIMEOUT",
    "FAILED_VERDICT_EXIT_CODE",
    "MAX_JOB_TIMEOUT",
    "REFUSED_EXIT_CODE",
    "RESUME_FROM_ENV",
    "RESUME_FROM_EXPECTED",
    "marker_reason",
    "owned_dirs",
    "resume_from_env",
    "verdict_exit_code",
    "whole_job_seconds",
]

# The sweeper's logger: what it logs of a job's directory reads as before
# this was split out of it (as sweeper.isolation's does).
logger = logging.getLogger("autoware_carla_scenario.sweeper.lanelet_constraint_sweeper")

#: Default hard timeout per job (seconds).  Can be overridden via
#: ``sweep.job_timeout_seconds`` in the scenario YAML.
DEFAULT_JOB_TIMEOUT = 120

#: The largest ``sweep.job_timeout_seconds``: ``signal.alarm`` takes a C
#: ``int`` and raises ``OverflowError`` above it (about 68 years).
MAX_JOB_TIMEOUT = 2**31 - 1

#: The exit status of a refused run (docs/usage.md, "Exit Codes").
REFUSED_EXIT_CODE = 2

#: The exit status of a forked case whose scenario ran to a FAILED verdict
#: (its result JSON says ``passed: false``).  Final, as in
#: :class:`~autoware_carla_scenario.scenario_queue.ScenarioQueue`, which
#: retries a run that raised but not one that completed: only a case exiting
#: 1 (it raised, or wrote no result), timing out or dying of a signal is
#: retried (``server.cooldown_max_retries``), and only when its attempt wrote
#: no verdict: a run that wrote its result and died winding down keeps it
#: (#2416).
FAILED_VERDICT_EXIT_CODE = 3

#: The environment variable ``scenario --resume-from N`` passes the sweeper
#: *N* in (``examples.run.RESUME_FROM_ENV``): Hydra's parser would reject the
#: flag.
RESUME_FROM_ENV = "SWEEP_RESUME_FROM"

#: What ``sweep.resume_from`` and :data:`RESUME_FROM_ENV` must be, as the
#: message refusing either says it.
RESUME_FROM_EXPECTED = (
    "a whole number of at least 1 (the 1-indexed case to start at; "
    "0 or unset for none)"
)


def whole_job_seconds(seconds: float) -> Optional[int]:
    """*seconds* as the whole number of seconds a job's timeout is, or
    ``None`` when it is not one from 0 (no timeout) to
    :data:`MAX_JOB_TIMEOUT`.

    A case is stopped by ``signal.alarm``, which counts whole seconds, so a
    fraction is refused rather than truncated (``0.5`` would become 0, no
    timeout at all), and so are a negative, NaN or infinite value, rather
    than silently disabling the timeout, and one above
    :data:`MAX_JOB_TIMEOUT`, which ``signal.alarm`` cannot take.
    """
    if not math.isfinite(seconds):
        return None
    if not 0.0 <= seconds <= float(MAX_JOB_TIMEOUT):
        return None
    if seconds != math.floor(seconds):
        return None
    return int(seconds)


def verdict_exit_code(verdict: Optional[bool]) -> int:
    """The exit status of a case that completed, by the verdict of the result
    JSON it wrote: 0 when it passed, :data:`FAILED_VERDICT_EXIT_CODE` when it
    failed, and 1 when it wrote no readable result (``None``), which is
    retried as a run that raised is."""
    if verdict is None:
        return 1
    return 0 if verdict else FAILED_VERDICT_EXIT_CODE


def marker_reason(exit_code: Optional[int], timeout: Optional[int] = None) -> str:
    """Why a job was given up, in words, from its last exit status."""
    if exit_code is None:
        return "timed out" if not timeout else f"timed out after {timeout}s"
    if exit_code < 0:
        try:
            name = signal.Signals(-exit_code).name
        except ValueError:
            name = f"signal {-exit_code}"
        return f"crashed ({name})"
    if exit_code == REFUSED_EXIT_CODE:
        return f"refused (exit code {exit_code})"
    return f"exit code {exit_code}"


def resume_from_env(env_value: Optional[str], from_config: int) -> int:
    """The 1-based case the sweep starts at, given :data:`RESUME_FROM_ENV`
    (*env_value*: ``--resume-from N``, or the variable set by hand) and
    *from_config*, ``sweep.resume_from`` already checked (0 when unset).

    The variable, when set to a whole number of at least 1, wins; an empty
    or unset variable, or ``0``, leaves *from_config*.

    Raises:
        ValueError: Naming the variable, when it is not a whole number of at
            least 0 (a sign, a fraction, an exponent or a non-ASCII digit).
    """
    text = (env_value or "").strip()
    if not text:
        return from_config
    if not (text.isascii() and text.isdigit()):
        # No sign, fraction, exponent or non-ASCII digit.
        raise ValueError(
            f"{RESUME_FROM_ENV} must be {RESUME_FROM_EXPECTED}, got {text!r}"
        )
    from_env = int(text)
    return from_env if from_env >= 1 else from_config


def owned_dirs(
    job_dirs: list[Optional[tuple[Path, Optional[str]]]],
    skipped: list[Optional[tuple[Path, Path, Optional[str]]]],
) -> set[Path]:
    """The directories of *job_dirs* (the jobs a sweep runs) that no other
    job of the sweep maps to, *skipped* (those a ``--resume-from`` skips,
    as ``LaneletConstraintSweeper._job_paths`` gives them) included.
    None at all when the directory of a job cannot be told: it may be any."""
    counts: dict[Path, int] = {}
    for job_dir in job_dirs:
        if job_dir is None:
            return set()
        counts[job_dir[0]] = counts.get(job_dir[0], 0) + 1
    for paths in skipped:
        if paths is None:
            return set()
        counts[paths[1]] = counts.get(paths[1], 0) + 1
    return {directory for directory, count in counts.items() if count == 1}


#: The logs that services a run starts write into its directory under names
#: of their own, which :func:`..run_outputs.run_output_stems` reads as the
#: CARLA recorder log of a run (``X.log``): a containerised backend's
#: (``backend_<name>.log``, appended to), Autoware's (``autoware-<n>-<name>.log``
#: or, in a batch, ``autoware-<stem>.log``, ``autoware-build.log``,
#: ``autoware-deps.log``) and the viewer runner's.
#: An earlier sweep's are not a run of the job, and stay.
_SERVICE_LOGS: tuple[str, ...] = (
    "backend_*.log",
    "autoware-*.log",
    "raw_output.log",
    "run.log",
)


def _is_service_log(directory: Path, stem: str) -> bool:
    """Whether all a run named *stem* has in *directory* is a log one of
    :data:`_SERVICE_LOGS` names: a service's log, not a run."""
    found = [path for path in run_output_paths(directory, stem) if path.exists()]
    return (
        len(found) == 1
        and found[0].name == f"{stem}.log"
        and any(
            fnmatch.fnmatchcase(found[0].name, pattern) for pattern in _SERVICE_LOGS
        )
    )


def _remove_earlier_outputs(job_dir: tuple[Path, str | None], label: str) -> None:
    """Remove the run outputs an earlier sweep left in *job_dir* (its
    directory and Hydra's job log), before the job's first attempt in this
    sweep: every stem :func:`..run_outputs.run_output_stems` finds there,
    with :func:`..run_outputs.remove_run_outputs`, as a retry removes the
    failed attempt's (#1956).  Hydra's job log, its ``.hydra`` directory, the
    job's marker (``case_errored.json``, replaced as the attempt starts), the
    logs of the services runs start (:data:`_SERVICE_LOGS`) and any other
    file that is not a run's output are left alone.  Logged, never
    raised, when one cannot be removed: the job runs anyway, under the next
    free stem."""
    _remove_run_stems(
        job_dir, label, "earlier run", ": the job runs again and replaces it."
    )


def _remove_run_stems(
    job_dir: tuple[Path, str | None],
    label: str,
    what: str,
    why: str,
    *,
    exclude: Optional[set[str]] = None,
) -> None:
    """Remove every run in *job_dir* (its directory and Hydra's job log) but
    those of *exclude* (none when ``None``) and the services' logs (:func:`_is_service_log`),
    with :func:`..run_outputs.remove_run_outputs`: the earlier sweep's
    (:func:`_remove_earlier_outputs`) or a failed attempt's, before its
    retry (#1373).  Each is logged as ``[<label>] Removing the outputs of
    the <what> <stem> in <directory><why>``; one that cannot be removed is
    logged, never raised (#2070)."""
    directory, job_log = job_dir
    skipped = set[str]() if exclude is None else exclude
    for stem in sorted(run_output_stems(directory, job_log) - skipped):
        if _is_service_log(directory, stem):
            continue
        logger.info(
            "[%s] Removing the outputs of the %s %s in %s%s",
            label,
            what,
            stem,
            directory,
            why,
        )
        try:
            remove_run_outputs(directory, stem)
        except OSError as exc:
            logger.warning(
                "Cannot remove the outputs of the %s %s in %s (%s).",
                what,
                stem,
                directory,
                exc,
            )


def _remove_marker(directory: Path) -> None:
    """Remove the marker (``case_errored.json``) of a job that ended with a
    verdict from *directory*; logged, never raised, when it cannot be."""
    try:
        remove_case_errored(directory)
    except OSError as exc:
        logger.warning("Cannot remove the error marker in %s (%s).", directory, exc)


def _write_unfinished_marker(
    job_dir: tuple[Path, str | None] | None, batch: tuple[str, ...], attempt: int
) -> Optional[float]:
    """Give a job the marker saying it did not finish
    (:func:`..gate.errored.write_case_unfinished`) before its attempt
    *attempt* (from 1), and return when it was written (its modification
    time): what the attempt writes is no older (#2416).  Logged, never
    raised, when it cannot be, and ``None``; ``None`` too when the job's
    directory is not known (the sweeper's
    ``LaneletConstraintSweeper._job_output_dir`` logged why)."""
    if job_dir is None:
        return None
    try:
        marker = write_case_unfinished(
            job_dir[0], overrides=list(batch), attempt=attempt
        )
        return marker.stat().st_mtime
    except OSError as exc:
        logger.warning(
            "Cannot write the unfinished-job marker in %s (%s).", job_dir[0], exc
        )
        return None


def _write_marker(
    job_dir: tuple[Path, str | None] | None,
    batch: tuple[str, ...],
    exit_code: int | None,
    attempts: int,
    timeout: int | None = None,
) -> None:
    """Leave the marker of a job given up (``case_errored.json``,
    :mod:`..gate.errored`) in its directory; logged, never raised, when it
    cannot be, or the directory is not known."""
    if job_dir is None:
        logger.warning(
            "Cannot tell the output directory of the job given up for "
            "overrides %s; it leaves no error marker, and scenario-gate will "
            "not see it.",
            batch,
        )
        return
    try:
        write_case_errored(
            job_dir[0],
            overrides=list(batch),
            exit_code=exit_code,
            attempts=attempts,
            reason=marker_reason(exit_code, timeout),
        )
    except OSError as exc:
        logger.warning("Cannot write the error marker in %s (%s).", job_dir[0], exc)
