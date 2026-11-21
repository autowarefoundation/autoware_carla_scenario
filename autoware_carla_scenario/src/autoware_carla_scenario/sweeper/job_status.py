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
directories no other job of the sweep shares (:func:`owned_dirs`).
"""

from __future__ import annotations

import math
import signal
from pathlib import Path
from typing import Optional

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
#: retried (``server.cooldown_max_retries``).
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
