"""The numbers of a lanelet sweep's case: its timeout and its exit status.

:class:`~autoware_carla_scenario.sweeper.lanelet_constraint_sweeper.LaneletConstraintSweeper`
gives each case a hard timeout, ``sweep.job_timeout_seconds`` in whole
seconds (:func:`whole_job_seconds`), and reads the exit status of a case off
its scenario's verdict (:func:`verdict_exit_code`).  This module is the
numeric half of that: the sweeper reads the setting as YAML gave it, and the
result JSON, and words what it refuses; nothing here needs Hydra, so the
library check compiles it (docs/typecheck.md).
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "DEFAULT_JOB_TIMEOUT",
    "FAILED_VERDICT_EXIT_CODE",
    "MAX_JOB_TIMEOUT",
    "REFUSED_EXIT_CODE",
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
