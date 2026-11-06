"""``scenario.timeout_seconds``, the run's own fail-safe, read and checked.

One check for every place a run's config is turned into a run: a local run
and a local batch (``examples/run.py``), a SLURM submission, which refuses the
whole batch before anything is submitted, and each SLURM task
(``slurm/worker.py``), which reads the same key again on its node.

Free of heavy imports, so the submitting process and the worker can both
import it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .value_text import shown

__all__ = [
    "DEFAULT_RUN_TIMEOUT_SECONDS",
    "check_run_timeout",
    "run_timeout_seconds",
]

#: The fail-safe of a config that sets no ``scenario.timeout_seconds``.
DEFAULT_RUN_TIMEOUT_SECONDS: float = 60.0


def check_run_timeout(seconds: float) -> float:
    """*seconds* as ``scenario.timeout_seconds``, which must be positive and finite.

    Hydra reads ``scenario.timeout_seconds=nan`` (or ``inf``) as a float, and
    a NaN timeout never fires: the run would have no time bound at all.

    Raises:
        ValueError: naming the key, when it is not positive and finite.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            f"scenario.timeout_seconds must be positive and finite, got {seconds}"
        )
    return seconds


def run_timeout_seconds(cfg: Mapping[str, Any]) -> float:
    """The run's ``scenario.timeout_seconds`` in *cfg* (a run's config), checked.

    :data:`DEFAULT_RUN_TIMEOUT_SECONDS` when the config sets none: when the
    key is absent, and when it is ``null``.  A ``DeclarativeScenarioConfig``
    leaves it ``null`` to fall back to its document's own timeout, and an
    exported package writes the key either way.  That needs no larger run
    timeout: this one is only the default ``TimeoutCondition`` of a scenario
    that registers none of its own, and a ``DeclarativeScenario`` always
    registers its document's.

    Raises:
        ValueError: naming the key, when it is not a number (a string, a
            bool, a list, ...), or not positive and finite.
    """
    scenario = cfg.get("scenario") or {}
    value = scenario.get("timeout_seconds")
    if value is None:
        return DEFAULT_RUN_TIMEOUT_SECONDS
    # A bool is an int to Python, and ``float("15")`` would read a string.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "scenario.timeout_seconds must be a number of seconds, "
            f"got {value!r} ({type(value).__name__})"
        )
    try:
        seconds = float(value)
    except OverflowError:
        # An int too big for a float (10**400): not finite (#1652).
        raise ValueError(
            f"scenario.timeout_seconds must be positive and finite, got {shown(value)}"
        ) from None
    return check_run_timeout(seconds)
