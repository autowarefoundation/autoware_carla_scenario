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
from typing import Any, Optional

from .value_text import shown

__all__ = [
    "DEFAULT_RUN_TIMEOUT_SECONDS",
    "check_run_timeout",
    "run_timeout_seconds",
]

#: The fail-safe of a config that sets no ``scenario.timeout_seconds``.
DEFAULT_RUN_TIMEOUT_SECONDS: float = 60.0


def check_run_timeout(
    seconds: float,
    written: Optional[str] = None,
    key: str = "scenario.timeout_seconds",
) -> float:
    """*seconds* as ``scenario.timeout_seconds``, which must be positive and finite.

    Hydra reads ``scenario.timeout_seconds=nan`` (or ``inf``) as a float, and
    a NaN timeout never fires: the run would have no time bound at all.

    *written* is the value as the config wrote it (its ``repr()``, as
    :func:`~autoware_carla_scenario.value_text.shown` gives it), which the
    message shows; *seconds*' own ``repr()`` without one.  A caller that
    converted the value (:func:`run_timeout_seconds`, of an int) passes it,
    so that ``timeout_seconds: 0`` is refused as ``got 0``, not ``got 0.0``
    (#2231).

    *key* is the name the message gives the value: ``timeout_seconds`` for
    :class:`~autoware_carla_scenario.ScenarioRunner`'s, ``run_scenario()``'s
    and :class:`~autoware_carla_scenario.ScenarioQueue`'s parameter of that
    name, which check it before anything is started, so that a timeout that
    never fires (or fires at once) is not found only once the ego is up
    (#2287).

    Raises:
        ValueError: naming the key, when it is not positive and finite.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        shown_value = repr(seconds) if written is None else written
        raise ValueError(f"{key} must be positive and finite, got {shown_value}")
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

    The ``scenario`` group is a mapping: an absent or null one sets no
    timeout either, but ``false``, ``0`` or a list is refused (#2143).

    Raises:
        ValueError: naming the key, when it is not a number (a string, a
            bool, a list, ...), or not positive and finite, or when
            ``scenario`` is not a mapping.
    """
    scenario = cfg.get("scenario")
    if scenario is None:
        return DEFAULT_RUN_TIMEOUT_SECONDS
    if not isinstance(scenario, Mapping):
        raise ValueError(
            f"scenario must be a mapping, got {shown(scenario)} "
            f"({type(scenario).__name__})"
        )
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
    return check_run_timeout(seconds, shown(value))
