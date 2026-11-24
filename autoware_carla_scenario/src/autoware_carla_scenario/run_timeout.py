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

from .number_settings import number_setting
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

    *written* is the value as the config wrote it, which the message shows;
    *seconds* as :func:`~autoware_carla_scenario.value_text.shown` gives it
    without one, so that ``timeout_seconds: 0`` is refused as ``got 0``, not
    ``got 0.0`` (#2231).

    *key* is the name the message gives the value: ``timeout_seconds`` for
    :class:`~autoware_carla_scenario.ScenarioRunner`'s, ``run_scenario()``'s
    and :class:`~autoware_carla_scenario.ScenarioQueue`'s parameter of that
    name, which check it before anything is started, so that a timeout that
    never fires (or fires at once) is not found only once the ego is up
    (#2287).

    The one check of both paths, the config's and the parameter's (#2403):
    a bool (``True`` is no 1 s timeout), a string (``"5"``) or anything else
    that is not a number is refused as
    :func:`~autoware_carla_scenario.number_settings.number_setting` refuses
    it, and an int too big for a float (``10**400``) as not finite, rather
    than with a ``TypeError`` or an ``OverflowError`` naming no key.

    Raises:
        ValueError: naming the key, when it is not a number, or not positive
            and finite.
    """
    if isinstance(seconds, int) and not isinstance(seconds, bool):
        try:
            number = float(seconds)
        except OverflowError:
            # An int too big for a float (10**400): not finite (#1652).
            number = math.inf
    else:
        number = number_setting(key, seconds)
    if not math.isfinite(number) or number <= 0:
        shown_value = shown(seconds) if written is None else written
        raise ValueError(f"{key} must be positive and finite, got {shown_value}")
    return number


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
    return check_run_timeout(value)
