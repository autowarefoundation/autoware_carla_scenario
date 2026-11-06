"""Which CommonRoad metrics are heavy, and whether a run evaluates them.

The heaviest CommonRoad metrics (:data:`HEAVY_METRICS`) take seconds a frame;
the environment turns them off (:data:`HEAVY_METRICS_ENV`), or the
``scenario`` command does for its runs (:data:`HEAVY_METRICS_DECIDED_ENV`),
and :func:`evaluates` says whether a metric is evaluated.

The plain environment parsing and checks of
:class:`~autoware_carla_scenario.commonroad.session.CommonRoadSettings`, kept
apart from its workers so that the library check compiles them;
``commonroad.session`` re-exports every name here.
"""

from __future__ import annotations

import math
import os
from typing import Optional

from .metrics import CRIME_TTR_S
from .value_text import shown

__all__ = [
    "DEFAULT_METRIC_PERIODS_S",
    "HEAVY_METRICS",
    "HEAVY_METRICS_DECIDED_ENV",
    "HEAVY_METRICS_ENV",
    "evaluates",
]

#: The CommonRoad metrics that take seconds a frame: CriMe's TTR simulates
#: the ego's evasive manoeuvres, about 2 s a frame with a collision ahead.
#: The library check reads ``CRIME_TTR_S`` through ``metrics``, as Codon 0.19
#: can report a constant imported by name and read at module level as not
#: defined (#1693, ``typecheck.transform.redirect_imports``).
HEAVY_METRICS: frozenset[str] = frozenset({CRIME_TTR_S})

#: The metrics evaluated less often than every
#: :attr:`~autoware_carla_scenario.commonroad.session.CommonRoadSettings.period_s`
#: by default, and how often (simulated seconds): the heavy ones every 0.5 s.
DEFAULT_METRIC_PERIODS_S: dict[str, float] = {key: 0.5 for key in HEAVY_METRICS}

#: Set to ``0`` (or ``false``, ``off``, ``no``) to turn the heavy metrics off
#: (:attr:`~autoware_carla_scenario.commonroad.session.CommonRoadSettings.heavy_metrics`).
HEAVY_METRICS_ENV = "AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS"

#: What the ``scenario`` command decided (``1`` or ``0``), from its configs'
#: ``metrics.heavy`` and :data:`HEAVY_METRICS_ENV`: set by it, not by the user,
#: to carry the decision to the runs' own processes.  Read before
#: :data:`HEAVY_METRICS_ENV` when set; the command never changes the user's
#: :data:`HEAVY_METRICS_ENV`.
HEAVY_METRICS_DECIDED_ENV = "AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS_DECIDED"


def _heavy_metrics_on(value: Optional[str]) -> bool:
    """Whether *value* (an environment variable's, ``None``: unset) leaves
    the heavy metrics on: unless it is ``0``, ``false``, ``off`` or ``no``."""
    return (value or "").strip().lower() not in ("0", "false", "off", "no")


def _heavy_metrics_from_user() -> bool:
    """Whether :data:`HEAVY_METRICS_ENV`, as the user set it, leaves the heavy
    metrics on."""
    return _heavy_metrics_on(os.environ.get(HEAVY_METRICS_ENV))


def _heavy_metrics_default() -> bool:
    """Whether the heavy metrics are on by default: as the ``scenario``
    command decided (:data:`HEAVY_METRICS_DECIDED_ENV`), else as
    :data:`HEAVY_METRICS_ENV` says."""
    decided = os.environ.get(HEAVY_METRICS_DECIDED_ENV, "").strip()
    if decided:
        return _heavy_metrics_on(decided)
    return _heavy_metrics_from_user()


def _check_period(name: str, period: float) -> None:
    try:
        finite = math.isfinite(period)
    except OverflowError:
        # An int too big for a float (10**400), which str() may refuse (#1652).
        raise ValueError(
            f"{name} must be positive and finite, got {shown(period)}"
        ) from None
    if not (finite and period > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {period}")


def evaluates(heavy: bool, key: str) -> bool:
    """Whether the metric *key* is evaluated, *heavy* saying whether the heavy
    metrics are on: not when it is heavy and they are off."""
    return heavy or key not in HEAVY_METRICS
