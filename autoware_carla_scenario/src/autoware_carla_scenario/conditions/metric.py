"""A condition on a metric: how the ego drives, read off the world."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..metrics import _FAILURE_CATEGORIES, BUILT_IN_METRICS, read_metric, want_metric
from .base import BaseCondition, ScenarioResult
from .comparison import ComparisonRule, ScalarComparisonRule
from ..failure import FailureCategory

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["MetricCondition"]


class MetricCondition(BaseCondition):
    """Satisfied while a built-in metric satisfies a comparison.

    A metric (:mod:`autoware_carla_scenario.metrics`) reads how the ego drives
    -- whether it collides, leaves the road, drives against the traffic, how
    close it comes to other road users.  An ODD does not map onto one (it is an
    outcome, not a condition of the domain); a run watches it with this
    condition instead, usually as a fail condition::

        scenario.register_fail_condition(
            MetricCondition(COLLISION_AT_FAULT, label="collision at fault")
        )
        scenario.register_fail_condition(
            MetricCondition(MIN_DISTANCE_TO_OBSTACLE_M, ComparisonRule.LESS_THAN,
                            0.5, label="closer than half a metre")
        )

    A frame the metric reads nothing in (no ego yet, no road user to measure
    from) does not satisfy it.

    A CommonRoad metric (CriMe's, CommonRoad-Reach's:
    :data:`~autoware_carla_scenario.metrics.COMMONROAD_METRICS`) is evaluated
    by a worker process every tick of a run that watches one; building the
    condition asks for it (:func:`~autoware_carla_scenario.metrics.want_metric`).

    Args:
        metric: The metric's key, e.g.
            :data:`~autoware_carla_scenario.metrics.OFFROAD`.
        rule: The comparison of the metric's value against *value*.  Defaults
            to :attr:`ComparisonRule.GREATER_THAN_OR_EQUAL`, so an event metric
            with *value* ``1`` reads "it happens".
        value: The threshold, in the metric's unit (``1`` or ``0`` for an
            event).
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.

    Raises:
        ValueError: If *metric* is not a built-in metric's key.
    """

    # Declared for the static check (docs/typecheck.md).
    _metric: str
    _unit: str
    _comparison: ScalarComparisonRule

    def __init__(
        self,
        metric: str,
        rule: ComparisonRule = ComparisonRule.GREATER_THAN_OR_EQUAL,
        value: float = 1.0,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        super().__init__(label=label)
        if metric not in BUILT_IN_METRICS:
            known = ", ".join(sorted(BUILT_IN_METRICS))
            raise ValueError(
                f"MetricCondition: {metric!r} is not a metric; known: {known}"
            )
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self._metric = metric
        self._unit = BUILT_IN_METRICS[metric].unit
        # A CommonRoad metric is evaluated only in a run that reads it.
        want_metric(metric)
        self._comparison = ScalarComparisonRule(
            field=metric, rule=rule, value=value, tolerance=tolerance
        )

    @property
    def metric(self) -> str:
        """The metric's key."""
        return self._metric

    def failure_category(self) -> Optional[FailureCategory]:
        """What the metric reads: a collision metric
        :attr:`~FailureCategory.COLLISION`, the weaving ones
        :attr:`~FailureCategory.STABILITY`; ``None`` for one that reads more
        than one thing (off the road *or* a collision) or no failure."""
        return _FAILURE_CATEGORIES.get(self._metric)

    def get_details(self) -> dict[str, Any]:
        return {
            "metric": self._metric,
            "rule": self._comparison.rule.name,
            "value": self._comparison.value,
            "unit": self._unit,
            "tolerance": self._comparison.tolerance,
        }

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a result while the metric satisfies the rule.

        Args:
            world: The CARLA world instance.
            elapsed: Simulated seconds since the run began.

        Returns:
            ScenarioResult with passed=True while the comparison holds, None
            otherwise.
        """
        actual = read_metric(self._metric, world)
        if actual is None or not self._comparison.satisfied(actual):
            return None
        unit = f" {self._unit}" if self._unit else ""
        return ScenarioResult(
            passed=True,
            message=(
                f"{self._metric} {actual:.3f}{unit} {self._comparison.rule.text} "
                f"{self._comparison.value:.3f}{unit} at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
