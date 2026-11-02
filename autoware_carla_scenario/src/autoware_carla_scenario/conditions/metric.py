"""A condition on a metric: how the ego drove, judged once the run is over."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from ..failure import FailureCategory
from ..metrics import _FAILURE_CATEGORIES, BUILT_IN_METRICS
from .base import BaseCondition, ScenarioResult
from .comparison import ComparisonRule, ScalarComparisonRule

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["MetricCondition"]


class MetricCondition(BaseCondition):
    """Fails a run whose metric was NG for longer than *max_ng_seconds*.

    A metric (:mod:`autoware_carla_scenario.metrics`) reads how the ego drives
    -- whether it collides, leaves the road, how close it comes to other road
    users, how critical CriMe and CommonRoad-Reach find the moment.  An ODD
    does not map onto one (it is an outcome, not a condition of the domain); a
    run is judged on it with this condition instead, a fail condition only::

        scenario.register_fail_condition(
            MetricCondition(COLLISION_AT_FAULT, label="collision at fault")
        )
        scenario.register_fail_condition(
            MetricCondition(CRIME_TTC_S, ComparisonRule.LESS_THAN, 1.5,
                            max_ng_seconds=0.5, label="TTC under 1.5 s")
        )

    Metrics are computed once the run is over, in a batch: the runner records
    the road users every tick, and when the run ends replays them on the
    world's map (``docs/metrics.md``, "How metrics are computed").  A frame
    is **NG** when the metric satisfies the comparison.  Each NG frame counts
    for the time until the next frame evaluated; the run fails when those add
    up to more than *max_ng_seconds* -- with ``0``, at its first NG frame --
    at the time they do, unless something else failed it earlier.  A frame the
    metric reads nothing in (no ego, no road user to measure from) is not NG.

    Being judged after the run, it neither ends a run early nor composes:
    :class:`~autoware_carla_scenario.conditions.PersistentCondition`, a NOT,
    an AND or an OR refuses one (*max_ng_seconds* is how long it must hold),
    and so does ``register_pass_condition``.

    Args:
        metric: The metric's key, e.g.
            :data:`~autoware_carla_scenario.metrics.OFFROAD`.
        rule: The comparison that makes a frame NG.  Defaults to
            :attr:`ComparisonRule.GREATER_THAN_OR_EQUAL`, so an event metric
            with *value* ``1`` reads "it happens".
        value: The threshold, in the metric's unit (``1`` or ``0`` for an
            event).
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        max_ng_seconds: How long, in total, the metric may be NG before the run
            fails (simulated seconds, ``>= 0``).

    Raises:
        ValueError: If *metric* is not a built-in metric's key, or
            *tolerance* or *max_ng_seconds* is negative.
    """

    #: What tells the runner, and the conditions that wrap others, that this
    #: one is judged after the run (:func:`~.base.is_post_run`).
    IS_POST_RUN: ClassVar[bool] = True

    # Declared for the static check (docs/typecheck.md).
    _metric: str
    _unit: str
    _comparison: ScalarComparisonRule
    _max_ng_seconds: float
    _judgement: dict[str, float]

    def __init__(
        self,
        metric: str,
        rule: ComparisonRule = ComparisonRule.GREATER_THAN_OR_EQUAL,
        value: float = 1.0,
        tolerance: float = 1e-6,
        *,
        max_ng_seconds: float = 0.0,
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
        if max_ng_seconds < 0:
            raise ValueError("max_ng_seconds must be non-negative")
        self._metric = metric
        self._unit = BUILT_IN_METRICS[metric].unit
        self._comparison = ScalarComparisonRule(
            field=metric, rule=rule, value=value, tolerance=tolerance
        )
        self._max_ng_seconds = float(max_ng_seconds)
        self._judgement = {}

    @property
    def metric(self) -> str:
        """The metric's key."""
        return self._metric

    @property
    def max_ng_seconds(self) -> float:
        """How long the metric may be NG before the run fails."""
        return self._max_ng_seconds

    def failure_category(self) -> Optional[FailureCategory]:
        """What the metric reads: a collision metric
        :attr:`~FailureCategory.COLLISION`, the weaving ones
        :attr:`~FailureCategory.STABILITY`; ``None`` for one that reads more
        than one thing (off the road *or* a collision) or no failure."""
        return _FAILURE_CATEGORIES.get(self._metric)

    def get_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "metric": self._metric,
            "rule": self._comparison.rule.name,
            "value": self._comparison.value,
            "unit": self._unit,
            "tolerance": self._comparison.tolerance,
            "max_ng_seconds": self._max_ng_seconds,
        }
        details.update(self._judgement)
        return details

    def is_ng(self, value: Optional[float]) -> bool:
        """Whether a frame reading *value* is NG."""
        return value is not None and self._comparison.satisfied(value)

    def judge(
        self, times: list[float], values: list[Optional[float]]
    ) -> Optional[ScenarioResult]:
        """Judge the run on its metric's series: *values* read at *times*
        (simulated seconds, ascending).

        Each NG frame counts until the next frame; the last, until the run's
        end at its own time.  Returns a result (``passed=True``: the fail
        condition holds) at the time the NG time first exceeds
        :attr:`max_ng_seconds`, ``None`` when it never does.  What it found --
        the NG time, when it first went NG, the most critical value -- is in
        :meth:`get_details` afterwards.
        """
        ng = 0.0
        first: Optional[float] = None
        worst: Optional[float] = None
        failed_at: Optional[float] = None
        for i in range(len(times)):
            value = values[i]
            if not self.is_ng(value):
                continue
            assert value is not None
            at = times[i]
            if first is None:
                first = at
            if worst is None or self._more_critical(value, worst):
                worst = value
            span = max(times[i + 1] - at, 0.0) if i + 1 < len(times) else 0.0
            if failed_at is None:
                if self._max_ng_seconds == 0.0:
                    failed_at = at
                elif ng + span > self._max_ng_seconds:
                    # When, within this frame, the NG time crossed the limit.
                    failed_at = at + (self._max_ng_seconds - ng)
            ng += span
        self._judgement = {"ng_seconds": ng}
        if first is not None:
            self._judgement["first_ng_at"] = first
        if worst is not None:
            self._judgement["worst"] = worst
        self._last_satisfied = failed_at is not None
        if failed_at is None:
            self._last_message = ""
            return None
        unit = f" {self._unit}" if self._unit else ""
        message = (
            f"{self._metric} {self._comparison.rule.text} "
            f"{self._comparison.value:.3f}{unit} for {ng:.2f}s in total "
            f"(more than {self._max_ng_seconds:.2f}s; first at {first:.2f}s, "
            f"worst {worst:.3f}{unit})"
        )
        self._last_message = message
        return ScenarioResult(passed=True, message=message, elapsed_seconds=failed_at)

    def _more_critical(self, value: float, than: float) -> bool:
        """Whether *value* is further into NG than *than*: below a threshold
        it is NG under, lower; above, higher."""
        if self._comparison.rule in (
            ComparisonRule.LESS_THAN,
            ComparisonRule.LESS_THAN_OR_EQUAL,
        ):
            return value < than
        return value > than

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Nothing while the run goes: a metric is judged once it is over
        (:meth:`judge`)."""
        return None

    def reset(self) -> None:
        super().reset()
        self._judgement = {}
