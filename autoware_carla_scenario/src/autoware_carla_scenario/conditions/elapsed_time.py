"""Elapsed-time-based scenario pass condition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import BaseCondition, ScenarioResult
from .comparison import ComparisonRule, ScalarComparisonRule

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


class ElapsedTimeCondition(BaseCondition):
    """Pass condition that triggers when elapsed time satisfies a comparison.

    By default the condition fires when elapsed time **reaches or exceeds**
    *duration_seconds* (``GREATER_THAN_OR_EQUAL``), which is backward-compatible
    with the original behaviour.

    Unlike :class:`TimeoutCondition` (which returns a *failure*), this condition
    returns a *success* result, making it suitable for scenarios that should pass
    after a certain amount of time (e.g. "drive for 30 seconds without issues").

    *elapsed* is a running sum of the simulator tick, so a duration that is a
    whole number of ticks often reads a hair short of (or past) its exact value
    on that tick.  Every rule therefore compares with a ``1e-9`` s time
    tolerance, as :class:`TimeoutCondition` and :class:`PersistentCondition`
    do:

    - ``>=`` holds once ``elapsed >= duration - 1e-9`` and ``<`` while
      ``elapsed < duration - 1e-9``, so both switch on the tick that reaches
      *duration*;
    - ``>`` holds once ``elapsed > duration + 1e-9`` and ``<=`` while
      ``elapsed <= duration + 1e-9``, so both switch on the tick after it;
    - ``==`` holds while ``|elapsed - duration| <= tolerance + 1e-9``, so a
      *tolerance* of ``0`` still matches the tick that reaches *duration*.

    Args:
        duration_seconds: Time threshold in seconds.  Must be positive.
        rule: Comparison operator applied to *elapsed* vs *duration_seconds*.
            Defaults to :attr:`ComparisonRule.GREATER_THAN_OR_EQUAL`.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
            Defaults to ``1e-6``.
    """

    # Declared for the static check (docs/typecheck.md).
    _comparison: ScalarComparisonRule

    def __init__(
        self,
        duration_seconds: float,
        rule: ComparisonRule = ComparisonRule.GREATER_THAN_OR_EQUAL,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        super().__init__(label=label)
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self._comparison = ScalarComparisonRule(
            field="elapsed", rule=rule, value=duration_seconds, tolerance=tolerance
        )

    def get_details(self) -> dict[str, Any]:
        return {
            "duration_seconds": self._comparison.value,
            "rule": self._comparison.rule.name,
            "tolerance": self._comparison.tolerance,
        }

    @property
    def duration_seconds(self) -> float:
        """The time threshold in seconds."""
        return self._comparison.value

    def _satisfied(self, elapsed: float) -> bool:
        """Return whether *elapsed* meets the rule, with a 1e-9 s time tolerance.

        A summed clock lands a hair either side of a whole-tick duration, so
        each rule moves its bound by ``1e-9`` toward the side where a tick
        that reaches *duration* counts as having reached it (#836).
        """
        rule = self._comparison.rule
        duration = self._comparison.value
        if rule == ComparisonRule.GREATER_THAN_OR_EQUAL:
            return elapsed >= duration - 1e-9
        if rule == ComparisonRule.GREATER_THAN:
            return elapsed > duration + 1e-9
        if rule == ComparisonRule.LESS_THAN:
            return elapsed < duration - 1e-9
        if rule == ComparisonRule.LESS_THAN_OR_EQUAL:
            return elapsed <= duration + 1e-9
        if rule == ComparisonRule.EQUAL_TO:
            return abs(elapsed - duration) <= self._comparison.tolerance + 1e-9
        raise ValueError(f"Unknown comparison rule: {rule}")

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a success result if the elapsed time satisfies the rule.

        Args:
            world: The CARLA world instance (unused).
            elapsed: Simulated seconds since the run began.

        Returns:
            ScenarioResult with passed=True if the condition is met, None otherwise.
        """
        if self._satisfied(elapsed):
            rule_text = self._comparison.rule.text
            return ScenarioResult(
                passed=True,
                message=(
                    f"Elapsed time {elapsed:.2f}s {rule_text}"
                    f" target duration {self.duration_seconds:.2f}s"
                ),
                elapsed_seconds=elapsed,
            )
        return None
