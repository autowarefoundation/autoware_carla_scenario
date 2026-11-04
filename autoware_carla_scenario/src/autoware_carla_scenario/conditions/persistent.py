"""Persistent condition that requires a child to remain True for a duration."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseCondition, ScenarioResult, refuse_post_run, reports_only_failure
from ..failure import FailureCategory

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


def _warn_if_failure_only(condition: BaseCondition) -> None:
    """Warn that a :class:`PersistentCondition` around *condition* never fires:
    *condition* only ever reports a failure (:func:`~.base.reports_only_failure`),
    and the persistent condition counts only passing results.

    A warning, not an error: such a scenario ran before this check, and it
    still runs as it did.
    """
    if not reports_only_failure(condition):
        return
    logger.warning(
        "PersistentCondition: its child %r (%s) only ever reports a failure "
        "(passed=False), and a PersistentCondition counts only passing "
        "results, so it never fires. Register the child on its own, or wrap "
        "it in NotCondition(child), which turns its failure into a pass the "
        "PersistentCondition counts; to require that a state has not held for "
        "a while, wrap the condition for that state in "
        "NotCondition(..., holds_when_pending=True), as the cut-in example does",
        condition.label,
        type(condition).__name__,
    )


class PersistentCondition(BaseCondition):
    """Requires child condition to remain True for *duration* consecutive seconds.

    Evaluation rules:

    - While the child returns ``passed=True``, a timer accumulates.
    - When the timer reaches *duration*, this condition returns ``passed=True``.
    - If the child returns ``None`` or ``passed=False``, the timer resets.

    So it never fires around a condition that only ever reports a failure
    (:func:`~.base.reports_only_failure`: a collision, off-road, existence or
    timeout condition), and says so in a warning when it is built.  Wrap such
    a child in a plain :class:`~.not_condition.NotCondition`, which turns its
    failure into a pass.

    Args:
        condition: The child condition to monitor.
        duration: Minimum consecutive seconds the child must pass before
            this condition passes.

    Raises:
        ValueError: If *duration* is not positive, or not finite: NaN fails
            every comparison, so a NaN duration would never fire.
    """

    # Declared for the static check (docs/typecheck.md).
    _condition: BaseCondition
    _duration: float
    _true_start: Optional[float]

    def __init__(
        self, condition: BaseCondition, duration: float, *, label: str | None = None
    ) -> None:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive and finite")
        refuse_post_run(condition, "PersistentCondition")
        _warn_if_failure_only(condition)
        super().__init__(label=label if label is not None else condition.label)
        self._condition = condition
        self._duration = duration
        self._true_start: Optional[float] = None

    def close(self) -> None:
        """Close the child (:meth:`BaseCondition.close`)."""
        self._condition.close()

    def reset(self) -> None:
        """Stop the timer, and reset the child (:meth:`BaseCondition.reset`):
        a child that held when an earlier run ended has not held since this
        run's start."""
        super().reset()
        self._true_start = None
        self._condition.reset()

    def failure_category(self) -> Optional[FailureCategory]:
        """What the wrapped condition says."""
        return self._condition.failure_category()

    def get_details(self) -> dict[str, Any]:
        return {
            "wrapper": "persistent",
            "required_duration_seconds": self._duration,
            "child": self._condition.to_summary_dict(),
        }

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result when the child has been passing for *duration* seconds.

        Args:
            world: The CARLA world instance.
            elapsed: Simulated seconds since the run began.

        Returns:
            :class:`ScenarioResult` with ``passed=True`` if the child has been
            continuously passing for at least *duration* seconds, ``None``
            otherwise.
        """
        result = self._condition.check(world, elapsed)

        if result is not None and result.passed:
            if self._true_start is None:
                self._true_start = elapsed
            # elapsed is a running sum of the tick, so a duration that is a
            # whole number of ticks can come out a hair short of it; without
            # the tolerance the condition would pass one tick late.
            if elapsed - self._true_start >= self._duration - 1e-9:
                return ScenarioResult(
                    passed=True,
                    message=(
                        f"Child condition '{self._condition.label}' passed"
                        f" continuously since {self._true_start:.2f}s, for"
                        f" {elapsed - self._true_start:.2f}s"
                        f" (threshold: {self._duration}s)"
                    ),
                    elapsed_seconds=elapsed,
                )
            return None

        # Child returned None or failed — reset timer.
        self._true_start = None
        return None
