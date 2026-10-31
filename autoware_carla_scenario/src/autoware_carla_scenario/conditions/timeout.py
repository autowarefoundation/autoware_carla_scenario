"""Timeout-based scenario fail condition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import BaseCondition, ScenarioResult
from ..failure import FailureCategory

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


class TimeoutCondition(BaseCondition):
    """Fail condition that triggers when elapsed time exceeds the timeout.

    Measured on the simulated clock, like every other duration: *timeout* is
    part of what the scenario says -- "this manoeuvre should be over within N
    seconds" -- and so has to mean the same thing on every host.

    ``ScenarioRunner`` registers one of these as a default fail condition on
    every scenario that registers no timeout of its own; see
    ``timeout_seconds`` there.  A scenario that registers one (bare, or inside
    an :class:`OrCondition` or a :class:`StickyCondition`) runs for as long as
    it says, and the run's default timeout does not apply to it.
    """

    # Declared for the static check (docs/typecheck.md).
    timeout_seconds: float

    def __init__(self, timeout_seconds: float = 60.0, *, label: str) -> None:
        """Initialize the timeout condition.

        Args:
            timeout_seconds: Number of seconds before failing. Defaults to 60.0.
            label: Human-readable label identifying this condition.
        """
        super().__init__(label=label)
        self.timeout_seconds = timeout_seconds

    def failure_category(self) -> Optional[FailureCategory]:
        """Running out of time: :attr:`~FailureCategory.PROGRESS`."""
        return FailureCategory.PROGRESS

    def get_details(self) -> dict[str, Any]:
        return {"timeout_seconds": self.timeout_seconds}

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a failure result if elapsed time exceeds the timeout.

        Args:
            world: The CARLA world instance (unused).
            elapsed: Simulated seconds since the run began.

        Returns:
            ScenarioResult with passed=False if timed out, None otherwise.
        """
        # elapsed is a difference of running sums of the tick, so a timeout
        # that is a whole number of ticks can come out a hair short of it;
        # without the tolerance the run would time out one tick late.
        if elapsed >= self.timeout_seconds - 1e-9:
            return ScenarioResult(
                passed=False,
                message=f"Timeout after {elapsed:.2f}s (limit: {self.timeout_seconds}s)",
                elapsed_seconds=elapsed,
            )
        return None
