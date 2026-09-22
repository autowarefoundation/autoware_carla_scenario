"""Timeout-based scenario fail condition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import carla


class TimeoutCondition(BaseCondition):
    """Fail condition that triggers when elapsed time exceeds the timeout.

    Measured on the simulated clock, like every other condition: *timeout* here
    is part of what the scenario says -- "this manoeuvre should be over within
    N seconds" -- and so has to mean the same thing on every host.

    It is therefore not the runner's watchdog.  Stopping a run that is making
    no useful progress is a question about the machine rather than about the
    scenario, and ``ScenarioRunner`` keeps a wall-clock guard of its own for
    it (see ``timeout_seconds`` there).
    """

    def __init__(self, timeout_seconds: float = 60.0, *, label: str) -> None:
        """Initialize the timeout condition.

        Args:
            timeout_seconds: Number of seconds before failing. Defaults to 60.0.
            label: Human-readable label identifying this condition.
        """
        super().__init__(label=label)
        self.timeout_seconds = timeout_seconds

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
        if elapsed >= self.timeout_seconds:
            return ScenarioResult(
                passed=False,
                message=f"Timeout after {elapsed:.2f}s (limit: {self.timeout_seconds}s)",
                elapsed_seconds=elapsed,
            )
        return None
