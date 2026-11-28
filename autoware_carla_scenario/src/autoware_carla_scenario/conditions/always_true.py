"""Conditions that always, or never, hold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


class AlwaysTrueCondition(BaseCondition):
    """A condition that returns a successful result on every check.

    Useful as a trigger for actions that should fire unconditionally
    (subject to the action's own ``once`` flag).
    """

    def __init__(self, *, label: str = "always_true") -> None:
        super().__init__(label=label)

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Always returns a passing :class:`ScenarioResult`."""
        return ScenarioResult(
            passed=True,
            message="AlwaysTrueCondition",
            elapsed_seconds=elapsed,
        )


class NeverCondition(BaseCondition):
    """A condition that never holds: the *until* of an action that runs for
    the rest of the run (a traffic source or sink, a speed that follows
    another entity's), where an action with none is over on the tick after it
    fired."""

    def __init__(self, *, label: str = "until_the_run_ends") -> None:
        super().__init__(label=label)

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Never a result."""
        # Through a typed local: Codon 0.19 cannot type a bare `return None`
        # that is a function's only return.
        never: Optional[ScenarioResult] = None
        return never
