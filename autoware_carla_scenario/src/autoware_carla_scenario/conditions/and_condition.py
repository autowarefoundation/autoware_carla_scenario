"""AND composite condition that requires all child conditions to pass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


class AndCondition(BaseCondition):
    """Composite condition that requires ALL child conditions to be satisfied.

    Every child is checked on every tick, whatever the others return, so a
    stateful child (:class:`PersistentCondition`, :class:`StickyCondition`, a
    condition that tracks progress) sees every tick and the verdict does not
    depend on the order of the children.  The results are then combined:

    - If any child returns ``passed=False``, return the first such failure.
    - Otherwise, if any child returns ``None`` (not yet determined), return
      ``None``.
    - If all children return ``passed=True``, return a combined pass result.

    Args:
        conditions: Child conditions to evaluate.  Must contain at least 2.
    """

    # Declared for the static check (docs/typecheck.md).
    _conditions: Sequence[BaseCondition]

    def __init__(
        self, conditions: Sequence[BaseCondition], *, label: str | None = None
    ) -> None:
        if len(conditions) < 2:
            raise ValueError("AndCondition requires at least 2 conditions")
        auto_label = " AND ".join(c.label for c in conditions)
        super().__init__(label=label if label is not None else auto_label)
        self._conditions: Sequence[BaseCondition] = conditions

    def close(self) -> None:
        """Close every child (:meth:`BaseCondition.close`)."""
        for condition in self._conditions:
            condition.close()

    def get_details(self) -> dict[str, Any]:
        return {
            "operator": "AND",
            "children": [c.to_summary_dict() for c in self._conditions],
        }

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Check all child conditions with AND logic.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            :class:`ScenarioResult` with ``passed=False`` if any child fails,
            ``passed=True`` if all children pass, or ``None`` if any child
            is still pending.
        """
        # Check every child before combining: stopping at the first pending
        # child would leave the later ones unchecked this tick (#663).
        messages: List[str] = []
        failure: Optional[ScenarioResult] = None
        pending = False
        for condition in self._conditions:
            result = condition.check(world, elapsed)
            if result is None:
                pending = True
            elif not result.passed:
                if failure is None:
                    failure = result
            else:
                messages.append(result.message)
        if failure is not None:
            return failure
        if pending:
            return None
        return ScenarioResult(
            passed=True,
            message=" AND ".join(messages),
            elapsed_seconds=elapsed,
        )
