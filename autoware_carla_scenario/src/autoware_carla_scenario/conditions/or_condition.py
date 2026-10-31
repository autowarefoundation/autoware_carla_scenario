"""OR composite condition that requires any child condition to pass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


class OrCondition(BaseCondition):
    """Composite condition that requires ANY child condition to be satisfied.

    Every child is checked on every tick, whatever the others return, so a
    stateful child (:class:`PersistentCondition`, :class:`StickyCondition`, a
    condition that tracks progress) sees every tick.  The results are then
    combined:

    - If any child returns a non-``None`` result, return the first such
      result, in the order the children were given.
    - If all children return ``None``, return ``None``.

    Args:
        conditions: Child conditions to evaluate.  Must contain at least 2.
    """

    # Declared for the static check (docs/typecheck.md).
    _conditions: Sequence[BaseCondition]

    def __init__(
        self, conditions: Sequence[BaseCondition], *, label: str | None = None
    ) -> None:
        if len(conditions) < 2:
            raise ValueError("OrCondition requires at least 2 conditions")
        auto_label = " OR ".join(c.label for c in conditions)
        super().__init__(label=label if label is not None else auto_label)
        self._conditions: Sequence[BaseCondition] = conditions

    def close(self) -> None:
        """Close every child (:meth:`BaseCondition.close`)."""
        for condition in self._conditions:
            condition.close()

    def reset(self) -> None:
        """Reset every child (:meth:`BaseCondition.reset`)."""
        super().reset()
        for condition in self._conditions:
            condition.reset()

    def get_details(self) -> dict[str, Any]:
        return {
            "operator": "OR",
            "children": [c.to_summary_dict() for c in self._conditions],
        }

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Check child conditions with OR logic.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            The first non-``None`` :class:`ScenarioResult` from any child,
            or ``None`` if all children are still pending.
        """
        # Check every child before choosing: stopping at the first result would
        # leave the later children unchecked this tick (#663).
        first: Optional[ScenarioResult] = None
        for condition in self._conditions:
            result = condition.check(world, elapsed)
            if first is None and result is not None:
                first = result
        return first
