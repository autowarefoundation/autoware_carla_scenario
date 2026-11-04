"""Condition on how many road users an :class:`~.entity_filter.EntityFilter` takes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseCondition, ScenarioResult
from .comparison import ComparisonRule, compare
from .entity_filter import EntityFilter

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["EntityCountCondition"]


class EntityCountCondition(BaseCondition):
    """Holds while the number of road users *entities* takes compares to
    *count* by *rule*: at least one by default -- "a vehicle within 30 m
    ahead of the ego", "no pedestrian on the crosswalk" (``EQUAL_TO`` 0).

    Args:
        entities: Which road users count (:class:`~.entity_filter.EntityFilter`).
        rule: How their number compares to *count*.
        count: The number compared with: a whole number (``2``, or ``2.0``
            read as 2), not negative.
        label: Human-readable identifier.

    Raises:
        ValueError: *count* a boolean, a fraction, NaN, infinite or below
            zero, or bounds of *entities* that can never hold.
    """

    # Declared for the static check (docs/typecheck.md).
    _entities: EntityFilter
    _rule: ComparisonRule
    _count: int
    _matched: list[str]

    def __init__(
        self,
        entities: EntityFilter,
        rule: ComparisonRule = ComparisonRule.GREATER_THAN_OR_EQUAL,
        count: int = 1,
        *,
        label: str,
    ) -> None:
        super().__init__(label=label)
        # int() alone would compare against 0.5 as 0 (">= 0.5" always holds)
        # and against True as 1, and fail on NaN with its own message.
        if isinstance(count, (bool, str)):
            raise ValueError(f"count must be a whole number, got {count!r}")
        number = float(count)
        if not math.isfinite(number):
            raise ValueError(f"count must be finite, got {count!r}")
        if math.floor(number) != number:
            raise ValueError(f"count must be a whole number, got {count!r}")
        if number < 0.0:
            raise ValueError(f"count must not be negative, got {count!r}")
        entities.validate()
        self._entities = entities
        self._rule = rule
        self._count = int(count)
        self._matched = []

    @property
    def entities(self) -> EntityFilter:
        """Which road users count."""
        return self._entities

    @property
    def matched(self) -> list[str]:
        """The names the last check took."""
        return list(self._matched)

    def get_details(self) -> dict[str, Any]:
        return {
            "entities": self._entities.describe(),
            "rule": self._rule.text,
            "count": self._count,
            "matched": list(self._matched),
        }

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Hold while the number taken compares to the count."""
        names: list[str] = []
        for entity in self._entities.select(world):
            names.append(entity.name)
        self._matched = names
        if not compare(float(len(names)), self._rule, float(self._count)):
            return None
        listed = ", ".join(names) if names else "none"
        return ScenarioResult(
            passed=True,
            message=(
                f"{len(names)} of the {self._entities.describe()} "
                f"({listed}), {self._rule.text} {self._count}"
            ),
            elapsed_seconds=elapsed,
        )
