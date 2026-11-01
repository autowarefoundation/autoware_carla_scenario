"""Condition on how many road users an :class:`~.entity_filter.EntityFilter` takes."""

from __future__ import annotations

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
        count: The number compared with.
        label: Human-readable identifier.

    Raises:
        ValueError: *count* below zero, or bounds of *entities* that can
            never hold.
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
        if count < 0:
            raise ValueError("count must not be negative")
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
