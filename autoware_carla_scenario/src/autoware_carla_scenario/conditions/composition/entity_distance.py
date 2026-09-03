"""Relative-distance condition between two entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import Vector3
from ..base import ScenarioResult, find_actor_in_list
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition

if TYPE_CHECKING:
    import carla


class EntityDistanceCondition(CompositionCondition):
    """Pass condition on the distance from a *source* entity to a *target* entity.

    The measured value is the Euclidean distance between the two actors'
    world positions.  ``vertical=False`` (the default) ignores the ``z``
    component, which is what scenario authors mean by "how far apart are
    these two cars".

    This is the relational counterpart of
    :class:`~autoware_carla_scenario.conditions.composition.speed.SpeedCondition`:
    it reads as ``source -> target | Distance | <rule> <value> m``.

    Args:
        source: ``role_name`` of the entity the distance is measured *from*.
        target: ``role_name`` of the entity the distance is measured *to*.
        value: Threshold distance in metres.
        rule: Comparison operator applied to ``distance`` vs *value*.
        vertical: Include the ``z`` component in the distance when ``True``.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative.
    """

    def __init__(
        self,
        source: Union[EntityRole, str],
        target: Union[EntityRole, str],
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        vertical: bool = False,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        super().__init__(entity_name=source, label=label)
        self._target = target
        self._vertical = vertical
        self._comparison = ScalarComparisonRule(
            field="distance", rule=rule, value=value, tolerance=tolerance
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update(
            {
                "source": str(self._entity_name),
                "target": str(self._target),
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
                "vertical": self._vertical,
            }
        )
        return details

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _measure(self, actors: "list[carla.Actor]") -> Optional[float]:
        """Return the source-to-target distance, or ``None`` if unavailable."""
        assert self._entity_name is not None  # noqa: S101
        source = find_actor_in_list(actors, self._entity_name)
        target = find_actor_in_list(actors, self._target)
        if source is None or target is None:
            return None

        src_loc = source.get_location()
        tgt_loc = target.get_location()
        delta = Vector3(
            tgt_loc.x - src_loc.x,
            tgt_loc.y - src_loc.y,
            (tgt_loc.z - src_loc.z) if self._vertical else 0.0,
        )
        return delta.magnitude()

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result when the distance satisfies the comparison rule."""
        actors: list[carla.Actor] = world.get_actors()
        distance = self._measure(actors)
        if distance is None:
            return None

        if not self._comparison.satisfied(distance):
            return None

        rule_text = self._comparison.rule.name.lower().replace("_", " ")
        return ScenarioResult(
            passed=True,
            message=(
                f"Distance '{self._entity_name}' -> '{self._target}'"
                f" ({distance:.2f} m) {rule_text}"
                f" {self._comparison.value:.2f} m at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
