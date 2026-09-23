"""Relative-distance condition between two entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ..base import ScenarioResult, find_actor_pair
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition
from .distance_measure import RelativeDistanceType, separation

if TYPE_CHECKING:
    import carla


class EntityDistanceCondition(CompositionCondition):
    """Pass condition on the distance from a *source* entity to a *target* entity.

    By default the measured value is the Euclidean distance between the two
    actors' centres, ignoring the ``z`` component -- which is what a scenario
    author means by "how far apart are these two cars".

    Two things change what is being asked, and both change the answer at the
    ranges scenarios care about:

    * *distance_type* picks the axis.  A car in the next lane is 20 m away in
      a straight line and 2 m away longitudinally, and a following-distance or
      cut-in scenario means the second.
    * *freespace* measures between bounding boxes rather than between centres.
      The difference is about a vehicle length, which at a 5 m threshold is
      most of the threshold.

    This is the relational counterpart of
    :class:`~autoware_carla_scenario.conditions.composition.speed.SpeedCondition`:
    it reads as ``source -> target | Distance | <rule> <value> m``.

    Args:
        source: ``role_name`` of the entity the distance is measured *from*.
        target: ``role_name`` of the entity the distance is measured *to*.
        value: Threshold distance in metres.
        rule: Comparison operator applied to ``distance`` vs *value*.
        vertical: Include the ``z`` component in the distance when ``True``.
            Only meaningful for :attr:`RelativeDistanceType.EUCLIDEAN`.
        distance_type: Which component of the separation to measure.  Defaults
            to :attr:`RelativeDistanceType.EUCLIDEAN`, the previous behaviour.
        freespace: Measure between bounding boxes rather than centres, clamped
            at zero once they overlap.  Off by default.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative, or if *vertical* is combined
            with a directional *distance_type*.  Longitudinal and lateral are
            components of the ground plane, so asking for height as well is a
            contradiction rather than a refinement -- and silently dropping one
            of the two would leave the document saying something the run does
            not do.
    """

    def __init__(
        self,
        source: Union[EntityRole, str],
        target: Union[EntityRole, str],
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        vertical: bool = False,
        distance_type: RelativeDistanceType = RelativeDistanceType.EUCLIDEAN,
        freespace: bool = False,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if vertical and distance_type is not RelativeDistanceType.EUCLIDEAN:
            raise ValueError(
                "vertical applies to a euclidean distance; "
                f"{distance_type.value} is a ground-plane component"
            )
        super().__init__(entity_name=source, label=label)
        self._target = target
        self._vertical = vertical
        self._distance_type = distance_type
        self._freespace = freespace
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
                "distance_type": self._distance_type.name,
                "freespace": self._freespace,
            }
        )
        return details

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _measure(self, actors: "list[carla.Actor]") -> Optional[float]:
        """Return the source-to-target distance, or ``None`` if unavailable."""
        assert self._entity_name is not None  # noqa: S101
        source, target = find_actor_pair(actors, self._entity_name, self._target)
        if source is None or target is None:
            return None

        return separation(
            source,
            target,
            distance_type=self._distance_type,
            freespace=self._freespace,
            vertical=self._vertical,
        )

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result when the distance satisfies the comparison rule."""
        actors: list[carla.Actor] = world.get_actors()
        distance = self._measure(actors)
        if distance is None:
            return None

        if not self._comparison.satisfied(distance):
            return None

        rule_text = self._comparison.rule.text
        return ScenarioResult(
            passed=True,
            message=(
                f"Distance '{self._entity_name}' -> '{self._target}'"
                f" [{self._distance_type.value}]"
                f" ({distance:.2f} m) {rule_text}"
                f" {self._comparison.value:.2f} m at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
