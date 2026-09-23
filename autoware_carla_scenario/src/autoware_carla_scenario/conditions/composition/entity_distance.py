"""Relative-distance condition between two entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import Vector3
from ...coordinate.lane_distance import lane_separation
from ...coordinate.poses import CarlaWorldPose
from ..base import ScenarioResult, find_actor_pair
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition, DistanceCoordinateSystem

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
            Meaningful only in the entity frame.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        coordinate_system: :attr:`~DistanceCoordinateSystem.ENTITY` (default)
            measures the straight line between the two.
            :attr:`~DistanceCoordinateSystem.LANE` measures along the road they
            share, which on a curve is the longer and more useful number, and
            which has no answer at all when they are on different roads.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative, or if a lane-frame distance is
            asked to be vertical.  A distance along the road is a length on a
            one-dimensional line, so there is no ``z`` to include; saying both
            is a mistake, and resolving it silently would hide which of the two
            the author meant.
    """

    def __init__(
        self,
        source: Union[EntityRole, str],
        target: Union[EntityRole, str],
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        vertical: bool = False,
        tolerance: float = 1e-6,
        coordinate_system: DistanceCoordinateSystem = (DistanceCoordinateSystem.ENTITY),
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if vertical and coordinate_system is DistanceCoordinateSystem.LANE:
            raise ValueError(
                "a lane-frame distance is measured along the road and has no "
                "vertical component; drop vertical=True, or measure in the "
                "entity frame"
            )
        super().__init__(entity_name=source, label=label)
        self._target = target
        self._vertical = vertical
        self._coordinate_system = coordinate_system
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
                "coordinate_system": self._coordinate_system.name,
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

        src_loc = source.get_location()
        tgt_loc = target.get_location()

        if self._coordinate_system is DistanceCoordinateSystem.LANE:
            # Unsigned, so neither entity has to be moving for this to mean
            # something -- unlike a headway, a separation has no direction.
            return lane_separation(
                CarlaWorldPose(x=src_loc.x, y=src_loc.y, z=src_loc.z, yaw=0.0),
                CarlaWorldPose(x=tgt_loc.x, y=tgt_loc.y, z=tgt_loc.z, yaw=0.0),
            )

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

        rule_text = self._comparison.rule.text
        return ScenarioResult(
            passed=True,
            message=(
                f"Distance '{self._entity_name}' -> '{self._target}'"
                f" ({distance:.2f} m) {rule_text}"
                f" {self._comparison.value:.2f} m at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
