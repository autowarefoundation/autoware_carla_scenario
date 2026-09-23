"""Distance from an entity to a fixed place on the map."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...coordinate.poses import AnyPose
from ...coordinate.transform import to_carla_location
from ...entity_role import EntityRole
from ...kinematics import Vector3
from ..base import ScenarioResult, find_actor_in_list
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition

if TYPE_CHECKING:
    import carla


class EntityPositionDistanceCondition(CompositionCondition):
    """Pass condition on the distance from an entity to a place on the map.

    The sibling of :class:`EntityDistanceCondition`, which measures between two
    *entities*.  This one measures to somewhere that does not move: a stop
    line, a conflict point, the mouth of a junction.  Both readings are wanted
    and neither substitutes for the other, which is why this is a second
    condition rather than a target that may be either.

    Like the other distance conditions the measurement is horizontal by
    default: two points on a slope are not far apart because of the height
    between them.

    .. note::
        The *position* is resolved to a CARLA world location **in this
        constructor**, and resolving a Lanelet2 pose reads the map, so
        :class:`~autoware_carla_scenario.coordinate.map_manager.MapManager`
        must be initialised first.  Conditions are built inside a scenario's
        ``setup()``, against a live world, so that holds there; a
        :class:`CarlaWorldPose` needs no map and can be built anywhere.

        Resolving once rather than on every tick is not only cheaper: it means
        a lanelet that does not exist on the map fails while the scenario is
        being built, rather than silently never firing during the run.

    Args:
        entity_name: The ``role_name`` attribute of the actor to measure from.
        position: Where to measure to, as a Lanelet2, OpenDRIVE or CARLA world
            pose.  Unlike :class:`EntityLanePositionCondition`, which uses only
            the lane a pose names, this uses the point: ``s`` along the lanelet
            is part of the answer.
        value: Threshold distance in metres.
        rule: Comparison operator applied to ``distance`` vs *value*.
        vertical: Include the height difference in the measurement.  Off by
            default.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        position: AnyPose,
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        vertical: bool = False,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        location = to_carla_location(position)
        super().__init__(entity_name=entity_name, label=label)
        self._target = Vector3(location.x, location.y, location.z)
        self._vertical = vertical
        self._comparison = ScalarComparisonRule(
            field="distance", rule=rule, value=value, tolerance=tolerance
        )

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update(
            {
                "target_x": self._target.x,
                "target_y": self._target.y,
                "target_z": self._target.z,
                "vertical": self._vertical,
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
            }
        )
        return details

    def _measure(self, actors: "list[carla.Actor]") -> Optional[float]:
        """Return the distance from the entity to the place, or ``None``."""
        assert self._entity_name is not None  # noqa: S101
        entity = find_actor_in_list(actors, self._entity_name)
        if entity is None:
            return None

        location = entity.get_location()
        dz = (self._target.z - location.z) if self._vertical else 0.0
        offset = Vector3(
            self._target.x - location.x,
            self._target.y - location.y,
            dz,
        )
        return offset.magnitude()

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result once the distance satisfies the rule."""
        distance = self._measure(world.get_actors())
        if distance is None:
            return None
        if not self._comparison.satisfied(distance):
            return None

        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' is {distance:.2f} m from the"
                f" position, which is {self._comparison.rule.text}"
                f" {self._comparison.value:.2f} m"
            ),
            elapsed_seconds=elapsed,
        )
