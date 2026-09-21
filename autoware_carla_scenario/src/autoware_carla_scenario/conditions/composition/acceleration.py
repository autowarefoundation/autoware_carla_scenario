"""Acceleration condition for scenario evaluation."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import AbsoluteAcceleration
from ..base import ScenarioResult, find_actor_in_list
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition, entity_axes

if TYPE_CHECKING:
    import carla


class AccelerationDirection(Enum):
    """Which acceleration component to evaluate.

    Mirrors :class:`~autoware_carla_scenario.conditions.SpeedDirection` member
    for member, and is deliberately a separate enum rather than a shared one:
    the speed enum is public API under a name that says "speed", and renaming
    it to something both could wear belongs to its own change.

    Attributes:
        LONGITUDINAL: Signed component along the entity's forward direction.
            Positive is acceleration, negative is braking -- which is the
            component a comfort or harsh-braking assertion is about, and the
            reason :attr:`MAGNITUDE` cannot be the only option.
        LATERAL: Signed component to the entity's left.
        MAGNITUDE: Scalar magnitude of the acceleration vector, always
            non-negative.
    """

    LONGITUDINAL = auto()
    LATERAL = auto()
    MAGNITUDE = auto()


class AccelerationCondition(CompositionCondition):
    """Fires when an entity's acceleration satisfies a comparison.

    CARLA reports acceleration directly (``Actor.get_acceleration()``), so
    nothing here differentiates a velocity: the value is read, decomposed and
    compared.

    Longitudinal and lateral components are taken in the **entity's own**
    frame, which is what makes a signed threshold mean "braking" rather than
    "accelerating west".

    Args:
        entity_name: The ``role_name`` attribute of the actor to evaluate.
        value: Threshold acceleration (m/s^2) to compare against.
        rule: Comparison operator.
        direction: Which acceleration component to evaluate.  Defaults to
            :attr:`AccelerationDirection.MAGNITUDE`.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
            Defaults to ``1e-6``.
        label: Identifier reported with the result.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        direction: AccelerationDirection = AccelerationDirection.MAGNITUDE,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        super().__init__(entity_name=entity_name, label=label)
        self._comparison = ScalarComparisonRule(
            field="acceleration", rule=rule, value=value, tolerance=tolerance
        )
        self._direction = direction

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update(
            {
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
                "direction": self._direction.name,
            }
        )
        return details

    def _extract_component(self, entity: "carla.Actor") -> Optional[float]:
        """Return the component of *entity*'s acceleration this rule compares.

        Returns:
            The value in m/s^2, or ``None`` when the entity's heading is
            degenerate and a directional component therefore has no meaning.
        """
        acceleration = AbsoluteAcceleration.from_carla_vector3d(
            entity.get_acceleration()
        )

        if self._direction == AccelerationDirection.MAGNITUDE:
            return acceleration.magnitude()

        axes = entity_axes(entity)
        if axes is None:
            return None
        forward_unit, left_unit = axes

        if self._direction == AccelerationDirection.LONGITUDINAL:
            return acceleration.vector.dot(forward_unit)
        return acceleration.vector.dot(left_unit)

    def _check(
        self, world: "carla.World", elapsed: float
    ) -> Optional[ScenarioResult]:
        """Return a pass result once the acceleration satisfies the rule."""
        assert self._entity_name is not None
        entity = find_actor_in_list(world.get_actors(), self._entity_name)
        if entity is None:
            return None

        component = self._extract_component(entity)
        if component is None:
            return None

        if not self._comparison.satisfied(component):
            return None

        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' acceleration"
                f" {self._direction.name.lower()}"
                f" ({component:.2f} m/s^2)"
                f" {self._comparison.rule.text} {self._comparison.value:.2f} m/s^2"
            ),
            elapsed_seconds=elapsed,
        )
