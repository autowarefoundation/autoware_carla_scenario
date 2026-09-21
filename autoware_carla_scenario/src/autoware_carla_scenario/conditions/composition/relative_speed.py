"""Relative speed condition for scenario evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import AbsoluteVelocity, RelativeVelocity
from ..base import ScenarioResult, find_actor_pair
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition, entity_axes
from .speed import SpeedDirection

if TYPE_CHECKING:
    import carla


class RelativeSpeedCondition(CompositionCondition):
    """Fires when one entity's speed *relative to another* satisfies a rule.

    The measured quantity is ``v_entity - v_reference``, so a negative
    longitudinal value means the subject is slower than the reference and the
    gap between them is opening or closing accordingly.  That sign is why this
    is not the absolute :class:`SpeedCondition` with arithmetic done by the
    author: "the NPC is 5 m/s slower than the ego" is one comparison, not two.

    Longitudinal and lateral components are taken in the **reference** entity's
    frame, which is what OpenSCENARIO's ``RelativeSpeedCondition`` means: the
    question is about the subject as seen from the reference.

    :attr:`SpeedDirection.MAGNITUDE` means something different here than it
    does on :class:`SpeedCondition`, and the difference is not cosmetic.  There
    it is the length of one velocity vector.  Here it is the **difference of
    the two scalar speeds**, ``|v_subject| - |v_reference|``, which is what
    OpenSCENARIO's relative speed is when no direction is named.  The obvious
    alternative -- the length of the velocity *difference* -- answers a
    different question: two cars each doing 10 m/s towards each other have a
    speed difference of 0 and a closing speed of 20, and a condition called
    "relative speed" that reported 20 would fire for a pair that are going
    exactly as fast as each other.

    Being a difference, it is signed: negative means the subject is the slower
    of the two, the same reading the directional components have.

    Args:
        entity_name: ``role_name`` of the subject entity.
        reference_entity_name: ``role_name`` of the entity the speed is
            measured relative to.
        value: Threshold relative speed (m/s).
        rule: Comparison operator.
        direction: Which component to evaluate.  Defaults to
            :attr:`SpeedDirection.MAGNITUDE`, the difference of the two scalar
            speeds; the other two are components of the velocity difference in
            the reference's frame.  All three are signed.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Identifier reported with the result.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        reference_entity_name: Union[EntityRole, str],
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        direction: SpeedDirection = SpeedDirection.MAGNITUDE,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        super().__init__(entity_name=entity_name, label=label)
        self._reference_entity_name = reference_entity_name
        self._comparison = ScalarComparisonRule(
            field="relative_speed", rule=rule, value=value, tolerance=tolerance
        )
        self._direction = direction

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update(
            {
                "reference_entity_name": str(self._reference_entity_name),
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
                "direction": self._direction.name,
            }
        )
        return details

    def _measure(self, actors: "list[carla.Actor]") -> Optional[float]:
        """Return the relative speed component, or ``None`` when unknowable."""
        assert self._entity_name is not None  # noqa: S101
        entity, reference = find_actor_pair(
            actors, self._entity_name, self._reference_entity_name
        )
        if entity is None or reference is None:
            return None

        subject_velocity = AbsoluteVelocity.from_carla_vector3d(entity.get_velocity())
        reference_velocity = AbsoluteVelocity.from_carla_vector3d(
            reference.get_velocity()
        )

        if self._direction == SpeedDirection.MAGNITUDE:
            # The difference of the speeds, not the speed of the difference.
            return subject_velocity.speed() - reference_velocity.speed()

        relative = RelativeVelocity.between(subject_velocity, reference_velocity)

        axes = entity_axes(reference)
        if axes is None:
            return None
        forward_unit, left_unit = axes

        if self._direction == SpeedDirection.LONGITUDINAL:
            return relative.vector.dot(forward_unit)
        return relative.vector.dot(left_unit)

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result once the relative speed satisfies the rule."""
        component = self._measure(world.get_actors())
        if component is None:
            return None
        if not self._comparison.satisfied(component):
            return None

        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' speed relative to"
                f" '{self._reference_entity_name}'"
                f" {self._direction.name.lower()}"
                f" ({component:.2f} m/s)"
                f" {self._comparison.rule.text} {self._comparison.value:.2f} m/s"
            ),
            elapsed_seconds=elapsed,
        )
