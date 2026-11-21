"""Acceleration condition for scenario evaluation."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import AbsoluteAcceleration, Vector3
from ...kinematics.walker_motion import WalkerVelocities, is_walker
from ..base import ScenarioResult, find_actor_by_role_name
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition, entity_axes

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


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

    CARLA reports a vehicle's acceleration directly
    (``Actor.get_acceleration()``): the value is read, decomposed and
    compared.  A walker's it does not -- CARLA's walker controller places its
    walker rather than driving it, so it reads 0 -- and a walker's
    acceleration is differentiated instead from its velocity as the speed
    conditions measure it
    (:class:`~autoware_carla_scenario.kinematics.walker_motion.WalkerVelocities`):
    the change since the last check over the simulated time between the two.
    It is known from the second velocity read on, so a walker's acceleration
    is not compared on the first check or two that see it.

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

    # Declared for the static check (docs/typecheck.md).
    _comparison: ScalarComparisonRule
    _direction: AccelerationDirection
    _velocities: WalkerVelocities
    #: Each walker's last velocity and the acceleration read with it, by
    #: actor id: ``(elapsed, vx, vy, vz, ax, ay, az, known)``, *known* whether
    #: an acceleration was measured (else there was nothing to measure from).
    _walker_reads: dict[
        int, tuple[float, float, float, float, float, float, float, bool]
    ]

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
        super().__init__(entity_name=entity_name, label=label)
        self._comparison = ScalarComparisonRule(
            field="acceleration", rule=rule, value=value, tolerance=tolerance
        )
        self._direction = direction
        # A walker's velocity, measured from where it stood when last checked,
        # and the acceleration differentiated from it (#2271).
        self._velocities = WalkerVelocities()
        self._walker_reads = {}

    def close(self) -> None:
        """Forget the walkers measured (the run is over), then close the
        guards (:meth:`CompositionCondition.close`): a condition registered
        once is checked again next run."""
        self._velocities.clear()
        self._walker_reads.clear()
        super().close()

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

    def _walker_acceleration(
        self, walker: "carla.Actor", elapsed: float
    ) -> Optional[Vector3]:
        """*walker*'s acceleration: the change of its measured velocity since
        the last check over the time between the two, or ``None`` until two
        velocities have been read (and again after the walker was teleported,
        or the clock went back)."""
        key = int(walker.id)
        read = self._velocities.velocity(walker, elapsed)
        if read is None:
            # Not measured yet, or teleported: nothing to differentiate from.
            self._walker_reads.pop(key, None)
            return None
        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        velocity: Vector3 = read
        acceleration = Vector3(0.0, 0.0, 0.0)
        known = False
        if key in self._walker_reads:
            when, vx, vy, vz, ax, ay, az, was = self._walker_reads[key]
            span = elapsed - when
            if span == 0.0:
                # Checked again at the same time.
                if was:
                    return Vector3(ax, ay, az)
                return None
            if span > 0.0:
                acceleration = Vector3(
                    (velocity.x - vx) / span,
                    (velocity.y - vy) / span,
                    (velocity.z - vz) / span,
                )
                known = True
        self._walker_reads[key] = (
            elapsed,
            velocity.x,
            velocity.y,
            velocity.z,
            acceleration.x,
            acceleration.y,
            acceleration.z,
            known,
        )
        if known:
            return acceleration
        return None

    def _extract_component(
        self, entity: "carla.Actor", elapsed: float = 0.0
    ) -> Optional[float]:
        """Return the component of *entity*'s acceleration this rule compares.

        Args:
            entity: The actor whose acceleration is read.
            elapsed: The scenario's elapsed time, to differentiate a walker's
                velocity by.

        Returns:
            The value in m/s^2, or ``None`` when the entity's heading is
            degenerate and a directional component therefore has no meaning,
            or the entity is a walker whose acceleration is not known yet.
        """
        vector = Vector3(0.0, 0.0, 0.0)
        if is_walker(entity):
            measured = self._walker_acceleration(entity, elapsed)
            if measured is None:
                return None
            vector = measured
        else:
            vector = AbsoluteAcceleration.from_carla_vector3d(
                entity.get_acceleration()
            ).vector

        if self._direction == AccelerationDirection.MAGNITUDE:
            return vector.magnitude()

        axes = entity_axes(entity)
        if axes is None:
            return None
        forward_unit, left_unit = axes

        if self._direction == AccelerationDirection.LONGITUDINAL:
            return vector.dot(forward_unit)
        return vector.dot(left_unit)

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result once the acceleration satisfies the rule."""
        assert self._entity_name is not None
        entity = find_actor_by_role_name(world, self._entity_name)
        if entity is None:
            return None

        component = self._extract_component(entity, elapsed)
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
