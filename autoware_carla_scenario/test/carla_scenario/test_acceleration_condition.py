"""Acceleration condition: the component, the sign, and the degenerate cases.

CARLA reports acceleration directly, so the world is faked and what is left to
test is the decomposition -- which is where the sign convention lives, and the
sign is the whole point of a harsh-braking assertion.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    AccelerationCondition,
    AccelerationDirection,
    ComparisonRule,
)


def _actor(
    role: str,
    acceleration: tuple[float, float, float],
    heading: tuple[float, float] = (1.0, 0.0),
) -> MagicMock:
    """Return a fake actor accelerating at *acceleration*, facing *heading*."""
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_acceleration.return_value = carla.Vector3D(*acceleration)
    transform = MagicMock()
    transform.get_forward_vector.return_value = carla.Vector3D(*heading, 0.0)
    actor.get_transform.return_value = transform
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


class TestLongitudinalComponent:
    def test_braking_is_negative_along_the_entity_heading(self) -> None:
        """A car facing +x and decelerating reports a negative longitudinal value."""
        condition = AccelerationCondition(
            entity_name="Ego",
            value=-3.0,
            rule=ComparisonRule.LESS_THAN,
            direction=AccelerationDirection.LONGITUDINAL,
            label="harsh_braking",
        )
        world = _world(_actor("Ego", (-4.0, 0.0, 0.0), heading=(1.0, 0.0)))
        result = condition.check(world, 1.0)
        assert result is not None
        assert result.passed
        assert "-4.00 m/s^2" in result.message

    def test_the_heading_decides_the_sign_not_the_world_axis(self) -> None:
        """The same world-frame vector is acceleration for a car facing the other way."""
        braking = AccelerationCondition(
            entity_name="Ego",
            value=-3.0,
            rule=ComparisonRule.LESS_THAN,
            direction=AccelerationDirection.LONGITUDINAL,
            label="braking",
        )
        # Facing -x while the acceleration vector points -x: speeding up.
        world = _world(_actor("Ego", (-4.0, 0.0, 0.0), heading=(-1.0, 0.0)))
        assert braking.check(world, 1.0) is None

    def test_gentle_braking_does_not_fire_a_harsh_threshold(self) -> None:
        condition = AccelerationCondition(
            entity_name="Ego",
            value=-3.0,
            rule=ComparisonRule.LESS_THAN,
            direction=AccelerationDirection.LONGITUDINAL,
            label="harsh_braking",
        )
        world = _world(_actor("Ego", (-1.0, 0.0, 0.0)))
        assert condition.check(world, 1.0) is None


class TestOtherComponents:
    def test_lateral_is_positive_to_the_entity_left(self) -> None:
        condition = AccelerationCondition(
            entity_name="Ego",
            value=1.0,
            rule=ComparisonRule.GREATER_THAN,
            direction=AccelerationDirection.LATERAL,
            label="lateral",
        )
        # Facing +x, CARLA's left-handed frame puts left at -y.
        world = _world(_actor("Ego", (0.0, -2.0, 0.0), heading=(1.0, 0.0)))
        result = condition.check(world, 0.0)
        assert result is not None
        assert "2.00 m/s^2" in result.message

    def test_magnitude_ignores_direction_and_is_never_negative(self) -> None:
        condition = AccelerationCondition(
            entity_name="Ego",
            value=3.0,
            rule=ComparisonRule.GREATER_THAN,
            direction=AccelerationDirection.MAGNITUDE,
            label="magnitude",
        )
        world = _world(_actor("Ego", (-3.0, 4.0, 0.0)))
        result = condition.check(world, 0.0)
        assert result is not None
        assert "5.00 m/s^2" in result.message

    def test_vertical_acceleration_stays_out_of_the_horizontal_components(self) -> None:
        """Gravity on a slope must not read as braking."""
        condition = AccelerationCondition(
            entity_name="Ego",
            value=-0.5,
            rule=ComparisonRule.LESS_THAN,
            direction=AccelerationDirection.LONGITUDINAL,
            label="longitudinal",
        )
        world = _world(_actor("Ego", (0.0, 0.0, -9.81), heading=(1.0, 0.0)))
        assert condition.check(world, 0.0) is None


class TestGuards:
    def test_an_absent_entity_yields_no_verdict(self) -> None:
        condition = AccelerationCondition(
            entity_name="Ego", value=0.0, rule=ComparisonRule.GREATER_THAN, label="a"
        )
        assert condition.check(_world(), 0.0) is None

    def test_a_degenerate_heading_is_unknown_rather_than_zero(self) -> None:
        """Zero is a value; "cannot tell" is not, and a rule must not read it as one."""
        condition = AccelerationCondition(
            entity_name="Ego",
            value=1.0,
            rule=ComparisonRule.LESS_THAN,
            direction=AccelerationDirection.LONGITUDINAL,
            label="longitudinal",
        )
        world = _world(_actor("Ego", (5.0, 0.0, 0.0), heading=(0.0, 0.0)))
        assert condition.check(world, 0.0) is None

    def test_a_degenerate_heading_still_allows_magnitude(self) -> None:
        condition = AccelerationCondition(
            entity_name="Ego",
            value=1.0,
            rule=ComparisonRule.GREATER_THAN,
            direction=AccelerationDirection.MAGNITUDE,
            label="magnitude",
        )
        world = _world(_actor("Ego", (5.0, 0.0, 0.0), heading=(0.0, 0.0)))
        assert condition.check(world, 0.0) is not None

    def test_a_negative_tolerance_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            AccelerationCondition(
                entity_name="Ego", value=0.0, tolerance=-1.0, label="a"
            )


class TestDetails:
    def test_details_report_the_rule_the_condition_was_built_with(self) -> None:
        condition = AccelerationCondition(
            entity_name="Ego",
            value=-3.0,
            rule=ComparisonRule.LESS_THAN,
            direction=AccelerationDirection.LONGITUDINAL,
            label="harsh_braking",
        )
        details = condition.get_details()
        assert details["entity_name"] == "Ego"
        assert details["rule"] == "LESS_THAN"
        assert details["direction"] == "LONGITUDINAL"
        assert math.isclose(details["value"], -3.0)
