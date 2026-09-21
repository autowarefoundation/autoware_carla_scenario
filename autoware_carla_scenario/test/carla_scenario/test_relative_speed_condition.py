"""Relative speed: the difference, the frame it is taken in, and the sign.

The quantity is ``v_subject - v_reference``, decomposed in the *reference*
entity's frame.  Both halves of that sentence are easy to get backwards, and
either mistake produces a condition that fires on the wrong vehicle, so they
are what these tests pin.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    ComparisonRule,
    RelativeSpeedCondition,
    SpeedDirection,
)


def _actor(
    role: str,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    heading: tuple[float, float] = (1.0, 0.0),
) -> MagicMock:
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_velocity.return_value = carla.Vector3D(*velocity)
    transform = MagicMock()
    transform.get_forward_vector.return_value = carla.Vector3D(*heading, 0.0)
    actor.get_transform.return_value = transform
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


class TestLongitudinal:
    def test_a_slower_subject_reads_negative(self) -> None:
        condition = RelativeSpeedCondition(
            entity_name="npc1",
            reference_entity_name="Ego",
            value=-3.0,
            rule=ComparisonRule.LESS_THAN,
            direction=SpeedDirection.LONGITUDINAL,
            label="npc_slower",
        )
        world = _world(
            _actor("npc1", (5.0, 0.0, 0.0)),
            _actor("Ego", (10.0, 0.0, 0.0)),
        )
        result = condition.check(world, 1.0)
        assert result is not None
        assert result.passed
        assert "-5.00 m/s" in result.message

    def test_a_faster_subject_does_not_fire_a_slower_than_rule(self) -> None:
        condition = RelativeSpeedCondition(
            entity_name="npc1",
            reference_entity_name="Ego",
            value=-3.0,
            rule=ComparisonRule.LESS_THAN,
            direction=SpeedDirection.LONGITUDINAL,
            label="npc_slower",
        )
        world = _world(
            _actor("npc1", (15.0, 0.0, 0.0)),
            _actor("Ego", (10.0, 0.0, 0.0)),
        )
        assert condition.check(world, 1.0) is None

    def test_the_reference_heading_defines_longitudinal_not_the_subject(self) -> None:
        """Two cars facing opposite ways: the answer must follow the reference."""
        condition = RelativeSpeedCondition(
            entity_name="npc1",
            reference_entity_name="Ego",
            value=0.0,
            rule=ComparisonRule.GREATER_THAN,
            direction=SpeedDirection.LONGITUDINAL,
            label="closing",
        )
        # Subject drives -x, reference faces -x while standing still.
        # Along the reference's own forward direction the subject is faster.
        world = _world(
            _actor("npc1", (-8.0, 0.0, 0.0), heading=(1.0, 0.0)),
            _actor("Ego", (0.0, 0.0, 0.0), heading=(-1.0, 0.0)),
        )
        result = condition.check(world, 0.0)
        assert result is not None
        assert "8.00 m/s" in result.message


class TestOtherComponents:
    def test_lateral_is_positive_to_the_reference_left(self) -> None:
        condition = RelativeSpeedCondition(
            entity_name="npc1",
            reference_entity_name="Ego",
            value=1.0,
            rule=ComparisonRule.GREATER_THAN,
            direction=SpeedDirection.LATERAL,
            label="lateral",
        )
        # Reference faces +x; CARLA's left-handed frame puts its left at -y.
        world = _world(
            _actor("npc1", (0.0, -3.0, 0.0)),
            _actor("Ego", (0.0, 0.0, 0.0), heading=(1.0, 0.0)),
        )
        result = condition.check(world, 0.0)
        assert result is not None
        assert "3.00 m/s" in result.message

    def test_magnitude_is_never_negative(self) -> None:
        condition = RelativeSpeedCondition(
            entity_name="npc1",
            reference_entity_name="Ego",
            value=4.0,
            rule=ComparisonRule.GREATER_THAN,
            direction=SpeedDirection.MAGNITUDE,
            label="magnitude",
        )
        world = _world(
            _actor("npc1", (0.0, 0.0, 0.0)),
            _actor("Ego", (3.0, 4.0, 0.0)),
        )
        result = condition.check(world, 0.0)
        assert result is not None
        assert "5.00 m/s" in result.message


class TestGuards:
    def test_a_missing_reference_yields_no_verdict(self) -> None:
        condition = RelativeSpeedCondition(
            entity_name="npc1",
            reference_entity_name="Ego",
            value=0.0,
            rule=ComparisonRule.GREATER_THAN,
            label="a",
        )
        assert condition.check(_world(_actor("npc1")), 0.0) is None

    def test_a_degenerate_reference_heading_is_unknown(self) -> None:
        condition = RelativeSpeedCondition(
            entity_name="npc1",
            reference_entity_name="Ego",
            value=-1.0,
            rule=ComparisonRule.LESS_THAN,
            direction=SpeedDirection.LONGITUDINAL,
            label="a",
        )
        world = _world(
            _actor("npc1", (0.0, 0.0, 0.0)),
            _actor("Ego", (10.0, 0.0, 0.0), heading=(0.0, 0.0)),
        )
        assert condition.check(world, 0.0) is None

    def test_a_negative_tolerance_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            RelativeSpeedCondition(
                entity_name="npc1",
                reference_entity_name="Ego",
                value=0.0,
                tolerance=-1.0,
                label="a",
            )


class TestDetails:
    def test_details_name_both_entities(self) -> None:
        condition = RelativeSpeedCondition(
            entity_name="npc1",
            reference_entity_name="Ego",
            value=-5.0,
            rule=ComparisonRule.LESS_THAN,
            direction=SpeedDirection.LONGITUDINAL,
            label="npc_slower",
        )
        details = condition.get_details()
        assert details["entity_name"] == "npc1"
        assert details["reference_entity_name"] == "Ego"
        assert details["rule"] == "LESS_THAN"
        assert details["direction"] == "LONGITUDINAL"
        assert math.isclose(details["value"], -5.0)
