"""Distance from an entity to a fixed place on the map.

A `CarlaWorldPose` is used throughout so the condition can be built without a
live map; the Lanelet2 path it also accepts is the same call
(`to_carla_location`) that `EntityLanePositionCondition` already relies on.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    CarlaWorldPose,
    ComparisonRule,
    EntityPositionDistanceCondition,
)


def _actor(role: str, position: tuple[float, float, float]) -> MagicMock:
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(*position)
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


_STOP_LINE = CarlaWorldPose(x=100.0, y=0.0, z=0.0)


class TestMeasurement:
    def test_fires_once_the_entity_is_inside_the_threshold(self) -> None:
        condition = EntityPositionDistanceCondition(
            entity_name="Ego",
            position=_STOP_LINE,
            value=10.0,
            rule=ComparisonRule.LESS_THAN,
            label="near_stop_line",
        )
        assert condition.check(_world(_actor("Ego", (85, 0, 0))), 1.0) is None

        result = condition.check(_world(_actor("Ego", (95, 0, 0))), 2.0)
        assert result is not None
        assert result.passed
        assert "5.00 m" in result.message

    def test_the_place_does_not_move_with_the_entity(self) -> None:
        """Measuring to a fixed point, not to another actor: approach shortens it."""
        condition = EntityPositionDistanceCondition(
            entity_name="Ego",
            position=_STOP_LINE,
            value=1000.0,
            rule=ComparisonRule.LESS_THAN,
            label="distance",
        )
        far = condition.check(_world(_actor("Ego", (0, 0, 0))), 0.0)
        near = condition.check(_world(_actor("Ego", (90, 0, 0))), 1.0)
        assert far is not None and near is not None
        assert "100.00 m" in far.message
        assert "10.00 m" in near.message

    def test_a_greater_than_rule_reads_as_being_far_enough_away(self) -> None:
        condition = EntityPositionDistanceCondition(
            entity_name="Ego",
            position=_STOP_LINE,
            value=50.0,
            rule=ComparisonRule.GREATER_THAN,
            label="clear_of_the_junction",
        )
        assert condition.check(_world(_actor("Ego", (80, 0, 0))), 0.0) is None
        assert condition.check(_world(_actor("Ego", (20, 0, 0))), 0.0) is not None


class TestHeight:
    def test_height_is_ignored_by_default(self) -> None:
        """A place on an overpass is not far away from the road beneath it."""
        condition = EntityPositionDistanceCondition(
            entity_name="Ego",
            position=CarlaWorldPose(x=0.0, y=0.0, z=100.0),
            value=5.0,
            rule=ComparisonRule.LESS_THAN,
            label="flat",
        )
        assert condition.check(_world(_actor("Ego", (3, 0, 0))), 0.0) is not None

    def test_height_counts_when_asked_for(self) -> None:
        condition = EntityPositionDistanceCondition(
            entity_name="Ego",
            position=CarlaWorldPose(x=0.0, y=0.0, z=100.0),
            value=5.0,
            rule=ComparisonRule.LESS_THAN,
            vertical=True,
            label="vertical",
        )
        assert condition.check(_world(_actor("Ego", (3, 0, 0))), 0.0) is None


class TestGuards:
    def test_an_absent_entity_yields_no_verdict(self) -> None:
        condition = EntityPositionDistanceCondition(
            entity_name="Ego", position=_STOP_LINE, value=10.0, label="a"
        )
        assert condition.check(_world(), 0.0) is None

    def test_a_negative_tolerance_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            EntityPositionDistanceCondition(
                entity_name="Ego",
                position=_STOP_LINE,
                value=10.0,
                tolerance=-1.0,
                label="a",
            )


class TestDetails:
    def test_details_report_the_resolved_place(self) -> None:
        """The resolved world point, because that is what was measured to."""
        condition = EntityPositionDistanceCondition(
            entity_name="Ego",
            position=_STOP_LINE,
            value=10.0,
            rule=ComparisonRule.LESS_THAN,
            label="near_stop_line",
        )
        details = condition.get_details()
        assert details["entity_name"] == "Ego"
        assert math.isclose(details["target_x"], 100.0)
        assert math.isclose(details["target_y"], 0.0)
        assert details["rule"] == "LESS_THAN"
        assert details["vertical"] is False
