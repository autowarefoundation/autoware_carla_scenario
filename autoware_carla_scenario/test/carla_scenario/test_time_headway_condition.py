"""Time headway: the measure itself, and the two cases it differs from TTC in.

Headway is ``range / own speed``.  What makes it worth having beside
:class:`TimeToCollisionCondition` is that it is defined when TTC is not -- a
steady gap -- and undefined where TTC happens to be defined, at standstill.
Both of those are tested here, because either one silently wrong turns a
following-distance assertion into one that never fires.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    ComparisonRule,
    TimeHeadwayCondition,
    TimeToCollisionCondition,
)


def _actor(
    role: str,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MagicMock:
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(*position)
    actor.get_velocity.return_value = carla.Vector3D(*velocity)
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


class TestMeasurement:
    def test_headway_is_the_gap_divided_by_the_follower_speed(self) -> None:
        condition = TimeHeadwayCondition(
            source="Ego",
            target="npc1",
            value=2.0,
            rule=ComparisonRule.LESS_THAN,
            label="tailgating",
        )
        # 20 m ahead at 10 m/s -> 2.0 s, which is not < 2.0.
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)),
            _actor("npc1", (20, 0, 0)),
        )
        assert condition.check(world, 1.0) is None

        # 15 m ahead at 10 m/s -> 1.5 s.
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)),
            _actor("npc1", (15, 0, 0)),
        )
        result = condition.check(world, 1.0)
        assert result is not None
        assert result.passed
        assert "1.50 s" in result.message

    def test_height_is_ignored(self) -> None:
        """Two cars on a slope are not further apart because of the z gap."""
        condition = TimeHeadwayCondition(
            source="Ego", target="npc1", value=1.1, label="headway"
        )
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)),
            _actor("npc1", (10, 0, 100)),
        )
        result = condition.check(world, 0.0)
        assert result is not None
        assert "1.00 s" in result.message


class TestDifferenceFromTimeToCollision:
    def test_a_steady_gap_has_a_headway_but_no_time_to_collision(self) -> None:
        """The case headway exists for: both moving, the gap not closing."""
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)),
            _actor("npc1", (15, 0, 0), velocity=(10.0, 0.0, 0.0)),
        )

        headway = TimeHeadwayCondition(
            source="Ego",
            target="npc1",
            value=2.0,
            rule=ComparisonRule.LESS_THAN,
            label="headway",
        )
        ttc = TimeToCollisionCondition(
            source="Ego",
            target="npc1",
            value=2.0,
            rule=ComparisonRule.LESS_THAN,
            label="ttc",
        )

        assert headway.check(world, 0.0) is not None
        assert ttc.check(world, 0.0) is None

    def test_a_stopped_follower_has_no_headway(self) -> None:
        """Undefined, not infinite: a `less than` rule must not read as false.

        Reporting an unbounded value would appear in a report as "the ego is
        keeping its distance", when the truth is that the question does not
        apply to a stationary car.
        """
        condition = TimeHeadwayCondition(
            source="Ego",
            target="npc1",
            value=2.0,
            rule=ComparisonRule.GREATER_THAN,
            label="headway",
        )
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(0.0, 0.0, 0.0)),
            _actor("npc1", (15, 0, 0)),
        )
        assert condition.check(world, 0.0) is None


class TestGuards:
    def test_a_missing_target_yields_no_verdict(self) -> None:
        condition = TimeHeadwayCondition(
            source="Ego", target="npc1", value=2.0, label="headway"
        )
        world = _world(_actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)))
        assert condition.check(world, 0.0) is None

    def test_a_negative_tolerance_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            TimeHeadwayCondition(
                source="Ego", target="npc1", value=2.0, tolerance=-1.0, label="a"
            )


class TestDetails:
    def test_details_name_both_entities(self) -> None:
        condition = TimeHeadwayCondition(
            source="Ego",
            target="npc1",
            value=2.0,
            rule=ComparisonRule.LESS_THAN,
            label="headway",
        )
        details = condition.get_details()
        assert details["source"] == "Ego"
        assert details["target"] == "npc1"
        assert details["rule"] == "LESS_THAN"
        assert math.isclose(details["value"], 2.0)
