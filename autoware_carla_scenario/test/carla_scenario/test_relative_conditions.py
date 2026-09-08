"""Relative conditions: distance and time-to-collision between two entities.

Both read positions and velocities off CARLA actors, so the world is faked: the
arithmetic and the guard conditions are what matter, and they are what a
scenario author gets wrong.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    ComparisonRule,
    EntityDistanceCondition,
    TimeToCollisionCondition,
)


def _actor(
    role: str,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MagicMock:
    """Return a fake CARLA actor at *position* moving at *velocity*."""
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(*position)
    actor.get_velocity.return_value = carla.Vector3D(*velocity)
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


class TestEntityDistanceCondition:
    def test_fires_when_the_gap_closes_below_the_threshold(self) -> None:
        condition = EntityDistanceCondition(
            source="npc1",
            target="Ego",
            value=20.0,
            rule=ComparisonRule.LESS_THAN,
            label="gap",
        )
        world = _world(_actor("npc1", (0, 0, 0)), _actor("Ego", (15, 0, 0)))
        result = condition.check(world, 1.0)
        assert result is not None
        assert result.passed
        assert "15.00 m" in result.message

    def test_stays_silent_while_the_gap_is_wide(self) -> None:
        condition = EntityDistanceCondition(
            source="npc1", target="Ego", value=20.0, label="gap"
        )
        world = _world(_actor("npc1", (0, 0, 0)), _actor("Ego", (25, 0, 0)))
        assert condition.check(world, 1.0) is None

    def test_height_is_ignored_by_default(self) -> None:
        """Two cars on a slope are not "far apart" because of the z difference."""
        flat = EntityDistanceCondition(source="a", target="b", value=5.0, label="flat")
        world = _world(_actor("a", (0, 0, 0)), _actor("b", (3, 0, 100)))
        assert flat.check(world, 0.0) is not None

        vertical = EntityDistanceCondition(
            source="a", target="b", value=5.0, vertical=True, label="vertical"
        )
        assert vertical.check(world, 0.0) is None

    def test_a_missing_actor_is_not_a_match(self) -> None:
        condition = EntityDistanceCondition(
            source="npc1", target="Ego", value=1000.0, label="gap"
        )
        assert condition.check(_world(_actor("npc1", (0, 0, 0))), 1.0) is None

    def test_details_describe_the_condition(self) -> None:
        condition = EntityDistanceCondition(
            source="npc1",
            target="Ego",
            value=20.0,
            rule=ComparisonRule.LESS_THAN,
            label="gap",
        )
        details = condition.get_details()
        assert details["source"] == "npc1"
        assert details["target"] == "Ego"
        assert details["value"] == 20.0
        assert details["rule"] == "LESS_THAN"

    def test_a_negative_tolerance_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            EntityDistanceCondition(
                source="a", target="b", value=1.0, tolerance=-1.0, label="x"
            )


class TestTimeToCollisionCondition:
    def test_closing_at_a_known_rate_gives_the_expected_ttc(self) -> None:
        # 20 m apart, closing at 10 m/s -> 2 s.
        condition = TimeToCollisionCondition(
            source="npc1",
            target="Ego",
            value=4.0,
            rule=ComparisonRule.LESS_THAN,
            label="ttc",
        )
        world = _world(
            _actor("npc1", (0, 0, 0), (10, 0, 0)),
            _actor("Ego", (20, 0, 0), (0, 0, 0)),
        )
        result = condition.check(world, 1.0)
        assert result is not None
        assert "2.00 s" in result.message

    def test_a_receding_pair_has_no_time_to_collision(self) -> None:
        """A car driving away must never trigger a TTC condition."""
        condition = TimeToCollisionCondition(
            source="npc1", target="Ego", value=1000.0, label="ttc"
        )
        world = _world(
            _actor("npc1", (0, 0, 0), (0, 0, 0)),
            _actor("Ego", (20, 0, 0), (10, 0, 0)),
        )
        assert condition.check(world, 1.0) is None

    def test_matching_speeds_never_collide(self) -> None:
        condition = TimeToCollisionCondition(
            source="npc1", target="Ego", value=1000.0, label="ttc"
        )
        world = _world(
            _actor("npc1", (0, 0, 0), (10, 0, 0)),
            _actor("Ego", (20, 0, 0), (10, 0, 0)),
        )
        assert condition.check(world, 1.0) is None

    def test_only_the_closing_component_counts(self) -> None:
        """Lateral motion does not shorten the time to collision."""
        condition = TimeToCollisionCondition(
            source="npc1",
            target="Ego",
            value=2.5,
            rule=ComparisonRule.LESS_THAN,
            label="ttc",
        )
        # 20 m ahead, but the NPC is moving sideways: no closing speed at all.
        world = _world(
            _actor("npc1", (0, 0, 0), (0, 10, 0)),
            _actor("Ego", (20, 0, 0), (0, 0, 0)),
        )
        assert condition.check(world, 1.0) is None

    def test_co_located_actors_do_not_divide_by_zero(self) -> None:
        condition = TimeToCollisionCondition(
            source="a", target="b", value=1000.0, label="ttc"
        )
        world = _world(_actor("a", (5, 5, 0), (1, 0, 0)), _actor("b", (5, 5, 0)))
        assert condition.check(world, 0.0) is None

    def test_a_missing_actor_is_not_a_match(self) -> None:
        condition = TimeToCollisionCondition(
            source="npc1", target="Ego", value=1000.0, label="ttc"
        )
        assert condition.check(_world(_actor("Ego", (0, 0, 0))), 1.0) is None

    def test_details_describe_the_condition(self) -> None:
        condition = TimeToCollisionCondition(
            source="npc1", target="Ego", value=4.0, label="ttc"
        )
        details = condition.get_details()
        assert details["source"] == "npc1"
        assert details["target"] == "Ego"
        assert details["value"] == 4.0
