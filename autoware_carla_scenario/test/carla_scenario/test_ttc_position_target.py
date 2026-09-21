"""TTC to a place, and TTC measured from the bumper.

The entity-to-entity form is unchanged and covered by
``test_relative_conditions.py``; what is new is a target that does not move and
a range that starts at the bounding box.
"""

from __future__ import annotations

import math
import re
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    CarlaWorldPose,
    ComparisonRule,
    TimeToCollisionCondition,
)


def _actor(
    role: str,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extent: tuple[float, float, float] | None = (2.5, 1.0, 0.75),
) -> MagicMock:
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(*position)
    actor.get_velocity.return_value = carla.Vector3D(*velocity)
    transform = MagicMock()
    transform.get_forward_vector.return_value = carla.Vector3D(1.0, 0.0, 0.0)
    actor.get_transform.return_value = transform
    if extent is None:
        actor.bounding_box = None
    else:
        box = MagicMock()
        box.extent = carla.Vector3D(*extent)
        actor.bounding_box = box
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


_STOP_LINE = CarlaWorldPose(x=100.0, y=0.0, z=0.0)


def _ttc(condition: TimeToCollisionCondition, world: MagicMock) -> float:
    result = condition.check(world, 0.0)
    assert result is not None, "the condition did not fire"
    match = re.search(r"\(([0-9.]+) s\)", result.message)
    assert match is not None, result.message
    return float(match.group(1))


def _to_stop_line(**kwargs: object) -> TimeToCollisionCondition:
    return TimeToCollisionCondition(
        source="Ego",
        position=_STOP_LINE,
        value=1e9,
        rule=ComparisonRule.LESS_THAN,
        label="ttc_to_stop_line",
        **kwargs,  # type: ignore[arg-type]
    )


class TestPositionTarget:
    def test_the_time_is_the_gap_over_the_subject_speed(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)))
        assert math.isclose(_ttc(_to_stop_line(), world), 10.0, abs_tol=0.01)

    def test_driving_away_from_the_place_never_fires(self) -> None:
        """A place behind you has no time-to-collision, exactly as an entity does not."""
        world = _world(_actor("Ego", (0, 0, 0), velocity=(-10.0, 0.0, 0.0)))
        condition = TimeToCollisionCondition(
            source="Ego",
            position=_STOP_LINE,
            value=1e9,
            rule=ComparisonRule.LESS_THAN,
            label="ttc",
        )
        assert condition.check(world, 0.0) is None

    def test_a_stationary_subject_never_fires(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0)))
        condition = TimeToCollisionCondition(
            source="Ego",
            position=_STOP_LINE,
            value=1e9,
            rule=ComparisonRule.LESS_THAN,
            label="ttc",
        )
        assert condition.check(world, 0.0) is None

    def test_only_the_component_towards_the_place_counts(self) -> None:
        """Crossing traffic closes on the conflict point more slowly than it drives."""
        world = _world(_actor("Ego", (0, 0, 0), velocity=(6.0, 8.0, 0.0)))
        # The place is due +x, so only the 6 m/s component closes on it.
        assert math.isclose(_ttc(_to_stop_line(), world), 100.0 / 6.0, abs_tol=0.01)

    def test_the_message_names_the_place(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)))
        result = _to_stop_line().check(world, 0.0)
        assert result is not None
        assert "(100.0, 0.0)" in result.message


class TestFreespace:
    def test_the_range_starts_at_the_subject_bumper(self) -> None:
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0), extent=(2.5, 1.0, 0.75))
        )
        assert math.isclose(
            _ttc(_to_stop_line(freespace=True), world), 9.75, abs_tol=0.01
        )

    def test_both_bumpers_count_for_an_entity_target(self) -> None:
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)),
            _actor("npc1", (100, 0, 0)),
        )
        condition = TimeToCollisionCondition(
            source="Ego",
            target="npc1",
            value=1e9,
            rule=ComparisonRule.LESS_THAN,
            freespace=True,
            label="ttc",
        )
        assert math.isclose(_ttc(condition, world), 9.5, abs_tol=0.01)

    def test_freespace_is_off_by_default(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)))
        assert math.isclose(_ttc(_to_stop_line(), world), 10.0, abs_tol=0.01)


class TestConstruction:
    def test_neither_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            TimeToCollisionCondition(source="Ego", value=4.0, label="neither")

    def test_both_targets_are_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            TimeToCollisionCondition(
                source="Ego",
                target="npc1",
                position=_STOP_LINE,
                value=4.0,
                label="both",
            )


class TestDetails:
    def test_a_place_target_reports_where_it_resolved_to(self) -> None:
        details = _to_stop_line().get_details()
        assert math.isclose(details["target_x"], 100.0)
        assert "target" not in details

    def test_an_entity_target_reports_the_entity(self) -> None:
        condition = TimeToCollisionCondition(
            source="Ego", target="npc1", value=4.0, label="ttc"
        )
        details = condition.get_details()
        assert details["target"] == "npc1"
        assert "target_x" not in details
