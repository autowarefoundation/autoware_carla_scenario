"""Distance from an entity to a fixed place on the map.

A `CarlaWorldPose` is used throughout so the condition can be built without a
live map; the Lanelet2 path it also accepts is the same call
(`to_carla_location`) that `EntityLanePositionCondition` already relies on.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import typesafe_carla.carla as carla
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


class TestARoutePose:
    """A route pose is placed on the scenario's route when first measured to."""

    @staticmethod
    def _pose():
        from autoware_carla_scenario.trajectory.model import RouteLanePose

        return RouteLanePose(anchor="junction:0:entry", ds=-10.0)

    def test_it_is_placed_once_when_first_measured_to(self) -> None:
        from unittest.mock import patch

        condition = EntityPositionDistanceCondition(
            entity_name="Ego", position=self._pose(), value=10.0, label="near"
        )
        assert condition.get_details()["position"] == "RouteLanePose"
        with patch(
            "autoware_carla_scenario.trajectory.resolve.resolve_position",
            return_value=(100.0, 0.0, None, None),
        ) as resolve:
            assert condition.check(_world(_actor("Ego", (85, 0, 0))), 1.0) is None
            result = condition.check(_world(_actor("Ego", (95, 0, 0))), 2.0)
        assert result is not None and result.passed
        assert resolve.call_count == 1
        assert math.isclose(condition.get_details()["target_x"], 100.0)

    def test_one_it_cannot_place_holds_nothing(self) -> None:
        from unittest.mock import patch

        condition = EntityPositionDistanceCondition(
            entity_name="Ego", position=self._pose(), value=10.0, label="near"
        )
        with patch(
            "autoware_carla_scenario.trajectory.resolve.resolve_position",
            side_effect=ValueError("no route"),
        ) as resolve:
            assert condition.check(_world(_actor("Ego", (99, 0, 0))), 1.0) is None
            assert condition.check(_world(_actor("Ego", (99, 0, 0))), 2.0) is None
        assert resolve.call_count == 1


class TestAPlaceWithoutHeight:
    """A place with no z stands on the road under it, not at CARLA z = 0 (#1030)."""

    @staticmethod
    def _condition(vertical: bool = True) -> EntityPositionDistanceCondition:
        from autoware_carla_scenario.trajectory.model import MapPose

        return EntityPositionDistanceCondition(
            entity_name="Ego",
            position=MapPose(x=5.0, y=-5.0),
            value=1.0,
            vertical=vertical,
            label="at_place",
        )

    @staticmethod
    def _measure(condition, road, at=(5.0, -5.0, -10.0)):
        from unittest.mock import patch

        reads: list[tuple[float, float]] = []

        def ground(_carla_map):
            def height(x: float, y: float):
                reads.append((x, y))
                return road

            return height

        with (
            patch(
                "autoware_carla_scenario.trajectory.resolve.resolve_position",
                return_value=(5.0, -5.0, None, None),
            ),
            patch("autoware_carla_scenario.trajectory.resolve.road_height", ground),
        ):
            first = condition._measure(_world(_actor("Ego", at)))
            second = condition._measure(_world(_actor("Ego", at)))
        return first, second, reads

    def test_the_height_is_measured_from_the_road_surface(self) -> None:
        condition = self._condition()
        first, second, reads = self._measure(condition, road=-10.0)
        assert first is not None and math.isclose(first, 0.0, abs_tol=1e-9)
        assert second is not None and math.isclose(second, 0.0, abs_tol=1e-9)
        assert reads == [(5.0, -5.0)]  # read once, then cached
        assert math.isclose(condition.get_details()["target_z"], -10.0)

    def test_the_height_above_the_road_still_counts(self) -> None:
        condition = self._condition()
        first, _, _ = self._measure(condition, road=-10.0, at=(5.0, -5.0, -7.0))
        assert first is not None and math.isclose(first, 3.0)

    def test_an_unreadable_road_leaves_the_height_out(self) -> None:
        condition = self._condition()
        first, _, reads = self._measure(condition, road=None)
        assert first is not None and math.isclose(first, 0.0, abs_tol=1e-9)
        assert len(reads) == 1
        assert "target_z" not in condition.get_details()

    def test_a_reset_reads_the_road_afresh(self) -> None:
        condition = self._condition()
        self._measure(condition, road=None)
        condition.reset()
        first, _, reads = self._measure(condition, road=-10.0, at=(5.0, -5.0, -8.0))
        assert first is not None and math.isclose(first, 2.0)
        assert len(reads) == 1

    def test_an_explicit_height_is_kept(self) -> None:
        from unittest.mock import patch

        condition = EntityPositionDistanceCondition(
            entity_name="Ego",
            position=CarlaWorldPose(x=0.0, y=0.0, z=0.0),
            value=1.0,
            vertical=True,
            label="explicit",
        )
        with patch("autoware_carla_scenario.trajectory.resolve.road_height") as ground:
            distance = condition._measure(_world(_actor("Ego", (0, 0, -10.0))))
        assert distance is not None and math.isclose(distance, 10.0)
        ground.assert_not_called()
