"""Which road users an EntityFilter takes, and EntityCountCondition over it.

The world is a fake: actors with a type id, a role name and a transform.  The
ego stands at the origin facing +x, so its left is -y (CARLA's frame is
left-handed).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    ComparisonRule,
    EntityCountCondition,
    EntityFilter,
    RoadUserKind,
)
from autoware_carla_scenario.conditions.entity_filter import (
    entity_filter_from_fields,
    name_matches,
)


class _Actor:
    def __init__(
        self,
        actor_id: int,
        type_id: str,
        role: str,
        x: float,
        y: float,
        z: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role} if role else {}
        self._transform = carla.Transform(
            carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)
        )

    def get_transform(self) -> carla.Transform:
        return self._transform


def _world(*actors: _Actor) -> Any:
    return SimpleNamespace(get_actors=lambda: list(actors))


EGO = _Actor(1, "vehicle.mini.cooper", "Ego", 0.0, 0.0)
AHEAD = _Actor(2, "vehicle.lincoln.mkz", "npc1", 20.0, 0.0)
BEHIND = _Actor(3, "vehicle.lincoln.mkz", "background4", -15.0, 0.0)
LEFT = _Actor(4, "vehicle.lincoln.mkz", "background7", 5.0, -3.5)
WALKER = _Actor(5, "walker.pedestrian.0015", "pedestrian1", 30.0, 4.0)
HIDDEN = _Actor(6, "vehicle.lincoln.mkz", "npc2", 0.0, 0.0, z=-500.0)
NAMELESS = _Actor(7, "vehicle.lincoln.mkz", "", 10.0, 0.0)
SENSOR = _Actor(8, "sensor.camera.rgb", "npc_camera", 1.0, 0.0)
WORLD = _world(EGO, AHEAD, BEHIND, LEFT, WALKER, HIDDEN, NAMELESS, SENSOR)


def _names(entities: EntityFilter, world: Any = WORLD) -> list[str]:
    return [entity.name for entity in entities.select(world)]


class TestEntityFilter:
    def test_with_no_bounds_it_takes_every_named_road_user_but_the_reference(
        self,
    ) -> None:
        # Not the ego, one parked under the map, a nameless one or a sensor.
        assert _names(EntityFilter()) == [
            "npc1",
            "background4",
            "background7",
            "pedestrian1",
        ]

    def test_names_are_patterns(self) -> None:
        assert _names(EntityFilter(names=("background*",))) == [
            "background4",
            "background7",
        ]
        assert _names(EntityFilter(names=("npc?", "pedestrian1"))) == [
            "npc1",
            "pedestrian1",
        ]

    @pytest.mark.parametrize(
        "pattern, name, matches",
        [
            ("*", "anything", True),
            ("sumo:*", "sumo:veh.3", True),
            ("background*", "background", True),
            ("npc?", "npc12", False),
            ("*car*", "my_car_1", True),
            ("a*b*c", "aXbYc", True),
            ("a*b*c", "aXbY", False),
            ("Ego", "ego", False),
        ],
    )
    def test_a_pattern_matches_as_a_glob(
        self, pattern: str, name: str, matches: bool
    ) -> None:
        assert name_matches(pattern, name) is matches

    def test_kind(self) -> None:
        assert _names(EntityFilter(kind=RoadUserKind.PEDESTRIAN)) == ["pedestrian1"]
        assert "pedestrian1" not in _names(EntityFilter(kind=RoadUserKind.VEHICLE))

    def test_distance_from_the_reference(self) -> None:
        assert _names(EntityFilter(max_distance_m=10.0)) == ["background7"]
        assert _names(EntityFilter(min_distance_m=18.0, max_distance_m=25.0)) == [
            "npc1"
        ]

    def test_ahead_and_behind_in_the_reference_s_frame(self) -> None:
        assert _names(EntityFilter(min_ahead_m=0.0, max_ahead_m=25.0)) == [
            "npc1",
            "background7",
        ]
        assert _names(EntityFilter(max_ahead_m=0.0)) == ["background4"]

    def test_left_and_right_in_the_reference_s_frame(self) -> None:
        # Facing +x in CARLA's left-handed frame, -y is the ego's left.
        assert _names(EntityFilter(min_left_m=1.0)) == ["background7"]
        assert _names(EntityFilter(max_left_m=-1.0)) == ["pedestrian1"]

    def test_the_offsets_turn_with_the_reference(self) -> None:
        # Facing +y (yaw 90): npc1, at +x, is to its left.
        turned = _Actor(1, "vehicle.mini.cooper", "Ego", 0.0, 0.0, yaw=90.0)
        (entity,) = [
            e
            for e in EntityFilter(names=("npc1",)).select(_world(turned, AHEAD))
            if e.name == "npc1"
        ]
        assert entity.ahead == pytest.approx(0.0, abs=1e-6)
        assert entity.left == pytest.approx(20.0)
        assert entity.distance == pytest.approx(20.0)

    def test_the_reference_is_any_entity_and_may_be_counted(self) -> None:
        around_npc = EntityFilter(reference="npc1", max_distance_m=21.0)
        assert _names(around_npc) == ["Ego", "background7", "pedestrian1"]
        assert "npc1" in _names(
            EntityFilter(reference="npc1", max_distance_m=1.0, include_reference=True)
        )

    def test_without_its_reference_a_relative_filter_takes_nothing(self) -> None:
        assert _names(EntityFilter(max_distance_m=100.0), _world(AHEAD)) == []
        # One with no relative bound does not need it.
        assert _names(EntityFilter(), _world(AHEAD)) == ["npc1"]
        (alone,) = EntityFilter().select(_world(AHEAD))
        assert math.isnan(alone.distance)

    def test_bounds_that_can_never_hold_are_refused(self) -> None:
        with pytest.raises(ValueError, match="above"):
            EntityFilter(min_ahead_m=5.0, max_ahead_m=1.0).validate()
        with pytest.raises(ValueError, match="negative"):
            EntityFilter(max_distance_m=-1.0).validate()

    def test_a_document_s_fields_build_one(self) -> None:
        built = entity_filter_from_fields(
            names="npc*, background*",
            kind="vehicle",
            reference=None,
            max_distance_m=30.0,
            lanelet_ids=[3, 4],
        )
        assert built == EntityFilter(
            names=("npc*", "background*"),
            kind=RoadUserKind.VEHICLE,
            reference="Ego",
            max_distance_m=30.0,
            lanelet_ids=(3, 4),
        )
        with pytest.raises(ValueError, match="road user kind"):
            entity_filter_from_fields(kind="bicycle")


class TestEntityCountCondition:
    def test_at_least_one_by_default(self) -> None:
        ahead = EntityCountCondition(
            EntityFilter(min_ahead_m=0.0, max_ahead_m=25.0, kind=RoadUserKind.VEHICLE),
            label="vehicle_ahead",
        )
        result = ahead.check(WORLD, 1.0)
        assert result is not None and result.passed
        assert "npc1" in result.message and "background7" in result.message
        assert ahead.matched == ["npc1", "background7"]

    def test_none_is_a_count_too(self) -> None:
        clear = EntityCountCondition(
            EntityFilter(kind=RoadUserKind.PEDESTRIAN, max_distance_m=10.0),
            ComparisonRule.EQUAL_TO,
            0,
            label="no_pedestrian_near",
        )
        assert clear.check(WORLD, 0.0) is not None
        crowded = EntityCountCondition(
            EntityFilter(), ComparisonRule.GREATER_THAN, 4, label="crowd"
        )
        assert crowded.check(WORLD, 0.0) is None

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="count"):
            EntityCountCondition(EntityFilter(), count=-1, label="x")
