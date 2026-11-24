"""Distance from an entity's body to an area of the map chosen by its ground,
and the closest it came (EntityAreaDistanceCondition, MapArea, utils.polygon)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    AreaGround,
    ComparisonRule,
    EntityAreaDistanceCondition,
    MapArea,
)
from autoware_carla_scenario.conditions import map_area_from_fields
from autoware_carla_scenario.conditions.map_area import (
    area_lanelet_ids,
    area_shapes,
    parse_attributes,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.utils.polygon import (
    bounding_circle,
    point_in_polygon,
    polygon_distance,
)

_DATA = Path(__file__).resolve().parents[3] / "data"

#: A crosswalk of the fixture map.
_CROSSWALK = 587


@pytest.fixture(scope="module")
def loaded_map() -> Iterator[MapManager]:
    """Load the fixture map once: an area is resolved on it."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    yield manager
    MapManager._instance = saved


@pytest.fixture
def run_map(loaded_map: MapManager) -> Iterator[MapManager]:
    saved = MapManager._instance
    MapManager._instance = loaded_map
    yield loaded_map
    MapManager._instance = saved


# ---------------------------------------------------------------------------
# Polygons on the ground
# ---------------------------------------------------------------------------

#: An L, not convex: its notch, round (1.5, 1.5), is outside it.
_L = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)]


class TestPolygons:
    def test_a_point_in_the_notch_of_a_polygon_that_is_not_convex_is_outside(
        self,
    ) -> None:
        assert point_in_polygon((0.5, 2.5), _L)
        assert point_in_polygon((2.5, 0.5), _L)
        assert not point_in_polygon((2.0, 2.0), _L)

    def test_a_point_on_an_edge_is_inside(self) -> None:
        assert point_in_polygon((2.0, 1.0), _L)
        assert point_in_polygon((0.0, 0.0), _L)

    def test_the_corner_order_does_not_matter(self) -> None:
        assert not point_in_polygon((2.0, 2.0), list(reversed(_L)))
        assert point_in_polygon((0.5, 2.5), list(reversed(_L)))

    def test_a_box_in_the_notch_is_as_far_as_its_nearest_edge(self) -> None:
        box = [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]
        assert polygon_distance(box, _L) == pytest.approx(0.5)

    def test_touching_overlapping_and_holding_are_all_zero(self) -> None:
        touching = [(3.0, 0.0), (4.0, 0.0), (4.0, 1.0), (3.0, 1.0)]
        overlapping = [(2.5, 0.5), (4.0, 0.5), (4.0, 2.0), (2.5, 2.0)]
        held = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        assert polygon_distance(touching, _L) == 0.0
        assert polygon_distance(overlapping, _L) == 0.0
        assert polygon_distance(held, _L) == 0.0
        assert polygon_distance(_L, held) == 0.0

    def test_a_diagonal_gap_is_corner_to_corner(self) -> None:
        box = [(4.0, 2.0), (5.0, 2.0), (5.0, 3.0), (4.0, 3.0)]
        assert polygon_distance(box, _L) == pytest.approx(math.hypot(1.0, 1.0))

    def test_the_bounding_circle_holds_every_corner(self) -> None:
        x, y, radius = bounding_circle(_L)
        for px, py in _L:
            assert math.hypot(px - x, py - y) <= radius + 1e-12


# ---------------------------------------------------------------------------
# MapArea: which ground
# ---------------------------------------------------------------------------


class TestMapArea:
    def test_attributes_are_key_value_pairs_and_a_bare_key_asks_for_any_value(
        self,
    ) -> None:
        assert parse_attributes("turn_direction=left, participant:pedestrian") == (
            ("turn_direction", "left"),
            ("participant:pedestrian", ""),
        )
        assert parse_attributes("") == ()

    @pytest.mark.parametrize("text", ["=left", "turn_direction="])
    def test_a_half_written_attribute_is_refused(self, text: str) -> None:
        with pytest.raises(ValueError, match="attribute"):
            parse_attributes(text)

    def test_from_fields_takes_the_ground_by_name(self) -> None:
        area = map_area_from_fields("junction", [12], "turn_direction=left")
        assert area == MapArea(
            AreaGround.JUNCTION, (12,), (("turn_direction", "left"),)
        )

    def test_an_unknown_ground_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown ground"):
            map_area_from_fields("PAVEMENT")

    def test_polygons_are_not_narrowed_by_picked_lanelets(self) -> None:
        with pytest.raises(ValueError, match="polygons, not lanelets"):
            MapArea(AreaGround.INTERSECTION_AREA, lanelet_ids=(12,)).validate()

    def test_it_describes_itself(self) -> None:
        area = MapArea(AreaGround.CROSSWALK, (587, 612), (("location", "urban"),))
        assert area.describe() == "crosswalks (lanelets 587, 612) where location=urban"


class TestMatching:
    def test_a_ground_of_lanelets_is_every_lanelet_of_its_subtype(
        self, run_map: MapManager
    ) -> None:
        lanelet_map = run_map.lanelet_map
        assert len(area_lanelet_ids(lanelet_map, MapArea(AreaGround.CROSSWALK))) == 84
        assert len(area_lanelet_ids(lanelet_map, MapArea(AreaGround.WALKWAY))) == 8

    def test_a_junction_is_its_turning_lanelets_narrowed_by_attributes(
        self, run_map: MapManager
    ) -> None:
        lefts = MapArea(AreaGround.JUNCTION, attributes=(("turn_direction", "left"),))
        assert len(area_lanelet_ids(run_map.lanelet_map, lefts)) == 103

    def test_a_ground_of_polygons_reads_the_polygon_layer(
        self, run_map: MapManager
    ) -> None:
        shapes = area_shapes(run_map.lanelet_map, MapArea(AreaGround.DETECTION_AREA))
        assert [s.layer for s in shapes] == ["polygon", "polygon"]
        assert (
            area_lanelet_ids(run_map.lanelet_map, MapArea(AreaGround.DETECTION_AREA))
            == []
        )

    def test_picked_lanelets_narrow_it(self, run_map: MapManager) -> None:
        area = MapArea(AreaGround.CROSSWALK, lanelet_ids=(_CROSSWALK,))
        assert area_lanelet_ids(run_map.lanelet_map, area) == [_CROSSWALK]

    def test_a_picked_lanelet_of_another_ground_is_a_mistake_said(
        self, run_map: MapManager
    ) -> None:
        road = area_lanelet_ids(run_map.lanelet_map, MapArea(AreaGround.ROAD))[0]
        with pytest.raises(ValueError, match="not one of the crosswalks"):
            area_shapes(
                run_map.lanelet_map, MapArea(AreaGround.CROSSWALK, lanelet_ids=(road,))
            )

    def test_a_picked_lanelet_the_map_lacks_is_a_mistake_said(
        self, run_map: MapManager
    ) -> None:
        with pytest.raises(ValueError, match="not on the map"):
            area_shapes(
                run_map.lanelet_map,
                MapArea(AreaGround.CROSSWALK, lanelet_ids=(99999999,)),
            )


# ---------------------------------------------------------------------------
# The condition
# ---------------------------------------------------------------------------


def _ego(x: float, y: float, yaw: float = 0.0) -> MagicMock:
    """A 4 m by 2 m vehicle whose box is centred on its origin."""
    actor = MagicMock()
    actor.attributes = {"role_name": "Ego"}
    transform = carla.Transform(carla.Location(x, y, 0.0), carla.Rotation(yaw=yaw))
    actor.get_transform.return_value = transform
    actor.get_location.return_value = transform.location
    actor.bounding_box = carla.BoundingBox(
        carla.Location(0.0, 0.0, 0.0), carla.Vector3D(2.0, 1.0, 0.8)
    )
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


def _east_of_crosswalk(manager: MapManager, gap: float) -> MagicMock:
    """The ego, facing +x, its rear *gap* m east of the crosswalk's easternmost
    corner and abreast of it: no point of the crosswalk is east of that
    corner, so the gap is the distance."""
    offset_x, offset_y = manager.mgrs_offset
    shape = area_shapes(
        manager.lanelet_map, MapArea(AreaGround.CROSSWALK, lanelet_ids=(_CROSSWALK,))
    )[0]
    corner = max(((x - offset_x, -(y - offset_y)) for x, y in shape.points))
    return _ego(corner[0] + 2.0 + gap, corner[1])


def _near_crosswalk(value: float = 0.5) -> EntityAreaDistanceCondition:
    return EntityAreaDistanceCondition(
        entity_name="Ego",
        area=MapArea(AreaGround.CROSSWALK, lanelet_ids=(_CROSSWALK,)),
        value=value,
        rule=ComparisonRule.LESS_THAN,
        label="near the crosswalk",
    )


class TestCondition:
    def test_it_measures_from_the_body_not_the_origin(
        self, run_map: MapManager
    ) -> None:
        condition = _near_crosswalk(value=3.5)
        result = condition.check(_world(_east_of_crosswalk(run_map, 3.0)), 1.0)
        assert result is not None
        # The origin is 5 m away; the body's rear, 3 m.
        assert (
            "3.00 m from crosswalks (lanelet 587), which is less than 3.50 m"
            in result.message
        )

    def test_it_keeps_the_closest_it_came_whether_or_not_it_fires(
        self, run_map: MapManager
    ) -> None:
        condition = _near_crosswalk()
        assert condition.closest_m is None
        for t, gap in ((1.0, 6.0), (2.0, 1.8), (3.0, 4.0)):
            assert condition.check(_world(_east_of_crosswalk(run_map, gap)), t) is None
        assert condition.closest_m == pytest.approx(1.8, abs=1e-3)
        assert condition.closest_at_s == 2.0
        details = condition.get_details()
        assert details["closest_m"] == pytest.approx(1.8, abs=1e-3)
        assert details["closest_at_s"] == 2.0
        assert details["closest_lanelet"] == _CROSSWALK
        assert details["area"] == "crosswalks (lanelet 587)"

    def test_over_the_area_is_zero(self, run_map: MapManager) -> None:
        condition = _near_crosswalk(value=0.1)
        result = condition.check(_world(_east_of_crosswalk(run_map, -1.0)), 1.0)
        assert result is not None
        assert condition.closest_m == 0.0

    def test_reset_forgets_the_closest(self, run_map: MapManager) -> None:
        condition = _near_crosswalk()
        condition.check(_world(_east_of_crosswalk(run_map, 2.0)), 1.0)
        condition.reset()
        assert condition.closest_m is None
        assert "closest_m" not in condition.get_details()

    def test_an_absent_entity_measures_nothing(self, run_map: MapManager) -> None:
        condition = _near_crosswalk()
        assert condition.check(_world(), 1.0) is None
        assert condition.closest_m is None

    def test_the_nearest_of_many_answers(self, run_map: MapManager) -> None:
        every = EntityAreaDistanceCondition(
            entity_name="Ego",
            area=MapArea(AreaGround.CROSSWALK),
            value=1000.0,
            label="any crosswalk",
        )
        picked = _near_crosswalk(value=1000.0)
        world = _world(_east_of_crosswalk(run_map, 1.0))
        every.check(world, 1.0)
        picked.check(world, 1.0)
        assert every.closest_m is not None and picked.closest_m is not None
        assert every.closest_m <= picked.closest_m

    def test_an_area_the_map_does_not_have_is_said_and_never_holds(
        self, run_map: MapManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Only the map can tell, so it is said, not refused: the validator
        accepts the card, and what it accepts builds (#1111)."""
        condition = EntityAreaDistanceCondition(
            entity_name="Ego",
            area=MapArea(AreaGround.PARKING_LOT),
            value=1000.0,
            label="parking",
        )
        assert condition.problem == "no parking lots on the map"
        assert "never holds" in caplog.text
        assert condition.check(_world(_east_of_crosswalk(run_map, 1.0)), 1.0) is None
        assert condition.get_details()["problem"] == "no parking lots on the map"
        assert condition.closest_m is None

    def test_a_picked_lanelet_of_another_ground_is_said_and_never_holds(
        self, run_map: MapManager
    ) -> None:
        road = area_lanelet_ids(run_map.lanelet_map, MapArea(AreaGround.ROAD))[0]
        condition = EntityAreaDistanceCondition(
            entity_name="Ego",
            area=MapArea(AreaGround.CROSSWALK, lanelet_ids=(road,)),
            value=1000.0,
            label="crosswalk",
        )
        assert "not one of the crosswalks" in condition.problem
        assert condition.check(_world(_east_of_crosswalk(run_map, 1.0)), 1.0) is None

    def test_an_area_no_map_could_hold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="polygons, not lanelets"):
            EntityAreaDistanceCondition(
                entity_name="Ego",
                area=MapArea(AreaGround.PARKING_LOT, lanelet_ids=(1,)),
                value=1.0,
                label="parking",
            )
