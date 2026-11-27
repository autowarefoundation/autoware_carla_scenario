"""Distance from an entity's body to an area of the map chosen by its ground,
and the closest it came (EntityAreaDistanceCondition, MapArea, utils.polygon),
across the levels of stacked roads (#2314)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import lanelet2
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
from autoware_carla_scenario.utils.clearance import level_gap, on_another_level
from autoware_carla_scenario.utils.polygon import (
    bounding_circle,
    edge_value,
    ground_box,
    point_in_polygon,
    polygon_distance,
    polygon_distance_at,
    polygon_gap,
    polygon_gap_at,
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


class TestWhereTheDistanceIs:
    """Where on an area's outline the distance is read: an edge (from that
    corner to the next) and how far along it, so a height kept per corner can
    be read there (#2314)."""

    def test_a_corner_of_the_box_to_an_edge_of_the_area(self) -> None:
        # East of the L's foot (x 3, y 0..1), abreast of y 0.25.
        box = [(4.0, 0.25), (5.0, 0.25), (5.0, -0.75), (4.0, -0.75)]
        distance, edge, along = polygon_gap_at(box, _L)
        assert distance == pytest.approx(1.0)
        # Edge 1 runs (3, 0) -> (3, 1).
        assert (edge, along) == (1, pytest.approx(0.25))

    def test_a_corner_of_the_area_to_an_edge_of_the_box(self) -> None:
        # Above the L's top-right corner (1, 3), wider than the L.
        box = [(-2.0, 4.0), (4.0, 4.0), (4.0, 5.0), (-2.0, 5.0)]
        distance, edge, along = polygon_gap_at(box, _L)
        assert distance == pytest.approx(1.0)
        # Corner 4, (1, 3), or corner 5, (0, 3): both a metre off; the first
        # found is kept, at the corner itself.
        assert edge in (3, 4, 5)
        assert edge_value([0.0, 0.0, 0.0, 0.0, 7.0, 7.0], edge, along) == 7.0

    def test_the_gap_and_the_distance_are_the_located_ones(self) -> None:
        for box in (
            [(4.0, 2.0), (5.0, 2.0), (5.0, 3.0), (4.0, 3.0)],
            [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)],
        ):
            assert polygon_gap_at(box, _L)[0] == polygon_gap(box, _L)
            assert polygon_distance_at(box, _L, (0.0, 0.0))[0] == (
                polygon_distance(box, _L)
            )

    def test_over_the_area_it_is_read_nearest_the_middle(self) -> None:
        held = [(0.1, 2.0), (0.7, 2.0), (0.7, 2.6), (0.1, 2.6)]
        distance, edge, along = polygon_distance_at(held, _L, (0.4, 2.3))
        assert distance == 0.0
        # Nearest (0.4, 2.3): the left edge, (0, 3) -> (0, 0), 0.4 off; the
        # one at x 1, 0.6, and the top one, 0.7.
        assert edge == 5
        assert along == pytest.approx(0.7 / 3.0)

    def test_a_value_is_interpolated_along_the_edge_and_wraps(self) -> None:
        square = [1.0, 2.0, 3.0, 5.0]
        assert edge_value(square, 0, 0.5) == pytest.approx(1.5)
        assert edge_value(square, 3, 0.25) == pytest.approx(4.0)
        assert edge_value(square, 2, 0.0) == 3.0

    def test_nothing_to_measure_to_is_infinitely_far(self) -> None:
        assert polygon_gap_at([], _L) == (math.inf, -1, 0.0)
        assert polygon_distance_at(_L, [], (0.0, 0.0)) == (math.inf, -1, 0.0)


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

    def test_every_point_of_an_outline_has_its_height(
        self, run_map: MapManager
    ) -> None:
        lanelet_map = run_map.lanelet_map
        for area in (MapArea(AreaGround.CROSSWALK), MapArea(AreaGround.DETECTION_AREA)):
            for shape in area_shapes(lanelet_map, area):
                assert len(shape.heights) == len(shape.points)
        (shape,) = area_shapes(
            lanelet_map, MapArea(AreaGround.CROSSWALK, lanelet_ids=(_CROSSWALK,))
        )
        corners = lanelet_map.laneletLayer[_CROSSWALK].polygon3d()
        assert shape.heights == [float(p.z) for p in corners]
        assert shape.points == [(float(p.x), float(p.y)) for p in corners]

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


def _ego(x: float, y: float, yaw: float = 0.0, z: float = 0.0) -> MagicMock:
    """A 4 m by 2 m vehicle, 1.6 m tall, whose box is centred on its origin in
    plan and stands on it: the origin is where its wheels meet the road at
    height *z*, as a CARLA vehicle's is."""
    actor = MagicMock()
    actor.attributes = {"role_name": "Ego"}
    transform = carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw))
    actor.get_transform.return_value = transform
    actor.get_location.return_value = transform.location
    actor.bounding_box = carla.BoundingBox(
        carla.Location(0.0, 0.0, 0.8), carla.Vector3D(2.0, 1.0, 0.8)
    )
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


def _east_of_crosswalk(manager: MapManager, gap: float, lift: float = 0.0) -> MagicMock:
    """The ego, facing +x, its rear *gap* m east of the crosswalk's easternmost
    corner and abreast of it: no point of the crosswalk is east of that
    corner, so the gap is the distance.  It stands on the road at the
    corner's height in CARLA's frame, or *lift* m above it -- on a bridge."""
    offset_x, offset_y = manager.mgrs_offset
    shape = area_shapes(
        manager.lanelet_map, MapArea(AreaGround.CROSSWALK, lanelet_ids=(_CROSSWALK,))
    )[0]
    corner = max(
        (x - offset_x, -(y - offset_y), z - manager.z_offset)
        for (x, y), z in zip(shape.points, shape.heights)
    )
    return _ego(corner[0] + 2.0 + gap, corner[1], z=corner[2] + lift)


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

    def test_levels_must_be_one_of_three(self, run_map: MapManager) -> None:
        with pytest.raises(ValueError, match="levels must be one of"):
            EntityAreaDistanceCondition(
                entity_name="Ego",
                area=MapArea(AreaGround.CROSSWALK),
                value=1.0,
                label="crosswalk",
                levels="above",
            )
        assert _near_crosswalk().levels == "measure"


# ---------------------------------------------------------------------------
# Levels of road (#2314)
# ---------------------------------------------------------------------------


def _levels_condition(levels: str, area: MapArea) -> EntityAreaDistanceCondition:
    return EntityAreaDistanceCondition(
        entity_name="Ego",
        area=area,
        value=0.5,
        label="at a crosswalk",
        levels=levels,
    )


class TestOneLevel:
    """On the fixture map, which has no stacked roads, the three ways of
    counting levels agree with each other -- and so with the plan the
    condition measured in before levels counted."""

    def test_the_closest_is_the_same_however_levels_count(
        self, run_map: MapManager
    ) -> None:
        offset_x, offset_y = run_map.mgrs_offset
        shapes = area_shapes(run_map.lanelet_map, MapArea(AreaGround.CROSSWALK))
        egos: list[MagicMock] = []
        # Over, beside and some way off a spread of the map's crosswalks,
        # standing on the road at the crosswalk's height there.
        for shape in shapes[::7]:
            x0, y0 = shape.points[0]
            z0 = shape.heights[0] - run_map.z_offset
            for dx, dy in ((0.0, 0.0), (3.0, 0.0), (0.0, -8.0), (20.0, 20.0)):
                egos.append(_ego(x0 - offset_x + dx, -(y0 - offset_y) + dy, z=z0))
        assert len(egos) >= 40
        area = MapArea(AreaGround.CROSSWALK)
        by_levels = {
            levels: _levels_condition(levels, area)
            for levels in ("measure", "ignore", "plan")
        }
        for index, ego in enumerate(egos):
            world = _world(ego)
            answers = []
            for condition in by_levels.values():
                condition.reset()
                condition.check(world, float(index))
                answers.append(
                    (condition.closest_m, condition.get_details().get("closest"))
                )
            assert answers[0] == answers[1] == answers[2], index

    def test_a_crosswalk_the_ego_stands_by_is_as_near_as_before(
        self, run_map: MapManager
    ) -> None:
        for levels in ("measure", "ignore", "plan"):
            condition = _levels_condition(
                levels, MapArea(AreaGround.CROSSWALK, lanelet_ids=(_CROSSWALK,))
            )
            condition.check(_world(_east_of_crosswalk(run_map, 1.8)), 1.0)
            assert condition.closest_m == pytest.approx(1.8, abs=1e-3)

    def test_a_crosswalk_a_level_below_is_that_far(self, run_map: MapManager) -> None:
        """The issue's example: the ego on a road 6 m over the crosswalk."""
        area = MapArea(AreaGround.CROSSWALK, lanelet_ids=(_CROSSWALK,))
        over = _world(_east_of_crosswalk(run_map, -1.0, lift=6.0))
        measured = _levels_condition("measure", area)
        assert measured.check(over, 1.0) is None
        assert measured.closest_m == pytest.approx(6.0, abs=0.1)
        plan = _levels_condition("plan", area)
        assert plan.check(over, 1.0) is not None
        assert plan.closest_m == 0.0
        ignored = _levels_condition("ignore", area)
        assert ignored.check(over, 1.0) is None
        assert ignored.closest_m is None


#: The two levels of the synthetic map: a street, and a bridge over it.
_STREET_Z = 0.0
_BRIDGE_Z = 6.5


def _crosswalk(x0: float, x1: float, z0: float, z1: float) -> "lanelet2.core.Lanelet":
    """A crosswalk lanelet from x *x0* to *x1*, y -3..3, rising from *z0* at
    its west end to *z1* at its east end, in the Lanelet2 map's frame."""
    core = lanelet2.core
    left = core.LineString3d(
        core.getId(),
        [
            core.Point3d(core.getId(), x0, 3.0, z0),
            core.Point3d(core.getId(), x1, 3.0, z1),
        ],
    )
    right = core.LineString3d(
        core.getId(),
        [
            core.Point3d(core.getId(), x0, -3.0, z0),
            core.Point3d(core.getId(), x1, -3.0, z1),
        ],
    )
    lanelet = core.Lanelet(core.getId(), left, right)
    lanelet.attributes["subtype"] = "crosswalk"
    return lanelet


@pytest.fixture
def stacked_map() -> Iterator[dict[str, int]]:
    """A map of two crosswalks, one over the other in plan: one on the street
    (x 0..10), one on a bridge over it (x 2..12), and a third climbing a ramp
    from the street to the bridge (x 30..60) -- built here rather than read,
    with the Lanelet2 map's frame CARLA's but for the flipped y."""
    street = _crosswalk(0.0, 10.0, _STREET_Z, _STREET_Z)
    bridge = _crosswalk(2.0, 12.0, _BRIDGE_Z, _BRIDGE_Z)
    ramp = _crosswalk(30.0, 60.0, _STREET_Z, _BRIDGE_Z)
    lanelet_map = lanelet2.core.LaneletMap()
    for lanelet in (street, bridge, ramp):
        lanelet_map.add(lanelet)
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._lanelet_map = lanelet_map
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    yield {"street": street.id, "bridge": bridge.id, "ramp": ramp.id}
    MapManager._instance = saved


class TestStacked:
    def test_over_both_the_one_on_the_ego_s_level_is_under_it(
        self, stacked_map: dict[str, int]
    ) -> None:
        area = MapArea(AreaGround.CROSSWALK)
        for z, level in ((_STREET_Z, "street"), (_BRIDGE_Z, "bridge")):
            for levels in ("measure", "ignore"):
                condition = _levels_condition(levels, area)
                assert condition.check(_world(_ego(6.0, 0.0, z=z)), 1.0) is not None
                assert condition.closest_m == 0.0
                assert condition.get_details()["closest_lanelet"] == stacked_map[level]

    def test_beside_both_the_other_level_is_measured_through_the_height(
        self, stacked_map: dict[str, int]
    ) -> None:
        """East of both, 2 m clear of the bridge's crosswalk and 4 m of the
        street's: on the street, the street's is 4 m away and the bridge's,
        4.9 m over the ego's roof, further."""
        area = MapArea(AreaGround.CROSSWALK)
        street = stacked_map["street"]
        condition = _levels_condition("measure", area)
        condition.check(_world(_ego(16.0, 0.0, z=_STREET_Z)), 1.0)
        assert condition.closest_m == pytest.approx(4.0)
        assert condition.get_details()["closest_lanelet"] == street
        plan = _levels_condition("plan", area)
        plan.check(_world(_ego(16.0, 0.0, z=_STREET_Z)), 1.0)
        assert plan.closest_m == pytest.approx(2.0)
        assert plan.get_details()["closest_lanelet"] == stacked_map["bridge"]

    def test_only_the_other_level_is_its_distance_through_the_height(
        self, stacked_map: dict[str, int]
    ) -> None:
        """Picked alone, the bridge's crosswalk from the street under it: its
        deck is 4.9 m over the ego's 1.6 m roof."""
        area = MapArea(AreaGround.CROSSWALK, lanelet_ids=(stacked_map["bridge"],))
        under = _world(_ego(6.0, 0.0, z=_STREET_Z))
        measured = _levels_condition("measure", area)
        assert measured.check(under, 1.0) is None
        assert measured.closest_m == pytest.approx(_BRIDGE_Z - 1.6)
        ignored = _levels_condition("ignore", area)
        assert ignored.check(under, 1.0) is None
        assert ignored.closest_m is None
        plan = _levels_condition("plan", area)
        assert plan.check(under, 1.0) is not None
        assert plan.closest_m == 0.0

    def test_on_a_ramp_the_height_is_read_where_the_ego_is(
        self, stacked_map: dict[str, int]
    ) -> None:
        """A ramp spans both levels; it is near the ego that stands on it
        wherever on it that is, which one height span per outline would not
        tell from an ego beside it on the other level."""
        area = MapArea(AreaGround.CROSSWALK, lanelet_ids=(stacked_map["ramp"],))
        for x in (31.0, 45.0, 59.0):
            z = _BRIDGE_Z * (x - 30.0) / 30.0
            condition = _levels_condition("measure", area)
            assert condition.check(_world(_ego(x, 0.0, z=z)), 1.0) is not None
            assert condition.closest_m == 0.0
        # Beside its foot, but on the bridge's level: 2 m in plan, 6.5 m up.
        beside = _levels_condition("measure", area)
        beside.check(_world(_ego(26.0, 0.0, z=_BRIDGE_Z)), 1.0)
        assert beside.closest_m == pytest.approx(math.hypot(2.0, _BRIDGE_Z))

    def test_the_skip_by_level_finds_what_measuring_every_outline_finds(
        self, stacked_map: dict[str, int]
    ) -> None:
        area = MapArea(AreaGround.CROSSWALK)
        condition = _levels_condition("measure", area)
        outlines = condition._outlines
        for x in (-8.0, 1.0, 6.0, 11.0, 14.0, 20.0, 29.0, 40.0, 63.0):
            for y in (-9.0, 0.0, 4.5):
                for z in (_STREET_Z, 2.0, _BRIDGE_Z, 9.0):
                    ego = _ego(x, y, yaw=30.0, z=z)
                    measured = condition._measure(_world(ego))
                    assert measured is not None
                    box = ground_box(ego, ego.get_transform())
                    body = (z, z + 1.6)
                    every = []
                    for outline in outlines:
                        planar, edge, along = polygon_distance_at(
                            box, outline.points, (x, y)
                        )
                        at = edge_value(outline.heights, edge, along)
                        if on_another_level(body, (at, at), planar):
                            planar = math.hypot(planar, level_gap(body, (at, at)))
                        every.append(planar)
                    assert measured[0] == pytest.approx(min(every)), (x, y, z)

    def test_further_down_a_hill_is_the_same_level(
        self, stacked_map: dict[str, int]
    ) -> None:
        """From 40 m east of the ramp's top, 3.5 m below it: its deck is clear
        of the ego's roof by 1.9 m, but within a road's grade (10 %) of the
        plan, so it is a road climbing to it, not another level: the plan
        distance.  3.5 m further down, it would be the next level."""
        area = MapArea(AreaGround.CROSSWALK, lanelet_ids=(stacked_map["ramp"],))
        condition = _levels_condition("measure", area)
        condition.check(_world(_ego(102.0, 0.0, z=_BRIDGE_Z - 3.5)), 1.0)
        assert condition.closest_m == pytest.approx(40.0)
        lower = _levels_condition("measure", area)
        lower.check(_world(_ego(102.0, 0.0, z=_BRIDGE_Z - 7.0)), 1.0)
        assert lower.closest_m == pytest.approx(math.hypot(40.0, 7.0 - 1.6))
