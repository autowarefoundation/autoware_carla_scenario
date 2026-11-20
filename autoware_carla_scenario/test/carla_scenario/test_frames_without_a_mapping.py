"""What tells lanes apart once Lanelet2 and OpenDRIVE are separate frames.

With no lanelet-to-road mapping, a pose crosses between the two maps through
CARLA world coordinates, so whatever identified the lane before has to be read
off the geometry: an OpenDRIVE road's ``<laneOffset>`` when its lanes are
stacked, and a lanelet's elevation when lanelets are stacked over one another.

Both run against stand-in maps, so they need no fixture.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pytest

# autoware_lanelet2_extension_python must be imported before lanelet2.
import autoware_lanelet2_extension_python.projection  # noqa: F401
import lanelet2.core

from autoware_carla_scenario.coordinate import transform
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose

_LANE_WIDTH = 3.0


class _Lane:
    def __init__(self, lane_id: int) -> None:
        self.id = lane_id
        self.lane_xml = ET.fromstring(
            f'<lane id="{lane_id}"><width sOffset="0" a="{_LANE_WIDTH}" '
            'b="0" c="0" d="0"/></lane>'
        )


class _Section:
    lane_section_xml = ET.fromstring('<laneSection s="0"/>')
    left_lanes = [_Lane(1), _Lane(2)]
    right_lanes = [_Lane(-1)]


class _Road:
    def __init__(self, lane_offset: float) -> None:
        self.road_xml = ET.fromstring(
            f'<road><lanes><laneOffset s="0" a="{lane_offset}" b="0" c="0" '
            'd="0"/></lanes></road>'
        )
        self.lane_sections = [_Section()]


class TestTheLaneOffsetShiftsTheLanes:
    """Lanes stack from the lane offset line, not the reference line."""

    def test_without_an_offset_the_reference_line_divides_the_sides(self) -> None:
        road = _Road(lane_offset=0.0)
        assert transform._find_lane_at_t(road, 5.0, 1.0) == 1
        assert transform._find_lane_at_t(road, 5.0, -1.0) == -1

    def test_an_offset_moves_every_band_with_it(self) -> None:
        # Shifted a lane to the left: the reference line now runs through the
        # middle of lane -1, and lane 1 starts a lane width out.
        road = _Road(lane_offset=_LANE_WIDTH)
        assert transform._find_lane_at_t(road, 5.0, 1.0) == -1
        assert transform._find_lane_at_t(road, 5.0, _LANE_WIDTH + 1.0) == 1
        assert transform._find_lane_at_t(road, 5.0, 2 * _LANE_WIDTH + 1.0) == 2


def _straight_lanelet(base_id: int, z: float) -> Any:
    """A ten-metre lanelet due east of the origin, three metres wide, at *z*."""

    def bound(offset: int, y: float) -> Any:
        return lanelet2.core.LineString3d(
            base_id + offset,
            [
                lanelet2.core.Point3d(base_id + offset + i + 1, x, y, z)
                for i, x in enumerate((0.0, 10.0))
            ],
        )

    return lanelet2.core.Lanelet(
        base_id, bound(10, _LANE_WIDTH / 2), bound(20, -_LANE_WIDTH / 2)
    )


class TestStackedLaneletsAreToldApartByElevation:
    """An overpass and the road beneath it are equally near in plan view."""

    LOWER = 1000
    UPPER = 2000

    @pytest.fixture
    def stacked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lanelet_map = lanelet2.core.LaneletMap()
        # Plan view alone picks the same deck for both heights.
        lanelet_map.add(_straight_lanelet(self.UPPER, z=8.0))
        lanelet_map.add(_straight_lanelet(self.LOWER, z=0.0))

        class _MapManager:
            mgrs_offset = (0.0, 0.0)
            z_offset = 0.0

            @classmethod
            def get_instance(cls) -> Any:
                return cls()

        _MapManager.lanelet_map = lanelet_map  # type: ignore[attr-defined]

        monkeypatch.setattr(transform, "MapManager", _MapManager)

    @pytest.mark.parametrize(("z", "expected"), [(0.0, LOWER), (8.0, UPPER)])
    def test_the_pose_lands_on_the_deck_at_its_own_height(
        self, stacked: None, z: float, expected: int
    ) -> None:
        pose = transform._carla_to_lanelet2(CarlaWorldPose(x=5.0, y=0.0, z=z))
        assert pose.lanelet_id == expected
        assert pose.s == pytest.approx(5.0)


def _lanelet_between(
    base_id: int, start: tuple[float, float, float], end: tuple[float, float, float]
) -> Any:
    """A three-metre-wide straight lanelet from *start* to *end* (x, y, z)."""
    (x0, y0, z0), (x1, y1, z1) = start, end
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    # The unit normal to the left of the direction of travel.
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length

    def bound(offset: int, side: float) -> Any:
        half = side * _LANE_WIDTH / 2
        return lanelet2.core.LineString3d(
            base_id + offset,
            [
                lanelet2.core.Point3d(
                    base_id + offset + 1, x0 + half * nx, y0 + half * ny, z0
                ),
                lanelet2.core.Point3d(
                    base_id + offset + 2, x1 + half * nx, y1 + half * ny, z1
                ),
            ],
        )

    return lanelet2.core.Lanelet(base_id, bound(10, 1.0), bound(20, -1.0))


def _load(monkeypatch: pytest.MonkeyPatch, *lanelets: Any) -> None:
    lanelet_map = lanelet2.core.LaneletMap()
    for lanelet in lanelets:
        lanelet_map.add(lanelet)

    class _MapManager:
        mgrs_offset = (0.0, 0.0)
        z_offset = 0.0

        @classmethod
        def get_instance(cls) -> Any:
            return cls()

    _MapManager.lanelet_map = lanelet_map  # type: ignore[attr-defined]
    monkeypatch.setattr(transform, "MapManager", _MapManager)


class TestOneRuleChoosesTheLanelet:
    """Elevation picks the level, then heading the lanelet on it (#2221)."""

    def test_the_level_wins_over_a_nearer_heading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A deck 10 m over a road it nearly parallels, heading about 0.005 rad.
        _load(
            monkeypatch,
            _lanelet_between(1000, (0.0, 0.0, 0.0), (100.0, 0.0, 0.0)),
            _lanelet_between(2000, (0.0, 0.0, 10.0), (100.0, 0.5, 10.0)),
        )
        # Facing along the ground road (CARLA yaw is clockwise, in degrees).
        on_deck = transform._carla_to_lanelet2(CarlaWorldPose(x=50.0, y=-0.2, z=10.0))
        assert on_deck.lanelet_id == 2000
        under = transform._carla_to_lanelet2(
            CarlaWorldPose(x=50.0, y=-0.2, z=0.0, yaw=-0.3)
        )
        assert under.lanelet_id == 1000

    def test_on_one_level_the_heading_chooses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two lanelets crossing at right angles on one level, the eastbound one
        # 0.5 m higher where they cross (within the stacked-height tolerance):
        # the height alone chose the northbound one.
        _load(
            monkeypatch,
            _lanelet_between(1000, (-10.0, 0.0, 0.5), (10.0, 0.0, 0.5)),
            _lanelet_between(2000, (0.0, -10.0, 0.0), (0.0, 10.0, 0.0)),
        )
        east = transform._carla_to_lanelet2(CarlaWorldPose(x=0.5, y=-0.5, z=0.0))
        assert east.lanelet_id == 1000
        assert east.heading == pytest.approx(0.0)
        north = transform._carla_to_lanelet2(
            CarlaWorldPose(x=0.5, y=-0.5, z=0.0, yaw=-90.0)
        )
        assert north.lanelet_id == 2000
        assert north.heading == pytest.approx(0.0)


class _OneSidedSection:
    lane_section_xml = ET.fromstring('<laneSection s="0"/>')
    left_lanes: list[_Lane] = []
    right_lanes = [_Lane(-1), _Lane(-2)]


class TestASideWithNoLanes:
    """Across the offset line of a one-sided road is the lane beside it, not
    the centre lane 0."""

    def test_the_innermost_lane_of_the_other_side(self) -> None:
        road = _Road(lane_offset=0.0)
        road.lane_sections = [_OneSidedSection()]  # type: ignore[list-item]
        assert transform._find_lane_at_t(road, 5.0, 0.2) == -1
        assert transform._find_lane_at_t(road, 5.0, -0.2) == -1
        assert transform._find_lane_at_t(road, 5.0, -_LANE_WIDTH - 0.2) == -2

    def test_no_lane_on_either_side_is_lane_0(self) -> None:
        road = _Road(lane_offset=0.0)
        road.lane_sections = []
        assert transform._find_lane_at_t(road, 5.0, 0.2) == 0
