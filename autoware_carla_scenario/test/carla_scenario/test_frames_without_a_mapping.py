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
