"""Unit tests for stop line utilities.

Tests are split between:
- ``TestGetStopLineLinestrings``: regulatory element traversal (``utils.stop_line``)
- ``TestGetStopLinePoses``: centroid → Lanelet2Pose conversion (``coordinate.stop_line``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_DATA = Path(__file__).resolve().parents[3] / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linestring(
    ls_id: int, points: list[tuple[float, float]], attrs: dict | None = None
) -> MagicMock:
    """Create a mock linestring with an id, points, and optional attributes."""
    mock_points = []
    for x, y in points:
        pt = MagicMock()
        pt.x = x
        pt.y = y
        mock_points.append(pt)

    ls = MagicMock()
    ls.id = ls_id
    ls.__iter__ = lambda self: iter(mock_points)
    ls.attributes = attrs or {}
    return ls


def _make_reg_elem(
    re_id: int,
    *,
    has_stop_line: bool = False,
    stop_line: object | None = None,
    subtype: str | None = None,
    params: dict | None = None,
) -> MagicMock:
    """Create a mock regulatory element."""
    re = MagicMock()
    re.id = re_id

    attrs = {}
    if subtype is not None:
        attrs["subtype"] = subtype
    re.attributes = attrs

    if has_stop_line:
        re.stopLine = stop_line
    else:
        del re.stopLine

    if params is not None:
        re.parameters = params
    else:
        re.parameters = {}

    return re


def _make_arc_coordinates(length: float) -> MagicMock:
    """Create a mock arc coordinates result."""
    arc = MagicMock()
    arc.length = length
    return arc


# ---------------------------------------------------------------------------
# TestGetStopLineLinestrings – utils/stop_line.py
# ---------------------------------------------------------------------------


class TestGetStopLineLinestrings:
    """Tests for get_stop_line_linestrings() (regulatory element traversal).

    The map is an argument, so a test supplies one directly rather than
    standing up the ``MapManager`` singleton and patching it into place.
    """

    def test_lanelet_not_found_raises(self) -> None:
        from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

        lanelet_map = MagicMock()
        lanelet_map.laneletLayer.__getitem__ = MagicMock(
            side_effect=KeyError("not found")
        )

        with pytest.raises(ValueError, match="Lanelet ID 999 not found"):
            get_stop_line_linestrings(lanelet_map, 999)

    def test_no_stop_lines_returns_empty(self) -> None:
        from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

        lanelet = MagicMock()
        lanelet.regulatoryElements = []

        lanelet_map = MagicMock()
        lanelet_map.laneletLayer.__getitem__ = MagicMock(return_value=lanelet)

        assert get_stop_line_linestrings(lanelet_map, 100) == []

    def test_traffic_light_stop_line(self) -> None:
        from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

        stop_ls = _make_linestring(10, [(1.0, 2.0)])
        reg_elem = _make_reg_elem(1, has_stop_line=True, stop_line=stop_ls)

        lanelet = MagicMock()
        lanelet.regulatoryElements = [reg_elem]

        lanelet_map = MagicMock()
        lanelet_map.laneletLayer.__getitem__ = MagicMock(return_value=lanelet)

        result = get_stop_line_linestrings(lanelet_map, 100)
        assert len(result) == 1
        assert result[0].id == 10

    def test_traffic_sign_ref_line(self) -> None:
        from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

        stop_ls = _make_linestring(20, [(5.0, 6.0)], attrs={"type": "stop_line"})
        reg_elem = _make_reg_elem(
            2, subtype="traffic_sign", params={"ref_line": [stop_ls]}
        )

        lanelet = MagicMock()
        lanelet.regulatoryElements = [reg_elem]

        lanelet_map = MagicMock()
        lanelet_map.laneletLayer.__getitem__ = MagicMock(return_value=lanelet)

        result = get_stop_line_linestrings(lanelet_map, 200)
        assert len(result) == 1
        assert result[0].id == 20

    def test_road_marking_refers(self) -> None:
        from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

        stop_ls = _make_linestring(30, [(7.0, 8.0)], attrs={"type": "stop_line"})
        reg_elem = _make_reg_elem(
            3, subtype="road_marking", params={"refers": [stop_ls]}
        )

        lanelet = MagicMock()
        lanelet.regulatoryElements = [reg_elem]

        lanelet_map = MagicMock()
        lanelet_map.laneletLayer.__getitem__ = MagicMock(return_value=lanelet)

        result = get_stop_line_linestrings(lanelet_map, 300)
        assert len(result) == 1
        assert result[0].id == 30

    def test_duplicate_stop_lines_deduplicated(self) -> None:
        from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

        stop_ls = _make_linestring(10, [(1.0, 2.0)])
        re1 = _make_reg_elem(1, has_stop_line=True, stop_line=stop_ls)
        re2 = _make_reg_elem(2, has_stop_line=True, stop_line=stop_ls)

        lanelet = MagicMock()
        lanelet.regulatoryElements = [re1, re2]

        lanelet_map = MagicMock()
        lanelet_map.laneletLayer.__getitem__ = MagicMock(return_value=lanelet)

        assert len(get_stop_line_linestrings(lanelet_map, 100)) == 1

    def test_multiple_different_stop_lines(self) -> None:
        from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

        stop_ls1 = _make_linestring(10, [(1.0, 2.0)])
        stop_ls2 = _make_linestring(20, [(3.0, 4.0)])
        re1 = _make_reg_elem(1, has_stop_line=True, stop_line=stop_ls1)
        re2 = _make_reg_elem(2, has_stop_line=True, stop_line=stop_ls2)

        lanelet = MagicMock()
        lanelet.regulatoryElements = [re1, re2]

        lanelet_map = MagicMock()
        lanelet_map.laneletLayer.__getitem__ = MagicMock(return_value=lanelet)

        result = get_stop_line_linestrings(lanelet_map, 100)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TestGetStopLinePoses – coordinate/stop_line.py
# ---------------------------------------------------------------------------


class TestGetStopLinePoses:
    """Tests for get_stop_line_poses() (centroid → Lanelet2Pose conversion)."""

    @patch("autoware_carla_scenario.coordinate.stop_line.get_stop_line_linestrings")
    def test_empty_linestrings_returns_empty(self, mock_get_ls: MagicMock) -> None:
        from autoware_carla_scenario.coordinate.stop_line import get_stop_line_poses

        mock_get_ls.return_value = []
        assert get_stop_line_poses(100) == []

    @patch("autoware_carla_scenario.coordinate.stop_line.lanelet2")
    @patch("autoware_carla_scenario.coordinate.stop_line.MapManager")
    @patch("autoware_carla_scenario.coordinate.stop_line.get_stop_line_linestrings")
    def test_centroid_computed_and_projected(
        self,
        mock_get_ls: MagicMock,
        mock_mm_cls: MagicMock,
        mock_ll2: MagicMock,
    ) -> None:
        from autoware_carla_scenario.coordinate.stop_line import get_stop_line_poses

        stop_ls = _make_linestring(10, [(1.0, 2.0), (3.0, 4.0)])
        mock_get_ls.return_value = [stop_ls]

        lanelet = MagicMock()
        mm = MagicMock()
        mock_mm_cls.get_instance.return_value = mm
        mm.lanelet_map.laneletLayer.__getitem__ = MagicMock(return_value=lanelet)

        mock_ll2.geometry.toArcCoordinates.return_value = _make_arc_coordinates(15.0)
        mock_ll2.geometry.to2D.return_value = lanelet

        result = get_stop_line_poses(100)
        assert len(result) == 1
        assert result[0].lanelet_id == 100
        assert result[0].s == pytest.approx(15.0)
        assert result[0].t == pytest.approx(0.0)

        # Centroid: (1+3)/2=2.0, (2+4)/2=3.0
        mock_ll2.core.BasicPoint2d.assert_called_once_with(2.0, 3.0)

    @patch("autoware_carla_scenario.coordinate.stop_line.lanelet2")
    @patch("autoware_carla_scenario.coordinate.stop_line.MapManager")
    @patch("autoware_carla_scenario.coordinate.stop_line.get_stop_line_linestrings")
    def test_multiple_linestrings(
        self,
        mock_get_ls: MagicMock,
        mock_mm_cls: MagicMock,
        mock_ll2: MagicMock,
    ) -> None:
        from autoware_carla_scenario.coordinate.stop_line import get_stop_line_poses

        ls1 = _make_linestring(10, [(1.0, 2.0)])
        ls2 = _make_linestring(20, [(3.0, 4.0)])
        mock_get_ls.return_value = [ls1, ls2]

        lanelet = MagicMock()
        mm = MagicMock()
        mock_mm_cls.get_instance.return_value = mm
        mm.lanelet_map.laneletLayer.__getitem__ = MagicMock(return_value=lanelet)

        mock_ll2.geometry.toArcCoordinates.side_effect = [
            _make_arc_coordinates(10.0),
            _make_arc_coordinates(20.0),
        ]
        mock_ll2.geometry.to2D.return_value = lanelet

        result = get_stop_line_poses(100)
        assert len(result) == 2
        assert result[0].s == pytest.approx(10.0)
        assert result[1].s == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Autoware detection areas (issue #857)
# ---------------------------------------------------------------------------


class _DetectionArea:
    """An Autoware ``DetectionArea`` stand-in: ``stopLine`` is a method."""

    def __init__(self, re_id: int, stop_line: object) -> None:
        self.id = re_id
        self.attributes = {"subtype": "detection_area"}
        self.parameters = {"ref_line": [stop_line], "refers": []}
        self._stop_line = stop_line

    def stopLine(self) -> object:  # noqa: N802 -- the binding's name
        return self._stop_line


class TestDetectionAreaMocked:
    def test_its_stop_line_is_read_from_the_ref_line_role(self) -> None:
        from autoware_carla_scenario.utils.stop_line import (
            regulatory_element_stop_lines,
        )

        stop_ls = _make_linestring(30, [(1.0, 2.0)], attrs={"type": "stop_line"})
        seen: set[int] = set()
        result = regulatory_element_stop_lines(_DetectionArea(3, stop_ls), seen)
        assert [ls.id for ls in result] == [30]
        assert seen == {30}

    def test_it_is_no_traffic_light_stop_line(self) -> None:
        from autoware_carla_scenario.sweeper.constraints import (
            HasStopLineConstraint,
            HasTrafficLightStopLineConstraint,
        )

        stop_ls = _make_linestring(30, [(1.0, 2.0)], attrs={"type": "stop_line"})
        lanelet = MagicMock()
        lanelet.regulatoryElements = [_DetectionArea(3, stop_ls)]
        assert HasStopLineConstraint().evaluate(lanelet)
        assert not HasTrafficLightStopLineConstraint().evaluate(lanelet)

    def test_a_traffic_light_still_counts(self) -> None:
        from autoware_carla_scenario.sweeper.constraints import (
            HasTrafficLightStopLineConstraint,
        )

        stop_ls = _make_linestring(10, [(1.0, 2.0)])
        reg_elem = _make_reg_elem(
            1, has_stop_line=True, stop_line=stop_ls, subtype="traffic_light"
        )
        lanelet = MagicMock()
        lanelet.regulatoryElements = [reg_elem]
        assert HasTrafficLightStopLineConstraint().evaluate(lanelet)


#: On the fixture map, lanelet 580 has a traffic sign (124129) and a detection
#: area (124149, whose stop line is linestring 1401), and no traffic light.
_SIGN_LANELET = 580
_DETECTION_AREA = 124149
_DETECTION_AREA_STOP_LINE = 1401


@pytest.fixture(scope="module")
def nishishinjuku() -> Any:
    from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

    return load_lanelet2_map(
        _DATA / "nishishinjuku.osm", _DATA / "nishishinjuku_carla.xodr"
    )


class TestDetectionAreaOnNishishinjuku:
    def test_the_detection_area_stop_line_is_found(self, nishishinjuku: Any) -> None:
        from autoware_carla_scenario.utils.stop_line import (
            regulatory_element_stop_lines,
        )

        element = nishishinjuku.regulatoryElementLayer[_DETECTION_AREA]
        lines = regulatory_element_stop_lines(element, set())
        assert [int(ls.id) for ls in lines] == [_DETECTION_AREA_STOP_LINE]
        assert int(element.stopLine().id) == _DETECTION_AREA_STOP_LINE

    def test_the_lanelet_has_no_traffic_light_stop_line(
        self, nishishinjuku: Any
    ) -> None:
        from autoware_carla_scenario.route.geometry import MapFeatures
        from autoware_carla_scenario.sweeper.constraints import (
            HasTrafficLightStopLineConstraint,
        )

        lanelet = nishishinjuku.laneletLayer[_SIGN_LANELET]
        assert not MapFeatures.has_traffic_light(lanelet)
        assert not HasTrafficLightStopLineConstraint().evaluate(lanelet)
        assert not MapFeatures.has_traffic_light_stop_line(lanelet)

    def test_the_constraint_agrees_with_the_traffic_lights(
        self, nishishinjuku: Any
    ) -> None:
        from autoware_carla_scenario.sweeper.constraints import (
            HasTrafficLightStopLineConstraint,
        )

        def lit(lanelet: Any) -> bool:
            for element in lanelet.regulatoryElements:
                attrs = element.attributes
                if (
                    "subtype" in attrs
                    and str(attrs["subtype"]) == "traffic_light"
                    and element.stopLine is not None
                ):
                    return True
            return False

        constraint = HasTrafficLightStopLineConstraint()
        disagree = [
            int(ll.id)
            for ll in nishishinjuku.laneletLayer
            if constraint.evaluate(ll) != lit(ll)
        ]
        assert disagree == []

    def test_the_temporary_stop_sweep_keeps_the_sign_controlled_lanelet(
        self, nishishinjuku: Any
    ) -> None:
        from autoware_carla_scenario.sweeper.constraints import parse_constraint

        constraint = parse_constraint(
            {
                "type": "and",
                "constraints": [
                    {"type": "has_stop_line"},
                    {
                        "type": "not",
                        "constraint": {"type": "has_traffic_light_stop_line"},
                    },
                ],
            }
        )
        assert constraint.evaluate(nishishinjuku.laneletLayer[_SIGN_LANELET])

    def test_the_lanelet_s_stop_lines_include_the_detection_area_s(
        self, nishishinjuku: Any
    ) -> None:
        from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

        ids = {
            int(ls.id) for ls in get_stop_line_linestrings(nishishinjuku, _SIGN_LANELET)
        }
        assert _DETECTION_AREA_STOP_LINE in ids
