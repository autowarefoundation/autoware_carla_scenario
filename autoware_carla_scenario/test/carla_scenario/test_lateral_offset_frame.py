"""A lateral offset is laid off the line it was measured against.

``t`` is the distance from a reference line -- a lanelet's centreline, or an
OpenDRIVE road's reference line -- positive to that line's left. Which way the
thing standing at ``(s, t)`` faces is a separate fact, carried in ``heading``.
Laying ``t`` off the pose's heading instead agrees with the measurement only
while the pose points along the line, and fails by exactly ``2t`` when it points
against it -- which on a left-hand-traffic map converted from Lanelet2 is the
common case, not the exception.

The transforms are exercised against a stand-in map, so these run without one.
"""

from __future__ import annotations

import math

import pytest

from autoware_carla_scenario.coordinate import transform
from autoware_carla_scenario.coordinate.poses import Lanelet2Pose, OpenDrivePose

#: A reference line running due east, so "left of it" is due north (+y).
_LINE_HEADING = 0.0
_OFFSET = 2.0


class _Point:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Lanelet:
    #: Ten metres due east at the origin.
    centerline = [_Point(x, 0.0) for x in (0.0, 5.0, 10.0)]


class _Road:
    reference_line = None  # filled in by the fixture, which needs numpy


class _RoadNetwork:
    road_ids_to_object = {"1": _Road()}


class _MapManager:
    lanelet_map = type("_M", (), {"laneletLayer": {7: _Lanelet()}})()
    road_network = _RoadNetwork()
    mgrs_offset = (0.0, 0.0)
    z_offset = 0.0
    carla_map = None
    road_lanelet_mapping = None

    @classmethod
    def get_instance(cls):
        return cls()


@pytest.fixture
def _map(monkeypatch: pytest.MonkeyPatch):
    import numpy as np

    _Road.reference_line = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    _Road.z_coordinates = np.array([0.0, 0.0, 0.0])
    monkeypatch.setattr(transform, "MapManager", _MapManager)


@pytest.mark.parametrize(
    ("heading", "facing"),
    [(0.0, "along the line"), (math.pi, "against it")],
)
def test_an_opendrive_offset_is_left_of_the_reference_line(
    _map: None, heading: float, facing: str
) -> None:
    """The point sits north of an eastward line whichever way it faces."""
    pose = OpenDrivePose(road_id="1", lane_id=1, s=5.0, t=_OFFSET, heading=heading)

    carla = transform.to_carla_world(pose)

    assert carla.x == pytest.approx(5.0), facing
    # CARLA mirrors y, so "north of the line" reads as -t there.
    assert carla.y == pytest.approx(-_OFFSET), facing


@pytest.mark.parametrize(
    ("heading", "facing"),
    [(0.0, "along the centreline"), (math.pi, "against it")],
)
def test_a_lanelet_offset_is_left_of_the_centreline(
    _map: None, heading: float, facing: str
) -> None:
    """Same for a Lanelet2 pose: t is from the centreline, not from the pose."""
    pose = Lanelet2Pose(lanelet_id=7, s=5.0, t=_OFFSET, heading=heading)

    carla = transform.to_carla_world(pose)

    assert carla.x == pytest.approx(5.0), facing
    assert carla.y == pytest.approx(-_OFFSET), facing


def test_facing_the_other_way_still_changes_the_yaw(_map: None) -> None:
    """The heading must still do its own job: it turns the pose, not moves it."""
    along = transform.to_carla_world(
        OpenDrivePose(road_id="1", lane_id=1, s=5.0, t=_OFFSET, heading=0.0)
    )
    against = transform.to_carla_world(
        OpenDrivePose(road_id="1", lane_id=1, s=5.0, t=_OFFSET, heading=math.pi)
    )

    assert abs((against.yaw - along.yaw + 180) % 360 - 180) == pytest.approx(180.0)
