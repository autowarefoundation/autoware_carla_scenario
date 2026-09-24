"""The lane centre a snap corrects to is on the side the pose is already on.

OpenDRIVE numbers lanes positive to the left of the reference line and negative
to the right, so the sign of a lane's centre follows its id. On a left-hand-
traffic map converted from Lanelet2 the id the lanelet-to-road mapping returns is
frequently for the other side, because the reference line runs against the lane.
Taking that centre verbatim reflects the pose across the reference line into the
opposing lane -- which for a goal is a goal in the wrong lane.
"""

from __future__ import annotations

import pytest

from autoware_carla_scenario.coordinate import snap as snap_module
from autoware_carla_scenario.coordinate.poses import (
    CarlaWorldPose,
    Lanelet2Pose,
    OpenDrivePose,
)

#: Where the projection puts the pose: right of the reference line, mid-lane.
_PROJECTED_T = -1.71
#: What the lane-centre calculation returns for the id the mapping gave.
_CENTRE_T_FAR_SIDE = 1.91
_CENTRE_T_SAME_SIDE = -1.91


class _FakeRoad:
    pass


class _FakeRoadNetwork:
    def __init__(self) -> None:
        self.road_ids_to_object = {"1028": _FakeRoad()}


class _FakeMapManager:
    road_network = _FakeRoadNetwork()

    @classmethod
    def get_instance(cls) -> "_FakeMapManager":
        return cls()


@pytest.fixture
def _transforms(monkeypatch: pytest.MonkeyPatch):
    """Stand in for the map-backed conversions; record the t that is used."""
    used: dict[str, float] = {}

    def _to_opendrive(pose):
        return OpenDrivePose(
            road_id="1028", lane_id=1, s=10.0, t=_PROJECTED_T, heading=0.0
        )

    def _to_carla_world(pose):
        if isinstance(pose, OpenDrivePose):
            used["t"] = pose.t
        return CarlaWorldPose(x=0.0, y=0.0, z=0.0, yaw=0.0)

    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.transform.to_opendrive", _to_opendrive
    )
    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.transform.to_carla_world", _to_carla_world
    )
    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.map_manager.MapManager", _FakeMapManager
    )
    monkeypatch.setattr(
        snap_module, "_z_from_nearest_spawn_point", lambda *_a, **_k: 0.0
    )
    monkeypatch.setattr(
        snap_module, "refine_z_with_ground_projection", lambda x, y, z, *_a, **_k: z
    )
    return used


def _snap(used, centre_t):
    from autoware_carla_scenario.coordinate import transform

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(transform, "_lane_center_t", lambda *_a, **_k: centre_t)
        snap_module._snap_lanelet2_via_opendrive(
            Lanelet2Pose(lanelet_id=176640, s=13.6),
            world=object(),  # type: ignore[arg-type]
            ground_projection=snap_module.GroundProjectionConfig(),
        )
    return used["t"]


def test_a_centre_on_the_far_side_is_mirrored_back(_transforms) -> None:
    """The correction must not carry the pose across the reference line."""
    assert _snap(_transforms, _CENTRE_T_FAR_SIDE) == pytest.approx(_CENTRE_T_SAME_SIDE)


def test_a_centre_already_on_the_right_side_is_used_as_is(_transforms) -> None:
    """The correction is still wanted: the lane centre is not the centreline."""
    assert _snap(_transforms, _CENTRE_T_SAME_SIDE) == pytest.approx(_CENTRE_T_SAME_SIDE)
