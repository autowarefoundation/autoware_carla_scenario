"""A Lanelet2 pose is placed where its own lanelet says.

``Lanelet2Pose`` is a Frenet pose on one named lanelet -- ``s`` along that
lanelet's centreline, ``t`` across it -- and the map Autoware plans on is the
Lanelet2 one. So the lanelet's own answer is not an approximation of the right
answer, it is the right answer, and the snap must not move the pose off it.

What CARLA is the authority on is the height: the ground under the point. That
part still comes from the world.
"""

from __future__ import annotations

import pytest

from autoware_carla_scenario.coordinate import snap as snap_module
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose

#: Where the lanelet itself puts the pose.
_ON_LANELET = CarlaWorldPose(x=-2861.25, y=2901.63, z=1.0, yaw=57.1)
#: The elevation CARLA reports for the ground there.
_GROUND_Z = 6.93


@pytest.fixture
def _world(monkeypatch: pytest.MonkeyPatch) -> list:
    """Stand in for the map and the world; record what the road was asked about."""
    asked: list = []

    def _to_carla_world(pose):
        asked.append(pose)
        return _ON_LANELET

    def _refuse(*_a, **_k):  # pragma: no cover - only reached on a regression
        raise AssertionError(
            "the snap consulted the OpenDRIVE road; a Lanelet2 pose needs only "
            "its lanelet"
        )

    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.transform.to_carla_world", _to_carla_world
    )
    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.transform.to_opendrive", _refuse
    )
    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.transform._lane_center_t", _refuse
    )
    monkeypatch.setattr(
        snap_module, "_z_from_nearest_spawn_point", lambda *_a, **_k: _GROUND_Z
    )
    monkeypatch.setattr(
        snap_module, "refine_z_with_ground_projection", lambda x, y, z, *_a, **_k: z
    )
    return asked


def _snap(pose):
    return snap_module.snap_to_carla_road(
        pose,
        object(),  # type: ignore[arg-type]
        ground_projection=snap_module.GroundProjectionConfig(),
    )


def test_the_pose_keeps_the_position_its_lanelet_gives_it(_world: list) -> None:
    snapped = _snap(Lanelet2Pose(lanelet_id=83, s=12.0))

    assert snapped.x == pytest.approx(_ON_LANELET.x)
    assert snapped.y == pytest.approx(_ON_LANELET.y)


def test_the_pose_keeps_the_heading_its_lanelet_gives_it(_world: list) -> None:
    snapped = _snap(Lanelet2Pose(lanelet_id=83, s=12.0))

    assert snapped.yaw == pytest.approx(_ON_LANELET.yaw)


def test_only_the_height_comes_from_the_world(_world: list) -> None:
    """A map with its own idea of elevation still lands on the road."""
    snapped = _snap(Lanelet2Pose(lanelet_id=83, s=12.0))

    assert snapped.z == pytest.approx(_GROUND_Z)
    assert snapped.z != pytest.approx(_ON_LANELET.z)


def test_the_road_network_is_never_consulted(_world: list) -> None:
    """The OpenDRIVE round trip is where every misplacement came from.

    Its stand-ins raise, so a snap that reaches for the road fails here rather
    than quietly returning a position off the lanelet the author named.
    """
    _snap(Lanelet2Pose(lanelet_id=83, s=12.0, t=1.5))

    assert [p.lanelet_id for p in _world] == [83]
