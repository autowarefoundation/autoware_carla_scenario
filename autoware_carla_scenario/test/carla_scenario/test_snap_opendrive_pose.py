"""An OpenDRIVE pose whose ``lane_id`` names no lane at its ``s`` is still put
on the ground (#2545).

``lane_id`` is context only (:class:`OpenDrivePose`), so
``get_waypoint_xodr(road, lane, s)`` can return ``None`` for a pose that is
perfectly placed; the ray cast onto the ground then starts from the OpenDRIVE
map's own height rather than being skipped.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoware_carla_scenario.coordinate import snap as snap_module
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, OpenDrivePose

#: Where the road's reference line puts the pose, at the map's height.
_PLACED = CarlaWorldPose(x=12.0, y=-3.0, z=1.25, yaw=30.0)
#: The ground CARLA finds under it.
_GROUND_Z = 0.4


def _snap(monkeypatch: pytest.MonkeyPatch, waypoint: object) -> tuple:
    rays: list = []
    carla_map = SimpleNamespace(get_waypoint_xodr=lambda road, lane, s: waypoint)
    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.map_manager.run_carla_map",
        lambda _world: carla_map,
    )
    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.transform.to_carla_world",
        lambda _pose: _PLACED,
    )

    def _ray(x, y, z, _world, **_kwargs):
        rays.append((x, y, z))
        return _GROUND_Z

    monkeypatch.setattr(snap_module, "refine_z_with_ground_projection", _ray)
    snapped = snap_module.snap_to_carla_road(
        OpenDrivePose(road_id="7", lane_id=0, s=0.0, t=1.0),
        object(),  # type: ignore[arg-type]
    )
    return snapped, rays


def test_no_waypoint_still_casts_the_ray_from_the_maps_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapped, rays = _snap(monkeypatch, None)

    assert rays == [(_PLACED.x, _PLACED.y, _PLACED.z)]
    assert (snapped.x, snapped.y, snapped.z, snapped.yaw) == (
        _PLACED.x,
        _PLACED.y,
        _GROUND_Z,
        _PLACED.yaw,
    )


def test_a_waypoint_still_starts_the_ray_at_its_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waypoint = SimpleNamespace(
        transform=SimpleNamespace(location=SimpleNamespace(z=2.5))
    )
    snapped, rays = _snap(monkeypatch, waypoint)

    assert rays == [(_PLACED.x, _PLACED.y, 2.5)]
    assert snapped.z == _GROUND_Z
