"""A snapped pose lands on its own level of stacked roads (issue #1874).

The ray that finds the ground starts above its height estimate and hits the
first surface below.  The estimate used to be the z of the spawn point nearest
in plan, whatever its height: where a road runs under an overpass, that spawn
point can be on the deck (and the pose on the lower road was placed on the
deck) or on the road below (and the pose on the deck was placed under it).

The fake world here has a deck at z=7 over a road at z=0, a spawn point on
each, half a metre up as CARLA's own maps hold them, and a ``ground_projection``
that returns the first of the two surfaces below the ray's origin.  None of
this needs CARLA.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Generator

import pytest

from autoware_carla_scenario.coordinate import snap as snap_module
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose

_DECK_Z = 7.0
_ROAD_Z = 0.0
#: The runner's shipped ray: 5 m up, 10 m down.
_RUNNER_RAY = snap_module.GroundProjectionConfig(
    ray_distance_upper=5.0, ray_distance_lower=10.0
)

#: A spawn point on the deck at x=1 and one on the road below at x=4.
_DECK_SPAWN = (1.0, 0.0, _DECK_Z + 0.5)
_ROAD_SPAWN = (4.0, 0.0, _ROAD_Z + 0.5)


def _spawn_point(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z))


class _Map:
    def __init__(self, spawn_points: list[tuple[float, float, float]]) -> None:
        self.spawn_points = [_spawn_point(*sp) for sp in spawn_points]
        self.asked: list[Any] = []

    def get_spawn_points(self) -> list[SimpleNamespace]:
        return self.spawn_points

    def get_waypoint(self, location: Any, project_to_road: bool = True) -> Any:
        """The road nearest in 3D, as CARLA picks it: the level nearest z."""
        self.asked.append(location)
        level = (
            _DECK_Z
            if abs(location.z - _DECK_Z) < abs(location.z - _ROAD_Z)
            else _ROAD_Z
        )
        return SimpleNamespace(
            transform=SimpleNamespace(
                location=SimpleNamespace(x=location.x, y=location.y, z=level)
            )
        )


class _World:
    def __init__(self, spawn_points: list[tuple[float, float, float]]) -> None:
        self.map = _Map(spawn_points)

    def get_map(self) -> _Map:
        return self.map

    def ground_projection(self, origin: Any, distance: float) -> Any:
        """The first of the deck and the road below *origin*, within *distance*."""
        for surface in (_DECK_Z, _ROAD_Z):
            if origin.z - distance <= surface <= origin.z:
                return SimpleNamespace(
                    location=SimpleNamespace(x=origin.x, y=origin.y, z=surface),
                    label=None,
                )
        return None


@pytest.fixture(autouse=True)
def _no_held_map() -> Generator[None, None, None]:
    """No MapManager map, so the snap reads the fake world's."""
    saved = MapManager._instance
    MapManager._instance = None
    yield
    MapManager._instance = saved


@pytest.fixture
def _lanelets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lanelet 1 runs on the road (z=0), lanelet 2 on the deck over it (z=7)."""
    levels = {1: _ROAD_Z, 2: _DECK_Z}

    def _to_carla_world(pose: Lanelet2Pose) -> CarlaWorldPose:
        return CarlaWorldPose(x=pose.s, y=0.0, z=levels[pose.lanelet_id])

    monkeypatch.setattr(
        "autoware_carla_scenario.coordinate.transform.to_carla_world", _to_carla_world
    )


def _snap(pose: Any, world: _World) -> CarlaWorldPose:
    return snap_module.snap_to_carla_road(
        pose,
        world,  # type: ignore[arg-type]
        ground_projection=_RUNNER_RAY,
    )


# ---------------------------------------------------------------------------
# Lanelet2Pose
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_lanelets")
def test_a_pose_under_the_deck_stays_on_the_road_below() -> None:
    # Nearest in plan (x=1) is the deck's spawn point; the road's is at x=4.
    world = _World([_DECK_SPAWN, _ROAD_SPAWN])
    snapped = _snap(Lanelet2Pose(lanelet_id=1, s=1.0), world)
    assert snapped.z == pytest.approx(_ROAD_Z)


@pytest.mark.usefixtures("_lanelets")
def test_a_pose_on_the_deck_stays_on_the_deck() -> None:
    # Nearest in plan (x=4) is the road's spawn point, under the deck.
    world = _World([_DECK_SPAWN, _ROAD_SPAWN])
    snapped = _snap(Lanelet2Pose(lanelet_id=2, s=4.0), world)
    assert snapped.z == pytest.approx(_DECK_Z)


@pytest.mark.usefixtures("_lanelets")
@pytest.mark.parametrize(("lanelet_id", "level"), [(1, _ROAD_Z), (2, _DECK_Z)])
def test_each_level_snaps_to_itself_where_its_spawn_point_is_nearest(
    lanelet_id: int, level: float
) -> None:
    world = _World([_DECK_SPAWN, _ROAD_SPAWN])
    s = _DECK_SPAWN[0] if level == _DECK_Z else _ROAD_SPAWN[0]
    assert _snap(Lanelet2Pose(lanelet_id=lanelet_id, s=s), world).z == pytest.approx(
        level
    )


@pytest.mark.usefixtures("_lanelets")
def test_without_a_spawn_point_on_its_level_the_pose_starts_from_its_own_height() -> (
    None
):
    world = _World([_DECK_SPAWN])
    snapped = _snap(Lanelet2Pose(lanelet_id=1, s=1.0), world)
    assert snapped.z == pytest.approx(_ROAD_Z)


# ---------------------------------------------------------------------------
# CarlaWorldPose
# ---------------------------------------------------------------------------


def test_a_carla_pose_on_the_deck_is_projected_at_its_own_height() -> None:
    # Nearest in plan (x=4) is the road's spawn point, under the deck.
    world = _World([_DECK_SPAWN, _ROAD_SPAWN])
    snapped = _snap(CarlaWorldPose(x=4.0, y=0.0, z=_DECK_Z), world)

    assert [location.z for location in world.map.asked] == [_DECK_Z]
    assert snapped.z == pytest.approx(_DECK_Z)


def test_a_carla_pose_under_the_deck_stays_on_the_road_below() -> None:
    world = _World([_DECK_SPAWN, _ROAD_SPAWN])
    snapped = _snap(CarlaWorldPose(x=1.0, y=0.0, z=_ROAD_Z), world)
    assert snapped.z == pytest.approx(_ROAD_Z)


# ---------------------------------------------------------------------------
# _z_from_nearest_spawn_point
# ---------------------------------------------------------------------------


def test_without_a_hint_the_spawn_point_nearest_in_plan_is_taken() -> None:
    world = _World([_DECK_SPAWN, _ROAD_SPAWN])
    z = snap_module._z_from_nearest_spawn_point(0.0, 0.0, world)  # type: ignore[arg-type]
    assert z == _DECK_SPAWN[2]


def test_with_a_hint_only_a_spawn_point_on_its_level_is_taken() -> None:
    world = _World([_DECK_SPAWN, _ROAD_SPAWN])
    z = snap_module._z_from_nearest_spawn_point(
        0.0,
        0.0,
        world,  # type: ignore[arg-type]
        z_hint=_ROAD_Z,
    )
    assert z == _ROAD_SPAWN[2]


def test_with_a_hint_and_no_spawn_point_on_its_level_there_is_none() -> None:
    world = _World([_DECK_SPAWN])
    z = snap_module._z_from_nearest_spawn_point(
        0.0,
        0.0,
        world,  # type: ignore[arg-type]
        z_hint=_ROAD_Z,
    )
    assert z is None
