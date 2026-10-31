"""LiveT4Scene: the rollout so far as a T4Scene, from the trajectory recorder's records."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from autoware_carla_scenario.t4_scene._proto import t4_scene_pb2 as pb
from autoware_carla_scenario.t4_scene.live import (
    EgoExtras,
    LiveT4Scene,
    _t4_points,
)
from autoware_carla_scenario.t4_scene.route import RouteGraph, _resample


def _vec(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


class _Actor:
    def __init__(self, actor_id: int, type_id: str, role: str = "") -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role}
        self.bounding_box = SimpleNamespace(
            extent=_vec(2.45, 1.07, 0.77), location=_vec(0.0, 0.0, 0.7)
        )

    def get_control(self) -> SimpleNamespace:
        return SimpleNamespace(throttle=0.5, steer=0.0, brake=0.0, gear=1)


class _ActorSnapshot:
    """An actor's state on one tick, facing CARLA's +y (yaw 90)."""

    def __init__(self, actor_id: int, x: float, vx: float, wz: float = 0.0) -> None:
        self.id = actor_id
        self._x, self._vx, self._wz = x, vx, wz

    def get_transform(self) -> SimpleNamespace:
        return SimpleNamespace(
            location=_vec(self._x, 1.0, 0.0),
            rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=90.0),
        )

    def get_velocity(self) -> SimpleNamespace:
        return _vec(self._vx)

    def get_angular_velocity(self) -> SimpleNamespace:
        return _vec(0.0, 0.0, self._wz)


class _World:
    """A world with the ego, no traffic light and no map files."""

    def __init__(self) -> None:
        self.id = 7
        self.actors = {1: _Actor(1, "vehicle.lincoln.mkz", "Ego")}
        self.snapshot: list[_ActorSnapshot] = []
        self.frame = 100

    def get_vehicles_light_states(self) -> dict[int, int]:
        return {}

    def get_map(self) -> SimpleNamespace:
        return SimpleNamespace(name="Carla/Maps/Town10HD_Opt")

    def get_actors(self, ids: list[int] | None = None):
        if ids is None:
            actors = MagicMock()
            actors.filter.return_value = []
            return actors
        return [self.actors[i] for i in ids if i in self.actors]

    def get_snapshot(self) -> SimpleNamespace:
        actors = self.snapshot
        snapshot = MagicMock()
        snapshot.frame = self.frame
        snapshot.__iter__.side_effect = lambda: iter(actors)
        return snapshot


def _numeric(array: pb.NumericArray) -> np.ndarray:
    dtype = {
        pb.NumericArray.BOOL: "?",
        pb.NumericArray.INT64: "<i8",
        pb.NumericArray.FLOAT32: "<f4",
        pb.NumericArray.FLOAT64: "<f8",
    }[array.dtype]
    return np.frombuffer(array.data, dtype=dtype).reshape(tuple(array.shape))


def _run(
    ticks: int, history_rows: int = 3, extras: bool = True, wz: float = 0.0
) -> LiveT4Scene:
    """The ego driving +x at 10 m/s, a tick (0.05 s) at a time, turning at
    *wz* (CARLA's deg/s)."""
    world = _World()
    live = LiveT4Scene(history_rows, wheel_base_m=2.85)
    live.start(world)
    for tick in range(ticks):
        world.frame = 100 + tick
        world.snapshot = [_ActorSnapshot(1, 0.5 * tick, 10.0, wz)]
        live.record(
            world,
            0.05 * tick,
            EgoExtras(
                acceleration_rig=np.array([1.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.1]),
                tire_steering_angle=0.02,
                velocity_rig=np.array([9.5, 0.25]),
            )
            if extras
            else None,
        )
    return live


def test_the_scene_is_the_last_rows_on_the_grid_the_current_one_last() -> None:
    live = _run(ticks=9)  # frames 100..108; the grid every other frame
    assert live.frames() == [104, 106, 108]
    scene = live.scene("run")
    assert scene is not None
    assert scene.current_index == 2
    assert list(_numeric(scene.frame_indices)) == [104, 106, 108]
    # On the run's clock: nanoseconds since its first tick.
    assert list(_numeric(scene.timestamps_ns)) == [
        200_000_000,
        300_000_000,
        400_000_000,
    ]
    assert list(_numeric(scene.ego.available)) == [True, True, True]


def test_a_tick_between_grid_rows_keeps_the_oldest_row_a_scene_shows() -> None:
    live = _run(ticks=10)  # frames 100..109; the last on the grid is 108
    assert live.frames() == [104, 106, 108]
    scene = live.scene("run")
    assert scene is not None
    assert list(_numeric(scene.ego.available)) == [True, True, True]


def test_a_short_run_has_the_rows_it_has() -> None:
    assert _run(ticks=1).frames() == [100]
    assert _run(ticks=4).frames() == [100, 102]
    assert _run(ticks=0).scene("run") is None


def test_the_ego_carries_what_the_policy_was_told() -> None:
    scene = _run(ticks=9).scene("run")
    assert scene is not None
    ego = scene.ego
    # The fake ego faces CARLA's +y (yaw 90), the map's -y: its rig's +x
    # acceleration points along the map's -y.
    np.testing.assert_allclose(_numeric(ego.accelerations)[-1], [0.0, -1.0], atol=1e-9)
    # The velocity the policy was told (base_link's), not the actor's origin's.
    np.testing.assert_allclose(_numeric(ego.velocities)[-1], [0.25, -9.5], atol=1e-9)
    np.testing.assert_allclose(_numeric(ego.angular_velocities)[-1], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(_numeric(ego.tire_steering_angles), 0.02)
    assert list(_numeric(ego.odometry_timestamps_ns)) == list(
        _numeric(scene.timestamps_ns)
    )
    assert ego.shape.wheel_base == pytest.approx(2.85)


def test_without_extras_they_are_masked_not_zero() -> None:
    scene = _run(ticks=5, extras=False).scene("run")
    assert scene is not None
    for array in (scene.ego.accelerations, scene.ego.tire_steering_angles):
        assert array.HasField("mask")
        assert np.frombuffer(array.mask, dtype="?").all()


def test_without_extras_the_angular_velocity_is_the_logs() -> None:
    scene = _run(ticks=5, extras=False, wz=-math.degrees(0.2)).scene("run")
    assert scene is not None
    array = scene.ego.angular_velocities
    assert not np.frombuffer(array.mask, dtype="?").any()
    # CARLA's clockwise -z is the right-handed frame's counter-clockwise +z.
    np.testing.assert_allclose(_numeric(array), [[0.0, 0.0, 0.2]] * 3, atol=1e-3)


def test_the_current_sweep_route_and_goal_ride_along() -> None:
    live = _run(ticks=5)
    points = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    scene = live.scene(
        "run",
        lidar_points=points,
        lidar_age_ns=50_000_000,
        route_lanelet_ids=[11, 12],
        goal_pose=(10.0, 20.0, math.pi / 2),
    )
    assert scene is not None
    rows = len(live.frames())
    assert [row.HasField("value") for row in scene.lidar.rows] == [False] * (
        rows - 1
    ) + [True]
    sweep = scene.lidar.rows[-1].value
    np.testing.assert_allclose(_numeric(sweep.lidar_pc), [[1.0, 2.0, 3.0, 127.5, 0.0]])
    # Taken a tick before the current row (frame 104, 0.2 s into the run).
    assert sweep.timestamp_ns == 150_000_000
    assert [list(_numeric(row.value.lane_ids)) for row in scene.routes.rows] == [
        [11, 12]
    ] * rows
    np.testing.assert_allclose(
        _numeric(scene.goal_poses)[-1], [10.0, 20.0, 0.0, 1.0], atol=1e-12
    )


def test_t4_points_scale_the_intensity_and_zero_the_fifth_column() -> None:
    out = _t4_points(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    assert out.shape == (1, 5) and out[0, 3] == 255.0 and out[0, 4] == 0.0


# -- the road ahead on the lanelets ---------------------------------------------


def test_a_scene_keeps_the_poses_unrounded() -> None:
    # The log keeps millimetres; a scene keeps what the policy is told.
    world = _World()
    live = LiveT4Scene(1)
    live.start(world)
    world.snapshot = [_ActorSnapshot(1, 0.1234567, 10.0)]
    live.record(world, 0.0)
    scene = live.scene("run")
    assert scene is not None
    # Facing the map's -y, base_link is behind the origin along y, not x.
    assert _numeric(scene.ego.poses)[-1, 0] == pytest.approx(0.1234567, abs=1e-12)


def _straight(*ids: int) -> RouteGraph:
    """Lanelets *ids*, 10 m each along y = 0 from x = 0, one after another."""
    centres = np.stack(
        [
            _resample(np.array([[10.0 * k, 0.0], [10.0 * k + 10.0, 0.0]]))
            for k in range(len(ids))
        ]
    )
    successors = {a: (b,) for a, b in zip(ids, ids[1:])}
    return RouteGraph(ids, centres, successors, {})


def test_the_route_keeps_the_lanelets_just_passed() -> None:
    graph = _straight(1, 2, 3, 4, 5, 6, 7, 8)
    live = LiveT4Scene(history_rows=1)
    road = np.array([[x, 0.0] for x in np.arange(0.0, 80.0, 2.0)])
    assert live._route_along(graph, road, np.array([1.0, 0.0])) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]  # noqa: SLF001
    ahead = np.array([[x, 0.0] for x in np.arange(61.0, 80.0, 2.0)])
    assert live._route_along(graph, ahead, np.array([61.0, 0.0])) == [3, 4, 5, 6, 7, 8]  # noqa: SLF001
