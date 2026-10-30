"""LiveT4Scene: the rollout so far as a T4Scene, from the trajectory recorder's records."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from autoware_carla_scenario.t4_scene._proto import t4_scene_pb2 as pb
from autoware_carla_scenario.t4_scene.convert import _Lanes
from autoware_carla_scenario.t4_scene.live import (
    EgoExtras,
    LiveT4Scene,
    _t4_points,
    lanelets_along,
)


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

    def __init__(self, actor_id: int, x: float, vx: float) -> None:
        self.id = actor_id
        self._x, self._vx = x, vx

    def get_transform(self) -> SimpleNamespace:
        return SimpleNamespace(
            location=_vec(self._x, 1.0, 0.0),
            rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=90.0),
        )

    def get_velocity(self) -> SimpleNamespace:
        return _vec(self._vx)


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


def _run(ticks: int, history_rows: int = 3, extras: bool = True) -> LiveT4Scene:
    """The ego driving +x at 10 m/s, a tick (0.05 s) at a time."""
    world = _World()
    live = LiveT4Scene(history_rows, wheel_base_m=2.85)
    live.start(world)
    for tick in range(ticks):
        world.frame = 100 + tick
        world.snapshot = [_ActorSnapshot(1, 0.5 * tick, 10.0)]
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


def _box(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    return np.array([[x0, y1], [x1, y1], [x1, y0], [x0, y0]], dtype=np.float64)


def _fork() -> _Lanes:
    """Lanelet 1 (x 0-10) on to 2 (10-20, straight on), or to 3 (10-20, which
    overlaps 2's first metres and then bends away to y 3-6 beyond x 13)."""
    outlines = {
        1: _box(0.0, 10.0, -1.5, 1.5),
        2: _box(10.0, 20.0, -1.5, 1.5),
        3: np.array(
            [
                [10.0, 1.5],
                [13.0, 6.0],
                [20.0, 6.0],
                [20.0, 3.0],
                [14.0, 3.0],
                [13.0, -1.5],
                [10.0, -1.5],
            ]
        ),
    }
    ids = tuple(outlines)
    return _Lanes(
        ids=ids,
        outlines=tuple(outlines[i] for i in ids),
        boxes=np.array([[*o.min(axis=0), *o.max(axis=0)] for o in outlines.values()]),
        lengths={1: 10.0, 2: 10.0, 3: 11.0},
        elements={1: (), 2: (), 3: ()},
        successors={1: (2, 3), 2: (), 3: ()},
    )


def test_the_road_ahead_takes_the_branch_it_keeps_to() -> None:
    straight = np.array([[x, 0.0] for x in np.arange(1.0, 20.0, 2.0)])
    assert lanelets_along(_fork(), straight) == [1, 2]
    bending = np.array(
        [[x, 0.0] for x in (1.0, 5.0, 9.0, 11.0)] + [[15.0, 4.5], [19.0, 4.5]]
    )
    assert lanelets_along(_fork(), bending) == [1, 3]


def test_a_road_ahead_on_no_lanelet_is_the_pin_alone() -> None:
    assert lanelets_along(_fork(), np.array([[50.0, 50.0], [52.0, 50.0]]), pin=1) == [1]
    assert lanelets_along(_fork(), np.array([[50.0, 50.0], [52.0, 50.0]])) == []
