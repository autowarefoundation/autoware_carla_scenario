"""LiveT4Scene: the rollout so far as a T4Scene, from the trajectory recorder's records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from autoware_carla_scenario.t4_scene._proto import t4_scene_pb2 as pb
from autoware_carla_scenario.t4_scene.convert import _Log, log_to_scene
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


def test_a_tick_between_grid_rows_is_the_current_row() -> None:
    live = _run(ticks=10)  # frames 100..109; 109 is off the run's grid
    assert live.frames() == [105, 107, 109]
    scene = live.scene("run")
    assert scene is not None
    assert list(_numeric(scene.frame_indices)) == [105, 107, 109]
    assert list(_numeric(scene.timestamps_ns)) == [
        250_000_000,
        350_000_000,
        450_000_000,
    ]
    assert list(_numeric(scene.ego.available)) == [True, True, True]


def test_the_next_tick_still_has_every_row_it_shows() -> None:
    live = _run(ticks=11)  # an off-grid tick kept what the next one shows
    assert live.frames() == [106, 108, 110]
    scene = live.scene("run")
    assert scene is not None
    assert list(_numeric(scene.ego.available)) == [True, True, True]
    assert not np.frombuffer(scene.ego.accelerations.mask, dtype="?").any()


def test_a_short_run_has_the_rows_it_has() -> None:
    assert _run(ticks=1).frames() == [100]
    assert _run(ticks=4).frames() == [101, 103]
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


def _drives(
    policy_timestep_s: float, ticks: int, history_rows: int = 3
) -> list[tuple[int, float, pb.T4Scene]]:
    """Each drive's frame, x and scene, the ego driving +x at 10 m/s with a
    policy stepping every *policy_timestep_s*, as ``CarlaDriverEntity`` does:
    every tick recorded, a scene on each policy step with the sweep taken on
    that tick."""
    world = _World()
    live = LiveT4Scene(history_rows)
    live.start(world)
    interval_us = int(round(policy_timestep_s * 1e6))
    sim_time_us = 0
    last_policy_us: int | None = None
    drives = []
    for tick in range(ticks):
        world.frame = 100 + tick
        x = 0.5 * tick
        world.snapshot = [_ActorSnapshot(1, x, 10.0)]
        sim_time_us += 50_000
        live.record(
            world,
            0.05 * tick,
            EgoExtras(
                acceleration_rig=np.array([1.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.1]),
            ),
        )
        if last_policy_us is None or sim_time_us - last_policy_us >= interval_us:
            last_policy_us = sim_time_us
            scene = live.scene(
                "run",
                lidar_points=np.zeros((1, 4), dtype=np.float32),
                lidar_age_ns=0,
            )
            assert scene is not None
            drives.append((world.frame, x, scene))
    return drives


@pytest.mark.parametrize("policy_timestep_s", [0.05, 0.15, 0.25])
def test_a_drive_off_the_grid_is_the_current_row(policy_timestep_s: float) -> None:
    drives = _drives(policy_timestep_s, ticks=20)
    assert any((frame - 100) % 2 for frame, _, _ in drives)
    for frame, x, scene in drives:
        frames = list(_numeric(scene.frame_indices))
        stamps = list(_numeric(scene.timestamps_ns))
        assert frames[-1] == frame
        assert scene.current_index == len(frames) - 1
        assert frames == list(range(frame - 2 * (len(frames) - 1), frame + 1, 2))
        now_ns = (frame - 100) * 50_000_000
        assert stamps[-1] == now_ns
        assert list(_numeric(scene.ego.odometry_timestamps_ns)) == stamps
        # The ego's state is the drive's tick's, not one 50 ms old.
        assert _numeric(scene.ego.poses)[-1, 0] == pytest.approx(x, abs=1e-6)
        assert list(_numeric(scene.ego.available)) == [True] * len(frames)
        assert not np.frombuffer(scene.ego.accelerations.mask, dtype="?").any()
        # The sweep taken on the drive's tick is on its row, at its time.
        assert [row.HasField("value") for row in scene.lidar.rows] == [False] * (
            len(frames) - 1
        ) + [True]
        assert scene.lidar.rows[-1].value.timestamp_ns == now_ns


def test_the_issue_repro_a_drive_on_frame_103() -> None:
    drives = {frame: scene for frame, _, scene in _drives(0.15, ticks=4)}
    scene = drives[103]
    # Not [100, 102] with the sweep stamped 100 ms on a row 50 ms old.
    assert list(_numeric(scene.frame_indices)) == [101, 103]
    assert list(_numeric(scene.timestamps_ns)) == [50_000_000, 150_000_000]
    assert scene.lidar.rows[-1].value.timestamp_ns == 150_000_000


def test_a_policy_on_the_grid_keeps_the_grid() -> None:
    for frame, _, scene in _drives(0.1, ticks=20):
        assert (frame - 100) % 2 == 0
        frames = list(_numeric(scene.frame_indices))
        assert frames[-1] == frame
        assert all((f - 100) % 2 == 0 for f in frames)
        assert scene.lidar.rows[-1].value.timestamp_ns == (frame - 100) * 50_000_000


def _with_a_car(history_rows: int = 3) -> tuple[_World, LiveT4Scene, _Log]:
    """A run with a second car, and the log the offline converter would
    read of the same records: only the ticks on its 0.1 s grid."""
    world = _World()
    world.actors[2] = _Actor(2, "vehicle.audi.tt")
    live = LiveT4Scene(history_rows)
    live.start(world)
    grid = _Log(step_frames=2)
    feed = live._log.feed

    def both(record: Mapping[str, Any]) -> None:
        feed(record)
        grid.feed(record)

    live._log.feed = both  # type: ignore[method-assign]
    return world, live, grid


def _tick(world: _World, live: LiveT4Scene, tick: int) -> None:
    world.frame = 100 + tick
    world.snapshot = [
        _ActorSnapshot(1, 0.5 * tick, 10.0),
        _ActorSnapshot(2, 20.0 + 0.25 * tick, 5.0),
    ]
    live.record(world, 0.05 * tick)


def test_a_drive_off_the_grid_has_the_road_users_of_its_rows() -> None:
    world, live, _ = _with_a_car()
    for tick in range(10):
        _tick(world, live, tick)
    scene = live.scene("run")
    assert scene is not None
    assert list(_numeric(scene.frame_indices)) == [105, 107, 109]
    annotations = scene.annotations
    assert list(_numeric(annotations.available)) == [True, True, True]
    assert list(_numeric(annotations.box_offsets)) == [0, 1, 2, 3]
    boxes = annotations.values.boxes
    assert [v.value for v in boxes.track_ids.values] == ["2", "2", "2"]
    # Where the car was on each (off-grid) tick: CARLA's x is the map's x.
    np.testing.assert_allclose(
        _numeric(boxes.positions)[:, 0],
        [20.0 + 0.25 * tick for tick in (5, 7, 9)],
        atol=1e-6,
    )


def test_a_drive_on_the_grid_is_what_the_log_converts_to() -> None:
    world, live, grid = _with_a_car(history_rows=4)
    for tick in range(15):
        _tick(world, live, tick)
        if tick % 2:
            continue
        scene = live.scene("run")
        assert scene is not None
        frames = live.frames()
        converted = log_to_scene(grid, frames, len(frames) - 1, "run", None)
        for field in ("timestamps_ns", "frame_indices", "annotations"):
            assert getattr(scene, field) == getattr(converted, field), field
        assert scene.ego.poses == converted.ego.poses
        assert scene.ego.available == converted.ego.available


def test_a_long_run_keeps_a_bounded_window() -> None:
    live = _run(ticks=500, history_rows=31)
    # A full history back from the last tick, every tick of it.
    assert sorted(live._log.rows) == list(range(539, 600))
    assert sorted(live._extras) == list(range(539, 600))
    assert live.frames() == list(range(539, 600, 2))


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
