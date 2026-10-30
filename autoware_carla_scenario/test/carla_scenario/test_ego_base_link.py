"""The ego's base_link as the run uses it, in the trajectory log and the T4
dataset (#739).

Each ego kind that places its frame at a rear-axle offset of its own -- the
driver entity's rig origin, the ROS 2 ego's odometry, the bridged Autoware
stack's base_link -- says where from its ``on_scenario_start``; the log's
``rear_axle`` record and the dataset's ego poses are then that offset, not the
one ``rear_axle_offset`` guesses from the wheels or the box.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoware_carla_scenario import t4
from autoware_carla_scenario.autoware_bridge import BridgePose
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.ego_base_link import (
    clear_ego_base_link,
    ego_base_link,
    set_ego_base_link,
)
from autoware_carla_scenario.trajectory_recorder import TrajectoryRecorder

from . import test_autoware_ego_entity as autoware_tests
from . import test_carla_driver_entity as driver_tests
from . import test_ros2_entity as ros2_tests
from . import test_trajectory_recorder as recorder_tests

#: The offset every ego kind here is configured with: neither the log's fake
#: ego (half its 2.45 m box: -1.225) nor the dataset's (-1.2) guesses it.
_CONFIGURED = -1.4

#: Starts an ego of one kind and returns what ends its scenario.
Starter = Callable[[pytest.MonkeyPatch], Callable[[], None]]


def _ros2(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    entity, _, _, clock = ros2_tests._started()  # rear_axle_offset_m=-1.4
    return lambda: entity.on_scenario_end(clock.world)


def _driver(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    entity, _ = driver_tests._entity()  # rear_axle_offset_m=-1.4
    world = driver_tests._world_with_route()
    entity.on_scenario_start(world)
    return lambda: entity.on_scenario_end(world)


def _autoware(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    # The map frame without a map, as test_autoware_ego_entity converts it.
    monkeypatch.setattr(
        "autoware_carla_scenario.entity.bridged_ego.to_map_frame",
        lambda pose: BridgePose.from_yaw(
            x=pose.x, y=-pose.y, z=pose.z, yaw=math.radians(-pose.yaw)
        ),
    )
    entity = autoware_tests._make_entity(base_link_offset_m=_CONFIGURED)
    world = autoware_tests._FakeWorld(
        [autoware_tests._FakeActor(1, str(EGO_ROLE_NAME))]
    )
    autoware_tests._place(entity, world)
    entity.on_scenario_start(world)
    return lambda: entity.on_scenario_end(world)


_KINDS = pytest.mark.parametrize(
    "start",
    [
        pytest.param(_driver, id="driver"),
        pytest.param(_ros2, id="ros2"),
        pytest.param(_autoware, id="autoware"),
    ],
)


@pytest.fixture(autouse=True)
def _nothing_left_running() -> Iterator[None]:
    yield
    t4.withdraw_driver_sensors()
    clear_ego_base_link()


def test_the_registry_holds_what_was_set_until_cleared() -> None:
    assert ego_base_link() is None
    set_ego_base_link(-1)
    assert ego_base_link() == -1.0
    assert isinstance(ego_base_link(), float)
    clear_ego_base_link()
    assert ego_base_link() is None


@_KINDS
def test_an_ego_sets_its_base_link_for_its_scenario(
    start: Starter, monkeypatch: pytest.MonkeyPatch
) -> None:
    end = start(monkeypatch)
    assert ego_base_link() == pytest.approx(_CONFIGURED)
    end()
    # Nothing of this scenario's ego is read as the next one's.
    assert ego_base_link() is None


@_KINDS
def test_the_log_s_rear_axle_is_where_the_ego_put_its_base_link(
    start: Starter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    end = start(monkeypatch)
    world = recorder_tests._World()
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [recorder_tests._ActorSnapshot(1, 0.0, 0.0)]
    try:
        recorder.record(world, 0.0)
    finally:
        end()
    recorder.close()
    rear_axles = [
        r for r in recorder_tests._lines(recorder.path) if r["type"] == "rear_axle"
    ]
    assert rear_axles == [{"type": "rear_axle", "id": 1, "offset": _CONFIGURED}]


def test_without_an_ego_s_base_link_the_log_guesses_it(tmp_path: Path) -> None:
    world = recorder_tests._World()
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [recorder_tests._ActorSnapshot(1, 0.0, 0.0)]
    recorder.record(world, 0.0)
    recorder.close()
    rear_axles = [
        r for r in recorder_tests._lines(recorder.path) if r["type"] == "rear_axle"
    ]
    assert rear_axles == [{"type": "rear_axle", "id": 1, "offset": -1.225}]


@_KINDS
def test_the_dataset_s_ego_pose_is_where_the_ego_put_its_base_link(
    start: Starter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("autoware_carla_scenario_t4")
    from . import test_t4_dataset as t4_tests  # noqa: PLC0415

    end = start(monkeypatch)
    world = t4_tests._World()
    try:
        recording = t4_tests._start(tmp_path, monkeypatch, world)
    finally:
        end()
    jpeg = t4_tests._jpeg()
    for frame in range(101, 111):
        world.frame = frame
        if frame % 2 == 1:
            if recording.fed_by_driver:
                recording.feed_lidar(
                    frame, np.array([[11.25, -3.5, 0.7, 0.9]], dtype=np.float32)
                )
                recording.feed_camera_jpeg(frame, "cam_front", jpeg)
            else:
                world.sense()
        recording.record(world)  # type: ignore[arg-type]
    root = recording.close()
    assert root is not None
    first: dict[str, Any] = t4_tests._table(root, "ego_pose")[0]
    # The fake ego's box would put it at -1.2 m; the driver config the
    # dataset's settings come from sets no offset.
    np.testing.assert_allclose(
        first["translation"][0],
        world.ego_x(101) + _CONFIGURED + t4_tests.OFFSETS[0],
        atol=1e-9,
    )
