"""LiDAR for a driver policy: the sensor, the rig frame, and the payload it rides in."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from carla_driver_interface.protocol import carla_driver_pb2, pack_lidar_sweep
from autoware_carla_scenario.driver.base import DriverClientConfig, DriverLidarConfig
from autoware_carla_scenario.driver.geometry import Pose
from autoware_carla_scenario.driver.observation import (
    lidar_points_to_rig,
    sensor_pose_in_rig,
)
from autoware_carla_scenario.driver.renderer import RendererDataBuilder
from autoware_carla_scenario.sensor import carla_lidar
from autoware_carla_scenario.sensor.carla_lidar import CarlaLidarSensor

from .test_driver_renderer import _ego, _world

# ---------------------------------------------------------------------------
# Rig frame
# ---------------------------------------------------------------------------


def test_a_level_mount_mirrors_y_and_lifts_by_the_mount_height() -> None:
    pose = sensor_pose_in_rig(0.5, 0.0, 2.0, 0.0, 0.0, 0.0, rear_axle_offset_m=-1.5)
    raw = np.array([[10.0, 3.0, -1.0, 0.25]], dtype=np.float32)  # CARLA: y right

    rig = lidar_points_to_rig(raw, pose)

    # x shifts by the mount and the actor->rig offset; y flips; z lifts.
    np.testing.assert_allclose(rig[0], [10.0 + 0.5 + 1.5, -3.0, 1.0, 0.25], atol=1e-6)
    assert rig.dtype == np.float32


def test_a_yawed_mount_rotates_the_sweep_the_way_carla_would() -> None:
    # CARLA yaw +90 degrees turns the sensor to the vehicle's right, so a point
    # straight ahead of the sensor lies on the rig's right: -y.
    pose = sensor_pose_in_rig(0.0, 0.0, 0.0, 0.0, 0.0, 90.0, rear_axle_offset_m=0.0)
    rig = lidar_points_to_rig(np.array([[5.0, 0.0, 0.0, 1.0]], dtype=np.float32), pose)
    np.testing.assert_allclose(rig[0, :3], [0.0, -5.0, 0.0], atol=1e-5)


def test_a_rolled_mount_puts_a_point_below_the_sensor_on_the_ground() -> None:
    # Rolled 180 degrees (mounted upside down), the sensor's "up" is the ground.
    pose = sensor_pose_in_rig(0.0, 0.0, 2.0, 180.0, 0.0, 0.0, rear_axle_offset_m=0.0)
    rig = lidar_points_to_rig(np.array([[0.0, 0.0, 2.0, 1.0]], dtype=np.float32), pose)
    np.testing.assert_allclose(rig[0, :3], [0.0, 0.0, 0.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_lidar_config_rejects_an_inverted_field_of_view() -> None:
    with pytest.raises(ValueError, match="lower_fov_deg"):
        DriverLidarConfig(upper_fov_deg=-30.0, lower_fov_deg=10.0)


def test_lidar_logical_ids_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        DriverClientConfig(lidars=(DriverLidarConfig(), DriverLidarConfig()))


def test_lidars_and_map_build_from_a_mapping() -> None:
    config = DriverClientConfig.from_mapping(
        {
            "lidars": [{"logical_id": "lidar_top", "channels": 32, "position_z": 1.9}],
            "map_dir": "/maps",
            "map_formats": ["lanelet2", "opendrive"],
        }
    )
    assert config.lidars == (
        DriverLidarConfig(logical_id="lidar_top", channels=32, position_z=1.9),
    )
    assert config.map_dir == "/maps"
    assert config.map_formats == ("lanelet2", "opendrive")
    with pytest.raises(ValueError, match="Unknown DriverLidarConfig"):
        DriverClientConfig.from_mapping({"lidars": [{"rotation_frequency": 10.0}]})


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------


def _measurement(frame: int, n: int = 3) -> SimpleNamespace:
    points = np.full((n, 4), float(frame), dtype=np.float32)
    return SimpleNamespace(frame=frame, raw_data=points.tobytes())


def test_the_sweep_of_the_frame_asked_for_is_taken_and_the_backlog_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    lidar = CarlaLidarSensor(DriverLidarConfig())
    for frame in (10, 11):
        lidar._receive(_measurement(frame))  # noqa: SLF001

    sweep = lidar.get_sweep(11)
    assert sweep is not None and sweep.shape == (3, 4) and sweep[0, 0] == 11.0
    # Nothing for frame 12 arrives: told nothing rather than the past.
    assert lidar.get_sweep(12) is None
    # A newer sweep than asked for is still this tick's best answer.
    lidar._receive(_measurement(14))  # noqa: SLF001
    sweep = lidar.get_sweep(13)
    assert sweep is not None and sweep[0, 0] == 14.0


def test_attach_pins_one_revolution_per_tick() -> None:
    blueprint = SimpleNamespace(attributes={}, has_attribute=lambda name: True)
    blueprint.set_attribute = blueprint.attributes.__setitem__
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: blueprint),
        spawn_actor=lambda bp, transform, attach_to: SimpleNamespace(
            listen=lambda cb: None
        ),
    )
    CarlaLidarSensor(DriverLidarConfig(channels=32)).attach(
        world, SimpleNamespace(type_id="vehicle.lincoln.mkz")
    )
    assert blueprint.attributes["rotation_frequency"] == "20.0"
    assert blueprint.attributes["channels"] == "32"
    assert blueprint.attributes["dropoff_general_rate"] == "0.0"


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def _sweep() -> carla_driver_pb2.LidarSweep:
    return pack_lidar_sweep(
        "lidar_top", 1_000, np.ones((5, 4)), Pose.identity().to_proto()
    )


def test_sweeps_ride_in_the_renderer_payload() -> None:
    builder = RendererDataBuilder(_world(), _ego(), DriverClientConfig())
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(builder.build(1_000, Pose.identity(), lidar=[_sweep()]))
    assert [sweep.logical_id for sweep in message.lidar] == ["lidar_top"]
    assert message.map_name == "Town10HD_Opt"


def test_a_ground_truth_failure_does_not_cost_the_sweep() -> None:
    world = _world()
    world.get_weather.side_effect = RuntimeError("CARLA hiccup")
    builder = RendererDataBuilder(world, _ego(), DriverClientConfig())
    assert builder.build(1_000, Pose.identity()) == b""
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(builder.build(1_000, Pose.identity(), lidar=[_sweep()]))
    assert len(message.lidar) == 1 and message.snapshot_timestamp_us == 1_000
