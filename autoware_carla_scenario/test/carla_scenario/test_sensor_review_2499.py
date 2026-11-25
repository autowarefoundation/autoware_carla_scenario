"""The sensor review of #2499: the LiDAR's fog scale, its noise seed, the HILS
LiDAR's lever arm and a camera's fps off the tick."""

from __future__ import annotations

import functools
import logging
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.driver.base import (
    DriverCameraConfig,
    DriverClientConfig,
    DriverLidarConfig,
)
from autoware_carla_scenario.ros2.config import Ros2CameraConfig
from autoware_carla_scenario.sensor import carla_camera, carla_lidar
from autoware_carla_scenario.sensor.base import frame_rate_text, frame_ticks
from autoware_carla_scenario.sensor.carla_lidar import CarlaLidarSensor

# ---------------------------------------------------------------------------
# 1. The fog scale
# ---------------------------------------------------------------------------


def _fog_scales(lidars: list[Any], server_scale: Any) -> list[float]:
    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    cfg = OmegaConf.create({"server": {"fog_density_scale": server_scale}})
    driver = run._with_fog_density_scale({"lidars": lidars}, cfg)  # noqa: SLF001
    config = DriverClientConfig.from_mapping(driver)
    return [lidar.fog_density_scale for lidar in config.lidars]


def test_a_null_fog_scale_takes_the_servers_as_an_absent_one_does() -> None:
    lidars = [
        {"logical_id": "a"},
        {"logical_id": "b", "fog_density_scale": None},
        {"logical_id": "c", "fog_density_scale": 0.05},
    ]
    assert _fog_scales(lidars, 0.01) == [0.01, 0.01, 0.05]
    # No server scale: CARLA's default, the LiDAR's own.
    assert _fog_scales(lidars, None) == [0.001, 0.001, 0.05]


def test_a_structured_lidar_takes_the_servers_fog_scale() -> None:
    from autoware_carla_scenario.scenario_config import (  # noqa: PLC0415
        DriverConfig,
        DriverLidarSpec,
    )

    assert DriverLidarSpec().fog_density_scale is None
    merged = OmegaConf.merge(
        OmegaConf.structured(DriverConfig), {"lidars": [{"logical_id": "c"}]}
    )
    container = OmegaConf.to_container(merged)
    assert isinstance(container, dict)
    lidars = container["lidars"]
    assert _fog_scales(lidars, 0.01) == [0.01]
    assert _fog_scales(lidars, None) == [0.001]


# ---------------------------------------------------------------------------
# 2. The noise seed
# ---------------------------------------------------------------------------


def _ring(n: int = 1000) -> np.ndarray:
    return np.tile(np.array([[30.0, 0.0, 0.0, 1.0]], np.float32), (n, 1))


def _rainy_sweeps(start: int, ticks: int) -> list[np.ndarray]:
    """The sweeps of a LiDAR attached when the server is on frame *start*."""
    lidar = CarlaLidarSensor(DriverLidarConfig(noise_seed=7), sensor_id="lidar_top")
    weather = SimpleNamespace(precipitation=80.0, fog_density=0.0, fog_distance=0.0)
    blueprint = SimpleNamespace(
        set_attribute=lambda name, value: None, has_attribute=lambda name: True
    )
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: blueprint),
        spawn_actor=lambda bp, transform, attach_to: SimpleNamespace(
            listen=lambda cb: None, stop=lambda: None, destroy=lambda: None
        ),
        get_weather=lambda: weather,
    )
    lidar.attach(world, SimpleNamespace(type_id="vehicle.test"))  # type: ignore[arg-type]
    sweeps = []
    for frame in range(start + 1, start + 1 + ticks):
        raw = _ring()
        lidar._receive(SimpleNamespace(frame=frame, raw_data=raw.tobytes()))  # type: ignore[arg-type]  # noqa: SLF001
        sweep = lidar.get_sweep(frame)
        assert sweep is not None
        sweeps.append(sweep)
    return sweeps


def test_the_noise_does_not_depend_on_the_servers_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused server keeps counting frames; the same seed and scenario
    drew different noise 4000 frames on (#2499)."""
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    first = _rainy_sweeps(1000, 3)
    later = _rainy_sweeps(5000, 3)
    for a, b in zip(first, later):
        np.testing.assert_array_equal(a, b)
    # Still a sweep's own noise, not every sweep's.
    assert not np.array_equal(first[0], first[1])


# ---------------------------------------------------------------------------
# 3. The HILS LiDAR's lever arm
# ---------------------------------------------------------------------------


def _vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _turning_motion(
    yaw_deg: float,
    mount: tuple[float, float, float],
    ego_location: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """The sensor motion of an ego turning in place at 30 deg/s, with its yaw
    at *yaw_deg*, a sensor mounted at *mount* (the ego's frame) and not
    rotated on it; the snapshot puts the ego at *ego_location*."""
    from autoware_carla_scenario.hils.bridge import _SensorDevice  # noqa: PLC0415
    from autoware_carla_scenario.hils.sensor_spec import SensorSpec  # noqa: PLC0415

    yaw = math.radians(yaw_deg)
    # The ego-to-world rotation of an ego yawed by *yaw* (CARLA, left-handed),
    # and the world-to-sensor rotation of the sensor on it.
    matrix = np.eye(4)
    matrix[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    inverse = matrix.T
    ego = SimpleNamespace(
        get_velocity=lambda: _vector(0.0, 0.0, 0.0),
        get_angular_velocity=lambda: _vector(0.0, 0.0, 30.0),
        get_transform=lambda: SimpleNamespace(
            location=_vector(*ego_location), get_matrix=lambda: matrix.tolist()
        ),
    )
    data = SimpleNamespace(
        transform=SimpleNamespace(get_inverse_matrix=lambda: inverse.tolist())
    )
    device = _SensorDevice.__new__(_SensorDevice)
    device._ego = ego  # noqa: SLF001
    device.sensor_spec = SensorSpec(
        blueprint="sensor.lidar.ray_cast", transform=(*mount, 0.0, 0.0, 0.0)
    )
    return np.array(device._sensor_motion(data))  # noqa: SLF001


def test_a_sensor_off_the_ego_origin_moves_as_the_ego_turns() -> None:
    """A LiDAR 1.5 m ahead of an ego turning at 30 deg/s moves right at
    0.785 m/s; the ego origin's velocity alone read 0 (#2499)."""
    w = math.radians(30.0)
    motion = _turning_motion(0.0, (1.5, 0.0, 0.0))
    np.testing.assert_allclose(motion[:3], [0.0, 1.5 * w, 0.0], atol=1e-12)
    np.testing.assert_allclose(motion[3:], [0.0, 0.0, w], atol=1e-12)
    # The ego yawed 90 degrees (facing +y): the sensor ahead of it, at +y in
    # the world, moves to the ego's right (-x in the world), its own +y.
    motion = _turning_motion(90.0, (1.5, 0.0, 0.0))
    np.testing.assert_allclose(motion[:3], [0.0, 1.5 * w, 0.0], atol=1e-12)


def test_the_lever_arm_is_the_mount_not_a_snapshot_a_tick_off() -> None:
    """The snapshot may be a tick off the measurement: where it puts the ego
    does not join the lever arm, the sensor's mount alone is it (#2499)."""
    w = math.radians(30.0)
    motion = _turning_motion(0.0, (1.5, 0.0, 0.0), ego_location=(-0.75, 0.0, 0.0))
    np.testing.assert_allclose(motion[:3], [0.0, 1.5 * w, 0.0], atol=1e-12)


def test_a_sensor_on_the_ego_origin_moves_as_the_origin_does() -> None:
    motion = _turning_motion(0.0, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(motion[:3], [0.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 4. A camera's fps off the tick
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fps", "ticks", "text"),
    [
        (20.0, (1, 1), "20 fps"),
        (10.0, (2, 2), "10 fps"),
        (30.0, (1, 1), "20 fps (30 fps set; one frame a tick)"),
        (15.0, (1, 2), "15 fps on average (a frame every 1-2 ticks)"),
    ],
)
def test_the_rate_a_camera_delivers(
    fps: float, ticks: tuple[int, int], text: str
) -> None:
    assert frame_ticks(fps) == ticks
    assert frame_rate_text(fps) == text


@pytest.mark.parametrize(
    "make",
    [DriverCameraConfig, functools.partial(Ros2CameraConfig, topic="/carla/ego/image")],
)
def test_a_camera_off_the_tick_is_warned_of_once_not_refused(
    make: Any, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(carla_camera, "_WARNED_FRAME_RATES", set())
    owner = "camera_2499"
    with caplog.at_level(logging.WARNING, logger=carla_camera.__name__):
        make(logical_id=owner, fps=10.0)
        assert caplog.records == []
        make(logical_id=owner, fps=30.0)
        make(logical_id=owner, fps=30.0)
        make(logical_id=owner, fps=15.0)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert "above one frame a 0.05 s tick" in messages[0]
    assert "20 fps (30 fps set" in messages[0]
    assert "not a whole number" in messages[1]
    assert "15 fps on average" in messages[1]
