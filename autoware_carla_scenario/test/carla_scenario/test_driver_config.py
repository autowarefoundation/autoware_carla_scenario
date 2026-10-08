"""Tests that the shipped driver YAML maps onto the runtime configs.

These deliberately avoid importing CARLA or Hydra: the mapping lives on the config
classes so that a typo in ``conf/driver/default.yaml`` is caught by a cheap unit test
rather than at the start of a simulation run.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from autoware_carla_scenario.driver.base import DriverCameraConfig, DriverClientConfig
from autoware_carla_scenario.driver.control import ControlConfig


_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)


def _shipped_driver_config() -> dict:
    """Return the ``driver`` node of the shipped default config group."""
    document = yaml.safe_load((_CONF / "driver" / "default.yaml").read_text())
    return document["driver"]


# ---------------------------------------------------------------------------
# The shipped YAML
# ---------------------------------------------------------------------------


def test_shipped_yaml_builds_a_client_config() -> None:
    config = DriverClientConfig.from_mapping(_shipped_driver_config())
    assert config.address == "localhost:50051"
    assert config.policy_timestep_s == pytest.approx(0.1)
    assert config.rear_axle_offset_m is None
    assert len(config.cameras) == 1
    assert config.cameras[0].logical_id == "camera_front_wide_120fov"
    assert isinstance(config.cameras[0], DriverCameraConfig)


def test_shipped_yaml_builds_control_gains() -> None:
    control = ControlConfig.from_mapping(_shipped_driver_config()["control"])
    assert control.max_steer_angle_rad == pytest.approx(math.radians(70.0))
    assert control.speed_kp == pytest.approx(0.6)


def test_shipped_policy_timestep_is_a_multiple_of_the_tick() -> None:
    """The runtime only queries the policy on tick boundaries."""
    config = DriverClientConfig.from_mapping(_shipped_driver_config())
    ticks = config.policy_timestep_s / 0.05
    assert ticks == pytest.approx(round(ticks))
    assert ticks >= 1


def test_ego_default_yaml_selects_the_autopilot_entity() -> None:
    document = yaml.safe_load((_CONF / "ego" / "default.yaml").read_text())
    assert document["ego"]["entity"] == "autopilot"


def test_driver_group_is_in_the_defaults_list() -> None:
    document = yaml.safe_load((_CONF / "config.yaml").read_text())
    assert {"driver": "default"} in document["defaults"]


# ---------------------------------------------------------------------------
# Mapping behaviour
# ---------------------------------------------------------------------------


def test_partial_mapping_keeps_the_defaults() -> None:
    config = DriverClientConfig.from_mapping({"address": "policy:9000"})
    assert config.address == "policy:9000"
    assert config.timeout_s == DriverClientConfig().timeout_s
    assert config.cameras == DriverClientConfig().cameras


def test_unknown_client_key_is_rejected() -> None:
    """A typo must not silently leave the default in place."""
    with pytest.raises(ValueError, match="Unknown DriverClientConfig key"):
        DriverClientConfig.from_mapping({"adress": "typo:1"})


def test_unknown_camera_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown DriverCameraConfig key"):
        DriverClientConfig.from_mapping({"cameras": [{"logical": "front"}]})


def test_unknown_control_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown ControlConfig key"):
        ControlConfig.from_mapping({"speed_kp": 1.0, "speed_kx": 2.0})


def test_control_mapping_accepts_radians_directly() -> None:
    control = ControlConfig.from_mapping({"max_steer_angle_rad": 1.0})
    assert control.max_steer_angle_rad == pytest.approx(1.0)


def test_client_mapping_ignores_the_control_section() -> None:
    """``control`` lives in the same YAML node but configures the follower."""
    config = DriverClientConfig.from_mapping(
        {"address": "policy:9000", "control": {"speed_kp": 1.0}}
    )
    assert config.address == "policy:9000"


def test_multiple_cameras_are_supported() -> None:
    config = DriverClientConfig.from_mapping(
        {
            "cameras": [
                {"logical_id": "front", "fov": 120.0},
                {"logical_id": "rear", "yaw": 180.0},
            ]
        }
    )
    assert [camera.logical_id for camera in config.cameras] == ["front", "rear"]
    assert config.cameras[1].yaw == pytest.approx(180.0)


def test_camera_maps_onto_a_carla_sensor_config() -> None:
    camera = DriverCameraConfig(logical_id="front", image_width=640, fov=90.0)
    sensor = camera.to_sensor_config()
    assert sensor.image_width == 640
    assert sensor.fov == pytest.approx(90.0)
    # 640 px across 90 deg.
    assert sensor.fx == pytest.approx(320.0, rel=1e-6)


# ---------------------------------------------------------------------------
# The VisionPilot preset (driver=vision_pilot)
# ---------------------------------------------------------------------------


def _composed(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(_CONF.resolve()), version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_vision_pilot_preset_sets_the_rig_and_hands_over_the_ego() -> None:
    cfg = _composed(["driver=vision_pilot"])
    assert cfg.ego.entity == "carla_driver"
    assert cfg.ego.vehicle_type == "vehicle.lincoln.mkz"

    config = DriverClientConfig.from_mapping(cfg.driver)
    (camera,) = config.cameras
    assert camera.logical_id == "camera_front_narrow_50fov"
    assert (camera.image_width, camera.image_height, camera.fov) == (1920, 1280, 50.0)
    assert (camera.position_x, camera.position_y, camera.position_z) == (
        1.544,
        0.0243,
        2.116,
    )
    assert (camera.roll, camera.pitch, camera.yaw) == (-0.10, -0.11, -0.23)
    assert config.policy_timestep_s == pytest.approx(0.1)
    assert ControlConfig.from_mapping(cfg.driver.control).wheelbase_m == pytest.approx(
        2.85
    )


def test_vision_pilot_preset_keeps_the_other_driver_defaults() -> None:
    """It extends driver/default rather than replacing it."""
    preset, default = _composed(["driver=vision_pilot"]).driver, _composed([]).driver
    for key in ("address", "route_horizon_m", "send_renderer_data", "image_quality"):
        assert preset[key] == default[key]
    assert preset.control.speed_kp == default.control.speed_kp
