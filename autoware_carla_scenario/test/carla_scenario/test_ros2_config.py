"""Tests for the ROS 2 ego's settings and the shipped ``ros2`` config group.

Like ``test_driver_config.py``, these need neither CARLA nor a DDS
implementation: a typo in ``conf/ros2/*.yaml`` fails here, not at the start of a
simulation run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoware_carla_scenario.driver.control import ControlConfig
from autoware_carla_scenario.examples.run import _to_dict
from autoware_carla_scenario.ros2 import (
    CommandSourceConfig,
    Ros2CameraConfig,
    Ros2EgoConfig,
)

_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)


def _composed(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(_CONF.resolve()), version_base=None):
        return compose(config_name="config", overrides=overrides)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_defaults_take_a_trajectory_from_the_planner() -> None:
    config = Ros2EgoConfig()
    (source,) = config.sources
    assert source.kind == "trajectory"
    assert source.topic == "/planning/scenario_planning/trajectory"
    assert config.on_stale == "hold"


def test_steer_accel_needs_both_topics() -> None:
    with pytest.raises(ValueError, match="steering_topic and acceleration_topic"):
        CommandSourceConfig(name="vp", kind="steer_accel", steering_topic="/s")


@pytest.mark.parametrize("kind", ["trajectory", "control"])
def test_a_single_topic_kind_needs_its_topic(kind: str) -> None:
    with pytest.raises(ValueError, match="needs a topic"):
        CommandSourceConfig(name="x", kind=kind)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        CommandSourceConfig(name="x", kind="joystick", topic="/joy")


def test_stale_s_must_be_positive() -> None:
    with pytest.raises(ValueError, match="stale_s"):
        CommandSourceConfig(name="x", kind="control", topic="/c", stale_s=0.0)


def test_source_names_must_be_unique() -> None:
    source = CommandSourceConfig(name="x", kind="control", topic="/c")
    with pytest.raises(ValueError, match="unique"):
        Ros2EgoConfig(sources=(source, source))


def test_unknown_on_stale_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_stale"):
        Ros2EgoConfig(on_stale="coast")


def test_a_source_lists_the_topics_it_reads() -> None:
    pair = CommandSourceConfig(
        name="vp", kind="steer_accel", steering_topic="/s", acceleration_topic="/a"
    )
    assert pair.topics == ("/s", "/a")
    assert CommandSourceConfig(name="c", kind="control", topic="/c").topics == ("/c",)


def test_from_mapping_rejects_an_unknown_key() -> None:
    with pytest.raises(ValueError, match="stale_sec"):
        Ros2EgoConfig.from_mapping(
            {
                "sources": [
                    {"name": "c", "kind": "control", "topic": "/c", "stale_sec": 1}
                ]
            }
        )
    with pytest.raises(ValueError, match="domain"):
        Ros2EgoConfig.from_mapping({"domain": 1})


def test_from_mapping_builds_the_sources_and_ignores_the_control_section() -> None:
    config = Ros2EgoConfig.from_mapping(
        {
            "domain_id": 3,
            "control": {"wheelbase_m": 2.85},
            "sources": [
                {"name": "c", "kind": "control", "topic": "/c", "priority": 5},
            ],
        }
    )
    assert config.domain_id == 3
    (source,) = config.sources
    assert (source.name, source.priority) == ("c", 5)


def test_a_camera_needs_a_topic() -> None:
    with pytest.raises(ValueError, match="needs a topic"):
        Ros2CameraConfig(logical_id="main_cam")


def test_camera_topics_must_be_unique() -> None:
    camera = Ros2CameraConfig(topic="/carla/hero/main_cam/image")
    with pytest.raises(ValueError, match="camera topics must be unique"):
        Ros2EgoConfig(cameras=(camera, camera))


def test_from_mapping_builds_the_cameras() -> None:
    config = Ros2EgoConfig.from_mapping(
        {
            "cameras": [
                {
                    "logical_id": "main_cam",
                    "topic": "/carla/hero/main_cam/image",
                    "frame_id": "main_cam",
                    "image_width": 1280,
                    "image_height": 720,
                    "fov": 60.0,
                    "fps": 10.0,
                }
            ]
        }
    )
    (camera,) = config.cameras
    assert camera.topic == "/carla/hero/main_cam/image"
    assert camera.frame_id == "main_cam"
    sensor = camera.to_sensor_config()
    assert (sensor.image_width, sensor.image_height, sensor.fov, sensor.fps) == (
        1280,
        720,
        60.0,
        10.0,
    )


def test_from_mapping_rejects_an_unknown_camera_key() -> None:
    with pytest.raises(ValueError, match="encoding"):
        Ros2EgoConfig.from_mapping(
            {"cameras": [{"topic": "/carla/hero/main_cam/image", "encoding": "rgb8"}]}
        )


# ---------------------------------------------------------------------------
# The shipped config group
# ---------------------------------------------------------------------------


def test_shipped_default_builds_the_config() -> None:
    cfg = _composed([])
    mapping = _to_dict(cfg.ros2)
    assert Ros2EgoConfig.from_mapping(mapping) == Ros2EgoConfig()
    assert ControlConfig.from_mapping(mapping["control"]) == ControlConfig()
    # Selecting the group selects nothing: the ego is still the autopilot's.
    assert cfg.ego.entity == "autopilot"


def test_visionpilot_safety_island_preset_replaces_the_bench_bridge() -> None:
    cfg = _composed(["ros2=visionpilot_safety_island"])
    assert cfg.ego.entity == "ros2"
    assert cfg.ego.vehicle_type == "vehicle.lincoln.mkz"
    assert cfg.server.max_tick_rate_hz == pytest.approx(20.0)

    mapping = _to_dict(cfg.ros2)
    config = Ros2EgoConfig.from_mapping(mapping)
    assert config.domain_id == 1
    sources = {source.name: source for source in config.sources}
    assert set(sources) == {"visionpilot", "safety_island"}
    assert sources["visionpilot"].topics == (
        "/vehicle/steering_cmd",
        "/vehicle/throttle_cmd",
    )
    assert sources["safety_island"].topic == "/control/trajectory_follower/control_cmd"
    assert sources["safety_island"].priority > sources["visionpilot"].priority
    # The state topics are the default's: what the bridge published.
    assert config.speed_topic == "/vehicle/speed"
    assert config.kinematic_state_topic == "/localization/kinematic_state"


def test_run_builds_the_ros2_entity() -> None:
    from autoware_carla_scenario.entity.ros2_entity import Ros2EgoEntity
    from autoware_carla_scenario.examples.run import build_ego_entity

    entity = build_ego_entity(_composed(["ros2=visionpilot_safety_island"]))
    assert isinstance(entity, Ros2EgoEntity)
    assert entity.config.domain_id == 1
    assert entity.use_autopilot is False
    # Nothing has joined the domain yet: that waits for the scenario to start.
    assert entity.transport is None


def test_run_warns_when_the_run_is_not_paced(caplog: pytest.LogCaptureFixture) -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    build_ego_entity(_composed(["ego.entity=ros2"]))
    assert "max_tick_rate_hz" in caplog.text

    caplog.clear()
    build_ego_entity(_composed(["ros2=visionpilot_safety_island"]))
    assert "max_tick_rate_hz" not in caplog.text


def test_run_refuses_ros2_without_its_group() -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import build_ego_entity

    with pytest.raises(ValueError, match="'ros2' config group"):
        build_ego_entity(OmegaConf.create({"ego": {"entity": "ros2"}}))
