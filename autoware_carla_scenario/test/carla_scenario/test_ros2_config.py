"""Tests for the ROS 2 ego's settings and the shipped ``ros2`` config group.

Like ``test_driver_config.py``, these need neither CARLA nor a DDS
implementation: a typo in ``conf/ros2/*.yaml`` fails here, not at the start of a
simulation run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

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
    # Tuples, not the lists the check reads them as (docs/typecheck.md).
    assert type(pair.topics) is tuple
    assert type(Ros2EgoConfig().sources) is tuple


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


_CAMERA_TOPIC = "/carla/hero/main_cam/image"


@pytest.mark.parametrize("fov", [0.0, -30.0, 180.0, 270.0])
def test_a_camera_refuses_a_fov_outside_0_to_180(fov: float) -> None:
    # CARLA's pinhole only spans (0, 180) degrees; outside it the camera would
    # only fail as it attaches, after the ego spawned (#1399).
    with pytest.raises(ValueError, match=r"fov must be in \(0, 180\) degrees"):
        Ros2CameraConfig(topic=_CAMERA_TOPIC, fov=fov)


def test_a_camera_refuses_a_nan_fov() -> None:
    with pytest.raises(ValueError, match="fov must be finite"):
        Ros2CameraConfig(topic=_CAMERA_TOPIC, fov=math.nan)


@pytest.mark.parametrize("fov", [0.001, 90.0, 179.9])
def test_a_camera_accepts_a_fov_inside_0_to_180(fov: float) -> None:
    assert Ros2CameraConfig(topic=_CAMERA_TOPIC, fov=fov).fov == fov


@pytest.mark.parametrize("key", ["image_width", "image_height"])
@pytest.mark.parametrize("value", [0, -1])
def test_a_camera_refuses_a_bad_image_size(key: str, value: Any) -> None:
    with pytest.raises(
        ValueError, match="image size and fps must be positive and finite"
    ):
        Ros2CameraConfig(topic=_CAMERA_TOPIC, **{key: value})


@pytest.mark.parametrize("key", ["image_width", "image_height"])
@pytest.mark.parametrize("value", [960.0, 960.5, math.nan, math.inf, True, "960"])
def test_a_camera_refuses_a_size_that_is_no_int(key: str, value: Any) -> None:
    """CARLA's image_size_x is an integer: 960.0 reached it as "960.0", as a
    driver camera's refused it only once the session started (#1472)."""
    match = rf"^camera 'camera': {key} must be a whole number"
    with pytest.raises(ValueError, match=match):
        Ros2CameraConfig(topic=_CAMERA_TOPIC, **{key: value})
    with pytest.raises(ValueError, match=match):
        Ros2EgoConfig.from_mapping({"cameras": [{"topic": _CAMERA_TOPIC, key: value}]})


def test_a_camera_hands_carla_an_int_size() -> None:
    sensor = Ros2CameraConfig(
        topic=_CAMERA_TOPIC, image_width=1280, image_height=720
    ).to_sensor_config()
    assert str(sensor.image_width) == "1280"
    assert str(sensor.image_height) == "720"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("fov", 0.0, r"fov must be in \(0, 180\) degrees"),
        ("fov", -30.0, r"fov must be in \(0, 180\) degrees"),
        ("fov", 180.0, r"fov must be in \(0, 180\) degrees"),
        ("fov", 270.0, r"fov must be in \(0, 180\) degrees"),
        ("fov", math.nan, "fov must be finite"),
        ("image_width", math.nan, "image_width must be a whole number"),
        ("image_height", math.nan, "image_height must be a whole number"),
        ("image_width", math.inf, "image_width must be a whole number"),
        ("image_height", 0, "image size and fps must be positive and finite"),
    ],
)
def test_from_mapping_refuses_a_camera_out_of_bounds(
    key: str, value: Any, match: str
) -> None:
    # The mapping is not coerced, so a YAML `.nan` reaches an int field as a
    # float; it is refused as the config is read.
    with pytest.raises(ValueError, match=match):
        Ros2EgoConfig.from_mapping({"cameras": [{"topic": _CAMERA_TOPIC, key: value}]})


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


def test_the_documented_ros2_run_composes() -> None:
    # docs/ros2.md's command line, and the one the "not paced" warning advises
    # (#1438): server.max_tick_rate_hz must be a key of the shipped server.
    cfg = _composed(["ego.entity=ros2", "server.max_tick_rate_hz=20"])
    assert cfg.ego.entity == "ros2"
    assert cfg.server.max_tick_rate_hz == 20


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

    caplog.clear()
    build_ego_entity(_composed(["ego.entity=ros2", "server.max_tick_rate_hz=20"]))
    assert "max_tick_rate_hz" not in caplog.text


@pytest.mark.parametrize("rate", ["0", "-20", "nan"])
def test_run_warns_when_the_rate_paces_nothing(
    rate: str, caplog: pytest.LogCaptureFixture
) -> None:
    # A run refuses such a rate first (check_run_config); built without that
    # check, the runner reads it as uncapped, so the warning says so (#1439).
    from autoware_carla_scenario.examples.run import build_ego_entity

    build_ego_entity(_composed(["ego.entity=ros2", f"server.max_tick_rate_hz={rate}"]))
    assert "max_tick_rate_hz" in caplog.text


def test_run_refuses_ros2_without_its_group() -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import build_ego_entity

    with pytest.raises(ValueError, match="'ros2' config group"):
        build_ego_entity(OmegaConf.create({"ego": {"entity": "ros2"}}))


def test_pair_window_s_defaults_to_measured_and_takes_null_or_seconds() -> None:
    base = {
        "name": "vp",
        "kind": "steer_accel",
        "steering_topic": "/s",
        "acceleration_topic": "/a",
    }
    assert CommandSourceConfig.from_mapping(base).pair_window_s is None
    assert (
        CommandSourceConfig.from_mapping({**base, "pair_window_s": None}).pair_window_s
        is None
    )
    assert (
        CommandSourceConfig.from_mapping({**base, "pair_window_s": 0.03}).pair_window_s
        == 0.03
    )


# ---------------------------------------------------------------------------
# Value types and the domain (#1468): refused as the config is built, naming
# the key, not once the ego has spawned and joined the domain
# ---------------------------------------------------------------------------


def _source(**overrides: Any) -> dict[str, Any]:
    return {"name": "planner", "kind": "trajectory", "topic": "/t", **overrides}


@pytest.mark.parametrize("domain_id", [0, 1, 101, 232])
def test_domain_id_takes_a_ros_domain_id(domain_id: int) -> None:
    assert Ros2EgoConfig(domain_id=domain_id).domain_id == domain_id


@pytest.mark.parametrize("domain_id", [-1, 233, 300, True, False, "7", 1.0, None])
def test_domain_id_refuses_what_is_no_ros_domain_id(domain_id: Any) -> None:
    with pytest.raises(ValueError, match=r"^domain_id must be a ROS domain id"):
        Ros2EgoConfig.from_mapping({"domain_id": domain_id})


def test_domain_id_shares_the_docker_launcher_s_bound() -> None:
    from autoware_carla_scenario.autoware_stack import docker
    from autoware_carla_scenario.constants import MAX_ROS_DOMAIN_ID

    assert MAX_ROS_DOMAIN_ID == 232
    assert docker.MAX_ROS_DOMAIN_ID is MAX_ROS_DOMAIN_ID


@pytest.mark.parametrize("priority", [-5, 0, 10])
def test_priority_takes_an_int(priority: int) -> None:
    assert CommandSourceConfig.from_mapping(_source(priority=priority)).priority == (
        priority
    )


@pytest.mark.parametrize("priority", ["10", True, 10.0, math.nan, None])
def test_priority_refuses_what_is_no_int(priority: Any) -> None:
    with pytest.raises(
        ValueError,
        match=r"^command source 'planner': priority must be a whole number",
    ):
        CommandSourceConfig.from_mapping(_source(priority=priority))


@pytest.mark.parametrize(
    "key", ["stale_s", "steering_sign", "speed_horizon_s", "jerk_mps3", "pair_window_s"]
)
@pytest.mark.parametrize("value", ["0.3", True, [0.3]])
def test_a_source_refuses_a_number_that_is_no_number(key: str, value: Any) -> None:
    with pytest.raises(
        ValueError, match=rf"^command source 'planner': {key} must be a number"
    ):
        CommandSourceConfig.from_mapping(_source(**{key: value}))


def test_a_source_takes_an_int_for_a_float() -> None:
    source = CommandSourceConfig.from_mapping(
        _source(stale_s=1, steering_sign=-1, speed_horizon_s=2, jerk_mps3=5)
    )
    assert source.stale_us == 1_000_000
    assert source.steering_sign == -1


@pytest.mark.parametrize(
    "key",
    [
        "fov",
        "fps",
        "position_x",
        "position_y",
        "position_z",
        "roll",
        "pitch",
        "yaw",
    ],
)
@pytest.mark.parametrize("value", ["1", True])
def test_a_camera_refuses_a_number_that_is_no_number(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=rf"^camera 'camera': {key} must be a number"):
        Ros2CameraConfig.from_mapping({"topic": "/c", key: value})


@pytest.mark.parametrize("value", ["0.5", True])
def test_rear_axle_offset_m_refuses_what_is_no_number(value: Any) -> None:
    with pytest.raises(ValueError, match=r"^rear_axle_offset_m must be a number"):
        Ros2EgoConfig.from_mapping({"rear_axle_offset_m": value})


def test_shipped_ros2_configs_build() -> None:
    for overrides in (["ego.entity=ros2"], ["ros2=visionpilot_safety_island"]):
        Ros2EgoConfig.from_mapping(_to_dict(_composed(overrides).ros2))


def test_the_run_check_refuses_a_bad_ros2_setting_up_front() -> None:
    from autoware_carla_scenario.examples import run

    run.check_run_config(_composed(["ros2=visionpilot_safety_island"]))

    cfg = _composed(["ego.entity=ros2", "ros2.domain_id=233"])
    with pytest.raises(run.RunConfigError, match=r"^ros2: domain_id must be"):
        run.check_run_config(cfg)

    cfg = _composed(["ros2=visionpilot_safety_island"])
    cfg.ros2.sources[1].priority = "10"
    with pytest.raises(
        run.RunConfigError,
        match=r"^ros2: command source 'safety_island': priority must be",
    ):
        run.check_run_config(cfg)

    cfg = _composed(["ego.entity=ros2", "+ros2.control.no_such_gain=1"])
    with pytest.raises(run.RunConfigError, match=r"^ros2: "):
        run.check_run_config(cfg)

    # Another ego reads no ros2 settings.
    run.check_run_config(_composed(["ros2.domain_id=233"]))
