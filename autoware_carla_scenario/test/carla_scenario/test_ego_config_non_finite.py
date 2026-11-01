"""The ego and driver configs refuse a NaN (or an infinity) that would hang a run.

Hydra reads ``=nan`` and ``=inf`` as floats, and a NaN fails every comparison: a
NaN ready or boot timeout never fires, so the run waits for the ego forever, and
a NaN staleness or gain silently brakes the ego for the whole run.  Every such
value is refused when the config is built, naming its key.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from autoware_carla_scenario.autoware_bridge import AutowareBridgeConfig
from autoware_carla_scenario.driver.base import DriverClientConfig
from autoware_carla_scenario.driver.control import ControlConfig
from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig
from autoware_carla_scenario.ros2 import (
    CommandSourceConfig,
    Ros2CameraConfig,
    Ros2EgoConfig,
)

_NAN = math.nan
_INF = math.inf

_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)


def _composed(overrides: List[str]) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(_CONF), version_base=None):
        return compose(config_name="config", overrides=overrides)


# -- AutowareBridgeConfig ------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ready_timeout_ticks", _NAN),
        ("ready_timeout_ticks", _INF),
        ("ready_timeout_ticks", 0),
        ("ready_timeout_ticks", -5),
        ("ready_timeout_ticks", 12.5),
        ("boot_timeout_s", _NAN),
        ("boot_timeout_s", _INF),
        ("boot_timeout_s", 0.0),
        ("timeout_s", _NAN),
        ("timeout_s", _INF),
        ("timeout_s", -1.0),
        ("boot_tick_period_s", _NAN),
        ("boot_tick_period_s", _INF),
        ("boot_tick_period_s", -0.05),
        ("base_link_offset_m", _NAN),
        ("base_link_offset_m", -_INF),
    ],
)
def test_the_autoware_bridge_config_refuses(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"^{key} must be"):
        AutowareBridgeConfig(**{key: value})  # type: ignore[arg-type]


def test_the_autoware_bridge_config_keeps_every_valid_value() -> None:
    AutowareBridgeConfig()
    # A whole-number float (what a YAML ``1200.0`` gives) and the edges that
    # mean something today: no wait between booting ticks, a derived offset.
    config = AutowareBridgeConfig(
        ready_timeout_ticks=1200.0,  # type: ignore[arg-type]
        boot_tick_period_s=0.0,
        base_link_offset_m=None,
        timeout_s=60,  # type: ignore[arg-type]
    )
    assert config.ready_timeout_ticks == 1200
    assert AutowareBridgeConfig(base_link_offset_m=-1.425).base_link_offset_m == -1.425


@pytest.mark.parametrize(
    "override",
    [
        "autoware.ready_timeout_ticks=nan",
        "autoware.boot_timeout_s=nan",
        "autoware.boot_timeout_s=inf",
        "+autoware.boot_tick_period_s=nan",
        "+autoware.timeout_s=nan",
        "+autoware.base_link_offset_m=nan",
    ],
)
def test_an_autoware_ego_with_a_nan_setting_is_refused(override: str) -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    key = override.lstrip("+").split("=")[0]
    with pytest.raises(ValueError, match=f"^{key} must be"):
        build_ego_entity(_composed(["ego.entity=autoware", override]))


def test_the_autoware_ego_builds_from_the_example_config() -> None:
    from autoware_carla_scenario.entity import AutowareEgoEntity
    from autoware_carla_scenario.examples.run import build_ego_entity

    entity = build_ego_entity(_composed(["ego.entity=autoware"]))
    assert isinstance(entity, AutowareEgoEntity)


@pytest.mark.parametrize(
    ("bridge", "key"),
    [
        ("{ready_timeout_ticks:nan}", "ready_timeout_ticks"),
        ("{boot_timeout_s:nan}", "boot_timeout_s"),
        ("{timeout_s:inf}", "timeout_s"),
    ],
)
def test_a_policy_ego_with_a_nan_bridge_setting_is_refused(
    bridge: str, key: str
) -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    with pytest.raises(ValueError, match=f"^driver\\.bridge\\.{key} must be"):
        build_ego_entity(
            _composed(["ego.entity=carla_driver", f"driver.bridge={bridge}"])
        )


# -- HilsEgoConfig -------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["connect_timeout_s", "ready_timeout_s", "end_timeout_s", "stale_s"]
)
@pytest.mark.parametrize("value", [_NAN, _INF, 0.0, -1.0])
def test_the_hils_config_refuses(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"^hils\\.{key} must be positive and finite"):
        HilsEgoConfig.from_mapping({key: value})


@pytest.mark.parametrize(
    "key", ["connect_timeout_s", "ready_timeout_s", "end_timeout_s", "stale_s"]
)
def test_a_hils_ego_with_a_nan_setting_is_refused(key: str) -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    with pytest.raises(ValueError, match=f"^hils\\.{key} must be"):
        build_ego_entity(_composed(["ego.entity=hils", f"hils.{key}=nan"]))


def test_the_hils_ego_builds_from_the_example_config() -> None:
    from autoware_carla_scenario.entity.hils_entity import HilsEgoEntity
    from autoware_carla_scenario.examples.run import build_ego_entity

    assert HilsEgoConfig() == HilsEgoConfig.from_mapping({})
    entity = build_ego_entity(_composed(["ego.entity=hils"]))
    assert isinstance(entity, HilsEgoEntity)


# -- DriverClientConfig and ControlConfig (ego.entity=carla_driver) ---------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("timeout_s", _NAN),
        ("timeout_s", _INF),
        ("timeout_s", 0.0),
        ("policy_timestep_s", _NAN),
        ("policy_timestep_s", _INF),
        ("policy_timestep_s", 0.0),
        ("warmup_s", _NAN),
        ("warmup_s", _INF),
        ("route_horizon_m", _NAN),
        ("route_horizon_m", _INF),
        ("route_resolution_m", _NAN),
        ("route_resolution_m", _INF),
        ("t4_scene_history_s", _NAN),
        ("t4_scene_history_s", _INF),
        ("rear_axle_offset_m", _NAN),
        ("t4_scene_wheel_base_m", _INF),
        ("traffic_light_sight_distance_m", _NAN),
        ("actor_horizon_m", _NAN),
    ],
)
def test_the_driver_config_refuses(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"^driver\\.{key} must be"):
        DriverClientConfig(**{key: value})  # type: ignore[arg-type]


def test_the_driver_config_keeps_every_valid_value() -> None:
    DriverClientConfig()
    DriverClientConfig(
        warmup_s=0.0,
        traffic_light_sight_distance_m=0.0,
        actor_horizon_m=_INF,
        rear_axle_offset_m=-1.4,
        t4_scene_wheel_base_m=None,
    )


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ("driver.timeout_s=nan", "driver.timeout_s"),
        ("driver.policy_timestep_s=nan", "driver.policy_timestep_s"),
        ("driver.warmup_s=inf", "driver.warmup_s"),
        ("+driver.control.kp=nan", "kp"),
        ("driver.control.max_steer_angle_deg=nan", "max_steer_angle_rad"),
    ],
)
def test_a_policy_ego_with_a_nan_setting_is_refused(override: str, key: str) -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    with pytest.raises(ValueError, match=key):
        build_ego_entity(_composed(["ego.entity=carla_driver", override]))


def test_the_control_config_refuses_a_nan() -> None:
    with pytest.raises(ValueError, match="lookahead_gain_s"):
        ControlConfig.from_mapping({"lookahead_gain_s": _NAN})
    assert ControlConfig.from_mapping({}) == ControlConfig()


def test_the_control_config_keeps_every_example_value() -> None:
    import yaml  # type: ignore[import-untyped]

    for name in ("driver/default.yaml", "driver/vision_pilot.yaml"):
        document: Dict[str, Any] = yaml.safe_load((_CONF / name).read_text())
        driver = document.get("driver", document)
        ControlConfig.from_mapping(driver.get("control") or {})


# -- Ros2EgoConfig ---------------------------------------------------------------


def _source(**values: Any) -> CommandSourceConfig:
    return CommandSourceConfig(
        name="planner", kind="trajectory", topic="/planning/trajectory", **values
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("stale_s", _NAN),
        ("stale_s", _INF),
        ("pair_window_s", _NAN),
        ("pair_window_s", _INF),
        ("steering_sign", _NAN),
        ("speed_horizon_s", _NAN),
        ("speed_horizon_s", _INF),
        ("jerk_mps3", _NAN),
    ],
)
def test_a_command_source_refuses(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"'planner': {key} must be"):
        _source(**{key: value})


@pytest.mark.parametrize(
    "key",
    ["fov", "fps", "position_x", "position_y", "position_z", "roll", "pitch", "yaw"],
)
def test_a_ros2_camera_refuses_a_nan(key: str) -> None:
    with pytest.raises(ValueError, match=f"{key} must be finite"):
        Ros2CameraConfig(topic="/camera/image", **{key: _NAN})  # type: ignore[arg-type]


def test_the_ros2_config_refuses_a_nan_rear_axle_offset() -> None:
    with pytest.raises(ValueError, match="rear_axle_offset_m must be finite"):
        Ros2EgoConfig(rear_axle_offset_m=_NAN)
    assert Ros2EgoConfig(rear_axle_offset_m=-1.4).rear_axle_offset_m == -1.4
    assert _source(pair_window_s=0.0).pair_window_s == 0.0


@pytest.mark.parametrize("profile", ["default", "visionpilot_safety_island"])
def test_the_ros2_ego_builds_from_the_example_configs(profile: str) -> None:
    from autoware_carla_scenario.entity.ros2_entity import Ros2EgoEntity
    from autoware_carla_scenario.examples.run import build_ego_entity

    entity = build_ego_entity(_composed(["ego.entity=ros2", f"ros2={profile}"]))
    assert isinstance(entity, Ros2EgoEntity)


def test_a_ros2_ego_with_a_nan_stale_s_is_refused() -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    cfg = _composed(["ego.entity=ros2"])
    sources = [dict(source) for source in cfg.ros2.sources]
    sources[0]["stale_s"] = _NAN
    cfg.ros2.sources = sources
    with pytest.raises(ValueError, match="stale_s must be positive and finite"):
        build_ego_entity(cfg)


def test_every_config_default_is_finite() -> None:
    """The defaults themselves pass, and none of them is a NaN."""
    for config in (
        AutowareBridgeConfig(),
        HilsEgoConfig(),
        DriverClientConfig(),
        ControlConfig(),
        Ros2EgoConfig(),
        _source(),
    ):
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if isinstance(value, float):
                assert not math.isnan(value), field.name
