"""The ego and driver configs refuse a NaN (or an infinity) that would hang a run.

Hydra reads ``=nan`` and ``=inf`` as floats, and a NaN fails every comparison: a
NaN ready or boot timeout never fires, so the run waits for the ego forever, and
a NaN staleness or gain silently brakes the ego for the whole run.  Every such
value is refused when the config is built, naming its key.
"""

from __future__ import annotations

import dataclasses
import logging
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
        ("liveness_timeout_s", _NAN),
        ("liveness_timeout_s", _INF),
        ("liveness_timeout_s", -1.0),
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


def test_timeout_s_is_deprecated(caplog: pytest.LogCaptureFixture) -> None:
    """It never had an effect (#2049): accepted, so a config that sets it still
    runs, and said to be ignored, naming the node parameter that does it."""
    logger = "autoware_carla_scenario.autoware_bridge.base"
    with caplog.at_level(logging.WARNING, logger=logger):
        AutowareBridgeConfig()
        AutowareBridgeConfig(liveness_timeout_s=0.0)
    assert caplog.records == []

    with caplog.at_level(logging.WARNING, logger=logger):
        AutowareBridgeConfig(timeout_s=10.0)
    (record,) = caplog.records
    assert "deprecated" in record.getMessage()
    assert "rpc_timeout_s" in record.getMessage()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("address", None),
        ("address", ""),
        ("address", "  "),
        ("address", 50052),
        ("ready_timeout_ticks", "1200"),
        ("ready_timeout_ticks", True),
        ("ready_timeout_ticks", None),
        ("timeout_s", "60"),
        ("timeout_s", True),
        ("boot_timeout_s", "300"),
        ("boot_timeout_s", True),
        ("boot_timeout_s", None),
        ("boot_tick_period_s", False),
        ("liveness_timeout_s", False),
        ("liveness_timeout_s", "30"),
        ("liveness_timeout_s", None),
        ("base_link_offset_m", True),
        ("base_link_offset_m", "-1.4"),
    ],
)
def test_the_autoware_bridge_config_refuses_what_is_no_number_naming_it(
    key: str, value: Any
) -> None:
    """#2217: a quoted number crashed minutes into the run, a bool was 0 or 1
    (``liveness_timeout_s=False`` turned liveness off), and None a TypeError."""
    with pytest.raises(ValueError, match=f"^{key} must be") as refused:
        AutowareBridgeConfig(**{key: value})
    assert repr(value) in str(refused.value)


@pytest.mark.parametrize(
    "address",
    [
        "localhost",  # gRPC binds port 443
        "localhost:abc",
        "localhost:99999",  # gRPC binds 99999 modulo 65536
        "localhost:65536",
        "localhost:-1",
        "localhost:",
        ":50052",
        ":",
        "localhost:²",  # a digit, but not an ASCII one
        "::1",  # gRPC binds ::1 at port 443
        "::1:50052",  # an unbracketed IPv6 host: gRPC binds nothing
        "[::1:50052",
    ],
)
def test_the_autoware_bridge_config_refuses_an_address_that_is_no_host_port(
    address: str,
) -> None:
    """#2275: refused naming the key, not when the bridge starts, or never."""
    with pytest.raises(ValueError, match="^address must be a host:port") as refused:
        AutowareBridgeConfig(address=address)
    assert repr(address) in str(refused.value)


@pytest.mark.parametrize(
    "address",
    ["localhost:50052", "localhost:0", "0.0.0.0:65535", "[::1]:50052", "[::]:0"],
)
def test_the_autoware_bridge_config_takes_a_host_port(address: str) -> None:
    assert AutowareBridgeConfig(address=address).address == address


def test_the_autoware_bridge_config_keeps_ints_as_ints() -> None:
    config = AutowareBridgeConfig(boot_timeout_s=300, liveness_timeout_s=0)
    assert type(config.ready_timeout_ticks) is int
    assert type(config.boot_timeout_s) is int
    assert type(config.liveness_timeout_s) is int


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("autoware.ready_timeout_ticks='1200'", "ready_timeout_ticks must be"),
        ("autoware.ready_timeout_ticks=true", "ready_timeout_ticks must be"),
        ("autoware.liveness_timeout_s=false", "liveness_timeout_s must be"),
        ("+autoware.base_link_offset_m=true", "base_link_offset_m must be"),
        ("autoware.address=''", "address must be"),
    ],
)
def test_an_autoware_setting_that_is_no_number_is_refused_naming_the_key(
    override: str, message: str
) -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    with pytest.raises(ValueError, match=f"^autoware\\.{message}"):
        build_ego_entity(_composed(["ego.entity=autoware", override]))


def test_a_null_autoware_setting_is_its_default() -> None:
    """#2217: absent or null is the default, not a TypeError."""
    from autoware_carla_scenario.examples.run import _autoware_bridge_config

    config = _autoware_bridge_config(
        _composed(
            [
                "ego.entity=autoware",
                "autoware.address=null",
                "autoware.ready_timeout_ticks=null",
                "autoware.boot_timeout_s=null",
                "autoware.liveness_timeout_s=null",
                "+autoware.boot_tick_period_s=null",
                "+autoware.timeout_s=null",
                "+autoware.base_link_offset_m=null",
            ]
        )
    )
    defaults = AutowareBridgeConfig()
    assert config.address == defaults.address
    assert config.ready_timeout_ticks == defaults.ready_timeout_ticks
    assert config.boot_timeout_s == defaults.boot_timeout_s
    assert config.liveness_timeout_s == defaults.liveness_timeout_s
    assert config.boot_tick_period_s == defaults.boot_tick_period_s
    assert config.timeout_s is None
    assert config.base_link_offset_m is None


@pytest.mark.parametrize(
    "override",
    [
        "autoware.ready_timeout_ticks=nan",
        "autoware.boot_timeout_s=nan",
        "autoware.boot_timeout_s=inf",
        "+autoware.boot_tick_period_s=nan",
        "+autoware.timeout_s=nan",
        "autoware.liveness_timeout_s=nan",
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
    with pytest.raises(
        ValueError, match=f"^hils\\.{key} must be a positive, finite number of seconds"
    ):
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
        ("+driver.control.kp=nan", "driver.control.kp"),
        (
            "driver.control.max_steer_angle_deg=nan",
            "driver.control.max_steer_angle_deg",
        ),
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
