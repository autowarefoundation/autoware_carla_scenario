"""Tests that the shipped driver YAML maps onto the runtime configs.

These deliberately avoid importing CARLA or Hydra: the mapping lives on the config
classes so that a typo in ``conf/driver/default.yaml`` is caught by a cheap unit test
rather than at the start of a simulation run.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from autoware_carla_scenario.carla_release import (
    default_max_steer_angle_deg,
    default_steer_exponent,
)
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
    # null in the YAML: the server's (carla_release).
    assert control.max_steer_angle_rad == pytest.approx(
        math.radians(default_max_steer_angle_deg())
    )
    assert control.steer_exponent == pytest.approx(default_steer_exponent())
    assert control.yaw_rate_ki == pytest.approx(3.0)
    assert control.longitudinal_controller == "pid"
    assert control.delay_compensation_time == pytest.approx(0.17)
    assert control.stopped_acc == pytest.approx(-3.4)


def test_shipped_policy_timestep_is_a_multiple_of_the_tick() -> None:
    """The runtime only queries the policy on tick boundaries."""
    config = DriverClientConfig.from_mapping(_shipped_driver_config())
    ticks = config.policy_timestep_s / 0.05
    assert ticks == pytest.approx(round(ticks))
    assert ticks >= 1


@pytest.mark.parametrize("seconds", [0.12, 0.07, math.inf, math.nan])
def test_policy_timestep_between_ticks_is_refused(seconds: float) -> None:
    """The policy is asked on tick boundaries only: 0.12 would run every 0.15 s
    while asking for plans 0.12 s ahead, so it is refused up front (#952)."""
    with pytest.raises(ValueError, match=r"driver\.policy_timestep_s"):
        DriverClientConfig(policy_timestep_s=seconds)


@pytest.mark.parametrize("seconds", [0.05, 0.1, 0.15, 0.2, 0.05 * 3, 0.1 + 0.2])
def test_policy_timestep_on_a_tick_is_accepted(seconds: float) -> None:
    """Multiples of the tick pass, floats not exact in binary (0.15) included."""
    assert DriverClientConfig(policy_timestep_s=seconds).policy_timestep_s == seconds


def test_from_mapping_refuses_a_policy_timestep_between_ticks() -> None:
    mapping = dict(_shipped_driver_config(), policy_timestep_s=0.07)
    with pytest.raises(ValueError, match=r"driver\.policy_timestep_s=0\.07"):
        DriverClientConfig.from_mapping(mapping)


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


def test_control_mapping_accepts_every_angle_in_degrees() -> None:
    control = ControlConfig.from_mapping({"yaw_rate_trim_limit_deg": 10.0})
    assert control.yaw_rate_trim_limit_rad == pytest.approx(math.radians(10.0))


def test_control_spec_mirrors_the_control_config() -> None:
    """Every DriverControlSpec field reaches ControlConfig, with the same default."""
    import dataclasses

    from autoware_carla_scenario.scenario_config import DriverControlSpec

    assert (
        ControlConfig.from_mapping(dataclasses.asdict(DriverControlSpec()))
        == ControlConfig()
    )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        # Bools and strings (#1662): read as 1, or failing mid-run.
        ({"wheelbase_m": True}, r"^wheelbase_m must be a number, got True$"),
        ({"kp": "0.5"}, r"^kp must be a number, got '0\.5'$"),
        ({"kp": [1.0]}, r"^kp must be a number, got \[1\.0\]$"),
        # Not finite, an int too large for a float included.
        ({"kp": math.inf}, r"^kp must be finite, got inf$"),
        ({"min_acc": -math.inf}, r"^min_acc must be finite, got -inf$"),
        ({"lookahead_gain_s": math.nan}, r"^lookahead_gain_s must be finite"),
        ({"ki": 10**400}, r"^ki must be finite, got 1000"),
        ({"ki": 10**5000}, r"^ki must be finite, got an integer of about 10\*\*"),
        # Divisors.
        ({"steer_exponent": 0}, r"^steer_exponent must be positive, got 0\.0$"),
        ({"max_steer_angle_rad": -0.5}, r"^max_steer_angle_rad must be positive"),
        ({"max_steer_angle_deg": 0}, r"^max_steer_angle_deg must be positive"),
        # An alias shows the degrees given, not the radians they make.
        (
            {"max_steer_angle_deg": -10},
            r"^max_steer_angle_deg must be positive, got -10$",
        ),
        (
            {"max_steer_angle_deg": -12.5},
            r"^max_steer_angle_deg must be positive, got -12\.5$",
        ),
        ({"wheelbase_m": 0.0}, r"^wheelbase_m must be positive"),
        ({"min_lookahead_m": 0.0}, r"^min_lookahead_m must be positive"),
        ({"velocity_lookahead_time": -1.0}, r"^velocity_lookahead_time must be"),
        # The lookahead's bounds.
        (
            {"min_lookahead_m": 5.0, "max_lookahead_m": 4.0},
            r"^max_lookahead_m must be at least min_lookahead_m \(5\.0\), got 4\.0$",
        ),
        # The *_deg aliases name themselves.
        ({"max_steer_angle_deg": "56"}, r"^max_steer_angle_deg must be a number"),
        ({"yaw_rate_trim_limit_deg": True}, r"^yaw_rate_trim_limit_deg must be a"),
        ({"max_steer_angle_deg": math.inf}, r"^max_steer_angle_deg must be finite"),
        ({"yaw_rate_trim_limit_deg": [10]}, r"^yaw_rate_trim_limit_deg must be a"),
        # Switches, the window and the controller.
        ({"enable_smooth_stop": "false"}, r"^enable_smooth_stop must be true or"),
        ({"enable_smooth_stop": 0}, r"^enable_smooth_stop must be true or false"),
        ({"velocity_smoothing_window": 2.5}, r"^velocity_smoothing_window must be"),
        ({"velocity_smoothing_window": True}, r"^velocity_smoothing_window must be"),
        ({"longitudinal_controller": "mpc"}, r"^longitudinal_controller must be one"),
        ({"speed_kx": 1.0}, r"^speed_kx: Unknown ControlConfig key"),
    ],
)
def test_control_mapping_refuses_a_bad_value_by_its_key(
    mapping: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ControlConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    "key",
    [
        "kp",
        "longitudinal_controller",
        "velocity_smoothing_window",
        "enable_smooth_stop",
        "max_steer_angle_rad",
        "max_steer_angle_deg",
    ],
)
def test_a_null_control_setting_is_its_default(key: str) -> None:
    """Null is the default, as an absent key is, for a field as for an
    alias (#2241)."""
    assert ControlConfig.from_mapping({key: None}) == ControlConfig()


@pytest.mark.parametrize("angle", ["max_steer_angle", "yaw_rate_trim_limit"])
def test_an_angle_set_in_degrees_and_in_radians_is_refused(angle: str) -> None:
    """Neither may silently win (#2241): the degrees used to override the
    radians."""
    with pytest.raises(
        ValueError,
        match=rf"^{angle}_deg and {angle}_rad set the same angle, .*"
        r"set one of them to null, got 30\.0 and 0\.1$",
    ):
        ControlConfig.from_mapping({f"{angle}_deg": 30.0, f"{angle}_rad": 0.1})
    # One of them null is the other's.
    control = ControlConfig.from_mapping({f"{angle}_deg": None, f"{angle}_rad": 0.1})
    assert getattr(control, f"{angle}_rad") == pytest.approx(0.1)
    control = ControlConfig.from_mapping({f"{angle}_deg": 30.0, f"{angle}_rad": None})
    assert getattr(control, f"{angle}_rad") == pytest.approx(math.radians(30.0))


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("driver.control.max_steer_angle_rad=0.6", 0.6),
        ("driver.control.max_steer_angle_deg=30", math.radians(30.0)),
    ],
)
def test_a_steer_angle_override_is_read_with_the_shipped_config(
    override: str, expected: float
) -> None:
    """driver/default.yaml leaves both of the full lock's keys null, so an
    override of either one is the angle the run steers with (#2241)."""
    from autoware_carla_scenario.examples import run

    cfg = _composed(["ego.entity=carla_driver", override])
    control = run._control_config(cfg.driver, "driver")
    assert control.max_steer_angle_rad == pytest.approx(expected)


@pytest.mark.parametrize("section", ["driver", "ros2"])
def test_the_run_heads_the_angle_conflict_with_its_section(section: str) -> None:
    """from_mapping alone refuses the conflict; the run heads it with the
    section, as every other control error (#2241)."""
    from autoware_carla_scenario.examples import run

    with pytest.raises(
        ValueError,
        match=rf"^{section}\.control\.max_steer_angle_deg and "
        r"max_steer_angle_rad set the same angle, .*"
        r"set one of them to null, got 30\.0 and 0\.6$",
    ):
        run._control_config(
            {"control": {"max_steer_angle_deg": 30.0, "max_steer_angle_rad": 0.6}},
            section,
        )
    cfg = _composed(
        [
            "ego.entity=carla_driver",
            "driver.control.max_steer_angle_deg=30",
            "driver.control.max_steer_angle_rad=0.6",
        ]
    )
    with pytest.raises(
        run.RunConfigError, match=r"^driver\.control\.max_steer_angle_deg and "
    ):
        run.check_run_config(cfg)


def test_control_mapping_reads_an_int_as_a_float() -> None:
    control = ControlConfig.from_mapping(
        {"kp": 2, "wheelbase_m": 3, "max_steer_angle_deg": 45, "min_acc": -4}
    )
    assert (control.kp, control.wheelbase_m, control.min_acc) == (2.0, 3.0, -4.0)
    assert all(isinstance(v, float) for v in (control.kp, control.wheelbase_m))
    assert control.max_steer_angle_rad == pytest.approx(math.radians(45.0))
    assert ControlConfig.from_mapping({"kp": np.float32(0.5)}).kp == 0.5
    assert ControlConfig.from_mapping({"velocity_smoothing_window": 4.0}) == (
        ControlConfig(velocity_smoothing_window=4)
    )
    # Equal bounds pin the lookahead; a null alias keeps the default.
    ControlConfig.from_mapping({"min_lookahead_m": 6, "max_lookahead_m": 6})
    assert ControlConfig.from_mapping({"max_steer_angle_deg": None}) == ControlConfig()


def test_every_shipped_control_section_loads() -> None:
    """Each ``control:`` block of the shipped config groups still loads."""
    for name in (
        "driver/default.yaml",
        "driver/vision_pilot.yaml",
        "ros2/default.yaml",
    ):
        document = yaml.safe_load((_CONF / name).read_text())
        section = document.get(name.split("/")[0], document)
        ControlConfig.from_mapping(section.get("control") or {})
    for overrides in (
        [],
        ["driver=vision_pilot"],
        ["ego.entity=ros2"],
        ["ros2=visionpilot_safety_island"],
    ):
        cfg = _composed(overrides)
        for section in (cfg.driver, cfg.ros2):
            ControlConfig.from_mapping(section.get("control") or {})


@pytest.mark.parametrize("section", ["driver", "ros2"])
@pytest.mark.parametrize(
    ("control", "shown"), [("abc", r"'abc'"), ([1.0], r"\[1\.0\]"), (0, "0")]
)
def test_the_run_refuses_a_control_section_that_is_no_mapping(
    section: str, control: Any, shown: str
) -> None:
    from autoware_carla_scenario.examples import run

    with pytest.raises(
        ValueError, match=rf"^{section}\.control must be a mapping, got {shown}$"
    ):
        run._control_config({"control": control}, section)
    # Null, or no entry, is the default gains.
    assert run._control_config({"control": None}, section) == ControlConfig()
    assert run._control_config({}, section) == ControlConfig()


def test_the_run_refuses_a_driver_control_string_up_front() -> None:
    from autoware_carla_scenario.examples import run

    cfg = _composed(["ego.entity=carla_driver", "driver.control=abc"])
    with pytest.raises(
        run.RunConfigError, match=r"^driver\.control must be a mapping, got 'abc'$"
    ):
        run.check_run_config(cfg)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            ["ego.entity=carla_driver", "driver.control.steer_exponent=0"],
            r"^driver\.control\.steer_exponent must be positive",
        ),
        (
            ["ego.entity=carla_driver", "+driver.control.kp='0.5'"],
            r"^driver\.control\.kp must be a number, got '0\.5'",
        ),
        (
            ["ego.entity=carla_driver", "driver.control.max_steer_angle_deg=0"],
            r"^driver\.control\.max_steer_angle_deg must be positive",
        ),
        (
            ["ego.entity=ros2", "+ros2.control.wheelbase_m=true"],
            r"^ros2\.control\.wheelbase_m must be a number, got True",
        ),
        (
            ["ego.entity=ros2", "+ros2.control.max_lookahead_m=1"],
            r"^ros2\.control\.max_lookahead_m must be at least",
        ),
    ],
)
def test_the_run_names_a_refused_control_key_by_its_section(
    overrides: list[str], message: str
) -> None:
    from autoware_carla_scenario.examples import run

    cfg = _composed(overrides)
    with pytest.raises(run.RunConfigError, match=message):
        run.check_run_config(cfg)
    with pytest.raises(ValueError, match=message):
        run.build_ego_entity(cfg)


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


def test_camera_logical_ids_must_be_unique() -> None:
    """Two cameras under one id would both attach, the second's mount winning (#1391)."""
    with pytest.raises(ValueError, match="camera logical ids must be unique"):
        DriverClientConfig.from_mapping(
            {
                "cameras": [
                    {"logical_id": "front"},
                    {"logical_id": "front", "yaw": 180.0},
                ]
            }
        )


@pytest.mark.parametrize(
    "values",
    [
        {"fps": 0.0},
        {"fps": -10.0},
        {"fps": float("nan")},
        {"fps": float("inf")},
        {"image_width": 0},
        {"image_height": -1},
    ],
)
def test_camera_refuses_a_non_positive_or_non_finite_size_or_fps(
    values: dict,
) -> None:
    """Each would only fail as the camera attaches, after the ego spawned (#1391)."""
    with pytest.raises(ValueError, match="image size and fps must be positive"):
        DriverClientConfig.from_mapping(
            {"cameras": [{"logical_id": "front", **values}]}
        )


@pytest.mark.parametrize("key", ["image_width", "image_height"])
@pytest.mark.parametrize("value", [960, 960.0, 9.6e2, np.int64(960), np.int32(960)])
def test_camera_keeps_a_whole_number_size_as_an_int(key: str, value: Any) -> None:
    """The egodriver contract's resolution is an integer field, which refused
    even 960.0 with a TypeError as the session started, after the ego spawned
    (#1472): a whole number is kept as the int it is."""
    for camera in (
        DriverCameraConfig(logical_id="front", **{key: value}),
        DriverClientConfig.from_mapping(
            {"cameras": [{"logical_id": "front", key: value}]}
        ).cameras[0],
    ):
        size = getattr(camera, key)
        assert type(size) is int
        assert size == 960
        assert type(getattr(camera.to_sensor_config(), key)) is int


@pytest.mark.parametrize("key", ["image_width", "image_height"])
@pytest.mark.parametrize("value", [960.5, True, "960", None])
def test_camera_refuses_a_size_that_is_no_whole_number(key: str, value: Any) -> None:
    """Not truncated to 960 or 1, as the config is read (#1472)."""
    match = rf"camera 'front': {key} must be a whole number"
    with pytest.raises(ValueError, match=f"^{match}"):
        DriverCameraConfig(logical_id="front", **{key: value})
    entry = {"cameras": [{"logical_id": "front", key: value}]}
    if value is None:
        # From a mapping, null is the default (#2228).
        camera = DriverClientConfig.from_mapping(entry).cameras[0]
        assert getattr(camera, key) == getattr(DriverCameraConfig(), key)
        return
    with pytest.raises(
        ValueError, match=rf"^driver\.cameras\[0\] \('front'\): {match}"
    ):
        DriverClientConfig.from_mapping(entry)


@pytest.mark.parametrize("key", ["image_width", "image_height"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_camera_refuses_a_size_that_is_not_finite(key: str, value: Any) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^driver\.cameras\[0\] \('front'\): camera 'front': {key} must be finite",
    ):
        DriverClientConfig.from_mapping(
            {"cameras": [{"logical_id": "front", key: value}]}
        )


def test_shipped_driver_cameras_have_int_sizes() -> None:
    for preset in ("default", "vision_pilot"):
        config = DriverClientConfig.from_mapping(_composed([f"driver={preset}"]).driver)
        for camera in config.cameras:
            assert type(camera.image_width) is int
            assert type(camera.image_height) is int


@pytest.mark.parametrize("fov", [0.0, -30.0, 180.0, 270.0, float("nan"), float("inf")])
def test_camera_refuses_a_fov_a_pinhole_cannot_have(fov: float) -> None:
    with pytest.raises(ValueError, match=r"fov must be in \(0, 180\)"):
        DriverCameraConfig(logical_id="front", fov=fov)


@pytest.mark.parametrize(
    "field", ["position_x", "position_y", "position_z", "roll", "pitch", "yaw"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_camera_refuses_a_non_finite_mount(field: str, value: float) -> None:
    changes: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        DriverCameraConfig(logical_id="front", **changes)


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
    assert (
        preset.control.longitudinal_controller
        == default.control.longitudinal_controller
    )
    assert preset.control.stopped_acc == default.control.stopped_acc


@pytest.mark.parametrize(
    "overrides",
    [
        {"t4_scene": True},
        {"t4_scene": True, "map_dir": "/maps", "send_renderer_data": False},
    ],
    ids=["no map_dir", "no renderer data"],
)
def test_t4_scenes_need_the_map_set_and_renderer_data(overrides: dict) -> None:
    with pytest.raises(ValueError, match="driver.t4_scene"):
        DriverClientConfig(**overrides)


def test_t4_scenes_are_off_by_default_and_reach_three_seconds_back() -> None:
    config = DriverClientConfig()
    assert config.t4_scene is False and config.t4_scene_history_s == 3.0
    assert DriverClientConfig(t4_scene=True, map_dir="/maps").t4_scene
    with pytest.raises(ValueError, match="t4_scene_history_s"):
        DriverClientConfig(t4_scene_history_s=-1.0)


def test_t4_scenes_take_the_scenarios_goal_and_the_runs_map_by_default() -> None:
    config = DriverClientConfig()
    assert (config.t4_scene_goal, config.t4_scene_map) == ("scenario", "run")
    assert DriverClientConfig(t4_scene_goal="road_ahead", t4_scene_map="map_set")
    with pytest.raises(ValueError, match="t4_scene_goal"):
        DriverClientConfig(t4_scene_goal="anywhere")
    with pytest.raises(ValueError, match="t4_scene_map"):
        DriverClientConfig(t4_scene_map="roadgen")


# ``driver.random_seed`` is a whole number, not negative, as the run's seed:
# it used to be kept as given and handed to the policy through int(), so 1.9
# and True seeded the session as 1, and "abc" failed only then (#1445).
@pytest.mark.parametrize(
    "seed", [1.9, True, False, "abc", "7", -3, math.nan, math.inf, 2**31]
)
def test_a_random_seed_that_is_no_seed_is_refused(seed: Any) -> None:
    with pytest.raises(ValueError, match=r"^driver\.random_seed must "):
        DriverClientConfig.from_mapping({"random_seed": seed})
    with pytest.raises(ValueError, match=r"^driver\.random_seed must "):
        DriverClientConfig(random_seed=seed)


@pytest.mark.parametrize(
    ("seed", "expected"),
    [(None, None), (0, 0), (7, 7), (2.0, 2), (2**31 - 1, 2**31 - 1)],
)
def test_a_whole_random_seed_is_kept(seed: Any, expected: Any) -> None:
    from autoware_carla_scenario.seed import resolve_seed

    config = DriverClientConfig.from_mapping({"random_seed": seed})
    if expected is None:
        # No seed stays no seed, so the session draws the run's seed.
        assert config.random_seed is None
    else:
        assert config.random_seed is not None
        assert resolve_seed(config.random_seed) == expected


def test_a_carla_driver_run_is_refused_for_it_before_anything_is_built() -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    cfg = _shipped(["ego.entity=carla_driver", "driver.random_seed=1.9"])
    with pytest.raises(run.RunConfigError, match=r"^driver\.random_seed must "):
        run.check_run_config(cfg)
    # Above 2**31 - 1, what CARLA and SUMO take (#1560).
    cfg = _shipped(["ego.entity=carla_driver", "driver.random_seed=2147483648"])
    with pytest.raises(
        run.RunConfigError, match=r"^driver\.random_seed must be at most"
    ):
        run.check_run_config(cfg)
    # Another ego reads no driver settings.
    run.check_run_config(_shipped(["driver.random_seed=1.9"]))


# ---------------------------------------------------------------------------
# A number field holding a bool or a string (#1506)
# ---------------------------------------------------------------------------

_DRIVER_NUMBERS = (
    "timeout_s",
    "policy_timestep_s",
    "warmup_s",
    "route_horizon_m",
    "route_resolution_m",
    "rear_axle_offset_m",
    "traffic_light_sight_distance_m",
    "actor_horizon_m",
    "t4_scene_history_s",
    "t4_scene_wheel_base_m",
    "image_quality",
)


@pytest.mark.parametrize("value", [True, False, "0.1", "90", None, [1.0]])
@pytest.mark.parametrize("key", _DRIVER_NUMBERS)
def test_a_driver_number_that_is_no_number_is_refused(key: str, value: Any) -> None:
    if value is None and key in ("rear_axle_offset_m", "t4_scene_wheel_base_m"):
        pytest.skip("None is this key's own value")
    with pytest.raises(ValueError, match=rf"^driver\.{key} must be "):
        DriverClientConfig(**{key: value})
    with pytest.raises(ValueError, match=rf"^driver\.{key} must be "):
        DriverClientConfig.from_mapping({key: value})


@pytest.mark.parametrize("quality", [0, 101, -5, 90.5, float("nan")])
def test_an_image_quality_out_of_range_or_fractional_is_refused(
    quality: Any,
) -> None:
    with pytest.raises(ValueError, match=r"^driver\.image_quality must be "):
        DriverClientConfig(image_quality=quality)


@pytest.mark.parametrize(
    ("quality", "expected"), [(90, 90), (90.0, 90), (1, 1), (100.0, 100)]
)
def test_a_whole_image_quality_is_kept_as_an_int(quality: Any, expected: int) -> None:
    kept = DriverClientConfig(image_quality=quality).image_quality
    assert kept == expected
    assert type(kept) is int


@pytest.mark.parametrize(
    ("key", "value"), [("timeout_s", 30), ("policy_timestep_s", np.float64(0.2))]
)
def test_an_int_or_a_numpy_float_is_a_number(key: str, value: Any) -> None:
    assert getattr(DriverClientConfig(**{key: value}), key) == value


@pytest.mark.parametrize("value", [True, "1.0"])
@pytest.mark.parametrize(
    "key",
    ["fov", "fps", "position_x", "position_y", "position_z", "roll", "pitch", "yaw"],
)
def test_a_camera_number_that_is_no_number_is_refused(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=rf"{key} must be a number"):
        DriverCameraConfig(**{key: value})


@pytest.mark.parametrize("value", [True, "1.0"])
@pytest.mark.parametrize(
    "key",
    ["range_m", "upper_fov_deg", "dropoff_general_rate", "weather_noise", "yaw"],
)
def test_a_lidar_number_that_is_no_number_is_refused(key: str, value: Any) -> None:
    from autoware_carla_scenario.driver.base import DriverLidarConfig

    with pytest.raises(ValueError, match=rf"^LiDAR {key} must be a number"):
        DriverLidarConfig.from_mapping({key: value})


@pytest.mark.parametrize(
    "key",
    [
        "principal_point",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "linear_cde",
        "max_angle",
    ],
)
def test_a_lens_number_that_is_no_number_is_refused(key: str) -> None:
    from autoware_carla_scenario.driver.base import DriverCameraLens

    lens: dict[str, Any] = {
        "principal_point": [480.0, 302.0],
        "pixeldist_to_angle_poly": [0.0, 0.002],
        "angle_to_pixeldist_poly": [0.0, 500.0],
        "max_angle": 1.5,
        "linear_cde": [1.0, 0.0, 0.0],
    }
    DriverCameraLens.from_mapping(lens)
    lens[key] = True if key == "max_angle" else [True] * len(lens[key])
    with pytest.raises(ValueError, match=rf"^lens\.{key} must be a number"):
        DriverCameraLens.from_mapping(lens)
    with pytest.raises(ValueError, match=rf"^lens\.{key} must be a number"):
        DriverCameraLens(**lens)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "key",
    [
        "principal_point",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "linear_cde",
        "max_angle",
    ],
)
def test_a_lens_number_that_is_not_finite_is_refused(key: str, value: float) -> None:
    """A NaN or an infinity in a lens is refused naming lens.<key> (#2277)."""
    from autoware_carla_scenario.driver.base import DriverCameraLens

    lens: dict[str, Any] = {
        "principal_point": [480.0, 302.0],
        "pixeldist_to_angle_poly": [0.0, 0.002],
        "angle_to_pixeldist_poly": [0.0, 500.0],
        "max_angle": 1.5,
        "linear_cde": [1.0, 0.0, 0.0],
    }
    if key == "max_angle":
        lens[key] = value
    else:
        lens[key] = [*lens[key][:-1], value]
    with pytest.raises(ValueError, match=rf"^lens\.{key} must be finite, got "):
        DriverCameraLens.from_mapping(lens)
    with pytest.raises(ValueError, match=r"^driver\.cameras\[0\] .*lens\."):
        DriverClientConfig.from_mapping({"cameras": [{"lens": lens}]})


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("warmup_s", -3.0, "must not be negative, got -3.0"),
        ("traffic_light_sight_distance_m", -1.0, "must not be negative, got -1.0"),
        ("actor_horizon_m", -1.0, "must not be negative, got -1.0"),
        ("actor_horizon_m", -math.inf, "must not be negative, got -inf"),
        ("route_horizon_m", 0.0, "must be positive, got 0.0"),
        ("route_horizon_m", -5.0, "must be positive, got -5.0"),
        ("route_resolution_m", 0.0, "must be positive, got 0.0"),
        ("route_resolution_m", -2, "must be positive, got -2"),
        ("t4_scene_wheel_base_m", -2.0, "must be positive, or null"),
        ("t4_scene_wheel_base_m", 0.0, "must be positive, or null"),
    ],
)
def test_a_driver_distance_or_duration_out_of_its_domain_is_refused(
    key: str, value: float, message: str
) -> None:
    """Each length and duration is refused out of its domain, naming the key
    (#2277), as t4_scene_history_s is."""
    with pytest.raises(ValueError, match=rf"^driver\.{key} {re.escape(message)}"):
        DriverClientConfig.from_mapping({key: value})


def test_a_driver_distance_at_its_documented_zero_is_kept() -> None:
    """0 keeps its meaning where it has one: no run-up, CARLA's own light
    trigger, no road user reported; an infinite horizon is every one."""
    config = DriverClientConfig.from_mapping(
        {
            "warmup_s": 0.0,
            "traffic_light_sight_distance_m": 0.0,
            "actor_horizon_m": 0,
            "t4_scene_wheel_base_m": 2.7,
        }
    )
    assert config.warmup_s == 0.0 and config.actor_horizon_m == 0
    assert config.traffic_light_sight_distance_m == 0.0
    assert DriverClientConfig(actor_horizon_m=math.inf).actor_horizon_m == math.inf
    assert DriverClientConfig() == DriverClientConfig.from_mapping({})


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"stop_brake": 2.0}, r"^stop_brake must be in \[0, 1\], got 2\.0$"),
        ({"stop_brake": -0.5}, r"^stop_brake must be in \[0, 1\], got -0\.5$"),
        ({"throttle_deadband": 1.5}, r"^throttle_deadband must be in \[0, 1\]"),
        ({"throttle_deadband": -0.1}, r"^throttle_deadband must be in \[0, 1\]"),
        ({"accel_lowpass_gain": 1.5}, r"^accel_lowpass_gain must be in \[0, 1\]"),
        ({"lpf_vel_error_gain": -0.1}, r"^lpf_vel_error_gain must be in \[0, 1\]"),
        ({"lpf_acc_error_gain": 2}, r"^lpf_acc_error_gain must be in \[0, 1\]"),
        ({"max_steer_rate": 0.0}, r"^max_steer_rate must be positive, got 0\.0$"),
        ({"max_steer_rate": -1.0}, r"^max_steer_rate must be positive"),
        ({"lookahead_gain_s": -1.0}, r"^lookahead_gain_s must not be negative"),
    ],
)
def test_control_mapping_refuses_pedals_gains_and_rates_out_of_range(
    mapping: dict, message: str
) -> None:
    """A brake of 2 or -0.5, a throttle floor above full throttle, a gain past
    1 and a steering rate limit of 0 are refused (#2277)."""
    with pytest.raises(ValueError, match=message):
        ControlConfig.from_mapping(mapping)


def test_control_mapping_keeps_the_ends_of_the_pedal_and_gain_ranges() -> None:
    config = ControlConfig.from_mapping(
        {
            "stop_brake": 1.0,
            "throttle_deadband": 0.0,
            "accel_lowpass_gain": 0.0,
            "lpf_vel_error_gain": 1,
            "lookahead_gain_s": 0.0,
        }
    )
    assert config.stop_brake == 1.0 and config.throttle_deadband == 0.0
    assert config.lpf_vel_error_gain == 1.0


@pytest.mark.parametrize(
    "override", ["driver.policy_timestep_s=true", "driver.image_quality=true"]
)
def test_a_carla_driver_run_refuses_a_bool_number(override: str) -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    key = override.split("=")[0].replace(".", r"\.")
    cfg = _shipped(["ego.entity=carla_driver", override])
    with pytest.raises(run.RunConfigError, match=rf"^{key} must be "):
        run.check_run_config(cfg)


@pytest.mark.parametrize(
    ("rate", "refused"),
    [
        ("45", True),
        ("-0.1", True),
        ("1.01", True),
        ("0", False),
        ("0.45", False),
        ("1", False),
    ],
)
def test_a_carla_driver_run_checks_the_lidar_dropoff_rate(
    rate: str, refused: bool
) -> None:
    """45 used to empty every sweep without a word (#1767)."""
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    lidars = f"driver.lidars=[{{logical_id:top,dropoff_general_rate:{rate}}}]"
    cfg = _shipped(["ego.entity=carla_driver", lidars])
    if refused:
        match = (
            r"^driver\.lidars\[0\] \('top'\): "
            r"LiDAR dropoff_general_rate must be in \[0, 1\], got "
        )
        with pytest.raises(run.RunConfigError, match=match):
            run.check_run_config(cfg)
    else:
        run.check_run_config(cfg)


@pytest.mark.parametrize("preset", ["default", "vision_pilot"])
def test_every_shipped_driver_preset_passes_the_run_check(preset: str) -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    run.check_run_config(_shipped(["ego.entity=carla_driver", f"driver={preset}"]))


_DRIVER_SWITCHES = (
    "send_ground_truth",
    "send_renderer_data",
    "send_actor_ground_truth",
    "t4_scene",
)


def _switch_config(name: str, value: Any) -> DriverClientConfig:
    """A config with switch *name* set to *value* (t4_scene with its map set)."""
    extra: dict[str, Any] = {"map_dir": "/maps"} if name == "t4_scene" else {}
    return DriverClientConfig.from_mapping({name: value, **extra})


@pytest.mark.parametrize("name", _DRIVER_SWITCHES)
@pytest.mark.parametrize("value", ["false", "0", " Off ", 0, False])
def test_a_driver_switch_spelled_false_is_false(name: str, value: Any) -> None:
    # "false" from a quoted override used to be bool()-true (#1760).
    assert getattr(_switch_config(name, value), name) is False


@pytest.mark.parametrize("name", _DRIVER_SWITCHES)
@pytest.mark.parametrize("value", ["true", "Yes", "1", 1, True])
def test_a_driver_switch_spelled_true_is_true(name: str, value: Any) -> None:
    assert getattr(_switch_config(name, value), name) is True


@pytest.mark.parametrize("name", _DRIVER_SWITCHES)
@pytest.mark.parametrize("value", ["ture", 2, None, 0.0, []])
def test_a_driver_switch_of_another_value_is_refused_by_its_key(
    name: str, value: Any
) -> None:
    with pytest.raises(ValueError, match=f"driver.{name} must be true or false"):
        _switch_config(name, value)


def test_the_driver_switches_keep_their_defaults() -> None:
    config = DriverClientConfig()
    assert config.send_ground_truth is False
    assert config.send_renderer_data is True
    assert config.send_actor_ground_truth is True
    assert config.t4_scene is False


def test_t4_scene_spelled_false_needs_no_map_set() -> None:
    # Read as true, "false" asked for driver.map_dir (#1760).
    config = DriverClientConfig.from_mapping({"t4_scene": "false"})
    assert config.t4_scene is False and config.map_dir is None


def test_a_mistyped_t4_scene_is_refused_as_a_switch_not_for_its_map_set() -> None:
    with pytest.raises(ValueError, match="driver.t4_scene must be true or false"):
        DriverClientConfig.from_mapping({"t4_scene": "ture"})


def test_send_renderer_data_spelled_false_is_seen_by_the_t4_scene_check() -> None:
    with pytest.raises(ValueError, match="needs driver.send_renderer_data"):
        DriverClientConfig.from_mapping(
            {"t4_scene": "true", "map_dir": "/maps", "send_renderer_data": "false"}
        )


# ---------------------------------------------------------------------------
# Null settings and logical ids of driver.cameras[] / driver.lidars[] (#2228)
# ---------------------------------------------------------------------------


def test_a_null_camera_setting_is_its_default() -> None:
    camera = DriverCameraConfig.from_mapping(
        {"logical_id": None, "fov": None, "image_width": None, "lens": None}
    )
    assert camera == DriverCameraConfig()


def test_a_null_lidar_setting_is_its_default() -> None:
    from autoware_carla_scenario.driver.base import DriverLidarConfig

    lidar = DriverLidarConfig.from_mapping(
        {"logical_id": None, "range_m": None, "channels": None, "noise_seed": None}
    )
    assert lidar == DriverLidarConfig()
    assert lidar.logical_id == "lidar_top"


def test_a_null_sensor_logical_id_in_the_driver_is_the_default() -> None:
    """The policy was offered sensors addressed by None, which it never found."""
    config = DriverClientConfig.from_mapping(
        {"cameras": [{"logical_id": None}], "lidars": [{"logical_id": None}]}
    )
    assert config.cameras[0].logical_id == "camera_front_wide_120fov"
    assert config.lidars[0].logical_id == "lidar_top"


@pytest.mark.parametrize("logical_id", ["", 3, True, ["a"]])
def test_a_camera_logical_id_that_is_no_name_is_refused(logical_id: Any) -> None:
    with pytest.raises(ValueError, match="camera logical_id must be a non-empty"):
        DriverCameraConfig(logical_id=logical_id)
    with pytest.raises(
        ValueError,
        match=r"^driver\.cameras\[1\] \(.*\): camera logical_id must be a non-empty",
    ):
        DriverClientConfig.from_mapping(
            {"cameras": [{"logical_id": "front"}, {"logical_id": logical_id}]}
        )


@pytest.mark.parametrize("logical_id", ["", 3, True])
def test_a_lidar_logical_id_that_is_no_name_is_refused(logical_id: Any) -> None:
    from autoware_carla_scenario.driver.base import DriverLidarConfig

    with pytest.raises(ValueError, match="LiDAR logical_id must be a non-empty"):
        DriverLidarConfig(logical_id=logical_id)
    with pytest.raises(
        ValueError,
        match=r"^driver\.lidars\[0\] \(.*\): LiDAR logical_id must be a non-empty",
    ):
        DriverClientConfig.from_mapping({"lidars": [{"logical_id": logical_id}]})


def test_a_lidar_error_names_its_entry_and_logical_id() -> None:
    with pytest.raises(
        ValueError,
        match=r"^driver\.lidars\[1\] \('rear'\): LiDAR channels must be positive",
    ):
        DriverClientConfig.from_mapping(
            {"lidars": [{"logical_id": "top"}, {"logical_id": "rear", "channels": 0}]}
        )
    # Given no id, the entry is named by the default it takes.
    with pytest.raises(
        ValueError, match=r"^driver\.lidars\[0\] \('lidar_top'\): LiDAR range_m"
    ):
        DriverClientConfig.from_mapping({"lidars": [{"range_m": -1.0}]})
    with pytest.raises(
        ValueError, match=r"^driver\.lidars\[0\] \('top'\): Unknown DriverLidarConfig"
    ):
        DriverClientConfig.from_mapping({"lidars": [{"logical_id": "top", "rang": 1}]})


def test_a_camera_error_names_its_entry() -> None:
    with pytest.raises(
        ValueError,
        match=r"^driver\.cameras\[0\] \('front'\): camera 'front': fov must be",
    ):
        DriverClientConfig.from_mapping(
            {"cameras": [{"logical_id": "front", "fov": 180.0}]}
        )


@pytest.mark.parametrize("key", ["cameras", "lidars"])
@pytest.mark.parametrize("value", [False, "lidar_top", [3]])
def test_a_sensor_list_that_is_no_list_of_mappings_is_refused(
    key: str, value: Any
) -> None:
    with pytest.raises(ValueError, match=rf"^driver\.{key}(\[0\])? must be a"):
        DriverClientConfig.from_mapping({key: value})


@pytest.mark.parametrize("address", [None, "", 50051])
def test_a_driver_address_that_is_no_address_is_refused(address: Any) -> None:
    """driver.address: null used to be taken as None, dialled as nothing."""
    with pytest.raises(ValueError, match="^driver.address must be a non-empty string"):
        DriverClientConfig.from_mapping({"address": address})


# ---------------------------------------------------------------------------
# policy, map_dir and map_formats, checked as they are read (#2241)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["policy", "map_dir"])
@pytest.mark.parametrize(
    ("value", "shown"), [("", "''"), ("  ", "'  '"), (False, "False"), (0, "0")]
)
def test_a_policy_or_map_dir_that_is_no_name_is_refused(
    key: str, value: Any, shown: str
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^driver\.{key} must be a non-empty string, or null, got {shown}$",
    ):
        DriverClientConfig.from_mapping({key: value})


@pytest.mark.parametrize("key", ["policy", "map_dir"])
def test_a_null_policy_or_map_dir_keeps_its_meaning(key: str) -> None:
    assert getattr(DriverClientConfig.from_mapping({key: None}), key) is None


@pytest.mark.parametrize(
    ("formats", "message"),
    [
        # A bare string would be read as its letters.
        ("lanelet2", r"^driver\.map_formats must be a list, not str 'lanelet2'"),
        (
            ["lanelet3"],
            r"^driver\.map_formats\[0\] must be one of \[.*\], got 'lanelet3'$",
        ),
        (
            ["lanelet2", "osm2"],
            r"^driver\.map_formats\[1\] must be one of \[.*\], got 'osm2'$",
        ),
        ([None], r"^driver\.map_formats\[0\] must be a string or a number"),
        (
            ["lanelet2", "t4"],
            r"^driver\.map_formats names t4, .* set driver\.t4_scene=true, or "
            r"leave t4 out, got \['lanelet2', 't4'\]$",
        ),
    ],
)
def test_map_formats_the_map_set_cannot_hold_are_refused(
    formats: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DriverClientConfig.from_mapping(
            {"map_dir": "/tmp/maps", "map_formats": formats}
        )


def test_map_formats_are_kept_as_a_tuple() -> None:
    config = DriverClientConfig.from_mapping(
        {"map_dir": "/tmp/maps", "map_formats": ["lanelet2", "opendrive"]}
    )
    assert config.map_formats == ("lanelet2", "opendrive")
    assert DriverClientConfig.from_mapping({"map_formats": None}).map_formats == (
        "lanelet2",
    )
    # t4 with the scenes that write it.
    config = DriverClientConfig.from_mapping(
        {"map_dir": "/tmp/maps", "t4_scene": True, "map_formats": ["t4"]}
    )
    assert config.map_formats == ("t4",)


def test_a_carla_driver_run_refuses_bare_map_formats_up_front() -> None:
    from autoware_carla_scenario.examples import run

    cfg = _composed(["ego.entity=carla_driver", "driver.map_formats=lanelet2"])
    with pytest.raises(
        run.RunConfigError, match=r"^driver\.map_formats must be a list"
    ):
        run.check_run_config(cfg)
