"""``hils.sensor_spec`` and ``hils.wire.command_value``: the plain-typed
parts of ``hils.config`` and ``hils.protocol`` the library check compiles
(#1622), still re-exported where they were."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from autoware_carla_scenario import hils
from autoware_carla_scenario.hils import config, protocol, sensor_spec, wire

_MOVED = (
    "CAMERA_PREFIX",
    "CLOCKS",
    "LIDAR_BLUEPRINT",
    "PHASES",
    "HilsConfigError",
    "SensorSpec",
    "_attribute_number",
    "_lidar_attributes",
    "_relative_file",
)


@pytest.mark.parametrize("name", _MOVED)
def test_config_re_exports_the_sensor_spec(name: str) -> None:
    assert getattr(config, name) is getattr(sensor_spec, name)


def test_the_package_re_exports_the_sensor_spec() -> None:
    assert hils.SensorSpec is sensor_spec.SensorSpec
    assert hils.HilsConfigError is sensor_spec.HilsConfigError
    assert issubclass(sensor_spec.HilsConfigError, ValueError)


@pytest.mark.parametrize("name", ["command_value", "COMMAND_VALUE_LIMIT"])
def test_protocol_re_exports_command_value(name: str) -> None:
    assert getattr(protocol, name) is getattr(wire, name)
    assert name in protocol.__all__
    assert name in wire.__all__


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, 0.5),
        (-1.0, -1.0),
        (wire.COMMAND_VALUE_LIMIT, wire.COMMAND_VALUE_LIMIT),
        (1e39, None),
        (-1e39, None),
        (math.inf, None),
        (-math.inf, None),
        (math.nan, None),
    ],
)
def test_command_value(value: float, expected: object) -> None:
    assert wire.command_value(value) == expected


def test_a_sensor_spec_s_defaults() -> None:
    spec = sensor_spec.SensorSpec(blueprint=sensor_spec.LIDAR_BLUEPRINT)
    assert spec.transform == (0.0,) * 6
    assert dict(spec.attributes) == {}
    assert spec.encoding == "bgra"
    assert spec.intensity_scale == 255.0
    assert spec.is_lidar
    assert not sensor_spec.SensorSpec(blueprint="sensor.camera.rgb").is_lidar


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"horizontal_fov": "abc"}, r"x\.horizontal_fov: expected a finite number"),
        ({"horizontal_fov": "400"}, r"x\.horizontal_fov: must be in \(0, 360\]"),
        ({"rotation_frequency": "0"}, r"x\.rotation_frequency: must be positive"),
        ({"rotation_frequency": "nan"}, r"expected a finite number, got 'nan'"),
        ({"sensor_tick": "0.1"}, r"x\.sensor_tick: a LiDAR measures every"),
    ],
)
def test_lidar_attributes_are_refused(attributes: dict[str, str], message: str) -> None:
    with pytest.raises(sensor_spec.HilsConfigError, match=message):
        sensor_spec._lidar_attributes(attributes, "x")


def test_lidar_attributes_are_taken() -> None:
    sensor_spec._lidar_attributes(
        {"horizontal_fov": "360", "rotation_frequency": "10", "sensor_tick": "0"}, "x"
    )


@pytest.mark.parametrize("key", ["dropoff_general_rate", "dropoff_zero_intensity"])
@pytest.mark.parametrize("value", ["45", "-0.1", "1.01"])
def test_a_dropoff_proportion_outside_0_to_1_is_refused(key: str, value: str) -> None:
    message = rf"^x\.{key}: must be in \[0, 1\] .*got '{re.escape(value)}'$"
    with pytest.raises(sensor_spec.HilsConfigError, match=message):
        sensor_spec._lidar_attributes({key: value}, "x")


@pytest.mark.parametrize("key", ["dropoff_general_rate", "dropoff_zero_intensity"])
def test_a_dropoff_proportion_that_is_no_number_is_refused(key: str) -> None:
    with pytest.raises(sensor_spec.HilsConfigError, match=rf"x\.{key}: expected"):
        sensor_spec._lidar_attributes({key: "most"}, "x")


@pytest.mark.parametrize("key", ["dropoff_general_rate", "dropoff_zero_intensity"])
@pytest.mark.parametrize("value", ["0", "0.45", "1", "0.0", "1.0"])
def test_a_dropoff_proportion_from_0_to_1_is_taken(key: str, value: str) -> None:
    sensor_spec._lidar_attributes({key: value}, "x")


def test_the_dropoff_intensity_limit_is_no_proportion() -> None:
    sensor_spec._lidar_attributes({"dropoff_intensity_limit": "1.5"}, "x")


def test_relative_file(tmp_path: Path) -> None:
    (tmp_path / "a.dbc").write_text("")
    assert sensor_spec._relative_file("a.dbc", tmp_path) == str(
        (tmp_path / "a.dbc").resolve()
    )
    assert sensor_spec._relative_file("vlp16", tmp_path) == "vlp16"
    assert sensor_spec._relative_file("/x/../y", tmp_path) == "/x/../y"
    assert sensor_spec._relative_file("~/z", tmp_path) == str(Path("~/z").expanduser())
