"""A null HILS setting is its default; an empty one is refused (#2170).

``hils.*`` (the run's side) crashed on a null or text timeout with a bare
``TypeError`` naming no key, took ``stale_s: true`` as 1 s and refused a null
host, port or on_stale.  The bridge's YAML refused a null scalar (``carla:
{port: null}``) instead of defaulting, and took an empty ``runner.listen`` as
no run link.  SUMO's mirrored pedestrians could be sent a negative speed.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig
from autoware_carla_scenario.hils.cli import example_config, main
from autoware_carla_scenario.hils.config import HilsConfigError, parse_config

_TIMEOUTS = ("connect_timeout_s", "ready_timeout_s", "end_timeout_s", "stale_s")


# -- Item 1: hils.* ----------------------------------------------------------


@pytest.mark.parametrize("key", [*_TIMEOUTS, "host", "port", "on_stale"])
def test_a_null_hils_setting_is_its_default(key: str) -> None:
    assert HilsEgoConfig.from_mapping({key: None}) == HilsEgoConfig()


@pytest.mark.parametrize("key", _TIMEOUTS)
@pytest.mark.parametrize("value", [True, False, "10", [1.0]])
def test_a_hils_timeout_that_is_not_a_number_is_refused(key: str, value: Any) -> None:
    with pytest.raises(ValueError) as error:
        HilsEgoConfig.from_mapping({key: value})
    assert str(error.value) == (
        f"hils.{key} must be a positive, finite number of seconds, got {value!r}"
    )


@pytest.mark.parametrize("key", _TIMEOUTS)
def test_a_whole_number_of_seconds_is_taken(key: str) -> None:
    assert getattr(HilsEgoConfig.from_mapping({key: 3}), key) == 3


@pytest.mark.parametrize("host", ["", "  ", "[ ]"])
def test_an_empty_hils_host_is_refused(host: str) -> None:
    with pytest.raises(ValueError, match=r"^hils\.host must be a host name"):
        HilsEgoConfig.from_mapping({"host": host})


# -- Item 2: null bridge settings are their defaults -------------------------


def _example() -> dict[str, Any]:
    return yaml.safe_load(example_config())


def _sensor_device(document: dict[str, Any]) -> dict[str, Any]:
    return next(d for d in document["devices"].values() if d.get("sensor"))


def test_null_scalars_take_their_defaults() -> None:
    document = _example()
    document["carla"] = {"host": None, "port": None, "timeout": None}
    document["ego"] = {"role_name": None, "wait": None}
    document["clock"] = None
    document["commands"] = {"mode": None}
    config = parse_config(document)
    assert (config.host, config.port, config.timeout) == ("localhost", 2000, 10.0)
    assert (config.role_name, config.ego_wait) == (str(EGO_ROLE_NAME), 120.0)
    assert config.clock == "simulation"
    assert config.command_mode == "control"


def test_a_null_sensor_setting_takes_its_default() -> None:
    expected = parse_config(_example())
    document = _example()
    sensor = _sensor_device(document)["sensor"]
    sensor["transform"] = {key: None for key in sensor.get("transform", {})}
    sensor["encoding"] = None
    sensor["intensity_scale"] = None
    nulled = parse_config(document)
    name = next(k for k, v in document["devices"].items() if v.get("sensor"))
    [before] = [d.sensor for d in expected.devices if d.name == name]
    [after] = [d.sensor for d in nulled.devices if d.name == name]
    assert after is not None and before is not None
    assert after.transform == (0.0,) * len(after.transform)
    assert (after.encoding, after.intensity_scale) == ("bgra", 255.0)


def test_a_null_codec_is_raw() -> None:
    document = {"devices": {"d": {"url": "mem://x", "codec": None}}}
    [device] = parse_config(document).devices
    assert device.codec == "raw"


def test_a_null_scale_offset_and_wait_timeout_take_their_defaults() -> None:
    document = {
        "devices": {
            "d": {
                "url": "mem://x",
                "codec": "nmea",
                "messages": [
                    {
                        "name": "GGA",
                        "fields": {
                            "lat": {"signal": "speed", "scale": None, "offset": None}
                        },
                    }
                ],
            }
        },
        "lifecycle": {"startup": [{"wait": "d", "name": "GGA", "timeout": None}]},
    }
    config = parse_config(document)
    [message] = config.devices[0].messages
    source = message.fields["lat"]
    assert (source.scale, source.offset) == (1.0, 0.0)
    [wait] = config.steps("startup")
    assert wait.seconds == 10.0


# -- Item 3: empty text names nothing -----------------------------------------


@pytest.mark.parametrize(
    ("section", "key"),
    [("runner", "listen"), ("carla", "host"), ("ego", "role_name")],
)
@pytest.mark.parametrize("value", ["", " "])
def test_empty_text_is_refused(section: str, key: str, value: str) -> None:
    document = copy.deepcopy(_example())
    document.setdefault(section, {})[key] = value
    with pytest.raises(HilsConfigError, match=rf"^{section}\.{key}: "):
        parse_config(document)


def test_only_a_null_listen_turns_the_run_link_off() -> None:
    document = _example()
    document["runner"] = {"listen": None}
    assert parse_config(document).runner_listen == ""


@pytest.mark.parametrize("flag", ["--listen", "--host", "--role-name"])
def test_an_empty_override_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    path = tmp_path / "hils.yaml"
    path.write_text(example_config())
    assert main([str(path), "--check", flag, ""]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"scenario-hils: {flag}: ''")


# -- Item 4: SUMO's pedestrians never walk at a negative speed ----------------


class _Walker:
    id = 1

    def __init__(self) -> None:
        self.control: Any = None

    def get_transform(self) -> Any:
        return SimpleNamespace(location=SimpleNamespace(z=0.0))

    def set_transform(self, transform: Any) -> None:
        pass

    def apply_control(self, control: Any) -> None:
        self.control = control


@pytest.mark.parametrize("speed", [-1.5, -0.1])
def test_a_negative_sumo_speed_stands_the_walker_still(speed: float) -> None:
    from autoware_carla_scenario.entity.pedestrian_entity import walker_control_speed
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.geometry import Pose2D

    walker = _Walker()
    backend = SimpleNamespace(_context=None)
    pose = Pose2D(0.0, 0.0, 0.0, 0.0)
    SumoTrafficBackend._walk(backend, walker, pose, 1.0)  # type: ignore[arg-type]
    assert walker.control.speed == pytest.approx(walker_control_speed(1.0))
    SumoTrafficBackend._walk(backend, walker, pose, speed)  # type: ignore[arg-type]
    assert walker.control.speed == 0.0


# -- #2241: an empty codec or url ------------------------------------------


@pytest.mark.parametrize(
    ("codec", "message"),
    [
        ("", r"expected a codec name .*, got ''$"),
        ("  ", r"expected a codec name .*, got '  '$"),
        (":vehicle.dbc", r"expected a codec name .*, got ':vehicle\.dbc'$"),
        ("dbc:", r"expected an argument after the colon .*, got 'dbc:'$"),
        ("lidar: ", r"expected an argument after the colon .*, got 'lidar: '$"),
    ],
)
def test_an_empty_codec_is_refused_naming_its_key(codec: str, message: str) -> None:
    document = {"devices": {"vehicle_can": {"url": "mem://x", "codec": codec}}}
    with pytest.raises(
        HilsConfigError, match=rf"^devices\.vehicle_can\.codec: {message}"
    ):
        parse_config(document)


@pytest.mark.parametrize(
    ("url", "shown"),
    [("", "''"), ("  ", "'  '"), (False, "the bool False"), (0, "the int 0")],
)
def test_an_empty_url_is_refused_naming_its_key_and_value(url: Any, shown: str) -> None:
    document = {"devices": {"d": {"url": url}}}
    with pytest.raises(HilsConfigError, match=r"^devices\.d\.url: ") as error:
        parse_config(document)
    assert shown in str(error.value)


@pytest.mark.parametrize(
    "document", [{"devices": {"d": {}}}, {"devices": {"d": {"url": None}}}]
)
def test_a_device_without_a_url_is_refused(document: Any) -> None:
    with pytest.raises(HilsConfigError, match=r"^devices\.d\.url: a device has a url$"):
        parse_config(document)
