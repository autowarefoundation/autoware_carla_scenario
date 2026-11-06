"""An integer of more digits than Python turns to or from text
(``sys.get_int_max_str_digits()``, 4300 by default) in a HILS configuration
is a :class:`HilsConfigError` naming where it is, not a bare ``ValueError``
(#1625)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.hils.config import (
    HilsConfigError,
    load_config,
    parse_config,
)

#: Too long for ``repr()`` and ``str()``.
HUGE = 10**5000


def _device(**extra: Any) -> dict[str, Any]:
    return {"devices": {"can": {"url": "mem://x", **extra}}}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            HUGE,
            r"configuration: expected a mapping, got an integer of about 10\*\*5000",
        ),
        ({"devices": HUGE}, r"devices: expected a mapping, got an integer of about"),
        ({HUGE: 1}, r"configuration: unknown key\(s\) an integer of about 10\*\*5000"),
        ({HUGE: 1, "zzz": 2}, r"unknown key\(s\) an integer of about .*, zzz"),
        ({"clock": HUGE}, r"clock: expected text, got an integer of about"),
        ({"runner": {"listen": HUGE}}, r"runner\.listen: expected host:port text"),
        (
            {**_device(), "carla": {"port": HUGE}},
            r"carla\.port: must be within 1-65535, got an integer of about",
        ),
        ({"devices": {HUGE: {}}}, r"devices: key an integer of about .*expected text"),
        (
            _device(messages=[{"name": "M", "fields": {"F": [HUGE]}}]),
            r"fields\.F: expected a constant.*a list holding an integer too long",
        ),
        (
            _device(options={HUGE: 1}),
            r"devices\.can\.options: key: expected an integer of fewer digits",
        ),
        (
            _device(
                codec="lidar:vlp16",
                sensor={"blueprint": "sensor.lidar.ray_cast", "attributes": {HUGE: 1}},
            ),
            r"sensor\.attributes: key: expected an integer of fewer digits",
        ),
        (
            {
                **_device(),
                "lifecycle": {"startup": [{"send": "can", "data": [HUGE]}]},
            },
            r"startup\[0\]\.data: a list holds byte values .*too long",
        ),
        (
            {**_device(), "lifecycle": {"startup": [{HUGE: 1}]}},
            r"startup\[0\]: a step is one of .*an integer of about",
        ),
    ],
    ids=[
        "document",
        "devices",
        "root-key",
        "root-keys",
        "clock",
        "listen",
        "port",
        "device-key",
        "field",
        "option-key",
        "attribute-key",
        "data",
        "step-key",
    ],
)
def test_parse_config_names_a_huge_integer(document: Any, message: str) -> None:
    with pytest.raises(HilsConfigError, match=message):
        parse_config(document)


@pytest.mark.parametrize(
    ("document", "where"),
    [
        (_device(messages=[{"name": HUGE}]), r"devices\.can\.messages\[0\]\.name"),
        (
            _device(inputs=[{"name": HUGE, "command": {"throttle": "T"}}]),
            r"devices\.can\.inputs\[0\]\.name",
        ),
        (
            {**_device(), "lifecycle": {"startup": [{"send": "can", "name": HUGE}]}},
            r"lifecycle\.startup\[0\]\.name",
        ),
    ],
    ids=["message", "input", "step"],
)
def test_a_number_name_too_long_for_text_is_refused(document: Any, where: str) -> None:
    # _text(number=True) takes an integer as its decimal text.
    with pytest.raises(
        HilsConfigError,
        match=where + r": expected an integer of fewer digits, got an integer of about",
    ):
        parse_config(document)


def test_a_huge_command_key_is_named() -> None:
    document = _device(inputs=[{"name": "M", "command": {HUGE: "T"}}])
    with pytest.raises(
        HilsConfigError, match=r"command: an integer of about .* is not a field"
    ):
        parse_config(document)


@pytest.mark.parametrize(
    "text",
    [
        "devices: " + "9" * 5000,
        "devices: {can: {url: 'mem://x', messages: [{name: " + "1" * 5000 + "}]}}",
        # A plain key is at most 1024 characters; an explicit one is not.
        "devices:\n  ? " + "7" * 5000 + "\n  : {}",
    ],
    ids=["value", "name", "key"],
)
def test_load_config_names_the_file_for_a_huge_integer(
    tmp_path: Path, text: str
) -> None:
    # PyYAML's int constructor raises a bare ValueError for it.
    path = tmp_path / "hils.yaml"
    path.write_text(text + "\n")
    # "the limit (4300 digits)" from 3.11 on, "the limit (4300)" on 3.10.
    with pytest.raises(HilsConfigError, match=r"hils\.yaml: .*limit \(4300"):
        load_config(path)


def test_load_config_still_reports_a_yaml11_scalar(tmp_path: Path) -> None:
    # The new ValueError handler must not swallow HilsConfigError (a
    # ValueError) into a message of its own.
    path = tmp_path / "hils.yaml"
    path.write_text("clock: 0102\n")
    with pytest.raises(HilsConfigError, match=r"hils\.yaml:1:8: clock: unquoted"):
        load_config(path)


def test_an_ordinary_value_is_still_shown_by_repr() -> None:
    with pytest.raises(HilsConfigError, match=r"carla\.port: .*got 70000$"):
        parse_config({**_device(), "carla": {"port": 70000}})
    with pytest.raises(HilsConfigError, match=r"clock: expected text, got the int 5;"):
        parse_config({**_device(), "clock": 5})


@pytest.mark.parametrize(
    "value",
    [HUGE, -HUGE, 10**6000, 10**4300 + 1, -(10**4300 + 1), 2**20000, 2**20000 - 1],
    # The default ids are repr(), which refuses these.
    ids=[
        "1e5000",
        "-1e5000",
        "1e6000",
        "1e4300+1",
        "-1e4300-1",
        "2^20000",
        "2^20000-1",
    ],
)
def test_a_huge_integer_is_shown_as_bit_length_says(value: int) -> None:
    # hils.config's _huge is the checked value_text.huge_integer (#1704); its
    # text is the one bit_length() gives, as it was.
    from autoware_carla_scenario.hils import config
    from autoware_carla_scenario.value_text import huge_integer, shown

    sign = "-" if value < 0 else ""
    expected = f"an integer of about {sign}10**{int(abs(value).bit_length() * 0.30103)}"
    assert config._huge is huge_integer
    assert huge_integer(value) == expected
    assert shown(value) == expected
    assert config._shown(value) == expected


def test_the_signal_check_is_the_sensor_specs() -> None:
    from autoware_carla_scenario.hils import config, sensor_spec

    assert config._signal is sensor_spec.checked_signal
    assert config._signal("speed", "x") == "speed"
    with pytest.raises(
        HilsConfigError, match=r"^x: unknown signal 'nope'; signals: time, frame, "
    ):
        config._signal("nope", "x")
