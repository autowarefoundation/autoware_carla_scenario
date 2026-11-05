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
    with pytest.raises(HilsConfigError, match=r"hils\.yaml: .*4300 digits"):
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
