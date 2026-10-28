"""The HILS configuration refuses the values YAML 1.1 turns into something
other than what was written (issue #695).

``load_config`` reads the file with ``yaml.safe_load``: unquoted, ``0102`` is
the octal int 66, ``0x10`` the int 16, ``On`` the bool ``True``, ``1.10`` the
float 1.1.  The text written is gone by the time the configuration is parsed,
so a field that holds text refuses anything but a string and says to quote it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from autoware_carla_scenario.hils.config import (
    HilsConfigError,
    load_config,
    parse_config,
)

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "hils.md"

_SEND = """
devices:
  ecu: {{url: "can://vcan0"}}
lifecycle:
  startup:
    - {{send: ecu, id: 0x7DF, data: {data}}}
"""


def _frame(data: str) -> bytes | None:
    config = parse_config(yaml.safe_load(_SEND.format(data=data)))
    return config.steps("startup")[0].data


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ('"0102"', b"\x01\x02"),
        ('"02 10 01"', b"\x02\x10\x01"),
        ("'00'", b"\x00"),
        ('"10"', b"\x10"),
    ],
)
def test_quoted_data_is_the_bytes_written(data: str, expected: bytes) -> None:
    assert _frame(data) == expected


@pytest.mark.parametrize(
    "data", ["0102", "0x10", "1234", "00", "1.5", "true", "[1, 2]"]
)
def test_unquoted_data_is_refused_with_a_hint_to_quote_it(data: str) -> None:
    # 0102 was sent as b"\x66", 0x10 as b"\x16"; 00 was refused as "not
    # hexadecimal bytes"; 1234 was right only by luck.
    with pytest.raises(HilsConfigError, match=r"\.data: .*quote"):
        _frame(data)


def test_quoted_data_that_is_not_hex_is_still_refused() -> None:
    with pytest.raises(HilsConfigError, match="not hexadecimal bytes"):
        _frame('"zz"')


_BASE = """
devices:
  can:
    url: mem://x
    messages:
      - name: Wake
        fields: {Level: 1}
"""


def _parse(text: str) -> None:
    parse_config(yaml.safe_load(text))


@pytest.mark.parametrize(
    ("text", "where"),
    [
        # A DBC signal named On, Off, Yes or No is a YAML bool as a key.
        (
            "devices:\n  can:\n    url: mem://x\n"
            "    messages: [{name: Wake, fields: {On: 1}}]\n",
            r"devices\.can\.messages\[0\]\.fields",
        ),
        (
            "devices:\n  can:\n    url: mem://x\n"
            "    messages: [{name: 0102, fields: {Level: 1}}]\n",
            r"devices\.can\.messages\[0\]\.name",
        ),
        (
            _BASE
            + "lifecycle:\n  startup:\n    - {send: can, name: Wake, fields: {Off: 0}}\n",
            r"lifecycle\.startup\[0\]\.fields",
        ),
        (
            _BASE + "lifecycle:\n  startup:\n    - {send: can, name: 0x10}\n",
            r"lifecycle\.startup\[0\]\.name",
        ),
        (
            _BASE + "lifecycle:\n  scenario_start:\n"
            "    - {wait: can, name: Ready, fields: {Yes: 1}}\n",
            r"lifecycle\.scenario_start\[0\]\.fields",
        ),
        (
            _BASE + "lifecycle:\n  scenario_start:\n    - {wait: can, name: 1234}\n",
            r"lifecycle\.scenario_start\[0\]\.name",
        ),
        (
            "devices:\n  0102: {url: mem://x}\n",
            r"devices: key 66",
        ),
        (
            "devices:\n  can:\n    url: mem://x\n"
            "    messages: [{name: Wake, fields: {Level: {signal: 1}}}]\n",
            r"fields\.Level\.signal",
        ),
        (
            "commands: {mode: control}\n"
            "devices:\n  can:\n    url: mem://x\n"
            "    inputs: [{name: 0x10, command: {throttle: Throttle}}]\n",
            r"devices\.can\.inputs\[0\]\.name",
        ),
        (
            "devices:\n  can:\n    url: mem://x\n"
            "    inputs: [{name: Cmd, command: {throttle: {field: On}}}]\n",
            r"command\.throttle\.field",
        ),
        ("ego: {role_name: 0102}\n" + _BASE, r"ego\.role_name"),
        ("carla: {host: 10.0}\n" + _BASE, r"carla\.host"),
        ("runner: {listen: 7700}\n" + _BASE, r"runner\.listen"),
        ("devices:\n  can: {url: mem://x, codec: 1}\n", r"devices\.can\.codec"),
    ],
)
def test_a_name_yaml_reads_as_another_type_is_refused(text: str, where: str) -> None:
    with pytest.raises(HilsConfigError, match=f"{where}.*quote"):
        _parse(text)


def test_quoted_names_are_kept_as_written() -> None:
    config = parse_config(
        yaml.safe_load(
            """
ego: {role_name: "0102"}
runner: {listen: null}
devices:
  "0102":
    url: mem://x
    messages: [{name: "0x10", fields: {"On": 1, Level: "$speed"}}]
lifecycle:
  startup:
    - {send: "0102", name: "1234", fields: {"Off": 0}}
"""
        )
    )
    [device] = config.devices
    assert device.name == "0102"
    assert config.role_name == "0102"
    assert config.runner_listen == ""
    [message] = device.messages
    assert message.name == "0x10"
    assert set(message.fields) == {"On", "Level"}
    [step] = config.steps("startup")
    assert (step.device, step.name, set(step.fields)) == ("0102", "1234", {"Off"})


def test_load_config_refuses_unquoted_data(tmp_path: Path) -> None:
    path = tmp_path / "hils.yaml"
    path.write_text(_SEND.format(data="0102"), encoding="utf-8")
    with pytest.raises(HilsConfigError, match="quote"):
        load_config(path)


def _yaml_blocks(text: str) -> list[tuple[int, str]]:
    return [
        (text[: m.start()].count("\n") + 1, m.group(1))
        for m in re.finditer(r"```ya?ml\n(.*?)```", text, re.S)
    ]


def test_the_docs_lifecycle_example_parses() -> None:
    """The documentation's lifecycle steps parse, raw frames to the bytes shown."""
    blocks = [
        (line, block)
        for line, block in _yaml_blocks(_DOCS.read_text(encoding="utf-8"))
        if block.startswith("lifecycle:")
    ]
    assert blocks, "docs/hils.md has no lifecycle example"
    for line, block in blocks:
        document = yaml.safe_load(block)
        devices = {
            step.get("send") or step.get("wait")
            for steps in document["lifecycle"].values()
            for step in steps
        } - {None}
        document["devices"] = {
            name: {"url": "mem://x", "messages": [{"name": "M", "fields": {"F": 1}}]}
            for name in devices
        }
        try:
            config = parse_config(document)
        except HilsConfigError as exc:  # pragma: no cover - the failure message
            raise AssertionError(f"docs/hils.md:{line}: {exc}") from exc
        raw = [s.data for p in config.lifecycle.values() for s in p if s.data]
        assert raw, f"docs/hils.md:{line}: no raw frame"
