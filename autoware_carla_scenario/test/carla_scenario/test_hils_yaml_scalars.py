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
    "data", ["0102", "0x10", "1234", "00", "1.5", "true", "{a: 1}"]
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
            "    messages: [{name: 1.10, fields: {Level: 1}}]\n",
            r"devices\.can\.messages\[0\]\.name",
        ),
        (
            _BASE
            + "lifecycle:\n  startup:\n    - {send: can, name: Wake, fields: {Off: 0}}\n",
            r"lifecycle\.startup\[0\]\.fields",
        ),
        (
            _BASE + "lifecycle:\n  startup:\n    - {send: can, name: true}\n",
            r"lifecycle\.startup\[0\]\.name",
        ),
        (
            _BASE + "lifecycle:\n  scenario_start:\n"
            "    - {wait: can, name: Ready, fields: {Yes: 1}}\n",
            r"lifecycle\.scenario_start\[0\]\.fields",
        ),
        (
            _BASE + "lifecycle:\n  scenario_start:\n    - {wait: can, name: 1.5}\n",
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
            "    inputs: [{name: On, command: {throttle: Throttle}}]\n",
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


@pytest.mark.parametrize(
    ("data", "expected"),
    [("[0x01, 0x02]", b"\x01\x02"), ("[]", b""), ("[255, 0]", b"\xff\x00")],
)
def test_data_can_be_a_list_of_byte_values(data: str, expected: bytes) -> None:
    assert _frame(data) == expected


@pytest.mark.parametrize("data", ["[256]", "[-1]", "[true]", "['01']", "[1.0]"])
def test_a_list_of_data_holds_bytes(data: str) -> None:
    with pytest.raises(HilsConfigError, match=r"\.data: a list holds byte values"):
        _frame(data)


def test_a_numeric_message_name_is_its_decimal_text() -> None:
    # RTCM names its messages by number: `name: 1005` is message "1005".
    config = parse_config(
        yaml.safe_load(
            """
devices:
  rtk:
    url: mem://x
    codec: rtcm
    messages: [{name: 1005, fields: {}}]
    inputs: [{name: 1077, command: {throttle: 0.0}}]
lifecycle:
  startup:
    - {send: rtk, name: 1005}
    - {wait: rtk, name: 1077}
"""
        )
    )
    [device] = config.devices
    assert device.messages[0].name == "1005"
    assert device.inputs[0].name == "1077"
    assert [s.name for s in config.steps("startup")] == ["1005", "1077"]


def test_runner_listen_takes_host_and_port_text() -> None:
    with pytest.raises(HilsConfigError, match=r"runner\.listen: expected host:port"):
        _parse("runner: {listen: 7700}\n" + _BASE)


_YAML11 = """\
devices:
  can:
    url: mem://x
    options: {strict: yes}
    messages:
      - name: Gear
        fields: {Position: Off, Mode: 1:30, Code: -012}
lifecycle:
  startup:
    - {send: can, id: 0x7DF, data: 00}
"""


@pytest.mark.parametrize(
    ("old", "new", "where", "reading"),
    [
        (
            "{strict: yes}",
            "{strict: true}",
            r":4:23: devices\.can\.options\.strict",
            "the boolean True; write true",
        ),
        (
            "Position: Off",
            "Position: P",
            r":7:28: devices\.can\.messages\[0\]\.fields\.Position",
            "the boolean False; write false",
        ),
        (
            "Mode: 1:30",
            "Mode: 90",
            r":7:37: devices\.can\.messages\[0\]\.fields\.Mode",
            "a sexagesimal",
        ),
        (
            "Code: -012",
            "Code: -10",
            r":7:47: devices\.can\.messages\[0\]\.fields\.Code",
            "the octal number -10; write -10 or -0xa",
        ),
        (
            "data: 00",
            'data: "00"',
            r":10:36: lifecycle\.startup\[0\]\.data",
            "the octal number 0",
        ),
    ],
)
def test_load_config_refuses_yaml11_scalars_anywhere(
    tmp_path: Path, old: str, new: str, where: str, reading: str
) -> None:
    """Each YAML 1.1-only form is refused where it is, with its line and path;
    the others are fixed so each case finds only its own."""
    text = _YAML11
    for o, n in [
        ("{strict: yes}", "{strict: true}"),
        ("Position: Off", "Position: P"),
        ("Mode: 1:30", "Mode: 90"),
        ("Code: -012", "Code: -10"),
        ("data: 00", 'data: "00"'),
    ]:
        if o != old:
            text = text.replace(o, n)
    path = tmp_path / "hils.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(HilsConfigError) as info:
        load_config(path)
    message = str(info.value)
    assert re.search(where, message), message
    assert reading in message
    assert "quote it" in message
    # Fixed, the file loads.
    path.write_text(text.replace(old, new), encoding="utf-8")
    load_config(path)


def test_load_config_refuses_a_yaml11_key(tmp_path: Path) -> None:
    path = tmp_path / "hils.yaml"
    path.write_text(
        "devices:\n  can:\n    url: mem://x\n"
        "    messages: [{name: M, fields: {On: 1}}]\n",
        encoding="utf-8",
    )
    with pytest.raises(
        HilsConfigError, match=r"devices\.can\.messages\[0\]\.fields: key: .*True"
    ):
        load_config(path)


def test_load_config_takes_what_yaml_12_reads_alike(tmp_path: Path) -> None:
    """Hex, underscores, floats, true/false, quoted text and the example."""
    path = tmp_path / "hils.yaml"
    path.write_text(
        """\
runner: {listen: 0.0.0.0:7700}
carla: {host: 10.0.0.1, timeout: 1.5}
devices:
  can:
    url: mem://x
    messages:
      - name: M
        fields: {A: 0x1F, B: 1_000, C: true, D: "Off", E: '0102', F: 0, G: 08}
lifecycle:
  startup:
    - {send: can, id: 0x7DF, data: "0102"}
""",
        encoding="utf-8",
    )
    config = load_config(path)
    [device] = config.devices
    fields = {k: v.constant for k, v in device.messages[0].fields.items()}
    assert fields == {
        "A": 31,
        "B": 1000,
        "C": True,
        "D": "Off",
        "E": "0102",
        "F": 0,
        "G": "08",
    }
    assert config.steps("startup")[0].data == b"\x01\x02"


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
