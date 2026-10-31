"""What YAML 1.1 makes of a plain scalar (``hils.yaml11``, #800), against
what ``hils.config`` computed before it was split out, and that
``hils.config`` still re-exports it."""

from __future__ import annotations

from typing import Callable, Optional

import pytest
import yaml

from autoware_carla_scenario.hils import config, yaml11
from autoware_carla_scenario.hils.yaml11 import (
    YAML_BOOL,
    YAML_FLOAT,
    YAML_INT,
    yaml11_reading,
)

# -- What hils.config computed before #800, verbatim -------------------------

_MASTER_INT = "tag:yaml.org,2002:int"
_MASTER_FLOAT = "tag:yaml.org,2002:float"
_MASTER_BOOL = "tag:yaml.org,2002:bool"


def _master_reading(tag: str, text: str) -> Optional[str]:
    digits = text.lstrip("+-").replace("_", "")
    if tag == _MASTER_BOOL and text.lower() in ("yes", "no", "on", "off"):
        truth = text.lower() in ("yes", "on")
        return f"the boolean {truth}; write {str(truth).lower()}"
    if tag in (_MASTER_INT, _MASTER_FLOAT) and ":" in digits:
        return "a sexagesimal (base 60) number; write it in decimal"
    if (
        tag == _MASTER_INT
        and len(digits) > 1
        and digits[0] == "0"
        and digits[1] in "01234567"
    ):
        value = int(digits, 8) * (-1 if text.startswith("-") else 1)
        return f"the octal number {value}; write {value} or {value:#x}"
    return None


def _outcome(
    reading: Callable[[str, str], Optional[str]], tag: str, text: str
) -> object:
    """What *reading* returns, or the type of what it raises."""
    try:
        return reading(tag, text)
    except Exception as exc:  # noqa: BLE001 - compared, not handled
        return type(exc)


def test_tags_are_masters() -> None:
    assert (YAML_INT, YAML_FLOAT, YAML_BOOL) == (
        _MASTER_INT,
        _MASTER_FLOAT,
        _MASTER_BOOL,
    )


def test_config_re_exports_them() -> None:
    assert config._yaml11_reading is yaml11.yaml11_reading
    assert config._YAML_INT == YAML_INT
    assert config._YAML_FLOAT == YAML_FLOAT
    assert config._YAML_BOOL == YAML_BOOL


_TEXTS = [
    "",
    "0",
    "00",
    "0102",
    "-0102",
    "+0102",
    "0_1_2",
    "07",
    "-07",
    "0x7DF",
    "0X10",
    "0o17",
    "0b101",
    "1_000",
    "1.5",
    "01.5",
    "1:30",
    "-1:30",
    "190:20:30",
    "1:30.5",
    "+1:30",
    "yes",
    "Yes",
    "YES",
    "no",
    "No",
    "on",
    "On",
    "ON",
    "off",
    "Off",
    "OFF",
    "true",
    "False",
    "y",
    "n",
    "hello",
    "-",
    "+-0102",
    "--07",
    "_0102",
]
_TAGS = [
    _MASTER_INT,
    _MASTER_FLOAT,
    _MASTER_BOOL,
    "tag:yaml.org,2002:str",
    "tag:yaml.org,2002:null",
]


@pytest.mark.parametrize("tag", _TAGS)
@pytest.mark.parametrize("text", _TEXTS)
def test_reading_matches_master(tag: str, text: str) -> None:
    assert _outcome(yaml11_reading, tag, text) == _outcome(_master_reading, tag, text)


@pytest.mark.parametrize("text", ["0109", "-0018", "01.5", "0_7_9"])
def test_an_invalid_octal_raises_as_master_did(text: str) -> None:
    with pytest.raises(ValueError):
        _master_reading(_MASTER_INT, text)
    with pytest.raises(ValueError):
        yaml11_reading(YAML_INT, text)


@pytest.mark.parametrize(
    "text", [t for t in _TEXTS if t not in ("", "-")] + ["0o7", "1e3", "~"]
)
def test_reading_matches_master_for_what_yaml_resolves(text: str) -> None:
    # The tag PyYAML's resolver gives the plain scalar, as load_config sees it.
    node = yaml.compose(f"key: {text}")
    assert node is not None
    scalar = node.value[0][1]
    if scalar.style is not None:
        pytest.skip("not a plain scalar")
    assert yaml11_reading(scalar.tag, scalar.value) == _master_reading(
        scalar.tag, scalar.value
    )
