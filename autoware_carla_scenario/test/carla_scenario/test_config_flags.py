"""config_flags.check_flag: a switch read as a boolean or its spelling (#1753)."""

from __future__ import annotations

from typing import Any

import pytest

from autoware_carla_scenario.config_flags import FALSE_TEXT, TRUE_TEXT, check_flag
from autoware_carla_scenario.slurm.settings import SlurmSettings


@pytest.mark.parametrize(
    "value", [True, "true", "True", "TRUE", " yes ", "on", "1", 1, " 1 "]
)
def test_a_spelling_of_true_is_true(value: Any) -> None:
    assert check_flag("rerun.enabled", value) is True


@pytest.mark.parametrize(
    "value", [False, "false", "False", " no ", "off", "OFF", "0", 0, -0]
)
def test_a_spelling_of_false_is_false(value: Any) -> None:
    assert check_flag("rerun.enabled", value) is False


@pytest.mark.parametrize(
    "value",
    ["ture", "", "  ", "2", "01", "y", "n", 2, -1, 0.0, 1.0, None, [True], {}],
)
def test_anything_else_is_refused_in_one_line_naming_the_key(value: Any) -> None:
    with pytest.raises(ValueError) as caught:
        check_flag("t4.enabled", value)
    message = str(caught.value)
    assert message == f"t4.enabled must be true or false, got {value!r}"
    assert "\n" not in message


def test_an_int_too_long_for_repr_is_refused_by_its_key() -> None:
    with pytest.raises(
        ValueError,
        match=r"^rerun\.strict must be true or false, got an integer of about 10\*\*",
    ):
        check_flag("rerun.strict", 10**5000)  # type: ignore[arg-type]


def test_the_spellings_are_the_authoring_registrys() -> None:
    from autoware_carla_scenario.authoring.registry import (
        FALSY_VALUES,
        TRUTHY_VALUES,
    )

    assert TRUE_TEXT == TRUTHY_VALUES
    assert FALSE_TEXT == FALSY_VALUES


@pytest.mark.parametrize("value", [True, "off", 1, "ture", None, 2, 0.0, [True]])
def test_slurm_reads_a_launcher_switch_with_it(value: Any) -> None:
    """``launcher.*`` switches agree with the run's: same values, same message."""
    key = "launcher.headless"
    try:
        expected: bool | str = check_flag(key, value)
    except ValueError as exc:
        expected = str(exc)
    try:
        got: bool | str = SlurmSettings.from_mapping({"headless": value}).headless
    except ValueError as exc:
        got = str(exc)
    assert got == expected
