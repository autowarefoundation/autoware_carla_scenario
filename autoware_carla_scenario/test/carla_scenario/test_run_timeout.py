"""``scenario.timeout_seconds`` refused with the value as it was written (#2231).

``run_timeout_seconds`` converts the value to a float before it checks it, and
the refusal showed the converted float: ``timeout_seconds: 0`` was refused as
``got 0.0``.
"""

from __future__ import annotations

import math

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.run_timeout import check_run_timeout, run_timeout_seconds

_REFUSED = "scenario.timeout_seconds must be positive and finite, got "


@pytest.mark.parametrize(
    ("timeout", "shown"),
    [
        (0, "0"),
        (-1, "-1"),
        (-15, "-15"),
        (0.0, "0.0"),
        (-0.5, "-0.5"),
        (math.nan, "nan"),
        (math.inf, "inf"),
        (-math.inf, "-inf"),
    ],
)
@pytest.mark.parametrize("omegaconf", [False, True])
def test_a_timeout_is_refused_as_written(
    timeout: float, shown: str, omegaconf: bool
) -> None:
    cfg = {"scenario": {"timeout_seconds": timeout}}
    with pytest.raises(ValueError) as refused:
        run_timeout_seconds(OmegaConf.create(cfg) if omegaconf else cfg)
    assert str(refused.value) == _REFUSED + shown


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [(0.0, "0.0"), (-2.5, "-2.5"), (math.nan, "nan"), (math.inf, "inf")],
)
def test_check_run_timeout_shows_the_float_it_was_given(
    seconds: float, shown: str
) -> None:
    with pytest.raises(ValueError) as refused:
        check_run_timeout(seconds)
    assert str(refused.value) == _REFUSED + shown


def test_check_run_timeout_shows_the_value_as_written() -> None:
    with pytest.raises(ValueError) as refused:
        check_run_timeout(0.0, "0")
    assert str(refused.value) == _REFUSED + "0"
    assert check_run_timeout(15.0, "15") == 15.0
    assert check_run_timeout(0.5) == 0.5


@pytest.mark.parametrize(
    ("seconds", "message"),
    [
        (True, "timeout_seconds must be a number, got True"),
        (False, "timeout_seconds must be a number, got False"),
        ("5", "timeout_seconds must be a number, got '5'"),
        ([5], "timeout_seconds must be a number, got [5]"),
        (10**400, "timeout_seconds must be positive and finite, got "),
        (-(10**400), "timeout_seconds must be positive and finite, got "),
    ],
    ids=["true", "false", "str", "list", "huge", "-huge"],
)
def test_check_run_timeout_refuses_what_is_no_float(
    seconds: object, message: str
) -> None:
    """#2403: the parameter path (``ScenarioRunner``, ``run_scenario()``,
    ``ScenarioQueue``) refuses what the config path does, with a keyed
    ValueError: ``True`` is no 1 s timeout, and neither a ``TypeError`` nor an
    ``OverflowError`` escapes."""
    with pytest.raises(ValueError) as refused:
        check_run_timeout(seconds, key="timeout_seconds")  # type: ignore[arg-type]
    assert str(refused.value).startswith(message)


def test_check_run_timeout_takes_an_int() -> None:
    assert check_run_timeout(5, key="timeout_seconds") == 5.0
    assert isinstance(check_run_timeout(5), float)
