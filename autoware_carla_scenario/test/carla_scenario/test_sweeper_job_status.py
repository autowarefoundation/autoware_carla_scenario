"""Tests for sweeper.job_status: a case's timeout and exit status (#1712)."""

from __future__ import annotations

import pytest

from autoware_carla_scenario.sweeper import job_status
from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as lcs
from autoware_carla_scenario.sweeper.job_status import (
    DEFAULT_JOB_TIMEOUT,
    FAILED_VERDICT_EXIT_CODE,
    MAX_JOB_TIMEOUT,
    REFUSED_EXIT_CODE,
    verdict_exit_code,
    whole_job_seconds,
)


def test_the_constants_keep_their_values() -> None:
    assert DEFAULT_JOB_TIMEOUT == 120
    assert MAX_JOB_TIMEOUT == 2**31 - 1
    assert REFUSED_EXIT_CODE == 2
    assert FAILED_VERDICT_EXIT_CODE == 3


def test_the_sweeper_keeps_its_private_names() -> None:
    assert lcs._DEFAULT_JOB_TIMEOUT == job_status.DEFAULT_JOB_TIMEOUT
    assert lcs._MAX_JOB_TIMEOUT == job_status.MAX_JOB_TIMEOUT
    assert lcs._REFUSED_EXIT_CODE == job_status.REFUSED_EXIT_CODE
    assert lcs._FAILED_VERDICT_EXIT_CODE == job_status.FAILED_VERDICT_EXIT_CODE


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, 0), (1.0, 1), (30.0, 30), (900.0, 900), (float(2**31 - 1), 2**31 - 1)],
)
def test_whole_job_seconds_takes_a_whole_number_in_range(
    seconds: float, expected: int
) -> None:
    whole = whole_job_seconds(seconds)
    assert whole == expected
    assert isinstance(whole, int)


@pytest.mark.parametrize(
    "seconds",
    [
        0.5,
        90.5,
        -1.0,
        -0.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        float(2**31),
        1e10,
    ],
)
def test_whole_job_seconds_refuses_the_rest(seconds: float) -> None:
    assert whole_job_seconds(seconds) is None


@pytest.mark.parametrize(("verdict", "expected"), [(True, 0), (False, 3), (None, 1)])
def test_verdict_exit_code(verdict: bool | None, expected: int) -> None:
    assert verdict_exit_code(verdict) == expected
