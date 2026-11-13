"""How the SLURM worker reads and takes a task's ROS domain ids
(``slurm.worker._env_domain_id`` and ``_take_domain``).

The ``_old_*`` oracles below are ``slurm/worker.py``'s ``_env_domain_id()``
and ``_take_domain()`` as they were before they moved into the library check
(#2124), copied verbatim but for their names and docstrings.  What the worker
has now must return the same, raise the same (type and message), and ask the
domains for the same ids in the same order.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from autoware_carla_scenario.constants import MAX_ROS_DOMAIN_ID
from autoware_carla_scenario.slurm import ports, worker
from autoware_carla_scenario.slurm.ports import RosDomains

# -- the oracles: slurm/worker.py as it was ------------------------------------


def _old_env_domain_id(value: str, key: str) -> int:
    text = value.strip()
    if not (text.isascii() and text.isdigit()):
        raise ValueError(
            f"{key} must be a ROS domain id, a whole number from 0 to "
            f"{MAX_ROS_DOMAIN_ID}, got {value!r}"
        )
    return int(text)


def _old_take_domain(domains: RosDomains, wanted: int, key: str) -> int:
    if not 0 <= wanted <= MAX_ROS_DOMAIN_ID:
        raise ValueError(
            f"{key} must be a ROS domain id, a whole number from 0 to "
            f"{MAX_ROS_DOMAIN_ID}, got {wanted}"
        )
    return domains.take(wanted)


# -- helpers --------------------------------------------------------------------


def _outcome(call: Any, *args: Any) -> tuple[Any, ...]:
    try:
        return ("returned", call(*args))
    except BaseException as exc:  # noqa: BLE001 -- compared, old and new
        return ("raised", type(exc), exc.args)


class _Domains:
    """What _take_domain() asks of a RosDomains: take(), recorded."""

    def __init__(self, answer: Any) -> None:
        self.asked: list[int] = []
        self._answer = answer

    def take(self, wanted: int) -> int:
        self.asked.append(wanted)
        if isinstance(self._answer, BaseException):
            raise self._answer
        return int(self._answer(wanted))


_VALUES = [
    "",
    " ",
    "0",
    "7",
    " 7 ",
    "\t42\n",
    " 7　",  # Unicode spaces, which str.strip() removes
    "007",
    str(MAX_ROS_DOMAIN_ID),
    str(MAX_ROS_DOMAIN_ID + 1),
    "999999999999999999999999",
    "-1",
    "+1",
    "1_000",
    "1.0",
    "1e3",
    "0x10",
    "٣",  # ARABIC-INDIC DIGIT THREE: isdigit(), not ASCII
    "²",  # SUPERSCRIPT TWO: isdigit(), int() refuses it
    "⅕",  # VULGAR FRACTION ONE FIFTH
    "1 2",
    "abc",
    "7\x00",
    "'quoted'",
    "a\ud800",
]


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("key", ["ROS_DOMAIN_ID", "env.ROS_DOMAIN_ID", ""])
def test_an_environments_domain_id_is_read_as_it_was(value: str, key: str) -> None:
    assert _outcome(worker._env_domain_id, value, key) == _outcome(
        _old_env_domain_id, value, key
    )


def test_random_domain_ids_are_read_as_they_were() -> None:
    rng = random.Random(2124)
    alphabet = "0123456789 \t\n+-_.xe٣²　a"
    for _ in range(3000):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randrange(6)))
        assert _outcome(worker._env_domain_id, value, "k") == _outcome(
            _old_env_domain_id, value, "k"
        ), value


_WANTED = [
    -(10**30),
    -2,
    -1,
    0,
    1,
    100,
    MAX_ROS_DOMAIN_ID - 1,
    MAX_ROS_DOMAIN_ID,
    MAX_ROS_DOMAIN_ID + 1,
    10**30,
    True,  # a bool is an int: RosDomains.take() refuses it, not this
    False,
]


@pytest.mark.parametrize("wanted", _WANTED)
@pytest.mark.parametrize(
    "answer",
    [
        lambda w: w,
        lambda w: w + 3,
        RuntimeError("no free ROS domain"),
        OSError(28, "no space"),
        KeyboardInterrupt(),
    ],
)
def test_a_domain_is_taken_as_it_was(wanted: int, answer: Any) -> None:
    new, old = _Domains(answer), _Domains(answer)
    assert _outcome(worker._take_domain, new, wanted, "ros.domain") == _outcome(
        _old_take_domain, old, wanted, "ros.domain"
    )
    assert new.asked == old.asked


def test_a_real_domain_is_taken_as_it_was(tmp_path: Any) -> None:
    new = RosDomains(tmp_path / "new")
    old = RosDomains(tmp_path / "old")
    try:
        for wanted in (5, 5, 0, MAX_ROS_DOMAIN_ID, MAX_ROS_DOMAIN_ID + 1, -1, True):
            assert _outcome(worker._take_domain, new, wanted, "k") == _outcome(
                _old_take_domain, old, wanted, "k"
            )
        assert new.mapping == old.mapping
    finally:
        new.release()
        old.release()


def test_the_worker_re_exports_the_checked_functions() -> None:
    # The library check compiles slurm.ports (#2124).
    assert worker._env_domain_id is ports._env_domain_id
    assert worker._take_domain is ports._take_domain
