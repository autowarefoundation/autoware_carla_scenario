"""The versioned cover-item names (``coverage.item_names``, #824), against
what ``coverage.collector`` computed before they were split out, and that
``coverage.collector`` still re-exports them."""

from __future__ import annotations

import itertools
from typing import Callable

import pytest

from autoware_carla_scenario.coverage import collector, item_names
from autoware_carla_scenario.coverage.item_names import (
    VERSION_MARK,
    _split_versioned,
    versioned_item_name,
)

# -- What coverage.collector computed before #824, verbatim ------------------

_MASTER_VERSION_MARK = "@v"


def _master_versioned_item_name(name: str, version: int) -> str:
    if version <= 1:
        return name
    return f"{name}{_MASTER_VERSION_MARK}{version}"


def _master_split_versioned(name: str) -> tuple[str, int, str]:
    rest = ""
    hash_at = name.find("#")
    if hash_at >= 0:
        name, rest = name[:hash_at], name[hash_at:]
    mark_at = name.rfind(_MASTER_VERSION_MARK)
    if mark_at >= 0 and name[mark_at + len(_MASTER_VERSION_MARK) :].isdigit():
        return (
            name[:mark_at],
            int(name[mark_at + len(_MASTER_VERSION_MARK) :]),
            rest,
        )
    return name, 1, rest


def _outcome(split: Callable[[str], tuple[str, int, str]], name: str) -> object:
    """What *split* returns, or the type of what it raises."""
    try:
        return split(name)
    except Exception as exc:  # noqa: BLE001 - compared, not handled
        return type(exc)


def test_the_mark_is_masters() -> None:
    assert VERSION_MARK == _MASTER_VERSION_MARK


def test_collector_re_exports_them() -> None:
    assert collector.VERSION_MARK == VERSION_MARK
    assert collector.versioned_item_name is item_names.versioned_item_name
    assert collector._split_versioned is item_names._split_versioned
    assert "versioned_item_name" in collector.__all__


_NAMES = [
    "",
    "odd.dynamic.vehicle_ahead_gap",
    "odd.dynamic.vehicle_ahead_gap@v2",
    "a@v",
    "a@v0",
    "a@v12",
    "a@vx",
    "a@v2@v3",
    "a@v2#3",
    "a#3",
    "a#3@v2",
    "#",
    "#@v2",
    "@v2",
    "@v2#",
    "a@v-2",
    "a@v 2",
    "a@v²",
    "a@v٣",
    "a@v2##4",
    "x.y@V2",
]


@pytest.mark.parametrize(
    ("name", "version"), itertools.product(_NAMES, [-1, 0, 1, 2, 3, 10, 123])
)
def test_versioned_item_name_is_masters(name: str, version: int) -> None:
    assert versioned_item_name(name, version) == _master_versioned_item_name(
        name, version
    )


# Names whose version is a non-ASCII digit, which master read as one (``٣``)
# or raised on (``²``: a digit to ``str.isdigit()`` but not to ``int()``).
_NON_ASCII = [name for name in _NAMES if not name.isascii()]


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in _NAMES
        + [
            versioned_item_name(n, v) + suffix
            for n, v, suffix in itertools.product(_NAMES, [1, 2, 7], ["", "#2", "#10"])
        ]
        if name.isascii()
    ],
)
def test_split_versioned_is_masters(name: str) -> None:
    assert _outcome(_split_versioned, name) == _outcome(_master_split_versioned, name)


def test_a_versioned_name_splits_back() -> None:
    assert _split_versioned(versioned_item_name("odd.x", 3) + "#2") == (
        "odd.x",
        3,
        "#2",
    )
    assert _split_versioned(versioned_item_name("odd.x", 1)) == ("odd.x", 1, "")


@pytest.mark.parametrize("name", _NON_ASCII)
def test_a_non_ascii_digit_is_no_version(name: str) -> None:
    """#1612: only an ASCII number is a version; anything else is part of
    the name, as ``a@vx`` is."""
    assert _split_versioned(name) == (name, 1, "")
    assert _split_versioned(name + "#2") == (name, 1, "#2")
