"""hils.cuts: what a scenario's cuts stop, as the bridge combined them before.

``_old_stopped`` and ``_old_checks`` are ``HilsBridge._apply_cuts()``'s
per-side combination and ``HilsBridge.disconnect()``'s checks as they stood
in ``hils/bridge.py`` before they moved into the library check (#2386),
verbatim but for reading the cuts and the devices from arguments instead of
the bridge.  The new functions, put together as the bridge puts them, must
answer the same for random cuts, devices, sides, directions and messages.
"""

from __future__ import annotations

import random
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Optional, Tuple

import pytest

from autoware_carla_scenario.hils import bridge, cuts
from autoware_carla_scenario.hils.wire import LINK_DIRECTIONS

_OldSide = Literal["send", "recv"]
_OLD_SIDES: Tuple[_OldSide, ...] = ("send", "recv")
_OLD_DIRECTION_SIDES: Dict[str, FrozenSet[_OldSide]] = {
    "to_ecu": frozenset({"send"}),
    "from_ecu": frozenset({"recv"}),
    "both": frozenset(_OLD_SIDES),
}


def _old_stopped(
    all_cuts: Iterable[cuts._Cut], name: str, side: str
) -> Optional[FrozenSet[str]]:
    cut_list = [c for c in all_cuts if c.device in (None, name) and side in c.sides]
    # Nothing (an empty set), everything (None), or these names.
    stopped: Optional[FrozenSet[str]] = (
        None
        if any(c.messages is None for c in cut_list)
        else frozenset().union(*(c.messages or () for c in cut_list))
    )
    return stopped


def _old_checks(
    channels: Dict[str, Any],
    device: Optional[str],
    messages: Any,
    direction: str,
) -> Tuple[str, Any]:
    """The old checks' error, or the names and sides of the cut made."""
    try:
        if direction not in LINK_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {', '.join(LINK_DIRECTIONS)}, "
                f"got {direction!r}"
            )
        if device is not None and device not in channels:
            raise ValueError(
                f"no device {device!r}; the devices are "
                f"{', '.join(sorted(channels)) or 'none'}"
            )
        if messages is not None and (
            isinstance(messages, str) or not isinstance(messages, (list, tuple))
        ):
            raise ValueError("messages must be a list of names")
        names = None if messages is None else frozenset(str(m) for m in messages)
        if names is not None and not names:
            raise ValueError("messages names none; null stops everything")
    except ValueError as exc:
        return ("refused", str(exc))
    return ("cut", (names, _OLD_DIRECTION_SIDES[direction]))


def _new_checks(
    channels: Dict[str, Any],
    device: Optional[str],
    messages: Any,
    direction: str,
) -> Tuple[str, Any]:
    """The same, as HilsBridge.disconnect() now makes them."""
    refusal = cuts.cut_refusal(direction, device, list(channels))
    if refusal is not None:
        return ("refused", refusal)
    if messages is not None and (
        isinstance(messages, str) or not isinstance(messages, (list, tuple))
    ):
        return ("refused", "messages must be a list of names")
    names = None if messages is None else frozenset(str(m) for m in messages)
    refusal = cuts.messages_refusal(names)
    if refusal is not None:
        return ("refused", refusal)
    return ("cut", (names, cuts._DIRECTION_SIDES[direction]))


_DEVICES = ("ecu", "gnss", "lidar", "")
_NAMES = ("a", "b", "c", "speed", "")


def _random_cut(rng: random.Random) -> cuts._Cut:
    device = rng.choice((None,) + _DEVICES + ("other",))
    messages = (
        None
        if rng.random() < 0.3
        else frozenset(rng.sample(_NAMES, rng.randint(0, len(_NAMES))))
    )
    return cuts._Cut(
        device, messages, cuts._DIRECTION_SIDES[rng.choice(LINK_DIRECTIONS)]
    )


def test_the_constants_are_the_same() -> None:
    assert cuts._SIDES == _OLD_SIDES
    assert cuts._DIRECTION_SIDES == _OLD_DIRECTION_SIDES
    assert cuts._Cut._fields == ("device", "messages", "sides")


@pytest.mark.parametrize(
    "name", ["Side", "_SIDES", "_DIRECTION_SIDES", "_Cut", "stopped_messages"]
)
def test_the_bridge_re_exports_them(name: str) -> None:
    assert getattr(bridge, name) is getattr(cuts, name)


def test_the_stopped_messages_are_the_same() -> None:
    rng = random.Random(2386)
    for _ in range(3000):
        made: List[cuts._Cut] = [_random_cut(rng) for _ in range(rng.randint(0, 6))]
        for name in _DEVICES:
            for side in _OLD_SIDES:
                old = _old_stopped(made, name, side)
                new = cuts.stopped_messages(made, name, side)
                assert new == old
                assert type(new) is type(old)


@pytest.mark.parametrize(
    "direction", list(LINK_DIRECTIONS) + ["", "BOTH", "to-ecu", "both "]
)
@pytest.mark.parametrize("device", [None, "ecu", "gnss", "", "missing"])
@pytest.mark.parametrize(
    "messages",
    [
        None,
        [],
        (),
        ["a"],
        ("a", "b", "a"),
        [1, 2.5, None],
        [""],
        "a",
        "",
        {"a"},
        {"a": 1},
        7,
    ],
)
@pytest.mark.parametrize("channels", [{}, {"ecu": 0}, {"gnss": 0, "ecu": 0}])
def test_the_checks_are_the_same(
    channels: Dict[str, Any], device: Optional[str], messages: Any, direction: str
) -> None:
    assert _new_checks(channels, device, messages, direction) == _old_checks(
        channels, device, messages, direction
    )
