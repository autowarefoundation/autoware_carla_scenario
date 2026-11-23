"""What a scenario's cuts stop on a HILS device's channel (#2386).

A scenario stops data as a break in the communication path would
(:meth:`~.bridge.HilsBridge.disconnect`): some or all of a device's messages,
or every device's, in one direction or both.  This module holds what needs no
channel: a cut (:class:`_Cut`), the sides of a channel each direction stops
(:data:`_DIRECTION_SIDES`), why a cut is refused (:func:`cut_refusal`,
:func:`messages_refusal`), and what a device's cuts stop together on one side
(:func:`stopped_messages`).  :mod:`.bridge` applies it to hilspy's channels
and re-exports it; it is split out so that the library check compiles it
(docs/typecheck.md, "Checking the framework itself").
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Literal, NamedTuple, Optional, Tuple

from .wire import LINK_DIRECTIONS

__all__ = [
    "Side",
    "cut_refusal",
    "messages_refusal",
    "stopped_messages",
]

Side = Literal["send", "recv"]

#: A channel's two sides, which cuts stop separately.
_SIDES: Tuple[Side, ...] = ("send", "recv")

#: The device channel's sides a ``disconnect``'s direction stops: the bridge
#: sends the ECU its inputs and receives its output.
_DIRECTION_SIDES: Dict[str, FrozenSet[Side]] = {
    "to_ecu": frozenset({"send"}),
    "from_ecu": frozenset({"recv"}),
    "both": frozenset(_SIDES),
}
assert set(_DIRECTION_SIDES) == set(LINK_DIRECTIONS)


class _Cut(NamedTuple):
    """Data a scenario stops (:meth:`~.bridge.HilsBridge.disconnect`)."""

    #: The device, or ``None`` for every device.
    device: Optional[str]
    #: The names of the messages, or ``None`` for everything.
    messages: Optional[FrozenSet[str]]
    #: The channel's sides it stops.
    sides: FrozenSet[Side]


def cut_refusal(
    direction: str, device: Optional[str], devices: List[str]
) -> Optional[str]:
    """Why a cut in *direction* of *device* (every device when ``None``) is
    refused, the bridge having *devices*; ``None`` when it is not."""
    if direction not in LINK_DIRECTIONS:
        return (
            f"direction must be one of {', '.join(LINK_DIRECTIONS)}, "
            f"got {direction!r}"
        )
    if device is not None and device not in devices:
        return (
            f"no device {device!r}; the devices are "
            f"{', '.join(sorted(devices)) or 'none'}"
        )
    return None


def messages_refusal(names: Optional[FrozenSet[str]]) -> Optional[str]:
    """Why a cut of the messages *names* (everything when ``None``) is
    refused: it names none; ``None`` when it is not."""
    if names is not None and not names:
        return "messages names none; null stops everything"
    return None


def stopped_messages(
    cuts: List[_Cut], device: str, side: str
) -> Optional[FrozenSet[str]]:
    """What *cuts* stop together on *side* of *device*'s channel: nothing
    (an empty set), everything (``None``), or these names."""
    names: set[str] = set()
    for c in cuts:
        if not (c.device is None or c.device == device) or side not in c.sides:
            continue
        messages = c.messages
        if messages is None:
            return None
        names.update(messages)
    return frozenset(names)
