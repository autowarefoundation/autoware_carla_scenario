"""The signal states the SUMO backend sets, from CARLA's lights and SUMO's programs.

With ``traffic_light_authority=carla``, CARLA's lights switch the SUMO links
they are matched to, and the links no CARLA light switches -- a bicycle ride's,
a pedestrian crossing's -- follow the vehicle links they go with or cross.
:mod:`.backend` reads the states off TraCI and writes them back; what it
writes is decided here, from strings and indices alone, as are SUMO's step
per CARLA tick and the links a program lets go together.

Pure functions of plain values, so they are tested without either simulator.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "CARLA_TO_SUMO_SIGNAL",
    "SECONDARY_SIGNAL_STATES",
    "SUMO_TO_CARLA_SIGNAL",
    "carla_light_state",
    "green_together",
    "link_states",
    "permissive_links",
    "ride_state",
    "sumo_step",
]

#: CARLA's light state names -> the SUMO link state a CARLA light sets.
CARLA_TO_SUMO_SIGNAL: dict[str, str] = {"Red": "r", "Yellow": "y", "Green": "G"}
#: SUMO's link states -> the CARLA light state name each shows as: red-yellow
#: (``u``, the light about to turn green) and a green right-turn arrow a
#: driver must stop at first (``s``, a right turn on red) stop a driver as
#: red does, and a light switched off (``o`` blinking, ``O`` dark) is
#: CARLA's ``Off``.
SUMO_TO_CARLA_SIGNAL: dict[str, str] = {
    "r": "Red",
    "u": "Red",
    "s": "Red",
    "y": "Yellow",
    "g": "Green",
    "G": "Green",
    "o": "Off",
    "O": "Off",
}


#: The link states :func:`carla_light_state` lets any other state of the same
#: light win over: a light switched off, and a right turn on red, say nothing
#: of the movements the light's other links make.
SECONDARY_SIGNAL_STATES = "soO"


def carla_light_state(states: str) -> Optional[str]:
    """The CARLA light state name of a light switching links in *states*
    (SUMO's states of its links, in the order they are set).

    The last link's with a regular state (``r u y g G``), as a light set once
    per link ends up showing; only a light with none of those shows its last
    :data:`SECONDARY_SIGNAL_STATES` link's.  ``None`` if no link's state
    maps to a CARLA one.
    """
    regular: Optional[str] = None
    secondary: Optional[str] = None
    for state in states:
        name = SUMO_TO_CARLA_SIGNAL.get(state)
        if name is None:
            continue
        if state in SECONDARY_SIGNAL_STATES:
            secondary = name
        else:
            regular = name
    return regular if regular is not None else secondary


def sumo_step(tick: float, step_length: Optional[float]) -> tuple[float, int]:
    """SUMO's step length and how many steps it makes per CARLA *tick*.

    Raises:
        ValueError: If *step_length* is longer than the tick, does not divide
            it, or is not a whole number of milliseconds -- SUMO's clock counts
            those, so it would round the step and drift from CARLA's.  With no
            *step_length*, if the tick itself is not a whole number of
            milliseconds (1/30 s), for the same reason.
    """
    if step_length is None:
        tick_millis = tick * 1000.0
        if abs(tick_millis - round(tick_millis)) > 1e-6:
            raise ValueError(
                f"traffic=sumo needs the world's fixed_delta_seconds ({tick:g} s) "
                "in whole milliseconds: SUMO steps once per CARLA tick, and its "
                "clock counts milliseconds, so it would round the step and drift "
                "from CARLA's"
            )
        return tick, 1
    steps = round(tick / step_length)
    millis = step_length * 1000.0
    if (
        steps < 1
        or abs(steps * step_length - tick) > 1e-6 * max(tick, 1.0)
        or abs(millis - round(millis)) > 1e-6
    ):
        raise ValueError(
            f"traffic.options.step_length_s ({step_length:g} s) must divide the "
            f"world's fixed_delta_seconds ({tick:g} s) in whole milliseconds: "
            "SUMO makes a whole number of steps per CARLA tick, and its clock "
            "counts milliseconds"
        )
    return tick / steps, steps


def green_together(phases: list[str], index: int) -> set[int]:
    """The links green in a phase of *phases* where link *index* is green."""
    together: set[int] = set()
    for state in phases:
        if index < len(state) and state[index] in "Gg":
            for i, c in enumerate(state):
                if c in "Gg":
                    together.add(i)
    return together


def permissive_links(phases: list[str], switched: set[int]) -> set[int]:
    """The links of *switched* that some phase of *phases* shows as a
    permissive green (``g``), also one that is ``G`` in another phase (a
    protected, then permissive turn): a CARLA Green does not say which."""
    permissive: set[int] = set()
    for state in phases:
        for index in switched:
            if index < len(state) and state[index] == "g":
                permissive.add(index)
    return permissive


def ride_state(follows: str, states: list[str]) -> str:
    """The state of a bicycle ride's signal link, from the *states* of the
    vehicle links it *follows* (``with``) or must be clear of (``clear``)."""
    if follows == "clear":
        return "G" if all(state in "rR" for state in states) else "r"
    if any(state in "Gg" for state in states):
        return "g"  # with the traffic, giving way as a bicycle does
    if any(state in "yY" for state in states):
        return "y"
    return "r"


def link_states(
    current: str,
    carla: dict[int, str],
    permissive: set[int],
    rides: list[tuple[int, str, list[int]]],
    crossings: list[tuple[int, list[int]]],
) -> str:
    """The state to set a traffic light's links to.

    Args:
        current: The state SUMO shows now, kept where nothing below sets one.
        carla: Link index -> the state name of the CARLA light switching it;
            a name with no SUMO state (``Off``, ``Unknown``) keeps the link's.
        permissive: The links a CARLA Green sets to a permissive green
            (``g``), as some phase of SUMO's own program shows them
            (:func:`permissive_links`); the rest get ``G``.
        rides: Each bicycle ride's link, how it follows (``with`` or
            ``clear``) and the vehicle links it follows (:func:`ride_state`).
        crossings: Each pedestrian crossing's link and the vehicle links over
            the road it crosses: it walks while every one of them is red.
    """
    state = list(current)
    for index, name in carla.items():
        if name not in CARLA_TO_SUMO_SIGNAL:
            continue
        char = CARLA_TO_SUMO_SIGNAL[name]
        if char == "G" and index in permissive:
            char = "g"  # a permissive green in SUMO's program stays one
        state[index] = char
    # A bicycle ride goes with the traffic beside it, or crosses while the
    # traffic it crosses is held; before the crossings, which wait for it as
    # for any vehicle over their road.
    for index, follows, links in rides:
        state[index] = ride_state(follows, [state[i] for i in links])
    # A crossing walks while nothing drives over the road it crosses: the
    # CARLA lights decide the vehicles, and with them the people.
    for index, foes in crossings:
        clear = all(state[foe] in "rR" for foe in foes)
        state[index] = "G" if clear else "r"
    return "".join(state)
