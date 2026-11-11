"""traffic.sumo.signal_states: the same states as the backend set before the split.

Each ``_before_*`` below is the code as it stood in ``traffic/sumo/backend.py``
before it moved out (#811), verbatim but for reading the CARLA light's state
name and SUMO's state from arguments instead of the light and TraCI.  A CARLA
light switched ``Off`` and the links switched off by one (``o``, ``O``) are
handled since (#2061), so the comparisons leave them out and the tests at the
end cover them.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.traffic.sumo.signal_states import (
    CARLA_TO_SUMO_SIGNAL,
    SUMO_TO_CARLA_SIGNAL,
    carla_light_state,
    green_together,
    clear_state,
    link_states,
    permissive_links,
    ride_state,
)

_CARLA_TO_SUMO_SIGNAL = {"Red": "r", "Yellow": "y", "Green": "G"}
_SUMO_TO_CARLA_SIGNAL = {
    "r": "Red",
    "u": "Red",
    "s": "Red",
    "y": "Yellow",
    "g": "Green",
    "G": "Green",
    "o": "Off",
    "O": "Off",
}


def _before_ride_state(follows: str, states: list[str]) -> str:
    if follows == "clear":
        return "G" if all(state in "rR" for state in states) else "r"
    if any(state in "Gg" for state in states):
        return "g"  # with the traffic, giving way as a bicycle does
    if any(state in "yY" for state in states):
        return "y"
    return "r"


def _before_green_together(phases: list[str], index: int) -> set[int]:
    together: set[int] = set()
    for state in phases:
        if index < len(state) and state[index] in "Gg":
            together.update(i for i, c in enumerate(state) if c in "Gg")
    return together


def _before_permissive(phases: list[str], switched: set[int]) -> set[int]:
    """``_match_permissive``'s set for one tls."""
    return {
        index
        for state in phases
        for index in switched
        if index < len(state) and state[index] == "g"
    }


def _before_signals_to_sumo(
    current: str,
    lights: dict[int, str],
    permissive: set[int],
    bicycle_rides: list[tuple[int, str, list[int]]],
    crossing_links: list[tuple[int, list[int]]],
) -> str:
    """``_signals_to_sumo``'s loop body for one tls."""
    state = list(current)
    for index, name in lights.items():
        char = _CARLA_TO_SUMO_SIGNAL.get(name)
        if char is None:
            continue
        if char == "G" and index in permissive:
            char = "g"  # a permissive green in SUMO's program stays one
        state[index] = char
    for index, follows, links in bicycle_rides:
        state[index] = _before_ride_state(follows, [state[i] for i in links])
    for index, foes in crossing_links:
        clear = all(state[foe] in "rR" for foe in foes)
        state[index] = "G" if clear else "r"
    return "".join(state)


def test_the_signal_tables_are_unchanged() -> None:
    # Off is SUMO's "off, blinking" since #2061.
    assert CARLA_TO_SUMO_SIGNAL == {**_CARLA_TO_SUMO_SIGNAL, "Off": "o"}
    assert SUMO_TO_CARLA_SIGNAL == _SUMO_TO_CARLA_SIGNAL


def test_every_sumo_signal_state_shows_as_a_carla_light_state() -> None:
    """Red-yellow and a stop sign stop as red does, a light switched off is
    Off: none leaves the CARLA light as it was (#1988)."""
    import typesafe_carla.carla as carla

    for state in "rugGysoO":
        name = SUMO_TO_CARLA_SIGNAL[state]
        assert hasattr(carla.TrafficLightState, name)
    assert SUMO_TO_CARLA_SIGNAL["u"] == "Red"
    assert SUMO_TO_CARLA_SIGNAL["s"] == "Red"
    assert SUMO_TO_CARLA_SIGNAL["o"] == SUMO_TO_CARLA_SIGNAL["O"] == "Off"


@pytest.mark.parametrize(
    ("states", "shown"),
    [
        ("G", "Green"),
        ("GO", "Green"),
        ("Go", "Green"),
        ("Gs", "Green"),
        ("sG", "Green"),
        ("OG", "Green"),
        ("rG", "Green"),  # the last regular link, as before
        ("Gr", "Red"),
        ("Gu", "Red"),
        ("uy", "Yellow"),
        ("O", "Off"),
        ("so", "Off"),  # no regular link: the last of the others
        ("os", "Red"),
        ("", None),
        ("R", None),  # not a SUMO state
    ],
)
def test_a_light_shows_its_regular_links_over_off_and_turn_on_red(
    states: str, shown: str | None
) -> None:
    """A light off, or a right turn on red, on one of a light's links does
    not make it show Off or Red over a green on another (#1988)."""
    assert carla_light_state(states) == shown


def test_a_carla_light_over_several_sumo_links_shows_their_regular_state() -> None:
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    class Light:
        def __init__(self) -> None:
            self.state = carla.TrafficLightState.Red

        def get_state(self) -> Any:
            return self.state

        def set_state(self, state: Any) -> None:
            self.state = state

    for links, shown in (
        ("GO", carla.TrafficLightState.Green),
        ("Gs", carla.TrafficLightState.Green),
        ("sG", carla.TrafficLightState.Green),
        ("oO", carla.TrafficLightState.Off),
    ):
        backend = SumoTrafficBackend()
        light = Light()
        backend._signals = {("j", 0): light, ("j", 1): light}
        backend._traci = SimpleNamespace(
            trafficlight=SimpleNamespace(
                getRedYellowGreenState=lambda tls, links=links: links
            )
        )
        backend._signals_to_carla()
        assert int(light.state) == int(shown), links


def _state(rng: random.Random, size: int, alphabet: str = "rRyYgGos") -> str:
    return "".join(rng.choice(alphabet) for _ in range(size))


#: Link states but those of a signal switched off, which #2061 changed.
_LIT = "rRyYgGs"


@pytest.mark.parametrize("seed", range(20))
def test_link_states_sets_what_the_backend_set(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        size = rng.randint(1, 12)
        links = list(range(size))
        current = _state(rng, size, _LIT)
        names = ["Red", "Yellow", "Green", "Unknown"]
        lights = {i: rng.choice(names) for i in rng.sample(links, rng.randint(0, size))}
        permissive = set(rng.sample(links, rng.randint(0, size)))
        rides: list[tuple[int, str, list[int]]] = [
            (
                rng.choice(links),
                rng.choice(["with", "clear"]),
                rng.sample(links, rng.randint(0, size)),
            )
            for _ in range(rng.randint(0, 3))
        ]
        crossings: list[tuple[int, list[int]]] = [
            (rng.choice(links), rng.sample(links, rng.randint(0, size)))
            for _ in range(rng.randint(0, 3))
        ]
        args: tuple[Any, ...] = (current, lights, permissive, rides, crossings)
        assert link_states(*args) == _before_signals_to_sumo(*args)


def test_link_states_keeps_a_permissive_green_after_a_red() -> None:
    """#807: a permissive link gets ``g`` on every CARLA Green, whatever the
    state SUMO shows."""
    shown = "Gg"
    seen = []
    for name in ("Green", "Yellow", "Red", "Green"):
        shown = link_states(shown, {0: name, 1: name}, {1}, [], [])
        seen.append(shown)
    assert seen == ["Gg", "yy", "rr", "Gg"]


@pytest.mark.parametrize("seed", range(10))
def test_the_program_reads_are_what_the_backend_read(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        phases = [_state(rng, rng.randint(0, 8)) for _ in range(rng.randint(0, 6))]
        switched = set(rng.sample(range(10), rng.randint(0, 10)))
        assert permissive_links(phases, switched) == _before_permissive(
            phases, switched
        )
        index = rng.randint(0, 9)
        assert green_together(phases, index) == _before_green_together(phases, index)


@pytest.mark.parametrize("follows", ["with", "clear", "other"])
def test_ride_state_is_unchanged(follows: str) -> None:
    rng = random.Random(follows)
    for _ in range(200):
        states = list(_state(rng, rng.randint(0, 5), _LIT))
        assert ride_state(follows, states) == _before_ride_state(follows, states)


def test_a_carla_light_switched_off_is_off_in_sumo() -> None:
    """#2061: a dark junction is SUMO's "off, blinking" (``o``), whose
    vehicles give way, not the colour the link had last; Unknown says
    nothing, so the link keeps its state."""
    assert link_states("GGrr", {0: "Off", 2: "Off"}, set(), [], []) == "oGor"
    # A permissive link is off as any other.
    assert link_states("rrgG", {0: "Off", 2: "Off"}, {2}, [], []) == "oroG"
    assert link_states("Gr", {0: "Unknown", 1: "Unknown"}, set(), [], []) == "Gr"
    # And back on.
    assert link_states("oo", {0: "Green", 1: "Red"}, set(), [], []) == "Gr"


def test_off_round_trips_between_carla_and_sumo() -> None:
    assert SUMO_TO_CARLA_SIGNAL[CARLA_TO_SUMO_SIGNAL["Off"]] == "Off"
    assert carla_light_state(CARLA_TO_SUMO_SIGNAL["Off"]) == "Off"


@pytest.mark.parametrize(
    ("states", "shown"),
    [
        ([], "G"),
        (["r", "R"], "G"),
        (["o"], "o"),
        (["r", "O"], "o"),
        (["o", "G"], "r"),
        (["o", "y"], "r"),
        (["g"], "r"),
    ],
)
def test_a_crossing_gives_way_at_a_dark_junction(states: list[str], shown: str) -> None:
    """A crossing, or a bicycle riding across, walks while the traffic it
    crosses is red, and gives way while that traffic is red or dark."""
    assert clear_state(states) == shown
    assert ride_state("clear", states) == shown
    crossings = [(len(states), list(range(len(states))))]
    assert link_states("".join(states) + "r", {}, set(), [], crossings)[-1] == shown


@pytest.mark.parametrize(
    ("states", "shown"),
    [
        (["o"], "o"),
        (["o", "r"], "o"),
        (["o", "G"], "g"),
        (["O", "y"], "y"),
        (["r"], "r"),
        ([], "r"),
    ],
)
def test_a_bicycle_riding_with_dark_traffic_gives_way(
    states: list[str], shown: str
) -> None:
    assert ride_state("with", states) == shown
