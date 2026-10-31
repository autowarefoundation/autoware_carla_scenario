"""traffic.sumo.signal_states: the same states as the backend set before the split.

Each ``_before_*`` below is the code as it stood in ``traffic/sumo/backend.py``
before it moved out (#811), verbatim but for reading the CARLA light's state
name and SUMO's state from arguments instead of the light and TraCI.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from autoware_carla_scenario.traffic.sumo.signal_states import (
    CARLA_TO_SUMO_SIGNAL,
    SUMO_TO_CARLA_SIGNAL,
    green_together,
    link_states,
    permissive_links,
    ride_state,
)

_CARLA_TO_SUMO_SIGNAL = {"Red": "r", "Yellow": "y", "Green": "G"}
_SUMO_TO_CARLA_SIGNAL = {"r": "Red", "y": "Yellow", "g": "Green", "G": "Green"}


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
    assert CARLA_TO_SUMO_SIGNAL == _CARLA_TO_SUMO_SIGNAL
    assert SUMO_TO_CARLA_SIGNAL == _SUMO_TO_CARLA_SIGNAL


def _state(rng: random.Random, size: int) -> str:
    return "".join(rng.choice("rRyYgGos") for _ in range(size))


@pytest.mark.parametrize("seed", range(20))
def test_link_states_sets_what_the_backend_set(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        size = rng.randint(1, 12)
        links = list(range(size))
        current = _state(rng, size)
        names = ["Red", "Yellow", "Green", "Off", "Unknown"]
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
        states = list(_state(rng, rng.randint(0, 5)))
        assert ride_state(follows, states) == _before_ride_state(follows, states)
