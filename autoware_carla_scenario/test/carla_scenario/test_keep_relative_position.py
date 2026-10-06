"""Holding a vehicle at a gap from another one, every tick.

Both vehicles are faked on a straight road along +x with no map loaded, so the
gap is the straight-line offset along the target's heading -- the fallback the
action uses where no road joins the two.  The point is the control loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import KeepRelativePositionAction
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity

_KMH_PER_MS = 3.6


class _Vehicle:
    """A vehicle on the x axis that drives at whatever it is told to."""

    def __init__(self, role_name: str, x: float, speed_ms: float) -> None:
        self.x = x
        self.speed_ms = speed_ms
        self.commanded: list[float] = []
        self.actor = MagicMock()
        self.actor.attributes = {"role_name": role_name}
        self.actor.get_location.side_effect = lambda: carla.Location(self.x, 0.0, 0.0)
        self.actor.get_velocity.side_effect = lambda: carla.Vector3D(
            self.speed_ms, 0.0, 0.0
        )
        transform = MagicMock()
        transform.get_forward_vector.return_value = carla.Vector3D(1.0, 0.0, 0.0)
        self.actor.get_transform.return_value = transform

    def set_desired_speed(self, world: object, speed_kmh: float) -> None:
        self.commanded.append(speed_kmh)
        self.speed_ms = speed_kmh / _KMH_PER_MS


class _World:
    def __init__(self, *vehicles: _Vehicle) -> None:
        self._vehicles = vehicles

    def get_actors(self) -> list[MagicMock]:
        return [v.actor for v in self._vehicles]


@pytest.fixture
def pair() -> Iterator[tuple[_Vehicle, _Vehicle]]:
    ego = _Vehicle("ego", x=0.0, speed_ms=10.0)
    npc = _Vehicle("npc1", x=5.0, speed_ms=10.0)
    register_entity("npc1", npc)
    yield ego, npc
    unregister_entity("npc1")


def _action(
    gap_m: float = 20.0,
    gain: float = 0.5,
    max_speed_kmh: float = 80.0,
    hold_seconds: float = 0.0,
) -> KeepRelativePositionAction:
    return KeepRelativePositionAction(
        "npc1",
        "ego",
        gap_m=gap_m,
        gain=gain,
        max_speed_kmh=max_speed_kmh,
        hold_seconds=hold_seconds,
    )


def _run(
    action, world, ego, npc, seconds: float, start: float = 0.0, step: float = 0.1
) -> float:
    """Tick for *seconds* of simulated time from *start*; return the clock."""
    elapsed = start
    for _ in range(int(round(seconds / step))):
        action.tick(world, elapsed)
        ego.x += ego.speed_ms * step
        npc.x += npc.speed_ms * step
        elapsed += step
    return elapsed


class TestTheCommand:
    def test_the_gap_is_measured_along_the_target_heading(self, pair) -> None:
        ego, npc = pair
        assert _action().gap(_World(ego, npc)) == (5.0, 10.0)

    def test_behind_where_it_should_be_it_speeds_up(self) -> None:
        # 15 m short of a 20 m gap at 0.5 1/s: 7.5 m/s on top of the ego's 10.
        assert _action().command_kmh(5.0, 10.0) == pytest.approx(17.5 * _KMH_PER_MS)

    def test_ahead_of_where_it_should_be_it_slows(self) -> None:
        assert _action().command_kmh(30.0, 10.0) == pytest.approx(5.0 * _KMH_PER_MS)

    def test_in_place_it_matches_the_target(self) -> None:
        assert _action().command_kmh(20.0, 10.0) == pytest.approx(36.0)

    def test_the_command_is_clamped(self) -> None:
        assert _action().command_kmh(200.0, 10.0) == 0.0
        assert _action(max_speed_kmh=50.0).command_kmh(-100.0, 10.0) == 50.0


class TestTheLoop:
    def test_it_closes_on_the_gap_and_holds_it(self, pair) -> None:
        ego, npc = pair
        world = _World(ego, npc)
        action = _action()
        _run(action, world, ego, npc, seconds=20.0)
        assert npc.x - ego.x == pytest.approx(20.0, abs=0.5)
        # Still holding: with no hold time it never completes.
        assert action.state is ActionState.RUNNING

    def test_it_follows_a_target_that_changes_speed(self, pair) -> None:
        ego, npc = pair
        world = _World(ego, npc)
        action = _action(gap_m=-15.0)
        clock = _run(action, world, ego, npc, seconds=10.0)
        ego.speed_ms = 4.0
        _run(action, world, ego, npc, seconds=20.0, start=clock)
        assert npc.x - ego.x == pytest.approx(-15.0, abs=0.5)

    def test_a_hold_ends_and_the_action_completes(self, pair) -> None:
        ego, npc = pair
        world = _World(ego, npc)
        action = _action(hold_seconds=2.0)
        clock = _run(action, world, ego, npc, seconds=1.0)
        assert action.state is ActionState.RUNNING
        _run(action, world, ego, npc, seconds=2.0, start=clock)
        assert action.state is ActionState.COMPLETE


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gain": 0.0}, "gain"),
        ({"max_speed_kmh": -1.0}, "max_speed_kmh"),
        ({"hold_seconds": -1.0}, "hold_seconds"),
    ],
)
def test_nonsense_is_refused(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _action(**kwargs)  # type: ignore[arg-type]
