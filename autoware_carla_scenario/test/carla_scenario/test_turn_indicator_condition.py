"""TurnIndicatorCondition and TurnIndicatorAction: a vehicle's blinker bits in CARLA."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from autoware_carla_scenario import (
    TurnIndicatorAction,
    TurnIndicatorCondition,
    TurnIndicatorState,
)
from autoware_carla_scenario.utils.turn_indicator import (
    turn_indicator_state,
    with_turn_indicators,
)

# CARLA's VehicleLightState bits.
_POSITION, _LOW_BEAM, _RIGHT_BLINKER, _LEFT_BLINKER = 0x1, 0x2, 0x10, 0x20


def _world(light_state: Optional[int], type_id: str = "vehicle.tesla.model3"):
    """A world holding the ego with *light_state*, or no ego when ``None``."""
    world = MagicMock()
    if light_state is None:
        world.get_actors.return_value = []
        return world
    actor = MagicMock()
    actor.attributes = {"role_name": "ego"}
    actor.type_id = type_id
    # Kept as CARLA keeps it: what was set is what is read back.
    state = [light_state]
    actor.get_light_state.side_effect = lambda: state[0]
    actor.set_light_state.side_effect = lambda value: state.__setitem__(0, value)
    world.get_actors.return_value = [actor]
    return world


@pytest.mark.parametrize(
    ("light_state", "expected"),
    [
        (0, TurnIndicatorState.OFF),
        (_POSITION | _LOW_BEAM, TurnIndicatorState.OFF),
        (_LOW_BEAM | _LEFT_BLINKER, TurnIndicatorState.LEFT),
        (_RIGHT_BLINKER, TurnIndicatorState.RIGHT),
        (_LEFT_BLINKER | _RIGHT_BLINKER, TurnIndicatorState.HAZARD),
    ],
)
def test_the_blinker_bits_name_the_state(
    light_state: int, expected: TurnIndicatorState
) -> None:
    assert turn_indicator_state(light_state) is expected


def test_passes_while_the_vehicle_shows_the_state() -> None:
    condition = TurnIndicatorCondition("ego", TurnIndicatorState.LEFT, label="left")
    result = condition.check(_world(_LOW_BEAM | _LEFT_BLINKER), elapsed=2.0)
    assert result is not None and result.passed
    assert "left" in result.message
    assert condition.check(_world(_RIGHT_BLINKER), elapsed=2.1) is None


def test_off_means_neither_indicator() -> None:
    condition = TurnIndicatorCondition("ego", TurnIndicatorState.OFF, label="off")
    assert condition.check(_world(_LOW_BEAM), elapsed=0.0) is not None
    assert condition.check(_world(_LEFT_BLINKER), elapsed=0.0) is None


def test_a_missing_entity_or_a_walker_never_passes() -> None:
    condition = TurnIndicatorCondition("ego", TurnIndicatorState.OFF, label="off")
    assert condition.check(_world(None), elapsed=0.0) is None
    walker = _world(0, type_id="walker.pedestrian.0001")
    assert condition.check(walker, elapsed=0.0) is None


def test_details_name_the_state() -> None:
    condition = TurnIndicatorCondition("ego", TurnIndicatorState.RIGHT, label="r")
    details = condition.get_details()
    assert details["state"] == "RIGHT"
    assert details["entity_name"] == "ego"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (TurnIndicatorState.OFF, _LOW_BEAM),
        (TurnIndicatorState.LEFT, _LOW_BEAM | _LEFT_BLINKER),
        (TurnIndicatorState.RIGHT, _LOW_BEAM | _RIGHT_BLINKER),
        (TurnIndicatorState.HAZARD, _LOW_BEAM | _LEFT_BLINKER | _RIGHT_BLINKER),
    ],
)
def test_setting_the_indicators_keeps_the_other_lights(
    state: TurnIndicatorState, expected: int
) -> None:
    assert with_turn_indicators(_LOW_BEAM | _RIGHT_BLINKER, state) == expected


def test_the_action_lights_another_vehicle_s_indicators() -> None:
    world = _world(_LOW_BEAM, type_id="vehicle.audi.tt")
    world.get_actors.return_value[0].attributes = {"role_name": "npc1"}
    action = TurnIndicatorAction("npc1", TurnIndicatorState.RIGHT)
    action.execute(world)

    (npc,) = world.get_actors.return_value
    npc.set_light_state.assert_called_once_with(_LOW_BEAM | _RIGHT_BLINKER)
    shows_right = TurnIndicatorCondition(
        "npc1", TurnIndicatorState.RIGHT, label="npc1_right"
    )
    assert shows_right.check(world, elapsed=0.0) is not None


def test_the_action_skips_a_missing_entity_or_a_walker() -> None:
    action = TurnIndicatorAction("ego", TurnIndicatorState.LEFT)
    action.execute(_world(None))  # nothing to set, nothing raised
    walker = _world(0, type_id="walker.pedestrian.0001")
    action.execute(walker)
    walker.get_actors.return_value[0].set_light_state.assert_not_called()
