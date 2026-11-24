"""An action still naming an entity that is not in the world -- despawned,
or never spawned -- says so once, not on every tick it executes (#2430),
as ``actions/despawn.py`` promises."""

from __future__ import annotations

import logging
from typing import Callable
from unittest.mock import MagicMock

import pytest

from autoware_carla_scenario import (
    EmergencyAction,
    LaneChangeAction,
    PerceptionFaultInjectionAction,
    SetSpeedAction,
    TurnIndicatorAction,
    TurnIndicatorState,
)
from autoware_carla_scenario.actions.base import BaseAction
from autoware_carla_scenario.traffic import LaneChangeDirection


def _world() -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = []
    return world


def _not_found(caplog: pytest.LogCaptureFixture) -> int:
    return sum(
        1
        for record in caplog.records
        if record.levelno == logging.WARNING and "not found" in record.getMessage()
    )


_ACTIONS: dict[str, Callable[[], BaseAction]] = {
    # Reissued every tick of its run, which never ends without the vehicle.
    "set_speed": lambda: SetSpeedAction(
        entity_name="gone", target_speed_kmh=30.0, rate_kmh_s=1.0
    ),
    # Instantaneous and repeating: executed on every other tick.
    "turn_indicator": lambda: TurnIndicatorAction(
        "gone", TurnIndicatorState.LEFT, once=False
    ),
    "emergency": lambda: EmergencyAction("gone", once=False),
    # Repeating, on a trigger of its own; its run waits on the vehicle.
    "lane_change": lambda: LaneChangeAction(
        "gone", LaneChangeDirection.LEFT, once=False
    ),
}


@pytest.mark.parametrize("make", list(_ACTIONS.values()), ids=list(_ACTIONS))
def test_said_once_per_run(
    make: Callable[[], BaseAction], caplog: pytest.LogCaptureFixture
) -> None:
    action = make()
    world = _world()
    with caplog.at_level(logging.WARNING):
        for tick in range(40):
            action.tick(world, tick * 0.05)
    assert _not_found(caplog) == 1

    # The next run says it again, once.
    caplog.clear()
    action.close()
    action.reset()
    with caplog.at_level(logging.WARNING):
        for tick in range(40):
            action.tick(world, tick * 0.05)
    assert _not_found(caplog) == 1


def test_each_action_says_it_for_itself(caplog: pytest.LogCaptureFixture) -> None:
    world = _world()
    speed = SetSpeedAction(entity_name="gone", target_speed_kmh=30.0, rate_kmh_s=1.0)
    lane = LaneChangeAction("gone", LaneChangeDirection.LEFT, once=False)
    with caplog.at_level(logging.WARNING):
        for tick in range(10):
            speed.tick(world, tick * 0.05)
            lane.tick(world, tick * 0.05)
    assert _not_found(caplog) == 2


def test_a_perception_fault_on_a_missing_actor_is_said_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    action = PerceptionFaultInjectionAction("gone", once=False)
    world = _world()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            action.execute(world)
    assert _not_found(caplog) == 1
