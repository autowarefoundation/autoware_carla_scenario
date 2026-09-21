"""The junction phase: one approach green, the rest of its group red.

CARLA's light group is faked.  What is under test is the partition -- that the
action applies it and the condition insists on the whole of it -- because the
partition is the entire reason for this pair to exist beside the single-signal
action and condition.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock, patch

import carla
import pytest

from autoware_carla_scenario import (
    TrafficSignalControllerAction,
    TrafficSignalControllerCondition,
)

_GREEN = carla.TrafficLightState.Green
_RED = carla.TrafficLightState.Red

_NORTH = 1001
"""Lanelet2 regulatory element id of the north approach."""


def _light(actor_id: int, state: "carla.TrafficLightState" = _RED) -> MagicMock:
    light = MagicMock()
    light.id = actor_id
    light.get_state.return_value = state
    light.get_opendrive_id.return_value = str(actor_id)
    return light


@pytest.fixture
def junction() -> Iterator[dict[str, MagicMock]]:
    """A junction of four lights, of which one is the north approach."""
    lights = {name: _light(id_) for name, id_ in zip("nesw", (1, 2, 3, 4))}
    group = list(lights.values())
    for light in group:
        light.get_group_traffic_lights.return_value = group

    def _find(world: object, lanelet2_id: int) -> list[MagicMock]:
        return [lights["n"]] if lanelet2_id == _NORTH else []

    with (
        patch(
            "autoware_carla_scenario.actions.traffic_signal_controller"
            ".find_traffic_lights_for_lanelet2_id",
            _find,
        ),
        patch(
            "autoware_carla_scenario.conditions.traffic_signal_controller"
            ".find_traffic_lights_for_lanelet2_id",
            _find,
        ),
    ):
        yield lights


class TestAction:
    def test_the_named_approach_goes_green_and_the_rest_red(
        self, junction: dict[str, MagicMock]
    ) -> None:
        TrafficSignalControllerAction(green_lanelet2_id=_NORTH).execute(MagicMock())

        junction["n"].set_state.assert_called_once_with(_GREEN)
        for name in "esw":
            junction[name].set_state.assert_called_once_with(_RED)

    def test_the_whole_junction_is_frozen_by_default(
        self, junction: dict[str, MagicMock]
    ) -> None:
        """A phase the simulator then cycles away from cannot be asserted about."""
        TrafficSignalControllerAction(green_lanelet2_id=_NORTH).execute(MagicMock())

        for light in junction.values():
            light.freeze.assert_called_once_with(True)

    def test_freezing_can_be_declined(self, junction: dict[str, MagicMock]) -> None:
        TrafficSignalControllerAction(green_lanelet2_id=_NORTH, freeze=False).execute(
            MagicMock()
        )

        for light in junction.values():
            light.freeze.assert_called_once_with(False)

    def test_an_unknown_approach_touches_nothing(
        self, junction: dict[str, MagicMock]
    ) -> None:
        TrafficSignalControllerAction(green_lanelet2_id=9999).execute(MagicMock())

        for light in junction.values():
            light.set_state.assert_not_called()


class TestCondition:
    def test_it_holds_when_the_whole_partition_holds(
        self, junction: dict[str, MagicMock]
    ) -> None:
        junction["n"].get_state.return_value = _GREEN
        condition = TrafficSignalControllerCondition(
            green_lanelet2_id=_NORTH, label="north_phase"
        )
        result = condition.check(MagicMock(), 1.0)
        assert result is not None
        assert result.passed

    def test_a_conflicting_green_is_not_the_phase(
        self, junction: dict[str, MagicMock]
    ) -> None:
        """The case a single-signal check cannot see.

        "Green for us" is true in both worlds; "green for us and red for the
        crossing traffic" is true only in one, and only the second is a
        junction state a real road produces.
        """
        junction["n"].get_state.return_value = _GREEN
        junction["e"].get_state.return_value = _GREEN
        condition = TrafficSignalControllerCondition(
            green_lanelet2_id=_NORTH, label="north_phase"
        )
        assert condition.check(MagicMock(), 1.0) is None

    def test_the_named_approach_being_red_is_not_the_phase(
        self, junction: dict[str, MagicMock]
    ) -> None:
        condition = TrafficSignalControllerCondition(
            green_lanelet2_id=_NORTH, label="north_phase"
        )
        assert condition.check(MagicMock(), 1.0) is None

    def test_an_unresolvable_approach_is_unknown_rather_than_wrong(
        self, junction: dict[str, MagicMock]
    ) -> None:
        """The map may not be loaded yet, which is not the phase being wrong."""
        condition = TrafficSignalControllerCondition(
            green_lanelet2_id=9999, label="missing"
        )
        assert condition.check(MagicMock(), 1.0) is None

    def test_details_name_the_approach(self) -> None:
        condition = TrafficSignalControllerCondition(
            green_lanelet2_id=_NORTH, label="north_phase"
        )
        assert condition.get_details() == {"green_lanelet2_id": _NORTH}
