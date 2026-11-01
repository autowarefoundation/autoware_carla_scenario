"""An AI-walked pedestrian waits at the kerb while the vehicles have the light.

typesafe_carla's ``TrafficLight.get_state()`` returns a plain int, whose
``str()`` is its value ("2"), so the state has to be read by name through
:func:`~autoware_carla_scenario.utils.traffic_light.traffic_light_state_name`.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.entity import _walker_ai

_DT_S = 0.05


class _Light:
    """A traffic light whose ``get_state()`` gives what typesafe_carla gives."""

    id = 7

    def __init__(self, state: Any) -> None:
        self._state = state

    def get_state(self) -> Any:
        return self._state


class _BrokenLight(_Light):
    def get_state(self) -> Any:
        raise RuntimeError("the light is gone")


def _walker() -> Any:
    world = SimpleNamespace(
        get_snapshot=lambda: SimpleNamespace(
            timestamp=SimpleNamespace(delta_seconds=_DT_S)
        )
    )
    return SimpleNamespace(id=1, get_world=lambda: world)


def _driven(light: _Light) -> Any:
    driven = _walker_ai._Driven(None, [], None)  # type: ignore[arg-type]
    driven.kerb_light_read = True
    driven.kerb_light = light  # type: ignore[assignment]
    return driven


def _wait(driven: Any) -> bool:
    return _walker_ai._kerb_wait(_walker(), driven, None, 0.0, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["Green", "Yellow"])
def test_a_walker_waits_while_the_vehicles_have_the_light(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    state = int(getattr(carla.TrafficLightState, name))
    driven = _driven(_Light(state))
    with caplog.at_level(logging.INFO, logger=_walker_ai.__name__):
        assert _wait(driven)
        assert _wait(driven)
    assert driven.kerb_waited == pytest.approx(2 * _DT_S)
    assert f"waits at the kerb for the light (7) {name} for the vehicles" in caplog.text


@pytest.mark.parametrize("name", ["Red", "Off"])
def test_a_walker_does_not_wait_on_a_red_or_dark_light(name: str) -> None:
    driven = _driven(_Light(int(getattr(carla.TrafficLightState, name))))
    assert not _wait(driven)
    assert driven.kerb_waited == 0.0


def test_a_light_that_cannot_be_read_does_not_hold_the_walker() -> None:
    driven = _driven(_BrokenLight(0))
    assert not _wait(driven)
    assert driven.kerb_waited == 0.0
