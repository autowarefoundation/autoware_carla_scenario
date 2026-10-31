"""A Lanelet2 light no OpenDRIVE controller is named after (#901).

On a roadgen map the action found such a light by its stop line (#659), but
the condition and the signal controllers did not: a condition on the same id
never fired, and a controller's phases drove nothing.  All three now resolve a
light through :func:`find_traffic_lights_for_lanelet2_id`; these tests pin
that they agree.  CARLA is faked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.actions.traffic_signal import TrafficSignalAction
from autoware_carla_scenario.conditions.traffic_signal import TrafficSignalCondition
from autoware_carla_scenario.coordinate.traffic_light import (
    find_traffic_lights_for_lanelet2_id,
)
from autoware_carla_scenario.signals import Phase, PhaseState, SignalController

_MODULE = "autoware_carla_scenario.coordinate.traffic_light"
_PATCH_CONTROLLER_ID = f"{_MODULE}.lanelet2_traffic_light_id_to_opendrive_controller_id"
_PATCH_SIGNAL_IDS = f"{_MODULE}.get_signal_ids_for_controller"
_PATCH_STOP_LINE = f"{_MODULE}.stop_line_in_carla"

_RED = carla.TrafficLightState.Red
_GREEN = carla.TrafficLightState.Green

#: Light 77's stop line runs across a northbound road at y = 0, x 0..7.
_LINE = ((0.0, 0.0), (7.0, 0.0))


@dataclass
class _Vector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class _WaypointTransform:
    location: _Vector
    forward: _Vector

    def get_forward_vector(self) -> _Vector:
        return self.forward


@dataclass
class _Waypoint:
    transform: _WaypointTransform


class _Light:
    """A traffic light whose lane stops at (x, y), heading (fx, fy)."""

    _ids = iter(range(1, 1_000_000))

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        fx: float = 0.0,
        fy: float = 1.0,
        opendrive_id: str = "",
    ) -> None:
        self.id = next(_Light._ids)
        self.type_id = "traffic.traffic_light"
        self.state: object = carla.TrafficLightState.Off
        self.frozen = False
        self._opendrive_id = opendrive_id
        self._stop = _Waypoint(_WaypointTransform(_Vector(x, y), _Vector(fx, fy)))

    def get_opendrive_id(self) -> str:
        return self._opendrive_id

    def get_stop_waypoints(self) -> list:
        return [self._stop]

    def get_state(self) -> object:
        return self.state

    def set_state(self, state: object) -> None:
        self.state = state

    def freeze(self, frozen: bool) -> None:
        self.frozen = frozen


class _Actors:
    def __init__(self, actors: List[_Light]) -> None:
        self._actors = actors

    def filter(self, pattern: str) -> List[_Light]:
        base = pattern.rstrip("*")
        return [a for a in self._actors if a.type_id.startswith(base)]

    def __iter__(self):
        return iter(self._actors)


class _World:
    def __init__(self, actors: List[_Light]) -> None:
        self._actors = actors
        self.now = 0.0

    def get_actors(self) -> _Actors:
        return _Actors(self._actors)

    def get_snapshot(self) -> MagicMock:
        snapshot = MagicMock()
        snapshot.timestamp.elapsed_seconds = self.now
        return snapshot


@pytest.fixture
def roadgen() -> Iterator[tuple[_World, _Light, _Light]]:
    """A roadgen map: light 77 has no controller, only a stop line.

    Yields the world, the light on 77's line, and the road beside's light,
    whose lane stops near the line's end but runs along it.
    """
    own = _Light(3.5, 0.5, 0.0, 1.0)
    beside = _Light(8.0, 1.0, -1.0, 0.0)
    world = _World([own, beside])

    def _line(lanelet2_id: int) -> object:
        return _LINE if lanelet2_id == 77 else None

    with (
        patch(_PATCH_CONTROLLER_ID, return_value=None),
        patch(_PATCH_STOP_LINE, side_effect=_line),
    ):
        yield world, own, beside


class TestResolver:
    def test_finds_the_light_on_the_stop_line(
        self, roadgen: tuple[_World, _Light, _Light]
    ) -> None:
        world, own, _beside = roadgen
        assert find_traffic_lights_for_lanelet2_id(world, 77) == [own]

    def test_an_id_with_no_line_resolves_to_nothing(
        self, roadgen: tuple[_World, _Light, _Light]
    ) -> None:
        world, _own, _beside = roadgen
        assert find_traffic_lights_for_lanelet2_id(world, 78) == []

    def test_a_controller_named_map_does_not_look_at_stop_lines(self) -> None:
        named = _Light(opendrive_id="sig_1")
        other = _Light(3.5, 0.5, 0.0, 1.0, opendrive_id="sig_2")
        stop_line = MagicMock(return_value=_LINE)
        with (
            patch(_PATCH_CONTROLLER_ID, return_value=100),
            patch(_PATCH_SIGNAL_IDS, return_value=["sig_1"]),
            patch(_PATCH_STOP_LINE, stop_line),
        ):
            found = find_traffic_lights_for_lanelet2_id(_World([named, other]), 77)
        assert found == [named]
        stop_line.assert_not_called()


class TestConditionByStopLine:
    def test_fires_on_the_light_the_action_set(
        self, roadgen: tuple[_World, _Light, _Light]
    ) -> None:
        world, own, beside = roadgen
        condition = TrafficSignalCondition(77, _RED, label="red_77")
        assert condition.check(world, elapsed=0.0) is None

        TrafficSignalAction(_RED, [77], label="set_red_77").execute(world)

        assert own.state == _RED and beside.state != _RED
        result = condition.check(world, elapsed=1.0)
        assert result is not None and result.passed
        assert condition.get_details()["cached_actor_ids"] == [own.id]

    def test_reads_the_cached_actor_and_does_not_resolve_again(
        self, roadgen: tuple[_World, _Light, _Light]
    ) -> None:
        world, own, _beside = roadgen
        own.state = _GREEN
        condition = TrafficSignalCondition(77, _RED, label="cached")
        with patch(
            "autoware_carla_scenario.conditions.traffic_signal"
            ".find_traffic_lights_for_lanelet2_id",
            wraps=find_traffic_lights_for_lanelet2_id,
        ) as find:
            assert condition.check(world, elapsed=0.0) is None
            own.state = _RED
            result = condition.check(world, elapsed=1.0)
        assert result is not None and result.passed
        find.assert_called_once()

    def test_an_unresolvable_id_warns_once(
        self,
        roadgen: tuple[_World, _Light, _Light],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        world, _own, _beside = roadgen
        condition = TrafficSignalCondition(78, _RED, label="nowhere_78")
        with caplog.at_level(logging.WARNING):
            for tick in range(5):
                assert condition.check(world, elapsed=float(tick)) is None
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "nowhere_78" in r.getMessage()
        ]
        assert len(warnings) == 1


class TestControllerByStopLine:
    def test_phases_drive_the_light_on_the_stop_line(
        self, roadgen: tuple[_World, _Light, _Light]
    ) -> None:
        world, own, beside = roadgen
        controller = SignalController(
            name="roadgen",
            phases=(
                Phase("go", 2.0, (PhaseState(77, "green"),)),
                Phase("stop", 2.0, (PhaseState(77, "red"),)),
            ),
        )
        controller.tick(world)
        assert own.state == _GREEN and own.frozen

        world.now = 2.0
        controller.tick(world)
        assert own.state == _RED
        assert beside.state == carla.TrafficLightState.Off and not beside.frozen
