"""A Lanelet2 light no OpenDRIVE controller is named after (#901).

On a roadgen map the action found such a light by its stop line (#659), but
the condition and the signal controllers did not: a condition on the same id
never fired, and a controller's phases drove nothing.  All three now resolve a
light through :func:`find_traffic_lights_for_lanelet2_id`; these tests pin
that they agree.  CARLA is faked.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.actions.traffic_signal import TrafficSignalAction
from autoware_carla_scenario.conditions.traffic_signal import TrafficSignalCondition
from autoware_carla_scenario.coordinate.traffic_light import (
    find_traffic_lights_for_lanelet2_id,
)
from autoware_carla_scenario.signals import Phase, PhaseState, SignalController
from autoware_carla_scenario.utils.light_lanes import LightLanes

_MODULE = "autoware_carla_scenario.coordinate.traffic_light"
_PATCH_CONTROLLER_ID = f"{_MODULE}.lanelet2_traffic_light_id_to_opendrive_controller_id"
_PATCH_SIGNAL_IDS = f"{_MODULE}.get_signal_ids_for_controller"
_PATCH_MAP = "autoware_carla_scenario.coordinate.traffic_light._loaded_map"
_PATCH_LANES_OF = "autoware_carla_scenario.utils.light_lanes._lanes_of"

_RED = carla.TrafficLightState.Red
_GREEN = carla.TrafficLightState.Green


def _lanes(*lanelets: tuple[int, float, float, tuple[int, ...]]) -> LightLanes:
    """Lanelets of a road from CARLA y -inf to +inf, x 0..7: each
    ``(id, y from, y to, traffic light elements)``, one after the other.
    Map frame, as the map is read: CARLA's y flipped."""
    ids = tuple(lanelet[0] for lanelet in lanelets)
    outlines = tuple(
        np.array([(0.0, -y0), (0.0, -y1), (7.0, -y1), (7.0, -y0)])
        for _, y0, y1, _ in lanelets
    )
    return LightLanes(
        ids=ids,
        outlines=outlines,
        boxes=np.array([[*o.min(axis=0), *o.max(axis=0)] for o in outlines]),
        lengths={i: y1 - y0 for i, y0, y1, _ in lanelets},
        elements={i: elements for i, _, _, elements in lanelets},
        successors={i: ids[k + 1 : k + 2] for k, i in enumerate(ids)},
    )


#: A northbound road: 60 m and 30 m of lanes, light 77's junction from y = 0,
#: then the road on.  CARLA's stop waypoints stand where the junction starts,
#: or before the map's stop line (on Town10HD_Opt, up to 6.7 m).
_ROAD = (
    _lanes(
        (8, -150.0, -90.0, ()),
        (9, -90.0, -30.0, ()),
        (10, -30.0, 0.0, ()),
        (11, 0.0, 15.0, (77,)),
        (12, 15.0, 40.0, ()),
    ),
    (0.0, 0.0),
)


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


@contextmanager
def _on(road: object) -> Iterator[None]:
    """The run's map is *road* (lanes, offset), or ``None`` for none: its
    lanes are the ones given, not read of a Lanelet2 map."""
    with (
        patch(_PATCH_MAP, return_value=road),
        patch(_PATCH_LANES_OF, side_effect=lambda lanes: lanes),
    ):
        yield


@pytest.fixture
def roadgen() -> Iterator[tuple[_World, _Light, _Light]]:
    """A roadgen map: light 77 has no controller, only its lanelets.

    Yields the world, the light whose lane stops on 77's junction, and the
    road beside's light, whose lane stops off 77's lanelets.
    """
    own = _Light(3.5, 0.5, 0.0, 1.0)
    beside = _Light(8.0, 1.0, -1.0, 0.0)
    world = _World([own, beside])
    with (
        patch(_PATCH_CONTROLLER_ID, return_value=None),
        _on(_ROAD),
    ):
        yield world, own, beside


class TestResolver:
    def test_finds_the_light_on_its_lanelets(
        self, roadgen: tuple[_World, _Light, _Light]
    ) -> None:
        world, own, _beside = roadgen
        assert find_traffic_lights_for_lanelet2_id(world, 77) == [own]

    def test_an_id_no_lanelet_carries_resolves_to_nothing(
        self, roadgen: tuple[_World, _Light, _Light]
    ) -> None:
        world, _own, _beside = roadgen
        assert find_traffic_lights_for_lanelet2_id(world, 78) == []

    def test_a_controller_named_map_does_not_look_at_lanelets(self) -> None:
        named = _Light(opendrive_id="sig_1")
        other = _Light(3.5, 0.5, 0.0, 1.0, opendrive_id="sig_2")
        lanes = MagicMock(return_value=_ROAD)
        with (
            patch(_PATCH_CONTROLLER_ID, return_value=100),
            patch(_PATCH_SIGNAL_IDS, return_value=["sig_1"]),
            patch(_PATCH_MAP, lanes),
        ):
            found = find_traffic_lights_for_lanelet2_id(_World([named, other]), 77)
        assert found == [named]
        lanes.assert_not_called()

    def test_no_map_resolves_to_nothing(self) -> None:
        with (
            patch(_PATCH_CONTROLLER_ID, return_value=None),
            _on(None),
        ):
            assert find_traffic_lights_for_lanelet2_id(_World([_Light(3.5)]), 77) == []


def _resolve(lights: List[_Light]) -> List[_Light]:
    """Light 77's lights among *lights*, on a map that names no controller."""
    with (
        patch(_PATCH_CONTROLLER_ID, return_value=None),
        _on(_ROAD),
    ):
        return find_traffic_lights_for_lanelet2_id(_World(lights), 77)


class TestResolverByLanes:
    """A light is 77's when its stop waypoint is on 77's lanelets or on the
    lanelets before them, up to 50 m back: not by how near it stands to the
    stop line (on Town10HD_Opt, CARLA's stand 0.7 m to 6.7 m before it)."""

    def test_a_stop_before_the_stop_line(self) -> None:
        own = _Light(3.5, -6.7, 0.0, 1.0)
        assert _resolve([own]) == [own]

    def test_a_stop_on_the_lanelet_before_that(self) -> None:
        own = _Light(3.5, -60.0, 0.0, 1.0)
        assert _resolve([own]) == [own]

    def test_not_one_more_than_50_m_of_lanes_back(self) -> None:
        assert _resolve([_Light(3.5, -120.0, 0.0, 1.0)]) == []

    def test_not_one_past_the_lanelets_77_governs(self) -> None:
        assert _resolve([_Light(3.5, 20.0, 0.0, 1.0)]) == []

    def test_not_one_beside_the_road(self) -> None:
        assert _resolve([_Light(9.5, -4.0, 0.0, 1.0)]) == []

    def test_a_light_before_another_s_junction_is_that_one_s(self) -> None:
        nearer = _lanes(
            (9, -90.0, -30.0, ()),
            (10, -30.0, 0.0, (55,)),
            (11, 0.0, 15.0, (77,)),
        )
        light = _Light(3.5, -40.0, 0.0, 1.0)
        with (
            patch(_PATCH_CONTROLLER_ID, return_value=None),
            _on((nearer, (0.0, 0.0))),
        ):
            assert find_traffic_lights_for_lanelet2_id(_World([light]), 77) == []
            assert find_traffic_lights_for_lanelet2_id(_World([light]), 55) == [light]


class TestConditionByLanes:
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


class TestConditionAfterWorldReload:
    """A condition the scenario's ``__init__`` registered outlives a run, and
    the world is reloaded between runs: its lights come back as new actors.
    The condition cached signal ids before #901, which a reload keeps; the
    actor ids it caches now are resolved again when they are gone."""

    def test_controller_named_map(self) -> None:
        condition = TrafficSignalCondition(77, _RED, label="reload_named")
        first = _Light(opendrive_id="sig_1")
        first.state = _RED
        with (
            patch(_PATCH_CONTROLLER_ID, return_value=100),
            patch(_PATCH_SIGNAL_IDS, return_value=["sig_1"]),
        ):
            result = condition.check(_World([first]), elapsed=0.0)
            assert result is not None and result.passed

            reloaded = _Light(opendrive_id="sig_1")
            reloaded.state = _RED
            result = condition.check(_World([reloaded]), elapsed=0.0)
        assert result is not None and result.passed
        assert condition.get_details()["cached_actor_ids"] == [reloaded.id]

    def test_roadgen_map(self, roadgen: tuple[_World, _Light, _Light]) -> None:
        world, own, _beside = roadgen
        own.state = _RED
        condition = TrafficSignalCondition(77, _RED, label="reload_roadgen")
        result = condition.check(world, elapsed=0.0)
        assert result is not None and result.passed

        reloaded = _Light(3.5, 0.5, 0.0, 1.0)
        reloaded.state = _GREEN
        reloaded_world = _World([reloaded])
        assert condition.check(reloaded_world, elapsed=0.0) is None
        reloaded.state = _RED
        result = condition.check(reloaded_world, elapsed=1.0)
        assert result is not None and result.passed


class TestControllerByLanes:
    def test_phases_drive_the_light_on_its_lanelets(
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
