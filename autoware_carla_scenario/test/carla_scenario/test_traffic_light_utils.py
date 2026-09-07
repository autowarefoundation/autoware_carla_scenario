"""Unit tests for traffic light utility functions (no CARLA required)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from xml.etree import ElementTree

import pytest

from autoware_carla_scenario.coordinate.traffic_light import (
    find_nearest_traffic_light,
)
from autoware_carla_scenario.utils.traffic_light import (
    get_signal_ids_for_controller,
    lanelet2_traffic_light_id_to_opendrive_controller_id,
)


# ---------------------------------------------------------------------------
# Lightweight stubs for CARLA types
# ---------------------------------------------------------------------------


@dataclass
class _FakeLocation:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance(self, other: "_FakeLocation") -> float:
        return (
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        ) ** 0.5


@dataclass
class _FakeTransform:
    location: _FakeLocation


class _FakeTrafficLight:
    """Minimal traffic light stub."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        opendrive_id: str = "",
    ) -> None:
        self._transform = _FakeTransform(location=_FakeLocation(x, y, z))
        self._state: object = None
        self._frozen: bool = False
        self._group: List["_FakeTrafficLight"] = [self]
        self._opendrive_id = opendrive_id
        self.type_id = "traffic.traffic_light"

    def get_transform(self) -> _FakeTransform:
        return self._transform

    def get_group_traffic_lights(self) -> List["_FakeTrafficLight"]:
        return self._group

    def get_opendrive_id(self) -> str:
        return self._opendrive_id

    def set_state(self, state: object) -> None:
        self._state = state

    def freeze(self, frozen: bool) -> None:
        self._frozen = frozen


class _FakeWorld:
    """Minimal CARLA world stub."""

    def __init__(self, actors: List[_FakeTrafficLight]) -> None:
        self._actors = actors

    def get_actors(self) -> "_FakeActorList":
        return _FakeActorList(self._actors)


class _FakeActorList:
    """Stub for CARLA ActorList with filter support."""

    def __init__(self, actors: List[_FakeTrafficLight]) -> None:
        self._actors = actors

    def filter(self, pattern: str) -> List[_FakeTrafficLight]:
        """Filter actors by type_id pattern (simplified glob match)."""
        base = pattern.rstrip("*")
        return [a for a in self._actors if a.type_id.startswith(base)]

    def __iter__(self):
        return iter(self._actors)

    def __len__(self):
        return len(self._actors)


def _make_group(*lights: _FakeTrafficLight) -> List[_FakeTrafficLight]:
    """Link traffic lights into a single group."""
    group = list(lights)
    for tl in group:
        tl._group = group
    return group


# ---------------------------------------------------------------------------
# find_nearest_traffic_light
# ---------------------------------------------------------------------------


class TestFindNearestTrafficLight:
    def test_returns_nearest(self) -> None:
        tl_close = _FakeTrafficLight(1.0, 0.0)
        tl_far = _FakeTrafficLight(100.0, 0.0)
        world = _FakeWorld([tl_far, tl_close])
        origin = _FakeLocation(0.0, 0.0, 0.0)

        nearest, dist = find_nearest_traffic_light(world, origin)

        assert nearest is tl_close
        assert dist == pytest.approx(1.0)

    def test_all_beyond_max_distance(self) -> None:
        tl = _FakeTrafficLight(200.0, 0.0)
        world = _FakeWorld([tl])
        origin = _FakeLocation(0.0, 0.0, 0.0)

        nearest, dist = find_nearest_traffic_light(world, origin, max_distance=50.0)

        assert nearest is None
        assert dist == float("inf")

    def test_empty_world(self) -> None:
        world = _FakeWorld([])
        origin = _FakeLocation(0.0, 0.0, 0.0)

        nearest, dist = find_nearest_traffic_light(world, origin)

        assert nearest is None
        assert dist == float("inf")

    def test_custom_max_distance(self) -> None:
        tl = _FakeTrafficLight(10.0, 0.0)
        world = _FakeWorld([tl])
        origin = _FakeLocation(0.0, 0.0, 0.0)

        nearest, dist = find_nearest_traffic_light(world, origin, max_distance=5.0)
        assert nearest is None

        nearest, dist = find_nearest_traffic_light(world, origin, max_distance=15.0)
        assert nearest is tl

    def test_multiple_lights_selects_closest(self) -> None:
        tl_a = _FakeTrafficLight(5.0, 0.0)
        tl_b = _FakeTrafficLight(3.0, 0.0)
        tl_c = _FakeTrafficLight(8.0, 0.0)
        world = _FakeWorld([tl_a, tl_b, tl_c])
        origin = _FakeLocation(0.0, 0.0, 0.0)

        nearest, dist = find_nearest_traffic_light(world, origin)

        assert nearest is tl_b
        assert dist == pytest.approx(3.0)

    def test_ignores_non_traffic_light_actors(self) -> None:
        tl = _FakeTrafficLight(1.0, 0.0)
        non_tl = _FakeTrafficLight(0.5, 0.0)
        non_tl.type_id = "vehicle.car"
        world = _FakeWorld([non_tl, tl])
        origin = _FakeLocation(0.0, 0.0, 0.0)

        nearest, dist = find_nearest_traffic_light(world, origin)

        assert nearest is tl
        assert dist == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# OpenDRIVE controller lookups
#
# These had no tests while they fetched the road network from the MapManager
# singleton: covering them meant standing up a whole map. They read an XML root
# now, so a few elements are enough.
# ---------------------------------------------------------------------------


class _FakeRoadNetwork:
    """Just enough of pyxodr's RoadNetwork: something with a `.root`."""

    def __init__(self, xml: str) -> None:
        self.root = ElementTree.fromstring(xml)


_CONTROLLERS = """
<OpenDRIVE>
  <controller id="7" name="Controller_TL_1234">
    <control signalId="700"/>
    <control signalId="701"/>
  </controller>
  <controller id="8" name="Controller_TL_5678">
    <control signalId="800"/>
    <control/>
  </controller>
</OpenDRIVE>
"""


class TestControllerIdForLanelet2TrafficLight:
    def test_the_naming_convention_is_followed_back(self) -> None:
        network = _FakeRoadNetwork(_CONTROLLERS)
        assert lanelet2_traffic_light_id_to_opendrive_controller_id(network, 1234) == 7
        assert lanelet2_traffic_light_id_to_opendrive_controller_id(network, 5678) == 8

    def test_an_unmapped_traffic_light_is_none(self) -> None:
        network = _FakeRoadNetwork(_CONTROLLERS)
        assert lanelet2_traffic_light_id_to_opendrive_controller_id(network, 1) is None

    def test_a_prefix_match_is_not_a_match(self) -> None:
        """`Controller_TL_123` must not answer for lanelet 1234, or vice versa."""
        network = _FakeRoadNetwork(_CONTROLLERS)
        assert (
            lanelet2_traffic_light_id_to_opendrive_controller_id(network, 123) is None
        )


class TestSignalIdsForController:
    def test_every_control_child_is_returned(self) -> None:
        network = _FakeRoadNetwork(_CONTROLLERS)
        assert get_signal_ids_for_controller(network, 7) == ["700", "701"]

    def test_a_control_without_a_signal_id_is_dropped(self) -> None:
        network = _FakeRoadNetwork(_CONTROLLERS)
        assert get_signal_ids_for_controller(network, 8) == ["800"]

    def test_an_unknown_controller_is_empty(self) -> None:
        network = _FakeRoadNetwork(_CONTROLLERS)
        assert get_signal_ids_for_controller(network, 99) == []
