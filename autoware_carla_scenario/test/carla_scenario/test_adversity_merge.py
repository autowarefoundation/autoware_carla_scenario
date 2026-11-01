"""TeraSim's Merge adversity, and what a merging lane is in Lanelet2 (#400).

A synthetic stretch of highway, eastbound, lanes 3.5 m wide::

    L1  (y =  0.0)  x -100..0, 0..100, 100..200
    L2  (y = -3.5)  x -100..0, 0..100, 100..200
    M   (y = -7.0)  x -100..0, 0..100: the acceleration lane, ending beside L2

Its lanes share their boundaries, so each is the other's neighbour.  Its
mirror image, where traffic keeps left, has the acceleration lane on the
other side (``y = +7.0``, ending beside ``y = +3.5``); the nishishinjuku
fixture map has two such merges (#1056).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import lanelet2
import pytest

from autoware_carla_scenario.adversity import (
    AdversarialCommand,
    AdversityKind,
    AdversityLocation,
    AgentFootprint,
    adversity_command,
)
from autoware_carla_scenario.adversity.idm import free_flow_acceleration
from autoware_carla_scenario.adversity.observation import (
    DEFAULT_DESIRED_SPEED,
    AgentSnapshot,
    Scene,
)
from autoware_carla_scenario.adversity.triggers import merge
from autoware_carla_scenario.coordinate.map_manager import build_routing_graph
from autoware_carla_scenario.coordinate.projection import load_lanelet2
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

CAR = AgentFootprint(5.0, 2.0)
L1, L2, M = 0.0, -3.5, -7.0
#: Lanelet ids: lane * 10 + segment (0: x -100..0, 1: 0..100, 2: 100..200).
L1_0, L2_0, M_0, M_1, L2_1 = 10, 20, 30, 31, 21


#: The nishishinjuku fixture map, kept at the repository root (left-hand
#: traffic), and its two lanelets that end into the lane on their right.
_OSM = Path(__file__).resolve().parents[3] / "data" / "nishishinjuku.osm"
FIXTURE_MERGES = (468, 124219)


def _highway(sign: float) -> MapFeatures:
    """The highway, with the acceleration lane on the right (*sign* 1.0) or,
    mirrored, on the left (*sign* -1.0)."""
    b = rm._Builder()
    for base, y, segments in ((10, L1, 3), (20, L2, 3), (30, M, 2)):
        for i in range(segments):
            x = -100.0 + 100.0 * i
            b.lanelet(base + i, rm._line((x, sign * y), (x + 100.0, sign * y)))
    return MapFeatures(b.map, rm.routing_graph(b.map))


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    return _highway(1.0)


@pytest.fixture(scope="module")
def mirrored() -> MapFeatures:
    return _highway(-1.0)


@pytest.fixture(scope="module")
def fixture_map() -> MapFeatures:
    lanelet_map, _, _ = load_lanelet2(_OSM, lanelet2.io.Origin(35.68855, 139.69142))
    graph = build_routing_graph(
        lanelet_map, lanelet2.traffic_rules.Participants.Vehicle
    )
    return MapFeatures(lanelet_map, graph)


def _car(
    name: str, x: float, y: float, speed: float = 10.0, yaw: float = 0.0
) -> AgentSnapshot:
    return AgentSnapshot(name, x, y, yaw, speed, 0.0, CAR)


def _mirror(agent: AgentSnapshot) -> AgentSnapshot:
    """*agent* on the mirrored highway."""
    return AgentSnapshot(
        agent.name, agent.x, -agent.y, -agent.yaw, agent.speed, 0.0, CAR
    )


def _command(features: MapFeatures, *agents: AgentSnapshot) -> object:
    scene = Scene(features, list(agents), {})
    return adversity_command(
        AdversityKind.MERGE, scene, "ego", AdversityLocation.ANY, random.Random(0)
    )


# -- the merging lane -------------------------------------------------------------


def _merges(features: MapFeatures, lanelet_id: int, side: str = "either") -> bool:
    lanelet: Any = features.lanelet(lanelet_id)
    return features.merges(lanelet, side)


def test_a_lane_ending_beside_one_that_goes_on_merges(features: MapFeatures) -> None:
    assert _merges(features, M_1)
    # Its first lanelet, within the window of where it ends.
    assert _merges(features, M_0)
    # The lanes it merges into go on, with nothing on their left ending.
    assert not _merges(features, L2_1)
    assert not _merges(features, L1_0)
    # Into the lane on its left, not on its right.
    assert _merges(features, M_1, "left")
    assert not _merges(features, M_1, "right")


def test_a_lane_ending_into_the_lane_on_its_right_merges(
    mirrored: MapFeatures,
) -> None:
    for lanelet_id in (M_1, M_0):
        assert _merges(mirrored, lanelet_id)
        assert _merges(mirrored, lanelet_id, "right")
        assert not _merges(mirrored, lanelet_id, "left")
    assert not _merges(mirrored, L2_1)
    assert not _merges(mirrored, L1_0)


@pytest.mark.parametrize("lanelet_id", FIXTURE_MERGES)
def test_the_fixture_maps_left_hand_merges_are_merges(
    fixture_map: MapFeatures, lanelet_id: int
) -> None:
    assert _merges(fixture_map, lanelet_id)
    assert _merges(fixture_map, lanelet_id, "right")
    assert not _merges(fixture_map, lanelet_id, "left")


# -- one lane over: speed up ------------------------------------------------------


def test_beside_a_merging_vehicle_ahead_it_speeds_up(features: MapFeatures) -> None:
    command = _command(features, _car("ego", 50.0, L2), _car("merging", 60.0, M, 5.0))
    free = free_flow_acceleration(10.0, DEFAULT_DESIRED_SPEED)
    assert command == AdversarialCommand.accelerate(free, 2.0)


@pytest.mark.parametrize(
    "others",
    [
        [],  # nobody merging
        [_car("merging", 35.0, M)],  # merging behind: nothing to give way to
        [_car("merging", 60.0, M), _car("ahead", 60.0, L2, 5.0)],  # held back
    ],
)
def test_no_speeding_up(features: MapFeatures, others: list[AgentSnapshot]) -> None:
    assert _command(features, _car("ego", 50.0, L2), *others) is None


# -- two lanes over: change in ----------------------------------------------------


def test_two_lanes_over_it_changes_into_the_lane_between(
    features: MapFeatures,
) -> None:
    command = _command(features, _car("ego", 50.0, L1), _car("merging", 60.0, M))
    assert command == AdversarialCommand.change_lane("right", 1.0)
    # A slow vehicle far ahead in that lane leaves time enough.
    far = _car("far", 90.0, L2, 5.0)
    assert _command(features, _car("ego", 50.0, L1), _car("merging", 60.0, M), far)


@pytest.mark.parametrize(
    "other",
    [
        _car("behind", 40.0, L2, 15.0),  # closing in, 1 s behind
        _car("ahead", 62.0, L2, 5.0),  # closing in on it, 1.4 s ahead
    ],
)
def test_no_change_in_within_the_headway(
    features: MapFeatures, other: AgentSnapshot
) -> None:
    agents = [_car("ego", 50.0, L1), _car("merging", 60.0, M), other]
    assert _command(features, *agents) is None


def test_without_a_merging_lane_beyond_nothing_changes(features: MapFeatures) -> None:
    scene = Scene(features, [_car("ego", 50.0, L1)], {})
    assert merge(scene.vehicle("ego"), None, features) is None


# -- traffic keeping left: the merging lane on the left -----------------------------


def test_beside_a_merging_vehicle_on_the_left_it_speeds_up(
    mirrored: MapFeatures,
) -> None:
    agents = [_car("ego", 50.0, L2), _car("merging", 60.0, M, 5.0)]
    command = _command(mirrored, *map(_mirror, agents))
    free = free_flow_acceleration(10.0, DEFAULT_DESIRED_SPEED)
    assert command == AdversarialCommand.accelerate(free, 2.0)
    # Held back by a vehicle ahead in its own lane, as on the right.
    held = [*agents, _car("ahead", 60.0, L2, 5.0)]
    assert _command(mirrored, *map(_mirror, held)) is None


def test_two_lanes_over_on_the_left_it_changes_left_into_the_lane_between(
    mirrored: MapFeatures,
) -> None:
    agents = [_car("ego", 50.0, L1), _car("merging", 60.0, M)]
    command = _command(mirrored, *map(_mirror, agents))
    assert command == AdversarialCommand.change_lane("left", 1.0)
    blocked = [*agents, _car("behind", 40.0, L2, 15.0)]
    assert _command(mirrored, *map(_mirror, blocked)) is None


@pytest.mark.parametrize(
    ("yaw", "fires"),
    [
        (math.radians(4.0), True),
        (math.radians(6.0), False),
        (math.radians(-6.0), False),
    ],
)
def test_a_vehicle_already_changing_lanes_is_left_alone(
    features: MapFeatures, yaw: float, fires: bool
) -> None:
    # TeraSim's is_lane_changing: 5 degrees or more off its lane (#440).
    ego = _car("ego", 50.0, L2, yaw=yaw)
    command = _command(features, ego, _car("merging", 60.0, M, 5.0))
    assert (command is not None) is fires
