"""TeraSim's Merge adversity, and what a merging lane is in Lanelet2 (#400).

A synthetic stretch of highway, eastbound, lanes 3.5 m wide::

    L1  (y =  0.0)  x -100..0, 0..100, 100..200
    L2  (y = -3.5)  x -100..0, 0..100, 100..200
    M   (y = -7.0)  x -100..0, 0..100: the acceleration lane, ending beside L2

Its lanes share their boundaries, so each is the other's neighbour.
"""

from __future__ import annotations

import math
import random
from typing import Any

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
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

CAR = AgentFootprint(5.0, 2.0)
L1, L2, M = 0.0, -3.5, -7.0
#: Lanelet ids: lane * 10 + segment (0: x -100..0, 1: 0..100, 2: 100..200).
L1_0, L2_0, M_0, M_1, L2_1 = 10, 20, 30, 31, 21


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    b = rm._Builder()
    for base, y, segments in ((10, L1, 3), (20, L2, 3), (30, M, 2)):
        for i in range(segments):
            x = -100.0 + 100.0 * i
            b.lanelet(base + i, rm._line((x, y), (x + 100.0, y)))
    return MapFeatures(b.map, rm.routing_graph(b.map))


def _car(
    name: str, x: float, y: float, speed: float = 10.0, yaw: float = 0.0
) -> AgentSnapshot:
    return AgentSnapshot(name, x, y, yaw, speed, 0.0, CAR)


def _command(features: MapFeatures, *agents: AgentSnapshot) -> object:
    scene = Scene(features, list(agents), {})
    return adversity_command(
        AdversityKind.MERGE, scene, "ego", AdversityLocation.ANY, random.Random(0)
    )


# -- the merging lane -------------------------------------------------------------


def _merges(features: MapFeatures, lanelet_id: int) -> bool:
    lanelet: Any = features.lanelet(lanelet_id)
    return features.merges(lanelet)


def test_a_lane_ending_beside_one_that_goes_on_merges(features: MapFeatures) -> None:
    assert _merges(features, M_1)
    # Its first lanelet, within the window of where it ends.
    assert _merges(features, M_0)
    # The lanes it merges into go on, with nothing on their left ending.
    assert not _merges(features, L2_1)
    assert not _merges(features, L1_0)


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
