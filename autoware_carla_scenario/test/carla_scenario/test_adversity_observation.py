"""What an agent sees, for the adversity triggers (#389).

On the synthetic crossroads of :mod:`._route_maps`: the eastbound W approach
is W_IN1 (x -110..-60) then W_IN2 (x -60..-10) at y = -1.75, with the outer
lane W_OUT_LANE1/2 at y = -5.25 on its right and the traffic light (stop
line at x = -12) on W_IN2.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from autoware_carla_scenario.adversity import AgentFootprint
from autoware_carla_scenario.adversity.observation import (
    DEFAULT_DESIRED_SPEED,
    AgentSnapshot,
    Scene,
    SignalState,
)
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

CAR = AgentFootprint(5.0, 2.0)
PERSON = AgentFootprint(0.5, 0.5, is_vehicle=False)
INNER = -1.75
OUTER = -5.25


def _car(
    name: str, x: float, y: float = INNER, yaw: float = 0.0, speed: float = 10.0
) -> AgentSnapshot:
    return AgentSnapshot(name, x, y, yaw, speed, 0.5, CAR)


def _person(name: str, x: float, y: float) -> AgentSnapshot:
    return AgentSnapshot(name, x, y, math.pi / 2, 1.0, 0.0, PERSON)


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    lanelet_map = rm.crossroads()
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


def test_a_vehicle_is_located_on_its_lane(features: MapFeatures) -> None:
    scene = Scene(
        features,
        [
            _car("ego", -40.0),
            _car("wrong_way", -40.0, yaw=math.pi),
            _person("p", -40.0, INNER),
        ],
        {},
    )
    lane = scene.lane_of("ego")
    assert lane is not None
    assert int(lane.lanelet.id) == rm.W_IN2
    assert lane.s == pytest.approx(20.0)
    assert scene.lane_of("wrong_way") is None
    assert scene.lane_of("p") is None


def test_the_leader_and_follower_along_the_lane_chain(features: MapFeatures) -> None:
    scene = Scene(
        features,
        [
            _car("ego", -40.0),
            _car("ahead", -25.0, speed=5.0),  # 15 m ahead: a 10 m gap
            _car("further", -15.0),
            _car("behind", -75.0),  # on W_IN1, 35 m behind: a 30 m gap
            _car("far_behind", -100.0),  # 60 m behind: out of the 40 m
            _car("beside", -40.0, OUTER),  # the other lane
        ],
        {},
    )
    seen = scene.vehicle("ego")
    assert seen.leader is not None and seen.leader.name == "ahead"
    assert seen.leader.gap == pytest.approx(10.0)
    assert seen.leader.speed == 5.0
    assert seen.follower is not None and seen.follower.name == "behind"
    assert seen.follower.gap == pytest.approx(30.0)


def test_the_lane_beside_and_who_is_on_it(features: MapFeatures) -> None:
    scene = Scene(
        features,
        [
            _car("ego", -40.0),
            _car("side_rear", -48.0, OUTER),
            _car("side_front", -20.0, OUTER),
        ],
        {},
    )
    seen = scene.vehicle("ego")
    # The lane on the left runs the other way: no lane beside on that side.
    assert seen.left is None
    assert seen.right is not None
    assert int(seen.right.lanelet.id) == rm.W_OUT_LANE2
    assert seen.right.s == pytest.approx(20.0)
    assert seen.right.follower is not None and seen.right.follower.name == "side_rear"
    assert seen.right.follower.gap == pytest.approx(3.0)
    assert seen.right.leader is not None and seen.right.leader.gap == pytest.approx(
        15.0
    )


def test_the_signal_ahead_is_measured_to_its_stop_line(features: MapFeatures) -> None:
    scene = Scene(
        features,
        [_car("ego", -40.0), _car("far", -100.0)],
        {rm.TRAFFIC_LIGHT: SignalState.RED},
    )
    signal = scene.vehicle("ego").signal
    assert signal is not None
    assert signal.distance == pytest.approx(28.0)
    assert signal.state is SignalState.RED
    assert signal.regulatory_element_id == rm.TRAFFIC_LIGHT
    # 88 m from the stop line: beyond the 40 m search.
    assert scene.vehicle("far").signal is None
    unknown = Scene(features, [_car("ego", -40.0)], {}).vehicle("ego").signal
    assert unknown is not None and unknown.state is SignalState.UNKNOWN


def test_past_the_stop_line_the_signal_is_behind(features: MapFeatures) -> None:
    scene = Scene(features, [_car("ego", -11.0)], {rm.TRAFFIC_LIGHT: SignalState.RED})
    assert scene.vehicle("ego").signal is None


def test_the_desired_speed_is_the_speed_limit() -> None:
    lanelet_map = rm.crossroads()
    lanelet_map.laneletLayer[rm.W_IN2].attributes["speed_limit"] = "30"
    features = MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))
    assert Scene(features, [_car("ego", -40.0)], {}).vehicle(
        "ego"
    ).desired_speed == pytest.approx(30.0 / 3.6)
    assert (
        Scene(features, [_car("ego", -100.0)], {}).vehicle("ego").desired_speed
        == DEFAULT_DESIRED_SPEED
    )
    # Untagged, the simulator's limit for the vehicle, as the ODD reads it.
    carla_limited = dataclasses.replace(_car("ego", -100.0), speed_limit=60.0)
    assert Scene(features, [carla_limited], {}).vehicle(
        "ego"
    ).desired_speed == pytest.approx(60.0 / 3.6)
    # The lanelet's tag wins over it.
    tagged = dataclasses.replace(_car("ego", -40.0), speed_limit=60.0)
    assert Scene(features, [tagged], {}).vehicle("ego").desired_speed == pytest.approx(
        30.0 / 3.6
    )


def test_a_vehicle_off_the_lanes_sees_nothing(features: MapFeatures) -> None:
    seen = Scene(
        features, [_car("ego", -40.0, -30.0), _car("other", -30.0)], {}
    ).vehicle("ego")
    assert seen.lane is None and seen.leader is None and seen.signal is None


def test_a_pedestrian_on_a_crosswalk(features: MapFeatures) -> None:
    seen = Scene(features, [_person("p", -15.0, 5.0)], {}).pedestrian("p")
    assert seen.crosswalk is not None and int(seen.crosswalk.id) == rm.CROSSWALK_W


def test_a_pedestrian_beside_the_road_sees_its_nearest_point(
    features: MapFeatures,
) -> None:
    seen = Scene(features, [_person("p", -40.0, -12.0)], {}).pedestrian("p")
    assert seen.crosswalk is None
    assert seen.nearest_road is not None
    assert int(seen.nearest_road.lanelet.id) == rm.W_OUT_LANE2
    assert (seen.nearest_road.x, seen.nearest_road.y) == pytest.approx((-40.0, OUTER))
    assert seen.nearest_road.distance == pytest.approx(12.0 + OUTER)
    assert (
        Scene(features, [_person("p", -40.0, -20.0)], {}).pedestrian("p").nearest_road
        is None
    )
