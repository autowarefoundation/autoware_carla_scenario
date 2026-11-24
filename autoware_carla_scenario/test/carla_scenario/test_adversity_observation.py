"""What an agent sees, for the adversity triggers (#389).

On the synthetic crossroads of :mod:`._route_maps`: the eastbound W approach
is W_IN1 (x -110..-60) then W_IN2 (x -60..-10) at y = -1.75, with the outer
lane W_OUT_LANE1/2 at y = -5.25 on its right and the traffic light (stop
line at x = -12) on W_IN2.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

import pytest

from autoware_carla_scenario.adversity import AgentFootprint, observation
from autoware_carla_scenario.adversity.observation import (
    DEFAULT_DESIRED_SPEED,
    AgentSnapshot,
    Scene,
    SignalState,
    on_another_level,
    vertical_span,
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


# -- stacked lanes (#1596) ----------------------------------------------------

#: A car's box: 1.5 m high, its centre 0.75 m above the road it stands on.
_HEIGHT = 1.5


def _on(name: str, x: float, road_z: float) -> AgentSnapshot:
    """A car on the stacked lanes at *x*, standing on a road at *road_z*."""
    return AgentSnapshot(
        name, x, 0.0, 0.0, 10.0, 0.0, CAR, z=road_z + _HEIGHT / 2, height=_HEIGHT
    )


@pytest.fixture(scope="module")
def stacked() -> MapFeatures:
    lanelet_map = rm.stacked(8.0)
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


def test_a_vehicle_is_located_on_its_own_deck(stacked: MapFeatures) -> None:
    scene = Scene(stacked, [_on("ego", 50.0, 8.0), _on("npc", 70.0, 0.0)], {})
    ego = scene.lane_of("ego")
    npc = scene.lane_of("npc")
    assert ego is not None and int(ego.lanelet.id) == rm.DECK
    assert npc is not None and int(npc.lanelet.id) == rm.STREET


def test_the_leader_on_the_deck_ignores_the_vehicle_below(
    stacked: MapFeatures,
) -> None:
    # The repro of #1596: the NPC on the street below was the ego's leader.
    below = Scene(stacked, [_on("ego", 50.0, 8.0), _on("npc", 70.0, 0.0)], {})
    assert below.vehicle("ego").leader is None
    assert below.vehicle("npc").follower is None
    # On the same deck it leads, 20 m ahead: a 15 m gap.
    same = Scene(stacked, [_on("ego", 50.0, 8.0), _on("npc", 70.0, 8.0)], {})
    leader = same.vehicle("ego").leader
    assert leader is not None and leader.name == "npc"
    assert leader.gap == pytest.approx(15.0)


def test_a_vehicle_on_another_level_is_no_neighbour_on_one_lanelet(
    features: MapFeatures,
) -> None:
    # Where only one lanelet is mapped -- the bridge left out of the map --
    # both are located on it, yet the one on the bridge is no leader.
    bridge = dataclasses.replace(_car("over", -25.0), z=8.75, height=_HEIGHT)
    ego = dataclasses.replace(_car("ego", -40.0), z=0.75, height=_HEIGHT)
    assert Scene(features, [ego, bridge], {}).vehicle("ego").leader is None
    level = dataclasses.replace(bridge, z=0.75)
    leader = Scene(features, [ego, level], {}).vehicle("ego").leader
    assert leader is not None and leader.name == "over"


def test_a_vehicle_up_a_grade_is_on_the_same_level() -> None:
    # 3 m higher 35 m ahead is a road climbing under 10 %, not another level,
    # though the boxes are 1.5 m apart in height; 3 m higher 25 m ahead is.
    ego = AgentSnapshot("ego", 0.0, 0.0, 0.0, 0.0, 0.0, CAR, z=0.75, height=1.5)
    up = AgentSnapshot("up", 35.0, 0.0, 0.0, 0.0, 0.0, CAR, z=3.75, height=1.5)
    assert not on_another_level(ego, up)
    assert on_another_level(ego, dataclasses.replace(up, x=25.0))


def test_without_an_elevation_an_agent_is_on_every_level() -> None:
    ego = AgentSnapshot("ego", 0.0, 0.0, 0.0, 0.0, 0.0, CAR)
    over = AgentSnapshot("over", 10.0, 0.0, 0.0, 0.0, 0.0, CAR, z=8.0)
    assert ego.z is None and ego.height == 0.0
    assert vertical_span(ego) is None
    assert vertical_span(over) == (8.0, 8.0)
    assert not on_another_level(ego, over)
    assert on_another_level(dataclasses.replace(ego, z=0.0), over)


# -- a pedestrian on stacked levels (#1606) -----------------------------------

#: A pedestrian's box: 1.8 m high, its centre 0.9 m above where it stands.
_PERSON_HEIGHT = 1.8


def _walker(name: str, x: float, ground_z: float | None) -> AgentSnapshot:
    """A pedestrian at *x* on the stacked lanes, standing at *ground_z*
    (None: its elevation not known)."""
    z = None if ground_z is None else ground_z + _PERSON_HEIGHT / 2
    return AgentSnapshot(
        name, x, 0.0, math.pi / 2, 1.0, 0.0, PERSON, z=z, height=_PERSON_HEIGHT
    )


@pytest.fixture(scope="module")
def stacked_crossings() -> MapFeatures:
    lanelet_map = rm.stacked(8.0, crosswalks=True)
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


def test_a_pedestrian_on_the_deck_is_not_on_the_crosswalk_below(
    stacked_crossings: MapFeatures,
) -> None:
    # Over the street's crosswalk, in plan view on it.
    seen = Scene(stacked_crossings, [_walker("p", 100.0, 8.0)], {}).pedestrian("p")
    assert seen.crosswalk is None
    assert seen.nearest_crosswalk is None
    assert seen.nearest_road is not None
    assert int(seen.nearest_road.lanelet.id) == rm.DECK


def test_a_pedestrian_on_the_deck_finds_the_deck_crosswalk(
    stacked_crossings: MapFeatures,
) -> None:
    # The street's crosswalk is 49 m nearer in plan view; the deck's is its own.
    scene = Scene(stacked_crossings, [_walker("p", 100.0, 8.0)], {})
    near = scene.pedestrian("p", road_radius=60.0).nearest_crosswalk
    assert near is not None and int(near.lanelet.id) == rm.DECK_CROSSWALK
    assert near.distance == pytest.approx(49.0)
    on = Scene(stacked_crossings, [_walker("p", 150.0, 8.0)], {}).pedestrian("p")
    assert on.crosswalk is not None and int(on.crosswalk.id) == rm.DECK_CROSSWALK


def test_a_pedestrian_on_the_street_is_on_its_crosswalk(
    stacked_crossings: MapFeatures,
) -> None:
    seen = Scene(stacked_crossings, [_walker("p", 100.0, 0.0)], {}).pedestrian("p")
    assert seen.crosswalk is not None
    assert int(seen.crosswalk.id) == rm.STREET_CROSSWALK
    assert seen.nearest_road is not None
    assert int(seen.nearest_road.lanelet.id) == rm.STREET
    # Under the deck's crosswalk, the street has none to be on.
    under = Scene(stacked_crossings, [_walker("p", 150.0, 0.0)], {}).pedestrian("p")
    assert under.crosswalk is None


def test_without_an_elevation_a_pedestrian_looks_around_in_plan_view(
    stacked_crossings: MapFeatures,
) -> None:
    seen = Scene(stacked_crossings, [_walker("p", 100.0, None)], {}).pedestrian("p")
    assert seen.crosswalk is not None
    assert int(seen.crosswalk.id) == rm.STREET_CROSSWALK
    assert seen.nearest_road is not None
    assert int(seen.nearest_road.lanelet.id) in (rm.STREET, rm.DECK)


# -- the chain prefilter (#2336) ---------------------------------------------


def _random_cars(features: MapFeatures, seed: int, count: int) -> list[AgentSnapshot]:
    """Cars all over the crossroads, most on its lanes -- some at the edge
    of the lateral limit off a lane's centre, some turned against it --
    and some off them, or on another level."""
    rng = random.Random(seed)
    roads = [
        [(float(p.x), float(p.y)) for p in lanelet.centerline]
        for lanelet in features.map.laneletLayer
        if MapFeatures.is_road(lanelet)
    ]
    cars = []
    for i in range(count):
        if rng.random() < 0.15:
            x, y = rng.uniform(-130.0, 130.0), rng.uniform(-130.0, 130.0)
            yaw = rng.uniform(-math.pi, math.pi)
        else:
            points = rng.choice(roads)
            k = rng.randrange(len(points) - 1)
            (x0, y0), (x1, y1) = points[k], points[k + 1]
            f = rng.random()
            heading = math.atan2(y1 - y0, x1 - x0)
            off = rng.choice([0.0, 0.0, 2.5, -2.5, rng.uniform(-3.0, 3.0)])
            x = x0 + f * (x1 - x0) - off * math.sin(heading)
            y = y0 + f * (y1 - y0) + off * math.cos(heading)
            yaw = heading + rng.choice([0.0, 0.0, 0.0, math.pi]) + rng.gauss(0, 0.3)
        z = rng.choice([None, None, None, 0.0, 8.0])
        cars.append(AgentSnapshot(f"car{i}", x, y, yaw, 10.0, 0.0, CAR, z=z))
    return cars


def _summary(seen: Any) -> tuple:
    def neighbour(found: Any) -> Any:
        return None if found is None else (found.name, found.gap)

    def side(lane: Any) -> Any:
        if lane is None:
            return None
        return (
            int(lane.lanelet.id),
            lane.s,
            neighbour(lane.leader),
            neighbour(lane.follower),
        )

    lane = None if seen.lane is None else (int(seen.lane.lanelet.id), seen.lane.s)
    return (
        lane,
        neighbour(seen.leader),
        neighbour(seen.follower),
        side(seen.left),
        side(seen.right),
    )


@pytest.mark.parametrize("seed", range(6))
def test_the_chain_prefilter_changes_no_observation(
    features: MapFeatures, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """Every vehicle's observation is the same as with every vehicle
    located: the prefilter only skips those that cannot be on the chain."""
    cars = _random_cars(features, seed, 80)
    filtered = Scene(features, cars, {})
    seen = {car.name: _summary(filtered.vehicle(car.name)) for car in cars}
    monkeypatch.setattr(observation, "_CHAIN_BOX_MARGIN_M", math.inf)
    unfiltered = Scene(features, cars, {})
    for car in cars:
        assert seen[car.name] == _summary(unfiltered.vehicle(car.name)), car


def test_the_chain_prefilter_keeps_every_vehicle_located_on_the_chain(
    features: MapFeatures,
) -> None:
    """Against locating every vehicle, on chains about every lanelet, with
    vehicles right at the bound: :data:`_MAX_LATERAL_M` off each corner of
    a chain lanelet's box, and just past it."""
    lanelets = list(features.map.laneletLayer)
    reach = observation._MAX_LATERAL_M
    rng = random.Random(2336)
    for lanelet in lanelets:
        if not MapFeatures.is_road(lanelet):
            continue
        chain = features.lane_chain(lanelet, 0.0, 40.0, 40.0)
        cars = _random_cars(features, rng.randrange(1 << 30), 40)
        for key, span in chain.items():
            points = [(float(p.x), float(p.y)) for p in span.lanelet.centerline]
            west = min(x for x, _y in points)
            east = max(x for x, _y in points)
            south = min(y for _x, y in points)
            north = max(y for _x, y in points)
            for n, (x, y, yaw) in enumerate(
                [
                    (points[0][0], points[0][1] - reach, 0.0),
                    (points[-1][0], points[-1][1] + reach, 0.0),
                    (west - reach, points[0][1], 0.0),
                    (east + reach, points[-1][1], math.pi),
                    (west - reach - 1e-3, south, 0.0),
                    (east + reach + 1e-3, north, math.pi),
                ]
            ):
                cars.append(AgentSnapshot(f"edge{key}_{n}", x, y, yaw, 1.0, 0.0, CAR))
        scene = Scene(features, cars, {})
        expected: dict[int, list[str]] = {}
        for car in cars:
            lane = scene.lane_of(car.name)
            if lane is not None and int(lane.lanelet.id) in chain:
                expected.setdefault(int(lane.lanelet.id), []).append(car.name)
        assert Scene(features, cars, {})._on_chain(chain) == expected


def test_far_vehicles_are_not_located(
    features: MapFeatures, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The map matching, the costly part, is done only for those near the
    observed vehicle's lanes."""
    located: list[float] = []
    original = observation.reference_lane_pose

    def counting(*args: Any) -> Any:
        located.append(args[1])
        return original(*args)

    monkeypatch.setattr(observation, "reference_lane_pose", counting)
    far = [_car(f"far{i}", 60.0 + i, -INNER, yaw=math.pi) for i in range(40)]
    scene = Scene(features, [_car("ego", -40.0), _car("ahead", -25.0), *far], {})
    seen = scene.vehicle("ego")
    assert seen.leader is not None and seen.leader.name == "ahead"
    # The ego and the one ahead; none of the 40 on the far arm.
    assert len(located) == 2
