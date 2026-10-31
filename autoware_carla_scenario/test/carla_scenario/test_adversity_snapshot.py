"""Reading CARLA actors and signals for the adversity observation (#389)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario.adversity import snapshot
from autoware_carla_scenario.adversity.observation import SignalState
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm


def _vector(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _actor(
    type_id: str,
    x: float,
    y: float,
    yaw_deg: float,
    velocity: Any,
    acceleration: Any,
    extent: Any,
    box_x: float = 0.0,
) -> SimpleNamespace:
    yaw = math.radians(yaw_deg)
    transform = SimpleNamespace(
        location=_vector(x, y, 0.3),
        rotation=SimpleNamespace(yaw=yaw_deg, pitch=0.0, roll=0.0),
        get_forward_vector=lambda: _vector(math.cos(yaw), math.sin(yaw)),
    )
    return SimpleNamespace(
        type_id=type_id,
        get_speed_limit=lambda: 40.0,
        bounding_box=SimpleNamespace(extent=extent, location=_vector(box_x, 0.0)),
        get_transform=lambda: transform,
        get_velocity=lambda: velocity,
        get_acceleration=lambda: acceleration,
    )


@pytest.fixture
def offset_map() -> Iterator[None]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._mgrs_offset = (100.0, 200.0)
    manager._z_offset = 0.0
    yield
    MapManager._instance = saved


@pytest.mark.usefixtures("offset_map")
def test_an_actor_is_read_into_the_lanelet2_frame() -> None:
    # CARLA yaw -30 degrees (clockwise is positive) is +30 degrees in the map.
    actor = _actor(
        "vehicle.tesla.model3",
        10.0,
        5.0,
        -30.0,
        velocity=_vector(3.0, -4.0, 1.0),
        acceleration=_vector(
            2.0 * math.cos(math.radians(-30.0)),
            2.0 * math.sin(math.radians(-30.0)),
            0.0,
        ),
        extent=_vector(2.4, 1.0, 0.8),
    )
    agent = snapshot.snapshot_actor(actor, "npc1")
    assert agent.name == "npc1"
    assert (agent.x, agent.y) == pytest.approx((110.0, 195.0))
    assert agent.yaw == pytest.approx(math.radians(30.0))
    assert agent.speed == pytest.approx(5.0)  # horizontal only
    assert agent.acceleration == pytest.approx(2.0)
    assert (agent.footprint.length, agent.footprint.width) == pytest.approx((4.8, 2.0))
    assert agent.footprint.is_vehicle
    assert agent.speed_limit == 40.0


@pytest.mark.usefixtures("offset_map")
def test_a_walker_is_no_vehicle() -> None:
    walker = _actor(
        "walker.pedestrian.0001",
        0.0,
        0.0,
        0.0,
        _vector(0.0, 0.0),
        _vector(-1.0, 0.0),
        _vector(0.2, 0.2, 0.9),
    )
    agent = snapshot.snapshot_actor(walker, "p1")
    assert not agent.footprint.is_vehicle
    assert agent.speed_limit is None
    assert agent.acceleration == pytest.approx(-1.0)


@pytest.mark.usefixtures("offset_map")
def test_the_position_is_the_centre_of_the_bounding_box() -> None:
    # The box 1.5 m ahead of the origin, heading +30 degrees in the map.
    car = _actor(
        "vehicle.audi.tt",
        10.0,
        5.0,
        -30.0,
        _vector(0.0, 0.0),
        _vector(0.0, 0.0),
        _vector(2.0, 1.0, 0.7),
        box_x=1.5,
    )
    agent = snapshot.snapshot_actor(car, "npc1")
    yaw = math.radians(30.0)
    assert (agent.x, agent.y) == pytest.approx(
        (110.0 + 1.5 * math.cos(yaw), 195.0 + 1.5 * math.sin(yaw))
    )


def _parked_or_not(z: float) -> SimpleNamespace:
    """A world of the ego at (-55, 1.75) driving east at 10 m/s on the
    crossroads' W approach, and ``npc1`` 15 m ahead of it at height *z*."""

    def car(role: str, x: float, actor_id: int) -> SimpleNamespace:
        actor = _actor(
            "vehicle.tesla.model3",
            x,
            1.75,
            0.0,
            velocity=_vector(10.0, 0.0),
            acceleration=_vector(0.0, 0.0),
            extent=_vector(2.5, 1.0, 0.8),
        )
        height = 0.3 if role == "ego" else z
        actor.get_transform().location.z = height
        actor.id = actor_id
        actor.attributes = {"role_name": role}
        return actor

    actors = [car("ego", -55.0, 1), car("npc1", -40.0, 2)]
    return SimpleNamespace(
        get_actors=lambda: SimpleNamespace(
            filter=lambda pattern: actors if pattern == "vehicle.*" else []
        )
    )


@pytest.fixture
def plain_map() -> Iterator[None]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    yield
    MapManager._instance = saved


@pytest.mark.usefixtures("plain_map")
@pytest.mark.parametrize(("z", "challenger"), [(0.3, "npc1"), (-499.7, None)])
def test_an_entity_parked_under_the_map_is_not_in_the_scene(
    z: float, challenger: Optional[str]
) -> None:
    # #853: NADE took the parked npc1 for a car on the road ahead of the ego,
    # brake-checked with it and moved the run's importance-sampling weight.
    from autoware_carla_scenario.adversity import AdversityKind, AdversityLocation
    from autoware_carla_scenario.adversity.nade import (
        ImportanceSampler,
        NadeAdversity,
        NadeSettings,
    )
    from autoware_carla_scenario.adversity.observation import Scene

    world = _parked_or_not(z)
    agents = snapshot.snapshot_world(world)
    assert [agent.name for agent in agents] == (
        ["ego", "npc1"] if challenger else ["ego"]
    )
    lanelet_map = rm.crossroads()
    scene = Scene(MapFeatures(lanelet_map, rm.routing_graph(lanelet_map)), agents, {})
    sampler = ImportanceSampler(settings=NadeSettings(fixed_is_probability=1.0))
    brake = NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, 0.01)
    choice = sampler.step(scene, "ego", "", {"npc1"}, [brake], 0.0)
    if challenger is None:
        assert choice is None
        assert sampler.weight == 1.0
    else:
        assert choice is not None and choice.name == challenger
        assert sampler.weight == pytest.approx(0.01)


def test_the_maps_traffic_lights() -> None:
    lanelet_map = rm.crossroads()
    features = MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))
    assert features.traffic_light_ids() == [rm.TRAFFIC_LIGHT]


def test_signal_states_by_regulatory_element(monkeypatch: pytest.MonkeyPatch) -> None:
    import typesafe_carla.carla as carla

    state = {"red": carla.TrafficLightState.Red}
    lights = {
        1: [SimpleNamespace(get_state=lambda: state["red"])],
        2: [SimpleNamespace(get_state=lambda: carla.TrafficLightState.Green)],
        3: [SimpleNamespace(get_state=lambda: carla.TrafficLightState.Off)],
    }
    resolved: list[int] = []

    def find(world: Any, regulatory_id: int) -> list[Any]:
        resolved.append(regulatory_id)
        return lights.get(regulatory_id, [])

    monkeypatch.setattr(snapshot, "find_traffic_lights_for_lanelet2_id", find)
    reader = snapshot.SignalReader(SimpleNamespace(), [1, 2, 3, 4])
    assert reader.read() == {
        1: SignalState.RED,
        2: SignalState.GREEN,
        3: SignalState.UNKNOWN,
    }
    state["red"] = carla.TrafficLightState.Yellow
    assert reader.read()[1] is SignalState.YELLOW
    # The lights are looked up once, not on every read.
    assert resolved == [1, 2, 3, 4]
