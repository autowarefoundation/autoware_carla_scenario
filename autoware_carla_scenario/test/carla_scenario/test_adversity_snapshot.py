"""Reading CARLA actors and signals for the adversity observation (#389)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

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
) -> SimpleNamespace:
    yaw = math.radians(yaw_deg)
    transform = SimpleNamespace(
        location=_vector(x, y, 0.3),
        rotation=SimpleNamespace(yaw=yaw_deg, pitch=0.0, roll=0.0),
        get_forward_vector=lambda: _vector(math.cos(yaw), math.sin(yaw)),
    )
    return SimpleNamespace(
        type_id=type_id,
        bounding_box=SimpleNamespace(extent=extent),
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
    assert agent.acceleration == pytest.approx(-1.0)


def test_the_maps_traffic_lights() -> None:
    lanelet_map = rm.crossroads()
    features = MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))
    assert snapshot.traffic_light_ids(features) == [rm.TRAFFIC_LIGHT]


def test_signal_states_by_regulatory_element(monkeypatch: pytest.MonkeyPatch) -> None:
    import typesafe_carla.carla as carla

    lights = {
        1: [SimpleNamespace(get_state=lambda: carla.TrafficLightState.Red)],
        2: [SimpleNamespace(get_state=lambda: carla.TrafficLightState.Green)],
        3: [SimpleNamespace(get_state=lambda: carla.TrafficLightState.Off)],
    }
    monkeypatch.setattr(
        snapshot,
        "find_traffic_lights_for_lanelet2_id",
        lambda world, i: lights.get(i, []),
    )
    states = snapshot.read_signal_states(SimpleNamespace(), [1, 2, 3, 4])
    assert states == {1: SignalState.RED, 2: SignalState.GREEN, 3: SignalState.UNKNOWN}
