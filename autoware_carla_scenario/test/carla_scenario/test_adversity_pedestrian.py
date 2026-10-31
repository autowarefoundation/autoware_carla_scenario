"""The pedestrian adversities: crossing on red and stopping mid-crossing (#399).

On the synthetic crossroads of :mod:`._route_maps`: the crosswalk across the W
leg runs north from (-15, -7) to (-15, 7), 14 m, over the W road's lanes.
Here it carries a traffic light, :data:`CROSSWALK_LIGHT`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from types import SimpleNamespace

import lanelet2.core
import pytest
from lanelet2.core import AttributeMap, LineString3d, Point3d

from autoware_carla_scenario.adversity import (
    AdversityKind,
    AdversityLocation,
    AgentFootprint,
    CommandKind,
    adversity_command,
)
from autoware_carla_scenario.adversity import snapshot
from autoware_carla_scenario.adversity.observation import (
    AgentSnapshot,
    Scene,
    SignalState,
)
from autoware_carla_scenario.adversity.triggers import (
    TriggerThresholds,
    running_red_light,
    stop_crossing,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

PERSON = AgentFootprint(0.5, 0.5, is_vehicle=False)
CAR = AgentFootprint(5.0, 2.0)
CROSSWALK_LIGHT = 961
RED = {CROSSWALK_LIGHT: SignalState.RED}


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    lanelet_map = rm.crossroads()
    light = LineString3d(
        9_000_001,
        [Point3d(9_000_002, -17.0, -8.0, 3.0), Point3d(9_000_003, -17.0, -7.5, 3.0)],
        AttributeMap({"type": "traffic_light"}),
    )
    signal = lanelet2.core.TrafficLight(
        CROSSWALK_LIGHT,
        AttributeMap({"type": "regulatory_element", "subtype": "traffic_light"}),
        [light],
    )
    lanelet_map.laneletLayer[rm.CROSSWALK_W].addRegulatoryElement(signal)
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


def _person(x: float, y: float, yaw: float = 0.0, speed: float = 0.0) -> AgentSnapshot:
    return AgentSnapshot("p", x, y, yaw, speed, 0.0, PERSON)


def _scene(
    features: MapFeatures,
    *agents: AgentSnapshot,
    signals: dict[int, SignalState] | None = None,
) -> Scene:
    return Scene(features, list(agents), signals or {})


# -- the observation --------------------------------------------------------------


def test_a_pedestrian_sees_the_crosswalk_near_it(features: MapFeatures) -> None:
    seen = _scene(features, _person(-15.0, -9.0)).pedestrian("p")
    near = seen.nearest_crosswalk
    assert seen.crosswalk is None
    assert near is not None and int(near.lanelet.id) == rm.CROSSWALK_W
    assert near.distance == pytest.approx(2.0)
    on = _scene(features, _person(-15.0, 0.0)).pedestrian("p")
    assert on.crosswalk is not None and int(on.crosswalk.id) == rm.CROSSWALK_W
    assert on.nearest_crosswalk is not None and on.nearest_crosswalk.distance == 0.0


# -- RUNNING_RED_LIGHT ------------------------------------------------------------


def test_a_waiting_pedestrian_crosses_on_red(features: MapFeatures) -> None:
    seen = _scene(features, _person(-15.0, -9.0), signals=RED).pedestrian("p")
    assert seen.nearest_crosswalk is not None
    assert seen.nearest_crosswalk.signal is SignalState.RED
    command = running_red_light(seen, features)
    assert command is not None and command.kind is CommandKind.TRAJECTORY
    # From the end it waits at to the far end: north, 14 m at 2.8 m/s.
    assert command.duration == pytest.approx(5.0)
    end = command.path[-1]
    assert (end.x, end.y) == pytest.approx((-15.0, 5.0))
    assert end.yaw == pytest.approx(math.pi / 2) and end.speed == 2.8


def test_from_the_other_end_it_crosses_the_other_way(features: MapFeatures) -> None:
    seen = _scene(features, _person(-15.0, 9.0), signals=RED).pedestrian("p")
    command = running_red_light(seen, features)
    assert command is not None
    assert command.path[-1].yaw == pytest.approx(-math.pi / 2)


@pytest.mark.parametrize(
    ("person", "signals"),
    [
        (_person(-15.0, -9.0), {CROSSWALK_LIGHT: SignalState.GREEN}),
        (_person(-15.0, -9.0), {}),  # its light unknown
        (_person(-15.0, -9.0, speed=1.0), RED),  # walking, not waiting
        (_person(-15.0, -11.0), RED),  # 4 m off the crosswalk
    ],
)
def test_no_crossing_on_red(
    features: MapFeatures, person: AgentSnapshot, signals: dict[int, SignalState]
) -> None:
    seen = _scene(features, person, signals=signals).pedestrian("p")
    assert running_red_light(seen, features) is None


def test_a_crosswalk_without_a_light_has_no_red_to_run(features: MapFeatures) -> None:
    # The E leg's crosswalk (x = 15) carries no light.
    seen = _scene(features, _person(15.0, -9.0), signals=RED).pedestrian("p")
    assert running_red_light(seen, features) is None


# -- STOP_CROSSING ----------------------------------------------------------------


def test_a_pedestrian_stops_on_the_crosswalk_then_hurries_across(
    features: MapFeatures,
) -> None:
    seen = _scene(features, _person(-15.0, -5.0, math.pi / 2, 1.0)).pedestrian("p")
    command = stop_crossing(seen, features, random.Random(3))
    assert command is not None and command.kind is CommandKind.TRAJECTORY
    wait = max(random.Random(3).gauss(2.0, 0.5), 0.0)
    path = command.path
    standing = [p for p in path if p.t <= wait + 1e-9]
    assert all(p.speed == 0.0 and (p.x, p.y) == (-15.0, -5.0) for p in standing)
    assert command.duration == pytest.approx(wait + 14.0 / 2.8)
    end = path[-1]
    assert (end.x, end.y) == pytest.approx((-15.0, 9.0)) and end.speed == 2.8
    assert [p.t for p in path] == sorted({p.t for p in path})  # no time twice


def test_no_stop_crossing_off_a_crosswalk(features: MapFeatures) -> None:
    seen = _scene(features, _person(-40.0, -12.0)).pedestrian("p")
    assert stop_crossing(seen, features, random.Random(0)) is None


def test_the_wait_is_never_negative(features: MapFeatures) -> None:
    seen = _scene(features, _person(-15.0, 0.0)).pedestrian("p")
    never = TriggerThresholds(stop_crossing_wait=(-5.0, 0.0))
    command = stop_crossing(seen, features, random.Random(0), never)
    assert command is not None and command.path[0].speed == 2.8


# -- who carries them out ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "x", "y"),
    [
        (AdversityKind.JAYWALKING, -40.0, -12.0),
        (AdversityKind.RUNNING_RED_LIGHT, -15.0, -9.0),
        (AdversityKind.STOP_CROSSING, -15.0, -1.75),
    ],
)
def test_a_vehicle_carries_out_no_pedestrian_adversity(
    features: MapFeatures, kind: AdversityKind, x: float, y: float
) -> None:
    # Where a pedestrian standing still would carry it out, a car does not.
    walker = AgentSnapshot("p", x, y, 0.0, 0.0, 0.0, PERSON)
    car = AgentSnapshot("car", x, y, 0.0, 0.0, 0.0, CAR)
    for agent, fires in ((walker, True), (car, False)):
        scene = _scene(features, agent, signals=RED)
        command = adversity_command(
            kind, scene, agent.name, AdversityLocation.ANY, random.Random(0)
        )
        assert (command is not None) is fires


def test_a_scene_looks_around_a_pedestrian_once(features: MapFeatures) -> None:
    scene = _scene(features, _person(-15.0, -9.0))
    assert scene.pedestrian("p") is scene.pedestrian("p")


def test_the_dispatcher_asks_the_pedestrian_triggers(features: MapFeatures) -> None:
    scene = _scene(features, _person(-15.0, -9.0), signals=RED)
    command = adversity_command(
        AdversityKind.RUNNING_RED_LIGHT,
        scene,
        "p",
        AdversityLocation.ANY,
        random.Random(0),
    )
    assert command == running_red_light(scene.pedestrian("p"), features)


# -- RUNNING_RED_LIGHT on what a world reports (#923) -----------------------------


@pytest.fixture
def plain_map() -> Iterator[None]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    yield
    MapManager._instance = saved


def _ai_walked(step_m: float) -> list[AgentSnapshot]:
    """A walker at map (-15, -9), at the red crosswalk, moved *step_m* east
    each 0.1 s tick while CARLA reports its velocity 0, as a walker on
    ``controller.ai.walker`` does; its snapshot on the third tick."""
    clock = {"elapsed": 0.0, "x": -15.0}

    def transform() -> SimpleNamespace:
        # CARLA's y is the map's flipped.
        return SimpleNamespace(
            location=SimpleNamespace(x=clock["x"], y=9.0, z=0.9),
            rotation=SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0),
            get_forward_vector=lambda: SimpleNamespace(x=1.0, y=0.0, z=0.0),
        )

    walker = SimpleNamespace(
        id=11,
        type_id="walker.pedestrian.0001",
        attributes={"role_name": "p"},
        bounding_box=SimpleNamespace(
            extent=SimpleNamespace(x=0.25, y=0.25, z=0.9),
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        ),
        get_transform=transform,
        get_location=lambda: transform().location,
        get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
        get_acceleration=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    world = SimpleNamespace(
        get_snapshot=lambda: SimpleNamespace(
            timestamp=SimpleNamespace(elapsed_seconds=clock["elapsed"])
        ),
        get_actors=lambda: SimpleNamespace(
            filter=lambda pattern: [walker] if pattern == "walker.*" else []
        ),
    )
    agents = snapshot.snapshot_world(world)
    for _ in range(2):
        clock["elapsed"] += 0.1
        clock["x"] += step_m
        agents = snapshot.snapshot_world(world)
    return agents


@pytest.mark.usefixtures("plain_map")
def test_an_ai_walked_pedestrian_at_a_red_crosswalk_is_not_waiting(
    features: MapFeatures,
) -> None:
    agents = _ai_walked(0.14)  # 1.4 m/s
    assert agents[0].speed == pytest.approx(1.4)
    seen = Scene(features, agents, RED).pedestrian("p")
    assert seen.nearest_crosswalk is not None
    assert running_red_light(seen, features) is None


@pytest.mark.usefixtures("plain_map")
def test_a_pedestrian_standing_at_a_red_crosswalk_crosses(
    features: MapFeatures,
) -> None:
    agents = _ai_walked(0.0)
    assert agents[0].speed == 0.0
    seen = Scene(features, agents, RED).pedestrian("p")
    command = running_red_light(seen, features)
    assert command is not None and command.kind is CommandKind.TRAJECTORY
