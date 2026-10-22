"""A pedestrian walking straight turns back when a vehicle comes too close.

Driven against the lane-distance fixture map (Nishi-Shinjuku), so the TTC and
the distance are the real lane-frame measures.  CARLA is faked: the walker
moves as its last ``WalkerControl`` says, and "on the road" is a band of the
fixture road around its reference line.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Optional
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    PedestrianEntity,
    PedestrianEntityConfig,
    SpawnTransform,
    WalkStraightAction,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions import walk_straight as walk_module
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, OpenDrivePose
from autoware_carla_scenario.coordinate.transform import to_carla_world, to_opendrive
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity

_DATA = Path(__file__).resolve().parents[3] / "data"
XODR_PATH = _DATA / "nishishinjuku_carla.xodr"
OSM_PATH = _DATA / "nishishinjuku.osm"

#: The fixture road the pedestrian crosses, and where along it.
ROAD = "173"
CROSSING_S = 60.0
#: The road's edge (the kerb), as an offset from its reference line (m).
KERB_T = -3.0
#: Where the pedestrian waits: a metre back from the kerb.
START_T = -4.0
DT = 0.05
SPEED = 2.0


@pytest.fixture(scope="module")
def loaded_map() -> Iterator[MapManager]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(XODR_PATH, OSM_PATH)
    yield manager
    MapManager._instance = saved


def _world_at(s: float, t: float) -> CarlaWorldPose:
    return to_carla_world(OpenDrivePose(road_id=ROAD, lane_id=-1, s=s, t=t))


def _t_of(location: Any) -> float:
    pose = to_opendrive(
        CarlaWorldPose(x=location.x, y=location.y, z=location.z, yaw=0.0)
    )
    assert pose.road_id == ROAD
    return pose.t


class _RoadMap:
    """A CARLA map with a driving lane wherever the fixture road is within
    the kerb."""

    def get_waypoint(self, location: Any, project_to_road: bool, lane_type: int) -> Any:
        assert not project_to_road
        assert lane_type == int(carla.LaneType.Driving)
        return object() if abs(_t_of(location)) < -KERB_T else None


class _Walker:
    """A walker that walks as its last control says (CARLA 0.10's scale)."""

    def __init__(self, at: CarlaWorldPose, yaw: float) -> None:
        self.location = carla.Location(x=at.x, y=at.y, z=at.z)
        self.yaw = yaw
        self.control: Optional[carla.WalkerControl] = None

    def get_transform(self) -> carla.Transform:
        return carla.Transform(self.location, carla.Rotation(yaw=self.yaw))

    def get_velocity(self) -> carla.Vector3D:
        if self.control is None:
            return carla.Vector3D(0.0, 0.0, 0.0)
        speed = self.control.speed / 20.5
        d = self.control.direction
        return carla.Vector3D(d.x * speed, d.y * speed, 0.0)

    def apply_control(self, control: carla.WalkerControl) -> None:
        self.control = control

    def step(self) -> None:
        v = self.get_velocity()
        self.location = carla.Location(
            x=self.location.x + v.x * DT,
            y=self.location.y + v.y * DT,
            z=self.location.z,
        )

    @property
    def speed(self) -> float:
        return 0.0 if self.control is None else self.control.speed / 20.5


class _Vehicle:
    """A vehicle on the fixture road, at *s*, driving at *speed* along it
    (negative: towards decreasing ``s``)."""

    def __init__(self, s: float, speed: float = 0.0) -> None:
        self.id = 7
        self.s = s
        self.speed = speed

    def get_location(self) -> carla.Location:
        at = _world_at(self.s, 0.0)
        return carla.Location(x=at.x, y=at.y, z=at.z)

    def get_velocity(self) -> carla.Vector3D:
        a, b = _world_at(self.s, 0.0), _world_at(self.s + 1.0, 0.0)
        length = math.hypot(b.x - a.x, b.y - a.y)
        return carla.Vector3D(
            (b.x - a.x) / length * self.speed, (b.y - a.y) / length * self.speed, 0.0
        )

    def step(self) -> None:
        self.s += self.speed * DT


class _Scene:
    def __init__(self, vehicles: list[_Vehicle], start_t: float = START_T) -> None:
        start = _world_at(CROSSING_S, start_t)
        across = _world_at(CROSSING_S, 0.0)
        yaw = math.degrees(math.atan2(across.y - start.y, across.x - start.x))
        self.walker = _Walker(start, yaw)
        self.entity = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1", spawn_location=SpawnTransform(carla.Transform())
            )
        )
        self.entity._walker = self.walker  # type: ignore[assignment]
        self.vehicles = vehicles
        self.world = MagicMock()
        self.world.get_actors.return_value.filter.side_effect = (
            lambda pattern: list(self.vehicles) if pattern == "vehicle.*" else []
        )
        self.elapsed = 0.0
        self.reversed = False

    @property
    def t(self) -> float:
        return _t_of(self.walker.location)

    def run(
        self, action: WalkStraightAction, seconds: float, on_tick: Any = None
    ) -> None:
        for _ in range(round(seconds / DT)):
            action.tick(self.world, self.elapsed)
            self.walker.step()
            for vehicle in self.vehicles:
                vehicle.step()
            if self.walker.control is not None and self.walker.speed > 0:
                d = self.walker.control.direction
                self.reversed |= (
                    d.x * math.cos(math.radians(self.walker.yaw))
                    + d.y * math.sin(math.radians(self.walker.yaw))
                    < 0
                )
            if on_tick is not None:
                on_tick(self)
            self.elapsed += DT


@pytest.fixture
def scene(loaded_map: MapManager, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setattr(walk_module, "run_carla_map", lambda world: _RoadMap())
    scenes: list[_Scene] = []

    def make(*vehicles: _Vehicle, start_t: float = START_T) -> _Scene:
        built = _Scene(list(vehicles), start_t)
        register_entity("walker1", built.entity)
        scenes.append(built)
        return built

    yield make
    if scenes:
        unregister_entity("walker1")


def _action(**options: Any) -> WalkStraightAction:
    return WalkStraightAction("walker1", speed_ms=SPEED, **options)


def test_with_no_vehicle_near_it_walks_on_across(scene: Any) -> None:
    s = scene(_Vehicle(CROSSING_S + 130.0))
    action = _action(turn_back_ttc_s=2.0, turn_back_distance_m=10.0)
    s.run(action, 3.0)
    # 6 m at 2 m/s: a metre to the kerb, then 5 m onto the road.
    assert s.t == pytest.approx(START_T + 6.0, abs=0.2)
    assert not s.reversed
    # Past turning back, the walk was settled and the action is over.
    assert action.settled
    assert action.state is ActionState.COMPLETE


def test_a_vehicle_too_close_along_the_road_turns_it_back_to_the_kerb(
    scene: Any,
) -> None:
    s = scene(_Vehicle(CROSSING_S + 30.0))
    action = _action(turn_back_distance_m=10.0, turn_back_within_m=1.0)

    def close_in(scene: _Scene) -> None:
        # Half a metre onto the road, the vehicle is 8 m away.
        if scene.t > KERB_T + 0.5:
            scene.vehicles[0].s = CROSSING_S + 8.0

    s.run(action, 0.9, close_in)
    assert s.reversed
    # It turned within a tick or two of the vehicle closing in.
    assert KERB_T + 0.3 < s.t < KERB_T + 0.7
    s.run(action, 1.5)
    # Walked back to half a metre behind the kerb, and stands there.
    assert s.walker.speed == 0.0
    assert KERB_T - 0.65 < s.t < KERB_T - 0.45
    assert action.state is ActionState.COMPLETE


def test_it_steps_back_no_further_than_where_it_set_off(scene: Any) -> None:
    # Set off 0.2 m behind the kerb: back there, not half a metre behind it.
    s = scene(_Vehicle(CROSSING_S + 30.0), start_t=KERB_T - 0.2)
    action = _action(turn_back_distance_m=10.0)

    def close_in(scene: _Scene) -> None:
        if scene.t > KERB_T + 0.5:
            scene.vehicles[0].s = CROSSING_S + 8.0

    s.run(action, 2.5, close_in)
    assert s.reversed
    assert s.walker.speed == 0.0
    assert KERB_T - 0.35 < s.t < KERB_T - 0.15


def test_a_vehicle_closing_within_the_ttc_turns_it_back(scene: Any) -> None:
    # 45 m away at 20 m/s: 1.5 s out after 0.75 s, when the pedestrian is
    # half a metre onto the road.
    s = scene(_Vehicle(CROSSING_S + 45.0, speed=-20.0))
    action = _action(turn_back_ttc_s=1.5)
    s.run(action, 0.7)
    assert not s.reversed
    s.run(action, 0.15)
    assert s.reversed
    s.run(action, 1.0)
    assert s.walker.speed == 0.0
    assert s.t < KERB_T - 0.4
    assert action.state is ActionState.COMPLETE


def test_a_vehicle_driving_away_has_no_ttc(scene: Any) -> None:
    s = scene(_Vehicle(CROSSING_S + 5.0, speed=15.0))
    action = _action(turn_back_ttc_s=3.0)
    s.run(action, 2.0)
    assert not s.reversed
    assert s.t > KERB_T + 2.0


def test_not_on_the_road_yet_it_stays_off_it(scene: Any) -> None:
    s = scene(_Vehicle(CROSSING_S + 8.0))
    action = _action(turn_back_distance_m=10.0)
    s.run(action, 1.0)
    # Stopped a step from where it set off, back no further than that.
    assert s.walker.speed == 0.0
    # (A tick of overshoot: 0.1 m at 2 m/s.)
    assert START_T - 0.15 < s.t < START_T + 0.15
    assert action.state is ActionState.COMPLETE


def test_too_far_onto_the_road_it_keeps_going(scene: Any) -> None:
    s = scene(_Vehicle(CROSSING_S + 30.0))
    action = _action(turn_back_distance_m=10.0, turn_back_within_m=1.0)

    def close_in(scene: _Scene) -> None:
        if scene.t > KERB_T + 1.5:
            scene.vehicles[0].s = CROSSING_S + 8.0

    s.run(action, 2.5, close_in)
    assert not s.reversed
    assert s.walker.speed == pytest.approx(SPEED)
    assert s.t > KERB_T + 2.5


def test_a_backend_that_walks_it_is_told_to_walk_it_back(scene: Any) -> None:
    s = scene(_Vehicle(CROSSING_S + 30.0))
    calls: list[tuple[float, Optional[float]]] = []

    class _Backend:
        def walk_straight(
            self, entity: Any, speed_ms: float, yaw_deg: Any = None
        ) -> bool:
            calls.append((speed_ms, yaw_deg))
            yaw = math.radians(s.walker.yaw if yaw_deg is None else yaw_deg)
            s.walker.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0),
                    speed=speed_ms * 20.5,
                )
            )
            return True

    s.entity.set_traffic_backend(_Backend())  # type: ignore[arg-type]
    action = _action(turn_back_distance_m=10.0)

    def close_in(scene: _Scene) -> None:
        if scene.t > KERB_T + 0.5:
            scene.vehicles[0].s = CROSSING_S + 8.0

    s.run(action, 2.0, close_in)
    assert calls[0] == (SPEED, None)
    assert calls[1] == (SPEED, pytest.approx(s.walker.yaw + 180.0))
    assert calls[-1][0] == 0.0
    assert action.state is ActionState.COMPLETE


def test_without_turn_back_options_the_walk_is_over_at_once(scene: Any) -> None:
    s = scene(_Vehicle(CROSSING_S + 2.0))
    action = _action()
    assert not action.turns_back
    s.run(action, 0.15)
    # Commanded once and over, as before: the vehicle right there changes nothing.
    assert action.state is ActionState.COMPLETE
    assert s.walker.speed == pytest.approx(SPEED)
    s.world.get_actors.assert_not_called()


@pytest.mark.parametrize(
    "option", ["turn_back_ttc_s", "turn_back_distance_m", "turn_back_within_m"]
)
def test_a_negative_turn_back_option_is_refused(option: str) -> None:
    with pytest.raises(ValueError, match=f"{option} must not be negative"):
        options: dict[str, Any] = {option: -1.0}
        WalkStraightAction("walker1", **options)
