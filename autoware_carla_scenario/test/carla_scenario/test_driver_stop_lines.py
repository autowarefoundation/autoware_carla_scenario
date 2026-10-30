"""driver.stop_lines, split out of driver.renderer so the library check
compiles it (#751): the light the renderer reports as governing the ego, the
distance to its stop line, and every call they make on CARLA (so the caches
fill and are read as before) are the ones the renderer made before the split.

:class:`_MasterRenderer` is the renderer's stop-line and governing-light logic
as it stood before the split (``driver/renderer.py`` at 2361b18d), copied
verbatim but for its constructor; each case runs it and the split renderer on
two copies of the same fake world, step after step, and compares them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from autoware_carla_scenario.driver import renderer, stop_lines
from autoware_carla_scenario.driver.base import DriverClientConfig
from autoware_carla_scenario.driver.geometry import Pose
from autoware_carla_scenario.driver.observation import to_local_pose
from autoware_carla_scenario.driver.renderer import RendererDataBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The renderer before the split
# ---------------------------------------------------------------------------

_STOP_LINE_SEARCH_M: float = 8.0
_STOP_LINE_STEP_M: float = 0.5
_NO_STOP_LINE_M: float = -1.0


class _MasterRenderer:
    """``RendererDataBuilder``'s traffic-light lookups before #751."""

    def __init__(self, world: Any, ego_actor: Any, config: Any) -> None:
        self._world = world
        self._ego = ego_actor
        self._config = config
        self._map = world.get_map()
        self._stop_lines: Optional[Dict[Tuple[int, int], List[Any]]] = None
        self._stop_line_points: Dict[int, List[NDArray[np.float64]]] = {}

    def _governing_traffic_light(self, ego_pose: Pose) -> Optional[Any]:
        if self._config.traffic_light_sight_distance_m > 0.0:
            waypoint = self._ego_waypoint()
            if waypoint is not None and waypoint.is_junction:
                return None
            lanes = self._lanes_ahead(self._config.traffic_light_sight_distance_m)
            light = self._nearest_light_ahead(lanes, ego_pose)
            if light is not None:
                return light
        try:
            if self._ego.is_at_traffic_light():
                return self._ego.get_traffic_light()
        except (RuntimeError, AttributeError):
            logger.debug("is_at_traffic_light() unavailable", exc_info=True)
        return None

    def _ego_waypoint(self) -> Optional[Any]:
        return self._map.get_waypoint(self._ego.get_location(), project_to_road=True)

    def _lanes_ahead(self, distance_m: float) -> List[Tuple[int, int]]:
        waypoint = self._ego_waypoint()
        if waypoint is None or waypoint.is_junction:
            return []

        step = max(1.0, self._config.route_resolution_m)
        lanes = [(waypoint.road_id, waypoint.lane_id)]
        travelled = 0.0
        while travelled < distance_m:
            options = waypoint.next(step)
            if not options:
                break
            waypoint = options[0]
            travelled += step
            lane = (waypoint.road_id, waypoint.lane_id)
            if lane != lanes[-1]:
                lanes.append(lane)
            if waypoint.is_junction:
                break
        return lanes

    def _nearest_light_ahead(
        self, lanes: List[Tuple[int, int]], ego_pose: Pose
    ) -> Optional[Any]:
        sight = self._config.traffic_light_sight_distance_m
        nearest: Optional[Any] = None
        nearest_distance = float("inf")
        for light in self._lights_on_lanes(lanes):
            distance = self._light_distance(light, ego_pose)
            if distance < 0.0 or distance > sight:
                continue
            if distance < nearest_distance:
                nearest, nearest_distance = light, distance
        return nearest

    def _lights_on_lanes(self, lanes: List[Tuple[int, int]]) -> List[Any]:
        if not lanes:
            return []
        index = self._stop_lines_by_lane()
        found: List[Any] = []
        for lane in lanes:
            found.extend(index.get(lane, ()))
        return found

    def _stop_lines_by_lane(self) -> Dict[Tuple[int, int], List[Any]]:
        if self._stop_lines is None:
            index: Dict[Tuple[int, int], List[Any]] = {}
            for light in self._world.get_actors().filter("traffic.traffic_light*"):
                for waypoint in self._stop_waypoints(light):
                    key = (waypoint.road_id, waypoint.lane_id)
                    index.setdefault(key, []).append(light)
            self._stop_lines = index
        return self._stop_lines

    @staticmethod
    def _stop_waypoints(light: Any) -> List[Any]:
        try:
            return list(light.get_stop_waypoints())
        except (RuntimeError, AttributeError):
            logger.debug("get_stop_waypoints() unavailable", exc_info=True)
            return []

    def _light_distance(self, light: Optional[Any], ego_pose: Pose) -> float:
        if light is None:
            return _NO_STOP_LINE_M

        points = self._stop_line_points_for(light)
        if not points:
            return _NO_STOP_LINE_M

        # +x in the rig frame is straight ahead.
        longitudinal = ego_pose.inverse().transform_points(np.stack(points))[:, 0]
        ahead = longitudinal[longitudinal >= 0.0]
        if len(ahead):
            return float(ahead.min())
        return float(longitudinal.max())

    def _stop_line_points_for(self, light: Any) -> List[NDArray[np.float64]]:
        key = int(getattr(light, "id", 0))
        cached = self._stop_line_points.get(key)
        if cached is None:
            cached = [
                to_local_pose(self._junction_mouth(waypoint).transform).position
                for waypoint in self._stop_waypoints(light)
            ]
            if not cached:
                cached = [to_local_pose(light.get_transform()).position]
            self._stop_line_points[key] = cached
        return cached

    @staticmethod
    def _junction_mouth(stop: Any) -> Any:
        waypoint, travelled = stop, 0.0
        while travelled < _STOP_LINE_SEARCH_M:
            options = waypoint.next(_STOP_LINE_STEP_M)
            if not options:
                return stop
            waypoint = options[0]
            travelled += _STOP_LINE_STEP_M
            if waypoint.is_junction:
                return waypoint
        return stop


# ---------------------------------------------------------------------------
# A fake world that logs every call made on it
# ---------------------------------------------------------------------------


def _transform(x: float, y: float, yaw: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=0.0),
        rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw),
    )


@dataclass
class _Road:
    """A straight lane: from *start* along *heading* (+1 is CARLA +x, -1 is -x)
    for *length* metres at CARLA ``y`` *y*.

    *segments* are ``(from_s, road_id, lane_id)``: the lane a waypoint at
    distance ``s`` along the road is on.  *junctions* are ``(from_s, to_s)``
    spans of junction waypoints.
    """

    log: List[str]
    start: float = 0.0
    heading: float = 1.0
    length: float = 200.0
    y: float = 0.0
    segments: Sequence[Tuple[float, int, int]] = ((0.0, 1, -1),)
    junctions: Sequence[Tuple[float, float]] = ()

    def at(self, s: float) -> "_Waypoint":
        return _Waypoint(self, s)


class _Waypoint:
    def __init__(self, road: _Road, s: float) -> None:
        self._road = road
        self._s = s
        lane = [seg for seg in road.segments if seg[0] <= s + 1e-9][-1]
        self.road_id = lane[1]
        self.lane_id = lane[2]
        self.is_junction = any(a <= s < b for a, b in road.junctions)
        self.transform = _transform(
            road.start + road.heading * s,
            road.y,
            0.0 if road.heading > 0 else 180.0,
        )

    def next(self, distance: float) -> List["_Waypoint"]:
        self._road.log.append(f"next({self._s:g}, {distance:g})")
        s = self._s + distance
        return [] if s > self._road.length else [_Waypoint(self._road, s)]


class _Actors(list):
    def __init__(self, log: List[str], actors: Sequence[Any]) -> None:
        super().__init__(actors)
        self._log = log

    def filter(self, pattern: str) -> List[Any]:
        self._log.append(f"filter({pattern})")
        needle = pattern.strip("*")
        return [a for a in self if needle in a.type_id]


class _Light:
    type_id = "traffic.traffic_light"

    def __init__(
        self,
        log: List[str],
        light_id: int,
        stops: Sequence[_Waypoint],
        at: Tuple[float, float] = (0.0, 0.0),
        fails: Optional[type] = None,
    ) -> None:
        self._log = log
        self.id = light_id
        self._stops = list(stops)
        self._at = at
        self._fails = fails

    def get_stop_waypoints(self) -> List[_Waypoint]:
        self._log.append(f"light {self.id}: get_stop_waypoints()")
        if self._fails is not None:
            raise self._fails("not in this CARLA")
        return list(self._stops)

    def get_transform(self) -> SimpleNamespace:
        self._log.append(f"light {self.id}: get_transform()")
        return _transform(*self._at)


class _Ego:
    type_id = "vehicle.tesla.model3"
    id = 1

    def __init__(
        self,
        log: List[str],
        at: Callable[[], float],
        trigger: Optional[_Light] = None,
        fails: Optional[type] = None,
    ) -> None:
        self._log = log
        self._at = at
        self._trigger = trigger
        self._fails = fails

    def get_location(self) -> SimpleNamespace:
        self._log.append("ego: get_location()")
        return SimpleNamespace(x=self._at(), y=0.0, z=0.0)

    def is_at_traffic_light(self) -> bool:
        self._log.append("ego: is_at_traffic_light()")
        if self._fails is not None:
            raise self._fails("not a vehicle")
        return self._trigger is not None

    def get_traffic_light(self) -> Optional[_Light]:
        self._log.append("ego: get_traffic_light()")
        return self._trigger


class _Map:
    def __init__(self, log: List[str], waypoint: Callable[[float], Any]) -> None:
        self._log = log
        self._waypoint = waypoint
        self.name = "Fake"

    def get_waypoint(self, location: Any, project_to_road: bool = True) -> Any:
        self._log.append(f"map: get_waypoint({location.x:g}, {project_to_road})")
        return self._waypoint(location.x)


@dataclass
class _World:
    log: List[str]
    map: _Map
    actors: _Actors
    ego: _Ego
    #: Where the ego is, set step by step: the CARLA x of the ego.
    x: List[float] = field(default_factory=lambda: [0.0])

    def get_map(self) -> _Map:
        return self.map

    def get_actors(self) -> _Actors:
        self.log.append("world: get_actors()")
        return self.actors


@dataclass
class _Case:
    """A world, the driver settings and where the ego is at each step."""

    build: Callable[[], _World]
    config: DriverClientConfig = field(default_factory=DriverClientConfig)
    #: ``(ego CARLA x, ego pose)`` at each step; the pose defaults to
    #: facing +x at that x.
    steps: Sequence[Tuple[float, Optional[Pose]]] = ((0.0, None),)


def _world(
    *,
    road: Callable[[List[str]], _Road],
    lights: Callable[[List[str], _Road], Sequence[_Light]] = lambda log, road: [],
    trigger: Optional[int] = None,
    ego_fails: Optional[type] = None,
    off_road: Optional[Tuple[float, float]] = None,
) -> Callable[[], _World]:
    """A world on *road*, the ego on it wherever the step puts it.

    *trigger* names the light (by id) whose trigger volume the ego stands in;
    *off_road* is a span of x where the map finds no waypoint for the ego.
    """

    def build() -> _World:
        log: List[str] = []
        lane = road(log)
        found = list(lights(log, lane))
        x = [0.0]

        def waypoint(at: float) -> Optional[_Waypoint]:
            if off_road is not None and off_road[0] <= at < off_road[1]:
                return None
            return lane.at((at - lane.start) * lane.heading)

        held = next((light for light in found if light.id == trigger), None)
        ego = _Ego(log, lambda: x[0], held, ego_fails)
        actors = _Actors(log, [*found, ego])
        return _World(log, _Map(log, waypoint), actors, ego, x)

    return build


def _straight(**kwargs: Any) -> Callable[[List[str]], _Road]:
    return lambda log: _Road(log, **kwargs)


def _stop_at(*stops: float, lane_id: int = -1, road_id: int = 1) -> Callable:
    """Lights 1, 2, ... with their stop waypoint at each of *stops* on the
    ego's road."""

    def lights(log: List[str], road: _Road) -> List[_Light]:
        return [
            _Light(log, i, [road.at(s)], at=(s + 3.0, 4.0))
            for i, s in enumerate(stops, start=1)
        ]

    return lights


_JUNCTION = ((40.0, 60.0), (120.0, 140.0))
_LANES = ((0.0, 1, -1), (15.0, 2, -1), (30.0, 3, -1), (60.0, 4, -1), (140.0, 5, -1))

_CASES: Dict[str, _Case] = {
    "no lights": _Case(_world(road=_straight(junctions=_JUNCTION))),
    "a light ahead, its junction past the line": _Case(
        _world(road=_straight(junctions=_JUNCTION), lights=_stop_at(35.0)),
        steps=[(0.0, None), (10.0, None), (34.0, None), (36.0, None), (45.0, None)],
    ),
    "two lights on the lane, the far one listed first": _Case(
        _world(
            road=_straight(junctions=((40.0, 60.0), (90.0, 100.0))),
            lights=_stop_at(85.0, 35.0),
        ),
        config=DriverClientConfig(traffic_light_sight_distance_m=120.0),
        steps=[(0.0, None), (20.0, None), (61.0, None), (80.0, None)],
    ),
    "a light behind the ego on its lane": _Case(
        _world(road=_straight(junctions=((100.0, 110.0),)), lights=_stop_at(5.0)),
        steps=[(20.0, None), (30.0, None)],
    ),
    "a light beyond the sight distance": _Case(
        _world(road=_straight(junctions=((85.0, 95.0),)), lights=_stop_at(80.0)),
        config=DriverClientConfig(traffic_light_sight_distance_m=40.0),
        steps=[(0.0, None), (30.0, None), (45.0, None)],
    ),
    "a light on another lane": _Case(
        _world(
            road=_straight(junctions=_JUNCTION),
            lights=lambda log, road: [
                _Light(
                    log,
                    7,
                    [
                        _Road(
                            log,
                            start=200.0,
                            heading=-1.0,
                            y=-3.5,
                            segments=((0.0, 1, 1),),
                            junctions=((140.0, 160.0),),
                        ).at(135.0)
                    ],
                )
            ],
        ),
        steps=[(0.0, None), (30.0, None)],
    ),
    "lane changes ahead, lights on later lanes": _Case(
        _world(
            road=_straight(segments=_LANES, junctions=_JUNCTION),
            lights=lambda log, road: [
                _Light(log, 3, [road.at(110.0)]),
                _Light(log, 4, [road.at(33.0)]),
                _Light(log, 5, [road.at(20.0), road.at(116.0)]),
            ],
        ),
        config=DriverClientConfig(
            traffic_light_sight_distance_m=150.0, route_resolution_m=3.0
        ),
        steps=[(0.0, None), (16.0, None), (25.0, None), (70.0, None), (112.0, None)],
    ),
    "a light with several stop lines, one on our lane": _Case(
        _world(
            road=_straight(junctions=_JUNCTION),
            lights=lambda log, road: [
                _Light(
                    log,
                    9,
                    [
                        _Road(log, y=3.5, segments=((0.0, 1, -2),)).at(36.0),
                        road.at(37.0),
                        _Road(
                            log,
                            start=100.0,
                            heading=-1.0,
                            y=-3.5,
                            segments=((0.0, 1, 1),),
                            junctions=((40.0, 60.0),),
                        ).at(35.0),
                    ],
                )
            ],
        ),
        steps=[(0.0, None), (38.0, None)],
    ),
    "a stop line on open road": _Case(
        _world(road=_straight(junctions=((90.0, 95.0),)), lights=_stop_at(30.0)),
        steps=[(0.0, None), (29.0, None), (31.0, None)],
    ),
    "a junction just within the mouth walk": _Case(
        _world(road=_straight(junctions=((38.0, 50.0),)), lights=_stop_at(30.0)),
        steps=[(0.0, None)],
    ),
    "the road ends before the sight distance": _Case(
        _world(road=_straight(length=40.0), lights=_stop_at(25.0, 38.0)),
        steps=[(0.0, None), (20.0, None), (39.0, None)],
    ),
    "the ego inside a junction": _Case(
        _world(road=_straight(junctions=_JUNCTION), lights=_stop_at(35.0), trigger=1),
        steps=[(45.0, None), (50.0, None), (61.0, None)],
    ),
    "the ego off the road, standing in a trigger volume": _Case(
        _world(
            road=_straight(junctions=_JUNCTION),
            lights=_stop_at(35.0),
            trigger=1,
            off_road=(0.0, 10.0),
        ),
        steps=[(0.0, None), (12.0, None)],
    ),
    "zero sight distance: the trigger volume decides": _Case(
        _world(road=_straight(junctions=_JUNCTION), lights=_stop_at(35.0), trigger=1),
        config=DriverClientConfig(traffic_light_sight_distance_m=0.0),
        steps=[(0.0, None), (34.0, None), (50.0, None)],
    ),
    "zero sight distance, no light": _Case(
        _world(road=_straight(junctions=_JUNCTION), lights=_stop_at(35.0)),
        config=DriverClientConfig(traffic_light_sight_distance_m=0.0),
    ),
    "the walk finds nothing, the trigger volume has a light": _Case(
        _world(
            road=_straight(junctions=_JUNCTION),
            lights=lambda log, road: [
                _Light(log, 1, [road.at(5.0)]),
                _Light(log, 2, [], at=(12.0, 0.0)),
            ],
            trigger=2,
        ),
        steps=[(20.0, None), (25.0, None)],
    ),
    "the trigger volume's light, its stop lines all behind": _Case(
        _world(
            road=_straight(junctions=_JUNCTION),
            lights=lambda log, road: [_Light(log, 1, [road.at(5.0), road.at(10.0)])],
            trigger=1,
        ),
        steps=[(20.0, None), (30.0, None)],
    ),
    "is_at_traffic_light() unavailable": _Case(
        _world(road=_straight(), lights=_stop_at(35.0), ego_fails=AttributeError),
        steps=[(0.0, None), (10.0, None)],
    ),
    "is_at_traffic_light() fails in CARLA": _Case(
        _world(road=_straight(), ego_fails=RuntimeError),
    ),
    "get_stop_waypoints() unavailable": _Case(
        _world(
            road=_straight(junctions=_JUNCTION),
            lights=lambda log, road: [
                _Light(log, 1, [road.at(35.0)], fails=AttributeError),
                _Light(log, 2, [road.at(35.0)], fails=RuntimeError),
                _Light(log, 3, [road.at(30.0)]),
            ],
            trigger=1,
        ),
        steps=[(0.0, None), (32.0, None), (33.0, None)],
    ),
    "a yawed and offset ego": _Case(
        _world(road=_straight(junctions=_JUNCTION), lights=_stop_at(35.0, 120.0)),
        config=DriverClientConfig(traffic_light_sight_distance_m=200.0),
        steps=[
            (0.0, Pose.from_xyz_yaw(0.0, 10.0, 0.0, 0.3)),
            (10.0, Pose.from_xyz_yaw(10.0, -2.0, 0.0, -0.2)),
            (20.0, Pose.from_xyz_yaw(20.0, 0.0, 0.0, math.pi)),
            (30.0, Pose.from_xyz_yaw(30.0, 0.0, 0.0, math.pi / 2)),
        ],
    ),
    "a fine route resolution": _Case(
        _world(
            road=_straight(segments=_LANES, junctions=_JUNCTION), lights=_stop_at(33.0)
        ),
        config=DriverClientConfig(route_resolution_m=0.25),
        steps=[(0.0, None), (14.5, None)],
    ),
    "a coarse route resolution walks past a junction": _Case(
        _world(
            road=_straight(segments=_LANES, junctions=((41.0, 43.0),)),
            lights=_stop_at(33.0, 100.0),
        ),
        config=DriverClientConfig(
            route_resolution_m=7.0, traffic_light_sight_distance_m=150.0
        ),
        steps=[(0.0, None), (35.0, None)],
    ),
}


@pytest.fixture(autouse=True)
def _the_world_s_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake world's map, not one a MapManager of another test holds."""
    monkeypatch.setattr(renderer, "run_carla_map", lambda world: world.get_map())


def _light_id(light: Optional[Any]) -> Optional[int]:
    return None if light is None else int(light.id)


def _run(case: _Case, split: bool) -> List[Tuple[Optional[int], float, List[str]]]:
    """Each step's governing light, its distance and the calls made on CARLA."""
    world = case.build()
    if split:
        lookup: Any = RendererDataBuilder(world, world.ego, case.config)
    else:
        lookup = _MasterRenderer(world, world.ego, case.config)
    world.log.clear()
    out = []
    for x, pose in case.steps:
        world.x[0] = x
        ego_pose = pose if pose is not None else Pose.from_xyz_yaw(x, 0.0, 0.0, 0.0)
        light = lookup._governing_traffic_light(ego_pose)
        distance = lookup._light_distance(light, ego_pose)
        out.append((_light_id(light), distance, list(world.log)))
        world.log.clear()
    return out


@pytest.mark.parametrize("name", sorted(_CASES))
def test_the_split_renderer_finds_the_same_light_as_before(name: str) -> None:
    case = _CASES[name]
    before = _run(case, split=False)
    after = _run(case, split=True)
    for step, (old, new) in enumerate(zip(before, after)):
        assert new[0] == old[0], f"step {step}: governing light"
        # Bit for bit: the same arithmetic on the same points.
        assert new[1] == old[1], f"step {step}: distance"
        assert new[2] == old[2], f"step {step}: calls made on CARLA"
    assert len(after) == len(before)


def test_the_cases_cover_what_they_are_named_for() -> None:
    """Lights found by the walk, by the fallback, and none; distances ahead,
    behind and absent; caches read without a call on CARLA."""
    results = {name: _run(case, split=True) for name, case in _CASES.items()}
    lights = [light for runs in results.values() for light, _, _ in runs]
    distances = [d for runs in results.values() for _, d, _ in runs]
    assert None in lights and len({light for light in lights if light}) >= 5
    assert any(d > 0.0 for d in distances) and any(
        d < 0.0 and d != -1.0 for d in distances
    )
    assert -1.0 in distances
    walked = results["a light ahead, its junction past the line"]
    assert walked[0][0] == 1 and 35.0 < walked[0][1] <= 40.0
    # The second step reads the lane index and the stop points from the
    # caches: no stop waypoints asked for again.
    assert not any("get_stop_waypoints" in call for call in walked[1][2])
    fallback = results["zero sight distance: the trigger volume decides"]
    assert [light for light, _, _ in fallback] == [1, 1, 1]
    assert results["the ego inside a junction"][0][0] is None


def test_the_renderer_delegates_to_one_stop_lines_per_builder() -> None:
    world = _CASES["no lights"].build()
    builder = RendererDataBuilder(world, world.ego, DriverClientConfig())
    assert isinstance(builder._stop_lines, stop_lines.StopLines)
    assert builder._stop_lines._map is builder._map
    assert renderer.stop_waypoints is stop_lines.stop_waypoints


def test_a_light_s_stop_points_are_cached_by_its_actor_id() -> None:
    """Two light objects of the same actor share the cache, as before."""
    log: List[str] = []
    road = _Road(log, junctions=_JUNCTION)
    world = _world(road=lambda _: road)()
    lines = stop_lines.StopLines(world, world.ego, world.map, DriverClientConfig())
    first = _Light(log, 4, [road.at(35.0)])
    again = _Light(log, 4, [road.at(10.0)])
    assert lines.stop_line_points(first) is lines.stop_line_points(again)
    assert lines.light_distance(again, Pose.identity()) == pytest.approx(40.0)
