"""Placing a road user so a measure reads a value at the start
(:mod:`autoware_carla_scenario.placement`), and cut_in's measured gap."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import placement
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.coordinate.transform import to_carla_world
from autoware_carla_scenario.measures import (
    VEHICLE_AHEAD_GAP_M,
    Bumpers,
    read_measure,
    set_measured_scenario,
)
from autoware_carla_scenario.placement import (
    place_at_vehicle_ahead_gap,
    solve_along,
    vehicle_bumpers,
)

from . import _lane_maps
from ._lane_maps import LaneMap, bend, build_road, straight

_DATA = Path(__file__).resolve().parents[3] / "data"


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


def test_the_solver_finds_a_linear_crossing() -> None:
    s = solve_along(lambda s: 2.0 * s - 4.0, 13.0, 0.0, 50.0)

    assert s == pytest.approx(8.5, abs=1e-3)


def test_the_solver_takes_the_first_crossing() -> None:
    # Up to 10 at s = 10, back down after: 5 is read at s = 5 first.
    s = solve_along(lambda s: 10.0 - abs(s - 10.0), 5.0, 0.0, 30.0, step=0.7)

    assert s == pytest.approx(5.0, abs=1e-3)


def test_the_solver_skips_where_nothing_is_read() -> None:
    s = solve_along(lambda s: None if s < 3.0 else s, 7.25, 0.0, 10.0)

    assert s == pytest.approx(7.25, abs=1e-3)


def test_the_solver_says_what_it_can_reach() -> None:
    with pytest.raises(
        ValueError, match=r"the gap cannot read 40: .* reads 0\.00 to 20\.00"
    ):
        solve_along(lambda s: s, 40.0, 0.0, 20.0, what="the gap")
    with pytest.raises(ValueError, match="reads nothing"):
        solve_along(lambda s: None, 1.0, 0.0, 5.0)


# ---------------------------------------------------------------------------
# Placement -> measure, on synthetic Lanelet2 maps
# ---------------------------------------------------------------------------

#: The ego's and the NPC's bumpers (a Lincoln MKZ and a Mini, about).
_EGO = Bumpers(front=2.6, rear=2.2)
_NPC = Bumpers(front=1.9, rear=1.8)


def _measured(ego: carla.Transform, npc: Lanelet2Pose) -> Any:
    """What the measure reads with the ego at *ego* and the NPC spawned at
    *npc*, as a Lanelet2Pose spawns (x, y and heading off the lanelet)."""
    at = to_carla_world(npc)
    world = _World(
        [
            _Actor(1, str(EGO_ROLE_NAME), ego, _EGO),
            _Actor(2, "npc1", at.to_carla_transform(), _NPC),
        ]
    )
    return read_measure(VEHICLE_AHEAD_GAP_M, world)


def _ego_at(built: LaneMap, lane: str, segment: int, s: float) -> carla.Transform:
    return to_carla_world(
        Lanelet2Pose(lanelet_id=built.id(lane, segment), s=s)
    ).to_carla_transform()


@pytest.fixture
def straight_road() -> Iterator[LaneMap]:
    built = build_road(
        straight(300.0),
        [0.0, 100.0, 200.0, 300.0],
        [-1.75, 1.75, 5.25],
        {"ego": (0, False), "left": (1, False)},
    )
    yield from _lane_maps.loaded(built)


_RADIUS = 120.0


@pytest.fixture
def curve() -> Iterator[LaneMap]:
    """Town04's highway: three lanes round a bend, the outer one longer, so
    the same ``s`` lies at different places on lanelets beside each other."""
    arc = _RADIUS * math.radians(90.0)
    built = build_road(
        bend(30.0, _RADIUS, 90.0, 80.0),
        [0.0, 30.0, 30.0 + arc / 2, 30.0 + arc, 30.0 + arc + 80.0],
        [-5.25, -1.75, 1.75, 5.25],
        {"outer": (0, False), "ego": (1, False), "inner": (2, False)},
    )
    yield from _lane_maps.loaded(built)


@pytest.fixture
def widths() -> Iterator[LaneMap]:
    built = build_road(
        straight(200.0),
        [0.0, 200.0],
        [-8.25, -1.25, 1.25, 3.75],
        {"wide": (0, False), "ego": (1, False), "narrow": (2, False)},
    )
    yield from _lane_maps.loaded(built)


@pytest.fixture(autouse=True)
def _no_scenario() -> Iterator[None]:
    set_measured_scenario(None)
    yield
    set_measured_scenario(None)


def test_on_a_straight_road_the_gap_is_bumper_to_bumper(straight_road: LaneMap) -> None:
    ego = _ego_at(straight_road, "ego", 0, 40.0)

    pose = place_at_vehicle_ahead_gap(
        ego, straight_road.id("left"), 15.0, ego_bumpers=_EGO, npc_bumpers=_NPC
    )

    # The NPC's origin 15 m plus both overhangs ahead of the ego's.
    assert pose.s == pytest.approx(40.0 + 2.6 + 15.0 + 1.8, abs=1e-3)
    assert _measured(ego, pose) == pytest.approx(15.0, abs=1e-3)


@pytest.mark.parametrize("lane", ["outer", "inner"])
@pytest.mark.parametrize("target", [2.0, 8.0, 15.0, 25.0])
@pytest.mark.parametrize("ego_s", [5.0, 40.0])
def test_on_a_curve_placement_reads_back_as_the_measure(
    curve: LaneMap, lane: str, target: float, ego_s: float
) -> None:
    ego = _ego_at(curve, "ego", 1, ego_s)

    # The NPC's lanelet is the one beside the ego's; a gap past its end is
    # not this lanelet's to stage.
    pose = place_at_vehicle_ahead_gap(
        ego, curve.id(lane, 1), target, ego_bumpers=_EGO, npc_bumpers=_NPC
    )

    assert _measured(ego, pose) == pytest.approx(target, abs=0.05)
    # The raw setting -- the NPC's s the ego's plus the gap and overhangs --
    # is off by the lanes' difference in length up to there.
    raw = Lanelet2Pose(lanelet_id=curve.id(lane, 1), s=ego_s + 2.6 + target + 1.8)
    error = _measured(ego, raw) - target
    assert (error < 0) == (lane == "outer")


def test_placement_on_the_next_lanelet_of_the_lane_beside(curve: LaneMap) -> None:
    """The ego near the end of its lanelet, the NPC on the lanelet after the
    one beside it: s runs on from one lanelet to the next."""
    ego = _ego_at(curve, "ego", 1, 85.0)

    pose = place_at_vehicle_ahead_gap(
        ego, curve.id("outer", 2), 20.0, ego_bumpers=_EGO, npc_bumpers=_NPC
    )

    assert _measured(ego, pose) == pytest.approx(20.0, abs=0.05)


def test_placement_in_a_wide_lane_beside(widths: LaneMap) -> None:
    ego = _ego_at(widths, "ego", 0, 30.0)

    pose = place_at_vehicle_ahead_gap(
        ego, widths.id("wide"), 12.0, ego_bumpers=_EGO, npc_bumpers=_NPC
    )

    assert _measured(ego, pose) == pytest.approx(12.0, abs=0.05)


def test_a_gap_the_lanelet_cannot_reach_is_refused(straight_road: LaneMap) -> None:
    ego = _ego_at(straight_road, "ego", 0, 60.0)

    with pytest.raises(ValueError, match=r"on lanelet \d+ \(length 100\.0 m\)"):
        place_at_vehicle_ahead_gap(
            ego, straight_road.id("left"), 50.0, ego_bumpers=_EGO, npc_bumpers=_NPC
        )


class _Probe:
    def __init__(self, actor_id: int, type_id: str = "vehicle.lincoln.mkz") -> None:
        self.id = actor_id
        self.type_id = type_id
        self.bounding_box = SimpleNamespace(
            location=SimpleNamespace(x=0.1), extent=SimpleNamespace(x=2.4)
        )
        self.destroyed = False
        #: Whether the world has ticked since it was spawned: CARLA's client
        #: destroys nothing it does not know yet.
        self.known = True

    def destroy(self) -> bool:
        if not self.known:
            return False
        self.destroyed = True
        return True


class _Blueprint:
    def __init__(self, bp_id: str) -> None:
        self.id = bp_id
        self.attributes: dict[str, str] = {}

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[key] = value


class _Library:
    def filter(self, pattern: str) -> list[_Blueprint]:
        return [_Blueprint("vehicle.lincoln.mkz")]

    def find(self, bp_id: str) -> _Blueprint:
        return _Blueprint(bp_id)


class _ProbeWorld:
    """A world that spawns the probe after *refusals* refusals."""

    def __init__(
        self,
        refusals: int = 0,
        actors: list[Any] | None = None,
        *,
        synchronous: bool = True,
        known: bool = True,
    ) -> None:
        self.refusals = refusals
        self.actors = actors or []
        self.heights: list[float] = []
        self.probes: list[_Probe] = []
        self.synchronous = synchronous
        self.known = known
        self.ticks: list[str] = []

    def get_settings(self) -> Any:
        return SimpleNamespace(synchronous_mode=self.synchronous)

    def _ticked(self, how: str) -> None:
        self.ticks.append(how)
        for probe in self.probes:
            probe.known = True

    def tick(self) -> None:
        self._ticked("tick")

    def wait_for_tick(self) -> None:
        self._ticked("wait_for_tick")

    def get_blueprint_library(self) -> _Library:
        return _Library()

    def get_actors(self) -> list[Any]:
        return self.actors

    def try_spawn_actor(self, blueprint: _Blueprint, at: Any) -> _Probe | None:
        self.heights.append(at.location.z)
        assert blueprint.attributes["role_name"] == placement._PROBE_ROLE
        if len(self.heights) <= self.refusals:
            return None
        probe = _Probe(7)
        probe.known = self.known
        self.probes.append(probe)
        return probe


@pytest.fixture
def _no_bumpers_known(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(placement, "_BUMPERS", {})


def test_the_probe_has_the_role_the_measure_skips() -> None:
    from autoware_carla_scenario.measures import PROBE_ROLE

    assert placement._PROBE_ROLE == PROBE_ROLE


@pytest.mark.usefixtures("_no_bumpers_known")
def test_the_bumpers_are_measured_once_per_blueprint() -> None:
    world: Any = _ProbeWorld()

    first = vehicle_bumpers(world, "vehicle.lincoln.mkz")
    again = vehicle_bumpers(world, "vehicle.lincoln.mkz")

    assert first == again
    assert (first.front, first.rear) == pytest.approx((2.5, 2.3))
    assert world.heights == [500.0]
    assert [p.destroyed for p in world.probes] == [True]


@pytest.mark.usefixtures("_no_bumpers_known")
@pytest.mark.parametrize(
    ("synchronous", "tick"), [(True, "tick"), (False, "wait_for_tick")]
)
def test_a_probe_the_client_does_not_know_yet_is_destroyed_after_a_tick(
    synchronous: bool, tick: str
) -> None:
    """Destroyed before the world has ticked, the probe outlived its destroy():
    it fell from on high for the whole run, and the TrafficManager given it
    stalled the next synchronous tick until the process ran out of memory."""
    world: Any = _ProbeWorld(synchronous=synchronous, known=False)

    vehicle_bumpers(world, "vehicle.lincoln.mkz")

    assert world.ticks == [tick]
    assert [p.destroyed for p in world.probes] == [True]


@pytest.mark.usefixtures("_no_bumpers_known")
def test_a_probe_that_cannot_be_destroyed_is_reported() -> None:
    world: Any = _ProbeWorld(known=False)
    world._ticked = lambda how: world.ticks.append(how)  # stays unknown

    with pytest.raises(RuntimeError, match="could not be destroyed"):
        vehicle_bumpers(world, "vehicle.lincoln.mkz")


@pytest.mark.usefixtures("_no_bumpers_known")
def test_a_failed_read_is_the_error_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(box: Any) -> Any:
        raise KeyError("no extent")

    monkeypatch.setattr(placement, "bumpers_of_box", broken)
    world: Any = _ProbeWorld(known=False)
    world._ticked = lambda how: world.ticks.append(how)  # destroy fails too

    with pytest.raises(KeyError, match="no extent"):
        vehicle_bumpers(world, "vehicle.lincoln.mkz")


@pytest.mark.usefixtures("_no_bumpers_known")
def test_a_known_probe_is_destroyed_without_a_tick() -> None:
    world: Any = _ProbeWorld()

    vehicle_bumpers(world, "vehicle.lincoln.mkz")

    assert world.ticks == []
    assert [p.destroyed for p in world.probes] == [True]


@pytest.mark.usefixtures("_no_bumpers_known")
def test_a_vehicle_of_the_blueprint_already_spawned_is_read_not_probed() -> None:
    world: Any = _ProbeWorld(actors=[_Probe(1, "vehicle.lincoln.mkz")])

    bumpers = vehicle_bumpers(world, "vehicle.lincoln.mkz")

    assert (bumpers.front, bumpers.rear) == pytest.approx((2.5, 2.3))
    assert world.heights == []


@pytest.mark.usefixtures("_no_bumpers_known")
def test_a_refused_probe_is_tried_higher() -> None:
    world: Any = _ProbeWorld(refusals=2)

    bumpers = vehicle_bumpers(world, "vehicle.lincoln.mkz")

    assert (bumpers.front, bumpers.rear) == pytest.approx((2.5, 2.3))
    assert world.heights == [500.0, 1000.0, 1500.0]


@pytest.mark.usefixtures("_no_bumpers_known")
def test_no_probe_spawned_or_no_such_blueprint_is_an_error() -> None:
    world: Any = _ProbeWorld(refusals=3)

    with pytest.raises(ValueError, match="could be spawned to measure"):
        vehicle_bumpers(world, "vehicle.lincoln.mkz")
    assert world.heights == [500.0, 1000.0, 1500.0]
    with pytest.raises(ValueError, match="not a vehicle blueprint"):
        vehicle_bumpers(world, "vehicle.no.such")


# ---------------------------------------------------------------------------
# cut_in: the NPC placed so the measure reads the gap asked for
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nishishinjuku() -> Iterator[MapManager]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    yield manager
    MapManager._instance = saved


class _Actor:
    def __init__(
        self, actor_id: int, role: str, transform: Any, bumpers: Bumpers
    ) -> None:
        self.id = actor_id
        self.type_id = "vehicle.test"
        self.attributes = {"role_name": role}
        self._transform = transform
        self.bounding_box = SimpleNamespace(
            location=SimpleNamespace(x=(bumpers.front - bumpers.rear) / 2),
            extent=SimpleNamespace(x=(bumpers.front + bumpers.rear) / 2),
        )

    def get_location(self) -> Any:
        return self._transform.location

    def get_transform(self) -> Any:
        return self._transform

    def get_velocity(self) -> Any:
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)


class _World:
    def __init__(self, actors: list[_Actor]) -> None:
        self._actors = actors

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=id(self))

    def get_actors(self) -> list[_Actor]:
        return self._actors


def _cut_in_npc_transform(monkeypatch, **config: Any) -> tuple[Any, Any]:
    """Run cut_in's setup with the world faked: the ego's and the NPC's spawn
    transforms, placed as the scenario places them."""
    from autoware_carla_scenario import EgoConfig, SpawnTransform
    from autoware_carla_scenario.coordinate.transform import to_opendrive
    from autoware_carla_scenario.examples import cut_in
    from autoware_carla_scenario.examples.configs import CutInConfig

    spawned: list[Any] = []

    class _Npc:
        def __init__(self, cfg: Any) -> None:
            self.config = cfg

        def spawn(self, world: Any) -> None:
            spawned.append(self.config)

    def snap(pose: Any, world: Any, **_kw: Any) -> CarlaWorldPose:
        # As the real one: x, y and heading off the lanelet, z from the world.
        return to_carla_world(pose)

    spawn = Lanelet2Pose(lanelet_id=20, s=5.0)
    ego_at = to_carla_world(spawn).to_carla_transform()

    def setup_ego_spawn(self: Any) -> Any:
        self.ego_config.spawn_location = SpawnTransform(ego_at)
        return to_opendrive(to_carla_world(spawn))

    monkeypatch.setattr(cut_in, "snap_to_carla_road", snap)
    monkeypatch.setattr(
        cut_in,
        "vehicle_bumpers",
        lambda world, vehicle_type: _EGO if vehicle_type == "vehicle.x" else _NPC,
    )
    monkeypatch.setattr(cut_in, "VehicleEntity", _Npc)
    monkeypatch.setattr(cut_in.CutInScenario, "_setup_ego_spawn", setup_ego_spawn)
    monkeypatch.setattr(cut_in.CutInScenario, "world", property(lambda self: None))
    monkeypatch.setattr(cut_in.CutInScenario, "register_entity", lambda self, e: None)

    scenario = cut_in.CutInScenario(
        EgoConfig(vehicle_type="vehicle.x", spawn_location=SpawnTransform(ego_at)),
        spawn_pose=spawn,
        config=CutInConfig(npc_lanelet_id=19, **config),
    )
    scenario.setup()
    (npc,) = spawned
    return ego_at, npc.spawn_location.value


@pytest.mark.parametrize("gap", [6.0, 12.5, 25.0])
def test_cut_in_places_the_npc_at_the_measured_gap(
    nishishinjuku: MapManager, monkeypatch, gap: float
) -> None:
    set_measured_scenario(None)
    ego_at, npc_at = _cut_in_npc_transform(monkeypatch, npc_gap_m=gap)

    world = _World(
        [
            _Actor(1, str(EGO_ROLE_NAME), ego_at, _EGO),
            _Actor(2, "npc1", npc_at, _NPC),
        ]
    )

    assert read_measure(VEHICLE_AHEAD_GAP_M, world) == pytest.approx(gap, abs=1e-2)


def test_cut_in_still_takes_the_raw_offset(
    nishishinjuku: MapManager, monkeypatch
) -> None:
    _ego_at, npc_at = _cut_in_npc_transform(monkeypatch, npc_ahead_m=10.0)

    expected = to_carla_world(Lanelet2Pose(lanelet_id=19, s=15.0))
    assert npc_at.location.x == pytest.approx(expected.x)
    assert npc_at.location.y == pytest.approx(expected.y)
