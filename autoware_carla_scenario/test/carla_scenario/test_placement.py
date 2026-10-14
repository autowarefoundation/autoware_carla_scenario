"""Placing a road user so a measure reads a value at the start
(:mod:`autoware_carla_scenario.placement`), and cut_in's measured gap."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import placement
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.coordinate.transform import to_carla_world
from autoware_carla_scenario.measures import (
    VEHICLE_AHEAD_GAP_M,
    in_ego_frame,
    read_measure,
    set_measured_scenario,
    vehicle_ahead_gap_between,
)
from autoware_carla_scenario.placement import place_at_vehicle_ahead_gap, solve_along

_DATA = Path(__file__).resolve().parents[3] / "data"


def _transform(x: float, y: float, yaw: float) -> carla.Transform:
    return carla.Transform(carla.Location(x=x, y=y, z=0.0), carla.Rotation(yaw=yaw))


# ---------------------------------------------------------------------------
# The measure's geometry
# ---------------------------------------------------------------------------


def test_the_ego_frame_is_ahead_and_to_the_right() -> None:
    ego = _transform(10.0, 5.0, 90.0)  # facing +y

    ahead, lateral = in_ego_frame(ego, carla.Location(x=7.0, y=25.0, z=0.0))

    assert ahead == pytest.approx(20.0)
    # CARLA is left-handed: facing +y, -x is on the right.
    assert lateral == pytest.approx(3.0)


def test_the_gap_is_read_only_for_a_vehicle_the_measure_counts() -> None:
    ego = _transform(0.0, 0.0, 0.0)

    assert vehicle_ahead_gap_between(ego, carla.Location(12.0, 3.5, 0.0)) == (
        pytest.approx(12.0)
    )
    assert vehicle_ahead_gap_between(ego, carla.Location(-1.0, 0.0, 0.0)) is None
    assert vehicle_ahead_gap_between(ego, carla.Location(10.0, 7.5, 0.0)) is None
    assert vehicle_ahead_gap_between(ego, carla.Location(150.0, 0.0, 0.0)) is None


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
# Synthetic lanelets: a straight pair, and a curve whose lanes differ in length
# ---------------------------------------------------------------------------

#: Lane width, in metres.
_WIDTH = 3.5


def _on(
    lanelets: dict[int, Callable[[float], CarlaWorldPose]],
    monkeypatch,
    length: float = 80.0,
) -> None:
    """Make *lanelets* (id -> pose at s, each *length* long) the map placement reads."""

    def to_world(pose: Lanelet2Pose) -> CarlaWorldPose:
        return lanelets[pose.lanelet_id](pose.s)

    monkeypatch.setattr(placement, "to_carla_world", to_world)
    monkeypatch.setattr(placement, "lanelet_length", lambda i: length)


def _straight(start_x: float, y: float) -> Callable[[float], CarlaWorldPose]:
    return lambda s: CarlaWorldPose(x=start_x + s, y=y, z=0.0, yaw=0.0)


#: The radius of the ego's lane on the curve, in metres.
_RADIUS = 400.0


def _arc(radius: float) -> Callable[[float], CarlaWorldPose]:
    """A lane turning right (CARLA's +yaw) about (0, _RADIUS), starting on the
    line x = 0: s along it is radius * angle, so an outer lane is longer and
    the same s is further back on it."""

    def at(s: float) -> CarlaWorldPose:
        angle = s / radius
        return CarlaWorldPose(
            x=radius * math.sin(angle),
            y=_RADIUS - radius * math.cos(angle),
            z=0.0,
            yaw=math.degrees(angle),
        )

    return at


def _gap(ego: carla.Transform, pose: Lanelet2Pose, lane: Callable) -> float:
    at = lane(pose.s)
    gap = vehicle_ahead_gap_between(ego, carla.Location(x=at.x, y=at.y, z=0.0))
    assert gap is not None
    return gap


def test_on_straight_lanes_the_gap_is_the_offset_along_them(monkeypatch) -> None:
    ego_lane, npc_lane = _straight(0.0, 0.0), _straight(-20.0, -_WIDTH)
    _on({1: ego_lane, 2: npc_lane}, monkeypatch)
    ego = _transform(5.0, 0.0, 0.0)

    pose = place_at_vehicle_ahead_gap(ego, 2, 15.0)

    # The NPC lanelet starts 20 m behind the ego's: 5 + 20 + 15.
    assert pose.lanelet_id == 2
    assert pose.s == pytest.approx(40.0, abs=1e-3)
    assert _gap(ego, pose, npc_lane) == pytest.approx(15.0, abs=1e-3)


@pytest.mark.parametrize("npc_radius", [_RADIUS - _WIDTH, _RADIUS + _WIDTH])
@pytest.mark.parametrize("target", [6.0, 15.0, 30.0])
def test_on_a_curve_the_gap_is_what_the_measure_reads(
    monkeypatch, npc_radius: float, target: float
) -> None:
    ego_lane, npc_lane = _arc(_RADIUS), _arc(npc_radius)
    _on({1: ego_lane, 2: npc_lane}, monkeypatch, length=600.0)
    ego_s = 200.0
    ego_at = ego_lane(ego_s)
    ego = _transform(ego_at.x, ego_at.y, ego_at.yaw)

    pose = place_at_vehicle_ahead_gap(ego, 2, target)

    assert _gap(ego, pose, npc_lane) == pytest.approx(target, abs=1e-3)
    # The raw setting -- the NPC's s the ego's plus the target -- is off by
    # the lanes' difference in length up to there (200 m * 3.5 / 400 = 1.75 m),
    # one way on the inner lane and the other on the outer.
    raw = Lanelet2Pose(lanelet_id=2, s=ego_s + target)
    error = _gap(ego, raw, npc_lane) - target
    assert abs(error) > 1.0
    assert (error > 0) == (npc_radius < _RADIUS)


def test_a_gap_the_lanelet_cannot_reach_is_refused(monkeypatch) -> None:
    _on({1: _straight(0.0, 0.0), 2: _straight(0.0, -_WIDTH)}, monkeypatch)
    ego = _transform(60.0, 0.0, 0.0)

    with pytest.raises(ValueError, match=r"lanelet 2 \(length 80\.0 m\).* to 20\.00"):
        place_at_vehicle_ahead_gap(ego, 2, 30.0)


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
    def __init__(self, actor_id: int, role: str, transform: Any) -> None:
        self.id = actor_id
        self.type_id = "vehicle.test"
        self.attributes = {"role_name": role}
        self._transform = transform

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

    world = _World([_Actor(1, str(EGO_ROLE_NAME), ego_at), _Actor(2, "npc1", npc_at)])

    assert read_measure(VEHICLE_AHEAD_GAP_M, world) == pytest.approx(gap, abs=1e-2)


def test_cut_in_still_takes_the_raw_offset(
    nishishinjuku: MapManager, monkeypatch
) -> None:
    _ego_at, npc_at = _cut_in_npc_transform(monkeypatch, npc_ahead_m=10.0)

    expected = to_carla_world(Lanelet2Pose(lanelet_id=19, s=15.0))
    assert npc_at.location.x == pytest.approx(expected.x)
    assert npc_at.location.y == pytest.approx(expected.y)
