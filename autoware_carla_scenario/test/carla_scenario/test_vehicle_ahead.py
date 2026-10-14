"""The vehicle-ahead measures on the Lanelet2 map (measures.py, version 2):
which vehicles count, from the map; the gap bumper to bumper along the ego's
lane; the relative speed in the lane's frame; and the geometric fallback."""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.coordinate.poses import Lanelet2Pose
from autoware_carla_scenario.coordinate.transform import to_carla_world
from autoware_carla_scenario.measures import (
    ALONG_LANE,
    GEOMETRIC,
    VEHICLE_AHEAD_GAP_M,
    VEHICLE_AHEAD_RELATIVE_SPEED_KPH,
    VEHICLE_AHEAD_VERSION,
    BUILT_IN_MEASURES,
    Bumpers,
    measure_version,
    read_measure,
    set_measured_scenario,
    vehicle_ahead_definition,
    vehicle_ahead_definitions,
)

from . import _lane_maps
from ._lane_maps import LaneMap, bend, build_road, no_map, straight

__all__ = ["no_map"]

#: An ego the size of a Lincoln MKZ, its box a little forward of its origin.
EGO = Bumpers(front=2.6, rear=2.2)
#: A lead the size of a Mini.
LEAD = Bumpers(front=1.9, rear=1.8)


# ---------------------------------------------------------------------------
# A world of actors
# ---------------------------------------------------------------------------


class _Actor:
    def __init__(
        self,
        actor_id: int,
        x: float,
        y: float,
        yaw: float,
        bumpers: Bumpers,
        speed: float = 0.0,
        role: str = "",
        type_id: str = "vehicle.test",
    ) -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role}
        self._location = SimpleNamespace(x=x, y=y, z=0.0)
        self._rotation = SimpleNamespace(yaw=yaw, pitch=0.0, roll=0.0)
        heading = math.radians(yaw)
        self._velocity = SimpleNamespace(
            x=speed * math.cos(heading), y=speed * math.sin(heading), z=0.0
        )
        # A box whose centre is (front - rear) / 2 ahead of the origin.
        self.bounding_box = SimpleNamespace(
            location=SimpleNamespace(x=(bumpers.front - bumpers.rear) / 2, y=0.0),
            extent=SimpleNamespace(x=(bumpers.front + bumpers.rear) / 2, y=0.9),
        )

    def get_location(self) -> Any:
        return self._location

    def get_transform(self) -> Any:
        return SimpleNamespace(location=self._location, rotation=self._rotation)

    def get_velocity(self) -> Any:
        return self._velocity


_FRAMES = iter(range(1, 10_000_000))


class _World:
    def __init__(self, actors: list[_Actor]) -> None:
        self._actors = actors
        self._frame = next(_FRAMES)

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=self._frame)

    def get_actors(self) -> list[_Actor]:
        return self._actors


def _on(built: LaneMap, lane: str, segment: int, s: float, t: float = 0.0) -> Any:
    """The CARLA world pose ``s`` along a lanelet, ``t`` to its left."""
    return to_carla_world(Lanelet2Pose(built.id(lane, segment), s=s, t=t))


def _vehicle(
    actor_id: int, pose: Any, bumpers: Bumpers = LEAD, speed: float = 0.0, **kw: Any
) -> _Actor:
    return _Actor(actor_id, pose.x, pose.y, pose.yaw, bumpers, speed, **kw)


def _ego(pose: Any, speed: float = 0.0) -> _Actor:
    return _vehicle(1, pose, EGO, speed, role=str(EGO_ROLE_NAME))


def _gap(*actors: _Actor) -> Any:
    return read_measure(VEHICLE_AHEAD_GAP_M, _World(list(actors)))


@pytest.fixture(autouse=True)
def _no_scenario() -> Iterator[None]:
    set_measured_scenario(None)
    yield
    set_measured_scenario(None)


# ---------------------------------------------------------------------------
# A straight road: the ego's lane, one beside it, one two over, an oncoming one
# ---------------------------------------------------------------------------


@pytest.fixture
def straight_road() -> Iterator[LaneMap]:
    built = build_road(
        straight(300.0),
        [0.0, 100.0, 200.0, 300.0],
        [-5.25, -1.75, 1.75, 5.25, 8.75],
        {
            "oncoming": (0, True),
            "ego": (1, False),
            "left": (2, False),
            "far": (3, False),
        },
    )
    yield from _lane_maps.loaded(built)


def test_a_vehicle_on_the_lane_beside_counts_one_two_over_does_not(
    straight_road: LaneMap,
) -> None:
    ego = _ego(_on(straight_road, "ego", 0, 50.0))

    beside = _vehicle(2, _on(straight_road, "left", 0, 70.0))
    two_over = _vehicle(3, _on(straight_road, "far", 0, 60.0))
    oncoming = _vehicle(4, _on(straight_road, "oncoming", 0, 45.0))

    assert _gap(ego, two_over) is None
    assert _gap(ego, oncoming) is None
    # Rear bumper at 70 - 1.8, front bumper at 50 + 2.6.
    assert _gap(ego, beside, two_over, oncoming) == pytest.approx(15.6, abs=1e-6)
    assert vehicle_ahead_definitions() == {ALONG_LANE: 3}


def test_the_gap_runs_on_across_lanelets(straight_road: LaneMap) -> None:
    ego = _ego(_on(straight_road, "ego", 0, 90.0))
    lead = _vehicle(2, _on(straight_road, "ego", 1, 50.0))  # s = 150 of the lane
    beyond = _vehicle(3, _on(straight_road, "ego", 2, 30.0))  # 135.6 m: too far
    beside = _vehicle(4, _on(straight_road, "left", 1, 10.0))

    assert _gap(ego, lead) == pytest.approx(150.0 - 1.8 - 92.6, abs=1e-6)
    assert _gap(ego, beyond) is None
    assert _gap(ego, beside, lead) == pytest.approx(110.0 - 1.8 - 92.6, abs=1e-6)


class _FrameZeroWorld(_World):
    """A world at frame 0, as every freshly made world is."""

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=0)


def test_a_new_world_is_not_read_off_the_last_ones_cache(
    straight_road: LaneMap,
) -> None:
    """Worlds made one after another, each freed before the next, may share
    an id(); each is read for itself, not off the cache of the one before."""
    ego = _ego(_on(straight_road, "ego", 0, 50.0))
    scenes = {
        ahead: [ego, _vehicle(2, _on(straight_road, "ego", 0, 50.0 + ahead))]
        for ahead in range(10, 30)
    }
    gaps = {
        # Each world is freed as soon as it is read, so the next one made
        # takes its place in memory, and its id().
        ahead: read_measure(VEHICLE_AHEAD_GAP_M, _FrameZeroWorld(actors))
        for ahead, actors in scenes.items()
    }
    assert gaps == {
        ahead: pytest.approx(ahead - 2.6 - 1.8, abs=1e-6) for ahead in scenes
    }


def test_a_vehicle_alongside_or_behind(straight_road: LaneMap) -> None:
    ego = _ego(_on(straight_road, "ego", 0, 50.0))

    # Its origin 1 m ahead of the ego's: ahead, its bumpers overlapping.
    alongside = _vehicle(2, _on(straight_road, "left", 0, 51.0))
    behind = _vehicle(3, _on(straight_road, "ego", 0, 40.0))

    assert _gap(ego, alongside) == pytest.approx(51.0 - 1.8 - 52.6, abs=1e-6)
    assert _gap(ego, behind) is None


# ---------------------------------------------------------------------------
# A curve: three lanes, the outer one longer (Town04's highway)
# ---------------------------------------------------------------------------

_RADIUS = 80.0


@pytest.fixture
def curve() -> Iterator[LaneMap]:
    built = build_road(
        bend(40.0, _RADIUS, 120.0, 60.0),
        [0.0, 40.0, 40.0 + _RADIUS * math.radians(120.0), 40.0 + _RADIUS * 2.0944 + 60],
        [-5.25, -1.75, 1.75, 5.25],
        {"outer": (0, False), "ego": (1, False), "inner": (2, False)},
    )
    yield from _lane_maps.loaded(built)


def _arc_length(lanelet_map: LaneMap, lane: str, segment: int) -> float:
    from autoware_carla_scenario.coordinate.transform import lanelet_length

    return lanelet_length(lanelet_map.id(lane, segment))


def test_the_lanes_beside_each_other_differ_in_length(curve: LaneMap) -> None:
    outer, ego, inner = (
        _arc_length(curve, lane, 1) for lane in ("outer", "ego", "inner")
    )

    assert outer == pytest.approx((_RADIUS + 3.5) * math.radians(120.0), rel=1e-4)
    assert outer - ego == pytest.approx(3.5 * math.radians(120.0), rel=1e-3)
    assert ego - inner == pytest.approx(outer - ego, rel=1e-3)


@pytest.mark.parametrize("lane", ["outer", "inner"])
def test_on_a_curve_the_gap_is_along_the_ego_lane(curve: LaneMap, lane: str) -> None:
    # The ego 10 degrees into the bend, the lead 40 degrees: 30 degrees of the
    # ego's lane between their origins, whichever lane the lead is on.
    ego_s = _RADIUS * math.radians(10.0)
    ego = _ego(_on(curve, "ego", 1, ego_s))
    lead_radius = _RADIUS + (3.5 if lane == "outer" else -3.5)
    lead = _vehicle(2, _on(curve, lane, 1, lead_radius * math.radians(40.0)))

    # Each bumper projected onto the ego's lane: radially, so the lead's rear
    # overhang shrinks or stretches by the ratio of the radii.
    rear_at = math.radians(40.0) - math.atan(LEAD.rear / lead_radius)
    front_at = math.radians(10.0) + math.atan(EGO.front / _RADIUS)
    assert _gap(ego, lead) == pytest.approx(_RADIUS * (rear_at - front_at), abs=0.01)


def test_on_a_curve_a_vehicle_far_round_the_bend_is_on_the_lane(curve: LaneMap) -> None:
    """60 degrees round an 80 m bend, the lead is 40 m to the side of the
    ego's heading: no lane band would count it, the map does."""
    ego = _ego(_on(curve, "ego", 1, 0.0))
    lead = _vehicle(2, _on(curve, "ego", 1, _RADIUS * math.radians(60.0)))

    rear_at = math.radians(60.0) - math.atan(LEAD.rear / _RADIUS)
    front_at = math.atan(EGO.front / _RADIUS)
    assert _gap(ego, lead) == pytest.approx(_RADIUS * (rear_at - front_at), abs=0.01)


def test_the_relative_speed_is_along_the_lane(curve: LaneMap) -> None:
    """Both at 20 m/s along their lanes: on the ego's lane they keep pace,
    though their headings are 30 degrees apart."""
    ego = _ego(_on(curve, "ego", 1, _RADIUS * math.radians(10.0)), speed=20.0)
    lead = _vehicle(
        2, _on(curve, "outer", 1, (_RADIUS + 3.5) * math.radians(40.0)), speed=20.0
    )
    world = _World([ego, lead])

    assert read_measure(VEHICLE_AHEAD_RELATIVE_SPEED_KPH, world) == pytest.approx(
        0.0, abs=0.3
    )


# ---------------------------------------------------------------------------
# Lanes of different widths
# ---------------------------------------------------------------------------


@pytest.fixture
def widths() -> Iterator[LaneMap]:
    # Right to left: a 7 m lane, the ego's 2.5 m lane, then two more 2.5 m lanes.
    built = build_road(
        straight(200.0),
        [0.0, 200.0],
        [-8.25, -1.25, 1.25, 3.75, 6.25],
        {
            "wide": (0, False),
            "ego": (1, False),
            "narrow": (2, False),
            "far": (3, False),
        },
    )
    yield from _lane_maps.loaded(built)


def test_membership_follows_the_lanes_widths(widths: LaneMap) -> None:
    ego = _ego(_on(widths, "ego", 0, 50.0))
    # Two narrow lanes over, 5 m from the ego's centre line: within three
    # half-lanes of 1.75 m, but not a lane beside the ego's.
    two_over = _vehicle(2, _on(widths, "far", 0, 60.0))
    # In the wide lane beside, 6.5 m to the ego's right.
    wide = _vehicle(3, _on(widths, "wide", 0, 80.0, t=-2.0))

    assert _gap(ego, two_over) is None
    assert _gap(ego, wide) == pytest.approx(80.0 - 1.8 - 52.6, abs=1e-6)


# ---------------------------------------------------------------------------
# Without a lane: the geometric definition
# ---------------------------------------------------------------------------


def test_without_a_map_the_gap_is_bumper_to_bumper_along_the_heading(
    no_map: None,
) -> None:
    ego = _Actor(1, 0.0, 0.0, 0.0, EGO, role=str(EGO_ROLE_NAME))
    beside = _Actor(2, 20.0, 3.5, 0.0, LEAD)
    two_over = _Actor(3, 10.0, 7.5, 0.0, LEAD)

    assert _gap(ego, beside, two_over) == pytest.approx(20.0 - 1.8 - 2.6)
    assert vehicle_ahead_definition(ego.get_transform()) == GEOMETRIC
    assert vehicle_ahead_definitions() == {GEOMETRIC: 1}


def test_off_the_map_the_gap_is_geometric(straight_road: LaneMap) -> None:
    ego = _Actor(1, 50.0, -500.0, 0.0, EGO, role=str(EGO_ROLE_NAME))
    lead = _Actor(2, 70.0, -500.0, 0.0, LEAD)

    assert vehicle_ahead_definition(ego.get_transform()) == GEOMETRIC
    assert _gap(ego, lead) == pytest.approx(20.0 - 1.8 - 2.6)
    assert vehicle_ahead_definitions() == {GEOMETRIC: 1}


def test_an_ego_across_its_lane_is_off_it(straight_road: LaneMap) -> None:
    """Turned 90 degrees, the ego is on no lanelet that runs its way."""
    at = _on(straight_road, "ego", 0, 50.0)

    ego = _Actor(1, at.x, at.y, at.yaw + 90.0, EGO, role=str(EGO_ROLE_NAME))

    assert vehicle_ahead_definition(ego.get_transform()) == GEOMETRIC
    on_lane = _Actor(1, at.x, at.y, at.yaw + 20.0, EGO, role=str(EGO_ROLE_NAME))
    assert vehicle_ahead_definition(on_lane.get_transform()) == ALONG_LANE


# ---------------------------------------------------------------------------
# The definition's version
# ---------------------------------------------------------------------------


def test_the_vehicle_ahead_measures_are_version_2() -> None:
    assert VEHICLE_AHEAD_VERSION == 2
    assert measure_version(VEHICLE_AHEAD_GAP_M) == 2
    assert measure_version(VEHICLE_AHEAD_RELATIVE_SPEED_KPH) == 2
    assert BUILT_IN_MEASURES[VEHICLE_AHEAD_GAP_M].version == 2
    assert measure_version("crossing_pedestrian_gap_m") == 1
    assert measure_version("a_scenario_s_own") == 1


def test_coverage_keeps_the_versions_apart(straight_road: LaneMap) -> None:
    """A coverage file names the vehicle-ahead items for their version and
    says what the measures meant; merged with a file of version 1, the two
    are reported apart rather than added up."""
    from autoware_carla_scenario.coverage import merge_coverage
    from autoware_carla_scenario.coverage.collector import CoverageCollector
    from autoware_carla_scenario.odd import default_odd

    ego = _ego(_on(straight_road, "ego", 0, 50.0))
    lead = _vehicle(2, _on(straight_road, "left", 0, 70.0))
    collector = CoverageCollector([], odd=default_odd())
    world = _World([ego, lead])
    collector.start(world, 0.0)
    collector.tick(world, 0.05)
    collector.end(world, 0.05)
    set_measured_scenario(None)  # as the runner does, before it writes the file

    new = collector.to_dict("cut_in")
    names = {item["name"] for item in new["items"]}
    assert "odd.dynamic.vehicle_ahead_gap#2" in names
    assert "odd.dynamic.vehicle_ahead_gap" not in names
    assert "odd.dynamic.crossing_pedestrian_gap" in names  # still version 1
    gap = new["measures"][VEHICLE_AHEAD_GAP_M]
    assert gap["version"] == 2
    assert gap["unit"] == "m"
    assert gap["definitions"] == {ALONG_LANE: 1}
    assert new["measures"]["crossing_pedestrian_gap_m"]["version"] == 1

    old = {
        **new,
        "items": [
            {**item, "name": item["name"].removesuffix("#2")} for item in new["items"]
        ],
    }
    merged = {entry.name for entry in merge_coverage([old, new]).entries}
    assert {
        "odd.dynamic.vehicle_ahead_gap",
        "odd.dynamic.vehicle_ahead_gap#2",
    } <= merged
