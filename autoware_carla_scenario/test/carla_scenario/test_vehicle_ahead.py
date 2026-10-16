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
    PROBE_ROLE,
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
from ._fake_world import ActorsById, FakeActorList
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
        self.is_alive = True
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


class _World(ActorsById):
    def __init__(self, actors: list[_Actor]) -> None:
        self._actors = actors
        self._frame = next(_FRAMES)

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=self._frame)

    def get_actors(self) -> FakeActorList:
        return FakeActorList(self._actors)


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

    # Its origin 1 m ahead of the ego's, its bumpers overlapping: in the
    # lane beside, it is alongside, not ahead; on the ego's own lane it is
    # ahead, and overlapping.
    alongside = _vehicle(2, _on(straight_road, "left", 0, 51.0))
    overlapping = _vehicle(4, _on(straight_road, "ego", 0, 51.0))
    behind = _vehicle(3, _on(straight_road, "ego", 0, 40.0))

    assert _gap(ego, alongside) is None
    assert _gap(ego, overlapping) == pytest.approx(51.0 - 1.8 - 52.6, abs=1e-6)
    assert _gap(ego, behind) is None


def test_a_vehicle_alongside_does_not_hide_the_one_ahead(
    straight_road: LaneMap,
) -> None:
    ego = _ego(_on(straight_road, "ego", 0, 50.0))
    alongside = _vehicle(2, _on(straight_road, "left", 0, 51.0))
    lead = _vehicle(3, _on(straight_road, "ego", 0, 72.0))
    # About to cut in: in the lane beside, ahead -- it still counts.
    cutting_in = _vehicle(4, _on(straight_road, "left", 0, 62.0))

    assert _gap(ego, alongside, lead) == pytest.approx(72.0 - 1.8 - 52.6, abs=1e-6)
    assert _gap(ego, alongside, lead, cutting_in) == pytest.approx(
        62.0 - 1.8 - 52.6, abs=1e-6
    )


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
    assert "odd.dynamic.vehicle_ahead_gap@v2" in names
    assert "odd.dynamic.vehicle_ahead_gap" not in names
    assert "odd.dynamic.crossing_pedestrian_gap" in names  # still version 1
    gap = new["measures"][VEHICLE_AHEAD_GAP_M]
    assert gap["version"] == 2
    assert gap["unit"] == "m"
    assert gap["definitions"] == {ALONG_LANE: 1}
    assert new["measures"]["crossing_pedestrian_gap_m"]["version"] == 1

    # A file of version 1, and one of version 1 with the item defined
    # differently (openodd names that one #2 when merged).
    old = {**new, "items": [_unversioned(item) for item in new["items"]]}
    old_variant = {
        **new,
        "items": [
            {**_unversioned(item), "min_stay": 2.0}
            if item["name"] == "odd.dynamic.vehicle_ahead_gap@v2"
            else _unversioned(item)
            for item in new["items"]
        ],
    }
    merged = [
        entry.name
        for entry in merge_coverage([old, old_variant, new]).entries
        if entry.name.startswith("odd.dynamic.vehicle_ahead_gap")
    ]
    assert sorted(merged) == [
        "odd.dynamic.vehicle_ahead_gap",
        "odd.dynamic.vehicle_ahead_gap#2",
        "odd.dynamic.vehicle_ahead_gap@v2",
    ]


def _unversioned(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "name": item["name"].removesuffix("@v2")}


def test_a_sampler_reads_only_the_current_version() -> None:
    """Version 1's hits are neither aimed past nor added to version 2's: the
    sampler reads the @v2 item, under the attribute's own name, alone."""
    from autoware_carla_scenario.coverage.collector import current_measure_coverage
    from autoware_carla_scenario.odd import default_odd

    def entry(name: str, hits: float) -> Any:
        return SimpleNamespace(
            name=name,
            buckets=["near", "far"],
            target=1.0,
            cover_by="hits",
            amount=lambda bucket: hits,
        )

    gap = "odd.dynamic.vehicle_ahead_gap"
    other = "odd.environment.rain"
    coverage = SimpleNamespace(
        entries=[
            entry(gap, 9.0),
            entry(f"{gap}#2", 8.0),
            entry(f"{gap}@v2", 1.0),
            entry(f"{gap}@v2#2", 2.0),
            entry(f"{gap}@v3", 7.0),
            entry(other, 5.0),
        ]
    )

    read = current_measure_coverage(coverage, default_odd())

    assert [(e.name, e.amount("near")) for e in read.entries] == [
        (gap, 1.0),
        (f"{gap}#2", 2.0),
        (other, 5.0),
    ]
    assert current_measure_coverage(None, default_odd()) is None


# ---------------------------------------------------------------------------
# Where the lane branches and where lanes merge
# ---------------------------------------------------------------------------


def _bounds(
    xs: list[float], centre: Any, half: float = 1.75
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Left and right bound points (Lanelet2 frame) of a lane along +x whose
    centre is at ``y = centre(x)``."""
    return [(x, centre(x) + half) for x in xs], [(x, centre(x) - half) for x in xs]


def _build(
    lanes: dict[str, tuple[list[tuple[float, float]], list[tuple[float, float]]]],
) -> LaneMap:
    """Lanelets from their bounds; a point shared by two bounds (by its
    coordinates) is one point, so lanelets that meet connect."""
    import lanelet2.routing
    import lanelet2.traffic_rules
    from lanelet2.core import AttributeMap, Lanelet, LaneletMap, LineString3d, Point3d

    ids = iter(range(500_000, 10_000_000))
    points: dict[tuple[float, float], Any] = {}

    def line(coords: list[tuple[float, float]]) -> Any:
        pts = []
        for x, y in coords:
            key = (round(x, 6), round(y, 6))
            if key not in points:
                points[key] = Point3d(next(ids), x, y, 0.0)
            pts.append(points[key])
        return LineString3d(
            next(ids), pts, AttributeMap({"type": "line_thin", "subtype": "dashed"})
        )

    lanelet_map = LaneletMap()
    built = LaneMap(map=lanelet_map, graph=None)
    for name, (left, right) in lanes.items():
        lanelet_id = next(ids)
        lanelet_map.add(
            Lanelet(lanelet_id, line(left), line(right), AttributeMap(_lane_maps._ROAD))
        )
        built.ids[(name, 0)] = lanelet_id
    rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )
    built.graph = lanelet2.routing.RoutingGraph(lanelet_map, rules)
    return built


def _xs(first: float, last: float) -> list[float]:
    return [first + i for i in range(int(last - first) + 1)]


@pytest.fixture
def fork() -> Iterator[LaneMap]:
    """A lanelet that branches: straight on, or bending away to the left."""
    built = _build(
        {
            "before": _bounds(_xs(0.0, 50.0), lambda x: 0.0),
            "straight": _bounds(_xs(50.0, 150.0), lambda x: 0.0),
            "branch": _bounds(_xs(50.0, 150.0), lambda x: (x - 50.0) ** 2 / 400.0),
        }
    )
    yield from _lane_maps.loaded(built)


def test_every_branch_ahead_is_the_lane(fork: LaneMap) -> None:
    """The measure does not know the route: a vehicle on either branch is on
    the ego's lane, measured along the branch it is on."""
    graph = fork.graph
    following = {
        int(ll.id) for ll in graph.following(fork.map.laneletLayer[fork.id("before")])
    }
    assert following == {fork.id("straight"), fork.id("branch")}
    ego = _ego(_on(fork, "before", 0, 20.0))
    on_branch = _vehicle(2, _on(fork, "branch", 0, 30.0))
    straight_on = _vehicle(3, _on(fork, "straight", 0, 40.0))

    assert _gap(ego, on_branch) == pytest.approx(50.0 + 30.0 - 1.8 - 22.6, abs=0.05)
    assert _gap(ego, straight_on) == pytest.approx(50.0 + 40.0 - 1.8 - 22.6, abs=0.05)
    assert _gap(ego, on_branch, straight_on) == pytest.approx(55.6, abs=0.05)


def test_a_moment_of_heading_drift_at_a_fork_does_not_lock_onto_the_branch(
    fork: LaneMap,
) -> None:
    """Where the branches overlap, one tick whose heading favours the other
    branch picks it; the next tick, heading straight on again, is back on
    the straight branch -- not held on the other until the ego leaves it."""
    lead = _vehicle(2, _on(fork, "straight", 0, 45.0))
    gaps = []
    for s in range(40, 80):
        pose = (
            _on(fork, "before", 0, float(s))
            if s < 50
            else _on(fork, "straight", 0, float(s - 50))
        )
        if s == 51:  # the first tick past the fork, a degree towards the branch
            pose = SimpleNamespace(x=pose.x, y=pose.y, yaw=pose.yaw - 1.0)
        gaps.append(_gap(_ego(pose), lead))

    fresh = [95.0 - 1.8 - (s + 2.6) for s in range(40, 80)]
    assert gaps[:11] == pytest.approx(fresh[:11], abs=0.05)
    # That tick read along the branch: the lead is not on it.
    assert gaps[11] is None
    assert gaps[12:] == pytest.approx(fresh[12:], abs=0.05)


@pytest.fixture
def long_fork() -> Iterator[LaneMap]:
    """A fork whose branches are cut in two, as a map splits an exit: the
    second pieces overlap where they start, but are not siblings in the
    routing graph (their predecessors differ)."""

    def bend(x: float) -> float:
        return (x - 50.0) ** 2 / 400.0

    built = _build(
        {
            "before": _bounds(_xs(0.0, 50.0), lambda x: 0.0),
            "straight": _bounds(_xs(50.0, 60.0), lambda x: 0.0),
            "straight_b": _bounds(_xs(60.0, 150.0), lambda x: 0.0),
            "branch": _bounds(_xs(50.0, 60.0), bend),
            "branch_b": _bounds(_xs(60.0, 150.0), bend),
        }
    )
    yield from _lane_maps.loaded(built)


def test_drift_onto_an_overlapping_lanelet_that_is_no_sibling_does_not_hold(
    long_fork: LaneMap,
) -> None:
    """The second piece of the other branch overlaps the ego's, though the
    routing graph does not make them siblings: a few degrees' drift on the
    tick the lane is chosen again picks it, and the next tick is back."""
    lead = _vehicle(2, _on(long_fork, "straight_b", 0, 40.0))
    gaps = []
    for s in range(52, 90):
        if s < 60:
            pose = _on(long_fork, "straight", 0, float(s - 50))
        else:
            pose = _on(long_fork, "straight_b", 0, float(s - 60))
        if s == 61:  # the first tick on straight_b, 4 degrees towards the branch
            pose = SimpleNamespace(x=pose.x, y=pose.y, yaw=pose.yaw - 4.0)
        gaps.append(_gap(_ego(pose), lead))

    fresh = [100.0 - 1.8 - (s + 2.6) for s in range(52, 90)]
    assert gaps[:9] == pytest.approx(fresh[:9], abs=0.05)
    assert gaps[9] is None  # that tick read along the branch
    assert gaps[10:] == pytest.approx(fresh[10:], abs=0.05)


@pytest.fixture
def merge() -> Iterator[LaneMap]:
    """Two lanelets that merge into one: the ego's, straight, and one coming
    in from the left."""
    built = _build(
        {
            "ego": _bounds(_xs(0.0, 50.0), lambda x: 0.0),
            "joining": _bounds(_xs(0.0, 50.0), lambda x: (50.0 - x) ** 2 / 200.0),
            "after": _bounds(_xs(50.0, 150.0), lambda x: 0.0),
        }
    )
    yield from _lane_maps.loaded(built)


def test_past_a_merge_the_lane_runs_on_and_the_lane_joining_is_not_it(
    merge: LaneMap,
) -> None:
    ego = _ego(_on(merge, "ego", 0, 20.0))
    lead = _vehicle(2, _on(merge, "after", 0, 10.0))
    # Ahead of the ego, on the lanelet joining: not the ego's lane, nor
    # beside it (they share no boundary), so not ahead of it.
    joining = _vehicle(3, _on(merge, "joining", 0, 10.0))

    assert _gap(ego, lead) == pytest.approx(50.0 + 10.0 - 1.8 - 22.6, abs=0.05)
    assert _gap(ego, joining) is None
    assert _gap(ego, joining, lead) == pytest.approx(35.6, abs=0.05)


def test_just_past_a_merge_the_lane_runs_back_one_way(merge: LaneMap) -> None:
    """Behind the ego's lanelet the lane runs back through the first of the
    lanelets before it (previous[0]); a vehicle ahead is measured as ever."""
    ego = _ego(_on(merge, "after", 0, 3.0))
    lead = _vehicle(2, _on(merge, "after", 0, 30.0))
    behind = _vehicle(3, _on(merge, "ego", 0, 45.0))

    assert _gap(ego, lead) == pytest.approx(30.0 - 1.8 - 5.6, abs=1e-3)
    assert _gap(ego, behind) is None


# ---------------------------------------------------------------------------
# What a reading costs
# ---------------------------------------------------------------------------


def test_the_lane_is_kept_while_the_ego_stays_on_its_lanelet(
    straight_road: LaneMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole map is searched for the ego's lanelet once, and again only
    when it leaves it; a lane is built for a lanelet and front bumper once."""
    from autoware_carla_scenario import measures

    searches: list[int] = []
    find_nearest = measures.lanelet2.geometry.findNearest

    def counted(*args: Any) -> Any:
        searches.append(1)
        return find_nearest(*args)

    monkeypatch.setattr(measures.lanelet2.geometry, "findNearest", counted)
    built = measures._FRAMES_BUILT["count"]
    lead = _vehicle(2, _on(straight_road, "ego", 1, 5.0))
    far = _vehicle(3, _on(straight_road, "ego", 1, 90.0))
    for s in range(10, 90, 2):
        assert _gap(_ego(_on(straight_road, "ego", 0, float(s))), lead) is not None
    assert len(searches) == 1
    assert measures._FRAMES_BUILT["count"] == built + 1

    # Onto the next lanelet: searched and built again, once.
    for s in range(5, 40, 2):
        _gap(_ego(_on(straight_road, "ego", 1, float(s))), far)
    assert len(searches) == 2
    assert measures._FRAMES_BUILT["count"] == built + 2

    # Another ego's front bumper is another lane frame.
    other = _vehicle(
        1,
        _on(straight_road, "ego", 1, 50.0),
        Bumpers(front=3.0, rear=2.0),
        role=str(EGO_ROLE_NAME),
    )
    assert _gap(other, far) == pytest.approx(190.0 - 1.8 - 153.0, abs=1e-6)
    assert measures._FRAMES_BUILT["count"] == built + 3


def test_off_the_lanelets_the_map_is_searched_again_only_once_the_ego_moves(
    straight_road: LaneMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario import measures

    searches: list[int] = []
    find_nearest = measures.lanelet2.geometry.findNearest

    def counted(*args: Any) -> Any:
        searches.append(1)
        return find_nearest(*args)

    monkeypatch.setattr(measures.lanelet2.geometry, "findNearest", counted)
    lead = _vehicle(2, _on(straight_road, "ego", 0, 80.0))
    for s in (50.0, 50.5, 51.0, 51.5):  # 30 m off the road, creeping on
        _gap(_ego(_on(straight_road, "ego", 0, s, t=-30.0)), lead)
    assert len(searches) == 1

    _gap(_ego(_on(straight_road, "ego", 0, 53.0, t=-30.0)), lead)  # 3 m on
    assert len(searches) == 2

    # Back on the road: found, and read along the lane.
    assert _gap(_ego(_on(straight_road, "ego", 0, 53.0)), lead) == pytest.approx(
        80.0 - 1.8 - 55.6, abs=1e-6
    )
    assert len(searches) == 3


def test_many_overlapping_lanelets_are_all_asked(
    straight_road: LaneMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At a junction more lanelets than first asked may hold a point: asked
    again with twice as many while every one asked holds it."""
    from autoware_carla_scenario import measures

    asked: list[int] = []

    def nearest(layer: Any, point: Any, n: int) -> list[Any]:
        asked.append(n)
        return [(0.0, SimpleNamespace(id=i)) for i in range(min(n, 20))]

    monkeypatch.setattr(measures.lanelet2.geometry, "findNearest", nearest)

    assert measures._lanelets_at(0.0, 0.0) == list(range(20))
    assert asked == [16, 32]


def test_a_reading_takes_milliseconds_on_a_long_road() -> None:
    """Ten vehicles round the ego on a two-kilometre, four-lane road of
    lanelets 500 m long with a point every 50 cm, as a real map has: a
    reading per tick, the ego moving, takes a few milliseconds (it took
    tens to hundreds when every vehicle searched the whole map)."""
    import time

    built = build_road(
        straight(2000.0),
        [0.0, 500.0, 1000.0, 1500.0, 2000.0],
        [-7.0, -3.5, 0.0, 3.5, 7.0],
        {"a": (0, False), "b": (1, False), "ego": (2, False), "d": (3, False)},
        step=0.5,
    )
    loaded = _lane_maps.loaded(built)
    next(loaded)
    try:
        lanes = ["a", "b", "ego", "d"]
        others = [
            _vehicle(k + 2, _on(built, lanes[k % 4], 0, 345.0 + 8.0 * k))
            for k in range(10)
        ]
        _gap(_ego(_on(built, "ego", 0, 300.0)), *others)  # the lines, read once
        times = []
        for tick in range(60):
            ego = _ego(_on(built, "ego", 0, 300.0 + 0.7 * tick))
            world = _World([ego, *others])
            start = time.perf_counter()
            read_measure(VEHICLE_AHEAD_GAP_M, world)
            times.append(time.perf_counter() - start)
    finally:
        next(loaded, None)
    mean_ms = 1000.0 * sum(times) / len(times)
    # Under a millisecond here (some 45 ms before); the bound leaves a slow
    # CI runner room.
    assert mean_ms < 15.0, f"{mean_ms:.1f} ms a reading"


# ---------------------------------------------------------------------------
# Before the run, and when a reading fails
# ---------------------------------------------------------------------------


def test_the_routing_graph_is_built_before_the_run(straight_road: LaneMap) -> None:
    from autoware_carla_scenario.coordinate.map_manager import MapManager
    from autoware_carla_scenario.coverage.collector import prepare_measures
    from autoware_carla_scenario.odd import OddDefinition, default_odd

    manager = MapManager.get_instance()
    manager._routing_graph = None
    prepare_measures(OddDefinition("nothing_measured", []))
    assert manager._routing_graph is None

    prepare_measures(default_odd())  # maps vehicle_ahead_gap_m
    assert manager._routing_graph is not None
    # And the lanelets' boxes the lane's rivals are found among.
    from autoware_carla_scenario import measures

    assert {box[0] for box in measures._LINES.boxes} == {
        int(lanelet.id) for lanelet in straight_road.map.laneletLayer
    }


def test_no_routing_graph_for_measures_the_scenario_replaced(
    straight_road: LaneMap,
) -> None:
    """A scenario that replaced the vehicle-ahead measures reads its own: the
    routing graph is not built for them before the run."""
    from autoware_carla_scenario.coordinate.map_manager import MapManager
    from autoware_carla_scenario.coverage.collector import prepare_measures
    from autoware_carla_scenario.measures import Measure
    from autoware_carla_scenario.odd import default_odd

    def replaced(*keys: str) -> SimpleNamespace:
        own = dict(BUILT_IN_MEASURES)
        for key in keys:
            own[key] = Measure(key, lambda world: None, own[key].unit)
        return SimpleNamespace(measures=own)

    manager = MapManager.get_instance()
    manager._routing_graph = None
    prepare_measures(
        default_odd(), replaced(VEHICLE_AHEAD_GAP_M, VEHICLE_AHEAD_RELATIVE_SPEED_KPH)
    )
    assert manager._routing_graph is None

    # One of them still the built-in one: that one reads the graph.
    prepare_measures(default_odd(), replaced(VEHICLE_AHEAD_GAP_M))
    assert manager._routing_graph is not None


def test_without_a_map_there_is_nothing_to_prepare(no_map: None) -> None:
    from autoware_carla_scenario.measures import prepare_vehicle_ahead

    prepare_vehicle_ahead()


def test_a_failed_reading_is_warned_of_once(
    straight_road: LaneMap, caplog: pytest.LogCaptureFixture
) -> None:
    class _Broken(_World):
        def get_actors(self) -> FakeActorList:
            raise RuntimeError("the server went away")

    with caplog.at_level("WARNING", logger="autoware_carla_scenario.measures"):
        assert read_measure(VEHICLE_AHEAD_GAP_M, _Broken([])) is None
        assert read_measure(VEHICLE_AHEAD_GAP_M, _Broken([])) is None
    warned = [r for r in caplog.records if "could not be read" in r.getMessage()]
    assert len(warned) == 1


def test_a_vehicle_far_above_the_ego_is_not_on_its_road(
    straight_road: LaneMap,
) -> None:
    """vehicle_bumpers() spawns its probe hundreds of metres up, over the
    map's origin: the measure does not count it."""
    ego = _ego(_on(straight_road, "ego", 0, 50.0))
    probe = _vehicle(2, _on(straight_road, "ego", 0, 60.0))
    probe._location.z = 500.0
    # Nor one by its role name, wherever it is.
    by_role = _vehicle(3, _on(straight_road, "ego", 0, 60.0), role=PROBE_ROLE)

    assert _gap(ego, probe) is None
    assert _gap(ego, by_role) is None
