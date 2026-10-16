"""The distance to the stop line ahead (``measures.STOP_LINE_DISTANCE_M``), and
the ego placed so it reads a value (``placement.place_at_stop_line_distance``)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from lanelet2.core import AttributeMap, LineString3d, Point3d, TrafficLight, getId

import typesafe_carla.carla as carla
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.coordinate.poses import Lanelet2Pose
from autoware_carla_scenario.coordinate.transform import to_carla_world
from autoware_carla_scenario.measures import (
    STOP_LINE_DISTANCE_M,
    Bumpers,
    read_measure,
    set_measured_scenario,
    stop_line_distance_between,
)
from autoware_carla_scenario.placement import place_at_stop_line_distance

from . import _lane_maps
from ._lane_maps import LaneMap, build_road, straight
from .test_placement import _Actor, _World

#: A Lincoln MKZ's bumpers, about.
_EGO = Bumpers(front=2.6, rear=2.2)
#: Where the stop line crosses the road, in metres along it.
_STOP_X = 150.0


def _with_stop_line(built: LaneMap, lanelet_id: int, x: float) -> None:
    """A traffic light's stop line across the lane at *x*, on *lanelet_id*."""
    stop_line = LineString3d(
        getId(),
        [Point3d(getId(), x, -1.75, 0.0), Point3d(getId(), x, 1.75, 0.0)],
        AttributeMap({"type": "stop_line"}),
    )
    light = LineString3d(
        getId(),
        [Point3d(getId(), x + 5, 1.0, 5.0), Point3d(getId(), x + 5, -1.0, 5.0)],
        AttributeMap({"type": "traffic_light"}),
    )
    built.map.laneletLayer[lanelet_id].addRegulatoryElement(
        TrafficLight(getId(), AttributeMap(), [light], stop_line)
    )


@pytest.fixture
def road() -> Iterator[LaneMap]:
    """One lane, 300 m, in lanelets of 100 m; a stop line 50 m into the second."""
    built = build_road(
        straight(300.0), [0.0, 100.0, 200.0, 300.0], [-1.75, 1.75], {"ego": (0, False)}
    )
    _with_stop_line(built, built.id("ego", 1), _STOP_X)
    yield from _lane_maps.loaded(built)


@pytest.fixture(autouse=True)
def _no_scenario() -> Iterator[None]:
    set_measured_scenario(None)
    yield
    set_measured_scenario(None)


def _at(built: LaneMap, segment: int, s: float) -> carla.Transform:
    return to_carla_world(
        Lanelet2Pose(lanelet_id=built.id("ego", segment), s=s)
    ).to_carla_transform()


def _measured(ego: carla.Transform) -> Any:
    return read_measure(
        STOP_LINE_DISTANCE_M, _World([_Actor(1, str(EGO_ROLE_NAME), ego, _EGO)])
    )


@pytest.mark.parametrize(("segment", "s"), [(0, 60.0), (0, 99.0), (1, 30.0)])
def test_it_is_from_the_front_bumper_along_the_lane(
    road: LaneMap, segment: int, s: float
) -> None:
    ego = _at(road, segment, s)
    expected = _STOP_X - (100.0 * segment + s + _EGO.front)

    assert stop_line_distance_between(ego, _EGO) == pytest.approx(expected, abs=1e-3)
    assert _measured(ego) == pytest.approx(expected, abs=1e-3)


def test_a_stop_line_passed_reads_nothing_past_it(road: LaneMap) -> None:
    assert _measured(_at(road, 1, 47.0)) == pytest.approx(0.4, abs=1e-3)
    assert _measured(_at(road, 1, 60.0)) is None


def test_the_next_stop_line_is_read_once_one_is_passed() -> None:
    built = build_road(
        straight(300.0), [0.0, 100.0, 200.0, 300.0], [-1.75, 1.75], {"ego": (0, False)}
    )
    _with_stop_line(built, built.id("ego", 1), _STOP_X)
    _with_stop_line(built, built.id("ego", 2), 250.0)
    for _ in _lane_maps.loaded(built):
        assert _measured(_at(built, 1, 40.0)) == pytest.approx(7.4, abs=1e-3)
        assert _measured(_at(built, 1, 61.0)) == pytest.approx(250.0 - 163.6, abs=1e-3)


def test_one_further_than_the_range_ahead_reads_nothing(road: LaneMap) -> None:
    assert _measured(_at(road, 0, 50.0)) == pytest.approx(97.4, abs=1e-3)
    assert _measured(_at(road, 0, 45.0)) is None


@pytest.mark.parametrize("distance", [1.0, 10.0, 30.0, 70.0])
def test_placement_reads_back_as_the_measure(road: LaneMap, distance: float) -> None:
    # Spawned 10 m before the stop line, on its lanelet.
    spawn = Lanelet2Pose(lanelet_id=road.id("ego", 1), s=37.4)

    pose = place_at_stop_line_distance(spawn, distance, ego_bumpers=_EGO)

    ego = to_carla_world(pose).to_carla_transform()
    assert _measured(ego) == pytest.approx(distance, abs=0.01)
    # Back onto the lanelet before where the stop line's is too short.
    expected = road.id("ego", 0 if distance > 47.4 else 1)
    assert pose.lanelet_id == expected


def test_a_spawn_on_a_lanelet_before_the_stop_line_s_is_placed_on_to_it(
    road: LaneMap,
) -> None:
    # 87.4 m before the stop line, on the lanelet before its lanelet.
    spawn = Lanelet2Pose(lanelet_id=road.id("ego", 0), s=60.0)

    pose = place_at_stop_line_distance(spawn, 10.0, ego_bumpers=_EGO)

    assert pose.lanelet_id == road.id("ego", 1)
    assert pose.s == pytest.approx(37.4, abs=0.01)
    assert _measured(to_carla_world(pose).to_carla_transform()) == pytest.approx(
        10.0, abs=0.01
    )


def test_placement_says_what_it_cannot_reach(road: LaneMap) -> None:
    spawn = Lanelet2Pose(lanelet_id=road.id("ego", 1), s=37.4)
    with pytest.raises(ValueError, match=f"{STOP_LINE_DISTANCE_M} cannot read 200"):
        place_at_stop_line_distance(spawn, 200.0, ego_bumpers=_EGO)
    past = Lanelet2Pose(lanelet_id=road.id("ego", 2), s=10.0)
    with pytest.raises(ValueError, match="no stop line"):
        place_at_stop_line_distance(past, 10.0, ego_bumpers=_EGO)
