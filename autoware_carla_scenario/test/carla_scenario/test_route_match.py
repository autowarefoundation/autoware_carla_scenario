"""Matching a track to the lanes it runs along (traffic.route_match)."""

from __future__ import annotations

import math

import pytest

from autoware_carla_scenario.traffic.route_match import TrackMatchError, match_track

from ._route_maps import E_OUT1, J, W_IN1, W_IN2, crossroads, routing_graph


@pytest.fixture(scope="module")
def roads() -> tuple:
    lanelet_map = crossroads()
    return lanelet_map, routing_graph(lanelet_map)


def _track(y: float, x0: float, x1: float, heading: float) -> list:
    step = 2.0 if x1 > x0 else -2.0
    count = int(abs(x1 - x0) / 2.0) + 1
    return [(x0 + i * step, y, heading) for i in range(count)]


def test_a_straight_track_runs_through_the_junction_on_its_lanes(roads: tuple) -> None:
    stops = match_track(_track(-1.75, -100.0, 40.0, 0.0), *roads)
    assert [s.lanelet_id for s in stops] == [W_IN1, W_IN1, W_IN2, J[("W", "E")], E_OUT1]
    assert stops[-1].s == pytest.approx(30.0, abs=1.0)


def test_the_heading_picks_the_lane_running_its_way(roads: tuple) -> None:
    """A track between the two directions' lanes, heading east: the
    eastbound lanes, never the westbound one beside them."""
    stops = match_track(_track(0.3, -100.0, -20.0, 0.0), *roads)
    assert [s.lanelet_id for s in stops] == [W_IN1, W_IN1, W_IN2]


def test_a_point_off_the_road_is_left_out(roads: tuple) -> None:
    track = _track(-1.75, -100.0, -70.0, 0.0) + [(-60.0, 40.0, 0.0)]
    stops = match_track(track, *roads)
    assert [s.lanelet_id for s in stops] == [W_IN1, W_IN1]


def test_a_track_on_no_lane_is_refused(roads: tuple) -> None:
    with pytest.raises(TrackMatchError):
        match_track([(500.0, 500.0, 0.0), (502.0, 500.0, 0.0)], *roads)


def test_a_track_the_wrong_way_down_a_one_way_lane_is_refused(roads: tuple) -> None:
    with pytest.raises(TrackMatchError):
        match_track(_track(-5.25, -20.0, -100.0, math.pi), *roads)
