"""Matching a track to the lanes it runs along (traffic.route_match)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from autoware_carla_scenario.traffic import route_match
from autoware_carla_scenario.traffic.route_match import (
    TrackMatchError,
    _Candidate,
    _transition,
    match_track,
)

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


class _Lanelet:
    def __init__(self, ident: int) -> None:
        self.id = ident


# Typed Any: they stand in for lanelet2's ConstLanelet.
_A: Any = _Lanelet(1)
_C: Any = _Lanelet(2)
_B: Any = _Lanelet(3)


class _Detour:
    """A routing graph where *a* reaches *b* only through *c*."""

    def __init__(self) -> None:
        self.searches = 0

    def following(self, lanelet: _Lanelet) -> list:
        return {1: [_C], 2: [_B]}.get(lanelet.id, [])

    def left(self, lanelet: _Lanelet) -> None:
        return None

    def right(self, lanelet: _Lanelet) -> None:
        return None

    def shortestPath(self, a: _Lanelet, b: _Lanelet) -> list:  # noqa: N802
        self.searches += 1
        return [a, _C, b]


def test_a_cached_path_is_measured_from_each_transitions_own_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path cache match_track shares across a track holds what the
    lanelets decide, not a distance worked out for other samples (#669)."""
    lengths = {1: 100.0, 2: 10.0, 3: 100.0}
    monkeypatch.setattr(route_match, "_length", lambda ll, cache: lengths[int(ll.id)])
    graph: Any = _Detour()
    fresh_graph: Any = _Detour()
    # Two pairs of consecutive samples, each 2 m apart.
    samples = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (100.0, 0.0, 0.0), (102.0, 0.0, 0.0)]
    paths: dict = {}

    # a@0 -> b@0 drives 100 + 10 + 0 = 110 m for 2 m: a detour.
    far = _transition(
        _Candidate(_A, 0.0, 0.0), _Candidate(_B, 0.0, 0.0), graph, {}, paths, samples, 0
    )
    assert far == math.inf

    # a@99 -> b@1 drives 1 + 10 + 1 = 12 m: drivable, at the same cost
    # with the shared cache as with a fresh one.
    fresh = _transition(
        _Candidate(_A, 99.0, 0.0),
        _Candidate(_B, 1.0, 0.0),
        fresh_graph,
        {},
        {},
        samples,
        2,
    )
    near = _transition(
        _Candidate(_A, 99.0, 0.0),
        _Candidate(_B, 1.0, 0.0),
        graph,
        {},
        paths,
        samples,
        2,
    )
    assert fresh != math.inf
    assert near == pytest.approx(fresh)
    # ...and the lanelet path was still searched only once.
    assert graph.searches == 1
