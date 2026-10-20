"""Where an agent is, for the adversities' location gate (#388).

On the synthetic crossroads of :mod:`._route_maps`: every junction lanelet
carries a ``turn_direction`` (the W approach is signalised, the others are
not), and the one-way road far away is tagged per test as a highway or a
roundabout.  The gate reads the map by the ODD's probes
(:mod:`autoware_carla_scenario.odd.probes`), which the tests check too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

import pytest

from autoware_carla_scenario.adversity import (
    AdversityLocation,
    location_matches,
    location_of,
)
from autoware_carla_scenario.odd import probes
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

INTERSECTION = AdversityLocation.INTERSECTION
HIGHWAY = AdversityLocation.HIGHWAY
ROUNDABOUT = AdversityLocation.ROUNDABOUT
JUNCTION = rm.J[("W", "E")]


def _features(tags: Optional[dict[int, dict[str, str]]] = None) -> MapFeatures:
    """The crossroads, with *tags* (by lanelet id) set."""
    lanelet_map = rm.crossroads()
    for lanelet_id, attributes in (tags or {}).items():
        lanelet = lanelet_map.laneletLayer[lanelet_id]
        for key, value in attributes.items():
            lanelet.attributes[key] = value
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    return _features()


def _lanelet(features: MapFeatures, lanelet_id: int) -> ConstLanelet:
    # The map's layer hands out a Lanelet, which the stubs do not relate to
    # the ConstLanelet the API takes (as a matcher or the routing graph gives).
    return cast("ConstLanelet", features.lanelet(lanelet_id))


def _at(
    features: MapFeatures, lanelet_id: int, s: float, **kw: Any
) -> Optional[AdversityLocation]:
    return location_of(features, _lanelet(features, lanelet_id), s, **kw)


@pytest.mark.parametrize(
    ("lanelet_id", "s", "expected"),
    [
        # In a junction, signalised or not: the ODD's scenery.junction.
        (JUNCTION, 0.0, INTERSECTION),
        (rm.J[("S", "N")], 0.0, INTERSECTION),
        # W_IN2 (50 m) ends at the junction: within 25 m of it from s = 25.
        (rm.W_IN2, 0.0, None),
        (rm.W_IN2, 24.0, None),
        (rm.W_IN2, 26.0, INTERSECTION),
        # N_IN (100 m) has no signal; its junction is ahead all the same.
        (rm.N_IN, 0.0, None),
        (rm.N_IN, 80.0, INTERSECTION),
        (rm.W_IN1, 49.0, None),
        # Just past the junction: what lies behind does not count.
        (rm.E_OUT1, 0.0, None),
        (rm.E_OUT1, 40.0, None),
    ],
)
def test_intersection(
    features: MapFeatures,
    lanelet_id: int,
    s: float,
    expected: Optional[AdversityLocation],
) -> None:
    assert _at(features, lanelet_id, s) is expected


def test_the_intersection_is_the_odds_junction(features: MapFeatures) -> None:
    for lanelet_id in rm.J.values():
        lanelet = _lanelet(features, lanelet_id)
        assert probes.lanelet_in_junction(lanelet)
        assert location_of(features, lanelet, 0.0) is INTERSECTION
    road = _lanelet(features, rm.W_IN1)
    assert not probes.lanelet_in_junction(road)


def test_the_lookahead_is_a_parameter(features: MapFeatures) -> None:
    assert _at(features, rm.W_IN1, 0.0, lookahead=100.0) is INTERSECTION
    assert _at(features, rm.W_IN2, 26.0, lookahead=10.0) is None


def test_a_highway_is_the_odds_road_type() -> None:
    features = _features({JUNCTION: {"subtype": "highway"}})
    junction = _lanelet(features, JUNCTION)
    assert probes.lanelet_road_type(junction) == "highway"
    # Highway wins over the junction on the same lanelet, as in TeraSim.
    assert location_of(features, junction, 0.0) is HIGHWAY


def test_a_fast_road_is_no_highway() -> None:
    # Only the road type decides, as for the ODD: not the speed limit.
    features = _features({rm.ONE_WAY1: {"speed_limit": "120"}})
    assert _at(features, rm.ONE_WAY1, 0.0) is None


def test_a_roundabout_ahead() -> None:
    features = _features({rm.ONE_WAY2: {"roundabout": "yes"}})
    # ONE_WAY1 is 60 m long and leads into the roundabout lanelet.
    assert _at(features, rm.ONE_WAY1, 0.0) is None
    assert _at(features, rm.ONE_WAY1, 40.0) is ROUNDABOUT
    assert _at(features, rm.ONE_WAY2, 0.0) is ROUNDABOUT


@pytest.mark.parametrize("tag", ["no", "False", "0"])
def test_a_roundabout_tag_of_no_is_none(tag: str) -> None:
    features = _features({rm.ONE_WAY2: {"roundabout": tag}})
    assert _at(features, rm.ONE_WAY2, 0.0) is None


def test_matches(features: MapFeatures) -> None:
    junction = _lanelet(features, JUNCTION)
    assert location_matches(INTERSECTION, features, junction, 0.0)
    assert not location_matches(HIGHWAY, features, junction, 0.0)
    elsewhere = _lanelet(features, rm.ONE_WAY1)
    assert location_matches(AdversityLocation.ANY, features, elsewhere, 0.0)
    assert not location_matches(INTERSECTION, features, elsewhere, 0.0)
    assert not location_matches(ROUNDABOUT, features, elsewhere, 0.0)


def test_matches_follows_the_precedence() -> None:
    # In a junction and a highway: only HIGHWAY matches.
    features = _features(
        {JUNCTION: {"subtype": "highway"}, rm.ONE_WAY2: {"roundabout": "yes"}}
    )
    junction = _lanelet(features, JUNCTION)
    assert location_matches(HIGHWAY, features, junction, 0.0)
    assert not location_matches(INTERSECTION, features, junction, 0.0)
    road = _lanelet(features, rm.ONE_WAY1)
    assert location_matches(ROUNDABOUT, features, road, 40.0)
    assert not location_matches(HIGHWAY, features, road, 40.0)


# -- the lane chain -------------------------------------------------------------


def test_the_chain_ahead_follows_every_branch_by_its_start(
    features: MapFeatures,
) -> None:
    # W_IN2 (50 m) at s = 40: its three junction lanelets start 10 m ahead.
    chain = features.lane_chain(_lanelet(features, rm.W_IN2), 40.0, ahead=15.0)
    assert chain[rm.W_IN2].start == -40.0
    for to in ("E", "N", "S"):
        assert chain[rm.J[("W", to)]].start == pytest.approx(10.0)
    assert rm.W_IN1 not in chain
    assert rm.E_OUT1 not in chain  # starts 30 m ahead, past the 15 m


def test_the_chain_behind_ends_within_its_distance(features: MapFeatures) -> None:
    own = _lanelet(features, rm.W_IN2)
    chain = features.lane_chain(own, 10.0, ahead=0.0, behind=20.0)
    assert chain[rm.W_IN1].start == pytest.approx(-60.0)
    assert chain[rm.W_IN1].end == pytest.approx(-10.0)
    assert set(chain) == {rm.W_IN1, rm.W_IN2}
    # Nothing behind by default, even at the very start of a lanelet.
    assert set(features.lane_chain(own, 0.0, ahead=0.0)) == {rm.W_IN2}
