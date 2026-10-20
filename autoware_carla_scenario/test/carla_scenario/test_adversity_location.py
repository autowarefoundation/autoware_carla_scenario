"""Where an agent is, for the adversities' location gate (#388).

On the synthetic crossroads of :mod:`._route_maps`: its W approach carries a
traffic light, every junction lanelet a ``turn_direction``, and the one-way
road far away is tagged per test as a highway or a roundabout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

import pytest

from autoware_carla_scenario.adversity import (
    AdversityLocation,
    location_matches,
    location_of,
)
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

INTERSECTION = AdversityLocation.INTERSECTION


def _features(**tags: dict[str, str]) -> MapFeatures:
    """The crossroads, with *tags* (by lanelet name in :mod:`._route_maps`) set."""
    lanelet_map = rm.crossroads()
    for name, attributes in tags.items():
        lanelet = lanelet_map.laneletLayer[getattr(rm, name)]
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
        # A lane the signal regulates is at the intersection all along.
        (rm.W_IN2, 0.0, INTERSECTION),
        (rm.J[("W", "E")], 0.0, INTERSECTION),
        # W_IN1 (50 m) leads into W_IN2 (50 m), whose stop line is at its
        # end: always more than 25 m ahead.
        (rm.W_IN1, 0.0, None),
        (rm.W_IN1, 49.0, None),
        # A junction without a signal is no intersection, as in TeraSim.
        (rm.J[("S", "N")], 0.0, None),
        (rm.N_IN, 80.0, None),
        # Just past the signalised junction: what lies behind does not count.
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


def test_the_signal_ahead_is_measured_to_its_stop_line(features: MapFeatures) -> None:
    # From W_IN1 at s the stop line at W_IN2's end is 100 - s ahead.
    assert _at(features, rm.W_IN1, 0.0, lookahead=100.0) is INTERSECTION
    assert _at(features, rm.W_IN1, 26.0, lookahead=74.0) is INTERSECTION
    assert _at(features, rm.W_IN1, 26.0, lookahead=73.0) is None


def test_a_junction_lanelet_fed_by_a_signalised_lane() -> None:
    features = _features()
    junction = features.lanelet(rm.J[("W", "N")])
    for element in list(junction.regulatoryElements):
        junction.removeRegulatoryElement(element)
    assert not MapFeatures.has_traffic_light(_lanelet(features, rm.J[("W", "N")]))
    assert _at(features, rm.J[("W", "N")], 0.0) is INTERSECTION


@pytest.mark.parametrize(
    "tags",
    [{"subtype": "highway"}, {"speed_limit": "100"}],
)
def test_a_highway_by_subtype_or_speed_limit(tags: dict[str, str]) -> None:
    features = _features(W_IN2=tags)
    # Highway wins over the traffic light on the same lanelet, as in TeraSim.
    assert _at(features, rm.W_IN2, 0.0) is AdversityLocation.HIGHWAY


@pytest.mark.parametrize("limit", ["90", "50", "fast"])
def test_a_speed_limit_up_to_90_kmh_is_no_highway(limit: str) -> None:
    features = _features(ONE_WAY1={"speed_limit": limit})
    assert _at(features, rm.ONE_WAY1, 0.0) is None


def test_a_roundabout_ahead() -> None:
    features = _features(ONE_WAY2={"roundabout": "yes"})
    # ONE_WAY1 is 60 m long and leads into the roundabout lanelet.
    assert _at(features, rm.ONE_WAY1, 0.0) is None
    assert _at(features, rm.ONE_WAY1, 40.0) is AdversityLocation.ROUNDABOUT
    assert _at(features, rm.ONE_WAY2, 0.0) is AdversityLocation.ROUNDABOUT


@pytest.mark.parametrize("tag", ["no", "False", "0"])
def test_a_roundabout_tag_of_no_is_none(tag: str) -> None:
    features = _features(ONE_WAY2={"roundabout": tag})
    assert _at(features, rm.ONE_WAY2, 0.0) is None


def test_matches(features: MapFeatures) -> None:
    lanelet = _lanelet(features, rm.W_IN2)
    assert location_matches(INTERSECTION, features, lanelet, 0.0)
    assert not location_matches(AdversityLocation.HIGHWAY, features, lanelet, 0.0)
    elsewhere = _lanelet(features, rm.ONE_WAY1)
    assert location_matches(AdversityLocation.ANY, features, elsewhere, 0.0)
    assert not location_matches(INTERSECTION, features, elsewhere, 0.0)
    assert not location_matches(AdversityLocation.ROUNDABOUT, features, elsewhere, 0.0)


def test_matches_follows_the_precedence() -> None:
    # Signalised and highway: only HIGHWAY matches.  A roundabout ahead of a
    # signalised lane: only INTERSECTION.
    features = _features(W_IN2={"speed_limit": "100"}, ONE_WAY2={"roundabout": "yes"})
    signalised = _lanelet(features, rm.W_IN2)
    assert location_matches(AdversityLocation.HIGHWAY, features, signalised, 0.0)
    assert not location_matches(INTERSECTION, features, signalised, 0.0)
    road = _lanelet(features, rm.ONE_WAY1)
    assert location_matches(AdversityLocation.ROUNDABOUT, features, road, 40.0)
    assert not location_matches(AdversityLocation.HIGHWAY, features, road, 40.0)


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
    chain = features.lane_chain(
        _lanelet(features, rm.W_IN2), 10.0, ahead=0.0, behind=20.0
    )
    assert chain[rm.W_IN1].start == pytest.approx(-60.0)
    assert chain[rm.W_IN1].end == pytest.approx(-10.0)
    assert set(chain) == {rm.W_IN1, rm.W_IN2}
    # Nothing behind by default, even at the very start of a lanelet.
    own = _lanelet(features, rm.W_IN2)
    assert set(features.lane_chain(own, 0.0, ahead=0.0)) == {rm.W_IN2}
