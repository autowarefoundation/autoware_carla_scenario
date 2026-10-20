"""Where an agent is, for an adversity's location gate (TeraSim's ``get_location``).

TeraSim only lets an adversity fire where it was configured to: a vehicle
"at an intersection" may run the light, one "on a highway" may cut in.  It
answers from SUMO's network; the port answers from the Lanelet2 map by the
same criteria the ODD's probes read (:mod:`..odd.probes`, the default ODD's
``scenery.road_type`` and ``scenery.junction``), so an adversity gated to a
location fires where coverage records that location.  TeraSim's precedence
(highway, then intersection, then roundabout) and look-ahead are kept:

* :attr:`AdversityLocation.HIGHWAY`: the lanelet's road type
  (:func:`~..odd.probes.lanelet_road_type`) is ``highway``;
* :attr:`AdversityLocation.INTERSECTION`: the lanelet is in a junction
  (:func:`~..odd.probes.lanelet_in_junction`, ``turn_direction`` tagged), or
  a junction lanelet starts within the look-ahead -- TeraSim's approach to
  an intersection.  Whether the junction is signalised is not the gate's
  question: the adversity that runs a light asks for the light itself;
* :attr:`AdversityLocation.ROUNDABOUT`: a lanelet tagged ``roundabout`` (any
  value but ``no``, ``false`` or ``0``), or one regulated by an element of
  subtype ``roundabout``, starts within the look-ahead.  Neither Lanelet2
  nor the default ODD has a roundabout attribute; this is the convention
  the port reads.

Anything else has no location (:func:`location_of` returns None), and only
:attr:`AdversityLocation.ANY` matches it.  Distances run along the lanes, by
the shortest chain of following lanelets, no lane change.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

from ..odd.probes import lanelet_in_junction, lanelet_road_type
from ..route.geometry import LaneSpan, MapFeatures

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

__all__ = [
    "DEFAULT_LOOKAHEAD_M",
    "AdversityLocation",
    "location_matches",
    "location_of",
]

#: How far ahead (metres, along the lanes) a junction or a roundabout puts an
#: agent in it, as TeraSim's 25 m.
DEFAULT_LOOKAHEAD_M = 25.0

_NO = ("no", "false", "0")


class AdversityLocation(Enum):
    """Where an adversity may fire."""

    ANY = "any"
    INTERSECTION = "intersection"
    HIGHWAY = "highway"
    ROUNDABOUT = "roundabout"


def _is_highway(lanelet: ConstLanelet) -> bool:
    return lanelet_road_type(lanelet) == "highway"


def _is_roundabout(lanelet: ConstLanelet) -> bool:
    if "roundabout" in lanelet.attributes:
        if str(lanelet.attributes["roundabout"]).lower() not in _NO:
            return True
    for element in lanelet.regulatoryElements:
        if (
            "subtype" in element.attributes
            and str(element.attributes["subtype"]) == "roundabout"
        ):
            return True
    return False


def _junction_ahead(chain: dict[int, LaneSpan]) -> bool:
    # The agent's own lanelet is in the chain, so being in a junction counts.
    for span in chain.values():
        if lanelet_in_junction(span.lanelet):
            return True
    return False


def _roundabout_ahead(chain: dict[int, LaneSpan]) -> bool:
    for span in chain.values():
        if _is_roundabout(span.lanelet):
            return True
    return False


def location_of(
    features: MapFeatures,
    lanelet: ConstLanelet,
    s: float,
    lookahead: float = DEFAULT_LOOKAHEAD_M,
) -> Optional[AdversityLocation]:
    """The location of an agent *s* metres along *lanelet*, or None."""
    if _is_highway(lanelet):
        return AdversityLocation.HIGHWAY
    chain = features.lane_chain(lanelet, s, lookahead)
    if _junction_ahead(chain):
        return AdversityLocation.INTERSECTION
    if _roundabout_ahead(chain):
        return AdversityLocation.ROUNDABOUT
    return None


def location_matches(
    location: AdversityLocation,
    features: MapFeatures,
    lanelet: ConstLanelet,
    s: float,
    lookahead: float = DEFAULT_LOOKAHEAD_M,
) -> bool:
    """Whether an agent *s* metres along *lanelet* is at *location*.

    :attr:`AdversityLocation.ANY` matches anywhere, without reading the map.
    """
    if location is AdversityLocation.ANY:
        return True
    return location_of(features, lanelet, s, lookahead) is location
