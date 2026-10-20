"""Where an agent is, for an adversity's location gate (TeraSim's ``get_location``).

TeraSim only lets an adversity fire where it was configured to: a vehicle
"at an intersection" may run the light, one "on a highway" may cut in.  It
answers from SUMO's network; the port answers from the Lanelet2 map, with
the same precedence (highway, then intersection, then roundabout) and the
same look-ahead:

* :attr:`AdversityLocation.HIGHWAY`: the lanelet has subtype ``highway``, or
  a ``speed_limit`` above :data:`HIGHWAY_SPEED_LIMIT_KMH` -- TeraSim's
  "motorway edge, or faster than 25 m/s";
* :attr:`AdversityLocation.INTERSECTION`: the lanelet is *signalised* -- a
  traffic light regulates it, or it is a junction lanelet (``turn_direction``
  tagged) entered from one a traffic light regulates -- or the end of a
  signalised lanelet, where its stop line is, lies within the look-ahead.
  TeraSim's "a lane the signal controls (or a junction lane fed by one), or
  the next signal within 25 m"; a junction without a signal is no
  intersection, as in TeraSim;
* :attr:`AdversityLocation.ROUNDABOUT`: a lanelet tagged ``roundabout`` (any
  value but ``no``, ``false`` or ``0``), or one regulated by an element of
  subtype ``roundabout``, starts within the look-ahead.  Lanelet2 has no
  standard roundabout tag; this is the convention the port reads.

Anything else has no location (:func:`location_of` returns None), and only
:attr:`AdversityLocation.ANY` matches it.  Distances run along the lanes, by
the shortest chain of following lanelets, no lane change.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

from ..route.geometry import MapFeatures

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

__all__ = [
    "DEFAULT_LOOKAHEAD_M",
    "HIGHWAY_SPEED_LIMIT_KMH",
    "AdversityLocation",
    "location_matches",
    "location_of",
]

#: How far ahead (metres, along the lanes) a signal or a roundabout puts an
#: agent in it, as TeraSim's 25 m.
DEFAULT_LOOKAHEAD_M = 25.0

#: A speed limit above this (km/h, 25 m/s) makes a road a highway.
HIGHWAY_SPEED_LIMIT_KMH = 90.0

_NO = ("no", "false", "0")


class AdversityLocation(Enum):
    """Where an adversity may fire."""

    ANY = "any"
    INTERSECTION = "intersection"
    HIGHWAY = "highway"
    ROUNDABOUT = "roundabout"


def _is_highway(lanelet: ConstLanelet) -> bool:
    if (
        "subtype" in lanelet.attributes
        and str(lanelet.attributes["subtype"]) == "highway"
    ):
        return True
    limit = MapFeatures.speed_limit(lanelet)
    return limit is not None and limit > HIGHWAY_SPEED_LIMIT_KMH


def _is_signalised(features: MapFeatures, lanelet: ConstLanelet) -> bool:
    if MapFeatures.has_traffic_light(lanelet):
        return True
    if not MapFeatures.is_junction(lanelet):
        return False
    for previous in features.graph.previous(lanelet):
        if MapFeatures.has_traffic_light(previous):
            return True
    return False


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


def _found_ahead(
    features: MapFeatures,
    lanelet: ConstLanelet,
    s: float,
    lookahead: float,
    roundabout: bool,
) -> bool:
    """Whether a signal (or, with *roundabout*, a roundabout) is within reach.

    A signal counts when the end of its lanelet is within *lookahead* of *s*
    on *lanelet*; a roundabout when the start of its lanelet is.  The search
    runs the following lanelets in order of their start, shortest chain
    first, and stops at the first find.
    """
    if roundabout and _is_roundabout(lanelet):
        return True
    if not roundabout and _is_signalised(features, lanelet):
        return True
    # Each lanelet reached, by id: how far ahead of the agent it starts.
    starts: dict[int, float] = {}
    frontier: list[tuple[float, ConstLanelet]] = [
        (features.length(lanelet) - s, lanelet)
    ]
    while frontier:
        index = 0
        for i in range(1, len(frontier)):
            if frontier[i][0] < frontier[index][0]:
                index = i
        start, current = frontier.pop(index)
        if start > lookahead:
            continue
        for following in features.graph.following(current):
            key = int(following.id)
            if key in starts and starts[key] <= start:
                continue
            starts[key] = start
            if roundabout and _is_roundabout(following):
                return True
            end = start + features.length(following)
            if (
                not roundabout
                and end <= lookahead
                and _is_signalised(features, following)
            ):
                return True
            frontier.append((end, following))
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
    if _found_ahead(features, lanelet, s, lookahead, False):
        return AdversityLocation.INTERSECTION
    if _found_ahead(features, lanelet, s, lookahead, True):
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

    Reads only what *location* needs: :attr:`AdversityLocation.ANY` nothing,
    a highway the lanelet itself, an intersection no search for a
    roundabout.
    """
    if location is AdversityLocation.ANY:
        return True
    if _is_highway(lanelet):
        return location is AdversityLocation.HIGHWAY
    if location is AdversityLocation.HIGHWAY:
        return False
    signal = _found_ahead(features, lanelet, s, lookahead, False)
    if location is AdversityLocation.INTERSECTION:
        return signal
    return not signal and _found_ahead(features, lanelet, s, lookahead, True)
