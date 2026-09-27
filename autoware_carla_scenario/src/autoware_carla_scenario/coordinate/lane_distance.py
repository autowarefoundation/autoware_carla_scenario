"""Distance measured along the road, rather than across the map.

OpenSCENARIO's ``coordinateSystem: lane`` asks for the gap a driver would
describe: how far along the lane the vehicle in front is.  The entity
coordinate system answers a different question -- a straight-line offset
projected onto a direction -- and the two agree only where the road is
straight.

This module is the shared half of that measurement.  All three distance-based
conditions reach for it, so the decisions that make it correct live here once
rather than three times.  Each asks a different question, so there is one
function per question rather than one that returns a tuple nobody wants whole:

* :func:`lane_separation` -- how far apart, unsigned.  Needs no velocity.
* :func:`lane_gap` -- how far *ahead*, signed by the source's travel.
* :func:`lane_closing_speed` -- how fast that gap is shrinking.

What is here, and what is not
-----------------------------
Both entities are projected onto the road network and their ``s`` compared,
following the chain of connected roads between them when they are not on the
same one.

Following the chain is not a refinement, it is the difference between the
measurement working and not.  An OpenDRIVE map is cut into short roads -- on
this project's own fixture the median road is 33 m and three quarters are
under 50 m -- so a leader at an ordinary following distance is usually on the
*next* road, not the one behind it.  Measuring within a single road would
answer "no measurement" to most of the following scenarios this coordinate
system exists for, which is exactly the silent never-fires it exists to
remove.  The links are read off the OpenDRIVE the converter already emits, so
this needs no routing graph.

What has no answer is a **junction**: several roads leave it, and which one a
vehicle will take is a route rather than a geometric fact.  A walk stops
there, and the pair is reported as having no measurement rather than
approximated or quietly swapped for the straight line -- a scenario that
silently changes what it measures passes for the wrong reason, which is worse
than not firing.  Reaching across a junction needs a lanelet2 routing graph in
the live runtime, which nothing holds yet.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from .map_manager import MapManager
from .poses import CarlaWorldPose, OpenDrivePose
from .transform import to_opendrive

logger = logging.getLogger(__name__)

__all__ = [
    "lane_closing_speed",
    "lane_gap",
    "lane_separation",
]

#: Nothing here needs a tuning constant: how a vehicle's motion relates to the
#: road is read from the projection itself, not sampled.  See
#: :func:`_speed_along_s`.
_NEAR_ZERO = 1e-9

#: Guard for the "no map" warning below.  A condition is evaluated every tick,
#: so an unguarded warning would print at the tick rate for a whole run and
#: bury everything else, and the fact it reports does not change between ticks.
_warned_without_map = False


def lane_separation(source: CarlaWorldPose, target: CarlaWorldPose) -> Optional[float]:
    """Return how far apart the two are along their shared road, unsigned.

    No velocity is needed: a separation has no direction, so neither entity has
    to be moving for this to mean something.

    Returns:
        Metres along the road, or ``None`` when there is no such measurement
        (see :func:`lane_gap` for what those cases are).
    """
    source_od = _project(source)
    target_od = _project(target)
    if source_od is None or target_od is None:
        return None
    reach = _reach(source_od, target_od)
    return None if reach is None else reach.distance


def lane_gap(
    source: CarlaWorldPose,
    travel_x: float,
    travel_y: float,
    target: CarlaWorldPose,
) -> Optional[float]:
    """Return how far *target* is ahead of *source* along their shared road.

    Positive is ahead of the source in its direction of travel, negative is
    behind it.  What to do with the sign is the caller's: a headway wants only
    what is in front.

    Every failure below is reported as no measurement rather than as a number,
    because a caller handed a number cannot tell it came from somewhere else:

    * **No map is loaded.**  Nothing can be projected onto a road.  Logged
      once, because unlike the others this is a misconfiguration rather than a
      fact about the scenario, and a condition that never fires for a whole run
      should say why.
    * **No chain of roads joins the two.**  A junction stands between them, or
      they are further apart than a measurement reaches; see the module
      docstring.
    * **The source is not moving.**  There is no direction of travel to measure
      along.
    * **The source is moving square across the road.**  It makes no progress
      along the road, so there is no gap in front of it to measure.

    The last three change from tick to tick as vehicles move, so they are
    silent: they are answers, not faults.

    Args:
        source: Where the measurement is taken from, in CARLA world
            coordinates.
        travel_x: The source's velocity, x component (CARLA world frame).
        travel_y: The source's velocity, y component.
        target: The other entity's position, in CARLA world coordinates.

    Returns:
        The signed along-road distance in metres, or ``None``.
    """
    fix = _speed_along_s(source, travel_x, travel_y)
    target_od = _project(target)
    if fix is None or target_od is None:
        return None
    source_od, along = fix
    if abs(along) < _NEAR_ZERO:
        # Moving, but square across the road: no progress along it to measure
        # a gap in front of.
        return None
    reach = _reach(source_od, target_od)
    if reach is None:
        return None

    # `reach.forward` is where the target lies in the source road's own `s`;
    # `along` is which way the source is driving in it.  Ahead is when the two
    # agree.
    driving_forward = along > 0
    sign = 1.0 if reach.forward == driving_forward else -1.0
    return sign * reach.distance


def lane_closing_speed(
    source: CarlaWorldPose,
    source_travel_x: float,
    source_travel_y: float,
    target: CarlaWorldPose,
    target_travel_x: float,
    target_travel_y: float,
) -> Optional[float]:
    """Return how fast the along-road separation between the two is shrinking.

    Positive means closing, negative means opening.  Both speeds are the
    component **along the road**, which is the whole point: a straight-line
    closing speed on a curve counts a vehicle's cornering as approach, and a
    time-to-collision built on it is short for the wrong reason.

    **Which way is "towards" comes from where the two are, not from how the
    source is driving.**  That distinction is the difference between this and
    :func:`lane_gap`, and it decides two cases a collision measure must not
    lose:

    * a **stationary** source with something bearing down on it -- the closing
      speed is the target's, and the pair has a perfectly finite time to
      collision;
    * a **faster target behind**, which is a rear-end collision and closes just
      as surely as one in front.

    Signing by the source's travel instead would answer ``None`` to both, which
    is the silent never-fires this whole coordinate system exists to remove.

    Returns:
        Metres per second, or ``None`` when the two are not on one road, when
        either cannot be placed on it, or when they are level along it -- with
        no separation there is no direction to close along.
    """
    source_od = _project(source)
    target_od = _project(target)
    if source_od is None or target_od is None:
        return None
    reach = _reach(source_od, target_od)
    if reach is None or reach.distance < _NEAR_ZERO:
        # Level along the road: no separation, so no direction to close along.
        return None

    source_along = _along_s(source, source_travel_x, source_travel_y)
    target_along = _along_s(target, target_travel_x, target_travel_y)
    if source_along is None or target_along is None:
        return None

    # Both speeds are read off their own road's `s`, and the two roads may
    # number it opposite ways round.  Putting each on the axis that runs from
    # the source to the target is what makes them subtractable: the separation
    # then shrinks at source minus target, whichever of the two is behind.
    source_sign = 1.0 if reach.forward else -1.0
    target_sign = 1.0 if reach.target_forward else -1.0
    return source_sign * source_along - target_sign * target_along


def _along_s(pose: CarlaWorldPose, travel_x: float, travel_y: float) -> Optional[float]:
    """Return the entity's speed along increasing ``s``, zero when it is still.

    A stationary entity is not an unanswerable one: its progress along the road
    is zero, which is a number a closing speed can be built from.  ``None`` is
    kept for the entity that cannot be placed on the road at all.
    """
    if math.hypot(travel_x, travel_y) < _NEAR_ZERO:
        return 0.0
    fix = _speed_along_s(pose, travel_x, travel_y)
    return None if fix is None else fix[1]


def _project(pose: CarlaWorldPose) -> Optional[OpenDrivePose]:
    """Return *pose* on its nearest OpenDRIVE road, or ``None``.

    ``to_opendrive`` raises when no map is loaded, which must not take a run
    down: a condition that cannot measure says so by not firing.
    """
    global _warned_without_map

    try:
        return to_opendrive(pose)
    except RuntimeError:
        if not _warned_without_map:
            _warned_without_map = True
            logger.warning(
                "A lane-coordinate distance needs a loaded map and none is "
                "initialized, so the condition cannot measure and will never "
                "fire. Load a map, or ask for the entity coordinate system."
            )
        return None
    except Exception:
        # Transient and positional -- a pose off the road network, a road the
        # network does not carry.  Debug rather than warning: it can be true
        # for a few ticks of a legitimate run.
        logger.debug(
            "lane-coordinate distance: (%.2f, %.2f) could not be placed on a road",
            pose.x,
            pose.y,
        )
        return None


def _speed_along_s(
    pose: CarlaWorldPose,
    travel_x: float,
    travel_y: float,
) -> "Optional[tuple[OpenDrivePose, float]]":
    """Return where the entity is on its road, and its speed along ``s``.

    ``s`` runs along the road's reference line and a vehicle may drive either
    way along it, so the sign has to come from somewhere.  It is not read off
    ``lane_id``: that sign says which *side* of the reference line a lane is
    on, and its relationship to the direction of travel is a convention -- one
    left-hand traffic inverts.

    It is read off the projection instead.  :attr:`OpenDrivePose.heading` is
    already the angle between the pose's own heading and the road's, so
    projecting the entity with its **velocity** in place of its yaw makes that
    field the angle between travel and road directly.  ``cos`` of it is then
    the fraction of the speed going along the road: the sign says which way,
    and the magnitude is what a closing speed in lane coordinates needs.

    One projection, exact, and with nothing to fall off: an earlier version
    sampled a point two metres ahead and compared ``s``, which refused to
    measure whenever that point crossed into the neighbouring road -- as it
    does for any vehicle within a probe's length of where its road begins.

    Returns:
        ``(pose on its road, signed speed along s)``, or ``None`` when the
        entity is stationary or cannot be placed on a road.  A caller that
        treats a stationary entity as having zero along-road speed must say so
        itself; the two are not the same answer.
    """
    speed = math.hypot(travel_x, travel_y)
    if speed < _NEAR_ZERO:
        return None

    # CARLA's world frame is left-handed, and `to_opendrive` reads a pose's
    # direction as `-radians(yaw)`, so the velocity's yaw follows the same
    # convention as any other CARLA heading.
    facing = CarlaWorldPose(
        x=pose.x,
        y=pose.y,
        z=pose.z,
        yaw=math.degrees(math.atan2(travel_y, travel_x)),
    )
    facing_od = _project(facing)
    if facing_od is None:
        return None

    return facing_od, speed * math.cos(facing_od.heading)


# ---------------------------------------------------------------------------
# Walking the chain of connected roads
# ---------------------------------------------------------------------------

#: How far along the chain of connected roads a measurement will reach.
#:
#: Bounded because the search runs over a graph the scenario does not control,
#: and because a pair further apart than this is not one any of these
#: conditions is written about.  200 m covers a time-to-collision at motorway
#: speed over the handful of seconds a threshold is set at, which is the
#: longest range asked for here.
_MAX_CHAIN_METRES = 200.0

#: Belt to the budget's braces: links that form a loop would otherwise be
#: walked until the distance cap caught them, once per condition per tick.
_MAX_CHAIN_HOPS = 24


@dataclass(frozen=True)
class _Reach:
    """How *target* was reached from *source* along the roads joining them.

    Attributes:
        distance: Along-road metres between the two points, never negative.
            Which way the target lies is *forward*.
        forward: Whether it lies in the direction its source's road numbers
            ``s``, rather than against it.
        target_forward: Whether the **target's** road numbers ``s`` the same
            way the measurement runs.  A chain can enter a road from its far
            end, and a speed read off that road's own ``s`` then has the wrong
            sign for a closing speed.
    """

    distance: float
    forward: bool
    target_forward: bool


def _road(road_id: str) -> Optional[object]:
    """Return the loaded road with *road_id*, or ``None``."""
    try:
        network = MapManager.get_instance().road_network
    except RuntimeError:
        return None
    return network.road_ids_to_object.get(str(road_id))


def _road_length(road_id: str) -> Optional[float]:
    """Return the road's reference-line length in metres, or ``None``."""
    road = _road(road_id)
    if road is None:
        return None
    try:
        return float(road["length"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


#: The road link graph, rebuilt when a different map is loaded.
#:
#: Keyed by the identity of the loaded network rather than by a map name: a
#: scenario queue re-initializes :class:`MapManager`, and a stale graph would
#: measure against the previous run's roads.
_links_for: "Optional[tuple[int, dict[tuple[str, str], tuple[str, str]]]]" = None


def _link_graph() -> "dict[tuple[str, str], tuple[str, str]]":
    """Return which road is reached by leaving a given road at a given end.

    A node is ``(road_id, end)`` -- the end being left by, ``"start"`` or
    ``"end"`` -- and the value is the road entered and the end entered at.

    Built in one pass and cached, for two reasons:

    * **Every link is read both ways.**  A road-to-road link is usually
      written from one side only: OpenDRIVE has a road adjoining a junction
      name the *junction*, not the connecting road on the far side, so B
      commonly says "predecessor: junction 7" where A says "successor: B".
      On this project's fixture 367 of 490 road-to-road successors have no
      matching road predecessor, and most of that is this idiom rather than
      anything wrong.  A walk that trusted each road's own ``predecessor``
      would measure A to B and then refuse B to A, and a separation that
      depends on which vehicle is asked is not a separation.
    * A condition is evaluated every tick, and the graph does not change
      between ticks.

    Links naming a **junction** are left out, which is what stops a walk
    there: several roads leave a junction and picking one is a route, not a
    geometric fact.  The map's own statement wins over the one derived by
    reversing another link, so a map that does spell out both directions is
    read as it is written.
    """
    global _links_for

    try:
        network = MapManager.get_instance().road_network
    except RuntimeError:
        return {}
    if _links_for is not None and _links_for[0] == id(network):
        return _links_for[1]

    explicit: "dict[tuple[str, str], tuple[str, str]]" = {}
    derived: "dict[tuple[str, str], tuple[str, str]]" = {}
    for road_id, road in network.road_ids_to_object.items():
        link = road.road_xml.find("link")
        if link is None:
            continue
        # A road is left by its `end` to reach a successor, by its `start` to
        # reach a predecessor.
        for kind, leaving in (("successor", "end"), ("predecessor", "start")):
            for element in link.findall(kind):
                if element.attrib.get("elementType") != "road":
                    continue
                other = element.attrib.get("elementId")
                if other is None:
                    continue
                entered = element.attrib.get("contactPoint", "start")
                explicit[(str(road_id), leaving)] = (str(other), entered)
                # Coming back the other way leaves the far road by the very
                # end this link met it at.
                derived.setdefault((str(other), entered), (str(road_id), leaving))

    graph = dict(derived)
    graph.update(explicit)
    _links_for = (id(network), graph)
    return graph


def _next_road(road_id: str, along_s: bool) -> Optional["tuple[str, bool]"]:
    """Return the road continuing past *road_id*, and how it is entered.

    Travelling along ``s`` leaves a road by its ``end``, and against ``s`` by
    its ``start``.  Entering the next road at its ``start`` leaves the walk
    travelling along that road's ``s``; entering at its ``end`` turns it
    around.

    ``None`` where the network ends, and -- the case worth naming -- where the
    only way on is through a **junction**, which is where a chain stops being
    a chain.
    """
    step = _link_graph().get((str(road_id), "end" if along_s else "start"))
    if step is None:
        return None
    next_id, entered = step
    return next_id, entered == "start"


def _reach(source: OpenDrivePose, target: OpenDrivePose) -> Optional[_Reach]:
    """Return how far *target* lies from *source* along the roads joining them.

    The same road needs no walk.  Otherwise the chain is followed both ways,
    because a leader on the next road along is the ordinary case rather than
    the exotic one: an OpenDRIVE map is cut into short roads -- a median of
    33 m on this project's own fixture, three quarters of them under 50 m --
    so a vehicle at a comfortable following distance is usually *not* on the
    road behind it.  Measuring only within one road would answer ``None`` to
    most of the following scenarios this coordinate system exists for, which
    is the silent never-fires it exists to remove.

    Returns:
        A :class:`_Reach`, or ``None`` when no chain of roads joins the two
        within :data:`_MAX_CHAIN_METRES`.
    """
    if source.road_id == target.road_id:
        offset = target.s - source.s
        forward = offset >= 0
        return _Reach(abs(offset), forward=forward, target_forward=forward)

    for forward in (True, False):
        found = _walk(source, target, forward)
        if found is not None:
            return found
    return None


def _walk(
    source: OpenDrivePose, target: OpenDrivePose, forward: bool
) -> Optional[_Reach]:
    """Follow the chain one way from *source*, and report reaching *target*."""
    source_length = _road_length(source.road_id)
    if source_length is None:
        return None

    # As far as the end of the source's own road that the walk leaves by.
    travelled = (source_length - source.s) if forward else source.s
    road_id = source.road_id
    along_s = forward

    for _ in range(_MAX_CHAIN_HOPS):
        step = _next_road(road_id, along_s)
        if step is None:
            return None
        road_id, along_s = step

        length = _road_length(road_id)
        if length is None:
            return None

        if road_id == target.road_id:
            # `along_s` says whether this road numbers `s` the way the walk is
            # travelling, which is what turns its `s` into a distance from
            # where the walk entered it.
            inner = target.s if along_s else (length - target.s)
            total = travelled + inner
            if total > _MAX_CHAIN_METRES:
                return None
            return _Reach(total, forward=forward, target_forward=along_s)

        travelled += length
        if travelled > _MAX_CHAIN_METRES:
            return None

    return None
