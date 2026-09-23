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
Only the **same-road** case: both entities projected onto one OpenDRIVE road,
and their ``s`` compared.  That covers the curvature error, which is where the
entity frame does real damage, and it needs no routing graph.

A pair on two different roads -- the junction case -- has no answer here.  It
is not approximated and it does not fall back to the straight line: a scenario
that silently swaps one measure for another passes for the wrong reason, which
is worse than not firing.  Reaching across roads needs a lanelet2 routing graph
in the live runtime, which nothing holds yet.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

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
    if source_od.road_id != target_od.road_id:
        return None
    return abs(target_od.s - source_od.s)


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
    * **The two are on different roads.**  The junction case, which needs a
      routing graph; see the module docstring.
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
    if source_od.road_id != target_od.road_id:
        return None
    if abs(along) < _NEAR_ZERO:
        # Moving, but square across the road: no progress along it to measure
        # a gap in front of.
        return None

    sign = 1.0 if along > 0 else -1.0
    return sign * (target_od.s - source_od.s)


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
    if source_od.road_id != target_od.road_id:
        return None

    offset = target_od.s - source_od.s
    if abs(offset) < _NEAR_ZERO:
        return None

    source_along = _along_s(source, source_travel_x, source_travel_y)
    target_along = _along_s(target, target_travel_x, target_travel_y)
    if source_along is None or target_along is None:
        return None

    # The separation is ``abs(offset)``, so it shrinks at
    # ``sign(offset) * (source_along - target_along)``: whoever is behind
    # closes by going faster along the road, whichever of them that is.
    sign = 1.0 if offset > 0 else -1.0
    return sign * (source_along - target_along)


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
