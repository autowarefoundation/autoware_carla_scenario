"""Resolve a :class:`~.model.RelativeLanePose` to an absolute lanelet pose.

The arithmetic of OpenSCENARIO's ``RelativeLanePosition`` on a Lanelet2 map:

1. find the lanelet the reference entity is on, and how far along it
   (:func:`reference_lane_pose`);
2. go ``ds`` metres along that lane, from lanelet to following lanelet
   (:func:`advance_along_lane`);
3. go ``d_lane`` lanes across, through the routing graph's neighbours
   (:func:`shift_lanes`);
4. stand ``offset`` metres from that lanelet's centreline
   (:func:`relative_lane_pose`).

Everything works on a lanelet map and its routing graph passed in, in the
Lanelet2 map frame (right-handed, x East, y North), so it is tested on a
synthetic map with no :class:`~autoware_carla_scenario.coordinate.MapManager`;
:func:`reference_lane_pose_of` is the one function that reads the loaded map.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

# autoware_lanelet2_extension_python must be imported before lanelet2.
from autoware_lanelet2_extension_python.projection import MGRSProjector as _  # noqa: F401
import lanelet2.core
import lanelet2.geometry

from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose
from .model import RelativeLanePose

__all__ = [
    "advance_along_lane",
    "reference_lane_pose",
    "reference_lane_pose_of",
    "relative_lane_pose",
    "shift_lanes",
]

#: How many lanelets nearest the reference entity are weighed.
_CANDIDATES = 8
#: How much farther (m) than the nearest lanelet another may be and still count
#: as one the entity is on: lanelets that overlap in a junction, or meet at a
#: seam, are all at distance 0.
_ON_TOLERANCE_M = 1e-3
#: Arc lengths closer than this (m) to a lanelet's end are at its end.
_EPSILON_M = 1e-9
#: How many lanelets a single ``ds`` may run through before it is taken to be
#: going round a loop with no end.
_MAX_HOPS = 10_000


# ---------------------------------------------------------------------------
# Centreline geometry
# ---------------------------------------------------------------------------


def _centreline(lanelet: Any) -> list[tuple[float, float]]:
    return [(float(p.x), float(p.y)) for p in lanelet.centerline]


def _length(lanelet: Any) -> float:
    return float(lanelet2.geometry.length(lanelet2.geometry.to2D(lanelet.centerline)))


def _heading_at(points: list[tuple[float, float]], s: float) -> float:
    """The centreline's direction (rad) at arc length *s*."""
    travelled = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        span = math.hypot(x1 - x0, y1 - y0)
        if span > 0.0 and travelled + span >= s:
            return math.atan2(y1 - y0, x1 - x0)
        travelled += span
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return math.atan2(y1 - y0, x1 - x0)


def _point_at(points: list[tuple[float, float]], s: float) -> tuple[float, float]:
    """The centreline point at arc length *s* (clamped to the centreline)."""
    travelled = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        span = math.hypot(x1 - x0, y1 - y0)
        if span > 0.0 and travelled + span >= s:
            fraction = max(0.0, (s - travelled) / span)
            return x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)
        travelled += span
    return points[-1]


def _start_heading(lanelet: Any) -> float:
    return _heading_at(_centreline(lanelet), 0.0)


def _end_heading(lanelet: Any) -> float:
    points = _centreline(lanelet)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return math.atan2(y1 - y0, x1 - x0)


def _turn(a: float, b: float) -> float:
    """The absolute angle (rad) between headings *a* and *b*."""
    return abs((b - a + math.pi) % (2.0 * math.pi) - math.pi)


def _arc(lanelet: Any, x: float, y: float) -> tuple[float, float]:
    """``(s, t)`` of a point against *lanelet*'s centreline; ``s`` clamped."""
    arc = lanelet2.geometry.toArcCoordinates(
        lanelet2.geometry.to2D(lanelet.centerline), lanelet2.core.BasicPoint2d(x, y)
    )
    return min(max(float(arc.length), 0.0), _length(lanelet)), float(arc.distance)


# ---------------------------------------------------------------------------
# 1. Where the reference entity is
# ---------------------------------------------------------------------------


def reference_lane_pose(
    lanelet_map: Any,
    x: float,
    y: float,
    heading: float,
    z: Optional[float] = None,
) -> Lanelet2Pose:
    """The lanelet pose of an entity at ``(x, y)`` facing *heading*.

    Args:
        lanelet_map: The Lanelet2 map.
        x, y: The entity's position in the map frame (m).
        heading: Its heading in the map frame (rad, counter-clockwise from
            East).
        z: Its elevation in the map frame, to tell stacked lanelets apart;
            ``None`` ignores elevation.

    Of the lanelets it is on -- several where lanelets overlap in a junction,
    or meet at a seam -- the one whose direction is nearest its heading wins,
    then the one nearest its elevation, then the lowest id.  ``s`` is exact
    along the centreline (not snapped to a vertex).

    Raises:
        ValueError: If the map has no lanelets.
    """
    found = lanelet2.geometry.findNearest(
        lanelet_map.laneletLayer, lanelet2.core.BasicPoint2d(x, y), _CANDIDATES
    )
    if not found:
        raise ValueError("the map has no lanelets to place a relative pose on")
    nearest = float(found[0][0])
    best: Optional[tuple[tuple[float, float, int], Any, float, float]] = None
    for distance, lanelet in found:
        if float(distance) > nearest + _ON_TOLERANCE_M:
            continue
        s, t = _arc(lanelet, x, y)
        turn = _turn(heading, _heading_at(_centreline(lanelet), s))
        dz = 0.0
        if z is not None:
            heights = [float(p.z) for p in lanelet.centerline]
            dz = abs(sum(heights) / len(heights) - z)
        key = (turn, dz, int(lanelet.id))
        if best is None or key < best[0]:
            best = (key, lanelet, s, t)
    assert best is not None
    _key, lanelet, s, t = best
    points = _centreline(lanelet)
    return Lanelet2Pose(
        lanelet_id=int(lanelet.id),
        s=s,
        t=t,
        heading=(heading - _heading_at(points, s) + math.pi) % (2.0 * math.pi)
        - math.pi,
    )


# ---------------------------------------------------------------------------
# 2. Along the lane
# ---------------------------------------------------------------------------


def _straightest(candidates: Iterable[Any], heading: float, *, ahead: bool) -> Any:
    """Of *candidates*, the one that turns least from *heading*; then lowest id.

    *ahead* compares each candidate's start with *heading* (a lanelet that
    follows), otherwise its end (a lanelet that precedes).
    """
    options = list(candidates)
    if not options:
        return None
    return min(
        options,
        key=lambda lanelet: (
            _turn(heading, _start_heading(lanelet) if ahead else _end_heading(lanelet)),
            int(lanelet.id),
        ),
    )


def advance_along_lane(
    lanelet_map: Any, routing_graph: Any, lanelet_id: int, s: float, ds: float
) -> tuple[int, float]:
    """``(lanelet_id, s)`` *ds* metres along the lane from *s* on *lanelet_id*.

    Past a lanelet's end the walk continues on the lanelet that follows it in
    *routing_graph*, before its start on the one that precedes it; where there
    are several, :func:`_straightest` chooses.

    Raises:
        ValueError: If the walk runs off a lanelet that nothing follows (or
            precedes).
    """
    lanelet = lanelet_map.laneletLayer[lanelet_id]
    target = s + ds
    for _hop in range(_MAX_HOPS):
        length = _length(lanelet)
        if -_EPSILON_M <= target <= length + _EPSILON_M:
            return int(lanelet.id), min(max(target, 0.0), length)
        if target > length:
            following = _straightest(
                routing_graph.following(lanelet), _end_heading(lanelet), ahead=True
            )
            if following is None:
                raise ValueError(
                    f"ds={ds:g} m runs {target - length:.2f} m past the end of "
                    f"lanelet {lanelet.id}, which no lanelet follows"
                )
            target -= length
            lanelet = following
        else:
            previous = _straightest(
                routing_graph.previous(lanelet), _start_heading(lanelet), ahead=False
            )
            if previous is None:
                raise ValueError(
                    f"ds={ds:g} m runs {-target:.2f} m before the start of "
                    f"lanelet {lanelet.id}, which no lanelet precedes"
                )
            lanelet = previous
            target += _length(lanelet)
    raise ValueError(f"ds={ds:g} m goes through more than {_MAX_HOPS} lanelets")


# ---------------------------------------------------------------------------
# 3. Across lanes
# ---------------------------------------------------------------------------


def _neighbour(routing_graph: Any, lanelet: Any, left: bool) -> Any:
    """The lane to the *left* (or right) of *lanelet*, lane-changeable or not."""
    if left:
        return routing_graph.left(lanelet) or routing_graph.adjacentLeft(lanelet)
    return routing_graph.right(lanelet) or routing_graph.adjacentRight(lanelet)


def shift_lanes(
    lanelet_map: Any, routing_graph: Any, lanelet_id: int, s: float, d_lane: int
) -> tuple[int, float]:
    """``(lanelet_id, s)`` *d_lane* lanes to the left (negative: right) of a point.

    Each step goes to the routing graph's neighbour of the lanelet and puts
    ``s`` where the point on the centreline projects onto the neighbour's, so a
    lane on the outside of a bend -- longer than the inside one -- keeps the
    point abreast rather than at the same distance from its start.

    Raises:
        ValueError: If a lanelet on the way has no neighbour on that side.
    """
    lanelet = lanelet_map.laneletLayer[lanelet_id]
    left = d_lane > 0
    side = "left" if left else "right"
    for step in range(abs(d_lane)):
        neighbour = _neighbour(routing_graph, lanelet, left)
        if neighbour is None:
            raise ValueError(
                f"d_lane={d_lane:+d}: lanelet {lanelet.id} has no lane to its "
                f"{side} (after {step} of {abs(d_lane)} lane changes)"
            )
        x, y = _point_at(_centreline(lanelet), s)
        s, _t = _arc(neighbour, x, y)
        lanelet = neighbour
    return int(lanelet.id), s


# ---------------------------------------------------------------------------
# 4. Altogether
# ---------------------------------------------------------------------------


def relative_lane_pose(
    lanelet_map: Any,
    routing_graph: Any,
    reference: Lanelet2Pose,
    position: RelativeLanePose,
) -> Lanelet2Pose:
    """The absolute lanelet pose *position* names, from the *reference* pose.

    The reference's own lateral offset and heading do not carry over:
    ``offset`` is measured from the target lane's centreline and ``yaw`` from
    its direction, as OpenSCENARIO measures them.  An unstated ``yaw`` gives
    ``heading=0`` here; the trajectory then takes its direction from the path.
    """
    lanelet_id, s = advance_along_lane(
        lanelet_map, routing_graph, reference.lanelet_id, reference.s, position.ds
    )
    if position.d_lane:
        lanelet_id, s = shift_lanes(
            lanelet_map, routing_graph, lanelet_id, s, position.d_lane
        )
    return Lanelet2Pose(
        lanelet_id=lanelet_id,
        s=s,
        t=float(position.offset),
        heading=0.0 if position.yaw is None else float(position.yaw),
    )


def reference_lane_pose_of(pose: CarlaWorldPose) -> Lanelet2Pose:
    """:func:`reference_lane_pose` of a CARLA world pose, on the loaded map."""
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415

    manager = MapManager.get_instance()
    offset_x, offset_y = manager.mgrs_offset
    return reference_lane_pose(
        manager.lanelet_map,
        pose.x + offset_x,
        -pose.y + offset_y,
        -math.radians(pose.yaw),
        pose.z + manager.z_offset,
    )
