"""Which of the lanelets under a pose it is on: one rule for every caller.

Where lanelets overlap in plan view a pose is "on" several of them at once:
lanelets that overlap on one level -- a junction's, or two that meet at a
seam -- and lanelets stacked over one another -- an overpass and the road
beneath it, metres apart.  :func:`choose_lanelet` tells them apart the same
way wherever a pose is resolved to a lanelet (:func:`.transform.to_lanelet2`,
:func:`~autoware_carla_scenario.trajectory.relative_lane.reference_lane_pose`):

1. the pose's elevation picks the level -- only the lanelets whose surface
   at the pose (the centreline's height interpolated at the pose's arc
   position, not the lanelet's mean height) is within
   :data:`STACKED_HEIGHT_TOLERANCE_M` of the nearest one stay;
2. of those, the one whose direction is nearest the pose's heading wins;
3. then the one nearest the pose's elevation, then the lowest id.

Elevation first because a deck and the road under it often run nearly
parallel: ranked by heading first, a hundredth of a radian between them
decides, and the pose lands on the wrong level (#2221).
"""

from __future__ import annotations

import math

__all__ = ["STACKED_HEIGHT_TOLERANCE_M", "choose_lanelet", "height_at"]

#: How much farther (m) from a point's elevation than the nearest lanelet
#: surface over or under it a lanelet's surface may be and still hold the
#: point.  Lanelets that overlap on one level -- a junction's -- all hold it;
#: of lanelets stacked over one another -- an overpass and the road beneath
#: it, metres apart -- only the one at the point's level does.
STACKED_HEIGHT_TOLERANCE_M = 1.0


def choose_lanelet(candidates: list[tuple[float, float, int]]) -> int:
    """The index in *candidates* of the lanelet a pose is on.

    Args:
        candidates: ``(dz, turn, lanelet_id)`` of each lanelet the pose is on
            in plan view: how far (m) the pose's elevation is from the
            lanelet's surface at the pose, and the absolute angle (rad)
            between the pose's heading and the lanelet's direction there.
            Pass ``dz = 0`` for every candidate when the elevation is not
            known; heading alone then decides.

    Of the candidates on the pose's level -- ``dz`` within
    :data:`STACKED_HEIGHT_TOLERANCE_M` of the smallest -- the one with the
    smallest ``turn`` wins, then the smallest ``dz``, then the lowest id.

    Raises:
        ValueError: If there are no candidates.
    """
    if not candidates:
        raise ValueError("no lanelet to choose from")
    lowest = min(dz for dz, _turn, _id in candidates)
    best = -1
    for index, (dz, turn, lanelet_id) in enumerate(candidates):
        if dz > lowest + STACKED_HEIGHT_TOLERANCE_M:
            continue
        if best < 0:
            best = index
            continue
        best_dz, best_turn, best_id = candidates[best]
        if (turn, dz, lanelet_id) < (best_turn, best_dz, best_id):
            best = index
    return best


def height_at(points: list[tuple[float, float, float]], s: float) -> float:
    """The height of the polyline *points* at plan-view arc length *s*.

    Linear between vertices; clamped to the first and last vertex.
    """
    if len(points) == 1 or s <= 0.0:
        return points[0][2]
    travelled = 0.0
    for index in range(len(points) - 1):
        x0, y0, z0 = points[index]
        x1, y1, z1 = points[index + 1]
        span = math.hypot(x1 - x0, y1 - y0)
        if span > 0.0 and travelled + span >= s:
            return z0 + (s - travelled) / span * (z1 - z0)
        travelled += span
    return points[-1][2]
