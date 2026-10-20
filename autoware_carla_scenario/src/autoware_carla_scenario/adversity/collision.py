"""Conflicts between two predicted trajectories (TeraSim's ``collision_check_cy``).

:func:`trajectories_intersect` is the test NADE's maneuver challenge and
avoidability run: two trajectories sampled on the same clock conflict when
their footprints overlap at a common sample, or when their paths cross within
a common step.
"""

from __future__ import annotations

from typing import Optional

from .geometry import (
    AgentFootprint,
    PredictedState,
    circle_centers,
    circle_offsets,
    circle_radius,
)

__all__ = [
    "DEFAULT_DISTANCE_THRESHOLD",
    "collision_time",
    "segments_intersect",
    "trajectories_intersect",
]

#: Two agents further apart than this (metres) at the first sample are not
#: checked at all, as in TeraSim.
DEFAULT_DISTANCE_THRESHOLD = 50.0


def collision_time(
    trajectory1: list[PredictedState],
    footprint1: AgentFootprint,
    trajectory2: list[PredictedState],
    footprint2: AgentFootprint,
    buffer: float = 0.0,
) -> Optional[float]:
    """Time of the first common sample at which the footprints overlap.

    Samples are paired by index, so the two trajectories must share their
    clock; the shorter one bounds the check.  *buffer* (metres) widens each
    footprint.  None when they never overlap.
    """
    reach = circle_radius(footprint1) + circle_radius(footprint2) + 2.0 * buffer
    reach2 = reach * reach
    offsets1 = circle_offsets(footprint1)
    offsets2 = circle_offsets(footprint2)
    for i in range(min(len(trajectory1), len(trajectory2))):
        point1 = trajectory1[i]
        centers1 = circle_centers(point1, offsets1)
        centers2 = circle_centers(trajectory2[i], offsets2)
        for x1, y1 in centers1:
            for x2, y2 in centers2:
                dx = x2 - x1
                dy = y2 - y1
                if dx * dx + dy * dy <= reach2:
                    return point1.t
    return None


def _orientation(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(
    a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]
) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[
        1
    ] <= max(a[1], b[1])


def segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """Whether segment *ab* and segment *cd* share a point (touching counts).

    What Shapely's ``LineString.intersects`` answers for two two-point lines
    in TeraSim.
    """
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    return (
        (o1 == 0.0 and _on_segment(a, b, c))
        or (o2 == 0.0 and _on_segment(a, b, d))
        or (o3 == 0.0 and _on_segment(c, d, a))
        or (o4 == 0.0 and _on_segment(c, d, b))
    )


def trajectories_intersect(
    trajectory1: list[PredictedState],
    footprint1: AgentFootprint,
    trajectory2: list[PredictedState],
    footprint2: AgentFootprint,
    buffer: float = 0.0,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> bool:
    """Whether two trajectories on a common clock conflict.

    False outright when they start more than *distance_threshold* apart.
    Otherwise they conflict when :func:`collision_time` finds an overlap, or
    when the paths cross within the same step (step *i* of one against step
    *i* of the other: two agents passing the same spot at different times do
    not conflict).
    """
    if not trajectory1 or not trajectory2:
        return False
    dx = trajectory2[0].x - trajectory1[0].x
    dy = trajectory2[0].y - trajectory1[0].y
    if dx * dx + dy * dy > distance_threshold * distance_threshold:
        return False
    if (
        collision_time(trajectory1, footprint1, trajectory2, footprint2, buffer)
        is not None
    ):
        return True
    for i in range(1, min(len(trajectory1), len(trajectory2))):
        if segments_intersect(
            (trajectory1[i - 1].x, trajectory1[i - 1].y),
            (trajectory1[i].x, trajectory1[i].y),
            (trajectory2[i - 1].x, trajectory2[i - 1].y),
            (trajectory2[i].x, trajectory2[i].y),
        ):
            return True
    return False
