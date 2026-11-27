"""Shapes on the ground: a road user's box and the areas of a map, and how far
apart two of them are.

Everything here is in the ground plane, ``(x, y)`` in CARLA's world frame:
a road user's bounding box is an oriented rectangle (:func:`ground_box`), an
area of the map any simple polygon -- a crosswalk's outline, a lane's,
Autoware's intersection area -- convex or not.  :func:`polygon_distance` is
the distance between two of them: ``0`` once they touch, overlap, or one
holds the other, and otherwise the shortest distance between their edges,
which for two polygons that do not meet is always from a corner of one to an
edge of the other.  :func:`polygon_distance_at` says where on the second
polygon's outline that distance is measured to, as an edge and how far along
it, so a value kept per corner -- the height of an area's outline -- can be
read there (:func:`edge_value`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "Box",
    "Point",
    "bounding_circle",
    "box_distance",
    "boxes_meet",
    "cross",
    "edge_value",
    "ground_box",
    "oriented_box",
    "point_in_box",
    "point_in_polygon",
    "point_segment_distance",
    "polygon_distance",
    "polygon_distance_at",
    "polygon_gap",
    "polygon_gap_at",
    "segment_meets_box",
    "segments_cross",
]

#: A point on the ground, ``(x, y)`` in CARLA's world frame.
Point = tuple[float, float]
#: An oriented box on the ground: its four corners, front left, front right,
#: rear right, rear left.
Box = list[Point]

#: How near (m) a point must be to an edge to be on it.
_ON_EDGE_M = 1e-9


def ground_box(actor: "carla.Actor", transform: "carla.Transform") -> Box:
    """*actor*'s bounding box on the ground, in the world frame, *transform*
    being where the actor is."""
    heading = math.radians(transform.rotation.yaw)
    box_yaw = 0.0
    extent_x, extent_y = 0.0, 0.0
    offset_x, offset_y = 0.0, 0.0
    # Not getattr() with a default, which Codon lacks (docs/typecheck.md).
    try:
        box = actor.bounding_box
        extent_x, extent_y = float(box.extent.x), float(box.extent.y)
        offset_x, offset_y = float(box.location.x), float(box.location.y)
        box_yaw = math.radians(box.rotation.yaw)
    except AttributeError:
        pass
    return oriented_box(
        transform.location.x,
        transform.location.y,
        heading,
        extent_x,
        extent_y,
        offset_x,
        offset_y,
        box_yaw,
    )


def oriented_box(
    x: float,
    y: float,
    heading: float,
    extent_x: float,
    extent_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    box_yaw: float = 0.0,
) -> Box:
    """The corners of a box on the ground (front left, front right, rear
    right, rear left): a road user's at *x*, *y* (its origin) heading
    *heading* (radians), *extent_x* and *extent_y* its half length and half
    width, *offset_x*, *offset_y* the box's centre in the road user's frame
    (turned with it) and *box_yaw* (radians) the box's own turn on top of
    *heading*.  :func:`ground_box` reads them off a CARLA actor; the viewer
    (``ui.drive_metrics.box``) off a trajectory log."""
    yaw = heading + box_yaw
    # CARLA's frame is left-handed: +y is to the right of +x.
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-forward[1], forward[0])
    centre = (
        x + offset_x * math.cos(heading) - offset_y * math.sin(heading),
        y + offset_x * math.sin(heading) + offset_y * math.cos(heading),
    )

    def corner(along: float, across: float) -> Point:
        return (
            centre[0] + along * forward[0] + across * right[0],
            centre[1] + along * forward[1] + across * right[1],
        )

    return [
        corner(extent_x, -extent_y),
        corner(extent_x, extent_y),
        corner(-extent_x, extent_y),
        corner(-extent_x, -extent_y),
    ]


def _segment_nearest(p: Point, a: Point, b: Point) -> tuple[float, float]:
    """Metres from *p* to the segment *ab*, and how far along it (``0`` at
    *a*, ``1`` at *b*) the nearest point of it is."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0
    if length2 > 0.0:
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return (math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy)), t)


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Metres from *p* to the segment *ab* (to *a* when the two ends meet)."""
    return _segment_nearest(p, a, b)[0]


def cross(o: Point, a: Point, b: Point) -> float:
    """The z of ``(a - o) x (b - o)``: positive when *b* is to the left of
    *o* -> *a* in a right-handed frame."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Whether segment *ab* meets segment *cd*, touching included."""
    if cross(c, d, a) * cross(c, d, b) < 0.0 and cross(a, b, c) * cross(a, b, d) < 0.0:
        return True
    return (
        point_segment_distance(a, c, d) <= _ON_EDGE_M
        or point_segment_distance(b, c, d) <= _ON_EDGE_M
        or point_segment_distance(c, a, b) <= _ON_EDGE_M
        or point_segment_distance(d, a, b) <= _ON_EDGE_M
    )


def point_in_box(point: Point, box: Box) -> bool:
    """Whether *point* is in the convex *box* or on its edge.

    A box of no area (a road user whose extent read as zero: a point, or a
    segment) holds only the points on it (#1997): every side's cross product
    is zero for it, which would otherwise hold every point on the line.
    """
    sign = 0.0
    for i in range(len(box)):
        side = cross(box[i], box[(i + 1) % len(box)], point)
        if abs(side) <= 1e-12:
            continue
        if sign == 0.0:
            sign = side
        elif (side > 0.0) != (sign > 0.0):
            return False
    if sign == 0.0:
        # Every side is in line with the point: the box is a point or a
        # segment (or the point is a corner of a box that is); on it or not.
        for i in range(len(box)):
            if point_segment_distance(point, box[i], box[(i + 1) % len(box)]) <= 1e-9:
                return True
        return False
    return True


def _box_edges(box: Box) -> list[tuple[Point, Point]]:
    return [(box[i], box[(i + 1) % len(box)]) for i in range(len(box))]


def segment_meets_box(a: Point, b: Point, box: Box) -> bool:
    """Whether segment *ab* meets the convex *box*, touching included."""
    if point_in_box(a, box) or point_in_box(b, box):
        return True
    for c, d in _box_edges(box):
        if segments_cross(a, b, c, d):
            return True
    return False


def boxes_meet(one: Box, other: Box) -> bool:
    """Whether two convex boxes on the ground touch or overlap."""
    for a, b in _box_edges(one):
        if segment_meets_box(a, b, other):
            return True
    # One wholly inside the other.
    return point_in_box(other[0], one)


def box_distance(one: Box, other: Box) -> float:
    """Metres between two convex boxes on the ground; ``0.0`` when they
    meet.  A box of no area is the point or segment it is (#1997)."""
    if boxes_meet(one, other):
        return 0.0
    one_edges, other_edges = _box_edges(one), _box_edges(other)
    best = math.inf
    for p in one:
        for a, b in other_edges:
            best = min(best, point_segment_distance(p, a, b))
    for p in other:
        for a, b in one_edges:
            best = min(best, point_segment_distance(p, a, b))
    return best


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Whether *point* is in the simple *polygon* or on its edge.

    By the crossings of a ray from *point* towards +x (even-odd), so a
    polygon that is not convex -- a crosswalk drawn along a curve, a lane
    through a junction -- is answered as drawn.  The order of the corners,
    either way round, does not matter; a polygon of fewer than three corners
    holds only the points on it.
    """
    count = len(polygon)
    if count == 0:
        return False
    for i in range(count):
        if (
            point_segment_distance(point, polygon[i], polygon[(i + 1) % count])
            <= _ON_EDGE_M
        ):
            return True
    if count < 3:
        return False
    x, y = point[0], point[1]
    inside = False
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if (yi > y) != (yj > y):
            at = xj + (y - yj) * (xi - xj) / (yi - yj)
            if x < at:
                inside = not inside
        j = i
    return inside


def _edges_meet(one: list[Point], other: list[Point]) -> bool:
    count, other_count = len(one), len(other)
    for i in range(count):
        a, b = one[i], one[(i + 1) % count]
        for j in range(other_count):
            if segments_cross(a, b, other[j], other[(j + 1) % other_count]):
                return True
    return False


def polygon_gap_at(one: list[Point], other: list[Point]) -> tuple[float, int, float]:
    """:func:`polygon_gap`, and where on *other*'s outline it is measured to:
    ``(distance, edge, along)``, the point *along* (``0`` to ``1``) the edge
    from *other*'s corner *edge* to the next.  ``(inf, -1, 0)`` when either
    has no corners."""
    best = math.inf
    edge = -1
    along = 0.0
    count, other_count = len(one), len(other)
    if count == 0:
        return (best, edge, along)
    for p in one:
        for j in range(other_count):
            distance, t = _segment_nearest(p, other[j], other[(j + 1) % other_count])
            if distance < best:
                best, edge, along = distance, j, t
    for j in range(other_count):
        for i in range(count):
            distance = _segment_nearest(other[j], one[i], one[(i + 1) % count])[0]
            if distance < best:
                best, edge, along = distance, j, 0.0
    return (best, edge, along)


def polygon_gap(one: list[Point], other: list[Point]) -> float:
    """The shortest distance from a corner of either polygon to an edge of the
    other: the distance between two polygons known not to meet (their
    bounding circles apart, say), without asking whether they do.  ``inf``
    when either has no corners."""
    return polygon_gap_at(one, other)[0]


def _meet(one: list[Point], other: list[Point]) -> bool:
    return (
        point_in_polygon(one[0], other)
        or point_in_polygon(other[0], one)
        or _edges_meet(one, other)
    )


def polygon_distance(one: list[Point], other: list[Point]) -> float:
    """Metres between two simple polygons on the ground: ``0.0`` when they
    touch, overlap or one holds the other, otherwise :func:`polygon_gap`.
    ``inf`` when either has no corners."""
    if not one or not other:
        return math.inf
    if _meet(one, other):
        return 0.0
    return polygon_gap(one, other)


def polygon_distance_at(
    one: list[Point], other: list[Point], centre: Point
) -> tuple[float, int, float]:
    """:func:`polygon_distance`, and where on *other*'s outline it is read:
    ``(distance, edge, along)`` as :func:`polygon_gap_at` says it.  Where the
    two meet, the distance is ``0`` and the point is the one of *other*'s
    outline nearest *centre* -- the middle of *one*, say, over the area.
    ``(inf, -1, 0)`` when either has no corners."""
    if not one or not other:
        return (math.inf, -1, 0.0)
    if not _meet(one, other):
        return polygon_gap_at(one, other)
    best = math.inf
    edge = 0
    along = 0.0
    count = len(other)
    for j in range(count):
        distance, t = _segment_nearest(centre, other[j], other[(j + 1) % count])
        if distance < best:
            best, edge, along = distance, j, t
    return (0.0, edge, along)


def edge_value(values: list[float], edge: int, along: float) -> float:
    """A value kept per corner of a polygon (*values*, one per corner, in
    order) at the point *along* (``0`` to ``1``) the edge from corner *edge*
    to the next, linearly between the two."""
    a = values[edge]
    b = values[(edge + 1) % len(values)]
    return a + (b - a) * along


def bounding_circle(polygon: list[Point]) -> tuple[float, float, float]:
    """A circle ``(x, y, radius)`` holding every corner of *polygon*: round
    the middle of its corners' extent, so no point of it is further from the
    centre than the radius.  ``(0, 0, 0)`` for one with no corners."""
    if not polygon:
        return (0.0, 0.0, 0.0)
    low_x, high_x = polygon[0][0], polygon[0][0]
    low_y, high_y = polygon[0][1], polygon[0][1]
    for p in polygon:
        low_x, high_x = min(low_x, p[0]), max(high_x, p[0])
        low_y, high_y = min(low_y, p[1]), max(high_y, p[1])
    cx, cy = 0.5 * (low_x + high_x), 0.5 * (low_y + high_y)
    radius = 0.0
    for p in polygon:
        radius = max(radius, math.hypot(p[0] - cx, p[1] - cy))
    return (cx, cy, radius)
