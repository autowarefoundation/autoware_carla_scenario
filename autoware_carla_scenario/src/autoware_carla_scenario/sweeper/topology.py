"""Map questions the routing graph does not answer: junctions, crossings, crosswalks.

The sweeper's constraints and bindings read the Lanelet2 routing graph, which
knows which lane follows which and which a vehicle may change into.  A logical
scenario also asks things the graph has no edge for:

* what *kind* of junction a lanelet belongs to -- a T-junction or a crossroad,
  told apart by how many roads (arms) meet there;
* which lane crosses the ego's path through a junction, and from which side;
* where a crosswalk lies across the ego's lane;
* whether a lane is straight or bends.

They are answered geometrically here, once per map: the per-junction answers
are cached against the map object, because a sweep asks them of every lanelet.

The junction clustering and the arm count follow CodSceneClassifier's own
(``utils/lanelet_util.py``): junction lanelets are grouped, the lanes entering
and leaving the group give one outward bearing each, and bearings within 30
degrees are one arm.  3 arms is a T-junction and 4 a crossroad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import lanelet2.core
import lanelet2.geometry

JunctionType = Literal["T-junction", "crossroad"]
RoadShape = Literal["straight", "curved_left", "curved_right"]

#: Bearings closer than this are one arm of a junction.
ARM_BIN_DEG = 30.0
#: Curvature above which a lane bends, in 1/m (CodSceneClassifier's threshold).
CURVATURE_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def _points(lanelet: Any) -> list[Any]:
    return list(lanelet2.geometry.to2D(lanelet.centerline))


def _heading(points: list[Any], index: int) -> float:
    a = points[max(index - 1, 0)]
    b = points[min(index + 1, len(points) - 1)]
    return math.atan2(b.y - a.y, b.x - a.x)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _is_junction(lanelet: Any) -> bool:
    return "turn_direction" in lanelet.attributes


def _subtype(lanelet: Any) -> str:
    return str(lanelet.attributes["subtype"]) if "subtype" in lanelet.attributes else ""


def _nearby(lanelet_map: Any, lanelet: Any, margin: float = 0.0) -> list[Any]:
    """Lanelets whose bounding box meets *lanelet*'s, grown by *margin*."""
    box = lanelet2.geometry.boundingBox2d(lanelet)
    grown = lanelet2.core.BoundingBox2d(
        lanelet2.core.BasicPoint2d(box.min.x - margin, box.min.y - margin),
        lanelet2.core.BasicPoint2d(box.max.x + margin, box.max.y + margin),
    )
    return [ll for ll in lanelet_map.laneletLayer.search(grown) if ll.id != lanelet.id]


def _segment_intersection(
    a: list[Any], b: list[Any]
) -> Optional[tuple[float, float, float]]:
    """The first crossing of polyline *b* along polyline *a*.

    Returns ``(arc length along a, x, y)``, or ``None`` when they do not cross.
    """
    travelled = 0.0
    for i in range(len(a) - 1):
        p, q = a[i], a[i + 1]
        rx, ry = q.x - p.x, q.y - p.y
        best: Optional[float] = None
        for j in range(len(b) - 1):
            u, v = b[j], b[j + 1]
            sx, sy = v.x - u.x, v.y - u.y
            denom = rx * sy - ry * sx
            if abs(denom) < 1e-12:
                continue
            t = ((u.x - p.x) * sy - (u.y - p.y) * sx) / denom
            w = ((u.x - p.x) * ry - (u.y - p.y) * rx) / denom
            if 0.0 <= t <= 1.0 and 0.0 <= w <= 1.0 and (best is None or t < best):
                best = t
        length = math.hypot(rx, ry)
        if best is not None:
            return travelled + best * length, p.x + best * rx, p.y + best * ry
        travelled += length
    return None


# ---------------------------------------------------------------------------
# Per-map cache
# ---------------------------------------------------------------------------


@dataclass
class _MapFacts:
    """What is worked out once per map."""

    #: Junction lanelet id -> id of the junction (cluster) it belongs to.
    junction_of: dict[int, int]
    #: Junction id -> its type, when it is one the classifier would name.
    junction_type: dict[int, Optional[JunctionType]]
    #: Lanelet id -> crosswalks across it: ``(arc length, crosswalk id, x, y)``,
    #: nearest the lanelet's start first.
    crosswalks_across: dict[int, list[tuple[float, int, float, float]]]


_FACTS: dict[int, tuple[Any, _MapFacts]] = {}


def _facts(lanelet_map: Any, routing_graph: Any) -> _MapFacts:
    cached = _FACTS.get(id(lanelet_map))
    if cached is not None and cached[0] is lanelet_map:
        return cached[1]
    facts = _build_facts(lanelet_map, routing_graph)
    _FACTS[id(lanelet_map)] = (lanelet_map, facts)
    return facts


def _build_facts(lanelet_map: Any, routing_graph: Any) -> _MapFacts:
    junctions = [ll for ll in lanelet_map.laneletLayer if _is_junction(ll)]
    parent = {ll.id: ll.id for ll in junctions}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    by_id = {ll.id: ll for ll in junctions}
    # An `intersection_area` tag names the junction outright where a map has it.
    by_area: dict[str, int] = {}
    for ll in junctions:
        if "intersection_area" in ll.attributes:
            area = str(ll.attributes["intersection_area"])
            if area in by_area:
                union(ll.id, by_area[area])
            else:
                by_area[area] = ll.id
    # Otherwise: lanelets that overlap, follow one another, leave one approach
    # lane or reach one exit lane are the same junction.
    for ll in junctions:
        for other in _nearby(lanelet_map, ll):
            if other.id in by_id and lanelet2.geometry.overlaps2d(ll, other):
                union(ll.id, other.id)
        for prev in routing_graph.previous(ll):
            if prev.id in by_id:
                union(ll.id, prev.id)
            for sibling in routing_graph.following(prev):
                if sibling.id in by_id:
                    union(ll.id, sibling.id)
        for nxt in routing_graph.following(ll):
            for sibling in routing_graph.previous(nxt):
                if sibling.id in by_id:
                    union(ll.id, sibling.id)

    clusters: dict[int, list[Any]] = {}
    for ll in junctions:
        clusters.setdefault(find(ll.id), []).append(ll)
    junction_type = {
        root: _classify(members, routing_graph) for root, members in clusters.items()
    }
    across: dict[int, list[tuple[float, int, float, float]]] = {}
    for crosswalk in lanelet_map.laneletLayer:
        if _subtype(crosswalk) != "crosswalk":
            continue
        cw_points = _points(crosswalk)
        for road in _nearby(lanelet_map, crosswalk):
            if _subtype(road) == "crosswalk" or not lanelet2.geometry.overlaps2d(
                crosswalk, road
            ):
                continue
            hit = _segment_intersection(_points(road), cw_points)
            if hit is not None:
                across.setdefault(road.id, []).append(
                    (hit[0], crosswalk.id, hit[1], hit[2])
                )
    for hits in across.values():
        hits.sort()
    return _MapFacts(
        junction_of={ll.id: find(ll.id) for ll in junctions},
        junction_type=junction_type,
        crosswalks_across=across,
    )


def _classify(members: list[Any], routing_graph: Any) -> Optional[JunctionType]:
    """Count the arms of one junction: the outward bearing of every road at it."""
    ids = {ll.id for ll in members}
    bearings: list[float] = []
    for ll in members:
        for prev in routing_graph.previous(ll):
            if prev.id not in ids:
                points = _points(prev)
                bearings.append(_wrap(_heading(points, len(points) - 1) + math.pi))
        for nxt in routing_graph.following(ll):
            if nxt.id not in ids:
                bearings.append(_heading(_points(nxt), 0))
    arms = _count_arms(bearings)
    return "T-junction" if arms == 3 else "crossroad" if arms == 4 else None


def _count_arms(bearings: list[float]) -> int:
    """How many groups the bearings form, ``ARM_BIN_DEG`` apart (wrapping)."""
    if not bearings:
        return 0
    ordered = sorted(b % (2 * math.pi) for b in bearings)
    gap = math.radians(ARM_BIN_DEG)
    groups = 1
    for a, b in zip(ordered, ordered[1:]):
        if b - a > gap:
            groups += 1
    if groups > 1 and (ordered[0] + 2 * math.pi) - ordered[-1] <= gap:
        groups -= 1
    return groups


# ---------------------------------------------------------------------------
# Public questions
# ---------------------------------------------------------------------------


def junction_type_of(
    lanelet: Any, lanelet_map: Any, routing_graph: Any
) -> Optional[JunctionType]:
    """The type of the junction *lanelet* lies in, or ``None``."""
    facts = _facts(lanelet_map, routing_graph)
    root = facts.junction_of.get(lanelet.id)
    return None if root is None else facts.junction_type[root]


def road_shape(lanelet: Any) -> RoadShape:
    """Whether *lanelet*'s centreline is straight or bends left or right.

    The 90th-percentile |curvature| against :data:`CURVATURE_THRESHOLD`, as the
    classifier labels a scene; the bend's direction is the sign of the
    curvature where it is strongest.
    """
    points = _points(lanelet)
    signed: list[float] = []
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        ab, bc, ca = (
            math.hypot(b.x - a.x, b.y - a.y),
            math.hypot(c.x - b.x, c.y - b.y),
            math.hypot(a.x - c.x, a.y - c.y),
        )
        if ab * bc * ca < 1e-9:
            continue
        cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        signed.append(2.0 * cross / (ab * bc * ca))
    if not signed:
        return "straight"
    magnitudes = sorted(abs(k) for k in signed)
    p90 = magnitudes[min(len(magnitudes) - 1, int(0.9 * len(magnitudes)))]
    if p90 <= CURVATURE_THRESHOLD:
        return "straight"
    strongest = max(signed, key=abs)
    # Positive curvature turns anticlockwise: to the left.
    return "curved_left" if strongest > 0 else "curved_right"


def ego_junction_lanelet(lanelet: Any, routing_graph: Any) -> Optional[Any]:
    """The junction lanelet the ego drives through from *lanelet*.

    *lanelet* itself when it is one; else the one following it, straight on
    where there is a choice (lowest id among equals).
    """
    if _is_junction(lanelet):
        return lanelet
    following = sorted(routing_graph.following(lanelet), key=lambda ll: ll.id)
    junctions = [ll for ll in following if _is_junction(ll)]
    straight = [
        ll for ll in junctions if str(ll.attributes["turn_direction"]) == "straight"
    ]
    return (straight or junctions or [None])[0]


def crossing_lanelet(
    lanelet: Any,
    lanelet_map: Any,
    routing_graph: Any,
    side: str,
) -> Optional[Any]:
    """The junction lanelet that crosses the ego's path from *side*.

    The ego's path is :func:`ego_junction_lanelet`.  Among the junction
    lanelets crossing it whose start lies on *side* of it, straight-on ones
    are preferred, then the crossing nearest the ego.
    """
    path = ego_junction_lanelet(lanelet, routing_graph)
    if path is None:
        return None
    points = _points(path)
    found: list[tuple[int, float, int, Any]] = []
    for other in _nearby(lanelet_map, path):
        if not _is_junction(other):
            continue
        other_points = _points(other)
        hit = _segment_intersection(points, other_points)
        if hit is None:
            continue
        along, x, y = hit
        # Which side of the ego's path the crossing lane comes from.
        index = min(
            range(len(points)),
            key=lambda i: (points[i].x - x) ** 2 + (points[i].y - y) ** 2,
        )
        h = _heading(points, index)
        start = other_points[0]
        cross = math.cos(h) * (start.y - y) - math.sin(h) * (start.x - x)
        if (cross > 0) != (side == "left"):
            continue
        # It has to come across, not merge in alongside.
        relative = abs(_wrap(_heading(other_points, 0) - h))
        if relative < math.radians(30) or relative > math.radians(150):
            continue
        straight = str(other.attributes["turn_direction"]) == "straight"
        found.append((0 if straight else 1, along, other.id, other))
    return min(found, key=lambda f: f[:3])[3] if found else None


def crossing_approach(
    lanelet: Any,
    lanelet_map: Any,
    routing_graph: Any,
    side: str,
    approach_m: float,
) -> tuple[int, float]:
    """Where a vehicle crossing from *side* starts: *approach_m* before it.

    Raises:
        ValueError: If nothing crosses the ego's path from that side.
    """
    crossing = crossing_lanelet(lanelet, lanelet_map, routing_graph, side)
    if crossing is None:
        raise ValueError(
            f"nothing crosses the path from lanelet {lanelet.id} from the {side}."
        )
    previous = routing_graph.previous(crossing)
    if not previous:
        return crossing.id, 0.0
    start = previous[0]
    return start.id, max(0.0, lanelet2.geometry.length2d(start) - approach_m)


@dataclass(frozen=True)
class CrosswalkStart:
    """Where a pedestrian starts across a crosswalk."""

    lanelet_id: int
    s: float
    #: Relative to the crosswalk lanelet's direction: 0, or pi to walk it backwards.
    heading: float


def crosswalk_start(
    lanelet: Any,
    lanelet_map: Any,
    routing_graph: Any,
    side: str,
    search_distance: float = 60.0,
    margin: float = 0.5,
) -> CrosswalkStart:
    """The first crosswalk across the ego's lane ahead, from its *side* kerb.

    The lane is walked forward from the pick's start, as ``route_offset``
    walks it, for *search_distance* metres.

    Raises:
        ValueError: If no crosswalk crosses the lane within that distance.
    """
    facts = _facts(lanelet_map, routing_graph)
    current, travelled = lanelet, 0.0
    while current is not None and travelled <= search_distance:
        hits = facts.crosswalks_across.get(current.id)
        if hits:
            _, crosswalk_id, x, y = hits[0]
            crosswalk = lanelet_map.laneletLayer[crosswalk_id]
            points = _points(current)
            cw_points = _points(crosswalk)
            index = min(
                range(len(points)),
                key=lambda i: (points[i].x - x) ** 2 + (points[i].y - y) ** 2,
            )
            h = _heading(points, index)
            first = cw_points[0]
            first_left = math.cos(h) * (first.y - y) - math.sin(h) * (first.x - x) > 0
            length = lanelet2.geometry.length2d(crosswalk)
            if first_left == (side == "left"):
                return CrosswalkStart(crosswalk.id, min(margin, length), 0.0)
            return CrosswalkStart(crosswalk.id, max(0.0, length - margin), math.pi)
        travelled += lanelet2.geometry.length2d(current)
        following = sorted(routing_graph.following(current), key=lambda ll: ll.id)
        current = following[0] if following else None
    raise ValueError(
        f"no crosswalk crosses the lane within {search_distance} m of lanelet "
        f"{lanelet.id}."
    )


__all__ = [
    "CrosswalkStart",
    "JunctionType",
    "RoadShape",
    "crossing_approach",
    "crossing_lanelet",
    "crosswalk_start",
    "ego_junction_lanelet",
    "junction_type_of",
    "road_shape",
]
