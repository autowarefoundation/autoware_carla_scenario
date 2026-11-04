"""Which traffic light regulatory elements a CARLA light shows, on the Lanelet2 map.

CARLA puts a light's stop waypoints where its own junction starts, which is
not where a Lanelet2 map draws the stop line: on Town10HD_Opt's
carla-ue5-maps map, 0.7 m to 6.7 m before it.  So a light is not matched to
an element by how near its waypoint is to the element's stop line.  The
match goes by the lanes instead: the lanelets the waypoint is on, then the
lanelets ahead of them, up to the first that carry a traffic light element
(:func:`elements_ahead`).  This is how ``carla_driver_interface.hdmap``
finds a light on a Lanelet2 map (``rules_ahead``).

Both readers of lights match the same way: a run's T4Scene
(:mod:`..t4_scene.convert`), and the action, the condition and the signal
controllers (:func:`lights_of_element`, through
:func:`~..coordinate.traffic_light.find_traffic_lights_for_lanelet2_id`).
A lit lanelet in a scene is therefore the light a scenario sets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

if TYPE_CHECKING:
    import lanelet2
    import typesafe_carla.carla as carla

__all__ = [
    "JOIN_M",
    "LIGHT_SEARCH_M",
    "LightLanes",
    "elements_ahead",
    "light_lanes",
    "lights_of_element",
]

#: How far (m) of lanelets past the one a light's stop waypoint is on its
#: traffic light regulatory elements are looked for; and how near (m) a
#: lanelet's start is to another's end to follow it.  As
#: ``carla_driver_interface.hdmap`` finds a light on a Lanelet2 map: a policy
#: that took its lights from there is shown the same lanelets lit.
LIGHT_SEARCH_M = 50.0
JOIN_M = 0.3


class LightLanes(NamedTuple):
    """A Lanelet2 map's lanelets, as far as finding a light's lanelets takes:
    each one's outline, length, traffic light regulatory elements and the
    lanelets that start where it ends.  Map frame."""

    ids: tuple[int, ...]
    outlines: tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], ...]
    #: ``(L, 4)``: each outline's min x, min y, max x, max y.
    boxes: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    lengths: Mapping[int, float]
    elements: Mapping[int, tuple[int, ...]]
    successors: Mapping[int, tuple[int, ...]]


def light_lanes(lanelet_map: lanelet2.core.LaneletMap) -> LightLanes:
    """The lanelets of *lanelet_map*, read for :func:`elements_ahead`."""
    ids: list[int] = []
    outlines: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
    lengths: dict[int, float] = {}
    elements: dict[int, tuple[int, ...]] = {}
    starts: dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]] = {}
    ends: dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]] = {}
    for lanelet in lanelet_map.laneletLayer:
        left = np.array([(p.x, p.y) for p in lanelet.leftBound], dtype=np.float64)
        right = np.array([(p.x, p.y) for p in lanelet.rightBound], dtype=np.float64)
        if len(left) < 2 or len(right) < 2:
            continue
        ids.append(int(lanelet.id))
        outlines.append(np.vstack([left, right[::-1]]))
        lengths[ids[-1]] = 0.5 * float(
            np.linalg.norm(np.diff(left, axis=0), axis=1).sum()
            + np.linalg.norm(np.diff(right, axis=0), axis=1).sum()
        )
        elements[ids[-1]] = tuple(
            int(element.id)
            for element in lanelet.regulatoryElements
            if "subtype" in element.attributes
            and element.attributes["subtype"] == "traffic_light"
        )
        starts[ids[-1]] = (left[0] + right[0]) / 2
        ends[ids[-1]] = (left[-1] + right[-1]) / 2
    start_ids = list(starts)
    start_points = np.array([starts[i] for i in start_ids]).reshape(-1, 2)
    successors = {
        lanelet_id: tuple(
            start_ids[k]
            for k in np.flatnonzero(
                np.hypot(start_points[:, 0] - end[0], start_points[:, 1] - end[1])
                < JOIN_M
            )
            if start_ids[k] != lanelet_id
        )
        for lanelet_id, end in ends.items()
    }
    boxes = np.array(
        [[*o.min(axis=0), *o.max(axis=0)] for o in outlines], dtype=np.float64
    ).reshape(-1, 4)
    return LightLanes(tuple(ids), tuple(outlines), boxes, lengths, elements, successors)


def _inside(
    point: np.ndarray[tuple[int], np.dtype[np.float64]],
    outline: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> bool:
    """Whether *point* is inside the polygon *outline* (even-odd rule)."""
    x, y = float(point[0]), float(point[1])
    xs, ys = outline[:, 0], outline[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (ys > y) != (ys2 > y)
    # Only an edge that crosses the point's y counts, and its ends differ in y:
    # the others divide by 1 instead of 0 (no warning to silence).
    at = xs + (y - ys) * (xs2 - xs) / np.where(crosses, ys2 - ys, 1.0)
    return bool(np.count_nonzero(crosses & (x < at)) % 2)


def elements_ahead(
    lanes: LightLanes, point: np.ndarray[tuple[int], np.dtype[np.float64]]
) -> list[int]:
    """The traffic light regulatory elements of the first lanelets on or ahead
    of the lanelets *point* (map frame) is on that carry one: each branch
    stops at its first such lanelet, or once :data:`LIGHT_SEARCH_M` of
    lanelets lie behind it (``carla_driver_interface.hdmap``'s
    ``rules_ahead``)."""
    b = lanes.boxes
    near = np.flatnonzero(
        (b[:, 0] <= point[0])
        & (point[0] <= b[:, 2])
        & (b[:, 1] <= point[1])
        & (point[1] <= b[:, 3])
    )
    start = {lanes.ids[i] for i in near if _inside(point, lanes.outlines[i])}
    found: dict[int, None] = {}
    queue = [(lanelet_id, 0.0) for lanelet_id in sorted(start)]
    seen: set[int] = set()
    while queue:
        lanelet_id, travelled = queue.pop(0)
        if lanelet_id in seen:
            continue
        seen.add(lanelet_id)
        if lanes.elements[lanelet_id]:
            for element in lanes.elements[lanelet_id]:
                found[element] = None
            continue
        ahead = (
            travelled if lanelet_id in start else travelled + lanes.lengths[lanelet_id]
        )
        if ahead <= LIGHT_SEARCH_M:
            queue += [(nxt, ahead) for nxt in lanes.successors[lanelet_id]]
    return list(found)


#: The lanes read of a Lanelet2 map, with the map they were read from
#: (:func:`_lanes_of`): one map at a time, the run's.  The map is held and
#: compared with ``is``, as :func:`..coordinate.transform._ref_lines` does:
#: keyed by its ``id`` alone, a new map allocated at a freed one's address
#: (a scenario queue reloads the map at each start) would be handed the old
#: map's lanes.
_LANES: "Optional[tuple[lanelet2.core.LaneletMap, LightLanes]]" = None


def _lanes_of(lanelet_map: lanelet2.core.LaneletMap) -> LightLanes:
    """:func:`light_lanes` of *lanelet_map*, read once."""
    global _LANES

    kept = _LANES
    if kept is not None and kept[0] is lanelet_map:
        return kept[1]
    lanes = light_lanes(lanelet_map)
    _LANES = (lanelet_map, lanes)
    return lanes


def lights_of_element(
    world: carla.World,
    lanelet_map: lanelet2.core.LaneletMap,
    offset: tuple[float, float],
    lanelet2_tl_id: int,
) -> list[carla.Actor]:
    """The CARLA lights of *world* that show the traffic light regulatory
    element *lanelet2_tl_id* of *lanelet_map*: those with a stop waypoint on a
    lanelet it governs, or on one before it (:func:`elements_ahead`).
    *offset* is the map's from CARLA's frame (the projector's)."""
    lanes = _lanes_of(lanelet_map)
    offset_x, offset_y = offset
    found: list[carla.Actor] = []
    for actor in world.get_actors().filter("traffic.traffic_light*"):
        for waypoint in actor.get_stop_waypoints():
            location = waypoint.transform.location
            # CARLA's frame into the map's: the inverse of map_pose_to_carla.
            point = np.array([location.x + offset_x, offset_y - location.y])
            if lanelet2_tl_id in elements_ahead(lanes, point):
                found.append(actor)
                break
    return found
