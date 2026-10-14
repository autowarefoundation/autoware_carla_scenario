"""Synthetic Lanelet2 maps of multi-lane roads, for the vehicle-ahead tests.

A road follows a reference path; its lane boundaries are offset from it
(positive to the left, in the Lanelet2 frame), and it is cut into lanelets at
the same places on every lane, so the lanelets beside each other share their
boundary and are neighbours in the routing graph.  On a curve the lanes'
centrelines are concentric arcs: the outer lanelet is longer than the inner
one beside it, and the same ``s`` lies at different places on them, as on
CARLA Town04's highway.

The Lanelet2 and CARLA frames are aligned (no MGRS offset): CARLA's ``y`` is
the Lanelet2 ``-y``, and its yaw the Lanelet2 heading negated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

# autoware_lanelet2_extension_python must be imported before lanelet2.
import autoware_lanelet2_extension_python.projection  # noqa: F401
import lanelet2.routing
import lanelet2.traffic_rules
import pytest
from lanelet2.core import AttributeMap, Lanelet, LaneletMap, LineString3d, Point3d

from autoware_carla_scenario.coordinate.map_manager import MapManager

_ROAD = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes"}

#: A path: arc length -> (x, y, heading) in the Lanelet2 frame.
Path = Callable[[float], tuple[float, float, float]]


def straight(length: float, x0: float = 0.0, y0: float = 0.0) -> tuple[Path, float]:
    """A straight path along +x from ``(x0, y0)``."""
    return (lambda u: (x0 + u, y0, 0.0)), length


def bend(
    straight_before: float, radius: float, angle_deg: float, straight_after: float
) -> tuple[Path, float]:
    """Straight along +x from the origin, then a left turn of *radius* through
    *angle_deg*, then straight again."""
    arc = radius * math.radians(angle_deg)
    end_heading = math.radians(angle_deg)
    end_x = straight_before + radius * math.sin(end_heading)
    end_y = radius * (1.0 - math.cos(end_heading))

    def at(u: float) -> tuple[float, float, float]:
        if u <= straight_before:
            return u, 0.0, 0.0
        if u <= straight_before + arc:
            angle = (u - straight_before) / radius
            return (
                straight_before + radius * math.sin(angle),
                radius * (1.0 - math.cos(angle)),
                angle,
            )
        rest = u - straight_before - arc
        return (
            end_x + rest * math.cos(end_heading),
            end_y + rest * math.sin(end_heading),
            end_heading,
        )

    return at, straight_before + arc + straight_after


@dataclass
class LaneMap:
    """A built map: the lanelets by (lane, segment), and the routing graph."""

    map: Any
    graph: Any
    ids: dict[tuple[str, int], int] = field(default_factory=dict)

    def id(self, lane: str, segment: int = 0) -> int:
        return self.ids[(lane, segment)]


def build_road(
    path: tuple[Path, float],
    cuts: list[float],
    boundaries: list[float],
    lanes: dict[str, tuple[int, bool]],
    *,
    step: float = 0.25,
) -> LaneMap:
    """A road along *path*, cut into lanelets at *cuts* (arc lengths of the
    reference path, from 0 to its length).

    *boundaries* are the lane lines' offsets from the reference, right to
    left; *lanes* names each lane by the index of its right boundary, and
    whether it runs against the path (``True``: the opposite direction).
    """
    along, length = path
    next_id = iter(range(100_000, 10_000_000))
    lanelet_map = LaneletMap()
    samples = sorted(
        {round(i * step, 9) for i in range(int(length / step) + 1)}
        | {round(c, 9) for c in cuts}
    )
    samples = [u for u in samples if u <= length + 1e-9]
    points: dict[tuple[int, float], Any] = {}

    def point(b: int, u: float) -> Any:
        key = (b, u)
        if key not in points:
            x, y, heading = along(u)
            d = boundaries[b]
            points[key] = Point3d(
                next(next_id), x - d * math.sin(heading), y + d * math.cos(heading), 0.0
            )
        return points[key]

    lines: dict[tuple[int, int], Any] = {}

    def line(b: int, segment: int) -> Any:
        key = (b, segment)
        if key not in lines:
            low, high = cuts[segment], cuts[segment + 1]
            us = [u for u in samples if low - 1e-9 <= u <= high + 1e-9]
            lines[key] = LineString3d(
                next(next_id),
                [point(b, u) for u in us],
                AttributeMap({"type": "line_thin", "subtype": "dashed"}),
            )
        return lines[key]

    built = LaneMap(map=lanelet_map, graph=None)
    for segment in range(len(cuts) - 1):
        for name, (right, opposite) in lanes.items():
            lanelet_id = next(next_id)
            if opposite:
                lanelet = Lanelet(
                    lanelet_id,
                    line(right, segment).invert(),
                    line(right + 1, segment).invert(),
                    AttributeMap(_ROAD),
                )
            else:
                lanelet = Lanelet(
                    lanelet_id,
                    line(right + 1, segment),
                    line(right, segment),
                    AttributeMap(_ROAD),
                )
            lanelet_map.add(lanelet)
            built.ids[(name, segment)] = lanelet_id
    rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )
    built.graph = lanelet2.routing.RoutingGraph(lanelet_map, rules)
    return built


def loaded(built: LaneMap) -> Iterator[LaneMap]:
    """Make *built* the map the MapManager holds (frames aligned); restore after."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._lanelet_map = built.map
    manager._routing_graph = built.graph
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    yield built
    MapManager._instance = saved


@pytest.fixture
def no_map() -> Iterator[None]:
    """No map loaded at all."""
    saved = MapManager._instance
    MapManager.reset()
    yield
    MapManager._instance = saved
