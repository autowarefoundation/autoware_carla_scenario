"""A small synthetic Lanelet2 map of a crossroads, for the route-search tests.

Right-hand traffic, junction box 20 m square at the origin, lanes 3.5 m wide::

                 N road (x = 0, y 10..110)
                    |  |
    W road  ========+  +======== E road (x 10..60, then 45 deg to the left)
    (x -110..-10)   |  |
                 S road (x = 0, y -110..-10)

* **W road**: eastbound (into the junction) two lanes -- inner ``W_IN1 W_IN2``
  at y = -1.75 and outer ``W_OUT_LANE1 W_OUT_LANE2`` at y = -5.25, which ends
  at the junction -- and westbound (out of it) one lane ``W_BACK2 W_BACK1``
  at y = +1.75.  ``W_IN2`` carries a traffic light with a stop line, and so do
  the junction lanelets leaving it.
* **E road**: eastbound ``E_OUT1`` (straight, 50 m) then ``E_OUT2`` (45 deg
  left on a 40 m radius), and westbound ``E_IN2 E_IN1`` back into the junction.
* **N road** and **S road**: one lane each way; nothing signalised.
* Junction lanelets join every incoming lane end to the three other roads,
  tagged ``turn_direction``; ids ``J<from><to>``.
* Crosswalks across the W leg (x = -15) and the E leg (x = 15).
* ``ONE_WAY``: a separate one-way road far away, for "no opposite lane".
"""

from __future__ import annotations

import math
from typing import Any

# autoware_lanelet2_extension_python must be imported before lanelet2.
import autoware_lanelet2_extension_python.projection  # noqa: F401
import lanelet2.core
import lanelet2.routing
import lanelet2.traffic_rules
from lanelet2.core import AttributeMap, Lanelet, LaneletMap, LineString3d, Point3d

W_IN1, W_IN2 = 101, 102
W_OUT_LANE1, W_OUT_LANE2 = 111, 112
W_BACK2, W_BACK1 = 121, 122  # westbound: W_BACK2 first (x -10..-60)
E_OUT1, E_OUT2 = 201, 202
E_IN2, E_IN1 = 211, 212  # westbound: E_IN2 first (on the arc), then E_IN1
N_OUT, N_IN = 301, 311
S_IN, S_OUT = 401, 411
ONE_WAY1, ONE_WAY2 = 501, 502
CROSSWALK_W, CROSSWALK_E = 901, 902
TRAFFIC_LIGHT = 951

#: Junction lanelets, by (from road, to road).
J = {
    ("W", "E"): 601,  # straight
    ("W", "N"): 602,  # left
    ("W", "S"): 603,  # right
    ("E", "W"): 611,
    ("E", "S"): 612,
    ("E", "N"): 613,
    ("N", "S"): 621,
    ("N", "E"): 622,
    ("N", "W"): 623,
    ("S", "N"): 631,
    ("S", "W"): 632,
    ("S", "E"): 633,
}

_ROAD = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes"}
_HALF = 1.75


class _Builder:
    def __init__(self) -> None:
        self.map = LaneletMap()
        self._next = iter(range(10_000, 10_000_000))
        self._points: dict[tuple[int, int], Any] = {}
        self._lines: dict[tuple[int, ...], Any] = {}

    def point(self, x: float, y: float) -> Any:
        key = (round(x * 100), round(y * 100))
        found = self._points.get(key)
        if found is None:
            found = Point3d(next(self._next), x, y, 0.0)
            self._points[key] = found
        return found

    def line(self, coords: list[tuple[float, float]], subtype: str = "dashed") -> Any:
        points = [self.point(x, y) for x, y in coords]
        ids = tuple(int(p.id) for p in points)
        if ids in self._lines:
            return self._lines[ids]
        if ids[::-1] in self._lines:
            return self._lines[ids[::-1]].invert()
        line = LineString3d(
            next(self._next),
            points,
            AttributeMap({"type": "line_thin", "subtype": subtype}),
        )
        self._lines[ids] = line
        return line

    def lanelet(
        self,
        lanelet_id: int,
        centre: list[tuple[float, float]],
        attributes: dict[str, str] | None = None,
        ends: tuple[float, float] | None = None,
    ) -> Any:
        """A lanelet on *centre*; *ends* are its exact start and end headings."""
        left, right = _offset(centre, _HALF, ends), _offset(centre, -_HALF, ends)
        lanelet = Lanelet(
            lanelet_id,
            self.line(left),
            self.line(right),
            AttributeMap({**_ROAD, **(attributes or {})}),
        )
        self.map.add(lanelet)
        return lanelet


def _normals(
    points: list[tuple[float, float]], ends: tuple[float, float] | None = None
) -> list[tuple[float, float]]:
    out = []
    for i in range(len(points)):
        if ends is not None and i in (0, len(points) - 1):
            heading = ends[0] if i == 0 else ends[1]
            out.append((-math.sin(heading), math.cos(heading)))
            continue
        a = points[max(i - 1, 0)]
        b = points[min(i + 1, len(points) - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        out.append((-dy / norm, dx / norm))
    return out


def _offset(
    points: list[tuple[float, float]],
    d: float,
    ends: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """*points* moved *d* to their left (negative: right)."""
    return [
        (x + nx * d, y + ny * d)
        for (x, y), (nx, ny) in zip(points, _normals(points, ends))
    ]


def _line(a: tuple[float, float], b: tuple[float, float], step: float = 10.0) -> list:
    n = max(1, round(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
    return [
        (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
        for i in range(n + 1)
    ]


def _arc(
    centre: tuple[float, float], radius: float, start: float, end: float, n: int = 12
) -> list[tuple[float, float]]:
    return [
        (
            centre[0] + radius * math.cos(start + (end - start) * i / n),
            centre[1] + radius * math.sin(start + (end - start) * i / n),
        )
        for i in range(n + 1)
    ]


def _bezier(
    a: tuple[float, float], ha: float, b: tuple[float, float], hb: float, n: int = 16
) -> list[tuple[float, float]]:
    d = math.hypot(b[0] - a[0], b[1] - a[1]) / 2.5
    p1 = (a[0] + d * math.cos(ha), a[1] + d * math.sin(ha))
    p2 = (b[0] - d * math.cos(hb), b[1] - d * math.sin(hb))
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append(
            (
                u**3 * a[0]
                + 3 * u * u * t * p1[0]
                + 3 * u * t * t * p2[0]
                + t**3 * b[0],
                u**3 * a[1]
                + 3 * u * u * t * p1[1]
                + 3 * u * t * t * p2[1]
                + t**3 * b[1],
            )
        )
    return out


#: With ``split_path``: a left turn from the N road into the E road drawn as
#: two junction lanelets -- ``SPLIT_FIRST`` from the N road's end to (3, 3),
#: heading -40 deg, then ``SPLIT_SECOND`` on to the E road.  The second one's
#: own heading where it starts is within 45 deg of an eastbound ego's.
SPLIT_FIRST, SPLIT_SECOND = 590, 591


def crossroads(split_path: bool = False) -> Any:
    """The map described in the module docstring (see also ``SPLIT_FIRST``)."""
    b = _Builder()
    # Road reference lines are the centre lines (y = 0 / x = 0).
    # --- W road ---------------------------------------------------------------
    for lid, xa, xb in ((W_IN1, -110.0, -60.0), (W_IN2, -60.0, -10.0)):
        b.lanelet(lid, _line((xa, -_HALF), (xb, -_HALF)))
    for lid, xa, xb in ((W_OUT_LANE1, -110.0, -60.0), (W_OUT_LANE2, -60.0, -10.0)):
        b.lanelet(lid, _line((xa, -3 * _HALF), (xb, -3 * _HALF)))
    for lid, xa, xb in ((W_BACK2, -10.0, -60.0), (W_BACK1, -60.0, -110.0)):
        b.lanelet(lid, _line((xa, _HALF), (xb, _HALF)))
    # --- E road ---------------------------------------------------------------
    b.lanelet(E_OUT1, _line((10.0, -_HALF), (60.0, -_HALF)))
    # 45 degrees to the left around (60, 40): eastbound lane on radius 41.75.
    b.lanelet(
        E_OUT2,
        _arc((60.0, 40.0), 40.0 + _HALF, -math.pi / 2, -math.pi / 4),
        ends=(0.0, math.pi / 4),
    )
    b.lanelet(
        E_IN2,
        _arc((60.0, 40.0), 40.0 - _HALF, -math.pi / 4, -math.pi / 2),
        ends=(math.pi + math.pi / 4, math.pi),
    )
    b.lanelet(E_IN1, _line((60.0, _HALF), (10.0, _HALF)))
    # --- N and S roads --------------------------------------------------------
    b.lanelet(N_OUT, _line((_HALF, 10.0), (_HALF, 110.0)))
    b.lanelet(N_IN, _line((-_HALF, 110.0), (-_HALF, 10.0)))
    b.lanelet(S_IN, _line((_HALF, -110.0), (_HALF, -10.0)))
    b.lanelet(S_OUT, _line((-_HALF, -10.0), (-_HALF, -110.0)))
    # --- the junction ---------------------------------------------------------
    ends = {
        "W": ((-10.0, -_HALF), 0.0),
        "E": ((10.0, _HALF), math.pi),
        "N": ((-_HALF, 10.0), -math.pi / 2),
        "S": ((_HALF, -10.0), math.pi / 2),
    }
    starts = {
        "E": ((10.0, -_HALF), 0.0),
        "W": ((-10.0, _HALF), math.pi),
        "N": ((_HALF, 10.0), math.pi / 2),
        "S": ((-_HALF, -10.0), -math.pi / 2),
    }
    for (src, dst), lid in J.items():
        a, ha = ends[src]
        z, hz = starts[dst]
        turn = (hz - ha + math.pi) % (2 * math.pi) - math.pi
        direction = "straight" if abs(turn) < 0.1 else ("left" if turn > 0 else "right")
        centre = _line(a, z, 5.0) if direction == "straight" else _bezier(a, ha, z, hz)
        b.lanelet(lid, centre, {"turn_direction": direction}, ends=(ha, hz))
    if split_path:
        middle, heading = (3.0, 3.0), math.radians(-40.0)
        b.lanelet(
            SPLIT_FIRST,
            _bezier((-_HALF, 10.0), -math.pi / 2, middle, heading),
            {"turn_direction": "left"},
            ends=(-math.pi / 2, heading),
        )
        b.lanelet(
            SPLIT_SECOND,
            _bezier(middle, heading, (10.0, -_HALF), 0.0),
            {"turn_direction": "left"},
            ends=(heading, 0.0),
        )
    # --- traffic light on the W approach --------------------------------------
    light = LineString3d(
        next(b._next),
        [
            Point3d(next(b._next), -8.0, -6.0, 5.0),
            Point3d(next(b._next), -8.0, -5.0, 5.0),
        ],
        AttributeMap({"type": "traffic_light"}),
    )
    stop_line = LineString3d(
        next(b._next),
        [
            Point3d(next(b._next), -12.0, -3.5, 0.0),
            Point3d(next(b._next), -12.0, 0.0, 0.0),
        ],
        AttributeMap({"type": "stop_line"}),
    )
    signal = lanelet2.core.TrafficLight(
        TRAFFIC_LIGHT,
        AttributeMap({"type": "regulatory_element", "subtype": "traffic_light"}),
        [light],
        stop_line,
    )
    for lid in (W_IN2, J[("W", "E")], J[("W", "N")], J[("W", "S")]):
        b.map.laneletLayer[lid].addRegulatoryElement(signal)
    # --- crosswalks -----------------------------------------------------------
    for lid, x in ((CROSSWALK_W, -15.0), (CROSSWALK_E, 15.0)):
        b.map.add(
            Lanelet(
                lid,
                b.line([(x - 1.0, -7.0), (x - 1.0, 7.0)], "solid"),
                b.line([(x + 1.0, -7.0), (x + 1.0, 7.0)], "solid"),
                AttributeMap({"type": "lanelet", "subtype": "crosswalk"}),
            )
        )
    # --- a one-way road elsewhere ---------------------------------------------
    b.lanelet(ONE_WAY1, _line((1000.0, 500.0), (1060.0, 500.0)))
    b.lanelet(ONE_WAY2, _line((1060.0, 500.0), (1120.0, 500.0)))
    return b.map


def routing_graph(lanelet_map: Any) -> Any:
    rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )
    return lanelet2.routing.RoutingGraph(lanelet_map, rules)
