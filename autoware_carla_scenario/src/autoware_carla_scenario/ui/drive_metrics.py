"""How the ego drove: the gaps between road users' boxes, and the metrics
the viewer reads off a trajectory log, fed one tick at a time.

``ui.results.case_metrics`` reads the log (its JSON lines, records of any
type) and hands each of the ego's ticks to a :class:`DriveMetrics`, and the
gap of the ego's box (:func:`box`) to each other road user's
(:func:`box_distance`). Split out of ``ui.results``, which re-exports what it
used of it, so that the library check compiles the arithmetic
(``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["DECEL_WINDOW_S", "DriveMetrics", "box", "box_distance"]

#: The span the hardest braking is measured over.
DECEL_WINDOW_S = 0.5


def box(
    x: float, y: float, yaw: float, extent: tuple[float, float]
) -> list[tuple[float, float]]:
    """The corners of a box on the ground: centred at *x*, *y*, turned by
    *yaw* (degrees), *extent* its half length and half width."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    ex, ey = extent
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((ex, ey), (-ex, ey), (-ex, -ey), (ex, -ey))
    ]


def _separated(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * px + ny * py for px, py in a]
            pb = [nx * px + ny * py for px, py in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return True
    return False


def _point_segment(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length = dx * dx + dy * dy
    t = (
        0.0
        if length == 0
        else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length))
    )
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def box_distance(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """The distance between two boxes on the ground (0 when they overlap)."""
    if not _separated(a, b):
        return 0.0
    return min(
        _point_segment(p, q[i], q[(i + 1) % 4])
        for p_box, q in ((a, b), (b, a))
        for p in p_box
        for i in range(4)
    )


class DriveMetrics:
    """The metrics of the ego's ticks, fed one at a time, in the log's order:
    :meth:`gap` with the gap to each other road user on a tick, then
    :meth:`tick` with the ego's time, position and speed (km/h) on it."""

    min_gap: float
    distance: float
    max_speed: float
    final_speed: Optional[float]
    max_decel: float
    stop_time: Optional[float]
    moved: bool
    #: The last tick's time, x, y and speed.
    last: Optional[tuple[float, float, float, float]]
    #: (t, speed) over the last window.
    window: list[tuple[float, float]]

    def __init__(self) -> None:
        self.min_gap = math.inf
        self.distance = 0.0
        self.max_speed = 0.0
        self.final_speed = None
        self.max_decel = 0.0
        self.stop_time = None
        self.moved = False
        self.last = None
        self.window = []

    def gap(self, gap: float) -> None:
        """The gap on the ground from the ego's box to another road user's."""
        self.min_gap = min(self.min_gap, gap)

    def tick(self, t: float, x: float, y: float, speed: float) -> None:
        """The ego at time *t*: at *x*, *y*, going at *speed* km/h."""
        last = self.last
        if last is not None:
            self.distance += math.hypot(x - last[1], y - last[2])
        # Braking over a window, not tick to tick: a single tick's
        # velocity jumps (a spawn, a teleport) would read as braking.
        self.window.append((t, speed))
        while len(self.window) > 1 and t - self.window[1][0] >= DECEL_WINDOW_S:
            self.window.pop(0)
        wt, ws = self.window[0]
        if t - wt >= DECEL_WINDOW_S:
            self.max_decel = max(self.max_decel, (ws - speed) / 3.6 / (t - wt))
        self.max_speed = max(self.max_speed, speed)
        self.moved = self.moved or speed > 5.0
        if self.moved and self.stop_time is None and speed < 0.5:
            self.stop_time = t
        self.final_speed = speed
        self.last = (t, x, y, speed)

    def to_dict(self) -> dict[str, float]:
        """The metrics, rounded (``ui.results.case_metrics``); empty before
        the first tick."""
        metrics: dict[str, float] = {}
        if self.last is not None:
            metrics["distance_m"] = round(self.distance, 2)
            metrics["max_speed_kmh"] = round(self.max_speed, 2)
            # `final_speed or 0.0`, written out for Codon.
            final = self.final_speed
            metrics["final_speed_kmh"] = round(
                final if final is not None and final else 0.0, 2
            )
            metrics["max_decel_ms2"] = round(self.max_decel, 2)
            if math.isfinite(self.min_gap):
                metrics["min_gap_m"] = round(self.min_gap, 3)
            stop = self.stop_time
            if stop is not None:
                metrics["stop_time_s"] = round(stop, 2)
        return metrics
