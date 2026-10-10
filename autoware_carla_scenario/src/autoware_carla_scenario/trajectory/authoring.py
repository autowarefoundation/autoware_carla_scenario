"""A trajectory as a scenario document states it, and the one it builds.

The editor's *Follow Trajectory* card says where the path comes from in one of
two ways (:data:`PATH_SOURCES`):

* ``vertices`` -- a list of map-frame vertices ``[x, y, yaw, time]`` kept in the
  document itself, ``yaw`` and ``time`` optional.  This is what a recording
  transcribes to, and what an author pastes or edits as text, one vertex per
  line;
* ``lanelets`` -- a route of lanelets picked on the map, followed along their
  centrelines at a lateral offset and timed at a constant speed.

The parsing here is pure Python, so the editor process -- which never imports
CARLA or lanelet2 -- can check a document with it.  Building a lanelet path
needs the loaded map and imports it only then.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from .model import (
    MapPose,
    ReferenceContext,
    Trajectory,
    TrajectoryTiming,
    TrajectoryVertex,
)

__all__ = [
    "PATH_SOURCES",
    "TIME_DOMAINS",
    "VertexRow",
    "authored_timing",
    "authored_trajectory",
    "format_vertices",
    "parse_vertices",
    "trajectory_summary",
    "vertex_problems",
]

#: Where a document's trajectory comes from.
PATH_SOURCES: tuple[str, ...] = ("vertices", "lanelets")

#: A document's time reference: ``none`` ignores the vertex times.
TIME_DOMAINS: tuple[str, ...] = ("none", "relative", "absolute")

#: One vertex: ``(x, y, yaw, time)`` in the ``map`` frame, the last two optional.
VertexRow = tuple[float, float, Optional[float], Optional[float]]

#: How far apart (m) the vertices of a lanelet path are.
LANELET_PATH_STEP_M = 2.0


# ---------------------------------------------------------------------------
# Vertices, as text and as data
# ---------------------------------------------------------------------------


def parse_vertices(value: Any) -> list[VertexRow]:
    """Read vertices from a document value or the editor's text.

    Accepts a list of rows (``[x, y]``, ``[x, y, yaw]`` or ``[x, y, yaw, time]``,
    with ``null`` for an unstated yaw or time), a list of mappings with those
    keys, or text with one vertex per line and the cells separated by commas
    -- ``81234.5, 50123.0, , 1.5`` leaves the yaw unstated.  Blank lines and
    lines starting with ``#`` are skipped.

    Raises:
        ValueError: On a row that is not two to four numbers, saying which.
    """
    if value is None:
        return []
    if isinstance(value, str):
        rows: list[Any] = []
        for line in value.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([cell.strip() for cell in line.split(",")])
    else:
        rows = list(value)
    return [_row(index, row) for index, row in enumerate(rows, start=1)]


def _row(index: int, row: Any) -> VertexRow:
    if isinstance(row, dict):
        cells: list[Any] = [row.get("x"), row.get("y"), row.get("yaw"), row.get("time")]
    elif isinstance(row, (list, tuple)):
        cells = list(row)
    else:
        raise ValueError(f"vertex {index}: expected x, y[, yaw][, time], got {row!r}")
    if not 2 <= len(cells) <= 4:
        raise ValueError(
            f"vertex {index}: expected x, y[, yaw][, time], got {len(cells)} values"
        )
    cells += [None] * (4 - len(cells))
    numbers: list[Optional[float]] = []
    for position, cell in enumerate(cells):
        if cell is None or (isinstance(cell, str) and not cell.strip()):
            if position < 2:
                raise ValueError(f"vertex {index}: x and y are required")
            numbers.append(None)
            continue
        try:
            number = float(cell)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"vertex {index}: {cell!r} is not a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"vertex {index}: {cell!r} is not a finite number")
        numbers.append(number)
    x, y, yaw, time = numbers
    assert x is not None and y is not None
    return x, y, yaw, time


def format_vertices(rows: Any) -> str:
    """The editor's text for *rows*: one ``x, y, yaw, time`` line per vertex."""
    if isinstance(rows, str):
        return rows

    def cell(value: Optional[float]) -> str:
        return "" if value is None else f"{value:.10g}"

    lines = []
    for x, y, yaw, time in parse_vertices(rows):
        cells = [cell(x), cell(y), cell(yaw), cell(time)]
        while cells and not cells[-1]:
            cells.pop()
        lines.append(", ".join(cells))
    return "\n".join(lines)


def vertex_problems(rows: Sequence[VertexRow]) -> list[str]:
    """What keeps *rows* from being a trajectory, or nothing.

    The rules :class:`~.model.Trajectory` enforces, stated without building
    one, so a document can be checked where the runtime is not importable.
    """
    if len(rows) < 2:
        return [f"needs at least two vertices, has {len(rows)}"]
    timed = [row[3] is not None for row in rows]
    if any(timed) and not all(timed):
        return ["either every vertex has a time or none has"]
    if all(timed):
        times = [row[3] for row in rows]
        if any(b < a for a, b in zip(times, times[1:])):  # type: ignore[operator]
            return ["the vertex times decrease"]
    return []


def trajectory_summary(value: Any) -> str:
    """One line saying what a trajectory field holds, for the inspector."""
    try:
        rows = parse_vertices(value)
    except ValueError as exc:
        return str(exc)
    if not rows:
        return "no vertices"
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(rows, rows[1:]))
    text = f"{len(rows)} vertices, {length:.1f} m"
    times = [row[3] for row in rows]
    if all(time is not None for time in times):
        text += f", {times[-1] - times[0]:.1f} s"  # type: ignore[operator]
    return text


# ---------------------------------------------------------------------------
# What the builder calls
# ---------------------------------------------------------------------------


def authored_trajectory(
    path_source: str,
    vertices: Any = None,
    lanelet_ids: Optional[Sequence[int]] = None,
    speed_kmh: Optional[float] = None,
    lateral_offset_m: float = 0.0,
    name: str = "trajectory",
) -> Trajectory:
    """The trajectory a *Follow Trajectory* card describes.

    Args:
        path_source: ``vertices`` or ``lanelets`` (see :data:`PATH_SOURCES`).
        vertices: The ``vertices`` source's rows (see :func:`parse_vertices`).
        lanelet_ids: The ``lanelets`` source's route, in driving order.
        speed_kmh: The ``lanelets`` source's speed, which times its vertices.
            ``None`` or zero leaves them untimed.
        lateral_offset_m: The ``lanelets`` source's offset from the
            centreline, positive to the left.
        name: What the trajectory is called.

    Raises:
        ValueError: On an unknown source, or a source without what it needs.
    """
    if path_source == "vertices":
        rows = parse_vertices(vertices)
        problems = vertex_problems(rows)
        if problems:
            raise ValueError(f"{name}: {problems[0]}")
        return Trajectory(
            name,
            [TrajectoryVertex(MapPose(x, y, yaw), time) for x, y, yaw, time in rows],
        )
    if path_source == "lanelets":
        return _lanelet_path(
            name, list(lanelet_ids or ()), speed_kmh, float(lateral_offset_m or 0.0)
        )
    raise ValueError(f"{name}: unknown path source {path_source!r}")


def _lanelet_path(
    name: str,
    lanelet_ids: list[int],
    speed_kmh: Optional[float],
    lateral_offset_m: float,
) -> Trajectory:
    """A path along the centrelines of *lanelet_ids*, timed at *speed_kmh*."""
    if not lanelet_ids:
        raise ValueError(f"{name}: a lanelet path needs at least one lanelet")
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415
    from ..coordinate.transform import lanelet_length  # noqa: PLC0415

    speed_ms = (speed_kmh or 0.0) / 3.6
    vertices: list[TrajectoryVertex] = []
    travelled = 0.0
    for position, lanelet_id in enumerate(lanelet_ids):
        length = lanelet_length(int(lanelet_id))
        steps = max(1, math.ceil(length / LANELET_PATH_STEP_M))
        # A lanelet's start is the previous one's end: taken once.
        first = 0 if position == 0 else 1
        for step in range(first, steps + 1):
            s = length * step / steps
            time = (travelled + s) / speed_ms if speed_ms > 0.0 else None
            vertices.append(
                TrajectoryVertex(
                    Lanelet2Pose(lanelet_id=int(lanelet_id), s=s, t=lateral_offset_m),
                    time,
                )
            )
        travelled += length
    return Trajectory(name, vertices)


def authored_timing(
    time_domain: str, scale: float = 1.0, offset: float = 0.0
) -> Optional[TrajectoryTiming]:
    """The time reference a card's ``time_domain`` names, or ``None`` for none."""
    if time_domain == "none":
        return None
    return TrajectoryTiming(
        domain=ReferenceContext(time_domain),
        scale=1.0 if scale is None else float(scale),
        offset=0.0 if offset is None else float(offset),
    )
