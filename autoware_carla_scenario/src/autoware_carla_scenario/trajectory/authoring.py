"""A trajectory as a scenario document states it, and the one it builds.

The editor's *Follow Trajectory* card says where the path comes from in one of
three ways (:data:`PATH_SOURCES`):

* ``vertices`` -- a list of map-frame vertices ``[x, y, yaw]`` kept in the
  document itself, ``yaw`` optional.  This is what a recording transcribes
  to, and what an author pastes or edits as text, one vertex per line;
* ``lanelets`` -- a route of lanelets picked on the map, followed along their
  centrelines at a lateral offset and timed at a constant speed;
* ``relative_lane`` -- vertices ``[ds, offset, d_lane, yaw]`` relative to an
  entity's lane (:class:`~.model.RelativeLanePose`), placed against where
  that entity is when the action starts: the shape of a manoeuvre -- pull out,
  overtake, cut in -- written once and played from wherever it begins.

What a written vertex departs on -- a time included -- is not in its row: it
is the card's waypoint condition on that vertex (``advance_conditions``; a
time is a ``trajectory_time`` condition).  A row with a time cell, as
documents had before, is refused with a message saying so.

The parsing here is pure Python, so the editor process -- which never imports
CARLA or lanelet2 -- can check a document with it.  Building a lanelet path
needs the loaded map and imports it only then.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from .model import (
    MapPose,
    ReferenceContext,
    RelativeLanePose,
    Trajectory,
    TrajectoryTiming,
    TrajectoryVertex,
)

if TYPE_CHECKING:
    from ..conditions.base import BaseCondition

__all__ = [
    "PATH_SOURCES",
    "RelativeVertexRow",
    "TIME_DOMAINS",
    "VertexRow",
    "authored_timing",
    "authored_trajectory",
    "format_relative_vertices",
    "format_vertices",
    "parse_relative_vertices",
    "parse_vertices",
    "relative_trajectory_summary",
    "relative_vertex_problems",
    "trajectory_summary",
    "vertex_problems",
]

#: Where a document's trajectory comes from.
PATH_SOURCES: tuple[str, ...] = ("vertices", "lanelets", "relative_lane")

#: A document's time reference: ``none`` ignores the vertex times.
TIME_DOMAINS: tuple[str, ...] = ("none", "relative", "absolute")

#: One vertex: ``(x, y, yaw)`` in the ``map`` frame, the yaw optional.
VertexRow = tuple[float, float, Optional[float]]

#: One lane-relative vertex: ``(ds, offset, d_lane, yaw)``, the yaw optional
#: (see :class:`~.model.RelativeLanePose`).
RelativeVertexRow = tuple[float, float, int, Optional[float]]

#: How far apart (m) the vertices of a lanelet path are.
LANELET_PATH_STEP_M = 2.0


def _no_time(index: int, time: Any) -> ValueError:
    """The error for a row that still states a time, saying how to write it now."""
    return ValueError(
        f"vertex {index}: a vertex row has no time cell any more; give vertex "
        f"{index} a Trajectory time waypoint condition instead "
        f"(advance_conditions: - vertex: {index}, condition: {{type: "
        f"trajectory_time, params: {{time: {time}}}}})"
    )


def _rows(value: Any) -> list[Any]:
    """The rows of a document value or of the editor's text."""
    if value is None:
        return []
    if isinstance(value, str):
        rows: list[Any] = []
        for line in value.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([cell.strip() for cell in line.split(",")])
        return rows
    return list(value)


def _number(index: int, cell: Any, name: str, required: bool) -> Optional[float]:
    """*cell* as a finite number; ``None`` for an empty optional one."""
    if cell is None or (isinstance(cell, str) and not cell.strip()):
        if required:
            raise ValueError(f"vertex {index}: {name} is required")
        return None
    if isinstance(cell, bool):
        raise ValueError(f"vertex {index}: {cell!r} is not a number")
    try:
        number = float(cell)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vertex {index}: {cell!r} is not a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"vertex {index}: {cell!r} is not a finite number")
    return number


def _cell(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.10g}"


# ---------------------------------------------------------------------------
# Vertices, as text and as data
# ---------------------------------------------------------------------------


def parse_vertices(value: Any) -> list[VertexRow]:
    """Read vertices from a document value or the editor's text.

    Accepts a list of rows (``[x, y]`` or ``[x, y, yaw]``, with ``null`` for an
    unstated yaw), a list of mappings with those keys, or text with one vertex
    per line and the cells separated by commas -- ``81234.5, 50123.0`` leaves
    the yaw unstated.  Blank lines and lines starting with ``#`` are skipped,
    and are not counted: vertex N is the Nth vertex.

    Raises:
        ValueError: On a row that is not two or three numbers, saying which --
            and, for a row that still has a time cell, how to give the vertex
            its time now.
    """
    return [_row(index, row) for index, row in enumerate(_rows(value), start=1)]


def _row(index: int, row: Any) -> VertexRow:
    expected = "x, y[, yaw]"
    if isinstance(row, dict):
        if row.get("time") is not None:
            raise _no_time(index, row["time"])
        cells: list[Any] = [row.get("x"), row.get("y"), row.get("yaw")]
    elif isinstance(row, (list, tuple)):
        cells = list(row)
    else:
        raise ValueError(f"vertex {index}: expected {expected}, got {row!r}")
    if len(cells) == 4:
        if cells[3] is None or (isinstance(cells[3], str) and not cells[3].strip()):
            cells = cells[:3]
        else:
            raise _no_time(index, cells[3])
    if not 2 <= len(cells) <= 3:
        raise ValueError(
            f"vertex {index}: expected {expected}, got {len(cells)} values"
        )
    cells += [None] * (3 - len(cells))
    x = _number(index, cells[0], "x", True)
    y = _number(index, cells[1], "y", True)
    yaw = _number(index, cells[2], "yaw", False)
    assert x is not None and y is not None
    return x, y, yaw


def format_vertices(rows: Any) -> str:
    """The editor's text for *rows*: one ``x, y[, yaw]`` line per vertex."""
    if isinstance(rows, str):
        return rows
    lines = []
    for x, y, yaw in parse_vertices(rows):
        cells = [_cell(x), _cell(y), _cell(yaw)]
        while cells and not cells[-1]:
            cells.pop()
        lines.append(", ".join(cells))
    return "\n".join(lines)


def vertex_problems(rows: Sequence[Any]) -> list[str]:
    """What keeps *rows* from being a trajectory, or nothing.

    The rule :class:`~.model.Trajectory` enforces on a row list, stated
    without building one, so a document can be checked where the runtime is
    not importable.  The timing rules are the waypoint conditions'.
    """
    if len(rows) < 2:
        return [f"needs at least two vertices, has {len(rows)}"]
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
    return f"{len(rows)} vertices, {length:.1f} m"


# ---------------------------------------------------------------------------
# Lane-relative vertices, as text and as data
# ---------------------------------------------------------------------------

#: The cells of a lane-relative vertex, in order.
_RELATIVE_CELLS = ("ds", "offset", "d_lane", "yaw")


def parse_relative_vertices(value: Any) -> list[RelativeVertexRow]:
    """Read lane-relative vertices from a document value or the editor's text.

    The same shapes as :func:`parse_vertices` -- rows, mappings with the keys
    ``ds``, ``offset``, ``d_lane`` and ``yaw``, or text with one vertex per
    line -- with the cells ``ds[, offset][, d_lane][, yaw]``.  Only ``ds`` is
    required: an empty ``offset`` is the centreline, an empty ``d_lane`` the
    reference entity's own lane, an empty ``yaw`` the direction of the path.
    ``d_lane`` must be a whole number.

    Raises:
        ValueError: On a row that is not one to four numbers, saying which --
            and, for a row that still has a time cell, how to give the vertex
            its time now.
    """
    return [
        _relative_row(index, row) for index, row in enumerate(_rows(value), start=1)
    ]


def _relative_row(index: int, row: Any) -> RelativeVertexRow:
    expected = "ds[, offset][, d_lane][, yaw]"
    if isinstance(row, dict):
        if row.get("time") is not None:
            raise _no_time(index, row["time"])
        cells: list[Any] = [row.get(name) for name in _RELATIVE_CELLS]
    elif isinstance(row, (list, tuple)):
        cells = list(row)
    else:
        raise ValueError(f"vertex {index}: expected {expected}, got {row!r}")
    if len(cells) == 5:
        if cells[4] is None or (isinstance(cells[4], str) and not cells[4].strip()):
            cells = cells[:4]
        else:
            raise _no_time(index, cells[4])
    if not 1 <= len(cells) <= 4:
        raise ValueError(
            f"vertex {index}: expected {expected}, got {len(cells)} values"
        )
    cells += [None] * (4 - len(cells))
    ds = _number(index, cells[0], "ds", True)
    offset = _number(index, cells[1], "offset", False)
    d_lane = _number(index, cells[2], "d_lane", False)
    yaw = _number(index, cells[3], "yaw", False)
    assert ds is not None
    if d_lane is not None and not d_lane.is_integer():
        raise ValueError(
            f"vertex {index}: d_lane is a number of lanes, a whole number, "
            f"got {cells[2]!r}"
        )
    return (
        ds,
        0.0 if offset is None else offset,
        0 if d_lane is None else int(d_lane),
        yaw,
    )


def format_relative_vertices(rows: Any) -> str:
    """The editor's text for *rows*: one ``ds, offset, d_lane[, yaw]`` line each."""
    if isinstance(rows, str):
        return rows
    lines = []
    for ds, offset, d_lane, yaw in parse_relative_vertices(rows):
        # ds, offset and d_lane are always written out, so a line reads the
        # same whichever cells follow; only an unstated yaw is dropped.
        cells = [_cell(ds), _cell(offset), str(d_lane), _cell(yaw)]
        if not cells[-1]:
            cells.pop()
        lines.append(", ".join(cells))
    return "\n".join(lines)


def relative_vertex_problems(rows: Sequence[Any]) -> list[str]:
    """What keeps lane-relative *rows* from being a trajectory, or nothing.

    Whether the lanes they name exist -- a lane to the left, a lanelet past
    the end of the reference's -- depends on where the reference entity is
    when the action starts, so that is only known then.
    """
    return vertex_problems(rows)


def relative_trajectory_summary(value: Any) -> str:
    """One line saying what a lane-relative trajectory field holds."""
    try:
        rows = parse_relative_vertices(value)
    except ValueError as exc:
        return str(exc)
    if not rows:
        return "no vertices"
    first, last = rows[0], rows[-1]
    lanes = sorted({row[2] for row in rows})
    text = f"{len(rows)} vertices, ds {first[0]:g} to {last[0]:g} m"
    if lanes != [0]:
        text += ", lanes " + ", ".join(f"{lane:+d}" for lane in lanes)
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
    relative_vertices: Any = None,
    reference_entity: Optional[str] = None,
    advance: Optional[Mapping[int, "BaseCondition"]] = None,
) -> Trajectory:
    """The trajectory a *Follow Trajectory* card describes.

    Args:
        path_source: ``vertices``, ``lanelets`` or ``relative_lane`` (see
            :data:`PATH_SOURCES`).
        vertices: The ``vertices`` source's rows (see :func:`parse_vertices`).
        lanelet_ids: The ``lanelets`` source's route, in driving order.
        speed_kmh: The ``lanelets`` source's speed, which times its vertices.
            ``None`` or zero leaves them untimed.
        lateral_offset_m: The ``lanelets`` source's offset from the
            centreline, positive to the left.
        name: What the trajectory is called.
        relative_vertices: The ``relative_lane`` source's rows (see
            :func:`parse_relative_vertices`).
        reference_entity: The ``relative_lane`` source's reference: the
            ``role_name`` of the entity its vertices are relative to, ``None``
            for the entity the action moves.
        advance: The card's waypoint conditions, by vertex index counted from
            0: what each of those vertices departs on, a time included.  Only
            for the two sources whose vertices are written out.

    Raises:
        ValueError: On an unknown source, or a source without what it needs.
    """
    gates = dict(advance or {})
    if path_source == "vertices":
        rows = parse_vertices(vertices)
        problems = vertex_problems(rows)
        if problems:
            raise ValueError(f"{name}: {problems[0]}")
        return Trajectory(
            name,
            [
                TrajectoryVertex(MapPose(x, y, yaw), gates.get(index))
                for index, (x, y, yaw) in enumerate(rows)
            ],
        )
    if path_source == "relative_lane":
        relative_rows = parse_relative_vertices(relative_vertices)
        problems = relative_vertex_problems(relative_rows)
        if problems:
            raise ValueError(f"{name}: {problems[0]}")
        reference = reference_entity or None
        return Trajectory(
            name,
            [
                TrajectoryVertex(
                    RelativeLanePose(
                        ds=ds,
                        offset=offset,
                        d_lane=d_lane,
                        yaw=yaw,
                        entity_ref=reference,
                    ),
                    gates.get(index),
                )
                for index, (ds, offset, d_lane, yaw) in enumerate(relative_rows)
            ],
        )
    if path_source == "lanelets":
        if gates:
            raise ValueError(
                f"{name}: a lanelet path's vertices are generated, so they take "
                "no waypoint conditions"
            )
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
    from ..conditions.trajectory_time import (  # noqa: PLC0415
        TrajectoryTimeCondition,
    )
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
            vertices.append(
                TrajectoryVertex(
                    Lanelet2Pose(lanelet_id=int(lanelet_id), s=s, t=lateral_offset_m),
                    TrajectoryTimeCondition((travelled + s) / speed_ms)
                    if speed_ms > 0.0
                    else None,
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
