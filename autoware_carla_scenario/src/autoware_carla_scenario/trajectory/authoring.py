"""A route as a scenario document states it, and the one it builds.

The editor's *Set Goal* card says where the route goes in one of five ways
(:data:`PATH_SOURCES`):

* ``goal`` -- a goal lanelet (and ``s`` along it) and the lanelets to pass
  on the way: the destination an ego is given;
* ``vertices`` -- a list of map-frame vertices ``[x, y, yaw]`` kept in the
  document itself, ``yaw`` optional.  This is what a recording transcribes
  to, and what an author pastes or edits as text, one vertex per line;
* ``lanelets`` -- a route of lanelets picked on the map, followed along their
  centrelines at a lateral offset and timed at a constant speed;
* ``relative_lane`` -- vertices ``[ds, offset, d_lane, yaw]`` relative to an
  entity's lane (:class:`~.model.RelativeLanePose`), placed against where
  that entity is when the action starts: the shape of a manoeuvre -- pull out,
  overtake, cut in -- written once and played from wherever it begins;
* ``route`` -- vertices placed against the scenario's *route* (a logical
  scenario's route search, ``docs/logical_scenarios.md``): one mapping per
  vertex with a ``kind`` -- ``lane``, ``opposite``, ``crossing``,
  ``crosswalk`` or ``roadside`` -- and that kind's keys
  (:data:`ROUTE_ROW_FIELDS`), edited as text ``kind key=value ...``.

What a written vertex departs on -- a time included -- is not in its row: it
is the card's waypoint condition on that vertex (``advance_conditions``; a
time is a ``trajectory_time`` condition).  A row with a time cell, as
documents had before, is refused with a message saying so.

The parsing here is pure Python, so the editor process -- which never imports
CARLA or lanelet2 -- can check a document with it.  Building a lanelet path
needs the loaded map and imports it only then.  The builders that take
values already typed -- the goal, the via lanelets, the timing and the
lanelet path -- are :mod:`.authored`'s, which the library check compiles,
re-exported here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from ..value_text import shown
from .authored import (
    LANELET_PATH_STEP_M,
    PATH_SOURCES,
    TIME_DOMAINS,
    _lanelet_path,
    authored_goal,
    authored_timing,
    authored_waypoints,
)
from .model import (
    MapPose,
    RelativeLanePose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteLanePose,
    RouteOppositePose,
    RouteRoadsidePose,
    Trajectory,
    TrajectoryVertex,
)
from .vertex_text import (
    RelativeVertexRow,
    VertexRow,
    cell_text,
    no_time,
    relative_vertices_summary,
    route_tokens,
    text_number,
    text_rows,
    vertex_count_problems,
    vertices_summary,
)

if TYPE_CHECKING:
    from ..conditions.base import BaseCondition

__all__ = [
    "LANELET_PATH_STEP_M",
    "PATH_SOURCES",
    "ROUTE_ROW_FIELDS",
    "RelativeVertexRow",
    "TIME_DOMAINS",
    "VertexRow",
    "authored_goal",
    "authored_place",
    "authored_route",
    "authored_timing",
    "authored_trajectory",
    "authored_waypoints",
    "format_relative_vertices",
    "format_route_vertices",
    "format_vertices",
    "parse_relative_vertices",
    "parse_route_vertices",
    "parse_vertices",
    "relative_trajectory_summary",
    "relative_vertex_problems",
    "route_pose",
    "route_trajectory_summary",
    "trajectory_summary",
    "vertex_problems",
]


def _no_time(index: int, time: Any) -> ValueError:
    """The error for a row that still states a time, saying how to write it now."""
    return no_time(index, f"{time}")


def _rows(value: Any) -> list[Any]:
    """The rows of a document value or of the editor's text."""
    if value is None:
        return []
    if isinstance(value, str):
        return text_rows(value)
    return list(value)


def _number(index: int, cell: Any, name: str, required: bool) -> Optional[float]:
    """*cell* as a finite number; ``None`` for an empty optional one."""
    if isinstance(cell, str):
        return text_number(index, cell, name, required)
    if cell is None:
        if required:
            raise ValueError(f"vertex {index}: {name} is required")
        return None
    if isinstance(cell, bool):
        raise ValueError(f"vertex {index}: {cell!r} is not a number")
    try:
        number = float(cell)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vertex {index}: {cell!r} is not a number") from exc
    except OverflowError:
        # An int too big for a float (10**400): not finite (#1652).
        raise ValueError(
            f"vertex {index}: {shown(cell)} is not a finite number"
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"vertex {index}: {cell!r} is not a finite number")
    return number


def _field_float(name: str, value: Any) -> float:
    """``float(value)`` of the card's field *name*; for a value ``float()``
    cannot read -- not a number, or an int too big for a float
    (``10**400``) -- a ``ValueError`` naming the field (#1660).  What
    ``float()`` reads is read as before."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {shown(value)}") from None
    except OverflowError:
        raise ValueError(f"{name} must be finite, got {shown(value)}") from None


def _field_int(name: str, value: Any) -> int:
    """``int(value)`` of the card's field *name*; for a value ``int()`` cannot
    read, a ``ValueError`` naming the field (#1660).  What ``int()`` reads is
    read as before."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a whole number, got {shown(value)}") from None


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
        cells = [cell_text(x), cell_text(y), cell_text(yaw)]
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
    return vertex_count_problems(len(rows))


def trajectory_summary(value: Any) -> str:
    """One line saying what a trajectory field holds, for the inspector."""
    try:
        rows = parse_vertices(value)
    except ValueError as exc:
        return str(exc)
    return vertices_summary(rows)


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
        cells = [cell_text(ds), cell_text(offset), str(d_lane), cell_text(yaw)]
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
    return relative_vertices_summary(rows)


# ---------------------------------------------------------------------------
# Route vertices, as text and as data
# ---------------------------------------------------------------------------

#: Each route vertex kind, its pose class, and its keys in order.
ROUTE_ROW_FIELDS: dict[str, tuple[type, tuple[str, ...]]] = {
    "lane": (RouteLanePose, ("ds", "offset", "d_lane", "yaw", "anchor")),
    "opposite": (RouteOppositePose, ("ds", "lane", "offset", "yaw", "anchor")),
    "crossing": (
        RouteCrossingPose,
        ("junction", "approach", "distance", "turn", "offset", "yaw", "d_lane", "path"),
    ),
    "crosswalk": (RouteCrosswalkPose, ("junction", "leg", "side", "along", "yaw")),
    "roadside": (
        RouteRoadsidePose,
        ("ds", "side", "kerb_distance", "yaw", "anchor"),
    ),
}

#: Keys holding text, not numbers.
_TEXT_KEYS = frozenset({"approach", "turn", "leg", "side", "anchor"})
#: Keys holding a whole number.
_WHOLE_KEYS = frozenset({"d_lane", "lane", "junction", "path"})
#: Keys that may be left unstated (``None``).
_OPTIONAL_KEYS = frozenset({"yaw", "turn", "anchor", "path"})
#: Keys the editor's text always writes out, default or not: they say *which*.
_ALWAYS_WRITTEN = frozenset({"junction", "approach", "leg", "side"})


def _route_value(index: int, kind: str, key: str, raw: Any) -> Any:
    blank = raw is None or (
        isinstance(raw, str) and raw.strip().lower() in ("", "none", "null")
    )
    if blank:
        if key in _OPTIONAL_KEYS:
            return None
        raise ValueError(f"vertex {index}: {kind} {key} needs a value")
    if key in _TEXT_KEYS:
        return str(raw).strip()
    number = _number(index, raw, key, True)
    assert number is not None
    if key in _WHOLE_KEYS:
        if not number.is_integer():
            raise ValueError(f"vertex {index}: {key} is a whole number, got {raw!r}")
        return int(number)
    return number


def _route_row(index: int, row: Any) -> dict[str, Any]:
    if isinstance(row, str):
        kind, pairs = route_tokens(index, row)
        mapping: dict[str, Any] = {"kind": kind}
        for key, value in pairs:
            mapping[key] = value
        row = mapping
    if not isinstance(row, Mapping):
        raise ValueError(f"vertex {index}: expected a mapping with a kind, got {row!r}")
    kind = str(row.get("kind", "")).strip()
    if kind not in ROUTE_ROW_FIELDS:
        raise ValueError(
            f"vertex {index}: kind must be one of {list(ROUTE_ROW_FIELDS)}, got {kind!r}"
        )
    cls, keys = ROUTE_ROW_FIELDS[kind]
    unknown = sorted(set(row) - {"kind", *keys})
    if unknown:
        raise ValueError(f"vertex {index}: a {kind} vertex does not take {unknown}")
    defaults = cls()
    out: dict[str, Any] = {"kind": kind}
    for key in keys:
        raw = row.get(key, getattr(defaults, key))
        out[key] = _route_value(index, kind, key, raw)
    try:
        cls(**{k: v for k, v in out.items() if k != "kind"})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vertex {index}: {exc}") from exc
    return out


def parse_route_vertices(value: Any) -> list[dict[str, Any]]:
    """Read route vertices from a document value or the editor's text.

    A list of mappings (``{kind: lane, ds: 10, d_lane: 1}``) or text with one
    vertex per line, ``kind key=value ...`` (commas allowed between pairs);
    blank lines and ``#`` comments are skipped and not counted.  Each row comes
    back as a mapping with its kind and every one of that kind's keys
    (:data:`ROUTE_ROW_FIELDS`), unstated keys at their defaults.

    Raises:
        ValueError: On a row that is not a route vertex, saying which and why.
    """
    if value is None:
        return []
    if isinstance(value, str):
        lines = [
            line.strip()
            for line in value.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return [_route_row(i, line) for i, line in enumerate(lines, start=1)]
    return [_route_row(i, row) for i, row in enumerate(list(value), start=1)]


def route_pose(row: Mapping[str, Any]) -> Any:
    """The route pose a parsed route vertex row names."""
    cls, keys = ROUTE_ROW_FIELDS[str(row["kind"])]
    return cls(**{key: row[key] for key in keys})


def format_route_vertices(rows: Any) -> str:
    """The editor's text for route *rows*: ``kind key=value ...``, defaults left out."""
    if isinstance(rows, str):
        return rows
    lines = []
    for row in parse_route_vertices(rows):
        cls, keys = ROUTE_ROW_FIELDS[row["kind"]]
        defaults = cls()
        cells = [row["kind"]]
        for key in keys:
            value = row[key]
            if value == getattr(defaults, key) and key not in _ALWAYS_WRITTEN:
                continue
            cells.append(
                f"{key}={cell_text(value) if isinstance(value, float) else value}"
            )
        lines.append(" ".join(cells))
    return "\n".join(lines)


def route_trajectory_summary(value: Any) -> str:
    """One line saying what a route trajectory field holds."""
    try:
        rows = parse_route_vertices(value)
    except ValueError as exc:
        return str(exc)
    if not rows:
        return "no vertices"
    kinds = sorted({row["kind"] for row in rows})
    return f"{len(rows)} route vertices ({', '.join(kinds)})"


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
    route_vertices: Any = None,
) -> Trajectory:
    """The trajectory a *Set Goal* card's written-out waypoints describe.

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
            for the sources whose vertices are written out.
        route_vertices: The ``route`` source's rows (see
            :func:`parse_route_vertices`).

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
    if path_source == "route":
        route_rows = parse_route_vertices(route_vertices)
        problems = vertex_problems(route_rows)
        if problems:
            raise ValueError(f"{name}: {problems[0]}")
        return Trajectory(
            name,
            [
                TrajectoryVertex(route_pose(row), gates.get(index))
                for index, row in enumerate(route_rows)
            ],
        )
    if path_source == "lanelets":
        if gates:
            raise ValueError(
                f"{name}: a lanelet path's vertices are generated, so they take "
                "no waypoint conditions"
            )
        return _lanelet_path(
            name,
            list(lanelet_ids or ()),
            speed_kmh,
            _field_float(f"{name}: lateral_offset_m", lateral_offset_m or 0.0),
        )
    raise ValueError(f"{name}: unknown path source {path_source!r}")


def authored_route(
    path_source: str,
    vertices: Any = None,
    lanelet_ids: Optional[Sequence[int]] = None,
    speed_kmh: Optional[float] = None,
    lateral_offset_m: float = 0.0,
    relative_vertices: Any = None,
    reference_entity: Optional[str] = None,
    advance: Optional[Mapping[int, "BaseCondition"]] = None,
    route_vertices: Any = None,
) -> Optional[Trajectory]:
    """The waypoints a *Set Goal* card writes out, as a trajectory; ``None``
    for a ``goal`` card, whose route is its goal and via lanelets.

    Raises:
        ValueError: As :func:`authored_trajectory`, and on waypoint
            conditions on a ``goal`` card, which writes no waypoints out.
    """
    if path_source == "goal":
        if advance:
            raise ValueError(
                "a route given as a goal and via lanelets writes no waypoints "
                "out to give conditions to"
            )
        return None
    return authored_trajectory(
        path_source,
        vertices=vertices,
        lanelet_ids=lanelet_ids,
        speed_kmh=speed_kmh,
        lateral_offset_m=lateral_offset_m,
        name="route",
        relative_vertices=relative_vertices,
        reference_entity=reference_entity,
        advance=advance,
        route_vertices=route_vertices,
    )


def authored_place(
    lanelet_id: Optional[int] = 0,
    s: Optional[float] = 0.0,
    t: Optional[float] = 0.0,
    heading: Optional[float] = 0.0,
    route_vertex: Any = None,
) -> Any:
    """A ``spawn`` or ``place_clear`` card's place: the route pose its
    ``route_pose`` names (one route vertex, as a ``route_vertices`` row),
    placed on the scenario's route when it is used; else *s* metres along
    lanelet *lanelet_id*, *t* off its centreline, turned *heading*.

    Raises:
        ValueError: For a ``route_pose`` of more than one vertex.
    """
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415

    rows = parse_route_vertices(route_vertex)
    if len(rows) > 1:
        raise ValueError(f"a place is one route vertex, got {len(rows)}")
    if rows:
        return route_pose(rows[0])
    return Lanelet2Pose(
        lanelet_id=_field_int("lanelet_id", lanelet_id or 0),
        s=_field_float("s", s or 0.0),
        t=_field_float("t", t or 0.0),
        heading=_field_float("heading", heading or 0.0),
    )
