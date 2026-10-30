"""The editor's text of a trajectory's vertices, and what is read off typed rows.

What :mod:`.authoring` does with text and with rows already parsed, which only
ever sees ``str``, numbers and the typed rows: the error for a row that still
states a time (:func:`no_time`), the editor's text split into rows of cells
(:func:`text_rows`) and a cell read as a number (:func:`text_number`), a
number written back as a cell (:func:`cell_text`), the rule on how many
vertices a trajectory needs (:func:`vertex_count_problems`), the inspector's
one-line summaries of parsed rows (:func:`vertices_summary`,
:func:`relative_vertices_summary`), and a route vertex's ``kind key=value``
line split into its kind and pairs (:func:`route_tokens`).

Split out of ``trajectory.authoring``, which keeps the dispatch on document
values of any type (YAML rows, mappings) and calls these, so that the library
check compiles it (``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

__all__ = [
    "RelativeVertexRow",
    "VertexRow",
    "cell_text",
    "no_time",
    "relative_vertices_summary",
    "route_tokens",
    "text_number",
    "text_rows",
    "vertex_count_problems",
    "vertices_summary",
]

#: One vertex: ``(x, y, yaw)`` in the ``map`` frame, the yaw optional.
VertexRow = Tuple[float, float, Optional[float]]

#: One lane-relative vertex: ``(ds, offset, d_lane, yaw)``, the yaw optional
#: (see :class:`~.model.RelativeLanePose`).
RelativeVertexRow = Tuple[float, float, int, Optional[float]]


def no_time(index: int, time: str) -> ValueError:
    """The error for a row that still states a time, saying how to write it now.

    *time* is the time cell as written (``str()`` of a document value).
    """
    return ValueError(
        f"vertex {index}: a vertex row has no time cell any more; give vertex "
        f"{index} a Trajectory time waypoint condition instead "
        f"(advance_conditions: - vertex: {index}, condition: {{type: "
        f"trajectory_time, params: {{time: {time}}}}})"
    )


def text_rows(text: str) -> List[List[str]]:
    """The rows of the editor's text: one per line, its cells split at commas.

    Blank lines and lines starting with ``#`` are skipped; every cell is
    stripped.
    """
    rows: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows


def text_number(index: int, cell: str, name: str, required: bool) -> Optional[float]:
    """The text *cell* as a finite number; ``None`` for an empty optional one."""
    if not cell.strip():
        if required:
            raise ValueError(f"vertex {index}: {name} is required")
        return None
    try:
        number = float(cell)
    except ValueError as exc:
        raise ValueError(f"vertex {index}: {cell!r} is not a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"vertex {index}: {cell!r} is not a finite number")
    return number


def cell_text(value: Optional[float]) -> str:
    """*value* as the editor writes a cell: empty for ``None``."""
    if value is None:
        return ""
    return f"{value:.10g}"


def vertex_count_problems(count: int) -> List[str]:
    """What keeps *count* vertices from being a trajectory, or nothing."""
    if count < 2:
        return [f"needs at least two vertices, has {count}"]
    return []


def vertices_summary(rows: List[VertexRow]) -> str:
    """One line saying what parsed map-frame *rows* hold: how many, how long."""
    if not rows:
        return "no vertices"
    length = 0.0
    for i in range(1, len(rows)):
        a = rows[i - 1]
        b = rows[i]
        length += math.hypot(b[0] - a[0], b[1] - a[1])
    return f"{len(rows)} vertices, {length:.1f} m"


def relative_vertices_summary(rows: List[RelativeVertexRow]) -> str:
    """One line saying what parsed lane-relative *rows* hold."""
    if not rows:
        return "no vertices"
    first = rows[0]
    last = rows[-1]
    lanes = sorted({row[2] for row in rows})
    text = f"{len(rows)} vertices, ds {first[0]:g} to {last[0]:g} m"
    if lanes != [0]:
        text += ", lanes " + ", ".join([f"{lane:+d}" for lane in lanes])
    return text


def route_tokens(index: int, line: str) -> Tuple[str, List[Tuple[str, str]]]:
    """A route vertex's ``kind key=value ...`` line: its kind and its pairs.

    Commas count as blanks.  The pairs come in the order written, repeats
    included (a later one wins once they are put in a mapping).

    Raises:
        ValueError: On an empty line, or a token that is not ``key=value``.
    """
    tokens = line.replace(",", " ").split()
    if not tokens:
        raise ValueError(f"vertex {index}: empty")
    pairs: List[Tuple[str, str]] = []
    for token in tokens[1:]:
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(
                f"vertex {index}: {token!r} is not key=value (write e.g. "
                "'lane ds=10 d_lane=1')"
            )
        pairs.append((key.strip(), value.strip()))
    return tokens[0], pairs
