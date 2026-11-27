"""The raised and grooved parts of a road's surface, read from the ASAM OpenCRG
grids an OpenDRIVE road refers to.

OpenDRIVE describes the surface a road was designed to.  What a wheel meets on
top of it -- a speed hump, a frost heave, wheel-path ruts, a pothole -- is an
**OpenCRG** grid: a regular grid of heights in the road's own ``(u, v)``,
referred to from the road's ``<surface>`` by a ``<CRG>``.  In ``attached``
mode (``attached0`` too) its values are added to the road's surface, so each
is how far the surface stands **above** (positive) or **below** (negative)
the design there, which is what a map generator such as roadgen writes
(``export_opendrive(..., crg=True)``).

:func:`relief_regions` turns those grids into regions: the connected parts of
a grid that stand at least a threshold above the design (**raised**) or below
it (**grooved**), each as an outline in the road's ``(s, t)`` -- station along
its reference line, metres to the left of it -- ready to be placed in CARLA's
world through the road's reference line, as any OpenDRIVE pose is.

What is read
------------
* Only ``purpose="elevation"`` grids (a ``friction`` grid says nothing about
  height), in ``attached`` or ``attached0`` mode: their ``u`` runs along the
  road from ``sStart`` (``orientation="same"``) or back from ``sEnd``
  (``"opposite"``, where ``v`` is mirrored too), offset by ``sOffset`` and
  ``tOffset``.  A ``genuine`` or ``global`` grid lays its own reference line
  over the world, which is not read: it is reported, and passed over.
* The OpenCRG ASCII formats ``LRFI`` (ten-character values, eight to a record)
  and ``LDFI`` (twenty, four to a record); a binary file is reported and
  passed over.  The long sections are the ``D:`` channels that are not the
  reference line's own (``reference line u``, ``phi``, ...); where they lie
  across the road is ``LONG_SECTION_V_RIGHT`` and ``_INCREMENT``.  A value
  that is not a finite number counts as the design (``0``).

The outline of a region
-----------------------
Each grid value stands for the cell round it, half an increment either way
(clipped at the grid's edge).  A region's outline runs along each of its
cross sections from its first cell to its last: exact for a region every cross
section crosses once -- a hump, a pothole, one rut -- and a little generous
for one a cross section crosses twice (a U-shaped region), whose gap it
takes in.  A region with a hole is outlined round the outside.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree.ElementTree import Element, iterparse

__all__ = [
    "DEFAULT_RELIEF_M",
    "CrgGrid",
    "CrgReference",
    "ReliefRegion",
    "crg_references",
    "read_crg",
    "relief_regions",
]

#: How far (m) off the design a surface has to stand to count, unless asked:
#: above a road's ordinary unevenness, below any hump or rut a wheel feels.
DEFAULT_RELIEF_M = 0.01

#: The kinds of a region: above the design, and below it.
RAISED = "raised"
GROOVED = "grooved"

#: How wide a value is in each ASCII format, and how many fill a record.
_FIELD_WIDTHS = {"LRFI": 10, "LDFI": 20}

#: The modes whose ``u`` runs along the road the grid is attached to.
_ATTACHED = ("attached", "attached0")


@dataclass(frozen=True)
class CrgReference:
    """One ``<CRG>`` of a road's ``<surface>``.

    Attributes:
        road_id: The road it is on, as the OpenDRIVE file names it.
        path: The OpenCRG file, resolved against the OpenDRIVE file's folder.
        s_start: Where on the road the grid starts (``sStart``).
        s_end: Where it ends (``sEnd``).
        s_offset: ``sOffset``.
        t_offset: ``tOffset``.
        orientation: ``same`` or ``opposite``.
        mode: ``attached``, ``attached0``, ``genuine`` or ``global``.
        purpose: ``elevation`` or ``friction``.
    """

    road_id: str
    path: Path
    s_start: float
    s_end: float
    s_offset: float
    t_offset: float
    orientation: str
    mode: str
    purpose: str


@dataclass(frozen=True)
class CrgGrid:
    """An OpenCRG grid: its values by cross section (``u``) and long section
    (``v``), ``values[row][column]``."""

    u: list[float]
    v: list[float]
    values: list[list[float]]


@dataclass(frozen=True)
class ReliefRegion:
    """A connected part of a road's surface that stands above or below its
    design by at least the threshold asked for.

    Attributes:
        road_id: The road it is on.
        kind: :data:`RAISED` or :data:`GROOVED`.
        extreme_m: The most it stands off the design: the highest crest of a
            raised region, the deepest point of a grooved one (negative).
        outline: Its outline, ``(s, t)`` on the road.
        relief: How far (m) the surface stands off its design at each point
            of *outline*, in the same order: the grid's value at the cell the
            point is a corner of.  Placed on the road's own height, it is the
            height of the region (#2314).
    """

    road_id: str
    kind: str
    extreme_m: float
    outline: list[tuple[float, float]]
    relief: list[float]

    def describe(self) -> str:
        """``raised 0.080 m on road 12``."""
        return f"{self.kind} {abs(self.extreme_m):.3f} m on road {self.road_id}"


def _number(text: Optional[str], default: float) -> float:
    if text is None or not text.strip():
        return default
    return float(text)


def _attribute(element: Element, key: str, default: str) -> str:
    """*element*'s attribute *key*, or *default* where it has none or an
    empty one."""
    # Branched: Codon types `get(key) or default` as an Optional.
    value = element.get(key)
    if value is None or not value:
        return default
    return value


def crg_references(xodr_path: Path) -> list[CrgReference]:
    """Every ``<CRG>`` of every road's ``<surface>`` in the OpenDRIVE file, in
    the file's order.

    Raises:
        OSError: The file cannot be read.
        xml.etree.ElementTree.ParseError: It is not XML.
    """
    folder = xodr_path.parent
    found: list[CrgReference] = []
    for _event, element in iterparse(str(xodr_path)):
        if element.tag != "road":
            continue
        surface = element.find("surface")
        if surface is not None:
            road_id = _attribute(element, "id", "")
            for crg in surface.findall("CRG"):
                name = _attribute(crg, "file", "")
                found.append(
                    CrgReference(
                        road_id=road_id,
                        path=folder / name,
                        s_start=_number(crg.get("sStart"), 0.0),
                        s_end=_number(crg.get("sEnd"), math.inf),
                        s_offset=_number(crg.get("sOffset"), 0.0),
                        t_offset=_number(crg.get("tOffset"), 0.0),
                        orientation=_attribute(crg, "orientation", "same").lower(),
                        mode=_attribute(crg, "mode", "attached").lower(),
                        purpose=_attribute(crg, "purpose", "elevation").lower(),
                    )
                )
        # A road's elements are not needed once it is read; the file may be
        # a city.
        element.clear()
    return found


def _header_value(header: dict[str, str], key: str) -> Optional[float]:
    if key not in header:
        return None
    text = header[key].split("!")[0].strip()
    if not text:
        return None
    return float(text)


def read_crg(path: Path) -> CrgGrid:
    """Read an OpenCRG ASCII file.

    Raises:
        OSError: The file cannot be read.
        ValueError: It is not an OpenCRG ASCII file this reads: a binary
            format, no long sections, or no grid extent.
    """
    text = path.read_text(encoding="latin-1")
    lines = text.splitlines()
    header: dict[str, str] = {}
    data_format = ""
    channels: list[str] = []
    u_channel = ""
    block = ""
    first_data = -1
    for index, line in enumerate(lines):
        if line.startswith("$$$$"):
            first_data = index + 1
            break
        if line.startswith("$"):
            # `!` starts a comment here as in a header value: `$ROAD_CRG ! ...`
            # is the ROAD_CRG block (#2541).
            block = line[1:].split("!")[0].strip().upper()
            continue
        if block == "ROAD_CRG" and "=" in line:
            key, _sep, value = line.partition("=")
            header[key.strip().upper()] = value
        elif block == "KD_DEFINITION":
            stripped = line.strip()
            if stripped.startswith("#:"):
                data_format = stripped[2:].split("!")[0].strip().upper()
            elif stripped.upper().startswith("D:"):
                channels.append(stripped[2:])
            elif stripped.upper().startswith("U:"):
                u_channel = stripped[2:]
    if first_data < 0:
        raise ValueError(f"{path.name}: no $$$$ line before the data")
    if data_format not in _FIELD_WIDTHS:
        raise ValueError(
            f"{path.name}: format {data_format or '(none)'!r} is not an OpenCRG "
            "ASCII format this reads (LRFI, LDFI)"
        )
    width = _FIELD_WIDTHS[data_format]
    long_sections: list[int] = []
    for index, channel in enumerate(channels):
        if not channel.strip().lower().startswith("reference line"):
            long_sections.append(index)
    if not long_sections:
        raise ValueError(f"{path.name}: no long sections")

    numbers: list[float] = []
    for line in lines[first_data:]:
        record = line.rstrip("\r\n")
        for start in range(0, len(record), width):
            field = record[start : start + width].strip()
            if not field:
                continue
            # Not `value`, the header's text above: Codon gives a name one type.
            try:
                number = float(field)
            except ValueError:
                number = math.nan
            numbers.append(number)
    columns = len(channels)
    rows = len(numbers) // columns

    u_start = _header_value(header, "REFERENCE_LINE_START_U")
    u_step = _header_value(header, "REFERENCE_LINE_INCREMENT")
    if u_channel:
        parts = u_channel.split(",")
        if len(parts) >= 4:
            u_start = float(parts[2])
            u_step = float(parts[3])
    v_right = _header_value(header, "LONG_SECTION_V_RIGHT")
    v_step = _header_value(header, "LONG_SECTION_V_INCREMENT")
    if u_start is None or u_step is None or v_right is None or v_step is None:
        raise ValueError(
            f"{path.name}: the grid's extent (REFERENCE_LINE_START_U, "
            "_INCREMENT, LONG_SECTION_V_RIGHT, _INCREMENT) is not in $ROAD_CRG"
        )
    values: list[list[float]] = []
    for row in range(rows):
        cells: list[float] = []
        for column in long_sections:
            cell = numbers[row * columns + column]
            cells.append(cell if math.isfinite(cell) else 0.0)
        values.append(cells)
    u = [u_start + row * u_step for row in range(rows)]
    v = [v_right + k * v_step for k in range(len(long_sections))]
    return CrgGrid(u=u, v=v, values=values)


def _find(parent: list[int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _components(
    grid: CrgGrid, sign: float, threshold_m: float
) -> list[list[tuple[int, int, int]]]:
    """The connected parts of *grid* standing at least *threshold_m* off the
    design on the side of *sign* (``1`` above, ``-1`` below), each as its runs
    ``(row, first column, last column)``, rows in order."""
    runs: list[tuple[int, int, int]] = []
    row_runs: list[list[int]] = []
    for row, cells in enumerate(grid.values):
        here: list[int] = []
        column = 0
        count = len(cells)
        while column < count:
            if sign * cells[column] >= threshold_m:
                first = column
                while column + 1 < count and sign * cells[column + 1] >= threshold_m:
                    column += 1
                here.append(len(runs))
                runs.append((row, first, column))
            column += 1
        row_runs.append(here)
    parent = list(range(len(runs)))
    for row in range(1, len(row_runs)):
        for a in row_runs[row - 1]:
            for b in row_runs[row]:
                _ra, first_a, last_a = runs[a]
                _rb, first_b, last_b = runs[b]
                if first_a <= last_b and first_b <= last_a:
                    root_a, root_b = _find(parent, a), _find(parent, b)
                    if root_a != root_b:
                        parent[root_b] = root_a
    by_root: dict[int, list[tuple[int, int, int]]] = {}
    order: list[int] = []
    for index in range(len(runs)):
        root = _find(parent, index)
        if root not in by_root:
            by_root[root] = []
            order.append(root)
        by_root[root].append(runs[index])
    return [by_root[root] for root in order]


def _cell_edges(values: list[float], index: int) -> tuple[float, float]:
    """Where the cell of sample *index* begins and ends: half-way to its
    neighbours, the grid's own ends at its edges."""
    low = values[index]
    high = values[index]
    if index > 0:
        low = 0.5 * (values[index - 1] + values[index])
    if index + 1 < len(values):
        high = 0.5 * (values[index] + values[index + 1])
    return (low, high)


def _outline(
    grid: CrgGrid, component: list[tuple[int, int, int]]
) -> list[tuple[float, float, float]]:
    """A component's outline in ``(u, v)``: along each cross section from its
    first cell to its last (the module's "The outline of a region"), each
    point with the value of the cell it is a corner of."""
    spans: dict[int, tuple[int, int]] = {}
    for row, first, last in component:
        if row in spans:
            low, high = spans[row]
            spans[row] = (min(low, first), max(high, last))
        else:
            spans[row] = (first, last)
    rows = sorted(spans)
    right: list[tuple[float, float, float]] = []
    left: list[tuple[float, float, float]] = []
    for row in rows:
        first, last = spans[row]
        u_low, u_high = _cell_edges(grid.u, row)
        v_low = _cell_edges(grid.v, first)[0]
        v_high = _cell_edges(grid.v, last)[1]
        at_first = grid.values[row][first]
        at_last = grid.values[row][last]
        right.append((u_low, v_low, at_first))
        right.append((u_high, v_low, at_first))
        left.append((u_high, v_high, at_last))
        left.append((u_low, v_high, at_last))
    left.reverse()
    points: list[tuple[float, float, float]] = []
    for point in right + left:
        if points and points[-1][0] == point[0] and points[-1][1] == point[1]:
            continue
        # A run of three on one line keeps only its ends.
        if len(points) >= 2 and (
            points[-2][0] == points[-1][0] == point[0]
            or points[-2][1] == points[-1][1] == point[1]
        ):
            points[-1] = point
            continue
        points.append(point)
    return points


def _on_road(
    reference: CrgReference, u0: float, u: float, v: float
) -> tuple[float, float]:
    """Where a grid point ``(u, v)`` is on the road, ``(s, t)``."""
    along = u - u0 + reference.s_offset
    if reference.orientation == "opposite":
        end = reference.s_end
        if not math.isfinite(end):
            end = reference.s_start
        return (end - along, -v + reference.t_offset)
    return (reference.s_start + along, v + reference.t_offset)


#: How many answers each cache below keeps before it starts again: a session
#: edits one map, so a handful covers a threshold being typed and a file being
#: saved; the bound only stops a long editor session growing without end.
_CACHE_SIZE = 32

#: Regions already read, by the OpenDRIVE file's path, its and its grids'
#: modification times, and what was asked.
_CACHE: dict[str, tuple[list[ReliefRegion], list[str]]] = {}

#: The ``<CRG>``s of an OpenDRIVE file, by its path and modification time:
#: the scenario editor asks on every edit, and the file may be a city.
_REFERENCES: dict[str, list[CrgReference]] = {}

#: The grids read, by their path and modification time: a threshold typed in
#: the editor asks again of the same grids, which are the slow part to read.
_GRIDS: dict[str, CrgGrid] = {}


def _stamp(path: Path) -> str:
    try:
        return f"{path}@{os.stat(str(path)).st_mtime_ns}"
    except OSError:
        return f"{path}@missing"


def relief_regions(
    xodr_path: Path,
    *,
    raised: bool = True,
    grooved: bool = True,
    threshold_m: float = DEFAULT_RELIEF_M,
) -> tuple[list[ReliefRegion], list[str]]:
    """The raised and/or grooved regions of every road of the OpenDRIVE file,
    and what could not be read (a missing or binary grid, a mode that lays its
    own reference line), each said in words and passed over.

    Raises:
        ValueError: A threshold that is not positive.
        OSError: The OpenDRIVE file cannot be read.
    """
    if not threshold_m > 0.0 or not math.isfinite(threshold_m):
        raise ValueError("the threshold of a raised or grooved part must be positive")
    xodr_stamp = _stamp(xodr_path)
    if xodr_stamp in _REFERENCES:
        references = _REFERENCES[xodr_stamp]
    else:
        references = crg_references(xodr_path)
        if len(_REFERENCES) >= _CACHE_SIZE:
            _REFERENCES.clear()
        _REFERENCES[xodr_stamp] = references
    grid_stamps: dict[str, str] = {}
    for reference in references:
        grid_stamps[str(reference.path)] = _stamp(reference.path)
    stamps = [xodr_stamp] + [grid_stamps[str(r.path)] for r in references]
    key = f"{'|'.join(stamps)}|{raised}|{grooved}|{threshold_m!r}"
    if key in _CACHE:
        held = _CACHE[key]
        return (list(held[0]), list(held[1]))
    regions: list[ReliefRegion] = []
    problems: list[str] = []
    for reference in references:
        if reference.purpose != "elevation":
            continue
        if reference.mode not in _ATTACHED:
            problems.append(
                f"road {reference.road_id}: {reference.path.name} is in "
                f"{reference.mode} mode, which lays its own reference line; "
                "only attached grids are read"
            )
            continue
        grid_stamp = grid_stamps[str(reference.path)]
        if grid_stamp not in _GRIDS:
            try:
                read = read_crg(reference.path)
            except (OSError, ValueError) as exc:
                problems.append(f"road {reference.road_id}: {exc}")
                continue
            if len(_GRIDS) >= _CACHE_SIZE:
                _GRIDS.clear()
            _GRIDS[grid_stamp] = read
        grid = _GRIDS[grid_stamp]
        if not grid.u:
            continue
        sides: list[tuple[str, float]] = []
        if raised:
            sides.append((RAISED, 1.0))
        if grooved:
            sides.append((GROOVED, -1.0))
        for kind, sign in sides:
            for component in _components(grid, sign, threshold_m):
                extreme = 0.0
                for row, first, last in component:
                    for column in range(first, last + 1):
                        value = grid.values[row][column]
                        if sign * value > sign * extreme:
                            extreme = value
                outline: list[tuple[float, float]] = []
                relief: list[float] = []
                for u, v, value in _outline(grid, component):
                    outline.append(_on_road(reference, grid.u[0], u, v))
                    relief.append(value)
                regions.append(
                    ReliefRegion(reference.road_id, kind, extreme, outline, relief)
                )
    if len(_CACHE) >= _CACHE_SIZE:
        _CACHE.clear()
    _CACHE[key] = (list(regions), list(problems))
    return (regions, problems)
