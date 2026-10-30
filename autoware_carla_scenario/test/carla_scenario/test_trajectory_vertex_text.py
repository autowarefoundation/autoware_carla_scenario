"""trajectory.vertex_text, split out of trajectory.authoring so the library
check compiles it (#774): what authoring did with the editor's text and with
parsed rows gives the same values, and raises the same errors, as before.

The ``_master_*`` functions are those parts of ``trajectory/authoring.py`` as
they stood before the split (at d689cc29), copied verbatim but for their
names; each case runs them and the split code on the same input and compares
the results, or the exception's type and message.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import pytest

from autoware_carla_scenario.trajectory import authoring, vertex_text

# ---------------------------------------------------------------------------
# trajectory/authoring.py before the split
# ---------------------------------------------------------------------------


def _master_no_time(index: int, time: Any) -> ValueError:
    return ValueError(
        f"vertex {index}: a vertex row has no time cell any more; give vertex "
        f"{index} a Trajectory time waypoint condition instead "
        f"(advance_conditions: - vertex: {index}, condition: {{type: "
        f"trajectory_time, params: {{time: {time}}}}})"
    )


def _master_rows(value: Any) -> list[Any]:
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


def _master_number(index: int, cell: Any, name: str, required: bool) -> Optional[float]:
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


def _master_cell(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.10g}"


def _master_vertex_problems(rows: Sequence[Any]) -> list[str]:
    if len(rows) < 2:
        return [f"needs at least two vertices, has {len(rows)}"]
    return []


def _master_trajectory_summary(rows: list[Any]) -> str:
    # trajectory_summary after parse_vertices.
    if not rows:
        return "no vertices"
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(rows, rows[1:]))
    return f"{len(rows)} vertices, {length:.1f} m"


def _master_relative_trajectory_summary(rows: list[Any]) -> str:
    # relative_trajectory_summary after parse_relative_vertices.
    if not rows:
        return "no vertices"
    first, last = rows[0], rows[-1]
    lanes = sorted({row[2] for row in rows})
    text = f"{len(rows)} vertices, ds {first[0]:g} to {last[0]:g} m"
    if lanes != [0]:
        text += ", lanes " + ", ".join(f"{lane:+d}" for lane in lanes)
    return text


def _master_route_mapping(index: int, row: str) -> dict[str, Any]:
    # The text branch of _route_row.
    tokens = row.replace(",", " ").split()
    if not tokens:
        raise ValueError(f"vertex {index}: empty")
    mapping: dict[str, Any] = {"kind": tokens[0]}
    for token in tokens[1:]:
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(
                f"vertex {index}: {token!r} is not key=value (write e.g. "
                "'lane ds=10 d_lane=1')"
            )
        mapping[key.strip()] = value.strip()
    return mapping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    """What *call* returned, or the type and message of what it raised."""
    try:
        return ("value", call())
    except Exception as exc:  # noqa: BLE001 - compared, not handled
        return (type(exc).__name__, str(exc))


def _same(left: Any, right: Any) -> bool:
    """Equal, NaN included, and of the same types."""
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_same(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


_TEXTS = [
    "",
    "\n\n",
    "# only a comment",
    "1, 2",
    "1,2,3",
    "  1 ,  2 , 0.5  \n# c\n\n3, 4",
    "1, 2, \n3, 4,",
    "1, 2, 3, 4",
    ",,,",
    "a, b",
    " # not a comment? yes, stripped first",
    "1\r\n2\r\n",
    "x,\ty ,z",
    "1, 2 # trailing",
]

_CELLS = [
    "",
    " ",
    "\t\n",
    "0",
    "-0",
    "1.5",
    " 2 ",
    "1e3",
    "1_000",
    "+7",
    "inf",
    "-Infinity",
    "nan",
    "NaN",
    "abc",
    "1,5",
    "0x10",
    "١٢",
    "1e400",
    "True",
]


# ---------------------------------------------------------------------------
# The moved helpers, against master's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("time", [0, 1.5, "3", "", None, True, [1, 2], {"t": 1}])
@pytest.mark.parametrize("index", [1, 7])
def test_no_time(index: int, time: Any) -> None:
    expected = _master_no_time(index, time)
    for got in (authoring._no_time(index, time), vertex_text.no_time(index, f"{time}")):
        assert type(got) is ValueError
        assert str(got) == str(expected)


@pytest.mark.parametrize("text", _TEXTS)
def test_text_rows(text: str) -> None:
    assert vertex_text.text_rows(text) == _master_rows(text)
    assert authoring._rows(text) == _master_rows(text)


@pytest.mark.parametrize("value", [None, [], [[1, 2]], ([1, 2], {"x": 1}), "1, 2"])
def test_rows_of_document_values(value: Any) -> None:
    assert authoring._rows(value) == _master_rows(value)


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("cell", _CELLS)
def test_text_number(cell: str, required: bool) -> None:
    expected = _outcome(lambda: _master_number(3, cell, "yaw", required))
    for got in (
        _outcome(lambda: vertex_text.text_number(3, cell, "yaw", required)),
        _outcome(lambda: authoring._number(3, cell, "yaw", required)),
    ):
        assert got[0] == expected[0]
        assert _same(got[1], expected[1])


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize(
    "cell",
    [None, 0, 1, -2.5, 1e308, math.inf, math.nan, True, False, [1], {"a": 1}, b"1"],
)
def test_number_of_document_values(cell: Any, required: bool) -> None:
    expected = _outcome(lambda: _master_number(2, cell, "x", required))
    got = _outcome(lambda: authoring._number(2, cell, "x", required))
    assert got[0] == expected[0]
    assert _same(got[1], expected[1])


@pytest.mark.parametrize(
    "value",
    [None, 0.0, -0.0, 1.0, 0.1, 1 / 3, 123456789.123456789, 1e-12, 1e20, -2.5, 1e300],
)
def test_cell_text(value: Optional[float]) -> None:
    assert vertex_text.cell_text(value) == _master_cell(value)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 10])
def test_vertex_count_problems(count: int) -> None:
    rows = [(0.0, 0.0, None)] * count
    assert vertex_text.vertex_count_problems(count) == _master_vertex_problems(rows)
    assert authoring.vertex_problems(rows) == _master_vertex_problems(rows)
    assert authoring.relative_vertex_problems(rows) == _master_vertex_problems(rows)


_VERTEX_ROWS: list[list[vertex_text.VertexRow]] = [
    [],
    [(0.0, 0.0, None)],
    [(0.0, 0.0, None), (3.0, 4.0, 0.5)],
    [(0.1, 0.2, None), (0.3, 0.4, None), (0.5, 0.6, None), (-7.0, 11.25, 1.0)],
    [(81234.5, 50123.0, None), (81240.25, 50130.5, None), (81250.0, 50131.0, 1.5)],
    [(float(i) * 0.37, float(i * i) * 0.11, None) for i in range(50)],
    [(1.0, 1.0, None)] * 5,
    [(0.04, 0.0, None), (0.0, 0.0, None)],
    [(0.05, 0.0, None), (0.0, 0.0, None)],
]


@pytest.mark.parametrize("rows", _VERTEX_ROWS)
def test_vertices_summary(rows: list[vertex_text.VertexRow]) -> None:
    assert vertex_text.vertices_summary(rows) == _master_trajectory_summary(rows)


_RELATIVE_ROWS: list[list[vertex_text.RelativeVertexRow]] = [
    [],
    [(0.0, 0.0, 0, None)],
    [(0.0, 0.0, 0, None), (30.0, 0.0, 0, None)],
    [(0.0, 0.0, 0, None), (10.0, 0.0, 1, None), (20.5, 0.5, 1, 0.1)],
    [(5.0, 0.0, -1, None), (2.5, 0.0, 2, None), (1e-5, 0.0, -12, None)],
    [(0.0, 0.0, 1, None)],
    [(1234567.0, 0.0, 0, None), (0.000123456, 0.0, -1, None)],
    [(-3.0, 0.0, 0, None), (-0.0, 0.0, 0, None), (1.0, 0.0, 10, None)],
]


@pytest.mark.parametrize("rows", _RELATIVE_ROWS)
def test_relative_vertices_summary(rows: list[vertex_text.RelativeVertexRow]) -> None:
    expected = _master_relative_trajectory_summary(rows)
    assert vertex_text.relative_vertices_summary(rows) == expected


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        ",,",
        "lane",
        "lane ds=10 d_lane=1",
        "lane,ds=10,d_lane=1",
        "lane ds=10, ds=20",
        "lane kind=opposite ds=1",
        "lane ds=",
        "lane =5",
        "lane ds==3",
        "lane a=b=c",
        "lane ds 10",
        "crossing junction=0 approach=north turn=left",
        "\tlane\tds=1　offset=2",
        "x y=1 z",
    ],
)
@pytest.mark.parametrize("index", [1, 4])
def test_route_tokens(index: int, line: str) -> None:
    expected = _outcome(lambda: _master_route_mapping(index, line))

    def mapping() -> dict[str, Any]:
        kind, pairs = vertex_text.route_tokens(index, line)
        out: dict[str, Any] = {"kind": kind}
        for key, value in pairs:
            out[key] = value
        return out

    got = _outcome(mapping)
    assert got[0] == expected[0]
    assert got[1] == expected[1]
    if got[0] == "value":
        assert list(got[1]) == list(expected[1])  # the same key order


# ---------------------------------------------------------------------------
# authoring still exports the same rows and summaries
# ---------------------------------------------------------------------------


def test_authoring_exports_the_row_types_of_vertex_text() -> None:
    assert authoring.VertexRow is vertex_text.VertexRow
    assert authoring.RelativeVertexRow is vertex_text.RelativeVertexRow


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "no vertices"),
        ("0, 0\n3, 4", "2 vertices, 5.0 m"),
        ("0, 0, 0, 5", None),
        ("0, 0\nx, 1", "vertex 2: 'x' is not a number"),
    ],
)
def test_trajectory_summary(text: str, expected: Optional[str]) -> None:
    got = authoring.trajectory_summary(text)
    if expected is None:
        assert got == str(_master_no_time(1, "5"))
    else:
        assert got == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "no vertices"),
        ("0\n20", "2 vertices, ds 0 to 20 m"),
        ("0, 0, -1\n12.5, 0, 2", "2 vertices, ds 0 to 12.5 m, lanes -1, +2"),
        ("0, 0, 0\n5, 0, 1", "2 vertices, ds 0 to 5 m, lanes +0, +1"),
        (
            "0, 0, 0.5",
            "vertex 1: d_lane is a number of lanes, a whole number, got '0.5'",
        ),
    ],
)
def test_relative_trajectory_summary(text: str, expected: str) -> None:
    assert authoring.relative_trajectory_summary(text) == expected


def test_route_rows_from_text() -> None:
    rows = authoring.parse_route_vertices("lane ds=10, d_lane=1\n# c\nlane ds=20")
    assert [row["kind"] for row in rows] == ["lane", "lane"]
    assert rows[0]["ds"] == 10.0 and rows[0]["d_lane"] == 1
    with pytest.raises(ValueError, match=r"^vertex 1: 'ds' is not key=value"):
        authoring.parse_route_vertices("lane ds")
    with pytest.raises(ValueError, match=r"^vertex 1: kind must be one of"):
        authoring.parse_route_vertices("lane kind=nope")
