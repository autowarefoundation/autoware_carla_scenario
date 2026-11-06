"""The package's ``Error:`` lines go to stderr (#1810).

A CLI that prints its refusal on stdout leaves stderr empty although it exits
non-zero, so a wrapper that keeps the two streams apart (a shell redirect, the
viewer's subprocess runs, a CI log) shows no reason.  Every ``print`` whose
first argument starts with ``Error`` must pass ``file=sys.stderr``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import autoware_carla_scenario

_PACKAGE = Path(autoware_carla_scenario.__file__).resolve().parent


def _leading_text(node: ast.expr) -> str | None:
    """The literal text *node*, a ``print`` argument, starts with, if any."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _writes_to_stderr(call: ast.Call) -> bool:
    return any(
        keyword.arg == "file"
        and isinstance(keyword.value, ast.Attribute)
        and keyword.value.attr == "stderr"
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "sys"
        for keyword in call.keywords
    )


def _error_prints_on_stdout(source: str) -> list[int]:
    """The lines of *source* with an ``Error`` ``print`` lacking stderr."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and node.args
        ):
            continue
        text = _leading_text(node.args[0])
        if (
            text is not None
            and text.startswith("Error")
            and not _writes_to_stderr(node)
        ):
            lines.append(node.lineno)
    return sorted(lines)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('print("Error: x")', [1]),
        ('print(f"Error: {x}")', [1]),
        ('print(\n    "Error: a "\n    "b"\n)', [1]),
        ('print("Error: x", file=sys.stdout)', [1]),
        ('print("Error: x", file=sys.stderr)', []),
        ('print(f"Error: {x}", file=sys.stderr)', []),
        ('print("Done")', []),
        ('print(f"{x} Error")', []),
        ("print()", []),
    ],
)
def test_the_guard_finds_an_error_print_on_stdout(
    source: str, expected: list[int]
) -> None:
    assert _error_prints_on_stdout(source) == expected


def test_no_error_line_of_the_package_goes_to_stdout() -> None:
    offenders = [
        f"{path.relative_to(_PACKAGE.parent)}:{line}"
        for path in sorted(_PACKAGE.rglob("*.py"))
        for line in _error_prints_on_stdout(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "print an 'Error' line with file=sys.stderr: " + ", ".join(
        offenders
    )
