"""The hints for a local Codon typed by its first assignment (#583, #584).

Pure AST: no Codon needed.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from autoware_carla_scenario.typecheck.check import (
    Diagnostic,
    _Module,
    _Sources,
    _data_model_hints,
    _none_first_binding_hints,
    _unplaced_none_mismatches,
)
from autoware_carla_scenario.typecheck.local_bindings import (
    int_literal_hint,
    none_first_bindings,
    none_mismatch_hint,
)

_INT = "'float' does not match expected type 'int'"
_OPT_INT = "'Optional[float]' does not match expected type 'Optional[int]'"
_OPT_NONE = "'Optional[float]' does not match expected type 'Optional[NoneType]'"


def _tree(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


def _line(source: str, needle: str) -> int:
    for number, text in enumerate(textwrap.dedent(source).splitlines(), start=1):
        if needle in text:
            return number
    raise AssertionError(needle)


_ACCUMULATOR = """
def mean_speed(speeds: list[float]) -> float:
    total = 0
    for v in speeds:
        total += v
    return total / max(len(speeds), 1)


class Acc:
    def mean(self, speeds: list[float]) -> float:
        acc = -1
        for v in speeds:
            acc = acc + v
        return acc
"""


def test_an_int_literal_first_binding_is_named_with_its_line() -> None:
    hint = int_literal_hint(
        _INT, _line(_ACCUMULATOR, "total += v"), _tree(_ACCUMULATOR)
    )
    assert hint is not None
    assert "`total = 0` on line 3" in hint
    assert "`total = 0.0`" in hint


def test_an_int_literal_first_binding_in_a_method() -> None:
    hint = int_literal_hint(
        _INT, _line(_ACCUMULATOR, "acc = acc + v"), _tree(_ACCUMULATOR)
    )
    assert hint is not None
    assert "`acc = -1` on line 11" in hint
    assert "`acc = -1.0`" in hint


_CONDITIONAL = """
def floor_speed(v: float | None) -> float:
    s = v if v is not None else 0
    return s * 2.0


def other(v: float | None) -> float:
    return f(v if v is not None else 0)
"""


def test_an_int_literal_branch_of_a_conditional_is_named() -> None:
    tree = _tree(_CONDITIONAL)
    hint = int_literal_hint(_OPT_INT, _line(_CONDITIONAL, "s = v"), tree)
    assert hint is not None
    assert "`s`" in hint and "`0`" in hint and "`0.0`" in hint
    unassigned = int_literal_hint(_OPT_INT, _line(_CONDITIONAL, "return f("), tree)
    assert unassigned is not None and "`0.0`" in unassigned


def test_no_int_literal_hint_for_other_lines_or_messages() -> None:
    source = """
    def f(xs: list[float], n: int) -> float:
        total: int = 0
        total += xs[0]
        count = n
        count += xs[0]
        flag = True
        flag += xs[0]
        return 0.0
    """
    tree = _tree(source)
    # An annotated int, a first binding of no literal, a bool literal.
    for needle in ("total += xs", "count += xs", "flag += xs"):
        assert int_literal_hint(_INT, _line(source, needle), tree) is None
    # Another refusal on a line that would match.
    acc = _tree(_ACCUMULATOR)
    line = _line(_ACCUMULATOR, "total += v")
    assert (
        int_literal_hint("'str' does not match expected type 'float'", line, acc)
        is None
    )


_RUNNING_MIN = """
def seen_low(xs: list[float]) -> bool:
    best = None
    for x in xs:
        if best is None or x < best:
            best = x
    return best is not None


def fine(xs: list[float], d: str) -> None:
    annotated: float | None = None
    annotated = 1.0
    reset = None
    reset = None
    label = None
    label = d
    other = None
    other = compute()
"""


def test_a_local_first_bound_to_none_and_rebound_is_found() -> None:
    found = {b.name: b for b in none_first_bindings(_tree(_RUNNING_MIN))}
    assert set(found) == {"best", "label", "other"}
    best = found["best"]
    assert best.line == _line(_RUNNING_MIN, "best = None")
    assert best.rebound == (_line(_RUNNING_MIN, "best = x"),)
    assert best.annotation() == "best: float | None = None"
    assert found["label"].annotation() == "label: str | None = None"
    assert found["other"].annotation() == "other: <T> | None = None"


def test_the_none_mismatch_names_the_first_binding() -> None:
    tree = _tree(_RUNNING_MIN)
    found = none_mismatch_hint(_OPT_NONE, _line(_RUNNING_MIN, "best = x"), tree)
    assert found is not None
    message, line = found
    assert line == _line(_RUNNING_MIN, "best = None")
    assert "annotate it: `best: float | None = None`" in message
    listed = none_mismatch_hint(
        "'Optional[List[int]]' does not match expected type 'Optional[NoneType]'",
        _line(_RUNNING_MIN, "best = x"),
        tree,
    )
    assert listed is not None and "`best: list[int] | None = None`" in listed[0]
    assert none_mismatch_hint(_OPT_NONE, 1, tree) is None
    assert none_mismatch_hint(_INT, _line(_RUNNING_MIN, "best = x"), tree) is None


def _sources(tmp_path: Path, source: str) -> _Sources:
    path = tmp_path / "scenario.py"
    text = textwrap.dedent(source)
    path.write_text(text)
    return _Sources(
        modules={"pkg.scenario": _Module(path, False, text, ast.parse(text))}
    )


def test_codon_refusals_get_the_hint_through_data_model_hints(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path, _ACCUMULATOR)
    path = str(next(iter(sources.modules.values())).path)
    line = _line(_ACCUMULATOR, "total += v")
    (out,) = _data_model_hints([Diagnostic(_INT, path, line, 9)], sources)
    assert out.message.startswith(_INT + ": Codon fixes the type of the local")
    # A file the check did not read gets no hint.
    (other,) = _data_model_hints([Diagnostic(_INT, "/elsewhere.py", line)], sources)
    assert other.message == _INT


def test_a_crash_or_an_unplaced_none_mismatch_lists_the_none_locals(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path, _RUNNING_MIN)
    hints = _none_first_binding_hints(sources, "Codon crashed (exit status -11)")
    assert [d.line for d in hints] == [
        _line(_RUNNING_MIN, "best = None"),
        _line(_RUNNING_MIN, "label = None"),
        _line(_RUNNING_MIN, "other = None"),
    ]
    assert hints[0].message.startswith("Codon crashed (exit status -11); ")
    assert "`best: float | None = None`" in hints[0].message

    unplaced = Diagnostic(_OPT_NONE, "/elsewhere.py", 3)
    assert len(_unplaced_none_mismatches([unplaced], sources)) == 3
    placed = Diagnostic(f"{_OPT_NONE}: Codon fixes the type of the local `best`")
    assert _unplaced_none_mismatches([placed], sources) == []
    assert _unplaced_none_mismatches([Diagnostic(_INT)], sources) == []
