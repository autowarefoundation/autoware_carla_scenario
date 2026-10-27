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
    _narrowed_none_first_hints,
    _none_first_binding_hints,
    _unplaced_none_mismatches,
)
from autoware_carla_scenario.typecheck.local_bindings import (
    int_literal_hint,
    narrowed_none_first_bindings,
    none_first_bindings,
    none_mismatch_hint,
    none_only_tests,
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


def test_an_int_literal_hint_needs_a_float_and_no_parameter() -> None:
    source = """
    def f(n: int, s: str, m=0) -> int:
        k = 0
        k = s
        n = 0
        n = 1.5
        m = 1.5
        return k
    """
    tree = _tree(source)
    # Codon refusing a str: storing a float is not the fix.
    assert (
        int_literal_hint(
            "'str' does not match expected type 'int'", _line(source, "k = s"), tree
        )
        is None
    )
    # A parameter is typed by its annotation or caller, not by `n = 0`.
    assert int_literal_hint(_INT, _line(source, "n = 1.5"), tree) is None
    assert int_literal_hint(_INT, _line(source, "m = 1.5"), tree) is None


def test_an_int_literal_hint_with_a_call_says_which_float_it_means() -> None:
    # Codon reports `half(y)`'s float argument on this line too.
    source = """
    def use(y: float) -> int:
        k = 0
        k = half(y)
        return k
    """
    hint = int_literal_hint(_INT, _line(source, "k = half"), _tree(source))
    assert hint is not None
    assert hint.startswith("if the float is the value stored in `k` here, ")
    plain = int_literal_hint(
        _INT, _line(_ACCUMULATOR, "total += v"), _tree(_ACCUMULATOR)
    )
    assert plain is not None and plain.startswith("Codon fixes")


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
    # Not wrapped in Optional: the type is kept whole.
    for codon, python in (("List[float]", "list[float]"), ("float", "float")):
        bare = none_mismatch_hint(
            f"'{codon}' does not match expected type 'Optional[NoneType]'",
            _line(_RUNNING_MIN, "best = x"),
            tree,
        )
        assert bare is not None and f"`best: {python} | None = None`" in bare[0]
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


# ---------------------------------------------------------------------------
# A None-first local a conditional expression tests (#629)
# ---------------------------------------------------------------------------

_NARROWED = """
def deltas(xs: list[float]) -> list:
    out = []
    prev = None
    for v in xs:
        d = v - prev if prev is not None else None
        out.append(d)
        prev = v
    return out


def none_only():
    z = None
    w = None
    w = None
    return (z + 1 if z is not None else None, 1 if w is None else 2)


def not_tested(n):
    q = None
    k: None = None
    return 1 if n else (2 if k is None else 3)


def parameter(p=None):
    return 1 if p is None else 2
"""


def test_a_local_only_ever_none_tested_by_a_conditional_is_found() -> None:
    found = none_only_tests(_tree(_NARROWED))
    assert [(t.name, t.line, t.test_line) for t in found] == [
        ("z", _line(_NARROWED, "z = None"), _line(_NARROWED, "return (z + 1")),
        ("w", _line(_NARROWED, "w = None"), _line(_NARROWED, "return (z + 1")),
    ]
    assert "`z: <T> | None = None`" in found[0].message()


def test_a_none_first_local_tested_by_a_conditional_is_named(tmp_path: Path) -> None:
    (binding,) = narrowed_none_first_bindings(_tree(_NARROWED))
    assert (binding.name, binding.line) == ("prev", _line(_NARROWED, "prev = None"))
    sources = _sources(tmp_path, _NARROWED)
    made_up = "'Optional[int]' does not match expected type 'Optional[float]'"
    (hint,) = _narrowed_none_first_hints([Diagnostic(made_up)], sources)
    assert hint.line == binding.line
    assert "`prev: float | None = None`" in hint.message
    # Any other refusal is not Codon's made-up int.
    assert _narrowed_none_first_hints([Diagnostic(_OPT_INT)], sources) == []
    assert _narrowed_none_first_hints([Diagnostic(_INT)], sources) == []
