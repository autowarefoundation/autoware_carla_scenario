"""The rewrites of valid Python Codon refuses (``typecheck/python_rewrites.py``).

None of these need Codon.  A rewrite keeps every line where it was and, where
CPython can run what it writes, the value of what it replaces (#2517).
"""

from __future__ import annotations

import ast
import textwrap
import types

import pytest

from autoware_carla_scenario.typecheck.python_rewrites import (
    python_rewrite_refusals,
    python_rewrites,
)

_REBOUND = "is assigned again later in the body of class"


def _rewritten(source: str) -> str:
    source = textwrap.dedent(source)
    text, _ = python_rewrites(source)
    assert text.count("\n") == source.count("\n")
    ast.parse(text)
    return text


def _run(source: str) -> object:
    """What *source* binds to ``r``, given a few names to read."""
    namespace: dict[str, object] = {
        "a": [0, 1, "", 2],
        "b": [3, 0],
        "c": True,
        "d": {"x": 1},
        "k": 5,
        "x": 7,
        "foo": types.SimpleNamespace(bar=3.5),
    }
    exec(source, namespace)  # noqa: S102 - the test's own snippets
    return namespace["r"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # A node pasted without the parentheses the source held it in (#2517).
        (
            "r = list(filter(None, a if c else b))\n",
            "r = list((_acs_v for _acs_v in (a if c else b) if _acs_v))\n",
        ),
        (
            "r = list(filter(None, (ys := a))); r = (r, ys)\n",
            "r = list((lambda _acs_xs: (_acs_v for _acs_v in _acs_xs if _acs_v))"
            "((ys := a))); r = (r, ys)\n",
        ),
        (
            'r = {**d, "k": (y := 2)}; r = (r, y)\n',
            'r = (d | {"k": (y := 2)}); r = (r, y)\n',
        ),
        (
            'r = {(z := "k"): 2, **d}; r = (r, z)\n',
            'r = ({(z := "k"): 2} | d); r = (r, z)\n',
        ),
        (
            "f = lambda x, k=k: (y := k + x); r = f(1)\n",
            "f = (lambda k: lambda x: (y := k + x))(k); r = f(1)\n",
        ),
        (
            'r = "{} {}".format((w := 3), w)\n',
            'r = (lambda _acs_f0, _acs_f1: f"{_acs_f0} {_acs_f1}")((w := 3), w)\n',
        ),
        # An int literal's attribute (#2517).
        ('r = "{0.real}".format(1)\n', 'r = f"{(1).real}"\n'),
        ('r = "{0.imag}{1}".format(2, x)\n', 'r = f"{(2).imag}{x}"\n'),
        # A value on two lines: never a line break in an f-string (#2517).
        (
            'r = "{}".format(foo\n    .bar)\n',
            'r = (lambda _acs_f0: f"{_acs_f0}")(foo\n    .bar)\n',
        ),
        (
            'r = "{}-{:>8}".format(x, foo.\n    bar)\n',
            'r = (lambda _acs_f0, _acs_f1: f"{_acs_f0}-{_acs_f1:>8}")(x, foo.\n'
            "    bar)\n",
        ),
        # A lone surrogate is written escaped (#2517).
        ('r = "\\ud800{}".format(x)\n', 'r = f"\\ud800{x}"\n'),
        ('r = f"\\ud800{x:+}"\n', 'r = ("\\ud800" + (x).__format__("+"))\n'),
        ('r = f"{x:\\ud800>5}"\n', 'r = ((x).__format__("\\ud800>5"))\n'),
        # `zip` the module defines keeps its keyword (#2517).
        (
            "def zip(a, b, strict=False):\n    return strict\nr = zip(1, 2, strict=True)\n",
            "def zip(a, b, strict=False):\n    return strict\nr = zip(1, 2, strict=True)\n",
        ),
        ("r = list(zip([1], [2], strict=True))\n", "r = list(zip([1], [2]))\n"),
        # The line breaks of a JSON isinstance stay inside its brackets.
        (
            "import json\nif isinstance(\n        x, int):\n    r = 1\n",
            "import json\nif (isinstance(x, int) or (isinstance(x, json.JSONValue)"
            " and (x)._acs_isjson())\n):\n    r = 1\n",
        ),
    ],
)
def test_a_rewrite_keeps_the_value_and_the_lines(source: str, expected: str) -> None:
    assert _rewritten(source) == expected
    if "json" not in source:
        assert _run(expected) == _run(source)


@pytest.mark.parametrize(
    ("source", "expected", "sent"),
    [
        (
            "def g():\n"
            '    r = {**d, "k": (yield)}\n'
            "    yield r\n"
            "it = g(); next(it); r = it.send(9)\n",
            '    r = (d | {"k": (yield)})\n',
            9,
        ),
        (
            "def g():\n"
            "    f = lambda k=k: (yield k)\n"
            "    yield f\n"
            "it = g(); r = list(next(it)())\n",
            "    f = (lambda k: lambda: (yield k))(k)\n",
            None,
        ),
        (
            "def g():\n"
            '    r = "{}".format((yield))\n'
            "    yield r\n"
            "it = g(); next(it); r = it.send(9)\n",
            '    r = (lambda _acs_f0: f"{_acs_f0}")((yield))\n',
            9,
        ),
    ],
)
def test_a_pasted_yield_keeps_its_parentheses(
    source: str, expected: str, sent: object
) -> None:
    text = _rewritten(source)
    assert text.splitlines(keepends=True)[1] == expected
    assert _run(text) == _run(source)


def test_a_rewrite_writes_no_fstring_syntax_of_python_3_12() -> None:
    """No line break in a single-quoted f-string (Codon's grammar, and
    Python's before 3.12)."""
    for source in ('"{}".format(foo\n    .bar)\n', '"{0}{0}".format(a\n.b)\n'):
        for node in ast.walk(ast.parse(_rewritten(source))):
            if isinstance(node, ast.JoinedStr):
                assert node.lineno == node.end_lineno


@pytest.mark.parametrize(
    "spec",
    ["{:\\ud800<5}", "{0:\\udfff^3}"],
)
def test_a_format_spec_with_a_lone_surrogate_is_refused(spec: str) -> None:
    source = f'r = "{spec}".format(x)\n'
    assert _rewritten(source) == source
    assert any("lone surrogate" in m for _, m in python_rewrite_refusals(source))


@pytest.mark.parametrize(
    "source",
    [
        """\
        class C:
            m = None
            v = m.x if m else None
            w = m.x if m is not None else None
        """,
        """\
        class C:
            m = None
            v = None if m is None else m.x
        """,
    ],
)
def test_a_class_constant_narrowed_gets_the_whole_rewrite(source: str) -> None:
    """The ``isinstance`` test is written although the class constant's own
    edit had the test's span (#597, #2517)."""
    text = _rewritten(source)
    for line in text.splitlines()[2:]:
        assert line.count("_acs_some(") == 1
        assert "isinstance(C.m, Optional[NoneType])" in line
        assert " m" not in line.replace(" m =", "")


@pytest.mark.parametrize(
    "where",
    [
        "    def f(self, v=m.x if m else None):\n        pass\n",
        "    @deco(m.x if m else None)\n    def f(self):\n        pass\n",
    ],
)
def test_a_narrowed_branch_is_wrapped_once(where: str) -> None:
    text = _rewritten("class C:\n    m = None\n" + where)
    assert text.count("_acs_some_if(") == 1
    assert (
        "_acs_some_if(lambda: C.m.x, C.m) if not isinstance(C.m, "
        "Optional[NoneType]) else None" in text
    )


@pytest.mark.parametrize(
    "later",
    [
        "D = [A for A in range(3)]",
        "D = {A for A in range(3)}",
        "D = {A: 0 for A in range(3)}",
        "D = tuple(A for A in range(3))",
        "D = lambda A: A",
    ],
)
def test_a_comprehension_variable_is_no_rebinding(later: str) -> None:
    source = f"class C:\n    A = 1\n    B = A + 1\n    {later}\n"
    assert "B = C.A + 1" in _rewritten(source)
    assert not any(_REBOUND in m for _, m in python_rewrite_refusals(source))


@pytest.mark.parametrize(
    "later",
    [
        "match 0:\n        case A:\n            pass",
        "match 0:\n        case [*A]:\n            pass",
        "match 0:\n        case {**A}:\n            pass",
        "for A in range(3):\n        pass",
        "A = 2",
    ],
)
def test_a_class_constant_rebound_later_is_refused(later: str) -> None:
    source = f"class C:\n    A = 1\n    B = A\n    {later}\n"
    assert any(
        line == 3 and _REBOUND in message
        for line, message in python_rewrite_refusals(source)
    )
