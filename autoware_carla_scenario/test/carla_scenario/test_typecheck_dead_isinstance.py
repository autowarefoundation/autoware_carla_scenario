"""The dead ``isinstance`` warning's AST pass (#565); it needs no Codon."""

from __future__ import annotations

import ast
import textwrap

import pytest

from autoware_carla_scenario.typecheck.dead_isinstance import dead_isinstances


def _found(source: str) -> list[str]:
    tree = ast.parse(textwrap.dedent(source))
    return [d.message.split(":")[0] for d in dead_isinstances(tree)]


def test_the_three_confirmed_shapes_are_found() -> None:
    assert _found(
        """
        class S:
            speed: float

            def m(self, r: float) -> None:
                x = 5
                if isinstance(x, str):
                    pass
                if isinstance(self.speed, int):
                    pass
                if isinstance(r, int):
                    pass
        """
    ) == [
        "isinstance(x, str)",
        "isinstance(self.speed, int)",
        "isinstance(r, int)",
    ]


def test_the_message_says_the_branch_is_not_checked() -> None:
    (found,) = dead_isinstances(
        ast.parse("def f() -> None:\n    x = 5\n    isinstance(x, str)\n")
    )
    assert found.line == 3
    assert "x is statically int here" in found.message
    assert "not checked by the static check" in found.message


@pytest.mark.parametrize(
    "source",
    [
        # bool is an int
        "def f(d: bool) -> None:\n    isinstance(d, int)\n",
        "def f(d: int) -> None:\n    isinstance(d, bool)\n",
        # one of a tuple may match
        "def f(d: int) -> None:\n    isinstance(d, (str, int))\n",
        # rebound: the static type is not the annotation
        "def f(d: int) -> None:\n    d = 'a'\n    isinstance(d, str)\n",
        "def f() -> None:\n    x = 1\n    x = 'a'\n    isinstance(x, str)\n",
        # a union, an unknown class, a nested function's own parameter
        "def f(d: int | str) -> None:\n    isinstance(d, str)\n",
        "def f(d: int) -> None:\n    isinstance(d, Unknown)\n",
        "def f(d: int) -> None:\n    def g(d):\n        isinstance(d, str)\n",
        # a subclass of the subject's class
        "class A: ...\nclass B(A): ...\ndef f(a: A) -> None:\n    isinstance(a, B)\n",
        # an unannotated self attribute
        "class S:\n    def m(self) -> None:\n        isinstance(self.x, str)\n",
    ],
)
def test_valid_code_is_not_flagged(source: str) -> None:
    assert dead_isinstances(ast.parse(source)) == []
