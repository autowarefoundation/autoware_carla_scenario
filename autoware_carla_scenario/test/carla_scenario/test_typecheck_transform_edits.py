"""The source edits of the transform keep the source valid (#2521).

Each case was once turned into source Python cannot parse, or crashed the
transform, or typed a field otherwise than its siblings.  None of these need
Codon: they pin down what reaches the compiler, and that every line stays on
its line (the origin map is the identity, but for moved decorators).
"""

from __future__ import annotations

import ast
import textwrap

from autoware_carla_scenario.typecheck.library_driver import render_library_checks
from autoware_carla_scenario.typecheck.transform import (
    PRELUDE,
    codon_annotation,
    init_false_refusals,
    transform_source_lines,
)


def _transform(source: str, *, in_place: bool = True) -> tuple[list[str], list[int]]:
    """The transformed lines of *source* (the prelude left out), checked to
    parse and, *in_place*, to keep every line of the module where it is (an
    Enum's methods are inserted after it instead), and the origin map."""
    source = textwrap.dedent(source)
    out, origin = transform_source_lines(source)
    ast.parse(out)
    assert out.startswith(PRELUDE)
    lines = out[len(PRELUDE) :].splitlines()
    assert len(origin) == len(lines)
    # Each line of the module stays a line of it (what follows is appended).
    count = len(source.splitlines())
    if in_place:
        assert sorted(origin[:count]) == list(range(1, count + 1))
    return lines, origin


def _module_lines(source: str) -> list[str]:
    """The transformed lines of the module itself, not what is appended."""
    lines, _ = _transform(source)
    return lines[: len(textwrap.dedent(source).splitlines())]


def _annotation(text: str, *, class_level: bool = False) -> str | None:
    return codon_annotation(ast.parse(text, mode="eval").body, class_level=class_level)


# A. A local annotation that is dropped.


def test_a_dropped_annotation_keeps_the_parentheses_around_the_value() -> None:
    lines = _module_lines(
        """\
        from typing import Any


        def f(rules: dict[str, int]) -> int:
            doc: dict[str, Any] = (
                {k: v for k, v in rules.items()}
                if rules
                else {}
            )
            return len(doc)
        """
    )
    assert lines[4] == "    doc = ("
    assert lines[5].strip() == "{k: v for k, v in rules.items()}"
    assert lines[8] == "    )"


def test_a_dropped_annotation_over_lines_puts_its_breaks_in_parentheses() -> None:
    lines = _module_lines(
        """\
        from typing import Union


        def f(x: int) -> None:
            w: Union[
                int, str
            ] = x
            print(w)
        """
    )
    assert lines[4:7] == ["    w = (", "", " x)"]
    assert lines[7] == "    print(w)"


def test_a_dropped_annotation_on_one_line_is_unchanged() -> None:
    lines = _module_lines(
        """\
        from typing import Union


        def f(x: int) -> None:
            w: Union[int, str] = x
            print(w)
        """
    )
    assert lines[4] == "    w = x"


# B. A parenthesised return annotation that is dropped.


def test_a_dropped_parenthesised_return_annotation_takes_its_parentheses() -> None:
    lines = _module_lines(
        """\
        from typing import Any


        def probes() -> (
            dict[str, Any]
        ):
            return {}
        """
    )
    assert lines[3:6] == ["def probes(", "", "):"]


# C. A multi-line annotation whose Codon spelling has no closing bracket.


def test_a_literal_over_lines_is_put_in_parentheses() -> None:
    lines = _module_lines(
        """\
        from typing import Literal


        def f(
            m: Literal[
                "a",
                "b",
            ],
        ) -> Literal[
            "a",
            "b",
        ]:
            k: Literal[
                "a",
                "b",
            ] = m
            return k
        """
    )
    assert lines[4] == "    m: (str"
    assert lines[7] == "),"
    assert lines[8] == ") -> (str"
    assert lines[11] == "):"
    assert lines[12] == "    k: (str"
    assert lines[15] == ") = m"


def test_annotated_and_final_over_lines_are_put_in_parentheses() -> None:
    lines = _module_lines(
        """\
        from dataclasses import dataclass
        from typing import Annotated, Final


        @dataclass
        class C:
            speed: Annotated[
                float, "m/s"
            ] = 1.0


        LIMIT: Final[
            int
        ] = 3
        """
    )
    assert lines[6:9] == ["    speed: (float", "", ") = 1.0"]
    assert lines[11:14] == ["LIMIT: (int", "", ") = 3"]


# D. dataclasses.replace with parenthesised keyword values.


def test_replace_keeps_the_parentheses_of_a_keyword_value() -> None:
    lines = _module_lines(
        """\
        from dataclasses import dataclass, replace


        @dataclass
        class C:
            a: float = 0.0
            b: int = 0


        def f(c: C, k: float) -> C:
            return replace(c, a=(k * 2.0), b=1)


        def g(c: C) -> C:
            return replace(c, a=(  # comment, here
                1.0
            ))
        """
    )
    assert lines[10] == "    return replace(c)._acs_with__a((k * 2.0))._acs_with__b(1)"
    assert lines[14] == "    return replace(c)._acs_with__a((  # comment, here"
    assert lines[15:17] == ["        1.0", "    ))"]


# G. Dropping a class decorator.


def test_a_dropped_class_decorator_takes_its_spaced_at() -> None:
    lines = _module_lines(
        """\
        from dataclasses import dataclass


        @ dataclass
        class B:
            x: int = 0
        """
    )
    assert lines[3].strip() == ""
    assert lines[4] == "class B:"


def test_a_kept_class_decorator_stays_contiguous_with_the_class() -> None:
    source = """\
        from dataclasses import dataclass


        def register(cls):
            return cls


        @register
        @dataclass
        class A:
            x: int = 0
        """
    lines, origin = _transform(source)
    assert lines[7:10] == ["", "@register", "class A:"]
    # The kept decorator moved down a line, the blank line up.
    assert origin[7:10] == [9, 8, 10]


# H. A `*` in a comment after the bare `*`.


def test_a_star_in_a_comment_is_not_the_bare_star() -> None:
    lines = _module_lines(
        """\
        def f(
            a: int,
            *,  # keyword-only below *
            b: int = 0,
        ) -> int:
            return a + b
        """
    )
    assert lines[2].startswith("    *_acs_kw: _acs_KeywordOnly[")
    assert lines[2].endswith("],  # keyword-only below *")


def test_a_star_in_a_comment_before_the_first_parameter() -> None:
    lines = _module_lines(
        """\
        def f(  # (*)
            *,
            b: int = 0,
        ) -> int:
            return b
        """
    )
    assert lines[0] == "def f(  # (*)"
    assert lines[1].startswith("    *_acs_kw: _acs_KeywordOnly[")


# K. A malformed string annotation of a dataclass field.


def test_a_malformed_string_annotation_does_not_crash() -> None:
    lines, _ = _transform(
        """\
        from dataclasses import dataclass


        @dataclass
        class C:
            x: "a b" = 0
        """
    )
    assert '    x: "a b" = 0' in lines


# E. field(init=False) of a type codon_annotation cannot express.


_INIT_FALSE = """\
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class C:
    a: int
    cache: dict[str, Any] = field(init=False)
    b: int = 0
"""


def test_an_init_false_field_naming_any_gets_an_empty_value() -> None:
    lines = _module_lines(_INIT_FALSE)
    assert lines[7] == "    cache: dict[str, Any] = dict[str, Any]()"


def test_the_fields_after_an_init_false_field_are_passed_by_keyword() -> None:
    calls = [
        line.strip()
        for line in render_library_checks(ast.parse(_INIT_FALSE)).source.splitlines()
        if line.strip().startswith("C(")
    ]
    assert "C(_acs_value(int), b=_acs_value(int))" in calls


def test_an_init_false_field_with_no_empty_value_is_refused() -> None:
    source = _INIT_FALSE.replace("b: int = 0", "u: Union[int, str] = field(init=False)")
    refused = init_false_refusals(source)
    assert [line for line, _ in refused] == [9]
    assert "C.u is a field(init=False) of `Union[int, str]`" in refused[0][1]
    assert init_false_refusals(_INIT_FALSE) == []
    # A default of its own: nothing to refuse.
    assert not init_false_refusals(
        source.replace("field(init=False)\n", "field(init=False, default=0)\n")
    )


# F. Empty defaults typed like their siblings.


def test_a_tuple_factory_of_a_sequence_is_a_typed_list() -> None:
    lines = _module_lines(
        """\
        from dataclasses import dataclass, field
        from typing import Sequence


        @dataclass
        class C:
            c1: Sequence[float] = ()
            c3: Sequence[float] = field(default_factory=tuple)
            c5: list[int] = field(default_factory=lambda: [])
            c6: dict[str, int] = field(default_factory=lambda: {})
        """
    )
    assert lines[6] == "    c1: Sequence[float] = List[float]()"
    assert lines[7] == "    c3: Sequence[float] = List[float]()"
    assert lines[8] == "    c5: list[int] = list[int]()"
    assert lines[9] == "    c6: dict[str, int] = dict[str, int]()"


# I. Annotation mappings.


def test_a_negative_literal_is_an_int() -> None:
    assert _annotation("Literal[-1]") == "int"
    assert _annotation("Literal[-1, 2]") == "int"


def test_a_nested_optional_is_flattened() -> None:
    assert _annotation("Optional[Union[int, None]]") == "Optional[int]"
    assert _annotation("Optional[Optional[int]]") == "Optional[int]"
    assert _annotation("Optional[int] | None") == "Optional[int]"


def test_an_empty_tuple_annotation_is_dropped() -> None:
    assert _annotation("tuple[()]") is None
    assert _annotation("Tuple[()]", class_level=True) is None


# J. ClassVar under an aliased typing import.


def test_an_aliased_class_var_is_no_field() -> None:
    lines, _ = _transform(
        """\
        import typing as t
        from dataclasses import dataclass


        @dataclass
        class A:
            x: int
            alias: t.ClassVar[int] = 0
        """
    )
    out = "\n".join(lines)
    assert "return self.x == other.x\n" in out
    assert "_acs_with__alias" not in out
    assert "A(x=x)" in out


# L. Enums.


def test_auto_of_a_str_mixin_enum_is_the_next_number() -> None:
    lines, _ = _transform(
        """\
        from enum import Enum, StrEnum, auto


        class M(str, Enum):
            A = auto()
            B = auto()


        class S(StrEnum):
            A = auto()
        """,
        in_place=False,
    )
    assert lines[4] == "    A: ClassVar[M] = M('A', '1')"
    assert lines[5] == "    B: ClassVar[M] = M('B', '2')"
    assert "    A: ClassVar[S] = S('A', 'a')" in lines


def test_an_alias_is_told_by_its_value() -> None:
    lines, _ = _transform(
        """\
        from enum import Enum


        class S(Enum):
            A = "x"
            B = 'x'


        class N(Enum):
            ONE = 1
            UNO = 0x1
            TWO = 2
        """,
        in_place=False,
    )
    out = "\n".join(lines)
    assert "return [S.A]" in out
    assert "return [N.ONE, N.TWO]" in out
