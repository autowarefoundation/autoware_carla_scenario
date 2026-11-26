"""The ``__repr__`` the static check gives a dataclass (#2536).

None of these need Codon: they pin down what the transform writes.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

from autoware_carla_scenario.typecheck.transform import (
    PRELUDE,
    transform_source_lines,
)


def _transform(source: str) -> tuple[str, list[int]]:
    return transform_source_lines(textwrap.dedent(source))


def _reprs(text: str) -> dict[str, str]:
    """Each class's generated ``__repr__`` body (its ``return`` expression)."""
    out = {}
    current = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"\s*class (\w+)", line)
        if match:
            current = match.group(1)
        if line.strip() == "def __repr__(self) -> str:" and current is not None:
            out[current] = lines[i + 1].strip().removeprefix("return ")
    return out


def _check_lines(source: str, text: str, origin: list[int]) -> None:
    """The transform's invariant: one origin per line, each a line of *source*."""
    ast.parse(text)
    body = text[len(PRELUDE) :]
    assert len(origin) == body.count("\n")
    assert all(1 <= line <= textwrap.dedent(source).count("\n") for line in origin)


def test_fields_in_python_format() -> None:
    source = """\
        from dataclasses import dataclass, field


        @dataclass
        class Pt:
            x: float
            name: str = "a"
            tags: list[str] = field(default_factory=list)
        """
    text, origin = _transform(source)
    _check_lines(source, text, origin)
    assert _reprs(text)["Pt"] == (
        '"Pt(x=" + repr(self.x) + ", name=" + repr(self.name)'
        ' + ", tags=" + repr(self.tags) + ")"'
    )
    # The appended method maps to its class's line.
    lines = text[len(PRELUDE) :].splitlines()
    at = lines.index("    def __repr__(self) -> str:")
    assert origin[at] == origin[at + 1] == 5


def test_no_field() -> None:
    text, _ = _transform(
        """\
        from dataclasses import dataclass


        @dataclass
        class Empty:
            pass
        """
    )
    assert _reprs(text)["Empty"] == '"Empty()"'


def test_repr_false_field_and_class_var_left_out() -> None:
    text, _ = _transform(
        """\
        from dataclasses import dataclass, field
        from typing import ClassVar, Union


        @dataclass
        class Cfg:
            LIMIT: ClassVar[int] = 3
            a: int = 0
            # Codon has no repr() of a Union: hidden, it does not matter.
            secret: Union[int, str] = field(default=0, repr=False)
            b: float = 1.0
        """
    )
    assert (
        _reprs(text)["Cfg"] == '"Cfg(a=" + repr(self.a) + ", b=" + repr(self.b) + ")"'
    )


def test_repr_false_class_and_own_repr_kept() -> None:
    text, _ = _transform(
        """\
        from dataclasses import dataclass


        @dataclass(repr=False)
        class Quiet:
            a: int = 0


        @dataclass
        class Own:
            a: int = 0

            def __repr__(self) -> str:
                return "own"


        @dataclass
        class Assigned:
            a: int = 0
            __repr__ = object.__repr__
        """
    )
    assert text.count("def __repr__") == 1  # Own's, as written
    assert "Quiet" not in _reprs(text)
    assert "Assigned" not in _reprs(text)
    assert _reprs(text)["Own"] == '"own"'


def test_nested_optional_enum_and_containers() -> None:
    text, _ = _transform(
        """\
        from __future__ import annotations

        import enum
        from dataclasses import dataclass, field
        from typing import Literal, Optional


        class Mode(enum.Enum):
            A = "a"


        @dataclass(frozen=True)
        class Inner:
            x: float = 0.0


        @dataclass
        class Outer:
            inner: Inner = field(default_factory=Inner)
            best: Optional[Inner] = None
            other: "Inner | None" = None
            mode: Mode = Mode.A
            legs: tuple[Inner, ...] = ()
            pair: tuple[int, str] = (0, "")
            by_name: dict[str, list[Inner]] = field(default_factory=dict)
            kind: Literal["a", "b"] = "a"
        """
    )
    assert _reprs(text)["Outer"] == (
        '"Outer(inner=" + repr(self.inner) + ", best=" + repr(self.best)'
        ' + ", other=" + repr(self.other) + ", mode=" + repr(self.mode)'
        ' + ", legs=" + repr(self.legs) + ", pair=" + repr(self.pair)'
        ' + ", by_name=" + repr(self.by_name) + ", kind=" + repr(self.kind) + ")"'
    )


def test_class_without_repr_only_directly() -> None:
    """A plain class has only Codon's default repr, which ``repr()`` falls
    back on but an ``Optional`` or a list's ``__repr__`` does not."""
    text, _ = _transform(
        """\
        from dataclasses import dataclass, field
        from typing import Optional


        class Handle:
            pass


        @dataclass
        class Direct:
            h: Handle = field(default_factory=Handle)


        @dataclass
        class Wrapped:
            h: Optional[Handle] = None


        @dataclass
        class Listed:
            hs: list[Handle] = field(default_factory=list)
        """
    )
    reprs = _reprs(text)
    assert reprs["Direct"] == '"Direct(h=" + repr(self.h) + ")"'
    assert "Wrapped" not in reprs and "Listed" not in reprs


def test_unknown_field_types_get_none() -> None:
    """Without a known ``__repr__`` for a field, the class keeps Codon's
    default: a generated one would break ``str()`` of it where it compiles."""
    text, _ = _transform(
        """\
        from dataclasses import dataclass, field
        from typing import Any, Optional, Union

        from somewhere import Pose

        Location = Union[int, str]


        @dataclass
        class U:
            u: Union[int, str] = 0


        @dataclass
        class Imported:
            p: Optional[Pose] = None


        @dataclass
        class Aliased:
            at: Location = 0


        @dataclass
        class Loose:
            extra: dict[str, Any] = field(default_factory=dict)


        @dataclass
        class Fine:
            a: int = 0
        """
    )
    assert set(_reprs(text)) == {"Fine"}


def test_holding_a_dataclass_without_repr() -> None:
    """A dataclass that gets no ``__repr__`` is no element of one that does."""
    text, _ = _transform(
        """\
        from dataclasses import dataclass, field
        from typing import Optional, Union


        @dataclass
        class NoRepr:
            u: Union[int, str] = 0


        @dataclass
        class Holder:
            a: NoRepr = field(default_factory=NoRepr)
            b: Optional[NoRepr] = None
        """
    )
    assert "Holder" not in _reprs(text)


def test_inherited_fields_first() -> None:
    text, _ = _transform(
        """\
        from dataclasses import dataclass, field


        @dataclass
        class Base:
            x: float = 0.0
            hidden: int = field(default=0, repr=False)


        @dataclass
        class Derived(Base):
            y: int = 0
            x: float = 1.0
            hidden: int = 1
        """
    )
    # A field declared again keeps its base's place, and its own repr flag.
    assert _reprs(text)["Derived"] == (
        '"Derived(x=" + repr(self.x) + ", hidden=" + repr(self.hidden)'
        ' + ", y=" + repr(self.y) + ")"'
    )
    assert _reprs(text)["Base"] == '"Base(x=" + repr(self.x) + ")"'


def test_self_reference() -> None:
    text, _ = _transform(
        """\
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Optional


        @dataclass
        class Node:
            value: int = 0
            next: Optional[Node] = None
        """
    )
    assert _reprs(text)["Node"] == (
        '"Node(value=" + repr(self.value) + ", next=" + repr(self.next) + ")"'
    )


def test_nested_dataclass_qualified() -> None:
    source = """\
        from dataclasses import dataclass


        class Plans:
            @dataclass
            class Leg:
                n: int = 0
        """
    text, origin = _transform(source)
    _check_lines(source, text, origin)
    assert '            return "Plans.Leg(n=" + repr(self.n) + ")"' in text.splitlines()


def test_repr_bound_by_module() -> None:
    text, _ = _transform(
        """\
        from dataclasses import dataclass


        def repr(x: int) -> str:
            return "x"


        @dataclass
        class Pt:
            a: int = 0
        """
    )
    assert "Pt" not in _reprs(text)


def test_repr_bound_by_module_seen_by_nested_dataclass() -> None:
    """A nested dataclass's method sees the module's ``repr`` too."""
    text, _ = _transform(
        """\
        from dataclasses import dataclass


        def repr(x: int) -> str:
            return "x"


        class Plans:
            @dataclass
            class Leg:
                n: float = 0.0
        """
    )
    assert '"Plans.Leg(' not in text


def test_enum_name_shadowed_by_a_plain_class() -> None:
    """Another module's enum, or a model enum, of the name of a plain class
    of this module is not the class an annotation here names."""
    text, _ = transform_source_lines(
        textwrap.dedent(
            """\
            from dataclasses import dataclass
            from typing import Optional


            class Phase:
                pass


            @dataclass
            class D:
                p: Optional[Phase] = None


            def f() -> None:
                class FailureCategory:
                    pass

                @dataclass
                class E:
                    c: Optional[FailureCategory] = None
            """
        ),
        enum_classes=["Phase"],
    )
    assert _reprs(text) == {}


def test_str_of_a_type_bypasses_codon_union_str() -> None:
    """``f"{type(u)}"`` of a Union value whose members have a ``__repr__``
    (a generated one included) compiles: every checked module imports the
    model's ``_tc``, whose ``str.__new__`` overload for a type comes before
    Codon's, which sends a type of a Union to ``union_str`` and fails on
    ``__ptr__()`` of the type (#2536, coordinate/transform.py's
    ``to_carla_world``)."""
    from autoware_carla_scenario.typecheck import transform

    assert "from autoware_carla_scenario._tc import " in PRELUDE
    model = Path(transform.__file__).parent / "codon" / "autoware_carla_scenario"
    text = (model / "_tc.codon").read_text(encoding="utf-8")
    block = text.split("@extend\nclass str:\n", 1)[1].split("\n\n\n", 1)[0]
    assert re.search(
        r"@overload\n    def __new__\(what\) -> str:\n"
        r"        if isinstance\(what, type\):\n"
        r"            return what\.__repr__\(\)\n"
        r"        elif isinstance\(what, Union\):\n",
        block,
    ), block
