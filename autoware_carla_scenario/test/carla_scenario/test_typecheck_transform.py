"""The source rewrite and the checks the static check makes before Codon runs.

None of these need Codon: they pin down what reaches the compiler.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import enum
import re
import sys
import textwrap
from typing import Any, Optional

import pytest

from autoware_carla_scenario.typecheck import model_dir
from autoware_carla_scenario.typecheck.driver import render_driver
from autoware_carla_scenario.typecheck.python_rewrites import (
    python_rewrite_refusals,
    python_rewrites,
)
from autoware_carla_scenario.typecheck.transform import (
    MISSING_BUILTIN_EXCEPTIONS,
    PRELUDE,
    classes_with_init,
    codon_annotation,
    conflicting_imports,
    flag_member_values,
    inherited_inits,
    relative_module_imports,
    split_tuple_excepts,
    transform_source,
    transform_source_lines,
    undeclared_attributes,
)


def test_the_prelude_imports_the_builtin_exceptions_codon_lacks() -> None:
    # Each is one of Python's built-in exceptions, declared in the model's
    # _exceptions.codon with its CPython base, and imported by every rewritten
    # module, where Python finds it among its builtins.
    tree = ast.parse(
        (model_dir() / "autoware_carla_scenario" / "_exceptions.codon").read_text()
    )
    bases = {
        node.name: ast.unparse(node.bases[0])
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name != "OSError"
    }
    assert sorted(bases) == sorted(MISSING_BUILTIN_EXCEPTIONS)
    for name in MISSING_BUILTIN_EXCEPTIONS:
        python = getattr(builtins, name)
        assert issubclass(python, BaseException), name
        assert bases[name] == f"Static[{python.__base__.__name__}]", name
    imported = ast.parse(PRELUDE).body[-1]
    assert isinstance(imported, ast.ImportFrom)
    assert imported.module == "autoware_carla_scenario._exceptions"
    assert [alias.name for alias in imported.names] == list(MISSING_BUILTIN_EXCEPTIONS)


def _transform(source: str) -> str:
    out = transform_source(textwrap.dedent(source))
    assert out.startswith(PRELUDE)
    return out[len(PRELUDE) :]


def _without_extensions(out: str) -> str:
    """*out* without its ``@extend`` blocks (each follows its class)."""
    return re.sub(r"\n@extend\nclass \w+:\n(?:(?:[ ].*)?\n)*", "", out)


def _annotation(text: str, *, class_level: bool = False) -> str | None:
    return codon_annotation(ast.parse(text, mode="eval").body, class_level=class_level)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("python", "codon"),
    [
        ("float | None", "Optional[float]"),
        ("None | Lanelet2Pose", "Optional[Lanelet2Pose]"),
        ("Optional[int]", "Optional[int]"),
        ("Union[str, None]", "Optional[str]"),
        ("list[int]", "list[int]"),
        ("dict[str, list[float | None]]", "dict[str, list[Optional[float]]]"),
        ("'carla.World'", "carla.World"),
        ("tuple[int, str]", "tuple[int, str]"),
        ("ClassVar[int]", "ClassVar[int]"),
    ],
)
def test_an_annotation_codon_can_express_is_kept(python: str, codon: str) -> None:
    assert _annotation(python) == codon


@pytest.mark.parametrize(
    "python",
    [
        "Union[EntityRole, str]",  # crashes Codon 0.19
        "EntityRole | str",
        "Any",
        "Callable[[int], None]",
        "type[BaseScenario]",
        "tuple[int, ...]",
        "Sequence[int]",  # a caller may pass a tuple
        "NDArray[np.uint8]",
    ],
)
def test_an_annotation_codon_cannot_express_is_dropped(python: str) -> None:
    assert _annotation(python) is None


@pytest.mark.parametrize(
    ("python", "codon"),
    [
        ("carla.TrafficLightState", "int"),
        ("'carla.TrafficLightState'", "int"),
        ("typesafe_carla.carla.LaneType", "int"),
        ("Optional[carla.LaneChange]", "Optional[int]"),
        ("list[carla.VehicleLightState]", "list[int]"),
        ("carla.TrafficLight", "carla.TrafficLight"),
        ("signals.TrafficLightState", "signals.TrafficLightState"),
    ],
)
def test_a_carla_integer_enumeration_is_an_int(python: str, codon: str) -> None:
    # typesafe_carla's Codon library has each as a value whose members are int.
    assert _annotation(python) == codon


def test_a_class_level_declaration_keeps_an_abstract_collection() -> None:
    # The typing shim maps it onto List; a declaration cannot be dropped.
    assert _annotation("Sequence[int]", class_level=True) == "Sequence[int]"


def test_parameters_and_returns_are_rewritten_in_place() -> None:
    out = _transform(
        """
        def f(a: float | None, b: Union[int, str], *, c: Any = 1) -> int | str:
            x: Any = 3
            return a
        """
    )
    assert "def f(a: Optional[float], b, *_acs_kw, c = 1):" in out
    assert "    x = 3" in out


def test_a_required_keyword_after_a_default_gets_a_placeholder_default() -> None:
    out = _transform(
        """
        def f(a: float = 0.0, *, label: str, n: int = 1, tag) -> None:
            pass

        def g(a: float, *, label: str) -> None:
            pass
        """
    )
    assert (
        "def f(a: float = 0.0, *_acs_kw, label: str = _acs_Required(), n: int = 1, "
        "tag = _acs_Required()) -> None:"
    ) in out
    assert "def g(a: float, *_acs_kw, label: str) -> None:" in out


def test_an_empty_collection_default_takes_the_type_of_its_field() -> None:
    out = _transform(
        """
        @dataclass
        class Result:
            statuses: list[Status] = field(default_factory=list)
            details: dict[str, Any] = field(default_factory=dict)
            names: set[str] = field(default_factory=list)
            ids: List[int] = field(default_factory=list)
            scores: typing.Dict[str, float] = field(default_factory=dict)
            tags: Set[str] = field(default_factory=set)
            loose: dict[str, Any] = field(default_factory=list)
        """
    )
    assert "    statuses: list[Status] = list[Status]()" in out
    # Codon cannot express Any, but the class keeps the declaration as
    # written, and its default is built as one (the typing shim's empty Any).
    assert "    details: dict[str, Any] = dict[str, Any]()" in out
    assert "    names: set[str] = []" in out  # not the annotation's collection
    # The typing spellings of the generics, as their builtins.
    assert "    ids: List[int] = List[int]()" in out
    assert "    scores: typing.Dict[str, float] = typing.Dict[str, float]()" in out
    assert "    tags: Set[str] = Set[str]()" in out
    assert "    loose: dict[str, Any] = []" in out  # not the annotation's either


def test_an_annotation_naming_a_class_defined_further_down_is_dropped() -> None:
    out = _transform(
        """
        class Absolute:
            vector: Relative  # a declaration is kept

            def __add__(self, other: Relative) -> Absolute:
                return self

            def __sub__(self, other: "list[Relative]") -> "Relative":
                return other[0]


        class Relative:
            def __add__(self, other: Absolute) -> Relative:
                return self


        def total(a: Absolute, r: Relative) -> Relative:
            return r
        """
    )
    assert "    vector: Relative  # a declaration is kept" in out
    assert "    def __add__(self, other) -> Absolute:" in out
    assert "    def __sub__(self, other):" in out
    assert "    def __add__(self, other: Absolute) -> Relative:" in out
    assert "def total(a: Absolute, r: Relative) -> Relative:" in out


def test_every_line_stays_where_it_was() -> None:
    source = textwrap.dedent(
        """
        from dataclasses import dataclass, field

        @dataclass(
            frozen=True,
        )
        class Config:
            ids: list[int] = field(
                default_factory=lambda: [460, 265]
            )
            npcs: list[NpcVehicleConfig] = field(default_factory=list)
            ground: GroundProjectionConfig = field(default_factory=GroundProjectionConfig)
            limit: float | None = field(default=None)
            later: int = 0  # marker
        """
    )
    out = _transform(source).partition("\n\n@extend\n")[0]  # the dunders follow
    assert out.count("\n") == source.count("\n")
    lines = out.splitlines()
    assert lines[source.splitlines().index("    later: int = 0  # marker")] == (
        "    later: int = 0  # marker"
    )
    assert "dataclass(" not in out.replace("import dataclass", "")
    assert "    ids: list[int] = ([460, 265])" in out
    assert "    npcs: list[NpcVehicleConfig] = list[NpcVehicleConfig]()" in out
    assert "    ground: GroundProjectionConfig = GroundProjectionConfig()" in out
    assert "    limit: Optional[float] = None" in out


def test_a_list_of_calls_or_names_becomes_an_acs_list() -> None:
    out = _transform(
        """
        a = AndCondition([TimeoutCondition(1.0, label="t"), SpeedCondition(x, label="s")])
        b = AndCondition([stopped, restarted])
        c = [460, 265]
        d = [only(1)]
        """
    )
    assert (
        'AndCondition(_acs_list(TimeoutCondition(1.0, label="t"), SpeedCondition(x, label="s")))'
        in out
    )
    assert "AndCondition(_acs_list(stopped, restarted))" in out
    assert "c = [460, 265]" in out
    assert "d = [only(1)]" in out
    assert "x = _acs_list(f(), g())" in transform_source("x = [f(), g()]\n")


def _transform_own(source: str) -> str:
    return _transform(_OWN_RULES + textwrap.dedent(source))


_OWN_RULES = """
class Rule: ...
class Dbl(Rule): ...
class Neg(Dbl): ...
class Other: ...
"""


def test_a_list_display_of_own_classes_is_a_list_of_the_annotation() -> None:
    # #480: Codon types a display by its first element, so a list of two
    # subclasses (or of one) is no list of their base: the display is built
    # into the annotated list.
    out = _transform_own(
        """
        RULES: list[Rule] = [Dbl()]
        def make() -> list[Rule]:
            return [Dbl(), Neg()]
        def total(rs: list[Rule], v: float = 1.0) -> float: ...
        total([Dbl()])
        total(rs=[Dbl(), Other()])
        def f():
            x: list[Rule] = [Dbl(), Neg()]
        """
    )
    assert "RULES: list[Rule] = _acs_list_into(List[Rule](), Dbl())" in out
    assert "return _acs_list_into(List[Rule](), Dbl(), Neg())" in out
    assert "total(_acs_list_into(List[Rule](), Dbl()))" in out
    assert "total(rs=_acs_list_into(List[Rule](), Dbl(), Other()))" in out
    assert "x: list[Rule] = _acs_list_into(List[Rule](), Dbl(), Neg())" in out


def test_a_dataclass_field_and_a_class_attribute_list_display() -> None:
    out = _transform_own(
        """
        @dataclass
        class H:
            rules: list[Rule] = field(default_factory=lambda: [Dbl()], repr=False)
        class R:
            rules: list[Rule] = [Dbl(), Neg()]
        """
    )
    assert "rules: list[Rule] = (_acs_list_into(List[Rule](), Dbl()))" in out
    assert "rules: list[Rule] = _acs_list_into(List[Rule](), Dbl(), Neg())" in out


def test_an_unannotated_list_display_of_own_classes_takes_their_common_base() -> None:
    out = _transform_own(
        """
        a = [Dbl(), Neg()]
        b = [Neg(), Dbl()]
        c = [Dbl(), Other()]
        d = [Dbl(), Dbl()]
        e = [1, 2]
        f: list[float] = [x, y]
        """
    )
    assert "a = _acs_list_into(List[Dbl](), Dbl(), Neg())" in out
    assert "b = _acs_list_into(List[Dbl](), Neg(), Dbl())" in out
    assert "c = _acs_list(Dbl(), Other())" in out
    assert "d = _acs_list(Dbl(), Dbl())" in out
    assert "e = [1, 2]" in out
    assert "f: list[float] = _acs_list(x, y)" in out


def test_a_display_of_classes_with_a_foreign_base_stays_an_acs_list() -> None:
    # A base from elsewhere may be BaseCondition: `_acs_list` makes those a
    # List[BaseCondition], a List[Mid] would be none (#480).
    out = _transform(
        textwrap.dedent(
            """
            from lib import BaseCondition
            class Mid(BaseCondition): ...
            class A(Mid): ...
            class B(Mid): ...
            class P: ...
            class Q(P): ...
            class R(P): ...
            x = [A(), B()]
            y = [Q(), R()]
            """
        )
    )
    assert "x = _acs_list(A(), B())" in out
    assert "y = _acs_list_into(List[P](), Q(), R())" in out


def test_a_dict_display_of_mixed_values_wraps_them_in_acs_base() -> None:
    # Codon types a dict display by its first item (#250): each value that may
    # be a condition or an action becomes its base class; the display, and
    # how Codon types any other value (a literal, None), stays as written.
    out = _transform(
        """
        a = ActionStateCondition("lc", s, {"turn": turn, "lc": lc}, label="x")
        b = {"t": TimeoutCondition(1.0, label="t"), "s": SpeedCondition(x, label="s")}
        c = {"a": {"b": p, "c": q}, "d": [r, s]}
        d = {"a": None, "b": 1, "c": y, "d": t if f else None, "e": [p, q][0]}
        """
    )
    assert (
        'a = ActionStateCondition("lc", s, '
        '{"turn": _acs_base(turn), "lc": _acs_base(lc)}, label="x")' in out
    )
    assert (
        'b = {"t": _acs_base(TimeoutCondition(1.0, label="t")), '
        '"s": _acs_base(SpeedCondition(x, label="s"))}' in out
    )
    assert (
        'c = {"a": {"b": _acs_base(p), "c": _acs_base(q)}, "d": _acs_list(r, s)}' in out
    )
    assert (
        'd = {"a": None, "b": 1, "c": _acs_base(y), '
        '"d": _acs_base(t if f else None), "e": _acs_base(_acs_list(p, q)[0])}' in out
    )


def test_an_evidently_homogeneous_dict_display_is_left_as_written() -> None:
    # Codon types these as Python does; the library's compile stays as it was.
    source = textwrap.dedent(
        """
        a = {"a": 1, "b": 2}
        b = {"a": Foo(1), "b": Foo(2)}
        c = {"a": f"{p}", "b": f"{q}"}
        d = {"l": Direction.LEFT, "r": Direction.RIGHT}
        f = {"a": p}
        g = {}
        h = {k: v for k, v in items}
        i: dict[str, int] = {"a": 1, "b": 2}
        """
    )
    out = transform_source(source)
    assert out == PRELUDE + source
    # Attributes of an instance are not members of one class.
    assert "_acs_base(" in transform_source('x = {"a": self.a, "b": self.b}\n')


def test_an_annotated_dict_display_is_built_as_the_annotated_type() -> None:
    out = _transform(
        """
        def setup(self) -> None:
            actions: dict[str, BaseAction] = {"turn": turn, "lc": lc}
            one: Dict[str, BaseAction] = {"turn": turn}
            loose: dict[str, Any] = {"turn": turn, "lc": lc}
            nested: dict[str, list[int]] = {"a": p, "b": q}
        """
    )
    assert (
        "actions: dict[str, BaseAction] = "
        '_acs_dict_into(Dict[str, BaseAction](), ("turn", turn), ("lc", lc))' in out
    )
    assert (
        'one: Dict[str, BaseAction] = _acs_dict_into(Dict[str, BaseAction](), ("turn", turn))'
        in out
    )
    # A type Codon cannot name: the display's own rewrite.
    assert 'loose = {"turn": _acs_base(turn), "lc": _acs_base(lc)}' in out
    assert '{"a": _acs_base(p), "b": _acs_base(q)}' in out


def test_a_dict_display_keeps_its_lines_and_comments() -> None:
    source = textwrap.dedent(
        """
        x = {
            # first: the turn
            ("turn"): turn,  # a: comment
            "lc"
            :
            (lc),
        }
        y = 1
        def setup(self) -> None:
            actions: dict[str, BaseAction] = {
                # first: the turn
                ("turn"): turn,  # a: comment
                "a:b,c"
                :
                (lc),
            }
            z = 2
        """
    )
    out = transform_source(source)[len(PRELUDE) :]
    assert out.count("\n") == source.count("\n")
    for line, original in zip(out.splitlines(), source.splitlines()):
        if original.strip() in ("y = 1", "z = 2"):
            assert line == original
    assert out.count("# first: the turn") == 2
    assert out.count("# a: comment") == 2
    tree = ast.parse(out)
    display = tree.body[0].value  # type: ignore[attr-defined]
    assert ast.unparse(display) == "{'turn': _acs_base(turn), 'lc': _acs_base(lc)}"
    into = tree.body[2].body[0].value  # type: ignore[attr-defined]
    assert ast.unparse(into) == (
        "_acs_dict_into(Dict[str, BaseAction](), ('turn', turn), ('a:b,c', lc))"
    )


def test_a_frozenset_is_checked_as_a_set() -> None:
    # Codon 0.19 has no frozenset; the check only types it, so a set will do.
    out = _transform(
        """
        def ids(self) -> FrozenSet[int]:
            return frozenset()

        def more(xs: frozenset[str]) -> None:
            ys = frozenset({"a"})

        def empty(keep: frozenset[str] = frozenset[str]()) -> None:
            pass
        """
    )
    lines = out.splitlines()
    assert lines[1] == "def ids(self) -> Set[int]:"
    assert lines[2] == "    return set()"
    assert lines[4] == "def more(xs: Set[str]) -> None:"
    assert lines[5] == '    ys = set({"a"})'
    # An empty one typed by its element, which a bare set() is not.
    assert lines[7] == "def empty(keep: Set[str] = Set[str]()) -> None:"
    assert _annotation("typing.FrozenSet[int]") == "Set[int]"
    assert _annotation("frozenset") is None


def test_bytes_is_checked_as_a_str() -> None:
    # Codon 0.19 has no bytes, and reads a b"..." literal as a str.
    out = _transform(
        """
        class Outcome:
            debug_info: bytes = b""

            def drive(self, data: bytes = b"") -> Optional[bytes]:
                return None
        """
    )
    lines = out.splitlines()
    assert lines[2] == '    debug_info: str = b""'
    assert lines[4] == '    def drive(self, data: str = b"") -> Optional[str]:'
    assert _annotation("builtins.bytes") == "str"
    assert _annotation("list[bytes]", class_level=True) == "list[str]"


def test_a_variable_length_tuple_attribute_is_checked_as_a_list() -> None:
    # Codon's tuples have a fixed length: the attribute holds a list, and
    # tuple() of a list (or a generator) makes one; a tuple stays itself.
    out = _transform(
        """
        class Path:
            vertices: Tuple[Vertex, ...]

            def __init__(self, vertices: Sequence[Vertex]) -> None:
                self.vertices = tuple(vertices)
                empty = tuple()
        """
    )
    lines = out.splitlines()
    assert lines[2] == "    vertices: List[Vertex]"
    assert lines[5] == "        self.vertices = _acs_tuple(vertices)"
    assert lines[6] == "        empty = tuple()"
    assert _annotation("tuple[int, ...]", class_level=True) == "List[int]"
    assert _annotation("typing.Tuple[str, ...]", class_level=True) == "List[str]"
    assert _annotation("tuple[int, ...]") is None  # a caller may pass a tuple
    assert _annotation("Sequence[int, ...]", class_level=True) is None


def test_a_variable_length_tuple_of_a_union_attribute_is_a_list_of_the_union() -> None:
    # Codon 0.19 holds a list of a union of classes, as it holds an attribute
    # of one (route.model's RouteSearchSpec.segments).
    out = _transform(
        """
        class Search:
            segments: tuple["LaneSpec | JunctionSpec", ...]
        """
    )
    assert out.splitlines()[2] == "    segments: List[Union[LaneSpec, JunctionSpec]]"
    assert (
        _annotation("Tuple[Union[A, B, C], ...]", class_level=True)
        == "List[Union[A, B, C]]"
    )
    # A parameter is still dropped (a caller may pass a tuple), and a union
    # with None, which Codon 0.19 cannot hold, is not translated.
    assert _annotation("tuple[A | B, ...]") is None
    assert _annotation("tuple[A | B | None, ...]", class_level=True) is None
    assert _annotation("tuple[A | Any, ...]", class_level=True) is None


def test_a_variable_length_tuple_return_is_checked_as_a_list() -> None:
    # Dropped, the annotation checked nothing the function returns (#191):
    # it is a List[X], as a class-level tuple[X, ...] is and `tuple(x)`
    # returns.  A parameter's is still dropped (a caller may pass a tuple).
    out = _transform(
        """
        def ids(xs: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(xs)
        def names() -> "Optional[Tuple[str, ...]]":
            return None
        def phases() -> "dict[str, tuple[Phase, ...]]":
            return {}
        def poses(
            n: int,
        ) -> tuple[
            Pose, ...
        ]:
            return tuple([])
        def seen() -> tuple[Sequence[int], ...]:
            return ()
        def later() -> tuple[Later, ...]:
            return ()
        def generic() -> tuple[T, ...]:
            return ()
        def either() -> Union[tuple[int, ...], str]:
            return ""
        def call() -> Callable[[], tuple[int, ...]]:
            return call
        def pair() -> tuple[int, str]:
            return (1, "a")
        T = TypeVar("T")
        class Later:
            pass
        """
    )
    lines = out.splitlines()
    assert lines[1] == "def ids(xs) -> List[int]:"
    assert lines[2] == "    return _acs_tuple(xs)"
    assert lines[3] == "def names() -> Optional[List[str]]:"
    assert lines[5] == "def phases() -> dict[str, List[Phase]]:"
    # An annotation over several lines keeps its line breaks.
    assert lines[7:12] == ["def poses(", "    n: int,", ") -> List[Pose", "", "]:"]
    # An abstract collection, a class defined further down, a TypeVar, a
    # Union and a Callable stay dropped.
    assert lines[13] == "def seen():"
    assert lines[15] == "def later():"
    assert lines[17] == "def generic():"
    assert lines[19] == "def either():"
    assert lines[21] == "def call():"
    # A fixed-length tuple is Codon's tuple, as before.
    assert lines[23] == "def pair() -> tuple[int, str]:"


def test_a_file_object_is_the_typing_shims_io() -> None:
    # typing.IO[AnyStr], what Path.open() returns: the typing shim's IO[T],
    # with bytes read as str.
    assert _annotation("Optional[IO[bytes]]") == "Optional[IO[str]]"
    assert _annotation("typing.IO[str]", class_level=True) == "typing.IO[str]"
    assert _annotation("IO") == "IO"


@pytest.mark.parametrize(
    ("python", "codon"),
    [
        (
            "np.ndarray[tuple[int, int], np.dtype[np.float32]]",
            "np.ndarray[np.float32, 2]",
        ),
        (
            "Optional[numpy.ndarray[Tuple[int, int, int], numpy.dtype[numpy.uint8]]]",
            "Optional[numpy.ndarray[numpy.uint8, 3]]",
        ),
        (
            "queue.Queue[tuple[int, np.ndarray[tuple[int], np.dtype[np.float64]]]]",
            "queue.Queue[tuple[int, np.ndarray[np.float64, 1]]]",
        ),
    ],
)
def test_a_shape_typed_array_is_codons_array_of_its_dimensions(
    python: str, codon: str
) -> None:
    # numpy types an array by its shape and dtype, Codon by its dtype and
    # its number of dimensions: a shape of N ints is N dimensions.
    assert _annotation(python) == codon
    assert _annotation(python, class_level=True) == codon


@pytest.mark.parametrize(
    "python",
    [
        "NDArray[np.float32]",  # no number of dimensions
        "np.ndarray[tuple[int, ...], np.dtype[np.float32]]",
        "np.ndarray[Any, np.dtype[np.float32]]",
        "np.ndarray[tuple[int, int], np.dtype[Any]]",
        "np.ndarray[tuple[int, int], np.float32]",
    ],
)
def test_an_array_without_its_dimensions_is_dropped(python: str) -> None:
    assert _annotation(python) is None


def test_an_annotation_broken_over_lines_keeps_its_statement_together() -> None:
    # The line breaks a rewritten annotation lost stay inside its brackets,
    # so the `= None` after it is still part of the statement.
    out = _transform(
        """
        def f() -> None:
            newest: Optional[
                tuple[int, np.ndarray[tuple[int, int], np.dtype[np.float32]]]
            ] = None
            after = 1
        """
    )
    lines = out.splitlines()
    assert lines[2] == "    newest: Optional[tuple[int, np.ndarray[np.float32, 2]]"
    assert lines[4] == "] = None"
    assert lines[5] == "    after = 1"
    ast.parse(out)


def test_a_literal_is_checked_as_the_type_of_its_values() -> None:
    # Codon's Literal types a compile-time value; Python's is checked as the
    # type of its values, and the typing shim's Literal is str.
    assert _annotation('Literal["any", "yes", "no"]') == "str"
    assert _annotation("typing.Literal[1, 2]") == "int"
    assert _annotation("Literal[True]") == "bool"
    assert _annotation('Literal["a", 1]') is None
    out = _transform(
        """
        Tristate = Literal["any", "yes", "no"]


        class Segment:
            shape: Literal["straight", "curved"] = "straight"
        """
    )
    lines = out.splitlines()
    assert lines[1] == "Tristate = str"
    assert lines[5] == '    shape: str = "straight"'
    # Inside a Union/Optional alias, a Literal's values are not forward
    # references to unquote.
    assert _transform('Maybe = Optional[Literal["a", "b"]]\n') == (
        "Maybe = Optional[str]\n"
    )
    assert _transform('Either = Union["Pose", Literal[1, 2]]\n') == (
        "Either = Union[Pose, int]\n"
    )


def test_callable_is_checked_by_the_stand_in() -> None:
    # Codon 0.19 has no callable(): _acs_callable decides it when compiling.
    assert "ok = _acs_callable(x.check)" in _transform("ok = callable(x.check)\n")


def test_object_setattr_with_a_constant_name_is_an_assignment() -> None:
    # A frozen dataclass sets its fields with object.__setattr__, which Codon
    # 0.19 lacks; the call keeps its lines inside brackets.
    source = textwrap.dedent(
        """
        class Pose:
            def __post_init__(self) -> None:
                object.__setattr__(self, "lane", int(self.lane))
                object.__setattr__(
                    self, "d_lane", _whole("Pose", "d_lane", self.d_lane)
                )
                object.__setattr__(self, name, 1)
        """
    )
    out = _transform(source)
    assert out.count("\n") == source.count("\n")
    lines = out.splitlines()
    assert lines[3] == "        self.lane = (int(self.lane))"
    assert lines[4].strip() == "self.d_lane = ("
    assert lines[5].strip() == '_whole("Pose", "d_lane", self.d_lane))'
    assert lines[7] == "        object.__setattr__(self, name, 1)"
    ast.parse(out)


def test_a_field_set_after_init_gets_an_empty_value_of_its_type() -> None:
    # Codon's constructor takes every field; one Python leaves out of it is
    # given a value, so a call without it still compiles.
    out = _transform(
        """
        @dataclass
        class Track:
            xs: list[float]
            _arc: list[float] = field(init=False, repr=False)
            _seen: bool = field(init=False, default=False)
        """
    )
    assert "    _arc: list[float] = list[float]()" in out
    assert "    _seen: bool = False" in out


def test_a_union_alias_of_forward_references_is_unquoted() -> None:
    # Codon takes no string as a type; under TYPE_CHECKING the name exists.
    out = _transform(
        """
        Position = Union["Lanelet2Pose", MapPose]
        Other = Optional["Lanelet2Pose"]
        label = Union["a", "b"] if False else "x"

        def f() -> None:
            local = Union["Lanelet2Pose", MapPose]
        """
    )
    lines = out.splitlines()
    assert lines[1] == "Position = Union[Lanelet2Pose, MapPose]"
    assert lines[2] == "Other = Optional[Lanelet2Pose]"
    assert lines[3] == 'label = Union["a", "b"] if False else "x"'
    assert lines[6] == '    local = Union["Lanelet2Pose", MapPose]'


def test_a_decorator_the_library_check_leaves_uncalled_is_dropped() -> None:
    # Codon 0.19 cannot decorate a method; a run-time-only wrapper is dropped
    # and the method is compiled as written.
    source = textwrap.dedent(
        """
        class Vertex:
            @_refusing_old_calls
            def __init__(self, position: Pose) -> None:
                self.position = position

            @other
            def f(self) -> None:
                pass
        """
    )
    out = transform_source(source, dropped_decorators=["_refusing_old_calls"])
    body = out[len(PRELUDE) :]
    assert body.count("\n") == source.count("\n")
    assert "@_refusing_old_calls" not in body
    assert "    @other" in body
    assert "@_refusing_old_calls" in _transform(source)


def test_an_annotation_naming_a_type_var_is_dropped() -> None:
    # A TypeVar types a run-time wrapper for mypy; Codon cannot type with it.
    out = _transform(
        """
        _Init = TypeVar("_Init", bound="Callable[..., None]")

        def wrap(init: _Init, n: int) -> _Init:
            return init

        def keep(x: int) -> int:
            return x
        """
    )
    lines = out.splitlines()
    assert lines[3] == "def wrap(init, n: int):"
    assert lines[6] == "def keep(x: int) -> int:"


def test_raise_from_none_loses_its_cause_and_raise_from_an_exception_keeps_it() -> None:
    out = _transform(
        """
        try:
            f()
        except AttributeError:
            raise RuntimeError(
                "no f"
            ) from None
        except ValueError as exc:
            raise RuntimeError("bad f") from exc
        """
    )
    lines = out.splitlines()
    assert lines[4:7] == ["    raise RuntimeError(", '        "no f"', "    )"]
    assert lines[8] == '    raise RuntimeError("bad f") from exc'


def test_a_builtin_exception_class_raised_as_it_is_is_instantiated() -> None:
    # Codon 0.19 crashes on `raise NotImplementedError` (#192 compiles every
    # abstract stub); Python raises an instance of the class either way.
    out = _transform(
        """
        def pick(ids: list[int]) -> int:
            if not ids:
                raise ValueError from None
            if ids[0] < 0:
                raise exc
            raise NotImplementedError
        """
    )
    lines = out.splitlines()
    assert lines[3] == "        raise ValueError()"
    assert lines[5] == "        raise exc"
    assert lines[6] == "    raise NotImplementedError()"


def test_super_new_in_a_singleton_s_new_allocates_the_class() -> None:
    # Codon 0.19's object.__new__ takes no class (MapManager's singleton).
    out = _transform(
        """
        class Single:
            def __new__(cls) -> "Single":
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance

            def other(self) -> None:
                def inner() -> None:
                    super().__new__(Single)
        """
    )
    lines = out.splitlines()
    assert lines[4] == "            cls._instance = Single.__new__()"
    assert lines[9] == "            Single.__new__()"


def test_a_callable_alias_is_none_on_its_lines() -> None:
    # Codon 0.19 reads the right-hand side as an expression ("expected type
    # expression"); an annotation spells the callable out.
    out = _transform(
        """
        Height = Callable[[float, float], Optional[float]]
        Resolver = Callable[
            [int],
            float,
        ]
        Pair = Tuple[float, float]


        def f() -> None:
            local = Callable[[int], int]
        """
    )
    lines = out.splitlines()
    assert lines[1] == "Height = None"
    assert lines[2:6] == ["Resolver = (None", "", "", ")"]
    assert lines[6] == "Pair = Tuple[float, float]"
    assert lines[10] == "    local = Callable[[int], int]"


def test_a_return_annotation_over_several_lines_is_dropped_in_place() -> None:
    out = _transform(
        """
        def make(
            n: int,
        ) -> Callable[
            [int],
            int,
        ]:
            return n
        """
    )
    lines = out.splitlines()
    assert lines[1:7] == ["def make(", "    n: int,", "", "", "", "):"]
    assert lines[7] == "    return n"


def test_only_an_int_or_str_enum_derives_from_the_value_comparing_classes() -> None:
    out = _transform(
        textwrap.dedent(
            """
            import enum
            class A(enum.IntEnum):
                X = 1
            class B(int, enum.Enum):
                X = 1
            class C(enum.IntFlag):
                X = 1
            class D(enum.StrEnum):
                X = "x"
            class E(str, enum.Enum):
                X = "x"
            class F(enum.Enum):
                X = 1
            class G(enum.Flag):
                X = 1
            class H(int, enum.Flag):
                X = 1
            """
        )
    )
    for line in (
        "class A(_acs_IntEnum[int]):",
        "class B(_acs_IntEnum[int]):",
        "class C(_acs_IntFlag[int]):",
        "class D(_acs_StrEnum[str]):",
        "class E(_acs_StrEnum[str]):",
        "class F(enum.Enum[int]):",
        "class G(enum.Flag[int]):",
        "class H(_acs_IntFlag[int]):",
    ):
        assert line in out


def test_an_enum_becomes_a_subclass_of_the_enum_shim() -> None:
    source = textwrap.dedent(
        """
        @unique
        class Turn(str, Enum):
            LEFT = "left"
            RIGHT = (
                "right"
            )

            def opposite(self) -> Turn:
                return self
        class Level(enum.Enum):
            LOW = auto()
            HIGH = auto()
            _ignored = 3
        class Kind(Enum):
            A = 4
            B = auto()
        class Mixed(Enum):
            A = 1
            B = "b"
        """
    )
    out = _transform(source)
    body = _without_extensions(out)  # the constructors by value follow
    assert body.count("\n") == source.count("\n")
    assert "@unique" not in out
    assert "class Turn(_acs_StrEnum[str]):" in out
    assert "    LEFT: ClassVar[Turn] = Turn('LEFT', \"left\")" in out
    assert "    RIGHT: ClassVar[Turn] = Turn('RIGHT', \"right\")" in out
    assert "    def opposite(self) -> Turn:" in out
    assert "class Level(enum.Enum[int]):" in out
    assert "    HIGH: ClassVar[Level] = Level('HIGH', 2)" in out
    assert "    _ignored = 3" in out
    assert "    B: ClassVar[Kind] = Kind('B', 5)" in out
    # Values of two types: left as written, for Codon to report.
    assert "class Mixed(Enum):" in out


def test_an_enum_is_given_a_constructor_by_value_after_the_module() -> None:
    source = textwrap.dedent(
        """
        class Turn(Enum):
            LEFT = "left"
        class Level(enum.IntEnum):
            LOW = 1
        def f():
            class Local(Enum):
                A = 1
        """
    )
    out = _transform(source)
    body = _without_extensions(out)
    assert body.count("\n") == source.count("\n")  # no line of the source moves
    assert "@extend\nclass Turn:\n    def __init__(self, value: str):" in out
    assert "@extend\nclass Level:\n    def __init__(self, value: int):" in out
    assert "@extend\nclass Local:" not in out  # the module's top-level enums only


def test_an_enum_is_given_its_members_and_a_lookup_by_name() -> None:
    source = textwrap.dedent(
        """
        class Turn(Enum):
            LEFT = "left"
            RIGHT = "right"
            ALIAS = "left"
            _hidden = 3
        """
    )
    out = _transform(source)
    # An alias is looked up by name, but iteration skips it.
    assert (
        "def _acs_members() -> List[Turn]:\n        return [Turn.LEFT, Turn.RIGHT]"
        in out
    )
    assert "if name == 'ALIAS':\n            return Turn.ALIAS" in out
    assert out.rstrip().endswith("raise KeyError(name)")


_FLAGS = """
class Perm(enum.IntFlag):
    NONE = 0
    R = 1
    W = 2
    RW = R | W
    X = enum.auto()
    BOTH = 3
    HIGH = 1 << 4
    Y = enum.auto()
    ALL = ~0
"""


def test_a_flag_s_composite_values_are_worked_out_and_skipped_by_iteration() -> None:
    # Python 3.11+ iterates the single-bit members: not zero, not a composite
    # (`RW`), not an alias of one.
    out = _transform(_FLAGS)
    assert "RW: ClassVar[Perm] = Perm('RW', 3)" in out
    assert "X: ClassVar[Perm] = Perm('X', 4)" in out  # above the highest bit (3)
    assert "HIGH: ClassVar[Perm] = Perm('HIGH', 16)" in out
    assert "Y: ClassVar[Perm] = Perm('Y', 32)" in out
    assert "ALL: ClassVar[Perm] = Perm('ALL', -1)" in out
    assert (
        "return [Perm.R, Perm.W, Perm.X, Perm.HIGH, Perm.Y]" in out
    ), "NONE, RW, BOTH and ALL are not iterated"
    for name in ("NONE", "RW", "BOTH", "ALL"):
        assert f"if name == '{name}':\n            return Perm.{name}" in out


def test_a_flag_is_given_its_operators_after_the_module() -> None:
    out = _transform(_FLAGS)
    extension = out.partition("\n\n@extend\n")[2]
    for method in ("or", "and", "xor"):
        assert f"def __{method}__(self, other: Perm) -> Perm:" in extension
        assert f"def __{method}__(self, other: int) -> Perm:" in extension
        assert f"def __r{method}__(self, other: int) -> Perm:" in extension
    assert "def __contains__(self, other: Perm) -> bool:" in extension
    # `~` flips every bit below the highest (32), as IntFlag does.
    assert "return Perm(63 & ~self._value_)" in extension
    plain = _transform("class Bits(enum.Flag):\n    A = 1\n    B = 2\n")
    assert "def __or__(self, other: Bits) -> Bits:" in plain
    assert "other: int" not in plain  # only an IntFlag takes an int
    assert "return Bits(3 & ~self._value_)" in plain


_FLAG_DEFINITIONS = [
    ("Flag", "A = auto()\nB = 4\nC = auto()\nD = 1\nE = auto()"),
    ("Flag", "A = 1\nB = 2\nAB = A | B\nC = auto()"),
    ("Flag", "R = 1\nW = 2\nRW = R | W\nX = 4"),
    ("Flag", "N = 0\nR = 1\nW = 2"),
    ("Flag", "R = 1\nW = 2\nREAD = 1\nX = 4"),
    ("Flag", "A = 3\nB = auto()"),
    ("Flag", "A = 1\nB = 2 * 2\nC = A + 6"),
    ("Flag", "A = 1\nB = 2\nC = 4\nAB = A ^ B\nBC = (A | B | C) & ~A"),
    ("Flag", "RW = 3\nR = 1\nW = 2"),
    ("Flag", "A = 1\nB = 2\nC = 3\nD = 3"),
    ("Flag", "A = 1\nB = 2\nBC = 6\nC = auto()"),
    ("IntFlag", "R = 1\nW = 2\nALL = ~0"),
    ("IntFlag", "R = 1\nW = 2\nM = -1"),
    ("IntFlag", "R = 1\nW = 2\nT = 3\nF = 5\nX = 8"),
    ("IntFlag", "A = 6\nB = 1\nC = 8"),
    ("IntFlag", "A = 1 << 3\nB = 1 << 5\nC = auto()"),
    ("IntFlag", "A = 16\nB = auto()\nC = auto()"),
    ("IntFlag", "A = 1\nB = -4\nC = auto()"),
    ("IntFlag", "R = 1\nW = 2\nBIG = 12"),
    ("IntFlag", "N = 0\nR = 2\nW = 4"),
]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Python 3.11 changed Flag iteration and ~"
)
@pytest.mark.parametrize(("kind", "body"), _FLAG_DEFINITIONS)
def test_a_flag_s_values_iteration_and_complement_are_python_s(
    kind: str, body: str
) -> None:
    # A differential test against CPython's own enum.Flag / IntFlag (#437).
    source = f"class P(enum.{kind}):\n" + textwrap.indent(body, "    ") + "\n"
    namespace: dict[str, Any] = {"enum": enum, "auto": enum.auto}
    exec(source, namespace)  # noqa: S102
    flag = namespace["P"]
    assert flag_member_values(ast.parse(source).body[0].body) == {  # type: ignore[attr-defined]
        name: member.value for name, member in flag.__members__.items()
    }
    out = _transform("import enum\nfrom enum import auto\n" + source)
    listed = re.search(r"return \[(.*?)\]", out.partition("_acs_members")[2])
    assert listed is not None
    assert [m.strip().split(".")[1] for m in listed[1].split(",") if m] == [
        member.name for member in flag
    ]
    flipped = re.search(r"return P\((-?\d+) & ~self\._value_\)", out)
    assert flipped is not None
    for member in flag.__members__.values():
        assert int(flipped[1]) & ~member.value == (~member).value, member


def test_a_flag_s_operators_follow_its_class_and_keep_the_lines_after_it() -> None:
    # Module-level code (`RW = P.R | P.W`) needs them while Codon reads the
    # module, so they cannot wait for its end; the lines after the class map
    # to their own (#437).
    source = textwrap.dedent(
        """
        import enum
        class P(enum.Flag):
            R = 1
            W = 2
        RW = P.R | P.W  # line 6
        class Q(enum.Enum):
            A = 1
        LAST = 1  # line 9
        """
    )
    text, origin = transform_source_lines(source)
    lines = text[len(PRELUDE) :].splitlines()
    assert len(origin) == len(lines)
    at = lines.index("RW = P.R | P.W  # line 6")
    assert origin[at] == 6
    assert any("def __or__" in line for line in lines[:at])
    assert origin[lines.index("LAST = 1  # line 9")] == 9
    # so does a plain Enum's (#469): a class body or module-level comprehension
    # over it reads its members while Codon reads the module
    assert any(
        "def _acs_by_name" in line
        for line in lines[: lines.index("LAST = 1  # line 9")]
    )


def test_no_other_enum_is_given_flag_operators() -> None:
    out = _transform(
        "class A(enum.IntEnum):\n    X = 1\nclass B(enum.Enum):\n    X = 1\n"
    )
    assert "__or__" not in out and "__invert__" not in out


def test_a_flag_value_that_is_not_worked_out_leaves_the_complement_to_codon() -> None:
    out = _transform("class P(enum.Flag):\n    A = 1\n    B = compute()\n")
    assert "B: ClassVar[P] = P('B', compute())" in out
    assert "__or__" in out and "__invert__" not in out


def test_the_framework_s_enums_are_iterated_without_being_named() -> None:
    from autoware_carla_scenario.typecheck.transform import model_enum_names

    assert {"TurnDirection", "ComparisonRule"} <= model_enum_names()
    out = _transform(
        "def f():\n    return list(TurnDirection), ComparisonRule['LESS_THAN']\n"
    )
    assert "list(TurnDirection._acs_members())" in out
    assert "ComparisonRule._acs_by_name('LESS_THAN')" in out


@pytest.mark.parametrize(
    "binding",
    [
        'TurnDirection = ["a", "b"]',
        "def TurnDirection():\n    return [1]",
        "import os as TurnDirection",
        "from os import path as TurnDirection",
        "for TurnDirection in [[1]]:\n    pass",
    ],
)
def test_a_module_binding_of_a_model_enum_s_name_is_not_iterated(binding: str) -> None:
    out = _transform(f"{binding}\ndef f():\n    return list(TurnDirection)\n")
    assert (
        "list(TurnDirection)" in out
        and "_acs_members" not in out.partition("def f")[2].partition("\n\n@extend")[0]
    )


def test_a_framework_import_of_a_model_enum_is_still_iterated() -> None:
    out = _transform(
        "from autoware_carla_scenario import TurnDirection\n"
        "def f():\n    return list(TurnDirection)\n"
    )
    assert "list(TurnDirection._acs_members())" in out


def test_a_model_enum_name_bound_inside_a_function_only_shadows_there() -> None:
    out = _transform(
        "def f():\n    TurnDirection = [1]\n    return list(TurnDirection)\n"
        "def g():\n    return list(TurnDirection)\n"
    )
    assert "list(TurnDirection)\n" in out
    assert "list(TurnDirection._acs_members())" in out


def test_a_class_of_a_model_enum_s_name_that_is_no_enum_is_not_iterated() -> None:
    out = _transform(
        "class TurnDirection:\n    pass\ndef f():\n    return list(TurnDirection)\n"
    )
    assert "list(TurnDirection)" in out


def test_iterating_over_an_enum_class_and_looking_up_a_member_by_name() -> None:
    source = textwrap.dedent(
        """
        class Mode(Enum):
            A = 1
        def f(n):
            for m in Mode:
                pass
            a = [x for x in Mode if x in Mode]
            b = list(Mode), len(Mode), sorted(Mode), (*Mode,)
            c = Mode[
                n
            ]
            d = Mode["A"].value
            e = Other[n]
            g = Mode.A
            h = {m: 1 for m in Mode}
        """
    )
    out = _transform(source)
    body = _without_extensions(out)
    assert body.count("\n") == source.count("\n")  # no line moves
    assert "for m in Mode._acs_members():" in out
    assert "[x for x in Mode._acs_members() if x in Mode._acs_members()]" in out
    assert "list(Mode._acs_members()), len(Mode._acs_members())" in out
    assert "sorted(Mode._acs_members()), (*Mode._acs_members(),)" in out
    assert "c = Mode._acs_by_name(\n" in out  # the line breaks stay
    assert 'd = Mode._acs_by_name("A").value' in out
    assert "e = Other[n]" in out and "g = Mode.A" in out
    assert "h = {m: 1 for m in Mode._acs_members()}" in out


def test_cls_of_an_enum_s_classmethod_is_the_enum() -> None:
    source = textwrap.dedent(
        """
        class Mode(Enum):
            A = 1
            @classmethod
            def f(cls, n):
                for m in cls:
                    pass
                return list(cls), len(cls), 1 in cls, cls[n], cls.A
            @classmethod
            def g(cls):
                return [m for m in cls], (*cls,)
        """
    )
    out = _without_extensions(_transform(source))
    assert "for m in Mode._acs_members():" in out
    assert "list(Mode._acs_members()), len(Mode._acs_members())" in out
    assert "1 in Mode._acs_members(), Mode._acs_by_name(n), Mode.A" in out
    assert "[m for m in Mode._acs_members()], (*Mode._acs_members(),)" in out


def test_cls_is_the_enum_in_its_classmethod_only() -> None:
    source = textwrap.dedent(
        """
        class Mode(Enum):
            A = 1
            def inst(self, cls):
                return list(cls)
            @staticmethod
            def st(cls):
                return list(cls)
            @classmethod
            def ok(klass):
                return list(klass)
        def free(cls):
            return list(cls)
        cls = Mode
        x = list(cls)
        """
    )
    out = _without_extensions(_transform(source))
    assert out.count("list(cls)") == 4
    assert "return list(Mode._acs_members())" in out


def test_cls_shadowed_in_a_classmethod_is_not_the_enum() -> None:
    source = textwrap.dedent(
        """
        class Mode(Enum):
            A = 1
            @classmethod
            def lam(cls):
                return list(cls), (lambda cls: list(cls))([1])
            @classmethod
            def nested(cls):
                def inner(cls):
                    return list(cls)
                return inner([1]), [list(cls) for cls in [[1]]]
            @classmethod
            def rebound(cls):
                cls = [1]
                return list(cls)
            @classmethod
            def loop(cls):
                for cls in [[1]]:
                    pass
                return list(cls)
            @classmethod
            def closure(cls):
                return (lambda: list(cls))()
        """
    )
    out = _without_extensions(_transform(source))
    assert "return list(Mode._acs_members()), (lambda cls: list(cls))([1])" in out
    assert "return list(cls)\n" in out  # inner(cls)
    assert "[list(cls) for cls in [[1]]]" in out
    assert "(lambda: list(Mode._acs_members()))()" in out
    # A rebound `cls` is no alias anywhere in the method.
    assert out.count("list(Mode._acs_members())") == 2


def test_cls_of_a_class_that_is_no_enum_is_untouched() -> None:
    source = textwrap.dedent(
        """
        class Plain:
            @classmethod
            def f(cls):
                return list(cls), cls["A"]
            def g(self):
                return list(type(self)), list(self.__class__)
        """
    )
    out = _transform(source)
    assert 'list(Plain), Plain["A"]' in out
    assert "list(type(self)), list(self.__class__)" in out
    assert "_acs_members" not in out


def test_type_of_self_and_dunder_class_are_the_enum_in_an_instance_method() -> None:
    source = textwrap.dedent(
        """
        class Mode(Enum):
            A = 1
            def peers(self):
                return list(type(self)), [m for m in self.__class__], type(self)["A"]
            def other(self, o):
                return list(type(o)), list(type(self, 1)), list(self.__class__.x)
            def shadow(self):
                type = len
                return list(type(self))
            def rebound(self):
                self = 3
                return list(type(self))
            @staticmethod
            def st(self):
                return list(type(self))
        def free(self):
            return list(type(self))
        """
    )
    out = _without_extensions(_transform(source))
    assert "list(Mode._acs_members()), [m for m in Mode._acs_members()]," in out
    assert 'Mode._acs_by_name("A")' in out
    assert "list(type(o)), list(type(self, 1)), list(self.__class__.x)" in out
    # Only `peers`: `type` bound, `self` rebound, a staticmethod, a function.
    assert out.count("list(Mode._acs_members())") == 1
    assert out.count("list(type(self))") == 4


def test_an_enum_of_another_module_is_iterated_when_named() -> None:
    source = "def f():\n    return list(Rule), Rule['A']\n"
    assert "list(Rule)" in _transform(source)
    out = transform_source(source, enum_classes=["Rule"])
    assert "list(Rule._acs_members()), Rule._acs_by_name('A')" in out


def test_a_name_that_is_not_the_enum_class_is_left_alone() -> None:
    source = textwrap.dedent(
        """
        class Mode(Enum):
            A = 1
        def f(Mode):
            return list(Mode), Mode["A"], 1 in Mode
        def g():
            Mode = {1: 2}
            return [m for m in Mode], Mode[1]
        def h(x: Mode[int]) -> Mode[int]:
            return list(Mode), self.Mode["A"]
        """
    )
    out = _without_extensions(_transform(source))
    assert 'list(Mode), Mode["A"], 1 in Mode' in out
    assert "[m for m in Mode], Mode[1]" in out
    assert "_acs_by_name(int)" not in out
    assert 'return list(Mode._acs_members()), self.Mode["A"]' in out


def test_a_str_enum_with_auto_values_takes_the_lowercase_names() -> None:
    out = _transform("class Mode(StrEnum):\n    FAST = auto()\n")
    assert "class Mode(_acs_StrEnum[str]):" in out
    assert "FAST: ClassVar[Mode] = Mode('FAST', 'fast')" in out


def test_an_exception_derives_statically() -> None:
    out = _transform(
        """
        class Mismatch(ValueError):
            pass
        class Worse(Mismatch):
            pass
        class Remote(errors.TransportError):
            pass
        class Plain(Base):
            pass
        """
    )
    assert "class Mismatch(Static[ValueError]):" in out
    assert "class Worse(Static[Mismatch]):" in out
    assert "class Remote(Static[errors.TransportError]):" in out
    assert "class Plain(Base):" in out


def test_abstractmethod_is_dropped() -> None:
    source = textwrap.dedent(
        """
        import abc
        from abc import ABC, abstractmethod

        class Bridge(ABC):
            @abstractmethod
            def is_ready(self) -> bool:
                \"\"\"Whether it is ready.\"\"\"

            @abc.abstractmethod
            def close(self) -> None:
                pass
        """
    )
    out = _transform(source)
    assert out.count("\n") == source.count("\n")
    assert "@" not in out
    assert "    def is_ready(self) -> bool:" in out
    assert "class Bridge(ABC):" in out


def test_a_classmethod_becomes_a_staticmethod_of_its_class() -> None:
    source = textwrap.dedent(
        """
        class Pose:
            @classmethod
            def origin(cls) -> Pose:
                return cls(0.0)

            @classmethod
            def at(cls, x: float | None, *, y: float = 0.0) -> Pose:
                return cls.make(cls(x))
        """
    )
    out = _transform(source)
    assert out.count("\n") == source.count("\n")
    assert "@classmethod" not in out
    assert "    def origin() -> Pose:\n        return Pose(0.0)" in out
    assert "    def at(x: Optional[float], *_acs_kw, y: float = 0.0) -> Pose:" in out
    assert "        return Pose.make(Pose(x))" in out


# ---------------------------------------------------------------------------
# Attribute declarations
# ---------------------------------------------------------------------------


_SCENARIO = """
class MyScenario(BaseScenario):
    _declared: int

    def __init__(self, config: Cfg | None = None) -> None:
        super().__init__()
        self._config = config or Cfg()
        self._declared = 1
        self.ego_config = None
        self._typed: float = 2.0
"""


def test_an_undeclared_attribute_of_a_subclass_is_reported_with_a_type() -> None:
    found = undeclared_attributes(
        ast.parse(_SCENARIO), {"BaseScenario": {"ego_config"}}
    )
    assert [(f.name, f.suggested_type) for f in found] == [
        ("_config", "Cfg"),
        ("_typed", "float"),
    ]
    assert found[0].lineno == 7
    assert "`_config: Cfg`" in found[0].message()


def test_a_class_outside_any_hierarchy_may_infer_its_attributes() -> None:
    source = "class Helper:\n    def __init__(self):\n        self.x = 1\n"
    assert undeclared_attributes(ast.parse(source), {}) == []


def test_a_class_with_a_class_level_annotation_declares_every_attribute() -> None:
    # Codon infers a plain class's attributes only while it declares none
    # (#218); its methods do not count as declarations.
    source = textwrap.dedent(
        """
        class Tracker:
            limit: float
            def __init__(self, limit: float):
                self.limit = limit
                self._samples = []
        class Methods:
            def __init__(self):
                self.x = 1
            def x_twice(self) -> int:
                return 2 * self.x
        class Constant:
            LIMIT = 3.0
            def __init__(self):
                self.y = 1.0
        """
    )
    found = undeclared_attributes(ast.parse(source), {})
    assert [(f.class_name, f.name, f.suggested_type) for f in found] == [
        ("Tracker", "_samples", "list[...]")
    ]


def test_a_class_named_in_also_is_checked_whatever_it_declares() -> None:
    source = "class Helper:\n    def __init__(self):\n        self.x = []\n"
    found = undeclared_attributes(ast.parse(source), {}, also={"Helper"})
    assert [(f.class_name, f.name, f.lineno) for f in found] == [("Helper", "x", 3)]


def test_a_method_codon_never_compiles_needs_no_declaration() -> None:
    # The library check leaves some methods uncalled (library.UNCALLED).
    source = textwrap.dedent(
        """
        class Region:
            _ids: list[int]
            def __init__(self, specs):
                self._specs = list(specs)
            def reset(self) -> None:
                self._count = 0
        """
    )
    found = undeclared_attributes(ast.parse(source), {}, uncalled={"Region.__init__"})
    assert [f.name for f in found] == ["_count"]


def test_a_declaration_on_a_base_in_the_same_module_counts() -> None:
    source = textwrap.dedent(
        """
        class Base:
            x: int
        class Child(Base):
            def __init__(self):
                self.x = 1
        """
    )
    assert undeclared_attributes(ast.parse(source), {}) == []


def test_a_base_imported_under_an_alias_is_looked_up_by_its_own_name() -> None:
    source = textwrap.dedent(
        """
        from .base import ScenarioBase as Parent
        class Child(Parent):
            def __init__(self):
                self.x = 1
        """
    )
    assert undeclared_attributes(ast.parse(source), {"ScenarioBase": {"x"}}) == []


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Npc:
    spawn_lanelet_id: int = 0
    spawn_s: float = 0.0


class _Turn(enum.Enum):
    LEFT = "left"


@dataclasses.dataclass
class _Config:
    timeout_seconds: float = 5.0
    min_speed_kmh: Optional[float] = None
    ids: list[int] = dataclasses.field(default_factory=list)
    npcs: list[_Npc] = dataclasses.field(default_factory=list)
    turn: _Turn = _Turn.LEFT

    def __post_init__(self) -> None:
        # As the built-in configs do: the runner passes the YAML mappings.
        self.npcs = [_Npc(**n) if isinstance(n, dict) else n for n in self.npcs]


class _Scenario:
    pass


def test_the_driver_renders_the_yaml_values_with_the_config_types() -> None:
    driver = render_driver(
        _Scenario,
        _Config,
        {
            "timeout_seconds": 5,
            "min_speed_kmh": 3,
            "ids": [460, 265],
            "npcs": [{"spawn_lanelet_id": 1, "spawn_s": 2}],
            "turn": _Turn.LEFT,
        },
    )
    lines = driver.source.splitlines()
    by_key = {key: lines[line - 1].strip() for line, key in driver.config_lines.items()}
    assert by_key == {
        "scenario.timeout_seconds": "config.timeout_seconds = 5.0",
        "scenario.min_speed_kmh": "config.min_speed_kmh = 3.0",
        "scenario.ids": "config.ids = [460, 265]",
        "scenario.npcs": "config.npcs = [_Npc(spawn_lanelet_id=1, spawn_s=2.0)]",
        "scenario.turn": "config.turn = _Turn.LEFT",
    }
    assert f"from {__name__} import _Npc" in lines
    assert "    scenario.setup()" in lines
    # The hooks' results, checked strictly after the module checks (#238):
    # `done: bool = scenario.is_done()` converted a str with __bool__.
    assert not [line for line in lines if "done: bool" in line]
    assert '    BaseScenario._acs_hook_returns("is_done", scenario.is_done())' in lines
    assert driver.trailer == "_acs_check_scenario_hooks()\n"


def test_a_mapping_the_config_does_not_convert_stays_a_dict() -> None:
    # config_cls(**scenario_dict) leaves it a dict at run time, so the driver
    # must not pass it off as the annotated dataclass.
    @dataclasses.dataclass
    class Unconverted:
        npc: Optional[_Npc] = None
        npcs: list[_Npc] = dataclasses.field(default_factory=list)

    driver = render_driver(
        _Scenario,
        Unconverted,
        {"npc": {"spawn_lanelet_id": 1}, "npcs": [{"spawn_lanelet_id": 2}]},
    )
    lines = driver.source.splitlines()
    by_key = {key: lines[line - 1].strip() for line, key in driver.config_lines.items()}
    assert by_key == {
        "scenario.npc": "config.npc = {'spawn_lanelet_id': 1}",
        "scenario.npcs": "config.npcs = [{'spawn_lanelet_id': 2}]",
    }


def test_a_config_with_a_mandatory_field_is_built_by_its_constructor() -> None:
    @dataclasses.dataclass
    class Mandatory:
        lanelet_id: int

    driver = render_driver(_Scenario, Mandatory, {"lanelet_id": 3})
    assert "        lanelet_id=3," in driver.source.splitlines()


# ---------------------------------------------------------------------------
# When the runner checks: the typecheck mode
# ---------------------------------------------------------------------------


class _ModeScenario:
    pass


def test_the_mode_comes_from_the_config_then_defaults_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.typecheck.mode import TYPECHECK_ENV, typecheck_mode

    monkeypatch.delenv(TYPECHECK_ENV, raising=False)
    assert typecheck_mode({}) == "auto"
    assert typecheck_mode({"typecheck": "required"}) == "required"
    assert typecheck_mode({"typecheck": False}) == "off"
    monkeypatch.setenv(TYPECHECK_ENV, "off")
    assert typecheck_mode({"typecheck": "required"}) == "off"
    monkeypatch.setenv(TYPECHECK_ENV, "sometimes")
    with pytest.raises(ValueError, match="typecheck must be one of"):
        typecheck_mode({})


def test_without_codon_auto_warns_and_required_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.registry import register_scenario, unregister_scenario
    from autoware_carla_scenario.typecheck import ENV_CODON, ScenarioTypeError
    from autoware_carla_scenario.typecheck.mode import check_registered_scenario

    monkeypatch.setenv(ENV_CODON, str(tmp_path / "no-such-codon"))
    register_scenario("typecheck_mode_case", _ModeScenario, _Config)  # type: ignore[arg-type]
    try:
        assert check_registered_scenario("typecheck_mode_case", {}, "off") is None
        result = check_registered_scenario("typecheck_mode_case", {}, "auto")
        assert result is not None and result.skipped is not None
        assert "static check skipped" in caplog.text
        with pytest.raises(ScenarioTypeError, match="no Codon compiler"):
            check_registered_scenario("typecheck_mode_case", {}, "required")
    finally:
        unregister_scenario("typecheck_mode_case")


def test_a_codon_of_another_release_series_counts_as_no_codon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from autoware_carla_scenario.typecheck import (
        ENV_CODON,
        ScenarioTypeError,
        ToolchainError,
        find_supported_codon,
    )
    from autoware_carla_scenario.registry import register_scenario, unregister_scenario
    from autoware_carla_scenario.typecheck.mode import check_registered_scenario

    codon = tmp_path / "bin" / "codon"
    codon.parent.mkdir()
    codon.write_text("#!/bin/sh\necho 0.18.2\n")
    codon.chmod(0o755)
    monkeypatch.setenv(ENV_CODON, str(codon))
    with pytest.raises(ToolchainError, match=r"Codon 0\.18\.2.*Codon 0\.19\.x"):
        find_supported_codon()
    register_scenario("typecheck_version_case", _ModeScenario, _Config)  # type: ignore[arg-type]
    try:
        result = check_registered_scenario("typecheck_version_case", {}, "auto")
        assert result is not None and result.skipped is not None
        with pytest.raises(ScenarioTypeError, match="no Codon compiler"):
            check_registered_scenario("typecheck_version_case", {}, "required")
    finally:
        unregister_scenario("typecheck_version_case")


def test_a_scenario_with_a_custom_builder_is_not_checked() -> None:
    from autoware_carla_scenario.registry import (
        get_scenario_classes,
        register_scenario,
        register_scenario_builder,
        unregister_scenario,
    )
    from autoware_carla_scenario.typecheck.mode import check_registered_scenario

    register_scenario("typecheck_builder_case", _ModeScenario, _Config)  # type: ignore[arg-type]
    assert get_scenario_classes("typecheck_builder_case") == (_ModeScenario, _Config)
    register_scenario_builder("typecheck_builder_case", lambda *args: None)  # type: ignore[arg-type,return-value]
    try:
        assert get_scenario_classes("typecheck_builder_case") is None
        assert (
            check_registered_scenario("typecheck_builder_case", {}, "required") is None
        )
    finally:
        unregister_scenario("typecheck_builder_case")
    assert get_scenario_classes("typecheck_builder_case") is None


# ---------------------------------------------------------------------------
# except (A, B): one handler per class (#118)
# ---------------------------------------------------------------------------


_TUPLE_EXCEPTS = textwrap.dedent(
    """
    def probe(repo):
        try:
            run(repo)
        except (OSError, subprocess.SubprocessError) as exc:  # why
            log(exc)
            try:
                retry()
            except (KeyError,
                    ValueError):
                pass
        except RuntimeError:
            return 2
        else:
            return 3
        return 0
    """
)


def _handlers(tree: ast.AST) -> list[list[tuple[Optional[str], Optional[str], str]]]:
    """Each try statement's handlers: (class, name, body) in order."""
    return [
        [
            (
                ast.unparse(h.type) if h.type else None,
                h.name,
                "; ".join(ast.unparse(s) for s in h.body),
            )
            for h in node.handlers
        ]
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
    ]


def test_a_tuple_except_becomes_one_handler_per_class_in_order() -> None:
    out, _ = split_tuple_excepts(_TUPLE_EXCEPTS)
    handlers = _handlers(ast.parse(out))
    # The outer try: OSError, then SubprocessError, each with the same body
    # (the inner try split too), then the handler that followed the tuple.
    outer = handlers[0]
    assert [(c, n) for c, n, _ in outer] == [
        ("OSError", "exc"),
        ("subprocess.SubprocessError", "exc"),
        ("RuntimeError", None),
    ]
    assert outer[0][2] == outer[1][2]
    inner = handlers[1:]
    assert len(inner) == 2  # the inner try, once in each copy
    for handlers_of_copy in inner:
        assert [c for c, _, _ in handlers_of_copy] == ["KeyError", "ValueError"]
    assert not [
        h
        for h in ast.walk(ast.parse(out))
        if isinstance(h, ast.ExceptHandler) and isinstance(h.type, ast.Tuple)
    ]


def test_a_tuple_except_catches_what_it_caught() -> None:
    # Python runs the split source as it ran the original: the same
    # exceptions reach the same body, and the others the handlers after it.
    source = textwrap.dedent(
        """
        def caught(error):
            try:
                raise error
            except (KeyError, LookupError) as exc:
                return "lookup " + type(exc).__name__
            except (ValueError,):
                return "value"
            except Exception:
                return "other"
        """
    )
    out, _ = split_tuple_excepts(source)
    assert "(KeyError, LookupError)" not in out and "(ValueError,)" not in out
    for text in (source, out):
        namespace: dict[str, Any] = {}
        exec(compile(text, "<split>", "exec"), namespace)  # noqa: S102
        caught = namespace["caught"]
        assert caught(KeyError()) == "lookup KeyError"
        assert caught(IndexError()) == "lookup IndexError"
        assert caught(ValueError()) == "value"
        assert caught(TypeError()) == "other"


def test_each_line_of_a_split_handler_maps_to_the_line_it_copies() -> None:
    out, origin = split_tuple_excepts(_TUPLE_EXCEPTS)
    source_lines = _TUPLE_EXCEPTS.splitlines()
    out_lines = out.splitlines()
    assert len(origin) == len(out_lines)
    for line, original in zip(out_lines, origin):
        if "except" not in line and line.strip() not in ("):", ""):
            assert line == source_lines[original - 1]
    # The last line of the module is the last line of the source.
    assert origin[-1] == len(source_lines)
    # transform_source_lines() gives the same map, after the prelude.
    text, mapped = transform_source_lines(_TUPLE_EXCEPTS)
    assert text.startswith(PRELUDE) and mapped == origin
    # Every other line is where it was.
    assert split_tuple_excepts("x = 1\ny = 2\n") == ("x = 1\ny = 2\n", [1, 2])


def test_a_class_written_over_lines_with_a_comment_is_copied_whole() -> None:
    # The comment inside the tuple would otherwise swallow the rest of the
    # class (`.Error`) and of the copy, which still parses.
    source = (
        "try:\n"
        "    pass\n"
        "except (errors  # why\n"
        "        .Error, KeyError) as exc:\n"
        "    pass\n"
    )
    out, origin = split_tuple_excepts(source)
    assert _handlers(ast.parse(out)) == [
        [("errors.Error", "exc", "pass"), ("KeyError", "exc", "pass")]
    ]
    assert origin == [1, 2, 3, 4, 5, 3, 4, 5]


@pytest.mark.parametrize(
    "handler",
    ["except errors:", "except (*errors,):", "except ():", "except Exception:"],
)
def test_a_tuple_that_is_not_written_out_is_left_as_it_is(handler: str) -> None:
    source = f"try:\n    pass\n{handler}\n    pass\n"
    assert split_tuple_excepts(source) == (source, [1, 2, 3, 4])


def test_an_empty_tuple_default_takes_the_type_of_its_field() -> None:
    out = _transform(
        """
        @dataclass(frozen=True)
        class Pin:
            warnings: tuple[str, ...] = field(default=())
            pair: tuple[str, int] = field(default=("a", 1))
            none: Optional[tuple[str, ...]] = field(default=None)
        """
    )
    # Checked as a List[str] (Codon's tuples have a fixed length).
    assert "    warnings: List[str] = List[str]()" in out
    assert '    pair: tuple[str, int] = ("a", 1)' in out
    assert "    none: Optional[List[str]] = None" in out


def test_a_tuple_display_default_of_a_sequence_field_is_a_list_of_its_type() -> None:
    # A `Sequence[X]` / `tuple[X, ...]` field is declared as a `List[X]` (the
    # typing shim's `Sequence` is `List`), and its tuple display would be a
    # fixed-length Codon tuple: a list of the
    # field's element type, which the constructor call leaving it out
    # compiles (#188).  Only the brackets are replaced, so an element is
    # rewritten as anywhere else.
    out = _transform(
        """
        @dataclass
        class Docker:
            colcon_args: Sequence[str] = ("--symlink-install", "--merge-install")
            one: Sequence[str] = ("a",)
            fixed: tuple[str, ...] = ("x", "y")
            wrapped: Sequence[int] = field(default=(1, 2))
            nested: Sequence[Set[str]] = (frozenset({"a"}),)
            pair: tuple[str, int] = ("a", 1)
            bare: Sequence[int] = 1, 2
            empty: Sequence["Pose"] = ()
            env: Mapping[str, str] = field(default_factory=dict)
            ids: Sequence[int] = field(default_factory=list)
            flags: AbstractSet[str] = field(default_factory=set)
        """
    )
    assert (
        '    colcon_args: Sequence[str] = List[str](["--symlink-install", '
        '"--merge-install"])' in out
    )
    assert '    one: Sequence[str] = List[str](["a",])' in out
    assert '    fixed: List[str] = List[str](["x", "y"])' in out
    assert "    wrapped: Sequence[int] = List[int]([1, 2])" in out
    assert '    nested: Sequence[Set[str]] = List[Set[str]]([set({"a"}),])' in out
    # A fixed-length tuple, a display without brackets, or no sequence: left
    # as written.
    assert '    pair: tuple[str, int] = ("a", 1)' in out
    assert "    bare: Sequence[int] = 1, 2" in out
    # An empty one is an empty list, as `field(default=())` is.
    assert "    empty: Sequence[Pose] = List[Pose]()" in out
    # A factory of an abstract collection builds the concrete one the typing
    # shim maps it onto, of the field's types (Codon would make a bare `{}` a
    # dict of None).
    assert "    env: Mapping[str, str] = Dict[str, str]()" in out
    assert "    ids: Sequence[int] = List[int]()" in out
    assert "    flags: AbstractSet[str] = Set[str]()" in out


def test_an_empty_tuple_default_of_a_sequence_parameter_is_an_empty_list() -> None:
    # The parameter's annotation is dropped (a caller may pass a tuple), so
    # its `()` would be Codon's empty tuple: it is typed as the list a
    # caller's sequence of the element type is, as `field(default=())` is.
    out = _transform(
        """
        def route(
            goal: Pose,
            waypoints: Sequence[Pose] = (),
            *,
            skip: Collection[int] = (),
            quoted: "Sequence[Pose]" = (),
            inner: Sequence["Pose"] = (),
            fixed: tuple[str, ...] = (),
            pair: tuple[str, int] = (),
            listed: list[int] = (),
            later: Sequence[Later] = (),
            filled: Sequence[int] = (1, 2),
            one: tuple[str, ...] = ("a",),
            filled_pair: tuple[str, int] = ("a", 1),
            starred: Sequence[int] = (*IDS,),
        ) -> None:
            pass


        class Later:
            pass
        """
    )
    assert "    waypoints = List[Pose]()," in out
    assert "    skip = List[int]()," in out
    assert "    quoted = List[Pose]()," in out
    assert "    inner = List[Pose]()," in out
    assert "    fixed = List[str]()," in out
    # Not a sequence of one element type, or annotated as Codon keeps it:
    # left as written.
    assert "    pair: tuple[str, int] = ()," in out
    assert "    listed: list[int] = ()," in out
    # A class defined further down, which Codon cannot name in a signature.
    assert "    later = ()," in out
    # A display is a list of its elements, as a field's (#188); a
    # fixed-length tuple, or one with a starred element, is left as written.
    assert "    filled = List[int]([1, 2])," in out
    assert '    one = List[str](["a",]),' in out
    assert '    filled_pair: tuple[str, int] = ("a", 1),' in out
    assert "    starred = (*IDS,)," in out


def test_a_none_default_of_a_dropped_optional_parameter_is_its_types_none() -> None:
    # An Optional of a sequence, or of a Callable, is dropped too, and a `None`
    # default would be Codon's NoneType, which has no element and no call:
    # it is the None of the Optional the library check calls with.
    out = _transform(
        """
        def configure(
            poses: Optional[Sequence[Pose]] = None,
            ground: Optional[Callable[[float, float], Optional[float]]] = None,
            lookup: Callable[[Union[A, B]], Pose] | None = None,
            label: Optional[str] = None,
            anything: Optional[Callable[..., float]] = None,
        ) -> None:
            pass
        """
    )
    assert "    poses = Optional[List[Pose]]()," in out
    assert (
        "    ground = Optional[Callable[[float, float], Optional[float]]](),"
    ) in out
    assert "    lookup = Optional[_acs_Returning[Pose, 1]]()," in out
    # An annotation Codon keeps types its None already.
    assert "    label: Optional[str] = None," in out
    # Nothing Codon can express: left as written.
    assert "    anything = None," in out


def test_an_empty_list_or_dict_default_takes_the_type_of_its_parameter() -> None:
    # Codon types a bare `[]` from what is put in it, and nothing is: a list
    # of None, which a `list[int]` parameter refuses.
    out = _transform(
        """
        def start(
            ids: list[int] = [],
            names: Dict[str, int] = {},
            seq: Sequence[str] = [],
            full: list[int] = [1],
            bare: list = [],
        ) -> None:
            pass
        """
    )
    assert "    ids: list[int] = list[int]()," in out
    assert "    names: Dict[str, int] = Dict[str, int]()," in out
    assert "    seq = List[str]()," in out
    assert "    full: list[int] = [1]," in out
    assert "    bare: list = []," in out


# ---------------------------------------------------------------------------
# An inherited __init__ (#201) and the None of an Optional return (#202)
# ---------------------------------------------------------------------------

_INHERITED_INIT = (
    "@extend\nclass {name}:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        super().__init__(*args, **kwargs)\n"
)


def test_a_class_inheriting_an_init_is_given_one_passing_everything_on() -> None:
    source = textwrap.dedent(
        """
        class A:
            def __init__(self, x: int, *, config: str = "") -> None:
                self.x = x
        class B(A):
            def f(self) -> int:
                return 1
        class C(B):
            pass
        class D(A):
            def __init__(self) -> None:
                super().__init__(1)
        class E:
            y: int = 0
        """
    )
    out, origin = transform_source_lines(source)
    body = out[len(PRELUDE) :]
    module, _, appended = body.partition("\n\n@extend\n")
    assert module.count("\n") == source.count("\n")  # no line of the source moves
    assert _INHERITED_INIT.format(name="B") in out
    assert _INHERITED_INIT.format(name="C") in out  # through B
    for name in ("A", "D", "E"):  # their own __init__, or none to inherit
        assert f"class {name}:" not in appended
    # An error in an appended constructor is reported at its class's line.
    lines = out.splitlines()
    prelude = PRELUDE.count("\n")
    assert len(origin) == len(lines) - prelude
    source_lines = source.splitlines()
    for number, line in enumerate(lines[prelude:], start=1):
        if "super().__init__(*args, **kwargs)" in line:
            assert source_lines[origin[number - 1] - 1].startswith(
                ("class B", "class C")
            )


def test_an_init_of_a_base_defined_elsewhere_is_inherited() -> None:
    source = textwrap.dedent(
        """
        from autoware_carla_scenario import BaseScenario
        from .parent import Parent as Base
        class Mine(BaseScenario):
            pass
        class Other(Base):
            pass
        class Unknown(Elsewhere):
            pass
        """
    )
    out = transform_source(source, init_classes={"BaseScenario", "Parent"})
    assert _INHERITED_INIT.format(name="Mine") in out
    assert _INHERITED_INIT.format(name="Other") in out  # by its own name
    assert "class Unknown:" not in out  # a base with no __init__ known
    assert "@extend" not in transform_source(source)


def test_an_init_of_a_later_base_is_called_through_that_base() -> None:
    # Python takes the __init__ of the first base that has one; Codon's
    # super() is the first base's, a mixin's generated constructor here.
    source = textwrap.dedent(
        """
        import base
        class Mixin:
            def hello(self) -> int:
                return 1
        class A(Mixin, base.Parent):
            pass
        class B(base.Parent, Mixin):
            pass
        """
    )
    out = transform_source(source, init_classes={"Parent"})
    assert (
        "@extend\nclass A:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        base.Parent.__init__(self, *args, **kwargs)\n"
    ) in out
    assert _INHERITED_INIT.format(name="B") in out
    assert "class Mixin:\n    def __init__" not in out


def test_a_dataclass_enum_or_exception_inherits_no_init() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            class A:
                def __init__(self, x: int) -> None:
                    self.x = x
            @dataclass
            class B(A):
                y: int = 0
            class C(B):
                pass
            class Color(A, Enum):
                RED = 1
            class Failed(ValueError):
                pass
            """
        )
    )
    # A dataclass generates its own constructor, and so does Codon for a class
    # deriving from it; an exception derives statically.
    assert inherited_inits(tree, {"ValueError"}, {"Failed"}) == []
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    assert classes_with_init(classes) == {"A", "Color"}
    assert classes_with_init(classes, {"ValueError"}) == {
        "A",
        "Color",
        "Failed",
        "ValueError",
    }


def test_return_none_of_an_optional_function_is_the_none_of_its_type() -> None:
    out = _transform(
        """
        def find(world: carla.World) -> Optional[carla.Actor]:
            if world:
                return
            def inner() -> int | None:
                return None
            fallback = lambda: None
            return None
        def where() -> carla.Transform | None:
            return None  # nothing
        def items() -> Optional[tuple[int, ...]]:
            return None
        def plain() -> None:
            return None
        def gen() -> Optional[int]:
            yield 1
            return None
        """
    )
    assert "    if world:\n        return Optional[carla.Actor]()\n" in out
    assert "\n    return Optional[carla.Actor]()\n" in out
    # A nested function keeps its own, and is rewritten for its own type.
    assert "        return Optional[int]()\n" in out
    assert "fallback = lambda: None" in out
    assert "    return Optional[carla.Transform]()  # nothing\n" in out
    assert "    return Optional[List[int]]()\n" in out
    assert "def plain() -> None:\n    return None\n" in out
    assert "    yield 1\n    return None\n" in out  # a generator


def test_return_none_is_left_when_the_annotation_is_dropped() -> None:
    out = _transform(
        """
        def f() -> Optional[Later]:
            return None
        def g() -> Optional[Any]:
            return None
        class Later:
            pass
        """
    )
    assert out.count("    return None\n") == 2


# ---------------------------------------------------------------------------
# A module of the scenario's package imported as a module (#310)
# ---------------------------------------------------------------------------

_OWN_MODULES = frozenset(
    {"pkg", "pkg.configs", "pkg.helpers", "pkg.sub", "pkg.sub.mod", "pkg.scenario"}
)


@pytest.mark.parametrize(
    ("written", "rewritten"),
    [
        ("from . import helpers", "import helpers as helpers"),
        ("from pkg import helpers", "import helpers as helpers"),
        ("import pkg.helpers as helpers", "import helpers as helpers"),
        ("from . import helpers as h", "import helpers as h"),
        ("from .sub import mod", "import sub.mod as mod"),
        ("import pkg.sub.mod as m", "import sub.mod as m"),
        ("from . import sub", "import sub as sub"),
        (
            "from . import helpers, configs as c",
            "import helpers as helpers; import configs as c",
        ),
        # A name beside a module stays as written, on the same line.
        (
            "from pkg import helpers, VALUE",
            "from pkg import VALUE; import helpers as helpers",
        ),
        ("import os, pkg.helpers as h", "import os; import helpers as h"),
        # Names, and other packages' modules, are left alone.
        ("from .helpers import speed_kph", "from .helpers import speed_kph"),
        ("from . import VALUE", "from . import VALUE"),
        ("from os import path", "from os import path"),
        ("import os.path as p", "import os.path as p"),
    ],
)
def test_an_own_module_imported_as_a_module_is_imported_from_its_directory(
    written: str, rewritten: str
) -> None:
    source = f"{written}\nx = 1\n"
    out, problems = relative_module_imports(
        source, "pkg.scenario", False, _OWN_MODULES.__contains__
    )
    assert problems == []
    assert out == f"{rewritten}\nx = 1\n"


def test_the_rewrite_keeps_every_line_and_reaches_functions() -> None:
    source = textwrap.dedent(
        """\
        from . import (
            helpers,
        )
        def f() -> int:
            from . import helpers as h
            return h.VALUE
        """
    )
    out, problems = relative_module_imports(
        source, "pkg.scenario", False, _OWN_MODULES.__contains__
    )
    assert problems == []
    lines = out.splitlines()
    assert len(lines) == len(source.splitlines())
    assert lines[0] == "import helpers as helpers"
    assert lines[3:] == [
        "def f() -> int:",
        "    import helpers as h",
        "    return h.VALUE",
    ]


def test_a_package_s_init_imports_its_modules_from_its_directory() -> None:
    out, problems = relative_module_imports(
        "from . import mod\n", "pkg.sub", True, _OWN_MODULES.__contains__
    )
    assert (out, problems) == ("import mod as mod\n", [])


@pytest.mark.parametrize(
    ("module", "written", "form"),
    [
        # Outside the importing module's package: no form Codon resolves.
        ("pkg.sub.mod", "from .. import helpers", "`from ..helpers import name`"),
        (
            "pkg.sub.mod",
            "import pkg.helpers as helpers",
            "`from ..helpers import name`",
        ),
        ("pkg.sub.mod", "import pkg as p", "`from .. import name`"),
        ("pkg.scenario", "import pkg as p", "`from . import name`"),
        # Bound without a name: `pkg`.
        ("pkg.scenario", "import pkg.helpers", "`import pkg.helpers as helpers`"),
    ],
)
def test_an_own_module_import_codon_cannot_resolve_is_refused_with_the_working_form(
    module: str, written: str, form: str
) -> None:
    source = f"x = 1\n{written}\n"
    out, problems = relative_module_imports(
        source, module, False, _OWN_MODULES.__contains__
    )
    assert out == source
    assert len(problems) == 1, problems
    line, message = problems[0]
    assert line == 2
    assert f"`{written}`" in message
    assert form in message, message


def test_a_top_level_module_imports_its_sibling_as_written() -> None:
    # No package: the importing file's directory is the root, so `import
    # helpers` is already the form Codon resolves, and is not refused.
    source = "import helpers\nimport helpers as h\n"
    out, problems = relative_module_imports(
        source, "scenario", False, {"helpers"}.__contains__
    )
    assert problems == []
    assert out == "import helpers as helpers\nimport helpers as h\n"


# ---------------------------------------------------------------------------
# An import repeated in a module (#354)
# ---------------------------------------------------------------------------


def _body(text: str) -> str:
    return transform_source(text)[len(PRELUDE) :]


def test_a_repeated_import_is_dropped_on_its_line() -> None:
    out = _body(
        "import logging\n"
        "from pkg import A, B\n"
        "import logging\n"
        "from pkg import A\n"
        "x = A\n"
    )
    assert out == ("import logging\nfrom pkg import A, B\npass\npass\nx = A\n")


def test_a_repeat_inside_one_statement_and_beside_new_names() -> None:
    out = _body("from pkg import A, A\nfrom pkg import A, C as D, A\nimport os, os\n")
    assert out == "from pkg import A\nfrom pkg import C as D\nimport os\n"


def test_a_repeat_under_the_same_alias_is_dropped_but_not_a_new_alias() -> None:
    out = _body(
        "from pkg import A as B\nfrom pkg import A as B\nfrom pkg import A as C\n"
        "from pkg import A\n"
    )
    assert (
        out
        == "from pkg import A as B\npass\nfrom pkg import A as C\nfrom pkg import A\n"
    )


def test_a_relative_repeat_is_dropped_only_at_the_same_level() -> None:
    out = _body("from .m import A\nfrom .m import A\nB = A\nfrom ..m import A\n")
    assert out == "from .m import A\npass\nB = A\nfrom ..m import A\n"


def test_imports_in_functions_and_branches_are_left_alone() -> None:
    text = (
        "from pkg import A\n"
        "def f():\n"
        "    from pkg import A\n"
        "    from pkg import A\n"
        "if True:\n"
        "    from pkg import A\n"
    )
    assert _body(text) == text


def test_different_dotted_imports_are_no_repeat() -> None:
    text = "import a.b\nimport a.c\nimport a.b\n"
    assert _body(text) == "import a.b\nimport a.c\npass\n"
    assert conflicting_imports(text) == []


def test_a_repeated_import_keeps_every_line() -> None:
    text = "from pkg import (\n    A,\n)\nfrom pkg import (\n    A,\n)\nx = 1\n"
    out, origin = transform_source_lines(text)
    assert out.count("\n") - PRELUDE.count("\n") == text.count("\n")
    assert len(origin) == text.count("\n")
    assert out.endswith("x = 1\n")


def test_a_name_read_before_it_is_rebound_is_reported_at_the_later_import() -> None:
    problems = conflicting_imports(
        "from a import X\nY = X\nimport b as X\nfrom c import Y2, Y2\n"
    )
    assert [line for line, _ in problems] == [3]
    assert "`X` is imported again" in problems[0][1]
    assert "line 1" in problems[0][1]
    assert conflicting_imports("from a import X\nfrom a import X\n") == []


def test_a_rebound_name_nothing_read_drops_the_earlier_import() -> None:
    text = (
        "from a import X, Z\ndef f():\n    return X\nfrom b import X\nfrom a import X\n"
    )
    assert _body(text) == (
        "from a import Z\ndef f():\n    return X\npass\nfrom a import X\n"
    )
    assert conflicting_imports(text) == []


def test_a_read_in_a_class_body_or_decorator_counts_as_a_use() -> None:
    for use in ("class C:\n    v = X\n", "@X\ndef f(): ...\n", "def f(d=X): ...\n"):
        text = f"from a import X\n{use}from b import X\n"
        assert [p[0] for p in conflicting_imports(text)] == [text.count("\n")], use


def test_a_name_assigned_between_two_identical_imports_keeps_the_second() -> None:
    text = "from a import X\nX = 1\nfrom a import X\n"
    assert _body(text) == text


def test_star_imports_and_future_imports_are_not_rebindings() -> None:
    text = "from __future__ import annotations\nfrom a import *\nfrom b import *\n"
    assert _body(text) == text
    assert conflicting_imports(text) == []


def test_a_try_import_fallback_is_left_alone() -> None:
    text = "from a import X\ntry:\n    from b import X\nexcept ImportError:\n    pass\n"
    assert _body(text) == text


# The methods @dataclass generates, and a NamedTuple as a @tuple (#365)


def _appended(source: str, name: str) -> str:
    """The ``@extend`` block appended for class *name*."""
    out = transform_source(textwrap.dedent(source))
    blocks = out.split("\n\n@extend\n")[1:]
    return next((b for b in blocks if b.startswith(f"class {name}:")), "")


def _methods(block: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(block))
        if isinstance(node, ast.FunctionDef)
    }


def test_a_dataclass_gets_eq_and_ne_comparing_its_fields() -> None:
    block = _appended(
        """
        @dataclass
        class Fz:
            a: int
            b: str = "x"
        """,
        "Fz",
    )
    assert _methods(block) == {"__eq__", "__ne__"}
    assert "return self.a == other.a and self.b == other.b" in block
    assert "return not self.__eq__(other)" in block
    ast.parse(block.replace("class Fz", "@extend\nclass Fz", 1))


@pytest.mark.parametrize(
    ("decorator", "methods"),
    [
        ("@dataclass", {"__eq__", "__ne__"}),
        ("@dataclass(eq=False)", set()),
        ("@dataclass(frozen=True)", {"__eq__", "__ne__", "__hash__"}),
        ("@dataclass(unsafe_hash=True)", {"__eq__", "__ne__", "__hash__"}),
        ("@dataclass(eq=False, unsafe_hash=True)", {"__hash__"}),
        (
            "@dataclasses.dataclass(order=True)",
            {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"},
        ),
    ],
)
def test_a_dataclasss_flags_pick_the_methods_generated(
    decorator: str, methods: set[str]
) -> None:
    source = f"{decorator}\nclass Fz:\n    a: int\n"
    out = transform_source(source)
    generated = {
        node.name
        for node in ast.walk(ast.parse(out.partition("\n\n@extend\n")[2] or "pass"))
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_acs_with__")
    }
    assert generated == methods, out


def test_order_compares_the_field_tuples_and_hash_hashes_them() -> None:
    out = transform_source(
        "@dataclass(frozen=True, order=True)\nclass Fz:\n    a: int\n    b: int\n"
    )
    assert "return (self.a, self.b,) < (other.a, other.b,)" in out
    assert "return (self.a, self.b,) >= (other.a, other.b,)" in out
    assert "return hash((self.a, self.b,))" in out


def test_a_dataclass_inheriting_a_dataclass_compares_the_base_fields_first() -> None:
    out = transform_source(
        textwrap.dedent(
            """
            @dataclass(frozen=True)
            class Base:
                a: int

            @dataclass(frozen=True)
            class Sub(Base):
                b: int
                a: int = 0
            """
        )
    )
    sub = out.split("class Sub:")[1]  # the dunders; the copies follow
    assert "self.a == other.a and self.b == other.b" in sub
    assert "hash((self.a, self.b,))" in sub


def test_fields_that_are_not_compared_are_left_out() -> None:
    out = transform_source(
        textwrap.dedent(
            """
            @dataclass
            class Fz:
                a: int
                b: ClassVar[int] = 1
                c: int = field(default=0, compare=False)
                d: InitVar[int] = 0
            """
        )
    )
    assert "return self.a == other.a\n" in out


def test_a_method_the_class_defines_is_kept() -> None:
    out = transform_source(
        textwrap.dedent(
            """
            @dataclass(frozen=True, order=True)
            class Fz:
                a: int

                def __eq__(self, other: "Fz") -> bool:
                    return True

                def __lt__(self, other: "Fz") -> bool:
                    return False
            """
        )
    )
    appended = out.partition("\n\n@extend\n")[2]
    assert "def __eq__" not in appended
    assert "def __lt__" not in appended
    assert "__hash__" in appended and "__le__" in appended


@pytest.mark.parametrize(
    "bases",
    ["FrameworkConfig", "Generic[T]", "Mixin, FrameworkConfig"],
)
def test_a_dataclass_with_a_base_it_cannot_see_is_left_alone(bases: str) -> None:
    out = transform_source(f"@dataclass\nclass Fz({bases}):\n    a: int\n")
    assert "@extend" not in out


def test_the_dunders_map_to_the_line_of_their_class() -> None:
    source = "x = 1\n\n\n@dataclass(frozen=True)\nclass Fz:\n    a: int\n"
    out, origin = transform_source_lines(source)
    extra = out.count("\n") - PRELUDE.count("\n") - len(origin)
    assert extra == 0
    assert len(origin) == out.count("\n") - PRELUDE.count("\n")
    assert origin[-1] == 5  # the class line, for every appended line
    assert origin[:6] == [1, 2, 3, 4, 5, 6]


def test_a_namedtuple_class_becomes_a_tuple_class_on_the_same_lines() -> None:
    source = (
        "import typing\n\n\nclass Pt(typing.NamedTuple):\n"
        "    x: int\n    y: float = 1.0\n\n\nz = Pt(1)\n"
    )
    out, origin = transform_source_lines(source)
    body = out[len(PRELUDE) :]
    assert "@tuple\nclass Pt():\n    x: int\n    y: float = 1.0\n" in body
    # The decorator is a new line, mapped to its class.
    lines = body.splitlines()
    assert origin[lines.index("@tuple")] == 4
    assert origin[lines.index("    x: int")] == 5
    assert origin[lines.index("z = Pt(1)")] == 9


def test_a_decorated_or_nested_namedtuple_gets_the_decorator_above_it() -> None:
    source = (
        "from typing import NamedTuple\n\n\n"
        "def f():\n    class Q(NamedTuple):\n        a: int\n    return Q\n"
    )
    out = transform_source(source)
    assert "    @tuple\n    class Q():\n" in out
    ast.parse(out[len(PRELUDE) :])


def test_other_tuple_classes_are_not_made_tuples() -> None:
    out = transform_source("class A(NamedTuple, Other):\n    x: int\n")
    assert "@tuple" not in out


def test_final_and_annotated_are_read_as_what_they_wrap() -> None:
    assert _annotation("Final[int]") == "int"
    assert _annotation("typing.Final[List[str]]") == "List[str]"
    assert _annotation('Annotated[float, "m/s"]') == "float"
    assert _annotation("Final") is None
    assert _annotation("NoReturn") is None
    assert _annotation("Sized") is None
    assert _annotation("Hashable") is None
    assert _transform("LIMIT: Final = 3\n") == "LIMIT = 3\n"
    out = _transform("class C:\n    LIMIT: Final = 3\n")
    assert out == "class C:\n    LIMIT = 3\n"


def test_cast_is_kept_for_a_codon_type_and_dropped_for_any_other() -> None:
    assert _transform("y = cast(int, x)\n") == "y = cast(int, x)\n"
    assert _transform('y = cast("int", x)\n') == "y = cast(int, x)\n"
    assert _transform("y = cast(Callable[[int], int], f)(2)\n") == "y = (f)(2)\n"
    assert _transform("y = typing.cast(Any, x)\n") == "y = (x)\n"


def test_functools_caching_decorators_are_dropped() -> None:
    source = textwrap.dedent(
        """
        import functools
        from functools import cache as memo, lru_cache

        @lru_cache(maxsize=8)
        def a(x: int) -> int:
            return x

        @functools.lru_cache()
        def b(x: int) -> int:
            return x

        @lru_cache
        def c(x: int) -> int:
            return x

        @functools.cache
        def d(x: int) -> int:
            return x

        class K:
            @memo
            def e(self, x: int) -> int:
                return x

            @lru_cache(
                maxsize=None,
            )
            def f(self, x: int) -> int:
                return x
        """
    )
    out = _transform(source)
    assert out.count("\n") == source.count("\n")
    assert "@" not in out
    assert "lru_cache" not in out.replace("import functools\n", "")
    assert out.count("def ") == 6


def test_functools_cached_property_becomes_a_property() -> None:
    source = textwrap.dedent(
        """
        import functools
        from functools import cached_property

        class K:
            @cached_property
            def a(self) -> int:
                return 1

            @functools.cached_property
            def b(self) -> int:
                return 2
        """
    )
    out = _transform(source)
    assert out.count("@property") == 2
    assert "cached_property" not in out


def test_functools_reduce_becomes_the_models() -> None:
    source = textwrap.dedent(
        """
        import functools as ft
        from functools import partial, reduce as fold

        def total(xs: list[int]) -> int:
            return fold(lambda a, b: a + b, xs, 0) + ft.reduce(max, xs)
        """
    )
    out = _transform(source)
    assert "_acs_reduce(lambda a, b: a + b, xs, 0) + _acs_reduce(max, xs)" in out
    # Only the names Codon lacks leave the import; partial stays.
    assert "from functools import partial\n" in out
    assert "fold" not in out.replace("_acs_reduce", "")


def test_a_functools_import_of_only_stubbed_names_is_dropped() -> None:
    source = "from functools import (\n    cache,\n    reduce,\n)\nx = 1\n"
    out = _transform(source)
    assert out.count("\n") == source.count("\n")
    assert "from functools" not in out
    assert "pass" in out


def test_another_modules_reduce_and_cache_are_left_alone() -> None:
    source = "from mylib import reduce, cache\n\n\ndef f(xs):\n    return reduce(xs)\n"
    out = _transform(source)
    assert "from mylib import reduce, cache" in out
    assert "_acs_reduce(xs)" not in out


def test_a_rebound_functools_name_is_left_alone() -> None:
    source = (
        "from functools import reduce, cache\n\n\n"
        "def f(reduce, xs):\n    return reduce(xs)\n\n\n"
        "def cache(g):\n    return g\n\n\n"
        "@cache\ndef h():\n    pass\n"
    )
    out = _transform(source)
    assert "_acs_reduce(xs)" not in out
    assert "return reduce(xs)" in out
    assert "@cache\ndef h()" in out


# ---------------------------------------------------------------------------
# Valid Python Codon refuses or crashes on (#376)
# ---------------------------------------------------------------------------


def _rewritten(source: str) -> str:
    text, _ = python_rewrites(textwrap.dedent(source))
    ast.parse(text)
    return text


def test_an_ellipsis_in_a_class_body_becomes_pass() -> None:
    text = _rewritten(
        """
        class A: ...
        class B(Exception):
            ...
        class C:
            x: int
            ...
        def f() -> None: ...
        """
    )
    assert "class A: pass" in text
    assert text.count("pass") == 3
    assert "def f() -> None: ..." in text


def test_a_dict_display_with_double_star_becomes_a_union() -> None:
    text = _rewritten(
        """
        d = {**a, **b}
        e = {**a}
        f = {"x": 1, **a, "y": 2, **(a or b)}
        """
    )
    assert "d = (a | b)" in text
    assert "e = dict(a)" in text
    assert 'f = ({"x": 1} | a | {"y": 2} | (a or b))' in text


def test_a_dict_display_with_double_star_keeps_its_lines() -> None:
    source = "d = {\n    **a,\n    'k': 1,\n    **b,\n}\nz = 1\n"
    text, _ = python_rewrites(source)
    assert text.count("\n") == source.count("\n")
    assert text.endswith("\nz = 1\n")
    assert text.startswith("d = (a | {'k': 1} | b)")


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ('"{:>5}|{}".format(1, 2)', 'f"{1:>5}|{2}"'),
        ('"{0}-{0}".format(x)', 'f"{x}-{x}"'),
        ('"{a}-{b:.2f}".format(a=x, b=y)', 'f"{x}-{y:.2f}"'),
        ('"{!r} {!s}".format(x, y)', 'f"{x!r} {y!s}"'),
        ('"{0.size[1]}".format(x)', 'f"{x.size[1]}"'),
        ('"{0[key]}".format(x)', "f\"{x['key']}\""),
        ('"plain".format()', '"plain"'),
        ('"{{}}".format()', '("{" + "}")'),
        ('"a{{{}}}".format(x)', '("a" + "{" + f"{x}" + "}")'),
    ],
)
def test_str_format_on_a_literal_becomes_an_fstring(call: str, expected: str) -> None:
    assert _rewritten(f"s = {call}\n") == f"s = {expected}\n"


def test_str_format_evaluates_a_complex_argument_once() -> None:
    text = _rewritten('s = "{0}{0}{k}".format(f(1), k=g("q"))\n')
    assert text == (
        's = (lambda _acs_f0, _acs_k_k: f"{_acs_f0}{_acs_f0}{_acs_k_k}")'
        '(f(1), g("q"))\n'
    )
    assert _rewritten('s = "x".format(f())\n') == 's = (lambda _acs_f0: "x")(f())\n'


def test_str_format_keeps_its_lines() -> None:
    source = 's = ("a {} "\n     "b {}").format(\n    1,\n    2,\n)\nz = 1\n'
    text, _ = python_rewrites(source)
    assert text.count("\n") == source.count("\n")
    assert text.endswith("\nz = 1\n")


def test_str_format_inside_str_format_is_rewritten_too() -> None:
    text = _rewritten('s = "{}".format("[{}]".format(f(1)))\n')
    assert ".format" not in text


@pytest.mark.parametrize(
    ("call", "hint"),
    [
        ('"{:>{}}".format(1, 2)', "a field inside it"),
        ('"{} {}".format(1)', "positional argument 1 but only 1 given"),
        ('"{k}".format(j=1)', "keyword 'k'"),
        ('"{} {1}".format(1, 2)', "mixes automatic and manual"),
        ('"{}".format(*a)', "*args or **kwargs"),
        ('"{}".format(**a)', "*args or **kwargs"),
        ('"{".format(1)', "invalid"),
        ('"{0!x}".format(1)', "unknown conversion"),
    ],
)
def test_str_format_that_cannot_be_expressed_is_refused(call: str, hint: str) -> None:
    refused = python_rewrite_refusals(f"x = 1\ns = {call}\n")
    assert [line for line, _ in refused] == [2]
    assert hint in refused[0][1]


def test_str_format_on_something_else_is_left_alone() -> None:
    text = _rewritten("s = fmt.format(1)\nt = obj.format(1)\n")
    assert text == "s = fmt.format(1)\nt = obj.format(1)\n"
    assert python_rewrite_refusals(text) == []


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ('f"{x:+.2f}"', '((x).__format__("+.2f"))'),
        ('f"{x: .2f}|{n:-d}"', '((x).__format__(" .2f") + "|" + (n).__format__("-d"))'),
        ('f"{x:^+10.2f}"', '((x).__format__("^+10.2f"))'),
        ('f"{x:>{w}.2f}"', '((x).__format__(">" + str(w) + ".2f"))'),
        ('f"{n:{w}d}"', '((n).__format__(str(w) + "d"))'),
        ('f"{s!r:{w}}"', "(repr(s).__format__(str(w)))"),
        ('f"{s!s:>{w!r}}"', '(str(s).__format__(">" + repr(w)))'),
        ('f"{x:{w:+d}}"', '((x).__format__((w).__format__("+d")))'),
        ('f"{x:q}"', '((x).__format__("q"))'),
        ('f"{x=:+.2f}"', '("x=" + (x).__format__("+.2f"))'),
        ('f"a {n} {x:+.1f} b"', '(f"a {n} " + (x).__format__("+.1f") + " b")'),
        ('f"{{{x:+d}}}"', '("{" + (x).__format__("+d") + "}")'),
        ('f"{f(a, b)[0]:+d}"', '((f(a, b)[0]).__format__("+d"))'),
        # ast.unparse spells a lambda "lambda : 1" before Python 3.11.
        (
            'f"{(lambda: 1)():+d}"',
            f"((({ast.unparse(ast.parse('lambda: 1', mode='eval').body)})())"
            '.__format__("+d"))',
        ),
    ],
)
def test_an_fstring_spec_codon_cannot_parse_becomes_a_format_call(
    field: str, expected: str
) -> None:
    assert _rewritten(f"r = {field}\n") == f"r = {expected}\n"


@pytest.mark.parametrize(
    "field",
    [
        'f"{x:.2f}|{x:e}|{x:g}|{x:.1%}|{x:010.3f}|{n:#x}|{n:08b}|{n:,}|{n:_x}"',
        'f"{n:>8}|{n:<8}|{n:^8}|{n:*^8}|{s!r:>8}|{s!s:^6}|{x=}|{x=:.2f}"',
        'f"{x}{y!r}"',
        "f\"{d['k']:>5}\"",
    ],
)
def test_an_fstring_codon_parses_is_left_alone(field: str) -> None:
    assert _rewritten(f"r = {field}\n") == f"r = {field}\n"


def test_a_rewritten_fstring_keeps_its_lines() -> None:
    source = 'r = f"""a\n{x:+.2f}\nb"""\nz = 1\nt = (f"{x:+d}"\n     f"{y}")\n'
    text, _ = python_rewrites(source)
    assert text.count("\n") == source.count("\n")
    assert "\nz = 1\n" in text
    assert '(x).__format__("+d")' in text


def test_str_format_with_a_sign_composes_with_the_fstring_rewrite() -> None:
    assert _rewritten('s = "{:+.2f}".format(x)\n') == ('s = ((x).__format__("+.2f"))\n')
    text = _rewritten('s = "{:+.2f}|{!r:>5}".format(f(x), y)\n')
    assert ".format(" not in text and '__format__("+.2f")' in text
    assert python_rewrite_refusals('s = "{:+.2f}".format(x)\n') == []


@pytest.mark.parametrize(
    ("field", "hint"),
    [
        ("f\"{'{}'.format(x)}\"", "a brace in an expression"),
        ("f\"{'{:>3}'.format(x):+d}\"", "a brace in an expression"),
        ('f"{len({1, 2})}"', "a brace in an expression"),
        ('f"{ {k: 1 for k in a}[1] }"', "a brace in an expression"),
        ('f"{ {**a}[1] }"', "a brace in an expression"),
        ("f\"{f'{x}'}\"", "f-string inside an f-string"),
        ('f"{x!a:+d}"', "!a together with a format spec"),
    ],
)
def test_an_fstring_with_a_brace_in_an_expression_is_refused(
    field: str, hint: str
) -> None:
    refused = python_rewrite_refusals(f"x = 1\nr = {field}\n")
    assert [line for line, _ in refused] == [2]
    assert hint in refused[0][1]


def test_zip_strict_is_dropped() -> None:
    assert _rewritten("zip(a, b, strict=True)\n") == "zip(a, b)\n"
    assert _rewritten("zip(strict=False)\n") == "zip()\n"
    assert _rewritten("zip(a, strict=x) + zip(b, strict=1)\n") == "zip(a) + zip(b)\n"
    source = "zip(\n    a,\n    b,\n    strict=True,\n)\n"
    text = _rewritten(source)
    assert text.count("\n") == source.count("\n")
    assert "strict" not in text
    assert _rewritten("other(a, strict=True)\n") == "other(a, strict=True)\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('s.startswith(("a", "b"))', '(s.startswith("a") or s.startswith("b"))'),
        ("self.p.endswith((x, y.z))", "(self.p.endswith(x) or self.p.endswith(y.z))"),
        ("s.startswith((a, b), 2)", "(s.startswith(a, 2) or s.startswith(b, 2))"),
        ('s.startswith(("a",))', '(s.startswith("a"))'),
    ],
)
def test_startswith_with_a_tuple_becomes_an_or(source: str, expected: str) -> None:
    assert _rewritten(source + "\n") == expected + "\n"


def test_startswith_with_a_tuple_evaluates_each_part_once() -> None:
    text = _rewritten("r = f(1).startswith((g(2), h(3)), k())\n")
    assert text.count("f(1)") == text.count("g(2)") == text.count("h(3)") == 1
    assert text.count("k()") == 1
    assert text == (
        "r = ((lambda _acs_a0, _acs_a1, _acs_a2, _acs_a3: "
        "_acs_a0.startswith(_acs_a1, _acs_a3) or "
        "_acs_a0.startswith(_acs_a2, _acs_a3))(f(1), g(2), h(3), k()))\n"
    )


def test_startswith_with_a_tuple_keeps_its_lines() -> None:
    source = "r = (\n    f(s)\n    .startswith(\n        (g(), 'z'),\n    )\n    and q\n)\nz = 1\n"
    text = _rewritten(source)
    assert text.count("\n") == source.count("\n")
    assert text.endswith("\nz = 1\n")
    assert "and q" in text.splitlines()[-3]


@pytest.mark.parametrize(
    "source",
    [
        "s.startswith(prefixes)",
        "s.startswith(('a', 'b'), *rest)",
        "s.startswith(())",
        "s.startswith(('a', 'b'), start=1)",
        "s.startswith('a')",
    ],
)
def test_startswith_it_cannot_rewrite_is_left_alone(source: str) -> None:
    assert _rewritten(source + "\n") == source + "\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("format(x, spec)", "x.__format__(spec)"),
        ('format(3.14159, ".2f")', '(3.14159).__format__(".2f")'),
        ('format(a + b, ">8")', '(a + b).__format__(">8")'),
        ("format(self.v, f(1))", "self.v.__format__(f(1))"),
        ("format(x)", "str(x)"),
        ("format(f(1))", "str(f(1))"),
        ("ascii(x)", "repr(x)"),
        ('ascii("b")', 'repr("b")'),
    ],
)
def test_format_and_ascii_are_rewritten(source: str, expected: str) -> None:
    assert _rewritten(source + "\n") == expected + "\n"


@pytest.mark.parametrize(
    "source",
    [
        "format()",
        "format(x, y, z)",
        "format(*a)",
        "format(x, spec=y)",
        "ascii(x, y)",
        "obj.format(x, y)",
    ],
)
def test_format_and_ascii_with_other_arguments_are_left_alone(source: str) -> None:
    assert _rewritten(source + "\n") == source + "\n"


@pytest.mark.parametrize(
    "source",
    [
        "def format(x, y):\n    return 1\nr = format(1, 2)",
        "from mod import ascii\nr = ascii(1)",
        "def f(format):\n    return format(1, 2)",
        "ascii = repr\nr = ascii(1)",
    ],
)
def test_format_and_ascii_bound_by_the_module_are_left_alone(source: str) -> None:
    assert _rewritten(source + "\n") == source + "\n"


@pytest.mark.parametrize(
    "source",
    [
        "class Rep:\n    def format(self):\n        return 'r'\nr = format(3, '05d')",
        "class Rep:\n    format = 1\nr = format(3, '05d')",
        "def render(format):\n    return format\nr = format(3, '05d')",
        "class Rep:\n    def ascii(self, x):\n        return 1\nr = ascii(3)",
        "def f():\n    ascii = 1\ng = lambda format: format\nr = format(3)",
    ],
)
def test_format_is_rewritten_where_no_enclosing_scope_binds_it(source: str) -> None:
    text = _rewritten(source + "\n")
    assert "r = " in text and (
        "__format__" in text or "str(3)" in text or "repr(3)" in text
    )


@pytest.mark.parametrize(
    "source",
    [
        "def f(format):\n    return format(1, 2)",
        "def f():\n    format = g\n    return format(1, 2)",
        "def f(format):\n    def h():\n        return format(1, 2)\n    return h",
        "r = (lambda ascii: ascii(1))(repr)",
    ],
)
def test_format_is_left_alone_inside_the_function_that_binds_it(source: str) -> None:
    assert _rewritten(source + "\n") == source + "\n"


def test_a_parameter_shadows_the_builtin_only_in_its_function() -> None:
    text = _rewritten("def f(format):\n    return format(1, 2)\nr = format(3, '05d')\n")
    assert "return format(1, 2)" in text
    assert "r = (3).__format__('05d')" in text


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    global format\n    format = g\ndef h():\n    return format(1, 2)",
        "def f(xs):\n    return [format(1, 2) for format in xs]",
        "class A:\n    r = [format(1, 2) for format in FS]",
        "def f(xs):\n    [(format := x) for x in xs]\n    return format(1, 2)",
        "def f(x):\n    match x:\n        case [format, *_]:\n            pass\n"
        "    return format(1, 2)",
        "def f():\n    try:\n        pass\n    except E as format:\n        pass\n"
        "    return format(1, 2)",
        "def f(x):\n    if x:\n        format = g\n    return format(1, 2)",
    ],
)
def test_format_follows_python_scoping_where_it_is_bound(source: str) -> None:
    assert _rewritten(source + "\n") == source + "\n"


@pytest.mark.parametrize(
    "source",
    [
        "class A:\n    format = 1\n    def m(self):\n        return format(1, 2)",
        "def f(xs):\n    a = [format for format in xs]\n    return format(1, 2)",
    ],
)
def test_format_is_rewritten_where_python_resolves_it_to_the_builtin(
    source: str,
) -> None:
    assert "format(1, 2)" not in _rewritten(source + "\n")


def test_a_global_declaration_of_a_model_enum_s_name_shadows_it() -> None:
    out = _transform(
        "def f():\n    global TurnDirection\n    TurnDirection = [1]\n"
        "def g():\n    return list(TurnDirection)\n"
    )
    assert "list(TurnDirection)\n" in out


def test_format_nested_in_startswith_is_rewritten_too() -> None:
    text = _rewritten('r = f(1).startswith((format(x, "d"), ascii(y)))\n')
    assert "format(" not in text and "ascii(" not in text
    assert 'x.__format__("d")' in text and "repr(y)" in text


def test_format_keeps_its_lines() -> None:
    source = "r = format(\n    x,\n    spec,\n)\nz = 1\n"
    text = _rewritten(source)
    assert text.count("\n") == source.count("\n")
    assert text.startswith("r = x.__format__(spec)")


def test_startswith_with_a_tuple_matches_cpython() -> None:
    # Differential: the rewritten text and the original give the same value and
    # evaluate the same parts in the same order (all of them, as CPython does,
    # even once an earlier prefix matched).
    import itertools

    templates = [
        "{R}.{M}(({A}, {B}))",
        "{R}.{M}(({A}, {B}), {S})",
        "{R}.{M}(({A}, {B}), {S}, {E})",
        "{R}.{M}(({A},), {S}, {E})",
    ]
    for template, r, a, b, s, e, m in itertools.product(
        templates,
        ["s", 'r("hello world")', '"hello world"'],
        ['"he"', 'a("wo")', "pre"],
        ['"ld"', 'b("ld")', '""'],
        ["1", 'sn("0")', "-3"],
        ["5", 'en("20")'],
        ["startswith", "endswith"],
    ):
        source = "v = " + template.format(R=r, A=a, B=b, S=s, E=e, M=m) + "\n"
        results = []
        for code in (source, _rewritten(source)):
            log: list[str] = []

            def note(name: str, value: Any, log: list[str] = log) -> Any:
                log.append(name)
                return value

            scope: dict[str, Any] = {
                "s": "hello world",
                "pre": "hello",
                "r": lambda v: note("r", v),
                "a": lambda v: note("a", v),
                "b": lambda v: note("b", v),
                "sn": lambda v: note("s", int(v)),
                "en": lambda v: note("e", int(v)),
            }
            exec(code, scope)
            results.append((scope["v"], log))
        assert results[0] == results[1], source


def test_except_star_is_refused_at_its_line() -> None:
    if sys.version_info < (3, 11):
        pytest.skip("except* needs Python 3.11")
    source = "try:\n    f()\nexcept* ValueError as eg:\n    g()\nz = 1\n"
    refused = python_rewrite_refusals(source)
    assert [line for line, _ in refused] == [3]
    assert "exception groups" in refused[0][1] and "use `except`" in refused[0][1]
    two = source.replace("z = 1", "").replace(
        "    g()\n", "    g()\nexcept* KeyError:\n    h()\n"
    )
    assert [line for line, _ in python_rewrite_refusals(two)] == [3, 5]
    assert python_rewrite_refusals("try:\n    f()\nexcept ValueError:\n    g()\n") == []


_LT_ONLY = """
class V:
    def __lt__(self, other: "V") -> bool:
        return True
"""


def test_a_class_with_only_lt_gets_gt_and_ge() -> None:
    text, blocks = python_rewrites(_LT_ONLY)
    assert text == _LT_ONLY
    [(name, block)] = blocks
    assert name == "V"
    assert "@extend\nclass V:" in block
    assert (
        "def __gt__(self, other: V) -> bool:\n        return other.__lt__(self)"
        in block
    )
    assert "return not self.__lt__(other)" in block
    assert "__le__" not in block


def test_a_class_with_every_comparison_gets_none() -> None:
    source = (
        _LT_ONLY
        + """
    def __gt__(self, other: "V") -> bool:
        return True

    def __ge__(self, other: "V") -> bool:
        return True
"""
    )
    assert python_rewrites(source)[1] == []
    ordered = "from dataclasses import dataclass\n@dataclass(order=True)\nclass D:\n    a: int\n    def __lt__(self, o): ...\n"
    assert python_rewrites(ordered)[1] == []
    assert python_rewrites("class N:\n    pass\n")[1] == []


def test_a_subclass_does_not_repeat_what_its_base_defines() -> None:
    source = (
        _LT_ONLY
        + "\nclass W(V):\n    def __lt__(self, other: V) -> bool:\n        return False\n"
    )
    blocks = dict(python_rewrites(source)[1])
    # W's own __lt__ is the one its comparisons must call.
    assert set(blocks) == {"V", "W"}


@pytest.mark.parametrize(
    ("root", "added", "absent"),
    [
        ("__lt__", ["__gt__", "__le__", "__ge__"], "__lt__"),
        ("__le__", ["__lt__", "__gt__", "__ge__"], "__le__"),
        ("__gt__", ["__lt__", "__le__", "__ge__"], "__gt__"),
        ("__ge__", ["__lt__", "__le__", "__gt__"], "__ge__"),
    ],
)
def test_total_ordering_adds_the_other_three_from_its_root(
    root: str, added: list[str], absent: str
) -> None:
    source = (
        "import functools\n"
        "@functools.total_ordering\n"
        f"class V:\n    def {root}(self, o): ...\n    def __eq__(self, o): ...\n"
    )
    text, [(name, block)] = python_rewrites(source)
    assert "total_ordering" not in text.replace("import functools", "")
    assert text.count("\n") == source.count("\n")
    assert name == "V"
    for dunder in added:
        assert f"def {dunder}(self, other: V) -> bool:" in block
    assert f"def {absent}(" not in block
    assert "self == other" in block


def test_total_ordering_without_eq_compares_identity() -> None:
    source = "from functools import total_ordering\n@total_ordering\nclass V:\n    def __lt__(self, o): ...\n"
    text, [(_, block)] = python_rewrites(source)
    assert "self is other" in block
    assert text.splitlines()[0] == "pass"
    assert "total_ordering" not in text


def test_total_ordering_takes_its_root_from_a_base() -> None:
    source = (
        "from functools import total_ordering\n"
        "class B:\n    def __lt__(self, o): ...\n"
        "@total_ordering\nclass V(B): ...\n"
    )
    _, blocks = python_rewrites(source)
    block = dict(blocks)["V"]
    assert "def __le__(self, other: V)" in block
    assert "def __ge__(self, other: V)" in block
    assert "def __lt__(" not in block


def test_the_total_ordering_import_loses_only_its_name() -> None:
    text = _rewritten(
        "from functools import partial, total_ordering as to\n@to\nclass V: ...\n"
    )
    assert text.splitlines()[0] == "from functools import partial"
    assert "@" not in text


def test_total_ordering_that_cannot_be_expressed_is_refused() -> None:
    nested = (
        "from functools import total_ordering\n"
        "def make():\n    @total_ordering\n    class V:\n"
        "        def __lt__(self, o): ...\n    return V\n"
    )
    [(line, message)] = python_rewrite_refusals(nested)
    assert line == 3 and "not at module level" in message
    empty = "from functools import total_ordering\n@total_ordering\nclass V: ...\n"
    [(line, message)] = python_rewrite_refusals(empty)
    assert line == 3 and "one of __lt__" in message


def test_a_lambda_default_that_is_not_a_literal_is_bound_by_a_call() -> None:
    assert (
        _rewritten("fs = [lambda x, k=k: x + k for k in range(3)]\n")
        == "fs = [(lambda k: lambda x: x + k)(k) for k in range(3)]\n"
    )
    assert (
        _rewritten("g = lambda a, b=1, *r, c=k, **kw: a\n")
        == "g = (lambda c: lambda a, b=1, *r, **kw: a)(k)\n"
    )
    # A default after a bound one is bound too, so a positional call keeps
    # meaning what it did.
    assert (
        _rewritten("g = lambda a, b=f(), c=2: a\n")
        == "g = (lambda b, c: lambda a: a)(f(), 2)\n"
    )
    assert _rewritten("lambda: 0\n") == "lambda: 0\n"
    assert _rewritten("g = lambda a, b=2, c=-1, d=None: a\n") == (
        "g = lambda a, b=2, c=-1, d=None: a\n"
    )


def test_the_rewrites_leave_a_module_with_none_of_them_unchanged() -> None:
    source = "class A:\n    x: int = 1\n\nd = {'a': 1}\ns = f'{d}'\nz = zip(d, d)\n"
    assert python_rewrites(source) == (source, [])
    assert python_rewrite_refusals(source) == []


def test_transform_source_lines_applies_the_rewrites_and_appends_the_ordering() -> None:
    source = _LT_ONLY + "x = {**a, **b}\nclass E(Exception): ...\n"
    text, origin = transform_source_lines(source)
    assert "x = (a | b)" in text
    assert "class E(Static[Exception]): pass" in text
    assert "@extend\nclass V:" in text
    assert len(origin) == text.count("\n") - PRELUDE.count("\n")
    assert origin[-1] == 2  # the appended methods map to their class


# ---------------------------------------------------------------------------
# __exit__ parameters (#435) and tables of functions (#436)
# ---------------------------------------------------------------------------


def _line_stable(source: str, text: str) -> None:
    assert len(text.splitlines()) == len(source.splitlines())
    ast.parse(text)


def test_exit_parameters_get_a_none_default_and_lose_their_annotation() -> None:
    source = (
        "class C:\n"
        "    def __exit__(\n"
        "        self,\n"
        "        exc_type: type[BaseException] | None,\n"
        "        exc: BaseException | None,\n"
        "        tb: TracebackType | None,\n"
        "    ) -> bool | None:\n"
        "        if exc_type is not None:\n"
        "            return False\n"
    )
    text, _ = python_rewrites(source)
    _line_stable(source, text)
    # type[...] and TracebackType are NoneType in Codon: dropped.  An exception
    # annotation stays, so `if exc is not None:` is still compiled.
    assert "exc_type=None," in text and "tb=None," in text
    assert "exc: BaseException | None = None," in text
    assert "type[BaseException]" not in text and "TracebackType" not in text
    assert "if exc_type is not None:" in text


def test_exit_exception_annotations_are_kept_as_optional() -> None:
    source = (
        "class C:\n"
        "    def __exit__(self, t, e: Optional[Exception], b: 'BaseException', "
        "m: ValueError | None = None, n: Exception | None = None) -> None:\n"
        "        pass\n"
    )
    text, _ = python_rewrites(source)
    assert "e: Exception | None = None," in text
    assert "b: BaseException | None = None," in text
    assert "m = None" in text  # any other annotation goes
    assert "n: Exception | None = None)" in text
    again, _ = python_rewrites(text)
    assert again == text


def test_exit_parameters_with_a_default_are_kept() -> None:
    source = (
        "class C:\n"
        "    def __exit__(self, et, ev: object | None = None, flag: bool = True) -> None:\n"
        "        pass\n"
        "    def __exit__2(self, *args: object) -> None:\n"
        "        pass\n"
    )
    text, _ = python_rewrites(source)
    assert "(self, et=None, ev = None, flag: bool = True)" in text
    assert "*args: object" in text


def test_exit_outside_a_class_is_left_alone() -> None:
    source = "def __exit__(self, a, b):\n    pass\n"
    assert python_rewrites(source) == (source, [])


def test_a_table_of_functions_is_wrapped_in_the_annotated_callable() -> None:
    source = (
        "import helpers\n"
        "def f(k):\n"
        "    fs: list[Callable[[int], int]] = [inc, helpers.dbl, lambda x: x + 1]\n"
        '    d: Dict[str, Callable[[int], int]] = {"a": inc, "b": lambda x: x}\n'
        "    return fs, d\n"
    )
    text, _ = python_rewrites(source)
    _line_stable(source, text)
    wrap = "Callable[Tuple[int], int]"
    assert f"[{wrap}(inc), {wrap}(helpers.dbl), {wrap}(lambda x: x + 1)]" in text
    assert f'{{"a": {wrap}(inc), "b": {wrap}(lambda x: x)}}' in text


def test_a_returned_table_and_a_nested_one_are_wrapped() -> None:
    source = (
        "def f() -> dict[str, list[Callable[[float, str], None]]]:\n"
        '    return {"a": [g, h], "b": [g]}\n'
    )
    text, _ = python_rewrites(source)
    wrap = "Callable[Tuple[float, str], None]"
    assert f'{{"a": [{wrap}(g), {wrap}(h)], "b": [{wrap}(g)]}}' in text


def test_a_rewritten_table_is_idempotent_and_keeps_inner_rewrites() -> None:
    source = (
        "def f():\n"
        "    fs: list[Callable[[int], int]] = [lambda x, k=len('a'): x + k, g]\n"
    )
    text, _ = python_rewrites(source)
    again, _ = python_rewrites(text)
    assert again == text
    assert "(lambda k: lambda x: x + k)(len('a'))" in text
    assert "Callable[Tuple[int], int](" in text


def test_what_a_table_does_not_cover_is_left_alone() -> None:
    cases = [
        # not annotated, a class attribute, a bound method, a capture
        "fs = [inc, dbl]\n",
        "class C:\n    fs: list[Callable[[int], int]] = [inc, dbl]\n",
        "def f(o):\n    fs: list[Callable[[int], int]] = [o.inc, o.dbl]\n",
        "def f(k):\n    fs: list[Callable[[int], int]] = [lambda x: x + k]\n",
        # a set (a Callable has no __hash__ in Codon)
        "fs: set[Callable[[int], int]] = {inc}\n",
        # a signature Codon cannot express
        "fs: list[Callable[..., int]] = [inc, dbl]\n",
        "fs: list[Callable[[Any], int]] = [inc, dbl]\n",
    ]
    for source in cases:
        assert python_rewrites(source) == (source, []), source


# -- a class constant read in the class scope (#433) --------------------------


def test_a_class_constant_in_another_class_attribute_is_qualified() -> None:
    assert _rewritten(
        """
        class H:
            LIMIT = 5
            DOUBLE = LIMIT * 2
            TABLE: list[int] = [LIMIT, LIMIT + 1]
            STEP: Final = 3
            TRIPLE: int = STEP * 3
        """
    ) == (
        "\nclass H:\n    LIMIT = 5\n    DOUBLE = H.LIMIT * 2\n"
        "    TABLE: list[int] = [H.LIMIT, H.LIMIT + 1]\n"
        "    STEP: Final = 3\n    TRIPLE: int = H.STEP * 3\n"
    )


def test_a_class_constant_in_a_default_or_decorator_is_qualified() -> None:
    text = _rewritten(
        """
        class H:
            LIMIT = 5
            @register(LIMIT)
            def f(self, n: int = LIMIT, *, m: int = LIMIT) -> int:
                return n
            @staticmethod
            def g(LIMIT=LIMIT):
                return LIMIT
        """
    )
    assert "@register(H.LIMIT)" in text
    assert "def f(self, n: int = H.LIMIT, *, m: int = H.LIMIT)" in text
    assert "def g(LIMIT=H.LIMIT):\n        return LIMIT" in text  # body untouched


def test_a_dataclass_field_default_is_qualified() -> None:
    text = _rewritten(
        """
        @dataclass
        class D:
            LIM = 5
            x: int = LIM
            y: list[int] = field(default_factory=lambda: [LIM])
        """
    )
    assert "x: int = D.LIM" in text
    assert "lambda: [LIM]" in text  # a lambda body does not see the class


def test_a_method_body_reads_the_module_global() -> None:
    source = (
        "LIMIT = 9\nclass H:\n    LIMIT = 5\n    def f(self):\n"
        "        return LIMIT + [LIMIT for _ in range(LIMIT)][0]\n"
    )
    assert _rewritten(source) == source


def test_a_comprehension_sees_the_class_scope_only_in_its_first_iterable() -> None:
    assert _rewritten(
        """
        class H:
            N = 3
            XS = [N for _ in range(N)]
            YS = {k: N for k in range(N) if k < N}
        """
    ) == (
        "\nclass H:\n    N = 3\n    XS = [N for _ in range(H.N)]\n"
        "    YS = {k: N for k in range(H.N) if k < N}\n"
    )


def test_a_lambda_default_is_in_the_class_scope_its_body_is_not() -> None:
    text = _rewritten("class H:\n    K = 2\n    F = lambda x, k=K: x + K\n")
    assert text == ("class H:\n    K = 2\n    F = (lambda k: lambda x: x + K)(H.K)\n")


def test_only_a_name_bound_earlier_in_the_body_is_qualified() -> None:
    source = "LIMIT = 9\nclass H:\n    A = LIMIT\n    LIMIT = 5\n    B = LIMIT\n"
    # A reads the global (LIMIT is not bound yet), B the class attribute; LIMIT
    # is assigned after A reads it, but A does not read it.
    assert _rewritten(source) == (
        "LIMIT = 9\nclass H:\n    A = LIMIT\n    LIMIT = 5\n    B = H.LIMIT\n"
    )


def test_a_name_that_is_not_an_assigned_constant_is_left_alone() -> None:
    source = (
        "class H:\n"
        "    @property\n    def x(self): ...\n"
        "    @x.setter\n    def x(self, v): ...\n"
        "    class In: ...\n    Y = In\n"
        "    if True:\n        C = 1\n    D = C\n"
        "    for i in range(2):\n        pass\n    E = i\n"
    )
    assert "H." not in _rewritten(source)


def test_a_nested_class_sees_only_its_own_body() -> None:
    text = _rewritten(
        """
        class O:
            A = 1
            class In:
                B = A
                C = 2
                D = C
                def f(self, x=C, y=A): ...
            E = A
        """
    )
    assert "B = A\n" in text  # the outer A is no name of the inner body
    assert "D = O.In.C" in text
    assert "def f(self, x=O.In.C, y=A)" in text
    assert "E = O.A" in text


def test_a_shadowing_parameter_or_comprehension_target_is_not_qualified() -> None:
    text = _rewritten(
        """
        class H:
            LIMIT = 5
            F = lambda LIMIT=LIMIT: LIMIT
            XS = [LIMIT for LIMIT in range(3)]
            def f(self, LIMIT=LIMIT):
                return LIMIT
        """
    )
    assert "(lambda LIMIT: lambda: LIMIT)(H.LIMIT)" in text
    assert "XS = [LIMIT for LIMIT in range(3)]" in text
    assert "def f(self, LIMIT=H.LIMIT):\n        return LIMIT" in text


def test_class_constants_keep_every_line_in_place() -> None:
    source = (
        "class H:\n    A = 1\n    B = (\n        A\n        + A\n    )\n"
        "    def f(\n        self,\n        n=A,\n    ): ...\n"
    )
    text = _rewritten(source)
    assert text.count("\n") == source.count("\n")
    assert "        H.A\n        + H.A\n" in text
    assert "        n=H.A,\n" in text


def test_an_enum_member_read_in_its_body_is_its_value() -> None:
    text = _rewritten(
        """
        class E(enum.IntEnum):
            A = 1
            B = A
            _K = 3
            C = _K
        """
    )
    assert "B = 1" in text  # an alias of A, as in Python
    assert "C = E._K" in text  # not a member: an attribute


def test_a_class_constant_that_cannot_be_qualified_is_refused() -> None:
    nested = "def make():\n    class L:\n        Q = 1\n        R = Q\n"
    [(line, message)] = python_rewrite_refusals(nested)
    assert line == 4 and "defined in a function" in message
    again = "class H:\n    X = 1\n    Y = X\n    X = 2\n"
    [(line, message)] = python_rewrite_refusals(again)
    assert line == 3 and "assigned again later" in message
    auto = "class E(enum.Enum):\n    A = enum.auto()\n    B = A\n"
    [(line, message)] = python_rewrite_refusals(auto)
    assert line == 3 and "write the value" in message


def test_an_annotated_literal_read_in_the_class_scope_is_written_as_its_value() -> None:
    # Codon makes `LIMIT: int = 5` a field of the instance, not an attribute
    # of the class, so there is nothing to qualify the read with: Python reads
    # the literal the body bound, and that is written in its place.
    source = (
        "class H:\n    LIMIT: int = 5\n    NEG: Final[int] = -2\n"
        "    def f(self, n: int = LIMIT, m: int = NEG ** 2): ...\n"
        "    X: int = NEG\n"
    )
    text = python_rewrites(source)[0]
    assert "n: int = 5, m: int = (-2) ** 2" in text and "X: int = -2" in text
    assert python_rewrite_refusals(source) == []
    # A value the check cannot write, or a second binding, is refused.
    computed = (
        "class H:\n    LIMIT: int = 2 * 3\n    def f(self, n: int = LIMIT): ...\n"
    )
    [(line, message)] = python_rewrite_refusals(computed)
    assert line == 3 and "Final" in message
    again = "class H:\n    X: int = 1\n    Y = X\n    X: int = 2\n"
    [(line, message)] = python_rewrite_refusals(again)
    assert line == 3 and "assigned again later" in message
    # A later plain assignment replaces the annotated value for the next read.
    plain = "class H:\n    X: int = 1\n    X = 2\n    Y = X\n"
    assert "Y = H.X" in python_rewrites(plain)[0]
    final = "from typing import Final\nclass H:\n    LIMIT: Final = 5\n    X = LIMIT\n"
    assert python_rewrite_refusals(final) == []


def test_a_negative_enum_alias_keeps_the_literal_of_its_member() -> None:
    source = "class E(enum.IntEnum):\n    A = -1\n    B = A\n"
    assert "B = -1\n" in python_rewrites(source)[0]


def test_a_namedtuple_field_default_reading_a_class_constant_is_refused() -> None:
    source = (
        "class P(NamedTuple):\n    LIM = 5\n    x: int = LIM\n"
        "    DOUBLE = LIM * 2\n    def f(self, n: int = LIM): ...\n"
    )
    [(line, message)] = python_rewrite_refusals(source)
    assert line == 3 and "NamedTuple" in message
    text = python_rewrites(source)[0]
    assert "DOUBLE = P.LIM * 2" in text and "n: int = P.LIM" in text


def test_a_class_without_a_constant_read_is_unchanged() -> None:
    source = (
        "class H:\n    A = 1\n    B = 2\n    def f(self, n=1): return self.A\n"
        "    C = [i for i in range(3)]\n"
    )
    assert python_rewrites(source) == (source, [])
    assert python_rewrite_refusals(source) == []


def test_class_constants_compose_with_the_other_rewrites() -> None:
    source = (
        "from dataclasses import dataclass\n"
        "from functools import total_ordering\n"
        "@total_ordering\n@dataclass\nclass V:\n    N = 2\n    x: int = N\n"
        "    def __lt__(self, o): ...\n"
        "    def f(self, a=N): return '{}'.format(a)\n"
    )
    text, [(name, block)] = python_rewrites(source)
    assert "x: int = V.N" in text and "a=V.N" in text and name == "V"
    assert "total_ordering" not in text and 'f"{a}"' in text


# -- a comprehension in a class body or a default_factory (#469) --------------


def test_a_class_body_comprehension_is_hoisted_above_its_class() -> None:
    out = transform_source(
        textwrap.dedent(
            """
            class Q:
                SQ = [i * i for i in range(3)]
            """
        )
    )
    assert (
        "def _acs_hoist_Q_3():\n    return ([i * i for i in range(3)])\n"
        "class Q:\n    SQ = _acs_hoist_Q_3()\n"
    ) in out


def test_a_default_factory_with_a_comprehension_is_hoisted() -> None:
    out = transform_source(
        textwrap.dedent(
            """
            from dataclasses import dataclass, field
            @dataclass
            class D:
                a: list[str] = field(default_factory=lambda: [str(i) for i in range(3)])
                b: tuple = field(default_factory=lambda: tuple(i for i in range(3)))
                c: dict = field(default_factory=lambda: dict(a=1))
                e: list = field(default_factory=lambda: [1, 2])
            """
        )
    )
    assert "a: list[str] = _acs_hoist_D_5()" in out
    assert "b: tuple = _acs_hoist_D_6()" in out
    assert "c: dict = _acs_hoist_D_7()" in out
    assert "e: list = ([1, 2])" in out  # nothing to hoist


def test_hoisting_keeps_the_lines_of_the_class_and_maps_the_new_ones() -> None:
    from autoware_carla_scenario.typecheck.transform import (
        PRELUDE,
        transform_source_lines,
    )

    source = "x = 1\nclass Q:\n    SQ = [i for i in range(3)]\n    Y = 2\n"
    text, origin = transform_source_lines(source)
    lines = text[len(PRELUDE) :].splitlines()
    assert len(lines) == len(origin)
    assert origin[lines.index("    SQ = _acs_hoist_Q_3()")] == 3
    assert origin[lines.index("    Y = 2")] == 4
    assert origin[lines.index("def _acs_hoist_Q_3():")] == 3


def test_an_enum_extension_follows_its_class() -> None:
    out = transform_source(
        "import enum\nclass C(enum.Enum):\n    A = 1\nX = [c for c in C]\n"
    )
    assert out.index("def _acs_members") < out.index("X = [c for c in C._acs")


def test_a_comprehension_reading_its_class_is_refused() -> None:
    refused = python_rewrite_refusals(
        "class Q:\n    B = 3\n    SQ = [i * 2 for i in range(B)]\n"
    )
    assert refused and "module-level function" in refused[0][1]
    assert refused[0][0] == 3


def test_dict_with_keywords_becomes_a_display() -> None:
    out = transform_source("d = dict(a=1, b = 2,)\ne = dict(**x)\nf = dict()\n")
    assert 'd = {"a": 1, "b": 2}' in out
    assert "e = dict(**x)" in out and "f = dict()" in out


@pytest.mark.parametrize(
    "source",
    [
        "def f(dict):\n    return dict(a=1)",
        "def f():\n    dict = g\n    return dict(a=1)",
        "def f():\n    global dict\n    dict = g\ndef h():\n    return dict(a=1)",
        "dict = g\nx = dict(a=1)",
        "class A:\n    dict = g\n    x = dict(a=1)",
    ],
)
def test_dict_with_keywords_is_left_alone_where_dict_is_bound(source: str) -> None:
    out = transform_source(source + "\n")
    assert "dict(a=1)" in out


@pytest.mark.parametrize(
    "source",
    [
        "def f(dict):\n    return dict\nx = dict(a=1)",
        "class A:\n    dict = g\n    def m(self):\n        return dict(a=1)",
        "class A:\n    def dict(self):\n        return 1\nx = dict(a=1)",
    ],
)
def test_dict_with_keywords_is_rewritten_where_dict_is_the_builtin(
    source: str,
) -> None:
    assert "dict(a=1)" not in transform_source(source + "\n")


def test_dict_with_keywords_is_left_alone_where_the_module_binds_dict() -> None:
    out = transform_source("def f(dict):\n    return dict(a=1)\nx = dict(b=2, **y)\n")
    assert "return dict(a=1)" in out and "x = dict(b=2, **y)" in out


# ---------------------------------------------------------------------------
# An f-string in a lambda that reads a captured name (#470)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('f = lambda v: f"{v}{p}"', "f = lambda v: (str(v) + str(p))"),
        (
            'f = lambda v: f"a{v}-{p}!"',
            'f = lambda v: ("a" + str(v) + "-" + str(p) + "!")',
        ),
        ('f = lambda v: f"{p!r}"', "f = lambda v: (repr(p))"),
        ('f = lambda v: f"{p:.2f}"', 'f = lambda v: ((p).__format__(".2f"))'),
        (
            'f = lambda v: f"{p!r:>{v}}"',
            'f = lambda v: (repr(p).__format__(">" + str(v)))',
        ),
        ('f = lambda: f"{p=:.2f}"', 'f = lambda: ("p=" + (p).__format__(".2f"))'),
        ('f = lambda: f"{p=}"', 'f = lambda: ("p=" + repr(p))'),
        ('f = lambda e: "{}{}".format(e, p)', "f = lambda e: (str(e) + str(p))"),
        (
            'f = lambda v, lab=cfg.name: f"{v}{lab}"',
            "f = (lambda lab: lambda v: (str(v) + str(lab)))(cfg.name)",
        ),
        (
            'g = (lambda lab: lambda v: f"{lab}")("a")',
            'g = (lambda lab: lambda v: (str(lab)))("a")',
        ),
        (
            's = sorted(xs, key=lambda e: f"{e}{p}")',
            "s = sorted(xs, key=lambda e: (str(e) + str(p)))",
        ),
    ],
)
def test_a_lambda_fstring_reading_a_captured_name_is_lowered(
    source: str, expected: str
) -> None:
    assert _rewritten(source + "\n") == expected + "\n"
    assert python_rewrite_refusals(source + "\n") == []


@pytest.mark.parametrize(
    "source",
    [
        'f = lambda v: f"{v}"',
        'f = lambda v, w: f"{v}{w:>5}"',
        'f = lambda: f"{3}"',
        'p = 1\nf = f"{p}"',
        'def g(p):\n    return f"{p}"',
        'def g(p):\n    return (lambda v: p)(1), f"{p}"',
    ],
)
def test_a_lambda_fstring_of_its_own_names_is_left_alone(source: str) -> None:
    assert _rewritten(source + "\n") == source + "\n"


def test_a_lambda_fstring_lowering_keeps_the_lines() -> None:
    source = 'f = lambda v: f"""a\n{v}\n{p}"""\nz = 1\n'
    text = _rewritten(source)
    assert text.count("\n") == source.count("\n")
    assert text.splitlines()[-1] == "z = 1"
    ast.parse(text)


def test_a_lambda_fstring_with_ascii_conversion_is_lowered_to_repr() -> None:
    assert _rewritten('f = lambda v: f"{p!a}"\n') == "f = lambda v: (repr(p))\n"
    assert (
        _rewritten('f = lambda v: f"{p!a:>5}"\n')
        == 'f = lambda v: (repr(p).__format__(">5"))\n'
    )


@pytest.mark.parametrize(
    "template",
    [
        'lambda v: f"{v}{p}"',
        'lambda v: f"<{v!r}|{p!r}|{p!s}|{p!s:>8}|{p!a}|{v!a:>9}>"',
        'lambda v: f"{p:{w}.{w}}|{p:>{w}}|{v:{w}}"',
        'lambda v: f"""a\\n{v}\\n{p}"""',
        'lambda v: (lambda u: f"{u}{v}{p}")(v)',
        'lambda v: [f"{u}{v}" for u in range(2)]',
        'lambda v: (y := v, f"{y}{p}")',
        'lambda v: f"{v:>6}|{p:.2f}|{p:+.1f}|{v:{w}}"',
        'lambda v: f"{p=}|{p=:.1f}|{v=}"',
        'lambda v: "{}-{:>4}-{!r}".format(v, p, p)',
        'lambda v: f"{{lit}} {v} {p}"',
        'lambda v, lab="q": f"{v}{lab}{p}"',
        'lambda v, lab=side(): f"{v}{lab}{p}"',
        '(lambda lab: lambda v: f"{v}{lab}{p}")(side())',
        'lambda v: f"{note(v)}{note(p)}{note(w)}"',
        'lambda v: [f"{i}{p}" for i in range(v)]',
    ],
)
def test_a_lowered_lambda_fstring_equals_the_original(template: str) -> None:
    results = []
    for code in (
        f"f = {template}\n",
        _rewritten(f"f = {template}\n"),
    ):
        log: list[str] = []

        def note(value: Any, log: list[str] = log) -> Any:
            log.append(repr(value))
            return value

        scope: dict[str, Any] = {"p": 2.5, "w": 7, "side": lambda: "S", "note": note}
        exec(code, scope)  # noqa: S102
        samples = (3,) if "range" in code else (1, 3.25)
        results.append(([scope["f"](v) for v in samples], log))
    assert results[0] == results[1], template


# -- a class-body lambda, a method's lambda default, nested classes (#482) -----


def _lines(source: str) -> tuple[list[str], list[int]]:
    from autoware_carla_scenario.typecheck.transform import (
        PRELUDE,
        transform_source_lines,
    )

    text, origin = transform_source_lines(source)
    lines = text[len(PRELUDE) :].splitlines()
    assert len(lines) == len(origin)
    return lines, origin


def test_a_class_body_lambda_with_self_becomes_a_method() -> None:
    out = transform_source("class T:\n    f = lambda self, v: v * 2\n")
    assert "    def f(self, v): return (v * 2)\n" in out
    assert "staticmethod" not in out


def test_a_class_body_lambda_without_self_becomes_a_static_method() -> None:
    lines, origin = _lines("class T:\n    key = lambda v: str(v) + 'x'\n    Y = 1\n")
    at = lines.index("    def key(v): return (str(v) + 'x')")
    assert lines[at - 1] == "    @staticmethod"
    # The decorator maps to the lambda's line, and the rest keeps its line.
    assert origin[at - 1] == origin[at] == 2
    assert origin[lines.index("    Y = 1")] == 3


def test_staticmethod_of_a_lambda_becomes_a_decorated_def() -> None:
    out = transform_source(
        "class T:\n    f = staticmethod(lambda a, b=2: a + b)\n    g = lambda: 1\n"
    )
    assert "    @staticmethod\n    def f(a, b=2): return (a + b)\n" in out
    assert "    @staticmethod\n    def g(): return (1)\n" in out


def test_a_multiline_class_body_lambda_keeps_its_lines() -> None:
    source = "class T:\n    f = lambda self, v: (\n        v + 1\n    )\n    Y = 1\n"
    lines, origin = _lines(source)
    assert origin[lines.index("    Y = 1")] == 5
    assert lines.index("    Y = 1") == lines.index("class T:") + 4


def test_a_lambda_in_a_nested_class_is_rewritten() -> None:
    out = transform_source(
        "class Outer:\n    class Inner:\n        f = lambda self, v: v * 2\n"
    )
    assert "        def f(self, v): return (v * 2)\n" in out


def test_a_lambda_on_the_line_of_its_class_is_left_alone() -> None:
    out = transform_source("class T: f = lambda self: 1\n")
    assert "class T: f = lambda self: 1\n" in out


def test_a_lambda_default_of_a_method_is_hoisted_above_the_class() -> None:
    lines, origin = _lines(
        "@dec\nclass P:\n    def sort(self, xs, key=lambda v: -v, *, k=lambda: 1): ...\n"
    )
    first = next(
        i for i, line in enumerate(lines) if line.startswith("def _acs_lambda")
    )
    assert lines[first] == "def _acs_lambda_P_3_27(v): return (-v)"
    assert lines.index("@dec") > first
    assert origin[first] == 3
    assert "key=_acs_lambda_P_3_27" in lines[lines.index("class P:") + 1]


def test_a_lambda_default_of_a_method_in_a_function_is_refused() -> None:
    from autoware_carla_scenario.typecheck.transform import class_body_refusals

    source = "def build():\n    class P:\n        def m(self, key=lambda v: v): ...\n"
    [(line, message)] = class_body_refusals(source)
    assert line == 3 and "module-level `def`" in message


def test_a_comprehension_in_a_nested_class_is_hoisted() -> None:
    lines, origin = _lines(
        "class Outer:\n    class Inner:\n        XS = [i for i in range(3)]\n"
    )
    first = lines.index("def _acs_hoist_Outer_Inner_3():")
    assert first < lines.index("class Outer:")
    assert origin[first] == 3
    assert "        XS = _acs_hoist_Outer_Inner_3()" in lines


def test_a_nested_class_comprehension_reading_an_outer_class_is_refused() -> None:
    from autoware_carla_scenario.typecheck.transform import class_body_refusals

    source = "class Outer:\n    N = 3\n    class Inner:\n        XS = [i for i in range(N)]\n"
    assert class_body_refusals(source) == []  # N is read, not a class name
    source = "class Outer:\n    class Inner:\n        XS = [Outer for i in range(3)]\n"
    [(line, message)] = class_body_refusals(source)
    assert line == 3 and "reads Outer itself" in message


@pytest.mark.parametrize(
    ("annotation", "advice"),
    [
        ("list", "list[X]"),
        ("dict", "dict[K, V]"),
        ("set", "set[X]"),
        ("tuple", "tuple[X, ...]"),
        ("typing.List", "List[X]"),
        ("'list'", "list[X]"),
    ],
)
def test_a_bare_collection_field_of_a_dataclass_is_refused(
    annotation: str, advice: str
) -> None:
    from autoware_carla_scenario.typecheck.transform import class_body_refusals

    source = (
        f"from dataclasses import dataclass\n@dataclass\nclass C:\n"
        f"    a: int = 1\n    xs: {annotation}\n"
    )
    [(line, message)] = class_body_refusals(source)
    assert line == 5 and advice in message and "`xs`" in message


def test_a_bare_collection_in_a_plain_class_body_is_refused() -> None:
    from autoware_carla_scenario.typecheck.transform import class_body_refusals

    # Codon cannot infer it in any class (#482), declared or assigned.
    for body in ("xs: list = []", "xs: list", "d: dict = {}"):
        [(line, message)] = class_body_refusals(f"class P:\n    {body}\n")
        assert line == 2 and "bare" in message, body
    source = "class P:\n    def __init__(self) -> None:\n        self.xs: list = []\n"
    assert class_body_refusals(source) == []


def test_a_parametrized_class_field_is_not_a_bare_collection() -> None:
    from autoware_carla_scenario.typecheck.transform import class_body_refusals

    source = (
        "from dataclasses import dataclass\n@dataclass\nclass C:\n"
        "    xs: list[int]\n    t: tuple[float, ...] = ()\n"
        "class P:\n    xs: list[int] = []\n"
    )
    assert class_body_refusals(source) == []


# dataclasses.replace lowered to typed copies (#487)

_REPLACE_DEFS = """
from dataclasses import dataclass, field, replace
import dataclasses

@dataclass
class W:
    lane: int = 0
    best: W | None = None
    cache: int = field(default=0, init=False)

@dataclass(frozen=True)
class V(W):
    speed: float = 1.0
"""


def _replacer_block(source: str, name: str) -> str:
    out = transform_source(textwrap.dedent(source))
    blocks = out.split("\n\n@extend\n")[1:]
    return next(
        (b for b in blocks if b.startswith(f"class {name}:") and "_acs_with__" in b),
        "",
    )


def test_a_dataclass_gets_a_typed_copy_method_per_init_field() -> None:
    w = _replacer_block(_REPLACE_DEFS, "W")
    assert _methods(w) == {"_acs_with__lane", "_acs_with__best"}
    assert "def _acs_with__best(self, best: Optional[W]) -> W:" in w
    assert "return W(lane=self.lane, best=best)" in w
    v = _replacer_block(_REPLACE_DEFS, "V")
    assert _methods(v) == {"_acs_with__lane", "_acs_with__best", "_acs_with__speed"}
    assert "return V(lane=lane, best=self.best, speed=self.speed)" in v


def test_a_class_with_an_init_or_an_unknown_base_gets_no_copy_method() -> None:
    source = """
    from dataclasses import dataclass
    from elsewhere import Base

    @dataclass
    class A(Base):
        x: int = 0

    @dataclass
    class B:
        x: int = 0

        def __init__(self, x: int = 0) -> None:
            self.x = x
    """
    assert _replacer_block(source, "A") == ""
    assert _replacer_block(source, "B") == ""


@pytest.mark.parametrize(
    ("call", "lowered"),
    [
        ("replace(w, lane=1)", "replace(w)._acs_with__lane(lane=1)"),
        (
            "replace(w, lane=1, best=None)",
            "replace(w)._acs_with__lane(lane=1)._acs_with__best(best=None)",
        ),
        ("dataclasses.replace(w, lane=f(1, 2))", None),
        ("replace((w), lane=1,)", "replace((w))._acs_with__lane(lane=1,)"),
        ("replace(w)", "replace(w)"),
    ],
)
def test_replace_becomes_a_chain_of_typed_copies(
    call: str, lowered: str | None
) -> None:
    out = transform_source(_REPLACE_DEFS + f"\nx = {call}\n")
    line = next(ln for ln in out.splitlines() if ln.startswith("x = "))
    if lowered is None:
        assert "._acs_with__lane(lane=f(1, 2))" in line
    else:
        assert line == f"x = {lowered}"


def test_replace_keeps_each_statement_on_its_lines() -> None:
    source = _REPLACE_DEFS + "\nx = replace(\n    w,\n    lane=1,\n    best=None,\n)\n"
    out, origin = transform_source_lines(source)
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("x = "))
    assert "".join(lines[start : start + 5]).replace(" ", "") == (
        "x=replace(w)._acs_with__lane(lane=1)._acs_with__best(best=None,)"
    )
    assert out.count("\n") == len(origin) + PRELUDE.count("\n")


@pytest.mark.parametrize(
    "call",
    [
        "replace(w, **kw)",
        "replace(w, 3)",
        "replace(obj=w, lane=1)",
        "dataclasses.replace(*args, lane=1)",
        "replace()",
    ],
)
def test_a_replace_call_that_cannot_be_lowered_is_refused(call: str) -> None:
    refused = python_rewrite_refusals(_REPLACE_DEFS + f"\nx = {call}\n")
    assert len(refused) == 1 and "dataclasses.replace" in refused[0][1]


def test_replace_that_is_not_dataclasses_replace_is_left_alone() -> None:
    source = "s = 'a'.replace('a', 'b')\n\ndef replace(a, b=1):\n    return a\n\nx = replace(1, b=2)\n"
    assert python_rewrite_refusals(source) == []
    assert "_acs_with__" not in transform_source(source)
