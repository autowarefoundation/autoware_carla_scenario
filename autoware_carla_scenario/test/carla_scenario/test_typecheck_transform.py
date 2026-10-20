"""The source rewrite and the checks the static check makes before Codon runs.

None of these need Codon: they pin down what reaches the compiler.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import enum
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
            """
        )
    )
    for line in (
        "class A(_acs_IntEnum[int]):",
        "class B(_acs_IntEnum[int]):",
        "class C(_acs_IntEnum[int]):",
        "class D(_acs_StrEnum[str]):",
        "class E(_acs_StrEnum[str]):",
        "class F(enum.Enum[int]):",
        "class G(enum.Flag[int]):",
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
    body = out.partition("\n\n@extend\n")[0]  # the constructors by value follow
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
    body, _, lookups = out.partition("\n\n@extend\n")
    assert body.count("\n") == source.count("\n")  # no line of the source moves
    assert "@extend\nclass Turn:\n    def __init__(self, value: str):" in out
    assert "@extend\nclass Level:\n    def __init__(self, value: int):" in out
    assert "class Local" not in lookups  # the module's top-level enums only


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
    body = out.partition("\n\n@extend\n")[0]
    assert body.count("\n") == source.count("\n")  # no line moves
    assert "for m in Mode._acs_members():" in out
    assert "[x for x in Mode._acs_members() if x in Mode._acs_members()]" in out
    assert "list(Mode._acs_members()), len(Mode._acs_members())" in out
    assert "sorted(Mode._acs_members()), (*Mode._acs_members(),)" in out
    assert "c = Mode._acs_by_name(\n" in out  # the line breaks stay
    assert 'd = Mode._acs_by_name("A").value' in out
    assert "e = Other[n]" in out and "g = Mode.A" in out
    assert "h = {m: 1 for m in Mode._acs_members()}" in out


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
    out = _transform(source).partition("\n\n@extend\n")[0]
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
        "from a import X, Z\n"
        "def f():\n"
        "    return X\n"
        "from b import X\n"
        "from a import X\n"
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
        if isinstance(node, ast.FunctionDef)
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
    sub = out.rpartition("class Sub:")[2]
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


def test_zip_strict_is_dropped() -> None:
    assert _rewritten("zip(a, b, strict=True)\n") == "zip(a, b)\n"
    assert _rewritten("zip(strict=False)\n") == "zip()\n"
    assert _rewritten("zip(a, strict=x) + zip(b, strict=1)\n") == "zip(a) + zip(b)\n"
    source = "zip(\n    a,\n    b,\n    strict=True,\n)\n"
    text = _rewritten(source)
    assert text.count("\n") == source.count("\n")
    assert "strict" not in text
    assert _rewritten("other(a, strict=True)\n") == "other(a, strict=True)\n"


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
