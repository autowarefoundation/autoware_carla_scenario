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
from autoware_carla_scenario.typecheck.transform import (
    MISSING_BUILTIN_EXCEPTIONS,
    PRELUDE,
    codon_annotation,
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
    out = _transform(source)
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
    assert "class Turn(Enum[str]):" in out
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


def test_a_str_enum_with_auto_values_takes_the_lowercase_names() -> None:
    out = _transform("class Mode(StrEnum):\n    FAST = auto()\n")
    assert "class Mode(StrEnum[str]):" in out
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
