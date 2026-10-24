"""Rewrite a scenario module into one Codon 0.19 compiles, line for line.

Codon reads Python syntax and checks it with static types, but it lacks some
of what typed Python writes. A scenario's source goes through
:func:`transform_source` before it is compiled, and every rewrite keeps each
statement on its line, so what Codon reports maps straight back onto the file
the author wrote (but the copies of a handler that catches a tuple, below,
which :func:`transform_source_lines` maps back onto the handler):

* ``X | None`` and ``Optional[X]`` / ``Union[X, None]`` become ``Optional[X]``.
  An annotation Codon cannot express (another ``Union``, which crashes Codon
  0.19, ``Any``, ``Callable``, ``type[...]``, ``Sequence[...]``, ...) is
  dropped: the parameter becomes generic, which Codon checks at each call.
  A ``Callable[[A], R]`` parameter the body stores (``self._cb = fn``,
  ``self._cbs.append(fn)``) is kept as ``Callable[Tuple[A], R]``: a generic
  one put in a typed ``Callable`` attribute aborts Codon (#488).
  Class-level declarations keep the abstract collections, which the
  ``typing`` shim maps onto ``List`` / ``Dict``.
* ``@dataclass`` is removed (a Codon class with annotated fields already gets
  the dataclass ``__init__``), and each ``field(default=...)`` /
  ``field(default_factory=...)`` is replaced by its default.  The methods
  ``@dataclass`` generates besides ``__init__`` are appended after the module
  with ``@extend`` (:func:`_dataclass_dunders`): ``__eq__`` / ``__ne__``,
  ``__hash__`` (``frozen`` or ``unsafe_hash``), ordering (``order=True``).
* ``replace(obj, a=x, b=y)`` (``dataclasses.replace``) becomes
  ``replace(obj)._acs_with__a(x)._acs_with__b(y)`` (:func:`lower_replace_calls`):
  each dataclass of the module gets one ``_acs_with__<field>`` method per field
  (:func:`_dataclass_replacers`), a typed copy of the object with that field
  changed, so Codon checks the field name and the value's type (the erased
  ``**changes`` it replaces crashed Codon on a ``None``, #487).
* A ``class Pt(NamedTuple)`` becomes a Codon ``@tuple`` class, the decorator
  on a new line above it (:func:`tuple_named_tuples`).
* ``@abstractmethod`` is removed (Codon 0.19 cannot decorate a method; the
  ``abc`` shim's ``ABC`` is an empty base), and so is ``@runtime_checkable``
  (the ``typing`` shim's ``Protocol`` is an empty base too).
* The names Codon's ``functools`` lacks (:data:`FUNCTOOLS_NAMES`): the
  decorators ``lru_cache`` / ``cache`` are dropped (caching changes no type),
  ``cached_property`` becomes ``property``, ``reduce`` becomes ``_acs_reduce``,
  and the names are left out of the ``from functools import`` line.
* A parameter or return annotation naming a class the module defines further
  down (``AbsoluteVelocity.__add__(self, other: RelativeVelocity)``) is
  dropped: Codon 0.19 cannot name a class in a signature before its
  definition, and two classes that take each other cannot both come first.
  So is one naming a class the module imports further down (the
  ``if TYPE_CHECKING:`` import moved to the end of a module in a cycle).
* An annotation naming one of CARLA's integer enumerations
  (``carla.TrafficLightState``) becomes ``int``: typesafe_carla's Codon
  library has each as a value whose members are ``int``, not as a type.
* ``frozenset``, which Codon 0.19 lacks, is checked as a ``set``: a call
  ``frozenset(...)`` becomes ``set(...)``, ``frozenset[X](...)`` becomes
  ``Set[X](...)``, and an annotation
  ``frozenset[X]`` / ``FrozenSet[X]`` becomes ``Set[X]``.
* A bare ``*`` in a signature (keyword-only parameters) becomes ``*_acs_kw``.
* A list display of two or more elements that are not all literals, e.g.
  ``[ElapsedTimeCondition(...), SpeedCondition(...)]``, becomes
  ``_acs_list(...)``: Codon types a display by its first element, and
  ``_acs_list`` makes it a list of conditions (or actions) as in Python.
* A dict display of two or more items whose values are not evidently of
  one type (all literals, all calls of the same name, all attributes of the
  same class), e.g. ``{"turn": turn, "lc": lc}``, becomes
  ``{"turn": _acs_base(turn), "lc": _acs_base(lc)}``: Codon types a display
  by its first item, and ``_acs_base`` makes each condition (or action) a
  ``BaseCondition`` (``BaseAction``), so the display is a dict of them as in
  Python; other values are left to Codon's display.  Assigned to an
  annotated ``x: dict[K, V]`` (``K`` and ``V`` plain names), such a display,
  of one item or more, becomes ``_acs_dict_into(Dict[K, V](), (k1, v1),
  ...)``, a dict of the annotated type.  A display of constructor calls of
  the module's own classes becomes one of the annotated type of a
  ``dict[K, B]`` parameter, or, unannotated, of ``Dict[str, B]`` with ``B``
  their nearest shared class (#489); a list display added to a ``list[B]``
  local or parameter (``a + [Sub()]``, ``+=``, ``extend()``) is a list of
  ``B``; and a class defined further down may be named in a function body.
* An ``Enum`` class derives from the ``enum`` shim's ``Enum[T]`` (``T`` the
  type of its values: ``str`` for ``class C(str, Enum)``, ``int`` for
  ``auto()``), and each member ``RED = "red"`` becomes the class variable
  ``RED: ClassVar[C] = C("RED", "red")``.  After the module, each such class
  is given a constructor by value (``C("red")``) with ``@extend``.
  An ``IntEnum`` / ``(int, Enum)`` derives from the shim's ``IntEnum[int]``
  (ordering, ``int()``), a ``StrEnum`` / ``(str, Enum)`` from ``StrEnum[str]``
  (equality with its string); no other enum has either.  An ``IntFlag`` /
  ``(int, Flag)`` derives from ``IntFlag[int]`` (an ``IntEnum``'s, and a truth
  value), a ``Flag`` from ``Flag[int]``; the ``@extend`` of each also declares
  its ``|``, ``&``, ``^``, ``~`` and ``in`` (:func:`_flag_operators`), typed by
  the class, which a method of the shim could not name.  A flag's value that
  is an expression of its members (``RW = R | W``, ``auto()``) is worked out
  (:func:`flag_member_values`); iteration skips composites, as Python 3.11+.
* A class deriving from an exception (a built-in one, one defined above it in
  the module, or a name ending in ``Error`` / ``Exception``) derives from
  ``Static[Base]``, as Codon 0.19 derives exceptions.
* A ``@classmethod`` becomes a ``@staticmethod`` without its ``cls``
  parameter; ``cls`` in its body names the class (Codon has no
  classmethods; a subclass calling it gets the base class's).
* ``super().__new__(cls)`` in a method of class ``C`` becomes
  ``C.__new__()``: Codon 0.19's ``object.__new__`` takes no class, and the
  class's own ``__new__()`` allocates an instance of it.
* A module-level ``Callable[...]`` alias becomes ``None``: Codon 0.19 reads
  the right-hand side as an expression and refuses it.  An annotation spells
  the callable out.
* ``raise X from None`` becomes ``raise X``: Codon 0.19 cannot type the
  ``None`` cause.  A builtin exception class raised as it is
  (``raise NotImplementedError``) becomes an instance of it
  (``raise NotImplementedError()``), as Python makes it: Codon 0.19 crashes
  on the class.
* The ``()`` default of a dropped ``Sequence[X]`` / ``Collection[X]`` /
  ``tuple[X, ...]`` parameter becomes ``List[X]()``, its tuple display
  default ``List[X]([...])``, and the ``None``
  default of a dropped ``Optional`` of one, or of a ``Callable``, the
  ``None`` of its type (``Optional[List[X]]()``): Codon would otherwise type
  the parameter as an empty tuple or as ``NoneType``.
* The tuple display default of a class-level ``Sequence[X]``-like or
  ``tuple[X, ...]`` field (``("a", "b")``, or ``field(default=(...))``)
  becomes ``List[X](["a", "b"])``, and its ``()`` ``List[X]()``: the field
  is a list, and a display a fixed-length tuple.  A ``list`` / ``dict`` /
  ``set`` ``default_factory`` of an abstract collection (``Mapping[str,
  str]``) becomes an empty one of the concrete type (``Dict[str, str]()``).
* A class-level ``tuple[X, ...]`` becomes ``List[X]``, and ``tuple(x)``
  becomes ``_acs_tuple(x)``, a list unless *x* is a tuple: Codon's tuples
  have a fixed length.  So does a return annotation ``tuple[X, ...]``
  (``Optional[tuple[X, ...]]`` an ``Optional[List[X]]``), so the value
  returned is checked; a parameter's is dropped.  ``callable(x)``, which
  Codon 0.19 lacks, becomes ``_acs_callable(x)``.
* ``object.__setattr__(obj, "name", value)`` (a frozen dataclass setting a
  field) becomes ``obj.name = (value)``; a ``field(init=False)`` without a
  default gets an empty value of its type (``List[float]()``), since Codon's
  constructor takes every field.
* A string inside a module-level ``Union[...]`` / ``Optional[...]`` alias (a
  forward reference) is unquoted: Codon takes no string as a type.
* ``Literal["a", "b"]`` (an annotation, or a module-level alias) becomes the
  type of its values (``str``): Codon's ``Literal`` types a compile-time
  value, and the ``typing`` shim's ``Literal`` is ``str``.
* A class-level ``tuple[A | B, ...]`` becomes ``List[Union[A, B]]``, which
  Codon 0.19 holds as it holds an attribute of a union of classes.
* ``except (A, B):``, which Codon 0.19 refuses ("expected type
  expression"), becomes one handler per class, in order, each with the same
  body (:func:`split_tuple_excepts`).
* A function decorator named in *dropped_decorators* is removed: the
  library check drops the run-time-only wrappers ``UNCALLED`` lists.  An
  annotation naming a module-level ``TypeVar`` (which types such a wrapper
  for mypy) is dropped.

* Iterating over an Enum class (``for m in Mode``, ``list(Mode)``,
  ``len(Mode)``, ``sorted(Mode)``, ``x in Mode``, a comprehension source)
  reads ``Mode._acs_members()`` and ``Mode[name]`` reads
  ``Mode._acs_by_name(name)``: two static methods added to each Enum class
  after the module, as is its lookup by value.  The Enum classes of other
  modules are named by *enum_classes* (:func:`enum_class_names`).

* ``return None`` (and a bare ``return``) of a function whose kept return
  annotation is ``Optional[X]`` becomes ``return Optional[X]()``: a bare
  ``None`` is Codon's ``NoneType``, which cannot be typed as one (#202).
* A top-level class with no ``__init__`` of its own whose base defines one
  (in the module, or among *init_classes*: another module's, the model's) is
  given one passing everything on to the base's, with ``@extend`` after the
  module: Codon 0.19 inherits no ``__init__`` (#201).

Before that, the scenario check gives each of the scenario's modules to
:func:`relative_module_imports`, which imports a module of the scenario's
package imported as a module (``from . import helpers``) the way Codon 0.19
resolves it (``import helpers as helpers``), on the same line (#310).

:func:`undeclared_attributes` reports what Codon additionally needs: an
attribute a class assigns on ``self`` must be declared at class level, with
its type, on any class in an inheritance hierarchy (Codon infers the type of
an undeclared one as ``None`` there).  Such a declaration is a bare
annotation, which Python ignores, so adding it changes nothing at run time.
"""

from __future__ import annotations

import ast
import builtins
import functools
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .scopes import ShadowScopes, _scope_bindings

__all__ = [
    "MISSING_BUILTIN_EXCEPTIONS",
    "PRELUDE",
    "UndeclaredAttribute",
    "class_declarations",
    "classes_with_init",
    "enum_class_names",
    "class_body_refusals",
    "hoist_class_bodies",
    "model_enum_names",
    "inherited_inits",
    "redirect_imports",
    "literal_all",
    "relative_module_imports",
    "split_tuple_excepts",
    "tuple_named_tuples",
    "uncalled_nodes",
    "transform_source",
    "transform_source_lines",
    "undeclared_attributes",
]

#: Python's built-in exceptions that Codon 0.19 lacks: the model's
#: ``_exceptions.codon`` declares them, and :data:`PRELUDE` imports them into
#: every transformed module, where Python finds them among its builtins.
MISSING_BUILTIN_EXCEPTIONS = (
    "BrokenPipeError",
    "ConnectionError",
    "FileNotFoundError",
    "ImportError",
    "ModuleNotFoundError",
    "PermissionError",
    "ProcessLookupError",
    "TimeoutError",
)

#: First lines of every transformed module; reported line numbers are shifted
#: back by as many.
PRELUDE = (
    "from autoware_carla_scenario._lists import _acs_list, _acs_base, "
    "_acs_dict_into, _acs_list_into\n"
    "from autoware_carla_scenario._tc import _Required as _acs_Required, "
    "_acs_Returning, _acs_callable, _acs_tuple, _acs_reduce\n"
    "from enum import IntEnum as _acs_IntEnum, IntFlag as _acs_IntFlag, "
    "StrEnum as _acs_StrEnum\n"
    "from autoware_carla_scenario._exceptions import "
    + ", ".join(MISSING_BUILTIN_EXCEPTIONS)
    + "\n"
)

_OPTIONAL_NAMES = {"Optional", "typing.Optional"}
_CALLABLE_NAMES = {"Callable", "typing.Callable", "collections.abc.Callable"}
_LITERAL_NAMES = {"Literal", "typing.Literal"}
_UNION_NAMES = {"Union", "typing.Union"}
_TUPLE_NAMES = {"tuple", "Tuple"}
_TYPE_VAR_NAMES = {"TypeVar", "typing.TypeVar"}
_CONCRETE_GENERICS = {
    "list",
    "List",
    "dict",
    "Dict",
    "set",
    "Set",
    "tuple",
    "Tuple",
    "ClassVar",
    "typing.List",
    "typing.Dict",
    "typing.Set",
    "typing.Tuple",
    "typing.ClassVar",
    # The `typing` shim's collections types (codon/typing.codon).
    "Deque",
    "Counter",
    "typing.Deque",
    "typing.Counter",
    # The `queue` shim's Queue[T] (codon/queue.codon).
    "queue.Queue",
    # The `subprocess` shim's (codon/subprocess.codon): what a process reads
    # and writes is a T (`bytes`, checked as `str`, or `str`).
    "subprocess.Popen",
    "subprocess.CompletedProcess",
    # The `typing` shim's IO[T] (codon/typing.codon), a file object.
    "IO",
    "typing.IO",
}
#: Abstract collections: kept in class-level declarations (the typing shim
#: maps them onto List / Dict), dropped from parameters, which a caller may
#: pass a tuple or another collection.
_ABSTRACT_GENERICS = {
    "Sequence",
    "MutableSequence",
    "Iterable",
    "Collection",
    "Mapping",
    "MutableMapping",
    "AbstractSet",
}
#: Python's immutable set, which Codon 0.19 lacks: checked as a ``Set`` (the
#: check only types it; nothing compiled is run, so nothing can mutate it).
_FROZENSET_NAMES = {"frozenset", "FrozenSet", "typing.FrozenSet"}
#: Annotations of a mapping whose dict display is built with
#: ``_acs_dict_into`` as the annotated type.
_DICT_ANNOTATIONS = {
    "dict",
    "Dict",
    "typing.Dict",
    "Mapping",
    "MutableMapping",
    "typing.Mapping",
    "typing.MutableMapping",
    "collections.abc.Mapping",
    "collections.abc.MutableMapping",
}
#: numpy's array class, which Codon's numpy has as ``ndarray[dtype, ndim]``.
_NDARRAY_NAMES = {"np.ndarray", "numpy.ndarray"}
_DTYPE_NAMES = {"np.dtype", "numpy.dtype"}
_LIST_ANNOTATIONS = {"list", "List", "typing.List"}
#: Bases that make a class no framework condition or action.
_PLAIN_BASES = {"object", "ABC", "abc.ABC"}
#: Types Codon displays on its own: a ``list[float]`` is not rewritten.
_SCALAR_NAMES = {"int", "float", "str", "bool", "bytes", "complex"}
_UNEXPRESSIBLE = {
    "Any",
    "object",
    "Callable",
    "type",
    "Type",
    "Literal",
    "Final",
    "NoReturn",
    "Never",
    "Sized",
    "Hashable",
    "Container",
    "Self",
    "TypeAlias",
}
#: ``TypeGuard[X]`` / ``TypeIs[X]``: what a function returns when it tells a
#: type checker about its argument, a ``bool`` (#514).
_BOOL_GUARDS = {"TypeGuard", "TypeIs"}
#: Annotations that wrap another: ``Final[X]`` and ``Annotated[X, ...]`` are
#: checked as ``X`` (a bare ``Final`` is dropped).
_WRAPPERS = {"Final", "Annotated"}
#: ``typing.cast``: ``cast(T, x)`` is ``x`` checked as ``T``.
_CAST_NAMES = {"cast", "typing.cast"}
#: Python's ``bytes``, which Codon 0.19 lacks: it reads a ``b"..."`` literal
#: as a ``str``, so an annotation naming ``bytes`` is checked as one.
_BYTES_NAMES = {"bytes", "builtins.bytes"}
#: CARLA's integer enumerations, which typesafe_carla's Codon library has as
#: values, not types (``TrafficLightState = _TrafficLightState()``, each member
#: an ``int``): an annotation naming one (``carla.TrafficLightState``) is the
#: ``int`` its members are.
_CARLA_INT_ENUMS = frozenset(
    {
        "ActorAttributeType",
        "ActorState",
        "CityObjectLabel",
        "ColorConverter",
        "GBufferTextureID",
        "LandmarkOrientation",
        "LaneChange",
        "LaneMarkingColor",
        "LaneMarkingType",
        "LaneType",
        "LightGroup",
        "MapLayer",
        "MaterialParameter",
        "TrafficLightState",
        "VehicleLightState",
    }
)
_CARLA_MODULES = {"carla", "typesafe_carla.carla"}

#: Class decorators dropped: Codon gives the class what they would.
_DROPPED_CLASS_DECORATORS = {
    "dataclass",
    "dataclasses.dataclass",
    "unique",
    "enum.unique",
    "runtime_checkable",
    "typing.runtime_checkable",
}
_FIELD_FUNCTIONS = {"field", "dataclasses.field"}
#: The names of ``functools`` that Codon's module (which has only ``partial``)
#: lacks and the transform handles (#378): ``lru_cache``, ``cache`` and
#: ``cached_property`` are decorators, and ``reduce`` a function.  The names
#: are left out of the ``from functools import`` line.
FUNCTOOLS_NAMES = ("cache", "cached_property", "lru_cache", "reduce")
_FUNCTOOLS_DROPPED = {"cache", "lru_cache"}
#: Function decorators dropped: Codon 0.19 rejects a decorated method, and
#: the method compiles the same without them.
_DROPPED_FUNCTION_DECORATORS = {"abstractmethod", "abc.abstractmethod"}
#: Builtins Codon 0.19 lacks, or has for fewer kinds of argument, called with
#: one argument: the ``_tc`` stand-in the call goes to instead.  ``tuple(x)``
#: of a list or a generator is a list (Codon's tuples have a fixed length; a
#: tuple stays itself); ``callable(x)`` is decided when Codon compiles.
_RENAMED_BUILTINS = {"tuple": "_acs_tuple", "callable": "_acs_callable"}

#: Enum bases (``enum.X`` too), and the value type a base implies.
_ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
_ENUM_VALUE_TYPES = {
    "IntEnum": "int",
    "StrEnum": "str",
    "Flag": "int",
    "IntFlag": "int",
}
_ENUM_MIXINS = {"str", "int", "float"}
_AUTO = {"auto", "enum.auto"}

#: Python's exception classes: a class deriving from one is an exception,
#: which Codon 0.19 derives with ``Static[...]``.
_BUILTIN_EXCEPTIONS = frozenset(
    name
    for name, obj in vars(builtins).items()
    if isinstance(obj, type) and issubclass(obj, BaseException)
)
_EXCEPTION_SUFFIXES = ("Error", "Exception")
_FACTORY_LITERALS = {"list": "[]", "dict": "{}", "set": "set()", "tuple": "()"}
#: The ``typing`` spellings of the generics a factory of
#: :data:`_FACTORY_LITERALS` builds (``List[int]`` for ``list``).
_TYPING_GENERICS = {
    "List": "list",
    "typing.List": "list",
    "Dict": "dict",
    "typing.Dict": "dict",
    "Set": "set",
    "typing.Set": "set",
}


@dataclass(frozen=True)
class UndeclaredAttribute:
    """``self.<name>`` assigned in *class_name* without a class-level declaration."""

    class_name: str
    name: str
    lineno: int
    suggested_type: str | None = None
    #: The method that annotates it (``self.x: T = ...``), when one does.
    annotated_in: str | None = None

    def message(self) -> str:
        hint = self.suggested_type or "<type>"
        if self.annotated_in is not None:
            return (
                f"{self.class_name}.{self.name} is annotated in "
                f"`{self.annotated_in}`; Codon needs it declared in the class "
                f"body: `{self.name}: {hint}` (a bare annotation, which Python "
                "ignores)"
            )
        return (
            f"{self.class_name}.{self.name} is assigned without a type: declare it "
            f"in the class body, e.g. `{self.name}: {hint}` (a bare annotation, "
            "which Python ignores; Codon needs it to type the attribute)"
        )


@dataclass
class _Edits:
    """Byte-span replacements on a source, applied back to front."""

    lines: list[bytes]
    starts: list[int] = field(default_factory=list)
    edits: list[tuple[int, int, str]] = field(default_factory=list)
    data: bytes = b""

    def __post_init__(self) -> None:
        self.data = b"".join(self.lines)
        offset = 0
        for line in self.lines:
            self.starts.append(offset)
            offset += len(line)

    def offset(self, lineno: int, col: int) -> int:
        return self.starts[lineno - 1] + col

    def text(self, start: int, end: int) -> str:
        return self.data[start:end].decode()

    def node_span(self, node: ast.AST) -> tuple[int, int]:
        return (
            self.offset(node.lineno, node.col_offset),  # type: ignore[attr-defined]
            self.offset(node.end_lineno, node.end_col_offset),  # type: ignore[attr-defined]
        )

    def replace(self, start: int, end: int, text: str) -> None:
        # Keep every statement on its line: a span that covered line breaks
        # keeps them (after the replacement, inside the brackets or the
        # statement it belonged to).
        lost = self.text(start, end).count("\n") - text.count("\n")
        self.edits.append((start, end, text + "\n" * max(lost, 0)))

    def supersede(self, start: int, end: int, text: str) -> None:
        """Like :meth:`replace`, but the edit wins over those already made on
        exactly this span (the first one made on a span otherwise wins)."""
        self.edits = [e for e in self.edits if (e[0], e[1]) != (start, end)]
        self.replace(start, end, text)

    def apply(self) -> str:
        data = self.data
        last = len(data) + 1
        for start, end, text in sorted(self.edits, key=lambda e: e[0], reverse=True):
            if end > last:
                continue  # nested in an edit already applied: the outer one wins
            data = data[:start] + text.encode() + data[end:]
            last = start
        return data.decode()


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _union_members(node: ast.AST) -> list[ast.AST] | None:
    """Members of an ``A | B | ...`` or ``Union[A, B, ...]`` annotation."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _union_members(node.left) or [node.left]
        right = _union_members(node.right) or [node.right]
        return left + right
    if isinstance(node, ast.Subscript) and _dotted(node.value) in _UNION_NAMES:
        inner = node.slice
        return list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
    return None


def _annotation_names(node: ast.AST) -> set[str]:
    """The bare names an annotation uses, inside string annotations too."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return set()
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out |= _annotation_names(sub)
    return out


def _names_abstract_collection(node: ast.AST) -> bool:
    """Whether an annotation names an abstract collection (``Sequence``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return False
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Name, ast.Attribute)):
            name = (_dotted(sub) or "").removeprefix("typing.")
            if name in _ABSTRACT_GENERICS:
                return True
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _names_abstract_collection(sub):
                return True
    return False


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _class_union(node: ast.AST) -> str | None:
    """``Union[A, B]`` for the element of a class-level ``tuple[A | B, ...]``.

    Codon 0.19 holds a list of a union of classes (``List[Union[A, B]]``), as
    it holds an attribute of one; a union with ``None`` in it, or one with a
    member Codon cannot express, gives ``None``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    members = _union_members(node)
    if members is None or len(members) < 2 or any(_is_none(m) for m in members):
        return None
    texts = [codon_annotation(m, class_level=True) for m in members]
    if any(t is None or t.startswith("Optional[") for t in texts):
        return None
    return f"Union[{', '.join(t for t in texts if t is not None)}]"


def _literal_type(node: ast.AST) -> str | None:
    """The type of the values of a ``Literal[...]`` (``"a", "b"``: ``str``).

    Codon's ``Literal`` is something else (a compile-time value's type, of
    ``int``, ``bool`` or ``str`` only), so a Python ``Literal`` of values of
    one type is checked as that type; ``None`` for anything else.
    """
    if not (isinstance(node, ast.Subscript) and _dotted(node.value) in _LITERAL_NAMES):
        return None
    inner = node.slice
    values = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
    kinds = {type(v.value) if isinstance(v, ast.Constant) else None for v in values}
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    return {bool: "bool", int: "int", str: "str"}.get(kind) if kind else None


def _ndarray(node: ast.Subscript, base_name: str) -> str | None:
    """Codon's ``np.ndarray[T, N]`` for numpy's shape-typed array, or ``None``.

    numpy types an array by its shape and its dtype
    (``np.ndarray[tuple[int, int], np.dtype[np.float32]]``: two dimensions of
    ``float32``); Codon's ``ndarray`` by its dtype and its number of
    dimensions (``np.ndarray[np.float32, 2]``).  A shape of a fixed number of
    ``int`` gives that number; anything else (``NDArray[np.float32]``, a shape
    of ``tuple[int, ...]`` or ``Any``) gives none, and the annotation is
    dropped.
    """
    if not (isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2):
        return None
    shape, dtype = node.slice.elts
    if not (
        isinstance(shape, ast.Subscript)
        and _dotted(shape.value) in _TUPLE_NAMES | {"typing.Tuple"}
    ):
        return None
    dims = shape.slice.elts if isinstance(shape.slice, ast.Tuple) else [shape.slice]
    if not dims or any(_dotted(d) != "int" for d in dims):
        return None
    if not (isinstance(dtype, ast.Subscript) and _dotted(dtype.value) in _DTYPE_NAMES):
        return None
    scalar = _dotted(dtype.slice)
    if scalar is None or scalar.removeprefix("typing.") in _UNEXPRESSIBLE:
        return None
    return f"{base_name}[{scalar}, {len(dims)}]"


def codon_annotation(node: ast.AST, *, class_level: bool = False) -> str | None:
    """The Codon spelling of a Python annotation, or ``None`` to drop it."""
    literal = _literal_type(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "None"
        if isinstance(node.value, str):
            try:
                inner = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return None
            return codon_annotation(inner, class_level=class_level)
        return None
    members = _union_members(node)
    if members is not None:
        rest = [m for m in members if not _is_none(m)]
        if len(rest) != 1:
            return None  # a real Union: Codon 0.19 cannot take one
        inner_text = codon_annotation(rest[0], class_level=class_level)
        if inner_text is None:
            return None
        return f"Optional[{inner_text}]" if len(rest) < len(members) else inner_text
    subscript = isinstance(node, ast.Subscript)
    base_name = _dotted(node.value if subscript else node)  # type: ignore[attr-defined]
    if base_name is None:
        return None
    base = base_name.removeprefix("typing.")
    if subscript and base_name in _OPTIONAL_NAMES:
        inner_text = codon_annotation(node.slice, class_level=class_level)  # type: ignore[attr-defined]
        return None if inner_text is None else f"Optional[{inner_text}]"
    if subscript and base in _WRAPPERS:
        wrapped = node.slice  # type: ignore[attr-defined]
        if isinstance(wrapped, ast.Tuple):
            wrapped = wrapped.elts[0]
        return codon_annotation(wrapped, class_level=class_level)
    if base in _UNEXPRESSIBLE or (base in _ABSTRACT_GENERICS and not class_level):
        return None
    if subscript and base.rsplit(".", 1)[-1] in _BOOL_GUARDS:
        return "bool"
    if base_name in _FROZENSET_NAMES:
        if not subscript:
            return None  # a bare frozenset: the element type is unknown
        inner_text = codon_annotation(node.slice, class_level=class_level)  # type: ignore[attr-defined]
        return None if inner_text is None else f"Set[{inner_text}]"
    if not subscript:
        if base_name in _BYTES_NAMES:
            return "str"
        module, _, name = base_name.rpartition(".")
        if module in _CARLA_MODULES and name in _CARLA_INT_ENUMS:
            return "int"
        return base_name
    if isinstance(node, ast.Subscript):
        if base_name in _NDARRAY_NAMES:
            return _ndarray(node, base_name)
        args = (
            list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        if base_name not in _CONCRETE_GENERICS and base not in _ABSTRACT_GENERICS:
            return None  # a generic Codon has no model of (NDArray, ...)
        if any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in args):
            # tuple[X, ...]: Codon's tuples have a fixed length.  A class-level
            # declaration holds a list instead, as `tuple(x)` is checked as one;
            # a parameter, which a caller may pass a tuple, is dropped.
            if not class_level or len(args) != 2 or base not in _TUPLE_NAMES:
                return None
            item = codon_annotation(args[0], class_level=class_level)
            if item is None:
                item = _class_union(args[0])
            return None if item is None else f"List[{item}]"
        converted = [codon_annotation(a, class_level=class_level) for a in args]
        if any(c is None for c in converted):
            return None
        return f"{base_name}[{', '.join(c for c in converted if c is not None)}]"
    return None


def callable_annotation(node: ast.AST) -> str | None:
    """``Callable[[A, B], R]`` in Codon's spelling, or ``None``.

    Codon 0.19 has a ``Callable[[A, B], R]`` type, which a function of that
    signature converts to.  The transform drops a ``Callable`` parameter
    annotation (the parameter becomes generic, so a lambda still checks) and
    leaves a class-level one as written, but the check needs a value to call
    with: one of that type, when Codon can express its argument and result
    types (not ``Any``).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    if not (isinstance(node, ast.Subscript) and _dotted(node.value) in _CALLABLE_NAMES):
        return None
    if not (isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2):
        return None
    params, result = node.slice.elts
    if not isinstance(params, ast.List):
        return None  # Callable[..., R]: no argument types to call with
    args = [codon_annotation(p, class_level=True) for p in params.elts]
    ret = codon_annotation(result, class_level=True)
    if ret is None or any(a is None for a in args):
        return None
    return f"Callable[[{', '.join(a for a in args if a is not None)}], {ret}]"


def kept_callable_parameter(node: ast.AST) -> str | None:
    """A ``Callable[[A, B], R]`` parameter annotation in Codon's spelling.

    ``Callable[Tuple[A, B], R]`` (the list of argument types a display the
    transform would rewrite), or ``None`` when the annotation is no
    ``Callable`` whose argument and result types Codon can express (it is
    dropped, as every ``Callable`` annotation once was).  A parameter stored
    in a ``Callable``-typed attribute or list (a callback registry) must keep
    the annotation: Codon aborts with ``std::bad_alloc`` on the generic
    parameter put in a typed container (#488).
    """
    member = optional_callable(node)
    if member is not None:
        # `Optional[Callable[[A], R]]` / `Callable[[A], R] | None`: Codon takes
        # a function, a lambda or `None` for the plain `Callable` (#495).
        node = member
    spelled = callable_annotation(node)
    if spelled is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node = ast.parse(node.value, mode="eval").body
    assert isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple)
    params, result = node.slice.elts
    assert isinstance(params, ast.List)
    if not params.elts:
        return spelled
    args = [codon_annotation(p, class_level=True) for p in params.elts]
    ret = codon_annotation(result, class_level=True)
    return f"Callable[Tuple[{', '.join(str(a) for a in args)}], {ret}]"


def optional_callable(node: ast.AST) -> ast.AST | None:
    """The ``Callable[...]`` of an ``Optional[Callable[...]]`` /
    ``Callable[...] | None`` annotation, or ``None`` for any other annotation."""
    member = _optional_member(node)
    if member is None:
        return None
    if isinstance(member, ast.Constant) and isinstance(member.value, str):
        try:
            member = ast.parse(member.value, mode="eval").body
        except SyntaxError:
            return None
    if isinstance(member, ast.Subscript) and _dotted(member.value) in _CALLABLE_NAMES:
        return member
    return None


def optional_callable_attribute(node: ast.AST) -> str | None:
    """``Optional[Callable[Tuple[A], R]]`` for a class-level optional callback.

    ``cb: Callable[[int], int] | None`` is a runtime ``|`` between a
    ``Callable`` and an ``Optional`` when Codon evaluates the class body, which
    it cannot (#495); ``Optional[...]`` is the same type, and the spelling of a
    parameter that is stored in it (:func:`kept_callable_parameter`).
    """
    member = optional_callable(node)
    if member is None:
        return None
    inner = kept_callable_parameter(member) or ast.unparse(member)
    return f"Optional[{inner}]"


#: Methods that put their argument in a collection.
_STORING_METHODS = {
    "append",
    "appendleft",
    "add",
    "insert",
    "extend",
    "setdefault",
    "put",
    "put_nowait",
}


#: A class's methods and, for each, the parameters it stores: the parameter
#: names in order (``self`` excluded) and the stored ones.
_Forwards = dict[str, tuple[list[str], set[str]]]


def forwarded_stores(cls: ast.ClassDef) -> _Forwards:
    """The parameters each method of *cls* stores, one that hands them to a
    method that stores them included (``register`` calls ``self._add(fn)``)."""
    methods = {
        m.name: m
        for m in cls.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forwards: _Forwards = {}
    for name, method in methods.items():
        every = [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]
        if not any(_dotted(d) == "staticmethod" for d in method.decorator_list):
            every = every[1:]
        forwards[name] = ([a.arg for a in every], set())
    changed = True
    while changed:
        changed = False
        for name, method in methods.items():
            params, stored = forwards[name]
            now = stored_parameters(method.body, set(params), forwards)
            if now - stored:
                stored |= now
                changed = True
    return forwards


def stored_parameters(
    body: list[ast.stmt], names: set[str], forwards: _Forwards | None = None
) -> set[str]:
    """The *names* the *body* puts in an attribute, a subscript or a collection.

    ``self.cb = fn``, ``self.cbs[key] = fn`` and ``self.cbs.append(fn)`` store
    ``fn``; calling it, or passing it on, does not.  A nested function or
    class is part of the body (a store there stores what it captured).
    """
    found: set[str] = set()
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        stack.extend(ast.iter_child_nodes(node))
        values: list[ast.expr] = []
        if isinstance(node, ast.Assign) and any(
            isinstance(t, (ast.Attribute, ast.Subscript)) for t in node.targets
        ):
            values = [node.value]
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and isinstance(node.target, (ast.Attribute, ast.Subscript))
        ):
            values = [node.value]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _STORING_METHODS
        ):
            values = list(node.args)
        if (
            forwards
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("self", "cls")
            and node.func.attr in forwards
        ):
            params, stored = forwards[node.func.attr]
            values += [
                a
                for i, a in enumerate(node.args)
                if i < len(params) and params[i] in stored
            ]
            values += [k.value for k in node.keywords if k.arg in stored]
        for value in values:
            if isinstance(value, ast.Name) and value.id in names:
                found.add(value.id)
    return found


def _members_or_self(node: ast.AST) -> list[ast.AST]:
    """The members of a union or an ``Optional`` (``None`` included), or *node*."""
    if isinstance(node, ast.Subscript) and _dotted(node.value) in (
        "Optional",
        "typing.Optional",
    ):
        return [*_members_or_self(node.slice), ast.Constant(None)]
    members = _union_members(node)
    if members is None:
        return [node]
    return [inner for member in members for inner in _members_or_self(member)]


def union_callable_annotation(node: ast.AST) -> str | None:
    """``_acs_Returning[R, 2]`` for a ``Callable[[A, B], R]`` with a union among
    ``A, B``, or an ``Optional`` of one; ``None`` for any other annotation.

    Codon 0.19 crashes on a ``Callable`` type with a union argument ("union
    already sealed").  The check calls with ``_acs_Returning[R, N]`` (the
    model's ``_tc.codon``) instead: a callable that takes ``N`` arguments of
    any type and returns an ``R``.  So the body is compiled with the number of
    arguments it passes checked, their types unchecked, and with the ``R`` it
    gets back checked.  Every member of each union, and ``R``,
    must be a type Codon can express.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    rest = [m for m in _members_or_self(node) if not _is_none(m)]
    if len(rest) != 1:
        return None
    node = rest[0]
    if not (isinstance(node, ast.Subscript) and _dotted(node.value) in _CALLABLE_NAMES):
        return None
    if not (isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2):
        return None
    params, result = node.slice.elts
    if not isinstance(params, ast.List):
        return None
    union = False
    for param in params.elts:
        members = [m for m in _members_or_self(param) if not _is_none(m)]
        union = union or len(members) > 1
        if any(codon_annotation(m, class_level=True) is None for m in members):
            return None
    ret = codon_annotation(result, class_level=True)
    if not union or ret is None:
        return None
    return f"_acs_Returning[{ret}, {len(params.elts)}]"


def _field_default(
    call: ast.Call, edits: _Edits, annotation: ast.expr | None = None
) -> str | None:
    """The default a ``field(...)`` call stands for, or ``None`` if it has none.

    An empty collection takes the type of the field's *annotation* when Codon
    can express it (``list[int]()``): Codon types a bare ``[]`` from what is
    put in it, and nothing is, so it would be a list of ``None``.  So does
    ``default=()`` of a ``tuple[X, ...]``, checked as a ``List[X]``.  One whose
    annotation names ``Any`` (``dict[str, Any]``, a free-form mapping) takes
    the annotation as written, as the class-level declaration keeps it: the
    ``Any`` of the ``typing`` shim, an empty class, so the default has the
    attribute's type and nothing can be put in it.
    """
    for kw in call.keywords:
        if kw.arg == "default":
            if (
                isinstance(kw.value, ast.Tuple)
                and not kw.value.elts
                and annotation is not None
            ):
                # `field(default=())` of a `tuple[X, ...]`, which is checked as
                # a `List[X]` (Codon's tuples have a fixed length): an empty
                # one of that type.
                typed = codon_annotation(annotation, class_level=True)
                if typed is not None and typed.startswith("List["):
                    return f"{typed}()"
            return edits.text(*edits.node_span(kw.value))
        if kw.arg == "default_factory":
            factory = kw.value
            if isinstance(factory, ast.Lambda):
                return "(" + edits.text(*edits.node_span(factory.body)) + ")"
            name = _dotted(factory)
            if name in _FACTORY_LITERALS:
                typed = (
                    codon_annotation(annotation, class_level=True)
                    if isinstance(annotation, ast.Subscript)
                    and _builtin_generic(annotation.value) == name
                    else None
                )
                if typed is None and annotation is not None:
                    typed = _abstract_factory_type(annotation, name)
                if (
                    typed is None
                    and isinstance(annotation, ast.Subscript)
                    and _builtin_generic(annotation.value) == name
                    and _names_any(annotation)
                ):
                    typed = edits.text(*edits.node_span(annotation))
                return f"{typed}()" if typed is not None else _FACTORY_LITERALS[name]
            return edits.text(*edits.node_span(factory)) + "()"
    if annotation is not None and any(
        kw.arg == "init"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
        for kw in call.keywords
    ):
        # `field(init=False)`, set by __post_init__: Codon's constructor takes
        # every field, so the field gets an empty value of its type to leave
        # it out of the call.
        typed = codon_annotation(annotation, class_level=True)
        if typed is not None:
            return f"{typed}()"
    return None


#: The abstract collections a ``default_factory`` builds a value of, by the
#: factory, and the concrete Codon collection that value is.
_ABSTRACT_FACTORIES = {
    "list": ({"Sequence", "MutableSequence", "Iterable", "Collection"}, "List"),
    "dict": ({"Mapping", "MutableMapping"}, "Dict"),
    "set": ({"AbstractSet"}, "Set"),
}


def _abstract_factory_type(annotation: ast.AST, factory: str) -> str | None:
    """``Dict[K, V]`` for ``Mapping[K, V] = field(default_factory=dict)``.

    The empty collection a factory builds for a field annotated with an
    abstract collection the ``typing`` shim maps onto it (``Mapping`` onto
    ``Dict``): Codon would type a bare ``{}`` from what is put in it, and
    nothing is, so a dict of ``None``.  ``None`` when the annotation is no
    such collection or names a type Codon cannot express.
    """
    if factory not in _ABSTRACT_FACTORIES or not isinstance(annotation, ast.Subscript):
        return None
    names, concrete = _ABSTRACT_FACTORIES[factory]
    base_name = _dotted(annotation.value) or ""
    if base_name.rsplit(".", 1)[-1] not in names:
        return None
    typed = codon_annotation(annotation, class_level=True)
    if typed is None or not typed.startswith(f"{base_name}["):
        return None
    return concrete + typed[len(base_name) :]


#: The abstract collections an empty tuple is a value of, checked as a list.
_SEQUENCE_NAMES = {"Sequence", "MutableSequence", "Iterable", "Collection"}


def _sequence_list(annotation: ast.AST) -> str | None:
    """``List[X]`` for a ``Sequence[X]``-like or ``tuple[X, ...]`` annotation.

    The type a parameter's ``()`` default is checked as (``Collection[int] =
    ()``): an empty list of the element type, as ``field(default=())`` is.
    ``None`` for any other annotation, or an element Codon cannot express.
    """
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None
    if not isinstance(annotation, ast.Subscript):
        return None
    base = (_dotted(annotation.value) or "").rsplit(".", 1)[-1]
    if base in _TUPLE_NAMES:
        typed = codon_annotation(annotation, class_level=True)
        return typed if typed is not None and typed.startswith("List[") else None
    if base not in _SEQUENCE_NAMES or isinstance(annotation.slice, ast.Tuple):
        return None
    item = codon_annotation(annotation.slice, class_level=True)
    return None if item is None else f"List[{item}]"


def _optional_member(annotation: ast.AST) -> ast.AST | None:
    """``X`` of an ``Optional[X]`` / ``X | None`` / ``Union[X, None]``, or ``None``."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None
    if isinstance(annotation, ast.Subscript) and _dotted(annotation.value) in (
        _OPTIONAL_NAMES
    ):
        return annotation.slice
    members = _union_members(annotation)
    if members is None or not any(_is_none(m) for m in members):
        return None
    rest = [m for m in members if not _is_none(m)]
    return rest[0] if len(rest) == 1 else None


def _names_any(annotation: ast.AST) -> bool:
    """Whether *annotation* names ``Any`` (``dict[str, Any]``)."""
    return any(
        _dotted(n) in ("Any", "typing.Any")
        for n in ast.walk(annotation)
        if isinstance(n, (ast.Name, ast.Attribute))
    )


def _is_builtin_exception(name: str) -> bool:
    """Whether *name* is a builtin exception class (``ValueError``)."""
    value = getattr(builtins, name, None)
    return isinstance(value, type) and issubclass(value, BaseException)


def _builtin_generic(node: ast.AST) -> str | None:
    """The builtin a generic's base names: ``list`` for ``list`` or ``List``."""
    name = _dotted(node)
    return _TYPING_GENERICS.get(name, name) if name is not None else None


class _Rewriter(ShadowScopes):
    def __init__(
        self,
        edits: _Edits,
        exceptions: set[str],
        dropped_decorators: Collection[str] = (),
    ) -> None:
        super().__init__()
        self.edits = edits
        #: Function decorators dropped besides :data:`_DROPPED_FUNCTION_DECORATORS`.
        self._dropped_decorators = set(dropped_decorators)
        self._class_depth = 0
        self._function_depth = 0
        #: Top-level classes of the module that are exceptions.
        self._exceptions = exceptions
        #: The class whose body is being visited (``None`` in a function).
        self._class_name: str | None = None
        self._forwards: _Forwards = {}
        #: The class whose method (or a function inside one) is being visited.
        self._method_class: str | None = None
        #: Top-level classes of the module defined below the statement
        #: being visited.
        self._later_classes: set[str] = set()
        #: Module-level ``TypeVar``s: an annotation naming one is dropped, as
        #: Codon (whose ``typing`` shim makes a ``TypeVar`` a string) cannot
        #: type with it.
        self._type_vars: set[str] = set()
        #: Enum classes rewritten at the module's top level: each is given a
        #: constructor by value, a list of its members and a lookup by name
        #: (:func:`_enum_lookups`).
        self.enums: list[_EnumInfo] = []
        #: Local name -> the ``functools`` name it is imported as.
        self._functools_names: dict[str, str] = {}
        #: The list displays (by ``id``) whose target type is known, and the
        #: class name of its elements: ``_acs_list_into(List[B](), ...)``.
        self._list_targets: dict[int, str] = {}
        #: The dict displays (by ``id``) whose target type is known (a
        #: parameter ``dict[str, B]``), and that type: ``Dict[str, B]``.
        self._dict_targets: dict[int, str] = {}
        #: The local lists (name -> the element class of its ``list[B]``, or
        #: ``None`` when it is no list of a known class) of the module and of
        #: each function being visited, innermost last: a list display added
        #: to one (``a + [Sub()]``) is a list of that class (#489).
        #: Each with the names the scope binds in any way and the depth of
        #: ``_local_bound`` it opens, so a name bound by a loop, a ``with``, a
        #: comprehension or a lambda is no tracked list.
        self._noted_targets: set[int] = set()
        self._list_scopes: list[tuple[dict[str, str | None], set[str] | None, int]] = [
            ({}, None, 0)
        ]
        #: The element class of the ``-> list[B]`` of each function being
        #: visited (``None``: no such return annotation).
        self._return_elements: list[str | None] = []
        #: Top-level classes of the module -> the names of their bases.
        self._class_bases: dict[str, list[str]] = {}
        #: Top-level functions of the module and the ``__init__`` / dataclass
        #: fields of its classes -> the parameters (name, annotation) a call
        #: passes its arguments to, in order.
        self._signatures: dict[str, list[tuple[str, ast.expr | None]]] = {}
        #: Local names of the ``functools`` module.
        self._functools_modules: set[str] = set()

    def _functools_name(self, node: ast.AST) -> str | None:
        """The :data:`FUNCTOOLS_NAMES` name *node* (``lru_cache`` or
        ``functools.lru_cache``) refers to, if any."""
        if isinstance(node, ast.Name):
            return self._functools_names.get(node.id)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in self._functools_modules
            and node.attr in FUNCTOOLS_NAMES
        ):
            return node.attr
        return None

    def visit_Name(self, node: ast.Name) -> None:
        # A loop variable, a `with` or tuple target, a walrus: no tracked list.
        if isinstance(node.ctx, ast.Store) and id(node) not in self._noted_targets:
            self._note_list_local(node.id, None)
        if (
            isinstance(node.ctx, ast.Load)
            and self._functools_names.get(node.id) == "reduce"
        ):
            self.edits.replace(*self.edits.node_span(node), "_acs_reduce")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._functools_name(node) == "reduce":
            self.edits.replace(*self.edits.node_span(node), "_acs_reduce")
        else:
            self.generic_visit(node)

    # -- annotations -----------------------------------------------------

    def visit_Module(self, node: ast.Module) -> None:
        self.enter_module(node)
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and _is_functools_import(stmt):
                for alias in stmt.names:
                    if alias.name in FUNCTOOLS_NAMES:
                        self._functools_names[alias.asname or alias.name] = alias.name
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name == "functools":
                        self._functools_modules.add(alias.asname or alias.name)
        # A name the module binds again (a parameter ``reduce``, a ``def cache``)
        # is no longer functools's: leave it, and Codon refuses what is missing.
        rebound = (
            {
                n.id
                for n in ast.walk(node)
                if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
            }
            | {
                n.name
                for n in ast.walk(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            | {n.arg for n in ast.walk(node) if isinstance(n, ast.arg)}
        )
        self._functools_names = {
            local: name
            for local, name in self._functools_names.items()
            if local not in rebound
        }
        later = {stmt.name for stmt in node.body if isinstance(stmt, ast.ClassDef)}
        # A name imported further down (the `if TYPE_CHECKING:` import of a
        # module cycle, moved to the end of the module) is as unnamable (#507).
        imported = {stmt: _imported_names(stmt) for stmt in node.body}
        for names in imported.values():
            later |= names
        self._scan_signatures(node)
        self._type_vars = {
            target.id
            for stmt in node.body
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and _dotted(stmt.value.func) in _TYPE_VAR_NAMES
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                later.discard(stmt.name)  # a class may name itself
            later -= imported[stmt]
            self._later_classes = set(later)
            self.visit(stmt)

    def _scan_signatures(self, module: ast.Module) -> None:
        """Note the module's classes and the parameters its callables take.

        For the list displays (:meth:`visit_List`): a class's bases tell the
        nearest class two elements share, and a parameter annotated
        ``list[B]`` the type of the display passed to it.  A name the module
        defines twice is left out (which one is called is not told here).
        """
        seen: set[str] = set()
        duplicated: set[str] = set()
        for stmt in module.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                (duplicated if stmt.name in seen else seen).add(stmt.name)
        for stmt in module.body:
            if getattr(stmt, "name", None) in duplicated:
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._signatures[stmt.name] = _parameters(stmt.args, skip_first=False)
            elif isinstance(stmt, ast.ClassDef):
                self._class_bases[stmt.name] = [
                    name
                    for base in stmt.bases
                    for name in [
                        _dotted(base.value if isinstance(base, ast.Subscript) else base)
                    ]
                    if name is not None
                ]
                init = next(
                    (
                        sub
                        for sub in stmt.body
                        if isinstance(sub, ast.FunctionDef) and sub.name == "__init__"
                    ),
                    None,
                )
                if init is not None:
                    self._signatures[stmt.name] = _parameters(
                        init.args, skip_first=True
                    )
                elif any(
                    _dotted(d.func if isinstance(d, ast.Call) else d)
                    in {"dataclass", "dataclasses.dataclass"}
                    for d in stmt.decorator_list
                ):
                    self._signatures[stmt.name] = [
                        (sub.target.id, sub.annotation)
                        for sub in stmt.body
                        if isinstance(sub, ast.AnnAssign)
                        and isinstance(sub.target, ast.Name)
                    ]

    def _later(self, name: str) -> bool:
        """Whether Codon cannot name the class *name* here: it is defined
        further down the module and this is not a function body.

        A module-level or class-level statement runs at once, and a signature
        is read when the function is defined; a function body is typed when
        the function is first called, after the whole module (#489).
        """
        return name in self._later_classes and self._function_depth == 0

    def _list_element(self, annotation: ast.AST | None) -> str | None:
        """``B`` for a ``list[B]`` annotation of a class Codon can name, else ``None``.

        ``B`` is a plain class name (not a scalar, ``Any``, a ``TypeVar`` or a
        class defined further down, unless in a function body).
        """
        if (
            not isinstance(annotation, ast.Subscript)
            or _dotted(annotation.value) not in _LIST_ANNOTATIONS
            or not isinstance(annotation.slice, ast.Name)
        ):
            return None
        name = annotation.slice.id
        if (
            name in _UNEXPRESSIBLE
            or name in _SCALAR_NAMES
            or name in _CONCRETE_GENERICS | _ABSTRACT_GENERICS
            or name in self._type_vars
            or self._later(name)
        ):
            return None
        return name

    def _mark_list(self, value: ast.AST | None, annotation: ast.AST | None) -> None:
        """Note that the list display *value* is to be a ``list[B]`` of *annotation*."""
        if isinstance(value, ast.List) and listable_display(value):
            element = self._list_element(annotation)
            if element is not None:
                self._list_targets[id(value)] = element

    def _mark_list_arguments(self, node: ast.Call) -> None:
        """Note the list displays passed to a ``list[B]`` parameter of the
        module's own function or class (``total([Dbl(), Neg()])``)."""
        callee = _dotted(node.func)
        parameters = self._signatures.get(callee) if callee is not None else None
        if parameters is None:
            return
        for (_, annotation), arg in zip(parameters, node.args):
            self._mark_list(arg, annotation)
            self._mark_dict(arg, annotation)
        by_name = dict(parameters)
        for kw in node.keywords:
            if kw.arg is not None and kw.arg in by_name:
                self._mark_list(kw.value, by_name[kw.arg])
                self._mark_dict(kw.value, by_name[kw.arg])

    def _mark_dict(self, value: ast.AST | None, annotation: ast.AST | None) -> None:
        """Note that the dict display *value*, of the module's own classes, is
        to be the ``dict[K, B]`` of *annotation* (``total({"a": Circle()})``)."""
        if isinstance(value, ast.Dict) and self._own_class_values(value):
            typed = self._dict_annotation_type(annotation)
            if typed is not None:
                self._dict_targets[id(value)] = typed

    def _own_class_values(self, node: ast.Dict) -> bool:
        """Whether every value of the dict display *node* constructs a class
        of the module (``{"a": Circle(), "b": Square()}``)."""
        return (
            bool(node.values)
            and all(key is not None for key in node.keys)
            and all(
                isinstance(value, ast.Call) and _dotted(value.func) in self._class_bases
                for value in node.values
            )
        )

    def _common_dict_type(self, node: ast.Dict) -> str | None:
        """``Dict[K, B]`` for the unannotated dict display *node* of classes
        of the module, ``B`` the nearest class they share and ``K`` the type
        of its literal keys; ``None`` when either is not told."""
        if len(node.values) < 2 or not self._own_class_values(node):
            return None
        keys = [key for key in node.keys if key is not None]
        if all(
            isinstance(k, ast.JoinedStr)
            or (isinstance(k, ast.Constant) and isinstance(k.value, str))
            for k in keys
        ):
            key = "str"
        elif all(
            isinstance(k, ast.Constant)
            and isinstance(k.value, int)
            and not isinstance(k.value, bool)
            for k in keys
        ):
            key = "int"
        else:
            return None
        base = self._common_base(node.values)
        return None if base is None else f"Dict[{key}, {base}]"

    def _list_scope_base(self, node: ast.AST) -> str | None:
        """The element class of the list *node* when it is a local known to be
        a ``list[B]`` (or a sum of such), else ``None``."""
        if isinstance(node, ast.Name):
            name = node.id
            if any(
                name in bound
                for bound, _ in self._local_bound[self._list_scopes[-1][2] :]
            ):
                return None  # a comprehension, lambda or class binds it
            for known, bound, _ in reversed(self._list_scopes):
                if name in known:
                    return known[name]
                if (
                    (name in bound)
                    if bound is not None
                    else (name in self._module_bound)
                ):
                    return None  # bound otherwise: a loop, a `with`, ...
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._list_scope_base(node.left) or self._list_scope_base(node.right)
        return None

    def _mark_added(self, base: str | None, display: ast.AST | None) -> None:
        """Make the list display *display*, added to a list of *base*, a list
        of it."""
        if base is not None and isinstance(display, ast.List):
            if listable_display(display) and not self._later(base):
                self._list_targets[id(display)] = base

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add):
            # `a + [Sub()]` with `a: list[Base]`: the display is a list of the
            # base too, as Python's is.
            self._mark_added(self._list_scope_base(node.left), node.right)
            self._mark_added(self._list_scope_base(node.right), node.left)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, ast.Add):
            self._noted_targets.add(id(node.target))  # `a += [..]` is still a list
        if isinstance(node.op, ast.Add):
            self._mark_added(self._list_scope_base(node.target), node.value)
        self.generic_visit(node)

    def _note_list_local(
        self, name: str, base: str | None, target: ast.AST | None = None
    ) -> None:
        """Note that the local *name* is a list of *base* (``None``: unknown)."""
        if target is not None:
            self._noted_targets.add(id(target))
        if self._class_depth == 0:
            self._list_scopes[-1][0][name] = base

    def _common_base(self, elements: list[ast.expr]) -> str | None:
        """The nearest class of the module that the classes constructed by
        *elements* (``Dbl()``, ``Neg()``) all derive from, if there is one."""
        names = []
        for element in elements:
            if not isinstance(element, ast.Call):
                return None
            callee = _dotted(element.func)
            if callee is None or callee not in self._class_bases:
                return None
            names.append(callee)
        if len(set(names)) < 2:
            return None
        lines = [self._ancestors(name) for name in names]
        shared = set.intersection(*(set(line) for line in lines))
        for ancestor in lines[0]:
            if ancestor in shared and not self._later(ancestor):
                # A base from elsewhere may be a framework condition or action,
                # whose displays `_acs_list` upcasts to BaseCondition /
                # BaseAction: leave those to it, a List[Mid] is no
                # List[BaseCondition].
                if any(
                    base not in self._class_bases and base not in _PLAIN_BASES
                    for line in self._ancestors(ancestor)
                    for base in self._class_bases[line]
                ):
                    return None
                return ancestor
        return None

    def _ancestors(self, name: str) -> list[str]:
        """*name* and the classes of the module it derives from, nearest first."""
        out: list[str] = []
        queue = [name]
        while queue:
            current = queue.pop(0)
            if current in out or current not in self._class_bases:
                continue
            out.append(current)
            queue.extend(self._class_bases[current])
        return out

    def _rewrite_annotation(self, node: ast.AST, *, class_level: bool) -> str | None:
        """Rewrite *node* in place; ``None`` when it must be dropped instead."""
        if not class_level and _annotation_names(node) & self._later_classes:
            return None  # a class defined further down: Codon cannot name it yet
        if _annotation_names(node) & self._type_vars:
            return None  # a TypeVar, which types for mypy only
        converted = codon_annotation(node, class_level=class_level)
        if converted is None and class_level:
            converted = optional_callable_attribute(node)
        if converted is not None:
            self._write_annotation(node, converted)
        return converted

    def _write_annotation(self, node: ast.AST, converted: str) -> None:
        start, end = self.edits.node_span(node)
        if self.edits.text(start, end) != converted:
            # An annotation broken over lines keeps its line breaks inside
            # its brackets: the rest of its statement may follow it
            # (`x: Optional[\n    T\n] = None`).
            lost = self.edits.text(start, end).count("\n")
            text = converted
            if lost and text.endswith("]"):
                text = text[:-1] + "\n" * lost + "]"
            self.edits.replace(start, end, text)

    def _list_return(self, node: ast.AST) -> str | None:
        """The ``List[X]`` a ``tuple[X, ...]`` return annotation is checked as.

        Codon's tuples have a fixed length, so the annotation would be
        dropped, and with it the check of what the function returns (#191).
        A class-level ``tuple[X, ...]`` is a ``List[X]``, and ``tuple(x)`` a
        list, so the function returns one: its annotation becomes the
        class-level spelling, ``List[X]`` (``Optional[List[X]]``,
        ``dict[str, List[X]]``).  ``None`` for any other annotation, and for
        one naming an abstract collection (dropped from a signature, a caller
        may pass a tuple for it), a class defined further down or a
        ``TypeVar``; a ``Union``, ``Any`` or ``Callable`` is dropped at any
        level.
        """
        if codon_annotation(node) is not None:
            return None  # kept as it is
        if _annotation_names(node) & (self._later_classes | self._type_vars):
            return None
        if _names_abstract_collection(node):
            return None
        # What is left that only the class-level spelling keeps is a
        # variable-length tuple: abstract collections are the only other.
        return codon_annotation(node, class_level=True)

    def _keep_callable(self, arg: ast.arg, body: list[ast.stmt]) -> bool:
        """Write the ``Callable`` annotation of a parameter the body stores (#488).

        A parameter put in an attribute, a list or a dict is a value of a
        ``Callable``-typed one, which Codon aborts on (``std::bad_alloc``)
        when the parameter is generic.  One the body only calls stays dropped:
        it is generic, so a closure that reads a local variable of its
        function, which Codon cannot convert to a ``Callable``, still checks.
        """
        node = arg.annotation
        if node is None or not stored_parameters(body, {arg.arg}, self._forwards):
            return False
        if _annotation_names(node) & (self._later_classes | self._type_vars):
            return False
        kept = kept_callable_parameter(node)
        if kept is None:
            return False
        self._write_annotation(node, kept)
        return True

    def _rewrite_arguments(
        self,
        args: ast.arguments,
        *,
        skip_first: bool,
        body: list[ast.stmt] | None = None,
    ) -> None:
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if skip_first:
            every = every[1:]  # removed as a whole (_rewrite_classmethod)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                every.append(extra)
        for arg in every:
            if arg.annotation is None:
                continue
            if self._rewrite_annotation(arg.annotation, class_level=False) is None:
                if self._keep_callable(arg, body or []):
                    continue
                name_end = self.edits.offset(arg.lineno, arg.col_offset) + len(
                    arg.arg.encode()
                )
                self.edits.replace(
                    name_end, self.edits.node_span(arg.annotation)[1], ""
                )
        if args.kwonlyargs and args.vararg is None:
            self._rewrite_bare_star(args)
        self._default_required_keywords(args)

    def _kept_optional_callable(self, arg: ast.arg, body: list[ast.stmt]) -> bool:
        """Whether *arg* is an optional ``Callable`` parameter the body stores."""
        node = arg.annotation
        return (
            node is not None
            and optional_callable(node) is not None
            and stored_parameters(body, {arg.arg}, self._forwards) != set()
            and not _annotation_names(node) & (self._later_classes | self._type_vars)
            and kept_callable_parameter(node) is not None
        )

    def _rewrite_defaults(
        self, args: ast.arguments, body: list[ast.stmt] | None = None
    ) -> None:
        """Type a default the dropped annotation of its parameter would type.

        A ``Sequence[X]`` / ``Collection[X]`` / ``tuple[X, ...]`` parameter is
        dropped (a caller may pass a tuple), so the parameter is generic, and
        its default ``()`` would make it Codon's empty tuple, which has no
        elements, no ``X`` and a fixed length.  It becomes ``List[X]()``, the
        value a caller's list of ``X`` is, as ``field(default=())`` does, and
        a tuple display (``("a", "b")``, a fixed-length ``Tuple[str, str]``)
        a list of its elements, ``List[X](["a", "b"])``, as a field's.  An
        empty ``[]`` / ``{}`` (of a ``list[X]`` / ``dict[K, V]`` parameter,
        or ``[]`` of a sequence) becomes an empty one of that type: Codon
        would make it a list of ``None``.

        A ``None`` default of an ``Optional`` of one (``Optional[Sequence[X]]
        = None``) or of a ``Callable`` (also dropped) would be Codon's
        ``NoneType``, which has nothing of ``X``, and no call: a body that
        tests it for ``None`` first is still compiled with it.  It becomes the
        ``None`` of the ``Optional`` (``Optional[List[X]]()``), a ``Callable``
        spelled as the library check calls with it (:func:`callable_annotation`,
        :func:`union_callable_annotation`).  The source keeps its default.
        """
        positional = [*args.posonlyargs, *args.args]
        pairs = [
            *zip(positional[len(positional) - len(args.defaults) :], args.defaults),
            *zip(args.kwonlyargs, args.kw_defaults),
        ]
        for arg, default in pairs:
            if default is None or arg.annotation is None:
                continue
            if _annotation_names(arg.annotation) & (
                self._later_classes | self._type_vars
            ):
                continue  # a type Codon cannot name here
            typed: str | None = None
            if isinstance(default, ast.Tuple) and default.elts:
                # `("a", "b")`: a list of the elements, as a field's (#188).
                listed = _sequence_list(arg.annotation)
                if listed is not None:
                    self._list_display(default, listed)
                continue
            if isinstance(default, ast.Tuple):
                typed = _sequence_list(arg.annotation)
            elif isinstance(default, (ast.List, ast.Dict)) and not (
                default.elts if isinstance(default, ast.List) else default.keys
            ):
                # `[]` / `{}`, which Codon types from what is put in it, and
                # nothing is: a list of `None` ("'List[NoneType]' does not
                # match expected type 'List[int]'").  An empty one of the
                # parameter's type, as `field(default_factory=list)` is.
                kept = codon_annotation(arg.annotation)
                kinds = (
                    ("list", "List")
                    if isinstance(default, ast.List)
                    else (
                        "dict",
                        "Dict",
                    )
                )
                if kept is not None and kept.split("[", 1)[0] in kinds:
                    typed = kept if "[" in kept else None
                elif isinstance(default, ast.List):
                    typed = _sequence_list(arg.annotation)
            elif _is_none(default):
                if self._kept_optional_callable(arg, body or []):
                    continue  # the annotation is the plain `Callable` (#495)
                member = _optional_member(arg.annotation)
                inner = (
                    None
                    if member is None
                    else _sequence_list(member)
                    or callable_annotation(member)
                    or union_callable_annotation(member)
                )
                typed = None if inner is None else f"Optional[{inner}]"
            if typed is not None:
                self.edits.replace(*self.edits.node_span(default), f"{typed}()")

    def _default_required_keywords(self, args: ast.arguments) -> None:
        """Give a required keyword-only parameter after a default one.

        Codon refuses a parameter without a default after one with
        (``def f(x: float = 0.0, *, label: str)``).  The default given,
        ``_acs_Required()`` (the model's ``_tc._Required``), is of no
        parameter's type, so a call that leaves the argument out is still a
        compile error, at the call.
        """
        defaulted = bool(args.defaults)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                defaulted = True
                continue
            if not defaulted:
                continue
            end = (
                self.edits.node_span(arg.annotation)[1]
                if arg.annotation is not None
                else self.edits.offset(arg.lineno, arg.col_offset)
                + len(arg.arg.encode())
            )
            self.edits.replace(end, end, " = _acs_Required()")

    def _rewrite_bare_star(self, args: ast.arguments) -> None:
        first_kw = args.kwonlyargs[0]
        end = self.edits.offset(first_kw.lineno, first_kw.col_offset)
        positional = [*args.posonlyargs, *args.args]
        if positional:
            last = positional[-1]
            last_end = self.edits.node_span(last)[1]
            defaults_end = (
                self.edits.node_span(args.defaults[-1])[1]
                if args.defaults
                else last_end
            )
            start = max(last_end, defaults_end)
        else:
            start = end - 1
            while start > 0 and self.edits.text(start, start + 1) != "(":
                start -= 1
        star = self.edits.text(start, end).rfind("*")
        if star >= 0:
            pos = start + len(self.edits.text(start, end)[:star].encode())
            self.edits.replace(pos, pos + 1, "*_acs_kw")

    def _rewrite_returns(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.returns is None:
            return
        listed = self._list_return(node.returns)
        if listed is not None:
            self._write_annotation(node.returns, listed)
            self._rewrite_none_returns(node, listed)
            return
        kept = self._rewrite_annotation(node.returns, class_level=False)
        if kept is not None:
            self._rewrite_none_returns(node, kept)
        else:
            ann_start, ann_end = self.edits.node_span(node.returns)
            head = self.edits.text(self.edits.offset(node.lineno, 0), ann_start)
            arrow = head.rfind("->")
            if arrow >= 0:
                before = head[:arrow].rstrip(" ")
                start = self.edits.offset(node.lineno, 0) + len(before.encode())
                breaks = self.edits.text(start, ann_end).count("\n")
                if breaks and before.endswith(")"):
                    # An annotation over several lines: its line breaks go
                    # inside the parameters' brackets, so no line moves.
                    self.edits.replace(start - 1, ann_end, "\n" * breaks + ")")
                else:
                    self.edits.replace(start, ann_end, "")

    def _rewrite_none_returns(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, kept: str
    ) -> None:
        """Make each ``return None`` of an ``Optional[X]`` function the ``None`` of X.

        A bare ``None`` is Codon's ``NoneType``: a function whose only return
        is ``return None`` (or ``return``) cannot be typed as an
        ``Optional[X]`` ("cannot typecheck 'None'").  Each becomes ``return
        Optional[X]()``, the ``None`` of the kept return annotation *kept*
        (#202).  Only the function's own returns: a nested function or class
        has its own annotation, and a generator's ``return`` ends it.
        """
        if not kept.startswith("Optional[") or not kept.endswith("]"):
            return
        returns: list[ast.Return] = []
        stack: list[ast.AST] = list(node.body)
        while stack:
            sub = stack.pop()
            if isinstance(
                sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue
            if isinstance(sub, (ast.Yield, ast.YieldFrom)):
                return  # a generator: its annotation is not what it returns
            if isinstance(sub, ast.Return):
                returns.append(sub)
            stack.extend(ast.iter_child_nodes(sub))
        for ret in returns:
            if ret.value is None:
                _, end = self.edits.node_span(ret)
                self.edits.replace(end, end, f" {kept}()")
            elif _is_none(ret.value):
                self.edits.replace(*self.edits.node_span(ret.value), f"{kept}()")

    # -- visitors ----------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        classmethod_ = (
            self._class_name is not None
            and self._function_depth == 0
            and bool([*node.args.posonlyargs, *node.args.args])
            and any(_dotted(d) == "classmethod" for d in node.decorator_list)
        )
        for decorator in node.decorator_list:
            wrapped = self._functools_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator
            )
            if classmethod_ and _dotted(decorator) == "classmethod":
                self.edits.replace(*self.edits.node_span(decorator), "staticmethod")
            elif wrapped == "cached_property":
                # Computed once, read as an attribute: a property types the same.
                self.edits.replace(*self.edits.node_span(decorator), "property")
            elif wrapped in _FUNCTOOLS_DROPPED:
                start, end = self.edits.node_span(decorator)
                # Keep the lines of a decorator broken over several.
                lines = "\n" * self.edits.text(start, end).count("\n")
                self.edits.replace(start - 1, end, lines)
            elif (
                _dotted(decorator) in _DROPPED_FUNCTION_DECORATORS
                or _dotted(decorator) in self._dropped_decorators
            ):
                start, end = self.edits.node_span(decorator)
                self.edits.replace(start - 1, end, "")  # with its "@"
            else:
                self.visit(decorator)
        if classmethod_ and self._class_name is not None:
            self._rewrite_classmethod(node, self._class_name)
        self._rewrite_defaults(node.args, node.body)
        self._rewrite_arguments(node.args, skip_first=classmethod_, body=node.body)
        self._rewrite_returns(node)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        class_depth, self._class_depth = self._class_depth, 0
        class_name, self._class_name = self._class_name, None
        method_class = self._method_class
        if class_name is not None:
            self._method_class = class_name
        self._function_depth += 1
        self._list_scopes.append(
            (
                {
                    name: self._list_element(annotation)
                    for name, annotation in _parameters(node.args, skip_first=False)
                },
                _scope_bindings(node),
                len(self._local_bound),
            )
        )
        self._return_elements.append(
            None
            if node.returns is None or isinstance(node, ast.AsyncFunctionDef)
            else self._list_element(node.returns)
        )
        for stmt in node.body:
            self.visit(stmt)
        self._return_elements.pop()
        self._list_scopes.pop()
        self._function_depth -= 1
        self._class_depth = class_depth
        self._class_name = class_name
        self._method_class = method_class

    def _rewrite_classmethod(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str
    ) -> None:
        """Drop the ``cls`` parameter; ``cls`` in the body names the class."""
        first = [*node.args.posonlyargs, *node.args.args][0]
        start, end = self.edits.node_span(first)
        after = self.edits.text(end, len(self.edits.data))
        comma = after.find(",")
        closing = after.find(")")
        if 0 <= comma < closing:
            rest = after[comma + 1 :]
            end += len(after[: comma + 1].encode())
            end += len(rest[: len(rest) - len(rest.lstrip(" "))].encode())
        self.edits.replace(start, end, "")
        for name in _parameter_reads(node.body, first.arg):
            self.edits.replace(*self.edits.node_span(name), class_name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _dotted(target) in _DROPPED_CLASS_DECORATORS:
                start, end = self.edits.node_span(decorator)
                self.edits.replace(start - 1, end, "")  # with its "@"
            else:
                self.visit(decorator)
        members = self._rewrite_enum(node)
        if _is_named_tuple(node):
            # `@tuple` is on the line above (tuple_named_tuples).
            self.edits.replace(*self.edits.node_span(node.bases[0]), "")
        elif (
            members is None
            and self._class_depth == 0
            and self._function_depth == 0
            and node.name in self._exceptions
        ):
            base_start, base_end = self.edits.node_span(node.bases[0])
            self.edits.replace(
                base_start, base_end, f"Static[{self.edits.text(base_start, base_end)}]"
            )
        elif members is None:
            for base in node.bases:
                self.visit(base)
        class_name, self._class_name = self._class_name, node.name
        forwards, self._forwards = self._forwards, forwarded_stores(node)
        self._class_depth += 1
        for stmt in node.body:
            if members is None or stmt not in members:
                self.visit(stmt)
        self._class_depth -= 1
        self._class_name = class_name
        self._forwards = forwards

    def _rewrite_enum(self, node: ast.ClassDef) -> list[ast.stmt] | None:
        """Rewrite an Enum class for the ``enum`` shim; its members, or ``None``.

        ``None`` too for an Enum whose value type cannot be told from its
        members (values of different types, or not literals): it is left as
        written, and Codon reports it.
        """
        bases = [_dotted(b) or "" for b in node.bases]
        enum_base = next(
            (b for b in bases if b.rsplit(".", 1)[-1] in _ENUM_BASES), None
        )
        if enum_base is None or node.keywords:
            return None
        kind = enum_base.rsplit(".", 1)[-1]
        members: list[tuple[ast.Assign, str, ast.expr]] = []
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and not stmt.targets[0].id.startswith("_")
            ):
                members.append((stmt, stmt.targets[0].id, stmt.value))

        def is_auto(value: ast.expr) -> bool:
            return isinstance(value, ast.Call) and _dotted(value.func) in _AUTO

        value_type = _ENUM_VALUE_TYPES.get(kind) or next(
            (b for b in bases if b in _ENUM_MIXINS), None
        )
        if value_type is None:

            def literal_type(value: ast.expr) -> str | None:
                if is_auto(value):
                    return "int"
                if isinstance(value, ast.Constant):
                    name = type(value.value).__name__  # bool is not int here
                    return name if name in _ENUM_MIXINS else None
                return None

            kinds = {literal_type(v) for _, _, v in members}
            value_type = kinds.pop() if len(kinds) == 1 else None
            if value_type is None:
                return None
        flag = kind in {"Flag", "IntFlag"}
        last = 0
        seen_values: set[str] = set()
        listed: list[str] = []
        #: A flag's value of each member so far, for a composite
        #: (``RW = R | W``) to be worked out, and the bits of the listed ones.
        known: dict[str, int] = {}
        numbers = flag_member_values(node.body) if flag else {}
        mask: int | None = 0 if flag else None
        every = 0  # the bits of all the members, composites too
        for stmt, name, value in members:
            number: int | None = None
            if is_auto(value):
                if value_type == "str":
                    text = repr(name.lower())
                elif flag:
                    number = numbers[name]
                    text = str(number)
                else:
                    last += 1
                    text = str(last)
            else:
                if flag:
                    number = numbers[name]
                if number is not None:
                    text = str(number)
                else:
                    text = self.edits.text(*self.edits.node_span(value))
                if isinstance(value, ast.Constant) and isinstance(value.value, int):
                    last = value.value
            if number is not None:
                known[name] = number
                every |= max(number, 0)
            if flag and number is not None:
                # Python 3.11+ iterates a Flag's single-bit members, in
                # definition order: not an alias of one, not a composite
                # (``RW = R | W``), not zero.
                single = number > 0 and number & (number - 1) == 0
                if single and text not in seen_values:
                    listed.append(name)
                    assert mask is not None
                    mask |= number
            elif text not in seen_values or is_auto(value):
                # A member whose value repeats an earlier one is an alias,
                # which iterating over the enum skips.
                listed.append(name)
            seen_values.add(text)
            self.edits.replace(
                *self.edits.node_span(stmt),
                f"{name}: ClassVar[{node.name}] = {node.name}({name!r}, {text})",
            )
        if flag and len(known) != len(members):
            mask = None  # a value that is not worked out: `~` is left to Codon
        elif kind == "IntFlag":
            # An IntFlag (boundary KEEP) flips every bit below the highest one
            # of any member, composites included; a Flag only those of its
            # single-bit members.
            mask = (1 << every.bit_length()) - 1
        first_start = self.edits.node_span(node.bases[0])[0]
        last_end = self.edits.node_span(node.bases[-1])[1]
        # Only an int-valued IntEnum / IntFlag / (int, Enum) orders and
        # converts like its value, and a StrEnum / (str, Enum) equals its
        # string: the shim has a class for each, a plain Enum has neither.
        shim = enum_base
        if kind == "IntFlag" or (kind == "Flag" and "int" in bases):
            shim = "_acs_IntFlag"
        elif kind == "IntEnum" or "int" in bases:
            shim = "_acs_IntEnum"
        elif kind == "StrEnum" or "str" in bases:
            shim = "_acs_StrEnum"
        self.edits.replace(first_start, last_end, f"{shim}[{value_type}]")
        if self._class_depth == 0 and self._function_depth == 0:
            self.enums.append(
                _EnumInfo(
                    node.name,
                    value_type,
                    listed,
                    [n for _, n, _ in members],
                    mask,
                    ("int" if shim == "_acs_IntFlag" else "flag") if flag else None,
                )
            )
        return [stmt for stmt, _, _ in members]

    def _rewrite_tuple_default(self, value: ast.expr, annotation: ast.expr) -> bool:
        """Make a field's tuple display default a list of its sequence type.

        A field ``Sequence[X]`` / ``tuple[X, ...]`` is declared as a
        ``List[X]``, and its default ``("a", "b")`` would be a fixed-length
        Codon tuple (``Tuple[str, str]``), which is no ``List[X]``: it becomes
        ``List[X](["a", "b"])``, a list of the field's element type, and
        ``()`` becomes ``List[X]()``.  Only the brackets are replaced, so the
        elements are rewritten as anywhere else.  *value* is the default
        itself or a ``field(default=(...))`` call; ``False`` (and nothing
        replaced) when it is no parenthesized tuple of such a field.
        """
        call: ast.Call | None = None
        if isinstance(value, ast.Call) and _dotted(value.func) in _FIELD_FUNCTIONS:
            call = value
            value = next(
                (kw.value for kw in call.keywords if kw.arg == "default"), value
            )
        if not isinstance(value, ast.Tuple):
            return False
        if any(isinstance(e, ast.Starred) for e in value.elts):
            return False
        if _annotation_names(annotation) & (self._later_classes | self._type_vars):
            return False  # a type Codon cannot name here
        typed = _sequence_list(annotation)
        start, end = self.edits.node_span(value)
        if typed is None or self.edits.text(start, start + 1) != "(":
            return False
        if call is not None:
            # `field(default=(...))`: the default alone, as `_field_default`
            # makes it.
            call_start, call_end = self.edits.node_span(call)
            self.edits.replace(call_start, start, "")
            self.edits.replace(end, call_end, "")
        if not value.elts:
            self.edits.replace(start, end, f"{typed}()")
            return True
        self._list_display(value, typed)
        for element in value.elts:
            self.visit(element)
        return True

    def _list_display(self, value: ast.Tuple, typed: str) -> bool:
        """Make the tuple display *value* the list *typed* of its elements.

        ``("a", "b")`` becomes ``List[str](["a", "b"])``.  Only the brackets
        are replaced, so the caller visits the elements.  ``False`` (and
        nothing replaced) for a display with no brackets or a starred
        element.
        """
        start, end = self.edits.node_span(value)
        if self.edits.text(start, start + 1) != "(" or any(
            isinstance(e, ast.Starred) for e in value.elts
        ):
            return False
        self.edits.replace(start, start + 1, f"{typed}([")
        self.edits.replace(end - 1, end, "])")
        return True

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        class_level = self._class_depth > 0 and self._function_depth == 0
        value = node.value
        self._mark_list(value, node.annotation)
        factory = (
            _lambda_factory(value)
            if class_level and isinstance(value, ast.Call)
            else None
        )
        listed: ast.expr | None = None
        if factory is not None:
            self._mark_list(factory.body, node.annotation)
            if id(factory.body) in self._list_targets:
                # `field(default_factory=lambda: [Dbl()])` of a `list[Rule]`:
                # the default alone, `(_acs_list_into(List[Rule](), Dbl()))`.
                assert isinstance(value, ast.Call)
                call_start, call_end = self.edits.node_span(value)
                body_start, body_end = self.edits.node_span(factory.body)
                self.edits.replace(call_start, body_start, "(")
                self.edits.replace(body_end, call_end, ")")
                listed = factory.body
                value = None  # replaced as a whole
        if (
            class_level
            and value is not None
            and self._rewrite_tuple_default(value, node.annotation)
        ):
            value = None  # replaced as a whole
        elif (
            class_level
            and isinstance(value, ast.Call)
            and _dotted(value.func) in _FIELD_FUNCTIONS
        ):
            default = _field_default(value, self.edits, node.annotation)
            if default is None:
                self.edits.replace(
                    self.edits.node_span(node.annotation)[1],
                    self.edits.node_span(value)[1],
                    "",
                )
            else:
                self.edits.replace(*self.edits.node_span(value), default)
            value = None  # replaced as a whole
        typed = (
            self._dict_annotation_type(node.annotation)
            if isinstance(value, ast.Dict)
            and not class_level
            and (_mixed_dict(value, least=1) or self._own_class_values(value))
            else None
        )
        if typed is not None:
            # `x: dict[str, BaseAction] = {...}`: a dict of the annotated
            # type, which a display of the first value's type would not be.
            assert isinstance(value, ast.Dict)
            self._rewrite_dict(value, f"_acs_dict_into({typed}(), ")
            self.generic_visit(value)
            value = None
        if isinstance(node.target, ast.Name):
            self._note_list_local(
                node.target.id, self._list_element(node.annotation), node.target
            )
        converted = self._rewrite_annotation(node.annotation, class_level=class_level)
        if (
            converted is None
            and node.value is not None
            and (
                not class_level
                or (_dotted(node.annotation) or "").removeprefix("typing.") == "Final"
            )
        ):
            self.edits.replace(
                self.edits.node_span(node.target)[1],
                self.edits.node_span(node.value)[0],
                " = ",
            )
        self.visit(node.target)
        if value is not None:
            self.visit(value)
        if listed is not None:
            self.visit(listed)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None and node.cause is not None and _is_none(node.cause):
            # `raise X from None`, which Codon 0.19 cannot type: `raise X`.
            _, end = self.edits.node_span(node.exc)
            self.edits.replace(end, self.edits.node_span(node)[1], "")
        if isinstance(node.exc, ast.Name) and _is_builtin_exception(node.exc.id):
            # `raise NotImplementedError`, a class, which Python instantiates
            # and Codon 0.19 crashes on: `raise NotImplementedError()`.
            start, end = self.edits.node_span(node.exc)
            self.edits.replace(start, end, f"{node.exc.id}()")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._mark_list_arguments(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "extend"
            and len(node.args) == 1
        ):
            # `a.extend([Sub()])` with `a: list[Base]`.
            self._mark_added(self._list_scope_base(node.func.value), node.args[0])
        if isinstance(node.func, ast.Name) and node.func.id == "frozenset":
            # Codon 0.19 has no frozenset: build a set, which types the same.
            start, end = self.edits.node_span(node.func)
            self.edits.replace(start, end, "set")
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id == "dict"
            # A ``dict`` the code binds itself is not Codon's concern.
            and not self._shadowed(node.func.id)
            and not node.args
            and node.keywords
            and all(kw.arg is not None for kw in node.keywords)
        ):
            # `dict(a=1)`: Codon's dict takes no keywords (#469); the display
            # `{"a": 1}` is the same dict.
            start, end = self.edits.node_span(node)
            self.edits.replace(start, self.edits.node_span(node.keywords[0])[0], "{")
            for kw in node.keywords:
                kw_start = self.edits.node_span(kw)[0]
                value_start = self.edits.node_span(kw.value)[0]
                self.edits.replace(kw_start, value_start, f'"{kw.arg}": ')
            last = self.edits.node_span(node.keywords[-1].value)[1]
            self.edits.replace(last, end, "}")
        elif (
            isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "frozenset"
        ):
            # `frozenset[str]()`, an empty one typed by its element: a Set.
            start, end = self.edits.node_span(node.func.value)
            self.edits.replace(start, end, "Set")
        elif (
            _dotted(node.func) in _CAST_NAMES
            and len(node.args) == 2
            and not node.keywords
        ):
            # `cast(T, x)`: `cast(T, x)` of the shim, which returns a T, or
            # `(x)` when T cannot be named in Codon (a Callable, a class
            # defined further down).
            if self._rewrite_annotation(node.args[0], class_level=False) is None:
                start, end = self.edits.node_span(node)
                value_start, value_end = self.edits.node_span(node.args[1])
                self.edits.replace(start, value_start, "(")
                self.edits.replace(value_end, end, ")")
            self.visit(node.args[1])
            return
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in _RENAMED_BUILTINS
            and len(node.args) == 1
            and not node.keywords
        ):
            start, end = self.edits.node_span(node.func)
            self.edits.replace(start, end, _RENAMED_BUILTINS[node.func.id])
        elif (
            self._method_class is not None
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__new__"
            and isinstance(node.func.value, ast.Call)
            and _dotted(node.func.value.func) == "super"
            and not node.func.value.args
            and len(node.args) == 1
            and not node.keywords
        ):
            # `super().__new__(cls)`, in a singleton's `__new__`: Codon 0.19's
            # `object.__new__` takes no class.  The class's own `__new__()`
            # allocates the same instance.
            self.edits.replace(
                *self.edits.node_span(node), f"{self._method_class}.__new__()"
            )
            return
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        call = node.value
        if (
            isinstance(call, ast.Call)
            and _dotted(call.func) == "object.__setattr__"
            and len(call.args) == 3
            and not call.keywords
            and isinstance(call.args[1], ast.Constant)
            and isinstance(call.args[1].value, str)
            and call.args[1].value.isidentifier()
        ):
            # A frozen dataclass sets its fields with object.__setattr__,
            # which Codon 0.19 lacks: assign the attribute, which types the
            # same.
            target, _, value = call.args
            name = call.args[1].value
            start, end = self.edits.node_span(call)
            self.edits.replace(
                start,
                self.edits.node_span(value)[0],
                # In brackets, so the line breaks a long call kept stay inside.
                f"{self.edits.text(*self.edits.node_span(target))}.{name} = (",
            )
            self.edits.replace(self.edits.node_span(value)[1], end, ")")
            self.visit(value)
            return
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = node.value
        if (
            self._class_depth == 0
            and self._function_depth == 0
            and isinstance(value, ast.Subscript)
            and _dotted(value.value) in _CALLABLE_NAMES
        ):
            # A Callable alias, `GroundHeight = Callable[[float, float], R]`,
            # which Codon 0.19 reads as an expression and refuses ("expected
            # type expression"): None, on as many lines.  An annotation must
            # spell the callable out, which the check then reads.
            start, end = self.edits.node_span(value)
            newlines = "\n" * self.edits.text(start, end).count("\n")
            self.edits.replace(start, end, f"(None{newlines})" if newlines else "None")
            return
        if (
            self._class_depth == 0
            and self._function_depth == 0
            and isinstance(value, ast.Subscript)
            and _dotted(value.value) in _UNION_NAMES | _OPTIONAL_NAMES
        ):
            # A type alias, `Position = Union["Lanelet2Pose", MapPose]`: Codon
            # takes no string as a type, so a forward reference is unquoted
            # (Codon reads the TYPE_CHECKING imports, which define the name).
            self._unquote_alias_members(value.slice)
        for target in node.targets:
            if isinstance(target, ast.Name):
                base = None
                if isinstance(value, ast.List) and listable_display(value):
                    base = self._common_base(value.elts)
                self._note_list_local(target.id, base, target)
        literal = _literal_type(value)
        if self._class_depth == 0 and self._function_depth == 0 and literal:
            # A Literal alias, `Tristate = Literal["any", "yes", "no"]`: the
            # type of its values (`str`), which Codon's Literal is not.
            self.edits.replace(*self.edits.node_span(value), literal)
            return
        self.generic_visit(node)

    def _unquote_alias_members(self, node: ast.AST) -> None:
        """Unquote the forward references in a ``Union``/``Optional`` alias.

        A ``Literal[...]`` in it is no forward reference: it becomes the type
        of its values (``Optional[Literal["a", "b"]]``: ``Optional[str]``),
        or, of values of several types, is left as written.
        """
        if isinstance(node, ast.Subscript) and _dotted(node.value) in _LITERAL_NAMES:
            literal = _literal_type(node)
            if literal is not None:
                self.edits.replace(*self.edits.node_span(node), literal)
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            self.edits.replace(*self.edits.node_span(node), node.value)
            return
        for child in ast.iter_child_nodes(node):
            self._unquote_alias_members(child)

    def visit_Dict(self, node: ast.Dict) -> None:
        typed = self._dict_targets.pop(id(node), None) or self._common_dict_type(node)
        if typed is not None:
            # `{"a": Circle(), "b": Square()}`: a dict of the target type, or
            # of the nearest class the values share, as the list display is.
            self._rewrite_dict(node, f"_acs_dict_into({typed}(), ")
            self.generic_visit(node)
            return
        if _mixed_dict(node):
            # `{"turn": turn, "lc": lc}`: `{"turn": _acs_base(turn), ...}`.
            # Each edit starts before the value it wraps (after its key),
            # so an edit inside the value, at its first character, still
            # applies; the display stays Codon's.
            for key, value in zip(node.keys, node.values):
                if not isinstance(value, _BASE_VALUES):
                    continue
                assert key is not None
                key_end = self.edits.node_span(key)[1]
                value_start, value_end = self.edits.node_span(value)
                self.edits.replace(
                    key_end,
                    value_start,
                    self.edits.text(key_end, value_start) + "_acs_base(",
                )
                self.edits.replace(value_end, value_end, ")")
        self.generic_visit(node)

    def _dict_annotation_type(self, annotation: ast.AST | None) -> str | None:
        """``Dict[K, V]`` for a ``dict[K, V]`` / ``Mapping[K, V]`` annotation.

        Only of two plain names Codon can type with (not ``Any``, a
        ``TypeVar``, or a class defined further down unless in a function
        body); ``None`` otherwise.
        """
        if not (
            isinstance(annotation, ast.Subscript)
            and _dotted(annotation.value) in _DICT_ANNOTATIONS
            and isinstance(annotation.slice, ast.Tuple)
            and len(annotation.slice.elts) == 2
        ):
            return None
        names = []
        for element in annotation.slice.elts:
            if (
                not isinstance(element, ast.Name)
                or element.id in _UNEXPRESSIBLE
                or element.id in _CONCRETE_GENERICS | _ABSTRACT_GENERICS
                or element.id in self._type_vars
                or self._later(element.id)
            ):
                return None
            names.append("str" if element.id in _BYTES_NAMES else element.id)
        return f"Dict[{names[0]}, {names[1]}]"

    def _rewrite_dict(self, node: ast.Dict, opening: str) -> None:
        """``{k1: v1, k2: v2}``: ``<opening>(k1, v1), (k2, v2))``.

        Each item becomes a ``(key, value)`` tuple; the text between the
        items (commas, line breaks, comments, a parenthesised key's
        brackets) is kept, so every item stays on its line.  No edit starts
        where a key or a value starts, so an edit inside one still applies.
        """
        edits = self.edits
        start, end = edits.node_span(node)
        previous = start + 1  # after the `{`
        for i, (key, value) in enumerate(zip(node.keys, node.values)):
            assert key is not None
            key_start, key_end = edits.node_span(key)
            value_start, value_end = edits.node_span(value)
            if i == 0:
                edits.replace(
                    start, key_start, opening + edits.text(previous, key_start) + "("
                )
            else:
                edits.replace(
                    previous, key_start, ")" + edits.text(previous, key_start) + "("
                )
            edits.replace(
                key_end, value_start, _colon_to_comma(edits.text(key_end, value_start))
            )
            previous = value_end
        edits.replace(previous, end, ")" + edits.text(previous, end - 1) + ")")

    def visit_List(self, node: ast.List) -> None:
        if not isinstance(node.ctx, ast.Load):
            self.generic_visit(node)
            return
        element = self._list_targets.pop(id(node), None)
        if element is None and len(node.elts) >= 2 and listable_display(node):
            # `[Dbl(), Neg()]`: a list of the nearest class they share.
            element = self._common_base(node.elts)
        start, end = self.edits.node_span(node)
        if element is not None:
            # `_acs_list_into(List[B](), a, b)`: a list of B that each element
            # is put into, as `append()` does (Codon types a display by its
            # first element and does not make a list of a subclass one of
            # its base).
            self.edits.replace(start, start + 1, f"_acs_list_into(List[{element}](), ")
            self.edits.replace(end - 1, end, ")")
        elif len(node.elts) >= 2 and listable_display(node):
            self.edits.replace(start, start + 1, "_acs_list(")
            self.edits.replace(end - 1, end, ")")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self._return_elements and self._return_elements[-1] is not None:
            if isinstance(node.value, ast.List) and listable_display(node.value):
                self._list_targets[id(node.value)] = self._return_elements[-1]  # type: ignore[assignment]
        self.generic_visit(node)


def listable_display(node: ast.List) -> bool:
    """Whether the list display *node* is one ``_acs_list`` / ``_acs_list_into``
    makes: no starred element, and not all literals (which Codon types)."""
    return (
        bool(node.elts)
        and not any(isinstance(e, ast.Starred) for e in node.elts)
        and not all(isinstance(e, ast.Constant) for e in node.elts)
        and not all(isinstance(e, ast.JoinedStr) for e in node.elts)
        and not _same_class_attributes(node.elts)
    )


def _same_class_attributes(elements: list[ast.expr]) -> bool:
    """Whether *elements* are all members of one class (``Color.RED``, ...)."""
    if not all(isinstance(e, ast.Attribute) for e in elements):
        return False
    bases = {_dotted(e.value) for e in elements}  # type: ignore[attr-defined]
    base = bases.pop() if len(bases) == 1 else None
    return base is not None and base.rsplit(".", 1)[-1][:1].isupper()


def _lambda_factory(call: ast.Call) -> ast.Lambda | None:
    """The lambda of ``field(default_factory=lambda: ...)``, if that is *call*."""
    if _dotted(call.func) not in _FIELD_FUNCTIONS:
        return None
    for kw in call.keywords:
        if kw.arg == "default_factory" and isinstance(kw.value, ast.Lambda):
            return kw.value
    return None


def _parameters(
    args: ast.arguments, *, skip_first: bool
) -> list[tuple[str, ast.expr | None]]:
    """The parameters (name, annotation) of *args*, positional ones first."""
    positional = [*args.posonlyargs, *args.args]
    if skip_first:
        positional = positional[1:]
    return [(a.arg, a.annotation) for a in [*positional, *args.kwonlyargs]]


#: The dict display values ``_acs_base`` wraps: those that may be a
#: condition or an action (not a literal, whose type Codon's display may
#: still promote or make optional).
_BASE_VALUES = (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.IfExp)


def _mixed_dict(node: ast.Dict, least: int = 2) -> bool:
    """Whether the dict display *node* may mix classes, for ``_acs_base``.

    At least *least* items, no ``**`` unpacking, and values that are not all
    of one evident type (:func:`_evident_values`), which are left to Codon:
    it types them as Python does.
    """
    if len(node.values) < least or any(key is None for key in node.keys):
        return False
    return not _evident_values(node.values)


def _evident_values(values: list[ast.expr]) -> bool:
    """Whether *values* are evidently of one type.

    All literals, all f-strings, all calls of the same name
    (``Foo(1)``, ``Foo(2)``), or all attributes of the same class
    (``Direction.LEFT``, ``Direction.RIGHT``).
    """
    if all(isinstance(v, ast.Constant) for v in values):
        return True
    if all(isinstance(v, ast.JoinedStr) for v in values):
        return True
    if all(isinstance(v, ast.Call) for v in values):
        callees = {_dotted(v.func) for v in values}  # type: ignore[attr-defined]
        return len(callees) == 1 and None not in callees
    if all(isinstance(v, ast.Attribute) for v in values):
        bases = {_dotted(v.value) for v in values}  # type: ignore[attr-defined]
        base = bases.pop() if len(bases) == 1 else None
        return base is not None and base.rsplit(".", 1)[-1][:1].isupper()
    return False


def _colon_to_comma(text: str) -> str:
    """The text between a dict key and its value, its ``:`` a ``,``.

    Only brackets, blanks and comments surround the colon; a comment may
    hold a colon of its own.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        code, hash_, comment = line.partition("#")
        if ":" in code:
            lines[i] = code.replace(":", ",", 1) + hash_ + comment
            return "\n".join(lines)
    raise ValueError(f"no colon between a dict key and its value: {text!r}")


def _splittable(handler: ast.ExceptHandler) -> bool:
    """Whether *handler* catches a tuple of classes written out (``(A, B)``)."""
    kind = handler.type
    return (
        isinstance(kind, ast.Tuple)
        and len(kind.elts) > 0
        and not any(isinstance(e, ast.Starred) for e in kind.elts)
    )


_NAMED_TUPLE_BASES = {"NamedTuple", "typing.NamedTuple"}


def _is_named_tuple(node: ast.ClassDef) -> bool:
    """``class Pt(NamedTuple):``, the class syntax (the only base, no keyword)."""
    return (
        len(node.bases) == 1
        and not node.keywords
        and _dotted(node.bases[0]) in _NAMED_TUPLE_BASES
    )


def tuple_named_tuples(source: str, origin: list[int]) -> tuple[str, list[int]]:
    """Put ``@tuple`` above each ``class Pt(NamedTuple)``.

    Codon refuses a class deriving from ``NamedTuple`` ("reference classes
    cannot inherit tuple classes"); its own ``@tuple`` class is the same
    thing: immutable, built from its fields in order (with defaults), indexed,
    unpacked and iterated, with ``==``, ordering and hashing by value.  The
    base is dropped by the rewriter; the decorator is a new line, which maps
    to the class's line (*origin* is :func:`split_tuple_excepts`'s, extended
    the same way).
    """
    lines = source.encode().splitlines(keepends=True)
    origin = list(origin)
    classes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and _is_named_tuple(node)
    ]
    for node in sorted(classes, key=lambda n: n.lineno, reverse=True):
        # Above the first decorator, if the class has any.
        first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        indent = b" " * node.col_offset
        lines.insert(first - 1, indent + b"@tuple\n")
        origin.insert(first - 1, origin[node.lineno - 1])
    return b"".join(lines).decode(), origin


def split_tuple_excepts(source: str) -> tuple[str, list[int]]:
    """Split each ``except (A, B):`` into one handler per class, in order.

    Codon 0.19 takes one class per handler (a tuple is "expected type
    expression").  ``except (A, B) as e: body`` becomes ``except (A) as e:
    body`` followed by ``except (B) as e: body``, the same handler for each
    class in the tuple's order, before the handlers that followed it: Python
    tries a tuple's classes in order and the handlers in order, so the same
    exceptions reach the same body.  A tuple that is not written out (a name,
    ``*classes``) and the empty tuple, which catches nothing, are left as
    they are.

    The copies are new lines, inserted after the handler.  Returns the source
    and, for each of its lines, the line of *source* it comes from: a copy's
    lines map to the handler they copy, so an error in either is reported at
    the line the author wrote.
    """
    lines = source.encode().splitlines(keepends=True)
    origin = list(range(1, len(lines) + 1))
    while True:
        handlers = [
            handler
            for node in ast.walk(ast.parse(b"".join(lines)))
            if isinstance(node, ast.Try)
            for handler in node.handlers
            if _splittable(handler)
        ]
        if not handlers:
            return b"".join(lines).decode(), origin
        # The last one first: one nested in a handler's body is split before
        # that handler is copied, and nothing above it moves.
        handler = max(handlers, key=lambda h: (h.lineno, h.col_offset))
        kind = handler.type
        assert isinstance(kind, ast.Tuple)
        first, last = handler.lineno, handler.end_lineno or handler.lineno
        block = lines[first - 1 : last]
        starts = [0]
        for line in block:
            starts.append(starts[-1] + len(line))
        text = b"".join(block)
        start = starts[kind.lineno - first] + kind.col_offset
        end = starts[(kind.end_lineno or kind.lineno) - first] + (
            kind.end_col_offset or 0
        )
        # A tuple written over several lines keeps its line breaks, inside
        # the brackets, so each copy has as many lines as the handler.
        breaks = b"\n" * text[start:end].count(b"\n")
        copies = [
            text[:start]
            + b"("
            # The class as one line, with no comment (`(errors  # why\n.Error,
            # ...)` would comment the rest of the copy out).
            + ast.unparse(element).encode()
            + breaks
            + b")"
            + text[end:]
            for element in kind.elts
        ]
        lines[first - 1 : last] = b"".join(copies).splitlines(keepends=True)
        origin[first - 1 : last] = origin[first - 1 : last] * len(copies)


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
#: Built-ins that build a collection from an argument (``list(Color)``,
#: ``tuple(gen)``, ``dict(a=1)``).
_COLLECTION_CALLS = {"list", "tuple", "set", "dict", "frozenset", "sorted"}


def _walk_eager(node: ast.AST) -> Iterator[ast.AST]:
    """*node* and what it evaluates itself: a lambda's body is left out."""
    yield node
    if isinstance(node, ast.Lambda):
        return
    for child in ast.iter_child_nodes(node):
        yield from _walk_eager(child)


def _factory_needs_hoist(body: ast.expr, enum_names: Collection[str]) -> bool:
    """Whether a ``default_factory`` lambda body is one Codon compiles only
    in a function: a comprehension, a collection built from an argument, or
    an Enum class (iterated)."""
    for sub in ast.walk(body):
        if isinstance(sub, _COMPREHENSIONS):
            return True
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id in _COLLECTION_CALLS
            and (sub.args or sub.keywords)
        ):
            return True
        if isinstance(sub, ast.Name) and sub.id in enum_names:
            return True
    return False


def _class_body_hoists(
    tree: ast.Module, enum_names: Collection[str] = ()
) -> tuple[
    list[tuple[ast.ClassDef, ast.expr, str, ast.Lambda | None]],
    list[tuple[int, str]],
]:
    """The class-body expressions to move into module-level functions (#469).

    Codon realizes a class body in the class's scope, where it does not bind
    the target of a comprehension ("name '%_i_945' is not defined"), and
    crashes on an Enum iterated there.  A function has neither problem, and
    Python evaluates a class body once, so a call of a function is the same
    value.  Each item is (the class, the expression, its function name,
    the ``default_factory`` lambda it is the body of, if it is); the second
    value is what cannot be moved, as (line, message).
    """
    hoists: list[tuple[ast.ClassDef, ast.expr, str, ast.Lambda | None]] = []
    refused: list[tuple[int, str]] = []

    def visit(node: ast.ClassDef, outer: ast.ClassDef, names: tuple[str, ...]) -> None:
        """*node*, nested in classes called *names* (the first is *outer*)."""
        if any(
            (_dotted(b) or "").rsplit(".", 1)[-1] in _ENUM_BASES for b in node.bases
        ):
            return
        names = (*names, node.name)
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                # A class nested in a class (#482): its function is placed
                # above the outermost class, named after the whole path.
                visit(stmt, outer, names)
                continue
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
                continue
            value = stmt.value
            factory: ast.Lambda | None = None
            if isinstance(value, ast.Call) and _dotted(value.func) in _FIELD_FUNCTIONS:
                lam = next(
                    (
                        kw.value
                        for kw in value.keywords
                        if kw.arg == "default_factory"
                        and isinstance(kw.value, ast.Lambda)
                        and not kw.value.args.args
                        and not kw.value.args.posonlyargs
                        and not kw.value.args.kwonlyargs
                        and kw.value.args.vararg is None
                        and kw.value.args.kwarg is None
                    ),
                    None,
                )
                if lam is None or not _factory_needs_hoist(lam.body, enum_names):
                    continue
                expr: ast.expr = lam.body
                factory = lam
            elif any(isinstance(n, _COMPREHENSIONS) for n in _walk_eager(value)):
                expr = value
            else:
                continue
            reads = {
                n.id
                for n in ast.walk(expr)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            read = next((n for n in reversed(names) if n in reads), None)
            if read is not None:
                refused.append(
                    (
                        stmt.lineno,
                        f"this expression in the body of class {node.name} "
                        f"reads {read} itself, which Codon has not "
                        "defined while it runs the body, and it needs a "
                        "function of its own (a comprehension or an Enum "
                        "iteration cannot run in a class body): build it in "
                        "a module-level function that does not read the "
                        "class",
                    )
                )
                continue
            hoists.append(
                (outer, expr, f"_acs_hoist_{'_'.join(names)}_{stmt.lineno}", factory)
            )

    for top in tree.body:
        if isinstance(top, ast.ClassDef):
            visit(top, top, ())
    return hoists, refused


def class_body_refusals(source: str) -> list[tuple[int, str]]:
    """What :func:`hoist_class_bodies` and :func:`class_body_lambdas` cannot
    move or rewrite, and the fields Codon cannot type, as (line, message)."""
    tree = ast.parse(source)
    return [
        *_class_body_hoists(tree)[1],
        *lambda_default_refusals(source),
        *bare_collection_refusals(tree),
        *replace_refusals(tree),
    ]


#: The bare collection annotations Codon cannot infer the elements of, and the
#: parametrized spelling to write.
_BARE_COLLECTIONS = {
    "list": "list[X]",
    "dict": "dict[K, V]",
    "set": "set[X]",
    "frozenset": "frozenset[X]",
    "tuple": "tuple[X, ...] (or tuple[float, float])",
    "List": "List[X]",
    "Dict": "Dict[K, V]",
    "Set": "Set[X]",
    "FrozenSet": "FrozenSet[X]",
    "Tuple": "Tuple[X, ...] (or Tuple[float, float])",
    "typing.List": "List[X]",
    "typing.Dict": "Dict[K, V]",
    "typing.Set": "Set[X]",
    "typing.FrozenSet": "FrozenSet[X]",
    "typing.Tuple": "Tuple[X, ...] (or Tuple[float, float])",
}


def bare_collection_refusals(tree: ast.Module) -> list[tuple[int, str]]:
    """The class attributes annotated with a bare ``list``, ``dict``, ``set`` or
    ``tuple``, as (line, message) (#482).

    Codon types an attribute declared in a class body (a dataclass field, a
    NamedTuple field, a plain ``xs: list = []``, or a bare ``xs: list``
    declaration) from its annotation and cannot infer the elements of a bare
    one ("type of attribute 'xs' of object 'Cfg' cannot be inferred", or a
    bare ``Tuple`` that matches no tuple), whatever the default is.  An
    annotation on ``self.xs: list = []`` inside a method is fine.
    """
    refused: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(
                stmt.target, ast.Name
            ):
                continue
            annotation = stmt.annotation
            if isinstance(annotation, ast.Constant) and isinstance(
                annotation.value, str
            ):
                try:
                    annotation = ast.parse(annotation.value, mode="eval").body
                except SyntaxError:
                    continue
            spelled = _dotted(annotation)
            if spelled in _BARE_COLLECTIONS:
                refused.append(
                    (
                        stmt.lineno,
                        f"attribute `{ast.unparse(stmt.target)}` of class "
                        f"{node.name} is annotated with a bare `{spelled}`, "
                        "whose elements Codon cannot infer: write "
                        f"`{_BARE_COLLECTIONS[spelled]}`",
                    )
                )
    return refused


def hoist_class_bodies(
    source: str, origin: list[int], enum_names: Collection[str] = ()
) -> tuple[str, list[int]]:
    """Move class-body comprehensions and ``default_factory`` lambdas that
    need a function into module-level functions placed above their class
    (#469).

    ``SQ = [i * i for i in range(3)]`` becomes ``SQ = _acs_hoist_Q_2()`` and
    ``field(default_factory=lambda: [...])`` ``field(default_factory=
    _acs_hoist_D_5)``, with ``def _acs_hoist_...(): return (...)`` above the
    class (above its decorators).  The expression keeps its statement's
    line; the two new lines map to the line of the expression (*origin* is
    extended the same way as :func:`tuple_named_tuples`'s).  A function is
    typed from what it returns, so nothing is lost.  A class nested in classes
    (#482) has its function placed above the outermost class, named after the
    whole path (``_acs_hoist_Outer_Inner_5``); a class inside a function is not
    hoisted.
    """
    tree = ast.parse(source)
    hoists, _ = _class_body_hoists(tree, enum_names)
    if not hoists:
        return source, origin
    edits = _Edits(source.encode().splitlines(keepends=True))
    inserts: dict[int, list[tuple[str, int]]] = {}
    for cls, expr, name, factory in hoists:
        body = edits.text(*edits.node_span(expr))
        if factory is None:
            edits.replace(*edits.node_span(expr), f"{name}()")
        else:
            edits.replace(*edits.node_span(factory), name)
        first = min([cls.lineno, *(d.lineno for d in cls.decorator_list)])
        inserts.setdefault(first, []).append(
            (f"def {name}():\n    return ({body})\n", origin[expr.lineno - 1])
        )
    lines = edits.apply().encode().splitlines(keepends=True)
    origin = list(origin)
    for first, items in sorted(inserts.items(), reverse=True):
        for text, line in reversed(items):
            new = text.encode().splitlines(keepends=True)
            # The body may span lines: keep each of the function's lines
            # on the line its text came from.
            lines[first - 1 : first - 1] = new
            origin[first - 1 : first - 1] = [line] * len(new)
    return b"".join(lines).decode(), origin


class _ClassWalker(ast.NodeVisitor):
    """Each class of a module with the classes it is nested in, outermost first,
    and whether a function or lambda is around it."""

    def __init__(self) -> None:
        self.found: list[tuple[ast.ClassDef, tuple[ast.ClassDef, ...], bool]] = []
        self._chain: list[ast.ClassDef] = []
        self._functions = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.found.append((node, tuple(self._chain), self._functions > 0))
        self._chain.append(node)
        self.generic_visit(node)
        self._chain.pop()

    def _function(self, node: ast.AST) -> None:
        self._functions += 1
        self.generic_visit(node)
        self._functions -= 1

    visit_FunctionDef = _function  # noqa: N815
    visit_AsyncFunctionDef = _function  # noqa: N815
    visit_Lambda = _function  # noqa: N815


def _lambda_def_text(name: str, lam: ast.Lambda, body: str) -> str:
    """``def name(args): return (body)``, the lambda ``lam`` as a function."""
    return f"def {name}({ast.unparse(lam.args)}): return ({body})"


def _wrapped_lambda(value: ast.expr) -> tuple[str | None, ast.Lambda | None]:
    """(the wrapper, the lambda) of ``lambda ...``, ``staticmethod(lambda ...)``
    and ``classmethod(lambda ...)``; the wrapper is ``None`` for a bare lambda."""
    if isinstance(value, ast.Lambda):
        return None, value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in ("staticmethod", "classmethod")
        and len(value.args) == 1
        and not value.keywords
        and isinstance(value.args[0], ast.Lambda)
    ):
        return value.func.id, value.args[0]
    return None, None


def class_body_lambdas(source: str) -> tuple[str, list[int]]:
    """Turn the lambdas of class bodies into functions (#482).

    Python makes ``f = lambda self, v: v * 2`` in a class body a method, and
    Codon cannot realize it ("cannot typecheck 'class ...'"); a ``def`` is
    what Python means.  ``name = lambda args: body`` becomes ``def name(args):
    return (body)`` on its line.  A lambda whose first parameter is not
    ``self`` or ``cls`` (``key = lambda v: v * 2``, called as ``C.key(3)``)
    is a ``@staticmethod``: Codon would bind the instance to ``v`` and check
    ``v * 2`` on it.  ``staticmethod(lambda ...)`` and ``classmethod(lambda
    ...)`` become a ``@staticmethod`` / ``@classmethod`` def.

    A lambda default of a method (``def sort(self, key=lambda v: -v)``),
    which Codon cannot realize either, becomes a module-level function above
    the outermost class (``key=_acs_lambda_P_3_30``): Python evaluates a
    default once, when the class body runs, and the lambda reads only
    globals and its own parameters, as the function does.

    Returns the source and, for each of its lines, the line of *source* it
    comes from: the ``@staticmethod`` / ``@classmethod`` line and the lines
    of a hoisted function map to the line they stand for.
    """
    lines = source.encode().splitlines(keepends=True)
    origin = list(range(1, len(lines) + 1))
    tree = ast.parse(source)
    edits = _Edits(list(lines))
    decorators: dict[int, tuple[str, int]] = {}
    inserts: dict[int, list[tuple[str, int]]] = {}
    walker = _ClassWalker()
    walker.visit(tree)
    for cls, chain, in_function in walker.found:
        path = "_".join(c.name for c in (*chain, cls))
        for stmt in cls.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and _alone_on_its_line(stmt, edits.lines)
            ):
                wrapper, lam = _wrapped_lambda(stmt.value)
                if lam is None:
                    continue
                positional = [*lam.args.posonlyargs, *lam.args.args]
                if wrapper is None:
                    wrapper = (
                        None
                        if positional and positional[0].arg in ("self", "cls")
                        else "staticmethod"
                    )
                body = edits.text(*edits.node_span(lam.body))
                edits.replace(
                    *edits.node_span(stmt),
                    _lambda_def_text(stmt.targets[0].id, lam, body),
                )
                if wrapper is not None:
                    decorators[stmt.lineno] = (
                        " " * stmt.col_offset + f"@{wrapper}",
                        stmt.lineno,
                    )
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _hoist_lambda_defaults(
                    stmt, chain, cls, in_function, path, edits, inserts
                )
    lines = edits.apply().encode().splitlines(keepends=True)
    # New lines, last first: above a decorated statement, and above the
    # outermost class (above its decorators) for a hoisted function.
    new_lines: dict[int, list[tuple[str, int]]] = {}
    for at, item in decorators.items():
        new_lines.setdefault(at, []).append(item)
    for at, items in inserts.items():
        new_lines.setdefault(at, []).extend(items)
    for at in sorted(new_lines, reverse=True):
        for text, line in reversed(new_lines[at]):
            added = (text + "\n").encode().splitlines(keepends=True)
            lines[at - 1 : at - 1] = added
            origin[at - 1 : at - 1] = [line] * len(added)
    return b"".join(lines).decode(), origin


def _alone_on_its_line(stmt: ast.stmt, lines: list[bytes]) -> bool:
    """Whether *stmt* starts its line and ends it (no ``;`` after it, no
    ``class C: f = ...`` before it), so a ``def`` can take its place."""
    start = lines[stmt.lineno - 1][: stmt.col_offset]
    end = lines[(stmt.end_lineno or stmt.lineno) - 1][stmt.end_col_offset or 0 :]
    rest = end.strip()
    return not start.strip() and (not rest or rest.startswith(b"#"))


def _hoist_lambda_defaults(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    chain: tuple[ast.ClassDef, ...],
    cls: ast.ClassDef,
    in_function: bool,
    path: str,
    edits: _Edits,
    inserts: dict[int, list[tuple[str, int]]],
) -> None:
    """Replace each lambda default of the method *func* by a function above the
    outermost class (see :func:`class_body_lambdas`)."""
    outermost = chain[0] if chain else cls
    defaults = [
        d
        for d in (*func.args.defaults, *func.args.kw_defaults)
        if isinstance(d, ast.Lambda)
    ]
    for lam in defaults:
        literal = all(
            isinstance(d, ast.Constant) or d is None
            for d in (*lam.args.defaults, *lam.args.kw_defaults)
        )
        if in_function or outermost.col_offset or not literal:
            continue  # refused: lambda_default_refusals
        name = f"_acs_lambda_{path}_{lam.lineno}_{lam.col_offset}"
        body = edits.text(*edits.node_span(lam.body))
        edits.replace(*edits.node_span(lam), name)
        first = min([outermost.lineno, *(d.lineno for d in outermost.decorator_list)])
        inserts.setdefault(first, []).append(
            (_lambda_def_text(name, lam, body), lam.lineno)
        )


def lambda_default_refusals(source: str) -> list[tuple[int, str]]:
    """The lambda defaults of methods :func:`class_body_lambdas` cannot move,
    as (line, message)."""
    walker = _ClassWalker()
    walker.visit(ast.parse(source))
    refused: list[tuple[int, str]] = []
    for cls, chain, in_function in walker.found:
        outermost = chain[0] if chain else cls
        for func in cls.body:
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for lam in (*func.args.defaults, *func.args.kw_defaults):
                if not isinstance(lam, ast.Lambda):
                    continue
                literal = all(
                    isinstance(d, ast.Constant) or d is None
                    for d in (*lam.args.defaults, *lam.args.kw_defaults)
                )
                if in_function or outermost.col_offset or not literal:
                    refused.append(
                        (
                            lam.lineno,
                            f"the lambda default of method {func.name} cannot be "
                            "compiled by Codon, and the check can move it to a "
                            "module-level function only for a class at module "
                            "level (not in a function or a block) and a lambda "
                            "whose own defaults are literals: write a "
                            "module-level `def` and use its name as the default",
                        )
                    )
    return refused


def transform_source(
    source: str,
    dropped_decorators: Collection[str] = (),
    init_classes: Collection[str] = (),
    enum_classes: Collection[str] = (),
    file: str | None = None,
) -> str:
    """Rewrite *source* for Codon; the result starts with :data:`PRELUDE`.

    *dropped_decorators* names function decorators to drop besides
    ``@abstractmethod``: the library check passes the run-time-only wrappers
    :data:`.library.UNCALLED` lists, which Codon never compiles.
    *init_classes* names the classes defined elsewhere (another module, the
    model) that define an ``__init__``, their own or a base's
    (:func:`classes_with_init`): a class of *source* deriving from one
    without an ``__init__`` of its own inherits it (:func:`inherited_inits`).
    *file* is the module's path, bound to ``__file__`` (#515).
    """
    return transform_source_lines(
        source, dropped_decorators, init_classes, enum_classes, file
    )[0]


def transform_source_lines(
    source: str,
    dropped_decorators: Collection[str] = (),
    init_classes: Collection[str] = (),
    enum_classes: Collection[str] = (),
    file: str | None = None,
) -> tuple[str, list[int]]:
    """:func:`transform_source`, and the line of *source* each line comes from.

    The second value has one entry per line after :data:`PRELUDE`: every
    rewrite keeps a line on its line but :func:`split_tuple_excepts`, whose
    copies of a handler map to the lines they copy.  The constructors
    appended after the module (an Enum's lookup by value, an inherited
    ``__init__``) map to the line of their class.
    """
    if not source.endswith("\n"):
        source += "\n"
    # Valid Python Codon refuses or crashes on (#376): python_rewrites.py.
    from .python_rewrites import python_rewrites

    # Type-only names, stored exceptions, operator getters (#512-#517): before
    # the class-body lambdas, which hoist the lambdas it writes.
    from .typing_rewrites import copy_dunders, typing_rewrites

    source = typing_rewrites(source)
    # Class-body lambdas and method defaults become functions first, so the
    # rewrites see the function (#482); its extra lines map to their origin.
    source, early = class_body_lambdas(source)
    source, ordering = python_rewrites(source)
    source, origin = split_tuple_excepts(source)
    origin = [early[line - 1] for line in origin]
    source, origin = tuple_named_tuples(source, origin)
    source, origin = hoist_class_bodies(
        source,
        origin,
        {
            n.name
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef)
            and any(
                (_dotted(b) or "").rsplit(".", 1)[-1] in _ENUM_BASES for b in n.bases
            )
        }
        | set(enum_classes)
        | model_enum_names(),
    )
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    exceptions = _exception_classes(tree)
    rewriter = _Rewriter(edits, exceptions, dropped_decorators)
    rewriter.visit(tree)
    own = {e.name for e in rewriter.enums} | set(enum_classes)
    # The model's enums too (#437), unless the module defines a class of that
    # name which is no enum.
    other = _module_level_bindings(tree) - own
    enum_iteration(tree, edits, own | (model_enum_names() - other))
    repeated_imports(tree, edits)
    lower_replace_calls(tree, edits)
    lines = {
        node.name: origin[node.lineno - 1]
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    # An Enum's extension (its members list, its lookups and, for a Flag, its
    # operators) follows its class instead of the module: module-level and
    # class-body code such as ``RW = P.R | P.W`` or a comprehension over the
    # class reads them while Codon reads the module (#437, #469).  Its lines
    # come from the class's line.
    after_class: dict[int, tuple[str, str]] = {}
    at_end = []
    for name, block in _enum_lookups(rewriter.enums):
        end = next(
            (
                n.end_lineno
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == name
            ),
            None,
        )
        if end is None:  # nested in something: after the module
            at_end.append((name, block))
        else:
            after_class[end] = (name, block)
    appended = [
        *at_end,
        *_inherited_init_extensions(_inherited_bases(tree, init_classes, exceptions)),
        *_dataclass_dunders(tree, exceptions),
        *_dataclass_replacers(tree, exceptions),
        *copy_dunders(tree, exceptions),
        *ordering,
    ]
    origin = list(origin)
    for end, (name, block) in sorted(after_class.items(), reverse=True):
        at = edits.starts[end] if end < len(edits.starts) else len(edits.data)
        edits.replace(at, at, block[1:])
        origin[end:end] = [lines[name]] * (block.count("\n") - 1)
    # Python sets a module's `__file__`, which Codon 0.19 does not (#515): it
    # is bound on the prelude's last line, before any code of the module and
    # with no line moved.
    prelude = PRELUDE if file is None else _with_file(PRELUDE, file)
    text = prelude + edits.apply()
    for name, block in appended:
        text += block
        origin += [lines[name]] * block.count("\n")
    return text, origin


def _with_file(prelude: str, file: str) -> str:
    """*prelude*, its last line also binding ``__file__`` to *file*."""
    return f"{prelude[:-1]}; __file__: str = {file!r}\n"


def _import_bindings(
    node: ast.Import | ast.ImportFrom,
) -> list[tuple[str, tuple[object, ...]]]:
    """The (bound name, source) of each name an import statement binds.

    Two imports bind the same thing when their sources are equal: the module
    (and its relative level) and the imported name, or the module for
    ``import a.b as c``.
    """
    bindings: list[tuple[str, tuple[object, ...]]] = []
    for alias in node.names:
        if isinstance(node, ast.ImportFrom):
            source: tuple[object, ...] = (node.level, node.module, alias.name)
            bound = alias.asname or alias.name
        else:
            source = (alias.name,)
            bound = alias.asname or alias.name.partition(".")[0]
            if alias.asname is None and "." in alias.name:
                # `import a.b` binds `a` and imports `a.b`: another `import
                # a.c` is neither a repeat of it nor a rebinding.
                bound = alias.name
        bindings.append((bound, source))
    return bindings


class _ModuleScope(ast.NodeVisitor):
    """The names a module-level statement reads and (re)binds when it runs.

    A function body, a lambda body and the bindings of a class body run later
    or in a scope of their own: they neither read nor bind module names here.
    Over-approximates (a comprehension target counts as a binding), which only
    makes the import analysis keep more imports.
    """

    def __init__(self) -> None:
        self.loads: set[str] = set()
        self.stores: set[str] = set()
        self._depth = 0

    def _store(self, name: str) -> None:
        if self._depth == 0:
            self.stores.add(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        else:
            self._store(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for name, _ in _import_bindings(node):
            self._store(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for name, _ in _import_bindings(node):
            self._store(name)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._store(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._store(node.name)
        for expr in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expr)
        self._depth += 1
        for statement in node.body:
            self.visit(statement)
        self._depth -= 1


@dataclass
class _Bound:
    source: tuple[object, ...]
    node: ast.Import | ast.ImportFrom
    index: int
    used: bool = False


def _import_plan(
    tree: ast.Module,
) -> tuple[set[tuple[int, int]], list[tuple[int, str]]]:
    """The module-level imports to drop, and the rebindings that cannot be.

    The drops are ``(id(statement), index of the name)``: an import that
    repeats an identical earlier one, and an import whose name a later import
    from elsewhere rebinds before anything has read it (the later one wins in
    Python, so dropping the earlier changes nothing).  What is left is a name
    rebound after it was read, reported as ``(line, message)``.
    """
    drops: set[tuple[int, int]] = set()
    problems: list[tuple[int, str]] = []
    bound: dict[str, _Bound] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            scope = _ModuleScope()
            scope.visit(node)
            for name in scope.loads & bound.keys():
                bound[name].used = True
            for name in scope.stores:
                for key in [k for k in bound if k == name or k.startswith(name + ".")]:
                    del bound[key]
            continue
        for index, (name, source) in enumerate(_import_bindings(node)):
            if name == "*":
                continue
            earlier = bound.get(name)
            if earlier is not None and earlier.source == source:
                drops.add((id(node), index))
                continue
            if earlier is not None:
                if earlier.used:
                    problems.append(
                        (
                            node.lineno,
                            f"`{name}` is imported again from somewhere "
                            f"else (first on line {earlier.node.lineno}) after "
                            "it was used, which Codon 0.19 cannot compile: "
                            "import each name once, under its own name "
                            "(`import ... as other` to tell two apart)",
                        )
                    )
                else:
                    drops.add((id(earlier.node), earlier.index))
            bound[name] = _Bound(source, node, index)
    return drops, problems


def _imported_names(stmt: ast.stmt) -> set[str]:
    """The names a top-level import statement binds, also in an ``if``."""
    if isinstance(stmt, ast.If):
        return {
            name
            for inner in [*stmt.body, *stmt.orelse]
            for name in _imported_names(inner)
        }
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {
            (alias.asname or alias.name).split(".")[0]
            for alias in stmt.names
            if alias.name != "*"
        }
    return set()


def _is_functools_import(node: ast.ImportFrom) -> bool:
    return node.module == "functools" and node.level == 0


def repeated_imports(tree: ast.Module, edits: _Edits) -> None:
    """Drop the module-level imports Codon 0.19 cannot compile twice.

    Python accepts ``from m import A`` twice (once in the import header, once
    further down), and so does Codon 0.19 in the module it compiles first; in
    any other module the repeat leaves the name undefined, so that a use of it
    fails the whole function with a bare "cannot typecheck".  The same goes for
    ``from a import X`` followed by ``from b import X``.  The earlier import
    of a rebound name and the repeat of an identical one are removed on their
    lines: a statement of such names becomes ``pass``, one with other names
    keeps those.  A name rebound after it was read stays
    (:func:`conflicting_imports`).
    """
    drops, _ = _import_plan(tree)
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        stubbed = isinstance(node, ast.ImportFrom) and _is_functools_import(node)
        kept = [
            a
            for i, a in enumerate(node.names)
            if (id(node), i) not in drops
            and not (stubbed and a.name in FUNCTOOLS_NAMES)
        ]
        if len(kept) == len(node.names):
            continue
        if not kept:
            text = "pass"
        else:
            names = ", ".join(_alias(a.name, a.asname) for a in kept)
            if isinstance(node, ast.ImportFrom):
                text = f"from {'.' * node.level}{node.module or ''} import {names}"
            else:
                text = f"import {names}"
        # An import broken over several lines keeps them: later lines stay put.
        start, end = edits.node_span(node)
        edits.replace(start, end, text + "\n" * edits.text(start, end).count("\n"))


def conflicting_imports(source: str) -> list[tuple[int, str]]:
    """(line, message) for each module-level name rebound after it was used.

    ``from a import X`` followed by ``from b import X`` is valid Python (the
    later import wins), but Codon 0.19 leaves the name undefined in any module
    it does not compile first.  :func:`repeated_imports` drops the earlier
    import when nothing read the name in between; when something did, the
    rebinding has no single reading and is reported here.
    """
    if not source.endswith("\n"):
        source += "\n"
    return _import_plan(ast.parse(source))[1]


@dataclass
class _EnumInfo:
    """A rewritten Enum class: its value type, iterated members, all names."""

    name: str
    value_type: str
    members: list[str]
    names: list[str]
    #: A Flag's bits of its members (what ``~`` flips), when all are known.
    mask: int | None = None
    #: ``"flag"`` for a ``Flag``, ``"int"`` for an ``IntFlag`` (whose operators
    #: take an int too), ``None`` for any other enum.
    flag: str | None = None


_FLAG_OPERATORS: dict[type, Callable[[int, int], int]] = {
    ast.BitOr: lambda a, b: a | b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitXor: lambda a, b: a ^ b,
    ast.LShift: lambda a, b: a << b if 0 <= b < 64 else 0,
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
}


def flag_member_values(body: list[ast.stmt]) -> dict[str, int | None]:
    """The int of each member of a Flag class body (``None`` if not worked out).

    A member ``auto()`` is the power of two above the highest bit so far, as
    in Python 3.11+; any other value is worked out by :func:`_flag_value`.
    """
    known: dict[str, int] = {}
    out: dict[str, int | None] = {}
    for stmt in body:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and not stmt.targets[0].id.startswith("_")
        ):
            continue
        name = stmt.targets[0].id
        value = stmt.value
        if isinstance(value, ast.Call) and _dotted(value.func) in _AUTO:
            number: int | None = 1 << max([0, *known.values()]).bit_length()
        else:
            number = _flag_value(value, known)
        out[name] = number
        if number is not None:
            known[name] = number
    return out


def _flag_value(value: ast.expr, known: Mapping[str, int]) -> int | None:
    """The int a Flag member's value works out to, or ``None``.

    A literal, an earlier member of the class and the bit operators of them
    (``RW = R | W``, ``ALL = ~0``, ``HIGH = 1 << 4``); what the class body
    computes in Python, where an earlier member is still an int.
    """
    if isinstance(value, ast.Constant):
        number = value.value
        return (
            number if isinstance(number, int) and not isinstance(number, bool) else None
        )
    if isinstance(value, ast.Name):
        return known.get(value.id)
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.Invert, ast.USub)):
        operand = _flag_value(value.operand, known)
        if operand is None:
            return None
        return ~operand if isinstance(value.op, ast.Invert) else -operand
    if isinstance(value, ast.BinOp) and type(value.op) in _FLAG_OPERATORS:
        left = _flag_value(value.left, known)
        right = _flag_value(value.right, known)
        if left is None or right is None:
            return None
        return _FLAG_OPERATORS[type(value.op)](left, right)
    return None


#: Builtins whose argument may be an enum class, which they iterate over.
_ITERATING_BUILTINS = {
    "list",
    "tuple",
    "set",
    "frozenset",
    "sorted",
    "len",
    "reversed",
    "enumerate",
    "iter",
    "min",
    "max",
    "sum",
    "any",
    "all",
    "zip",
    "map",
    "filter",
}


@functools.lru_cache(maxsize=None)
def model_enum_names() -> frozenset[str]:
    """The names of the framework's enums in the model (``typecheck/codon``).

    A model enum is a ``@tuple`` class that declares ``_acs_members()`` (what
    iterating over it reads) and ``_acs_by_name()``; the names come from the
    model files, so a new enum is iterable as soon as it declares them.
    """
    root = Path(__file__).resolve().parent / "codon" / "autoware_carla_scenario"
    names = set()
    for path in sorted(root.rglob("*.codon")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and any(
                isinstance(stmt, ast.FunctionDef) and stmt.name == "_acs_members"
                for stmt in node.body
            ):
                names.add(node.name)
    return frozenset(names)


def enum_class_names(classes: Mapping[str, ast.ClassDef]) -> set[str]:
    """The names of the Enum classes among *classes* (by their bases).

    What another module's :func:`transform_source_lines` is given as
    *enum_classes*: it iterates over, and looks up by name in, these too.
    """
    return {
        name
        for name, node in classes.items()
        if any((_dotted(b) or "").rsplit(".", 1)[-1] in _ENUM_BASES for b in node.bases)
    }


def _module_level_bindings(tree: ast.Module) -> set[str]:
    """The names *tree* binds at its own level other than by importing them
    from the framework: an assignment, a loop or ``with`` target, a ``def``,
    a ``class``, any other import, a ``global`` declaration (#471).  A name
    the module binds so is not the model's enum of that name.  The bodies of
    functions, lambdas and classes are other scopes."""
    names: set[str] = set()
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            # `global X` anywhere in it binds the module's X.
            for inner in ast.walk(node):
                if isinstance(inner, ast.Global):
                    names.update(inner.names)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            framework = not node.level and (node.module or "").split(".")[0] == (
                "autoware_carla_scenario"
            )
            for alias in node.names:
                if not framework or alias.asname not in (None, alias.name):
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _parameter_reads(body: list[ast.stmt], name: str) -> list[ast.Name]:
    """The uses of the name *name* in *body* (a function's) that are the
    function's own: not those of a nested function or lambda with a parameter
    of that name, or a comprehension with a loop variable of it (#481)."""
    found: list[ast.Name] = []
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            declared = {
                a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)
            }
            declared.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
            if name in declared:
                # Its defaults and decorators are still read in this scope.
                stack.extend(d for d in (*args.defaults, *args.kw_defaults) if d)
                stack.extend(getattr(node, "decorator_list", []))
                continue
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            targets = {
                n.id
                for g in node.generators
                for n in ast.walk(g.target)
                if isinstance(n, ast.Name)
            }
            if name in targets:
                # Its loop variable is its own; only the first iterable is
                # read in this scope.
                stack.append(node.generators[0].iter)
                continue
        if isinstance(node, ast.Name) and node.id == name:
            found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _enum_alias_rebound(
    function: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> bool:
    """Whether *function* binds *name* other than as its parameter: an
    assignment, a loop, ``with`` or comprehension target, an import, a nested
    ``def`` or ``class``, ``global``/``nonlocal`` (#481).  Conservative: a
    binding anywhere inside it, a nested scope's too, counts."""
    body = ast.Module(body=function.body, type_ignores=[])
    if name in _scope_bindings(body):
        return True
    for child in ast.walk(body):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            if child.id == name:
                return True
        elif isinstance(child, (ast.Global, ast.Nonlocal)) and name in child.names:
            return True
    return False


class _EnumIteration(ast.NodeVisitor):
    """Rewrite the iteration over, and the lookup by name in, an Enum class.

    ``for m in Mode`` / ``list(Mode)`` / ``len(Mode)`` / ``x in Mode`` read
    ``Mode._acs_members()``, and ``Mode[name]`` is ``Mode._acs_by_name(name)``
    (:func:`_enum_lookups`).  Only a bare name of an Enum class is rewritten,
    in place, so no line moves.

    Inside the Enum's own methods the class has other spellings (#481): the
    first parameter of a ``@classmethod`` (``cls``), and, in an instance
    method, ``type(self)`` and ``self.__class__``.  They read the same, unless
    the function rebinds the parameter, or a nested function or lambda
    declares one of the same name.
    """

    #: What a name stands for: the Enum class itself (``cls``) or an instance
    #: of it (``self``).
    _CLASS = "class"
    _INSTANCE = "instance"

    def __init__(self, edits: _Edits, enums: Collection[str]) -> None:
        self.edits = edits
        self.enums = enums
        #: The names bound inside each enclosing function or lambda: a
        #: parameter or local called like an Enum class is not that class.
        #: With, for a method of an Enum, its first parameter as (the Enum,
        #: what it is).
        self._bound: list[tuple[set[str], dict[str, tuple[str, str]]]] = []
        #: The Enum whose body is being visited, for a function directly in it.
        self._owner: str | None = None
        self._module_bound: set[str] = set()

    def visit_Module(self, node: ast.Module) -> None:
        self._module_bound = _scope_bindings(node)
        self.generic_visit(node)

    def _resolve(self, name: str) -> tuple[str, str] | None:
        """The Enum (and whether the class or an instance) *name* stands for."""
        for bound, aliases in reversed(self._bound):
            if name in aliases:
                return aliases[name]
            if name in bound:
                return None
        return (name, self._CLASS) if name in self.enums else None

    def _enum_of(self, node: ast.AST) -> str | None:
        """The Enum class *node* is: its name, or an alias of it in one of
        its methods (``cls``, ``type(self)``, ``self.__class__``)."""
        if isinstance(node, ast.Name):
            resolved = self._resolve(node.id)
            return resolved[0] if resolved and resolved[1] == self._CLASS else None
        instance = None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and "type" not in self._module_bound
            and not any("type" in bound for bound, _ in self._bound)
        ):
            instance = node.args[0].id
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "__class__"
            and isinstance(node.value, ast.Name)
        ):
            instance = node.value.id
        if instance is None:
            return None
        resolved = self._resolve(instance)
        return resolved[0] if resolved and resolved[1] == self._INSTANCE else None

    def _members(self, node: ast.AST) -> None:
        enum = self._enum_of(node)
        if enum is not None:
            # `cls` is already the class's name (`_Rewriter._rewrite_classmethod`).
            self.edits.supersede(*self.edits.node_span(node), f"{enum}._acs_members()")

    def _method_alias(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, bound: set[str]
    ) -> dict[str, tuple[str, str]]:
        """The first parameter of a method of the Enum being visited, as an
        alias, unless it is rebound or the Enum's name is (#481)."""
        owner = self._owner
        positional = [*node.args.posonlyargs, *node.args.args]
        if owner is None or not positional:
            return {}
        decorators = {_dotted(d) for d in node.decorator_list}
        if "staticmethod" in decorators:
            return {}
        kind = self._CLASS if "classmethod" in decorators else self._INSTANCE
        param = positional[0].arg
        if (
            _enum_alias_rebound(node, param)
            or owner in bound
            or self._resolve(owner) != (owner, self._CLASS)
        ):
            return {}
        return {param: (owner, kind)}

    def _scope(self, node: ast.AST, args: ast.arguments) -> None:
        bound = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        bound.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                bound.add(child.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(child.name)
        aliases = {}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            aliases = self._method_alias(node, bound)
        self._owner = None  # a function nested in the method is no method
        self._bound.append((bound, aliases))
        for attr, value in ast.iter_fields(node):
            if attr == "returns":
                continue  # an annotation (`Mode[int]`), not an expression
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, ast.AST):
                    self.visit(item)
        self._bound.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        owner = self._owner
        is_enum = node.name in self.enums and any(
            (_dotted(b) or "").rsplit(".", 1)[-1] in _ENUM_BASES for b in node.bases
        )
        for stmt in node.body:
            self._owner = node.name if is_enum else None
            self.visit(stmt)
        for expr in (*node.decorator_list, *node.bases):
            self._owner = None
            self.visit(expr)
        self._owner = owner

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        owner = self._owner
        self._scope(node, node.args)
        self._owner = owner

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        owner, self._owner = self._owner, None
        self._scope(node, node.args)
        self._owner = owner

    def visit_arg(self, node: ast.arg) -> None:
        pass  # an annotation is not an expression to evaluate

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._members(node.iter)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._members(node.iter)
        self.generic_visit(node)

    def visit_Starred(self, node: ast.Starred) -> None:
        self._members(node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _ITERATING_BUILTINS:
            for arg in node.args:
                self._members(arg)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                self._members(right)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        value = node.value
        enum = self._enum_of(value)
        if enum is not None and isinstance(node.ctx, ast.Load):
            value_start, value_end = self.edits.node_span(value)
            slice_start, slice_end = self.edits.node_span(node.slice)
            _, end = self.edits.node_span(node)
            if not isinstance(value, ast.Name):  # `type(self)`, `self.__class__`
                self.edits.replace(value_start, value_end, enum)
            self.edits.replace(value_end, slice_start, "._acs_by_name(")
            self.edits.replace(slice_end, end, ")")
        self.generic_visit(node)


def enum_iteration(tree: ast.Module, edits: _Edits, enums: Collection[str]) -> None:
    """Apply :class:`_EnumIteration` to *tree* for the Enum classes *enums*."""
    if enums:
        _EnumIteration(edits, enums).visit(tree)


def _flag_operators(info: _EnumInfo) -> str:
    """The methods ``|``, ``&``, ``^``, ``~`` and ``in`` of a Flag class (#437).

    Each returns the class, built from the combined int (any int is a value of
    the constructor by value, so a combination of members is representable
    though no member).  An ``IntFlag`` takes an int on either side too.  ``in``
    is true when every bit of the other is set.  ``~`` flips the bits of the
    members (Python's complement), and is left out when a member's value is not
    worked out.
    """
    name = info.name
    out = ""
    for method, symbol in (("or", "|"), ("and", "&"), ("xor", "^")):
        out += (
            f"\n    def __{method}__(self, other: {name}) -> {name}:\n"
            f"        return {name}(self._value_ {symbol} other._value_)\n"
        )
        if info.flag == "int":
            out += (
                f"\n    def __{method}__(self, other: int) -> {name}:\n"
                f"        return {name}(self._value_ {symbol} other)\n"
                f"\n    def __r{method}__(self, other: int) -> {name}:\n"
                f"        return {name}(other {symbol} self._value_)\n"
            )
    out += (
        f"\n    def __contains__(self, other: {name}) -> bool:\n"
        f"        return (self._value_ & other._value_) == other._value_\n"
    )
    if info.mask is not None:
        out += (
            f"\n    def __invert__(self) -> {name}:\n"
            f"        return {name}({info.mask} & ~self._value_)\n"
        )
    return out


def _enum_lookups(enums: list[_EnumInfo]) -> list[tuple[str, str]]:
    """A constructor by value (``Color("red")``) for each rewritten Enum class.

    Also its members as a list (``Color._acs_members()``, what iterating over
    the class reads) and a lookup by name (``Color._acs_by_name("RED")``,
    what ``Color["RED"]`` reads; ``KeyError`` for an unknown name).

    Codon gives a class without an ``__init__`` one that takes each field
    (``Color("RED", "red")``, what a member is built with); the lookup by
    value is added to it with ``@extend``, after the module, so no line of
    the source moves.  It is typed only: the ``Color`` it makes is not the
    member, which nothing the check compiles needs, since nothing runs.
    Each is (the class, the lines appended).
    """
    out = []
    for info in enums:
        name = info.name
        listing = ", ".join(f"{name}.{m}" for m in info.members)
        lookups = "".join(
            f"        if name == {n!r}:\n            return {name}.{n}\n"
            for n in info.names
        )
        operators = _flag_operators(info) if info.flag else ""
        out.append(
            (
                name,
                (
                    f"\n\n@extend\nclass {name}:\n"
                    f"    def __init__(self, value: {info.value_type}):\n"
                    f'        self._name_ = ""\n'
                    f"        self._value_ = value\n"
                    f"\n    @staticmethod\n"
                    f"    def _acs_members() -> List[{name}]:\n"
                    f"        return [{listing}]\n"
                    f"\n    @staticmethod\n"
                    f"    def _acs_by_name(name: str) -> {name}:\n"
                    f"{lookups}"
                    f"        raise KeyError(name)\n"
                    f"{operators}"
                ),
            )
        )
    return out


#: Bases of a dataclass that add no field and no generated method.
_PLAIN_BASES = {"object", "ABC", "abc.ABC"}
_NOT_FIELDS = {"ClassVar", "typing.ClassVar", "InitVar", "dataclasses.InitVar"}
_NOT_FIELDS |= {"KW_ONLY", "dataclasses.KW_ONLY"}


def _dataclass_options(node: ast.ClassDef) -> dict[str, bool] | None:
    """The flags of *node*'s ``@dataclass`` (Python's defaults where not
    written as a literal), or ``None`` for a class that is no dataclass."""
    options = {"eq": True, "order": False, "frozen": False, "unsafe_hash": False}
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _dotted(target) not in ("dataclass", "dataclasses.dataclass"):
            continue
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if (
                    keyword.arg in options
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, bool)
                ):
                    options[keyword.arg] = keyword.value.value
        return options
    return None


def _compared_fields(node: ast.ClassDef) -> list[str]:
    """The fields *node* declares that ``==``, ordering and hashing compare:
    not a ``ClassVar``, an ``InitVar``, ``KW_ONLY`` or ``field(compare=False)``."""
    names = []
    for stmt in node.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        annotation = stmt.annotation
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        if isinstance(annotation, ast.Subscript):
            annotation = annotation.value
        if _dotted(annotation) in _NOT_FIELDS:
            continue
        value = stmt.value
        if isinstance(value, ast.Call) and _dotted(value.func) in _FIELD_FUNCTIONS:
            if any(
                k.arg == "compare"
                and isinstance(k.value, ast.Constant)
                and k.value.value is False
                for k in value.keywords
            ):
                continue
        names.append(stmt.target.id)
    return names


def _defined_methods(node: ast.ClassDef) -> set[str]:
    return {
        m.name
        for m in node.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _dataclass_dunders(tree: ast.Module, exceptions: set[str]) -> list[tuple[str, str]]:
    """The methods Python's ``@dataclass`` generates that Codon does not, per class.

    The check drops ``@dataclass`` (Codon's constructor takes the fields), so
    ``==``, ``!=`` and ``in``, hashing (``frozen=True``, ``unsafe_hash=True``)
    and ordering (``order=True``) are missing.  Each is appended after the
    module with ``@extend``, comparing the fields in order (the fields of a
    dataclass base first), as Python does: ``eq=False`` and a method the
    class defines itself are respected.  A top-level dataclass with a base the
    module does not define (a framework dataclass, ``Generic``) is left as it
    is: its fields are not known here.  Each is (the class, the lines).
    """
    fields: dict[str, list[str]] = {}
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        options = _dataclass_options(node)
        if options is None or node.name in exceptions or node.keywords:
            continue
        inherited: list[str] | None = []
        for base in node.bases:
            name = _dotted(base)
            if name in fields:
                inherited = [*(inherited or []), *fields[name]]
            elif name not in _PLAIN_BASES:
                inherited = None
                break
        if inherited is None:
            continue
        names = [*inherited, *(n for n in _compared_fields(node) if n not in inherited)]
        fields[node.name] = names
        own = _defined_methods(node)
        lines: list[str] = []

        def method(
            name: str,
            result: str,
            body: str,
            lines: list[str] = lines,
            own: set[str] = own,
        ) -> None:
            if name in own:
                return
            other = "" if name == "__hash__" else f", other: {node.name}"
            lines.append(
                f"    def {name}(self{other}) -> {result}:\n        return {body}\n"
            )

        def tuples(op: str) -> str:
            if not names:
                return "True" if "=" in op else "False"
            mine = ", ".join(f"self.{n}" for n in names)
            theirs = ", ".join(f"other.{n}" for n in names)
            return f"({mine},) {op} ({theirs},)"

        if options["eq"]:
            same = " and ".join(f"self.{n} == other.{n}" for n in names) or "True"
            method("__eq__", "bool", same)
            method("__ne__", "bool", "not self.__eq__(other)")
            if options["frozen"] or options["unsafe_hash"]:
                method(
                    "__hash__",
                    "int",
                    f"hash(({', '.join(f'self.{n}' for n in names)},))"
                    if names
                    else "0",
                )
        elif options["unsafe_hash"]:
            method(
                "__hash__",
                "int",
                f"hash(({', '.join(f'self.{n}' for n in names)},))" if names else "0",
            )
        if options["order"]:
            for dunder, op in (
                ("__lt__", "<"),
                ("__le__", "<="),
                ("__gt__", ">"),
                ("__ge__", ">="),
            ):
                method(dunder, "bool", tuples(op))
        if lines:
            out.append(
                (node.name, f"\n\n@extend\nclass {node.name}:\n" + "".join(lines))
            )
    return out


# ---------------------------------------------------------------------------
# dataclasses.replace (#487)
# ---------------------------------------------------------------------------


def _replace_spellings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """The names ``dataclasses.replace`` is bound to, and those ``dataclasses``
    is: ``from dataclasses import replace as r`` and ``import dataclasses as d``.

    A name another import, a ``def``, a ``class`` or an assignment of the module
    also binds is left out: it is not known to be ``dataclasses``'s.
    """
    functions: set[str] = set()
    modules: set[str] = set()
    other: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound = a.asname or a.name.split(".")[0]
                if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
                    if a.name == "replace":
                        functions.add(bound)
                        continue
                elif isinstance(node, ast.Import) and a.name == "dataclasses":
                    modules.add(bound)
                    continue
                other.add(bound)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            other.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            other.update(
                n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)
            )
    return functions - other, modules - other


class _ReplaceCalls(ShadowScopes):
    """The ``dataclasses.replace`` calls of a module, where the name is not
    rebound by a parameter, a local or a nested definition (``def f(replace)``)."""

    def __init__(self, functions: set[str], modules: set[str]) -> None:
        super().__init__()
        self.functions = functions
        self.modules = modules
        self.calls: list[ast.Call] = []
        self.imported_locally: set[str] = set()

    def _rebound(self, name: str) -> bool:
        # The import of ``replace`` inside a function binds it there too.
        if name in self.imported_locally:
            return False
        return any(name in bound for bound, _is_class in self._local_bound)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            hit = func.id in self.functions and not self._rebound(func.id)
        else:
            hit = (
                isinstance(func, ast.Attribute)
                and func.attr == "replace"
                and isinstance(func.value, ast.Name)
                and func.value.id in self.modules
                and not self._rebound(func.value.id)
            )
        if hit:
            self.calls.append(node)
        self.generic_visit(node)


def _replace_calls(tree: ast.Module) -> list[ast.Call]:
    functions, modules = _replace_spellings(tree)
    if not functions and not modules:
        return []
    visitor = _ReplaceCalls(functions, modules)
    visitor.enter_module(tree)
    top = {id(n) for n in tree.body}
    visitor.imported_locally = {
        a.asname or a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) not in top
        for a in n.names
    } & (functions | modules)
    visitor.visit(tree)
    return visitor.calls


def _replace_problem(call: ast.Call) -> str | None:
    """Why *call* cannot be lowered, or ``None`` when it can."""
    if len(call.args) != 1 or isinstance(call.args[0], ast.Starred):
        return (
            "dataclasses.replace takes the object as its one positional "
            "argument and the fields to change as keywords: write "
            "replace(obj, field=value)"
        )
    if any(kw.arg is None for kw in call.keywords):
        return (
            "dataclasses.replace(obj, **changes) cannot be checked, as the "
            "fields it changes are not known: write each as a keyword "
            "(replace(obj, speed=1.0)), or build the dataclass yourself"
        )
    return None


def replace_refusals(tree: ast.Module) -> list[tuple[int, str]]:
    """The ``dataclasses.replace`` calls :func:`lower_replace_calls` cannot
    lower, as (line, message)."""
    out = []
    for call in _replace_calls(tree):
        problem = _replace_problem(call)
        if problem is not None:
            out.append((call.lineno, problem))
    return out


def replace_method(field_name: str) -> str:
    """The name of the method that copies a dataclass with one field changed."""
    return f"_acs_with__{field_name}"


def lower_replace_calls(tree: ast.Module, edits: _Edits) -> None:
    """``replace(obj, a=x, b=y)`` becomes ``replace(obj)._acs_with__a(x)
    ._acs_with__b(y)``: the copy is made by the typed methods
    :func:`_dataclass_replacers` appends, which Codon checks.

    The value goes by position: a field a base declares has the method on the
    base and on each subclass (so that it returns the subclass), which makes
    it virtual, and Codon 0.19 cannot pass a keyword through the dispatch
    thunk ("'a' is an invalid keyword argument for class_thunk_dispatch()",
    #494).

    Only the text between the arguments is edited, so the values keep the
    rewrites made inside them and every call stays on its lines.
    """
    for call in _replace_calls(tree):
        if _replace_problem(call) is not None or not call.keywords:
            continue
        previous = call.args[0]
        for keyword in call.keywords:
            start = edits.node_span(previous)[1]
            end = edits.node_span(keyword.value)[0]
            comma = edits.text(start, end).find(",")
            if comma < 0:
                break
            edits.replace(start + comma, end, f").{replace_method(str(keyword.arg))}(")
            previous = keyword.value


def _replaceable_fields(node: ast.ClassDef) -> list[tuple[str, ast.expr, bool]] | None:
    """The fields *node* declares, as (name, annotation, whether ``__init__``
    takes it); ``None`` for a class ``replace`` cannot copy (an ``InitVar``)."""
    out = []
    for stmt in node.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        annotation = stmt.annotation
        text = annotation.value if isinstance(annotation, ast.Constant) else None
        if isinstance(text, str):
            annotation = ast.parse(text, mode="eval").body
        head = annotation.value if isinstance(annotation, ast.Subscript) else annotation
        kind = _dotted(head)
        if kind in ("InitVar", "dataclasses.InitVar"):
            return None
        if kind in _NOT_FIELDS:
            continue
        init = True
        value = stmt.value
        if isinstance(value, ast.Call) and _dotted(value.func) in _FIELD_FUNCTIONS:
            init = not any(
                k.arg == "init"
                and isinstance(k.value, ast.Constant)
                and k.value.value is False
                for k in value.keywords
            )
        out.append((stmt.target.id, annotation, init))
    return out


def _dataclass_replacers(
    tree: ast.Module, exceptions: set[str]
) -> list[tuple[str, str]]:
    """The ``_acs_with__<field>`` methods of each dataclass, per class.

    Each returns the object's copy built through the constructor with that
    field changed (``Plan(points=self.points, best=best)``), typed as the
    field is declared, so ``replace(plan, best=None)`` and ``replace(plan,
    speed="fast")`` are checked and Codon needs no ``**changes``.  A field
    declared ``init=False`` gets none, as Python's ``replace`` refuses it (the
    hint of the "no attribute" error says so).  A class with a base the module does not
    define, an ``__init__`` of its own or an ``InitVar`` has none: ``replace``
    of it is refused ("no attribute").  Each is (the class, the lines).
    """
    fields: dict[str, list[tuple[str, ast.expr, bool]]] = {}
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if _dataclass_options(node) is None or node.name in exceptions or node.keywords:
            continue
        inherited: list[tuple[str, ast.expr, bool]] | None = []
        for base in node.bases:
            name = _dotted(base)
            if name in fields:
                inherited = [*(inherited or []), *fields[name]]
            elif name not in _PLAIN_BASES:
                inherited = None
                break
        own_fields = _replaceable_fields(node)
        if inherited is None or own_fields is None:
            continue
        merged = {n: (n, a, i) for n, a, i in [*inherited, *own_fields]}
        listed = list(merged.values())
        fields[node.name] = listed
        if "__init__" in _defined_methods(node):
            continue
        lines = []
        for name, annotation, init in listed:
            method = replace_method(name)
            if not init:
                continue  # Python refuses to replace it too
            # As a parameter does: an abstract collection (Mapping, Sequence)
            # is left generic, and the constructor checks what is passed.
            typed = codon_annotation(annotation)
            param = f"{name}: {typed}" if typed is not None else name
            call = ", ".join(
                f"{n}={n}" if n == name else f"{n}=self.{n}" for n, _, i in listed if i
            )
            lines.append(
                f"    def {method}(self, {param}) -> {node.name}:\n"
                f"        return {node.name}({call})\n"
            )
        if lines:
            out.append(
                (node.name, f"\n\n@extend\nclass {node.name}:\n" + "".join(lines))
            )
    return out


def _inherited_init_extensions(
    bases: list[tuple[str, str | None]],
) -> list[tuple[str, str]]:
    """The ``__init__`` each class of *bases* inherits, appended with ``@extend``.

    Codon 0.19 inherits no ``__init__``: a class without one gets a
    constructor taking each of its fields, which refuses what the base's
    takes ("'config' is an invalid keyword argument", #201).  The one
    appended passes everything on to the base's, which checks it: through
    ``super()``, or, when the ``__init__`` is a later base's (``class
    S(Mixin, Base)``), through that base by name, as Codon's ``super()`` is
    the first base's.
    """
    return [
        (
            name,
            (
                f"\n\n@extend\nclass {name}:\n"
                "    def __init__(self, *args, **kwargs):\n"
                + (
                    "        super().__init__(*args, **kwargs)\n"
                    if base is None
                    else f"        {base}.__init__(self, *args, **kwargs)\n"
                )
            ),
        )
        for name, base in bases
    ]


def _defines_init(node: ast.ClassDef) -> bool:
    return any(
        isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "__init__"
        for m in node.body
    )


def _is_dataclass_node(node: ast.ClassDef) -> bool:
    return any(
        _dotted(d.func if isinstance(d, ast.Call) else d)
        in ("dataclass", "dataclasses.dataclass")
        for d in node.decorator_list
    )


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """``Parent`` -> ``Base`` for each ``from m import Base as Parent``."""
    return {
        alias.asname: alias.name
        for stmt in tree.body
        if isinstance(stmt, ast.ImportFrom)
        for alias in stmt.names
        if alias.asname is not None and alias.asname != alias.name
    }


def classes_with_init(
    classes: Mapping[str, ast.ClassDef], known: Collection[str] = ()
) -> set[str]:
    """The classes of *classes* that have an explicit ``__init__``, inherited too.

    A class defines one when its body does, or when it is no dataclass (which
    has its constructor generated) and a base does: a base among *classes*,
    or one *known* names (the model's classes that define one).  Bases are
    looked up by their last name (``base.Base``: ``Base``).  The result
    includes *known*.
    """
    known = set(known)
    memo: dict[str, bool] = {}

    def has(name: str, seen: frozenset[str]) -> bool:
        if name in memo:
            return memo[name]
        node = classes.get(name)
        if node is None:
            return name in known
        if name in seen:
            return False
        if _defines_init(node):
            found = True
        elif _is_dataclass_node(node):
            found = False
        else:
            found = any(
                has((_dotted(b) or "").rsplit(".", 1)[-1], seen | {name})
                for b in node.bases
            )
        memo[name] = found
        return found

    return {name for name in classes if has(name, frozenset())} | known


def inherited_inits(
    tree: ast.Module,
    init_classes: Collection[str] = (),
    exceptions: Collection[str] = (),
) -> list[str]:
    """The top-level classes of *tree* that inherit an ``__init__`` (#201).

    A class without an ``__init__`` of its own whose base defines one: a base
    of the module (its own ``__init__`` or one it inherits), or one
    *init_classes* names (looked up by the name it is defined with, through
    ``from m import Base as Parent``).  A dataclass has its constructor
    generated, and an Enum or an exception (*exceptions*, derived
    statically) is constructed as Codon builds it, so none of them is one.
    """
    return [name for name, _ in _inherited_bases(tree, init_classes, exceptions)]


def _inherited_bases(
    tree: ast.Module,
    init_classes: Collection[str] = (),
    exceptions: Collection[str] = (),
) -> list[tuple[str, str | None]]:
    """:func:`inherited_inits`, each with the base whose ``__init__`` it is.

    ``None`` when that is the first base (``super()`` reaches it), else the
    base as the source spells it: Python takes the ``__init__`` of the first
    base that has one (``class S(Mixin, Base)``: ``Base``'s).
    """
    local = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    aliases = _import_aliases(tree)
    known = {
        alias
        for alias, name in aliases.items()
        if name in init_classes and alias not in local
    } | (set(init_classes) - set(local))
    with_init = classes_with_init(local, known)
    out: list[tuple[str, str | None]] = []
    for name, node in local.items():
        if _defines_init(node) or _is_dataclass_node(node) or name in exceptions:
            continue
        bases = [(_dotted(b) or "").rsplit(".", 1)[-1] for b in node.bases]
        if any(b in _ENUM_BASES for b in bases) or node.keywords:
            continue
        for index, base in enumerate(bases):
            if base in with_init:
                out.append(
                    (name, None if index == 0 else ast.unparse(node.bases[index]))
                )
                break
    return out


def _exception_classes(tree: ast.Module) -> set[str]:
    """Top-level classes of *tree* deriving (only) from an exception."""
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or len(node.bases) != 1:
            continue
        base = (_dotted(node.bases[0]) or "").rsplit(".", 1)[-1]
        if (
            base in _BUILTIN_EXCEPTIONS
            or base in out
            or base.endswith(_EXCEPTION_SUFFIXES)
        ):
            out.add(node.name)
    return out


# ---------------------------------------------------------------------------
# Imports of the framework, in a module of the framework (the library check)
# ---------------------------------------------------------------------------


def _absolute_module(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    if not node.level:
        return node.module or ""
    package = module if is_package else module.rpartition(".")[0]
    base = package.split(".")
    base = base[: len(base) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


def _alias(name: str, asname: str | None) -> str:
    return f"{name} as {asname}" if asname and asname != name else name


def _import_as(dest: str, asname: str) -> str:
    parent, _, last = dest.rpartition(".")
    if not parent:
        return f"import {dest} as {asname}"
    return f"from {parent} import {_alias(last, asname)}"


def uncalled_nodes(
    tree: ast.Module, uncalled: Collection[str]
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The definitions in *tree* of the functions *uncalled* names.

    *uncalled* names a function as ``function`` or ``Class.method``; a name
    with no definition in *tree* is ignored.
    """
    wanted = set(uncalled)
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in wanted:
                out.append(node)
        elif isinstance(node, ast.ClassDef):
            out += [
                m
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and f"{node.name}.{m.name}" in wanted
            ]
    return out


def redirect_imports(
    source: str,
    module: str,
    is_package: bool,
    target: Callable[[str], str | None],
    is_module: Callable[[str], bool],
    package: str = "autoware_carla_scenario",
    uncalled: Collection[str] = (),
) -> tuple[str, list[tuple[int, str]]]:
    """Point every import of *package* in *source* at the module standing in for it.

    *source* is module *module* of the framework, compiled by the library
    check.  ``target(name)`` is the module of the check's workspace that
    stands in for the framework module *name*: its own (rewritten) source, or
    its Codon model; ``None`` when there is neither.  ``is_module(name)``
    tells a submodule (``from . import frames``) from a name.  Each import
    statement keeps its line; ``from a import b, c`` that imports both a
    submodule and names becomes two statements on it.  An import inside a
    function *uncalled* names (as :func:`uncalled_nodes` takes them) is left
    as written: Codon compiles no import in a function nothing calls, so a
    module may import one there that has nothing to stand in for it.

    Returns:
        The rewritten source, and (line, message) for each import that could
        not be pointed anywhere (left as written).
    """
    if not source.endswith("\n"):
        source += "\n"
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    problems: list[tuple[int, str]] = []
    skipped = {
        id(node)
        for function in uncalled_nodes(tree, uncalled)
        for node in ast.walk(function)
    }

    def inside(name: str) -> bool:
        return name == package or name.startswith(package + ".")

    def missing(name: str) -> str:
        return (
            f"{name} is neither checked (typecheck/library.py) nor modelled "
            "(typecheck/codon/), so a checked module cannot import it"
        )

    for node in ast.walk(tree):
        if id(node) in skipped:
            continue
        pieces: list[str] = []
        failed = False
        if isinstance(node, ast.ImportFrom):
            name = _absolute_module(node, module, is_package)
            if not inside(name):
                continue
            plain: list[str] = []
            for alias in node.names:
                if alias.name == "*":
                    problems.append(
                        (node.lineno, f"`from {name} import *`: import each name")
                    )
                    failed = True
                elif is_module(f"{name}.{alias.name}"):
                    dest = target(f"{name}.{alias.name}")
                    if dest is None:
                        problems.append((node.lineno, missing(f"{name}.{alias.name}")))
                        failed = True
                    else:
                        pieces.append(_import_as(dest, alias.asname or alias.name))
                else:
                    plain.append(_alias(alias.name, alias.asname))
            if plain:
                dest = target(name)
                if dest is None:
                    problems.append((node.lineno, missing(name)))
                    failed = True
                else:
                    pieces.insert(0, f"from {dest} import {', '.join(plain)}")
        elif isinstance(node, ast.Import):
            if not any(inside(alias.name) for alias in node.names):
                continue
            for alias in node.names:
                if not inside(alias.name):
                    pieces.append(f"import {_alias(alias.name, alias.asname)}")
                    continue
                dest = target(alias.name)
                if alias.asname is None:
                    problems.append(
                        (
                            node.lineno,
                            f"`import {alias.name}`: import it with a name "
                            f"(`from ... import ...` or `import {alias.name} as ...`)",
                        )
                    )
                    failed = True
                elif dest is None:
                    problems.append((node.lineno, missing(alias.name)))
                    failed = True
                else:
                    pieces.append(_import_as(dest, alias.asname))
        else:
            continue
        if not failed:
            edits.replace(*edits.node_span(node), "; ".join(pieces))
    return edits.apply(), problems


def _relative_from(package: str, target: str) -> str:
    """*target* as a relative module name in the package *package*
    (``pkg.sub``, ``pkg.helpers`` -> ``..helpers``)."""
    mine, theirs = package.split("."), target.split(".")
    shared = 0
    while shared < min(len(mine), len(theirs)) and mine[shared] == theirs[shared]:
        shared += 1
    return "." * (len(mine) - shared + 1) + ".".join(theirs[shared:])


def literal_all(tree: ast.Module) -> list[str] | None:
    """The names of ``__all__`` when *tree* sets it only as a literal list or
    tuple of strings (``=``, ``: list[str] =`` or ``+=``); ``None`` when it
    sets none or builds it any other way (``from X import *`` then binds
    every public name, as Codon does)."""
    names: list[str] | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        elif isinstance(stmt, ast.AugAssign):
            targets, value = [stmt.target], stmt.value
        else:
            if any(
                isinstance(n, ast.Name) and n.id == "__all__" for n in ast.walk(stmt)
            ):
                return None  # __all__.append(...), a conditional, a function...
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        if not isinstance(value, (ast.List, ast.Tuple)) or not all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
        ):
            return None
        literal = [
            e.value
            for e in value.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        if isinstance(stmt, ast.AugAssign):
            if names is None:
                return None
            names += literal
        else:
            names = literal
    return names


def relative_module_imports(
    source: str,
    module: str,
    is_package: bool,
    is_own_module: Callable[[str], bool],
    star_names: Callable[[str], list[str] | None] | None = None,
) -> tuple[str, list[tuple[int, str]]]:
    """Import each of the scenario's own modules imported as a module the way
    Codon 0.19 resolves it (#310).

    Codon resolves ``from X import name`` only as a name ``X/__init__``
    defines, never as the submodule ``X.name``, and an absolute ``import
    a.b`` only relative to the importing file's directory (or as a ``.codon``
    file of ``CODON_PATH``, which a scenario's modules are not).  So in
    ``pkg/scenario.py`` (*module* ``pkg.scenario``) ``from . import
    helpers``, ``from pkg import helpers`` and ``import pkg.helpers as
    helpers`` all fail, and ``import helpers as helpers`` is the form that
    works.  Each import of a module *is_own_module* names (one of the
    scenario's own, ``pkg.helpers``) is rewritten into that form: the
    module's name relative to the importing module's package (``from .sub
    import mod`` -> ``import sub.mod as mod``).  ``from a import b, c`` that
    imports both a module and names becomes two statements on its line, so
    no line moves.

    ``from helpers import *`` of an own module whose ``__all__`` is a literal
    (``star_names(module)`` returns its names, see :func:`literal_all`) is
    rewritten into an import of those names, as Python binds only them (#506).

    A module that is not inside the importing module's package (``from ..
    import helpers`` in ``pkg/sub/scenario.py``) has no such form, and
    neither has a module imported without a name (``import pkg.helpers``,
    which binds ``pkg``): such an import is left as written, and reported.

    Returns:
        The rewritten source, and (line, message) for each import that could
        not be rewritten.
    """
    if not source.endswith("\n"):
        source += "\n"
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    package = module if is_package else module.rpartition(".")[0]
    problems: list[tuple[int, str]] = []

    def form(target: str, asname: str, written: str, line: int) -> str | None:
        prefix = package + "." if package else ""
        if target.startswith(prefix):
            return f"import {target[len(prefix) :]} as {asname}"
        problems.append(
            (
                line,
                f"`{written}` imports the module {target} as a module, which "
                f"Codon 0.19 resolves only inside the importing module's own "
                f"package ({package}): import the names instead "
                f"(`from {_relative_from(package, target)} import name`)",
            )
        )
        return None

    for node in ast.walk(tree):
        pieces: list[str] = []
        failed = False
        if isinstance(node, ast.ImportFrom):
            name = _absolute_module(node, module, is_package)
            plain: list[str] = []
            aliases = list(node.names)
            exports = (
                star_names(name)
                if star_names is not None
                and any(a.name == "*" for a in aliases)
                and is_own_module(name)
                else None
            )
            if exports is not None:
                aliases = [a for a in aliases if a.name != "*"] + [
                    ast.alias(name=n) for n in exports
                ]
            for alias in aliases:
                target = f"{name}.{alias.name}" if name else alias.name
                if alias.name == "*" or not is_own_module(target):
                    plain.append(_alias(alias.name, alias.asname))
                    continue
                written = (
                    f"from {'.' * node.level}{node.module or ''} import "
                    f"{_alias(alias.name, alias.asname)}"
                )
                piece = form(target, alias.asname or alias.name, written, node.lineno)
                if piece is None:
                    failed = True
                else:
                    pieces.append(piece)
            if not pieces and not failed and exports is None:
                continue
            if plain:
                pieces.insert(
                    0,
                    f"from {'.' * node.level}{node.module or ''} import "
                    f"{', '.join(plain)}",
                )
            elif not pieces:
                pieces.append("pass")  # an empty __all__ binds nothing
        elif isinstance(node, ast.Import):
            if not any(is_own_module(alias.name) for alias in node.names):
                continue
            for alias in node.names:
                written = f"import {_alias(alias.name, alias.asname)}"
                if not is_own_module(alias.name):
                    pieces.append(written)
                elif alias.asname is None and "." in alias.name:
                    last = alias.name.rpartition(".")[2]
                    problems.append(
                        (
                            node.lineno,
                            f"`{written}` binds {alias.name.split('.')[0]}, "
                            f"which Codon 0.19 cannot resolve from {module}: "
                            f"import the module with a name (`import "
                            f"{alias.name} as {last}`) or import its names",
                        )
                    )
                    failed = True
                else:
                    piece = form(
                        alias.name, alias.asname or alias.name, written, node.lineno
                    )
                    if piece is None:
                        failed = True
                    else:
                        pieces.append(piece)
        else:
            continue
        if not failed:
            edits.replace(*edits.node_span(node), "; ".join(pieces))
    return edits.apply(), problems


# ---------------------------------------------------------------------------
# Attribute declarations
# ---------------------------------------------------------------------------


def class_declarations(tree: ast.Module) -> dict[str, tuple[list[str], set[str]]]:
    """Each top-level class of *tree*: its base names and the names it declares.

    A class declares its class-level annotations and assignments, its
    methods (properties included) and, for an exception, nothing it needs.
    """
    out: dict[str, tuple[list[str], set[str]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        declared: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                declared.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                declared.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declared.add(stmt.name)
        bases = [b for b in (_dotted(base) for base in node.bases) if b is not None]
        out[node.name] = (bases, declared)
    return out


def _suggest_type(value: ast.AST | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
        return _suggest_type(value.values[-1])
    if isinstance(value, ast.IfExp):
        return _suggest_type(value.orelse) or _suggest_type(value.body)
    if isinstance(value, ast.Call):
        name = _dotted(value.func)
        if name is not None and name[:1].isupper():
            return name
        if name in {"list", "dict", "set"}:
            return name
    if isinstance(value, ast.Constant) and value.value is not None:
        return type(value.value).__name__
    if isinstance(value, (ast.List, ast.ListComp)):
        return "list[...]"
    if isinstance(value, (ast.Dict, ast.DictComp)):
        return "dict[...]"
    return None


def undeclared_attributes(
    tree: ast.Module,
    declared_elsewhere: dict[str, set[str]],
    *,
    also: Collection[str] = (),
    uncalled: Collection[str] = (),
) -> list[UndeclaredAttribute]:
    """Attributes assigned on ``self`` that a class of *tree* does not declare.

    Only classes in an inheritance hierarchy are checked, with a base class
    (other than ``object``) or with a subclass in *tree*, and classes with a
    class-level annotation (#218): Codon types a plain class's attributes
    from its ``__init__`` only while it declares none, and once it declares
    one, an attribute it does not declare is not there.  A name counts as
    declared when the class, a base class in *tree*, or a base class named in
    *declared_elsewhere* (class name -> declared names; the Codon model's
    classes, and classes of other checked modules) declares it.  A base
    imported under another name (``from .base import Base as Parent``) is
    looked up by the name it is defined with.  The classes named in *also*
    are checked whatever they declare, and the methods *uncalled* names
    (``Class.method``, which Codon never compiles) are not read.
    """
    classes = class_declarations(tree)
    subclassed = {b.rsplit(".", 1)[-1] for bases, _ in classes.values() for b in bases}
    aliases = {
        alias.asname: alias.name
        for stmt in tree.body
        if isinstance(stmt, ast.ImportFrom)
        for alias in stmt.names
        if alias.asname is not None and alias.asname != alias.name
    }

    def declared(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
        if name in seen:
            return set()
        if name in classes:
            bases, own = classes[name]
            out = set(own)
            for base in bases:
                out |= declared(base.rsplit(".", 1)[-1], seen | {name})
            return out
        return set(declared_elsewhere.get(aliases.get(name, name), set()))

    found: list[UndeclaredAttribute] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases, _ = classes[node.name]
        annotated = any(
            isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            for stmt in node.body
        )
        if (
            not [b for b in bases if b != "object"]
            and node.name not in subclassed
            and not annotated
            and node.name not in also
        ):
            continue
        names = declared(node.name)
        reported: set[str] = set()
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if f"{node.name}.{method.name}" in uncalled:
                continue
            for stmt in ast.walk(method):
                targets: list[tuple[ast.AST, ast.AST | None, str | None]] = []
                if isinstance(stmt, ast.Assign):
                    targets = [(t, stmt.value, None) for t in stmt.targets]
                elif isinstance(stmt, ast.AnnAssign):
                    annotation = codon_annotation(stmt.annotation, class_level=True)
                    targets = [(stmt.target, stmt.value, annotation)]
                elif isinstance(stmt, ast.AugAssign):
                    targets = [(stmt.target, None, None)]
                for target, value, annotation in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr not in names
                        and target.attr not in reported
                    ):
                        reported.add(target.attr)
                        found.append(
                            UndeclaredAttribute(
                                node.name,
                                target.attr,
                                target.lineno,
                                annotation or _suggest_type(value),
                                method.name
                                if isinstance(stmt, ast.AnnAssign)
                                else None,
                            )
                        )
    return found
