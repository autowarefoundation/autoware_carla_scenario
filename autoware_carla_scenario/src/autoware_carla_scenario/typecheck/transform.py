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
  Class-level declarations keep the abstract collections, which the
  ``typing`` shim maps onto ``List`` / ``Dict``.
* ``@dataclass`` is removed (a Codon class with annotated fields already gets
  the dataclass ``__init__``), and each ``field(default=...)`` /
  ``field(default_factory=...)`` is replaced by its default.
* ``@abstractmethod`` is removed (Codon 0.19 cannot decorate a method; the
  ``abc`` shim's ``ABC`` is an empty base), and so is ``@runtime_checkable``
  (the ``typing`` shim's ``Protocol`` is an empty base too).
* A parameter or return annotation naming a class the module defines further
  down (``AbsoluteVelocity.__add__(self, other: RelativeVelocity)``) is
  dropped: Codon 0.19 cannot name a class in a signature before its
  definition, and two classes that take each other cannot both come first.
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
* An ``Enum`` class derives from the ``enum`` shim's ``Enum[T]`` (``T`` the
  type of its values: ``str`` for ``class C(str, Enum)``, ``int`` for
  ``auto()``), and each member ``RED = "red"`` becomes the class variable
  ``RED: ClassVar[C] = C("RED", "red")``.  After the module, each such class
  is given a constructor by value (``C("red")``) with ``@extend``.
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

* ``return None`` (and a bare ``return``) of a function whose kept return
  annotation is ``Optional[X]`` becomes ``return Optional[X]()``: a bare
  ``None`` is Codon's ``NoneType``, which cannot be typed as one (#202).
* A top-level class with no ``__init__`` of its own whose base defines one
  (in the module, or among *init_classes*: another module's, the model's) is
  given one passing everything on to the base's, with ``@extend`` after the
  module: Codon 0.19 inherits no ``__init__`` (#201).

:func:`undeclared_attributes` reports what Codon additionally needs: an
attribute a class assigns on ``self`` must be declared at class level, with
its type, on any class in an inheritance hierarchy (Codon infers the type of
an undeclared one as ``None`` there).  Such a declaration is a bare
annotation, which Python ignores, so adding it changes nothing at run time.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

__all__ = [
    "MISSING_BUILTIN_EXCEPTIONS",
    "PRELUDE",
    "UndeclaredAttribute",
    "class_declarations",
    "classes_with_init",
    "inherited_inits",
    "redirect_imports",
    "split_tuple_excepts",
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
    "from autoware_carla_scenario._lists import _acs_list\n"
    "from autoware_carla_scenario._tc import _Required as _acs_Required, "
    "_acs_Returning, _acs_callable, _acs_tuple\n"
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
#: numpy's array class, which Codon's numpy has as ``ndarray[dtype, ndim]``.
_NDARRAY_NAMES = {"np.ndarray", "numpy.ndarray"}
_DTYPE_NAMES = {"np.dtype", "numpy.dtype"}
_UNEXPRESSIBLE = {"Any", "object", "Callable", "type", "Type", "Literal", "Final"}
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

    def message(self) -> str:
        hint = self.suggested_type or "<type>"
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
    if base in _UNEXPRESSIBLE or (base in _ABSTRACT_GENERICS and not class_level):
        return None
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


class _Rewriter(ast.NodeVisitor):
    def __init__(
        self,
        edits: _Edits,
        exceptions: set[str],
        dropped_decorators: Collection[str] = (),
    ) -> None:
        self.edits = edits
        #: Function decorators dropped besides :data:`_DROPPED_FUNCTION_DECORATORS`.
        self._dropped_decorators = set(dropped_decorators)
        self._class_depth = 0
        self._function_depth = 0
        #: Top-level classes of the module that are exceptions.
        self._exceptions = exceptions
        #: The class whose body is being visited (``None`` in a function).
        self._class_name: str | None = None
        #: The class whose method (or a function inside one) is being visited.
        self._method_class: str | None = None
        #: Top-level classes of the module defined below the statement
        #: being visited.
        self._later_classes: set[str] = set()
        #: Module-level ``TypeVar``s: an annotation naming one is dropped, as
        #: Codon (whose ``typing`` shim makes a ``TypeVar`` a string) cannot
        #: type with it.
        self._type_vars: set[str] = set()
        #: Enum classes rewritten at the module's top level, with the type of
        #: their values: each is given a constructor by value
        #: (:func:`_enum_lookups`).
        self.enums: list[tuple[str, str]] = []

    # -- annotations -----------------------------------------------------

    def visit_Module(self, node: ast.Module) -> None:
        later = {stmt.name for stmt in node.body if isinstance(stmt, ast.ClassDef)}
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
            self._later_classes = set(later)
            self.visit(stmt)

    def _rewrite_annotation(self, node: ast.AST, *, class_level: bool) -> str | None:
        """Rewrite *node* in place; ``None`` when it must be dropped instead."""
        if not class_level and _annotation_names(node) & self._later_classes:
            return None  # a class defined further down: Codon cannot name it yet
        if _annotation_names(node) & self._type_vars:
            return None  # a TypeVar, which types for mypy only
        converted = codon_annotation(node, class_level=class_level)
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

    def _rewrite_arguments(self, args: ast.arguments, *, skip_first: bool) -> None:
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
                name_end = self.edits.offset(arg.lineno, arg.col_offset) + len(
                    arg.arg.encode()
                )
                self.edits.replace(
                    name_end, self.edits.node_span(arg.annotation)[1], ""
                )
        if args.kwonlyargs and args.vararg is None:
            self._rewrite_bare_star(args)
        self._default_required_keywords(args)

    def _rewrite_defaults(self, args: ast.arguments) -> None:
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
            if classmethod_ and _dotted(decorator) == "classmethod":
                self.edits.replace(*self.edits.node_span(decorator), "staticmethod")
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
        self._rewrite_defaults(node.args)
        self._rewrite_arguments(node.args, skip_first=classmethod_)
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
        for stmt in node.body:
            self.visit(stmt)
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
        for stmt in node.body:
            for name in ast.walk(stmt):
                if isinstance(name, ast.Name) and name.id == first.arg:
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
        if (
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
        self._class_depth += 1
        for stmt in node.body:
            if members is None or stmt not in members:
                self.visit(stmt)
        self._class_depth -= 1
        self._class_name = class_name

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
        for stmt, name, value in members:
            if is_auto(value):
                if value_type == "str":
                    text = repr(name.lower())
                else:
                    last = (last * 2 if last else 1) if flag else last + 1
                    text = str(last)
            else:
                if isinstance(value, ast.Constant) and isinstance(value.value, int):
                    last = value.value
                text = self.edits.text(*self.edits.node_span(value))
            self.edits.replace(
                *self.edits.node_span(stmt),
                f"{name}: ClassVar[{node.name}] = {node.name}({name!r}, {text})",
            )
        first_start = self.edits.node_span(node.bases[0])[0]
        last_end = self.edits.node_span(node.bases[-1])[1]
        self.edits.replace(first_start, last_end, f"{enum_base}[{value_type}]")
        if self._class_depth == 0 and self._function_depth == 0:
            self.enums.append((node.name, value_type))
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
        converted = self._rewrite_annotation(node.annotation, class_level=class_level)
        if converted is None and node.value is not None and not class_level:
            self.edits.replace(
                self.edits.node_span(node.target)[1],
                self.edits.node_span(node.value)[0],
                " = ",
            )
        self.visit(node.target)
        if value is not None:
            self.visit(value)

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
        if isinstance(node.func, ast.Name) and node.func.id == "frozenset":
            # Codon 0.19 has no frozenset: build a set, which types the same.
            start, end = self.edits.node_span(node.func)
            self.edits.replace(start, end, "set")
        elif (
            isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "frozenset"
        ):
            # `frozenset[str]()`, an empty one typed by its element: a Set.
            start, end = self.edits.node_span(node.func.value)
            self.edits.replace(start, end, "Set")
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

    def visit_List(self, node: ast.List) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and len(node.elts) >= 2
            and not any(isinstance(e, ast.Starred) for e in node.elts)
            and not all(isinstance(e, ast.Constant) for e in node.elts)
        ):
            start, end = self.edits.node_span(node)
            self.edits.replace(start, start + 1, "_acs_list(")
            self.edits.replace(end - 1, end, ")")
        self.generic_visit(node)


def _splittable(handler: ast.ExceptHandler) -> bool:
    """Whether *handler* catches a tuple of classes written out (``(A, B)``)."""
    kind = handler.type
    return (
        isinstance(kind, ast.Tuple)
        and len(kind.elts) > 0
        and not any(isinstance(e, ast.Starred) for e in kind.elts)
    )


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


def transform_source(
    source: str,
    dropped_decorators: Collection[str] = (),
    init_classes: Collection[str] = (),
) -> str:
    """Rewrite *source* for Codon; the result starts with :data:`PRELUDE`.

    *dropped_decorators* names function decorators to drop besides
    ``@abstractmethod``: the library check passes the run-time-only wrappers
    :data:`.library.UNCALLED` lists, which Codon never compiles.
    *init_classes* names the classes defined elsewhere (another module, the
    model) that define an ``__init__``, their own or a base's
    (:func:`classes_with_init`): a class of *source* deriving from one
    without an ``__init__`` of its own inherits it (:func:`inherited_inits`).
    """
    return transform_source_lines(source, dropped_decorators, init_classes)[0]


def transform_source_lines(
    source: str,
    dropped_decorators: Collection[str] = (),
    init_classes: Collection[str] = (),
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
    source, origin = split_tuple_excepts(source)
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    exceptions = _exception_classes(tree)
    rewriter = _Rewriter(edits, exceptions, dropped_decorators)
    rewriter.visit(tree)
    lines = {
        node.name: origin[node.lineno - 1]
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    appended = [
        *_enum_lookups(rewriter.enums),
        *_inherited_init_extensions(inherited_inits(tree, init_classes, exceptions)),
    ]
    text = PRELUDE + edits.apply()
    origin = list(origin)
    for name, block in appended:
        text += block
        origin += [lines[name]] * block.count("\n")
    return text, origin


def _enum_lookups(enums: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """A constructor by value (``Color("red")``) for each rewritten Enum class.

    Codon gives a class without an ``__init__`` one that takes each field
    (``Color("RED", "red")``, what a member is built with); the lookup by
    value is added to it with ``@extend``, after the module, so no line of
    the source moves.  It is typed only: the ``Color`` it makes is not the
    member, which nothing the check compiles needs, since nothing runs.
    Each is (the class, the lines appended).
    """
    return [
        (
            name,
            (
                f"\n\n@extend\nclass {name}:\n"
                f"    def __init__(self, value: {value_type}):\n"
                f'        self._name_ = ""\n'
                f"        self._value_ = value\n"
            ),
        )
        for name, value_type in enums
    ]


def _inherited_init_extensions(names: list[str]) -> list[tuple[str, str]]:
    """The ``__init__`` each class of *names* inherits, appended with ``@extend``.

    Codon 0.19 inherits no ``__init__``: a class without one gets a
    constructor taking each of its fields, which refuses what the base's
    takes ("'config' is an invalid keyword argument", #201).  The one
    appended passes everything on to the base's, which checks it.
    """
    return [
        (
            name,
            (
                f"\n\n@extend\nclass {name}:\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        super().__init__(*args, **kwargs)\n"
            ),
        )
        for name in names
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
    local = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    aliases = _import_aliases(tree)
    known = {
        alias
        for alias, name in aliases.items()
        if name in init_classes and alias not in local
    } | (set(init_classes) - set(local))
    with_init = classes_with_init(local, known)
    out: list[str] = []
    for name, node in local.items():
        if _defines_init(node) or _is_dataclass_node(node) or name in exceptions:
            continue
        bases = [(_dotted(b) or "").rsplit(".", 1)[-1] for b in node.bases]
        if any(b in _ENUM_BASES for b in bases) or node.keywords:
            continue
        if any(b in with_init for b in bases):
            out.append(name)
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
) -> list[UndeclaredAttribute]:
    """Attributes assigned on ``self`` that a class of *tree* does not declare.

    Only classes in an inheritance hierarchy are checked: with a base class
    (other than ``object``) or with a subclass in *tree*.  A name counts as
    declared when the class, a base class in *tree*, or a base class named in
    *declared_elsewhere* (class name -> declared names; the Codon model's
    classes, and classes of other checked modules) declares it.  A base
    imported under another name (``from .base import Base as Parent``) is
    looked up by the name it is defined with.
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
        if not [b for b in bases if b != "object"] and node.name not in subclassed:
            continue
        names = declared(node.name)
        reported: set[str] = set()
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
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
                            )
                        )
    return found
