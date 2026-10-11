"""Rewrite a scenario module into one Codon 0.19 compiles, line for line.

Codon reads Python syntax and checks it with static types, but it lacks some
of what typed Python writes. A scenario's source goes through
:func:`transform_source` before it is compiled, and every rewrite keeps each
statement on its line, so what Codon reports maps straight back onto the file
the author wrote:

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
  ``abc`` shim's ``ABC`` is an empty base).
* A parameter or return annotation naming a class the module defines further
  down (``AbsoluteVelocity.__add__(self, other: RelativeVelocity)``) is
  dropped: Codon 0.19 cannot name a class in a signature before its
  definition, and two classes that take each other cannot both come first.
* A bare ``*`` in a signature (keyword-only parameters) becomes ``*_acs_kw``.
* A list display of two or more elements that are not all literals, e.g.
  ``[ElapsedTimeCondition(...), SpeedCondition(...)]``, becomes
  ``_acs_list(...)``: Codon types a display by its first element, and
  ``_acs_list`` makes it a list of conditions (or actions) as in Python.
* An ``Enum`` class derives from the ``enum`` shim's ``Enum[T]`` (``T`` the
  type of its values: ``str`` for ``class C(str, Enum)``, ``int`` for
  ``auto()``), and each member ``RED = "red"`` becomes the class variable
  ``RED: ClassVar[C] = C("RED", "red")``.
* A class deriving from an exception (a built-in one, one defined above it in
  the module, or a name ending in ``Error`` / ``Exception``) derives from
  ``Static[Base]``, as Codon 0.19 derives exceptions.
* A ``@classmethod`` becomes a ``@staticmethod`` without its ``cls``
  parameter; ``cls`` in its body names the class (Codon has no
  classmethods; a subclass calling it gets the base class's).
* ``raise X from None`` becomes ``raise X``: Codon 0.19 cannot type the
  ``None`` cause.

:func:`undeclared_attributes` reports what Codon additionally needs: an
attribute a class assigns on ``self`` must be declared at class level, with
its type, on any class in an inheritance hierarchy (Codon infers the type of
an undeclared one as ``None`` there).  Such a declaration is a bare
annotation, which Python ignores, so adding it changes nothing at run time.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "PRELUDE",
    "UndeclaredAttribute",
    "class_declarations",
    "redirect_imports",
    "transform_source",
    "undeclared_attributes",
]

#: First line of every transformed module; reported line numbers are shifted
#: back by one.
PRELUDE = "from autoware_carla_scenario._lists import _acs_list\n"

_OPTIONAL_NAMES = {"Optional", "typing.Optional"}
_UNION_NAMES = {"Union", "typing.Union"}
_CONCRETE_GENERICS = {
    "list",
    "List",
    "dict",
    "Dict",
    "set",
    "Set",
    "frozenset",
    "tuple",
    "Tuple",
    "ClassVar",
    "typing.List",
    "typing.Dict",
    "typing.Set",
    "typing.Tuple",
    "typing.ClassVar",
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
_UNEXPRESSIBLE = {"Any", "object", "Callable", "type", "Type", "Literal", "Final"}

#: Class decorators dropped: Codon gives the class what they would.
_DROPPED_CLASS_DECORATORS = {
    "dataclass",
    "dataclasses.dataclass",
    "unique",
    "enum.unique",
}
_FIELD_FUNCTIONS = {"field", "dataclasses.field"}
#: Function decorators dropped: Codon 0.19 rejects a decorated method, and
#: the method compiles the same without them.
_DROPPED_FUNCTION_DECORATORS = {"abstractmethod", "abc.abstractmethod"}

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


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def codon_annotation(node: ast.AST, *, class_level: bool = False) -> str | None:
    """The Codon spelling of a Python annotation, or ``None`` to drop it."""
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
    if not subscript:
        return base_name
    if isinstance(node, ast.Subscript):
        args = (
            list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        if base_name not in _CONCRETE_GENERICS and base not in _ABSTRACT_GENERICS:
            return None  # a generic Codon has no model of (NDArray, ...)
        if any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in args):
            return None  # tuple[X, ...]: Codon tuples have a fixed length
        converted = [codon_annotation(a, class_level=class_level) for a in args]
        if any(c is None for c in converted):
            return None
        return f"{base_name}[{', '.join(c for c in converted if c is not None)}]"
    return None


def _field_default(call: ast.Call, edits: _Edits) -> str | None:
    """The default a ``field(...)`` call stands for, or ``None`` if it has none."""
    for kw in call.keywords:
        if kw.arg == "default":
            return edits.text(*edits.node_span(kw.value))
        if kw.arg == "default_factory":
            factory = kw.value
            if isinstance(factory, ast.Lambda):
                return "(" + edits.text(*edits.node_span(factory.body)) + ")"
            name = _dotted(factory)
            if name in _FACTORY_LITERALS:
                return _FACTORY_LITERALS[name]
            return edits.text(*edits.node_span(factory)) + "()"
    return None


class _Rewriter(ast.NodeVisitor):
    def __init__(self, edits: _Edits, exceptions: set[str]) -> None:
        self.edits = edits
        self._class_depth = 0
        self._function_depth = 0
        #: Top-level classes of the module that are exceptions.
        self._exceptions = exceptions
        #: The class whose body is being visited (``None`` in a function).
        self._class_name: str | None = None
        #: Top-level classes of the module defined below the statement
        #: being visited.
        self._later_classes: set[str] = set()

    # -- annotations -----------------------------------------------------

    def visit_Module(self, node: ast.Module) -> None:
        later = {stmt.name for stmt in node.body if isinstance(stmt, ast.ClassDef)}
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                later.discard(stmt.name)  # a class may name itself
            self._later_classes = set(later)
            self.visit(stmt)

    def _rewrite_annotation(self, node: ast.AST, *, class_level: bool) -> str | None:
        """Rewrite *node* in place; ``None`` when it must be dropped instead."""
        if not class_level and _annotation_names(node) & self._later_classes:
            return None  # a class defined further down: Codon cannot name it yet
        converted = codon_annotation(node, class_level=class_level)
        if converted is not None:
            start, end = self.edits.node_span(node)
            if self.edits.text(start, end) != converted:
                self.edits.replace(start, end, converted)
        return converted

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
        if self._rewrite_annotation(node.returns, class_level=False) is None:
            ann_start, ann_end = self.edits.node_span(node.returns)
            head = self.edits.text(self.edits.offset(node.lineno, 0), ann_start)
            arrow = head.rfind("->")
            if arrow >= 0:
                before = head[:arrow].rstrip(" ")
                start = self.edits.offset(node.lineno, 0) + len(before.encode())
                self.edits.replace(start, ann_end, "")

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
            elif _dotted(decorator) in _DROPPED_FUNCTION_DECORATORS:
                start, end = self.edits.node_span(decorator)
                self.edits.replace(start - 1, end, "")  # with its "@"
            else:
                self.visit(decorator)
        if classmethod_ and self._class_name is not None:
            self._rewrite_classmethod(node, self._class_name)
        self._rewrite_arguments(node.args, skip_first=classmethod_)
        self._rewrite_returns(node)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        class_depth, self._class_depth = self._class_depth, 0
        class_name, self._class_name = self._class_name, None
        self._function_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._function_depth -= 1
        self._class_depth = class_depth
        self._class_name = class_name

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
        return [stmt for stmt, _, _ in members]

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        class_level = self._class_depth > 0 and self._function_depth == 0
        value = node.value
        if (
            class_level
            and isinstance(value, ast.Call)
            and _dotted(value.func) in _FIELD_FUNCTIONS
        ):
            default = _field_default(value, self.edits)
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
        self.generic_visit(node)

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


def transform_source(source: str) -> str:
    """Rewrite *source* for Codon; the result starts with :data:`PRELUDE`."""
    if not source.endswith("\n"):
        source += "\n"
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    rewriter = _Rewriter(edits, _exception_classes(tree))
    rewriter.visit(tree)
    return PRELUDE + edits.apply()


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


def redirect_imports(
    source: str,
    module: str,
    is_package: bool,
    target: Callable[[str], str | None],
    is_module: Callable[[str], bool],
    package: str = "autoware_carla_scenario",
) -> tuple[str, list[tuple[int, str]]]:
    """Point every import of *package* in *source* at the module standing in for it.

    *source* is module *module* of the framework, compiled by the library
    check.  ``target(name)`` is the module of the check's workspace that
    stands in for the framework module *name*: its own (rewritten) source, or
    its Codon model; ``None`` when there is neither.  ``is_module(name)``
    tells a submodule (``from . import frames``) from a name.  Each import
    statement keeps its line; ``from a import b, c`` that imports both a
    submodule and names becomes two statements on it.

    Returns:
        The rewritten source, and (line, message) for each import that could
        not be pointed anywhere (left as written).
    """
    if not source.endswith("\n"):
        source += "\n"
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    problems: list[tuple[int, str]] = []

    def inside(name: str) -> bool:
        return name == package or name.startswith(package + ".")

    def missing(name: str) -> str:
        return (
            f"{name} is neither checked (typecheck/library.py) nor modelled "
            "(typecheck/codon/), so a checked module cannot import it"
        )

    for node in ast.walk(tree):
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
