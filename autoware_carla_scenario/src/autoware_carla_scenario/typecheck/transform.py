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
* A bare ``*`` in a signature (keyword-only parameters) becomes ``*_acs_kw``.
* A list display of two or more elements that are not all literals, e.g.
  ``[ElapsedTimeCondition(...), SpeedCondition(...)]``, becomes
  ``_acs_list(...)``: Codon types a display by its first element, and
  ``_acs_list`` makes it a list of conditions (or actions) as in Python.

:func:`undeclared_attributes` reports what Codon additionally needs: an
attribute a class assigns on ``self`` must be declared at class level, with
its type, on any class in an inheritance hierarchy (Codon infers the type of
an undeclared one as ``None`` there).  Such a declaration is a bare
annotation, which Python ignores, so adding it changes nothing at run time.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

__all__ = [
    "PRELUDE",
    "Transformed",
    "UndeclaredAttribute",
    "class_declarations",
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

_DATACLASS_DECORATORS = {"dataclass", "dataclasses.dataclass"}
_FIELD_FUNCTIONS = {"field", "dataclasses.field"}
_FACTORY_LITERALS = {"list": "[]", "dict": "{}", "set": "set()", "tuple": "()"}


@dataclass
class Transformed:
    """A transformed module and what the rewrite needed."""

    source: str
    uses_acs_list: bool = False


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

    def __post_init__(self) -> None:
        offset = 0
        for line in self.lines:
            self.starts.append(offset)
            offset += len(line)

    def offset(self, lineno: int, col: int) -> int:
        return self.starts[lineno - 1] + col

    def text(self, start: int, end: int) -> str:
        return b"".join(self.lines)[start:end].decode()

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
        data = b"".join(self.lines)
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
    name = _dotted(node)
    if name is not None:
        base = name.rsplit(".", 1)[-1] if name.startswith("typing.") else name
        if base in _UNEXPRESSIBLE:
            return None
        if base in _ABSTRACT_GENERICS and not class_level:
            return None
        return name
    if isinstance(node, ast.Subscript):
        base_name = _dotted(node.value)
        if base_name is None:
            return None
        base = (
            base_name.rsplit(".", 1)[-1]
            if base_name.startswith("typing.")
            else base_name
        )
        args = (
            list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        if base_name in _OPTIONAL_NAMES:
            inner_text = codon_annotation(args[0], class_level=class_level)
            return None if inner_text is None else f"Optional[{inner_text}]"
        if base in _UNEXPRESSIBLE:
            return None
        if base in _ABSTRACT_GENERICS and not class_level:
            return None
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
    def __init__(self, edits: _Edits) -> None:
        self.edits = edits
        self.uses_acs_list = False
        self._class_depth = 0
        self._function_depth = 0

    # -- annotations -----------------------------------------------------

    def _rewrite_annotation(self, node: ast.AST, *, class_level: bool) -> str | None:
        """Rewrite *node* in place; ``None`` when it must be dropped instead."""
        converted = codon_annotation(node, class_level=class_level)
        if converted is not None:
            start, end = self.edits.node_span(node)
            if self.edits.text(start, end) != converted:
                self.edits.replace(start, end, converted)
        return converted

    def _rewrite_arguments(self, args: ast.arguments) -> None:
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
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
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._rewrite_arguments(node.args)
        self._rewrite_returns(node)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        class_depth, self._class_depth = self._class_depth, 0
        self._function_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._function_depth -= 1
        self._class_depth = class_depth

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _dotted(target) in _DATACLASS_DECORATORS:
                start, end = self.edits.node_span(decorator)
                self.edits.replace(start - 1, end, "")  # with its "@"
            else:
                self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        self._class_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._class_depth -= 1

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
            self.uses_acs_list = True
        self.generic_visit(node)


def transform_source(source: str) -> Transformed:
    """Rewrite *source* for Codon; the result starts with :data:`PRELUDE`."""
    if not source.endswith("\n"):
        source += "\n"
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    rewriter = _Rewriter(edits)
    rewriter.visit(tree)
    return Transformed(PRELUDE + edits.apply(), uses_acs_list=rewriter.uses_acs_list)


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
    classes, and classes of other checked modules) declares it.
    """
    classes = class_declarations(tree)
    subclassed = {b.rsplit(".", 1)[-1] for bases, _ in classes.values() for b in bases}

    def declared(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
        if name in seen:
            return set()
        if name in classes:
            bases, own = classes[name]
            out = set(own)
            for base in bases:
                out |= declared(base.rsplit(".", 1)[-1], seen | {name})
            return out
        return set(declared_elsewhere.get(name, set()))

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
