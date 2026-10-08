"""Rewrite the framework's own modules so Codon compiles them as they are.

A standalone binary is compiled from the framework's real sources, not from a
re-implementation of them. :func:`~..typecheck.transform.transform_source`
already rewrites what Codon cannot express; :func:`transform_for_runtime` adds
what Codon cannot *run* (or parse). A rewrite either stays on the lines it
rewrites or inserts whole lines marked :data:`INSERTED`, so
:func:`original_line` maps an error back to the line the author wrote:

* ``a or b`` -> ``_acs_or(a, lambda: b)``: Codon types ``a or b`` as a Union
  and fails to read an ``Optional[T] or T`` back at run time.
* ``x if x is not None else d`` -> ``_acs_default(x, lambda: d)``: a Codon
  conditional expression needs one type, and ``d`` may be a subclass of T.
* A required keyword-only parameter after defaulted ones (``*, label: str``),
  which Codon refuses, gets a sentinel default; the function body checks it
  (``_acs_unrequire``), at compile time, on a line inserted before its first
  statement.
* An ``Optional[C]`` parameter (C a class), which in Codon takes neither a
  subclass of C nor an Optional of one, is made generic, and the body
  converts it to ``Optional[C]`` (``_acs_opt``), on that line.
* ``@abstractmethod`` is dropped (Codon allows no decorator on a method but its
  own), and so is an ``ABC`` or ``Protocol`` base (a level of inheritance
  Codon miscompiles) and ``@runtime_checkable``.
* ``@classmethod`` -> ``@staticmethod``, with ``cls`` meaning the class itself
  (Codon has no class methods).
* The framework's own imports under ``if TYPE_CHECKING:`` are dropped, and
  so are the annotations naming them (the parameter turns generic): they
  run in Codon and close import cycles Python never runs into.
* ``__slots__`` is dropped; ``from collections.abc import ...`` imports from
  ``typing`` (Codon has no ``collections.abc``).
* An attribute a class assigns on ``self`` without declaring it at class
  level, which Codon needs declared (docs/typecheck.md), is declared on a
  line inserted at the top of the class body when its type can be read: an
  annotation where it is assigned, the annotation of the parameter assigned
  to it, or a literal or constructor call.
* An ``Enum`` (Codon has no ``enum``) becomes a class deriving from
  ``_AcsEnum[T]`` whose members are class-level singletons, so ``is``,
  ``==``, ``.name`` and ``.value`` behave as they do on Python's; ``auto()``
  numbers members from 1, as Python's does.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Callable, Mapping

from .hierarchy import ModuleContext
from ..typecheck.transform import (
    PRELUDE,
    _Edits,
    _is_none,
    _suggest_type,
    _union_members,
    codon_annotation,
    transform_source,
    undeclared_attributes,
)

__all__ = [
    "INSERTED",
    "RUNTIME_PRELUDE",
    "original_line",
    "rewrite_for_runtime",
    "rewrite_or",
    "transform_for_runtime",
]

#: The first line of every rewritten module (one line, as :data:`PRELUDE` is).
RUNTIME_PRELUDE = (
    "from _acs_helpers import (_acs_class, _acs_default, _acs_list, _acs_opt, _acs_or, _acs_or_empty,"
    " _AcsClass, _AcsEnum,"
    " _AcsRequired, _acs_unrequire, frozenset)\n"
)
_ABSTRACT = {"abstractmethod", "abc.abstractmethod"}
_ABC = {"ABC", "abc.ABC", "Protocol", "typing.Protocol"}
_CLASS_DECORATORS_DROPPED = {"runtime_checkable", "typing.runtime_checkable"}
_ENUM = {"Enum", "enum.Enum"}
#: typing names Codon has as builtins.
_CODON_BUILTIN_TYPES = {
    "Optional", "Union", "List", "Dict", "Set", "Tuple", "Callable", "ClassVar", "Literal",
}  # fmt: skip
_AUTO = {"auto", "enum.auto"}


def _is_or(node: ast.AST) -> bool:
    return isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)


def _innermost_ors(tree: ast.AST) -> list[ast.BoolOp]:
    """The ``or`` expressions with no other ``or`` inside, outside f-strings."""
    found: list[ast.BoolOp] = []

    def visit(node: ast.AST) -> bool:
        if isinstance(node, ast.JoinedStr):
            return False
        contains = False
        for child in ast.iter_child_nodes(node):
            contains = visit(child) or contains
        if _is_or(node):
            if not contains:
                found.append(node)  # type: ignore[arg-type]
            return True
        return contains

    visit(tree)
    return found


def _none_test(test: ast.AST) -> tuple[str, bool] | None:
    """(name, is_not) of a ``name is None`` / ``name is not None`` test."""
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], (ast.Is, ast.IsNot))
        and isinstance(test.left, (ast.Name, ast.Attribute))
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    ):
        return ast.unparse(test.left), isinstance(test.ops[0], ast.IsNot)
    return None


def rewrite_defaults(source: str) -> str:
    """``x if x is not None else d`` (or ``d if x is None else x``) ->
    ``_acs_default(x, lambda: d)``: the two branches of a Codon conditional
    expression must have one type, and these have Optional[T] and a T (or a
    subclass of T)."""
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        found = _none_test(node.test)
        if found is None:
            continue
        name, is_not = found
        value, fallback = (
            (node.body, node.orelse) if is_not else (node.orelse, node.body)
        )
        if ast.unparse(value) != name:
            continue
        edits.replace(
            *edits.node_span(node),
            f"_acs_default({name}, lambda: {edits.text(*edits.node_span(fallback))})",
        )
    return edits.apply()


def _empty_literal(node: ast.AST) -> bool:
    """``[]``, ``{}``, ``()``, ``set()``, ``list()``, ``dict()``, ``tuple()``."""
    if isinstance(node, (ast.List, ast.Tuple)) and not node.elts:
        return True
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("set", "list", "dict", "tuple")
        and not node.args
        and not node.keywords
    )


def rewrite_or(source: str) -> str:
    """Every ``a or b [or c ...]`` in *source*, as nested ``_acs_or`` calls."""
    while True:
        targets = _innermost_ors(ast.parse(source))
        if not targets:
            return source
        edits = _Edits(source.encode().splitlines(keepends=True))
        for node in targets:
            parts = [edits.text(*edits.node_span(v)) for v in node.values]
            text = parts[-1]
            last = node.values[-1]
            if len(parts) == 2 and _empty_literal(last):
                # `x or []`: an empty literal has no element type of its own
                # in Codon; the result is x's type, made empty
                edits.replace(*edits.node_span(node), f"(_acs_or_empty({parts[0]}))")
                continue
            for part in reversed(parts[:-1]):
                text = f"_acs_or({part}, lambda: {text})"
            edits.replace(*edits.node_span(node), f"({text})")
        source = edits.apply()


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _decorator_span(edits: _Edits, deco: ast.expr) -> tuple[int, int]:
    """The span of ``@deco`` (the ``@`` sits just before the expression)."""
    start, end = edits.node_span(deco)
    return start - 1, end


def _enum_members(cls: ast.ClassDef) -> list[tuple[ast.Assign, str, object]] | None:
    """(statement, name, value) of each member of an Enum class; None if not one."""
    if not any(_dotted(b) in _ENUM for b in cls.bases):
        return None
    members: list[tuple[ast.Assign, str, object]] = []
    counter = 0
    for stmt in cls.body:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and not stmt.targets[0].id.startswith("_")
        ):
            continue
        value = stmt.value
        if isinstance(value, ast.Call) and _dotted(value.func) in _AUTO:
            counter += 1
            members.append((stmt, stmt.targets[0].id, counter))
        elif isinstance(value, ast.Constant) and isinstance(value.value, (int, str)):
            if isinstance(value.value, int):
                counter = value.value
            members.append((stmt, stmt.targets[0].id, value.value))
        else:
            return None  # a member Codon cannot be given: left for Codon to report
    return members


def _rewrite_enum(cls: ast.ClassDef, edits: _Edits) -> bool:
    members = _enum_members(cls)
    if not members:
        return False
    kinds = {type(v) for _s, _n, v in members}
    if len(kinds) != 1:
        return False
    kind = "int" if kinds == {int} else "str"
    for base in cls.bases:
        if _dotted(base) in _ENUM:
            edits.replace(*edits.node_span(base), f"_AcsEnum[{kind}]")
    for stmt, name, value in members:
        edits.replace(
            *edits.node_span(stmt),
            f"{name}: ClassVar[{cls.name}] = {cls.name}({value!r}, {name!r}, {cls.name!r})",
        )
    return True


def _unoptional(annotation: ast.AST) -> ast.AST:
    """``Optional[T]`` / ``T | None`` -> ``T``; anything else as it is."""
    if isinstance(annotation, ast.Subscript) and _dotted(annotation.value) in (
        "Optional",
        "typing.Optional",
        "_Optional",
    ):
        return annotation.slice
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        for side, other in (
            (annotation.left, annotation.right),
            (annotation.right, annotation.left),
        ):
            if isinstance(other, ast.Constant) and other.value is None:
                return side
    return annotation


def _param_annotation(
    value: ast.AST | None, params: Mapping[str, ast.AST]
) -> ast.AST | None:
    """The type a parameter gives *value*: ``param``, or ``param if ... else default``
    (the parameter's type, not Optional: the other branch fills its None)."""
    if isinstance(value, ast.Name) and value.id in params:
        return params[value.id]
    if isinstance(value, ast.IfExp):
        for branch in (value.body, value.orelse):
            if isinstance(branch, ast.Name) and branch.id in params:
                return _unoptional(params[branch.id])
    return None


class _Unquote(ast.NodeTransformer):
    """``'carla.World'`` inside an annotation -> ``carla.World``."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            try:
                return ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return node
        return node


@functools.cache
def carla_enums() -> frozenset[str]:
    """The CARLA enums typesafe_carla's Codon library spells as ``int``.

    Its enums are instances (``TrafficLightState = _TrafficLightState()``)
    whose fields are the members, which are ints: ``carla.TrafficLightState``
    is a value there, not a type.
    """
    from ..typecheck.toolchain import codon_path_dir  # noqa: PLC0415

    found: set[str] = set()
    for path in (codon_path_dir() / "typesafe_carla").glob("*.codon"):
        found.update(_ENUM_INSTANCE.findall(path.read_text(encoding="utf-8")))
    return frozenset(found)


_ENUM_INSTANCE = re.compile(r"^([A-Z]\w*) = _\1\(\)$", re.MULTILINE)


def _carla_enum(node: ast.AST) -> bool:
    name = _dotted(node)
    return bool(name) and name.startswith("carla.") and name[6:] in carla_enums()


class _RuntimeTypes(ast.NodeTransformer):
    """Python types with a different name in Codon: ``IO[str]`` -> ``File``,
    ``carla.TrafficLightState`` -> ``int``."""

    _FILE = {"IO", "TextIO", "typing.IO", "typing.TextIO"}

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if _dotted(node.value) in self._FILE:
            return ast.Name("File", ast.Load())
        if _dotted(node.value) in _CLASS_TYPES:
            node.value = ast.Name("_AcsClass", ast.Load())
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.Name("File", ast.Load()) if node.id in self._FILE else node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        return ast.Name("int", ast.Load()) if _carla_enum(node) else node


def _carla_enum_annotations(tree: ast.Module, edits: _Edits) -> None:
    """``carla.TrafficLightState`` in an annotation -> ``int`` (:func:`carla_enums`)."""
    annotations: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
            annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    for annotation in annotations:
        for node in ast.walk(annotation):
            if _carla_enum(node):
                edits.replace(*edits.node_span(node), "int")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                try:  # a forward reference: "carla.TrafficLightState"
                    inner = ast.parse(node.value, mode="eval").body
                except SyntaxError:
                    continue
                if any(_carla_enum(n) for n in ast.walk(inner)):
                    text = ast.unparse(_RuntimeTypes().visit(inner))
                    edits.replace(*edits.node_span(node), text)


def _annotation_text(annotation: ast.AST) -> str | None:
    """The Codon spelling of an attribute's *annotation*, or None to leave it out."""
    import copy  # noqa: PLC0415

    node = _RuntimeTypes().visit(_Unquote().visit(copy.deepcopy(annotation)))
    # Codon has Callable[[A], R] (the checker drops it): each one converted
    # on its own, through a placeholder name the conversion keeps as it is
    callables: dict[str, str] = {}

    class _Callables(ast.NodeTransformer):
        def visit_Subscript(self, sub: ast.Subscript) -> ast.AST:
            self.generic_visit(sub)
            if _dotted(sub.value) not in ("Callable", "typing.Callable"):
                return sub
            parts = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else []
            if len(parts) != 2 or not isinstance(parts[0], ast.List):
                return sub
            texts = [_convert(a) for a in parts[0].elts] + [_convert(parts[1])]
            if any(t is None for t in texts):
                return sub
            key = f"_ACS_CALLABLE_{len(callables)}"
            callables[key] = f"Callable[[{', '.join(texts[:-1])}], {texts[-1]}]"  # type: ignore[arg-type]
            return ast.Name(key, ast.Load())

    def _convert(part: ast.expr) -> str | None:
        if isinstance(part, ast.Name) and part.id in callables:
            return callables[part.id]
        if isinstance(part, ast.Constant) and part.value is None:
            return "None"
        return codon_annotation(part, class_level=True)

    node = _Callables().visit(node)
    text = codon_annotation(node, class_level=True) or _union_text(node)
    if text is None:
        return None
    for key, value in callables.items():
        text = re.sub(rf"\b{key}\b", value, text)
    return text


def _union_text(node: ast.AST) -> str | None:
    """``Union[A, B]`` (``Optional[...]`` of one with a None member) for a
    real union: the checker drops one, but a Codon field takes it."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    members = _union_members(node)
    if members is None and _unoptional(node) is not node:
        inner = _union_text(_unoptional(node))
        return None if inner is None else f"Optional[{inner}]"
    if members is None:
        return None
    rest = [m for m in members if not _is_none(m)]
    texts = [codon_annotation(m, class_level=True) for m in rest]
    if len(rest) < 2 or any(t is None for t in texts):
        return None
    text = f"Union[{', '.join(t for t in texts if t)}]"
    return f"Optional[{text}]" if len(rest) < len(members) else text


def _attribute_types(cls: ast.ClassDef) -> dict[str, str]:
    """Attribute -> the source text of its type, as the class's methods assign it."""
    types: dict[str, str] = {}
    for method in cls.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {
            a.arg: a.annotation
            for a in method.args.posonlyargs + method.args.args + method.args.kwonlyargs
            if a.annotation is not None
        }
        for stmt in ast.walk(method):
            target: ast.AST | None = None
            annotation: ast.AST | None = None
            value: ast.AST | None = None
            if isinstance(stmt, ast.AnnAssign):
                target, annotation, value = stmt.target, stmt.annotation, stmt.value
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue
            if target.attr in types:
                continue
            if (
                annotation is None
                and isinstance(value, ast.BoolOp)
                and isinstance(value.op, ast.Or)
                and len(value.values) == 2
                and isinstance(value.values[0], ast.Name)
                and _class_param(params.get(value.values[0].id)) is not None
            ):
                # `param or Default` of a class parameter (_rewrite_parameters)
                types[target.attr] = f"_AcsClass[{ast.unparse(value.values[1])}]"
                continue
            if annotation is None:
                annotation = _param_annotation(value, params)
            if annotation is not None:
                text = _annotation_text(annotation)
                if text is not None:
                    types[target.attr] = text
                continue
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id[:1].isupper()
            ):
                types[target.attr] = value.value.id  # an enum member: Enum.MEMBER
                continue
            suggested = _suggest_type(value)
            if suggested is not None and "..." not in suggested:
                types[target.attr] = suggested
    return types


def _own_declarations(cls: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
    return names


def _declare_attributes(
    tree: ast.Module,
    edits: _Edits,
    dropped: set[str],
    context: ModuleContext | None = None,
) -> None:
    missing: dict[str, list[str]] = {}
    for attr in undeclared_attributes(tree, {}):
        missing.setdefault(attr.class_name, []).append(attr.name)
    # undeclared_attributes() looks at class hierarchies only (where the check
    # needs a declaration); a compiled class needs one wherever it is, or its
    # type is incomplete (a field of it then cannot be typed either).
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.bases or cls.name in missing:
            continue
        declared = _own_declarations(cls)
        assigned = [name for name in _attribute_types(cls) if name not in declared]
        if assigned:
            missing[cls.name] = assigned
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in missing:
            continue
        types = _attribute_types(cls)
        # a class attribute is kept on the class (_class_attributes)
        skip: set[str] = set()
        if context is not None:
            skip = set(context.attributes) - context.fields.get(cls.name, set())
        decls = [
            f"{name}: {types[name]}"
            for name in missing[cls.name]
            if name in types
            and name not in skip
            and not _uses(ast.parse(types[name], mode="eval").body, dropped)
        ]
        if context is not None:
            bound = _module_names(tree)
            for decl in decls:
                for node in ast.walk(ast.parse(decl.split(": ", 1)[1], mode="eval")):
                    if not isinstance(node, ast.Name) or node.id in bound:
                        continue
                    statement = context.hoistable.get(node.id)
                    if statement is not None and statement not in context.hoisted:
                        # imported in a function only: the declaration needs it
                        context.hoisted.append(statement)
        if decls:
            _insert_before(edits, cls.body[0], "; ".join(decls))


def _module_names(tree: ast.Module) -> set[str]:
    """Names *tree* binds at module level (TYPE_CHECKING imports included)."""
    out: set[str] = set()
    for node in tree.body:
        stmts = node.body if isinstance(node, ast.If) else [node]
        for stmt in stmts:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                out.update((a.asname or a.name).split(".")[0] for a in stmt.names)
            elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef)):
                out.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                out.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                out.add(stmt.target.id)
    return out


def _typed_factories(tree: ast.Module, edits: _Edits) -> None:
    """``x: list[T] = field(default_factory=list)`` -> ``x: list[T] = list[T]()``.

    The check's rewrite puts ``[]`` in place, which Codon types as a list of
    nothing; the annotation says what the list holds.
    """
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for stmt in cls.body:
            if not (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.value, ast.Call)
                and _dotted(stmt.value.func) in ("field", "dataclasses.field")
                and isinstance(stmt.annotation, ast.Subscript)
            ):
                continue
            factory = next(
                (k.value for k in stmt.value.keywords if k.arg == "default_factory"),
                None,
            )
            if isinstance(factory, ast.Name) and factory.id in ("list", "dict", "set"):
                edits.replace(
                    *edits.node_span(stmt.value), f"{ast.unparse(stmt.annotation)}()"
                )


#: Ends every line a rewrite inserts; :func:`original_line` counts them back out.
INSERTED = "  # acs:inserted"


def _insert_before(edits: _Edits, stmt: ast.stmt, text: str) -> None:
    """A new line *text* before *stmt*, at its indentation."""
    line_start = edits.offset(stmt.lineno, 0)
    edits.replace(
        line_start, line_start, " " * stmt.col_offset + text + INSERTED + "\n"
    )


def original_line(transformed: str, line: int) -> int:
    """The line of the author's file that *line* of the rewritten module is."""
    lines = transformed.splitlines()
    inserted = sum(1 for text in lines[: line - 1] if text.endswith(INSERTED))
    return line - 1 - inserted  # less the prelude line


_NOT_CLASSES = {"str", "int", "float", "bool", "bytes"}


def _optional_class(annotation: ast.AST | None) -> str | None:
    """``C`` of an ``Optional[C]`` / ``C | None`` parameter annotation, C a class."""
    if annotation is None:
        return None
    union = _union_text(annotation)
    if union is not None and union.startswith("Optional["):
        return union[len("Optional[") : -1]  # Optional[Union[A, B]] -> Union[A, B]
    inner = _unoptional(annotation)
    if inner is annotation:
        return None
    name = _dotted(inner)
    if not name or name in _NOT_CLASSES or not name.rsplit(".", 1)[-1][:1].isupper():
        return None
    return name


_CLASS_TYPES = {"type", "Type", "typing.Type"}


def _class_param(annotation: ast.AST | None) -> str | None:
    """``C`` of a ``type[C]`` / ``Optional[type[C]]`` parameter annotation."""
    if annotation is None:
        return None
    inner = _unoptional(annotation)
    if isinstance(inner, ast.Subscript) and _dotted(inner.value) in _CLASS_TYPES:
        return ast.unparse(inner.slice)
    return None


def _rewrite_parameters(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    edits: _Edits,
    touched: frozenset[int] | set[int] = frozenset(),
) -> None:
    """Required keyword-only parameters and ``Optional[C]`` parameters.

    Codon refuses a required parameter after defaulted ones, and its
    ``Optional[C]`` takes neither a C subclass nor an ``Optional`` of one: such
    a parameter is made generic, and the body converts it on entry
    (``_acs_unrequire`` / ``_acs_opt``), on a line inserted before its first
    statement.
    """
    has_defaults = bool(fn.args.defaults) or any(
        d is not None for d in fn.args.kw_defaults
    )
    required = {
        arg.arg
        for arg, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
        if d is None and has_defaults
    }
    entry: list[str] = []
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        if _class_param(arg.annotation) is not None:
            # `param or Default`, how a class parameter is defaulted: the
            # default is the class (_AcsClass) -- the annotation may name one
            # the module only imports under TYPE_CHECKING
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.BoolOp)
                    and isinstance(node.op, ast.Or)
                    and len(node.values) == 2
                    and isinstance(node.values[0], ast.Name)
                    and node.values[0].id == arg.arg
                ):
                    default = edits.text(*edits.node_span(node.values[1]))
                    edits.replace(
                        *edits.node_span(node), f"_acs_class({arg.arg}, {default})"
                    )
        if id(arg) in touched:
            # its annotation is already gone (_drop_annotations): only the
            # sentinel default, after the span that edit rewrites
            if arg.arg in required:
                end = edits.node_span(arg)[1]
                edits.replace(end, end, " = _AcsRequired()")
                entry.append(f'{arg.arg} = _acs_unrequire({arg.arg}, "{arg.arg}")')
            continue
        klass = _class_param(arg.annotation)
        if klass is not None and arg.arg not in required:
            # a class as a value (_AcsClass)
            entry.append(f"{arg.arg} = _acs_class({arg.arg}, {klass})")
            edits.replace(*edits.node_span(arg), arg.arg)
            continue
        cls = _optional_class(arg.annotation)
        if arg.arg not in required and cls is None:
            continue
        text = arg.arg
        if arg.arg in required:
            text += " = _AcsRequired()"
            entry.append(f'{arg.arg} = _acs_unrequire({arg.arg}, "{arg.arg}")')
        elif arg.annotation is not None and cls is None:
            text += ": " + ast.unparse(arg.annotation)
        if cls is not None:
            entry.append(f"{arg.arg} = _acs_opt({arg.arg}, {cls})")
        edits.replace(*edits.node_span(arg), text)
    if entry:
        _insert_before(edits, fn.body[0], "; ".join(entry))


def _qualify_class_defaults(cls: ast.ClassDef, edits: _Edits) -> None:
    """``def f(self, x=LIMIT)``, ``LIMIT = 0`` in the class body -> ``x=0``.

    Python evaluates a method's defaults in the class body's scope; Codon looks
    them up in the module's, and the class is not defined yet when it does.
    Only a constant is inlined; any other such default stays a build error.
    """
    names: dict[str, str] = {}
    for stmt in cls.body:
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, ast.Constant):
            targets = (
                stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]  # type: ignore[attr-defined]
            )
            names.update(
                (t.id, repr(value.value)) for t in targets if isinstance(t, ast.Name)
            )
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = [*stmt.args.defaults, *(d for d in stmt.args.kw_defaults if d)]
            for default in defaults:
                for node in ast.walk(default):
                    if isinstance(node, ast.Name) and node.id in names:
                        edits.replace(*edits.node_span(node), names[node.id])


def _class_attributes(tree: ast.Module, edits: _Edits, context: ModuleContext) -> None:
    """Class attributes (:meth:`.hierarchy.Hierarchy.class_attributes`).

    ``name: T = value`` in the class body stays a class variable,
    ``name: ClassVar[T] = value``, beside an instance field ``_acs_name`` and
    the flag saying an instance assigned it. ``obj.name`` reads
    ``obj._acs_get_name()`` -- the field once assigned, else the class
    variable of the class that last gave it a value, the method being
    dispatched as Python's lookup is -- and ``obj.name = v`` is
    ``obj._acs_set_name(v)``. A class that assigns an attribute of that name
    as an ordinary field gets the two methods for it too.
    """
    lines: dict[str, list[str]] = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        out = lines.setdefault(cls.name, [])
        for name in sorted(context.fields.get(cls.name, ())):
            out.append(f"def _acs_get_{name}(self): return self.{name}")
            out.append(f"def _acs_set_{name}(self, value): self.{name} = value")
        defines = context.defines.get(cls.name, {})
        for stmt in cls.body:
            if not (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id in defines
                and stmt.value is not None
            ):
                continue
            name = stmt.target.id
            kind = _annotation_text(stmt.annotation)
            given = edits.text(*edits.node_span(stmt.value))
            if kind is None:
                raise TransformError(
                    f"{cls.name}.{name}: the annotation of a class attribute has no "
                    "Codon spelling"
                )
            text = f"{name}: ClassVar[{kind}] = {given}"
            if defines[name]:  # the first class to have it holds the field
                text += f"; _acs_{name}: {kind}; _acs_{name}_set: bool"
                convert = "value"
                optional = _optional_class(stmt.annotation)
                if optional is not None:
                    convert = f"_acs_opt(value, {optional})"
                out.append(
                    f"def _acs_set_{name}(self, value) -> None: "
                    f"self._acs_{name} = {convert}; self._acs_{name}_set = True"
                )
            out.append(
                f"def _acs_get_{name}(self) -> {kind}: "
                f"return self._acs_{name} if self._acs_{name}_set else {cls.name}.{name}"
            )
            edits.replace(*edits.node_span(stmt), text)
        if out:
            indent = " " * cls.body[0].col_offset
            _insert_before(edits, cls.body[0], (INSERTED + "\n" + indent).join(out))
    names = context.attributes
    for node in ast.walk(tree):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Attribute)
            and target.attr in names
            and value is not None
        ):
            # `recv.name = value` -> `recv._acs_set_name(value)`; the edits
            # inside value stay separate
            start = edits.node_span(node)[0]
            recv = edits.text(*edits.node_span(target.value))
            edits.replace(
                start, edits.node_span(value)[0], f"{recv}._acs_set_{target.attr}("
            )
            end = edits.node_span(value)[1]
            edits.replace(end, end, ")")
            continue
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr in names
        ):
            continue
        recv_node = node.value
        if isinstance(recv_node, ast.Name) and recv_node.id[:1].isupper():
            continue  # read off the class: its class variable
        if (
            isinstance(recv_node, ast.Call)
            and isinstance(recv_node.func, ast.Name)
            and recv_node.func.id == "type"
            and len(recv_node.args) == 1
        ):
            # type(obj).name: what obj.name reads unless obj assigned its own
            inner = edits.text(*edits.node_span(recv_node.args[0]))
            edits.replace(*edits.node_span(node), f"{inner}._acs_get_{node.attr}()")
            continue
        end = edits.node_span(node)[1]
        edits.replace(edits.node_span(recv_node)[1], end, f"._acs_get_{node.attr}()")


def _static_exception(cls: ast.ClassDef, edits: _Edits) -> None:
    """``class E(RuntimeError)`` -> ``class E(Static[RuntimeError])``, with the
    two constructors Codon's own exceptions have (``E(message)``, and the one
    a subclass calls with its own name). The author's ``__init__``, if any,
    comes after them and wins a call both would take."""
    for base in cls.bases:
        text = edits.text(*edits.node_span(base))
        edits.replace(*edits.node_span(base), f"Static[{text}]")
    lines = [
        'def __init__(self, typename: str, message: str = ""): '
        "super().__init__(typename, message)",
    ]
    if not any(
        isinstance(s, ast.FunctionDef) and s.name == "__init__" for s in cls.body
    ):
        lines.append(
            f'def __init__(self, message: str = ""): super().__init__("{cls.name}", message)'
        )
    indent = " " * cls.body[0].col_offset
    _insert_before(edits, cls.body[0], (INSERTED + "\n" + indent).join(lines))


class TransformError(ValueError):
    """A construct the standalone rewrite cannot give Codon."""


def _type_checking_names(
    tree: ast.Module, edits: _Edits, kept: frozenset[int] = frozenset()
) -> set[str]:
    """Drop the framework's own imports under ``if TYPE_CHECKING:``; their names.

    They exist for annotations only, and run in Codon (the typing stand-in
    sets TYPE_CHECKING, so annotations on external types such as carla's
    resolve), where they close import cycles Python never runs into.
    """
    dropped: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test)):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.level > 0:
                if stmt.lineno in kept:
                    continue  # closes no cycle (.hierarchy)
                dropped.update(a.asname or a.name for a in stmt.names)
                edits.replace(*edits.node_span(stmt), "pass")
    return dropped


def _uses(annotation: ast.AST | None, names: set[str]) -> bool:
    if annotation is None or not names:
        return False
    text = ast.unparse(annotation)
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        text = annotation.value
    return any(re.search(rf"\b{re.escape(n)}\b", text) for n in names)


def _drop_annotations(tree: ast.Module, edits: _Edits, names: set[str]) -> set[int]:
    """Annotations that name a dropped import go (the parameter turns generic);
    the ids of the arguments rewritten, which other rules leave alone."""
    touched: set[int] = set()
    if not names:
        return touched
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        all_args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        for extra in (fn.args.vararg, fn.args.kwarg):
            if extra is not None:
                all_args.append(extra)
        for arg in all_args:
            if _uses(arg.annotation, names):
                assert arg.annotation is not None
                edits.replace(
                    edits.node_span(arg)[0], edits.node_span(arg.annotation)[1], arg.arg
                )
                touched.add(id(arg))
        if _uses(fn.returns, names):
            assert fn.returns is not None
            # `) -> T:` -> `):`
            end_args = edits.node_span(fn.returns)[0]
            data = edits.data
            arrow = data.rindex(b"->", 0, end_args)
            edits.replace(arrow, edits.node_span(fn.returns)[1], "")
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and _uses(node.annotation, names):
            if node.value is None:
                edits.replace(*edits.node_span(node), "pass")
            else:
                edits.replace(
                    edits.node_span(node.target)[1],
                    edits.node_span(node.annotation)[1],
                    "",
                )
    return touched


def _unavailable_names(
    tree: ast.Module, unavailable: Callable[[str], bool]
) -> set[str]:
    """Names *tree* binds to modules a binary does not have (or to their names)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if unavailable(a.name):
                    names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if unavailable(node.module):
                names.update(a.asname or a.name for a in node.names)
    return names


def rewrite_for_runtime(
    source: str,
    unavailable: Callable[[str], bool] | None = None,
    context: ModuleContext | None = None,
) -> str:
    """The structural rewrites of the module docstring, on *source*.

    *unavailable* says which imported modules a binary does not have: an
    annotation naming one of their names is dropped, as one naming a
    TYPE_CHECKING-only import is.
    """
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    dropped = _type_checking_names(
        tree, edits, context.kept if context else frozenset()
    )
    if unavailable is not None:
        dropped |= _unavailable_names(tree, unavailable)
    touched = _drop_annotations(tree, edits, dropped)
    _carla_enum_annotations(tree, edits)
    _declare_attributes(tree, edits, dropped, context)
    if context is not None:
        _class_attributes(tree, edits, context)
    _typed_factories(tree, edits)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            # Codon's builtins: imported from the typing stand-in they would
            # name something else (a Union field then cannot be typed).
            kept = [a for a in node.names if a.name not in _CODON_BUILTIN_TYPES]
            if len(kept) != len(node.names):
                text = (
                    "from typing import "
                    + ", ".join(
                        a.name + (f" as {a.asname}" if a.asname else "") for a in kept
                    )
                    if kept
                    else "pass"
                )
                edits.replace(*edits.node_span(node), text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "collections.abc":
            # Codon has no collections.abc; the typing stand-in names the same ABCs.
            start, end = edits.node_span(node)
            names = ", ".join(
                a.name + (f" as {a.asname}" if a.asname else "") for a in node.names
            )
            edits.replace(start, end, f"from typing import {names}")
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        _rewrite_enum(cls, edits)
        for deco in cls.decorator_list:
            if _dotted(deco) in _CLASS_DECORATORS_DROPPED:
                edits.replace(*_decorator_span(edits, deco), "")
        for base in cls.bases:
            if _dotted(base) in _ABC:
                start, end = edits.node_span(base)
                data = edits.data  # bytes: the offsets are byte offsets
                if len(cls.bases) == 1 and not cls.keywords:
                    start = data.rindex(b"(", 0, start)  # `(ABC)` -> nothing
                    end = data.index(b")", end) + 1
                elif data[end:].lstrip().startswith(b","):
                    end = data.index(b",", end) + 1  # `ABC, Other` -> `Other`
                else:
                    start = data.rindex(b",", 0, start)  # `Other, ABC` -> `Other`
                edits.replace(start, end, "")
        _qualify_class_defaults(cls, edits)
        for stmt in cls.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "__slots__"
            ):
                edits.replace(*edits.node_span(stmt), "pass")
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in stmt.decorator_list:
                name = _dotted(deco)
                if name in _ABSTRACT:
                    edits.replace(*_decorator_span(edits, deco), "")
                elif name == "classmethod":
                    edits.replace(*_decorator_span(edits, deco), "@staticmethod")
                    first = stmt.args.posonlyargs + stmt.args.args
                    if first:
                        cls_arg = first[0]
                        start, end = edits.node_span(cls_arg)
                        # `cls, x` -> `x`; `cls` alone -> `` (byte offsets)
                        data = edits.data
                        if data[end:].lstrip().startswith(b","):
                            end = data.index(b",", end) + 1
                            while data[end : end + 1] == b" ":
                                end += 1
                        edits.replace(start, end, "")
                        for node in ast.walk(stmt):
                            if (
                                isinstance(node, ast.Name)
                                and node.id == cls_arg.arg
                                and node is not cls_arg
                            ):
                                edits.replace(*edits.node_span(node), cls.name)
    for fn in [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        _rewrite_parameters(fn, edits, touched)
    if context is not None:
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name in context.exceptions:
                _static_exception(stmt, edits)
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and context.class_lines.get(stmt.name):
                # one edit: several at one offset would come out reversed
                indent = " " * stmt.body[0].col_offset
                joined = (INSERTED + "\n" + indent).join(context.class_lines[stmt.name])
                _insert_before(edits, stmt.body[0], joined)
    return edits.apply()


def transform_for_runtime(
    source: str,
    unavailable: Callable[[str], bool] | None = None,
    context: ModuleContext | None = None,
) -> str:
    """Rewrite a framework or scenario module for a standalone build.

    *context* is what the rest of the program adds to it (:mod:`.hierarchy`).
    """
    structural = rewrite_for_runtime(
        source if source.endswith("\n") else source + "\n", unavailable, context
    )
    checked = transform_source(structural)
    if not checked.startswith(PRELUDE):
        raise AssertionError("transform_source() no longer starts with PRELUDE")
    prelude = RUNTIME_PRELUDE
    if context is not None and (context.imports or context.hoisted):
        # on the prelude's line: the author's lines keep their numbers
        extra = context.imports + context.hoisted
        prelude = prelude[:-1] + "; " + "; ".join(extra) + "\n"
    return prelude + rewrite_or(rewrite_defaults(checked[len(PRELUDE) :]))
