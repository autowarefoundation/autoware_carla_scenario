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
  own), and so is an ``ABC`` base (a level of inheritance Codon miscompiles).
* ``@classmethod`` -> ``@staticmethod``, with ``cls`` meaning the class itself
  (Codon has no class methods).
* ``__slots__`` is dropped.
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
from collections.abc import Mapping

from ..typecheck.transform import (
    PRELUDE,
    _Edits,
    _suggest_type,
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
    "from _acs_helpers import (_acs_default, _acs_list, _acs_opt, _acs_or, _AcsEnum,"
    " _AcsRequired, _acs_unrequire)\n"
)
_ABSTRACT = {"abstractmethod", "abc.abstractmethod"}
_ABC = {"ABC", "abc.ABC"}
_ENUM = {"Enum", "enum.Enum"}
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
            if annotation is None:
                annotation = _param_annotation(value, params)
            if annotation is not None:
                types[target.attr] = ast.unparse(annotation)
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


def _declare_attributes(tree: ast.Module, edits: _Edits) -> None:
    missing: dict[str, list[str]] = {}
    for attr in undeclared_attributes(tree, {}):
        missing.setdefault(attr.class_name, []).append(attr.name)
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in missing:
            continue
        types = _attribute_types(cls)
        decls = [
            f"{name}: {types[name]}" for name in missing[cls.name] if name in types
        ]
        if decls:
            _insert_before(edits, cls.body[0], "; ".join(decls))


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
    inner = _unoptional(annotation)
    if inner is annotation:
        return None
    name = _dotted(inner)
    if not name or name in _NOT_CLASSES or not name.rsplit(".", 1)[-1][:1].isupper():
        return None
    return name


def _rewrite_parameters(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, edits: _Edits
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


def rewrite_for_runtime(source: str) -> str:
    """The structural rewrites of the module docstring, on *source*."""
    tree = ast.parse(source)
    edits = _Edits(source.encode().splitlines(keepends=True))
    _declare_attributes(tree, edits)
    _typed_factories(tree, edits)
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        _rewrite_enum(cls, edits)
        for base in cls.bases:
            if _dotted(base) in _ABC:
                start, end = edits.node_span(base)
                # `(ABC)` alone, or `ABC, ` / `, ABC` among other bases.
                text = edits.data.decode()
                if len(cls.bases) == 1:
                    while text[start - 1] != "(":
                        start -= 1
                    start -= 1
                    while text[end] != ")":
                        end += 1
                    end += 1
                elif text[end:].lstrip().startswith(","):
                    end = text.index(",", end) + 1
                else:
                    start = text.rindex(",", 0, start)
                edits.replace(start, end, "")
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
                        # `cls, x` -> `x`; `cls` alone -> ``
                        text = edits.data.decode()
                        rest = text[end:]
                        if rest.lstrip().startswith(","):
                            end = text.index(",", end) + 1
                            while text[end] == " ":
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
        _rewrite_parameters(fn, edits)
    return edits.apply()


def transform_for_runtime(source: str) -> str:
    """Rewrite a framework or scenario module for a standalone build; same lines."""
    structural = rewrite_for_runtime(source if source.endswith("\n") else source + "\n")
    checked = transform_source(structural)
    if not checked.startswith(PRELUDE):
        raise AssertionError("transform_source() no longer starts with PRELUDE")
    return RUNTIME_PRELUDE + rewrite_or(rewrite_defaults(checked[len(PRELUDE) :]))
