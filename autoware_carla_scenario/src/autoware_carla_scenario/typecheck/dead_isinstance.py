"""An ``isinstance`` branch the check does not compile (#565).

Codon decides ``isinstance(x, T)`` at compile time.  When the static type of
``x`` cannot be ``T`` it drops the branch before typing it, so an error in
the body (``q: int = "wrong"``, ``x.nonexistent()``) is not reported.  Python
may well take the branch (a ``float`` parameter that is given an ``int``), so
the check cannot refuse it; it warns that the branch is not checked.

Only a subject whose static type is certain from the syntax is considered:

* a parameter or a local with a concrete annotation (``int``, ``float``,
  ``str``, ``bool``, ``bytes``, ``list[..]``, ``dict[..]``, ``set[..]``,
  ``tuple[..]``, a class of the module);
* a local bound exactly once, to a literal, an f-string or a display;
* ``self.<attr>`` with a class-level annotation of the enclosing class.

The classes may be a name or a tuple of names, every one of them a builtin
or a class of the module.  A subject bound more than once, a ``Union`` /
``Optional`` / ``Any`` / ``object`` annotation, or a class the module does not
define (an import, an ABC) is skipped, and so is a class that is a subclass
of the subject's type (``bool`` is an ``int``).
"""

from __future__ import annotations

import ast
from typing import NamedTuple

__all__ = ["DeadIsinstance", "dead_isinstances"]


class DeadIsinstance(NamedTuple):
    line: int
    column: int
    message: str


# Each builtin and the builtins an instance of it is also an instance of.
_BUILTINS: dict[str, tuple[str, ...]] = {
    "int": ("int",),
    "float": ("float",),
    "str": ("str",),
    "bool": ("bool", "int"),
    "bytes": ("bytes",),
    "list": ("list",),
    "dict": ("dict",),
    "tuple": ("tuple",),
    "set": ("set",),
}
_CONTAINERS = {"list", "dict", "set", "tuple", "List", "Dict", "Set", "Tuple"}
_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _annotation_type(
    node: ast.expr | None, classes: dict[str, list[str]]
) -> str | None:
    """The concrete type an annotation names, or ``None`` (unknown, a union)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name) and base.id in _CONTAINERS:
            return base.id.lower()
        return None
    if isinstance(node, ast.Name) and (node.id in _BUILTINS or node.id in classes):
        return node.id
    return None


def _value_type(node: ast.expr) -> str | None:
    """The type of a literal, an f-string or a display, or ``None``."""
    if isinstance(node, ast.Constant):
        name = type(node.value).__name__
        return name if name in _BUILTINS else None
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    return None


def _ancestry(name: str, classes: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    todo = [name]
    while todo:
        current = todo.pop()
        if current not in seen:
            seen.add(current)
            todo.extend(classes.get(current, []))
    return seen


def _can_be(subject: str, wanted: str, classes: dict[str, list[str]]) -> bool:
    """Whether a value of static type *subject* may be an instance of *wanted*."""
    if subject in _BUILTINS:
        if wanted in _BUILTINS:
            return wanted in _BUILTINS[subject] or subject in _BUILTINS[wanted]
        return wanted not in classes  # an unknown class: do not guess
    if wanted in classes:
        return wanted in _ancestry(subject, classes) or subject in _ancestry(
            wanted, classes
        )
    if wanted in _BUILTINS:
        return wanted in _ancestry(subject, classes)
    return True


def _names_bound(function: _Function) -> dict[str, int]:
    """How many times each name is bound in *function* (its parameters included);
    a binding that is not a plain assignment counts twice, so it is never
    taken for the only one."""
    counts: dict[str, int] = {}

    def add(name: str, n: int) -> None:
        counts[name] = counts.get(name, 0) + n

    def add_targets(target: ast.AST, n: int = 2) -> None:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                add(sub.id, n)

    for node in ast.walk(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                add(a.arg, 1 if node is function else 2)
            for extra in (args.vararg, args.kwarg):
                if extra is not None:
                    add(extra.arg, 2)
            if node is not function and not isinstance(node, ast.Lambda):
                add(node.name, 2)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                add_targets(t, 1 if isinstance(t, ast.Name) else 2)
        elif isinstance(node, ast.AnnAssign):
            add_targets(node.target, 1)
        elif isinstance(
            node, (ast.AugAssign, ast.For, ast.AsyncFor, ast.comprehension)
        ):
            add_targets(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            add_targets(node.optional_vars)
        elif isinstance(node, ast.NamedExpr):
            add_targets(node.target)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name, 2)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                add((alias.asname or alias.name).split(".")[0], 2)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                add(name, 2)
        elif isinstance(node, ast.ClassDef):
            add(node.name, 2)
    return counts


def _local_types(
    function: _Function, classes: dict[str, list[str]]
) -> dict[str, str | None]:
    types: dict[str, str | None] = {}
    args = function.args
    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        types[a.arg] = _annotation_type(a.annotation, classes)
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    types[t.id] = _value_type(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            types[node.target.id] = _annotation_type(node.annotation, classes)
    return types


def _wanted(node: ast.expr) -> list[str] | None:
    items = list(node.elts) if isinstance(node, ast.Tuple) else [node]
    if all(isinstance(i, ast.Name) for i in items) and items:
        return [i.id for i in items if isinstance(i, ast.Name)]
    return None


def dead_isinstances(tree: ast.Module) -> list[DeadIsinstance]:
    """Each ``isinstance(x, T)`` of *tree* whose subject's static type cannot be
    a ``T``, in source order."""
    classes: dict[str, list[str]] = {}
    attributes: dict[str, dict[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            attributes[node.name] = {
                s.target.id: _annotation_type(s.annotation, classes)
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            }
    found: list[DeadIsinstance] = []

    def visit(function: _Function, owner: str | None) -> None:
        bound = _names_bound(function)
        types = _local_types(function, classes)
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance"
                and len(node.args) == 2
                and not node.keywords
            ):
                continue
            subject, target = node.args
            static: str | None = None
            if isinstance(subject, ast.Name):
                if bound.get(subject.id) == 1:
                    static = types.get(subject.id)
            elif (
                isinstance(subject, ast.Attribute)
                and isinstance(subject.value, ast.Name)
                and subject.value.id == "self"
                and owner is not None
            ):
                static = attributes.get(owner, {}).get(subject.attr)
            wanted = _wanted(target)
            if static is None or wanted is None:
                continue
            if any(_can_be(static, w, classes) for w in wanted):
                continue
            text = ast.unparse(subject)
            found.append(
                DeadIsinstance(
                    node.lineno,
                    node.col_offset + 1,
                    f"{ast.unparse(node)}: {text} is statically {static} here, so "
                    "this branch is not checked by the static check; an error in "
                    "it is not reported",
                )
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(stmt, node.name)
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(stmt, None)
    return sorted(set(found))
