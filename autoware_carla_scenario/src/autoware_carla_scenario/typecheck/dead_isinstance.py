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
* a local bound exactly once, to a literal, an f-string or a display, and
  by no other binder (:func:`.scopes.node_bindings`: ``except ... as``, a
  match capture, an import, a ``def`` / ``class``, a walrus);
* ``self.<attr>`` with a class-level annotation of the enclosing class.

A class of the module is known by its name only where no other class of the
module has it (a function's own ``class Holder`` beside the module's names
neither), and none derived from such a name is known at all.

The classes may be a name or a tuple of names, every one of them a builtin
or a class of the module.  A subject bound more than once, a ``Union`` /
``Optional`` / ``Any`` / ``object`` annotation, or a class the module does not
define (an import, an ABC) is skipped, and so is a class that is a subclass
of the subject's type (``bool`` is an ``int``).
"""

from __future__ import annotations

import ast
from typing import NamedTuple

from .scopes import node_bindings, parse_string_annotation

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
    node = parse_string_annotation(node)
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
    """How many times each name is bound in *function* (its parameters
    included), nested functions, lambdas and comprehensions too; a binding
    that is not a plain assignment (:func:`.scopes.node_bindings`) counts
    twice, so it is never taken for the only one."""
    counts: dict[str, int] = {}

    def add(name: str, n: int) -> None:
        counts[name] = counts.get(name, 0) + n

    args = function.args
    own = {id(a) for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        add(a.arg, 1)
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            own.add(id(extra))
            add(extra.arg, 2)
    for node in ast.walk(function):
        if node is function or id(node) in own:
            continue
        for bound in node_bindings(node):
            add(bound.name, 1 if bound.kind in ("assign", "annotated") else 2)
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
    # By name, as annotations name them: a name two classes of the module
    # share (a function's own `class Holder` beside the module's) names
    # neither for certain, so it is left out (#2513).
    defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    names = [node.name for node in defs]
    classes: dict[str, list[str]] = {
        node.name: [b.id for b in node.bases if isinstance(b, ast.Name)]
        for node in defs
        if names.count(node.name) == 1
    }
    # Nor is a class derived from such a name: its ancestry is not known
    # (`class A(Holder)` may be an `int` through either `Holder`).
    shared = {name for name in names if names.count(name) > 1}
    while unknown := [
        name
        for name, bases in classes.items()
        if any(b in shared or (b in names and b not in classes) for b in bases)
    ]:
        for name in unknown:
            del classes[name]
    # By the class itself: `self.x` reads the enclosing class's own.
    attributes: dict[int, dict[str, str | None]] = {
        id(node): {
            s.target.id: _annotation_type(s.annotation, classes)
            for s in node.body
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
        }
        for node in defs
    }
    found: list[DeadIsinstance] = []

    def visit(function: _Function, owner: ast.ClassDef | None) -> None:
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
                static = attributes.get(id(owner), {}).get(subject.attr)
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

    for node in defs:
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(stmt, node)
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(stmt, None)
    return sorted(set(found))
