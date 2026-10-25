"""Hints for a local that Codon typed by its first assignment (#583, #584).

Codon fixes the type of an unannotated local at its first assignment, where
Python and mypy let it widen.  Two patterns of scenario code then fail with an
error that never names that assignment:

* a local first bound to an int literal (``total = 0``) that a float is later
  stored in: ``'float' does not match expected type 'int'``; likewise a
  conditional expression with an int literal branch (``v if v is not None
  else 0``): ``'Optional[float]' does not match expected type
  'Optional[int]'``;
* a local first bound to ``None`` and later rebound (``best = None`` of a
  running minimum): ``'Optional[float]' does not match expected type
  'Optional[NoneType]'``, or Codon 0.19 crashing with no diagnostic at all.

Everything here reads the author's source only (the AST of a checked module);
the check keeps refusing, and these name the assignment to change.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_OTHER_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

#: Codon's refusal of a float where an int literal fixed the type (#583).
_INT_MISMATCH = re.compile(
    r"^'(?P<got>[^']*float[^']*)' does not match expected type "
    r"'(?P<want>int|Optional\[int\])'"
)
#: Codon's refusal of a value where a ``None`` fixed the type (#584).
_NONE_MISMATCH = re.compile(
    r"^'(?:Optional\[(?P<optional>[^']*)\]|(?P<got>[^']*))' does not match "
    r"expected type 'Optional\[NoneType\]'"
)


@dataclass(frozen=True)
class _Binding:
    name: str
    line: int
    column: int
    #: The value bound (``None`` for a binding with no single value: a loop
    #: target, an augmented assignment, ``with ... as``).
    value: ast.expr | None
    annotated: bool = False
    augmented: bool = False


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node of *scope*'s own body: not those of a nested function,
    class, lambda or comprehension (other scopes)."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, _OTHER_SCOPES):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _bindings(scope: ast.AST) -> list[_Binding]:
    """The bindings of plain names in *scope*'s own body, in source order."""
    found: list[_Binding] = []
    for node in _own_nodes(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.append(
                        _Binding(target.id, node.lineno, target.col_offset, node.value)
                    )
                else:
                    found.extend(
                        _Binding(n.id, n.lineno, n.col_offset, None)
                        for n in ast.walk(target)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.append(
                _Binding(
                    node.target.id,
                    node.lineno,
                    node.target.col_offset,
                    node.value,
                    annotated=True,
                )
            )
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            found.append(
                _Binding(
                    node.target.id,
                    node.lineno,
                    node.target.col_offset,
                    node.value,
                    augmented=True,
                )
            )
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.withitem, ast.NamedExpr)):
            bound: ast.expr | None = (
                node.optional_vars if isinstance(node, ast.withitem) else node.target
            )
            if bound is None:
                continue
            value = node.value if isinstance(node, ast.NamedExpr) else None
            found.extend(
                _Binding(n.id, n.lineno, n.col_offset, value)
                for n in ast.walk(bound)
                if isinstance(n, ast.Name)
            )
    found.sort(key=lambda b: (b.line, b.column))
    return found


def _declared_elsewhere(scope: ast.AST) -> set[str]:
    """Names *scope* declares ``global`` or ``nonlocal``: not its locals."""
    return {
        name
        for node in _own_nodes(scope)
        if isinstance(node, (ast.Global, ast.Nonlocal))
        for name in node.names
    }


def _scope_at(tree: ast.Module, line: int) -> ast.AST:
    """The innermost function holding *line*, or the module."""
    best: ast.AST = tree
    for node in ast.walk(tree):
        if (
            isinstance(node, _SCOPES)
            and node.lineno <= line <= (node.end_lineno or node.lineno)
            and (not isinstance(best, _SCOPES) or node.lineno >= best.lineno)
        ):
            best = node
    return best


def _parameters(scope: ast.AST) -> set[str]:
    """The parameter names of *scope*, a function (none for the module)."""
    if not isinstance(scope, _SCOPES):
        return set()
    params = scope.args
    return {
        arg.arg
        for arg in [
            *params.posonlyargs,
            *params.args,
            *params.kwonlyargs,
            *filter(None, (params.vararg, params.kwarg)),
        ]
    }


def _int_literal(node: ast.expr | None) -> str | None:
    """The source of *node* when it is an int literal (``0``, ``-1``)."""
    inner = node
    if isinstance(inner, ast.UnaryOp) and isinstance(inner.op, (ast.USub, ast.UAdd)):
        inner = inner.operand
    if (
        isinstance(inner, ast.Constant)
        and isinstance(inner.value, int)
        and not isinstance(inner.value, bool)
    ):
        return ast.unparse(node) if node is not None else None
    return None


def _int_branch(node: ast.expr | None) -> str | None:
    """The int literal branch of a conditional expression *node*, when its
    other branch is no int literal."""
    if not isinstance(node, ast.IfExp):
        return None
    body, orelse = _int_literal(node.body), _int_literal(node.orelse)
    if (body is None) == (orelse is None):
        return None
    return body if body is not None else orelse


def _float_of(literal: str) -> str:
    return f"{literal}.0"


def int_literal_hint(message: str, line: int, tree: ast.Module) -> str | None:
    """Why Codon refuses a float at *line* where a local was first bound to an
    int literal, or a conditional expression has an int literal branch (#583).

    *message* is Codon's (``'float' does not match expected type 'int'``),
    *tree* the author's module holding *line*.  ``None`` when *line* is
    neither.
    """
    if _INT_MISMATCH.match(message) is None:
        return None
    scope = _scope_at(tree, line)
    bindings = _bindings(scope)
    # A parameter is typed by its annotation or its caller, not by a literal.
    elsewhere = _declared_elsewhere(scope) | _parameters(scope)
    first: dict[str, _Binding] = {}
    for binding in bindings:
        first.setdefault(binding.name, binding)
    for binding in bindings:
        if binding.line != line or binding.name in elsewhere:
            continue
        start = first[binding.name]
        if start.annotated:
            continue
        literal = _int_literal(start.value)
        if literal is not None and start.line < line:
            # With a call in the value, the refused float may be one of its
            # arguments instead: say which case the hint is for.
            when = (
                f"if the float is the value stored in `{binding.name}` here, "
                if binding.value is not None
                and any(isinstance(n, ast.Call) for n in ast.walk(binding.value))
                else ""
            )
            return (
                f"{when}Codon fixes the type of the local `{binding.name}` at "
                f"its first assignment, `{binding.name} = {literal}` on line "
                f"{start.line}, which makes it an int: write "
                f"`{binding.name} = {_float_of(literal)}` to store a float in it"
            )
        branch = _int_branch(start.value)
        if branch is not None:
            where = "here" if start.line == line else f"on line {start.line}"
            return (
                f"Codon fixes the type of the local `{binding.name}` at its "
                f"first assignment ({where}), a conditional expression whose "
                f"int literal branch `{branch}` makes it an int: write "
                f"`{_float_of(branch)}`"
            )
    # Not assigned: a conditional expression elsewhere on the line.
    for node in _own_nodes(scope):
        if isinstance(node, ast.IfExp) and node.lineno <= line <= (
            node.end_lineno or node.lineno
        ):
            branch = _int_branch(node)
            if branch is not None:
                return (
                    f"Codon types a conditional expression by its first "
                    f"branch, and the int literal `{branch}` here makes it an "
                    f"int: write `{_float_of(branch)}`"
                )
    return None


@dataclass(frozen=True)
class NoneFirstBinding:
    """A local first bound to ``None`` with no annotation, and rebound."""

    name: str
    #: The line of its ``name = None``.
    line: int
    #: The lines it is rebound on.
    rebound: tuple[int, ...]
    #: The type it is rebound to, where the source says (``float``).
    rebound_type: str | None

    def annotation(self, rebound_type: str | None = None) -> str:
        """The annotated first binding to write instead."""
        kind = rebound_type or self.rebound_type or "<T>"
        return f"{self.name}: {kind} | None = None"


def _is_none(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


_CONTAINERS = frozenset(
    {
        "list",
        "List",
        "Sequence",
        "MutableSequence",
        "Iterable",
        "Iterator",
        "Collection",
        "set",
        "Set",
        "frozenset",
        "FrozenSet",
        "AbstractSet",
        "deque",
    }
)


def _element(annotation: ast.expr) -> str | None:
    """The element type of a container annotation (``list[float]``)."""
    if not isinstance(annotation, ast.Subscript):
        return None
    head = annotation.value
    name = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", "")
    if name not in _CONTAINERS or isinstance(annotation.slice, ast.Tuple):
        return None
    return ast.unparse(annotation.slice)


def _annotations(scope: ast.AST) -> dict[str, ast.expr]:
    """The annotated parameters and locals of *scope*."""
    found: dict[str, ast.expr] = {}
    if isinstance(scope, _SCOPES):
        params = scope.args
        for arg in [*params.posonlyargs, *params.args, *params.kwonlyargs]:
            if arg.annotation is not None:
                found[arg.arg] = arg.annotation
    for node in _own_nodes(scope):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.setdefault(node.target.id, node.annotation)
    return found


def _type_of(
    binding: _Binding, scope: ast.AST, annotations: dict[str, ast.expr]
) -> str | None:
    """The type *binding* binds, where its source says it plainly."""
    value = binding.value
    if isinstance(value, ast.Constant) and value.value is not None:
        return type(value.value).__name__
    if isinstance(value, ast.Name) and value.id in annotations:
        return ast.unparse(annotations[value.id])
    if value is None:
        # A loop target over an annotated container: `for x in xs`.
        for node in _own_nodes(scope):
            if (
                isinstance(node, (ast.For, ast.AsyncFor))
                and isinstance(node.target, ast.Name)
                and node.target.id == binding.name
                and node.lineno == binding.line
                and isinstance(node.iter, ast.Name)
                and node.iter.id in annotations
            ):
                return _element(annotations[node.iter.id])
        return None
    if isinstance(value, ast.Name):
        # A loop variable of an annotated container: `best = x`.
        for node in _own_nodes(scope):
            if (
                isinstance(node, (ast.For, ast.AsyncFor))
                and isinstance(node.target, ast.Name)
                and node.target.id == value.id
                and isinstance(node.iter, ast.Name)
                and node.iter.id in annotations
            ):
                return _element(annotations[node.iter.id])
    return None


def none_first_bindings(tree: ast.Module) -> list[NoneFirstBinding]:
    """Each local of *tree*'s functions first bound to ``None`` with no
    annotation, and rebound to something else later (#584)."""
    found: list[NoneFirstBinding] = []
    for scope in ast.walk(tree):
        if not isinstance(scope, _SCOPES):
            continue
        bindings = _bindings(scope)
        elsewhere = _declared_elsewhere(scope)
        annotations = _annotations(scope)
        by_name: dict[str, list[_Binding]] = {}
        for binding in bindings:
            by_name.setdefault(binding.name, []).append(binding)
        for name, each in by_name.items():
            start = each[0]
            if (
                name in elsewhere
                or name in annotations
                or start.annotated
                or start.augmented
                or not _is_none(start.value)
            ):
                continue
            later = [b for b in each[1:] if not _is_none(b.value)]
            if not later:
                continue
            kinds = {_type_of(b, scope, annotations) for b in later}
            found.append(
                NoneFirstBinding(
                    name,
                    start.line,
                    tuple(b.line for b in later),
                    kinds.pop() if len(kinds) == 1 else None,
                )
            )
    found.sort(key=lambda f: f.line)
    return found


def none_mismatch_hint(
    message: str, line: int, tree: ast.Module
) -> tuple[str, int] | None:
    """Why Codon refuses a value of a local it typed ``Optional[NoneType]``
    at *line* (#584), and the line of that local's ``= None``."""
    found = _NONE_MISMATCH.match(message)
    if found is None:
        return None
    got = found["optional"] if found["optional"] is not None else found["got"]
    for codon, python in (("List[", "list["), ("Dict[", "dict["), ("Set[", "set[")):
        got = got.replace(codon, python)
    got = got.replace("Tuple[", "tuple[")
    for binding in none_first_bindings(tree):
        if line in binding.rebound:
            return (
                none_first_binding_message(binding, got or None),
                binding.line,
            )
    return None


def none_first_binding_message(
    binding: NoneFirstBinding, rebound_type: str | None = None
) -> str:
    return (
        f"Codon fixes the type of the local `{binding.name}` at its first "
        f"assignment, `{binding.name} = None` on line {binding.line}, and "
        f"cannot widen it when it is rebound (line"
        f"{'s' if len(binding.rebound) > 1 else ''} "
        f"{', '.join(map(str, binding.rebound))}): annotate it: "
        f"`{binding.annotation(rebound_type)}`"
    )
