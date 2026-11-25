"""Python's name scoping for the passes that rewrite a builtin (#471).

A builtin such as ``format`` or ``dict`` is rewritten only where the name is
not bound by the module or an enclosing function: ``ShadowScopes`` is the
visitor mixin that tracks it.

``node_bindings`` and ``own_bindings`` are the one walk of Python's binders
the static analyses share (``local_bindings``, ``dead_isinstance``,
``static_raises``, ``return_paths``, #2513): each says how a name is bound,
so a reader that only trusts some kinds (a plain assignment) can tell them
from the rest; ``parse_string_annotation`` reads a quoted annotation.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass

_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
#: ``type X = ...`` (Python 3.12+) and the type parameters of 3.12's generic
#: syntax; absent before.
_TYPE_ALIAS: tuple[type, ...] = tuple(
    t for t in (getattr(ast, "TypeAlias", None),) if t is not None
)
_TYPE_PARAMS: tuple[type, ...] = tuple(
    t
    for t in (getattr(ast, n, None) for n in ("TypeVar", "ParamSpec", "TypeVarTuple"))
    if t is not None
)


@dataclass(frozen=True)
class Bound:
    """A name a node binds, and how.

    *kind* is one of ``assign`` (``x = v``, a plain name target), ``unpack``
    (a name inside a tuple, list or starred target), ``annotated``
    (``x: T`` / ``x: T = v``), ``augmented`` (``x += v``), ``loop`` (a ``for``
    or comprehension target), ``with`` (``with ... as x``), ``walrus``
    (``x := v``), ``except`` (``except E as x``), ``capture`` (a match
    capture, ``*x`` or ``**x`` pattern), ``import``, ``definition`` (``def`` /
    ``class``), ``alias`` (``type X = ...``), ``type parameter``,
    ``parameter``, ``delete`` (``del x``) and ``declaration`` (``global`` /
    ``nonlocal``).
    """

    name: str
    line: int
    column: int
    kind: str
    #: The value bound, where there is one: an assignment's, an annotated
    #: or augmented assignment's, a walrus's, a type alias's.
    value: ast.expr | None = None
    #: The binding node itself (the target ``Name``, the ``ExceptHandler``,
    #: the ``FunctionDef``, ...).
    node: ast.AST | None = None


def parse_string_annotation(node: ast.expr | None) -> ast.expr | None:
    """*node*, an annotation, with a quoted one (``"Optional[int]"``) parsed;
    ``None`` for a quoted one that is not an expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return ast.parse(node.value.strip(), mode="eval").body
        except SyntaxError:
            return None
    return node


def _targets(target: ast.AST, kind: str, value: ast.expr | None) -> list[Bound]:
    """The names a target stores: ``a`` of ``a``, ``(a, *b)``; not those it
    only reads (``x`` of ``x.y`` or ``x[i]``)."""
    return [
        Bound(n.id, n.lineno, n.col_offset, kind, value, n)
        for n in ast.walk(target)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    ]


def node_bindings(node: ast.AST) -> list[Bound]:
    """The names *node* itself binds, not those of the nodes under it (a
    ``for`` binds its target, not what its body assigns).  Visiting every
    node of a tree once with it finds every binding."""
    if isinstance(node, ast.Assign):
        out: list[Bound] = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                out.append(
                    Bound(
                        target.id,
                        node.lineno,
                        target.col_offset,
                        "assign",
                        node.value,
                        target,
                    )
                )
            else:
                out.extend(_targets(target, "unpack", None))
        return out
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            return [
                Bound(
                    node.target.id,
                    node.lineno,
                    node.target.col_offset,
                    "annotated",
                    node.value,
                    node.target,
                )
            ]
        return []
    if isinstance(node, ast.AugAssign):
        if isinstance(node.target, ast.Name):
            return [
                Bound(
                    node.target.id,
                    node.lineno,
                    node.target.col_offset,
                    "augmented",
                    node.value,
                    node.target,
                )
            ]
        return []
    if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        return _targets(node.target, "loop", None)
    if isinstance(node, ast.withitem):
        if node.optional_vars is None:
            return []
        return _targets(node.optional_vars, "with", None)
    if isinstance(node, ast.NamedExpr):
        return _targets(node.target, "walrus", node.value)
    if isinstance(node, ast.Delete):
        out = []
        for target in node.targets:
            out.extend(_targets(target, "delete", None))
        return out
    if isinstance(node, ast.ExceptHandler):
        if not node.name:
            return []
        return [Bound(node.name, node.lineno, node.col_offset, "except", None, node)]
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        if not node.name:
            return []
        return [Bound(node.name, node.lineno, node.col_offset, "capture", None, node)]
    if isinstance(node, ast.MatchMapping):
        if not node.rest:
            return []
        return [Bound(node.rest, node.lineno, node.col_offset, "capture", None, node)]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [
            Bound(
                (alias.asname or alias.name).split(".")[0],
                node.lineno,
                node.col_offset,
                "import",
                None,
                node,
            )
            for alias in node.names
            if alias.name != "*"
        ]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [
            Bound(node.name, node.lineno, node.col_offset, "definition", None, node)
        ]
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return [
            Bound(name, node.lineno, node.col_offset, "declaration", None, node)
            for name in node.names
        ]
    if isinstance(node, ast.arg):
        return [Bound(node.arg, node.lineno, node.col_offset, "parameter", None, node)]
    if _TYPE_ALIAS and isinstance(node, _TYPE_ALIAS):
        name = getattr(node, "name", None)
        if isinstance(name, ast.Name):
            return [
                Bound(
                    name.id,
                    name.lineno,
                    name.col_offset,
                    "alias",
                    getattr(node, "value", None),
                    name,
                )
            ]
        return []
    if _TYPE_PARAMS and isinstance(node, _TYPE_PARAMS):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            return [
                Bound(
                    name,
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                    "type parameter",
                    None,
                    node,
                )
            ]
    return []


def _own_scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes of *scope*'s own body whose bindings are *scope*'s: not
    the bodies of the functions, classes and lambdas it defines (their names
    are its own), and of a comprehension only the walrus targets."""
    body = getattr(scope, "body", [])
    stack: list[ast.AST] = list(body) if isinstance(body, list) else [body]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _NESTED_SCOPES):
            continue
        if isinstance(node, _COMPREHENSIONS):
            # Its loop targets are its own; a walrus binds in *scope*.
            inner: list[ast.AST] = list(ast.iter_child_nodes(node))
            while inner:
                sub = inner.pop()
                if isinstance(sub, _NESTED_SCOPES):
                    continue
                if isinstance(sub, ast.NamedExpr):
                    yield sub
                inner.extend(ast.iter_child_nodes(sub))
            continue
        stack.extend(ast.iter_child_nodes(node))


def own_bindings(scope: ast.AST) -> list[Bound]:
    """Every binding of *scope*'s own body (a module, a function, a lambda or
    a class), in source order: assignments of every kind, ``except ... as``,
    match captures, imports, nested ``def`` / ``class`` names, a walrus inside
    a comprehension, ``del`` and ``global`` / ``nonlocal``.  Not the scope's
    parameters, nor what nested functions, classes, lambdas and
    comprehensions bind for themselves."""
    found = [b for node in _own_scope_nodes(scope) for b in node_bindings(node)]
    found.sort(key=lambda b: (b.line, b.column))
    return found


def _scope_bindings(scope: ast.AST) -> set[str]:
    """The names *scope* (a module, function or lambda) binds in its own
    scope: its parameters, variables, imports, and the names of the functions
    and classes it defines.  The bodies of nested functions, lambdas and
    classes are other scopes (a method name never binds in the scope of its
    class's module, a class attribute is not read from a method as a name)."""
    names: set[str] = set()
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, _COMPREHENSIONS):
            # Its loop targets are its own; a walrus binds in *scope*.
            names.update(
                n.target.id
                for n in ast.walk(node)
                if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name)
            )
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        else:
            for attr in ("name", "rest"):  # match: `case x`, `case *x`, `case {**x}`
                value = getattr(node, attr, None)
                if isinstance(value, str) and type(node).__name__.startswith("Match"):
                    names.add(value)
        stack.extend(ast.iter_child_nodes(node))
    return names


class ShadowScopes(ast.NodeVisitor):
    """A visitor that knows which names its scopes bind.  Call
    ``enter_module`` from ``visit_Module``; ``_shadowed`` then answers for the
    node being visited."""

    def __init__(self) -> None:
        #: The names the module binds at its own level, and those each
        #: enclosing function or lambda binds: a builtin of that name is
        #: shadowed only where one of them is in scope.
        self._module_bound: set[str] = set()
        #: Each with whether it is a class body, whose names only its own
        #: statements read: a method or a comprehension in it skips it.
        self._local_bound: list[tuple[set[str], bool]] = []

    def enter_module(self, node: ast.Module) -> None:
        self._module_bound = _scope_bindings(node)
        # `global format` in a function binds the module's name.
        for inner in ast.walk(node):
            if isinstance(inner, ast.Global):
                self._module_bound.update(inner.names)

    def _shadowed(self, name: str) -> bool:
        """Whether *name* is bound by the module or an enclosing function, so
        a builtin of that name is not the one in scope."""
        if name in self._module_bound:
            return True
        for index, (bound, is_class) in enumerate(reversed(self._local_bound)):
            if name in bound and (index == 0 or not is_class):
                return True
        return False

    def visit(self, node: ast.AST) -> None:
        scope: tuple[set[str], bool] | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scope = (_scope_bindings(node), False)
        elif isinstance(node, ast.ClassDef):
            scope = (_scope_bindings(node), True)
        elif isinstance(node, _COMPREHENSIONS):
            scope = (
                {
                    n.id
                    for g in node.generators
                    for n in ast.walk(g.target)
                    if isinstance(n, ast.Name)
                },
                False,
            )
        if scope is not None:
            self._local_bound.append(scope)
            try:
                super().visit(node)
            finally:
                self._local_bound.pop()
        else:
            super().visit(node)
