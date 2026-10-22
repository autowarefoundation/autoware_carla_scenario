"""Python's name scoping for the passes that rewrite a builtin (#471).

A builtin such as ``format`` or ``dict`` is rewritten only where the name is
not bound by the module or an enclosing function: ``ShadowScopes`` is the
visitor mixin that tracks it.
"""

from __future__ import annotations

import ast

_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


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
