"""The two ways a function's declared return type goes unenforced (#529).

Codon coerces the value of a ``return`` to a declared ``-> bool`` (any type
with ``__bool__`` converts) and compiles a function that falls off its end
on one path although it declares a return type.  Python returns the str, or
``None``, there.  Both are decided from the syntax alone:

* ``return "no"`` (a literal, an f-string or a display: its type is certain)
  in a function declared ``-> bool``;
* a function declared to return a type that is not ``None``, ``Optional``,
  ``Any`` or ``NoReturn`` whose body can reach its end.  Control flow is
  followed through ``if``, ``while``, ``for``, ``try``, ``with`` and
  ``match`` (a match without a wildcard case is
  taken to be exhaustive); a ``while True`` without a ``break``, ``return``, ``raise``,
  ``assert False`` and a call of ``sys.exit`` or of a function declared
  ``NoReturn`` end a path.  Anything not understood is taken to end the path,
  so the check never refuses a function Python and mypy accept.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from typing import NamedTuple, cast


class ReturnProblem(NamedTuple):
    path: str
    line: int
    column: int
    message: str


#: Calls that never return, whichever object they are made on.
_NO_RETURN_CALLS = frozenset(
    {"exit", "_exit", "abort", "assert_never", "quit", "fail", "skip", "xfail"}
)
_NO_RETURN_TYPES = frozenset({"NoReturn", "Never"})
_ANY_TYPES = frozenset({"Any", "object"})

_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _annotation(function: _Function) -> ast.expr | None:
    node = function.returns
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    return node


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _may_be_none(node: ast.expr) -> bool:
    """Whether the declared type allows ``None`` or is not decidable."""
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, str)
    name = _name(node)
    if name in _NO_RETURN_TYPES or name in _ANY_TYPES or name == "Optional":
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _may_be_none(node.left) or _may_be_none(node.right)
    if isinstance(node, ast.Subscript):
        outer = _name(node.value)
        if outer in ("Optional", "Awaitable", "Coroutine"):
            return True
        if outer == "Union":
            members = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            return any(_may_be_none(m) for m in members)
        if outer in ("Annotated", "Final"):
            return True  # not followed
    return False


def _walk_own(nodes: list[ast.stmt]) -> Iterator[ast.AST]:
    """Every node of *nodes* outside the functions, classes and lambdas they
    define."""
    stack: list[ast.AST] = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                stack.append(child)


def _breaks(body: list[ast.stmt]) -> bool:
    """Whether a ``break`` in *body* leaves the loop that holds *body*."""
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Break):
            return True
        if isinstance(node, (ast.While, ast.For, ast.AsyncFor)):
            # a break of a nested loop ends that loop; its else branch
            # still belongs to this one
            stack.extend(node.orelse)
            continue
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _truthy(test: ast.expr) -> bool:
    return isinstance(test, ast.Constant) and bool(test.value)


def _falsy(test: ast.expr) -> bool:
    return isinstance(test, ast.Constant) and not test.value


class _Flow:
    def __init__(self, no_return: frozenset[str]) -> None:
        self.no_return = no_return

    def ends(self, expr: ast.expr) -> bool:
        if isinstance(expr, ast.Await):
            expr = expr.value
        if isinstance(expr, ast.Call):
            name = _name(expr.func)
            return name in _NO_RETURN_CALLS or name in self.no_return
        return False

    def falls(self, body: list[ast.stmt]) -> bool:
        """Whether execution can leave *body* by running to its end."""
        for stmt in body:
            if not self.falls_stmt(stmt):
                return False
        return True

    def falls_stmt(self, stmt: ast.stmt) -> bool:
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return False
        if isinstance(stmt, ast.Assert):
            return not _falsy(stmt.test)
        if isinstance(stmt, ast.Expr):
            return not self.ends(stmt.value)
        if isinstance(stmt, ast.If):
            if _truthy(stmt.test):
                return self.falls(stmt.body)
            if _falsy(stmt.test):
                return self.falls(stmt.orelse)
            return self.falls(stmt.body) or self.falls(stmt.orelse)
        if isinstance(stmt, ast.While):
            if _truthy(stmt.test):
                return _breaks(stmt.body)
            return self.falls(stmt.orelse) or _breaks(stmt.body)
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            return self.falls(stmt.orelse) or _breaks(stmt.body)
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            return self.falls(stmt.body)
        if type(stmt).__name__ in ("Try", "TryStar"):  # TryStar: Python 3.11+
            try_ = cast(ast.Try, stmt)
            body = self.falls(try_.body) and self.falls(try_.orelse)
            handlers = any(self.falls(h.body) for h in try_.handlers)
            if try_.finalbody and not self.falls(try_.finalbody):
                return False
            return body or handlers
        if isinstance(stmt, ast.Match):
            # Without a wildcard case the match may still be exhaustive (an
            # enum, a union), which mypy accepts: only a case body that falls
            # off the end counts.
            return any(self.falls(c.body) for c in stmt.cases)
        return True


def _stub(function: _Function) -> bool:
    """A body that is only a docstring, ``...`` or ``pass``: an abstract
    method, an overload or a protocol member."""
    return all(
        isinstance(s, ast.Pass)
        or (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Constant)
            and (s.value.value is Ellipsis or isinstance(s.value.value, str))
        )
        for s in function.body
    )


def _certain_type(expr: ast.expr | None) -> str | None:
    """The type of a returned expression whose type is certain, as written."""
    if expr is None:
        return None
    if isinstance(expr, ast.Constant):
        value = expr.value
        if isinstance(value, bool) or value is None or value is Ellipsis:
            return None
        return type(value).__name__
    if isinstance(expr, ast.JoinedStr):
        return "str"
    if isinstance(expr, ast.List):
        return "list"
    if isinstance(expr, ast.Dict):
        return "dict"
    if isinstance(expr, ast.Set):
        return "set"
    if isinstance(expr, ast.IfExp):
        # certain only when both branches are certain, and of the same type
        body = _certain_type(expr.body)
        return body if body is not None and body == _certain_type(expr.orelse) else None
    return None


def _no_return_functions(trees: list[ast.Module]) -> frozenset[str]:
    names = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ann = _annotation(node)
                if ann is not None and _name(ann) in _NO_RETURN_TYPES:
                    names.add(node.name)
    return frozenset(names)


def return_problems(
    modules: Mapping[str, tuple[str, ast.Module]],
) -> list[ReturnProblem]:
    """The functions of *modules* (name -> (path, tree)) whose declared return
    type is not kept."""
    flow = _Flow(_no_return_functions([tree for _, tree in modules.values()]))
    problems: list[ReturnProblem] = []
    for _, (path, tree) in sorted(modules.items()):
        owners = {
            id(member): cls.name
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef)
            for member in cls.body
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # ``XScenario.f``, as the author calls a method
            shown = (
                f"{owners[id(node)]}.{node.name}" if id(node) in owners else node.name
            )
            declared = _annotation(node)
            if declared is None:
                continue
            own = list(_walk_own(node.body))
            if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in own):
                continue
            if _name(declared) == "bool" and isinstance(declared, ast.Name):
                for stmt in own:
                    if isinstance(stmt, ast.Return):
                        found = _certain_type(stmt.value)
                        if found is not None:
                            problems.append(
                                ReturnProblem(
                                    path,
                                    stmt.lineno,
                                    stmt.col_offset + 1,
                                    f"{shown}() returns {found}, but the "
                                    "function is declared -> bool: Codon "
                                    "converts the value to a bool, Python "
                                    f"returns the {found} itself",
                                )
                            )
            if _may_be_none(declared) or _stub(node):
                continue
            if any(
                isinstance(d, (ast.Name, ast.Attribute))
                and _name(d) in ("abstractmethod", "overload")
                for d in node.decorator_list
            ):
                continue
            if flow.falls(node.body):
                problems.append(
                    ReturnProblem(
                        path,
                        node.lineno,
                        node.col_offset + 1,
                        f"{shown}() has a path that returns nothing, but "
                        f"the function is declared -> {ast.unparse(declared)}: "
                        "Python returns None there",
                    )
                )
    return problems
