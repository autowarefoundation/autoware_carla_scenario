"""The two ways a function's declared return type goes unenforced (#529).

Codon coerces the value of a ``return`` to a declared ``-> bool`` (any type
with ``__bool__`` converts) and compiles a function that falls off its end
on one path although it declares a return type.  Python returns the str, or
``None``, there.  Both are decided from the syntax alone:

* ``return "no"`` (a literal, an f-string or a display: its type is certain)
  in a function declared ``-> bool``;
* a function declared to return a type that is not ``None``, ``Optional``,
  ``Any`` or ``NoReturn`` (written out, or through a module-level alias of
  the scenario's modules: ``MaybeF = Optional[float]``, ``type MaybeF =
  float | None``) whose body can reach its end.

Control flow is followed through ``if``, ``while``, ``for``, ``try``,
``with`` and ``match``.  A ``while True`` without a ``break``, ``return``,
``raise``, ``assert False`` and a call that never returns end a path: one of
``sys.exit()``, ``os._exit()``, ``os.abort()``, the ``os.exec*()`` family,
``pytest.fail()`` / ``skip()`` / ``xfail()`` / ``exit()``, ``exit()``,
``quit()``, ``assert_never()``, or of a function declared ``NoReturn``:
a module-level function of the scenario's modules called by its name (or
through an import), or a method called on ``self``, ``cls``, ``super()`` or
a parameter annotated with a class of those modules, found in that class or
its bases.  Where the receiver's class is not known (a base class from
outside those modules, a local variable), a method whose name is one of
those or of any function declared ``NoReturn`` still ends the path, as the
check cannot tell.

A ``with contextlib.suppress(...)`` block falls through when its body
raises, so it never ends a path.  A ``match`` falls through when a case body
does, and when no case can catch every subject: every case has a guard
(``case _ if y``; one that always holds, ``True`` or a ``sys.version_info``,
``sys.platform`` or ``TYPE_CHECKING`` test mypy decides, is none), or the subject is a parameter annotated ``int``, ``float``,
``str``, ``bytes`` or ``complex`` and the unguarded cases are all literals.
Otherwise a match without a wildcard case is taken to be exhaustive (an enum,
a union).  Anything not understood is taken to end the path, so the check
never refuses a function Python and mypy accept.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from typing import NamedTuple, cast

from .scopes import Bound, node_bindings, own_bindings, parse_string_annotation


class ReturnProblem(NamedTuple):
    path: str
    line: int
    column: int
    message: str


#: The calls that never return, by the dotted name they stand for.
_NO_RETURN_QUALIFIED = frozenset(
    {
        "sys.exit",
        "os._exit",
        "os.abort",
        *(
            f"os.{name}"
            for name in (
                "execl",
                "execle",
                "execlp",
                "execlpe",
                "execv",
                "execve",
                "execvp",
                "execvpe",
            )
        ),
        "pytest.fail",
        "pytest.skip",
        "pytest.xfail",
        "pytest.exit",
        "typing.assert_never",
        "typing_extensions.assert_never",
        "builtins.exit",
        "builtins.quit",
    }
)
#: The builtins of :data:`_NO_RETURN_QUALIFIED`, called by their bare name.
_NO_RETURN_BUILTINS = frozenset({"exit", "quit"})
#: Where the callee is not known (a method of an object whose class is not
#: one of the modules'), a call of one of these names still ends a path:
#: the check cannot tell it is not ``sys.exit`` or ``pytest.fail``.
_NO_RETURN_NAMES = frozenset(
    {"exit", "_exit", "abort", "assert_never", "quit", "fail", "skip", "xfail"}
)
_NO_RETURN_TYPES = frozenset({"NoReturn", "Never"})
_ANY_TYPES = frozenset({"Any", "object"})
#: The types a literal pattern cannot exhaust, as a parameter's annotation.
_OPEN_TYPES = frozenset({"int", "float", "str", "bytes", "complex"})
#: Bases that define no method a call could resolve to.
_EMPTY_BASES = frozenset(
    {
        "builtins.object",
        "typing.Generic",
        "typing.Protocol",
        "typing_extensions.Protocol",
        "abc.ABC",
    }
)
_SUPPRESS = "contextlib.suppress"
#: The attributes of a test mypy decides from its own settings.
_MYPY_DECIDED = frozenset({"version_info", "platform", "TYPE_CHECKING"})
#: How deep an alias of an alias is followed.
_ALIAS_DEPTH = 8

_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _annotation(function: _Function) -> ast.expr | None:
    return parse_string_annotation(function.returns)


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _Alias(NamedTuple):
    value: ast.expr
    #: The type parameters of ``type X[T] = ...``: any type, ``None`` too.
    params: frozenset[str]


def _may_be_none(
    node: ast.expr | None,
    aliases: Mapping[str, list[_Alias]] | None = None,
    depth: int = 0,
    params: frozenset[str] = frozenset(),
) -> bool:
    """Whether the declared type allows ``None`` or is not decidable.

    *aliases* are the module-level type aliases in scope (name -> each value
    it is bound to); a name bound to one that may be ``None`` may be too."""
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, str)
    if depth > _ALIAS_DEPTH:
        return True

    def alias(name: str | None) -> bool:
        found = (aliases or {}).get(name or "", [])
        return any(_may_be_none(a.value, aliases, depth + 1, a.params) for a in found)

    name = _name(node)
    if name in _NO_RETURN_TYPES or name in _ANY_TYPES or name == "Optional":
        return True
    if isinstance(node, ast.Name) and (node.id in params or alias(node.id)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _may_be_none(node.left, aliases, depth, params) or _may_be_none(
            node.right, aliases, depth, params
        )
    if isinstance(node, ast.Subscript):
        outer = _name(node.value)
        if outer in ("Optional", "Awaitable", "Coroutine"):
            return True
        if outer == "Union":
            members = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            return any(_may_be_none(m, aliases, depth, params) for m in members)
        if outer in ("Annotated", "Final"):
            return True  # not followed
        if isinstance(node.value, ast.Name) and alias(node.value.id):
            return True  # a generic alias: `type Opt[T] = T | None`
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


def _narrowing_guard(guard: ast.expr | None) -> bool:
    """Whether a case's *guard* may fail.  ``True`` does not, and a test mypy
    decides itself (``sys.version_info``, ``sys.platform``,
    ``TYPE_CHECKING``) is taken not to, as mypy may hold it always true."""
    if guard is None or _truthy(guard):
        return False
    return not any(
        (isinstance(n, ast.Name) and n.id == "TYPE_CHECKING")
        or (isinstance(n, ast.Attribute) and n.attr in _MYPY_DECIDED)
        for n in ast.walk(guard)
    )


def _irrefutable(pattern: ast.pattern) -> bool:
    """Whether *pattern* matches every subject: ``_``, a capture, or an
    ``|`` with such an alternative."""
    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None or _irrefutable(pattern.pattern)
    if isinstance(pattern, ast.MatchOr):
        return any(_irrefutable(p) for p in pattern.patterns)
    return False


def _literal_pattern(pattern: ast.pattern) -> bool:
    """A literal (``1``, ``-1.5``, ``"a"``) or an ``|`` of literals: a
    pattern that cannot exhaust an ``int``, ``float``, ``str`` ... subject."""
    if isinstance(pattern, ast.MatchValue):
        value = pattern.value
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            value = value.operand
        if isinstance(value, ast.BinOp):  # a complex literal, `1 + 2j`
            return isinstance(value.left, ast.Constant) and isinstance(
                value.right, ast.Constant
            )
        return isinstance(value, ast.Constant)
    if isinstance(pattern, ast.MatchOr):
        return all(_literal_pattern(p) for p in pattern.patterns)
    return False


class _Module:
    """What one module of the scenario's binds at its own level, and imports."""

    def __init__(self, name: str, path: str, tree: ast.Module) -> None:
        self.name = name
        self.tree = tree
        package = name if path.endswith("__init__.py") else name.rpartition(".")[0]
        self.imports = self._imports(tree, package)
        counts: dict[str, int] = {}
        bindings = own_bindings(tree)
        for bound in bindings:
            counts[bound.name] = counts.get(bound.name, 0) + 1
        once = {n for n, c in counts.items() if c == 1}
        #: The functions and classes the module defines at its own level, by
        #: a name nothing else binds there.
        self.functions: dict[str, _Function] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name in once:
                    self.functions[stmt.name] = stmt
            elif isinstance(stmt, ast.ClassDef) and stmt.name in once:
                self.classes[stmt.name] = stmt
        #: Each value a module-level name is assigned: a type alias, maybe.
        self.assigned: dict[str, list[_Alias]] = {}
        for bound in bindings:
            if bound.kind in ("assign", "annotated", "alias") and bound.value:
                params: frozenset[str] = frozenset()
                if bound.kind == "alias":
                    holder = next(
                        (
                            n
                            for n in ast.walk(tree)
                            if getattr(n, "name", None) is bound.node
                        ),
                        None,
                    )
                    params = frozenset(
                        str(getattr(t, "name", ""))
                        for t in getattr(holder, "type_params", [])
                    )
                self.assigned.setdefault(bound.name, []).append(
                    _Alias(bound.value, params)
                )

    @staticmethod
    def _imports(tree: ast.Module, package: str) -> dict[str, str | None]:
        """Each name an import of *tree* binds, and the dotted name it stands
        for; ``None`` for one two imports bind differently."""
        names: dict[str, str | None] = {}

        def bind(name: str, dotted: str) -> None:
            if names.setdefault(name, dotted) != dotted:
                names[name] = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname is not None:
                        bind(alias.asname, alias.name)
                    else:
                        top = alias.name.split(".")[0]
                        bind(top, top)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".") if package else []
                    keep = len(parts) - (node.level - 1)
                    if keep < 0:
                        continue
                    base = ".".join([*parts[:keep], *([base] if base else [])])
                for alias in node.names:
                    if alias.name != "*":
                        dotted = f"{base}.{alias.name}" if base else alias.name
                        bind(alias.asname or alias.name, dotted)
        return names


class _Index:
    """The scenario's modules, to resolve what a name or a call stands for."""

    def __init__(self, modules: Mapping[str, tuple[str, ast.Module]]) -> None:
        self.modules = {
            name: _Module(name, path, tree) for name, (path, tree) in modules.items()
        }
        #: Every function declared ``NoReturn``, by bare name: where the
        #: callee is not known, a call of one of these names ends a path.
        self.no_return_names = frozenset(
            node.name
            for _, tree in modules.values()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and self.no_return(node)
        )

    @staticmethod
    def no_return(function: _Function) -> bool:
        declared = _annotation(function)
        return declared is not None and _name(declared) in _NO_RETURN_TYPES

    def member(self, dotted: str | None) -> ast.AST | None:
        """The function or class *dotted* (``pkg.mod.f``) names, defined at
        the level of one of the modules; else ``None``."""
        if not dotted or "." not in dotted:
            return None
        owner, _, attr = dotted.rpartition(".")
        module = self.modules.get(owner)
        if module is None:
            return None
        return module.functions.get(attr) or module.classes.get(attr)

    def aliases(self, module: _Module) -> dict[str, list[_Alias]]:
        """The values each name of *module*'s level may be an alias of: its
        own assignments, and those of the name it imports from another of
        the modules."""
        out = {k: list(v) for k, v in module.assigned.items()}
        for local, dotted in module.imports.items():
            if not dotted or local in out:
                continue
            owner, _, attr = dotted.rpartition(".")
            source = self.modules.get(owner)
            if source is not None and attr in source.assigned:
                out[local] = list(source.assigned[attr])
        return out


class _Flow:
    def __init__(
        self,
        index: _Index,
        module: _Module,
        function: _Function | None = None,
        owner: ast.ClassDef | None = None,
        local: frozenset[str] = frozenset(),
    ) -> None:
        self.index = index
        self.module = module
        self.function = function
        #: The class *function* is a method of.
        self.owner = owner
        #: The names *function* and the functions around it bind.
        self.local = local

    # -- what a call stands for ---------------------------------------------

    def _dotted(self, node: ast.expr) -> str | None:
        """The dotted name *node* stands for through the module's imports
        (``sys.exit`` for ``exit`` after ``from sys import exit``)."""
        if isinstance(node, ast.Attribute):
            base = self._dotted(node.value)
            return None if base is None else f"{base}.{node.attr}"
        if not isinstance(node, ast.Name) or node.id in self.local:
            return None
        if node.id in self.module.imports:
            return self.module.imports[node.id]
        if node.id in self.module.functions or node.id in self.module.classes:
            return f"{self.module.name}.{node.id}"
        if node.id in _NO_RETURN_BUILTINS and node.id not in self.module.assigned:
            return f"builtins.{node.id}"
        return None

    def _class_of(self, annotation: ast.expr | None) -> ast.ClassDef | None:
        node = parse_string_annotation(annotation)
        if node is None:
            return None
        found = self.index.member(self._dotted(node))
        return found if isinstance(found, ast.ClassDef) else None

    def _module_of(self, cls: ast.ClassDef) -> _Module | None:
        for module in self.index.modules.values():
            if module.classes.get(cls.name) is cls:
                return module
        return None

    def _method(
        self, cls: ast.ClassDef, attr: str, *, bases_only: bool = False
    ) -> bool | None:
        """Whether *cls*'s method *attr* never returns; ``None`` when not
        known: a base from outside the modules, an attribute that is no
        method, definitions that disagree, none found."""
        verdicts: set[bool] = set()
        seen: set[int] = set()
        todo: list[tuple[ast.ClassDef, bool]] = [(cls, bases_only)]
        while todo:
            current, skip = todo.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if not skip:
                for stmt in current.body:
                    if (
                        isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and stmt.name == attr
                    ):
                        verdicts.add(self.index.no_return(stmt))
                    elif any(b.name == attr for b in _class_level_bindings(stmt)):
                        return None  # a class attribute: any callable
            module = self._module_of(current)
            if module is None:
                return None
            for base in current.bases:
                head = base.value if isinstance(base, ast.Subscript) else base
                flow = _Flow(self.index, module)
                dotted = flow._dotted(head)
                if dotted is None and isinstance(head, ast.Name):
                    dotted = f"builtins.{head.id}" if head.id == "object" else None
                if dotted in _EMPTY_BASES:
                    continue
                found = self.index.member(dotted)
                if not isinstance(found, ast.ClassDef):
                    return None
                todo.append((found, False))
            if current.keywords:  # a metaclass: it may make any attribute
                return None
        return verdicts.pop() if len(verdicts) == 1 else None

    def _resolved(self, call: ast.Call) -> bool | None:
        """Whether *call* never returns, where its callee is known."""
        func = call.func
        if isinstance(func, ast.Name):
            if func.id in self.local:
                return None
            found = self.index.member(self._dotted(func))
            if isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return self.index.no_return(found)
            if isinstance(found, ast.ClassDef):
                return False  # an instance is built
            return None
        if not isinstance(func, ast.Attribute):
            return None
        receiver = func.value
        if isinstance(receiver, ast.Name) and receiver.id not in self.local:
            # a module or a class of the modules: `helpers.stop()`
            found = self.index.member(self._dotted(func))
            if isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return self.index.no_return(found)
        cls: ast.ClassDef | None = None
        bases_only = False
        function = self.function
        if function is not None and isinstance(receiver, ast.Name):
            args = function.args
            params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            rebound = {
                b.name for b in own_bindings(function) if b.kind != "declaration"
            }
            if receiver.id in rebound:
                return None
            first = (
                params[0].arg if params and (args.posonlyargs or args.args) else None
            )
            static = any(_name(d) == "staticmethod" for d in function.decorator_list)
            if receiver.id == first and self.owner is not None and not static:
                cls = self.owner
            else:
                for param in params:
                    if param.arg == receiver.id:
                        cls = self._class_of(param.annotation)
        elif (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "super"
            and not receiver.args
            and self.owner is not None
        ):
            cls, bases_only = self.owner, True
        if cls is None:
            return None
        return self._method(cls, func.attr, bases_only=bases_only)

    def ends(self, expr: ast.expr) -> bool:
        if isinstance(expr, ast.Await):
            expr = expr.value
        if not isinstance(expr, ast.Call):
            return False
        if self._dotted(expr.func) in _NO_RETURN_QUALIFIED:
            return True
        verdict = self._resolved(expr)
        if verdict is not None:
            return verdict
        # Not known: by name, as a call of `sys.exit` or a NoReturn may be.
        name = _name(expr.func)
        return name in _NO_RETURN_NAMES or name in self.index.no_return_names

    def _suppresses(self, stmt: ast.With | ast.AsyncWith) -> bool:
        """Whether *stmt* enters ``contextlib.suppress(...)``, which goes on
        after the block when its body raises."""
        return any(
            isinstance(item.context_expr, ast.Call)
            and self._dotted(item.context_expr.func) == _SUPPRESS
            for item in stmt.items
        )

    def _open_subject(self, subject: ast.expr) -> bool:
        """Whether *subject* is a parameter declared a type no literal
        pattern exhausts (``x: int``)."""
        function = self.function
        if function is None or not isinstance(subject, ast.Name):
            return False
        if any(b.name == subject.id for b in own_bindings(function)):
            return False
        args = function.args
        for param in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            if param.arg == subject.id:
                declared = parse_string_annotation(param.annotation)
                return isinstance(declared, ast.Name) and declared.id in _OPEN_TYPES
        return False

    # -- control flow -------------------------------------------------------

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
            # contextlib.suppress goes on after the block when it raises
            return self._suppresses(stmt) or self.falls(stmt.body)
        if type(stmt).__name__ in ("Try", "TryStar"):  # TryStar: Python 3.11+
            try_ = cast(ast.Try, stmt)
            body = self.falls(try_.body) and self.falls(try_.orelse)
            handlers = any(self.falls(h.body) for h in try_.handlers)
            if try_.finalbody and not self.falls(try_.finalbody):
                return False
            return body or handlers
        if isinstance(stmt, ast.Match):
            if any(self.falls(c.body) for c in stmt.cases):
                return True
            unguarded = [c for c in stmt.cases if not _narrowing_guard(c.guard)]
            if any(_irrefutable(c.pattern) for c in unguarded):
                return False
            # No case catches every subject.  Without a guard, the cases may
            # still be exhaustive (an enum, a union), which mypy accepts; a
            # guarded case narrows nothing, and literals cannot exhaust an
            # int, a float or a str.
            if not unguarded:
                return True
            return self._open_subject(stmt.subject) and all(
                _literal_pattern(c.pattern) for c in unguarded
            )
        return True


def _class_level_bindings(stmt: ast.stmt) -> list[Bound]:
    """The names a class-level statement binds in the class body."""
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return node_bindings(stmt)
    return []


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


def _enclosing(tree: ast.Module) -> dict[int, list[ast.AST]]:
    """Each function of *tree* (by id) and the functions and classes around
    it, innermost first."""
    out: dict[int, list[ast.AST]] = {}

    def visit(node: ast.AST, around: list[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[id(child)] = around
                visit(child, [child, *around])
            elif isinstance(child, ast.ClassDef):
                visit(child, [child, *around])
            else:
                visit(child, around)

    visit(tree, [])
    return out


def _bound_in(function: ast.AST) -> set[str]:
    """The names *function* binds: its parameters and its own bindings, but
    for its imports (which :class:`_Module` resolves, as Python does) and
    ``global`` / ``nonlocal`` names."""
    names = {
        b.name
        for b in own_bindings(function)
        if b.kind not in ("declaration", "import")
    }
    if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = function.args
        names.update(
            a.arg
            for a in [
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                *filter(None, (args.vararg, args.kwarg)),
            ]
        )
    return names


def return_problems(
    modules: Mapping[str, tuple[str, ast.Module]],
) -> list[ReturnProblem]:
    """The functions of *modules* (name -> (path, tree)) whose declared return
    type is not kept."""
    index = _Index(modules)
    problems: list[ReturnProblem] = []
    for name, (path, tree) in sorted(modules.items()):
        module = index.modules[name]
        aliases = index.aliases(module)
        around = _enclosing(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            outer = around.get(id(node), [])
            owner = outer[0] if outer and isinstance(outer[0], ast.ClassDef) else None
            # ``XScenario.f``, as the author calls a method
            shown = f"{owner.name}.{node.name}" if owner is not None else node.name
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
            if _may_be_none(declared, aliases) or _stub(node):
                continue
            if any(
                isinstance(d, (ast.Name, ast.Attribute))
                and _name(d) in ("abstractmethod", "overload")
                for d in node.decorator_list
            ):
                continue
            local: set[str] = _bound_in(node)
            for scope in outer:
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    local |= _bound_in(scope)
            flow = _Flow(index, module, node, owner, frozenset(local))
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
