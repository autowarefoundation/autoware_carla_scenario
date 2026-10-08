"""The class hierarchy of the modules a binary compiles, across modules.

Codon 0.19 dispatches a method a subclass overrides through a table, and it
miscompiles one case Python code is full of: a method defined in a class,
overridden further down, and called through a class in between that inherits
it (``'M' does not match expected type 'E'`` in ``class_thunk_dispatch``). A
class in between that defines the method itself is dispatched correctly, so
:meth:`Hierarchy.forwarders` gives every class that inherits such a method a
definition of its own that calls ``super()`` -- with the parameter types
spelled out, because a method with untyped parameters is generic and cannot
be dispatched at all.

The rewrite is per module (:func:`.transform.transform_for_runtime`); this is
the part of it that needs every module at once.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .collect import PACKAGE, _Resolver, absolute, package_root

__all__ = ["ClassInfo", "Hierarchy", "ModuleContext"]


@dataclass
class ClassInfo:
    module: str
    name: str
    node: ast.ClassDef
    #: (module, class) of each base the index knows, in order
    bases: list[tuple[str, str]] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return self.module, self.name

    def method(self, name: str) -> ast.FunctionDef | None:
        for stmt in self.node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
                return stmt
        return None


@dataclass
class ModuleContext:
    """What the rewrite of one module adds from the rest of the program."""

    #: class name -> lines to insert at the top of its body
    class_lines: dict[str, list[str]] = field(default_factory=dict)
    #: import statements the inserted lines need
    imports: list[str] = field(default_factory=list)
    #: every class attribute of the program (:meth:`Hierarchy.class_attributes`)
    attributes: frozenset[str] = frozenset()
    #: class name -> {its class attribute: whether no ancestor has it too}
    defines: dict[str, dict[str, bool]] = field(default_factory=dict)
    #: class name -> the class attributes it keeps as plain fields
    fields: dict[str, set[str]] = field(default_factory=dict)
    #: lines of the imports under ``if TYPE_CHECKING:`` that close no cycle
    #: (:meth:`Hierarchy.kept_type_checking`); the others are dropped
    kept: frozenset[int] = frozenset()
    #: name -> the function-level import binding it, which can run at module
    #: level instead (an attribute's declared type may need the name)
    hoistable: dict[str, str] = field(default_factory=dict)
    #: filled by the rewrite: the hoistable imports it used
    hoisted: list[str] = field(default_factory=list)
    #: classes of the module that are exceptions (:meth:`Hierarchy.exceptions`)
    exceptions: frozenset[str] = frozenset()


def _bindings(module: str, tree: ast.Module) -> dict[str, str]:
    """Top-level name -> the import statement that binds it in *module*."""
    is_package = False
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.If):  # `if TYPE_CHECKING:` imports run in Codon
            body = node.body
        else:
            body = [node]
        for stmt in body:
            if isinstance(stmt, ast.Import):
                for a in stmt.names:
                    if a.asname:
                        out[a.asname] = f"import {a.name} as {a.asname}"
                    else:
                        out[a.name.split(".")[0]] = f"import {a.name}"
            elif isinstance(stmt, ast.ImportFrom):
                source = absolute(module, is_package, stmt)
                for a in stmt.names:
                    bound = a.asname or a.name
                    alias = f" as {bound}" if bound != a.name else ""
                    out[bound] = f"from {source} import {a.name}{alias}"
            elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.Assign)):
                names = (
                    [stmt.name]
                    if not isinstance(stmt, ast.Assign)
                    else [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                )
                for name in names:
                    out[name] = f"from {module} import {name}"
    return out


BUILTIN_EXCEPTIONS = {
    "Exception", "RuntimeError", "ValueError", "TypeError", "KeyError", "IndexError",
    "LookupError", "OSError", "IOError", "TimeoutError", "NotImplementedError",
    "AttributeError", "ArithmeticError", "ZeroDivisionError", "AssertionError",
    "StopIteration", "EOFError", "FileNotFoundError", "PermissionError",
    "ConnectionError",
}  # fmt: skip

_BUILTIN_NAMES = {
    "int", "float", "str", "bool", "bytes", "object", "None", "List", "Dict", "Set",
    "Tuple", "Optional", "Union", "Callable", "ClassVar", "Literal", "File",
}  # fmt: skip


class Hierarchy:
    """Every class in *modules* (module name -> source), with resolved bases."""

    def __init__(self, modules: dict[str, str]):
        self.resolver = _Resolver({PACKAGE: package_root()})
        self.trees = {m: ast.parse(s) for m, s in modules.items()}
        self.classes: dict[tuple[str, str], ClassInfo] = {}
        self.bindings: dict[str, dict[str, str]] = {}
        self._edges: dict[str, set[str]] | None = None
        for module, tree in self.trees.items():
            self.bindings[module] = _bindings(module, tree)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    self.classes[(module, node.name)] = ClassInfo(
                        module, node.name, node
                    )
        for info in self.classes.values():
            for base in info.node.bases:
                found = self._resolve(info.module, base)
                if found is not None:
                    info.bases.append(found)

    def _imported(self, module: str, node: ast.Import | ast.ImportFrom) -> set[str]:
        """The compiled modules an import statement in *module* names (each
        name resolved to the module that defines it)."""
        out: set[str] = set()
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        else:
            source = absolute(module, False, node)
            for a in node.names:
                if f"{source}.{a.name}" in self.trees:
                    out.add(f"{source}.{a.name}")
                    continue
                target = self.resolver.resolve(source, a.name)
                out.add(target[0] if target is not None else source)
        return {m for m in out if m in self.trees}

    def _graph(self) -> dict[str, set[str]]:
        """Module -> the compiled modules it imports when it runs: its
        module-level imports, those under TYPE_CHECKING it keeps included."""
        if self._edges is not None:
            return self._edges
        graph: dict[str, set[str]] = {m: set() for m in self.trees}
        pending: list[tuple[str, ast.ImportFrom]] = []
        for module, tree in self.trees.items():
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    graph[module] |= self._imported(module, node)
                elif isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(
                    node.test
                ):
                    for stmt in node.body:
                        if isinstance(stmt, ast.ImportFrom) and stmt.level > 0:
                            pending.append((module, stmt))
        self._kept: set[tuple[str, int]] = set()
        for module, stmt in pending:  # in order: a kept one is an edge for the next
            targets = self._imported(module, stmt)
            if not any(self._reaches(graph, t, module) for t in targets):
                graph[module] |= targets
                self._kept.add((module, stmt.lineno))
        self._edges = graph
        return graph

    @staticmethod
    def _reaches(graph: dict[str, set[str]], start: str, goal: str) -> bool:
        seen: set[str] = set()
        stack = [start]
        while stack:
            m = stack.pop()
            if m == goal:
                return True
            if m in seen:
                continue
            seen.add(m)
            stack.extend(graph.get(m, ()))
        return False

    def kept_type_checking(self, module: str) -> frozenset[int]:
        """Lines of *module*'s TYPE_CHECKING imports that close no import
        cycle: Codon runs them (the typing stand-in sets TYPE_CHECKING), and a
        cycle Python never runs into is one Codon cannot compile."""
        self._graph()
        return frozenset(line for m, line in self._kept if m == module)

    def hoistable(self, module: str) -> dict[str, str]:
        """Function-level imports of *module* that close no cycle at module level."""
        graph = self._graph()
        out: dict[str, str] = {}
        for fn in ast.walk(self.trees[module]):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.ImportFrom) or not node.level:
                    continue
                if any(
                    self._reaches(graph, t, module)
                    for t in self._imported(module, node)
                ):
                    continue
                for a in node.names:
                    alias = f" as {a.asname}" if a.asname else ""
                    out[a.asname or a.name] = (
                        f"from {'.' * node.level}{node.module or ''} import {a.name}{alias}"
                    )
        return out

    def exceptions(self) -> set[tuple[str, str]]:
        """The classes deriving from a built-in exception, directly or not.

        Codon's exceptions inherit statically (``Static[RuntimeError]``); one
        that derives from one the Python way makes the built-in polymorphic,
        and then not even the built-in can be raised
        (:func:`.transform.rewrite_for_runtime`).
        """
        out: set[tuple[str, str]] = set()
        changed = True
        while changed:
            changed = False
            for info in self.classes.values():
                if info.key in out:
                    continue
                if any(
                    isinstance(b, ast.Name) and b.id in BUILTIN_EXCEPTIONS
                    for b in info.node.bases
                ) or any(b in out for b in info.bases):
                    out.add(info.key)
                    changed = True
        return out

    def base_context(self, module: str) -> ModuleContext:
        """What the rewrite of *module* needs before any module is rewritten."""
        return ModuleContext(
            kept=self.kept_type_checking(module),
            hoistable=self.hoistable(module),
            exceptions=frozenset(n for m, n in self.exceptions() if m == module),
        )

    def _resolve(self, module: str, expr: ast.expr) -> tuple[str, str] | None:
        """The (module, class) a base expression in *module* names, if indexed."""
        if not isinstance(expr, ast.Name):
            return None
        if (module, expr.id) in self.classes:
            return module, expr.id
        tree = self.trees[module]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if (a.asname or a.name) != expr.id:
                        continue
                    source = absolute(module, False, node)
                    target = self.resolver.resolve(source, a.name)
                    if target is not None and target in self.classes:
                        return target
        return None

    def mro(self, key: tuple[str, str]) -> list[ClassInfo]:
        """The class and its indexed ancestors, depth first, left to right."""
        out: list[ClassInfo] = []
        seen: set[tuple[str, str]] = set()

        def visit(k: tuple[str, str]) -> None:
            if k in seen or k not in self.classes:
                return
            seen.add(k)
            out.append(self.classes[k])
            for base in self.classes[k].bases:
                visit(base)

        visit(key)
        return out

    def class_attributes(self, info: ClassInfo) -> dict[str, ast.AnnAssign]:
        """``name: T = value`` in a plain class's body (not a ClassVar, not a
        dataclass field, not an enum member).

        Python reads one off the class until an instance assigns its own, and
        a subclass may give it another value. Codon makes it an instance field
        that the class's own ``__init__`` leaves at zero, whatever the value;
        the rewrite keeps it a class variable read through a method
        (:func:`.transform.rewrite_for_runtime`).
        """
        node = info.node
        if any(
            "dataclass" in ast.unparse(d) or "tuple" == ast.unparse(d)
            for d in node.decorator_list
        ):
            return {}
        if any(
            "Enum" in ast.unparse(b) or "NamedTuple" in ast.unparse(b)
            for b in node.bases
        ):
            return {}
        out: dict[str, ast.AnnAssign] = {}
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.value is not None
                and "ClassVar" not in ast.unparse(stmt.annotation)
                and not stmt.target.id.startswith("__")
            ):
                out[stmt.target.id] = stmt
        return out

    def attribute_names(self) -> frozenset[str]:
        """The class attributes some class gives a value other than zero.

        One that is None, False, 0 or "" wherever it is given is what a Codon
        field starts as anyway, and stays one.
        """
        values: dict[str, list[ast.expr]] = {}
        for info in self.classes.values():
            for name, stmt in self.class_attributes(info).items():
                assert stmt.value is not None
                values.setdefault(name, []).append(stmt.value)
        return frozenset(
            name
            for name, given in values.items()
            if not all(
                isinstance(v, ast.Constant) and v.value in (None, False, 0, "")
                for v in given
            )
        )

    def _assigned(self, info: ClassInfo) -> set[str]:
        """Attributes *info*'s methods assign on ``self``."""
        out: set[str] = set()
        for stmt in ast.walk(info.node):
            targets: list[ast.expr] = []
            if isinstance(stmt, ast.Assign):
                targets = list(stmt.targets)
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                targets = [stmt.target]
            for t in targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    out.add(t.attr)
        return out

    def _overridden(self) -> set[tuple[tuple[str, str], str]]:
        """(defining class, method) of every method some subclass redefines."""
        out: set[tuple[tuple[str, str], str]] = set()
        for info in self.classes.values():
            ancestors = self.mro(info.key)[1:]
            for stmt in info.node.body:
                if not isinstance(stmt, ast.FunctionDef) or stmt.name.startswith("__"):
                    continue
                for ancestor in ancestors:
                    if ancestor.method(stmt.name) is not None:
                        out.add((ancestor.key, stmt.name))
        return out

    def context(self, module: str, transformed: dict[str, str]) -> ModuleContext:
        """The forwarders the classes of *module* need (module docstring).

        *transformed* holds each module's rewritten source: a forwarder takes
        the parameter types its target has after the rewrite.
        """
        ctx = self.base_context(module)
        ctx.attributes = self.attribute_names()
        overridden = self._overridden()
        own = self.bindings.get(module, {})
        exceptions = self.exceptions()
        for info in [c for c in self.classes.values() if c.module == module]:
            if info.key in exceptions:
                continue  # inherits statically: nothing is dispatched
            ancestors = self.mro(info.key)[1:]
            inherited = {
                n
                for a in ancestors
                for n in self.class_attributes(a)
                if n in ctx.attributes
            }
            mine = {
                n: v
                for n, v in self.class_attributes(info).items()
                if n in ctx.attributes
            }
            if mine:
                ctx.defines[info.name] = {n: n not in inherited for n in mine}
            # an attribute some unrelated class has as a class attribute, which
            # this one assigns as an ordinary field: reads of it anywhere are
            # rewritten, so the field gets the same accessors (once per chain)
            plain = (self._assigned(info) & ctx.attributes) - inherited - set(mine)
            for ancestor in ancestors:
                plain -= self._assigned(ancestor)
            if plain:
                ctx.fields[info.name] = plain
            defined = {s.name for s in info.node.body if isinstance(s, ast.FunctionDef)}
            done: set[str] = set(defined)
            for ancestor in self.mro(info.key)[1:]:
                for stmt in ancestor.node.body:
                    if not isinstance(stmt, ast.FunctionDef) or stmt.name in done:
                        continue
                    done.add(stmt.name)
                    if (ancestor.key, stmt.name) not in overridden:
                        continue
                    lines = self._forwarder(ancestor, stmt.name, transformed)
                    if lines is None:
                        continue
                    text, names = lines
                    ctx.class_lines.setdefault(info.name, []).extend(text)
                    for name in sorted(names):
                        statement = self.bindings[ancestor.module].get(name)
                        if statement is not None:
                            statement = _relative(module, statement)
                        if statement is not None and name not in own:
                            if statement not in ctx.imports:
                                ctx.imports.append(statement)
        return ctx

    def _forwarder(
        self, owner: ClassInfo, name: str, transformed: dict[str, str]
    ) -> tuple[list[str], set[str]] | None:
        """``def name(self, a: T) -> R: return super().name(a)`` for *owner*'s
        method as the rewrite left it; None for one that is not plain
        positional (defaults, keyword-only, ``*args``)."""
        original = owner.method(name)
        assert original is not None
        args = original.args
        if (
            args.defaults
            or args.kwonlyargs
            or args.vararg
            or args.kwarg
            or args.posonlyargs
        ):
            return None
        decorators = [ast.unparse(d) for d in original.decorator_list]
        if any(d not in ("property", "staticmethod") for d in decorators):
            return None
        if "staticmethod" in decorators:
            return None  # not dispatched
        tree = ast.parse(transformed[owner.module])
        method: ast.FunctionDef | None = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == owner.name:
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
                        method = stmt
        if method is None:
            return None
        params = method.args.args
        names: set[str] = set()
        rendered = []
        for i, param in enumerate(params):
            if i == 0:
                rendered.append(param.arg)
                continue
            if param.annotation is None:
                return None  # generic: Codon cannot dispatch it anyway
            rendered.append(f"{param.arg}: {ast.unparse(param.annotation)}")
            names |= _names(param.annotation)
        returns = ""
        if method.returns is not None:
            returns = f" -> {ast.unparse(method.returns)}"
            names |= _names(method.returns)
        call = ", ".join(p.arg for p in params[1:])
        body = f"return super().{name}"
        if "property" not in decorators:
            body += f"({call})"
        lines = []
        if "property" in decorators:
            lines.append("@property")
        lines.append(f"def {name}({', '.join(rendered)}){returns}: {body}")
        return lines, names - _BUILTIN_NAMES


def _names(annotation: ast.AST) -> set[str]:
    """The top-level names an annotation uses (``carla`` of ``carla.World``)."""
    out: set[str] = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name):
            out.add(node.id)
    return out


def _relative(module: str, statement: str) -> str:
    """``from <package module> import x`` as a relative import from *module*:
    Codon resolves the package's absolute name only once its __init__ ran."""
    prefix = "from "
    if not statement.startswith(prefix + PACKAGE + "."):
        return statement
    source, _, rest = statement[len(prefix) :].partition(" import ")
    here = module.split(".")[:-1]
    there = source.split(".")
    common = 0
    while common < min(len(here), len(there)) and here[common] == there[common]:
        common += 1
    dots = "." * (len(here) - common + 1)
    return f"from {dots}{'.'.join(there[common:])} import {rest}"
