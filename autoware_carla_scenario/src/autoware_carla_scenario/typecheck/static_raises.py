"""Calls that raise when Python runs them, decided from the syntax alone.

Codon checks types, not values, so a call it compiles may still raise in
CPython.  Two kinds are decidable without running the scenario:

* ``isinstance`` / ``issubclass`` with a subscripted generic as the classes
  (``isinstance(v, list[int])``, ``issubclass(C, typing.Dict[str, int])``),
  also inside a tuple of classes, a ``|`` union or a ``typing.Union`` /
  ``Optional``: Python raises ``TypeError`` ("argument 2 cannot be a
  parameterized generic"), while Codon compiles it (#603).  A plain
  ``int | float`` union, an unsubscripted ``typing.List`` and an index into a
  tuple of classes (``TYPES[0]``) are left alone.
* A call of a framework class whose arguments are all literals and that the
  constructor refuses (#602): ``MetricCondition("no_such_metric", label="m")``,
  ``ElapsedTimeCondition(1.0, label="")``,
  ``RouteCrossingPose(approach="Left")``, ``RouteRoadsidePose(side="kerb")``.
  The class is built here, with those literals, and a ``ValueError`` it raises
  is the refusal, in the framework's own words: the validation is the real
  one, so the check refuses exactly what the run would.  Only the classes of
  :data:`LITERAL_CHECKED` are built: each one's constructor only validates and
  stores its arguments, without CARLA, a world, a map or any other side
  effect.  An argument that is not a literal (a name, a call, an enum member,
  ``*args``) leaves the call to the run.
"""

from __future__ import annotations

import ast
import builtins
import importlib
from functools import lru_cache
from typing import Any, NamedTuple

__all__ = [
    "LITERAL_CHECKED",
    "StaticRaise",
    "literal_argument_refusals",
    "subscripted_isinstances",
]

_PACKAGE = "autoware_carla_scenario"

#: The framework classes a call with literal arguments is built for: each
#: constructor is pure (it validates and stores its arguments, nothing else).
LITERAL_CHECKED: tuple[str, ...] = (
    f"{_PACKAGE}.AlwaysTrueCondition",
    f"{_PACKAGE}.ElapsedTimeCondition",
    f"{_PACKAGE}.MetricCondition",
    f"{_PACKAGE}.OffRoadFractionCondition",
    f"{_PACKAGE}.RouteProgressCondition",
    f"{_PACKAGE}.TimeoutCondition",
    f"{_PACKAGE}.RouteCrossingPose",
    f"{_PACKAGE}.RouteCrosswalkPose",
    f"{_PACKAGE}.RouteLanePose",
    f"{_PACKAGE}.RouteOppositePose",
    f"{_PACKAGE}.RouteRoadsidePose",
)

#: The generic classes a subscript of is a parameterized generic.
_GENERICS = frozenset(
    {
        *(f"builtins.{n}" for n in ("list", "dict", "set", "frozenset", "tuple")),
        "builtins.type",
        *(
            f"{module}.{n}"
            for module in ("typing", "collections.abc")
            for n in (
                "AbstractSet",
                "AsyncIterable",
                "AsyncIterator",
                "Awaitable",
                "Callable",
                "Collection",
                "Container",
                "Generator",
                "ItemsView",
                "Iterable",
                "Iterator",
                "KeysView",
                "Mapping",
                "MappingView",
                "MutableMapping",
                "MutableSequence",
                "MutableSet",
                "Reversible",
                "Sequence",
                "Set",
                "ValuesView",
            )
        ),
        *(
            f"typing.{n}"
            for n in (
                "Annotated",
                "ChainMap",
                "ClassVar",
                "Counter",
                "DefaultDict",
                "Deque",
                "Dict",
                "Final",
                "FrozenSet",
                "List",
                "Literal",
                "OrderedDict",
                "Tuple",
                "Type",
            )
        ),
        *(
            f"collections.{n}"
            for n in ("ChainMap", "Counter", "OrderedDict", "defaultdict", "deque")
        ),
    }
)
#: The unions whose members are classes in their own right.
_UNIONS = frozenset({"typing.Union", "typing.Optional"})
#: What to test instead of a subscript of these, when not the bare class.
_INSTEAD = {
    "typing.Literal": "compare the value itself (`x == 1`, `x in (1, 2)`)",
    "typing.Annotated": "test against its first argument, the class",
    "typing.ClassVar": "test against the class it wraps",
    "typing.Final": "test against the class it wraps",
}


class StaticRaise(NamedTuple):
    line: int
    column: int
    message: str


def _imported_names(tree: ast.Module) -> dict[str, str]:
    """Each name an absolute import binds, and the dotted name it stands for."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if alias.name != "*":
                    names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    names[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    names[top] = top
    return names


def _rebound(tree: ast.Module) -> set[str]:
    """Each name *tree* binds other than by an import: what an import or a
    builtin of that name stands for is not certain there."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


class _Names:
    def __init__(self, tree: ast.Module) -> None:
        self.imported = _imported_names(tree)
        self.rebound = _rebound(tree)

    def dotted(self, node: ast.expr) -> str | None:
        """The dotted name *node* stands for (``typing.List`` for ``t.List``
        after ``import typing as t``; ``builtins.list`` for ``list``)."""
        if isinstance(node, ast.Attribute):
            base = self.dotted(node.value)
            return None if base is None else f"{base}.{node.attr}"
        if not isinstance(node, ast.Name) or node.id in self.rebound:
            return None
        if node.id in self.imported:
            return self.imported[node.id]
        if hasattr(builtins, node.id):
            return f"builtins.{node.id}"
        return None


# ---------------------------------------------------------------------------
# isinstance / issubclass with a parameterized generic (#603)
# ---------------------------------------------------------------------------


def _generic(node: ast.expr, names: _Names) -> tuple[ast.Subscript, str] | None:
    """The first subscripted generic of the classes *node*, and its name."""
    if isinstance(node, ast.Tuple):
        for item in node.elts:
            found = _generic(item, names)
            if found is not None:
                return found
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _generic(node.left, names) or _generic(node.right, names)
    if isinstance(node, ast.Subscript):
        name = names.dotted(node.value)
        if name in _UNIONS:
            members = node.slice
            return _generic(members, names)
        if name in _GENERICS:
            return node, name
    return None


def subscripted_isinstances(tree: ast.Module) -> list[StaticRaise]:
    """Each ``isinstance`` / ``issubclass`` of *tree* whose classes include a
    subscripted generic, which Python refuses with ``TypeError``."""
    names = _Names(tree)
    found: list[StaticRaise] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("isinstance", "issubclass")
            and names.dotted(node.func) == f"builtins.{node.func.id}"
            and len(node.args) == 2
        ):
            continue
        hit = _generic(node.args[1], names)
        if hit is None:
            continue
        subscript, name = hit
        instead = _INSTEAD.get(
            name,
            f"test against the bare class (`{ast.unparse(subscript.value)}`) and "
            "check the elements separately if they matter",
        )
        found.append(
            StaticRaise(
                node.lineno,
                node.col_offset + 1,
                f"{node.func.id}() cannot take the subscripted generic "
                f"`{ast.unparse(subscript)}`: Python raises TypeError "
                f"({node.func.id}() argument 2 cannot be a parameterized "
                f"generic) when it evaluates it, though Codon compiles it; "
                f"{instead}",
            )
        )
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Framework classes built with literal arguments (#602)
# ---------------------------------------------------------------------------


def _resolve(dotted: str) -> Any:
    """The object *dotted* names, importing its module, or ``None``."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:i]))
        except Exception:  # noqa: BLE001 -- not a module: try a shorter one
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except Exception:  # noqa: BLE001
            return None
        return obj
    return None


@lru_cache(maxsize=1)
def _checked_classes() -> dict[Any, str]:
    """Each class of :data:`LITERAL_CHECKED`, to its name."""
    out: dict[Any, str] = {}
    for dotted in LITERAL_CHECKED:
        cls = _resolve(dotted)
        if cls is not None:
            out[cls] = dotted.rsplit(".", 1)[1]
    return out


_NOT_LITERAL = object()


def _literal(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _NOT_LITERAL


def literal_argument_refusals(tree: ast.Module) -> list[StaticRaise]:
    """Each call of a class of :data:`LITERAL_CHECKED` in *tree* whose
    arguments are all literals and that raises ``ValueError`` when built."""
    checked = _checked_classes()
    if not checked:
        return []
    names = _Names(tree)
    found: list[StaticRaise] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = names.dotted(node.func)
        if dotted is None or not (
            dotted == _PACKAGE or dotted.startswith(_PACKAGE + ".")
        ):
            continue
        cls = _resolve(dotted)
        if not isinstance(cls, type) or cls not in checked:
            continue
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            continue
        args = [_literal(a) for a in node.args]
        kwargs = {k.arg: _literal(k.value) for k in node.keywords if k.arg}
        if any(v is _NOT_LITERAL for v in (*args, *kwargs.values())):
            continue
        try:
            cls(*args, **kwargs)
        except ValueError as exc:
            found.append(
                StaticRaise(
                    node.lineno,
                    node.col_offset + 1,
                    f"{checked[cls]}(...) with these literal arguments "
                    f"raises ValueError when the scenario runs: {exc}",
                )
            )
        except Exception:  # noqa: BLE001 -- a type error is Codon's to report
            continue
    return sorted(set(found))
