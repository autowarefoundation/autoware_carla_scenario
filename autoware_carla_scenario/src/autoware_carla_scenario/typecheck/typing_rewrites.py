"""Type-only names, caught exceptions that are stored, ``operator`` getters and
``copy`` (#512, #513, #514, #517), rewritten for Codon 0.19.

A pass of its own, run by :func:`.transform.transform_source_lines` before the
other rewrites.  Each rewrite is on the same lines as what it replaces, and
only the checked copy of a module changes: what runs is the code as written.

* ``@override`` is dropped.
* ``X: TypeAlias = T`` becomes ``X = T``.
* ``Self`` in an annotation of a class is the class (``-> Self`` is
  ``-> Counter``): a subclass's method is checked as returning its base.
* ``X = NewType("X", T)`` becomes ``X = T`` and ``X(v)`` becomes ``(v)``: the
  new type is checked as its base, as Python runs it.
* ``assert_never(x)`` as a statement or a ``return`` becomes ``raise
  AssertionError(...)``.
* ``TypeVar("T", bound=B)`` and ``TypeVar("N", int, float)`` become
  ``TypeVar("T")``: Codon makes a function generic over its unannotated
  parameters, and its ``TypeVar`` takes a name only.
* ``operator.itemgetter(1)`` and ``attrgetter("n")`` with literal arguments
  become lambdas: Codon's call the function with the argument wrapped in a
  tuple, so they cannot be a sort key.
* ``from collections.abc import Sequence, Callable`` becomes ``from typing
  import (Sequence, Callable)`` (#586): Codon has no ``collections.abc``, and
  the typing shim has the abstract collections (``Set`` is its
  ``AbstractSet``).  A name the shim does not have, ``import
  collections.abc`` and ``from collections import abc`` are refused with a
  hint.
* A caught exception put in a variable, attribute, list or parameter
  annotated ``Exception`` is rebuilt as an ``Exception`` of its type name and
  message: Codon does not upcast an exception to its base.

Python 3.12+ syntax (#624):

* A string literal in an f-string's replacement field quoted with the
  f-string's own quote (PEP 701: ``f"{d["k"]}"``) is quoted with the other
  quote (:func:`requote_fstring_fields`), on the same columns: Codon 0.19's
  parser, as Python's before 3.12, ends the f-string at its own quote.  One
  that cannot be (it holds the other quote too, or a backslash) is refused,
  unless its f-string is one :mod:`.python_rewrites` writes as ``+`` parts
  (a literal brace, a spec Codon does not parse, ``!a``), where such a field
  becomes a call and the quotes no longer matter (#704).
* ``type X = T`` becomes ``X = T``.  A generic alias (``type Pair[T] =
  tuple[T, T]``) is refused: Codon has no generic alias.
* The type parameters of a function or class (PEP 695: ``def first[T](xs:
  list[T]) -> T``, ``class Box[T]:``) are Codon's own generics syntax and
  stay; a bound, constraints or a default (``[T: Base]``, ``[T = int]``) are
  dropped (Codon would read ``T: int`` as a static integer parameter), and
  ``*Ts`` / ``**P`` are refused.

``Never`` and ``Container`` annotations are dropped, and ``TypeGuard[X]`` is
read as ``bool``, where the transform reads annotations.  ``copy.copy`` and
``copy.deepcopy`` of a class of the module get the ``__copy__`` and
``__deepcopy__`` Codon calls (:func:`copy_dunders`).
"""

from __future__ import annotations

import ast
import io
import re
import tokenize

from .scopes import ShadowScopes
from .transform import _dotted, _Edits, _union_members

__all__ = [
    "TYPING_HINTS",
    "copy_dunders",
    "requote_fstring_fields",
    "typing_rewrite_refusals",
    "typing_rewrites",
    "typing_rewrites_lines",
]

#: Codon's refusals of what this pass leaves alone, and what to write.
TYPING_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"'(?:Optional\[)?\w+\]?' does not match expected type "
            r"'(?:Optional\[)?(?:Base)?Exception\]?'"
            r"|'(?:Optional\[)?(?:Base)?Exception\]?' does not match expected type "
            r"'(?:Optional\[)?\w+(?:Error|Exception)\]?'"
        ),
        "Codon does not convert an exception to its base class, so a value "
        "typed as a subclass cannot go where `Exception` is expected. The check "
        "rebuilds a caught exception (`except ValueError as e:`) that is "
        "assigned, appended or passed to a variable, attribute, list or "
        "parameter annotated `Exception`; for any other exception store "
        "`str(e)`, or catch `Exception` itself",
    ),
    (
        re.compile(
            r"'Optional\[\w*(?:Condition|Action|Scenario)\]' "
            r"does not match expected type 'Optional\[\w*(?:Condition|Action|Scenario)\]'"
        ),
        "Codon 0.19 does not convert an Optional of a subclass to an Optional of "
        "its base (nor the reverse), so an `Optional[BaseCondition]` or "
        "`Optional[BaseAction]` that holds an `ElapsedTimeCondition`, a "
        "`SetSpeedAction` or any other subclass is refused, as is an Optional "
        "helper parameter given one. Declare the value without Optional "
        "(`c: BaseCondition = ElapsedTimeCondition(...)`) and test a separate "
        "flag, register it where it is made, or pass the helper a plain "
        "`BaseCondition` and a bool",
    ),
    (
        re.compile(r"object has no attribute '__(?:deep)?copy__'"),
        "Codon's copy.copy() and copy.deepcopy() call the class's __copy__ and "
        "__deepcopy__. The check writes them for the classes of a module that "
        "imports copy, from their declared fields: define them on this class "
        "(`def __copy__(self) -> Self`), or build the copy from its fields",
    ),
    (
        re.compile(r"tuple index out of range.*item: \d+"),
        "operator.itemgetter() and attrgetter() cannot be a key here: write a "
        "lambda (`key=lambda v: v[1]`, `lambda v: v.n`)",
    ),
    (
        re.compile(r"cannot import name '\w+' from 'typing_extensions'"),
        "the check models only the names of typing_extensions that typing has "
        'too: docs/typecheck.md ("typing names") lists them',
    ),
)

_MAX_PASSES = 5
_TYPING_MODULES = {"typing", "typing_extensions"}
_ATTRIBUTE_PATH = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
_ASSERT_NEVER = 'raise AssertionError("Expected code to be unreachable")'
_ADDERS = {"append", "add", "appendleft"}
#: The names of ``collections.abc`` the typing shim has, by their name there
#: (#586): ``from collections.abc import X`` is read as ``from typing import X``.
_ABC_NAMES = {
    **{
        name: name
        for name in (
            "Callable",
            "Collection",
            "Container",
            "Generator",
            "Hashable",
            "Iterable",
            "Iterator",
            "Mapping",
            "MutableMapping",
            "MutableSequence",
            "Sequence",
            "Sized",
        )
    },
    "Set": "AbstractSet",
}
_ABC_HINT = (
    "the check reads `from collections.abc import ...` as an import from "
    "typing, of these names only: {names} (`Set` is typing's `AbstractSet`)"
).format(names=", ".join(f"`{n}`" for n in sorted(_ABC_NAMES)))
_LISTS = {"list", "List", "set", "Set", "Deque", "deque", "Sequence", "Iterable"}
_DICTS = {"dict", "Dict", "Mapping", "MutableMapping", "DefaultDict", "defaultdict"}


def _literal_key(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, str)) and not isinstance(node.value, bool)
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    )


class _Names:
    """What the module's imports of ``typing``, ``typing_extensions``,
    ``operator`` and ``copy`` are called in it."""

    def __init__(self, tree: ast.Module) -> None:
        #: local name -> the name it has in typing / operator
        self.typing: dict[str, str] = {}
        self.operator: dict[str, str] = {}
        self.typing_modules: set[str] = set()
        self.operator_modules: set[str] = set()
        self.copy_imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                for alias in node.names:
                    local = alias.asname or alias.name
                    if node.module in _TYPING_MODULES:
                        self.typing[local] = alias.name
                    elif node.module == "operator":
                        self.operator[local] = alias.name
                    elif node.module == "copy":
                        self.copy_imported = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name
                    if alias.name in _TYPING_MODULES:
                        self.typing_modules.add(local)
                    elif alias.name == "operator":
                        self.operator_modules.add(local)
                    elif alias.name == "copy":
                        self.copy_imported = True

    def typing_name(self, node: ast.AST) -> str | None:
        """The ``typing`` name *node* (``Self``, ``t.Self``) refers to."""
        if isinstance(node, ast.Name):
            return self.typing.get(node.id)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.typing_modules
        ):
            return node.attr
        return None

    def operator_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.operator.get(node.id)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.operator_modules
        ):
            return node.attr
        return None


def _exception_base(annotation: ast.AST | None) -> str | None:
    """``Exception`` / ``BaseException`` for an annotation of one, alone or
    with ``None`` (``Optional[...]``, ``X | None``), else ``None``."""
    if annotation is None:
        return None
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None
    members = _union_members(annotation)
    if members is None and isinstance(annotation, ast.Subscript):
        if (_dotted(annotation.value) or "").removeprefix("typing.") == "Optional":
            members = [annotation.slice]
    if members is not None:
        rest = [
            m for m in members if not (isinstance(m, ast.Constant) and m.value is None)
        ]
        return _exception_base(rest[0]) if len(rest) == 1 else None
    name = _dotted(annotation)
    return name if name in ("Exception", "BaseException") else None


def _element(annotation: ast.AST | None, mapping: bool) -> ast.AST | None:
    """The element annotation of a ``list[X]`` (or the value of a ``dict[K, X]``)."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None
    if not isinstance(annotation, ast.Subscript):
        return None
    name = (_dotted(annotation.value) or "").removeprefix("typing.").split(".")[-1]
    slice_ = annotation.slice
    if mapping and name in _DICTS:
        if isinstance(slice_, ast.Tuple) and len(slice_.elts) == 2:
            return slice_.elts[1]
    elif not mapping and name in _LISTS:
        return slice_ if not isinstance(slice_, ast.Tuple) else None
    return None


def _own_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Every statement of *body* and of the blocks inside it, but for those of
    nested functions and classes."""
    out: list[ast.stmt] = []
    stack = list(body)
    while stack:
        stmt = stack.pop()
        out.append(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for name in ("body", "orelse", "finalbody"):
            stack.extend(getattr(stmt, name, []) or [])
        for handler in getattr(stmt, "handlers", []) or []:
            stack.extend(handler.body)
        for case in getattr(stmt, "cases", []) or []:
            stack.extend(case.body)
    return out


class _Pass(ShadowScopes):
    def __init__(self, edits: _Edits, tree: ast.Module, names: _Names) -> None:
        super().__init__()
        self.edits = edits
        self.names = names
        self.refused: list[tuple[int, str]] = []
        #: The classes being visited, innermost last; ``None`` marks a function.
        self._scope: list[str | None] = []
        #: The ``NewType``s of the module (class and module level).
        self._new_types: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and self._new_type(node.value) is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._new_types.add(target.id)
        #: Class -> attribute -> annotation, for ``self.x`` stores.
        self._attributes: dict[str, dict[str, ast.expr]] = {}
        #: Class -> bases.
        self._bases: dict[str, list[str]] = {}
        #: Class -> method -> parameters (name, annotation) without ``self``;
        #: "" is the module's functions.
        self._signatures: dict[str, dict[str, list[tuple[str, ast.expr | None]]]] = {
            "": {}
        }
        self._scan(tree)
        #: Annotations of the names bound in the function being visited, and
        #: in the module; the handler variables of the specific exceptions
        #: being handled; the return annotations of the functions.
        self._annotations: list[dict[str, ast.expr]] = [self._module_annotations(tree)]
        self._handlers: list[str] = []
        self._returns: list[ast.expr | None] = []
        self._classes: list[str] = []

    def visit_Module(self, node: ast.Module) -> None:
        self.enter_module(node)
        self.generic_visit(node)

    def _local(self, node: ast.AST) -> bool:
        """Whether *node* is a name a function, lambda or class around it binds
        (a parameter called ``LaneId``, say), not the module's import or alias."""
        if not isinstance(node, ast.Name):
            return False
        for bound, _ in reversed(self._local_bound):
            if node.id in bound:
                return True
        return False

    # -- scanning ---------------------------------------------------------

    def _scan(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                attrs = self._attributes.setdefault(node.name, {})
                self._bases[node.name] = [
                    b for b in (_dotted(x) for x in node.bases) if b
                ]
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(
                        stmt.target, ast.Name
                    ):
                        attrs[stmt.target.id] = stmt.annotation
                    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._signatures.setdefault(node.name, {})[stmt.name] = (
                            self._parameters(stmt, skip=True)
                        )
                        for sub in _own_statements(stmt.body):
                            if (
                                isinstance(sub, ast.AnnAssign)
                                and isinstance(sub.target, ast.Attribute)
                                and isinstance(sub.target.value, ast.Name)
                                and sub.target.value.id == "self"
                            ):
                                attrs[sub.target.attr] = sub.annotation
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._signatures[""][stmt.name] = self._parameters(stmt, skip=False)
            elif isinstance(stmt, ast.ClassDef):
                init = self._signatures.get(stmt.name, {}).get("__init__")
                if init is not None:
                    self._signatures[""][stmt.name] = init

    @staticmethod
    def _parameters(
        node: ast.FunctionDef | ast.AsyncFunctionDef, skip: bool
    ) -> list[tuple[str, ast.expr | None]]:
        args = [*node.args.posonlyargs, *node.args.args]
        if skip and args:
            args = args[1:]
        return [(a.arg, a.annotation) for a in [*args, *node.args.kwonlyargs]]

    @staticmethod
    def _module_annotations(tree: ast.Module) -> dict[str, ast.expr]:
        return {
            s.target.id: s.annotation
            for s in _own_statements(tree.body)
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
        }

    # -- type-only names ---------------------------------------------------

    def _new_type(self, node: ast.AST | None) -> ast.expr | None:
        """The base of ``NewType("N", base)``."""
        if (
            isinstance(node, ast.Call)
            and self.names.typing_name(node.func) == "NewType"
            and len(node.args) == 2
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], (ast.Name, ast.Attribute, ast.Subscript))
        ):
            return node.args[1]
        return None

    def _in_class(self) -> str | None:
        names: list[str] = []
        for entry in reversed(self._scope):
            if entry is None:
                break
            names.append(entry)
        return ".".join(reversed(names)) or None

    def _self_to_class(self, annotation: ast.AST | None) -> None:
        owner = self._in_class()
        if annotation is None or owner is None:
            return
        for sub in ast.walk(annotation):
            if isinstance(sub, (ast.Name, ast.Attribute)) and (
                self.names.typing_name(sub) == "Self"
            ):
                self.edits.replace(*self.edits.node_span(sub), owner)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._type_params(node)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        self._scope.append(node.name)
        self._classes.append(node.name)
        for stmt in node.body:
            self.visit(stmt)
        self._classes.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._type_params(node)
        dropped: list[ast.expr] = []
        for decorator in node.decorator_list:
            if self.names.typing_name(decorator) == "override":
                dropped.append(decorator)
            else:
                self.visit(decorator)
        # The kept ones stay contiguous with the `def` (#595).
        self.edits.drop_decorators(node, dropped)
        every = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        ]
        for arg in every:
            if arg is not None:
                self._self_to_class(arg.annotation)
        self._self_to_class(node.returns)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        scope = {a.arg: a.annotation for a in every if a and a.annotation}
        for stmt in _own_statements(node.body):
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                scope[stmt.target.id] = stmt.annotation
        self._annotations.append(scope)
        self._returns.append(node.returns)
        self._scope.append(None)
        for stmt in node.body:
            self.visit(stmt)
        self._scope.pop()
        self._returns.pop()
        self._annotations.pop()

    # -- Python 3.12+ type parameters and aliases (#624) ---------------------

    def _type_params(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        """The type parameters of *node* (PEP 695), as Codon has them: a
        bare name each.  ``[T: Base]`` loses its bound (Codon would read
        ``T: int`` as a static integer), ``[T: (int, float)]`` its
        constraints and ``[T = int]`` (3.13) its default, as a ``TypeVar``
        does (``TypeVar("T", bound=B)`` becomes ``TypeVar("T")``).  ``*Ts``
        and ``**P`` are refused."""
        for param in getattr(node, "type_params", ()):
            kind = type(param).__name__
            name = getattr(param, "name", "")
            if kind != "TypeVar":
                spelling = "*" if kind == "TypeVarTuple" else "**"
                self.refused.append(
                    (
                        param.lineno,
                        f"the type parameter `{spelling}{name}` of `{node.name}` "
                        "(PEP 646 / PEP 612) is not supported by the check: "
                        "Codon 0.19 has no variadic generics; type the "
                        "parameters (`*args: int`), or leave them unannotated",
                    )
                )
            elif (
                getattr(param, "bound", None) is not None
                or getattr(param, "default_value", None) is not None
            ):
                self.edits.replace(*self.edits.node_span(param), name)

    def visit_TypeAlias(self, node: ast.AST) -> None:
        """``type X = T`` (Python 3.12+) becomes ``X = T``; a generic alias
        (``type Pair[T] = tuple[T, T]``) is refused."""
        name = getattr(node, "name", None)
        value = getattr(node, "value", None)
        if not isinstance(name, ast.Name) or not isinstance(value, ast.expr):
            return
        if getattr(node, "type_params", None):
            params = ", ".join(
                getattr(p, "name", "?") for p in getattr(node, "type_params", ())
            )
            self.refused.append(
                (
                    node.lineno,  # type: ignore[attr-defined]
                    f"the generic type alias `type {name.id}[{params}] = ...` is "
                    "not supported by the check: Codon 0.19 has no generic "
                    "alias. Write the type out where it is used "
                    f"(`{ast.unparse(value)}` with the parameters filled in), "
                    f"or make the alias concrete (`type {name.id} = ...` "
                    "naming no type parameter)",
                )
            )
            return
        start = self.edits.offset(node.lineno, node.col_offset)  # type: ignore[attr-defined]
        end = self.edits.offset(value.lineno, value.col_offset)
        self.edits.replace(start, end, f"{name.id} = ")
        self.visit(value)

    # -- from collections.abc import ... (#586) ------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "collections.abc" or alias.name.startswith(
                "collections.abc."
            ):
                self.refused.append(
                    (
                        node.lineno,
                        "`import collections.abc` is not supported by the check "
                        "(Codon has no collections.abc): write `from "
                        "collections.abc import Sequence` (or `from typing "
                        "import Sequence`) and annotate with the bare name; "
                        + _ABC_HINT,
                    )
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            return
        if node.module == "collections":
            if any(alias.name == "abc" for alias in node.names):
                self.refused.append(
                    (
                        node.lineno,
                        "`from collections import abc` is not supported by the "
                        "check (Codon has no collections.abc): write `from "
                        "collections.abc import Sequence` (or `from typing "
                        "import Sequence`) and annotate with the bare name; "
                        + _ABC_HINT,
                    )
                )
            return
        if node.module != "collections.abc":
            return
        kept: list[str] = []
        for alias in node.names:
            name = _ABC_NAMES.get(alias.name)
            if name is None:
                what = (
                    "`from collections.abc import *`"
                    if alias.name == "*"
                    else f"collections.abc.{alias.name}"
                )
                self.refused.append(
                    (
                        getattr(alias, "lineno", node.lineno),
                        f"{what} is not modelled by the check (Codon has no "
                        f"collections.abc): {_ABC_HINT}; annotate with one of "
                        "them, or with a concrete type (list, dict, set), "
                        'instead (docs/typecheck.md, "typing names")',
                    )
                )
                continue
            local = alias.asname or alias.name
            kept.append(name if name == local else f"{name} as {local}")
        start, end = self.edits.node_span(node)
        lost = self.edits.text(start, end).count("\n")
        if kept:
            text = "from typing import (" + ", ".join(kept) + "\n" * lost + ")"
        else:  # refused: kept on its lines for the line numbers
            text = "(" + "\n" * lost + ")" if lost else "pass"
        self.edits.replace(start, end, text)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.names.typing_name(node.annotation) == "TypeAlias" and node.value:
            target = node.target
            assert target.end_lineno is not None and target.end_col_offset is not None
            start = self.edits.offset(target.end_lineno, target.end_col_offset)
            end = self.edits.offset(node.value.lineno, node.value.col_offset)
            self.edits.replace(start, end, " = ")
        else:
            self._self_to_class(node.annotation)
        if node.value is not None:
            self._store(
                self._target_annotation(node.target, node.annotation), node.value
            )
            self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        base = self._new_type(node.value)
        if base is not None and all(isinstance(t, ast.Name) for t in node.targets):
            self.edits.replace(
                *self.edits.node_span(node.value),
                self.edits.text(*self.edits.node_span(base)),
            )
            return
        if node.targets:
            annotation = self._target_annotation(node.targets[0])
            self._store(annotation, node.value)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if self._assert_never(node.value):
            self.edits.replace(*self.edits.node_span(node), _ASSERT_NEVER)
        else:
            self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and self._assert_never(node.value):
            self.edits.replace(*self.edits.node_span(node), _ASSERT_NEVER)
            return
        if node.value is not None and self._returns:
            self._store(self._returns[-1], node.value)
        self.generic_visit(node)

    def _assert_never(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and not self._local(node.func)
            and self.names.typing_name(node.func) == "assert_never"
            and len(node.args) == 1
            and not node.keywords
        )

    # -- calls --------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        local = self._local(node.func)
        typing = None if local else self.names.typing_name(node.func)
        operator = None if local else self.names.operator_name(node.func)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in self._new_types
            and not self._local(node.func)
            and len(node.args) == 1
            and not node.keywords
            and not isinstance(node.args[0], ast.Starred)
        ):
            # Only the name goes: `LaneId(v)` is `(v)`, and a call inside the
            # argument is rewritten on its own.
            self.edits.replace(*self.edits.node_span(node.func), "")
            self.generic_visit(node)
            return
        if typing == "assert_never":
            # Only a statement or a `return` is rewritten: Codon types the
            # call of the function that always raises as `None`, which would
            # refuse a correct `1 if ok else assert_never(x)`.
            self.refused.append(
                (
                    node.lineno,
                    "assert_never() is checked only as a statement or after "
                    "`return`: write `assert_never(x)` on a line of its own",
                )
            )
        if (
            typing == "TypeVar"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and (len(node.args) > 1 or node.keywords)
        ):
            func = self.edits.text(*self.edits.node_span(node.func))
            first = self.edits.text(*self.edits.node_span(node.args[0]))
            self.edits.replace(*self.edits.node_span(node), f"{func}({first})")
            return
        if operator in ("itemgetter", "attrgetter"):
            lambda_text = self._getter(node, operator)
            if lambda_text is not None:
                self.edits.replace(*self.edits.node_span(node), lambda_text)
                return
        self._call_arguments(node)
        self._append(node)
        self.generic_visit(node)

    def _getter(self, node: ast.Call, which: str) -> str | None:
        """``(lambda v: v[1])`` for ``itemgetter(1)``; ``None`` (and a refusal)
        when an argument is not a literal."""
        ok = bool(node.args) and not node.keywords
        parts: list[str] = []
        for arg in node.args:
            if which == "itemgetter" and _literal_key(arg):
                parts.append(f"v[{self.edits.text(*self.edits.node_span(arg))}]")
            elif (
                which == "attrgetter"
                and isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and _ATTRIBUTE_PATH.match(arg.value)
            ):
                parts.append(f"v.{arg.value}")
            else:
                ok = False
        if not ok:
            self.refused.append(
                (
                    node.lineno,
                    f"operator.{which}() is checked only with literal "
                    f"{'keys' if which == 'itemgetter' else 'attribute names'} "
                    "(Codon's passes its argument wrapped in a tuple): write a "
                    "lambda (`key=lambda v: v[i]`)",
                )
            )
            return None
        body = parts[0] if len(parts) == 1 else f"({', '.join(parts)})"
        return f"(lambda v: {body})"

    # -- caught exceptions put where an Exception is expected ----------------

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        specific = (
            node.name is not None
            and node.type is not None
            and _dotted(node.type) not in ("Exception", "BaseException")
        )
        if node.type is not None:
            self.visit(node.type)
        if specific:
            assert node.name is not None
            self._handlers.append(node.name)
        for stmt in node.body:
            self.visit(stmt)
        if specific:
            self._handlers.pop()

    def _lookup(self, name: str) -> ast.expr | None:
        for scope in reversed(self._annotations):
            if name in scope:
                return scope[name]
        return None

    def _attribute(self, owner: str, name: str, seen: set[str]) -> ast.expr | None:
        if owner in seen:
            return None
        seen.add(owner)
        if name in self._attributes.get(owner, {}):
            return self._attributes[owner][name]
        for base in self._bases.get(owner, []):
            found = self._attribute(base, name, seen)
            if found is not None:
                return found
        return None

    def _target_annotation(
        self, target: ast.AST, declared: ast.expr | None = None
    ) -> ast.expr | None:
        """The annotation of what *target* holds, if known."""
        if declared is not None:
            return declared
        if isinstance(target, ast.Name):
            return self._lookup(target.id)
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and self._classes
        ):
            return self._attribute(self._classes[-1], target.attr, set())
        if isinstance(target, ast.Subscript):
            container = self._target_annotation(target.value)
            return _element(container, mapping=True)  # type: ignore[return-value]
        return None

    def _store(self, annotation: ast.expr | None, value: ast.AST | None) -> None:
        """Rebuild the caught exception *value* if *annotation* wants an
        ``Exception``; a display of them (``[e]``) element by element."""
        if value is None or annotation is None or not self._handlers:
            return
        base = _exception_base(annotation)
        if base is not None and isinstance(value, ast.Name):
            if value.id in self._handlers:
                self.edits.replace(
                    *self.edits.node_span(value),
                    f"{base}({value.id}.typename, {value.id}.message)",
                )
            return
        if isinstance(value, (ast.List, ast.Set, ast.Tuple)):
            element = _element(annotation, mapping=False)
            for item in value.elts:
                self._store(element, item)  # type: ignore[arg-type]

    def _append(self, node: ast.Call) -> None:
        func = node.func
        if (
            not self._handlers
            or not isinstance(func, ast.Attribute)
            or func.attr not in _ADDERS
            or len(node.args) != 1
        ):
            return
        container = self._target_annotation(func.value)
        self._store(_element(container, mapping=False), node.args[0])  # type: ignore[arg-type]

    def _call_arguments(self, node: ast.Call) -> None:
        """A caught exception passed to a parameter of the module's own
        function, method or class annotated ``Exception``."""
        if not self._handlers:
            return
        func = node.func
        parameters: list[tuple[str, ast.expr | None]] | None = None
        if isinstance(func, ast.Name):
            parameters = self._signatures[""].get(func.id)
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and self._classes
        ):
            for owner in [self._classes[-1], *self._bases.get(self._classes[-1], [])]:
                parameters = self._signatures.get(owner, {}).get(func.attr)
                if parameters is not None:
                    break
        if parameters is None:
            return
        for (_, annotation), arg in zip(parameters, node.args):
            self._store(annotation, arg)
        by_name = dict(parameters)
        for keyword in node.keywords:
            if keyword.arg in by_name:
                self._store(by_name[keyword.arg], keyword.value)


_FSTRING_PREFIX = re.compile(r"[fF][rR]?['\"]|[rR][fF]['\"]")
_FSTRING_QUOTE_REFUSAL = (
    "an f-string's replacement field holds a string quoted with the "
    "f-string's own quote (Python 3.12+, PEP 701), which Codon 0.19 cannot "
    "parse, and the check cannot quote it with the other quote, which it "
    "holds too: put the value in a variable before the f-string "
    """(`key = "it's"`, then `f"{d[key]}"`)"""
)
_FSTRING_BACKSLASH_REFUSAL = (
    "an f-string's replacement field holds a backslash (Python 3.12+, PEP "
    "701), which Codon 0.19 cannot parse: put the value in a variable before "
    """the f-string (`sep = "\\n"`, then `f"{sep.join(names)}"`)"""
)


def requote_fstring_fields(source: str) -> tuple[str, list[tuple[int, str]]]:
    """*source* with each string literal of an f-string's replacement field
    that is quoted with the quote of an f-string around it (PEP 701,
    ``f"{name.replace("_", "-")}"``) quoted with the other quote
    (``f"{name.replace('_', '-')}"``), and what cannot be requoted, as
    (line, message) (#624).

    Python before 3.12, and Codon 0.19's parser, end an f-string at its own
    quote, so Codon reports "unexpected dedent" on a later line.  A nested
    f-string is requoted in the same way (its opening and closing quote).
    Only the quote characters change, so every line and column stays.  A
    literal that holds the other quote, a triple-quoted one inside an
    f-string of its quote, and a backslash anywhere in a field (which
    Codon's parser refuses too) are refused, and then nothing is requoted,
    except in an f-string that :mod:`.python_rewrites` writes as ``+``
    parts (:func:`_rewritten_fstring_spans`): its fields are written back
    from the tree there, the field that cannot be requoted as a call
    (#704), so nothing of it is refused here.  Before Python 3.12 the
    tokenizer has no f-string tokens, and such source does not parse: the
    source is returned as it is.
    """
    if "FSTRING_START" not in tokenize.tok_name.values() or not _FSTRING_PREFIX.search(
        source
    ):
        return source, []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source, []
    # The tokenizer's rows: split at "\n" only, as `str.splitlines` would
    # also split at a form feed or "\u2028" a line of the source may hold.
    lines = io.StringIO(source).readlines()
    rewritten = _rewritten_fstring_spans(source, lines)
    #: Whether the outermost f-string the token is in is in *rewritten*:
    #: nothing of it is refused or requoted.
    exempt = False
    #: (row, column, the quote there now, the quote to put there).
    changes: list[tuple[int, int, str, str]] = []
    refused: list[tuple[int, str]] = []
    #: The f-strings around the token: (its quote as it will be, triple,
    #: its quote as written).
    opened: list[tuple[str, bool, str]] = []

    def refuse(line: int, message: str) -> None:
        if not exempt and (line, message) not in refused:
            refused.append((line, message))

    def change(row: int, col: int, quote: str, new: str) -> None:
        # An exempt f-string is written back from the tree: as written, its
        # `{x=}` text keeps its own quotes.
        if not exempt:
            changes.append((row, col, quote, new))

    def quote_of(text: str) -> tuple[str, str, bool]:
        """(prefix, quote character, triple) of a literal or FSTRING_START."""
        body = text.lstrip("rRbBuUfFtT")
        prefix = text[: len(text) - len(body)]
        return prefix, body[:1], body[:3] == body[:1] * 3

    def requoted(quote: str, triple: bool, body: str, line: int) -> str:
        """The quote a literal of *quote* takes inside :data:`opened`."""
        single = {q for q, t, _ in opened if not t}
        every = {q for q, _, _ in opened}
        if triple:
            if quote in every:
                refuse(line, _FSTRING_QUOTE_REFUSAL)
            return quote
        if quote not in single:
            return quote
        other = "'" if quote == '"' else '"'
        if other in single or other in body:
            refuse(line, _FSTRING_QUOTE_REFUSAL)
            return quote
        return other

    for token in tokens:
        name = tokenize.tok_name[token.type]
        row, col = token.start
        if name == "FSTRING_START":
            if not opened:
                exempt = any(start <= token.start < end for start, end in rewritten)
            prefix, quote, triple = quote_of(token.string)
            new = requoted(quote, triple, "", row) if opened else quote
            if new != quote:
                at = col + len(prefix)
                change(row, at, quote, new)
            opened.append((new, triple, quote))
        elif name == "FSTRING_MIDDLE" and len(opened) > 1:
            new, _, quote = opened[-1]
            if "\\" in token.string:
                refuse(row, _FSTRING_BACKSLASH_REFUSAL)
            elif new != quote and new in token.string:
                refuse(row, _FSTRING_QUOTE_REFUSAL)
        elif name == "FSTRING_END" and opened:
            new, _, quote = opened.pop()
            if new != quote:
                change(row, col, quote, new)
        elif token.type == tokenize.STRING and opened:
            if "\\" in token.string:
                refuse(row, _FSTRING_BACKSLASH_REFUSAL)
                continue
            prefix, quote, triple = quote_of(token.string)
            size = 3 if triple else 1
            body = token.string[len(prefix) + size : -size]
            new = requoted(quote, triple, body, row)
            if new != quote:
                change(row, col + len(prefix), quote, new)
                end_row, end_col = token.end
                change(end_row, end_col - 1, quote, new)
    if not changes or refused:
        # Refused: the check fails, and the source stays as it parses.
        return source, refused
    edited = [list(line) for line in lines]
    for row, col, _, new in changes:
        edited[row - 1][col] = new
    return "".join("".join(line) for line in edited), refused


def _rewritten_fstring_spans(
    source: str, lines: list[str]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The (start, end) tokenizer positions of each f-string that
    :mod:`.python_rewrites` writes as ``+`` parts, its fields rebuilt from the
    tree: a field whose expression holds a quote or a backslash is a call
    there (#704), so its quotes need no requoting.  One it refuses (a brace
    or a nested f-string in an expression) is not among them."""
    from .python_rewrites import _fstring_hazard, _fstring_needs_rewrite

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    def column(row: int, offset: int) -> int:
        # ast counts UTF-8 bytes, the tokenizer characters.
        return len(lines[row - 1].encode()[:offset].decode(errors="replace"))

    spans = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.JoinedStr)
            and node.end_lineno is not None
            and node.end_col_offset is not None
            and _fstring_needs_rewrite(node)
            and _fstring_hazard(node) is None
        ):
            spans.append(
                (
                    (node.lineno, column(node.lineno, node.col_offset)),
                    (node.end_lineno, column(node.end_lineno, node.end_col_offset)),
                )
            )
    return spans


def _run(source: str) -> tuple[str, list[tuple[int, str]], list[int]]:
    if not source.endswith("\n"):
        source += "\n"
    source, refused = requote_fstring_fields(source)
    origin = list(range(1, len(source.encode().splitlines()) + 1))
    for _ in range(_MAX_PASSES):
        tree = ast.parse(source)
        edits = _Edits(source.encode().splitlines(keepends=True))
        walker = _Pass(edits, tree, _Names(tree))
        walker.visit(tree)
        for line, message in walker.refused:
            problem = (origin[line - 1], message)
            if problem not in refused:
                refused.append(problem)
        if not edits.edits:
            break
        source = edits.apply()
        origin = [origin[line - 1] for line in edits.line_origin()]
    return source, sorted(refused), origin


def typing_rewrites(source: str) -> str:
    """*source* rewritten, every statement on its line."""
    return _run(source)[0]


def typing_rewrites_lines(source: str) -> tuple[str, list[int]]:
    """:func:`typing_rewrites`, and the line of *source* each line comes from:
    a decorator moved down to its ``def`` (#595) maps to its line."""
    text, _, origin = _run(source)
    return text, origin


def typing_rewrite_refusals(source: str) -> list[tuple[int, str]]:
    """What the rewrites cannot express, as (line, message), in line order."""
    return _run(source)[1]


# ---------------------------------------------------------------------------
# copy.copy / copy.deepcopy of a class of the module (#517)
# ---------------------------------------------------------------------------

_PLAIN = {"object", "ABC", "abc.ABC"}


def _class_var(annotation: ast.expr) -> bool:
    node = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    return (_dotted(node) or "").removeprefix("typing.") == "ClassVar"


def _declared_fields(node: ast.ClassDef) -> list[str]:
    return [
        stmt.target.id
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and not _class_var(stmt.annotation)
    ]


def copy_dunders(tree: ast.Module, exceptions: set[str]) -> list[tuple[str, str]]:
    """``__copy__`` and ``__deepcopy__`` for the classes of a module that
    imports ``copy``, as (the class, the lines) to append after the module.

    Codon's ``copy.copy(x)`` calls ``x.__copy__()`` and ``deepcopy`` calls
    ``x.__deepcopy__()``, which a class of the module lacks.  Each is written
    from the declared fields (a base of the module first), as Python's copies
    the instance's attributes: ``copy`` shares them, ``deepcopy`` copies each.
    A class with a base the module does not define (a framework class), an
    exception, an Enum, a NamedTuple or a class defining either method itself
    is left as it is.
    """
    if not _Names(tree).copy_imported:
        return []
    fields: dict[str, list[str]] = {}
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name in exceptions:
            continue
        if node.keywords:
            continue
        inherited: list[str] | None = []
        for base in node.bases:
            name = _dotted(base)
            if name in fields:
                inherited = [*(inherited or []), *fields[name]]
            elif name not in _PLAIN:
                inherited = None
                break
        if inherited is None:
            continue
        names = [*inherited, *(f for f in _declared_fields(node) if f not in inherited)]
        fields[node.name] = names
        own = {
            m.name
            for m in node.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        lines: list[str] = []
        if "__copy__" not in own:
            body = "".join(f"        r.{f} = self.{f}\n" for f in names)
            lines.append(
                f"    def __copy__(self) -> {node.name}:\n"
                f"        r = {node.name}.__new__()\n{body}        return r\n"
            )
        if "__deepcopy__" not in own:
            body = "".join(f"        r.{f} = _acs_deepcopy(self.{f})\n" for f in names)
            lines.append(
                f"    def __deepcopy__(self) -> {node.name}:\n"
                "        from copy import deepcopy as _acs_deepcopy\n\n"
                f"        r = {node.name}.__new__()\n{body}        return r\n"
            )
        if lines:
            out.append(
                (
                    node.name,
                    "\n\n" f"@extend\nclass {node.name}:\n" + "".join(lines),
                )
            )
    return out
