"""Two class mistakes Codon compiles and Python refuses at run time.

Codon gives every field of a class its zero value when the object is built,
so an attribute nothing assigns reads as ``0``, ``""`` or an empty list, where
Python raises ``AttributeError``.  Two ways an author gets there:

* **A framework base's ``__init__`` skipped** (#587).  A class of the
  scenario's modules derives from a framework class whose ``__init__`` sets
  attributes (``BaseCondition`` its ``label``, ``BaseScenario`` its
  ``ego_config`` and spawn pose, ...) and defines an ``__init__`` of its own
  that never calls the base's.  The runner then reads ``condition.label``.
  Refused at the ``def __init__`` line.  The framework classes are the real
  ones: each class of the method resolution order after the author's, from
  the framework package, whose ``__init__`` (as written, or the one
  ``@dataclass`` generates) assigns an attribute of ``self``.  One whose
  ``__init__`` assigns nothing is left alone, as is a class deriving from no
  framework class, and so is an ``__init__`` that itself assigns every
  attribute the base's would.  The base counts as called when the ``__init__`` names
  ``__init__`` anywhere in its body, on any path (``super().__init__(...)``,
  ``super(Cls, self).__init__(...)``, ``Base.__init__(self, ...)``, inside
  an ``if`` or a ``try``), or calls a method of ``self`` while a method of the
  class names it: the check does not follow paths.

* **A declared attribute never assigned** (#588).  A bare class-level
  annotation (``_count: int``) on a class of the scenario's modules, as the
  check asks attributes to be declared, that nothing assigns, but that a
  method of the class (or of a subclass) reads (``self._count``, ``self._count
  += 1``).  Refused at the annotation.  The rule is conservative: the
  attribute counts as assigned when anything in the scenario's modules
  stores an attribute of that name, on any object (``x.name = ...``,
  ``self.name: T = ...``, ``self.a, self.name = ...``, ``for self.name in``,
  ``setattr(x, "name", ...)``, ``object.__setattr__(x, "name", ...)``), when
  the source of a framework base stores one, or the framework stores one on
  an object not its own ``self`` (``scenario.name = ...``), and when the
  class (or a base) gives it a value (a class attribute, a property, a
  slot); the class is left alone when it is a dataclass field, a
  ``NamedTuple``, ``TypedDict``, ``Protocol`` or enum, when its family assigns
  attributes by a computed name (``setattr`` with a variable, ``__dict__``,
  ``vars()``), defines ``__getattr__``, or derives from a class neither of the
  scenario, of the framework nor built in.  An augmented assignment
  (``self._count += 1``) reads before it stores, so it is a read.

Both rules read the classes Python built (the modules are imported by then)
beside the syntax: a class that is not a module-level name is left alone.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import sys
import textwrap
from collections.abc import Callable, Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

_PACKAGE = "autoware_carla_scenario"

#: Modules whose classes add no instance attributes an author's class relies on.
_PLAIN_MODULES = frozenset({"builtins", "abc", "typing", "collections.abc"})

_Function = ast.FunctionDef | ast.AsyncFunctionDef


class ClassProblem(NamedTuple):
    path: str
    line: int
    message: str


def _in_package(module: str) -> bool:
    return module == _PACKAGE or module.startswith(_PACKAGE + ".")


def _classes(
    modules: Mapping[str, tuple[str, ast.Module]],
) -> Iterator[tuple[str, str, ast.ClassDef, type]]:
    """(module, path, node, class) for each module-level class Python built."""
    for name, (path, tree) in sorted(modules.items()):
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            cls = getattr(loaded, node.name, None)
            if (
                inspect.isclass(cls)
                and cls.__module__ == name
                and cls.__qualname__ == node.name
            ):
                yield name, path, node, cls


def _self_name(function: _Function) -> str | None:
    args = [*function.args.posonlyargs, *function.args.args]
    if not args:
        return None
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id in (
            "staticmethod",
            "classmethod",
        ):
            return None
    return args[0].arg


def _stored_attributes(tree: ast.AST) -> Iterator[tuple[ast.expr | None, str]]:
    """(object, attribute) for each attribute *tree* assigns.

    The object is ``None`` for a ``setattr`` whose object is not a plain name.
    An augmented assignment and a bare ``self.x: T`` are not assignments.
    """
    not_stores: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) or (
            isinstance(node, ast.AnnAssign) and node.value is None
        ):
            not_stores.add(id(node.target))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and id(node) not in not_stores
        ):
            yield node.value, node.attr
        elif isinstance(node, ast.Call):
            named = _setattr_name(node)
            if isinstance(named, str):
                yield node.args[0], named


def _setattr_name(call: ast.Call) -> str | bool | None:
    """The literal name a ``setattr`` / ``__setattr__`` call assigns; ``True``
    for one whose name is computed, ``None`` for any other call."""
    func = call.func
    if not (
        (isinstance(func, ast.Name) and func.id == "setattr")
        or (isinstance(func, ast.Attribute) and func.attr == "__setattr__")
    ):
        return None
    # object.__setattr__(self, "x", v), setattr(self, "x", v): the name is the
    # second argument; self.__setattr__("x", v) has it first.
    position = 0 if isinstance(func, ast.Attribute) and len(call.args) == 2 else 1
    if len(call.args) <= position:
        return True
    name = call.args[position]
    if isinstance(name, ast.Constant) and isinstance(name.value, str):
        return name.value
    return True


def _source_tree(obj: Any) -> ast.Module | None:
    try:
        return ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None


# ---------------------------------------------------------------------------
# #587: a framework base's __init__ skipped
# ---------------------------------------------------------------------------


def _init_assigns(base: type) -> list[str]:
    """The attributes of ``self`` *base*'s own ``__init__`` assigns."""
    init = inspect.unwrap(base.__dict__["__init__"])
    code = getattr(init, "__code__", None)
    if code is None:
        return []
    if dataclasses.is_dataclass(base) and code.co_filename.startswith("<"):
        return [f.name for f in dataclasses.fields(base) if f.init]
    tree = _source_tree(init)
    if tree is None or not tree.body or not isinstance(tree.body[0], _Function):
        return []
    function = tree.body[0]
    me = _self_name(function)
    names: list[str] = []
    for obj, attr in _stored_attributes(function):
        if isinstance(obj, ast.Name) and obj.id == me and attr not in names:
            names.append(attr)
    return names


def _names_init(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr == "__init__" for n in ast.walk(node)
    )


def _calls_base_init(cls_node: ast.ClassDef, init: _Function) -> bool:
    if _names_init(ast.Module(body=init.body, type_ignores=[])):
        return True
    me = _self_name(init)
    calls_own = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == me
        for n in ast.walk(init)
    )
    # A helper of the class calling the base's __init__ for it.
    return calls_own and any(
        isinstance(stmt, _Function) and stmt is not init and _names_init(stmt)
        for stmt in cls_node.body
    )


def skipped_base_inits(
    modules: Mapping[str, tuple[str, ast.Module]],
    *,
    framework: Callable[[type], bool] | None = None,
) -> list[ClassProblem]:
    """Each ``__init__`` of *modules* that never calls a framework base's (#587).

    *modules* maps a module's dotted name to its path and syntax tree; the
    module must be imported.  *framework* says which classes are the
    framework's: by default those of the package outside *modules*.
    """
    if framework is None:

        def framework(base: type) -> bool:
            return _in_package(base.__module__) and base.__module__ not in modules

    problems: list[ClassProblem] = []
    for _name, path, node, cls in _classes(modules):
        # The last one: Python binds it, an earlier one is an @overload stub.
        init = next(
            (
                stmt
                for stmt in reversed(node.body)
                if isinstance(stmt, _Function) and stmt.name == "__init__"
            ),
            None,
        )
        if init is None or "__init__" not in cls.__dict__:
            continue
        if _calls_base_init(node, init):
            continue
        me = _self_name(init)
        own = {
            attr
            for obj, attr in _stored_attributes(init)
            if isinstance(obj, ast.Name) and obj.id == me
        }
        skipped = [
            (base, assigned)
            for base in cls.__mro__[1:]
            if "__init__" in base.__dict__ and framework(base)
            for assigned in [_init_assigns(base)]
            if assigned
        ]
        # One that sets all the bases would itself is left alone.
        if not skipped or own.issuperset(a for _b, names in skipped for a in names):
            continue
        base, assigned = skipped[0]
        shown = ", ".join(assigned[:4]) + (", ..." if len(assigned) > 4 else "")
        problems.append(
            ClassProblem(
                path,
                init.lineno,
                f"{node.name}.__init__ never calls {base.__name__}.__init__, "
                f"which sets {shown} on self: Codon zero-initialises them, "
                "and Python raises AttributeError where the framework reads "
                f"them; call super().__init__(...) in {node.name}.__init__",
            )
        )
    return problems


# ---------------------------------------------------------------------------
# #588: a declared attribute never assigned
# ---------------------------------------------------------------------------


def _python_files(package: str) -> list[Path]:
    spec = sys.modules.get(package)
    paths = getattr(spec, "__path__", None)
    if not paths:
        return []
    return sorted(p for root in paths for p in Path(root).rglob("*.py"))


@lru_cache(maxsize=1)
def _framework_stores() -> frozenset[str]:
    """Each attribute name the framework package assigns on an object other
    than the method's own (``scenario.goal_pose = ...``): the framework may
    assign it on an author's object from outside."""
    out: set[str] = set()
    for path in _python_files(_PACKAGE):
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        out.update(
            attr
            for obj, attr in _stored_attributes(tree)
            if not (isinstance(obj, ast.Name) and obj.id in ("self", "cls"))
        )
    return frozenset(out)


@lru_cache(maxsize=256)
def _class_stores(cls: type) -> frozenset[str] | None:
    """Each attribute name the source of *cls* assigns, ``None`` when there is
    no source to read or it assigns by a computed name."""
    tree = _source_tree(cls)
    if tree is None or _dynamic(tree):
        return None
    return frozenset(attr for _obj, attr in _stored_attributes(tree))


def _not_plain_class(cls: type) -> bool:
    """A class whose annotations are not instance attributes to assign."""
    return (
        dataclasses.is_dataclass(cls)
        or issubclass(cls, (tuple, dict, enum.Enum))
        or bool(getattr(cls, "_is_protocol", False))
        or hasattr(cls, "__attrs_attrs__")
        or hasattr(cls, "model_fields")
    )


def _dynamic(tree: ast.AST) -> bool:
    """Whether *tree* may assign an attribute of ``self`` by a computed name
    (``setattr(self, name, v)``, ``self.__dict__``, ``vars(self)``)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _setattr_name(node) is True:
                return True
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "vars"
                and (not node.args or _is_self(node.args[0]))
            ):
                return True
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "__dict__"
            and _is_self(node.value)
        ):
            return True
    return False


def _is_self(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "self"


def _reads(tree: ast.AST, name: str) -> int | None:
    """The line of the first read of ``self.<name>`` in a method of *tree*."""
    lines: list[int] = []
    for function in ast.walk(tree):
        if not isinstance(function, _Function):
            continue
        me = _self_name(function)
        if me is None:
            continue
        augmented = {
            id(n.target) for n in ast.walk(function) if isinstance(n, ast.AugAssign)
        }
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == name
                and isinstance(node.value, ast.Name)
                and node.value.id == me
                and (isinstance(node.ctx, ast.Load) or id(node) in augmented)
            ):
                lines.append(node.lineno)
    return min(lines) if lines else None


def _is_class_var(annotation: ast.expr) -> bool:
    for node in ast.walk(annotation):
        name = (
            node.id
            if isinstance(node, ast.Name)
            else node.attr
            if isinstance(node, ast.Attribute)
            else node.value
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            else None
        )
        if isinstance(name, str) and any(
            word in name for word in ("ClassVar", "Final", "InitVar")
        ):
            return True
    return False


def _inherited_stores(cls: type, own: Mapping[type, Any]) -> set[str] | None:
    """What the bases of *cls* outside *own* assign; ``None`` when one of
    them is neither the framework's (with a source) nor built in."""
    out: set[str] = set()
    for base in cls.__mro__[1:]:
        if base in own or base.__module__ in _PLAIN_MODULES:
            continue
        assigned = _class_stores(base) if _in_package(base.__module__) else None
        if assigned is None:
            return None
        out |= assigned
    return out


def unassigned_attributes(
    modules: Mapping[str, tuple[str, ast.Module]],
) -> list[ClassProblem]:
    """Each bare class-level annotation of *modules* nothing assigns but a
    method reads (#588).  The modules must be imported."""
    built = list(_classes(modules))
    if not built:
        return []
    nodes: dict[type, ast.ClassDef] = {cls: node for _m, _p, node, cls in built}
    stored = set(_framework_stores())
    for _path, tree in modules.values():
        stored.update(attr for _obj, attr in _stored_attributes(tree))

    def family(cls: type) -> list[type]:
        return [
            other for other in nodes if other in cls.__mro__ or cls in other.__mro__
        ]

    problems: list[ClassProblem] = []
    for _name, path, node, cls in built:
        declared = [
            stmt
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign)
            and stmt.value is None
            and stmt.simple
            and isinstance(stmt.target, ast.Name)
            and not _is_class_var(stmt.annotation)
        ]
        if not declared or _not_plain_class(cls):
            continue
        if getattr(cls, "__getattr__", None) is not None:
            continue
        if cls.__getattribute__ is not object.__getattribute__:
            continue
        inherited = _inherited_stores(cls, nodes)
        if inherited is None:
            continue
        kin = family(cls)
        if any(_dynamic(nodes[k]) for k in kin):
            continue
        for stmt in declared:
            name = stmt.target.id  # type: ignore[union-attr]
            if name in stored or name in inherited or hasattr(cls, name):
                continue
            fields = getattr(cls, "__dataclass_fields__", {})
            if name in fields:
                continue
            read = [
                line
                for k in kin
                if cls in k.__mro__
                for line in [_reads(nodes[k], name)]
                if line is not None
            ]
            if not read:
                continue
            problems.append(
                ClassProblem(
                    path,
                    stmt.lineno,
                    f"{node.name}.{name} is declared but never assigned: Codon "
                    "zero-initialises it, and Python raises AttributeError "
                    f"where a method reads self.{name}; assign self.{name} in "
                    f"{node.name}.__init__, or give it a default "
                    f"({name}: ... = <value>)",
                )
            )
    return problems
