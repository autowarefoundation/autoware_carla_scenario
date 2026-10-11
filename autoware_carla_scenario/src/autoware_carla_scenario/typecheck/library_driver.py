"""The calls the library check compiles: everything public, with typed values.

Codon only checks a function it compiles, and it compiles a function only
when something calls it.  To check a framework module on its own, the library
check (:func:`.check.typecheck_library`) appends a function to the module
that calls every public function and every method of every public class,
each with a value of every parameter's annotated type::

    def _acs_library_check():
        normalize_angle_deg(_acs_value(float))
        Vector3(_acs_value(float), _acs_value(float), _acs_value(float))
        _acs_value(Vector3).dot(_acs_value(Vector3))
        Vector3.zero()

``_acs_value(T)`` (the model's ``_tc.codon``) is a ``T`` read through a null
pointer: Codon types it as a ``T``, and nothing compiled is ever run.  The
calls are written in the module's own namespace, so an annotation means what
it means in the module.  A parameter annotated with a union
(``Lanelet2Pose | OpenDrivePose``) is called once with each member.

Some annotations give the check nothing to call with (``Any``, ``object``,
``Callable``, none at all): each is reported as a problem of the module, to
be fixed there with a type Codon can check.  ``object`` is accepted for the
other operand of ``__eq__`` and ``__ne__``, which Python requires; it is
given a value of the class itself.
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Collection
from dataclasses import dataclass, field

from .driver import Driver
from .transform import (
    _ENUM_BASES,
    _dotted,
    _is_none,
    _union_members,
    codon_annotation,
)

__all__ = [
    "LIBRARY_CHECK",
    "LibraryChecks",
    "render_library_checks",
    "render_library_driver",
    "uncalled_nodes",
    "workspace_module",
]

#: The function appended to every checked module.
LIBRARY_CHECK = "_acs_library_check"
#: Where the checked modules live in the check's workspace.
LIBRARY_PACKAGE = "_acs_lib"
#: The most calls made for one function: past it, a union parameter is
#: varied one at a time, the others taking their first member.
MAX_CALLS = 16

_DATACLASS = {"dataclass", "dataclasses.dataclass"}
_SKIPPED_DECORATORS = {"overload", "typing.overload"}
_PROPERTIES = {"property", "functools.cached_property", "cached_property"}
_OBJECT_OPERAND = {"__eq__", "__ne__"}


def workspace_module(module: str) -> str:
    """The name a checked framework module is compiled under.

    Codon names a file only by its base name in its errors, and a package
    has many ``frames.py`` and ``__init__.py``: each checked module is one
    file of :data:`LIBRARY_PACKAGE` named by its whole dotted name
    (``autoware_carla_scenario.coordinate.frames`` ->
    ``_acs_lib.autoware_carla_scenario__coordinate__frames``).
    """
    return f"{LIBRARY_PACKAGE}.{module.replace('.', '__')}"


@dataclass
class LibraryChecks:
    """The function appended to one checked module."""

    source: str
    #: Line of :attr:`source` (1-based) -> the call on it, as the author reads it.
    labels: dict[int, str] = field(default_factory=dict)
    #: (line of the module, message): a parameter the check cannot call with.
    problems: list[tuple[int, str]] = field(default_factory=list)


class _Renderer:
    def __init__(self, uncalled: Collection[str] = ()) -> None:
        self.uncalled = set(uncalled)
        self.lines = [
            "",
            "",
            "from autoware_carla_scenario._tc import _acs_value",
            "",
            "",
            f"def {LIBRARY_CHECK}():",
        ]
        self.labels: dict[int, str] = {}
        self.problems: list[tuple[int, str]] = []

    def emit(self, text: str, label: str) -> None:
        self.lines.append(f"    {text}")
        self.labels[len(self.lines)] = label

    def types(self, arg: ast.arg, owner: str | None, where: str) -> list[str] | None:
        """The Codon types to call with for *arg*; ``None`` after a problem."""
        annotation = arg.annotation
        if annotation is None:
            self.problems.append(
                (arg.lineno, f"{where}: parameter `{arg.arg}` has no annotation")
            )
            return None
        text = codon_annotation(annotation)
        if text is not None:
            return [text]
        members = _union_members(annotation)
        if members is not None:
            texts = [
                codon_annotation(m, class_level=True)
                for m in members
                if not _is_none(m)
            ]
        else:
            texts = [codon_annotation(annotation, class_level=True)]
        if texts and all(t is not None for t in texts):
            return [t for t in texts if t is not None]
        method = where.rsplit(".", 1)[-1]
        if owner is not None and method in _OBJECT_OPERAND:
            if _dotted(annotation) == "object":
                return [owner]
        self.problems.append(
            (
                arg.lineno,
                f"{where}: Codon cannot express `{ast.unparse(annotation)}`, the "
                f"annotation of parameter `{arg.arg}`; annotate it with a type "
                "Codon can check",
            )
        )
        return None

    def calls(
        self,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        head: str,
        where: str,
        *,
        drop_first: bool,
        owner: str | None = None,
    ) -> None:
        args = func.args
        positional = [*args.posonlyargs, *args.args][1 if drop_first else 0 :]
        every = [*positional, *args.kwonlyargs]
        choices = [self.types(arg, owner, where) for arg in every]
        if any(c is None for c in choices):
            return
        self._emit_calls(
            head, where, positional, args.kwonlyargs, [c for c in choices if c]
        )

    def _emit_calls(
        self,
        head: str,
        where: str,
        positional: list[ast.arg],
        keyword: list[ast.arg],
        choices: list[list[str]],
    ) -> None:
        for combo in _combinations(choices):
            values = [f"_acs_value({t})" for t in combo[: len(positional)]]
            values += [
                f"{a.arg}=_acs_value({t})"
                for a, t in zip(keyword, combo[len(positional) :])
            ]
            signature = ", ".join(
                f"{a.arg}: {t}" for a, t in zip([*positional, *keyword], combo)
            )
            self.emit(f"{head}({', '.join(values)})", f"{where}({signature})")

    def function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name not in self.uncalled:
            self.calls(node, node.name, node.name, drop_first=False)

    def cls(self, node: ast.ClassDef) -> None:
        name = node.name
        bases = [(_dotted(b) or "").rsplit(".", 1)[-1] for b in node.bases]
        is_enum = any(b in _ENUM_BASES for b in bases)
        decorators = {
            _dotted(d.func if isinstance(d, ast.Call) else d)
            for d in node.decorator_list
        }
        methods = [
            m
            for m in node.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        init = next((m for m in methods if m.name == "__init__"), None)
        if f"{name}.__init__" in self.uncalled:
            pass
        elif init is not None and not is_enum:
            self.calls(init, name, f"{name}.__init__", drop_first=True, owner=name)
        elif decorators & _DATACLASS and not is_enum:
            self._dataclass_init(node)
        for method in methods:
            dunder = method.name.startswith("__") and method.name.endswith("__")
            if method.name == "__init__" or (
                method.name.startswith("_") and not dunder
            ):
                continue
            kinds = {_dotted(d) or "" for d in method.decorator_list}
            where = f"{name}.{method.name}"
            if where in self.uncalled:
                continue
            if kinds & _SKIPPED_DECORATORS or any(
                k.endswith((".setter", ".deleter")) for k in kinds
            ):
                continue
            if kinds & _PROPERTIES:
                self.emit(f"_acs_value({name}).{method.name}", where)
            elif "staticmethod" in kinds:
                self.calls(method, f"{name}.{method.name}", where, drop_first=False)
            elif "classmethod" in kinds:
                self.calls(method, f"{name}.{method.name}", where, drop_first=True)
            else:
                self.calls(
                    method,
                    f"_acs_value({name}).{method.name}",
                    where,
                    drop_first=True,
                    owner=name,
                )

    def _dataclass_init(self, node: ast.ClassDef) -> None:
        """The generated ``__init__`` of a dataclass: one value per field."""
        fields: list[ast.arg] = []
        for stmt in node.body:
            if not (
                isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ):
                continue
            if (_dotted(stmt.annotation) or "").endswith("KW_ONLY"):
                break  # keyword-only fields: not modelled, so not called
            if isinstance(stmt.annotation, ast.Subscript) and (
                _dotted(stmt.annotation.value) or ""
            ).endswith("ClassVar"):
                continue
            value = stmt.value
            if isinstance(value, ast.Call) and any(
                kw.arg == "init"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in value.keywords
            ):
                continue
            arg = ast.arg(stmt.target.id, stmt.annotation)
            arg.lineno = stmt.lineno
            fields.append(arg)
        choices = []
        for arg in fields:
            text = (
                codon_annotation(arg.annotation, class_level=True)
                if arg.annotation is not None
                else None
            )
            if text is None:
                self.problems.append(
                    (
                        arg.lineno,
                        f"{node.name}: Codon cannot express the annotation of "
                        f"field `{arg.arg}`; annotate it with a type Codon can check",
                    )
                )
                return
            choices.append([text])
        self._emit_calls(node.name, f"{node.name}.__init__", fields, [], choices)


def _combinations(choices: list[list[str]]) -> list[tuple[str, ...]]:
    total = 1
    for c in choices:
        total *= len(c)
    if total <= MAX_CALLS:
        return list(itertools.product(*choices))
    first = tuple(c[0] for c in choices)
    out = [first]
    for i, c in enumerate(choices):
        for alternative in c[1:]:
            out.append(first[:i] + (alternative,) + first[i + 1 :])
    return out


def uncalled_nodes(
    tree: ast.Module, uncalled: Collection[str]
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The definitions in *tree* of the functions *uncalled* names.

    *uncalled* names a function as ``function`` or ``Class.method``; a name
    with no definition in *tree* is ignored.
    """
    wanted = set(uncalled)
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in wanted:
                out.append(node)
        elif isinstance(node, ast.ClassDef):
            out += [
                m
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and f"{node.name}.{m.name}" in wanted
            ]
    return out


def render_library_checks(
    tree: ast.Module, uncalled: Collection[str] = ()
) -> LibraryChecks:
    """The function to append to the module *tree* is the source of.

    *uncalled* names the functions the check leaves out (``function`` or
    ``Class.method``; ``Class.__init__`` for the constructor, generated or
    not), as :data:`.library.UNCALLED` lists them.
    """
    renderer = _Renderer(uncalled)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                renderer.function(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            renderer.cls(node)
    renderer.lines.append("    pass")
    return LibraryChecks(
        "\n".join(renderer.lines) + "\n", renderer.labels, renderer.problems
    )


def render_library_driver(modules: list[str]) -> Driver:
    """The program that runs the appended function of each module in *modules*."""
    lines: list[str] = []
    calls: list[str] = []
    for i, module in enumerate(modules):
        lines.append(
            f"from {workspace_module(module)} import {LIBRARY_CHECK} as _acs_check_{i}"
        )
        calls.append(f"_acs_check_{i}()")
    return Driver("\n".join([*lines, "", "", *calls]) + "\n")
