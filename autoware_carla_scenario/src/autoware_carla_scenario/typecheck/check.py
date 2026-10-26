"""Compile a scenario with Codon before it runs (docs/typecheck.md).

:func:`typecheck_scenario` copies the scenario's own modules into a scratch
workspace, next to the Codon model of the framework (``codon/``), rewrites
them for Codon (:mod:`.transform`), adds a driver program that builds the
scenario from its config and calls ``setup()`` (:mod:`.driver`), and compiles
the lot with ``codon build -llvm``.  Nothing compiled is ever run: the
compile *is* the check, and a scenario that does not compile is refused before
the runner touches CARLA.

What is checked is everything ``setup()`` and ``is_done()`` reach: the
framework's public API (as far as the model goes), the CARLA API, which is
typesafe_carla's Codon library (``import typesafe_carla.carla as carla``,
the import the runtime uses too), and the scenario package's own modules.  A module outside those
(yaml, say) has no Codon model, and a scenario importing one fails the check
with a message saying so.  Each of the scenario's own modules also has the
library check's calls appended (:func:`.library_driver.render_library_checks`),
which the driver runs, so every function, method and class-level default in
it is compiled, whether ``setup()`` reaches it or not; one the check cannot
call (a parameter without an annotation it can call with, or a return
annotation naming ``Any``) is left to what reaches it, with a warning.  The
result of each method overriding a framework hook is checked against the
hook's return type (#238), and, after the compile, mypy reports each value
that may be ``None`` used where it may not be, which Codon unwraps
implicitly (:mod:`.none_check`, #239).

:func:`typecheck_library` compiles the framework's own modules instead (those
:mod:`.library` lists as checked), each with calls to every function and method,
protected ones included, appended (:mod:`.library_driver`), next to the same model.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import enum
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

from .driver import (
    DRIVER_MODULE,
    render_driver,
    render_module_checks,
    render_odd_driver,
)
from .toolchain import (
    SUPPORTED_CODON_SERIES,
    Toolchain,
    ToolchainError,
    codon_environment,
    codon_path_dir,
    find_codon,
    is_supported_version,
)
from .library import (
    CHECKED,
    MODELLED_IMPORTS,
    REPLACED_MODELS,
    UNCALLED,
    package_modules,
)
from .dead_isinstance import dead_isinstances
from .none_check import (
    BYTES_STR,
    CLASS_VAR,
    ENUM_EQ,
    FINAL,
    FROZEN,
    OVERRIDE,
    READ_ONLY_ABC,
    STR_FORMAT,
    NoneCheckSkipped,
    NoneFinding,
    none_given,
    run_none_check,
    search_path,
)
from .not_modelled import NOT_MODELLED, not_modelled_member
from .library_driver import (
    call_sites,
    render_library_checks,
    render_library_driver,
    uncalled_nodes,
    workspace_module,
)
from .local_bindings import (
    int_literal_hint,
    none_first_binding_message,
    none_first_bindings,
    none_mismatch_hint,
)
from .module_cycles import cycle_hint
from .python_rewrites import REWRITE_HINTS, python_rewrite_refusals
from .transform import (
    PRELUDE,
    class_declarations,
    classes_with_init,
    conflicting_imports,
    enum_class_names,
    literal_all,
    redirect_imports,
    relative_module_imports,
    transform_source_lines,
    undeclared_attributes,
)
from .call_diagnostics import (
    Refiner,
    _Files,
    is_internal_frame,
    plain_literal_types,
    slip_hint,
)
from .class_rules import skipped_base_inits, unassigned_attributes
from .return_paths import return_problems
from .static_raises import literal_argument_refusals, subscripted_isinstances
from .typing_rewrites import TYPING_HINTS, typing_rewrite_refusals

__all__ = [
    "Diagnostic",
    "ScenarioTypeError",
    "TypeCheckResult",
    "available_toolchain",
    "find_supported_codon",
    "model_dir",
    "typecheck_library",
    "typecheck_odd",
    "typecheck_scenario",
]

#: Seconds a single compile may take.
DEFAULT_TIMEOUT_SECONDS = 600.0

logger = logging.getLogger(__name__)

_PACKAGE = "autoware_carla_scenario"
#: typesafe_carla's Codon library, in its CODON_PATH directory.
_LIBRARY = "typesafe_carla"
#: Lines :data:`.transform.PRELUDE` adds above every rewritten module.
_PRELUDE_LINES = PRELUDE.count("\n")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LOCATION = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>\d+)(?: \((?P<col>\d+)(?:-\d+)?\))?: "
)
_ERROR = re.compile(r"error: ?(?P<message>.*)$")
_CRASH_MARKERS = ("Assert failed", "Segmentation fault", "Aborted")
#: Codon 0.19's code generation refusing the IR it generated itself: a bug of
#: Codon's, no error of the program (#263).
_LLVM_ERROR = "LLVM: error:"
#: Codon 0.19 aborting with no location, as for a ``Callable`` attribute that
#: a value of another type is stored in (#488).
_BAD_ALLOC = "std::bad_alloc"
#: LLVM refusing a call that passes a function where a ``Callable`` is
#: expected, with the signature of another: Codon 0.19 converts the function
#: without comparing the two and only the verifier of the IR it generated
#: notices (#488).
_SIGNATURE_MISMATCH = "Call parameter type does not match function signature"
#: Calls that take a zero-argument lambda as a factory, not as a ``Callable``.
_FACTORY_CALLS = {"field", "defaultdict"}
_PASSED_FUNCTION = re.compile(r'ptr @"std\.[\w.:]*?\.(?P<fn>\w+)\.\d+:\d+\[')
#: The class an LLVM error's frame names, as in
#: ``(__internal__.opt_ref_new_rtti:0[,std.pkg.module.Cls.0])``.
_LLVM_CLASS = re.compile(
    r"\(__internal__\.\w+:\d+\[.*?(?P<cls>[A-Za-z_][\w.]*)\.\d+\]+\)"
)
#: Where an error in the generated driver is reported: the scenario does not
#: build the way the runner builds it (its constructor, setup() or is_done()).
_DRIVER_LABEL = "<the scenario as register_scenario() builds it>"


#: The model's directory of additions to Codon's own stdlib modules (#378):
#: ``itertools.codon`` there is appended to Codon's ``itertools.codon``, so the
#: workspace's module has every name Codon's has and the additions.  (A module
#: that replaced Codon's would lose what the compiler finds by name, such as
#: ``functools.partial``; the names of ``functools`` are the transform's.)
_STDLIB_EXTENSIONS = "_stdlib"

#: Why a module Codon has no model for is refused, and what to write instead
#: (#378): appended to the refusal at the import line.
_UNMODELLED_HINTS = {
    "warnings": "log with logger.warning(...) instead of warnings.warn(...)",
    "types": "name the type itself (SimpleNamespace is a @dataclass or a plain "
    "class; MappingProxyType is a dict; annotate the tb of __exit__ as object)",
    "textwrap": "build the string with str methods and join()",
    "traceback": "log the exception with logger.exception(...)",
    "io": "use a str, or a pathlib.Path opened with open()",
    "csv": "read the file with Path.read_text().splitlines() and split(',') "
    "on each line (a quoted field needs more care), and write one with "
    "','.join(fields) and Path.write_text()",
    "uuid": "use a counter, or os.getpid() for an id that differs per process",
}


def model_dir() -> Path:
    """The directory holding the Codon model (the workspace's base)."""
    return Path(__file__).resolve().parent / "codon"


@lru_cache(maxsize=8)
def _supported(tc: Toolchain) -> Toolchain:
    version = tc.version()
    if not is_supported_version(version):
        raise ToolchainError(
            f"{tc.source}: {tc.executable} is Codon {version or '(unknown version)'}; "
            f"the check is written for Codon {SUPPORTED_CODON_SERIES}.x"
        )
    return tc


def find_supported_codon() -> Toolchain:
    """The Codon :func:`.toolchain.find_codon` finds, if it is a supported release.

    The source rewrite and the model target one Codon release series, so
    another is treated as no Codon at all.  typesafe_carla's ``CODON_PATH``
    directory has to be there too (it is created here, once, before any
    concurrent check reads it): scenarios are compiled against it.

    Raises:
        ToolchainError: No Codon or no typesafe_carla was found, the Codon
            is of another release series, or typesafe_carla's ``CODON_PATH``
            directory cannot be created.
    """
    tc = _supported(find_codon())
    codon_path_dir()
    return tc


def available_toolchain() -> Toolchain | None:
    """The Codon the checker runs (:func:`find_supported_codon`), or ``None``."""
    try:
        return find_supported_codon()
    except ToolchainError:
        return None


@lru_cache(maxsize=8)
def _codon_stdlib(tc: Toolchain) -> frozenset[str]:
    """Top-level module names of Codon's standard library."""
    stdlib = tc.codon_dir / "lib" / "codon" / "stdlib"
    if stdlib.is_dir():
        return frozenset(p.name.split(".")[0] for p in stdlib.iterdir())
    return frozenset(
        {"math", "random", "itertools", "collections", "functools", "sys", "os"}
    )


@lru_cache(maxsize=1)
def _shims() -> frozenset[str]:
    """Modules the model directory provides besides the framework's own."""
    return frozenset(p.name.split(".")[0] for p in model_dir().iterdir()) - {
        _PACKAGE,
        _STDLIB_EXTENSIONS,
        "__pycache__",
    }


@lru_cache(maxsize=1)
def _model_modules() -> frozenset[str]:
    # One module per public package (coordinate.codon), and the odd module of
    # a checked package that is modelled on its own (utils/traffic_light.codon):
    # the package itself is checked, so its __init__.codon models nothing.
    root = model_dir() / _PACKAGE
    names = {_PACKAGE}
    names.update(
        ".".join([_PACKAGE, *p.relative_to(root).with_suffix("").parts])
        for p in root.rglob("*.codon")
        if p.stem != "__init__"
    )
    return frozenset(names)


def _model_file(module: str) -> Path:
    """The model file of *module*, one of :func:`_model_modules`."""
    root = model_dir() / _PACKAGE
    parts = module.split(".")[1:]
    if not parts:
        return root / "__init__.codon"
    return root.joinpath(*parts[:-1], f"{parts[-1]}.codon")


@lru_cache(maxsize=None)
def _model_names(module: str) -> frozenset[str]:
    """The names model module *module* declares: what it defines or imports."""
    names: set[str] = set()
    for node in ast.parse(_model_file(module).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return frozenset(names)


def _not_modelled(module: str, names: list[str]) -> list[str]:
    """Why ``from module import names`` fails the check: each name the model of
    *module* leaves out on purpose (:data:`.not_modelled.NOT_MODELLED`)."""
    reasons = NOT_MODELLED.get(module, {})
    return [
        f"{module}.{name} is not part of the Codon model, on purpose: {reasons[name]}"
        for name in names
        if name in reasons
    ]


def _python_attribute(module: str, name: str) -> tuple[bool, Any]:
    """(found, value) of *name* in Python's *module*."""
    try:
        found = importlib.import_module(module)
        return hasattr(found, name), getattr(found, name, None)
    except Exception:  # a module that does not import: not one to suggest
        return False, None


def _unmodelled_module(module: str, names: list[str]) -> str:
    """Why ``from module import names`` fails the check, *module* being one of
    the package's modules the model does not have: where else the model
    declares each name as Python exports it, if anywhere.

    A module of the model is suggested only when its Python module has the
    same object under that name, so the suggested import runs too (#237).
    """
    message = f"{module} is not part of the Codon model"
    wanted = [n for n in names if n != "*"]
    if not wanted:
        return message
    modelled = _model_modules()
    parts = module.split(".")
    ancestors = [
        ".".join(parts[:end])
        for end in range(len(parts) - 1, 0, -1)
        if ".".join(parts[:end]) in modelled
    ]
    candidates = [*ancestors, *sorted(modelled - set(ancestors))]
    found: dict[str, str] = {}
    reasons: list[str] = []
    for name in wanted:
        has, value = _python_attribute(module, name)
        for candidate in candidates:
            if name not in _model_names(candidate) or name in NOT_MODELLED.get(
                candidate, {}
            ):
                continue
            there, other = _python_attribute(candidate, name)
            if there and (not has or other is value):
                found[name] = candidate
                break
        else:
            reasons += [
                f"{candidate}.{name} is left out on purpose: "
                f"{NOT_MODELLED[candidate][name]}"
                for candidate in candidates
                if name in NOT_MODELLED.get(candidate, {})
            ][:1]
    missing = [n for n in wanted if n not in found]
    if not missing:
        by_module: dict[str, list[str]] = {}
        for name in wanted:
            by_module.setdefault(found[name], []).append(name)
        where = "; ".join(
            f"import {', '.join(each)} from {candidate}"
            for candidate, each in by_module.items()
        )
        return f"{message}; {where} instead"
    it = "it" if len(missing) == 1 else "them"
    tail = (
        f", and no module of the model declares {', '.join(missing)} as Python "
        f"exports {it}"
    )
    tail += "".join(f"; {reason}" for reason in reasons)
    tail += "".join(
        f"; import {name} from {candidate}" for name, candidate in found.items()
    )
    return message + tail


@lru_cache(maxsize=1)
def _model_declarations() -> dict[str, set[str]]:
    """Class name -> the names it declares, inherited ones included, in the model."""
    classes: dict[str, tuple[list[str], set[str]]] = {}
    for path in sorted(model_dir().rglob("*.codon")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        classes.update(class_declarations(tree))

    def closure(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
        if name not in classes or name in seen:
            return set()
        bases, own = classes[name]
        out = set(own)
        for base in bases:
            out |= closure(base.rsplit(".", 1)[-1], seen | {name})
        return out

    return {name: closure(name) for name in classes}


def _init_classes(classes: dict[str, ast.ClassDef]) -> set[str]:
    """The classes that define an ``__init__``, of *classes* and of the model.

    What a class without one of its own inherits (#201): the transform gives
    such a class an ``__init__`` passing everything on to its base's.
    """
    model = {name for name, own in _model_declarations().items() if "__init__" in own}
    return classes_with_init(classes, model - set(classes))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostic:
    """One compile error (or warning), located in the file the author wrote."""

    message: str
    path: str | None = None
    line: int | None = None
    column: int | None = None
    #: Codon's "during the realization of ..." lines, outermost last.
    trace: tuple[str, ...] = ()
    #: A warning does not fail the check (:attr:`TypeCheckResult.warnings`).
    severity: Literal["error", "warning"] = "error"

    def format(self) -> str:
        where = ""
        if self.path is not None:
            where = self.path
            if self.line is not None:
                where += f":{self.line}"
                if self.column is not None:
                    where += f":{self.column}"
            where += ": "
        text = f"{where}{self.severity}: {self.message}"
        for frame in self.trace:
            text += f"\n    {frame}"
        return text


@dataclass
class TypeCheckResult:
    """The outcome of :func:`typecheck_scenario`."""

    scenario: str
    ok: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)
    #: Why the scenario was not compiled at all (no Codon, no source file).
    skipped: str | None = None
    #: The compiler crashed: no verdict on the scenario either way.
    crashed: bool = False
    output: str = ""
    seconds: float = 0.0
    #: What the check left unchecked and why; none fails it.  Last, so the
    #: fields before it keep their positions.
    warnings: list[Diagnostic] = field(default_factory=list)

    def format(self) -> str:
        if self.skipped is not None:
            return f"{self.scenario}: static check skipped: {self.skipped}"
        if self.ok:
            head = f"{self.scenario}: static check passed ({self.seconds:.1f}s)"
            shown = self.warnings
        else:
            head = f"{self.scenario}: static check failed"
            if self.crashed:
                head += " (the Codon compiler crashed)"
            shown = [*self.diagnostics, *self.warnings]
        if not shown:
            return head
        lines = [head + ":"]
        lines += ["  " + d.format().replace("\n", "\n  ") for d in shown]
        return "\n".join(lines)


class ScenarioTypeError(Exception):
    """A scenario failed its static check, so it is not run."""

    def __init__(self, result: TypeCheckResult) -> None:
        super().__init__(result.format())
        self.result = result


# ---------------------------------------------------------------------------
# Collecting the scenario's own modules
# ---------------------------------------------------------------------------


def _module_file(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _user_prefix(module: str) -> str:
    """Modules under this prefix are the scenario's own and are compiled as written."""
    parts = module.split(".")
    if parts[0] == _PACKAGE:
        # A scenario inside the framework (the built-in examples): its own
        # package, not the framework, which the model stands for.
        return ".".join(parts[:-1]) if len(parts) > 2 else module
    return parts[0]


def _imports(
    tree: ast.Module,
    module: str,
    is_package: bool,
    *,
    skip: Collection[int] = (),
) -> list[tuple[str, list[str], int]]:
    """(absolute module, imported names, line) for every import in *tree*.

    *skip* holds the ``id()`` of import nodes to leave out.
    """
    package = module if is_package else module.rpartition(".")[0]
    out: list[tuple[str, list[str], int]] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Import):
            out += [(alias.name, [], node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            out.append((name, [alias.name for alias in node.names], node.lineno))
    return out


@dataclass(frozen=True)
class _Module:
    path: Path
    is_package: bool  # a package __init__
    text: str
    tree: ast.Module


@dataclass
class _Sources:
    modules: dict[str, _Module] = field(default_factory=dict)
    problems: list[Diagnostic] = field(default_factory=list)


@lru_cache(maxsize=4)
def _linked(codon_path: Path) -> frozenset[str]:
    """Top-level modules of typesafe_carla's CODON_PATH directory: not
    modelled here but linked into the workspace (``_build_workspace``), so a
    scenario's ``import typesafe_carla.carla`` resolves to the library."""
    return frozenset(p.name.split(".")[0] for p in codon_path.iterdir())


def _check_time_only_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Every node of the code only Codon reads: the body of each ``if
    TYPE_CHECKING:`` and the ``else`` branch of each ``if not TYPE_CHECKING:``
    (the counterpart of :func:`_run_time_only_nodes`)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if _is_type_checking(node.test):
            check_time_only = node.body
        elif (
            isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and _is_type_checking(node.test.operand)
        ):
            check_time_only = node.orelse
        else:
            continue
        for stmt in check_time_only:
            yield from ast.walk(stmt)


def _missing_at_run_time(module: str, names: list[str]) -> list[str]:
    """Why ``from module import names`` (or ``import module``) fails in Python.

    *module* is one the model has: the check passes an import of it whatever
    Python's module holds, so a model-only module (``_tc``) or a name the
    model declares beyond Python's (#214) would pass and raise ``ImportError``
    when the scenario runs, where an import inside a function is first run.
    An import that fails for another reason (a missing dependency) is left
    to the run.
    """
    try:
        found = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if module == missing or module.startswith(missing + "."):
            return [
                f"{module} is a module of the Codon model only: the package "
                "has no such module, so importing it fails when the scenario runs"
            ]
        return []
    except Exception:  # not this check's to report
        return []

    def attribute(name: str) -> bool:
        try:
            return hasattr(found, name)
        except Exception:  # a lazy attribute that fails to import: the run's to report
            return True

    def submodule(name: str) -> bool:
        if not hasattr(found, "__path__"):
            return False
        try:
            return importlib.util.find_spec(f"{module}.{name}") is not None
        except (ImportError, ValueError):
            return False

    return [
        f"{module} has no {name} in Python, whatever its Codon model declares: "
        f"`from {module} import {name}` raises ImportError when the scenario runs"
        for name in names
        if name != "*" and not attribute(name) and not submodule(name)
    ]


def _collect(roots: list[str], tc: Toolchain, codon_path: Path) -> _Sources:
    stdlib = _codon_stdlib(tc)
    modeled = _model_modules()
    shims = _shims() | _linked(codon_path)
    prefixes = {_user_prefix(root) for root in roots}
    out = _Sources()
    queue = list(roots)
    while queue:
        name = queue.pop()
        if name in out.modules:
            continue
        path = _module_file(name)
        if path is None:
            continue
        is_package = path.name == "__init__.py"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        out.modules[name] = _Module(path, is_package, text, tree)
        # What only Codon reads (`if TYPE_CHECKING:`) Python never imports.
        check_time = {id(node) for node in _check_time_only_nodes(tree)}
        run_time = {
            (imported, lineno)
            for imported, _names, lineno in _imports(
                tree, name, is_package, skip=check_time
            )
        }
        for imported, names, lineno in _imports(tree, name, is_package):
            top = imported.split(".")[0]
            if imported in modeled:
                # A name the model leaves out on purpose: said with the
                # reason, rather than as Codon's "cannot import name" (#237).
                out.problems += [
                    Diagnostic(message, str(path), lineno)
                    for message in _not_modelled(imported, names)
                ]
                # The model may declare what Python's module lacks (a
                # model-only helper module, a name the model needs for
                # itself): an import Python runs must find it there too (#214).
                if (imported, lineno) in run_time:
                    out.problems += [
                        Diagnostic(message, str(path), lineno)
                        for message in _missing_at_run_time(imported, names)
                    ]
                continue
            if top in shims:
                continue
            if any(imported == p or imported.startswith(p + ".") for p in prefixes):
                submodules = [
                    f"{imported}.{n}" for n in names if _module_file(f"{imported}.{n}")
                ]
                queue += submodules
                plain = [n for n in names if f"{imported}.{n}" not in submodules]
                if plain or not names:
                    queue.append(imported)  # names from the module (or package) itself
                continue
            if top in stdlib and top != _PACKAGE:
                continue
            if top == _PACKAGE:
                what = _unmodelled_module(imported, names)
            else:
                what = f"{imported} has no Codon model, so a scenario importing it cannot be checked"
                if top in _UNMODELLED_HINTS:
                    what += f": {_UNMODELLED_HINTS[top]}"
            out.problems.append(Diagnostic(what, str(path), lineno))
    return out


# ---------------------------------------------------------------------------
# The workspace and the compile
# ---------------------------------------------------------------------------


@dataclass
class _Workspace:
    root: Path
    driver: Any
    #: workspace-relative file -> original path
    files: dict[str, str] = field(default_factory=dict)
    #: workspace-relative file -> (its lines before the calls the library
    #: check appended, appended line -> the call on it)
    appended: dict[str, tuple[int, dict[int, str]]] = field(default_factory=dict)
    #: workspace-relative file -> (appended line -> the line of the file
    #: holding what the call on it compiles), where an error on an appended
    #: line is reported at that line (the scenario check) rather than at the
    #: call (the library check, whose appended calls are not the author's)
    appended_origins: dict[str, dict[int, int]] = field(default_factory=dict)
    #: What an appended call is named as in a diagnostic.
    appended_label: str = "library check"
    #: Problems that do not fail the check (:attr:`TypeCheckResult.warnings`).
    warnings: list[Diagnostic] = field(default_factory=list)
    #: workspace-relative file -> the line of the original each of its lines
    #: after the prelude comes from (:func:`.transform.transform_source_lines`)
    origins: dict[str, list[int]] = field(default_factory=dict)
    #: What an error in the driver program is reported at.
    driver_label: str = _DRIVER_LABEL


def _base_workspace(root: Path, driver: Any, codon_path: Path) -> _Workspace:
    """The model, the shims and the CARLA API, in *root*."""
    shutil.copytree(model_dir(), root, dirs_exist_ok=True)
    extensions = root / _STDLIB_EXTENSIONS
    stdlib = find_codon().codon_dir / "lib" / "codon" / "stdlib"
    for extension in sorted(extensions.glob("*.codon")):
        (root / extension.name).write_text(
            (stdlib / extension.name).read_text() + "\n\n" + extension.read_text(),
        )
    shutil.rmtree(extensions)
    # The CARLA API: typesafe_carla's CODON_PATH directory (the library and
    # the compile-time switches it reads), linked in, as Codon reads only one.
    for entry in codon_path.iterdir():
        (root / entry.name).symlink_to(entry.resolve())
    return _Workspace(root, driver)


def _declarations(sources: _Sources) -> dict[str, set[str]]:
    """Class name -> the names it declares, in the model and in *sources*."""
    declared = {cls: set(names) for cls, names in _model_declarations().items()}
    for module in sources.modules.values():
        for cls, (_bases, own) in class_declarations(module.tree).items():
            declared.setdefault(cls, set()).update(own)
    return declared


#: The label of an appended call in a scenario's (or an ODD's) check.
_SCENARIO_CHECK_LABEL = "scenario check"


def _unchecked_warning(message: str, path: Path, line: int, reach: str) -> Diagnostic:
    """A function the appended calls cannot call: compiled only as far as reached."""
    return Diagnostic(
        f"{message}; the check does not call it on its own, so it is compiled "
        f"only where {reach} reach it",
        str(path),
        line,
        severity="warning",
    )


def _build_workspace(
    root: Path, sources: _Sources, driver: Any, codon_path: Path
) -> tuple[_Workspace, list[Diagnostic]]:
    """The scenario's (or the ODD's) modules, rewritten, beside the model.

    Each module has the library check's calls appended
    (:func:`.library_driver.render_library_checks`) and the driver runs
    them, so every function, method and class-level default of the module is
    compiled, not only what the driver's own calls reach.  A function the
    calls cannot be made for (a parameter with no annotation, or one Codon
    cannot express or naming a ``TypeVar``), or one returning ``Any``
    (``get_details() -> dict[str, Any]``), is left out of them with a
    warning: the driver still compiles it where it reaches it, as before.
    A function with a ``**kwargs`` or ``*args`` the modules call is compiled
    through those calls only (:func:`.library_driver.call_sites`, #266).
    """
    ws = _base_workspace(root, driver, codon_path)
    ws.appended_label = _SCENARIO_CHECK_LABEL
    problems: list[Diagnostic] = []
    declared = _declarations(sources)
    classes = {
        node.name: node
        for module in sources.modules.values()
        for node in module.tree.body
        if isinstance(node, ast.ClassDef)
    }
    init_classes = _init_classes(classes)
    callers = call_sites([module.tree for module in sources.modules.values()])
    for name, module in sorted(sources.modules.items()):
        rel = Path(*name.split("."))
        rel = rel / "__init__.py" if module.is_package else rel.with_suffix(".py")
        for attr in undeclared_attributes(module.tree, declared):
            problems.append(Diagnostic(attr.message(), str(module.path), attr.lineno))
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        # An own module imported as a module, in the form Codon resolves (#310).
        source, refused = relative_module_imports(
            module.text,
            name,
            module.is_package,
            sources.modules.__contains__,
            lambda m: literal_all(sources.modules[m].tree),
        )
        problems += [
            Diagnostic(message, str(module.path), line) for line, message in refused
        ]
        # What Python raises for when it runs, decided from the syntax
        # (#602, #603): Codon checks types, so it compiles these.
        problems += [
            Diagnostic(r.message, str(module.path), r.line, r.column)
            for r in [
                *subscripted_isinstances(module.tree),
                *literal_argument_refusals(module.tree),
            ]
        ]
        problems += [
            Diagnostic(message, str(module.path), line)
            for line, message in [
                *conflicting_imports(source),
                *python_rewrite_refusals(source),
                *typing_rewrite_refusals(source),
            ]
        ]
        text, origin = transform_source_lines(
            source,
            init_classes=init_classes,
            enum_classes=enum_class_names(classes),
            file=str(module.path),
        )
        checks = render_library_checks(
            module.tree,
            classes=classes,
            skip_any_returns=True,
            hook_returns=True,
            callers=callers,
        )
        for line, message in checks.problems:
            ws.warnings.append(
                _unchecked_warning(message, module.path, line, driver.reaches)
            )
        ws.warnings += [
            Diagnostic(
                d.message, str(module.path), d.line, d.column, severity="warning"
            )
            for d in dead_isinstances(module.tree)
        ]
        dest.write_text(text + checks.source)
        key = rel.as_posix()
        ws.files[key] = str(module.path)
        ws.origins[key] = origin
        ws.appended[key] = (text.count("\n"), checks.labels)
        ws.appended_origins[key] = checks.lines
    # Every package on the way needs an __init__; one not compiled is empty.
    for rel_name in list(ws.files):
        parent = Path(rel_name).parent
        while parent != Path("."):
            init_py, init_codon = (
                root / parent / "__init__.py",
                root / parent / "__init__.codon",
            )
            if not init_py.exists() and not init_codon.exists():
                init_codon.write_text("")
            parent = parent.parent
    (root / DRIVER_MODULE).write_text(
        driver.source + render_module_checks(sorted(sources.modules)) + driver.trailer
    )
    return ws, problems


#: Codon's refusals of valid Python on a ``**kwargs`` or ``*args`` (#266),
#: or on any tuple's truth, and what to write instead.
_VARIADIC_LIMITS = (
    (
        # An empty container with no annotation, an unannotated class
        # attribute `items = []` most often (#611): Codon types it from what
        # it holds, and it holds nothing, so its elements are `None`.
        re.compile(r"^'[^']+' does not match expected type 'NoneType'$"),
        "Codon typed the target from its first value with no element type: "
        "an empty `[]`, `{}` or `set()` with no annotation (a class attribute "
        "`items = []`) holds `None`, and so does a name first bound to `None`: "
        "annotate it (`items: list[str] = []`, `x: int | None = None`)",
    ),
    (
        # A tuple display given for a `tuple[X, ...]` the check could not
        # rewrite (#614): Codon's tuples have a fixed length.
        re.compile(r"^'Tuple\[[^']*\]' does not match expected type 'List\[[^']*\]'"),
        "a variable-length `tuple[X, ...]` is checked as a list (Codon's "
        "tuples have a fixed length): build it with `tuple([...])` "
        "(`tuple([1, 2, 1])`) instead of a tuple display",
    ),
    (
        re.compile(r"^'NamedTuple(?:\[[^']*\])?' object has no attribute '"),
        "Codon makes **kwargs a NamedTuple of the keywords passed, not a dict: "
        'read one with kw.get("name", default) or declare it as a keyword '
        "parameter, and do not iterate it or call .items(), .keys(), "
        ".values() or len() on it",
    ),
    (
        # Any tuple, not only *args (#278): the message names no variable.
        re.compile(r"^'Tuple\[[^']*\]' object has no attribute '__bool__'"),
        "Codon gives no tuple a truth value, *args included: test "
        "len(t) > 0 instead of the tuple t itself",
    ),
    (
        # Any other class without __bool__ (#603): a plain object, or one
        # with only __len__.  Not a bound method (`'flag(Cls)'`), whose
        # forgotten call has its own hint (call_diagnostics.forgotten_self).
        re.compile(
            r"^'[A-Za-z_][\w.]*(?:\[[^']*\])?' object has no attribute '__bool__'$"
        ),
        "Codon gives an object a truth value only through __bool__, and this "
        "class has none (Python takes a plain object as true, and one with "
        "__len__ as len() != 0): test what you mean, `x is not None` for a "
        "value that may be missing, `len(x) > 0` for a container, or define "
        "__bool__ on the class",
    ),
    (
        re.compile(r"^unsupported class decorator$"),
        "Codon refuses a decorator on a class, other than those the check "
        "drops (@dataclass, @enum.unique, @typing.runtime_checkable, "
        "@functools.total_ordering): a decorator that registers the class "
        "(`@register`) does not compile. Drop it, and choose the class with an "
        "if/elif chain (or a `match`) on the name, as Codon cannot hold a "
        "class as a value",
    ),
    (
        re.compile(r"^async not yet supported$"),
        "Codon compiles no `async def`, `await`, `async for` or `async with` "
        "anywhere in a checked module, called or not (the runner calls the "
        "scenario synchronously): make it a plain `def`, or move it to a "
        "module the scenario does not import",
    ),
    (
        # getattr(x, "name", default) with a literal name and a plain receiver
        # is rewritten (python_rewrites.py); this is any other (#590).
        re.compile(r"^getattr\(\) takes 2 arguments \(3 given\)"),
        "Codon's getattr takes no default: write "
        '`x.name if hasattr(x, "name") else default`, which Codon decides '
        'when it compiles (the check rewrites getattr(x, "name", default) '
        "itself where the name is a literal, x a name or an attribute chain "
        "and the default on one line); a name built at run time has no Codon "
        "spelling, so spell the "
        "attributes out",
    ),
    (
        re.compile(r"cannot import name 'Generic' from 'typing'"),
        "typing.Generic is not available in the check: leave the type "
        "parameters of a class or function unannotated (Codon makes it generic "
        "over them), or write the concrete type; a container with a typed "
        "attribute (`items: list[T]`) declares the concrete element type "
        "(`items: list[int]`)",
    ),
    (
        re.compile(r"cannot import name 'TypedDict' from 'typing'"),
        "typing.TypedDict is not available in the check: write a @dataclass "
        "(or a plain class with annotated fields), or a Dict[str, X]",
    ),
    (
        re.compile(r"cannot import name 'asdict' from 'dataclasses'"),
        "dataclasses.asdict returns a dict[str, Any], which the check cannot "
        "type: build the dict from the fields you need "
        '({"x": point.x, "y": point.y})',
    ),
    (
        re.compile(r"cannot import name '(?:InitVar|KW_ONLY)' from 'dataclasses'"),
        "dataclasses.InitVar and KW_ONLY are not modelled: Codon's constructor "
        "of a dataclass takes every annotated field and does not pass an "
        "init-only value to __post_init__. For an InitVar, make the value a "
        "parameter of a @classmethod or a function that builds the instance "
        "(or keep it as an ordinary field); for KW_ONLY, drop the marker and "
        "pass the fields by keyword (a field without a default goes before "
        "the defaulted ones)",
    ),
    (
        re.compile(r"cannot import name '\w+' from 'dataclasses'"),
        "the check models only dataclass, field and replace of dataclasses: "
        'docs/typecheck.md ("standard library names") lists the names',
    ),
    (
        re.compile(
            r"cannot import name '\w+' from "
            "'(functools|itertools|collections|math|random|statistics)'"
        ),
        "the check adds only some names to Codon's functools, itertools, "
        'collections, math, random and statistics: docs/typecheck.md ("standard '
        'library names") lists them',
    ),
    (
        re.compile(r"cannot import name '\w+' from 'contextlib'"),
        "the check models only suppress of contextlib: write a class with "
        "__enter__ and __exit__(self, exc_type, exc, tb) for contextmanager "
        "(the check calls __exit__ with no arguments, so those parameters are "
        "None in it)",
    ),
    (
        re.compile(r"cannot import name '\w+' from 'json'"),
        "the check models only dumps, dump, loads, load and JSONDecodeError of "
        'json: docs/typecheck.md ("standard library names") lists what they '
        "take and return",
    ),
    (
        re.compile(r"^'Path' object has no attribute '\w+'"),
        "the check models only some members of pathlib.Path: if CPython's Path "
        'has it, it is not modelled (docs/typecheck.md, "pathlib" among the '
        "shims lists the members); build the path with str methods and the "
        "declared members (name, stem, suffix, parent, with_name(), "
        "with_suffix(), joinpath(), /, ...)",
    ),
    (
        re.compile(r"cannot import name '\w+' from 'os(?:\.__init__)?'"),
        "the check models only some names of os: if CPython's os has it, it "
        'is not modelled (docs/typecheck.md, "os" among the shims lists the '
        "names); use the declared names or pathlib.Path",
    ),
    (
        re.compile(r"cannot import name '\w+' from 'os\.path'"),
        "the check models only some names of os.path: if CPython's os.path has "
        'it, it is not modelled (docs/typecheck.md, "os.path" among the shims '
        "lists the names); use the declared names or pathlib.Path",
    ),
    (
        re.compile(r"cannot import name '\w+' from 'shutil'"),
        "the check models only some names of shutil: if CPython's shutil has "
        'it, it is not modelled (docs/typecheck.md, "shutil" among the shims '
        "lists the names)",
    ),
    (
        re.compile(
            r"cannot import name '(?:version_info|version|hexversion)' from 'sys'"
        ),
        "sys.version_info (and sys.version, sys.hexversion) is not modelled: "
        "the check compiles every branch of a version test, so the code has "
        "to be valid on every supported Python (3.10 and later) anyway: "
        "write that code without the test, or keep the version-dependent part "
        "out of the checked modules",
    ),
    (
        re.compile(r"cannot import name '\w+' from 'sys'"),
        "the check models only argv, stdin, stdout, stderr, exit, maxsize, "
        'platform and executable of sys (docs/typecheck.md, "sys" among the '
        "shims): if CPython's sys has it, it is not modelled",
    ),
    (
        re.compile(r"cannot import name 'strftime' from 'time'"),
        "time.strftime is not modelled: take now = datetime.datetime.now() and "
        "format its fields, f'{now.year:04d}{now.month:02d}{now.day:02d}'",
    ),
    (
        re.compile(r"^'datetime' object has no attribute 'strftime'"),
        "datetime.strftime is not modelled (Codon's datetime has no "
        "strftime() or format): format the fields, "
        "f'{now.year:04d}{now.month:02d}{now.day:02d}'",
    ),
    (
        re.compile(r"cannot import name '\w+' from 'typing'"),
        "the check models only some names of typing: docs/typecheck.md "
        '("typing names") lists them',
    ),
)


_LIST_DISPLAY_FRAME = re.compile(r"during the realization of _acs_list(?:_into)?\(")
_LIST_DISPLAY_HINT = (
    "a list display is typed by its first element, so its elements must share "
    "a class of the scenario's own: annotate the target as the common base "
    "(`rules: list[Base] = [A(), B()]`, a `-> list[Base]` return, a `list[Base]` "
    "parameter), or declare a local `list[Base]` and `append()` each element; "
    "an element that is no subclass of the base is refused"
)


#: A list display assigned to an annotated target (``xs: list[B] = [...]``),
#: which the check already makes a list of the annotated element type (#611).
_ANNOTATED_DISPLAY = re.compile(r"^\s*[A-Za-z_][\w.]*\s*:[^=]+=\s*\[")
_ANNOTATED_DISPLAY_HINT = (
    "the display is a list of the target's annotated element type, so an "
    "element that is not of it (or of a subclass) is refused"
)


_DICT_DISPLAY_HINT = (
    "a dict display is typed by its first value, so its values must share a "
    "class of the scenario's own that the check can infer from literal keys "
    "and constructor calls: annotate the target as the common base "
    "(`d: dict[str, Base] = {...}`, a `dict[str, Base]` parameter), or declare "
    "a local `dict[str, Base]` and assign each item; a value that is no "
    "subclass of the base is refused"
)
_SUM_DISPLAY_HINT = (
    "a list display is typed by its first element, so it is no list of the "
    "base: put it in a local annotated `list[Base]` (`more: list[Base] = "
    "[Sub()]`) and add that, or `append()` each element; the check types the "
    "display itself when the other operand is a local or parameter annotated "
    "`list[Base]`"
)
_DICT_MISMATCH = re.compile(r"^'\w+' does not match expected type '\w+'")
_SCALAR_MISMATCH = re.compile(
    r"^'(?:int|float|str|bool)' does not match expected type "
    r"'(?:int|float|str|bool)'"
)
_DICT_SCALAR_HINT = (
    "a dict display is typed by its first value, and Codon has no dict of "
    "values of different types (no usable `Any` or `Union` value type, so "
    "annotating the target does not help): keep one value type (`str(...)` "
    "every value, or `dict[str, str]`), use a small dataclass with typed "
    "fields for the record, or use one dict per type"
)
_DICT_LINE = re.compile(r"\{|^\s*[\w\"'.]+\s*:\s*\w+\(")
_LIST_SUM = re.compile(
    r"^unsupported operand type\(s\) for \+=?: 'List\[(\w+)\]' and 'List\[(\w+)\]'"
)


_ANY_MISMATCH = re.compile(r"^'[^']+' does not match expected type '[^']+'$")
_CONDITIONAL_HINT = (
    "a conditional expression (`a if c else b`) has one type in Codon: if "
    "its two branches have different types (Python allows that), convert one "
    "(`str(n)`, `float(n)`), or write an if/else statement that uses each "
    "value in its own branch"
)


def _conditional_expression_at(diagnostic: Diagnostic) -> bool:
    """Whether *diagnostic* is at the start of a conditional expression of
    the author's file, where Codon reports its branches' mismatch (#603)."""
    if diagnostic.path is None or diagnostic.line is None or not diagnostic.column:
        return False
    try:
        tree = ast.parse(Path(diagnostic.path).read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(node, ast.IfExp)
        and node.lineno == diagnostic.line
        and node.col_offset == diagnostic.column - 1
        for node in ast.walk(tree)
    )


def _source_line(diagnostic: Diagnostic) -> str:
    """The line of the author's file *diagnostic* is at, or ``""``."""
    if diagnostic.path is None or diagnostic.line is None:
        return ""
    try:
        lines = Path(diagnostic.path).read_text().splitlines()
        return lines[diagnostic.line - 1]
    except (OSError, UnicodeDecodeError, IndexError):
        return ""


def _in_dict_display(diagnostic: Diagnostic) -> bool:
    """Whether the line of *diagnostic* is inside a dict display of the
    author's file: a ``{`` of an f-string or a set is no dict display."""
    if diagnostic.path is None or diagnostic.line is None:
        return False
    try:
        tree = ast.parse(Path(diagnostic.path).read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(node, ast.Dict)
        and node.lineno <= diagnostic.line <= (node.end_lineno or node.lineno)
        for node in ast.walk(tree)
    )


def _classes_below(diagnostic: Diagnostic) -> tuple[set[str], bool]:
    """The classes of the file defined below the line of *diagnostic*, and
    whether that line is in a function body (which may name them)."""
    if diagnostic.path is None or diagnostic.line is None:
        return set(), False
    try:
        tree = ast.parse(Path(diagnostic.path).read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set(), False
    line = diagnostic.line
    in_body = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line <= (node.end_lineno or node.lineno)
        for node in ast.walk(tree)
    )
    below = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.lineno > line
    }
    return below, in_body


def _define_above(name: str) -> str:
    return (
        f"Codon cannot name `{name}` here: it is defined further down the "
        f"module, and a class body or a module-level statement runs at once: "
        f"define `{name}` above its use (a function body may name it)"
    )


_NOT_DEFINED = re.compile(r"^name '(\w+)' is not defined$")


def _display_hint(diagnostic: Diagnostic, in_display: bool) -> str | None:
    """Why Codon refused a display of the scenario's own classes, when no
    other hint fits: a list or dict display typed by its first element, a
    list display added to a list, or a class defined below its use."""
    below, in_body = _classes_below(diagnostic)
    undefined = _NOT_DEFINED.match(diagnostic.message)
    if undefined is not None:
        return _define_above(undefined[1]) if undefined[1] in below else None
    if in_display:
        later = next(
            (
                n
                for n in re.findall(r"[A-Za-z_]\w*", _source_line(diagnostic))
                if n in below
            ),
            None,
        )
        if later is not None and not in_body:
            return _define_above(later)
        if _ANNOTATED_DISPLAY.match(_source_line(diagnostic)):
            return _ANNOTATED_DISPLAY_HINT
        return _LIST_DISPLAY_HINT
    if any("_acs_dict_into" in frame for frame in diagnostic.trace):
        return None  # already a dict of the target type: the value is refused
    if _ANY_MISMATCH.match(diagnostic.message) and _conditional_expression_at(
        diagnostic
    ):
        return _CONDITIONAL_HINT
    sum_ = _LIST_SUM.match(diagnostic.message)
    if sum_ is not None and sum_[1] != sum_[2] and "+" in _source_line(diagnostic):
        return _SUM_DISPLAY_HINT
    if _DICT_MISMATCH.match(diagnostic.message) and _DICT_LINE.search(
        _source_line(diagnostic)
    ):
        if _SCALAR_MISMATCH.match(diagnostic.message) and _in_dict_display(diagnostic):
            return _DICT_SCALAR_HINT
        return _DICT_DISPLAY_HINT
    return None


#: The frame Codon reports when a value of the wrong type goes to a
#: ``dataclasses.replace`` method that a subclass overrides (#494): a call to a
#: virtual method goes through a dispatch thunk, which names no field.
_REPLACE_THUNK_FRAME = re.compile(
    r"^(?:during the realization of )?class_thunk_dispatch\(slf: \w+, cls_id: int, \*args: Tuple\[(?P<got>[^\]]*)\], "
    r"F: _acs_with__(?P<field>\w+)\[(?P<cls>\w+),(?P<want>[^\]]*)\]\)"
)


#: The frame of the method ``replace(obj, a=x)`` is lowered to (#614): its
#: lines are appended after the module and map to the class's line, while the
#: frame names the line of the call.
_REPLACE_HELPER_FRAME = re.compile(
    r"^during the realization of _acs_with__\w+\(self: \w+\b.*"
    r" \[(?:[^\]]*, )?(?P<file>[^\]\s:]+):(?P<line>\d+)\]$"
)


def _replace_helper_location(diagnostic: Diagnostic) -> Diagnostic:
    """*diagnostic* raised inside a ``replace`` method, at the ``replace(...)``
    call of the same file that reached it, rather than at the class."""
    if not diagnostic.trace or diagnostic.path is None:
        return diagnostic
    found = _REPLACE_HELPER_FRAME.match(diagnostic.trace[0])
    if found is None or Path(diagnostic.path).name != found["file"]:
        return diagnostic
    return dataclasses.replace(
        diagnostic, line=int(found["line"]), column=None, trace=diagnostic.trace[1:]
    )


def _replace_thunk_message(diagnostic: Diagnostic) -> Diagnostic:
    """Say what ``replace(obj, a="x")`` got wrong when the field is inherited."""
    for frame in diagnostic.trace:
        found = _REPLACE_THUNK_FRAME.match(frame)
        if found is None or "does not match expected type" not in diagnostic.message:
            continue
        message = (
            f"dataclasses.replace: field '{found['field']}' of {found['cls']} "
            f"is declared {found['want']}, not {found['got']}"
        )
        trace = tuple(f for f in diagnostic.trace if f is not frame)
        return dataclasses.replace(diagnostic, message=message, trace=trace)
    return diagnostic


_SHIM_MEMBER = re.compile(
    r"^'Path' object has no attribute '(?P<path_name>\w+)'"
    r"|cannot import name '(?P<name>\w+)' from "
    r"'(?P<module>os|os\.__init__|os\.path|shutil)'"
)


@lru_cache(maxsize=8)
def _shim_members(shim: str) -> tuple[str, ...]:
    """The public names the shim *shim* (``pathlib``, ``os``, ``os/path`` or
    ``shutil``) declares: the methods of ``Path``, or the module's functions
    and constants."""
    file = model_dir() / f"{shim}.codon"
    if not file.is_file():
        file = model_dir() / shim / "__init__.codon"
    try:
        text = file.read_text(encoding="utf-8")
    except OSError:
        return ()
    if shim == "pathlib":
        text = text[text.index("\nclass Path:") :]
        names = re.findall(r"^    def (\w+)", text, re.MULTILINE)
    else:
        pairs = re.findall(r"^(?:def (\w+)|(\w+) =)", text, re.MULTILINE)
        names = [a or b for a, b in pairs]
    return tuple(sorted({n for n in names if not n.startswith("_")}))


def _misspelt_member(message: str) -> str | None:
    """The hint for a refused ``Path`` member or ``os``, ``os.path`` or
    ``shutil`` name that CPython lacks too (a typo), or None when CPython has
    it (then it is only not modelled)."""
    found = _SHIM_MEMBER.search(message)
    if found is None:
        return None
    owner: object
    if found["path_name"] is not None:
        name, shim, owner, label = found["path_name"], "pathlib", Path, "Path"
    else:
        name, module = found["name"], found["module"]
        if module == "shutil":
            shim, owner = "shutil", shutil
        elif module == "os.path":
            shim, owner = "os/path", os.path
        else:
            shim, owner = "os", os
        label = module.removesuffix(".__init__")
    if hasattr(owner, name):
        return None
    close = difflib.get_close_matches(name, _shim_members(shim), n=1)
    suggestion = f", did you mean '{close[0]}'?" if close else ""
    return f"CPython's {label} has no '{name}' either (a typo?){suggestion}"


def _variadic_hints(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    """*diagnostics*, each refusal of a Codon limit on ``**kwargs``, on
    ``*args`` or on a tuple saying why and what to write instead (#266)."""
    out = []
    files = _Files()
    for diagnostic in diagnostics:
        diagnostic = _replace_helper_location(_replace_thunk_message(diagnostic))
        in_display = any(_LIST_DISPLAY_FRAME.match(frame) for frame in diagnostic.trace)
        if in_display:
            # A list display `_acs_list` could not make one list (#480): the
            # helper is the check's, so the refusal does not name it.
            diagnostic = dataclasses.replace(
                diagnostic,
                trace=tuple(
                    f for f in diagnostic.trace if not _LIST_DISPLAY_FRAME.match(f)
                ),
            )
        typo = _misspelt_member(diagnostic.message)
        if typo is not None:
            diagnostic = dataclasses.replace(
                diagnostic, message=f"{diagnostic.message}: {typo}"
            )
            out.append(diagnostic)
            continue
        try:
            slip = slip_hint(
                diagnostic.message,
                diagnostic.trace,
                diagnostic.path,
                diagnostic.line,
                diagnostic.column,
                files,
            )
        except Exception:  # noqa: BLE001 - only a hint: never lose Codon's error
            slip = None
        if slip is not None:
            # The refiner adds it, to its own wording (_authors_terms): no
            # other hint here, which would not apply.
            out.append(diagnostic)
            continue
        for pattern, hint in (*_VARIADIC_LIMITS, *REWRITE_HINTS, *TYPING_HINTS):
            if pattern.search(diagnostic.message):
                diagnostic = dataclasses.replace(
                    diagnostic, message=f"{diagnostic.message}: {hint}"
                )
                break
        else:
            # No other hint fits (a table of functions has its own, #462).
            shown = _display_hint(diagnostic, in_display)
            if shown is not None:
                diagnostic = dataclasses.replace(
                    diagnostic, message=f"{diagnostic.message}: {shown}"
                )
        out.append(diagnostic)
    return out


_UNINFERRED = re.compile(
    r"type of attribute '(?P<attr>\w+)' of object '(?P<cls>\w+)' cannot be inferred"
)


def _uninferred_attributes(
    diagnostics: list[Diagnostic], sources: _Sources
) -> list[Diagnostic]:
    """What to declare when Codon could not type an attribute (#218).

    A class that declares no attribute at class level has its attributes
    typed from what its ``__init__`` assigns, which Codon cannot always do:
    not an empty list, and not where the scenario check calls a method on a
    value it did not build with ``__init__`` (``self: Counter[NoneType]``).
    Its error is then Codon's own, at a line of the class, or naming the
    attribute of the class holding one (``type of attribute '_tracker' of
    object 'GapCondition' cannot be inferred``), not the class that could not
    be typed.  For each such class, name each attribute it assigns without
    declaring it, at its line, with the type to declare.
    """
    classes = {
        node.name: (name, node)
        for name, module in sources.modules.items()
        for node in module.tree.body
        if isinstance(node, ast.ClassDef)
    }

    def undeclaring(cls: str) -> bool:
        node = classes[cls][1]
        return not any(isinstance(stmt, ast.AnnAssign) for stmt in node.body)

    suspects: list[str] = []
    for diagnostic in diagnostics:
        for cls, (name, node) in classes.items():
            if (
                diagnostic.path == str(sources.modules[name].path)
                and diagnostic.line is not None
                and node.lineno <= diagnostic.line <= (node.end_lineno or node.lineno)
                and undeclaring(cls)
                and cls not in suspects
            ):
                suspects.append(cls)
        found = _UNINFERRED.search(diagnostic.message)
        if found is None or found["cls"] not in classes:
            continue
        holder = classes[found["cls"]][1]
        named = {found["cls"]}
        for stmt in holder.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == found["attr"]
            ):
                named |= {
                    n.id for n in ast.walk(stmt.annotation) if isinstance(n, ast.Name)
                }
                named |= {
                    n.value.split(".")[-1]
                    for n in ast.walk(stmt.annotation)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                }
        suspects += sorted(
            n for n in named if n in classes and undeclaring(n) and n not in suspects
        )
    declared = _declarations(sources)
    out: list[Diagnostic] = []
    for cls in suspects:
        name, _node = classes[cls]
        module = sources.modules[name]
        out += [
            Diagnostic(attr.message(), str(module.path), attr.lineno)
            for attr in undeclared_attributes(module.tree, declared, also={cls})
            if attr.class_name == cls
        ]
    return out


_NO_ATTRIBUTE = re.compile(r"'(?P<cls>\w+)' object has no attribute '(?P<attr>\w+)'")


def _lookup_order(sources: _Sources, path: str | None) -> list[str]:
    """The modules to look a name up in: the one in the file *path* first
    (its name is what that file's calls mean, whatever another module
    defines), then the other checked modules, then the modelled ones."""
    own = [n for n, m in sources.modules.items() if path and str(m.path) == path]
    rest = [n for n in sources.modules if n not in own]
    return [*own, *rest, *sorted(_model_modules())]


def _python_class(name: str, sources: _Sources, path: str | None = None) -> type | None:
    """The Python class Codon calls *name*: one of the checked modules' (a
    scenario's own subclass), else one a modelled module defines."""
    for module in _lookup_order(sources, path):
        try:
            found = getattr(importlib.import_module(module), name, None)
        except Exception:  # noqa: BLE001 - only a hint: a module that does not import
            continue
        if isinstance(found, type):
            return found
    return None


_EXPECTED_TYPE = re.compile(
    r"^'(?P<cls>\w+)' does not match expected type '(?P<want>str|int|float)'"
)
_PROTOCOL_EXPECTED = re.compile(
    r"^'[^']+' does not match expected type "
    r"'(?:(?:List|Set|Optional)\[)*(?P<cls>[A-Za-z_]\w*)"
)
_OPERAND = re.compile(
    r"^unsupported operand type\(s\) for (?P<op>[^:]+): '(?P<cls>\w+)' and"
)
_GETITEM = re.compile(r"__getitem__\(self: (?P<cls>\w+),")
_NO_HASH = re.compile(
    r"^'(?P<cls>\w+)(?:\[[^']*\])?' object has no attribute '__hash__'"
)
_NAMED_TUPLE_ONLY = ("_replace", "_asdict", "_fields", "_field_defaults", "_make")


def _is_named_tuple_class(cls: type | None) -> bool:
    return (
        cls is not None and issubclass(cls, tuple) and hasattr(cls, "_fields")
    )  # typing.NamedTuple / collections.namedtuple


def _data_model_hints(
    diagnostics: list[Diagnostic], sources: _Sources
) -> list[Diagnostic]:
    """Why Codon refuses a scenario's own dataclass, enum or NamedTuple, and
    what to write instead (#365).

    The check generates a dataclass's ``==``, hash and ordering and gives an
    ``IntEnum`` / ``(str, Enum)`` member its comparisons and ``int()``, and
    makes a ``NamedTuple`` a Codon ``@tuple``; what remains is Codon's own
    refusal, which names none of this.
    """
    out = []
    for diagnostic in diagnostics:
        hint = None
        message = diagnostic.message
        try:
            hint = _data_model_hint(message, diagnostic.trace, sources)
            if hint is None:  # a cycle of the package's modules (#507)
                hint = cycle_hint(
                    message, diagnostic.path, diagnostic.line, sources.modules
                )
        except Exception:  # noqa: BLE001 - only a hint: never lose Codon's error
            pass
        if hint is None and message.startswith("expected iterable expression"):
            hint = _self_iterator_hint(diagnostic, sources)
        if hint is None:
            try:
                hint = _local_binding_hint(diagnostic, sources)
            except Exception:  # noqa: BLE001 - only a hint: never lose Codon's error
                pass
        if hint is not None:
            diagnostic = dataclasses.replace(diagnostic, message=f"{message}: {hint}")
        out.append(diagnostic)
    return out


def _author_tree(path: str | None, sources: _Sources) -> ast.Module | None:
    """The AST of the checked module at *path* (the author's file)."""
    if path is None:
        return None
    for module in sources.modules.values():
        if str(module.path) == path:
            return module.tree
    return None


def _local_binding_hint(diagnostic: Diagnostic, sources: _Sources) -> str | None:
    """A local Codon typed by its first assignment, an int literal or
    ``None``, where the refused value is stored in it (#583, #584)."""
    tree = _author_tree(diagnostic.path, sources)
    if tree is None or diagnostic.line is None:
        return None
    hint = int_literal_hint(diagnostic.message, diagnostic.line, tree)
    if hint is None:
        found = none_mismatch_hint(diagnostic.message, diagnostic.line, tree)
        hint = found[0] if found is not None else None
    return hint


def _none_first_binding_hints(sources: _Sources, why: str) -> list[Diagnostic]:
    """Each unannotated local of the checked modules first bound to ``None``
    and rebound later, at its ``= None`` line, as a likely cause of *why*
    (Codon's crash, or its ``Optional[NoneType]`` refusal elsewhere) (#584)."""
    found: list[Diagnostic] = []
    for module in sources.modules.values():
        for binding in none_first_bindings(module.tree):
            found.append(
                Diagnostic(
                    f"{why}; a likely cause: " f"{none_first_binding_message(binding)}",
                    str(module.path),
                    binding.line,
                )
            )
    return found


_FOR_LINE = re.compile(r"\bfor\b.*?\bin\b(?P<iterable>.*)$")


def _self_iterator_hint(diagnostic: Diagnostic, sources: _Sources) -> str | None:
    """Why ``for x in C(...)`` is refused when *C* is an iterator class with
    ``__next__`` (#504): Codon has no ``__next__`` protocol."""
    found = _FOR_LINE.search(_source_line(diagnostic))
    if found is None:
        return None
    for name in re.findall(r"[A-Za-z_]\w*", found["iterable"]):
        cls = _python_class(name, sources)
        if (
            cls is not None
            and hasattr(cls, "__next__")
            and not issubclass(cls, tuple)
            and "__iter__" in dir(cls)
            and not inspect.isgeneratorfunction(getattr(cls, "__iter__", None))
        ):
            return (
                f"Codon iterates a class only through an `__iter__` that is a "
                f"generator (it has no `__next__` protocol): write "
                f"`{name}.__iter__` with `yield` instead of returning self "
                f"(valid Python too)"
            )
    return None


def _annotated_class_attribute(
    cls_name: str, attr: str, sources: _Sources
) -> str | None:
    """Why ``C.X`` is no attribute of *C* when the class annotates ``X`` (#471):
    Codon makes an annotated name of a class body a field of the instance."""
    cls = _python_class(cls_name, sources)
    if cls is None or attr not in cls.__dict__.get("__annotations__", {}):
        return None
    if attr not in cls.__dict__:
        return None  # annotated with no value: a field in Python too
    if dataclasses.is_dataclass(cls):
        return (
            f"`{attr}` is a field of the @dataclass {cls_name} (its default "
            f"is read as `{cls_name}.{attr}` in Python only): read it from an "
            f"instance, or use a module-level constant"
        )
    return (
        f"Codon makes the annotated `{attr}` of class {cls_name} a field of "
        f"the instance, not a class attribute (`ClassVar[...]` and "
        f"`Final[int]` included): write `{attr}: Final = ...` (or no "
        f"annotation) to read it as `{cls_name}.{attr}`"
    )


def _data_model_hint(
    message: str, trace: tuple[str, ...], sources: _Sources
) -> str | None:
    expected = _EXPECTED_TYPE.match(message)
    if expected is not None:
        cls = _python_class(expected["cls"], sources)
        if (
            cls is not None
            and issubclass(cls, enum.Enum)
            and issubclass(cls, (int, str))
        ):
            return (
                f"an {cls.__name__} member is not a {expected['want']} in the "
                f"check (only its comparisons and int() are modelled): pass "
                f"its .value"
            )
    wanted = _PROTOCOL_EXPECTED.match(message)
    if wanted is not None:
        cls = _python_class(wanted["cls"], sources)
        if cls is not None and getattr(cls, "_is_protocol", False):
            return (
                f"{cls.__name__} is a typing.Protocol, which Codon matches by "
                f"class name, not by shape: derive from it (class Sq("
                f"{cls.__name__})), give the classes a common base class, or "
                f"annotate a Union of the classes (a list is invariant: "
                f"annotate it with the class it holds)"
            )
    operand = _OPERAND.match(message)
    if operand is not None:
        cls = _python_class(operand["cls"], sources)
        if cls is not None and issubclass(cls, enum.Flag):
            # Python's IntFlag adds ints (to an int, not to a flag), a Flag
            # adds nothing; both combine with | & ^ ~ and `in`.
            return (
                f"a {cls.__name__} member combines with `|`, `&`, `^`, `~` "
                f"and `in` in the check, and does no other arithmetic"
                + (
                    " (only comparisons and int()): use its .value"
                    if issubclass(cls, int)
                    else ""
                )
            )
        if cls is not None and issubclass(cls, enum.Enum) and issubclass(cls, int):
            return (
                f"an {cls.__name__} member does no arithmetic in the check "
                f"(only comparisons and int()): use its .value"
            )
        if cls is not None and dataclasses.is_dataclass(cls):
            if operand["op"] in ("==", "!="):
                return (
                    "the check gives a @dataclass its == and != only with "
                    "bases it can see the fields of (dataclasses of this "
                    "module): compare the fields"
                )
            return (
                "the check gives a @dataclass ordering only with "
                "order=True (Python refuses it otherwise) and bases it can "
                "see the fields of (dataclasses of this module); compare "
                "the fields instead"
            )
    no_hash = _NO_HASH.match(message)
    if no_hash is not None:
        cls = _python_class(no_hash["cls"], sources)
        if cls is not None and dataclasses.is_dataclass(cls):
            return (
                "the check generates hash() for a @dataclass(frozen=True) "
                "(or unsafe_hash=True) whose bases are dataclasses of this "
                "module; key on a field or a tuple of fields instead"
            )
        if cls is not None and cls.__hash__ is None:
            # Python sets __hash__ = None on a class that defines __eq__ and
            # not __hash__ (#596).
            return (
                f"{cls.__name__} defines __eq__ and not __hash__, so Python "
                f"makes it unhashable (it sets __hash__ = None): a set of "
                f"them, or a dict keyed by one, raises TypeError; define "
                f"__hash__ consistent with __eq__ (`return hash((self.x, "
                f"self.y))`), or key on a field"
            )
    if message.startswith("expected iterable expression"):
        for frame in trace:
            found = _GETITEM.search(frame)
            if found and _is_named_tuple_class(_python_class(found["cls"], sources)):
                return (
                    "Codon indexes and unpacks a NamedTuple only when all its "
                    "fields have one type: read the fields by name "
                    "(p.x, p.y)"
                )
    missing = _NO_ATTRIBUTE.search(message)
    if missing is not None:
        annotated = _annotated_class_attribute(missing["cls"], missing["attr"], sources)
        if annotated is not None:
            return annotated
    if missing is not None and missing["attr"] in _NAMED_TUPLE_ONLY:
        if _is_named_tuple_class(_python_class(missing["cls"], sources)):
            return (
                f"a NamedTuple is a Codon @tuple in the check, which has no "
                f"{missing['attr']}(): build a new {missing['cls']}(...) from "
                f"the fields instead"
            )
    return None


def _with_member_reasons(
    diagnostics: list[Diagnostic], sources: _Sources
) -> list[Diagnostic]:
    """*diagnostics*, each of Codon's ``'X' object has no attribute 'm'``
    naming a member the model leaves out on purpose given its reason
    (:data:`.not_modelled.NOT_MODELLED_MEMBERS`, #265), as an import of a
    name the model leaves out is (#237)."""
    out: list[Diagnostic] = []
    for diagnostic in diagnostics:
        found = _NO_ATTRIBUTE.search(diagnostic.message)
        cls = _python_class(found["cls"], sources) if found is not None else None
        listed = None
        if found is not None and cls is not None:
            member = found["attr"]
            listed = not_modelled_member(cls, member, found["cls"])
        if listed is None:
            out.append(diagnostic)
            continue
        owner, reason = listed
        out.append(
            dataclasses.replace(
                diagnostic,
                message=(
                    f"{diagnostic.message}: {owner}.{member} is not part of "
                    f"the Codon model, on purpose: {reason}"
                ),
            )
        )
    return out


def _original_line(origin: list[int] | None, line: int) -> int:
    """The line of the original file that *line* of its rewrite comes from."""
    line -= _PRELUDE_LINES
    if not origin or line < 1:
        return line
    if line <= len(origin):
        return origin[line - 1]
    # After the module (an Enum's constructor): as many lines past its end.
    return line - len(origin) + max(origin)


_UNEXPECTED_TOKEN = re.compile(r"unexpected '(?P<token>[^']+)'")


def _fstring_line(
    text: str, origin: list[int] | None, column: int | None, message: str
) -> int | None:
    """The line of the original of the f-string Codon's parser failed on.

    Codon parses an f-string field on its own and reports a syntax error in
    it at the line *within the field* (1), so the line comes out before the
    file's first one; the column is within the field's text
    (``{x:+.2f}``).  The f-strings of the file are searched for the field
    whose character at that column is the unexpected token, else the first
    field with a format spec; ``None`` if the file holds none.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    token = _UNEXPECTED_TOKEN.search(message)
    fields: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.FormattedValue) and value.format_spec:
                    inner = value.format_spec
                    spec = "".join(
                        str(v.value) if isinstance(v, ast.Constant) else "{}"
                        for v in (
                            inner.values if isinstance(inner, ast.JoinedStr) else []
                        )
                    )
                    shown = f"{{{ast.unparse(value.value)}:{spec}}}"
                    fields.append((node.lineno, shown))
    fields.sort()
    if not fields:
        return None
    if token is not None and column is not None:
        for line, shown in fields:
            if column < len(shown) and shown[column] == token["token"]:
                return _original_line(origin, line)
    return _original_line(origin, fields[0][0])


def _rewritten_lines(
    checked: Path, line: int, original: str, original_line: int
) -> tuple[str, str] | None:
    """The (compiled, author's) text of line *line* of the file the check
    compiled and line *original_line* of the author's *original*, when they
    differ."""
    try:
        compiled = checked.read_text().splitlines()[line - 1]
        written = Path(original).read_text().splitlines()[original_line - 1]
    except (OSError, IndexError, ValueError):
        return None
    return None if compiled == written else (compiled, written)


def _column_survives(column: int, compiled: str, written: str) -> bool:
    """Whether *column* of the compiled line is also that of the author's:
    what precedes it is the same text in both."""
    before = max(column - 1, 0)
    return before <= len(written) and compiled[:before] == written[:before]


#: Codon's internal name of a lambda's type (``%_lambda_981(...)``,
#: ``%_lambda_88[T1]``), which names nothing in the scenario's source.
_LAMBDA_TYPE = re.compile(r"%_lambda_\d+(?:\(\.\.\.\)|\[[^\]]*\])?")


def _python_callable(name: str, sources: _Sources, path: str | None = None) -> Any:
    """The Python function Codon calls *name*: one of the checked modules',
    else one a modelled module defines."""
    for module in _lookup_order(sources, path):
        try:
            found = getattr(importlib.import_module(module), name, None)
        except Exception:  # noqa: BLE001 - only a hint: a module that does not import
            continue
        if inspect.isfunction(found):
            return found
    return None


def _authors_terms(diagnostic: Diagnostic, refiner: Refiner) -> Diagnostic:
    """*diagnostic* in the author's terms, not Codon's (#527, #528, #530): a
    refused call or ``return`` says what is wrong with it, a static string
    type is ``str``, and the frames of the model's own helpers are gone."""
    message, line, column = refiner.refine(
        diagnostic.message,
        diagnostic.path,
        diagnostic.line,
        diagnostic.column,
        diagnostic.trace,
    )
    return dataclasses.replace(
        diagnostic,
        message=plain_literal_types(message),
        line=line,
        column=column,
        trace=tuple(
            plain_literal_types(t) for t in diagnostic.trace if not is_internal_frame(t)
        ),
    )


def _without_lambda_names(diagnostic: Diagnostic) -> Diagnostic:
    """*diagnostic* with the lambda type names of its message and trace plain.

    Done once the hints, which match Codon's own wording, have been applied.
    """
    return dataclasses.replace(
        diagnostic,
        message=_plain_lambda_names(diagnostic.message),
        trace=tuple(_plain_lambda_names(t) for t in diagnostic.trace),
    )


def _plain_lambda_names(message: str) -> str:
    """*message* with Codon's lambda type names read as "a lambda" (#495)."""
    return _LAMBDA_TYPE.sub("a lambda", message)


def _parse_output(output: str, ws: _Workspace) -> list[Diagnostic]:
    """Codon's errors, each located at its innermost frame in the scenario's files."""
    by_basename: dict[str, list[str]] = {}
    for rel in ws.files:
        by_basename.setdefault(Path(rel).name, []).append(rel)

    #: The (compiled, author's) text of each (file, line) of Codon's that the
    #: check rewrote: a column after the rewrite is not the author's.
    moved: dict[tuple[str, int], tuple[str, str]] = {}

    def locate(
        file: str, line: int, column: int | None = None, message: str = ""
    ) -> tuple[str, int, bool, str | None] | None:
        """(path to show, line, is the scenario's own file, the appended call)."""
        if file == DRIVER_MODULE:
            key = ws.driver.config_lines.get(line)
            return (key if key else ws.driver_label, 0 if key else line, False, None)
        candidates = by_basename.get(Path(file).name, [])
        if len(candidates) != 1:
            return None
        rel = candidates[0]
        if rel in ws.appended and line > ws.appended[rel][0]:
            # A call the check appended: name the call it makes, and, in a
            # scenario, report it at the line of what it compiles (a
            # class-level default, a parameter's default).
            appended = line - ws.appended[rel][0]
            call = ws.appended[rel][1].get(appended)
            note = f"<{ws.appended_label}: {call or rel}>"
            origin = ws.appended_origins.get(rel, {}).get(appended)
            if origin is not None:
                return (ws.files[rel], origin, True, note)
            return (note, 0, False, None)
        original = _original_line(ws.origins.get(rel), line)
        if (
            original < 1
            and line <= _PRELUDE_LINES
            and message.startswith("syntax error")
        ):
            # Codon's position inside an f-string field: not a line of the file.
            found = _fstring_line(
                (ws.root / rel).read_text(), ws.origins.get(rel), column, message
            )
            if found is not None:
                original = found
        if original >= 1:
            texts = _rewritten_lines(ws.root / rel, line, ws.files[rel], original)
            if texts is not None:
                moved[(file, line)] = texts
        return (ws.files[rel], original, True, None)

    errors: list[tuple[str, list[tuple[str, int, int | None, str]]]] = []
    for raw in output.splitlines():
        text = _ANSI.sub("", raw)
        nested = text.lstrip().startswith(("├─", "╰─", "│"))
        body = text.lstrip().lstrip("├╰─│ ").strip() if nested else text.strip()
        match = _LOCATION.match(body)
        rest = body[match.end() :] if match else body
        found = _ERROR.search(rest)
        if found is None:
            continue
        text_ = found["message"].strip()
        frame = (
            (
                match["file"],
                int(match["line"]),
                int(match["col"]) if match["col"] else None,
                text_,
            )
            if match
            else ("", 0, None, text_)
        )
        if nested and errors:
            errors[-1][1].append(frame)
        else:
            errors.append((frame[3], [frame]))

    diagnostics: list[Diagnostic] = []
    for message, frames in errors:
        located = [
            (f, locate(f[0], f[1], f[2], f[3])) if f[0] else (f, None) for f in frames
        ]
        pf, primary = next(
            ((f, loc) for f, loc in located if loc and loc[2]), None
        ) or next(((f, loc) for f, loc in located if loc), (None, None))
        column = pf[2] if pf is not None else None
        if column is not None and pf is not None and (pf[0], pf[1]) in moved:
            if not _column_survives(column, *moved[(pf[0], pf[1])]):
                column = None  # a column of the rewritten line (#494)
        trace = []
        if primary is not None and primary[3] is not None:
            # The error is on an appended line itself: say which call it is.
            trace.append(f"reached by {primary[3]}")
            column = None  # a column of the appended line, not of the file's
        for f, loc in located[1:]:
            if f[0] == DRIVER_MODULE:
                continue  # the generated program: nothing the author wrote
            if loc is not None:
                where = f"{Path(loc[0]).name}:{loc[1]}" if loc[1] else loc[0]
                if loc[3] is not None:
                    where = f"{loc[3]}, {where}"
            else:
                where = f"{f[0]}:{f[1]}" if f[0] else ""
            trace.append(f"{f[3]} [{where}]" if where else f[3])
        if primary is None:
            diagnostics.append(Diagnostic(message, trace=tuple(trace)))
        else:
            diagnostics.append(
                Diagnostic(
                    message,
                    primary[0],
                    primary[1] or None,
                    column if primary[1] else None,
                    tuple(trace),
                )
            )
    return diagnostics


_CACHE: dict[str, TypeCheckResult] = {}
#: The case (:attr:`Driver.case`) each verdict of :data:`_CACHE` was made
#: for, so that another case of the same shape names it when it reuses it.
_CHECKED_FOR: dict[str, dict[str, str]] = {}
#: The verdict of each key being made right now, for the checks of the same
#: key that start meanwhile to wait for (``scenario-check`` runs configs in
#: threads side by side); ``None`` when the compile timed out.
_IN_FLIGHT: dict[str, Future[TypeCheckResult | None]] = {}
_IN_FLIGHT_LOCK = threading.Lock()

#: A directory to keep verdicts in beyond the process, one file per cache key:
#: the test suite's xdist workers share it, and CI restores it from one run to
#: the next. Unset, the user's cache directory; set but empty, a verdict lives
#: as long as the process.
CACHE_DIR_ENV = "AUTOWARE_CARLA_SCENARIO_TYPECHECK_CACHE"


def _relocated(result: TypeCheckResult, paths: Mapping[str, str]) -> TypeCheckResult:
    """*result* with each string in *paths* in its text replaced by its value."""
    order = sorted(paths, key=len, reverse=True)

    def text(value: str) -> str:
        for old in order:
            value = value.replace(old, paths[old])
        return value

    def diagnostic(d: Diagnostic) -> Diagnostic:
        return dataclasses.replace(
            d,
            message=text(d.message),
            path=None if d.path is None else text(d.path),
            trace=tuple(text(t) for t in d.trace),
        )

    return dataclasses.replace(
        result,
        diagnostics=[diagnostic(d) for d in result.diagnostics],
        output=text(result.output),
        warnings=[diagnostic(d) for d in result.warnings],
    )


def _module_markers(sources: _Sources) -> dict[str, str]:
    """Each module's file, and what stands for it in a verdict kept on disk.

    The key has each module's name and text, not its file: the same scenario
    in another directory (another run's ``tmp_path``) is the same program.
    """
    return {str(m.path): f"\0module:{name}\0" for name, m in sources.modules.items()}


def _cache_dir() -> Path | None:
    """``$CACHE_DIR_ENV``, or the user's cache directory when it is unset.

    ``None`` (verdicts kept for the process only) when it is set but empty,
    or when there is no home directory to put the default in. A relative
    ``XDG_CACHE_HOME`` is ignored, as the XDG spec says, and so is a home
    that is not absolute (``HOME=""``): either would put the cache in
    whatever directory the check runs from.
    """
    configured = os.environ.get(CACHE_DIR_ENV)
    if configured is not None:
        return Path(configured) if configured else None
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    if os.path.isabs(xdg):
        base = Path(xdg)
    else:
        try:
            home = Path.home()
        except RuntimeError:  # no home directory
            return None
        if not home.is_absolute():
            return None
        base = home / ".cache"
    return base / "autoware_carla_scenario" / "typecheck"


def _cache_file(key: str) -> Path | None:
    root = _cache_dir()
    return root / f"{key}.json" if root is not None else None


def _cached(key: str, sources: _Sources) -> TypeCheckResult | None:
    """The verdict kept under *key*, in this process or in ``$CACHE_DIR_ENV``.

    One read from the directory names the files of *sources*.

    An entry that cannot be read is a miss. One that is read has its times
    touched, so that a cache pruned by access time keeps it.
    """
    if key in _CACHE:
        return _CACHE[key]
    path = _cache_file(key)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))

        def diagnostic(d: dict[str, Any]) -> Diagnostic:
            return Diagnostic(**{**d, "trace": tuple(d["trace"])})

        result = TypeCheckResult(
            **{
                **data,
                "diagnostics": [diagnostic(d) for d in data["diagnostics"]],
                "warnings": [diagnostic(d) for d in data["warnings"]],
            }
        )
        os.utime(path)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    markers = _module_markers(sources)
    result = _relocated(result, {marker: path for path, marker in markers.items()})
    _CACHE[key] = result
    return result


def _store(key: str, result: TypeCheckResult, sources: _Sources) -> TypeCheckResult:
    """Keep *result* under *key*; in ``$CACHE_DIR_ENV`` too unless Codon crashed.

    A crash says nothing about the program (Codon may have run out of memory),
    so it is not worth keeping for another run. The file is written whole or
    not at all: workers that compile the same program race harmlessly.
    """
    _CACHE[key] = result
    path = _cache_file(key)
    if path is None or result.crashed:
        return result
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            kept = _relocated(result, _module_markers(sources))
            json.dump(dataclasses.asdict(kept), out)
        os.replace(tmp, path)
    except OSError:
        pass
    return result


def _tree_digest(digest: Any, paths: list[Path]) -> None:
    for path in sorted(paths):
        digest.update(path.read_bytes())


@lru_cache(maxsize=4)
def _framework_digest(codon_path: Path) -> str:
    """The Codon inputs that do not change while the process runs: the CARLA
    library and the model."""
    library = (codon_path / _LIBRARY).resolve()
    digest = hashlib.sha256(str(library).encode())
    _tree_digest(digest, list(library.rglob("*.codon")))
    _tree_digest(digest, list(model_dir().rglob("*.codon")))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _package_digest() -> str:
    """The framework's Python, the checker's included: every ``*.py`` (and
    ``*.pyi``, which mypy prefers) of the package, by its path in the package
    and its text.

    mypy reads it for the types of what a scenario calls (the None check), and
    the checks Python makes before the compile import it (``sources.problems``).
    It cannot change while the process runs: Python has already imported it.
    """
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), *root.rglob("*.pyi")]):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _mypy_version() -> str:
    """The version of the mypy the None check runs (``python -m mypy``, this
    interpreter's), or ``""`` without one."""
    try:
        return importlib.metadata.version("mypy")
    except importlib.metadata.PackageNotFoundError:
        return ""


@lru_cache(maxsize=1)
def _distributions_digest() -> str:
    """The name and version of every distribution installed beside the
    framework (its own aside, which :func:`_package_digest` covers by its
    text): mypy follows a scenario's imports into them (numpy, lanelet2,
    typesafe_carla's CPython package), and their types decide what may be
    None. An editable install's version says nothing of its edits; those
    are :func:`_outside_digest`'s."""
    own = _PACKAGE.replace("_", "-")
    names = set()
    for dist in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"] or "").lower()
        if name != own:
            names.add(f"{name}=={dist.version}")
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def _python_digest() -> str:
    """What the verdict depends on beyond the compile (#268): the framework's
    Python, the mypy that reads it, the distributions it follows imports
    into, and the Python (its typeshed, and what the checks made in Python
    find) that runs both."""
    digest = hashlib.sha256(_package_digest().encode())
    digest.update(f"\0mypy {_mypy_version()}".encode())
    digest.update(f"\0dists {_distributions_digest()}".encode())
    digest.update(f"\0python {sys.version_info[0]}.{sys.version_info[1]}".encode())
    return digest.hexdigest()


#: What a source tree holds at its top that is none of its importable
#: source: a build's or a colcon workspace's output, a virtualenv. Only at
#: the top: deeper down, ``mypkg/log/`` is a subpackage like any other.
_NOT_SOURCE_DIRS = frozenset(
    {"build", "dist", "env", "install", "log", "node_modules", "venv"}
)


@lru_cache(maxsize=None)
def _source_tree_digest(root: str, own: tuple[str, ...] = ()) -> str:
    """Each ``.py``, ``.pyi`` and ``py.typed`` under *root* by its path, size
    and modification time: what mypy may read there, without reading it
    (a ``PYTHONPATH`` entry can be a whole ROS ``site-packages``).

    Under a directory of *own* (the top packages of the modules being
    checked, whose ``.py`` files are in the key by their text) only stubs
    and markers count: an edit to one scenario of a package does not stale
    the verdicts on the others, but a ``.pyi`` mypy prefers to a module does.
    Symlinked directories are followed (``colcon build --symlink-install``),
    each directory once. A tree that is not there is a digest of its own.
    """
    if not os.path.isdir(root):
        return "missing"
    digest = hashlib.sha256()
    seen: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(directory)
        if real in seen:  # a symlink back up the tree
            dirnames[:] = []
            continue
        seen.add(real)
        top = directory == root
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and d != "__pycache__"
            and not (top and d in _NOT_SOURCE_DIRS)
            and not os.path.isfile(os.path.join(directory, d, "pyvenv.cfg"))
        )
        inside_own = any(
            directory == o or directory.startswith(os.path.join(o, "")) for o in own
        )
        for filename in sorted(filenames):
            if filename.endswith(".pyi") or filename == "py.typed":
                pass
            elif not filename.endswith(".py"):
                continue
            path = os.path.join(directory, filename)
            if inside_own or path in own:
                if filename.endswith(".py"):
                    continue
            try:
                stat = os.stat(path)
            except OSError:
                continue
            relative = os.path.relpath(path, root)
            digest.update(f"\0{relative} {stat.st_size} {stat.st_mtime_ns}".encode())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _editable_roots() -> tuple[tuple[str, str], ...]:
    """``(name, directory)`` of every distribution but the framework that is
    installed editable (PEP 610's ``direct_url.json``)."""
    own = _PACKAGE.replace("_", "-")
    roots = set()
    for dist in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"] or "").lower()
        if name == own:
            continue
        try:
            info = json.loads(dist.read_text("direct_url.json") or "{}")
            if not info.get("dir_info", {}).get("editable"):
                continue
            url = urlparse(info["url"])
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
        if url.scheme == "file" and url.netloc in ("", "localhost"):
            roots.add((name, os.path.abspath(url2pathname(url.path))))
    return tuple(sorted(roots))


def _top_package(path: str) -> str:
    """The directory of the top package holding the module *path* (the last
    one up with an ``__init__.py``), or *path* itself for a lone module."""
    top = path
    directory = os.path.dirname(path)
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        top = directory
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return top


def _outside_digest(sources: _Sources) -> str:
    """What mypy reads that no version names: the source of each editable
    distribution, and the directories of ``MYPYPATH`` and ``PYTHONPATH``
    (mypy searches both), but the ``.py`` files of the packages being
    checked, which are in the key by their text (:func:`_source_tree_digest`)."""
    own = tuple(
        sorted(
            {_top_package(os.path.abspath(m.path)) for m in sources.modules.values()}
        )
    )
    digest = hashlib.sha256()
    trees = [(f"editable {name}", root) for name, root in _editable_roots()]
    trees += [
        (variable, entry)
        for variable in ("MYPYPATH", "PYTHONPATH")
        for entry in search_path(variable)
    ]
    for label, root in trees:
        prefix = os.path.join(root, "")
        inside = tuple(o for o in own if o.startswith(prefix))
        digest.update(f"\0{label} {root} {_source_tree_digest(root, inside)}".encode())
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _toolchain_digest(tc: Toolchain) -> str:
    """The compiler itself: its version and its standard library.

    The path alone names the same file across an upgrade, which a verdict kept
    beyond the process (``$CACHE_DIR_ENV``) would outlive.
    """
    digest = hashlib.sha256(str(tc.executable).encode())
    digest.update(tc.version().encode())
    for directory in tc.library_dirs():
        _tree_digest(digest, list(directory.rglob("*.codon")))
    return digest.hexdigest()


def _cache_key(
    tc: Toolchain, codon_path: Path, sources: _Sources, driver_source: str
) -> str:
    digest = hashlib.sha256(_toolchain_digest(tc).encode())
    digest.update(_framework_digest(codon_path).encode())
    digest.update(_python_digest().encode())
    digest.update(_outside_digest(sources).encode())
    for name, module in sorted(sources.modules.items()):
        digest.update(name.encode())
        digest.update(module.text.encode())
    # The driver names each module's file (its `__file__`); a verdict names it
    # by its marker (_relocated), so the same program elsewhere is a hit.
    markers = _module_markers(sources)
    # What Python found wrong before the compile (an import the framework's
    # Python lacks, an abstract class built) is the verdict when there is
    # any, and a hit returns before _compile reads it (#268).
    problems = _relocated(
        TypeCheckResult("", ok=False, diagnostics=list(sources.problems)), markers
    ).diagnostics
    digest.update(json.dumps([dataclasses.asdict(d) for d in problems]).encode())
    for path in sorted(markers, key=len, reverse=True):
        driver_source = driver_source.replace(path, markers[path])
    digest.update(driver_source.encode())
    return digest.hexdigest()


def typecheck_scenario(
    scenario_cls: type,
    config_cls: type,
    scenario_dict: dict[str, Any] | None = None,
    *,
    toolchain: Toolchain | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> TypeCheckResult:
    """Compile *scenario_cls* built with ``config_cls(**scenario_dict)``.

    Args:
        scenario_cls: The scenario class, as given to ``register_scenario``.
        config_cls: Its config class.
        scenario_dict: The ``scenario`` section of the resolved config, as the
            runner passes it to *config_cls*; ``None`` checks the defaults.
        toolchain: The Codon to compile with; found as typesafe_carla finds
            it, and of a supported release (:func:`find_supported_codon`), by
            default.
        timeout: Seconds the compile may take.

    Returns:
        The result; ``result.ok`` is ``False`` when the scenario must not run.
        A result with ``skipped`` set made no check at all (no Codon, or a
        scenario class without a source file).
    """
    name = f"{scenario_cls.__module__}.{scenario_cls.__qualname__}"
    roots = sorted({scenario_cls.__module__, config_cls.__module__})
    built: list[Any] = []

    def driver() -> Any:
        built.append(render_driver(scenario_cls, config_cls, dict(scenario_dict or {})))
        return built[0]

    result = _typecheck(
        name,
        roots,
        driver,
        toolchain,
        timeout,
        what="scenario",
        refuse=lambda sources: [
            *_abstract_classes_built(sources, scenario_cls),
            *_returns_not_kept(sources),
            *_class_problems(sources),
        ],
    )
    if built and built[0].build_error and result.skipped is None and not result.crashed:
        # The runner builds the config as the driver does, so a constructor
        # or __post_init__ that raises stops the run, whatever Codon says (#531).
        error = Diagnostic(
            f"building {config_cls.__qualname__}(**scenario) raised "
            f"{built[0].build_error}: the runner builds the config the same way",
            "scenario",
        )
        result = dataclasses.replace(
            result, ok=False, diagnostics=[*result.diagnostics, error]
        )
    return result


def _returns_not_kept(sources: _Sources) -> list[Diagnostic]:
    """Each function of the scenario's modules whose declared return type
    Codon does not enforce: a literal returned from a ``-> bool`` and a path
    that returns nothing (#529, :mod:`.return_paths`)."""
    modules = {
        name: (str(module.path), module.tree)
        for name, module in sources.modules.items()
    }
    return [
        Diagnostic(p.message, p.path, p.line, p.column)
        for p in return_problems(modules)
    ]


def _of_package(cls: type) -> bool:
    return cls.__module__ == _PACKAGE or cls.__module__.startswith(_PACKAGE + ".")


def _class_problems(sources: _Sources, *, library: bool = False) -> list[Diagnostic]:
    """Each class of the collected modules Codon builds and Python does not
    (:mod:`.class_rules`): an ``__init__`` that skips a framework base's
    (#587) and an attribute declared at class level that nothing assigns
    (#588).  The framework is the package outside the scenario's modules;
    in the library check, the whole package."""
    modules = {
        name: (str(module.path), module.tree)
        for name, module in sources.modules.items()
    }
    framework: Callable[[type], bool] | None = None
    if library:
        framework = _of_package
    return [
        Diagnostic(p.message, p.path, p.line)
        for p in [
            *skipped_base_inits(modules, framework=framework),
            *unassigned_attributes(modules),
        ]
    ]


def _abstract_classes_built(sources: _Sources, scenario_cls: type) -> list[Diagnostic]:
    """Each class of the scenario's modules that is built but still abstract (#215).

    Codon has no ``@abstractmethod``, and the model gives ``is_done()``,
    ``check()`` and ``execute()`` bodies (its built-in conditions and actions
    inherit them), so a scenario, condition or action that leaves one
    undefined compiles, and Python refuses to build it.  Python says which
    classes are abstract (:func:`inspect.isabstract`); a class counts as built
    when it is the scenario class, which the runner builds, or when the
    scenario's modules call it by name (``GapCondition(...)``,
    ``conditions.GapCondition(...)``).  A class only derived from, such as an
    abstract base of the scenario's own conditions, is never called, so it
    may stay abstract.
    """
    calls: dict[str, tuple[Path, int]] = {}
    for module in sources.modules.values():
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if called is not None:
                calls.setdefault(called, (module.path, node.lineno))
    problems: list[Diagnostic] = []
    for name, module in sorted(sources.modules.items()):
        loaded = sys.modules.get(name)
        if loaded is None:
            continue  # imported nowhere yet, so none of its classes is built yet
        for node in module.tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            cls = getattr(loaded, node.name, None)
            if not (inspect.isclass(cls) and inspect.isabstract(cls)):
                continue
            if cls is scenario_cls:
                where = "by the runner"
            elif node.name in calls:
                path, line = calls[node.name]
                where = f"at {path.name}:{line}"
            else:
                continue
            missing = sorted(getattr(cls, "__abstractmethods__", ()))
            them = "them" if len(missing) > 1 else "it"
            problems.append(
                Diagnostic(
                    f"{node.name} is built ({where}) but leaves abstract "
                    f"{', '.join(missing)} undefined: define {them} in "
                    f"{node.name}, or Python refuses to build it (TypeError: "
                    "Can't instantiate abstract class)",
                    str(module.path),
                    node.lineno,
                )
            )
    return problems


def _typecheck(
    name: str,
    roots: list[str],
    driver: Any,
    toolchain: Toolchain | None,
    timeout: float,
    *,
    what: str,
    refuse: Callable[[_Sources], list[Diagnostic]] | None = None,
) -> TypeCheckResult:
    """Compile the modules *roots* with the driver *driver()* renders, cached.

    *refuse* names what Python already knows is wrong with the collected
    modules; any of it fails the check before Codon runs.
    """
    if toolchain is None:
        try:
            toolchain = find_supported_codon()
        except ToolchainError as exc:
            return TypeCheckResult(name, ok=True, skipped=f"no Codon compiler: {exc}")
    try:
        codon_path = codon_path_dir()
    except ToolchainError as exc:
        return TypeCheckResult(name, ok=True, skipped=f"no CARLA API to check: {exc}")
    if any(_module_file(root) is None for root in roots):
        return TypeCheckResult(
            name, ok=True, skipped=f"the {what} has no source file to compile"
        )
    sources = _collect(roots, toolchain, codon_path)
    if refuse is not None:
        sources.problems += refuse(sources)
    rendered = driver()
    # Keyed by the driver's shape, each config value replaced by its type
    # (Driver.shape): Codon types the program statically, so a sweep's cases
    # that differ only in their values are one program, compiled once.
    key = _cache_key(
        toolchain,
        codon_path,
        sources,
        (rendered.shape or rendered.source) + rendered.trailer,
    )
    cached = _cached(key, sources)
    if cached is not None:
        return _reused(name, key, rendered.case, cached)
    # scenario-check runs configs side by side: one of the same shape that is
    # being compiled right now is waited for rather than compiled again. A
    # compile that timed out is no verdict: who waited for it compiles under
    # its own timeout.
    while True:
        with _IN_FLIGHT_LOCK:
            pending = _IN_FLIGHT.get(key)
            if pending is None:
                if key in _CACHE:  # kept since the miss above
                    return _reused(name, key, rendered.case, _CACHE[key])
                owned: Future[TypeCheckResult | None] = Future()
                _IN_FLIGHT[key] = owned
        if pending is None:
            break
        shared = pending.result()
        if shared is not None:
            return _reused(name, key, rendered.case, shared)
    result: TypeCheckResult | None = None
    try:
        _CHECKED_FOR[key] = rendered.case
        none_check = _none_check_started(sources, timeout)
        result = _compile(name, sources, rendered, toolchain, codon_path, timeout)
        if result is not None:
            result = _none_checked(key, result, sources, none_check)
    except BaseException as exc:
        owned.set_exception(exc)
        raise
    else:
        owned.set_result(result)
    finally:
        with _IN_FLIGHT_LOCK:
            del _IN_FLIGHT[key]
    if result is None:  # timed out: no verdict worth keeping
        return _timed_out(name, timeout)
    return result


def _reused(
    name: str, key: str, case: dict[str, str], result: TypeCheckResult
) -> TypeCheckResult:
    """*result*, the verdict kept (or being made) under *key*, for the check
    *name* of the case *case*: logged when it was made for another case."""
    checked_for = _CHECKED_FOR.setdefault(key, case)
    changed = [k for k, v in case.items() if checked_for.get(k) != v]
    if changed:
        logger.info(
            "%s: reusing the static check made for %s for %s: the config "
            "has the same types, so it is the same program to Codon",
            name,
            ", ".join(f"{k}={checked_for.get(k)}" for k in changed),
            ", ".join(f"{k}={case[k]}" for k in changed),
        )
    return result


def _none_check_started(
    sources: _Sources, timeout: float
) -> "Future[list[NoneFinding]]":
    """The None check of *sources*' modules, started on a thread of its own.

    mypy reads only the modules, so it runs while Codon compiles them rather
    than after it: an uncached check takes the longer of the two, not their
    sum. Its findings are read once the compile is done (and not at all when
    the compile crashed or timed out).
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="none-check")
    future = executor.submit(
        run_none_check, sorted(m.path for m in sources.modules.values()), timeout
    )
    executor.shutdown(wait=False)
    return future


def _none_checked(
    key: str,
    result: TypeCheckResult,
    sources: _Sources,
    none_check: "Future[list[NoneFinding]]",
) -> TypeCheckResult:
    """*result* with the None check's findings (:func:`_with_none_misuses`),
    kept under *key*.

    The key covers what the None check reads beyond the modules
    (:func:`_python_digest`: the framework's Python, the CARLA stubs'
    writer among it, and mypy; :func:`_framework_digest`: the CARLA library
    the stubs are written from), so a kept verdict holds the check's too.
    """
    checked = _with_none_misuses(result, none_check)
    if len(checked.warnings) > len(result.warnings):
        # The None check did not run (mypy missing, out of time): a verdict
        # for this process only, so the next run checks it again.
        _CACHE[key] = checked
        return checked
    return _store(key, checked, sources)


def _with_none_misuses(
    result: TypeCheckResult, none_check: "Future[list[NoneFinding]]"
) -> TypeCheckResult:
    """*result*, failed by each ``None`` misused in the modules (#239).

    Codon unwraps an ``Optional`` implicitly, so the compile accepts a value
    that may be ``None`` used where it may not be; mypy's strict Optional
    does not (:mod:`.none_check`).  Its findings come after Codon's errors.
    Without a verdict from mypy (not installed, out of time, crashed) the
    result keeps Codon's verdict, with a warning saying so.
    """
    if result.crashed:
        return result
    try:
        findings = none_check.result()
    except NoneCheckSkipped as exc:
        warning = Diagnostic(
            f"the None check did not run ({exc}), so an Optional used without "
            "a test for None is not reported",
            severity="warning",
        )
        return dataclasses.replace(result, warnings=[*result.warnings, warning])
    if not findings:
        return result
    diagnostics = [
        Diagnostic(_none_check_advice(f), f.path, f.line, f.column, f.notes)
        for f in findings
    ]
    return dataclasses.replace(
        result, ok=False, diagnostics=[*result.diagnostics, *diagnostics]
    )


def _none_check_advice(finding: NoneFinding) -> str:
    """The message of a :class:`.none_check.NoneFinding`, with what to do."""
    if finding.kind == FROZEN and finding.read_only is not None:
        cls, attr = finding.read_only
        return (
            f"{cls} is a frozen dataclass; use dataclasses.replace(x, {attr}=...) "
            f"instead of assigning its field {attr!r}, or Python raises "
            f"FrozenInstanceError here (mypy: {finding.code})"
        )
    if finding.kind == BYTES_STR:
        return (
            f"{finding.message}: bytes and str do not mix in Python: decode with "
            f".decode() / encode with .encode(); Codon's bytes is a str, so this "
            f"compiles, and Python raises TypeError here or the comparison is "
            f"always False (mypy: {finding.code})"
        )
    if finding.kind == ENUM_EQ:
        return (
            f"{finding.message}: an Enum member never equals its raw value: "
            f"compare with Color.RED, or use Color.RED.value; Python evaluates "
            f"this comparison to False (mypy: {finding.code})"
        )
    if finding.kind == FINAL:
        return (
            f"{finding.message}: a name declared Final is bound once; drop the "
            f"assignment, bind a new name, or drop Final from the declaration. "
            f"Python does not stop it and Codon ignores Final, so this runs "
            f"with the value the declaration promised never to change "
            f"(mypy: {finding.code})"
        )
    if finding.kind == CLASS_VAR:
        return (
            f"{finding.message}: a ClassVar is shared by the class; assign it "
            f"through the class (ClassName.name = ...), or declare it without "
            f"ClassVar to make it an instance field. Python creates a "
            f"shadowing instance attribute here and Codon treats the "
            f"annotated name as an instance field, so what runs can differ "
            f"(mypy: {finding.code})"
        )
    if finding.kind == OVERRIDE:
        return (
            f"{finding.message}: the override must accept what the overridden "
            f"method accepts and return what it returns, as the caller (the "
            f"framework, for a hook of BaseScenario) calls it with the "
            f"superclass's signature; Python fails on the call that does not "
            f"fit, and Codon does not compare the two. Match the "
            f"superclass's signature (extra parameters need a default), or "
            f"add `# type: ignore[override]` if the narrowing is intended "
            f"(mypy: {finding.code})"
        )
    if finding.kind == READ_ONLY_ABC:
        return (
            f"{finding.message}: a Mapping / Sequence / AbstractSet parameter "
            f"is read-only by its annotation; copy it first (dict(x), "
            f"list(x), set(x)) and mutate the copy, or annotate it with the "
            f"mutable type (dict, list, set). Python fails here when the "
            f"caller passes a tuple or a MappingProxyType, and Codon does not "
            f"enforce the annotation (mypy: {finding.code})"
        )
    if finding.kind == STR_FORMAT:
        return (
            f"{finding.message}: Python raises ValueError, TypeError, "
            f"IndexError or KeyError on this format string with these values "
            f'("{{:d}}" and "{{:x}}" take an int, "%d" and "{{:.2f}}" a number; '
            f"each field needs a value); convert the value first (int(x), "
            f'str(x)) or use "{{}}" / "%s". Codon formats with its own '
            f"rules, so this compiles (mypy: {finding.code})"
        )
    union = finding.union
    if union is None:
        wanted = none_given(finding.message)
        if wanted is not None:  # None itself: there is no value to test
            return _none_not_taken(finding, wanted)
        return (
            f"{finding.message}: Codon unwraps an Optional implicitly, so test "
            f"the value for None before this use, or Python fails here when it "
            f"is None (mypy: {finding.code})"
        )
    return (
        f"{finding.message}: Codon holds this union of the framework's "
        f"({union.name}) and converts it to the member this use needs at run "
        f"time, so narrow it with isinstance first, or Python fails here on "
        f"any other member (mypy: {finding.code})"
    )


_ARGUMENT = re.compile(r'^Argument (?P<n>\d+) to "(?P<callee>[^"]+)"')


def _none_not_taken(finding: NoneFinding, wanted: str) -> str:
    """The message for ``None`` itself given where *wanted* is expected."""
    argument = _ARGUMENT.match(finding.message)
    if argument is not None:
        what = (
            f"argument {argument['n']} of {argument['callee']}() does not take "
            f"None, it takes {wanted}"
        )
    elif finding.message.startswith("Incompatible return value type"):
        what = f"the function is declared -> {wanted}, so it cannot return None"
    elif finding.message.startswith("Incompatible types in assignment"):
        what = f"this is declared {wanted}, so it cannot be assigned None"
    else:
        what = f"None is given where {wanted} is expected, and it does not take None"
    return f"{what} (mypy: {finding.code})"


def typecheck_odd(
    builder: Any,
    *,
    toolchain: Toolchain | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> TypeCheckResult:
    """Compile the ODD *builder* returns, as :func:`typecheck_scenario` compiles a scenario.

    *builder* is a module-level function that takes nothing and returns an
    :class:`~autoware_carla_scenario.OddDefinition`.  The driver calls it, so
    everything it reaches is checked: each attribute's probe is called with a
    world, each condition must come from an attribute, and the result must
    be an ODD.

    Returns:
        The result; ``result.ok`` is ``False`` when the ODD must not be used.
    """
    module = getattr(builder, "__module__", "") or ""
    qualname = getattr(builder, "__qualname__", "<callable>")
    name = f"{module}.{qualname}"
    if "." in qualname or "<" in qualname:
        return TypeCheckResult(
            name, ok=True, skipped="only a module-level function can be compiled"
        )
    return _typecheck(
        name,
        [module],
        lambda: render_odd_driver(module, qualname),
        toolchain,
        timeout,
        what="ODD",
    )


def _timed_out(name: str, timeout: float) -> TypeCheckResult:
    return TypeCheckResult(
        name,
        ok=False,
        crashed=True,
        diagnostics=[
            Diagnostic(f"Codon did not finish compiling within {timeout:.0f}s")
        ],
        seconds=timeout,
    )


def _compile(
    name: str,
    sources: _Sources,
    driver: Any,
    toolchain: Toolchain,
    codon_path: Path,
    timeout: float,
    build: Callable[[Path], tuple[_Workspace, list[Diagnostic]]] | None = None,
) -> TypeCheckResult | None:
    """The verdict on *sources*, or ``None`` when Codon ran out of time.

    *build* writes the workspace into the directory it is given
    (:func:`_build_workspace` by default).
    """
    if sources.problems:
        return TypeCheckResult(name, ok=False, diagnostics=list(sources.problems))

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="acs-typecheck-") as tmp:
        if build is None:
            ws, problems = _build_workspace(Path(tmp), sources, driver, codon_path)
        else:
            ws, problems = build(Path(tmp))
        if problems:
            return TypeCheckResult(
                name, ok=False, diagnostics=problems, warnings=list(ws.warnings)
            )
        env = codon_environment(toolchain, ws.root)
        try:
            proc = subprocess.run(  # noqa: S603 - a fixed compiler invocation
                [
                    str(toolchain.executable),
                    "build",
                    "-llvm",
                    "-o",
                    os.devnull,
                    DRIVER_MODULE,
                ],
                cwd=ws.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        output = proc.stdout + proc.stderr
        diagnostics = _data_model_hints(
            _with_member_reasons(_variadic_hints(_parse_output(output, ws)), sources),
            sources,
        )
        if proc.returncode != 0:
            try:
                diagnostics += _uninferred_attributes(diagnostics, sources)
            except Exception:  # only a hint: never lose Codon's own errors to it
                pass
            try:
                diagnostics += _unplaced_none_mismatches(diagnostics, sources)
            except Exception:  # only a hint: never lose Codon's own errors to it
                pass
    seconds = time.monotonic() - started
    refiner = Refiner(
        lambda name, path: _python_class(name, sources, path),
        lambda name, path: _python_callable(name, sources, path),
    )
    codegen = proc.returncode != 0 and _LLVM_ERROR in output
    if codegen:
        diagnostics = [_codegen_crash(output, sources)]
    crashed = proc.returncode != 0 and (
        proc.returncode < 0
        or proc.returncode > 128
        or codegen
        or any(m in output for m in _CRASH_MARKERS)
    )
    if proc.returncode != 0 and not diagnostics and _SIGNATURE_MISMATCH in output:
        try:
            diagnostics = _callable_signature_hints(output, sources)
        except Exception:  # only a hint: the crash is reported either way
            diagnostics = []
    if proc.returncode != 0 and not diagnostics and _BAD_ALLOC in output:
        try:
            diagnostics = _bad_alloc_hints(sources)
        except Exception:  # only a hint: the crash is reported either way
            diagnostics = []
    if proc.returncode != 0 and not diagnostics:
        tail = _ANSI.sub("", output).strip().splitlines()[-1:] or [
            f"exit status {proc.returncode}"
        ]
        diagnostics = [Diagnostic(f"Codon failed without a diagnostic: {tail[0]}")]
        try:
            # Codon 0.19 segfaults on some locals first bound to None (#584):
            # name each, after the crash itself (which may have another cause).
            diagnostics += _none_first_binding_hints(
                sources, f"Codon crashed ({tail[0]})"
            )
        except Exception:  # only a hint: the crash is reported either way
            pass
    return TypeCheckResult(
        name,
        ok=proc.returncode == 0,
        diagnostics=[
            _without_lambda_names(_authors_terms(d, refiner)) for d in diagnostics
        ],
        warnings=list(ws.warnings),
        crashed=crashed,
        output=output,
        seconds=seconds,
    )


def _unplaced_none_mismatches(
    diagnostics: list[Diagnostic], sources: _Sources
) -> list[Diagnostic]:
    """The locals first bound to ``None`` when Codon refused a value as
    ``Optional[NoneType]`` at a line that rebinds none of them (#584)."""
    if not any(
        _NONE_TYPE_MISMATCH in d.message and "Codon fixes the type" not in d.message
        for d in diagnostics
    ):
        return []
    return _none_first_binding_hints(
        sources, f"Codon refused a value as {_NONE_TYPE_MISMATCH}"
    )


_NONE_TYPE_MISMATCH = "expected type 'Optional[NoneType]'"


def _callable_signature_hints(output: str, sources: _Sources) -> list[Diagnostic]:
    """A function of another signature than the ``Callable`` it is passed as (#488).

    Codon 0.19 does not compare a function passed to a ``Callable`` parameter
    with the parameter's signature (a lambda it does), and generates a call
    LLVM refuses.  The function is named in the IR; each call of the checked
    modules that passes it is reported, or the crash once if none is found.
    """
    text = _ANSI.sub("", output)
    found = _PASSED_FUNCTION.search(text)
    name = found["fn"] if found is not None else None
    message = (
        f"{'`' + name + '`' if name else 'A function'} is passed where a "
        "`Callable` is expected, and has another signature than it: Codon 0.19 "
        "does not compare them and fails in code generation; give the function "
        "the argument and result types of the `Callable`"
    )
    lambda_message = (
        "A lambda that takes no arguments passed where a `Callable` is expected "
        "is a likely cause: Codon 0.19 converts it without comparing the "
        "signature and fails in code generation; pass a named function "
        "(`def f() -> None`)"
    )
    hints: list[Diagnostic] = []
    for module in sources.modules.values():
        for call in ast.walk(module.tree):
            if not isinstance(call, ast.Call):
                continue
            for arg in [*call.args, *(k.value for k in call.keywords)]:
                if name is not None and (
                    (isinstance(arg, ast.Name) and arg.id == name)
                    or (isinstance(arg, ast.Attribute) and arg.attr == name)
                ):
                    hints.append(Diagnostic(message, str(module.path), call.lineno))
                elif (
                    name is None
                    and isinstance(arg, ast.Lambda)
                    and not ast.unparse(arg.args)
                    and (_dotted_name(call.func) or "").rpartition(".")[2]
                    not in _FACTORY_CALLS
                ):
                    # No function is named in the IR: a lambda has no name.
                    hints.append(
                        Diagnostic(lambda_message, str(module.path), arg.lineno)
                    )
    return hints or [Diagnostic(message)]


def _bad_alloc_hints(sources: _Sources) -> list[Diagnostic]:
    """Codon aborting with ``std::bad_alloc``, as what is likely to cause it (#488).

    Codon 0.19 gives no file, line or message for it.  The cause seen is a
    class attribute annotated with a ``Callable`` (alone, or inside a list or
    dict) that a value of no declared type is stored in; the transform now
    keeps the ``Callable`` annotation of a parameter, so what is left is a
    stored value of another unannotated kind.  Each such attribute is named at
    its line, and the crash is reported once if none is found.
    """
    found: list[Diagnostic] = []
    for module in sources.modules.values():
        for cls in ast.walk(module.tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and any(
                    isinstance(n, (ast.Name, ast.Attribute))
                    and (_dotted_name(n) or "").rpartition(".")[2] == "Callable"
                    for n in ast.walk(stmt.annotation)
                ):
                    found.append(
                        Diagnostic(
                            "Codon aborted with `std::bad_alloc` (it gives no "
                            "location); a `Callable` attribute is a likely cause: "
                            "annotate every parameter or variable stored in it "
                            "with the same `Callable[[...], R]`, or store a "
                            "function the attribute's signature names",
                            str(module.path),
                            stmt.lineno,
                        )
                    )
    return found or [
        Diagnostic(
            "Codon aborted with `std::bad_alloc` (it gives no location): it ran "
            "out of memory, or was given a type it cannot hold, often a "
            "`Callable` annotation on an attribute that a value of another "
            "type is stored in"
        )
    ]


def _codegen_crash(output: str, sources: _Sources) -> Diagnostic:
    """Codon 0.19's code generation failing, as what to change (#263).

    Codon checked the program, then generated LLVM IR that LLVM refuses: a
    bug of Codon's. Every case seen is an ``Optional`` of a class with
    subclasses, which Codon lowers as one of them before it knows of the
    other. Named by the class the error's frame names, at the ``class`` line
    of that class or of a subclass of it in the checked modules.
    """
    text = _ANSI.sub("", output)
    llvm = next(
        (line.strip() for line in text.splitlines() if _LLVM_ERROR in line),
        _LLVM_ERROR,
    )
    llvm = llvm[llvm.index(_LLVM_ERROR) :]
    found = _LLVM_CLASS.search(text)
    if found is None:
        return Diagnostic(
            "Codon 0.19 generated code it cannot compile, a bug of Codon's "
            "rather than an error of the program (#263); the cases seen are an "
            "Optional of a class with subclasses (`x: X | None = None`) where "
            "no object of the class or of a subclass is built: build one where "
            "setup() reaches it, or hold the value without Optional",
            trace=(llvm,),
        )
    qualified = found["cls"].removeprefix("std.")
    cls = qualified.rpartition(".")[2]
    path: str | None = None
    line: int | None = None
    sub: str | None = None
    for module in sources.modules.values():
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                b.id if isinstance(b, ast.Name) else b.attr
                for b in node.bases
                if isinstance(b, (ast.Name, ast.Attribute))
            }
            if node.name == cls or cls in bases:
                path, line = str(module.path), node.lineno
                sub = node.name if node.name != cls else None
                break
        if path is not None:
            break
    build = f"a {sub}" if sub is not None else f"a {cls} (or one of its subclasses)"
    return Diagnostic(
        f"Codon 0.19 generated code it cannot compile for {qualified}, a class "
        "with subclasses: a bug of Codon's rather than an error of the program "
        "(#263). It does so for an Optional of such a class "
        f"(`x: {cls} | None = None`, a parameter that defaults to None) when "
        "the program builds no object of the class or of a subclass, or "
        "defines a subclass after Codon compiled the class. Build "
        f"{build} where setup() reaches it, or hold the value without Optional",
        path,
        line,
        trace=(llvm,),
    )


# ---------------------------------------------------------------------------
# The library check: the framework's own modules (library.py)
# ---------------------------------------------------------------------------

_LIBRARY_DRIVER_LABEL = "<the library check's driver>"


def _model_module(name: str) -> str | None:
    """The model module standing in for the framework module *name*, if any.

    The model has one module per public package (``coordinate.codon``) where
    the framework has a package (``coordinate/transform.py``): a module
    stands in for every module under it.
    """
    modeled = _model_modules()
    parts = name.split(".")
    for end in range(len(parts), 1, -1):
        candidate = ".".join(parts[:end])
        if candidate in modeled:
            return candidate
    return name if name == _PACKAGE else None


@dataclass(frozen=True)
class _LibraryFile:
    """A checked module, as it is compiled: rewritten, with the calls appended."""

    module: str
    rel: str
    text: str
    #: Lines of :attr:`text` before the appended calls.
    offset: int
    labels: dict[int, str]
    #: The line of the module each line of :attr:`text` after the prelude
    #: comes from (:func:`.transform.transform_source_lines`).
    origin: list[int]


def _uncalled(module: str) -> list[str]:
    """What :data:`.library.UNCALLED` lists of *module*, relative to it."""
    # Keyed by what the lists hold now, not by the module alone: tests add to
    # them, and a library check asks for each module several times.
    return list(_uncalled_of(module, tuple(CHECKED), tuple(UNCALLED)))


@lru_cache(maxsize=1024)
def _uncalled_of(
    module: str, checked: tuple[str, ...], uncalled: tuple[str, ...]
) -> tuple[str, ...]:
    # A name belongs to the longest module it starts with:
    # ``trajectory.model._wrap`` is of ``trajectory.model``, not of the
    # ``trajectory`` package.
    modules = {*checked, module}
    out: list[str] = []
    for name in uncalled:
        owners = [m for m in modules if name.startswith(f"{m}.")]
        if owners and max(owners, key=len) == module:
            out.append(name.removeprefix(f"{module}."))
    return tuple(out)


def _read_module(path: Path) -> _Module:
    text = path.read_text(encoding="utf-8")
    return _Module(path, path.name == "__init__.py", text, ast.parse(text))


def _library_dependencies(
    names: list[str], sources: _Sources, every: dict[str, Path]
) -> list[str]:
    """The :data:`.library.CHECKED` modules *names* import, transitively.

    Each is read into *sources*.  An import the importing module makes
    through the model (:data:`.library.MODELLED_IMPORTS`) or inside a function
    the check leaves out (:data:`.library.UNCALLED`) is not followed: the
    module's source is not imported there in the full run either.

    Once one module of a :data:`.library.REPLACED_MODELS` entry is compiled,
    the others come too (when they are all checked), so that model module is
    a re-export of the checked definitions, as in the full run, rather than
    a second set of the same types (the vehicle entity's config, which holds
    the checked `GroundProjectionConfig` once `entity._spawn` is compiled).
    """
    checked = set(CHECKED)
    found: list[str] = []
    pending = list(names)

    def add(dep: str) -> None:
        if dep in checked and dep not in sources.modules:
            sources.modules[dep] = _read_module(every[dep])
            found.append(dep)
            pending.append(dep)

    while pending:
        while pending:
            name = pending.pop()
            module = sources.modules[name]
            modelled = set(MODELLED_IMPORTS.get(name, ()))
            skipped = {
                id(node)
                for function in uncalled_nodes(module.tree, _uncalled(name))
                for node in ast.walk(function)
            }
            for imported, aliases, _line in _imports(
                module.tree, name, module.is_package, skip=skipped
            ):
                # `from a import b`: b may be a submodule, not a name of a.
                subs = [f"{imported}.{alias}" for alias in aliases]
                for dep in sorted({s if s in every else imported for s in subs}):
                    if dep not in modelled:
                        add(dep)
                if not aliases and imported not in modelled:
                    add(imported)
        for modules in REPLACED_MODELS.values():
            if any(m in sources.modules for m in modules):
                for dep in modules:
                    add(dep)
    return sorted(found)


def _library_files(
    names: list[str], sources: _Sources, every: dict[str, Path]
) -> list[_LibraryFile]:
    """Each module of *sources* rewritten for the library workspace.

    A problem found on the way (an import that cannot be redirected, a
    parameter the check cannot call with) is added to ``sources.problems``.
    """
    checked = set(names)

    def targets(name: str) -> Callable[[str], str | None]:
        modelled = set(MODELLED_IMPORTS.get(name, ()))

        def target(module: str) -> str | None:
            if module in checked and module not in modelled:
                return workspace_module(module)
            return _model_module(module)

        return target

    # The classes of the checked modules, by name: a dataclass derived from
    # one is constructed with the fields of its base too.
    classes = {
        node.name: node
        for name in names
        for node in sources.modules[name].tree.body
        if isinstance(node, ast.ClassDef)
    }
    init_classes = _init_classes(classes)
    files: list[_LibraryFile] = []
    for name in names:
        module = sources.modules[name]
        redirected, problems = redirect_imports(
            module.text,
            name,
            module.is_package,
            targets(name),
            every.__contains__,
            uncalled=_uncalled(name),
        )
        uncalled = _uncalled(name)
        checks = render_library_checks(module.tree, uncalled, classes)
        for line, message in [
            *problems,
            *checks.problems,
            *python_rewrite_refusals(redirected),
            *typing_rewrite_refusals(redirected),
        ]:
            sources.problems.append(Diagnostic(message, str(module.path), line))
        # A function the check never compiles cannot decorate one it does: a
        # run-time-only wrapper (trajectory.model's TrajectoryVertex.__init__)
        # is dropped, and the function it wraps is compiled as written.
        body, origin = transform_source_lines(
            redirected,
            dropped_decorators=[n for n in uncalled if "." not in n],
            init_classes=init_classes,
            enum_classes=enum_class_names(classes),
            file=str(module.path),
        )
        rel = workspace_module(name).replace(".", "/") + ".codon"
        files.append(
            _LibraryFile(
                name,
                rel,
                body + checks.source,
                body.count("\n"),
                checks.labels,
                origin,
            )
        )
    return files


def _run_time_only_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Every node of the code only Python runs: the ``else`` branch of each
    ``if TYPE_CHECKING:`` and the body of each ``if not TYPE_CHECKING:``.

    Codon reads ``TYPE_CHECKING`` as a compile-time true (the ``typing``
    shim), so it compiles only the branch taken when it is true.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if _is_type_checking(node.test):
            run_time_only = node.orelse
        elif (
            isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and _is_type_checking(node.test.operand)
        ):
            run_time_only = node.body
        else:
            continue
        for stmt in run_time_only:
            yield from ast.walk(stmt)


def _is_type_checking(node: ast.expr) -> bool:
    return _dotted_name(node) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _library_imports(sources: _Sources, tc: Toolchain, codon_path: Path) -> None:
    """Report each import of a module Codon has nothing for."""
    known = _codon_stdlib(tc) | _shims() | _linked(codon_path) | {_PACKAGE}
    for name, module in sources.modules.items():
        # Codon compiles no import in a function nothing calls.
        skipped = {
            id(node)
            for function in uncalled_nodes(module.tree, _uncalled(name))
            for node in ast.walk(function)
        }
        # Nor one in the `else` branch of `if TYPE_CHECKING:` or the body of
        # `if not TYPE_CHECKING:`, a compile-time true there (the typing shim).
        skipped |= {id(node) for node in _run_time_only_nodes(module.tree)}
        for imported, _names, lineno in _imports(
            module.tree, name, module.is_package, skip=skipped
        ):
            if imported.split(".")[0] not in known:
                sources.problems.append(
                    Diagnostic(
                        f"{imported} has no Codon model, so a module importing it "
                        "cannot be checked: move the module to EXCLUDED "
                        "(typecheck/library.py) with that reason",
                        str(module.path),
                        lineno,
                    )
                )


def _build_library_workspace(
    root: Path,
    sources: _Sources,
    files: list[_LibraryFile],
    driver: Any,
    codon_path: Path,
) -> tuple[_Workspace, list[Diagnostic]]:
    """The model as the scenario check has it, and the checked modules beside it.

    The model stays at ``autoware_carla_scenario/``; each checked module is
    one file of ``_acs_lib/`` (:func:`.library_driver.workspace_module`), and
    every import of the framework in it names either another checked module
    there or the model (:func:`.transform.redirect_imports`).
    """
    ws = _base_workspace(root, driver, codon_path)
    ws.driver_label = _LIBRARY_DRIVER_LABEL
    problems: list[Diagnostic] = []
    declared = _declarations(sources)
    for file in files:
        module = sources.modules[file.module]
        # What Codon never compiles needs no declaration (an __init__ the
        # check leaves uncalled, LaneletRegion's).
        for attr in undeclared_attributes(
            module.tree, declared, uncalled=_uncalled(file.module)
        ):
            problems.append(Diagnostic(attr.message(), str(module.path), attr.lineno))
        dest = root / file.rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(file.text)
        ws.files[file.rel] = str(module.path)
        ws.appended[file.rel] = (file.offset, file.labels)
        ws.origins[file.rel] = file.origin
    (root / Path(files[0].rel).parent / "__init__.codon").write_text("")
    for stem, text in replaced_models(sources.modules, problems).items():
        (root / _PACKAGE / f"{stem}.codon").write_text(text)
    (root / DRIVER_MODULE).write_text(driver.source)
    return ws, problems


def _top_level_names(tree: ast.Module) -> set[str]:
    """The classes, functions and variables a module defines at top level."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def replaced_models(
    checked: Mapping[str, Any], problems: list[Diagnostic]
) -> dict[str, str]:
    """Model module stem -> its text in the library check (:data:`.library.REPLACED_MODELS`).

    A model module all of whose modules are in *checked* (dotted name ->
    anything with the module's ``tree``) becomes a re-export of each name it
    declares, from the checked module that defines it.  A name no checked
    module defines is added to *problems*: the model and the source disagree.
    """
    out: dict[str, str] = {}
    for stem, modules in REPLACED_MODELS.items():
        if not all(m in checked for m in modules):
            continue
        model = model_dir() / _PACKAGE / f"{stem}.codon"
        imports: dict[str, list[str]] = {m: [] for m in modules}
        defined = {m: _top_level_names(checked[m].tree) for m in modules}
        for name in sorted(_top_level_names(ast.parse(model.read_text()))):
            owner = next((m for m in modules if name in defined[m]), None)
            if owner is None:
                problems.append(
                    Diagnostic(
                        f"{name}, which the model declares, is defined by none "
                        f"of {', '.join(modules)}",
                        str(model),
                    )
                )
                continue
            imports[owner].append(name)
        out[stem] = "".join(
            f"from {workspace_module(m)} import {', '.join(names)}\n"
            for m, names in imports.items()
            if names
        )
    return out


def typecheck_library(
    modules: Iterable[str] | None = None,
    *,
    toolchain: Toolchain | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> TypeCheckResult:
    """Compile the framework's own modules, as :data:`.library.CHECKED` lists them.

    Each module is compiled from its source, every function and method
    (protected ``_`` ones included) called with a value of each parameter's annotated type
    (:mod:`.library_driver`), so its body is checked.  A framework module
    one of them imports that is not checked itself stands in through the
    model, as it does for a scenario.  All of *modules* are compiled
    together, in one Codon build.

    Named *modules* come with the checked modules they import, transitively
    (:func:`_library_dependencies`), compiled and called the same way, so
    the run is a slice of the full one; an error in any of them is reported.

    Args:
        modules: Dotted module names; :data:`.library.CHECKED` by default.
        toolchain: The Codon to compile with (:func:`find_supported_codon`
            by default).
        timeout: Seconds the compile may take.

    The modules compiled are then given the None check, as a scenario's are
    (:func:`_with_none_misuses`): an ``Optional`` CARLA or the framework
    returns used without a test for ``None`` fails the check (#277).

    Returns:
        The result; ``result.ok`` is ``False`` when a module does not compile
        or uses a value that may be ``None`` where it may not be.
    """
    names = sorted(CHECKED if modules is None else set(modules))
    title = f"{_PACKAGE} library ({len(names)} checked modules)"
    if not names:
        return TypeCheckResult(title, ok=True, skipped="no module to check")
    if toolchain is None:
        try:
            toolchain = find_supported_codon()
        except ToolchainError as exc:
            return TypeCheckResult(title, ok=True, skipped=f"no Codon compiler: {exc}")
    try:
        codon_path = codon_path_dir()
    except ToolchainError as exc:
        return TypeCheckResult(title, ok=True, skipped=f"no CARLA API to check: {exc}")
    every = package_modules()
    unknown = [name for name in names if name not in every]
    if unknown:
        return TypeCheckResult(
            title,
            ok=False,
            diagnostics=[
                Diagnostic(
                    f"{name} is not a module of {_PACKAGE} the check can compile"
                )
                for name in unknown
            ],
        )
    sources = _Sources()
    for name in names:
        sources.modules[name] = _read_module(every[name])
    dependencies = _library_dependencies(names, sources, every)
    if dependencies:
        title = (
            f"{_PACKAGE} library ({len(names)} checked modules, "
            f"{len(dependencies)} more they import)"
        )
    names = sorted([*names, *dependencies])
    _library_imports(sources, toolchain, codon_path)
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - its classes are then left alone
            logger.debug("%s does not import: its classes are not checked", name)
    sources.problems += _class_problems(sources, library=True)
    files = _library_files(names, sources, every)
    driver = render_library_driver(names)
    key = _cache_key(
        toolchain,
        codon_path,
        sources,
        driver.source + "".join(f.text for f in files),
    )
    cached = _cached(key, sources)
    if cached is not None:
        return cached
    none_check = _none_check_started(sources, timeout)
    result = _compile(
        title,
        sources,
        driver,
        toolchain,
        codon_path,
        timeout,
        build=lambda root: _build_library_workspace(
            root, sources, files, driver, codon_path
        ),
    )
    if result is None:
        return _timed_out(title, timeout)
    # Codon unwraps an Optional in the framework's modules as it does in a
    # scenario's, so they get the scenario's None check too (#277).
    return _none_checked(key, result, sources, none_check)
