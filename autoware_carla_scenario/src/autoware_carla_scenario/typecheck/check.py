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
from .none_check import NoneCheckSkipped, NoneFinding, run_none_check, search_path
from .not_modelled import NOT_MODELLED, not_modelled_member
from .library_driver import (
    call_sites,
    render_library_checks,
    render_library_driver,
    uncalled_nodes,
    workspace_module,
)
from .python_rewrites import REWRITE_HINTS, python_rewrite_refusals
from .transform import (
    PRELUDE,
    class_declarations,
    classes_with_init,
    conflicting_imports,
    enum_class_names,
    redirect_imports,
    relative_module_imports,
    transform_source_lines,
    undeclared_attributes,
)

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
    "contextlib": "write a class with __enter__ and __exit__(self, exc_type, "
    "exc, tb) for contextmanager (the check calls __exit__ with no arguments, "
    "so those parameters are None in it), and try/except X: pass for suppress",
    "warnings": "log with logger.warning(...) instead of warnings.warn(...)",
    "json": "the check has no dict[str, Any] to load or dump: read the "
    "configuration into a @dataclass, and leave serialisation to a module the "
    "check does not compile",
    "types": "name the type itself (SimpleNamespace is a @dataclass or a plain "
    "class; MappingProxyType is a dict; annotate the tb of __exit__ as object)",
    "textwrap": "build the string with str methods and join()",
    "traceback": "log the exception with logger.exception(...)",
    "io": "use a str, or a pathlib.Path opened with open()",
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
            module.text, name, module.is_package, sources.modules.__contains__
        )
        problems += [
            Diagnostic(message, str(module.path), line) for line, message in refused
        ]
        problems += [
            Diagnostic(message, str(module.path), line)
            for line, message in [
                *conflicting_imports(source),
                *python_rewrite_refusals(source),
            ]
        ]
        text, origin = transform_source_lines(
            source, init_classes=init_classes, enum_classes=enum_class_names(classes)
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
        re.compile(r"cannot import name 'Generic' from 'typing'"),
        "typing.Generic is not available in the check: leave the type "
        "parameters of a class or function unannotated (Codon makes it generic "
        "over them), or write the concrete type",
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
        re.compile(r"cannot import name '\w+' from 'dataclasses'"),
        "the check models only dataclass, field and replace of dataclasses: "
        'docs/typecheck.md ("standard library names") lists the names',
    ),
    (
        re.compile(
            r"cannot import name '\w+' from '(functools|itertools|collections)'"
        ),
        "the check adds only some names to Codon's functools, itertools and "
        'collections: docs/typecheck.md ("standard library names") lists them',
    ),
    (
        re.compile(r"cannot import name '\w+' from 'typing'"),
        "the check models only some names of typing: docs/typecheck.md "
        '("typing names") lists them',
    ),
)


def _variadic_hints(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    """*diagnostics*, each refusal of a Codon limit on ``**kwargs``, on
    ``*args`` or on a tuple saying why and what to write instead (#266)."""
    out = []
    for diagnostic in diagnostics:
        for pattern, hint in (*_VARIADIC_LIMITS, *REWRITE_HINTS):
            if pattern.search(diagnostic.message):
                diagnostic = dataclasses.replace(
                    diagnostic, message=f"{diagnostic.message}: {hint}"
                )
                break
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


def _python_class(name: str, sources: _Sources) -> type | None:
    """The Python class Codon calls *name*: one of the checked modules' (a
    scenario's own subclass), else one a modelled module defines."""
    for module in [*sources.modules, *sorted(_model_modules())]:
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
_NO_HASH = re.compile(r"^'(?P<cls>\w+)' object has no attribute '__hash__'")
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
        except Exception:  # noqa: BLE001 - only a hint: never lose Codon's error
            pass
        if hint is not None:
            diagnostic = dataclasses.replace(diagnostic, message=f"{message}: {hint}")
        out.append(diagnostic)
    return out


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
                    "eq=True (the default) and bases it can see the fields "
                    "of (dataclasses of this module); with eq=False Python "
                    "compares by identity: write `a is b`, or compare the "
                    "fields"
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


def _parse_output(output: str, ws: _Workspace) -> list[Diagnostic]:
    """Codon's errors, each located at its innermost frame in the scenario's files."""
    by_basename: dict[str, list[str]] = {}
    for rel in ws.files:
        by_basename.setdefault(Path(rel).name, []).append(rel)

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
    return _typecheck(
        name,
        roots,
        lambda: render_driver(scenario_cls, config_cls, dict(scenario_dict or {})),
        toolchain,
        timeout,
        what="scenario",
        refuse=lambda sources: _abstract_classes_built(sources, scenario_cls),
    )


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
    union = finding.union
    if union is None:
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
    seconds = time.monotonic() - started
    codegen = proc.returncode != 0 and _LLVM_ERROR in output
    if codegen:
        diagnostics = [_codegen_crash(output, sources)]
    crashed = proc.returncode != 0 and (
        proc.returncode < 0
        or proc.returncode > 128
        or codegen
        or any(m in output for m in _CRASH_MARKERS)
    )
    if proc.returncode != 0 and not diagnostics:
        tail = _ANSI.sub("", output).strip().splitlines()[-1:] or [
            f"exit status {proc.returncode}"
        ]
        diagnostics = [Diagnostic(f"Codon failed without a diagnostic: {tail[0]}")]
    return TypeCheckResult(
        name,
        ok=proc.returncode == 0,
        diagnostics=diagnostics,
        warnings=list(ws.warnings),
        crashed=crashed,
        output=output,
        seconds=seconds,
    )


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
        )
        rel = workspace_module(name).replace(".", "/") + ".codon"
        # Python sets a module's `__file__`, which Codon 0.19 does not: a
        # checked module importing it from another (`from ..autoware_bridge
        # import __file__`, in bridge_node) finds it here, after the checks so
        # no line of the module moves.
        file_global = f"\n__file__: str = {str(module.path)!r}\n"
        files.append(
            _LibraryFile(
                name,
                rel,
                body + checks.source + file_global,
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
