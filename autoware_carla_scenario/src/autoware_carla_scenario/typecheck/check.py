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
annotation naming ``Any``) is left to what reaches it, with a warning.

:func:`typecheck_library` compiles the framework's own modules instead (those
:mod:`.library` lists as checked), each with calls to every function and method,
protected ones included, appended (:mod:`.library_driver`), next to the same model.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

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
from .library_driver import (
    render_library_checks,
    render_library_driver,
    uncalled_nodes,
    workspace_module,
)
from .transform import (
    PRELUDE,
    class_declarations,
    classes_with_init,
    redirect_imports,
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
#: Where an error in the generated driver is reported: the scenario does not
#: build the way the runner builds it (its constructor, setup() or is_done()).
_DRIVER_LABEL = "<the scenario as register_scenario() builds it>"


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
        for imported, names, lineno in _imports(tree, name, is_package):
            top = imported.split(".")[0]
            if imported in modeled or top in shims:
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
                what = f"{imported} is not part of the Codon model; import from {_PACKAGE} instead"
            else:
                what = f"{imported} has no Codon model, so a scenario importing it cannot be checked"
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
    for name, module in sorted(sources.modules.items()):
        rel = Path(*name.split("."))
        rel = rel / "__init__.py" if module.is_package else rel.with_suffix(".py")
        for attr in undeclared_attributes(module.tree, declared):
            problems.append(Diagnostic(attr.message(), str(module.path), attr.lineno))
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        text, origin = transform_source_lines(module.text, init_classes=init_classes)
        checks = render_library_checks(
            module.tree, classes=classes, skip_any_returns=True
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
        driver.source + render_module_checks(sorted(sources.modules))
    )
    return ws, problems


def _original_line(origin: list[int] | None, line: int) -> int:
    """The line of the original file that *line* of its rewrite comes from."""
    line -= _PRELUDE_LINES
    if not origin or line < 1:
        return line
    if line <= len(origin):
        return origin[line - 1]
    # After the module (an Enum's constructor): as many lines past its end.
    return line - len(origin) + max(origin)


def _parse_output(output: str, ws: _Workspace) -> list[Diagnostic]:
    """Codon's errors, each located at its innermost frame in the scenario's files."""
    by_basename: dict[str, list[str]] = {}
    for rel in ws.files:
        by_basename.setdefault(Path(rel).name, []).append(rel)

    def locate(file: str, line: int) -> tuple[str, int, bool, str | None] | None:
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
        return (ws.files[rel], _original_line(ws.origins.get(rel), line), True, None)

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
        located = [(f, locate(f[0], f[1])) if f[0] else (f, None) for f in frames]
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


def _tree_digest(digest: Any, paths: list[Path]) -> None:
    for path in sorted(paths):
        digest.update(path.read_bytes())


@lru_cache(maxsize=4)
def _framework_digest(codon_path: Path) -> str:
    """The inputs that do not change while the process runs: the CARLA
    library, the model and the checker itself."""
    library = (codon_path / _LIBRARY).resolve()
    digest = hashlib.sha256(str(library).encode())
    _tree_digest(digest, list(library.rglob("*.codon")))
    _tree_digest(digest, list(model_dir().rglob("*.codon")))
    _tree_digest(digest, list(Path(__file__).parent.glob("*.py")))
    return digest.hexdigest()


def _cache_key(
    tc: Toolchain, codon_path: Path, sources: _Sources, driver_source: str
) -> str:
    digest = hashlib.sha256(str(tc.executable).encode())
    digest.update(_framework_digest(codon_path).encode())
    for name, module in sorted(sources.modules.items()):
        digest.update(name.encode())
        digest.update(module.text.encode())
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
    )


def _typecheck(
    name: str,
    roots: list[str],
    driver: Any,
    toolchain: Toolchain | None,
    timeout: float,
    *,
    what: str,
) -> TypeCheckResult:
    """Compile the modules *roots* with the driver *driver()* renders, cached."""
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
    rendered = driver()
    key = _cache_key(toolchain, codon_path, sources, rendered.source)
    if key not in _CACHE:
        result = _compile(name, sources, rendered, toolchain, codon_path, timeout)
        if result is None:  # timed out: no verdict worth keeping
            return _timed_out(name, timeout)
        _CACHE[key] = result
    return _CACHE[key]


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
        diagnostics = _parse_output(output, ws)
    seconds = time.monotonic() - started
    crashed = proc.returncode != 0 and (
        proc.returncode < 0
        or proc.returncode > 128
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
    # A name belongs to the longest module it starts with:
    # ``trajectory.model._wrap`` is of ``trajectory.model``, not of the
    # ``trajectory`` package.
    modules = {*CHECKED, module}
    out: list[str] = []
    for name in UNCALLED:
        owners = [m for m in modules if name.startswith(f"{m}.")]
        if owners and max(owners, key=len) == module:
            out.append(name.removeprefix(f"{module}."))
    return out


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
        for line, message in [*problems, *checks.problems]:
            sources.problems.append(Diagnostic(message, str(module.path), line))
        # A function the check never compiles cannot decorate one it does: a
        # run-time-only wrapper (trajectory.model's TrajectoryVertex.__init__)
        # is dropped, and the function it wraps is compiled as written.
        body, origin = transform_source_lines(
            redirected,
            dropped_decorators=[n for n in uncalled if "." not in n],
            init_classes=init_classes,
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
        for attr in undeclared_attributes(module.tree, declared):
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

    Returns:
        The result; ``result.ok`` is ``False`` when a module does not compile.
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
    if key not in _CACHE:
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
        _CACHE[key] = result
    return _CACHE[key]
