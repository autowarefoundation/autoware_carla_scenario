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
(numpy, say) has no Codon model, and a scenario importing one fails the check
with a message saying so.

:func:`typecheck_library` compiles the framework's own modules instead (those
:mod:`.library` lists as checked), each with calls to everything public
appended (:mod:`.library_driver`), next to the same model.
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
from collections.abc import Callable, Collection, Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .driver import DRIVER_MODULE, render_driver, render_odd_driver
from .toolchain import (
    SUPPORTED_CODON_SERIES,
    Toolchain,
    ToolchainError,
    codon_environment,
    codon_path_dir,
    find_codon,
    is_supported_version,
)
from .library import CHECKED, REPLACED_MODELS, UNCALLED, package_modules
from .library_driver import (
    render_library_checks,
    render_library_driver,
    uncalled_nodes,
    workspace_module,
)
from .transform import (
    PRELUDE,
    class_declarations,
    redirect_imports,
    transform_source,
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


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostic:
    """One compile error, located in the file the author wrote."""

    message: str
    path: str | None = None
    line: int | None = None
    column: int | None = None
    #: Codon's "during the realization of ..." lines, outermost last.
    trace: tuple[str, ...] = ()

    def format(self) -> str:
        where = ""
        if self.path is not None:
            where = self.path
            if self.line is not None:
                where += f":{self.line}"
                if self.column is not None:
                    where += f":{self.column}"
            where += ": "
        text = f"{where}error: {self.message}"
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

    def format(self) -> str:
        if self.skipped is not None:
            return f"{self.scenario}: static check skipped: {self.skipped}"
        if self.ok:
            return f"{self.scenario}: static check passed ({self.seconds:.1f}s)"
        head = f"{self.scenario}: static check failed"
        if self.crashed:
            head += " (the Codon compiler crashed)"
        lines = [head + ":"]
        lines += ["  " + d.format().replace("\n", "\n  ") for d in self.diagnostics]
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


def _build_workspace(
    root: Path, sources: _Sources, driver: Any, codon_path: Path
) -> tuple[_Workspace, list[Diagnostic]]:
    ws = _base_workspace(root, driver, codon_path)
    problems: list[Diagnostic] = []
    declared = _declarations(sources)
    for name, module in sorted(sources.modules.items()):
        rel = Path(*name.split("."))
        rel = rel / "__init__.py" if module.is_package else rel.with_suffix(".py")
        for attr in undeclared_attributes(module.tree, declared):
            problems.append(Diagnostic(attr.message(), str(module.path), attr.lineno))
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(transform_source(module.text))
        ws.files[rel.as_posix()] = str(module.path)
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
    (root / DRIVER_MODULE).write_text(driver.source)
    return ws, problems


def _parse_output(output: str, ws: _Workspace) -> list[Diagnostic]:
    """Codon's errors, each located at its innermost frame in the scenario's files."""
    by_basename: dict[str, list[str]] = {}
    for rel in ws.files:
        by_basename.setdefault(Path(rel).name, []).append(rel)

    def locate(file: str, line: int) -> tuple[str, int, bool] | None:
        """(path to show, line, is the scenario's own file)."""
        if file == DRIVER_MODULE:
            key = ws.driver.config_lines.get(line)
            return (key if key else ws.driver_label, 0 if key else line, False)
        candidates = by_basename.get(Path(file).name, [])
        if len(candidates) != 1:
            return None
        rel = candidates[0]
        if rel in ws.appended and line > ws.appended[rel][0]:
            # A call the library check appended: name the call it makes.
            call = ws.appended[rel][1].get(line - ws.appended[rel][0])
            return (f"<library check: {call or rel}>", 0, False)
        return (ws.files[rel], line - _PRELUDE_LINES, True)

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
        for f, loc in located[1:]:
            if f[0] == DRIVER_MODULE:
                continue  # the generated program: nothing the author wrote
            if loc is not None:
                where = f"{Path(loc[0]).name}:{loc[1]}" if loc[1] else loc[0]
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
            return TypeCheckResult(name, ok=False, diagnostics=problems)
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


def _uncalled(module: str) -> list[str]:
    """What :data:`.library.UNCALLED` lists of *module*, relative to it."""
    prefix = f"{module}."
    return [name.removeprefix(prefix) for name in UNCALLED if name.startswith(prefix)]


def _library_files(
    names: list[str], sources: _Sources, every: dict[str, Path]
) -> list[_LibraryFile]:
    """Each module of *sources* rewritten for the library workspace.

    A problem found on the way (an import that cannot be redirected, a
    parameter the check cannot call with) is added to ``sources.problems``.
    """
    checked = set(names)

    def target(module: str) -> str | None:
        return workspace_module(module) if module in checked else _model_module(module)

    files: list[_LibraryFile] = []
    for name in names:
        module = sources.modules[name]
        redirected, problems = redirect_imports(
            module.text, name, module.is_package, target, every.__contains__
        )
        checks = render_library_checks(module.tree, _uncalled(name))
        for line, message in [*problems, *checks.problems]:
            sources.problems.append(Diagnostic(message, str(module.path), line))
        body = transform_source(redirected)
        rel = workspace_module(name).replace(".", "/") + ".codon"
        files.append(
            _LibraryFile(
                name, rel, body + checks.source, body.count("\n"), checks.labels
            )
        )
    return files


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

    Each module is compiled from its source, every public function and
    method called with a value of each parameter's annotated type
    (:mod:`.library_driver`), so its body is checked.  A framework module
    one of them imports that is not checked itself stands in through the
    model, as it does for a scenario.  All of *modules* are compiled
    together, in one Codon build.

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
        path = every[name]
        text = path.read_text(encoding="utf-8")
        sources.modules[name] = _Module(
            path, path.name == "__init__.py", text, ast.parse(text)
        )
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
