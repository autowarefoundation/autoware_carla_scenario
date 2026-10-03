"""Compile a scenario with Codon before it runs (docs/typecheck.md).

:func:`typecheck_scenario` copies the scenario's own modules into a scratch
workspace, next to the Codon model of the framework (``codon/``), rewrites
them for Codon (:mod:`.transform`), adds a driver program that builds the
scenario from its config and calls ``setup()`` (:mod:`.driver`), and compiles
the lot with ``codon build -llvm``.  Nothing compiled is ever run: the
compile *is* the check, and a scenario that does not compile is refused before
the runner touches CARLA.

What is checked is everything ``setup()`` and ``is_done()`` reach, as far as
the model goes: the framework's public API, the part of the CARLA API in
``codon/carla``, and the scenario package's own modules.  A module outside
those (numpy, say) has no Codon model, and a scenario importing one fails the
check with a message saying so.
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
from functools import lru_cache
from pathlib import Path
from typing import Any

from .driver import DRIVER_MODULE, render_driver
from .toolchain import Toolchain, ToolchainError, codon_environment, find_codon
from .transform import (
    PRELUDE,
    class_declarations,
    transform_source,
    undeclared_attributes,
)

__all__ = [
    "Diagnostic",
    "ScenarioTypeError",
    "TypeCheckResult",
    "available_toolchain",
    "model_dir",
    "typecheck_scenario",
]

#: Seconds a single compile may take.
DEFAULT_TIMEOUT_SECONDS = 600.0

_PACKAGE = "autoware_carla_scenario"
#: Modules the model directory provides besides the framework's own.
_SHIMS = frozenset({"carla", "__future__", "logging", "dataclasses", "typing"})

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


def available_toolchain() -> Toolchain | None:
    """The Codon the checker runs (:func:`.toolchain.find_codon`), or ``None``."""
    try:
        return find_codon()
    except ToolchainError:
        return None


@lru_cache(maxsize=8)
def _codon_stdlib(tc: Toolchain) -> frozenset[str]:
    """Top-level module names of Codon's standard library."""
    for candidate in (tc.codon_dir / "lib" / "codon" / "stdlib",):
        if candidate.is_dir():
            return frozenset(p.name.split(".")[0] for p in candidate.iterdir())
    return frozenset(
        {"math", "random", "itertools", "collections", "functools", "sys", "os"}
    )


@lru_cache(maxsize=1)
def _model_modules() -> frozenset[str]:
    root = model_dir() / _PACKAGE
    names = {_PACKAGE}
    names.update(
        f"{_PACKAGE}.{p.stem}" for p in root.glob("*.codon") if p.stem != "__init__"
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
    tree: ast.Module, module: str, is_package: bool
) -> list[tuple[str, list[str], int]]:
    """(absolute module, imported names, line) for every import in *tree*."""
    package = module if is_package else module.rpartition(".")[0]
    out: list[tuple[str, list[str], int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(alias.name, [], node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            out.append((name, [alias.name for alias in node.names], node.lineno))
    return out


@dataclass
class _Sources:
    #: module name -> (source file, is a package __init__)
    modules: dict[str, tuple[Path, bool]] = field(default_factory=dict)
    problems: list[Diagnostic] = field(default_factory=list)


def _collect(roots: list[str], tc: Toolchain) -> _Sources:
    stdlib = _codon_stdlib(tc)
    modeled = _model_modules()
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
        out.modules[name] = (path, is_package)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported, names, lineno in _imports(tree, name, is_package):
            top = imported.split(".")[0]
            if imported in modeled or top in _SHIMS:
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
    #: workspace-relative file -> (original path, line offset)
    files: dict[str, tuple[str, int]] = field(default_factory=dict)


def _build_workspace(
    root: Path, sources: _Sources, driver: Any
) -> tuple[_Workspace, list[Diagnostic]]:
    shutil.copytree(model_dir(), root, dirs_exist_ok=True)
    ws = _Workspace(root, driver)
    problems: list[Diagnostic] = []
    declared = {cls: set(names) for cls, names in _model_declarations().items()}
    trees: dict[str, ast.Module] = {}
    for name, (path, _is_package) in sources.modules.items():
        trees[name] = ast.parse(path.read_text(encoding="utf-8"))
    for tree in trees.values():
        for cls, (_bases, own) in class_declarations(tree).items():
            declared.setdefault(cls, set()).update(own)
    for name, (path, is_package) in sorted(sources.modules.items()):
        rel = Path(*name.split("."))
        rel = rel / "__init__.py" if is_package else rel.with_suffix(".py")
        for attr in undeclared_attributes(trees[name], declared):
            problems.append(Diagnostic(attr.message(), str(path), attr.lineno))
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(transform_source(path.read_text(encoding="utf-8")).source)
        ws.files[rel.as_posix()] = (str(path), PRELUDE.count("\n"))
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
            return (key if key else _DRIVER_LABEL, 0 if key else line, False)
        candidates = by_basename.get(Path(file).name, [])
        if len(candidates) == 1:
            original, offset = ws.files[candidates[0]]
            return (original, line - offset, True)
        return None

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
        primary = next((loc for f, loc in located if loc and loc[2]), None)
        primary = primary or next((loc for f, loc in located if loc), None)
        column = None
        for f, loc in located:
            if loc is not None and loc == primary:
                column = f[2]
                break
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


def _cache_key(tc: Toolchain, sources: _Sources, driver_source: str) -> str:
    digest = hashlib.sha256(str(tc.executable).encode())
    for path in sorted(model_dir().rglob("*.codon")):
        digest.update(path.read_bytes())
    for name, (path, _) in sorted(sources.modules.items()):
        digest.update(name.encode())
        digest.update(path.read_bytes())
    digest.update(driver_source.encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
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
            it (:func:`.toolchain.find_codon`) by default.
        timeout: Seconds the compile may take.

    Returns:
        The result; ``result.ok`` is ``False`` when the scenario must not run.
        A result with ``skipped`` set made no check at all (no Codon, or a
        scenario class without a source file).
    """
    name = f"{scenario_cls.__module__}.{scenario_cls.__qualname__}"
    if toolchain is None:
        try:
            toolchain = find_codon()
        except ToolchainError as exc:
            return TypeCheckResult(name, ok=True, skipped=f"no Codon compiler: {exc}")
    roots = sorted({scenario_cls.__module__, config_cls.__module__})
    if any(_module_file(root) is None for root in roots):
        return TypeCheckResult(
            name, ok=True, skipped="the scenario has no source file to compile"
        )
    sources = _collect(roots, toolchain)
    driver = render_driver(scenario_cls, config_cls, dict(scenario_dict or {}))
    key = _cache_key(toolchain, sources, driver.source)
    if key in _CACHE:
        return _CACHE[key]
    if sources.problems:
        result = TypeCheckResult(name, ok=False, diagnostics=list(sources.problems))
        _CACHE[key] = result
        return result

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="acs-typecheck-") as tmp:
        ws, problems = _build_workspace(Path(tmp), sources, driver)
        if problems:
            result = TypeCheckResult(name, ok=False, diagnostics=problems)
            _CACHE[key] = result
            return result
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
            return TypeCheckResult(
                name,
                ok=False,
                crashed=True,
                diagnostics=[
                    Diagnostic(f"Codon did not finish compiling within {timeout:.0f}s")
                ],
                seconds=time.monotonic() - started,
            )
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
    result = TypeCheckResult(
        name,
        ok=proc.returncode == 0,
        diagnostics=diagnostics,
        crashed=crashed,
        output=output,
        seconds=seconds,
    )
    _CACHE[key] = result
    return result
