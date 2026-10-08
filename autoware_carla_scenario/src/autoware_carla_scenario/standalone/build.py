"""Compile a scenario, with the framework's own runner, into a native binary.

The workspace Codon compiles (docs/standalone.md):

* the framework modules the run reaches (:func:`.collect.collect`) and the
  scenario's own, each rewritten by :func:`.transform.transform_for_runtime`,
  with generated package ``__init__`` files re-exporting only what is used;
* the modules the runtime provides instead (``codon/autoware_carla_scenario/``:
  the Lanelet2/pyxodr-backed coordinate code, the gRPC bridge server);
* stand-ins for the standard library Codon lacks (``codon/*.codon``);
* an empty stub for every other package the compiled modules import (numpy
  aside, which Codon has): a module-level import resolves, and a function the
  run calls that uses one is a build error naming it;
* typesafe_carla's Codon library, the CARLA client the code runs on.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..typecheck.toolchain import (
    Toolchain,
    codon_environment,
    codon_path_dir,
    find_codon,
)
from .collect import PACKAGE, Collected, collect
from .hierarchy import Hierarchy, ModuleContext
from .transform import INSERTED, original_line, transform_for_runtime

__all__ = [
    "BuildError",
    "Diagnostic",
    "REPLACED",
    "Workspace",
    "compile_workspace",
    "prepare_workspace",
    "runtime_dir",
]

#: Modules the runtime provides (``codon/<module path>.codon``).
REPLACED = (
    f"{PACKAGE}.coordinate.map_manager",
    f"{PACKAGE}.coordinate.projection",
    f"{PACKAGE}.coordinate.transform",
    f"{PACKAGE}.autoware_bridge.grpc_server",
    f"{PACKAGE}.server",
)

#: Top-level modules Codon provides or the runtime shims (not stubbed).
_PROVIDED = {
    "__future__", "abc", "bisect", "collections", "copy", "dataclasses", "enum",
    "functools", "heapq", "itertools", "json", "logging", "math", "numpy", "omegaconf",
    "operator",
    "os", "pathlib", "random", "re", "statistics", "string", "sys", "threading",
    "time", "typing", "typesafe_carla", "_acs_helpers", "_acs_fmt",
}  # fmt: skip


def runtime_dir() -> Path:
    return Path(__file__).resolve().parent / "codon"


@dataclass(frozen=True)
class Diagnostic:
    message: str
    path: str | None = None
    line: int | None = None
    trace: tuple[str, ...] = ()

    def format(self) -> str:
        where = f"{self.path}:{self.line}: " if self.path else ""
        text = f"{where}error: {self.message}"
        for frame in self.trace:
            text += f"\n    {frame}"
        return text


class BuildError(RuntimeError):
    def __init__(
        self, message: str, diagnostics: list[Diagnostic] | None = None
    ) -> None:
        self.diagnostics = list(diagnostics or [])
        super().__init__(
            "\n".join(
                [message]
                + ["  " + d.format().replace("\n", "\n  ") for d in self.diagnostics]
            )
        )


@dataclass
class Workspace:
    root: Path
    collected: Collected
    #: workspace-relative file -> the author's file
    files: dict[str, Path] = field(default_factory=dict)
    #: workspace-relative file -> its rewritten text
    texts: dict[str, str] = field(default_factory=dict)


def _external_imports(tree: ast.Module) -> set[str]:
    """Modules imported, and ``module.name`` for each name taken from one --
    by ``from module import name`` or as ``module.name`` after ``import module``."""
    out: set[str] = set()
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
                aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = []
            value: ast.AST = node
            while isinstance(value, ast.Attribute):
                dotted.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name) and value.id in aliases:
                out.add(f"{aliases[value.id]}.{dotted[-1]}")
    return out


def _stub(root: Path, module: str, names: set[str]) -> None:
    """An empty stand-in for a package Codon does not have."""
    path = root.joinpath(*module.split("."))
    path.mkdir(parents=True, exist_ok=True)
    init = path / "__init__.codon"
    lines = [
        f"# Stand-in for {module}, which a standalone binary does not include: using it",
        "# on the run path is a build error.",
    ]
    for name in sorted(names):
        lines.append(f"class {name}:\n    pass")
    init.write_text("\n".join(lines) + "\n")


def _unavailable(search: dict[str, Path] | None) -> Callable[[str], bool]:
    """Whether a binary lacks the module (it is stubbed)."""

    def unavailable(module: str) -> bool:
        top = module.split(".")[0]
        return not (
            top in _PROVIDED or top == PACKAGE or (search is not None and top in search)
        )

    return unavailable


def prepare_workspace(
    root: Path, roots: list[str], program: str, search: dict[str, Path] | None = None
) -> Workspace:
    """Fill *root* with everything :func:`compile_workspace` compiles."""
    collected = collect(roots, replaced=REPLACED, search=search)
    shutil.copytree(runtime_dir(), root, dirs_exist_ok=True)
    for entry in codon_path_dir().iterdir():
        (root / entry.name).symlink_to(entry.resolve())
    ws = Workspace(root, collected)
    externals: dict[str, set[str]] = {}
    sources = {
        module: path.read_text(encoding="utf-8")
        for module, path in sorted(collected.modules.items())
    }
    # Two passes: what a module gets from the others (.hierarchy) is read off
    # their rewritten sources.
    hierarchy = Hierarchy(sources)
    first = {
        m: transform_for_runtime(s, _unavailable(search), hierarchy.base_context(m))
        for m, s in sources.items()
    }
    for module, path in sorted(collected.modules.items()):
        source = sources[module]
        rel = Path(*module.split(".")).with_suffix(".py")
        text = transform_for_runtime(
            source, _unavailable(search), hierarchy.context(module, first)
        )
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        ws.files[rel.as_posix()] = path
        ws.texts[rel.as_posix()] = text
        for imported in _external_imports(ast.parse(source)):
            top = imported.split(".")[0]
            if top in _PROVIDED or top == PACKAGE or (search and top in search):
                continue
            externals.setdefault(
                imported.rsplit(".", 1)[0] if "." in imported else top, set()
            )
            if "." in imported:
                externals[imported.rsplit(".", 1)[0]].add(imported.rsplit(".", 1)[1])
    for module, names in externals.items():
        if not (root / Path(*module.split("."))).exists():
            _stub(root, module, {n for n in names if n[:1].isalpha()})
    for package in collected.exports:
        pkg_dir = root.joinpath(*package.split("."))
        pkg_dir.mkdir(parents=True, exist_ok=True)
        init = pkg_dir / "__init__.codon"
        provided = runtime_dir().joinpath(*package.split(".")) / "__init__.codon"
        if not provided.exists():
            init.write_text(collected.init_source(package))
    attributes = sorted(hierarchy.attribute_names())
    (root / "_acs_classes.codon").write_text(_class_extensions(attributes))
    (root / "_acs_main.py").write_text(
        transform_for_runtime(
            program, context=ModuleContext(imports=["import _acs_classes"])
        )
    )
    ws.texts["_acs_main.py"] = (root / "_acs_main.py").read_text()
    return ws


def _class_extensions(names: list[str]) -> str:
    """The class attributes read off a class held as a value (``_AcsClass``):
    T's accessor, on an instance nothing initialized -- it reads the class
    variable, the instance having assigned none."""
    lines = [
        "# Generated by standalone/build.py: class attributes read off _AcsClass.",
        "from _acs_helpers import _AcsClass",
        "",
        "@extend",
        "class _AcsClass:",
        "    def _acs_is_class(self) -> bool:",
        "        return True",
    ]
    for name in names:
        lines.append(f"    def _acs_get_{name}(self):")
        lines.append(f"        return T.__new__()._acs_get_{name}()")
    return "\n".join(lines) + "\n"


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LOCATION = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+)(?: \(\d+(?:-\d+)?\))?: ")
_ERROR = re.compile(r"error: ?(?P<message>.*)$")


def _diagnostics(output: str, ws: Workspace) -> list[Diagnostic]:
    by_name: dict[str, list[str]] = {}
    for rel in ws.texts:
        by_name.setdefault(Path(rel).name, []).append(rel)

    def locate(file: str, line: int) -> tuple[str, int]:
        candidates = by_name.get(Path(file).name, [])
        if len(candidates) == 1:
            rel = candidates[0]
            author = ws.files.get(rel)
            mapped = original_line(ws.texts[rel], line)
            return (str(author) if author else rel, mapped)
        return file, line

    out: list[Diagnostic] = []
    for raw in output.splitlines():
        text = _ANSI.sub("", raw).strip()
        nested = text.startswith(("├─", "╰─", "│"))
        body = text.lstrip("├╰─│ ").strip()
        loc = _LOCATION.match(body)
        found = _ERROR.search(body[loc.end() :] if loc else body)
        if not found:
            continue
        where = locate(loc["file"], int(loc["line"])) if loc else (None, None)
        if nested and out:
            last = out[-1]
            frame = (
                f"{found['message']} [{where[0]}:{where[1]}]"
                if loc
                else found["message"]
            )
            out[-1] = Diagnostic(
                last.message, last.path, last.line, (*last.trace, frame)
            )
        else:
            out.append(Diagnostic(found["message"], where[0], where[1]))
    return out


def compile_workspace(
    ws: Workspace,
    output: Path,
    *,
    toolchain: Toolchain | None = None,
    release: bool = True,
    rpath: str | None = None,
    timeout: float = 1800.0,
) -> float:
    """``codon build -exe`` the workspace's program into *output*; seconds taken."""
    import time

    tc = toolchain or find_codon()
    args = [str(tc.executable), "build", "-exe"]
    if release:
        args.append("-release")
    if rpath is not None:
        args.append(f"--linker-flags=-Wl,--disable-new-dtags,-rpath,{rpath}")
    args += ["-o", str(output), "_acs_main.py"]
    started = time.monotonic()
    proc = subprocess.run(  # noqa: S603 - a fixed compiler invocation
        args,
        cwd=ws.root,
        env=codon_environment(tc, ws.root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        output_text = proc.stdout + proc.stderr
        diagnostics = _diagnostics(output_text, ws) or [
            Diagnostic(
                _ANSI.sub("", output_text).strip().splitlines()[-1:][0]
                if output_text.strip()
                else f"codon exited with {proc.returncode}"
            )
        ]
        raise BuildError("the scenario does not build", diagnostics)
    return time.monotonic() - started


assert INSERTED  # re-exported for callers that read the rewritten files
_ = os
