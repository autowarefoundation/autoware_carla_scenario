"""Which of the framework's modules a standalone binary compiles.

Codon runs a package's ``__init__`` whenever one of its modules is imported,
and the framework's ``__init__`` files re-export everything -- the editor,
Hydra, the sweeper. A binary compiles only what its run reaches, so
:func:`collect` follows imports from the scenario (and the runner) and, for
each package on the way, writes an ``__init__`` re-exporting only the names
the compiled modules take from it, each resolved through the real
``__init__``'s re-exports (``from .x import y``, or the top-level package's
``_LAZY_IMPORTS`` table) to the module that defines it.

A module named in *replaced* is not compiled from its source: the runtime
provides it (``standalone/codon/``), because it stands at a boundary the
binary crosses differently (Lanelet2 and pyxodr, gRPC).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Collected", "collect", "package_root"]

PACKAGE = "autoware_carla_scenario"


def package_root() -> Path:
    """The directory holding the ``autoware_carla_scenario`` package."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Collected:
    """The modules to compile, and the ``__init__`` each package gets."""

    #: module name -> its source file (a package's own __init__ is not here)
    modules: dict[str, Path] = field(default_factory=dict)
    #: package name -> {exported name: (defining module, attribute)}
    exports: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    #: modules the runtime provides instead
    replaced: set[str] = field(default_factory=set)
    #: package name -> submodules imported from it by name (`from pkg import mod`)
    submodules: dict[str, set[str]] = field(default_factory=dict)

    def init_source(self, package: str) -> str:
        """The generated ``__init__`` of *package*."""
        lines = []
        for name, (module, attr) in sorted(self.exports.get(package, {}).items()):
            alias = "" if attr == name else f" as {name}"
            # Relative within the package: Codon does not resolve the package's
            # own absolute name while its __init__ runs.
            source = (
                "." + module[len(package) + 1 :]
                if module.startswith(package + ".")
                else module
            )
            lines.append(f"from {source} import {attr}{alias}")
        for name in sorted(self.submodules.get(package, ())):
            lines.append(f"from . import {name}")
        return "\n".join(lines) + "\n"


class _Resolver:
    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots  # top-level package -> directory containing it
        self._trees: dict[str, tuple[ast.Module, bool] | None] = {}

    def path(self, module: str) -> tuple[Path, bool] | None:
        base = self.roots.get(module.split(".")[0])
        if base is None:
            return None
        p = base.joinpath(*module.split("."))
        if (p / "__init__.py").exists():
            return p / "__init__.py", True
        if p.with_suffix(".py").exists():
            return p.with_suffix(".py"), False
        return None

    def tree(self, module: str) -> tuple[ast.Module, bool] | None:
        if module not in self._trees:
            found = self.path(module)
            self._trees[module] = (
                None if found is None else (ast.parse(found[0].read_text()), found[1])
            )
        return self._trees[module]

    def defines(self, module: str, name: str) -> bool:
        found = self.tree(module)
        if found is None:
            return False
        for node in found[0].body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    return True
            elif isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    return True
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == name:
                    return True
        return False

    def reexport(self, package: str, name: str) -> tuple[str, str] | None:
        """Where *package*'s ``__init__`` takes *name* from: (module, attribute)."""
        found = self.tree(package)
        if found is None:
            return None
        tree, _ = found
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        return absolute(package, True, node), alias.name
            if (
                isinstance(node, ast.AnnAssign | ast.Assign)
                and isinstance(node.value, ast.Dict)
                and "_LAZY_IMPORTS" in ast.unparse(node)
            ):
                for key, value in zip(node.value.keys, node.value.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == name
                        and isinstance(value, ast.Tuple)
                    ):
                        rel, attr = (ast.literal_eval(e) for e in value.elts)
                        level = len(rel) - len(rel.lstrip("."))
                        parts = package.split(".")[
                            : len(package.split(".")) - level + 1
                        ]
                        return ".".join([*parts, rel.lstrip(".")]), attr
        return None

    def resolve(self, module: str, name: str) -> tuple[str, str] | None:
        """The module that defines what ``from module import name`` imports."""
        seen: set[tuple[str, str]] = set()
        while (module, name) not in seen:
            seen.add((module, name))
            if self.path(f"{module}.{name}") is not None:
                return None  # a submodule, not a name
            found = self.tree(module)
            if found is None:
                return None
            _, is_package = found
            if not is_package or self.defines(module, name):
                return module, name
            nxt = self.reexport(module, name)
            if nxt is None:
                return None
            module, name = nxt
        return None


def absolute(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    """The absolute module an ``ImportFrom`` in *module* names."""
    if not node.level:
        return node.module or ""
    base = (module if is_package else module.rpartition(".")[0]).split(".")
    base = base[: len(base) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


def _imports(
    tree: ast.Module, module: str, is_package: bool
) -> Iterable[tuple[str, list[str]]]:
    """(module, names) of every import in *tree*, function-level ones included."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, []
        elif isinstance(node, ast.ImportFrom):
            yield absolute(module, is_package, node), [a.name for a in node.names]


def collect(
    roots: Iterable[str],
    *,
    replaced: Iterable[str] = (),
    search: dict[str, Path] | None = None,
) -> Collected:
    """The modules *roots* (module names) need, with generated package inits.

    Args:
        roots: The modules to start from (the scenario's, the runner's).
        replaced: Modules the runtime provides; not followed into.
        search: Top-level package -> the directory holding it (the framework's
            by default; a scenario package adds its own).
    """
    roots_map = {PACKAGE: package_root(), **(search or {})}
    resolver = _Resolver(roots_map)
    replaced_set = set(replaced)
    out = Collected(replaced=replaced_set)
    queue = list(roots)

    def export(package: str, name: str, target: tuple[str, str]) -> None:
        out.exports.setdefault(package, {})[name] = target
        queue.append(target[0])

    def note_packages(module: str) -> None:
        parts = module.split(".")
        for i in range(1, len(parts)):
            out.exports.setdefault(".".join(parts[:i]), {})

    while queue:
        module = queue.pop()
        if module in out.modules or module in replaced_set:
            continue
        found = resolver.tree(module)
        if found is None:
            continue  # not ours: the standard library, typesafe_carla, a shim
        tree, is_package = found
        note_packages(module)
        if is_package:
            out.exports.setdefault(module, {})
            continue  # its __init__ is generated
        path = resolver.path(module)
        assert path is not None
        out.modules[module] = path[0]
        for imported, names in _imports(tree, module, is_package):
            if imported.split(".")[0] not in roots_map:
                continue
            target = resolver.path(imported)
            if target is None and imported not in replaced_set:
                continue
            if not names:
                queue.append(imported)
                continue
            if imported in replaced_set or (target is not None and not target[1]):
                queue.append(imported)  # a module: its names come with it
                continue
            for name in names:
                if resolver.path(f"{imported}.{name}") is not None:
                    queue.append(f"{imported}.{name}")
                    note_packages(f"{imported}.{name}")
                    out.submodules.setdefault(imported, set()).add(name)
                    continue
                resolved = resolver.resolve(imported, name)
                if resolved is not None:
                    export(imported, name, resolved)
                    # every package between the importer and the definer
                    # re-exports it, as the real ones do
                    note_packages(resolved[0])
    return out
