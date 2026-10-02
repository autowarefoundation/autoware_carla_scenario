"""A workspace member this package imports has to be declared, and found.

``uv sync`` installs every member of a workspace, so a sibling lands on
``sys.path`` whether or not anything asked for it.  The omission only surfaces
once the package is resolved on its own -- which is what
``authoring.wheelhouse`` does: it resolves the lockfile into wheels, an
undeclared dependency is not in the resolution, its wheel is never downloaded,
and the scenario runner started from the resulting venv dies on a
``ModuleNotFoundError``.  Members are read from the workspace root rather than
listed here, so ones added later are covered too.

The converter, ``autoware_lanelet2_to_opendrive``, was such a dependency once.
It is not any more: Lanelet2 and OpenDRIVE are loaded as two independent
coordinate systems, and nothing the package ships may import it again.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

# tomllib landed in 3.11, and 3.10 is still the floor of the supported range.
# Branching on sys.version_info rather than catching ImportError keeps mypy from
# reading the fallback as a redefinition when it checks against 3.11+.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _PACKAGE_ROOT.parent
_SRC = _PACKAGE_ROOT / "src"


def _canonical(name: str) -> str:
    """Normalise a distribution name the way PEP 503 does."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _manifest(directory: Path) -> dict:
    return tomllib.loads((directory / "pyproject.toml").read_text())


#: The converter's top-level module and distribution.
_CONVERTER_MODULE = "autoware_lanelet2_to_opendrive"
_CONVERTER_DISTRIBUTION = "autoware-lanelet2-to-opendrive"


def _siblings() -> dict[str, str]:
    """Map every other workspace member's top-level module to its distribution."""
    members = _manifest(_WORKSPACE_ROOT)["tool"]["uv"]["workspace"]["members"]
    modules: dict[str, str] = {}
    for member in members:
        root = _WORKSPACE_ROOT / member
        if root == _PACKAGE_ROOT:
            continue
        distribution = _manifest(root)["project"]["name"]
        source = root / "src"
        if not source.is_dir():  # pragma: no cover - every member is src-layout
            continue
        for child in source.iterdir():
            if child.is_dir() and child.name != "__pycache__":
                modules[child.name] = distribution
    return modules


def _sources() -> dict:
    """The `[tool.uv.sources]` that apply here: the root's, overridden by ours."""

    def table(directory: Path) -> dict:
        return _manifest(directory).get("tool", {}).get("uv", {}).get("sources", {})

    return {**table(_WORKSPACE_ROOT), **table(_PACKAGE_ROOT)}


def _declared() -> set[str]:
    """Every distribution the package requires, canonicalised."""
    project = _manifest(_PACKAGE_ROOT)["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    # A requirement starts with the name and ends at the first version
    # specifier, extras bracket, marker or space.
    return {
        _canonical(re.split(r"[\[<>=!~;\s]", text, maxsplit=1)[0])
        for text in requirements
    }


def _imported_top_level_modules() -> dict[str, set[Path]]:
    """Map each top-level module the shipped sources import to where from."""
    found: dict[str, set[Path]] = {}

    def record(name: str, path: Path) -> None:
        found.setdefault(name.split(".")[0], set()).add(path.relative_to(_SRC))

    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    record(alias.name, path)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                record(node.module, path)
    return found


@pytest.mark.parametrize("module", sorted(_siblings()))
def test_an_imported_first_party_project_is_a_declared_dependency(
    module: str,
) -> None:
    importers = _imported_top_level_modules().get(module)
    if not importers:
        pytest.skip(f"{module} is not imported by this package")
    distribution = _siblings()[module]
    assert _canonical(distribution) in _declared(), (
        f"{module} is imported by {sorted(map(str, importers))} but "
        f"{distribution} is not in this package's dependencies -- a wheelhouse "
        "built from the lockfile would leave its wheel out."
    )


@pytest.mark.parametrize("module", sorted(_siblings()))
def test_a_declared_sibling_resolves_from_the_workspace(module: str) -> None:
    """Nothing publishes these to an index, so uv has to be told where to look."""
    distribution = _siblings()[module]
    if _canonical(distribution) not in _declared():
        pytest.skip(f"{distribution} is not a dependency of this package")
    matching = {
        name: value
        for name, value in _sources().items()
        if _canonical(name) == _canonical(distribution)
    }
    assert matching, f"{distribution} has no [tool.uv.sources] entry"
    assert all(
        value.get("workspace") is True for value in matching.values()
    ), f"{distribution} must resolve with {{ workspace = true }}, got {matching}"


def test_the_converter_is_not_imported() -> None:
    """Lanelet2 and OpenDRIVE are two frames of their own; no mapping is shared."""
    importers = _imported_top_level_modules().get(_CONVERTER_MODULE)
    assert (
        not importers
    ), f"{_CONVERTER_MODULE} is imported by {sorted(map(str, importers))}"


def test_the_converter_is_not_a_dependency() -> None:
    assert _canonical(_CONVERTER_DISTRIBUTION) not in _declared()
    assert not any(
        _canonical(name) == _canonical(_CONVERTER_DISTRIBUTION) for name in _sources()
    ), f"{_CONVERTER_DISTRIBUTION} still has a [tool.uv.sources] entry"
