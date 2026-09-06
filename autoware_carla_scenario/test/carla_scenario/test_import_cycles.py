"""Every subpackage must import on its own.

A cycle between two packages does not necessarily raise.  Whether it does
depends on which one an entry point reaches first, so a cycle can sit in the
tree indefinitely while ``import autoware_carla_scenario`` -- the order every
test happens to take -- keeps working, and surface only when something imports a
subpackage directly.  That is what happened to ``coordinate`` and ``utils``:
``coordinate/__init__`` imports ``coordinate.stop_line``, which imports
``utils.stop_line``, which imported ``coordinate.map_manager`` straight back.

Each import runs in its own interpreter, because the check is about import
*order* and a module already in ``sys.modules`` cannot demonstrate anything.
"""

from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import autoware_carla_scenario


def _subpackages() -> list[str]:
    """Return every importable subpackage, discovered rather than listed."""
    return sorted(
        f"{autoware_carla_scenario.__name__}.{info.name}"
        for info in pkgutil.iter_modules(autoware_carla_scenario.__path__)
        if info.ispkg
    )


@pytest.mark.parametrize("module", _subpackages())
def test_a_subpackage_imports_on_its_own(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"{module} does not import on its own:\n{result.stderr}"


def test_the_sweeper_map_loader_imports_on_its_own() -> None:
    """The path that first exposed the cycle, named so a regression is legible.

    ``sweeper/__init__`` -> ``bindings`` -> ``utils/__init__`` ->
    ``utils.stop_line`` -> ``coordinate`` -> ``coordinate.stop_line`` -> back
    into a half-built ``utils.stop_line``.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import autoware_carla_scenario.sweeper.map_loader"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class TestLayering:
    """`utils` sits underneath `coordinate`, and the imports have to say so.

    A cycle is only the symptom that shows up as an `ImportError`. The cause is
    a leaf package reaching up into the one that imports it, and that reads as
    fine right up until an entry point takes the other order -- so it is worth
    checking directly rather than waiting for the traceback.
    """

    @staticmethod
    def _imported_packages(module_path: Path) -> set[str]:
        """Return the sibling packages *module_path* imports, at any depth."""
        found: set[str] = set()
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.module:
                found.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[0] == "autoware_carla_scenario" and len(parts) > 1:
                        found.add(parts[1])
        return found

    def test_utils_does_not_import_coordinate(self) -> None:
        utils = Path(autoware_carla_scenario.__file__).parent / "utils"
        offenders = {
            path.name: sorted(self._imported_packages(path))
            for path in sorted(utils.glob("*.py"))
            if "coordinate" in self._imported_packages(path)
        }
        assert not offenders, (
            "utils must read what it is given, not fetch it from coordinate: "
            f"{offenders}"
        )
