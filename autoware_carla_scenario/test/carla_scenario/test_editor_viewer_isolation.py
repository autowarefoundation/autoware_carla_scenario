"""The Scenario Editor must not reach into the Scenario Result Viewer.

Integrating the two is a later phase. Until then the editor is a separate
application with its own entry point, and these tests are what make "we did not
touch the viewer" a checkable claim rather than a promise.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# tomllib landed in 3.11; the workspace is capped at 3.10 by the CARLA wheel.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.11+ has it in the stdlib
    import tomli as tomllib

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _source_of(module_name: str) -> str:
    """Return the source text of an imported module."""
    module = sys.modules.get(module_name) or importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    assert path is not None, f"{module_name} has no source file"
    return Path(path).read_text()


@pytest.fixture(scope="module")
def scripts() -> dict[str, str]:
    """Return the package's console scripts."""
    data = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text())
    return data["project"]["scripts"]


class TestEntryPoints:
    def test_the_editor_has_its_own_entry_point(self, scripts: dict[str, str]) -> None:
        assert scripts["scenario-editor"] == "autoware_carla_scenario.editor:main"

    def test_the_viewer_entry_point_is_unchanged(self, scripts: dict[str, str]) -> None:
        assert scripts["viewer"] == "autoware_carla_scenario.ui:main"

    def test_they_are_different_applications(self) -> None:
        from autoware_carla_scenario import editor, ui

        assert editor.main is not ui.main


class TestNoCoupling:
    def test_the_editor_does_not_import_the_viewer(self) -> None:
        """A shared import today is a shared regression tomorrow."""
        import autoware_carla_scenario.editor.app  # noqa: F401
        import autoware_carla_scenario.editor.routes  # noqa: F401
        import autoware_carla_scenario.editor.service  # noqa: F401

        for module in (
            "autoware_carla_scenario.editor.app",
            "autoware_carla_scenario.editor.routes",
            "autoware_carla_scenario.editor.service",
            "autoware_carla_scenario.editor.map_preview",
        ):
            source = _source_of(module)
            assert "from ..ui" not in source, module
            assert "autoware_carla_scenario.ui" not in source, module

    def test_the_viewer_does_not_import_the_editor(self) -> None:
        ui_dir = _PACKAGE_ROOT / "src" / "autoware_carla_scenario" / "ui"
        for path in ui_dir.rglob("*.py"):
            source = path.read_text()
            assert "editor" not in source, path

    def test_the_editor_is_not_mounted_on_the_viewer(self) -> None:
        from autoware_carla_scenario.ui.app import app as viewer_app

        paths = {getattr(route, "path", "") for route in viewer_app.routes}
        assert not any(path.startswith("/draft") for path in paths)
        assert "/new" not in paths

    def test_the_viewer_still_serves_its_own_pages(self, tmp_path: Path) -> None:
        """The regression this whole file exists to catch."""
        from autoware_carla_scenario.ui import app as viewer_module

        original = viewer_module.BASE_PATH
        viewer_module.BASE_PATH = tmp_path
        try:
            client = TestClient(viewer_module.app)
            assert client.get("/").status_code == 200
            assert client.get("/api/scenarios").status_code == 200
        finally:
            viewer_module.BASE_PATH = original

    def test_the_editor_serves_its_own_pages(self, tmp_path: Path) -> None:
        from autoware_carla_scenario.editor.app import create_app

        client = TestClient(create_app(draft_dir=tmp_path / "drafts"))
        assert client.get("/").status_code == 200


class TestSharedRuntime:
    """Reuse flows one way: the editor builds the framework's own primitives."""

    def test_the_editor_does_not_ship_a_second_constraint_engine(self) -> None:
        from autoware_carla_scenario.editor import map_preview

        source = Path(map_preview.__file__).read_text()
        assert "from ..sweeper.constraints import" in source

    def test_authoring_stays_importable_without_carla(self) -> None:
        """The editor process has no simulator; keep its imports light."""
        for module in (
            "autoware_carla_scenario.authoring.models",
            "autoware_carla_scenario.authoring.registry",
            "autoware_carla_scenario.authoring.compiler",
            "autoware_carla_scenario.authoring.validator",
            "autoware_carla_scenario.authoring.builders",
            "autoware_carla_scenario.authoring.hydra_config",
        ):
            assert "\nimport carla" not in _source_of(module), module
