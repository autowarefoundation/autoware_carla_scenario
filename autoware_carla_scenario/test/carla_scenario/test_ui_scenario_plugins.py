"""The viewer sees the scenarios of installed scenario packages (#893).

A package registers its ``conf/`` through the ``autoware_carla_scenario.scenarios``
entry point, as the ``scenario`` CLI loads it: the viewer lists its scenarios
(``/api/scenarios``), matches them in globs, and expands their sweeps.
"""

from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import autoware_carla_scenario
import autoware_carla_scenario.registry as registry
from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import scanner, sweep_resolver

# Located from the package rather than ``scanner.BUILTIN_CONF_DIR``, so that a
# scanner without it fails the tests that use it, not their collection.
_BUILTIN_LEFT = (
    Path(autoware_carla_scenario.__file__).resolve().parent
    / "examples"
    / "conf"
    / "scenario"
    / "lane_change"
    / "left.yaml"
)


@pytest.fixture
def probe_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installed scenario package whose ``conf/`` has ``probe_pkg/left``.

    Its entry point is the only one installed; the registry's conf dirs and
    its loaded-once latch are restored after the test.
    """
    conf_dir = tmp_path / "probe_conf"
    (conf_dir / "scenario" / "probe_pkg").mkdir(parents=True)
    shutil.copy(_BUILTIN_LEFT, conf_dir / "scenario" / "probe_pkg" / "left.yaml")

    def register() -> None:
        registry.register_conf_dir(conf_dir)

    entry_point = importlib.metadata.EntryPoint(
        name="probe_pkg",
        value="probe_pkg:register",
        group=registry.SCENARIO_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(importlib.metadata.EntryPoint, "load", lambda self: register)

    def fake_entry_points(**kwargs: Any) -> list[importlib.metadata.EntryPoint]:
        if kwargs.get("group") == registry.SCENARIO_ENTRY_POINT_GROUP:
            return [entry_point]
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(registry, "_CONF_DIRS", list(registry.get_conf_dirs()))
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    return conf_dir.resolve()


def test_api_scenarios_lists_package_and_builtin_scenarios(
    probe_package: Path,
) -> None:
    names = TestClient(app_module.app).get("/api/scenarios").json()
    assert "probe_pkg/left" in names
    assert "lane_change/left" in names
    assert len(names) == len(set(names))
    assert probe_package in registry.get_conf_dirs()


def test_glob_matches_package_scenarios(probe_package: Path) -> None:
    assert app_module._resolve_scenario_names("probe_pkg/*") == ["probe_pkg/left"]
    assert "lane_change/left" in app_module._resolve_scenario_names("lane_change/*")


@pytest.mark.usefixtures("at_repo_root")
def test_resolve_sweep_composes_package_scenarios(probe_package: Path) -> None:
    jobs = sweep_resolver.resolve_sweep("probe_pkg/left", ["map=nishishinjuku"])
    assert jobs
    for job in jobs:
        assert job[0] == "scenario=probe_pkg/left"
        assert any(o.startswith("ego.spawn_lanelet_id=") for o in job)


def test_builtin_dir_listed_once_and_first(
    probe_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The built-in dir comes first even when ``examples.run`` registered it
    (or not), and a dir registered twice is walked once."""
    registry.register_conf_dir(scanner.BUILTIN_CONF_DIR)
    dirs = scanner.scenario_conf_dirs()
    assert dirs[0] == scanner.BUILTIN_CONF_DIR
    assert dirs.count(scanner.BUILTIN_CONF_DIR) == 1
    assert dirs.count(probe_package) == 1

    monkeypatch.setattr(registry, "_CONF_DIRS", [probe_package])
    assert scanner.scenario_conf_dirs() == [scanner.BUILTIN_CONF_DIR, probe_package]


def test_broken_package_leaves_builtin_scenarios(
    probe_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package whose entry point raises (e.g. its scenario module fails to
    import) is logged and skipped: the viewer still lists the built-ins."""

    def broken() -> None:
        raise ImportError("probe_pkg is broken")

    monkeypatch.setattr(importlib.metadata.EntryPoint, "load", lambda self: broken)
    names = TestClient(app_module.app).get("/api/scenarios").json()
    assert "lane_change/left" in names
    assert "probe_pkg/left" not in names
