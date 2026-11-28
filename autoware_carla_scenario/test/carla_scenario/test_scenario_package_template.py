"""The scenario package template's packaging registers what it ships (#2624).

``examples/scenario_package_template`` is never installed by the suite, so a
broken entry point in its ``pyproject.toml`` -- or a ``register()`` that no
longer registers its scenario or conf dir -- would pass CI while the
template's README "Try it" breaks.  This reads the entry point the way the
``scenario`` CLI does and calls it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

# tomllib landed in 3.11; tomli is the framework's own dependency on 3.10.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

from autoware_carla_scenario import registry

_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "examples" / "scenario_package_template"
)
_PACKAGE = "my_scenario_package"


def _package_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == _PACKAGE or name.startswith(f"{_PACKAGE}.")
    }


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Copies of the scenario registry and the conf dirs, restored afterwards,
    and the template importable from its ``src`` for the test only."""
    for name in ("_SCENARIO_REGISTRY", "_SCENARIO_CLASSES", "_SCENARIO_DOCUMENTS"):
        monkeypatch.setattr(registry, name, dict(getattr(registry, name)))
    monkeypatch.setattr(registry, "_CONF_DIRS", list(registry._CONF_DIRS))
    monkeypatch.setattr(registry, "_CONFIG_INDEX", dict(registry._CONFIG_INDEX))
    monkeypatch.setattr(registry, "_INDEXED_DIRS", list(registry._INDEXED_DIRS))
    monkeypatch.setattr(registry, "_DIR_CONFIG_PATHS", dict(registry._DIR_CONFIG_PATHS))
    monkeypatch.syspath_prepend(str(_TEMPLATE / "src"))
    saved = _package_modules()
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in _package_modules():
            del sys.modules[name]
        sys.modules.update(saved)  # type: ignore[arg-type]


@pytest.mark.usefixtures("isolated_registry")
def test_the_template_entry_point_registers_its_scenario_and_conf_dir() -> None:
    with (_TEMPLATE / "pyproject.toml").open("rb") as manifest:
        project = tomllib.load(manifest)["project"]
    entry_points = project["entry-points"][registry.SCENARIO_ENTRY_POINT_GROUP]
    assert entry_points, "the template advertises no scenario entry point"

    for name, value in entry_points.items():
        EntryPoint(name, value, registry.SCENARIO_ENTRY_POINT_GROUP).load()()

    assert "reach_goal" in registry.get_scenario_registry()
    classes = registry.get_scenario_classes("reach_goal")
    assert classes is not None
    assert [cls.__name__ for cls in classes] == ["ReachGoalScenario", "ReachGoalConfig"]
    conf_dir = (_TEMPLATE / "src" / _PACKAGE / "conf").resolve()
    assert conf_dir in registry.get_conf_dirs()
    assert (conf_dir / "scenario" / "reach_goal" / "default.yaml").is_file()
