"""Unit tests for the smoke test the packing action runs inside an image.

The script only ever runs inside a freshly built image, so a check in it that
cannot fail goes unnoticed until a broken image ships.  These tests run it
against fake registries: the config-directory check must reject a package
whose wheel lost its ``conf/`` even though the framework's own config
directory, always registered and always full of YAML, is on the search path
(#2055).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "actions"
    / "pack-scenario-image"
    / "smoke-test.py"
)


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    """Import the script by path -- it ships with the action, not the package."""
    spec = importlib.util.spec_from_file_location("smoke_test_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "a: 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def framework_conf(tmp_path: Path) -> Path:
    """A stand-in for the framework's own conf dir, full of scenario YAML."""
    conf = tmp_path / "framework" / "conf"
    _write(conf / "scenario" / "builtin" / "default.yaml")
    _write(conf / "map" / "nishishinjuku.yaml")
    return conf


def _plugin_conf(tmp_path: Path) -> Path:
    conf = tmp_path / "my_scenario_package" / "conf"
    _write(conf / "scenario" / "reach_goal" / "default.yaml")
    return conf


def test_framework_conf_dir_alone_fails(
    smoke: ModuleType, framework_conf: Path
) -> None:
    problem = smoke.check_conf_dirs([framework_conf], framework_conf)
    assert problem is not None
    assert "framework's own" in problem


def test_missing_plugin_conf_dir_fails(
    smoke: ModuleType, framework_conf: Path, tmp_path: Path
) -> None:
    # What a generated package registers when its wheel dropped conf/.
    missing = tmp_path / "my_scenario_package" / "conf"
    problem = smoke.check_conf_dirs([framework_conf, missing], framework_conf)
    assert problem is not None
    assert str(missing) in problem


def test_plugin_conf_dir_without_scenario_group_fails(
    smoke: ModuleType, framework_conf: Path, tmp_path: Path
) -> None:
    conf = tmp_path / "my_scenario_package" / "conf"
    _write(conf / "map" / "custom.yaml")
    (conf / "scenario").mkdir()
    assert smoke.check_conf_dirs([framework_conf, conf], framework_conf) is not None


def test_plugin_conf_dir_with_scenario_yaml_passes(
    smoke: ModuleType, framework_conf: Path, tmp_path: Path
) -> None:
    conf = _plugin_conf(tmp_path)
    assert smoke.check_conf_dirs([framework_conf, conf], framework_conf) is None


def test_one_good_plugin_conf_dir_is_enough(
    smoke: ModuleType, framework_conf: Path, tmp_path: Path
) -> None:
    conf = _plugin_conf(tmp_path)
    missing = tmp_path / "other_package" / "conf"
    dirs = [framework_conf, missing, conf]
    assert smoke.check_conf_dirs(dirs, framework_conf) is None


def test_framework_conf_dir_is_matched_after_resolving(
    smoke: ModuleType, framework_conf: Path
) -> None:
    unresolved = framework_conf / ".." / "conf"
    assert smoke.check_conf_dirs([unresolved], framework_conf) is not None


# ---------------------------------------------------------------------------
# main() against fakes: no CARLA client, no installed scenario package.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_image(
    monkeypatch: pytest.MonkeyPatch, framework_conf: Path
) -> SimpleNamespace:
    """Fake the client and the registry; return the knobs a test turns."""
    state = SimpleNamespace(
        conf_dirs=[framework_conf],
        scenarios={"my_scenario_package/reach_goal": object()},
    )
    monkeypatch.setenv("TYPESAFE_CARLA_PYCARLA_BUILD", "0")

    carla = ModuleType("typesafe_carla.carla")
    carla.Location = lambda *args: args  # type: ignore[attr-defined]
    carla.Transform = lambda location: location  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "typesafe_carla.carla", carla)
    import typesafe_carla
    from typesafe_carla import paths

    monkeypatch.setattr(typesafe_carla, "carla", carla, raising=False)
    monkeypatch.setattr(paths, "native_info", lambda: {"carla_git_ref": "fake"})

    registry = ModuleType("autoware_carla_scenario.registry")
    registry._FRAMEWORK_CONF_DIR = framework_conf  # type: ignore[attr-defined]
    registry.load_scenario_plugins = lambda: None  # type: ignore[attr-defined]
    registry.get_conf_dirs = lambda: list(state.conf_dirs)  # type: ignore[attr-defined]
    registry.get_scenario_registry = lambda: dict(state.scenarios)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoware_carla_scenario.registry", registry)

    from hydra.core import plugins

    plugin = type("AutowareScenarioSearchPathPlugin", (), {})

    class _Plugins:
        @staticmethod
        def instance() -> _Plugins:
            return _Plugins()

        def discover(self, _kind: object) -> list[object]:
            return [plugin]

    monkeypatch.setattr(plugins, "Plugins", _Plugins)
    return state


def test_main_fails_without_plugin_conf(
    smoke: ModuleType,
    fake_image: SimpleNamespace,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_image.conf_dirs.append(tmp_path / "my_scenario_package" / "conf")
    assert smoke.main(["smoke-test.py"]) == 1
    assert "conf/ missing from the wheel" in capsys.readouterr().err


def test_main_passes_with_plugin_conf(
    smoke: ModuleType, fake_image: SimpleNamespace, tmp_path: Path
) -> None:
    fake_image.conf_dirs.append(_plugin_conf(tmp_path))
    assert smoke.main(["smoke-test.py"]) == 0


def test_main_fails_without_scenarios(
    smoke: ModuleType, fake_image: SimpleNamespace, tmp_path: Path
) -> None:
    fake_image.conf_dirs.append(_plugin_conf(tmp_path))
    fake_image.scenarios = {}
    assert smoke.main(["smoke-test.py"]) == 1


def test_framework_conf_dir_constant_exists() -> None:
    """The script reads the registry's private constant: keep the name alive."""
    from autoware_carla_scenario import registry

    assert registry._FRAMEWORK_CONF_DIR.name == "conf"
