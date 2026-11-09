"""Unit tests for the pluggable scenario registry (Issue #420)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import typesafe_carla.carla as carla
import pytest
from omegaconf import DictConfig, OmegaConf

from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyConfig:
    """Minimal config class that stores kwargs as attributes."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _DummyScenario(BaseScenario):
    """Scenario stub for registry tests."""

    def __init__(self, ego_config: EgoConfig, **kwargs: Any) -> None:
        super().__init__(ego_config)
        self.init_kwargs = kwargs

    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return True


def _make_ego() -> EgoConfig:
    return EgoConfig(
        spawn_location=SpawnTransform(
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        ),
        vehicle_type="vehicle.mini.cooper",
    )


def _make_unknown_scenario_cfg() -> DictConfig:
    """Config with an unregistered scenario name, suitable for error-path tests."""
    return OmegaConf.create(
        {
            "scenario": {"name": "__nonexistent__"},
            "entity": {
                "ground_projection_ray_distance_upper": 10.0,
                "ground_projection_ray_distance_lower": 1.0,
                "spawn_retry_max_count": 3,
                "spawn_retry_t_step": 0.1,
                "spawn_retry_z_step": 0.1,
            },
            "ego": {
                "vehicle_type": "vehicle.mini.cooper",
                "initial_speed_kmh": 0.0,
                "spawn_lanelet_id": 1,
                "spawn_s": 0.0,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests for registry API
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    """Tests for register_scenario, register_scenario_builder, get_scenario_registry."""

    def test_builtin_scenarios_are_registered(self) -> None:
        """All four built-in scenarios should be present in the registry."""
        from autoware_carla_scenario.examples.run import get_scenario_registry

        registry = get_scenario_registry()
        expected = {
            "intersection_passing",
            "traffic_light_compliance",
            "lane_change",
            "temporary_stop",
        }
        assert expected.issubset(set(registry.keys()))

    def test_get_scenario_registry_returns_copy(self) -> None:
        """Mutating the returned dict must not affect the internal registry."""
        from autoware_carla_scenario.examples.run import get_scenario_registry

        copy1 = get_scenario_registry()
        copy1["__test_sentinel__"] = lambda *_: None  # type: ignore[assignment]
        copy2 = get_scenario_registry()
        assert "__test_sentinel__" not in copy2

    def test_register_scenario(self) -> None:
        """register_scenario() should create a builder in the registry."""
        from autoware_carla_scenario import (
            get_scenario_builder,
            get_scenario_registry,
            register_scenario,
            unregister_scenario,
        )

        name = "__test_register_scenario__"
        try:
            register_scenario(name, _DummyScenario, _DummyConfig)
            assert name in get_scenario_registry()

            # Verify the builder works.
            ego = _make_ego()
            spawn_pose = MagicMock()
            ground_projection = MagicMock()
            scenario_dict: dict[str, Any] = {"name": name, "foo": "bar"}
            builder = get_scenario_builder(name)
            assert builder is not None
            result = builder(ego, scenario_dict, spawn_pose, ground_projection)
            assert isinstance(result, _DummyScenario)
        finally:
            unregister_scenario(name)

    def test_register_scenario_builder(self) -> None:
        """register_scenario_builder() should accept a custom callable."""
        from autoware_carla_scenario import (
            get_scenario_builder,
            register_scenario_builder,
            unregister_scenario,
        )

        name = "__test_register_builder__"
        sentinel = _DummyScenario(_make_ego())

        def custom_builder(
            ego: EgoConfig,
            scenario_dict: dict[str, Any],
            spawn_pose: Any,
            ground_projection: Any,
        ) -> BaseScenario:
            return sentinel

        try:
            register_scenario_builder(name, custom_builder)
            builder = get_scenario_builder(name)
            assert builder is custom_builder
            result = builder(_make_ego(), {}, MagicMock(), MagicMock())
            assert result is sentinel
        finally:
            unregister_scenario(name)

    def test_register_overwrites_existing(self) -> None:
        """Re-registering under the same name should overwrite silently."""
        from autoware_carla_scenario import (
            get_scenario_builder,
            register_scenario,
            unregister_scenario,
        )

        name = "__test_overwrite__"
        try:
            register_scenario(name, _DummyScenario, _DummyConfig)
            old_builder = get_scenario_builder(name)
            register_scenario(name, _DummyScenario, _DummyConfig)
            # Builder should be a new closure.
            assert get_scenario_builder(name) is not old_builder
        finally:
            unregister_scenario(name)


class TestPluginReplacement:
    """A plugin that takes a registered name is reported (#1802)."""

    @staticmethod
    def _load(
        monkeypatch: pytest.MonkeyPatch, register: Any, name: str = "my_pkg"
    ) -> None:
        from autoware_carla_scenario import registry  # noqa: PLC0415

        entry_point = MagicMock()
        entry_point.name = name
        entry_point.value = f"{name}:register"
        entry_point.load.return_value = register
        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda *, group: [entry_point]
        )
        monkeypatch.setattr(registry, "_plugins_loaded", False)
        monkeypatch.setattr(registry, "_PLUGIN_SCENARIOS", {})
        registry.load_scenario_plugins()

    @staticmethod
    def _keep(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        """Restore the registration of *name* once the test is over."""
        from autoware_carla_scenario import registry  # noqa: PLC0415

        for table in (registry._SCENARIO_REGISTRY, registry._SCENARIO_CLASSES):
            if name in table:
                monkeypatch.setitem(table, name, table[name])
            else:
                monkeypatch.delitem(table, name, raising=False)

    def test_a_plugin_replacing_a_built_in_is_warned_about(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import autoware_carla_scenario.examples.run  # noqa: F401, PLC0415
        from autoware_carla_scenario import register_scenario  # noqa: PLC0415

        self._keep(monkeypatch, "lane_change")
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                lambda: register_scenario("lane_change", _DummyScenario, _DummyConfig),
            )
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "'my_pkg'" in message
        assert "my_pkg:register" in message
        assert "'lane_change'" in message

    def test_a_plugin_adding_a_new_name_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario import register_scenario  # noqa: PLC0415

        name = "__test_plugin_new_name__"
        self._keep(monkeypatch, name)
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                lambda: register_scenario(name, _DummyScenario, _DummyConfig),
            )
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_failing_plugin_is_still_checked(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """What a plugin registered before it raised stays registered."""
        from autoware_carla_scenario import register_scenario  # noqa: PLC0415

        name = "__test_plugin_taken__"
        self._keep(monkeypatch, name)
        register_scenario(name, _DummyScenario, _DummyConfig)

        class _OtherScenario(_DummyScenario):
            pass

        def register() -> None:
            register_scenario(name, _OtherScenario, _DummyConfig)
            raise RuntimeError("boom")

        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(monkeypatch, register, name="broken_pkg")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert f"'{name}'" in warnings[0].getMessage()
        assert "'broken_pkg'" in warnings[0].getMessage()

    def test_registering_the_same_pair_again_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ``register()`` called explicitly, then by its entry point.

        The tutorial's "No entry point? Register explicitly." tip, with the
        package installed: the same pair is registered twice, which replaces
        nothing, although each call makes a new builder.
        """
        from autoware_carla_scenario import register_scenario  # noqa: PLC0415

        name = "__test_plugin_same_pair__"
        self._keep(monkeypatch, name)

        def register() -> None:
            register_scenario(name, _DummyScenario, _DummyConfig)

        register()
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(monkeypatch, register)
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_distribution_seen_twice_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An editable install and a wheel of the same package on the path."""
        from autoware_carla_scenario import registry  # noqa: PLC0415
        from autoware_carla_scenario import register_scenario  # noqa: PLC0415

        name = "__test_plugin_twice__"
        self._keep(monkeypatch, name)

        def register() -> None:
            register_scenario(name, _DummyScenario, _DummyConfig)

        entry_points = []
        for _ in range(2):
            entry_point = MagicMock()
            entry_point.name = "my_pkg"
            entry_point.value = "my_pkg:register"
            entry_point.load.return_value = register
            entry_points.append(entry_point)
        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda *, group: entry_points
        )
        monkeypatch.setattr(registry, "_plugins_loaded", False)
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            registry.load_scenario_plugins()
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_different_config_class_is_a_replacement(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario import register_scenario  # noqa: PLC0415

        name = "__test_plugin_other_config__"
        self._keep(monkeypatch, name)
        register_scenario(name, _DummyScenario, _DummyConfig)

        class _OtherConfig(_DummyConfig):
            pass

        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                lambda: register_scenario(name, _DummyScenario, _OtherConfig),
            )
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert f"'{name}'" in warnings[0].getMessage()


def _entry_point(name: str, register: Any, dist: str | None = None) -> Any:
    """An entry point of the distribution *dist* that calls *register*."""
    entry_point = MagicMock()
    entry_point.name = name
    entry_point.value = f"{name}:register"
    entry_point.dist.name = dist or name
    entry_point.load.return_value = register
    return entry_point


def _conf_dir(root: Path, name: str, configs: dict[str, str]) -> Path:
    """A throwaway package config directory holding *configs*."""
    conf_dir = root / name / "conf"
    for relative, text in configs.items():
        (conf_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        (conf_dir / relative).write_text(text)
    return conf_dir


def _isolate_conf_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the conf-dir registry and its index from outliving a test."""
    import autoware_carla_scenario.examples.run  # noqa: F401, PLC0415
    from autoware_carla_scenario import registry  # noqa: PLC0415

    monkeypatch.setattr(registry, "_CONF_DIRS", list(registry._CONF_DIRS))
    monkeypatch.setattr(registry, "_CONFIG_INDEX", dict(registry._CONFIG_INDEX))
    monkeypatch.setattr(registry, "_INDEXED_DIRS", list(registry._INDEXED_DIRS))
    monkeypatch.setattr(registry, "_DIR_CONFIG_PATHS", dict(registry._DIR_CONFIG_PATHS))


class TestPluginCollisions:
    """Two packages that provide the same scenario (#2034)."""

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep every registration a test makes from outliving it.

        The runner is imported first: importing it registers the built-ins,
        which must land in the real tables, not in this test's copies.
        """
        import autoware_carla_scenario.examples.run  # noqa: F401, PLC0415
        from autoware_carla_scenario import registry  # noqa: PLC0415

        for table in (
            "_SCENARIO_REGISTRY",
            "_SCENARIO_CLASSES",
            "_SCENARIO_DOCUMENTS",
            "_BUILTIN_SCENARIOS",
        ):
            monkeypatch.setattr(registry, table, dict(getattr(registry, table)))
        monkeypatch.setattr(registry, "_PLUGIN_SCENARIOS", {})
        _isolate_conf_dirs(monkeypatch)

    @staticmethod
    def _load(monkeypatch: pytest.MonkeyPatch, entry_points: list[Any]) -> None:
        from autoware_carla_scenario import registry  # noqa: PLC0415

        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda *, group: list(entry_points)
        )
        monkeypatch.setattr(registry, "_plugins_loaded", False)
        registry.load_scenario_plugins()

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    @staticmethod
    def _packages(
        tmp_path: Path, fail_b: bool = False
    ) -> tuple[Any, Any, type, type, Path, Path]:
        """Packages ``pkga`` and ``pkgb``, both providing scenario ``dup``.

        ``pkgb`` also provides ``b_only``, and a config only it ships that
        names ``dup``.
        """
        from autoware_carla_scenario import (  # noqa: PLC0415
            register_conf_dir,
            register_scenario,
        )

        class _ScenarioA(_DummyScenario):
            pass

        class _ScenarioB(_DummyScenario):
            pass

        conf_a = _conf_dir(
            tmp_path, "pkga", {"scenario/dup/default.yaml": "name: dup\nfrom_a: 42\n"}
        )
        conf_b = _conf_dir(
            tmp_path,
            "pkgb",
            {
                "scenario/dup/default.yaml": "name: dup\nfrom_b: 7\n",
                "scenario/dup/b_only.yaml": "name: dup\nfrom_b: 1\n",
            },
        )

        def register_a() -> None:
            register_scenario("dup", _ScenarioA, _DummyConfig)
            register_conf_dir(conf_a)

        def register_b() -> None:
            register_scenario("b_only", _ScenarioB, _DummyConfig)
            register_conf_dir(conf_b)
            register_scenario("dup", _ScenarioB, _DummyConfig)
            if fail_b:
                raise RuntimeError("boom")

        return (
            _entry_point("pkga", register_a, dist="plugtest-a"),
            _entry_point("pkgb", register_b, dist="plugtest_b"),
            _ScenarioA,
            _ScenarioB,
            conf_a.resolve(),
            conf_b.resolve(),
        )

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("fail_b", [False, True])
    def test_the_later_package_is_refused_as_a_whole(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
        reverse: bool,
        fail_b: bool,
    ) -> None:
        """Whatever order the environment lists them in, even if it raised."""
        from autoware_carla_scenario import registry  # noqa: PLC0415

        first, second, scenario_a, _, conf_a, conf_b = self._packages(
            tmp_path, fail_b=fail_b
        )
        listed = [second, first] if reverse else [first, second]
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(monkeypatch, listed)

        assert registry.get_scenario_classes("dup") == (scenario_a, _DummyConfig)
        # Everything pkgb registered here is gone: its other scenario and its
        # conf dir, so no config of its own reaches Hydra.
        assert registry.get_scenario_builder("b_only") is None
        assert conf_a in registry.get_conf_dirs()
        assert conf_b not in registry.get_conf_dirs()

        warnings = self._warnings(caplog)
        refusal = next(w for w in warnings if "was not loaded" in w)
        assert "'pkgb' (pkgb:register) was not loaded" in refusal
        assert "'dup' from 'pkga' (pkga:register)" in refusal
        assert "traffic backends" in refusal

    def test_the_composed_config_and_the_class_agree(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """End to end through Hydra's search path (the issue's reproduction)."""
        from hydra.errors import MissingConfigException  # noqa: PLC0415

        import autoware_carla_scenario.examples.run as run  # noqa: PLC0415
        from autoware_carla_scenario import registry  # noqa: PLC0415

        first, second, scenario_a, _, _, _ = self._packages(tmp_path)
        self._load(monkeypatch, [second, first])
        cfg = run._compose_config("dup/default", [])
        assert cfg.scenario.name == "dup"
        assert cfg.scenario.from_a == 42
        assert "from_b" not in cfg.scenario
        assert registry.get_scenario_classes("dup") == (scenario_a, _DummyConfig)
        # The refused package's own config cannot be selected at all.
        with pytest.raises(MissingConfigException):
            run._compose_config("dup/b_only", [])

    def test_the_first_document_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A declarative scenario keeps the document of the package that won."""
        from autoware_carla_scenario import registry  # noqa: PLC0415

        builder_a = MagicMock()
        builder_b = MagicMock()
        document_a = MagicMock()

        def register_a() -> None:
            registry.register_scenario_builder("decl", builder_a)
            registry.register_scenario_document("decl", lambda: document_a)

        def register_b() -> None:
            registry.register_scenario_builder("decl", builder_b)
            registry.register_scenario_document("decl", MagicMock)

        self._load(
            monkeypatch,
            [_entry_point("b_pkg", register_b), _entry_point("a_pkg", register_a)],
        )
        assert registry.get_scenario_builder("decl") is builder_a
        assert registry.get_scenario_document("decl") is document_a

    def test_a_document_alone_for_another_plugins_name_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario import registry  # noqa: PLC0415

        builder_a = MagicMock()

        def register_b() -> None:
            registry.register_scenario_document("decl", MagicMock)

        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                [
                    _entry_point(
                        "a_pkg",
                        lambda: registry.register_scenario_builder("decl", builder_a),
                    ),
                    _entry_point("b_pkg", register_b),
                ],
            )
        assert registry.get_scenario_builder("decl") is builder_a
        assert registry.get_scenario_document("decl") is None
        (message,) = self._warnings(caplog)
        assert "'b_pkg' (b_pkg:register) was not loaded" in message

    def test_a_package_registered_twice_with_a_document_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A new loader closure per ``register()`` is still the same document."""
        from autoware_carla_scenario import registry  # noqa: PLC0415

        builder = MagicMock()
        document = MagicMock()

        def register() -> None:
            registry.register_scenario_builder("decl", builder)
            registry.register_scenario_document("decl", lambda: document)

        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                [_entry_point("pkg", register), _entry_point("pkg", register)],
            )
        assert not self._warnings(caplog)

    def test_a_plugin_replacing_a_built_in_still_replaces_it(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """#1802's behaviour: a name registered before the plugins is taken."""
        import autoware_carla_scenario.examples.run  # noqa: F401, PLC0415
        from autoware_carla_scenario import registry  # noqa: PLC0415

        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                [
                    _entry_point(
                        "my_pkg",
                        lambda: registry.register_scenario(
                            "lane_change", _DummyScenario, _DummyConfig
                        ),
                    )
                ],
            )
        assert registry.get_scenario_classes("lane_change") == (
            _DummyScenario,
            _DummyConfig,
        )
        (message,) = self._warnings(caplog)
        # The text says which config is used: the framework's, not the plugin's.
        assert "Hydra still composes" in message
        assert "framework's conf directory" in message
        assert "gets the plugin's" not in message

    @staticmethod
    def _before_the_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
        """A process that has not imported ``examples.run``.

        Empties the registry, and writes a module ``fake_run_<n>`` that, like
        ``examples.run``, registers the framework's conf dir and a built-in
        scenario ``fake_builtin`` when it is imported; returns its name.
        """
        from autoware_carla_scenario import registry  # noqa: PLC0415

        for table in (
            "_SCENARIO_REGISTRY",
            "_SCENARIO_CLASSES",
            "_SCENARIO_DOCUMENTS",
            "_BUILTIN_SCENARIOS",
        ):
            monkeypatch.setattr(registry, table, {})
        monkeypatch.setattr(registry, "_CONF_DIRS", [])
        module = f"fake_run_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from autoware_carla_scenario import registry\n"
            "class Scenario: pass\n"
            "class Config: pass\n"
            "registry.register_conf_dir(registry._FRAMEWORK_CONF_DIR)\n"
            "registry._register_builtin_scenario('fake_builtin', Scenario, Config)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, module, raising=False)
        return module

    def test_a_refused_plugin_keeps_the_built_ins_its_import_registered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Only what the plugin registered itself is undone."""
        import importlib  # noqa: PLC0415

        from autoware_carla_scenario import registry  # noqa: PLC0415

        module = self._before_the_runner(monkeypatch, tmp_path)
        conf_b = _conf_dir(tmp_path, "pkgb", {"scenario/dup/b.yaml": "name: dup\n"})

        def register_b() -> None:
            importlib.import_module(module)  # e.g. examples.configs
            registry.register_scenario("dup", _DummyScenario, _DummyConfig)
            registry.register_conf_dir(conf_b)

        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                [
                    _entry_point(
                        "pkga",
                        lambda: registry.register_scenario_builder("dup", MagicMock()),
                    ),
                    _entry_point("pkgb", register_b),
                ],
            )
        refused = "'pkgb' (pkgb:register) was not loaded"
        assert any(refused in w for w in self._warnings(caplog))
        fake = sys.modules[module]
        assert registry.get_scenario_classes("fake_builtin") == (
            fake.Scenario,
            fake.Config,
        )
        assert registry.get_scenario_classes("dup") is None  # pkga's builder
        assert registry._FRAMEWORK_CONF_DIR in registry.get_conf_dirs()
        assert conf_b.resolve() not in registry.get_conf_dirs()

    def test_the_framework_conf_dir_survives_a_refusal_that_registered_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from autoware_carla_scenario import registry  # noqa: PLC0415

        self._before_the_runner(monkeypatch, tmp_path)
        real_register = registry.register_conf_dir
        framework_first: list[bool] = []

        def register_b() -> None:
            # As if the walk had not registered it before (an older caller).
            registry._CONF_DIRS.remove(registry._FRAMEWORK_CONF_DIR)
            real_register(registry._FRAMEWORK_CONF_DIR)
            registry.register_scenario("dup", _DummyScenario, _DummyConfig)

        def register_a() -> None:
            first = registry.get_conf_dirs()[:1]
            framework_first.append(first == [registry._FRAMEWORK_CONF_DIR])
            registry.register_scenario_builder("dup", MagicMock())

        self._load(
            monkeypatch,
            [_entry_point("pkga", register_a), _entry_point("pkgb", register_b)],
        )
        # The walk registered the framework's conf dir before any plugin.
        assert framework_first == [True]
        assert registry.get_conf_dirs() == [registry._FRAMEWORK_CONF_DIR]

    def test_a_winner_importing_the_built_ins_does_not_own_them(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """A later plugin replacing that built-in replaces it (#1802)."""
        import importlib  # noqa: PLC0415

        from autoware_carla_scenario import registry  # noqa: PLC0415

        module = self._before_the_runner(monkeypatch, tmp_path)

        class _Replacement(_DummyScenario):
            pass

        def register_a() -> None:
            importlib.import_module(module)
            registry.register_scenario("dup", _DummyScenario, _DummyConfig)

        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                [
                    _entry_point("pkga", register_a),
                    _entry_point(
                        "pkgb",
                        lambda: registry.register_scenario(
                            "fake_builtin", _Replacement, _DummyConfig
                        ),
                    ),
                ],
            )
        assert registry.get_scenario_classes("fake_builtin") == (
            _Replacement,
            _DummyConfig,
        )
        (message,) = self._warnings(caplog)
        assert "'pkgb' (pkgb:register) replaced 'fake_builtin'" in message

    def test_a_built_in_replaced_before_the_runner_is_imported(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Warned about then, and not silently undone by the import later."""
        from autoware_carla_scenario import registry  # noqa: PLC0415

        self._before_the_runner(monkeypatch, tmp_path)
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                [
                    _entry_point(
                        "my_pkg",
                        lambda: registry.register_scenario(
                            "lane_change", _DummyScenario, _DummyConfig
                        ),
                    )
                ],
            )
        (message,) = self._warnings(caplog)
        assert "'my_pkg' (my_pkg:register) replaced 'lane_change'" in message

        class _BuiltIn(_DummyScenario):
            pass

        # What importing examples.run now does.
        registry._register_builtin_scenario("lane_change", _BuiltIn, _DummyConfig)
        assert registry.get_scenario_classes("lane_change") == (
            _DummyScenario,
            _DummyConfig,
        )
        # A built-in no plugin took is registered as usual.
        registry._register_builtin_scenario("cut_in", _BuiltIn, _DummyConfig)
        assert registry.get_scenario_classes("cut_in") == (_BuiltIn, _DummyConfig)

    def test_entry_points_load_sorted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """By normalised distribution name, then entry-point name."""
        calls: list[str] = []
        entry_points = [
            _entry_point(name, lambda name=name: calls.append(name), dist=dist)
            for name, dist in [
                ("c", "Plugtest_C"),
                ("a2", "plugtest-a"),
                ("b", "plugtest.b"),
                ("a1", "plugtest_a"),
            ]
        ]
        self._load(monkeypatch, entry_points)
        assert calls == ["a1", "a2", "b", "c"]


class TestConfDirCollisions:
    """A config path two directories provide is reported (#2034)."""

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _isolate_conf_dirs(monkeypatch)

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    def test_a_config_only_shadow_of_a_built_in_path_warns(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Even before the framework's own directory is registered."""
        from autoware_carla_scenario import registry  # noqa: PLC0415

        conf = _conf_dir(
            tmp_path,
            "pkg",
            {
                "scenario/lane_change/left.yaml": "name: lane_change\n",
                "scenario/mine/default.yaml": "name: mine\n",
            },
        )
        registry._CONF_DIRS.clear()
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            registry.register_conf_dir(conf)
        (message,) = self._warnings(caplog)
        assert "scenario/lane_change/left.yaml" in message
        assert "scenario/mine" not in message
        assert f"already provided by {registry._FRAMEWORK_CONF_DIR}," in message
        assert message.count(str(conf.resolve())) == 1

    def test_other_groups_are_compared_too(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        from autoware_carla_scenario import registry  # noqa: PLC0415

        first = _conf_dir(tmp_path, "one", {"map/my_map.yaml": "a: 1\n"})
        second = _conf_dir(tmp_path, "two", {"map/my_map.yaml": "a: 2\n"})
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            registry.register_conf_dir(first)
            assert not self._warnings(caplog)
            registry.register_conf_dir(second)
            registry.register_conf_dir(second)  # a no-op, and quiet
        (message,) = self._warnings(caplog)
        assert "map/my_map.yaml" in message
        assert f"already provided by {first.resolve()}," in message

    def test_an_unregistered_directory_shadows_nothing(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        from autoware_carla_scenario import registry  # noqa: PLC0415

        first = _conf_dir(tmp_path, "one", {"map/my_map.yaml": "a: 1\n"})
        second = _conf_dir(tmp_path, "two", {"map/my_map.yaml": "a: 2\n"})
        registry.register_conf_dir(first)
        registry.unregister_conf_dir(first)
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            registry.register_conf_dir(second)
        assert not self._warnings(caplog)

    def test_distinct_paths_are_quiet(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        from autoware_carla_scenario import registry  # noqa: PLC0415

        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            registry.register_conf_dir(registry._FRAMEWORK_CONF_DIR)
            registry.register_conf_dir(
                _conf_dir(tmp_path, "pkg", {"scenario/mine/a.yaml": "name: mine\n"})
            )
            registry.register_conf_dir(tmp_path / "missing")
        assert not self._warnings(caplog)

    def test_each_directory_is_scanned_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Registering a directory does not rescan the ones before it."""
        from autoware_carla_scenario import registry  # noqa: PLC0415

        monkeypatch.setattr(registry, "_CONF_DIRS", [])
        monkeypatch.setattr(registry, "_CONFIG_INDEX", {})
        monkeypatch.setattr(registry, "_INDEXED_DIRS", [])
        monkeypatch.setattr(registry, "_DIR_CONFIG_PATHS", {})
        scanned: list[Path] = []
        scan = registry._config_paths

        def counting(directory: Path) -> frozenset[str]:
            scanned.append(directory)
            return scan(directory)

        monkeypatch.setattr(registry, "_config_paths", counting)
        registry.register_conf_dir(registry._FRAMEWORK_CONF_DIR)
        dirs = [
            _conf_dir(tmp_path, f"pkg{i}", {f"scenario/s{i}/a.yaml": "{}\n"})
            for i in range(20)
        ]
        for conf_dir in dirs:
            registry.register_conf_dir(conf_dir)
        # Unregistering one rebuilds the index from what was scanned.
        registry.unregister_conf_dir(dirs[3])
        registry.register_conf_dir(_conf_dir(tmp_path, "last", {"x/y.yaml": "{}\n"}))
        assert len(scanned) == len(set(scanned)) == 22
        assert registry._FRAMEWORK_CONF_DIR in scanned


# ---------------------------------------------------------------------------
# Tests for build_scenario with injectable builder
# ---------------------------------------------------------------------------


class TestBuildScenarioInjectable:
    """Tests for the build_scenario_fn parameter."""

    def test_build_scenario_fn_overrides_registry(self) -> None:
        """When build_scenario_fn is given, it should be used instead of the registry."""
        from autoware_carla_scenario.examples.run import build_scenario

        ego = _make_ego()
        sentinel = _DummyScenario(ego)

        def custom_fn(cfg: DictConfig) -> tuple[EgoConfig, BaseScenario]:
            return ego, sentinel

        cfg = OmegaConf.create({"scenario": {"name": "nonexistent"}})
        result_ego, result_scenario = build_scenario(cfg, build_scenario_fn=custom_fn)
        assert result_ego is ego
        assert result_scenario is sentinel

    def test_an_injected_scenario_is_given_the_configured_waypoints(self) -> None:
        """The way there must reach an injected scenario, like the goal does.

        `build_scenario_fn` bypasses the constructor `build_ego_and_spawn`
        fills, so whatever `_apply_ego_config` leaves behind is dropped
        silently.  Dropping the waypoints would let Autoware plan the shortcut
        they exist to prevent, on exactly the path downstream projects use.
        """
        from autoware_carla_scenario.examples.run import build_scenario

        ego = _make_ego()
        sentinel = _DummyScenario(ego)

        cfg = OmegaConf.create(
            {
                "scenario": {"name": "nonexistent"},
                "ego": {"waypoint_lanelet_ids": [11, 22]},
            }
        )
        build_scenario(cfg, build_scenario_fn=lambda _cfg: (ego, sentinel))

        assert [p.lanelet_id for p in sentinel.waypoint_poses] == [11, 22]

    def test_an_injected_scenario_keeps_its_own_waypoints_when_none_configured(
        self,
    ) -> None:
        """No waypoints configured means none were asked for, not none intended.

        A scenario may derive its own in `setup()`; an empty config list must
        not overwrite them, for the same reason a missing goal does not.
        """
        from autoware_carla_scenario.coordinate import Lanelet2Pose
        from autoware_carla_scenario.examples.run import build_scenario

        ego = _make_ego()
        sentinel = _DummyScenario(ego)
        sentinel.waypoint_poses = [Lanelet2Pose(lanelet_id=99, s=0.0)]

        cfg = OmegaConf.create({"scenario": {"name": "nonexistent"}})
        build_scenario(cfg, build_scenario_fn=lambda _cfg: (ego, sentinel))

        assert [p.lanelet_id for p in sentinel.waypoint_poses] == [99]

    def test_build_scenario_unknown_name_raises(self) -> None:
        """build_scenario should raise ValueError for unregistered names."""
        from autoware_carla_scenario.examples.run import build_scenario

        with pytest.raises(ValueError, match="Unknown scenario name"):
            build_scenario(_make_unknown_scenario_cfg())

    def test_build_scenario_unknown_name_lists_registered(self) -> None:
        """The error message should list registered scenario names."""
        from autoware_carla_scenario.examples.run import build_scenario

        with pytest.raises(ValueError, match="Registered scenarios"):
            build_scenario(_make_unknown_scenario_cfg())


# ---------------------------------------------------------------------------
# Tests for build_ego_entity
# ---------------------------------------------------------------------------


class TestBuildEgoEntity:
    """Tests for the ego entity selected by build_ego_entity()."""

    def test_missing_ego_group_selects_autopilot(self) -> None:
        """A config with no 'ego' group at all should select autopilot, not raise.

        build_scenario() calls build_ego_entity() even when an injected
        build_scenario_fn supplied the EgoConfig, and such a config has no reason
        to carry the group Hydra would compose.
        """
        from autoware_carla_scenario.examples.run import build_ego_entity

        cfg = OmegaConf.create({"scenario": {"name": "whatever"}})
        assert build_ego_entity(cfg) is None

    def test_empty_ego_group_selects_autopilot(self) -> None:
        """An 'ego' group without an 'entity' key should select autopilot."""
        from autoware_carla_scenario.examples.run import build_ego_entity

        cfg = OmegaConf.create({"ego": {"vehicle_type": "vehicle.tesla.model3"}})
        assert build_ego_entity(cfg) is None

    def test_unknown_entity_raises(self) -> None:
        """An unrecognised ego.entity should raise rather than silently autopilot."""
        from autoware_carla_scenario.examples.run import build_ego_entity

        cfg = OmegaConf.create({"ego": {"entity": "__nope__"}})
        with pytest.raises(ValueError, match="Unknown ego.entity"):
            build_ego_entity(cfg)


class TestBuildAutowareLauncher:
    """``autoware.launcher`` selects who starts Autoware for each scenario."""

    @staticmethod
    def _cfg(entity: str = "autoware", **launcher: object) -> DictConfig:
        return OmegaConf.create(
            {
                "ego": {"entity": entity},
                "scenario": {"name": "left_turn"},
                "autoware": {"address": "localhost:0", "launcher": launcher},
            }
        )

    def test_none_by_default(self) -> None:
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        assert build_autoware_launcher(self._cfg(type="none")) is None
        assert (
            build_autoware_launcher(OmegaConf.create({"ego": {"entity": "autoware"}}))
            is None
        )

    def test_only_an_autoware_ego_has_one(self) -> None:
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg("autopilot", type="command", command=["true"])
        assert build_autoware_launcher(cfg) is None

    def test_docker_is_shared_by_every_ego_of_the_batch(self, tmp_path) -> None:
        from autoware_carla_scenario.autoware_stack import DockerAutowareLauncher
        from autoware_carla_scenario.examples.run import (
            build_autoware_launcher,
            build_ego_entity,
        )

        cfg = self._cfg(
            type="docker",
            workspace=str(tmp_path),
            launch=[
                "autoware_launch",
                "x.launch.xml",
                "bridge_address:={bridge_address}",
            ],
            ros_domain_ids=[3, 4],
            # The command launcher's key: the group holds both types' keys.
            command=["true"],
        )
        launcher = build_autoware_launcher(cfg)

        assert isinstance(launcher, DockerAutowareLauncher)
        assert launcher.config.workspace == tmp_path
        assert launcher.config.ros_domain_ids == (3, 4)
        assert build_autoware_launcher(cfg) is launcher
        entity = build_ego_entity(cfg)
        assert entity is not None
        assert entity.launcher is launcher  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        ("value", "rosdep"),
        [
            ("false", False),
            (" Off ", False),
            (0, False),
            ("0", False),
            (False, False),
            ("true", True),
            ("yes", True),
            (1, True),
            (True, True),
        ],
    )
    def test_rosdep_is_read_strictly(
        self, tmp_path: Path, value: object, rosdep: bool
    ) -> None:
        """#1761: the launcher reads it by truthiness, so "false" ran rosdep."""
        from autoware_carla_scenario.autoware_stack import DockerAutowareLauncher
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        launcher = build_autoware_launcher(
            self._cfg(
                type="docker",
                workspace=str(tmp_path),
                launch=["autoware_launch", "x.launch.xml"],
                rosdep=value,
            )
        )
        assert isinstance(launcher, DockerAutowareLauncher)
        assert launcher.config.rosdep is rosdep

    def test_rosdep_defaults_to_on(self, tmp_path: Path) -> None:
        from autoware_carla_scenario.autoware_stack import DockerAutowareLauncher
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        launcher = build_autoware_launcher(
            self._cfg(
                type="docker",
                workspace=str(tmp_path),
                launch=["autoware_launch", "x.launch.xml"],
            )
        )
        assert isinstance(launcher, DockerAutowareLauncher)
        assert launcher.config.rosdep is True

    @pytest.mark.parametrize("value", ["ture", "", 2, 0.0, None, [True]])
    def test_a_rosdep_that_is_no_switch_is_refused(
        self, tmp_path: Path, value: object
    ) -> None:
        from autoware_carla_scenario.examples import run

        cfg = self._cfg(
            type="docker",
            workspace=str(tmp_path),
            launch=["autoware_launch", "x.launch.xml"],
            rosdep=value,
        )
        match = r"^autoware\.launcher\.rosdep must be true or false"
        with pytest.raises(ValueError, match=match):
            run.build_autoware_launcher(cfg)
        # Before anything starts, as the launcher's other settings are.
        with pytest.raises(ValueError, match=match):
            run._check_batch_queue_settings(cfg)

    @pytest.mark.parametrize("value", ["false", 0, None])
    def test_the_config_refuses_a_rosdep_that_is_no_bool(
        self, tmp_path: Path, value: object
    ) -> None:
        from autoware_carla_scenario.autoware_stack import DockerAutowareConfig

        with pytest.raises(ValueError, match=r"^rosdep must be true or false"):
            DockerAutowareConfig(
                workspace=tmp_path,
                launch=("autoware_launch", "x.launch.xml"),
                rosdep=value,  # type: ignore[arg-type]
            )

    def test_command(self) -> None:
        from autoware_carla_scenario.autoware_stack import CommandAutowareLauncher
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        launcher = build_autoware_launcher(
            self._cfg(type="command", command=["ros2", "launch", "{bridge_address}"])
        )

        assert isinstance(launcher, CommandAutowareLauncher)
        assert tuple(launcher.argv) == ("ros2", "launch", "{bridge_address}")

    def test_unknown_type_raises(self) -> None:
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        with pytest.raises(ValueError, match="Unknown autoware.launcher.type"):
            build_autoware_launcher(self._cfg(type="kubernetes"))

    @pytest.mark.parametrize("kind", ["none", "docker", "command"])
    @pytest.mark.parametrize("typo", ["stop_timout_s", "workspce"])
    def test_an_unknown_key_is_refused(self, kind: str, typo: str) -> None:
        """#1413: a key neither launcher reads is a typo, not a default."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
        )
        cfg.autoware.launcher[typo] = 60
        with pytest.raises(
            ValueError,
            match=rf"^Unknown autoware\.launcher key\(s\): "
            rf"\['autoware\.launcher\.{typo}'\]\. Known keys: .*'stop_timeout_s'",
        ):
            build_autoware_launcher(cfg)

    def test_the_shipped_launcher_keys_are_accepted(self) -> None:
        """The shipped config holds the keys of both types; each type takes it."""
        from hydra import compose, initialize_config_dir

        from autoware_carla_scenario.autoware_stack import (
            CommandAutowareLauncher,
            DockerAutowareLauncher,
        )
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        conf = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "autoware_carla_scenario"
            / "examples"
            / "conf"
        )
        common = [
            "ego.entity=autoware",
            "autoware.launcher.launch=[autoware_launch,x.launch.xml]",
            "autoware.launcher.command=[ros2,launch]",
            "autoware.launcher.stop_timeout_s=7",
            "+autoware.launcher.docker=podman",
            "+autoware.launcher.container_prefix=acs-test",
        ]
        with initialize_config_dir(config_dir=str(conf), version_base=None):
            docker = compose(
                config_name="config",
                overrides=[*common, "autoware.launcher.type=docker"],
            )
            command = compose(
                config_name="config",
                overrides=[*common, "autoware.launcher.type=command"],
            )
            none = compose(config_name="config", overrides=common)

        docker_launcher = build_autoware_launcher(docker)
        assert isinstance(docker_launcher, DockerAutowareLauncher)
        assert docker_launcher.config.stop_timeout_s == 7.0
        assert docker_launcher.config.docker == "podman"
        assert docker_launcher.config.container_prefix == "acs-test"
        command_launcher = build_autoware_launcher(command)
        assert isinstance(command_launcher, CommandAutowareLauncher)
        assert tuple(command_launcher.argv) == ("ros2", "launch")
        assert build_autoware_launcher(none) is None

    @pytest.mark.parametrize("kind", ["docker", "command"])
    @pytest.mark.parametrize(
        "value",
        [
            *(float("nan"), float("inf"), -5, 0, "abc", None, True),
            pytest.param(10**400, id="huge"),  # too big for a float (#1652)
        ],
    )
    def test_a_stop_timeout_that_is_not_positive_and_finite_is_refused(
        self, kind: str, value: object
    ) -> None:
        """#1450: a NaN never escalates a stack that ignores SIGINT, a negative
        one skips its grace, and a string used to fail only at stop time."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
            stop_timeout_s=value,
        )
        with pytest.raises(
            ValueError,
            match=r"^autoware\.launcher\.stop_timeout_s must be a finite number > 0",
        ):
            build_autoware_launcher(cfg)

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            # The range is DockerAutowareConfig's, prefixed with the group.
            ([-3, 999], r"must be ROS domain ids, whole numbers from 0 to 232, got -3"),
            ([3, 233], r"must be ROS domain ids, whole numbers from 0 to 232, got 233"),
            # The types are build_autoware_launcher's: no raw TypeError.
            (7, r"must be a list of ROS domain ids"),
            ("7", r"must be a list of ROS domain ids"),
            ([3, True], r"must be a list of ROS domain ids"),
            ([3, 4.5], r"must be a list of ROS domain ids"),
            ([3, "4"], r"must be a list of ROS domain ids"),
        ],
    )
    def test_a_ros_domain_id_that_is_not_one_is_refused(
        self, value: object, message: str
    ) -> None:
        """#1450: refused up front, not by the stack's DDS in the container."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type="docker",
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            ros_domain_ids=value,
        )
        with pytest.raises(
            ValueError, match=rf"^autoware\.launcher\.ros_domain_ids {message}"
        ):
            build_autoware_launcher(cfg)

    def test_the_ros_domain_id_range_and_null_are_taken(self) -> None:
        from autoware_carla_scenario.autoware_stack import DockerAutowareLauncher
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        launch = ["autoware_launch", "x.launch.xml"]
        launcher = build_autoware_launcher(
            self._cfg(
                type="docker",
                workspace="/nowhere",
                launch=launch,
                ros_domain_ids=[0, 232],
                stop_timeout_s=1,
            )
        )
        assert isinstance(launcher, DockerAutowareLauncher)
        assert launcher.config.ros_domain_ids == (0, 232)
        assert launcher.config.stop_timeout_s == 1.0
        launcher = build_autoware_launcher(
            self._cfg(
                type="docker",
                workspace="/nowhere",
                launch=launch,
                ros_domain_ids=None,
            )
        )
        assert isinstance(launcher, DockerAutowareLauncher)
        assert launcher.config.ros_domain_ids == ()

    @pytest.mark.parametrize(
        ("kind", "name"),
        [
            ("docker", "launch"),
            ("docker", "colcon_args"),
            ("docker", "docker_args"),
            ("command", "command"),
        ],
    )
    @pytest.mark.parametrize("value", ["--shm-size=8g", "ros2 launch x y", 7, {}])
    def test_an_argument_list_that_is_not_a_list_is_refused(
        self, kind: str, name: str, value: object
    ) -> None:
        """#1484: a string used to be split into its characters."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
        )
        cfg.autoware.launcher[name] = value
        with pytest.raises(
            ValueError,
            match=rf"^autoware\.launcher\.{name} must be a list of arguments, not ",
        ):
            build_autoware_launcher(cfg)

    @pytest.mark.parametrize(
        ("kind", "name"), [("docker", "docker_args"), ("command", "command")]
    )
    @pytest.mark.parametrize("item", [["a"], {"a": 1}, None, True])
    def test_an_argument_that_is_not_a_string_or_number_is_refused(
        self, kind: str, name: str, item: object
    ) -> None:
        """#1484: its str() (``"['a']"``, ``"None"``) is not the argument meant."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
        )
        cfg.autoware.launcher[name] = ["ok", item]
        with pytest.raises(
            ValueError,
            match=rf"^autoware\.launcher\.{name}\[1\] must be a string or a number",
        ):
            build_autoware_launcher(cfg)

    def test_argument_lists_are_made_strings(self) -> None:
        from autoware_carla_scenario.autoware_stack import (
            CommandAutowareLauncher,
            DockerAutowareLauncher,
        )
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        docker = build_autoware_launcher(
            self._cfg(
                type="docker",
                workspace="/nowhere",
                launch=["autoware_launch", "x.launch.xml", "rate:=10"],
                colcon_args=None,
                docker_args=["--shm-size=8g", "--cpus", 4],
                env={"A": 1, "B": "two words"},
                mounts={"/data": "/home/aw/data"},
            )
        )
        assert isinstance(docker, DockerAutowareLauncher)
        assert docker.config.launch == ("autoware_launch", "x.launch.xml", "rate:=10")
        assert docker.config.colcon_args == ()
        assert docker.config.docker_args == ("--shm-size=8g", "--cpus", "4")
        assert docker.config.env == {"A": "1", "B": "two words"}
        assert docker.config.mounts == {"/data": "/home/aw/data"}
        command = build_autoware_launcher(
            self._cfg(type="command", command=["sleep", 5], env=None)
        )
        assert isinstance(command, CommandAutowareLauncher)
        assert tuple(command.argv) == ("sleep", "5")
        assert command.env == {}

    @pytest.mark.parametrize(
        ("kind", "name", "value", "message"),
        [
            ("docker", "env", "A=1", r"env must be a mapping"),
            ("command", "env", ["A=1"], r"env must be a mapping"),
            ("docker", "mounts", "/data:/data", r"mounts must be a mapping"),
            ("docker", "env", {"A": None}, r"env\.A must be a string or a number"),
            ("command", "env", {"A": [1]}, r"env\.A must be a string or a number"),
            (
                "docker",
                "bridge_parameters",
                "x",
                r"bridge_parameters must be a mapping",
            ),
            (
                "command",
                "bridge_parameters",
                [1],
                r"bridge_parameters must be a mapping",
            ),
        ],
    )
    def test_a_mapping_setting_that_is_not_one_is_refused(
        self, kind: str, name: str, value: object, message: str
    ) -> None:
        """#1484: refused naming the key, not by an AttributeError at run time."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
        )
        cfg.autoware.launcher[name] = value
        with pytest.raises(ValueError, match=rf"^autoware\.launcher\.{message}"):
            build_autoware_launcher(cfg)

    def test_a_config_refusal_names_the_group(self) -> None:
        """DockerAutowareConfig's own refusals are prefixed with the group."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        with pytest.raises(
            ValueError, match=r"^autoware\.launcher\.launch is required"
        ):
            build_autoware_launcher(self._cfg(type="docker", workspace="/nowhere"))


# ---------------------------------------------------------------------------
# Tests for build_ego_and_spawn helper
# ---------------------------------------------------------------------------


def _ego_and_spawn_cfg(**ego_overrides: object) -> DictConfig:
    """Return a resolved-config shape build_ego_and_spawn() accepts."""
    ego: dict[str, object] = {
        "vehicle_type": "vehicle.tesla.model3",
        "initial_speed_kmh": 30.0,
        "spawn_lanelet_id": 42,
        "spawn_s": 5.0,
    }
    ego.update(ego_overrides)
    return OmegaConf.create(
        {
            "entity": {
                "ground_projection_ray_distance_upper": 10.0,
                "ground_projection_ray_distance_lower": 1.0,
                "spawn_retry_max_count": 5,
                "spawn_retry_t_step": 0.2,
                "spawn_retry_z_step": 0.3,
            },
            "ego": ego,
        }
    )


class TestBuildEgoAndSpawn:
    """Tests for the build_ego_and_spawn() helper."""

    def test_returns_ego_spawn_and_ground_projection(self) -> None:
        """build_ego_and_spawn should return a 3-tuple from the config."""
        from autoware_carla_scenario import (
            GroundProjectionConfig,
            Lanelet2Pose,
        )
        from autoware_carla_scenario.examples.run import build_ego_and_spawn

        ego, spawn_pose, ground_projection = build_ego_and_spawn(_ego_and_spawn_cfg())

        assert isinstance(ego, EgoConfig)
        assert ego.vehicle_type == "vehicle.tesla.model3"
        assert ego.initial_speed_kmh == 30.0

        assert isinstance(spawn_pose, Lanelet2Pose)
        assert spawn_pose.lanelet_id == 42
        assert spawn_pose.s == 5.0

        assert isinstance(ground_projection, GroundProjectionConfig)
        assert ground_projection.ray_distance_upper == 10.0
        assert ground_projection.ray_distance_lower == 1.0

    def test_the_configured_goal_travels_with_the_ego_config(self) -> None:
        """The goal is part of the ego's configuration, not a separate parcel."""
        from autoware_carla_scenario.examples.run import build_ego_and_spawn

        cfg = _ego_and_spawn_cfg(goal_lanelet_id=265, goal_s=12.5)
        ego, _, _ = build_ego_and_spawn(cfg)

        assert ego.goal_pose is not None
        assert ego.goal_pose.lanelet_id == 265
        assert ego.goal_pose.s == 12.5

    def test_a_config_without_a_goal_waits_for_the_scenario(self) -> None:
        """A scenario may know the destination the config does not.

        ``IntersectionPassingScenario`` derives the ego's goal in ``setup()``
        from the route it asserts, so a config that names none is not yet wrong;
        ``register_route_to_goal`` refuses an ego that still has nowhere to go.
        """
        from autoware_carla_scenario.examples.run import build_ego_and_spawn

        ego, _, _ = build_ego_and_spawn(_ego_and_spawn_cfg(entity="autoware"))

        assert ego.goal_pose is None


class TestEgoNumbersAreRefusedByTheirKey:
    """#1660: what ``float()`` or ``int()`` cannot read of the ego's and the
    entity's numbers is refused with a ``ValueError`` naming the key, not an
    unkeyed ``ValueError`` (``"abc"``) or ``OverflowError`` (``10**400``)."""

    @staticmethod
    def _cfg(group: str, key: str, value: object) -> DictConfig:
        cfg = _ego_and_spawn_cfg(goal_lanelet_id=265)
        cfg[group][key] = value
        return cfg

    @pytest.mark.parametrize(
        ("group", "key"),
        [
            ("ego", "initial_speed_kmh"),
            ("entity", "spawn_retry_t_step"),
            ("entity", "spawn_retry_z_step"),
            ("ego", "spawn_s"),
            ("ego", "spawn_t"),
            ("ego", "spawn_heading"),
            ("ego", "goal_s"),
        ],
    )
    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("abc", "must be a number, got 'abc'"),
            (10**400, "must be finite, got 1000"),
            (-(10**5000), "must be finite, got an integer of about -10\\*\\*"),
        ],
        ids=["text", "huge", "too-long-for-repr"],
    )
    def test_a_number_float_cannot_read(
        self, group: str, key: str, value: object, message: str
    ) -> None:
        from autoware_carla_scenario.examples.run import (
            build_ego_and_spawn,
            build_planned_route,
            build_spawn_pose,
        )

        cfg = self._cfg(group, key, value)
        match = rf"^{group}\.{key} {message}"
        with pytest.raises(ValueError, match=match):
            build_ego_and_spawn(cfg)
        if key in ("spawn_s", "spawn_t", "spawn_heading"):
            with pytest.raises(ValueError, match=match):
                build_spawn_pose(cfg)
            with pytest.raises(ValueError, match=match):
                build_planned_route(cfg)

    @pytest.mark.parametrize("value", ["abc", "1.5", math.inf])
    def test_a_retry_count_int_cannot_read(self, value: object) -> None:
        from autoware_carla_scenario.examples.run import build_ego_and_spawn

        cfg = self._cfg("entity", "spawn_retry_max_count", value)
        with pytest.raises(
            ValueError, match=r"^entity\.spawn_retry_max_count must be a whole number"
        ):
            build_ego_and_spawn(cfg)

    def test_what_float_and_int_read_is_read_as_before(self) -> None:
        """A numeric string, a bool and an int read as ``float()``/``int()`` do."""
        from autoware_carla_scenario.examples.run import (
            build_ego_and_spawn,
            build_planned_route,
        )

        cfg = _ego_and_spawn_cfg(
            goal_lanelet_id=265,
            initial_speed_kmh="12.5",
            spawn_s=True,
            spawn_t="-1",
            spawn_heading=0,
            goal_s="3",
        )
        cfg.entity.spawn_retry_max_count = "4"
        cfg.entity.spawn_retry_t_step = 2
        ego, spawn, _ = build_ego_and_spawn(cfg)
        assert ego.initial_speed_kmh == 12.5
        assert ego.spawn_retry_max_count == 4
        assert ego.spawn_retry_t_step == 2.0
        assert (spawn.s, spawn.t, spawn.heading) == (1.0, -1.0, 0.0)
        assert ego.goal_pose is not None and ego.goal_pose.s == 3.0
        route = build_planned_route(cfg)
        assert (route.start.s, route.start.t) == (1.0, -1.0)


#: The ego's and entity's numbers a run refuses unless finite (#1809): the
#: speed and the retry steps also below zero, the poses at any finite value.
_FINITE_EGO_NUMBERS = [
    ("ego", "initial_speed_kmh", True),
    ("entity", "spawn_retry_t_step", True),
    ("entity", "spawn_retry_z_step", True),
    ("ego", "spawn_s", False),
    ("ego", "spawn_t", False),
    ("ego", "spawn_heading", False),
    ("ego", "goal_s", False),
]


def _shipped(overrides: list[str]) -> DictConfig:
    from .test_video_removed import _shipped as compose

    return compose(overrides)  # type: ignore[no-any-return]


class TestEgoNumbersMustBeFinite:
    """#1809: a NaN, an infinity or a negative speed or retry step of the
    ego's and the entity's numbers is refused with a ``ValueError`` naming the
    key, by the build and by the batch check before anything starts."""

    @staticmethod
    def _bad_values(non_negative: bool) -> list[float]:
        values = [math.nan, math.inf, -math.inf]
        return [*values, -1.0] if non_negative else values

    @pytest.mark.parametrize(("group", "key", "non_negative"), _FINITE_EGO_NUMBERS)
    def test_the_build_refuses_it(
        self, group: str, key: str, non_negative: bool
    ) -> None:
        from autoware_carla_scenario.examples.run import (
            build_ego_and_spawn,
            build_goal_pose,
            build_planned_route,
            build_spawn_pose,
        )

        bound = "finite and at least 0" if non_negative else "finite"
        for value in self._bad_values(non_negative):
            cfg = TestEgoNumbersAreRefusedByTheirKey._cfg(group, key, value)
            match = rf"^{group}\.{key} must be {bound}, got "
            with pytest.raises(ValueError, match=match):
                build_ego_and_spawn(cfg)
            if key in ("spawn_s", "spawn_t", "spawn_heading"):
                with pytest.raises(ValueError, match=match):
                    build_spawn_pose(cfg)
                with pytest.raises(ValueError, match=match):
                    build_planned_route(cfg)
            if key == "goal_s":
                with pytest.raises(ValueError, match=match):
                    build_goal_pose(cfg)

    @pytest.mark.parametrize(("group", "key", "non_negative"), _FINITE_EGO_NUMBERS)
    def test_the_batch_check_refuses_it(
        self, group: str, key: str, non_negative: bool
    ) -> None:
        from autoware_carla_scenario.examples import run

        for value in self._bad_values(non_negative):
            cfg = _shipped([])
            cfg[group][key] = value
            with pytest.raises(run.RunConfigError) as raised:
                run.check_run_config(cfg)
            message = str(raised.value)
            assert message.startswith(f"{group}.{key} must be finite"), message
            assert "\n" not in message
            with pytest.raises(run.RunConfigError, match=rf"^case-2: {group}\.{key}"):
                run.check_batch_configs([_shipped([]), cfg], ["case-1", "case-2"])

    def test_a_negative_pose_and_a_zero_speed_or_step_are_accepted(self) -> None:
        from autoware_carla_scenario.examples import run

        cfg = TestEgoNumbersAreRefusedByTheirKey._cfg("ego", "spawn_t", -1.5)
        cfg.ego.spawn_heading = -3.0
        cfg.ego.initial_speed_kmh = 0
        cfg.entity.spawn_retry_t_step = 0.0
        cfg.entity.spawn_retry_z_step = 0.0
        ego, spawn, _ = run.build_ego_and_spawn(cfg)
        assert (spawn.t, spawn.heading) == (-1.5, -3.0)
        assert (ego.initial_speed_kmh, ego.spawn_retry_t_step) == (0.0, 0.0)
        shipped = _shipped([])
        shipped.ego.spawn_t = -1.5
        shipped.ego.initial_speed_kmh = 0
        run.check_run_config(shipped)

    def test_the_batch_check_refuses_an_npc_vehicle_speed(self) -> None:
        from autoware_carla_scenario.examples import run

        cfg = _shipped(["scenario=intersection_passing/straight_3npc"])
        run.check_run_config(cfg)
        cfg.scenario.npc_vehicles[1].initial_speed_kmh = math.nan
        with pytest.raises(
            run.RunConfigError,
            match=(
                r"^scenario\.npc_vehicles\[1\]\.initial_speed_kmh "
                "must be finite and at least 0"
            ),
        ):
            run.check_run_config(cfg)


class TestNpcVehiclesAndRetryCountAreCheckedBeforeTheRun:
    """#1984: a ``scenario.npc_vehicles`` entry without a lanelet, with one
    that is no whole number or that is no mapping at all, and an
    ``entity.spawn_retry_max_count`` that is no whole number of 0 or more, are
    refused by the batch check, naming the key, before anything starts."""

    @pytest.mark.parametrize(
        ("entries", "match"),
        [
            (
                [{"spawn_s": 5.0}],
                r"^scenario\.npc_vehicles\[0\]\.spawn_lanelet_id is required",
            ),
            (
                [{"spawn_lanelet_id": "abc"}],
                r"^scenario\.npc_vehicles\[0\]\.spawn_lanelet_id must be a whole number",
            ),
            (
                [{"spawn_lanelet_id": 1.5}],
                r"^scenario\.npc_vehicles\[0\]\.spawn_lanelet_id must be a whole number",
            ),
            (
                [{"spawn_lanelet_id": True}],
                r"^scenario\.npc_vehicles\[0\]\.spawn_lanelet_id must be a whole number",
            ),
            ([7], r"^scenario\.npc_vehicles\[0\] must be a mapping"),
            (
                [{"spawn_lanelet_id": 1}, "npc"],
                r"^scenario\.npc_vehicles\[1\] must be a mapping",
            ),
            (
                [{"spawn_lanelet_id": 1, "speed": 3}],
                r"^scenario\.npc_vehicles\[0\]: .*speed",
            ),
        ],
        ids=["missing", "text", "fraction", "bool", "number", "string", "unknown-key"],
    )
    def test_the_batch_check_refuses_a_bad_npc_vehicle(
        self, entries: list[Any], match: str
    ) -> None:
        from autoware_carla_scenario.examples import run

        cfg = _shipped(["scenario=intersection_passing/straight_3npc"])
        cfg.scenario.npc_vehicles = entries
        with pytest.raises(run.RunConfigError, match=match):
            run.check_run_config(cfg)

    def test_a_whole_float_lanelet_is_read_as_an_int(self) -> None:
        from autoware_carla_scenario import NpcVehicleConfig
        from autoware_carla_scenario.examples import run

        npc = NpcVehicleConfig(spawn_lanelet_id=12.0)  # type: ignore[arg-type]
        assert npc.spawn_lanelet_id == 12 and type(npc.spawn_lanelet_id) is int
        cfg = _shipped(["scenario=intersection_passing/straight_3npc"])
        cfg.scenario.npc_vehicles[0].spawn_lanelet_id = 12.0
        run.check_run_config(cfg)

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (2.5, "must be a whole number"),
            (True, "must be a whole number"),
            ("abc", "must be a whole number"),
            (-1, "must be >= 0"),
        ],
    )
    def test_the_retry_count_is_refused(self, value: object, message: str) -> None:
        from autoware_carla_scenario.examples import run

        match = rf"^entity\.spawn_retry_max_count {message}"
        cfg = _shipped([])
        cfg.entity.spawn_retry_max_count = value
        with pytest.raises(run.RunConfigError, match=match):
            run.check_run_config(cfg)
        with pytest.raises(ValueError, match=match):
            run.build_ego_and_spawn(
                TestEgoNumbersAreRefusedByTheirKey._cfg(
                    "entity", "spawn_retry_max_count", value
                )
            )

    def test_a_zero_or_whole_float_retry_count_is_accepted(self) -> None:
        from autoware_carla_scenario.examples import run

        for value in (0, 3.0, "4"):
            cfg = _shipped([])
            cfg.entity.spawn_retry_max_count = value
            run.check_run_config(cfg)
        ego, _, _ = run.build_ego_and_spawn(
            TestEgoNumbersAreRefusedByTheirKey._cfg(
                "entity", "spawn_retry_max_count", 3.0
            )
        )
        assert ego.spawn_retry_max_count == 3

    @pytest.mark.parametrize("value", [None, "", "  ", 7])
    def test_the_batch_check_refuses_a_server_host_that_names_nothing(
        self, value: object
    ) -> None:
        from autoware_carla_scenario.examples import run

        cfg = _shipped([])
        cfg.server.host = value
        with pytest.raises(run.RunConfigError, match=r"^server\.host must be"):
            run.check_run_config(cfg)


class TestEntityConfigNumbersMustBeFinite:
    """#1809: ``NpcVehicleConfig`` and ``VehicleEntityConfig`` (and the ego's
    and the other vehicles' configs built on it) refuse a NaN, an infinity or
    a negative speed or retry step, and ``NpcVehicleConfig`` a non-finite
    ``spawn_s``, naming the field."""

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -20.0])
    def test_npc_vehicle_config_refuses_a_speed(self, value: float) -> None:
        from autoware_carla_scenario import NpcVehicleConfig

        with pytest.raises(
            ValueError, match=r"^initial_speed_kmh must be finite and at least 0"
        ):
            NpcVehicleConfig(spawn_lanelet_id=1, initial_speed_kmh=value)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_npc_vehicle_config_refuses_a_spawn_s(self, value: float) -> None:
        from autoware_carla_scenario import NpcVehicleConfig

        with pytest.raises(ValueError, match=r"^spawn_s must be finite"):
            NpcVehicleConfig(spawn_lanelet_id=1, spawn_s=value)

    def test_npc_vehicle_config_accepts_a_still_npc_and_a_negative_s(self) -> None:
        from autoware_carla_scenario import NpcVehicleConfig

        npc = NpcVehicleConfig(spawn_lanelet_id=1, spawn_s=-2.0, initial_speed_kmh=0)
        assert (npc.spawn_s, npc.initial_speed_kmh) == (-2.0, 0)

    def test_the_scenario_config_names_the_npc(self) -> None:
        from autoware_carla_scenario.examples.configs import IntersectionPassingConfig

        with pytest.raises(
            ValueError,
            match=r"^scenario\.npc_vehicles\[0\]\.initial_speed_kmh must be",
        ):
            IntersectionPassingConfig(
                npc_vehicles=[{"spawn_lanelet_id": 1, "initial_speed_kmh": -5.0}]  # type: ignore[list-item]
            )

    @pytest.mark.parametrize(
        "field", ["initial_speed_kmh", "spawn_retry_t_step", "spawn_retry_z_step"]
    )
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.5])
    def test_vehicle_configs_refuse_it(self, field: str, value: float) -> None:
        from autoware_carla_scenario import (
            EmergencyVehicleEntityConfig,
            MotorcycleEntityConfig,
            VehicleEntityConfig,
        )

        spawn = SpawnTransform(carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)))
        match = rf"^{field} must be finite and at least 0, got "
        bad: dict[str, Any] = {field: value}
        with pytest.raises(ValueError, match=match):
            VehicleEntityConfig(role_name="npc", spawn_location=spawn, **bad)
        with pytest.raises(ValueError, match=match):
            EgoConfig(spawn_location=spawn, **bad)
        with pytest.raises(ValueError, match=match):
            MotorcycleEntityConfig(role_name="m", spawn_location=spawn, **bad)
        with pytest.raises(ValueError, match=match):
            EmergencyVehicleEntityConfig(role_name="e", spawn_location=spawn, **bad)

    def test_vehicle_configs_accept_zero(self) -> None:
        from autoware_carla_scenario import VehicleEntityConfig

        spawn = SpawnTransform(carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)))
        config = VehicleEntityConfig(
            role_name="npc",
            spawn_location=spawn,
            initial_speed_kmh=0.0,
            spawn_retry_t_step=0.0,
            spawn_retry_z_step=0.0,
        )
        assert config.initial_speed_kmh == 0.0


# ---------------------------------------------------------------------------
# Tests for build_goal_pose helper
# ---------------------------------------------------------------------------


class TestBuildGoalPose:
    """``ego.goal_lanelet_id`` is what routes an ego that plans for itself."""

    def test_a_configured_goal_becomes_a_lanelet2_pose(self) -> None:
        from autoware_carla_scenario import Lanelet2Pose
        from autoware_carla_scenario.examples.run import build_goal_pose

        cfg = OmegaConf.create({"ego": {"goal_lanelet_id": 265, "goal_s": 12.5}})
        goal = build_goal_pose(cfg)

        assert isinstance(goal, Lanelet2Pose)
        assert goal.lanelet_id == 265
        assert goal.s == 12.5

    def test_no_goal_configured_returns_none(self) -> None:
        # The ordinary case: only an ego that routes itself needs a goal, so an
        # unset one must not become lanelet 0.
        from autoware_carla_scenario.examples.run import build_goal_pose

        cfg = OmegaConf.create({"ego": {"spawn_lanelet_id": 242}})
        assert build_goal_pose(cfg) is None

    def test_missing_ego_group_returns_none(self) -> None:
        from autoware_carla_scenario.examples.run import build_goal_pose

        cfg = OmegaConf.create({"scenario": {"name": "whatever"}})
        assert build_goal_pose(cfg) is None

    def test_goal_s_defaults_to_the_start_of_the_lanelet(self) -> None:
        from autoware_carla_scenario.examples.run import build_goal_pose

        cfg = OmegaConf.create({"ego": {"goal_lanelet_id": 265}})
        goal = build_goal_pose(cfg)

        assert goal is not None
        assert goal.s == 0.0


def test_run_batch_gives_each_scenario_its_own_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    built: list[object] = []

    def fake_build(cfg: object, **_: object) -> tuple[None, object]:
        built.append(object())
        return None, built[-1]

    added: list[tuple[object, float | None]] = []

    class FakeQueue:
        def __init__(self, **_: object) -> None:
            pass

        def add(
            self,
            scenario: object,
            *,
            timeout_seconds: float | None = None,
            case: object = None,
            file_stem: str | None = None,
        ) -> None:
            added.append((scenario, timeout_seconds))

        def __enter__(self) -> "FakeQueue":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def run_all(self, **_: object) -> list[object]:
            return []

        def errored_indices(self) -> list[int]:
            return []

    monkeypatch.setattr(run, "build_scenario", fake_build)
    monkeypatch.setattr(run, "ScenarioQueue", FakeQueue)
    monkeypatch.setattr(
        run,
        "resolve_map_paths",
        lambda _m: SimpleNamespace(
            install_xodr=None,
            overwrite_xodr=False,
            opendrive_path=None,
            lanelet2_path=None,
            name="M",
            projector_type=None,
        ),
    )
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: None)
    monkeypatch.setattr(run, "_print_summary", lambda *a, **k: True)

    with pytest.raises(SystemExit):
        run.run_batch(["lane_change/left", "lane_change_fail/right"], [])

    assert added == [(built[0], 10.0), (built[1], 1.5)]


def test_a_batch_lets_go_of_each_scenario_once_it_has_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A batch keeps each case's result, not the finished scenario: one kept to
    the end keeps everything its run built (entities, conditions, the ego's
    driver), and a long batch grows with every case."""
    import gc
    import weakref
    from types import SimpleNamespace

    from autoware_carla_scenario import ScenarioResult, scenario_queue
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")

    class _Built:
        """What a built scenario stands for here; weakly referenceable."""

    built: list[weakref.ref[_Built]] = []

    def fake_build(cfg: object, **_: object) -> tuple[None, object]:
        scenario = _Built()
        built.append(weakref.ref(scenario))
        return None, scenario

    alive_while_running: list[list[int]] = []

    class FakeRunner:
        """Keeps no scenario (a MagicMock keeps every argument it is given)."""

        def __init__(self, *_: object, **__: object) -> None:
            self.xodr_path = None

        def load_map_by_name(self, _name: str) -> None:
            pass

        def reset_world(self, scenario_name: str = "") -> None:
            pass

        def run_scenario(self, scenario: object, **_: object) -> ScenarioResult:
            gc.collect()
            index = next(k for k, ref in enumerate(built) if ref() is scenario)
            alive_while_running.append(
                [k for k, ref in enumerate(built[:index]) if ref() is not None]
            )
            return ScenarioResult(passed=True, message="ok", elapsed_seconds=0.0)

    summarised: list[object] = []

    def fake_summary(_labels: object, results: list[object], **_: object) -> bool:
        summarised.extend(results)
        return True

    monkeypatch.setattr(run, "build_scenario", fake_build)
    monkeypatch.setattr(scenario_queue, "CarlaServerManager", MagicMock())
    monkeypatch.setattr(scenario_queue, "ScenarioRunner", FakeRunner)
    monkeypatch.setattr(
        run,
        "resolve_map_paths",
        lambda _m: SimpleNamespace(
            install_xodr=None,
            overwrite_xodr=False,
            opendrive_path=None,
            lanelet2_path=None,
            name="M",
            projector_type=None,
        ),
    )
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: tmp_path)
    monkeypatch.setattr(run, "_print_summary", fake_summary)

    with pytest.raises(SystemExit):
        run.run_batch(
            ["lane_change/left", "lane_change_fail/right", "lane_change/left"], []
        )

    # While each case ran, every case before it was already gone ...
    assert alive_while_running == [[], [], []]
    # ... and the summary still has every result.
    assert len(summarised) == 3


def test_a_batch_goes_on_past_a_case_that_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A case that raises on every attempt is reported as errored, the cases
    after it still run, and the batch exits non-zero."""
    import json
    from types import SimpleNamespace

    from autoware_carla_scenario import ScenarioResult, scenario_queue
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    built: list[object] = []

    def fake_build(cfg: object, **_: object) -> tuple[None, object]:
        built.append(object())
        return None, built[-1]

    ran: list[int] = []

    class FakeRunner:
        def __init__(self, *_: object, **__: object) -> None:
            self.xodr_path = None

        def load_map_by_name(self, _name: str) -> None:
            pass

        def reset_world(self, scenario_name: str = "") -> None:
            pass

        def run_scenario(self, scenario: object, **_: object) -> ScenarioResult:
            index = next(k for k, b in enumerate(built) if b is scenario)
            ran.append(index)
            if index == 1:
                raise RuntimeError("ground_projection returned no hit")
            return ScenarioResult(passed=True, message="ok", elapsed_seconds=1.0)

    monkeypatch.setattr(run, "build_scenario", fake_build)
    monkeypatch.setattr(scenario_queue, "CarlaServerManager", MagicMock())
    monkeypatch.setattr(scenario_queue, "ScenarioRunner", FakeRunner)
    monkeypatch.setattr(
        run,
        "resolve_map_paths",
        lambda _m: SimpleNamespace(
            install_xodr=None,
            overwrite_xodr=False,
            opendrive_path=None,
            lanelet2_path=None,
            name="M",
            projector_type=None,
        ),
    )
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: tmp_path)
    monkeypatch.setattr(scenario_queue, "_wait_with_progress", lambda *_, **__: None)

    with pytest.raises(SystemExit) as exited:
        run.run_batch(
            ["lane_change/left", "lane_change_fail/right", "lane_change/left"], []
        )

    assert exited.value.code == 1
    assert ran == [0, 1, 2]
    out = capsys.readouterr().out
    assert "[ERROR] lane_change_fail/right" in out
    assert "ground_projection returned no hit" in out
    assert out.count("[PASS] lane_change/left") == 2
    assert "2/3 scenarios passed, 1 errored" in out
    entries = json.loads((tmp_path / "batch_results.json").read_text())
    assert [e.get("error", False) for e in entries] == [False, True, False]
    assert [e["passed"] for e in entries] == [True, False, True]


def test_a_batch_goes_on_past_a_reset_that_fails_after_an_errored_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1941: the reset after an errored case that fails is the failure of
    the next case's attempt: the errored case is reported, and the batch's
    summary lists every case, none pending."""
    import json
    from types import SimpleNamespace

    from autoware_carla_scenario import ScenarioResult, scenario_queue
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    built: list[object] = []

    def fake_build(cfg: object, **_: object) -> tuple[None, object]:
        built.append(object())
        return None, built[-1]

    ran: list[int] = []
    resets: list[str] = []

    class FakeRunner:
        def __init__(self, *_: object, **__: object) -> None:
            self.xodr_path = None

        def load_map_by_name(self, _name: str) -> None:
            pass

        def reset_world(self, scenario_name: str = "") -> None:
            resets.append(scenario_name)
            if len(resets) == 2:
                raise RuntimeError("load_world timed out")

        def run_scenario(self, scenario: object, **_: object) -> ScenarioResult:
            index = next(k for k, b in enumerate(built) if b is scenario)
            ran.append(index)
            if index == 1:
                raise RuntimeError("ground_projection returned no hit")
            return ScenarioResult(passed=True, message="ok", elapsed_seconds=1.0)

    monkeypatch.setattr(run, "build_scenario", fake_build)
    monkeypatch.setattr(scenario_queue, "CarlaServerManager", MagicMock())
    monkeypatch.setattr(scenario_queue, "ScenarioRunner", FakeRunner)
    monkeypatch.setattr(
        run,
        "resolve_map_paths",
        lambda _m: SimpleNamespace(
            install_xodr=None,
            overwrite_xodr=False,
            opendrive_path=None,
            lanelet2_path=None,
            name="M",
            projector_type=None,
        ),
    )
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: tmp_path)
    monkeypatch.setattr(scenario_queue, "_wait_with_progress", lambda *_, **__: None)

    with pytest.raises(SystemExit) as exited:
        run.run_batch(
            ["lane_change/left", "lane_change_fail/right", "lane_change/left"], []
        )

    assert exited.value.code == 1
    # The third case's only attempt was the failed reset: it never ran.
    assert ran == [0, 1]
    out = capsys.readouterr().out
    assert "[ERROR] lane_change_fail/right" in out
    assert "ground_projection returned no hit" in out
    assert "load_world timed out" in out
    entries = json.loads((tmp_path / "batch_results.json").read_text())
    assert not any(e.get("pending", False) for e in entries)
    assert [e.get("error", False) for e in entries] == [False, True, True]
    assert [e["passed"] for e in entries] == [True, False, False]
