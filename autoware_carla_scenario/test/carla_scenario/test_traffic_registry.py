"""Selecting a traffic backend by name, including one from another package.

The registry is the whole of what makes the seam pluggable, so the tests here
are about names: what a name resolves to, what an unknown one says, and how a
config that predates the ``traffic`` group still picks the right backend with
the right port.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.constants import DEFAULT_TM_PORT

from autoware_carla_scenario.traffic import (
    NullTrafficBackend,
    TrafficBackend,
    available_backends,
    build_backend,
    get_backend_factory,
    register_backend,
    register_builtin_backends,
    unregister_backend,
)
from autoware_carla_scenario.traffic.config import TrafficConfig
from autoware_carla_scenario.traffic.traffic_manager import TrafficManagerBackend


class _Fake(TrafficBackend):
    name = "fake"

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.options = dict(options)


@pytest.fixture
def registered_fake():
    """A backend from somewhere else, removed again so it cannot leak."""
    register_backend("fake", _Fake)
    yield "fake"
    unregister_backend("fake")


class TestTheRegistry:
    def test_the_built_ins_are_there(self) -> None:
        assert {"traffic_manager", "none"} <= set(available_backends())

    def test_a_name_resolves_to_its_backend(self, registered_fake: str) -> None:
        assert isinstance(build_backend(registered_fake, {}), _Fake)

    def test_options_reach_the_backend_verbatim(self, registered_fake: str) -> None:
        """The shared config knows a backend's options only as a mapping.

        That is what keeps a third-party backend from needing a field in a
        dataclass this package owns.
        """
        backend = build_backend(registered_fake, {"anything": 1})
        assert isinstance(backend, _Fake)
        assert backend.options == {"anything": 1}

    def test_an_unknown_name_says_what_is_registered(self) -> None:
        with pytest.raises(ValueError, match="Unknown traffic backend") as error:
            get_backend_factory("__nonexistent__")
        assert "traffic_manager" in str(error.value)

    def test_a_backend_can_be_replaced(self, registered_fake: str) -> None:
        """A package overriding a built-in means to, and gets to."""
        register_backend("traffic_manager", _Fake)
        try:
            assert isinstance(build_backend("traffic_manager", {}), _Fake)
        finally:
            register_builtin_backends()
        assert isinstance(build_backend("traffic_manager", {}), TrafficManagerBackend)

    def test_the_entry_point_walk_runs_once(self, monkeypatch) -> None:
        """A --multirun sweep builds a backend per job, in one process.

        Re-walking every installed distribution each time would also re-run
        every third-party registration callable, which is unbounded work this
        package does not control.
        """
        from autoware_carla_scenario.traffic import registry as traffic_registry

        walks = []
        monkeypatch.setattr(traffic_registry, "_plugins_loaded", False)
        monkeypatch.setattr(
            "autoware_carla_scenario.registry.load_entry_point_plugins",
            lambda group, what, *args, **kwargs: walks.append(group),
        )

        traffic_registry.load_traffic_backend_plugins()
        traffic_registry.load_traffic_backend_plugins()
        assert walks == [traffic_registry.TRAFFIC_BACKEND_ENTRY_POINT_GROUP]


def _entry_point(name: str, register: Any, dist: str | None = None) -> Any:
    """An entry point of the distribution *dist* that calls *register*."""
    entry_point = MagicMock()
    entry_point.name = name
    entry_point.value = f"{name}:register"
    entry_point.dist.name = dist or name
    entry_point.load.return_value = register
    return entry_point


class TestPluginCollisions:
    """Plugins that take a backend name already registered are reported (#2034)."""

    @staticmethod
    def _load(monkeypatch: pytest.MonkeyPatch, entry_points: list[Any]) -> None:
        from autoware_carla_scenario.traffic import registry as traffic_registry

        monkeypatch.setattr(
            traffic_registry, "_BACKENDS", dict(traffic_registry._BACKENDS)
        )
        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda *, group: list(entry_points)
        )
        monkeypatch.setattr(traffic_registry, "_plugins_loaded", False)
        traffic_registry.load_traffic_backend_plugins()

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    def test_overriding_a_built_in_is_reported_softly(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                [
                    _entry_point(
                        "my_tm", lambda: register_backend("traffic_manager", _Fake)
                    )
                ],
            )
            assert isinstance(build_backend("traffic_manager", {}), _Fake)
        (message,) = self._warnings(caplog)
        assert "'my_tm' (my_tm:register) overrides 'traffic_manager'" in message
        assert "now selects the plugin's traffic backend" in message

    def test_two_plugins_with_one_name_are_reported_and_the_last_wins(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _Other(_Fake):
            pass

        from autoware_carla_scenario.traffic import registry as traffic_registry

        builtins = dict(traffic_registry._BACKENDS)
        first = _entry_point("sumo_a", lambda: register_backend("dup_sim", _Fake))
        second = _entry_point("sumo_b", lambda: register_backend("dup_sim", _Other))
        for order in ([first, second], [second, first]):
            caplog.clear()
            monkeypatch.setattr(traffic_registry, "_BACKENDS", dict(builtins))
            with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
                self._load(monkeypatch, order)
                # Sorted by distribution name: sumo_b loads last, whichever
                # order the environment lists them in.
                assert isinstance(build_backend("dup_sim", {}), _Other)
            (message,) = self._warnings(caplog)
            assert "'sumo_a' (sumo_a:register)" in message
            assert "'sumo_b' (sumo_b:register)" in message
            assert "'sumo_b' loaded last" in message

    def test_a_plugin_adding_a_new_backend_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="autoware_carla_scenario.registry"):
            self._load(
                monkeypatch,
                [_entry_point("new", lambda: register_backend("brand_new", _Fake))],
            )
        assert not self._warnings(caplog)


class TestConcurrentPluginLoading:
    """Callers that make the first call at the same time (#2037)."""

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from autoware_carla_scenario.traffic import registry as traffic_registry

        monkeypatch.setattr(
            traffic_registry, "_BACKENDS", dict(traffic_registry._BACKENDS)
        )
        monkeypatch.setattr(traffic_registry, "_plugins_loaded", False)

    def test_every_first_caller_returns_with_the_backends_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.traffic import registry as traffic_registry

        started = threading.Event()
        release = threading.Event()

        def register() -> None:
            started.set()
            assert release.wait(10)
            register_backend("slow_sim", _Fake)

        entry_point = _entry_point("slow", register)
        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda *, group: [entry_point]
        )
        seen: dict[str, bool] = {}

        def call(caller: str) -> None:
            traffic_registry.load_traffic_backend_plugins()
            seen[caller] = "slow_sim" in available_backends()

        first = threading.Thread(target=call, args=("first",))
        second = threading.Thread(target=call, args=("second",))
        first.start()
        assert started.wait(10)
        second.start()
        second.join(0.2)
        assert second.is_alive()
        release.set()
        first.join(10)
        second.join(10)
        assert not first.is_alive() and not second.is_alive()
        assert seen == {"first": True, "second": True}
        entry_point.load.assert_called_once()

    def test_a_plugin_that_loads_the_plugins_again_does_not_deadlock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nor one that loads the scenario plugins, which share the lock."""
        from autoware_carla_scenario import registry
        from autoware_carla_scenario.traffic import registry as traffic_registry

        monkeypatch.setattr(registry, "_plugins_loaded", True)

        def register() -> None:
            traffic_registry.load_traffic_backend_plugins()
            registry.load_scenario_plugins()
            register_backend("reentrant_sim", _Fake)

        entry_point = _entry_point("reentrant", register)
        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda *, group: [entry_point]
        )
        loader = threading.Thread(
            target=traffic_registry.load_traffic_backend_plugins, daemon=True
        )
        loader.start()
        loader.join(10)
        assert not loader.is_alive()
        assert "reentrant_sim" in available_backends()
        entry_point.load.assert_called_once()


class TestTheConfig:
    def test_the_default_is_the_traffic_manager(self) -> None:
        """Every scenario written before this group existed keeps its traffic."""
        assert TrafficConfig().backend == "traffic_manager"

    def test_an_unknown_key_is_refused(self) -> None:
        """A typo that silently left a default in place is the failure guarded."""
        with pytest.raises(ValueError, match="Unknown TrafficConfig key"):
            TrafficConfig.from_mapping({"backedn": "sumo"})


class TestBuildingFromAHydraConfig:
    """``build_traffic_backend`` is to traffic what ``build_ego_entity`` is to the ego."""

    @staticmethod
    def _build(cfg: dict) -> TrafficBackend:
        from autoware_carla_scenario.examples.run import build_traffic_backend

        return build_traffic_backend(OmegaConf.create(cfg))

    def test_no_traffic_group_at_all_is_the_traffic_manager(self) -> None:
        """An exported scenario package composed before the group existed.

        The port is deliberately not the framework default: a config that never
        composes ``conf/traffic/traffic_manager.yaml`` cannot be served by the
        interpolation in it, so dropping the legacy key here would silently move
        such a run onto a different TrafficManager while still looking right.
        """
        backend = self._build({"traffic_manager": {"port": 9000}})
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == 9000

    def test_a_config_that_names_no_port_at_all_gets_the_default(self) -> None:
        backend = self._build({})
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == DEFAULT_TM_PORT

    def test_a_named_backend_is_built(self) -> None:
        backend = self._build({"traffic": {"backend": "none", "options": {}}})
        assert isinstance(backend, NullTrafficBackend)

    def test_an_unknown_backend_fails_before_anything_is_spawned(self) -> None:
        with pytest.raises(ValueError, match="Unknown traffic backend"):
            self._build({"traffic": {"backend": "__nonexistent__", "options": {}}})


class TestSelectingFromTheRealConfig:
    """The four selection paths, composed the way ``scenario`` composes them.

    The legacy-port bridge is an interpolation in
    ``conf/traffic/traffic_manager.yaml``, so a test that hand-builds a config
    dict would pin the Python that no longer does the work.  These compose the
    shipped config group instead.
    """

    @staticmethod
    def _build(overrides: list[str]) -> TrafficBackend:
        from hydra import compose, initialize_config_dir

        from autoware_carla_scenario.examples import conf as conf_package
        from autoware_carla_scenario.examples.run import build_traffic_backend

        conf_dir = str(Path(conf_package.__file__).parent.resolve())
        with initialize_config_dir(config_dir=conf_dir, version_base=None):
            return build_traffic_backend(
                compose(config_name="config", overrides=overrides)
            )

    def test_the_default_is_the_traffic_manager_on_the_framework_port(self) -> None:
        backend = self._build([])
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == DEFAULT_TM_PORT

    def test_the_legacy_port_still_decides(self) -> None:
        """``traffic_manager.port`` is where the port lived, and CI still sets it."""
        backend = self._build(["traffic_manager.port=9000"])
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == 9000

    def test_an_explicit_option_wins_over_the_legacy_port(self) -> None:
        backend = self._build(
            ["traffic_manager.port=9000", "traffic.options.port=8123"]
        )
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == 8123

    def test_a_group_selects_the_backend(self) -> None:
        assert isinstance(self._build(["traffic=none"]), NullTrafficBackend)

    def test_the_traffic_parameters_reach_the_backend(self) -> None:
        from autoware_carla_scenario.traffic import Spread

        backend = self._build(
            [
                "traffic.vehicles.speed_factor.mean=0.8",
                "traffic.vehicles.speed_factor.std=0.1",
                "traffic.vehicles.ignore_lights=0.1",
            ]
        )
        assert backend.vehicle_parameters.speed_factor == Spread(0.8, 0.1)
        # A plain number is a spread of no width.
        assert backend.vehicle_parameters.ignore_lights == Spread(0.1)

    def test_a_parameter_the_backend_cannot_honour_fails_the_run(self) -> None:
        from autoware_carla_scenario.traffic import TrafficParameterError

        with pytest.raises(
            TrafficParameterError,
            match="reaction_time_s can only be used with the 'sumo' backend",
        ):
            self._build(["traffic.vehicles.reaction_time_s=1.5"])

    def test_every_group_declares_every_parameter(self) -> None:
        """So that `traffic.vehicles.x=y` overrides work under struct mode."""
        from hydra import compose, initialize_config_dir

        from autoware_carla_scenario.examples import conf as conf_package
        from autoware_carla_scenario.traffic import (
            PEDESTRIAN_PARAMETER_BACKENDS,
            VEHICLE_PARAMETER_BACKENDS,
        )

        conf_dir = str(Path(conf_package.__file__).parent.resolve())
        for group in ("traffic_manager", "sumo", "none"):
            with initialize_config_dir(config_dir=conf_dir, version_base=None):
                cfg = compose(config_name="config", overrides=[f"traffic={group}"])
            assert set(cfg.traffic.vehicles) == set(VEHICLE_PARAMETER_BACKENDS), group
            assert set(cfg.traffic.pedestrians) == set(
                PEDESTRIAN_PARAMETER_BACKENDS
            ), group
