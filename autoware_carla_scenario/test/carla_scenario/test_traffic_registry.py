"""Selecting a traffic backend by name, including one from another package.

The registry is the whole of what makes the seam pluggable, so the tests here
are about names: what a name resolves to, what an unknown one says, and how a
config that predates the ``traffic`` group still picks the right backend with
the right port.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.traffic import (
    NullTrafficBackend,
    TrafficBackend,
    available_backends,
    build_backend,
    get_backend_factory,
    register_backend,
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
            from autoware_carla_scenario.traffic import _build_traffic_manager

            register_backend("traffic_manager", _build_traffic_manager)
        assert isinstance(build_backend("traffic_manager", {}), TrafficManagerBackend)


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
        """An exported scenario package composed before the group existed."""
        backend = self._build({"traffic_manager": {"port": 8100}})
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == 8100

    def test_the_legacy_port_still_decides(self) -> None:
        """``traffic_manager.port`` is where the port lived, and CI still sets it."""
        backend = self._build(
            {
                "traffic_manager": {"port": 9000},
                "traffic": {"backend": "traffic_manager", "options": {"port": None}},
            }
        )
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == 9000

    def test_an_explicit_option_wins_over_the_legacy_port(self) -> None:
        backend = self._build(
            {
                "traffic_manager": {"port": 9000},
                "traffic": {"backend": "traffic_manager", "options": {"port": 8123}},
            }
        )
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == 8123

    def test_a_named_backend_is_built(self) -> None:
        backend = self._build({"traffic": {"backend": "none", "options": {}}})
        assert isinstance(backend, NullTrafficBackend)

    def test_an_unknown_backend_fails_before_anything_is_spawned(self) -> None:
        with pytest.raises(ValueError, match="Unknown traffic backend"):
            self._build({"traffic": {"backend": "__nonexistent__", "options": {}}})
