"""An NPC motorcycle: a vehicle entity ridden on a motorcycle's blueprint.

:class:`MotorcycleEntity` picks the motorcycle of the server the run is on
when its config names none, refuses a blueprint that is not a motorcycle's,
and is otherwise every NPC vehicle: the TrafficManager puts it on autopilot
and takes its manoeuvres like any vehicle's.  How SUMO takes it on (as a
vehicle of its ``motorcycle`` class) is in ``test_traffic_sumo.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

import autoware_carla_scenario as acs
from autoware_carla_scenario.entity import (
    ANY_MOTORCYCLE,
    MotorcycleEntity,
    MotorcycleEntityConfig,
    SpawnPointIndex,
    VehicleEntity,
    VehicleEntityConfig,
)
from autoware_carla_scenario.entity import vehicle_entity
from autoware_carla_scenario.entity.registry import (
    clear_entities,
    find_vehicle_entity,
    register_entity,
)
from autoware_carla_scenario.entity.vehicle_entity import (
    is_motorcycle_blueprint,
    motorcycle_blueprints,
    resolve_motorcycle_type,
)
from autoware_carla_scenario.entity_role import EntityRole
from autoware_carla_scenario.traffic.config import TrafficManagerBackendConfig
from autoware_carla_scenario.traffic.driven import BackendDriven
from autoware_carla_scenario.traffic.traffic_manager import TrafficManagerBackend

from .test_traffic_backend import _Actor, _Client, _World


class _Attribute:
    def __init__(self, value: str) -> None:
        self._value = value

    def as_str(self) -> str:
        return self._value


class _Blueprint:
    def __init__(self, blueprint_id: str, base_type: str | None) -> None:
        self.id = blueprint_id
        self._base_type = base_type

    def has_attribute(self, name: str) -> bool:
        return name == "base_type" and self._base_type is not None

    def get_attribute(self, name: str) -> _Attribute:
        assert self._base_type is not None
        return _Attribute(self._base_type)


class _Library:
    def __init__(self, blueprints: List[_Blueprint]) -> None:
        self._blueprints = blueprints

    def filter(self, pattern: str) -> List[_Blueprint]:
        assert pattern == "vehicle.*"
        return list(self._blueprints)


class _BlueprintWorld:
    def __init__(self, blueprints: List[_Blueprint]) -> None:
        self._library = _Library(blueprints)

    def get_blueprint_library(self) -> _Library:
        return self._library


def _world() -> Any:
    return _BlueprintWorld(
        [
            _Blueprint("vehicle.mini.cooper", "car"),
            _Blueprint("vehicle.yamaha.yzf", "motorcycle"),
            _Blueprint("vehicle.kawasaki.ninja", "Motorcycle"),
            _Blueprint("vehicle.bh.crossbike", "bicycle"),
            _Blueprint("vehicle.old.server", None),
        ]
    )


def _config(**kwargs: Any) -> MotorcycleEntityConfig:
    return MotorcycleEntityConfig(
        role_name="rider", spawn_location=SpawnPointIndex(0), **kwargs
    )


class TestTheBlueprint:
    def test_a_motorcycle_is_a_blueprint_whose_base_type_says_so(self) -> None:
        assert is_motorcycle_blueprint(_Blueprint("vehicle.a.b", "motorcycle"))
        assert is_motorcycle_blueprint(_Blueprint("vehicle.a.b", "Motorcycle"))
        assert not is_motorcycle_blueprint(_Blueprint("vehicle.a.b", "bicycle"))
        # An older server, with no base type: not known to be one.
        assert not is_motorcycle_blueprint(_Blueprint("vehicle.a.b", None))

    def test_the_motorcycles_of_a_world_are_listed_by_id(self) -> None:
        assert motorcycle_blueprints(_world()) == [
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
        ]

    def test_none_named_is_the_first_motorcycle_of_the_server(self) -> None:
        assert resolve_motorcycle_type(_world(), ANY_MOTORCYCLE) == (
            "vehicle.kawasaki.ninja"
        )

    def test_a_motorcycle_named_is_the_one_ridden(self) -> None:
        assert resolve_motorcycle_type(_world(), "vehicle.yamaha.yzf") == (
            "vehicle.yamaha.yzf"
        )

    @pytest.mark.parametrize(
        "name", ["vehicle.mini.cooper", "vehicle.bh.crossbike", "vehicle.no.such"]
    )
    def test_a_blueprint_that_is_no_motorcycle_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="not a motorcycle") as raised:
            resolve_motorcycle_type(_world(), name)
        assert "vehicle.yamaha.yzf" in str(raised.value)

    def test_a_server_without_motorcycles_is_refused(self) -> None:
        world = _BlueprintWorld([_Blueprint("vehicle.mini.cooper", "car")])
        with pytest.raises(ValueError, match="no motorcycle blueprint"):
            resolve_motorcycle_type(world, ANY_MOTORCYCLE)


class TestTheEntity:
    def test_it_is_an_npc_vehicle(self) -> None:
        rider = MotorcycleEntity(_config())
        assert isinstance(rider, VehicleEntity)
        assert isinstance(rider, BackendDriven)
        assert rider.role_name == "rider"

    def test_the_config_names_no_blueprint_by_default(self) -> None:
        config = _config()
        assert isinstance(config, VehicleEntityConfig)
        assert config.vehicle_type == ANY_MOTORCYCLE
        assert config.initial_speed_kmh == 0.0

    def test_the_config_stores_the_role_as_its_string(self) -> None:
        config = MotorcycleEntityConfig(
            role_name=EntityRole.npc(4),  # type: ignore[arg-type]
            spawn_location=SpawnPointIndex(0),
        )
        assert type(config.role_name) is str
        assert config.role_name == "npc4"

    def test_it_is_spawned_as_the_motorcycle_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: List[Any] = []
        actor = SimpleNamespace(id=7)

        def spawn(
            world: Any, vehicle_type: str, role_name: str, *a: Any, **k: Any
        ) -> Any:
            spawned.append((vehicle_type, role_name))
            return actor

        monkeypatch.setattr(vehicle_entity, "spawn_vehicle_actor", spawn)
        rider = MotorcycleEntity(_config(initial_speed_kmh=20.0))
        assert rider.spawn(_world()) is actor
        assert spawned == [("vehicle.kawasaki.ninja", "rider")]
        assert rider.vehicle_type == "vehicle.kawasaki.ninja"
        assert rider.actor is actor
        assert rider.initial_speed_kmh == 20.0

    def test_a_car_blueprint_is_refused_before_anything_is_spawned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            vehicle_entity,
            "spawn_vehicle_actor",
            lambda *a, **k: pytest.fail("spawned"),
        )
        rider = MotorcycleEntity(_config(vehicle_type="vehicle.mini.cooper"))
        with pytest.raises(ValueError, match="not a motorcycle"):
            rider.spawn(_world())
        assert rider.actor is None

    def test_spawning_after_the_warm_up_is_refused_as_any_vehicle_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vehicle_entity, "_warmup_done", True)
        with pytest.raises(RuntimeError, match="warm-up"):
            MotorcycleEntity(_config()).spawn(_world())

    def test_it_is_registered_as_a_vehicle(self) -> None:
        clear_entities()
        rider = MotorcycleEntity(_config())
        try:
            register_entity("rider", rider)
            assert find_vehicle_entity("rider") is rider
        finally:
            clear_entities()

    def test_the_package_exports_it(self) -> None:
        assert acs.MotorcycleEntity is MotorcycleEntity
        assert acs.MotorcycleEntityConfig is MotorcycleEntityConfig


class TestTheTrafficManagerRidesIt:
    """A motorcycle is a vehicle to the TrafficManager: put on autopilot with
    the rest and given its manoeuvres as any vehicle is."""

    def _ridden(self) -> tuple[TrafficManagerBackend, MotorcycleEntity, _Actor, Any]:
        client = _Client()
        backend = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=8123), client=client
        )
        bike = _Actor(5)
        bike.attributes = {"role_name": "rider", "base_type": "motorcycle"}
        rider = MotorcycleEntity(_config())
        rider._vehicle = bike  # type: ignore[assignment]
        rider.set_traffic_backend(backend)
        backend.adopt(rider)
        backend.start(_World([_Actor(1), bike]))
        return backend, rider, bike, client.tm

    def test_it_is_put_on_autopilot(self) -> None:
        backend, _rider, bike, _tm = self._ridden()
        try:
            assert bike.autopilot == [(True, 8123)]
        finally:
            backend.close()

    def test_its_desired_speed_goes_to_the_traffic_manager(self) -> None:
        backend, rider, bike, tm = self._ridden()
        try:
            rider.set_desired_speed(_World([bike]), 35.0)
            assert tm.desired_speeds[bike.id] == [35.0]
        finally:
            backend.close()

    def test_released_and_resumed_as_any_vehicle(self) -> None:
        backend, rider, bike, _tm = self._ridden()
        try:
            rider.release_from_traffic(_World([bike]))
            rider.resume_traffic(_World([bike]))
            assert bike.autopilot == [(True, 8123), (False, 8123), (True, 8123)]
        finally:
            backend.close()
