"""An NPC emergency vehicle: a vehicle entity on an emergency vehicle's
blueprint, which can be put on a call.

:class:`EmergencyVehicleEntity` picks the emergency vehicle of the server the
run is on when its config names none and refuses a blueprint that is not one.
On a call its siren lights (``VehicleLightState.Special1``) are on and the
run's traffic backend drives it as one: the TrafficManager has it run red
lights and stop signs.  How SUMO takes it on (as a vehicle of its
``emergency`` class with a bluelight device) is in ``test_traffic_sumo.py``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest
import typesafe_carla.carla as carla

import autoware_carla_scenario as acs
from autoware_carla_scenario.actions import EmergencyAction
from autoware_carla_scenario.control_domain import ControlDomain
from autoware_carla_scenario.entity import (
    ANY_EMERGENCY_VEHICLE,
    EmergencyVehicleEntity,
    EmergencyVehicleEntityConfig,
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
    emergency_vehicle_blueprints,
    is_emergency_vehicle_blueprint,
    resolve_emergency_vehicle_type,
)
from autoware_carla_scenario.traffic.base import DrivenEntity, NullTrafficBackend
from autoware_carla_scenario.traffic.config import TrafficManagerBackendConfig
from autoware_carla_scenario.traffic.context import TrafficContext
from autoware_carla_scenario.traffic.parameters import Spread, VehicleParameters
from autoware_carla_scenario.traffic.traffic_manager import (
    EMERGENCY_IGNORE_PCT,
    TrafficManagerBackend,
)
from autoware_carla_scenario.utils.vehicles import (
    is_emergency_vehicle_type,
    set_siren_lights,
    with_siren_lights,
)

from .test_traffic_backend import _Actor, _Client, _TrafficManager, _World

_SIREN = int(carla.VehicleLightState.Special1)
_LOW_BEAM = int(carla.VehicleLightState.LowBeam)
_LEFT = int(carla.VehicleLightState.LeftBlinker)


class _Attribute:
    def __init__(self, value: str) -> None:
        self._value = value

    def as_str(self) -> str:
        return self._value


class _Blueprint:
    def __init__(self, blueprint_id: str, special_type: Optional[str]) -> None:
        self.id = blueprint_id
        self._special_type = special_type

    def has_attribute(self, name: str) -> bool:
        return name == "special_type" and self._special_type is not None

    def get_attribute(self, name: str) -> _Attribute:
        assert self._special_type is not None
        return _Attribute(self._special_type)


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
            _Blueprint("vehicle.mini.cooper", ""),
            _Blueprint("vehicle.ford.ambulance", "emergency"),
            _Blueprint("vehicle.dodge.charger_police", "Emergency"),
            _Blueprint("vehicle.mercedes.sprinter", "taxi"),
            _Blueprint("vehicle.old.server", None),
        ]
    )


class _LitActor(_Actor):
    """A vehicle with lights, as CARLA keeps its light state."""

    def __init__(self, actor_id: int, light_state: int = 0) -> None:
        super().__init__(actor_id)
        self.light_state = light_state

    def get_light_state(self) -> int:
        return self.light_state

    def set_light_state(self, light_state: int) -> None:
        self.light_state = int(light_state)


class _SirenTrafficManager(_TrafficManager):
    """Keeps the percentages of red lights and stop signs each vehicle runs."""

    def __init__(self) -> None:
        super().__init__()
        self.lights: dict = {}
        self.signs: dict = {}

    def ignore_lights_percentage(self, actor: Any, percent: float) -> None:
        self.lights.setdefault(actor.id, []).append(percent)

    def ignore_signs_percentage(self, actor: Any, percent: float) -> None:
        self.signs.setdefault(actor.id, []).append(percent)


def _config(**kwargs: Any) -> EmergencyVehicleEntityConfig:
    return EmergencyVehicleEntityConfig(
        role_name="ambulance", spawn_location=SpawnPointIndex(0), **kwargs
    )


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str]]:
    """Each (blueprint, role) spawned; the actor is a :class:`_LitActor`."""
    calls: List[Tuple[str, str]] = []

    def spawn(world: Any, vehicle_type: str, role_name: str, *a: Any, **k: Any) -> Any:
        calls.append((vehicle_type, role_name))
        return _LitActor(7, _LOW_BEAM)

    monkeypatch.setattr(vehicle_entity, "spawn_vehicle_actor", spawn)
    return calls


class TestTheSirenLights:
    def test_the_siren_is_the_light_bar_bit(self) -> None:
        assert with_siren_lights(0, True) == _SIREN
        assert with_siren_lights(_SIREN, False) == 0

    def test_switching_it_keeps_every_other_light(self) -> None:
        state = _LOW_BEAM | _LEFT
        assert with_siren_lights(state, True) == state | _SIREN
        assert with_siren_lights(state | _SIREN, False) == state

    def test_a_vehicle_shows_it(self) -> None:
        actor = _LitActor(1, _LOW_BEAM)
        set_siren_lights(actor, True)  # type: ignore[arg-type]
        assert actor.light_state == _LOW_BEAM | _SIREN
        set_siren_lights(actor, False)  # type: ignore[arg-type]
        assert actor.light_state == _LOW_BEAM

    def test_a_vehicle_already_so_is_sent_nothing(self) -> None:
        sent: List[int] = []

        class _Counting(_LitActor):
            def set_light_state(self, light_state: int) -> None:
                sent.append(int(light_state))
                super().set_light_state(light_state)

        actor = _Counting(1, _SIREN)
        set_siren_lights(actor, True)  # type: ignore[arg-type]
        assert sent == []


class TestTheBlueprint:
    @pytest.mark.parametrize(
        "type_id",
        [
            "vehicle.ford.ambulance",
            "vehicle.carlamotors.firetruck",
            "vehicle.dodge.charger_police_2020",
            "vehicle.dodgecop.charger",
        ],
    )
    def test_with_no_special_type_its_id_says_so_as_on_carla_0_10(
        self, type_id: str
    ) -> None:
        assert is_emergency_vehicle_type(type_id, "")
        assert is_emergency_vehicle_blueprint(_Blueprint(type_id, None))

    def test_a_special_type_wins_over_the_id(self) -> None:
        assert not is_emergency_vehicle_type("vehicle.ford.ambulance", "car")
        assert not is_emergency_vehicle_type("vehicle.mini.cooper", "")

    def test_an_emergency_vehicle_is_a_blueprint_whose_special_type_says_so(
        self,
    ) -> None:
        assert is_emergency_vehicle_blueprint(_Blueprint("vehicle.a.b", "emergency"))
        assert is_emergency_vehicle_blueprint(_Blueprint("vehicle.a.b", "Emergency"))
        assert not is_emergency_vehicle_blueprint(_Blueprint("vehicle.a.b", "taxi"))
        assert not is_emergency_vehicle_blueprint(_Blueprint("vehicle.a.b", ""))
        # An older server, with no special type: not known to be one.
        assert not is_emergency_vehicle_blueprint(_Blueprint("vehicle.a.b", None))

    def test_the_emergency_vehicles_of_a_world_are_listed_by_id(self) -> None:
        assert emergency_vehicle_blueprints(_world()) == [
            "vehicle.dodge.charger_police",
            "vehicle.ford.ambulance",
        ]

    def test_none_named_is_the_first_emergency_vehicle_of_the_server(self) -> None:
        assert resolve_emergency_vehicle_type(_world(), ANY_EMERGENCY_VEHICLE) == (
            "vehicle.dodge.charger_police"
        )

    def test_one_named_is_the_one_driven(self) -> None:
        assert resolve_emergency_vehicle_type(_world(), "vehicle.ford.ambulance") == (
            "vehicle.ford.ambulance"
        )

    @pytest.mark.parametrize(
        "name", ["vehicle.mini.cooper", "vehicle.mercedes.sprinter", "vehicle.no.such"]
    )
    def test_a_blueprint_that_is_no_emergency_vehicle_is_refused(
        self, name: str
    ) -> None:
        with pytest.raises(ValueError, match="not an emergency vehicle") as raised:
            resolve_emergency_vehicle_type(_world(), name)
        assert "vehicle.ford.ambulance" in str(raised.value)

    def test_a_server_without_emergency_vehicles_is_refused(self) -> None:
        world = _BlueprintWorld([_Blueprint("vehicle.mini.cooper", "")])
        with pytest.raises(ValueError, match="no emergency vehicle blueprint"):
            resolve_emergency_vehicle_type(world, ANY_EMERGENCY_VEHICLE)


class TestTheEntity:
    def test_it_is_an_npc_vehicle_off_duty_by_default(self) -> None:
        ambulance = EmergencyVehicleEntity(_config())
        assert isinstance(ambulance, VehicleEntity)
        assert ambulance.role_name == "ambulance"
        assert not ambulance.in_emergency

    def test_the_config_names_no_blueprint_by_default(self) -> None:
        config = _config()
        assert isinstance(config, VehicleEntityConfig)
        assert config.vehicle_type == ANY_EMERGENCY_VEHICLE
        assert config.emergency is False

    def test_it_is_spawned_as_the_emergency_vehicle_resolved_siren_off(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        ambulance = EmergencyVehicleEntity(_config())
        actor = ambulance.spawn(_world())
        assert spawned == [("vehicle.dodge.charger_police", "ambulance")]
        assert ambulance.vehicle_type == "vehicle.dodge.charger_police"
        assert actor.light_state == _LOW_BEAM  # type: ignore[attr-defined]

    def test_one_on_a_call_from_its_config_spawns_with_its_siren_on(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        ambulance = EmergencyVehicleEntity(_config(emergency=True))
        assert ambulance.in_emergency
        actor = ambulance.spawn(_world())
        assert actor.light_state == _LOW_BEAM | _SIREN  # type: ignore[attr-defined]

    def test_a_car_blueprint_is_refused_before_anything_is_spawned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            vehicle_entity,
            "spawn_vehicle_actor",
            lambda *a, **k: pytest.fail("spawned"),
        )
        ambulance = EmergencyVehicleEntity(_config(vehicle_type="vehicle.mini.cooper"))
        with pytest.raises(ValueError, match="not an emergency vehicle"):
            ambulance.spawn(_world())
        assert ambulance.actor is None

    def test_before_its_spawn_a_call_only_sets_the_state_it_spawns_in(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        ambulance = EmergencyVehicleEntity(_config())
        ambulance.set_emergency(True)
        assert ambulance.in_emergency
        actor = ambulance.spawn(_world())
        assert actor.light_state & _SIREN  # type: ignore[attr-defined]

    def test_a_call_switches_its_siren_and_tells_the_backend(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        told: List[Tuple[Any, bool]] = []

        class _Backend(NullTrafficBackend):
            def set_emergency(self, entity: DrivenEntity, active: bool) -> None:
                told.append((entity, active))

        ambulance = EmergencyVehicleEntity(_config())
        actor = ambulance.spawn(_world())
        ambulance.set_traffic_backend(_Backend())
        assert told == []  # off duty: nothing to tell
        ambulance.set_emergency(True)
        assert actor.light_state == _LOW_BEAM | _SIREN  # type: ignore[attr-defined]
        ambulance.set_emergency(False)
        assert actor.light_state == _LOW_BEAM  # type: ignore[attr-defined]
        assert told == [(ambulance, True), (ambulance, False)]

    def test_a_backend_injected_before_its_spawn_drives_it_on_a_call(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        told: List[bool] = []

        class _Backend(NullTrafficBackend):
            def set_emergency(self, entity: DrivenEntity, active: bool) -> None:
                told.append(active)

        ambulance = EmergencyVehicleEntity(_config(emergency=True))
        ambulance.set_traffic_backend(_Backend())
        assert told == []  # no actor yet
        ambulance.spawn(_world())
        assert told == [True]

    def test_the_traffic_manager_adopting_one_on_a_call_drives_it_as_one(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        tm = _SirenTrafficManager()
        backend = TrafficManagerBackend(client=_Client(tm))
        ambulance = EmergencyVehicleEntity(_config(emergency=True))
        actor = ambulance.spawn(_world())
        ambulance.set_traffic_backend(backend)
        backend.adopt(ambulance)
        assert tm.lights[actor.id] == [EMERGENCY_IGNORE_PCT]

    def test_a_backend_that_cannot_drive_one_says_so(
        self, spawned: List[Tuple[str, str]], caplog: pytest.LogCaptureFixture
    ) -> None:
        ambulance = EmergencyVehicleEntity(_config())
        ambulance.spawn(_world())
        ambulance.set_traffic_backend(NullTrafficBackend())
        ambulance.set_emergency(True)
        assert "cannot drive ambulance as an emergency vehicle" in caplog.text

    def test_the_actor_a_backend_brings_it_back_as_is_lit_as_it_was(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        ambulance = EmergencyVehicleEntity(_config(emergency=True))
        ambulance.spawn(_world())
        back = _LitActor(8)
        assert ambulance._replace_actor(back)  # type: ignore[arg-type]
        assert ambulance.actor is back
        assert back.light_state == _SIREN

    def test_it_is_registered_as_a_vehicle(self) -> None:
        clear_entities()
        ambulance = EmergencyVehicleEntity(_config())
        try:
            register_entity("ambulance", ambulance)
            assert find_vehicle_entity("ambulance") is ambulance
        finally:
            clear_entities()

    def test_the_package_exports_it(self) -> None:
        assert acs.EmergencyVehicleEntity is EmergencyVehicleEntity
        assert acs.EmergencyVehicleEntityConfig is EmergencyVehicleEntityConfig


class TestTheTrafficManagerDrivesIt:
    """On a call it runs every red light and stop sign; off duty it obeys
    them as it drew."""

    def _driven(
        self, vehicles: Optional[VehicleParameters] = None, emergency: bool = False
    ) -> Tuple[TrafficManagerBackend, EmergencyVehicleEntity, _LitActor, Any]:
        tm = _SirenTrafficManager()
        client = _Client(tm)
        backend = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=8123), client=client, vehicles=vehicles
        )
        backend.prepare(TrafficContext(client=client, random_seed=1))
        actor = _LitActor(5)
        actor.attributes = {"role_name": "ambulance", "special_type": "emergency"}
        ambulance = EmergencyVehicleEntity(_config(emergency=emergency))
        ambulance._vehicle = actor  # type: ignore[assignment]
        ambulance.set_traffic_backend(backend)
        backend.adopt(ambulance)
        backend.start(_World([actor]))
        return backend, ambulance, actor, tm

    def test_on_a_call_it_runs_red_lights_and_stop_signs(self) -> None:
        backend, ambulance, actor, tm = self._driven()
        try:
            assert actor.autopilot == [(True, 8123)]
            ambulance.set_emergency(True)
            assert tm.lights[actor.id] == [EMERGENCY_IGNORE_PCT]
            assert tm.signs[actor.id] == [EMERGENCY_IGNORE_PCT]
            assert actor.light_state & _SIREN
        finally:
            backend.close()

    def test_off_duty_it_obeys_them_again(self) -> None:
        backend, ambulance, actor, tm = self._driven()
        try:
            ambulance.set_emergency(True)
            ambulance.set_emergency(False)
            assert tm.lights[actor.id] == [EMERGENCY_IGNORE_PCT, 0.0]
            assert tm.signs[actor.id] == [EMERGENCY_IGNORE_PCT, 0.0]
            assert not actor.light_state & _SIREN
        finally:
            backend.close()

    def test_off_duty_it_goes_back_to_its_own_draw(self) -> None:
        vehicles = VehicleParameters(
            ignore_lights=Spread(0.25), ignore_signs=Spread(0.5)
        )
        backend, ambulance, actor, tm = self._driven(vehicles)
        try:
            assert tm.lights[actor.id] == [25.0]
            ambulance.set_emergency(True)
            ambulance.set_emergency(False)
            assert tm.lights[actor.id] == [25.0, EMERGENCY_IGNORE_PCT, 25.0]
            assert tm.signs[actor.id] == [50.0, EMERGENCY_IGNORE_PCT, 50.0]
        finally:
            backend.close()

    def test_a_call_before_the_autopilot_outlasts_its_first_draw(self) -> None:
        vehicles = VehicleParameters(ignore_lights=Spread(0.25))
        backend, ambulance, actor, tm = self._driven(vehicles, emergency=True)
        try:
            # Told when it was given the backend; its draw is kept, unsent.
            assert tm.lights[actor.id] == [EMERGENCY_IGNORE_PCT]
            ambulance.set_emergency(False)
            assert tm.lights[actor.id] == [EMERGENCY_IGNORE_PCT, 25.0]
        finally:
            backend.close()

    def test_a_vehicle_never_on_a_call_is_left_as_it_drew(self) -> None:
        backend, ambulance, actor, tm = self._driven()
        try:
            ambulance.set_emergency(False)
            assert actor.id not in tm.lights
        finally:
            backend.close()

    def test_a_new_run_forgets_the_calls(self) -> None:
        backend, ambulance, actor, _tm = self._driven()
        ambulance.set_emergency(True)
        backend.close()
        assert backend._emergencies == set()


class TestTheEmergencyAction:
    """Puts the named emergency vehicle on a call, or takes it off one."""

    @pytest.fixture(autouse=True)
    def _registry(self) -> Any:
        clear_entities()
        yield
        clear_entities()

    def _registered(self, spawned: List[Tuple[str, str]]) -> EmergencyVehicleEntity:
        ambulance = EmergencyVehicleEntity(_config())
        ambulance.spawn(_world())
        register_entity("ambulance", ambulance)
        return ambulance

    def test_it_puts_the_vehicle_on_a_call(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        ambulance = self._registered(spawned)
        EmergencyAction("ambulance").execute(_World())  # type: ignore[arg-type]
        assert ambulance.in_emergency
        assert ambulance.actor.light_state & _SIREN  # type: ignore[union-attr]

    def test_inactive_it_takes_the_vehicle_off_its_call(
        self, spawned: List[Tuple[str, str]]
    ) -> None:
        ambulance = self._registered(spawned)
        ambulance.set_emergency(True)
        EmergencyAction("ambulance", active=False).execute(_World())  # type: ignore[arg-type]
        assert not ambulance.in_emergency
        assert not ambulance.actor.light_state & _SIREN  # type: ignore[union-attr]

    def test_it_claims_the_vehicles_lights(self) -> None:
        assert EmergencyAction("ambulance")._control_claims() == [
            ("ambulance", ControlDomain.LIGHTING)
        ]

    def test_a_vehicle_that_is_no_emergency_vehicle_is_left_as_it_is(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        npc = VehicleEntity(VehicleEntityConfig("npc1", SpawnPointIndex(0)))
        register_entity("npc1", npc)
        EmergencyAction("npc1").execute(_World())  # type: ignore[arg-type]
        assert "is not an emergency vehicle" in caplog.text

    def test_an_unknown_entity_is_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        EmergencyAction("nobody").execute(_World())  # type: ignore[arg-type]
        assert "entity 'nobody' not found" in caplog.text

    def test_the_package_exports_it(self) -> None:
        assert acs.EmergencyAction is EmergencyAction
