"""NPCs cannot take a role name the framework reads as the ego's (#2046).

``canonical_role_name`` reads ``Ego``, ``ego`` and ``hero``, in any case, as
the ego (#2039).  A config that takes its role name as a plain string skips
``EntityRole``'s pattern, so it refuses those names itself; the ego's own
construction path still uses them.  The end-of-run sweep reads the names it
keeps the same way.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest

from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.entity import (
    AutowareEgoEntity,
    EmergencyVehicleEntityConfig,
    MotorcycleEntityConfig,
    PedestrianEntityConfig,
    VehicleEntityConfig,
)
from autoware_carla_scenario.entity._spawn import SpawnPointIndex, SpawnTransform
from autoware_carla_scenario.entity.ego import EgoConfig
from autoware_carla_scenario.entity_role import (
    NATIVE_AUTOWARE_EGO_ROLE_NAME,
    EntityRole,
    _refuse_ego_role_name,
)
from autoware_carla_scenario.hils.config import HilsConfigError, parse_config
from autoware_carla_scenario.scenario_runner import _kept_by_role

_ALIASES = ["ego", "hero", "HERO", "Hero", "EGO", "eGo"]


def _vehicle(role: str) -> VehicleEntityConfig:
    return VehicleEntityConfig(role_name=role, spawn_location=SpawnPointIndex(0))


def _motorcycle(role: str) -> VehicleEntityConfig:
    return MotorcycleEntityConfig(role_name=role, spawn_location=SpawnPointIndex(0))


def _emergency(role: str) -> VehicleEntityConfig:
    return EmergencyVehicleEntityConfig(
        role_name=role, spawn_location=SpawnPointIndex(0)
    )


def _pedestrian(role: str) -> PedestrianEntityConfig:
    return PedestrianEntityConfig(
        role_name=role,
        spawn_location=SpawnTransform(None),  # type: ignore[arg-type]
    )


_NPC_CONFIGS: list[Callable[[str], Any]] = [
    _vehicle,
    _motorcycle,
    _emergency,
    _pedestrian,
]
_NPC_ONLY_CONFIGS: list[Callable[[str], Any]] = [_motorcycle, _emergency, _pedestrian]


@pytest.mark.parametrize("build", _NPC_CONFIGS)
@pytest.mark.parametrize("role", _ALIASES)
def test_an_npc_named_as_the_ego_is_refused(
    build: Callable[[str], Any], role: str
) -> None:
    with pytest.raises(ValueError, match=f"Invalid role name {role!r}"):
        build(role)


@pytest.mark.parametrize("build", _NPC_ONLY_CONFIGS)
def test_an_entity_that_is_never_the_ego_is_refused_ego_itself(
    build: Callable[[str], Any],
) -> None:
    with pytest.raises(ValueError, match="'Ego'"):
        build("Ego")


@pytest.mark.parametrize("build", _NPC_CONFIGS)
@pytest.mark.parametrize("role", ["npc1", "car2", "heron1", "ego_car", "rider"])
def test_other_names_are_accepted(build: Callable[[str], Any], role: str) -> None:
    assert build(role).role_name == role


@pytest.mark.parametrize("build", _NPC_CONFIGS)
def test_an_entity_role_is_still_accepted(build: Callable[[str], Any]) -> None:
    assert build(EntityRole.npc(3)).role_name == "npc3"  # type: ignore[arg-type]


def test_the_message_says_how_to_name_an_npc() -> None:
    with pytest.raises(ValueError) as caught:
        _refuse_ego_role_name("hero", "an NPC vehicle")
    message = str(caught.value)
    assert "an NPC vehicle" in message
    assert "'npc1'" in message


# ---------------------------------------------------------------------------
# The ego's own construction path
# ---------------------------------------------------------------------------


def test_the_ego_config_keeps_its_role_name() -> None:
    assert EgoConfig(spawn_location=SpawnPointIndex(0)).role_name == "Ego"


def test_a_vehicle_config_may_still_name_the_ego() -> None:
    # EgoConfig is built on VehicleEntityConfig, and EntityRole allows "Ego".
    assert _vehicle(str(EGO_ROLE_NAME)).role_name == "Ego"
    assert _vehicle(EntityRole.ego()).role_name == "Ego"  # type: ignore[arg-type]


def test_the_native_ego_is_still_spawned_as_ego() -> None:
    entity = AutowareEgoEntity(
        bridge=FakeAutowareBridge(), native_autoware_interface=True
    )
    assert entity._ros2_autoware_role_name() == NATIVE_AUTOWARE_EGO_ROLE_NAME


# ---------------------------------------------------------------------------
# HILS sensor devices, whose name is the sensor's role_name
# ---------------------------------------------------------------------------


def _sensor_device(name: str) -> dict[str, Any]:
    return {
        "devices": {
            name: {
                "url": "mem://x",
                "codec": "lidar:vlp16",
                "sensor": {"blueprint": "sensor.lidar.ray_cast"},
            }
        }
    }


@pytest.mark.parametrize("name", ["Ego", *_ALIASES])
def test_a_hils_sensor_named_as_the_ego_is_refused(name: str) -> None:
    with pytest.raises(HilsConfigError, match=f"devices.{name}: .*as the ego"):
        parse_config(_sensor_device(name))


def test_a_hils_sensor_with_another_name_is_read() -> None:
    assert parse_config(_sensor_device("lidar_top")).devices[0].name == "lidar_top"


# ---------------------------------------------------------------------------
# The end-of-run sweep's keep set
# ---------------------------------------------------------------------------


def _actor(role: str) -> Any:
    return SimpleNamespace(attributes={"role_name": role})


@pytest.mark.parametrize("keep", ["Ego", "ego", "hero", "HERO"])
@pytest.mark.parametrize("role", ["Ego", "ego", "hero", "Hero"])
def test_a_keep_set_naming_the_ego_any_way_keeps_it(keep: str, role: str) -> None:
    assert _kept_by_role(_actor(role), frozenset({keep}))


def test_a_keep_set_of_ego_keeps_no_npc() -> None:
    assert not _kept_by_role(_actor("npc1"), frozenset({"ego"}))
    assert not _kept_by_role(SimpleNamespace(attributes={}), frozenset({"ego"}))


def test_other_kept_names_match_as_they_are() -> None:
    assert _kept_by_role(_actor("sensor1"), frozenset({"sensor1"}))
    assert not _kept_by_role(_actor("Sensor1"), frozenset({"sensor1"}))
