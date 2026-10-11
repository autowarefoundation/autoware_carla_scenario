"""An entity config stores its role name as CARLA's ``role_name`` string.

``VehicleEntityConfig``, ``PedestrianEntityConfig`` and ``EgoConfig`` accept an
:class:`EntityRole` or a ``str`` and hold the string either way (#72), so what
a config holds has one type, which the static check needs (docs/typecheck.md).
"""

from __future__ import annotations

import dataclasses

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    EgoConfig,
    EntityRole,
    PedestrianEntity,
    PedestrianEntityConfig,
    SpawnPointIndex,
    SpawnTransform,
    VehicleEntity,
    VehicleEntityConfig,
)


@pytest.mark.parametrize("role", [EntityRole.npc(3), "npc3"])
def test_a_vehicle_config_stores_the_role_as_its_string(role: EntityRole | str) -> None:
    # The field is typed `str`; an EntityRole is still taken at run time.
    config = VehicleEntityConfig(role_name=role, spawn_location=SpawnPointIndex(0))  # type: ignore[arg-type]
    assert type(config.role_name) is str
    assert config.role_name == "npc3"
    assert VehicleEntity(config).role_name == "npc3"


@pytest.mark.parametrize("role", [EntityRole("pedestrian1"), "pedestrian1"])
def test_a_pedestrian_config_stores_the_role_as_its_string(
    role: EntityRole | str,
) -> None:
    config = PedestrianEntityConfig(
        role_name=role,  # type: ignore[arg-type]
        spawn_location=SpawnTransform(carla.Transform()),
    )
    assert type(config.role_name) is str
    assert config.role_name == "pedestrian1"
    assert PedestrianEntity(config).role_name == "pedestrian1"


def test_the_ego_config_stores_the_ego_role_as_its_string() -> None:
    config = EgoConfig(SpawnPointIndex(0))
    assert type(config.role_name) is str
    assert config.role_name == str(EGO_ROLE_NAME) == "Ego"


def test_a_role_name_is_not_newly_validated() -> None:
    # Only EntityRole validates; a plain string is stored as it was given.
    config = VehicleEntityConfig(
        role_name="Not A Role", spawn_location=SpawnPointIndex(0)
    )
    assert config.role_name == "Not A Role"


def test_a_config_rebuilt_from_its_fields_keeps_the_string() -> None:
    config = VehicleEntityConfig(
        role_name=EntityRole.npc(1),  # type: ignore[arg-type]
        spawn_location=SpawnPointIndex(0),
    )
    copy = dataclasses.replace(config, vehicle_type="vehicle.tesla.model3")
    assert copy.role_name == "npc1"
    assert copy == dataclasses.replace(config, vehicle_type="vehicle.tesla.model3")
