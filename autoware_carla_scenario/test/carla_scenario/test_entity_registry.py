"""The entity registry: every lookup by role, and the typed lookups by kind."""

from __future__ import annotations

import logging
from typing import Iterator
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EntityRole,
    PedestrianEntity,
    PedestrianEntityConfig,
    SpawnTransform,
    VehicleEntity,
    VehicleEntityConfig,
    WalkStraightAction,
)
from autoware_carla_scenario.entity.registry import (
    clear_entities,
    find_entity_by_role_name,
    find_pedestrian_entity,
    find_vehicle_entity,
    register_entity,
    unregister_entity,
)


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    clear_entities()
    yield
    clear_entities()


def _vehicle() -> VehicleEntity:
    return VehicleEntity(
        VehicleEntityConfig(
            role_name="npc1", spawn_location=SpawnTransform(carla.Transform())
        )
    )


def _pedestrian() -> PedestrianEntity:
    return PedestrianEntity(
        PedestrianEntityConfig(
            role_name="walker1", spawn_location=SpawnTransform(carla.Transform())
        )
    )


def test_a_vehicle_is_found_as_a_vehicle_only() -> None:
    vehicle = _vehicle()
    register_entity(EntityRole("npc1"), vehicle)

    assert find_entity_by_role_name("npc1") is vehicle
    assert find_vehicle_entity("npc1") is vehicle
    assert find_pedestrian_entity("npc1") is None


def test_a_pedestrian_is_found_as_a_pedestrian_only() -> None:
    walker = _pedestrian()
    register_entity("walker1", walker)

    assert find_entity_by_role_name(EntityRole("walker1")) is walker
    assert find_pedestrian_entity("walker1") is walker
    assert find_vehicle_entity("walker1") is None


def test_an_entity_of_neither_kind_is_found_by_every_lookup() -> None:
    double = MagicMock()
    register_entity("npc1", double)

    assert find_entity_by_role_name("npc1") is double
    assert find_vehicle_entity("npc1") is double
    assert find_pedestrian_entity("npc1") is double


def test_re_registering_a_role_replaces_it_across_kinds() -> None:
    register_entity("npc1", MagicMock())
    vehicle = _vehicle()
    register_entity("npc1", vehicle)

    assert find_entity_by_role_name("npc1") is vehicle
    assert find_pedestrian_entity("npc1") is None


def test_unregistering_forgets_every_kind() -> None:
    register_entity("npc1", MagicMock())
    unregister_entity("npc1")
    unregister_entity("npc1")  # forgetting an unknown role is fine

    assert find_entity_by_role_name("npc1") is None
    assert find_vehicle_entity("npc1") is None
    assert find_pedestrian_entity("npc1") is None


def test_walking_a_vehicle_says_it_is_not_a_pedestrian(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_entity("npc1", _vehicle())

    with caplog.at_level(logging.WARNING):
        WalkStraightAction(entity_name="npc1").execute(MagicMock())

    assert "is not a pedestrian" in caplog.text
