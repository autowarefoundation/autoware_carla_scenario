"""The entity registry: every lookup by role, and the typed lookups by kind."""

from __future__ import annotations

import logging
from typing import Iterator
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    BaseAction,
    EntityRole,
    LaneChangeAction,
    LaneChangeDirection,
    PedestrianEntity,
    PedestrianEntityConfig,
    SetSpeedAction,
    SpawnTransform,
    TurnAction,
    TurnDirection,
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


@pytest.mark.parametrize(
    ("action", "method"),
    [
        (LaneChangeAction("walker1", LaneChangeDirection.LEFT), "change_lane"),
        (TurnAction("walker1", TurnDirection.LEFT), "turn_at_junction"),
        (SetSpeedAction("walker1", 30.0), "set_desired_speed"),
        (SetSpeedAction("walker1", 30.0, rate_kmh_s=5.0), "set_desired_speed"),
    ],
    ids=["lane_change", "turn", "set_speed", "set_speed_rate"],
)
def test_a_vehicle_action_on_a_pedestrian_raises_as_before(
    action: BaseAction, method: str
) -> None:
    # The vehicle lookup does not find a pedestrian, but these actions used
    # to call the method on whatever entity had the role, which raised.
    register_entity("walker1", _pedestrian())

    with pytest.raises(AttributeError, match=f"'PedestrianEntity'.*'{method}'"):
        action.execute(MagicMock())


def test_a_vehicle_action_on_an_unknown_role_says_not_found(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        LaneChangeAction("npc9", LaneChangeDirection.LEFT).execute(MagicMock())

    assert "not found" in caplog.text


def test_clearing_forgets_every_kind() -> None:
    register_entity("npc1", MagicMock())
    register_entity("npc2", _vehicle())
    register_entity("walker1", _pedestrian())
    clear_entities()

    for role in ("npc1", "npc2", "walker1"):
        assert find_entity_by_role_name(role) is None
        assert find_vehicle_entity(role) is None
        assert find_pedestrian_entity(role) is None
