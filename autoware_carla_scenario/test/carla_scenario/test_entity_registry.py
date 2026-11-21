"""The entity registry: every lookup by role, and the typed lookups by kind."""

from __future__ import annotations

import logging
from typing import Iterator
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    BaseAction,
    DespawnAction,
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
    register_pedestrian_entity,
    register_vehicle_entity,
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


def test_register_vehicle_entity_finds_it_as_a_vehicle_only() -> None:
    vehicle = _vehicle()
    register_vehicle_entity(EntityRole("npc1"), vehicle)

    assert find_entity_by_role_name("npc1") is vehicle
    assert find_vehicle_entity("npc1") is vehicle
    assert find_pedestrian_entity("npc1") is None


def test_register_pedestrian_entity_finds_it_as_a_pedestrian_only() -> None:
    walker = _pedestrian()
    register_pedestrian_entity(EntityRole("walker1"), walker)

    assert find_entity_by_role_name("walker1") is walker
    assert find_pedestrian_entity("walker1") is walker
    assert find_vehicle_entity("walker1") is None


def test_the_typed_registrations_replace_a_role_across_kinds() -> None:
    register_entity("x", MagicMock())
    walker = _pedestrian()
    register_pedestrian_entity("x", walker)
    assert find_vehicle_entity("x") is None
    assert find_pedestrian_entity("x") is walker

    vehicle = _vehicle()
    register_vehicle_entity("x", vehicle)
    assert find_vehicle_entity("x") is vehicle
    assert find_pedestrian_entity("x") is None

    register_pedestrian_entity("x", walker)
    assert find_vehicle_entity("x") is None
    assert find_entity_by_role_name("x") is walker


def test_the_typed_registrations_are_forgotten_and_cleared() -> None:
    register_vehicle_entity("npc1", _vehicle())
    register_pedestrian_entity("walker1", _pedestrian())
    unregister_entity("npc1")
    assert find_entity_by_role_name("npc1") is None
    assert find_entity_by_role_name("walker1") is not None

    clear_entities()
    assert find_entity_by_role_name("walker1") is None


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


def test_the_emergency_and_traffic_parameter_actions_on_a_pedestrian_raise() -> None:
    """As the other vehicle actions do, rather than saying "not found" (#2273)."""
    from autoware_carla_scenario import EmergencyAction, TrafficParametersAction
    from autoware_carla_scenario.traffic import (
        NullTrafficBackend,
        Spread,
        VehicleParameters,
    )

    register_entity("walker1", _pedestrian())
    with pytest.raises(AttributeError, match="'PedestrianEntity'.*'set_emergency'"):
        EmergencyAction("walker1").execute(MagicMock())
    parameters = TrafficParametersAction(
        VehicleParameters(speed_factor=Spread(0.5)), entity_name="walker1"
    )
    # Given straight to the action: NullTrafficBackend changes nothing, and
    # the pedestrian check comes before any change is asked for.
    parameters._backend = NullTrafficBackend()
    with pytest.raises(
        AttributeError, match="'PedestrianEntity'.*'override_vehicle_parameters'"
    ):
        parameters.execute(MagicMock())


def test_the_emergency_action_on_an_unknown_role_still_says_not_found(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from autoware_carla_scenario import EmergencyAction

    with caplog.at_level(logging.WARNING):
        EmergencyAction("npc9").execute(MagicMock())
    assert "entity 'npc9' not found" in caplog.text


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


# ---------------------------------------------------------------------------
# The ego by any of its names (#2271)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Ego", "ego", "EGO", "hero", "Hero"])
def test_the_ego_is_found_by_any_name_the_actor_lookups_take(name: str) -> None:
    ego = MagicMock()
    register_entity("Ego", ego)
    assert find_vehicle_entity(name) is ego
    assert find_entity_by_role_name(name) is ego
    assert find_pedestrian_entity(name) is ego


def test_registering_under_an_alias_keeps_the_ego_under_its_canonical_name() -> None:
    ego = MagicMock()
    register_entity("hero", ego)
    assert find_entity_by_role_name("Ego") is ego
    unregister_entity("ego")
    assert find_entity_by_role_name("Ego") is None


def test_a_vehicle_registered_as_the_ego_is_found_as_ego() -> None:
    ego = _vehicle()
    register_vehicle_entity(EntityRole.ego(), ego)
    assert find_vehicle_entity("ego") is ego


def test_other_names_still_match_exactly() -> None:
    vehicle = _vehicle()
    register_vehicle_entity("npc1", vehicle)
    assert find_vehicle_entity("npc1") is vehicle
    assert find_vehicle_entity("NPC1") is None


@pytest.mark.parametrize(
    "name", ["Ego", "ego", "EGO", "hero", "HERO", EntityRole.ego()]
)
def test_the_ego_cannot_be_despawned_by_any_of_its_names(
    name: str | EntityRole,
) -> None:
    with pytest.raises(ValueError, match="the ego cannot be despawned"):
        DespawnAction(name)


@pytest.mark.parametrize(
    "name", ["Ego", "ego", "EGO", "hero", "HERO", EntityRole.ego()]
)
def test_the_ego_cannot_be_spawned_by_any_of_its_names(
    name: str | EntityRole,
) -> None:
    from autoware_carla_scenario.actions.spawn import SpawnAction

    with pytest.raises(ValueError, match="the ego is in the scene from the start"):
        SpawnAction(name)


def test_an_npc_can_still_be_spawned() -> None:
    from autoware_carla_scenario.actions.spawn import SpawnAction

    assert SpawnAction("npc1").entity_name == "npc1"


def test_an_npc_can_still_be_despawned() -> None:
    assert DespawnAction("npc1").entity_name == "npc1"
