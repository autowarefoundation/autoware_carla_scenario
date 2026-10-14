"""The ego entity hierarchy: who drives, and what follows from it."""

from __future__ import annotations

import logging
from typing import Callable

import pytest

from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
from autoware_carla_scenario.entity import (
    AutowareEgoEntity,
    BridgedEgoEntity,
    CarlaDriverEntity,
    EgoVehicle,
    ExternallyDrivenEgo,
)
from autoware_carla_scenario.entity.ros2_entity import Ros2EgoEntity


@pytest.mark.parametrize(
    "entity_class",
    [AutowareEgoEntity, CarlaDriverEntity, Ros2EgoEntity],
)
def test_every_self_driven_ego_is_externally_driven(entity_class: type) -> None:
    assert issubclass(entity_class, ExternallyDrivenEgo)
    assert entity_class.use_autopilot is False


def test_the_traffic_manager_ego_is_not() -> None:
    assert not issubclass(EgoVehicle, ExternallyDrivenEgo)
    assert EgoVehicle.use_autopilot is True


def test_the_autoware_ego_takes_its_mission_over_the_bridge() -> None:
    assert issubclass(AutowareEgoEntity, BridgedEgoEntity)
    assert BridgedEgoEntity.requires_goal is True
    assert not issubclass(CarlaDriverEntity, BridgedEgoEntity)
    assert not issubclass(Ros2EgoEntity, BridgedEgoEntity)


def test_terminate_records_the_reason() -> None:
    entity = ExternallyDrivenEgo()
    assert not entity.termination_requested
    assert entity.termination_reason is None

    entity._terminate("the stack went down")

    assert entity.termination_requested
    assert entity.termination_reason == "the stack went down"
    entity._reset_termination()
    assert not entity.termination_requested
    assert entity.termination_reason is None


@pytest.mark.parametrize(
    "refuse",
    [
        lambda entity: entity.change_lane(None, None),
        lambda entity: entity.set_desired_speed(None, 30.0),
        lambda entity: entity.turn_at_junction(None, None),
    ],
    ids=["change_lane", "set_desired_speed", "turn_at_junction"],
)
def test_a_refusal_names_what_drives(
    refuse: Callable[[ExternallyDrivenEgo], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    entity = AutowareEgoEntity(bridge=FakeAutowareBridge())
    with caplog.at_level(logging.WARNING):
        refuse(entity)
    assert "not driven by the TrafficManager" in caplog.text
    assert "left to the Autoware stack" in caplog.text
