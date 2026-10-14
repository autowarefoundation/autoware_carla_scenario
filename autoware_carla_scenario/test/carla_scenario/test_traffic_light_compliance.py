"""``TrafficLightComplianceScenario``'s speed thresholds, with CARLA stubbed."""

from __future__ import annotations

import unittest.mock
from typing import Any
from unittest.mock import MagicMock

import pytest

import typesafe_carla.carla as carla
from autoware_carla_scenario import EgoConfig, Lanelet2Pose, SpawnTransform
from autoware_carla_scenario.examples.configs import TrafficLightComplianceConfig
from autoware_carla_scenario.examples.traffic_light_compliance import (
    TrafficLightComplianceScenario,
)


def _speeds(config: TrafficLightComplianceConfig) -> dict[str, float]:
    """Each labelled ``SpeedCondition``'s threshold (m/s) in the set-up scenario."""
    scenario = TrafficLightComplianceScenario(
        EgoConfig(
            spawn_location=SpawnTransform(carla.Transform(carla.Location())),
            vehicle_type="vehicle.mini.cooper",
        ),
        spawn_pose=Lanelet2Pose(lanelet_id=1, s=5.0),
        config=config,
    )
    scenario.set_client(MagicMock(), 8000)
    with unittest.mock.patch.object(scenario, "_setup_ego_spawn"):
        scenario.setup()
    found: dict[str, float] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("condition_type") == "SpeedCondition":
                found[node["label"]] = node["value"]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for condition in [*scenario._pass_conditions, *scenario._fail_conditions]:
        walk(condition.get_details())
    return found


class TestDepartureSpeed:
    def test_unset_it_is_the_moving_speed(self) -> None:
        speeds = _speeds(TrafficLightComplianceConfig(moving_speed_kmh=1.0))
        assert speeds["ego_moving_after_green"] == pytest.approx(1.0 / 3.6)
        assert speeds["ego_moving_during_red"] == pytest.approx(1.0 / 3.6)

    def test_set_it_only_raises_the_green_check(self) -> None:
        speeds = _speeds(
            TrafficLightComplianceConfig(moving_speed_kmh=1.0, departure_speed_kmh=10.0)
        )
        assert speeds["ego_moving_after_green"] == pytest.approx(10.0 / 3.6)
        assert speeds["ego_moving_during_red"] == pytest.approx(1.0 / 3.6)
