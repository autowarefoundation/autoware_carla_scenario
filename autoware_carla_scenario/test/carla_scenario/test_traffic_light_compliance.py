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


def _set_up(config: TrafficLightComplianceConfig, monkeypatch=None) -> Any:
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
    return scenario


class TestControls:
    def test_the_ego_is_placed_before_the_stop_line_when_asked(
        self, monkeypatch
    ) -> None:
        from autoware_carla_scenario.examples import traffic_light_compliance as module
        from autoware_carla_scenario.measures import Bumpers

        asked: list[tuple[Any, float]] = []

        def place(spawn: Any, distance: float, *, ego_bumpers: Any) -> Any:
            asked.append((spawn, distance))
            return Lanelet2Pose(lanelet_id=2, s=7.0)

        monkeypatch.setattr(module, "place_at_stop_line_distance", place)
        monkeypatch.setattr(module, "vehicle_bumpers", lambda world, kind: Bumpers())

        scenario = _set_up(TrafficLightComplianceConfig(stop_line_distance_m=12.5))

        assert asked == [(Lanelet2Pose(lanelet_id=1, s=5.0), 12.5)]
        assert scenario._spawn_pose == Lanelet2Pose(lanelet_id=2, s=7.0)
        # Unset, the spawn pose is left as it is.
        assert _set_up(TrafficLightComplianceConfig())._spawn_pose == Lanelet2Pose(
            lanelet_id=1, s=5.0
        )

    def test_the_red_phase_is_a_measure(self) -> None:
        from autoware_carla_scenario.examples.traffic_light_compliance import (
            RED_LIGHT_DURATION_S,
        )

        scenario = _set_up(TrafficLightComplianceConfig(light_switch_delay_seconds=7.5))

        assert scenario.measure(RED_LIGHT_DURATION_S, MagicMock()) == 7.5

    def test_the_timeout_can_count_from_the_green_light(self) -> None:
        def timeout(config: TrafficLightComplianceConfig) -> float:
            scenario = _set_up(config)
            (found,) = [
                c.get_details()
                for c in scenario._fail_conditions
                if "timeout_seconds" in c.get_details()
            ]
            return float(found["timeout_seconds"])

        assert timeout(TrafficLightComplianceConfig(timeout_seconds=9.0)) == 9.0
        assert (
            timeout(
                TrafficLightComplianceConfig(
                    timeout_seconds=9.0,
                    light_switch_delay_seconds=8.0,
                    timeout_after_green_seconds=6.0,
                )
            )
            == 14.0
        )
