"""``TrafficLightComplianceScenario``'s judgement, with CARLA stubbed."""

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

    def test_set_it_raises_the_green_check(self) -> None:
        speeds = _speeds(
            TrafficLightComplianceConfig(moving_speed_kmh=1.0, departure_speed_kmh=10.0)
        )
        assert speeds["ego_moving_after_green"] == pytest.approx(10.0 / 3.6)

    def test_moving_on_red_is_no_longer_judged_by_speed(self) -> None:
        # Approaching the stop line on red is not a failure: only crossing it
        # is (TestStopLineWatch).
        speeds = _speeds(TrafficLightComplianceConfig(moving_speed_kmh=1.0))
        assert "ego_moving_during_red" not in speeds


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


# ---------------------------------------------------------------------------
# Where the ego stops against the stop line
# ---------------------------------------------------------------------------

#: The stop line the fake ego drives up to, along x: its front bumper is at
#: ``x + _FRONT``.
_LINE_X = 100.0
_FRONT = 2.0


class _Ego:
    """A fake ego on a straight lane along +x, facing +x."""

    def __init__(self, x: float) -> None:
        self.x = x

    def get_transform(self) -> Any:
        return carla.Transform(carla.Location(x=self.x), carla.Rotation(yaw=0.0))


class _Run:
    """Drives a :class:`StopLineWatch` and the scenario's two conditions
    tick by tick, with the ego, its standstill and the stop-line measure
    faked.

    Past the line the measure reads *beyond* (the next line ahead, or
    ``None``), as the real one does."""

    def __init__(
        self,
        monkeypatch: Any,
        start_x: float,
        *,
        beyond: Any = 40.0,
        red_until: float = 5.0,
        tolerance_m: float = 0.0,
    ) -> None:
        from autoware_carla_scenario.examples import traffic_light_compliance as module
        from autoware_carla_scenario.measures import Bumpers

        self.ego = _Ego(start_x)
        self.standing = False
        self.beyond = beyond
        self.read: list[float] = []

        def distance(transform: Any, bumpers: Any) -> Any:
            gap = _LINE_X - (float(transform.location.x) + bumpers.front)
            self.read.append(gap)
            return gap if gap >= 0.0 else self.beyond

        monkeypatch.setattr(module, "stop_line_distance_between", distance)
        monkeypatch.setattr(module, "bumpers_of", lambda actor: Bumpers(_FRONT, 2.0))
        monkeypatch.setattr(
            module, "find_actor_by_role_name", lambda world, role: self.ego
        )
        self.watch = module.StopLineWatch(
            red_until=red_until,
            stop_zone_min_m=0.0,
            stop_zone_max_m=3.0,
            tolerance_m=tolerance_m,
            standstill_seconds=0.5,
        )
        standstill = MagicMock()
        standstill.check.side_effect = lambda world, elapsed: (
            MagicMock() if self.standing else None
        )
        self.watch._standstill = standstill
        self.stopped = module.StoppedInStopZoneCondition(self.watch, label="in_zone")
        self.violation = module.RedLightViolationCondition(
            self.watch, label="violation"
        )
        self.elapsed = 0.0

    def tick(self, x: float, *, standing: bool = False) -> Any:
        """Move the ego to *x*, advance 0.1 s, and return the fail result."""
        self.elapsed = round(self.elapsed + 0.1, 6)
        self.ego.x = x
        self.standing = standing
        world = MagicMock()
        self.stopped.check(world, self.elapsed)
        return self.violation.check(world, self.elapsed)

    def hold(self, until: float, *, standing: bool = True) -> Any:
        """Keep the ego where it is until *until*; the first fail result."""
        while self.elapsed < until - 1e-9:
            failed = self.tick(self.ego.x, standing=standing)
            if failed is not None:
                return failed
        return None


class TestStopLineWatch:
    def test_approaching_slowly_from_far_back_and_stopping_in_the_zone_passes(
        self, monkeypatch
    ) -> None:
        # 32 m before the line, rolling up at 1 m per tick to 1.5 m before it.
        run = _Run(monkeypatch, _LINE_X - _FRONT - 32.0, red_until=6.0)
        x = run.ego.x
        while _LINE_X - (x + _FRONT) > 1.5 + 1e-9:
            x += 1.0 if _LINE_X - (x + 1.0 + _FRONT) >= 1.5 else 0.5
            assert run.tick(x) is None
        assert run.hold(6.0) is None
        assert run.watch.stopped_in_zone_at is not None
        assert run.watch.stopped_at_m == pytest.approx(1.5, abs=1e-4)
        # At and after the green the judgement holds.
        assert run.tick(run.ego.x) is None
        result = run.stopped.check(MagicMock(), run.elapsed)
        assert result is not None and result.passed
        assert "1.50 m before the stop line" in result.message
        assert run.stopped.get_details()[
            "stopped_at_stop_line_distance_m"
        ] == pytest.approx(1.5, abs=1e-4)

    @pytest.mark.parametrize("beyond", [40.0, None, 0.4])
    def test_crossing_the_line_on_red_fails(self, monkeypatch, beyond: Any) -> None:
        # Past the line the measure reads the next line, nothing, or even a
        # line just past this one: the watch carries its own line on.
        run = _Run(monkeypatch, _LINE_X - _FRONT - 2.0, beyond=beyond)
        assert run.tick(run.ego.x + 1.0) is None
        assert run.tick(run.ego.x + 0.95) is None
        failed = run.tick(run.ego.x + 0.3)
        assert failed is not None and not failed.passed
        assert "crossed the stop line" in failed.message
        assert run.watch.crossed_by_m == pytest.approx(0.25, abs=1e-4)
        assert run.read[-1] < 0.0

    def test_creeping_out_of_the_zone_over_the_line_fails(self, monkeypatch) -> None:
        # Starting in the zone, it has to stay there.
        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0)
        assert run.hold(2.0) is None
        assert run.watch.stopped_in_zone_at is not None
        assert run.tick(run.ego.x + 0.6) is None
        failed = run.tick(run.ego.x + 0.6)
        assert failed is not None and "crossed the stop line" in failed.message

    def test_the_tolerance_forgives_a_nose_over_the_line(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0, tolerance_m=0.3)
        assert run.hold(2.0) is None
        assert run.tick(run.ego.x + 1.2, standing=True) is None
        assert run.watch.distance == pytest.approx(-0.2, abs=1e-4)
        assert run.hold(6.0) is None

        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0, tolerance_m=0.3)
        assert run.hold(2.0) is None
        failed = run.tick(run.ego.x + 1.4)
        assert failed is not None and "crossed the stop line" in failed.message

    def test_stopping_short_of_the_zone_fails_at_the_green(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 8.0)
        failed = run.hold(6.0)
        assert failed is not None and not failed.passed
        assert failed.elapsed_seconds == pytest.approx(5.0)
        assert "stopped 8.00 m before the stop line" in failed.message
        assert "outside the stop zone" in failed.message

    def test_still_rolling_at_the_green_fails_on_its_own(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 38.0)
        failed = None
        while failed is None:
            failed = run.tick(run.ego.x + 0.5)
        assert failed.elapsed_seconds == pytest.approx(5.0)
        assert "had not stopped by the green" in failed.message
        assert "crossed" not in failed.message

    def test_no_stop_line_ahead_has_not_stopped(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X + 10.0, beyond=None)
        failed = run.hold(6.0)
        assert failed is not None
        assert "had not stopped by the green" in failed.message

    def test_crossing_after_the_green_is_not_running_the_red(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0)
        assert run.hold(5.2) is None
        for _ in range(5):
            assert run.tick(run.ego.x + 1.0) is None
        assert run.watch.crossed_at is not None

    def test_a_second_check_at_the_same_time_travels_nothing(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 10.0, beyond=None)
        run.tick(run.ego.x)
        run.ego.x += 11.0
        world = MagicMock()
        run.watch.update(world, 0.2)
        run.watch.update(world, 0.2)
        assert run.watch.distance == pytest.approx(-1.0, abs=1e-4)

    def test_a_stop_zone_that_ends_before_it_starts_is_refused(self) -> None:
        from autoware_carla_scenario.examples.traffic_light_compliance import (
            StopLineWatch,
        )

        with pytest.raises(ValueError, match="stop_zone_min_m"):
            StopLineWatch(5.0, 3.0, 1.0, 0.0, 0.5)

    def test_the_scenario_judges_by_the_stop_zone(self) -> None:
        scenario = _set_up(
            TrafficLightComplianceConfig(stop_zone_min_m=0.5, stop_zone_max_m=4.0)
        )
        (fail,) = [
            c for c in scenario._fail_conditions if c.label == "ego_red_light_violation"
        ]
        details = fail.get_details()
        assert details["stop_zone_min_m"] == 0.5
        assert details["stop_zone_max_m"] == 4.0
        assert details["green_at_seconds"] == 3.0
