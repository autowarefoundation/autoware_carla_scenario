"""``TrafficLightComplianceScenario``'s judgement, with CARLA stubbed."""

from __future__ import annotations

import math
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
_STANDSTILL_S = 0.5


class _Ego:
    """A fake ego: where it is, which way it faces, how fast it goes."""

    def __init__(self, x: float) -> None:
        self.x = x
        self.y = 0.0
        self.yaw_deg = 0.0
        self.speed = 0.0

    def get_transform(self) -> Any:
        return carla.Transform(
            carla.Location(x=self.x, y=self.y), carla.Rotation(yaw=self.yaw_deg)
        )

    def get_velocity(self) -> Any:
        return carla.Vector3D(x=self.speed)


def _straight(beyond: Any) -> Any:
    """The measure on a straight lane along +x with one line at
    :data:`_LINE_X`: past it, *beyond* (the next line ahead, or ``None``)."""

    def read(front_x: float, front_y: float) -> Any:
        gap = _LINE_X - front_x
        return gap if gap >= 0.0 else beyond

    return read


class _Run:
    """Drives a :class:`StopLineWatch` and the scenario's two conditions
    tick by tick (0.1 s), with the ego, its standstill and the stop-line
    measure faked.  *read* is the measure, from the front bumper."""

    def __init__(
        self,
        monkeypatch: Any,
        start_x: float,
        *,
        beyond: Any = 40.0,
        read: Any = None,
        red_until: float = 5.0,
        tolerance_m: float = 0.0,
    ) -> None:
        from autoware_carla_scenario.examples import traffic_light_compliance as module
        from autoware_carla_scenario.measures import Bumpers

        self.ego = _Ego(start_x)
        self.read = _straight(beyond) if read is None else read
        self.still_since: Any = None

        def distance(transform: Any, bumpers: Any) -> Any:
            yaw = math.radians(float(transform.rotation.yaw))
            return self.read(
                float(transform.location.x) + bumpers.front * math.cos(yaw),
                float(transform.location.y) + bumpers.front * math.sin(yaw),
            )

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
            standstill_seconds=_STANDSTILL_S,
        )
        # StandstillCondition's rule: slow enough for its duration.
        standstill = MagicMock()
        standstill.check.side_effect = lambda world, elapsed: (
            MagicMock()
            if self.still_since is not None
            and elapsed - self.still_since >= _STANDSTILL_S - 1e-9
            else None
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
        self.ego.speed = 0.0 if standing else 1.0
        if not standing:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = self.elapsed
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

    def front(self) -> float:
        return self.ego.x + _FRONT


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
        assert run.hold(6.5) is None
        assert run.watch.stopped_in_zone_at is not None
        assert run.watch.stopped_at_m == pytest.approx(1.5, abs=1e-4)
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

    def test_the_tick_ending_at_the_green_still_ran_under_red(
        self, monkeypatch
    ) -> None:
        # The light turns green before the world steps from red_until, so the
        # step that ends at red_until was driven under red.
        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0)
        assert run.hold(4.9) is None
        failed = run.tick(run.ego.x + 2.0)
        assert run.elapsed == pytest.approx(5.0)
        assert failed is not None and "ran the red" in failed.message

        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0)
        assert run.hold(5.0) is None
        assert run.tick(run.ego.x + 2.0) is None
        assert run.watch.crossed_at == pytest.approx(5.1)

    def test_creeping_out_of_the_zone_over_the_line_fails(self, monkeypatch) -> None:
        # Starting in the zone, it has to stay there.
        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0)
        assert run.hold(2.0) is None
        assert run.watch.stopped_in_zone_at is not None
        assert run.tick(run.ego.x + 0.6) is None
        failed = run.tick(run.ego.x + 0.6)
        assert failed is not None and "crossed the stop line" in failed.message

    def test_once_stopped_in_the_zone_moving_back_still_passes(
        self, monkeypatch
    ) -> None:
        # Latched: having stood in the zone before the green is what counts.
        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0)
        assert run.hold(2.0) is None
        assert run.tick(run.ego.x - 3.0) is None
        assert run.tick(run.ego.x + 2.5) is None
        assert run.hold(6.0, standing=False) is None
        assert run.stopped.check(MagicMock(), run.elapsed) is not None

    def test_the_tolerance_counts_a_nose_over_the_line_as_at_it(
        self, monkeypatch
    ) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0, tolerance_m=0.3)
        assert run.tick(run.ego.x) is None
        assert run.tick(run.ego.x + 1.2) is None
        assert run.watch.distance == pytest.approx(-0.2, abs=1e-4)
        assert run.hold(6.0) is None
        assert run.watch.stopped_in_zone_at is not None
        assert run.watch.stopped_at_m == pytest.approx(-0.2, abs=1e-4)

        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0, tolerance_m=0.3)
        assert run.tick(run.ego.x) is None
        failed = run.tick(run.ego.x + 1.4)
        assert failed is not None and "crossed the stop line" in failed.message

    def test_stopping_short_of_the_zone_fails_at_the_green(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 8.0)
        failed = run.hold(6.0)
        assert failed is not None and not failed.passed
        # The first tick driven under green.
        assert failed.elapsed_seconds == pytest.approx(5.1)
        assert "stopped 8.00 m before the stop line" in failed.message
        assert "outside the stop zone" in failed.message

    def test_still_rolling_at_the_green_fails_on_its_own(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 38.0)
        failed = None
        while failed is None:
            failed = run.tick(run.ego.x + 0.5)
        assert failed.elapsed_seconds == pytest.approx(5.1)
        assert "had not stopped by the green: still moving" in failed.message
        assert "13.00 m before" in failed.message

    @pytest.mark.parametrize("stop_at, still_for", [(4.7, 0.3), (4.6, 0.4)])
    def test_stopping_in_the_zone_too_late_says_so(
        self, monkeypatch, stop_at: float, still_for: float
    ) -> None:
        # Stopped in the zone 0.3 s before the green; and 0.4 s, so that the
        # standstill completes only on the first green tick, which does not
        # count.
        run = _Run(monkeypatch, _LINE_X - _FRONT - 20.0)
        while run.elapsed < stop_at - 0.1 - 1e-9:
            assert run.tick(run.ego.x + 0.4) is None
        failed = run.hold(6.0)
        assert failed is not None
        assert failed.elapsed_seconds == pytest.approx(5.1)
        assert "stopped in the stop zone" in failed.message
        assert f"only {still_for:.2f}s before the green" in failed.message
        assert "standstill_seconds" in failed.message
        assert run.watch.stopped_in_zone_at is None

    def test_no_stop_line_ahead_has_not_stopped(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X + 10.0, beyond=None)
        failed = run.hold(6.0)
        assert failed is not None
        assert "no stop line was read" in failed.message

    def test_crossing_after_the_green_is_not_running_the_red(self, monkeypatch) -> None:
        run = _Run(monkeypatch, _LINE_X - _FRONT - 1.0)
        assert run.hold(5.2) is None
        for _ in range(5):
            assert run.tick(run.ego.x + 1.0) is None
        assert run.watch.crossed_at is not None

    def test_a_line_on_the_other_branch_gives_way_to_the_egos_own(
        self, monkeypatch
    ) -> None:
        # At a fork the measure first reads line A on the other branch, 12 m
        # ahead; past the fork, the ego's own line B, 3 m beyond A.  It stops
        # 1 m before B, 2 m past where A is.
        line_a, line_b = 100.0, 103.0

        def read(front_x: float, front_y: float) -> Any:
            return (line_a if front_x < 92.0 else line_b) - front_x

        run = _Run(monkeypatch, line_a - 12.0 - _FRONT, read=read)
        while run.front() < line_b - 1.0 - 1e-9:
            assert run.tick(run.ego.x + 1.0) is None
        assert run.hold(6.0) is None
        assert run.watch.crossed_at is None
        assert run.watch.stopped_at_m == pytest.approx(1.0, abs=1e-4)

    def test_a_lane_change_onto_a_nearer_line_takes_it(self, monkeypatch) -> None:
        # Changing lanes onto one whose stop line is 3 m nearer.
        def read(front_x: float, front_y: float) -> Any:
            return (_LINE_X if front_x < 90.0 else _LINE_X - 3.0) - front_x

        run = _Run(monkeypatch, _LINE_X - 12.0 - _FRONT, read=read)
        while run.front() < _LINE_X - 4.0 - 1e-9:
            assert run.tick(run.ego.x + 1.0) is None
        assert run.hold(6.0) is None
        assert run.watch.stopped_at_m == pytest.approx(1.0, abs=1e-4)

    def test_on_a_curve_the_line_is_carried_on_the_egos_travel(
        self, monkeypatch
    ) -> None:
        # A left curve of radius 20 m about (0, 20), the line 20.25 m along
        # it.  The lane is lost 6 m before the line, and the ego drives on
        # past it.
        radius, line_s = 20.0, 20.25
        readings: list[Any] = []

        def front_s(front_x: float, front_y: float) -> float:
            return radius * math.atan2(front_x, radius - front_y)

        def read(front_x: float, front_y: float) -> Any:
            gap = line_s - front_s(front_x, front_y)
            readings.append(gap if gap > 6.0 else None)
            return readings[-1]

        run = _Run(monkeypatch, 0.0, read=read)

        def drive_to(s: float) -> Any:
            theta = s / radius
            run.ego.y = radius * (1.0 - math.cos(theta))
            run.ego.yaw_deg = math.degrees(theta)
            return run.tick(radius * math.sin(theta))

        assert drive_to(0.0) is None
        failed = None
        s = 0.0
        while failed is None and s < 25.0:
            s += 0.5
            failed = drive_to(s)
            if readings[-1] is not None:
                assert run.watch.distance == pytest.approx(readings[-1], abs=1e-4)
        assert failed is not None and "crossed the stop line" in failed.message
        # On the step that took the front bumper past the line, and by about
        # as far as it went past it.
        theta = s / radius
        front_x = radius * math.sin(theta) + _FRONT * math.cos(theta)
        front_y = radius * (1.0 - math.cos(theta)) + _FRONT * math.sin(theta)
        past = front_s(front_x, front_y) - line_s
        assert 0.0 < past < 0.5
        assert run.watch.crossed_by_m == pytest.approx(past, abs=0.05)

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
            TrafficLightComplianceConfig(
                stop_zone_min_m=0.5,
                stop_zone_max_m=4.0,
                standstill_seconds=1.5,
                standstill_speed_mps=0.3,
            )
        )
        (fail,) = [
            c for c in scenario._fail_conditions if c.label == "ego_red_light_violation"
        ]
        details = fail.get_details()
        assert details["stop_zone_min_m"] == 0.5
        assert details["stop_zone_max_m"] == 4.0
        assert details["green_at_seconds"] == 3.0
        standstill = fail._watch._standstill.get_details()
        assert standstill["speed_threshold"] == 0.3
        assert standstill["duration_seconds"] == 1.5
