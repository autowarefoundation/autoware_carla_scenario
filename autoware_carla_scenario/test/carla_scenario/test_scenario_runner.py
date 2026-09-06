"""Unit tests for ScenarioRunner's tick pacing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario import ScenarioRunner


def _make_runner(max_tick_rate_hz: float | None) -> ScenarioRunner:
    """Build a runner without touching CARLA."""
    with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
        return ScenarioRunner(MagicMock(), max_tick_rate_hz=max_tick_rate_hz)


class _FakeClock:
    """A clock the test moves by hand, and the sleeps taken from it."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestTickPacing:
    """The runner steps the world no faster than the slowest client can read it.

    Whoever ticks the world decides what every other client sees: one that
    cannot service every tick sees the simulation jump rather than step, and
    reads its own missing samples as sensor dropouts.
    """

    @staticmethod
    def _pace(runner: ScenarioRunner, clock: _FakeClock, times: int) -> None:
        with (
            patch(
                "autoware_carla_scenario.scenario_runner.time.monotonic",
                clock.monotonic,
            ),
            patch("autoware_carla_scenario.scenario_runner.time.sleep", clock.sleep),
        ):
            for _ in range(times):
                runner._pace_tick()

    def test_uncapped_never_sleeps(self):
        clock = _FakeClock()
        self._pace(_make_runner(None), clock, times=3)
        assert clock.slept == []

    def test_capped_sleeps_out_the_rest_of_the_slot(self):
        clock = _FakeClock()
        # The first tick starts the schedule, so only the ones after it wait.
        self._pace(_make_runner(5.0), clock, times=3)
        assert clock.slept == pytest.approx([0.2, 0.2])

    def test_an_overrunning_tick_does_not_push_the_next_one_back(self):
        # A tick that takes longer than its slot has already spent the budget:
        # the one after it is due immediately, and the schedule restarts from
        # now rather than firing a burst to catch up.
        runner = _make_runner(5.0)
        clock = _FakeClock()
        self._pace(runner, clock, times=1)
        clock.now += 0.5
        self._pace(runner, clock, times=1)
        assert clock.slept == []
        self._pace(runner, clock, times=1)
        assert clock.slept == pytest.approx([0.2])
