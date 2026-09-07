"""Unit tests for ScenarioRunner's tick pacing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario import ScenarioRunner
from autoware_carla_scenario.scenario_runner import _destroy_all_dynamic_actors


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


class _FakeActor:
    """The parts of a CARLA actor the cleanup looks at."""

    def __init__(self, actor_id: int, role_name: str = "", parent=None) -> None:
        self.id = actor_id
        self.attributes = {"role_name": role_name} if role_name else {}
        self.parent = parent
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _FakeActorList:
    """``world.get_actors()``: a list that filters by type wildcard."""

    def __init__(self, vehicles: list, sensors: list) -> None:
        self._by_kind = {"vehicle.*": vehicles, "sensor.*": sensors}

    def filter(self, pattern: str) -> list:
        return list(self._by_kind[pattern])


class TestCleanupExemption:
    """A run may start beside an ego the scenario does not own."""

    @staticmethod
    def _world(vehicles: list, sensors: list) -> MagicMock:
        world = MagicMock()
        world.get_actors.return_value = _FakeActorList(vehicles, sensors)
        return world

    def test_leftovers_are_destroyed(self):
        leftover = _FakeActor(1, "npc1")
        sensor = _FakeActor(2, parent=leftover)
        _destroy_all_dynamic_actors(self._world([leftover], [sensor]), "S")
        assert leftover.destroyed
        assert sensor.destroyed

    def test_a_kept_ego_survives_with_its_sensors(self):
        # The interface node spawns this ego and drives it; destroying it would
        # leave the run polling for an actor that is never coming back.
        ego = _FakeActor(1, "Ego")
        lidar = _FakeActor(2, parent=ego)
        other = _FakeActor(3, "npc1")
        _destroy_all_dynamic_actors(
            self._world([ego, other], [lidar]), "S", keep_role_names=frozenset({"Ego"})
        )
        assert not ego.destroyed
        assert not lidar.destroyed
        assert other.destroyed


class _FakeEgo:
    """An ego that becomes ready after a given number of ticks."""

    attaches_to_existing_actor = True

    def __init__(self, ready_after: int, gives_up_after: int | None = None) -> None:
        self._ready_after = ready_after
        self._gives_up_after = gives_up_after
        self.ticks = 0

    @property
    def is_initialized(self) -> bool:
        return self.ticks >= self._ready_after

    @property
    def termination_requested(self) -> bool:
        return self._gives_up_after is not None and self.ticks >= self._gives_up_after

    def on_tick(self, world, elapsed) -> None:
        del world, elapsed
        self.ticks += 1


class TestWaitForEgo:
    """The scenario is judged only once its ego can act on it."""

    def test_an_ego_that_is_ready_is_not_waited_for(self):
        runner = _make_runner(None)
        world = MagicMock()
        runner._wait_for_ego(world, _FakeEgo(ready_after=0), "S")
        world.tick.assert_not_called()

    def test_the_world_ticks_until_the_ego_is_ready(self):
        # Autoware only makes progress while simulation time advances, so the
        # wait has to tick rather than sleep.
        runner = _make_runner(None)
        world = MagicMock()
        ego = _FakeEgo(ready_after=3)
        runner._wait_for_ego(world, ego, "S")
        assert world.tick.call_count == 3
        assert ego.is_initialized

    def test_an_ego_that_gives_up_ends_the_wait(self):
        runner = _make_runner(None)
        world = MagicMock()
        ego = _FakeEgo(ready_after=100, gives_up_after=2)
        runner._wait_for_ego(world, ego, "S")
        assert world.tick.call_count == 2
        assert not ego.is_initialized


class TestInitPhaseOrdering:
    """Init actions run before the ego is asked to start, and before the clock.

    The order is what makes the phase useful: an ego that plans its own route
    is handed its mission by an init action, and the runner then asks it to
    start and waits for it to be ready.  Registering the hand-over on the tick
    loop instead could never work -- the loop begins after that wait, and the
    wait is for a stack that has not been told where to go.
    """

    def test_init_runs_before_on_scenario_start_and_the_wait(self):
        import carla

        from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform

        order: list[str] = []

        class _Scenario(BaseScenario):
            def setup(self) -> None:
                self.register_init(lambda _w: order.append("init"))

            def is_done(self) -> bool:
                return True

        scenario = _Scenario(
            EgoConfig(
                spawn_location=SpawnTransform(
                    carla.Transform(carla.Location(x=0, y=0, z=0))
                )
            )
        )
        scenario.setup()

        ego = _FakeEgo(ready_after=1)
        ego.on_scenario_start = lambda _w: order.append("on_scenario_start")  # type: ignore[method-assign]

        world = MagicMock()
        scenario.run_init(world)
        ego.on_scenario_start(world)
        _make_runner(None)._wait_for_ego(world, ego, "S")

        assert order == ["init", "on_scenario_start"]
