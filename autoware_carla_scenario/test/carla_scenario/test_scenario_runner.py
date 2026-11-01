"""Unit tests for ScenarioRunner's tick pacing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario import ScenarioRunner
from autoware_carla_scenario.constants import FIXED_DELTA_SECONDS
from autoware_carla_scenario.scenario_runner import (
    _destroy_all_dynamic_actors,
    _ScenarioClock,
)


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


class _FakeWorld:
    """A world whose simulated clock the test advances by hand."""

    def __init__(self, simulated_now: float = 0.0) -> None:
        self.simulated_now = simulated_now

    def tick(self, ticks: int = 1) -> None:
        """Advance simulated time by *ticks* of ``fixed_delta_seconds``."""
        self.simulated_now += ticks * FIXED_DELTA_SECONDS

    def get_snapshot(self) -> MagicMock:
        snapshot = MagicMock()
        snapshot.timestamp.elapsed_seconds = self.simulated_now
        return snapshot


class TestScenarioClock:
    """A scenario is measured on the world's clock, not on the machine's.

    The loop steps the world as fast as the slowest client allows, so how much
    simulated time fits into a second of real time is a property of the host.
    Reading a scenario's durations off the wall clock would make the same
    scenario fire its triggers in different places on a fast machine and a slow
    one -- exactly the determinism synchronous mode exists to provide.
    """

    def test_it_starts_at_zero_however_long_the_world_has_been_up(self):
        """Init ticks the world before the run, and must not count against it."""
        world = _FakeWorld(simulated_now=1234.5)

        assert _ScenarioClock(world).simulated == 0.0

    def test_it_advances_by_one_fixed_delta_per_tick(self):
        world = _FakeWorld(simulated_now=1234.5)
        clock = _ScenarioClock(world)

        world.tick()
        assert clock.simulated == pytest.approx(FIXED_DELTA_SECONDS)

        world.tick(3)
        assert clock.simulated == pytest.approx(4 * FIXED_DELTA_SECONDS)

    def test_real_time_passing_does_not_move_it(self):
        """The whole point: twenty ticks are one second on every host.

        Whether those twenty ticks took a tenth of a real second on an idle
        machine or ten real seconds with an autonomy stack attached, the
        scenario is one second further along.
        """
        world = _FakeWorld()
        wall = _FakeClock()

        with patch(
            "autoware_carla_scenario.scenario_runner.time.monotonic", wall.monotonic
        ):
            clock = _ScenarioClock(world)
            world.tick(20)
            wall.now += 10.0

            assert clock.simulated == pytest.approx(1.0)

    def test_a_stalled_world_stops_the_clock(self):
        """A consequence worth stating: no ticks, no time.

        A run whose simulator has stopped is not protected by this clock at
        all.  That is the CARLA client's RPC timeout and the sweeper's
        ``job_timeout_seconds``, which are about the machine rather than about
        the scenario.
        """
        world = _FakeWorld()
        wall = _FakeClock()

        with patch(
            "autoware_carla_scenario.scenario_runner.time.monotonic", wall.monotonic
        ):
            clock = _ScenarioClock(world)
            wall.now += 90.0

            assert clock.simulated == 0.0


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
    """``world.get_actors()``: a list that filters by type wildcard.

    An unknown pattern is empty rather than an error, because the real
    ``ActorList`` answers every wildcard: a test that has no walkers should
    read as a world with no walkers, not as one that has never heard of them.
    """

    def __init__(
        self,
        vehicles: list,
        sensors: list,
        walkers: list | None = None,
        controllers: list | None = None,
    ) -> None:
        self._by_kind = {
            "vehicle.*": vehicles,
            "sensor.*": sensors,
            "walker.*": walkers or [],
            "controller.*": controllers or [],
        }

    def filter(self, pattern: str) -> list:
        return list(self._by_kind.get(pattern, []))


class TestCleanupExemption:
    """A run may start beside an ego the scenario does not own."""

    @staticmethod
    def _world(vehicles: list, sensors: list, walkers: list | None = None) -> MagicMock:
        world = MagicMock()
        world.get_actors.return_value = _FakeActorList(
            vehicles, sensors, walkers=walkers
        )
        return world

    def test_leftovers_are_destroyed(self):
        leftover = _FakeActor(1, "npc1")
        sensor = _FakeActor(2, parent=leftover)
        _destroy_all_dynamic_actors(self._world([leftover], [sensor]), "S")
        assert leftover.destroyed
        assert sensor.destroyed

    def test_a_leftover_walker_is_destroyed_too(self):
        """A queue runs several scenarios against one world.

        A pedestrian left standing from the last one is a pedestrian the next
        one did not put there, and nothing in the next scenario would explain
        why the ego stopped for it.
        """
        walker = _FakeActor(1, "walker1")
        _destroy_all_dynamic_actors(self._world([], [], walkers=[walker]), "S")
        assert walker.destroyed

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
    carried_actor_ids: frozenset[int] = frozenset()
    actor = None

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
        import typesafe_carla.carla as carla

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


class _SnapshotLight:
    """A traffic light whose ``is_frozen()`` reads as CARLA's client reads it.

    ``freeze()`` reaches the server at once, but ``is_frozen()`` answers from
    the snapshot the last tick sent, so a freeze reads only after a tick.  Only
    the freeze is modelled; ``set_state()`` is accepted and ignored.
    """

    type_id = "traffic.traffic_light"

    def __init__(self) -> None:
        self.frozen = False  # on the server
        self._seen = False  # in the client's snapshot

    def set_state(self, state: object) -> None:
        del state

    def freeze(self, frozen: bool) -> None:
        self.frozen = frozen

    def is_frozen(self) -> bool:
        return self._seen

    def snapshot(self) -> None:
        self._seen = self.frozen


class _SnapshotWorld:
    """A world of traffic lights whose tick refreshes what the client reads."""

    def __init__(self, lights: list[_SnapshotLight]) -> None:
        self.lights = lights
        self.ticks = 0

    def get_actors(self) -> "_LightList":
        return _LightList(self.lights)

    def tick(self) -> int:
        self.ticks += 1
        for light in self.lights:
            light.snapshot()
        return self.ticks


class _LightList:
    def __init__(self, lights: list[_SnapshotLight]) -> None:
        self._lights = lights

    def filter(self, pattern: str) -> list:
        return list(self._lights) if pattern.startswith("traffic.traffic_light") else []


class TestTheEgoStartsOnTheInitPhasesWorld:
    """The runner ticks after the init actions, before the ego is started.

    A client reads ``is_frozen()`` from the last tick, so an ego started on
    the init actions' own tick reads every light they froze as running.  The
    policy run-up then froze those lights as its own and thawed them on
    arrival: CARLA's cycle turned the scenario's red lights yellow mid-run.
    """

    _LIGHTS = 3

    def _start(self, runner: ScenarioRunner, tmp_path) -> tuple[list[str], list]:
        """Run a scenario up to ``ego.on_scenario_start``, through run_scenario.

        Returns what happened, in order (``init``, ``tick``, ``start``), and the
        lights the policy run-up, started there, would take over as its own.
        """
        from autoware_carla_scenario.entity._policy_warmup import PolicyWarmup

        world = _SnapshotWorld([_SnapshotLight() for _ in range(self._LIGHTS)])
        calls: list[str] = []
        taken: list = []

        def tick(*_a, **_k) -> int:
            calls.append("tick")
            return _SnapshotWorld.tick(world)

        def run_init(w) -> None:
            # What TrafficSignalAction(Red, ALL) does: freeze every light.
            for light in w.get_actors().filter("traffic.traffic_light*"):
                light.freeze(True)
            calls.append("init")

        def on_scenario_start(w) -> None:
            calls.append("start")
            # What CarlaDriverEntity does there with warmup_s > 0.
            taken.extend(PolicyWarmup._freeze_lights(w))
            raise _Stop

        world_mock = MagicMock()
        world_mock.get_actors.side_effect = world.get_actors
        world_mock.tick.side_effect = tick
        runner.output_dir = tmp_path
        runner._world = world_mock
        runner._pace_tick = lambda: None  # type: ignore[method-assign]

        scenario = MagicMock()
        scenario.STABILIZE_TICKS = 2
        scenario.run_init.side_effect = run_init
        ego = scenario.create_ego.return_value
        ego.spawn.return_value.id = 1
        ego.on_scenario_start.side_effect = on_scenario_start

        with patch("autoware_carla_scenario.scenario_runner.prepare_measures"):
            with pytest.raises(_Stop):
                runner.run_scenario(scenario, reset_world=False)
        return calls, taken

    def test_the_world_ticks_between_the_init_actions_and_the_egos_start(
        self, tmp_path
    ):
        calls, _ = self._start(_make_runner(None), tmp_path)
        assert calls[calls.index("init") :] == ["init", "tick", "start"]

    def test_the_run_up_takes_over_no_light_the_init_phase_froze(self, tmp_path):
        _, taken = self._start(_make_runner(None), tmp_path)
        assert taken == []

    def test_without_that_tick_the_run_up_takes_them_all_over(self, tmp_path):
        """The fault the tick prevents, through the same run: an ego started on
        the init actions' own tick reads their lights as running."""
        runner = _make_runner(None)
        runner._run_init = lambda w, s: s.run_init(w)  # type: ignore[method-assign]
        calls, taken = self._start(runner, tmp_path)
        assert calls[calls.index("init") :] == ["init", "start"]
        assert len(taken) == self._LIGHTS


class TestNothingMovesDuringInit:
    """The clock ticks through init, but no vehicle does anything with it.

    Simulation time has to advance -- a stack localizes, routes and engages
    only while it does, and its sensors only publish then -- so the phase
    cannot be a pause.  What must not happen is a vehicle using that time: an
    ego that engages partway through the wait, or a car rolling down a slope,
    would both have moved before the run started measuring anything.
    """

    @staticmethod
    def _vehicle() -> MagicMock:
        actor = MagicMock()
        actor.attributes = {"role_name": "npc1"}
        return actor

    def _world(self, vehicles: list) -> MagicMock:
        world = MagicMock()
        world.get_actors.return_value = _FakeActorList(vehicles, [])
        return world

    def test_the_wait_holds_every_vehicle(self):
        vehicle = self._vehicle()
        world = self._world([vehicle])
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._pace_tick = lambda: None  # type: ignore[method-assign]

        ego = _FakeEgo(ready_after=3)
        runner._wait_for_ego(world, ego, "S")

        assert vehicle.apply_control.call_count == 3
        held = vehicle.apply_control.call_args[0][0]
        assert held.hand_brake is True
        assert held.brake == 1.0
        assert held.throttle == 0.0

    def test_the_hold_comes_off_again(self):
        vehicle = self._vehicle()
        world = self._world([vehicle])

        ScenarioRunner._release_vehicles(world)

        released = vehicle.apply_control.call_args[0][0]
        assert released.hand_brake is False
        assert released.brake == 0.0


def test_the_init_hold_spares_an_ego_that_is_moving_itself():
    from autoware_carla_scenario.utils.vehicles import hold_vehicles_still

    ego, other = MagicMock(id=1), MagicMock(id=2)
    world = MagicMock()
    world.get_actors.return_value.filter.return_value = [ego, other]
    hold_vehicles_still(world, spare={ego.id})
    ego.apply_control.assert_not_called()
    other.apply_control.assert_called_once()


def test_the_release_is_not_the_default_control():
    """CARLA's client drops a control equal to the last one sent through the same
    handle, and a fresh handle's last is the default: a release equal to the
    default never reaches the vehicle."""
    from autoware_carla_scenario.utils.vehicles import release_vehicle

    vehicle = MagicMock()
    release_vehicle(vehicle)
    released = vehicle.apply_control.call_args[0][0]
    assert (released.throttle, released.brake, released.hand_brake) == (0.0, 0.0, False)
    assert released.gear == 1 and released.manual_gear_shift is False


class TestReloadingTheMap:
    """The world is reset by loading its map by name, not with reload_world()."""

    @staticmethod
    def _client(map_name: str) -> MagicMock:
        client = MagicMock()
        client.get_world.return_value.get_map.return_value.name = map_name
        return client

    def test_the_map_is_loaded_by_its_name(self):
        from autoware_carla_scenario.world_reset import reload_current_map

        client = self._client("Carla/Maps/Town10HD_Opt")

        world = reload_current_map(client)

        # FindMapPath on a CARLA UE5 server matches the name, not the path,
        # and an empty name (what reload_world sends) matches nothing.
        client.load_world.assert_called_once_with("Town10HD_Opt")
        client.reload_world.assert_not_called()
        assert world is client.load_world.return_value

    def test_a_map_without_a_name_falls_back_to_reload_world(self):
        from autoware_carla_scenario.world_reset import reload_current_map

        client = self._client("")

        reload_current_map(client)

        client.reload_world.assert_called_once_with()
        client.load_world.assert_not_called()


class TestRecordRun:
    """What a run's coverage file says of its outcome."""

    def test_a_failed_run_records_when_and_on_what(self) -> None:
        from autoware_carla_scenario.conditions import ScenarioResult
        from autoware_carla_scenario.environment import capture
        from autoware_carla_scenario.scenario_runner import _record_run

        coverage = MagicMock()
        result = ScenarioResult(passed=False, message="crash", elapsed_seconds=123.45)
        _record_run(coverage, None, result)
        coverage.record_case.assert_not_called()
        coverage.record_outcome.assert_called_once_with(
            False, failed_at=123.45, environment=capture().id, weight=None
        )

    def test_a_passed_run_records_no_failure_time(self) -> None:
        from autoware_carla_scenario.conditions import ScenarioResult
        from autoware_carla_scenario.environment import capture
        from autoware_carla_scenario.scenario_runner import _record_run

        coverage = MagicMock()
        case = (MagicMock(), MagicMock())
        _record_run(
            coverage,
            case,
            ScenarioResult(passed=True, message="ok", elapsed_seconds=9.0),
        )
        coverage.record_case.assert_called_once_with(*case)
        coverage.record_outcome.assert_called_once_with(
            True, failed_at=None, environment=capture().id, weight=None
        )

    def test_an_unreadable_environment_is_none(self) -> None:
        from autoware_carla_scenario.conditions import ScenarioResult
        from autoware_carla_scenario.scenario_runner import _record_run

        coverage = MagicMock()
        with patch(
            "autoware_carla_scenario.environment.capture",
            side_effect=RuntimeError("no"),
        ):
            _record_run(
                coverage,
                None,
                ScenarioResult(passed=False, message="x", elapsed_seconds=1.0),
            )
        coverage.record_outcome.assert_called_once_with(
            False, failed_at=1.0, environment=None, weight=None
        )

    def test_an_importance_sampled_run_records_its_weight(self) -> None:
        from autoware_carla_scenario.conditions import ScenarioResult
        from autoware_carla_scenario.environment import capture
        from autoware_carla_scenario.scenario_runner import _record_run

        coverage = MagicMock()
        result = ScenarioResult(passed=False, message="crash", elapsed_seconds=2.0)
        _record_run(coverage, None, result, 0.0125)
        coverage.record_outcome.assert_called_once_with(
            False, failed_at=2.0, environment=capture().id, weight=0.0125
        )


def test_only_an_importance_sampled_run_has_an_outcome_weight() -> None:
    from types import SimpleNamespace

    from autoware_carla_scenario import AdversarialTrafficAction, BaseScenario
    from autoware_carla_scenario.actions import SetSpeedAction

    outcome_weight = BaseScenario.outcome_weight.fget  # type: ignore[attr-defined]
    actions = SimpleNamespace(
        _init_actions=[],
        _pre_tick_actions=[SetSpeedAction("npc1", 1.0)],
        _post_tick_actions=[],
    )
    assert outcome_weight(actions) is None
    # NADE samples, even before it has sampled anything (weight 1).
    actions._post_tick_actions.append(AdversarialTrafficAction([]))
    assert outcome_weight(actions) == 1.0


def test_what_the_odds_measures_read_is_built_before_the_first_tick(
    tmp_path,
) -> None:
    """The vehicle-ahead measures' routing graph takes tens of seconds on a
    large map: the runner builds it (prepare_measures) in setup, before the
    world is ticked -- not on the first tick that reads it."""

    class _Stop(Exception):
        pass

    calls: list[str] = []
    runner = _make_runner(None)
    runner.output_dir = tmp_path
    world = MagicMock()
    world.tick.side_effect = lambda *a, **k: calls.append("tick")
    runner._world = world
    scenario = MagicMock()
    scenario.setup.side_effect = lambda: calls.append("setup")

    def prepare(odd: object, measured: object) -> None:
        assert odd is runner.odd
        # The scenario, whose replacements of the built-in measures need none.
        assert measured is scenario
        calls.append("prepare")
        raise _Stop  # enough: what follows is the ego, then the ticks

    with patch(
        "autoware_carla_scenario.scenario_runner.prepare_measures", side_effect=prepare
    ):
        try:
            runner.run_scenario(scenario, reset_world=False)
        except _Stop:
            pass

    assert calls[:2] == ["setup", "prepare"]


class _SharedServer:
    """A CARLA client on a server other clients may load worlds on.

    Every load starts a new episode, as on a real server; the worlds it hands
    out say which episode and which map they belong to, and every map name
    asked for is counted (``get_map()`` builds the whole road network).
    """

    def __init__(self, maps: tuple[str, ...], current: str, episode: int = 1) -> None:
        self.maps = maps
        self.current = current
        self.episode = episode
        self.loads: list[str] = []
        self.map_reads = 0

    def get_available_maps(self) -> list[str]:
        return [f"/Game/Carla/Maps/{name}" for name in self.maps]

    def load_world(self, name: str) -> MagicMock:
        self.loads.append(name)
        self.foreign_load(name)
        return self.get_world()

    def reload_world(self) -> MagicMock:  # pragma: no cover - must not be called
        raise AssertionError("reload_world() cannot reload a map on CARLA UE5")

    def foreign_load(self, name: str) -> None:
        """Another client loads *name* on the same server."""
        self.current = name
        self.episode += 1

    def get_world(self) -> MagicMock:
        world = MagicMock()
        world.id = self.episode
        loaded = f"Carla/Maps/{self.current}"

        def get_map() -> MagicMock:
            self.map_reads += 1
            carla_map = MagicMock()
            carla_map.name = loaded
            return carla_map

        world.get_map.side_effect = get_map
        return world


class _Stop(Exception):
    """Ends a run once the part under test is over."""


def _runner_on(server: _SharedServer) -> ScenarioRunner:
    runner = _make_runner(None)
    runner._client = server  # type: ignore[assignment]
    return runner


def _set_up_once(runner: ScenarioRunner) -> None:
    """Start a run and stop it where the scenario would build its ego.

    That is right after the runner has settled which world the run is on.
    """
    scenario = MagicMock()
    scenario.create_ego.side_effect = _Stop
    with pytest.raises(_Stop):
        runner.run_scenario(scenario, reset_world=False)


class TestTheBatchMap:
    """A batch runs on the map it loaded, even when another client loads a
    world of its own on the same server between (or during) its runs."""

    def test_reload_map_loads_the_named_map_not_the_current_one(self):
        from autoware_carla_scenario.world_reset import reload_map

        server = _SharedServer(("Odaiba_generated", "Town10HD_Opt"), "Town10HD_Opt")

        reload_map(server, "/Game/Carla/Maps/Odaiba_generated")  # type: ignore[arg-type]

        assert server.loads == ["Odaiba_generated"]

    def test_the_reset_reloads_the_batch_map_not_a_foreign_one(self):
        server = _SharedServer(("Odaiba_generated", "Town10HD_Opt"), "Town10HD_Opt")
        runner = _runner_on(server)
        runner.load_map_by_name("Odaiba_generated")
        server.foreign_load("Town10HD_Opt")

        runner.reset_world("Case9")

        assert server.loads == ["Odaiba_generated", "Odaiba_generated"]
        assert server.current == "Odaiba_generated"

    def test_a_run_on_a_foreign_map_loads_the_batch_map_first(self, caplog):
        server = _SharedServer(("Odaiba_generated", "Town10HD_Opt"), "Town10HD_Opt")
        runner = _runner_on(server)
        runner.load_map_by_name("Odaiba_generated")
        server.foreign_load("Town10HD_Opt")

        with caplog.at_level("WARNING", logger="autoware_carla_scenario"):
            _set_up_once(runner)

        assert server.loads == ["Odaiba_generated", "Odaiba_generated"]
        assert server.current == "Odaiba_generated"
        assert runner._world is not None and runner._world.id == server.episode
        warning = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "'Odaiba_generated' again" in message for message in warning
        ), warning

    def test_an_installed_opendrive_is_installed_again(self, tmp_path, monkeypatch):
        xodr = tmp_path / "odaiba.xodr"
        xodr.write_text("<OpenDRIVE/>")
        installed = tmp_path / "installed.xodr"
        monkeypatch.setenv("ODAIBA_GENERATED_PATH", str(installed))
        server = _SharedServer(("Odaiba_generated", "Town10HD_Opt"), "Town10HD_Opt")
        runner = _runner_on(server)
        with patch(
            "autoware_carla_scenario.scenario_runner.map_asset_env_var",
            return_value="ODAIBA_GENERATED_PATH",
        ):
            runner.load_map_by_overwriting_xodr(xodr, "Odaiba_generated")
            installed.write_text("<OpenDRIVE>someone else's</OpenDRIVE>")
            server.foreign_load("Town10HD_Opt")

            _set_up_once(runner)

        assert server.loads == ["Odaiba_generated", "Odaiba_generated"]
        assert installed.read_text() == "<OpenDRIVE/>"

    def test_a_foreign_reload_of_the_same_map_is_loaded_again(self):
        """Another client's world of the same map holds its own actors,
        settings and weather: the run gets a clean one."""
        server = _SharedServer(("Odaiba_generated",), "Odaiba_generated")
        runner = _runner_on(server)
        runner.load_map_by_name("Odaiba_generated")
        server.foreign_load("Odaiba_generated")

        _set_up_once(runner)

        assert server.loads == ["Odaiba_generated", "Odaiba_generated"]
        assert runner._world is not None and runner._world.id == server.episode

    def test_the_reset_installs_the_opendrive_again(self, tmp_path, monkeypatch):
        xodr = tmp_path / "odaiba.xodr"
        xodr.write_text("<OpenDRIVE/>")
        installed = tmp_path / "installed.xodr"
        monkeypatch.setenv("ODAIBA_GENERATED_PATH", str(installed))
        server = _SharedServer(("Odaiba_generated",), "Odaiba_generated")
        runner = _runner_on(server)
        with patch(
            "autoware_carla_scenario.scenario_runner.map_asset_env_var",
            return_value="ODAIBA_GENERATED_PATH",
        ):
            runner.load_map_by_overwriting_xodr(xodr, "Odaiba_generated")
            # Another queue installs its own OpenDRIVE for the same map.
            installed.write_text("<OpenDRIVE>someone else's</OpenDRIVE>")

            runner.reset_world("Case1")

        assert server.loads == ["Odaiba_generated", "Odaiba_generated"]
        assert installed.read_text() == "<OpenDRIVE/>"

    def test_a_runner_that_loaded_no_map_resets_to_the_current_one(self):
        """As before: the server's map, and no world fetched per run."""
        server = _SharedServer(("Odaiba_generated", "Town10HD_Opt"), "Town10HD_Opt")
        runner = _runner_on(server)
        held = server.get_world()
        runner._world = held

        _set_up_once(runner)
        assert runner._world is held
        runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt"]

    def test_the_normal_path_loads_no_more_than_before(self):
        """One load at the start and one per reset; no map read to check."""
        server = _SharedServer(("Odaiba_generated",), "Town10HD_Opt")
        runner = _runner_on(server)
        runner.load_map_by_name("Odaiba_generated")

        for case in range(3):
            _set_up_once(runner)
            runner.reset_world(f"Case{case}")

        assert server.loads == ["Odaiba_generated"] * 4
        assert server.map_reads == 0

    def test_a_negative_episode_id_is_an_id(self, caplog):
        """The id is a random 64-bit number, which typesafe-carla hands over
        signed: a negative one is the same episode for all that, not an
        unknown one to load the map again for."""
        server = _SharedServer(
            ("Odaiba_generated",), "Town10HD_Opt", episode=-1240847540305438359
        )
        runner = _runner_on(server)
        runner.load_map_by_name("Odaiba_generated")

        with caplog.at_level("WARNING", logger="autoware_carla_scenario"):
            for case in range(3):
                _set_up_once(runner)
                runner.reset_world(f"Case{case}")

        assert server.loads == ["Odaiba_generated"] * 4
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_an_unknown_episode_is_never_the_batch_one(self):
        from autoware_carla_scenario.world_reset import episode_of, same_episode

        world = MagicMock()
        type(world).id = property(lambda self: (_ for _ in ()).throw(RuntimeError()))

        assert episode_of(world) is None
        assert not same_episode(None, None)
        assert not same_episode(None, 7)
        assert same_episode(-7, -7)


class TestTheResetReleasesTheRunsSensors:
    """The reset destroys the run's actors through the server, then reloads.

    The reload alone does not release a sensor that is still listening. A
    ``CollisionCondition`` leaves one in every run, and its stream keeps the
    server and the client busy for the rest of the batch, so the tick loop
    slows a little more with every case (#290).
    """

    @staticmethod
    def _server(events: list, actors) -> _SharedServer:
        server = _SharedServer(("Town10HD_Opt",), "Town10HD_Opt")
        plain_get_world = server.get_world
        plain_load_world = server.load_world

        def get_world() -> MagicMock:
            world = plain_get_world()
            world.get_actors.side_effect = actors
            return world

        def load_world(name: str) -> MagicMock:
            events.append(("load", name))
            return plain_load_world(name)

        server.get_world = get_world  # type: ignore[method-assign]
        server.load_world = load_world  # type: ignore[method-assign]
        return server

    def test_a_listening_sensor_is_destroyed_before_the_reload(self):
        events: list = []

        class _Logged(_FakeActor):
            def destroy(self) -> None:
                super().destroy()
                events.append(("destroy", self.id))

        npc = _Logged(1, "npc1")
        collision = _Logged(2, parent=npc)
        left = [npc, collision]

        def actors() -> _FakeActorList:
            alive = [a for a in left if not a.destroyed]
            return _FakeActorList(
                [a for a in alive if a.parent is None],
                [a for a in alive if a.parent is not None],
            )

        runner = _runner_on(self._server(events, actors))
        runner.load_map_by_name("Town10HD_Opt")
        events.clear()

        runner.reset_world("Case0")

        assert events == [("destroy", 1), ("destroy", 2), ("load", "Town10HD_Opt")]

    def test_the_map_is_reloaded_even_when_the_destroy_fails(self, caplog):
        def actors() -> _FakeActorList:
            raise RuntimeError("time-out of 10000ms while waiting for the simulator")

        events: list = []
        runner = _runner_on(self._server(events, actors))
        runner.load_map_by_name("Town10HD_Opt")
        events.clear()

        with caplog.at_level("WARNING", logger="autoware_carla_scenario"):
            runner.reset_world("Case0")

        assert events == [("load", "Town10HD_Opt")]
        assert any("before the reload failed" in r.getMessage() for r in caplog.records)


def _never(label: str):
    """A condition that never fires: the other half of a composite."""
    from autoware_carla_scenario import BaseCondition

    class _Never(BaseCondition):
        def check(self, world, elapsed):
            return None

    return _Never(label=label)


class TestTheDefaultTimeout:
    """The runner's timeout is a default: a scenario's own timeout replaces it.

    A red light whose drawn red phase is 13.4 s, with seven seconds allowed
    after the green, has to run for 20.4 s; a run whose default is twelve
    used to fail it before the light ever turned green.
    """

    @staticmethod
    def _scenario(*fail_conditions):
        import typesafe_carla.carla as carla

        from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform

        class _Scenario(BaseScenario):
            def setup(self) -> None:
                for condition in fail_conditions:
                    self.register_fail_condition(condition)

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
        return scenario

    @staticmethod
    def _timeouts(scenario) -> list[tuple[str, float]]:
        from autoware_carla_scenario import TimeoutCondition

        return [
            (c.label, c.timeout_seconds)
            for c in scenario._fail_conditions
            if isinstance(c, TimeoutCondition)
        ]

    def test_a_scenario_without_a_timeout_gets_the_default(self):
        from autoware_carla_scenario.scenario_runner import _register_default_timeout

        scenario = self._scenario()
        assert _register_default_timeout(scenario, 12.0, "S") is True
        assert self._timeouts(scenario) == [("default_timeout", 12.0)]

    def test_a_scenario_with_its_own_timeout_keeps_it_alone(self):
        from autoware_carla_scenario import TimeoutCondition
        from autoware_carla_scenario.scenario_runner import _register_default_timeout

        scenario = self._scenario(
            TimeoutCondition(13.4 + 7.0, label="scenario_timeout")
        )
        assert _register_default_timeout(scenario, 12.0, "S") is False
        assert self._timeouts(scenario) == [("scenario_timeout", 20.4)]

    def test_a_shorter_own_timeout_is_kept_too(self):
        from autoware_carla_scenario import TimeoutCondition
        from autoware_carla_scenario.scenario_runner import _register_default_timeout

        scenario = self._scenario(TimeoutCondition(5.0, label="scenario_timeout"))
        assert _register_default_timeout(scenario, 60.0, "S") is False
        assert self._timeouts(scenario) == [("scenario_timeout", 5.0)]

    def test_a_timeout_inside_an_or_counts(self):
        from autoware_carla_scenario import OrCondition, TimeoutCondition
        from autoware_carla_scenario.scenario_runner import _register_default_timeout

        scenario = self._scenario(
            OrCondition(
                [
                    _never("other"),
                    TimeoutCondition(20.0, label="scenario_timeout"),
                ]
            )
        )
        assert _register_default_timeout(scenario, 12.0, "S") is False
        assert scenario._fail_conditions[-1].label != "default_timeout"

    def test_a_timeout_inside_a_sticky_counts(self):
        from autoware_carla_scenario import StickyCondition, TimeoutCondition
        from autoware_carla_scenario.scenario_runner import _register_default_timeout

        sticky = StickyCondition(TimeoutCondition(20.0, label="scenario_timeout"))
        scenario = self._scenario(sticky)
        assert _register_default_timeout(scenario, 12.0, "S") is False
        assert scenario._fail_conditions == [sticky]
        # And it does fail the run when the timeout does.
        assert sticky.check(MagicMock(), 20.0) is not None

    @pytest.mark.parametrize("wrapper", ["and", "not", "persistent"])
    def test_a_timeout_that_does_not_bound_the_run_does_not_count(self, wrapper):
        from autoware_carla_scenario import (
            AndCondition,
            NotCondition,
            PersistentCondition,
            TimeoutCondition,
        )
        from autoware_carla_scenario.scenario_runner import _register_default_timeout

        timeout = TimeoutCondition(20.0, label="t")
        condition = {
            "and": lambda: AndCondition([_never("other"), timeout]),
            "not": lambda: NotCondition(timeout),
            "persistent": lambda: PersistentCondition(timeout, duration=0.1),
        }[wrapper]()
        scenario = self._scenario(condition)
        assert _register_default_timeout(scenario, 12.0, "S") is True
        assert scenario._fail_conditions[-1].label == "default_timeout"

    def test_a_persistent_timeout_never_fires(self):
        """Why a PersistentCondition does not count: it resets on a failure."""
        from autoware_carla_scenario import PersistentCondition, TimeoutCondition

        persistent = PersistentCondition(TimeoutCondition(1.0, label="t"), duration=0.1)
        assert all(persistent.check(MagicMock(), t) is None for t in (0.5, 1.0, 99.0))

    @pytest.mark.parametrize("timeout", [float("nan"), float("inf")])
    def test_a_non_finite_timeout_does_not_bound_the_run(self, timeout):
        """It never fires, so the default timeout is still added (#934)."""
        from autoware_carla_scenario import OrCondition, TimeoutCondition
        from autoware_carla_scenario.scenario_runner import (
            _bounds_the_run,
            _register_default_timeout,
        )

        # The constructor refuses one; a timeout changed afterwards is the
        # case left.
        own = TimeoutCondition(20.0, label="scenario_timeout")
        own.timeout_seconds = timeout
        assert _bounds_the_run(own) is False
        assert _bounds_the_run(OrCondition([_never("other"), own])) is False
        scenario = self._scenario(own)
        assert _register_default_timeout(scenario, 12.0, "S") is True
        assert scenario._fail_conditions[-1].label == "default_timeout"

    @pytest.mark.parametrize("timeout", ["nan", "inf", "-inf", "0", "-1"])
    def test_the_command_line_refuses_a_timeout_that_is_not_positive_and_finite(
        self, timeout
    ):
        """Hydra reads ``scenario.timeout_seconds=nan`` as a float (#934)."""
        from omegaconf import OmegaConf

        from autoware_carla_scenario.run_timeout import run_timeout_seconds

        cfg = OmegaConf.create({"scenario": {"timeout_seconds": float(timeout)}})
        with pytest.raises(ValueError, match="scenario.timeout_seconds"):
            run_timeout_seconds(cfg)

    def test_the_command_line_timeout_is_read_as_given(self):
        from omegaconf import OmegaConf

        from autoware_carla_scenario.run_timeout import run_timeout_seconds

        assert run_timeout_seconds(OmegaConf.create({"scenario": {}})) == 60.0
        cfg = OmegaConf.create({"scenario": {"timeout_seconds": 15}})
        assert run_timeout_seconds(cfg) == 15.0

    def test_the_registered_conditions_fire_at_the_scenarios_own_limit(self):
        """What is left registered fails at 20.4 s, and not at 12 s."""
        from autoware_carla_scenario import TimeoutCondition
        from autoware_carla_scenario.scenario_runner import _register_default_timeout

        scenario = self._scenario(TimeoutCondition(20.4, label="scenario_timeout"))
        _register_default_timeout(scenario, 12.0, "S")
        world = MagicMock()

        def fired(elapsed: float) -> list[str]:
            return [
                c.label
                for c in scenario._fail_conditions
                if c.check(world, elapsed) is not None
            ]

        assert fired(12.0) == []
        assert fired(20.4) == ["scenario_timeout"]


class TestRunScenarioRegistersTheDefaultTimeout:
    """The wiring in ``run_scenario``: after setup and ``ego_existence``.

    The run is stopped where the clock would start, right after the default
    timeout is (or is not) registered.
    """

    @staticmethod
    def _run(tmp_path, scenario, timeout_seconds: float) -> None:
        runner = _make_runner(None)
        runner.output_dir = tmp_path
        runner._world = MagicMock()
        with (
            patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
            patch(
                "autoware_carla_scenario.scenario_runner._ScenarioClock",
                side_effect=_Stop,
            ),
            pytest.raises(_Stop),
        ):
            runner.run_scenario(
                scenario, timeout_seconds=timeout_seconds, reset_world=False
            )

    @staticmethod
    def _scenario(*own):
        """A stand-in scenario whose fail conditions are a real list."""
        scenario = MagicMock()
        scenario.STABILIZE_TICKS = 0
        scenario._fail_conditions = []
        scenario.register_fail_condition.side_effect = scenario._fail_conditions.append
        scenario.setup.side_effect = lambda: [
            scenario.register_fail_condition(c) for c in own
        ]
        ego = _FakeEgo(ready_after=0)
        ego.set_traffic_backend = MagicMock()  # type: ignore[attr-defined]
        ego.spawn = MagicMock()  # type: ignore[attr-defined]
        ego.set_initial_speeds = MagicMock()  # type: ignore[attr-defined]
        ego.on_scenario_start = MagicMock()  # type: ignore[attr-defined]
        ego.on_scenario_end = MagicMock()  # type: ignore[attr-defined]
        ego.destroy = MagicMock()  # type: ignore[attr-defined]
        ego.use_autopilot = True  # type: ignore[attr-defined]
        scenario.create_ego.return_value = ego
        return scenario

    @staticmethod
    def _labels(scenario) -> list[str]:
        return [c.label for c in scenario._fail_conditions]

    def test_the_default_comes_after_setup_and_ego_existence(self, tmp_path):
        scenario = self._scenario(_never("own"))
        self._run(tmp_path, scenario, 12.0)
        assert self._labels(scenario) == ["own", "ego_existence", "default_timeout"]
        assert scenario._fail_conditions[-1].timeout_seconds == 12.0

    def test_no_default_for_a_scenario_with_its_own(self, tmp_path):
        from autoware_carla_scenario import TimeoutCondition

        scenario = self._scenario(TimeoutCondition(20.4, label="scenario_timeout"))
        self._run(tmp_path, scenario, 12.0)
        assert self._labels(scenario) == ["scenario_timeout", "ego_existence"]

    def test_a_retry_does_not_add_a_second_default(self, tmp_path):
        """setup() runs again on the same object, after the earlier attempt's
        registrations are reset: the retry is given the default again, once."""
        scenario = self._scenario()
        scenario._reset_registrations.side_effect = scenario._fail_conditions.clear
        self._run(tmp_path, scenario, 12.0)
        self._run(tmp_path, scenario, 12.0)
        assert self._labels(scenario).count("default_timeout") == 1


class TestRunScenarioForgetsTheLastRunsWeather:
    """A soft world reset keeps the episode, so every run starts by forgetting
    the visibility and the tire friction the last run left (#795)."""

    def test_the_tire_friction_the_last_run_set_is_forgotten(self, tmp_path):
        from types import SimpleNamespace

        from autoware_carla_scenario.actions.environment import (
            apply_tire_friction,
            world_tire_friction,
        )
        from autoware_carla_scenario.visibility import (
            set_world_visibility,
            world_visibility,
        )

        class _Actors(list):
            def filter(self, _pattern):
                return []

        last_run = SimpleNamespace(
            id=795, get_actors=lambda ids=None: _Actors(), get_snapshot=lambda: []
        )
        apply_tire_friction(last_run, 0.55)  # type: ignore[arg-type]
        set_world_visibility(795, 80.0)

        runner = _make_runner(None)
        runner.output_dir = tmp_path
        runner._world = MagicMock()
        scenario = MagicMock()
        scenario.STABILIZE_TICKS = 0
        scenario.setup.side_effect = _Stop
        with (
            patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
            pytest.raises(_Stop),
        ):
            runner.run_scenario(scenario, reset_world=False)

        assert world_visibility(795) is None
        assert world_tire_friction(795) is None
