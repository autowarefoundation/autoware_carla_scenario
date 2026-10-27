"""Unit tests for ScenarioQueue."""

from __future__ import annotations

import gc
import weakref
from unittest.mock import MagicMock, patch

import typesafe_carla.carla as carla
import pytest

from autoware_carla_scenario import (
    BaseScenario,
    EgoConfig,
    ScenarioQueue,
    ScenarioResult,
    ScenarioRunner,
    SpawnTransform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ego_config() -> EgoConfig:
    return EgoConfig(
        spawn_location=SpawnTransform(carla.Transform(carla.Location(x=0, y=0, z=0))),
        vehicle_type="vehicle.mini.cooper",
    )


class _NullScenario(BaseScenario):
    """Minimal scenario: setup is a no-op, is_done always returns True."""

    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return True


class _HeavyScenario(_NullScenario):
    """A scenario holding what a run leaves behind (a stand-in for its
    entities, conditions and the ego's driver)."""

    payload: bytearray

    def __init__(self, ego_config: EgoConfig) -> None:
        super().__init__(ego_config)
        self.payload = bytearray(1 << 20)


class _RecordingRunner:
    """A runner that keeps no scenario: it records, when each one starts,
    which scenarios it ran before are still alive.  (A MagicMock keeps every
    argument it was called with.)"""

    def __init__(self) -> None:
        self._ran: list[weakref.ref[BaseScenario]] = []
        self.alive_before: list[list[int]] = []
        self.timeouts: list[float | None] = []
        self.results: list[ScenarioResult] = []

    def run_scenario(
        self,
        scenario: BaseScenario,
        *,
        timeout_seconds: float | None = None,
        case: object = None,
        reset_world: bool = True,
    ) -> ScenarioResult:
        gc.collect()
        self.alive_before.append(
            [k for k, ref in enumerate(self._ran) if ref() is not None]
        )
        self._ran.append(weakref.ref(scenario))
        self.timeouts.append(timeout_seconds)
        result = ScenarioResult(passed=True, message="ok", elapsed_seconds=0.0)
        self.results.append(result)
        return result

    def reset_world(self) -> None:
        pass


class _SecondCaseRaises:
    """A runner whose run of one scenario (``failing``) raises *error* on
    every attempt; it records which scenarios it ran, by position, and when
    it reloaded the world."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.failing: BaseScenario | None = None
        self.order: list[int] = []
        self.ran: list[int] = []
        self.events: list[str] = []

    def run_scenario(self, scenario: BaseScenario, **_: object) -> ScenarioResult:
        if id(scenario) not in self.order:
            self.order.append(id(scenario))
        self.ran.append(self.order.index(id(scenario)))
        self.events.append("run")
        if scenario is self.failing:
            raise self.error
        return ScenarioResult(passed=True, message="ok", elapsed_seconds=1.0)

    def reset_world(self) -> None:
        self.events.append("reset")


def _make_queue_with_mock_runner() -> tuple[ScenarioQueue, MagicMock]:
    """Return a ScenarioQueue whose internal runner is a MagicMock.

    The server is also mocked so no real CARLA connection is attempted.
    """
    with (
        patch(
            "autoware_carla_scenario.scenario_queue.CarlaServerManager"
        ) as mock_server_cls,
        patch(
            "autoware_carla_scenario.scenario_queue.ScenarioRunner"
        ) as mock_runner_cls,
    ):
        mock_server_cls.return_value = MagicMock()
        mock_runner = MagicMock(spec=ScenarioRunner)
        mock_runner_cls.return_value = mock_runner

        queue = ScenarioQueue(map_name="Town10HD_Opt")
        queue.start()  # sets self._runner to mock_runner

    return queue, mock_runner


# ---------------------------------------------------------------------------
# ScenarioQueue – unit tests (no CARLA required)
# ---------------------------------------------------------------------------


class TestScenarioQueueUnit:
    def test_start_drops_the_map_an_earlier_queue_left(self) -> None:
        """A queue with no Lanelet2 map builds no MapManager map, and must not
        run on the one an earlier queue of this process left there: the runs
        read that map (run_carla_map) for snaps, spawns and the map's name."""
        from autoware_carla_scenario.coordinate.map_manager import MapManager

        found = MapManager._instance
        try:
            MapManager.reset()
            MapManager.get_instance()._carla_map = MagicMock(name="Odaiba map")
            _make_queue_with_mock_runner()  # map_name only: no Lanelet2 map
            assert MapManager.get_instance().carla_map is None
        finally:
            MapManager._instance = found

    def test_negative_cooldown_max_retries_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cooldown_max_retries must be >= 0"):
            ScenarioQueue(cooldown_max_retries=-1)

    def test_empty_queue_len_is_zero(self) -> None:
        queue = ScenarioQueue()
        assert len(queue) == 0

    def test_add_increases_length(self) -> None:
        queue = ScenarioQueue()
        assert len(queue) == 0
        queue.add(_NullScenario(_make_ego_config()))
        assert len(queue) == 1
        queue.add(_NullScenario(_make_ego_config()))
        assert len(queue) == 2

    def test_run_all_raises_if_not_started(self) -> None:
        queue = ScenarioQueue()
        queue.add(_NullScenario(_make_ego_config()))
        with pytest.raises(RuntimeError, match="started"):
            queue.run_all()

    def test_result_for_raises_before_run_all(self) -> None:
        queue = ScenarioQueue()
        scenario = _NullScenario(_make_ego_config())
        queue.add(scenario)
        with pytest.raises(KeyError):
            queue.result_for(scenario)

    def test_run_all_calls_run_scenario_for_each(self) -> None:
        queue, mock_runner = _make_queue_with_mock_runner()
        s1 = _NullScenario(_make_ego_config())
        s2 = _NullScenario(_make_ego_config())
        queue.add(s1)
        queue.add(s2)

        expected = ScenarioResult(passed=True, message="ok", elapsed_seconds=1.0)
        mock_runner.run_scenario.return_value = expected

        results = queue.run_all()

        assert mock_runner.run_scenario.call_count == 2
        assert results == [expected, expected]

    def test_run_all_preserves_order(self) -> None:
        queue, mock_runner = _make_queue_with_mock_runner()
        scenarios = [_NullScenario(_make_ego_config()) for _ in range(3)]
        for s in scenarios:
            queue.add(s)

        expected_results = [
            ScenarioResult(passed=True, message=f"r{i}", elapsed_seconds=float(i))
            for i in range(3)
        ]
        mock_runner.run_scenario.side_effect = expected_results

        results = queue.run_all()
        assert results == expected_results

    def test_result_for_returns_correct_result(self) -> None:
        queue, mock_runner = _make_queue_with_mock_runner()
        s1 = _NullScenario(_make_ego_config())
        s2 = _NullScenario(_make_ego_config())
        queue.add(s1)
        queue.add(s2)

        r1 = ScenarioResult(passed=True, message="s1", elapsed_seconds=1.0)
        r2 = ScenarioResult(passed=False, message="s2", elapsed_seconds=2.0)
        mock_runner.run_scenario.side_effect = [r1, r2]

        queue.run_all()

        assert queue.result_for(s1) is r1
        assert queue.result_for(s2) is r2

    def test_run_all_passes_correct_scenario_to_runner(self) -> None:
        queue, mock_runner = _make_queue_with_mock_runner()
        s1 = _NullScenario(_make_ego_config())
        s2 = _NullScenario(_make_ego_config())
        queue.add(s1)
        queue.add(s2)
        mock_runner.run_scenario.return_value = ScenarioResult(
            passed=True, message="ok", elapsed_seconds=0.0
        )

        queue.run_all()

        called_scenarios = [
            call.args[0] for call in mock_runner.run_scenario.call_args_list
        ]
        assert called_scenarios[0] is s1
        assert called_scenarios[1] is s2

    def test_run_all_passes_each_scenarios_own_timeout(self) -> None:
        queue, mock_runner = _make_queue_with_mock_runner()
        s1 = _NullScenario(_make_ego_config())
        s2 = _NullScenario(_make_ego_config())
        queue.add(s1, timeout_seconds=1.5)
        queue.add(s2)
        mock_runner.run_scenario.return_value = ScenarioResult(
            passed=True, message="ok", elapsed_seconds=0.0
        )

        queue.run_all()

        timeouts = [
            call.kwargs["timeout_seconds"]
            for call in mock_runner.run_scenario.call_args_list
        ]
        assert timeouts == [1.5, None]  # None: the runner's own default

    def test_run_all_passes_each_scenarios_odd_case(self) -> None:
        queue, mock_runner = _make_queue_with_mock_runner()
        s1 = _NullScenario(_make_ego_config())
        s2 = _NullScenario(_make_ego_config())
        case = (MagicMock(), MagicMock())
        queue.add(s1, case=case)
        queue.add(s2)
        mock_runner.run_scenario.return_value = ScenarioResult(
            passed=True, message="ok", elapsed_seconds=0.0
        )

        queue.run_all()

        cases = [
            call.kwargs["case"] for call in mock_runner.run_scenario.call_args_list
        ]
        assert cases == [case, None]

    def test_the_world_is_reset_after_every_scenario_but_the_last(self) -> None:
        queue, mock_runner = _make_queue_with_mock_runner()
        for _ in range(3):
            queue.add(_NullScenario(_make_ego_config()))
        mock_runner.run_scenario.return_value = ScenarioResult(
            passed=True, message="ok", elapsed_seconds=0.0
        )

        queue.run_all()

        resets = [
            call.kwargs["reset_world"]
            for call in mock_runner.run_scenario.call_args_list
        ]
        assert resets == [True, True, False]
        mock_runner.reset_world.assert_not_called()

    def test_a_second_run_all_first_resets_the_world_the_first_left(self) -> None:
        queue, mock_runner = _make_queue_with_mock_runner()
        queue.add(_NullScenario(_make_ego_config()))
        mock_runner.run_scenario.return_value = ScenarioResult(
            passed=True, message="ok", elapsed_seconds=0.0
        )

        queue.run_all()
        mock_runner.reset_world.assert_not_called()
        queue.run_all()

        mock_runner.reset_world.assert_called_once_with()

    def test_a_releasing_run_lets_go_of_each_scenario_once_it_has_run(
        self,
    ) -> None:
        queue = ScenarioQueue(server=MagicMock())
        runner = _RecordingRunner()
        queue._runner = runner  # type: ignore[assignment]
        refs = []
        for k in range(4):
            scenario = _HeavyScenario(_make_ego_config())
            refs.append(weakref.ref(scenario))
            queue.add(scenario, timeout_seconds=float(k + 1), case=MagicMock())
        del scenario

        results = queue.run_all(release=True)

        # While each case ran, every case before it was already gone.
        assert runner.alive_before == [[], [], [], []]
        assert runner.timeouts == [1.0, 2.0, 3.0, 4.0]
        assert results == runner.results
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert len(queue) == 0
        assert queue._timeouts == {} and queue._cases == {}
        assert queue._scenario_results == {}

    def test_a_releasing_run_retries_with_the_same_scenario(self) -> None:
        """A retry gets the scenario, timeout and case it was added with,
        though the queue has let go of them by then."""

        class _FailsOnce(_RecordingRunner):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[int, float | None, object]] = []

            def run_scenario(
                self, scenario, *, timeout_seconds=None, case=None, reset_world=True
            ):  # noqa: ANN001, ANN202
                self.calls.append((id(scenario), timeout_seconds, case))
                if len(self.calls) == 1:
                    raise RuntimeError("CARLA went away")
                return super().run_scenario(
                    scenario,
                    timeout_seconds=timeout_seconds,
                    case=case,
                    reset_world=reset_world,
                )

        queue = ScenarioQueue(server=MagicMock(), cooldown_max_retries=1)
        runner = _FailsOnce()
        queue._runner = runner  # type: ignore[assignment]
        scenario = _HeavyScenario(_make_ego_config())
        case = MagicMock()
        queue.add(scenario, timeout_seconds=7.0, case=case)

        results = queue.run_all(release=True)

        assert len(results) == 1
        assert runner.calls == [(id(scenario), 7.0, case)] * 2

    def test_a_releasing_run_that_fails_leaves_nothing_keyed(self) -> None:
        """A case that fails every attempt ends the run; the cases after it
        are out of the queue and leave nothing keyed by their ids."""

        class _AlwaysFails(_RecordingRunner):
            def run_scenario(self, scenario, **_):  # noqa: ANN001, ANN003, ANN202
                raise RuntimeError("no")

        queue = ScenarioQueue(server=MagicMock())
        queue._runner = _AlwaysFails()  # type: ignore[assignment]
        for k in range(3):
            queue.add(
                _HeavyScenario(_make_ego_config()),
                timeout_seconds=float(k),
                case=MagicMock(),
            )

        with pytest.raises(RuntimeError, match="failed after 1 attempt"):
            queue.run_all(release=True)

        assert len(queue) == 0
        assert queue._timeouts == {} and queue._cases == {}

    @staticmethod
    def _queue_whose_second_case_cannot_run(
        error: BaseException, *, retries: int = 1, cases: int = 3
    ) -> tuple[ScenarioQueue, "_SecondCaseRaises", list[BaseScenario]]:
        queue = ScenarioQueue(server=MagicMock(), cooldown_max_retries=retries)
        runner = _SecondCaseRaises(error)
        queue._runner = runner  # type: ignore[assignment]
        scenarios: list[BaseScenario] = [
            _NullScenario(_make_ego_config()) for _ in range(cases)
        ]
        for scenario in scenarios:
            queue.add(scenario)
        runner.failing = scenarios[1]
        return queue, runner, scenarios

    @pytest.mark.parametrize("release", [False, True])
    def test_a_run_that_continues_on_error_records_the_case_and_goes_on(
        self, release: bool
    ) -> None:
        """A case that raises on every attempt is a failed result naming the
        error; the world is reloaded and the next case runs."""
        queue, runner, scenarios = self._queue_whose_second_case_cannot_run(
            RuntimeError("ground_projection returned no hit at (1.0, 2.0)")
        )

        results = queue.run_all(release=release, continue_on_error=True)

        assert len(results) == 3
        assert results[0].passed and results[2].passed
        assert not results[1].passed
        assert results[1].message == (
            "Errored after 2 attempt(s): RuntimeError: "
            "ground_projection returned no hit at (1.0, 2.0)"
        )
        # Both attempts of the second case, then the third case.
        assert runner.ran == [0, 1, 1, 2]
        assert queue.errored_indices() == [1]
        # Reloaded after the errored case, which never reached its own reload.
        assert runner.events == ["run", "run", "run", "reset", "run"]
        if not release:
            assert queue.result_for(scenarios[1]) is results[1]

    def test_an_errored_last_case_leaves_the_world_to_the_next_run(self) -> None:
        queue, runner, _ = self._queue_whose_second_case_cannot_run(
            RuntimeError("no"), retries=0, cases=2
        )

        results = queue.run_all(continue_on_error=True)

        assert [r.passed for r in results] == [True, False]
        assert queue.errored_indices() == [1]
        assert runner.events == ["run", "run"]
        assert queue._world_left_as_is

    def test_a_run_forgets_the_errors_of_the_last(self) -> None:
        queue, runner, _ = self._queue_whose_second_case_cannot_run(
            RuntimeError("no"), retries=0
        )
        queue.run_all(continue_on_error=True)
        runner.failing = None

        queue.run_all(continue_on_error=True)

        assert queue.errored_indices() == []

    def test_a_run_ends_at_a_case_that_raises_by_default(self) -> None:
        queue, runner, _ = self._queue_whose_second_case_cannot_run(RuntimeError("no"))

        with pytest.raises(RuntimeError, match="failed after 2 attempt"):
            queue.run_all()

        assert runner.ran == [0, 1, 1]

    def test_an_interrupt_ends_a_run_that_continues_on_error(self) -> None:
        queue, runner, _ = self._queue_whose_second_case_cannot_run(KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            queue.run_all(continue_on_error=True)

        # Not retried, and the third case never ran.
        assert runner.ran == [0, 1]

    def test_a_run_that_does_not_release_keeps_its_scenarios(self) -> None:
        queue = ScenarioQueue(server=MagicMock())
        runner = _RecordingRunner()
        queue._runner = runner  # type: ignore[assignment]
        scenarios = [_HeavyScenario(_make_ego_config()) for _ in range(2)]
        for scenario in scenarios:
            queue.add(scenario)

        queue.run_all()
        again = queue.run_all()

        assert len(queue) == 2
        assert len(again) == 2
        assert queue.result_for(scenarios[1]) is runner.results[-1]

    def test_owns_server_flag_when_no_server_provided(self) -> None:
        with patch("autoware_carla_scenario.scenario_queue.CarlaServerManager"):
            queue = ScenarioQueue()
        assert queue._owns_server is True

    def test_borrows_server_when_provided(self) -> None:
        mock_server = MagicMock()
        queue = ScenarioQueue(server=mock_server)
        assert queue._owns_server is False
        assert queue._server is mock_server

    def test_stop_does_not_call_server_stop_when_borrowed(self) -> None:
        mock_server = MagicMock()
        with patch("autoware_carla_scenario.scenario_queue.ScenarioRunner"):
            queue = ScenarioQueue(server=mock_server)
            queue.start()
            queue.stop()

        mock_server.stop.assert_not_called()

    def test_context_manager_calls_start_and_stop(self) -> None:
        with (
            patch(
                "autoware_carla_scenario.scenario_queue.CarlaServerManager"
            ) as mock_server_cls,
            patch("autoware_carla_scenario.scenario_queue.ScenarioRunner"),
        ):
            mock_server = MagicMock()
            mock_server_cls.return_value = mock_server

            queue = ScenarioQueue()
            with queue:
                assert queue._runner is not None

            mock_server.start.assert_called_once()
            mock_server.stop.assert_called_once()
            assert queue._runner is None


# ---------------------------------------------------------------------------
# Integration tests – real CARLA (skipped if CARLA unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestScenarioQueueIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_carla(self, carla_queue: ScenarioQueue) -> None:
        """Require the session-scoped carla_queue fixture."""

    def test_queue_has_results_after_run(self, carla_queue: ScenarioQueue) -> None:
        """After run_all(), the queue must have non-empty results."""
        assert len(carla_queue._results) > 0

    def test_all_results_are_scenario_result_instances(
        self, carla_queue: ScenarioQueue
    ) -> None:
        for result in carla_queue._results:
            assert isinstance(result, ScenarioResult)
