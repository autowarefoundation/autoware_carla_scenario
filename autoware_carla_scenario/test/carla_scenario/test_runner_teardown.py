"""Unit tests for what run_scenario does at its end (no CARLA server)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest

from autoware_carla_scenario import ScenarioRunner
from autoware_carla_scenario.debug_markers import DebugMarkers, DebugPolyline

from .test_scenario_runner import _mock_scenario


class _Ego:
    attaches_to_existing_actor = False
    carried_actor_ids: frozenset[int] = frozenset()
    actor = None
    use_autopilot = True
    is_initialized = True
    termination_requested = False

    def __init__(self) -> None:
        self.markers = DebugMarkers()
        self.markers.add_polyline(DebugPolyline([(1.0, 0.0, 0.3), (1.0, 1.0, 0.3)]))
        for name in (
            "set_traffic_backend",
            "set_initial_speeds",
            "on_scenario_start",
            "on_scenario_end",
            "destroy",
        ):
            setattr(self, name, MagicMock())
        self.spawn = MagicMock(return_value=SimpleNamespace(id=7, type_id="vehicle.x"))

    def on_tick(self, world: Any, elapsed: float) -> None:
        del world, elapsed

    def debug_markers(self) -> DebugMarkers:
        return self.markers


def _npc_monitor(finished: list[Any] | None = None) -> MagicMock:
    """A stand-in NpcCollisionMonitor that hears nothing on a tick and ends
    *finished* when the run does."""
    monitor = MagicMock()
    monitor.tick.return_value = ([], [], [])
    monitor.finish.return_value = [] if finished is None else finished
    return monitor


def _run(
    tmp_path: Path,
    trajectory: Any,
    *,
    stop_recorder_fails: bool = False,
    scenario_out: list[Any] | None = None,
    traffic_backend: Any = None,
    write_result: Any = None,
    rerun: Any = None,
    npc_monitor: Any = None,
    tick_error: BaseException | None = None,
    coverage_end_error: BaseException | None = None,
    settings_before: Any = None,
) -> ScenarioRunner:
    """Run a scenario of three ticks through run_scenario, on stand-ins."""
    with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
        runner = ScenarioRunner(MagicMock(), output_dir=tmp_path)
    if stop_recorder_fails:
        runner._client.stop_recorder.side_effect = RuntimeError("std::exception")
    if traffic_backend is not None:
        runner._traffic_backend = traffic_backend
    runner.rerun = rerun
    world = MagicMock()
    frame = {"n": 0}

    def tick() -> None:
        frame["n"] += 1

    world.tick.side_effect = tick
    if settings_before is not None:
        found = settings_before
        world.get_settings.side_effect = lambda: SimpleNamespace(**vars(found))
    world.get_snapshot.side_effect = lambda: SimpleNamespace(frame=frame["n"])
    runner._world = world
    runner._pace_tick = lambda: None  # type: ignore[method-assign]
    scenario = _mock_scenario()
    scenario.STABILIZE_TICKS = 2
    scenario._pass_conditions = []
    scenario._fail_conditions = []
    scenario._pre_tick_actions = []
    scenario._post_tick_actions = []
    scenario._pre_tick_callbacks = []
    scenario._post_tick_callbacks = []
    if tick_error is not None:
        error = tick_error

        def fail(world: Any) -> None:
            raise error

        scenario._post_tick_callbacks = [fail]
    scenario.is_done.side_effect = [False, False, False, True]
    scenario.create_ego.return_value = _Ego()
    if scenario_out is not None:
        scenario_out.append(scenario)
    with (
        patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
        patch("autoware_carla_scenario.scenario_runner.CoverageCollector") as cov,
        patch(
            "autoware_carla_scenario.scenario_runner._write_result",
            new=write_result or MagicMock(),
        ),
        patch("autoware_carla_scenario.scenario_runner._record_run"),
        patch(
            "autoware_carla_scenario.scenario_runner._npc_collision_monitor",
            new=MagicMock(return_value=npc_monitor or _npc_monitor()),
        ),
        patch("autoware_carla_scenario.scenario_runner._ScenarioClock") as clock,
        patch(
            "autoware_carla_scenario.scenario_runner.TrajectoryRecorder",
            new=MagicMock(return_value=trajectory),
        ),
    ):
        clock.return_value.simulated = 0.1
        if coverage_end_error is not None:
            cov.return_value.end.side_effect = coverage_end_error
        result = runner.run_scenario(scenario, reset_world=False)
    assert result is not None and result.passed
    return runner


def test_the_trajectory_log_ends_with_the_verdict(tmp_path: Path) -> None:
    trajectory = MagicMock()
    trajectory.active = True
    _run(tmp_path, trajectory)
    trajectory.record_result.assert_called_once()
    assert trajectory.record_result.call_args.args[0].passed
    trajectory.close.assert_called_once()


def test_the_log_goes_to_rerun_once_the_result_is_written(tmp_path: Path) -> None:
    # A log without its result record is converted with the result JSON
    # beside it: handed over before that is written, the converter could
    # read it half written.
    order: list[str] = []
    written: list[Any] = []

    def write_result(json_path: Path, *_: Any) -> None:
        order.append("result")
        written.append(_[3])

    trajectory = MagicMock()
    trajectory.active = True
    trajectory.path = tmp_path / "S.trajectory.jsonl"
    rerun = MagicMock()
    rerun.submit.side_effect = lambda path, label: order.append("rerun")
    _run(tmp_path, trajectory, write_result=write_result, rerun=rerun)
    assert order == ["result", "rerun"]
    rerun.submit.assert_called_once_with(trajectory.path, ANY)
    # The result names its recording all the same.
    assert written[0].rerun_recording == str(tmp_path / "S.rrd")


def test_the_coverage_and_gate_record_are_written_before_the_result(
    tmp_path: Path,
) -> None:
    # A job killed after writing its result then holds its gate record too:
    # a result without a record is a scenario without a gate (#1840).
    order: list[str] = []

    def write_result(*_: Any) -> None:
        order.append("result")

    with patch(
        "autoware_carla_scenario.scenario_runner._write_gate_record",
        new=MagicMock(side_effect=lambda *_, **kw: order.append(f"gate {kw}")),
    ):
        _run(tmp_path, MagicMock(active=False), write_result=write_result)
    assert order == ["gate {'result_follows': True}", "result"]


def test_a_result_that_cannot_be_written_leaves_the_gate_record(
    tmp_path: Path,
) -> None:
    write_gate_record = MagicMock()
    with (
        patch(
            "autoware_carla_scenario.scenario_runner._write_gate_record",
            new=write_gate_record,
        ),
        pytest.raises(OSError),
    ):
        _run(
            tmp_path,
            MagicMock(active=False),
            write_result=MagicMock(side_effect=OSError("disk full")),
        )
    write_gate_record.assert_called_once()


def test_the_log_goes_to_rerun_when_the_result_cannot_be_written(
    tmp_path: Path,
) -> None:
    trajectory = MagicMock()
    trajectory.active = True
    trajectory.path = tmp_path / "S.trajectory.jsonl"
    rerun = MagicMock()
    with pytest.raises(OSError):
        _run(
            tmp_path,
            trajectory,
            write_result=MagicMock(side_effect=OSError("disk full")),
            rerun=rerun,
        )
    rerun.submit.assert_called_once_with(trajectory.path, ANY)


@pytest.mark.parametrize(
    "error", [RuntimeError("CommonRoad worker died"), KeyboardInterrupt()]
)
def test_a_run_whose_metrics_raise_ends_its_log_without_the_loop_verdict(
    tmp_path: Path, error: BaseException
) -> None:
    # The verdict of the tick loop alone is not the run's: a MetricCondition
    # judged after it may fail the run.  No result is written, so the log
    # mirrors that and ends without one, and is still converted (#2067).
    trajectory = MagicMock()
    trajectory.active = True
    trajectory.path = tmp_path / "S.trajectory.jsonl"
    rerun = MagicMock()
    write_result = MagicMock()
    with (
        patch(
            "autoware_carla_scenario.scenario_runner._judge_metrics",
            new=MagicMock(side_effect=error),
        ),
        pytest.raises(type(error)),
    ):
        _run(tmp_path, trajectory, write_result=write_result, rerun=rerun)
    trajectory.record_result.assert_not_called()
    trajectory.close.assert_called_once()
    rerun.submit.assert_called_once_with(trajectory.path, ANY)
    write_result.assert_not_called()


def test_a_run_whose_coverage_end_raises_ends_its_log_without_a_verdict(
    tmp_path: Path,
) -> None:
    trajectory = MagicMock()
    trajectory.active = True
    trajectory.path = tmp_path / "S.trajectory.jsonl"
    rerun = MagicMock()
    write_result = MagicMock()
    with pytest.raises(ValueError, match="coverage broke"):
        _run(
            tmp_path,
            trajectory,
            write_result=write_result,
            rerun=rerun,
            coverage_end_error=ValueError("coverage broke"),
        )
    trajectory.record_result.assert_not_called()
    trajectory.close.assert_called_once()
    rerun.submit.assert_called_once_with(trajectory.path, ANY)
    write_result.assert_not_called()


def test_the_log_ends_with_the_verdict_the_metrics_returned(tmp_path: Path) -> None:
    # A MetricCondition judged after the loop decides the log's result, as
    # it does the result JSON's.
    trajectory = MagicMock()
    trajectory.active = True
    judged = MagicMock(
        side_effect=lambda scenario, recorder, world, result, *_: type(result)(
            passed=False, message="metric NG", elapsed_seconds=0.1
        )
    )
    with (
        patch("autoware_carla_scenario.scenario_runner._judge_metrics", new=judged),
        pytest.raises(AssertionError),
    ):
        # _run asserts a passing result; this one is not.
        _run(tmp_path, trajectory)
    trajectory.record_result.assert_called_once()
    assert trajectory.record_result.call_args.args[0].message == "metric NG"


def test_a_failing_recorder_stop_still_ends_the_trajectory_log(tmp_path: Path) -> None:
    trajectory = MagicMock()
    trajectory.active = True
    runner = _run(tmp_path, trajectory, stop_recorder_fails=True)
    runner._client.stop_recorder.assert_called_once()
    trajectory.record_result.assert_called_once()
    trajectory.close.assert_called_once()


def test_the_conditions_are_closed_before_the_ego_is_destroyed(tmp_path: Path) -> None:
    """A condition's sensor is attached to the ego: it is stopped and destroyed
    through its own handle while the run's world, and the ego, still exist."""
    trajectory = MagicMock()
    trajectory.active = False
    scenarios: list[Any] = []
    order: list[str] = []
    real_run_scenario = ScenarioRunner.run_scenario

    def run_scenario(self: ScenarioRunner, scenario: Any, **kwargs: Any) -> Any:
        scenario.close_conditions.side_effect = lambda: order.append("close")
        ego = scenario.create_ego.return_value
        ego.destroy.side_effect = lambda: order.append("destroy ego")
        return real_run_scenario(self, scenario, **kwargs)

    with patch.object(ScenarioRunner, "run_scenario", run_scenario):
        _run(tmp_path, trajectory, scenario_out=scenarios)
    scenarios[0].close_conditions.assert_called_once_with()
    assert order == ["close", "destroy ego"]


def test_the_result_records_the_traffic_as_it_was_before_the_backend_closed(
    tmp_path: Path,
) -> None:
    """What drove the traffic, with how many vehicles were at each level, is
    read before close(), which forgets it."""
    from autoware_carla_scenario.traffic import NullTrafficBackend

    class _Backend(NullTrafficBackend):
        closed = False

        def fidelity_census(self) -> dict[str, int]:
            return {} if self.closed else {"microscopic": 3, "mesoscopic": 9}

        def close(self) -> None:
            self.closed = True

    write_result = MagicMock()
    trajectory = MagicMock()
    trajectory.active = False
    _run(tmp_path, trajectory, traffic_backend=_Backend(), write_result=write_result)
    traffic = write_result.call_args.args[5]
    assert traffic == {
        "backend": "none",
        "fidelity": {"microscopic": 3, "mesoscopic": 9},
    }


def test_a_backend_that_cannot_count_its_traffic_does_not_end_the_run(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.traffic import NullTrafficBackend

    class _Backend(NullTrafficBackend):
        def fidelity_census(self) -> dict[str, int]:
            raise RuntimeError("gone")

    write_result = MagicMock()
    trajectory = MagicMock()
    trajectory.active = False
    _run(tmp_path, trajectory, traffic_backend=_Backend(), write_result=write_result)
    assert write_result.call_args.args[5] is None


def test_the_traffic_record_is_written_into_the_result(tmp_path: Path) -> None:
    import json

    from autoware_carla_scenario.conditions.base import ScenarioResult
    from autoware_carla_scenario.scenario_runner import _write_result

    path = tmp_path / "x_result.json"
    with patch("autoware_carla_scenario.environment.write", return_value=None):
        _write_result(
            path,
            tmp_path,
            "x",
            MagicMock(),
            ScenarioResult(passed=True, message="ok", elapsed_seconds=1.0),
            {"backend": "sumo", "fidelity": {"mesoscopic": 2}},
        )
    data = json.loads(path.read_text())
    assert data["traffic"] == {"backend": "sumo", "fidelity": {"mesoscopic": 2}}


def _ordered_npc_monitor(trajectory: MagicMock, order: list[str]) -> MagicMock:
    """A monitor with one contact still under way, which notes in *order* when
    it is finished and closed, and when *trajectory* takes its record."""
    contact = SimpleNamespace(phase="end", duration=0.05)
    monitor = _npc_monitor([contact])
    real_finish = monitor.finish.return_value

    def finish(elapsed: float, frame: int = -1) -> list[Any]:
        order.append(f"finish {elapsed}")
        return real_finish

    monitor.finish.side_effect = finish
    monitor.close.side_effect = lambda: order.append("close")
    trajectory.record_collision.side_effect = lambda c: order.append(c.phase)
    trajectory.close.side_effect = lambda: order.append("trajectory close")
    return monitor


def test_the_contacts_still_going_end_with_the_run(tmp_path: Path) -> None:
    """#1539: the end record of a contact the run ended in is logged, at the
    last tick's time, before the monitor and the trajectory log close."""
    trajectory = MagicMock()
    trajectory.active = True
    order: list[str] = []
    monitor = _ordered_npc_monitor(trajectory, order)
    frames: list[int] = []
    trajectory.record.side_effect = lambda world, t: frames.append(
        world.get_snapshot().frame
    )
    _run(tmp_path, trajectory, npc_monitor=monitor)
    # With the frame of the tick that time is of (#2019).
    monitor.finish.assert_called_once_with(0.1, frames[-1])
    assert order == ["finish 0.1", "end", "close", "trajectory close"]


def test_the_contacts_end_with_a_run_that_raised(tmp_path: Path) -> None:
    trajectory = MagicMock()
    trajectory.active = True
    order: list[str] = []
    monitor = _ordered_npc_monitor(trajectory, order)
    with pytest.raises(ValueError, match="tick broke"):
        _run(
            tmp_path,
            trajectory,
            npc_monitor=monitor,
            tick_error=ValueError("tick broke"),
        )
    assert order == ["finish 0.1", "end", "close", "trajectory close"]


def test_a_run_that_raised_before_the_monitors_tick_ends_contacts_on_one_tick(
    tmp_path: Path,
) -> None:
    """#2019: a tick cut short between the world's tick and the NPC monitor's
    still ends the contacts under way with the frame and time of one tick --
    the tick the world last advanced to, not the monitor's last."""
    from autoware_carla_scenario.npc_collisions import NpcCollisionMonitor

    trajectory = MagicMock()
    trajectory.active = True
    frames: list[int] = []
    trajectory.record.side_effect = lambda world, t: frames.append(
        world.get_snapshot().frame
    )
    monitor = NpcCollisionMonitor()
    # A contact heard, never taken up by a tick of the monitor.
    monitor._names.update({2: "npc1", 3: "npc2"})  # noqa: SLF001
    monitor._heard.append((1, 2, 3, "npc2", 5.0))  # noqa: SLF001
    with pytest.raises(ValueError, match="tick broke"):
        _run(
            tmp_path,
            trajectory,
            npc_monitor=monitor,
            tick_error=ValueError("tick broke"),
        )
    records = [c.args[0] for c in trajectory.record_collision.call_args_list]
    assert [r.phase for r in records] == ["start", "end"]
    assert frames[-1] > 0
    assert [(r.frame, r.t) for r in records] == [(frames[-1], 0.1)] * 2
    assert all(r.start_frame == 1 for r in records)


def test_a_failing_finish_does_not_hide_why_the_run_ended(tmp_path: Path) -> None:
    trajectory = MagicMock()
    trajectory.active = True
    monitor = _npc_monitor()
    monitor.finish.side_effect = RuntimeError("finish broke")
    with pytest.raises(ValueError, match="tick broke"):
        _run(
            tmp_path,
            trajectory,
            npc_monitor=monitor,
            tick_error=ValueError("tick broke"),
        )
    # The rest of the cleanup still went.
    monitor.close.assert_called_once_with()
    trajectory.close.assert_called_once_with()


# -- An ego that gives up waiting, a run that raises, the walkers' controllers


class _GivesUp(_Ego):
    """An ego whose stack never becomes ready: it gives up on its first wait
    tick."""

    is_initialized = False
    termination_reason = "Autoware never became ready"

    def on_tick(self, world: Any, elapsed: float) -> None:
        del world, elapsed
        self.termination_requested = True


def _stand_in_run(
    tmp_path: Path,
    *,
    ego: Any = None,
    pass_conditions: list[Any] | None = None,
    tick_error: BaseException | None = None,
    reset_world: bool = True,
    reset_error: BaseException | None = None,
    setup: Any = None,
    runner_out: list[ScenarioRunner] | None = None,
) -> tuple[ScenarioRunner, MagicMock, MagicMock, Any]:
    """Run a scenario of one tick through run_scenario on stand-ins, with
    ``reset_world`` replaced by a mock: the runner, its world, the scenario's
    result writer, and the result.  An error of the run propagates: the runner
    is then in *runner_out*."""
    with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
        runner = ScenarioRunner(MagicMock(), output_dir=tmp_path)
    world = MagicMock()
    frame = {"n": 0}
    world.tick.side_effect = lambda: frame.__setitem__("n", frame["n"] + 1)
    world.get_snapshot.side_effect = lambda: SimpleNamespace(frame=frame["n"])
    runner._world = world
    runner._pace_tick = lambda: None  # type: ignore[method-assign]
    runner._traffic_backend = MagicMock()
    runner.reset_world = MagicMock(side_effect=reset_error)  # type: ignore[method-assign]
    if runner_out is not None:
        runner_out.append(runner)
    scenario = _mock_scenario()
    scenario.STABILIZE_TICKS = 1
    scenario._pass_conditions = pass_conditions or []
    scenario._fail_conditions = []
    scenario._pre_tick_actions = []
    scenario._post_tick_actions = []
    scenario._pre_tick_callbacks = []
    scenario._post_tick_callbacks = []
    if tick_error is not None:
        error = tick_error

        def fail(world: Any) -> None:
            raise error

        scenario._post_tick_callbacks = [fail]
    if setup is not None:
        scenario.setup.side_effect = setup
    scenario.is_done.side_effect = [False, True]
    scenario.create_ego.return_value = ego or _Ego()
    write_result = MagicMock()
    with (
        patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
        patch("autoware_carla_scenario.scenario_runner.CoverageCollector"),
        patch(
            "autoware_carla_scenario.scenario_runner._write_result", new=write_result
        ),
        patch("autoware_carla_scenario.scenario_runner._record_run"),
        patch(
            "autoware_carla_scenario.scenario_runner._npc_collision_monitor",
            new=MagicMock(return_value=_npc_monitor()),
        ),
        patch("autoware_carla_scenario.scenario_runner._ScenarioClock") as clock,
        patch(
            "autoware_carla_scenario.scenario_runner.TrajectoryRecorder",
            new=MagicMock(return_value=MagicMock(active=False)),
        ),
    ):
        clock.return_value.simulated = 0.05
        result = runner.run_scenario(scenario, reset_world=reset_world)
    return runner, world, write_result, result


def test_an_ego_that_gives_up_waiting_fails_the_run_before_it_begins(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.conditions import AlwaysTrueCondition
    from autoware_carla_scenario.failure import FailureCategory

    ego: Any = _GivesUp()
    with patch.object(ScenarioRunner, "_release_vehicles") as release:
        runner, _world, write_result, result = _stand_in_run(
            tmp_path,
            ego=ego,
            # True at the initial pose: it would pass the run on its first tick.
            pass_conditions=[AlwaysTrueCondition(label="ego_stopped_before_line")],
            reset_world=False,
        )
    assert not result.passed
    assert result.message == (
        "Ego entity requested session termination: Autoware never became ready"
    )
    assert result.elapsed_seconds == 0.0
    assert result.failure is not None
    assert result.failure.category == FailureCategory.DRIVER
    assert result.failure.condition == ""
    # No condition was ever checked: the always-true one is not reported
    # as satisfied beside the failure.
    assert result.condition_statuses == []
    # The vehicles were never released, nor the traffic started.
    backend: Any = runner._traffic_backend
    release.assert_not_called()
    backend.start.assert_not_called()
    # The cleanup went as for any other run, and the result was written.
    ego.on_scenario_end.assert_called_once()
    ego.destroy.assert_called_once_with()
    backend.close.assert_called_once_with()
    write_result.assert_called_once()


def test_a_run_its_ego_gave_up_on_checks_no_condition(tmp_path: Path) -> None:
    from autoware_carla_scenario.conditions import AlwaysTrueCondition
    from autoware_carla_scenario.failure import FailureCategory

    class _NotReady(AlwaysTrueCondition):
        def check(self, world: Any, elapsed: float) -> Any:
            raise RuntimeError("checked before its first tick")

    _runner, _world, _write, result = _stand_in_run(
        tmp_path,
        ego=_GivesUp(),
        pass_conditions=[_NotReady(label="not_ready")],
        reset_world=False,
    )
    assert not result.passed
    assert result.failure is not None
    assert result.failure.category == FailureCategory.DRIVER


def test_a_run_that_raises_resets_the_world_before_the_error_propagates(
    tmp_path: Path,
) -> None:
    runners: list[ScenarioRunner] = []
    with pytest.raises(RuntimeError, match="time-out"):
        _stand_in_run(
            tmp_path,
            tick_error=RuntimeError("time-out of 10000ms"),
            runner_out=runners,
        )
    runners[0].reset_world.assert_called_once_with("MagicMock")  # type: ignore[attr-defined]


def test_a_reset_that_fails_after_a_run_that_raised_does_not_hide_its_error(
    tmp_path: Path,
) -> None:
    runners: list[ScenarioRunner] = []
    with pytest.raises(ValueError, match="tick broke"):
        _stand_in_run(
            tmp_path,
            tick_error=ValueError("tick broke"),
            reset_error=RuntimeError("reset broke"),
            runner_out=runners,
        )
    runners[0].reset_world.assert_called_once()  # type: ignore[attr-defined]


def test_a_run_that_raises_is_not_reset_when_the_caller_resets(tmp_path: Path) -> None:
    # A queue passes reset_world=False and resets before the next attempt.
    runners: list[ScenarioRunner] = []
    with pytest.raises(ValueError, match="tick broke"):
        _stand_in_run(
            tmp_path,
            tick_error=ValueError("tick broke"),
            reset_world=False,
            runner_out=runners,
        )
    runners[0].reset_world.assert_not_called()  # type: ignore[attr-defined]


def test_an_interrupted_run_propagates_without_a_reset(tmp_path: Path) -> None:
    runners: list[ScenarioRunner] = []
    with pytest.raises(KeyboardInterrupt):
        _stand_in_run(
            tmp_path,
            tick_error=KeyboardInterrupt(),
            runner_out=runners,
        )
    runners[0].reset_world.assert_not_called()  # type: ignore[attr-defined]


def test_a_run_that_ends_normally_is_reset_once(tmp_path: Path) -> None:
    runner, _world, _write, result = _stand_in_run(tmp_path)
    assert result.passed
    runner.reset_world.assert_called_once_with("MagicMock")  # type: ignore[attr-defined]


def _driven_walker(walker_id: int) -> MagicMock:
    """Put a walker on a stand-in controller in the walker AI's state, as a
    run before left it; the controller."""
    from autoware_carla_scenario.entity import _walker_ai

    controller = MagicMock()
    driven = _walker_ai._Driven(controller, [], None)
    driven.running = True
    _walker_ai._DRIVEN[walker_id] = driven
    _walker_ai._GIVING_WAY[walker_id] = 1.0
    return controller


def test_the_walkers_a_run_before_left_are_forgotten_at_its_start(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.entity import _walker_ai

    controller = _driven_walker(42)
    seen: list[tuple[dict[int, Any], dict[int, float]]] = []

    def setup() -> None:
        seen.append((dict(_walker_ai._DRIVEN), dict(_walker_ai._GIVING_WAY)))

    try:
        _stand_in_run(tmp_path, setup=setup)
    finally:
        _walker_ai.forget_all()
    assert seen == [({}, {})]
    # Forgotten, not destroyed: the leftovers' sweep destroys the controller.
    assert controller.mock_calls == []


def test_the_walkers_of_a_run_are_forgotten_without_a_call_per_walker(
    tmp_path: Path,
) -> None:
    """A run's cleanup calls nothing on a walker's controller: after a server
    that died mid-run, each call could wait out the client's timeout."""
    from autoware_carla_scenario.entity import _walker_ai

    controllers: list[MagicMock] = []

    def setup() -> None:
        # Walkers the run drives on their controllers.
        controllers.extend(_driven_walker(walker_id) for walker_id in (7, 8, 9))

    try:
        with pytest.raises(ValueError, match="tick broke"):
            _stand_in_run(tmp_path, setup=setup, tick_error=ValueError("tick broke"))
        assert _walker_ai._DRIVEN == {}
        assert _walker_ai._GIVING_WAY == {}
    finally:
        _walker_ai.forget_all()
    assert [c.mock_calls for c in controllers] == [[], [], []]


def test_forget_all_forgets_every_walker_without_calling_the_server() -> None:
    from autoware_carla_scenario.entity import _walker_ai

    controller = _driven_walker(1)
    controller.stop.side_effect = RuntimeError("time-out of 60000ms")
    controller.destroy.side_effect = RuntimeError("time-out of 60000ms")
    _walker_ai._WALKING._placed[4] = (0.0,) * 7 + (False,)  # type: ignore[assignment]
    _walker_ai.forget_all()
    assert controller.mock_calls == []
    assert _walker_ai._DRIVEN == {}
    assert _walker_ai._GIVING_WAY == {}
    assert _walker_ai._WALKING._placed == {}
