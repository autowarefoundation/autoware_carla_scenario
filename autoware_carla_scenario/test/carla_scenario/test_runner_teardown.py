"""Unit tests for what run_scenario does at its end (no CARLA server)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from autoware_carla_scenario import ScenarioRunner
from autoware_carla_scenario.debug_markers import DebugMarkers, DebugPolyline


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


def _run(
    tmp_path: Path,
    trajectory: Any,
    *,
    stop_recorder_fails: bool = False,
    scenario_out: list[Any] | None = None,
    traffic_backend: Any = None,
    write_result: Any = None,
) -> ScenarioRunner:
    """Run a scenario of three ticks through run_scenario, on stand-ins."""
    with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
        runner = ScenarioRunner(MagicMock(), output_dir=tmp_path)
    if stop_recorder_fails:
        runner._client.stop_recorder.side_effect = RuntimeError("std::exception")
    if traffic_backend is not None:
        runner._traffic_backend = traffic_backend
    world = MagicMock()
    frame = {"n": 0}

    def tick() -> None:
        frame["n"] += 1

    world.tick.side_effect = tick
    world.get_snapshot.side_effect = lambda: SimpleNamespace(frame=frame["n"])
    runner._world = world
    runner._pace_tick = lambda: None  # type: ignore[method-assign]
    scenario = MagicMock()
    scenario.STABILIZE_TICKS = 2
    scenario._pass_conditions = []
    scenario._fail_conditions = []
    scenario._pre_tick_actions = []
    scenario._post_tick_actions = []
    scenario._pre_tick_callbacks = []
    scenario._post_tick_callbacks = []
    scenario.is_done.side_effect = [False, False, False, True]
    scenario.create_ego.return_value = _Ego()
    if scenario_out is not None:
        scenario_out.append(scenario)
    with (
        patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
        patch("autoware_carla_scenario.scenario_runner.CoverageCollector"),
        patch(
            "autoware_carla_scenario.scenario_runner._write_result",
            new=write_result or MagicMock(),
        ),
        patch("autoware_carla_scenario.scenario_runner._record_run"),
        patch("autoware_carla_scenario.scenario_runner._ScenarioClock") as clock,
        patch(
            "autoware_carla_scenario.scenario_runner.TrajectoryRecorder",
            new=MagicMock(return_value=trajectory),
        ),
    ):
        clock.return_value.simulated = 0.1
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
