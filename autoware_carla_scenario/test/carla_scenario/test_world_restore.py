"""The world handed back when a queue stops on a server it does not stop (#2067).

Every queued run leaves the world synchronous at 20 Hz with its NPCs in place
(its TrafficManager it shuts down itself), for the queue to reset before the next
attempt.  After the last one nothing ticks it: on a server that outlives the
queue (one already running, or given) the world would be frozen for every
other client.  No CARLA here: the server is test_world_soft_reset's fake.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario import ScenarioQueue, ScenarioResult, ScenarioRunner

from .test_scenario_queue import _make_ego_config, _NullScenario
from .test_world_soft_reset import _play_a_case, _runner, _Server


class _ServerWithTm(_Server):
    """The fake server, counting the TrafficManagers asked for: asking starts
    one on the port, or connects to another client's there."""

    def __init__(self) -> None:
        super().__init__()
        self.tm_ports: list[int] = []

    def get_trafficmanager(self, port: int) -> MagicMock:
        self.tm_ports.append(port)
        return MagicMock()


@pytest.fixture()
def server() -> _ServerWithTm:
    return _ServerWithTm()


def _after_a_run(runner: ScenarioRunner, server: _Server) -> list[Any]:
    """The world as a run leaves it: what it found kept by the runner, then
    made synchronous, with the run's actors in it."""
    runner._settings_before_runs = server.world.get_settings()  # type: ignore[assignment]
    return _play_a_case(server.world)


class TestRunnerRestoreWorld:
    @pytest.mark.parametrize("world_reset", ["reload", "soft"])
    def test_the_world_is_handed_back_asynchronous_and_empty(
        self, server: _ServerWithTm, world_reset: str
    ) -> None:
        runner = _runner(server, world_reset=world_reset, tm_port=8123)
        runner.load_map_by_name("Town10HD_Opt")
        map_actors = len(server.world.actors)
        left = _after_a_run(runner, server)

        runner.restore_world("Case0")

        world = server.world
        assert world.settings.synchronous_mode is False
        assert world.settings.fixed_delta_seconds is None
        assert len(world.actors) == map_actors
        assert all(actor.id not in world.actors for actor in left[:1])
        # Each run's backend shut its TrafficManager down: none is asked for,
        # which would start one on that port, or take over another client's.
        assert server.tm_ports == []
        assert runner._world is world

    def test_settings_found_synchronous_are_put_back_as_found(
        self, server: _ServerWithTm
    ) -> None:
        # The server's owner ticks it: its settings are its own.
        runner = _runner(server)
        server.world.settings.synchronous_mode = True
        server.world.settings.fixed_delta_seconds = 0.1
        _after_a_run(runner, server)
        server.world.settings.fixed_delta_seconds = 0.05

        runner.restore_world("Case0")

        assert server.world.settings.synchronous_mode is True
        assert server.world.settings.fixed_delta_seconds == 0.1

    def test_a_runner_that_never_ran_touches_nothing(
        self, server: _ServerWithTm
    ) -> None:
        runner = _runner(server)
        runner.load_map_by_name("Town10HD_Opt")

        runner.restore_world("Case0")

        assert server.loads == ["Town10HD_Opt"]
        assert server.tm_ports == []
        assert ("apply_settings", "") not in server.world.events

    def test_a_failed_reload_still_destroys_the_actors_and_restores_settings(
        self, server: _ServerWithTm, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _runner(server)
        runner.load_map_by_name("Town10HD_Opt")
        world = server.world
        _after_a_run(runner, server)
        server.load_world = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("std::exception")
        )

        with caplog.at_level(logging.WARNING):
            runner.restore_world("Case0")

        assert server.world is world
        # The run's vehicles, walkers and sensors (its props are the run's
        # own cleanup's: clear_props).
        assert not [
            actor
            for actor in world.actors.values()
            if actor.type_id.split(".")[0] in ("vehicle", "walker", "sensor")
        ]
        assert world.settings.synchronous_mode is False
        assert world.settings.fixed_delta_seconds is None
        assert server.tm_ports == []
        assert any("Reloading the map failed" in r.getMessage() for r in caplog.records)

    def test_each_failure_is_logged_and_the_next_step_runs(
        self, server: _ServerWithTm, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _runner(server)
        runner.load_map_by_name("Town10HD_Opt")
        _after_a_run(runner, server)
        runner.reset_world = MagicMock(side_effect=RuntimeError("reset broke"))  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            runner.restore_world("Case0")

        assert server.world.settings.synchronous_mode is False
        messages = [r.getMessage() for r in caplog.records]
        assert any("Resetting the world after the last run" in m for m in messages)
        assert server.tm_ports == []

    def test_a_failure_to_restore_the_settings_is_logged(
        self, server: _ServerWithTm, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _runner(server)
        _after_a_run(runner, server)
        server.world.apply_settings = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("time-out")
        )
        server.load_world = MagicMock(return_value=server.world)  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            runner.restore_world("Case0")

        assert any(
            "Restoring the world's settings failed" in r.getMessage()
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


def _queue(
    *, given: bool, reused: bool = False, events: list[str] | None = None
) -> tuple[ScenarioQueue, MagicMock, MagicMock]:
    """A started queue on a fake server and runner, with one scenario."""
    log = events if events is not None else []
    server = MagicMock()
    server.reused = reused
    server.stop.side_effect = lambda: log.append("server stop")
    runner = MagicMock(spec=ScenarioRunner)
    runner.run_scenario.return_value = ScenarioResult(
        passed=True, message="ok", elapsed_seconds=0.0
    )
    runner.restore_world.side_effect = lambda *_: log.append("restore")
    launcher = MagicMock()
    launcher.close.side_effect = lambda: log.append("launcher close")
    with (
        patch(
            "autoware_carla_scenario.scenario_queue.CarlaServerManager",
            return_value=server,
        ),
        patch(
            "autoware_carla_scenario.scenario_queue.ScenarioRunner",
            return_value=runner,
        ),
    ):
        queue = ScenarioQueue(
            server=server if given else None, autoware_launcher=launcher
        )
        queue.start()
    queue.add(_NullScenario(_make_ego_config()))
    return queue, server, runner


class TestQueueStop:
    def test_a_given_server_is_restored_before_anything_stops(self) -> None:
        events: list[str] = []
        queue, server, runner = _queue(given=True, events=events)
        queue.run_all()

        queue.stop()

        runner.restore_world.assert_called_once()
        assert events == ["restore", "launcher close"]
        server.stop.assert_not_called()
        assert not queue._world_dirty

    def test_a_reused_server_is_restored_then_stopped_as_a_no_op(self) -> None:
        events: list[str] = []
        queue, server, runner = _queue(given=False, reused=True, events=events)
        queue.run_all()

        queue.stop()

        runner.restore_world.assert_called_once()
        assert events == ["restore", "launcher close", "server stop"]

    def test_a_server_the_queue_stops_is_not_restored(self) -> None:
        events: list[str] = []
        queue, server, runner = _queue(given=False, events=events)
        queue.run_all()

        queue.stop()

        runner.restore_world.assert_not_called()
        assert events == ["launcher close", "server stop"]

    def test_a_clean_world_is_not_restored(self) -> None:
        queue, _server, runner = _queue(given=True)

        queue.stop()

        runner.restore_world.assert_not_called()

    def test_a_failed_restore_is_logged_and_the_queue_still_stops(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        events: list[str] = []
        queue, _server, runner = _queue(given=True, events=events)
        queue.run_all()
        runner.restore_world.side_effect = RuntimeError("time-out")

        with caplog.at_level(logging.WARNING):
            queue.stop()

        assert events == ["launcher close"]
        assert queue._runner is None
        assert any("failed" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("answer", [False, RuntimeError("refused")])
    def test_a_server_that_does_not_answer_is_not_restored(
        self, answer: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Each call of the restore would wait for the client's timeout, or
        # abort the process (arm64), before the launcher is stopped.
        events: list[str] = []
        queue, server, runner = _queue(given=True, events=events)
        queue.run_all()
        if isinstance(answer, BaseException):
            server.is_alive.side_effect = answer
        else:
            server.is_alive.return_value = answer
        server.host = "localhost"
        server.port = 2000

        with caplog.at_level(logging.WARNING):
            queue.stop()

        runner.restore_world.assert_not_called()
        assert events == ["launcher close"]
        assert not queue._world_dirty
        assert any("does not answer" in r.getMessage() for r in caplog.records)

    def test_a_run_all_that_raised_leaves_a_world_that_is_restored(self) -> None:
        queue, _server, runner = _queue(given=True)
        runner.run_scenario.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            queue.run_all()
        queue.stop()

        runner.restore_world.assert_called_once()

    def test_with_restores_the_world(self) -> None:
        queue, _server, runner = _queue(given=True)
        with patch.object(queue, "start"), queue:
            queue.run_all()

        runner.restore_world.assert_called_once()

    def test_the_pytest_fixture_restores_the_world(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from autoware_carla_scenario.server import CarlaServerManager

        monkeypatch.setattr(
            CarlaServerManager, "executable", classmethod(lambda cls: tmp_path)
        )
        queue, _server, runner = _queue(given=True)
        monkeypatch.setattr(queue, "start", lambda: None)
        generator = queue.as_fixture().__wrapped__()  # type: ignore[attr-defined]
        assert next(generator) is queue
        runner.restore_world.assert_not_called()

        with pytest.raises(StopIteration):
            next(generator)

        runner.restore_world.assert_called_once()


def test_a_runs_first_settings_are_what_it_found(tmp_path: Path) -> None:
    """The runner keeps the settings its first run found, before it made the
    world synchronous."""
    from .test_runner_teardown import _run

    found = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)
    runner = _run(tmp_path, MagicMock(active=False), settings_before=found)
    kept = runner._settings_before_runs
    assert kept is not None
    assert kept.synchronous_mode is False  # type: ignore[attr-defined]
    assert kept.fixed_delta_seconds is None  # type: ignore[attr-defined]
