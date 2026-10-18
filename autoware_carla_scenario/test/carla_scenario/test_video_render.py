"""Unit tests for video.enabled and rendering a run's video later (video_render.py)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario import ScenarioRunner
from autoware_carla_scenario.debug_markers import (
    DebugMarkerLog,
    DebugMarkers,
    DebugPolyline,
)
from autoware_carla_scenario.scenario_base import SpectatorCameraConfig
from autoware_carla_scenario.scenario_runner import render_video_from_recording
from autoware_carla_scenario.video_render import (
    RENDER_JOB_SUFFIX,
    find_render_jobs,
    main,
    read_render_job,
    render_job,
    write_render_job,
)

_SPECTATOR = SpectatorCameraConfig(offset_back=8.0, offset_up=5.0, pitch=-15.0)


def _markers(x: float) -> DebugMarkers:
    markers = DebugMarkers()
    markers.add_polyline(
        DebugPolyline([(x, 0.0, 0.3), (x, 1.0, 0.3)], (0, 255, 0), 0.1, 0.1)
    )
    return markers


def _log() -> DebugMarkerLog:
    """Frames 2-4 hold one plan, 6 another, 7 a third (as an entity hands them)."""
    log = DebugMarkerLog()
    log.start(100)
    plan = _markers(1.0)
    for frame in (102, 103, 104):
        log.record(frame, plan)
    log.record(106, _markers(2.0))
    log.record(107, _markers(3.0))
    return log


def _write(
    tmp_path: Path, stem: str = "Run-2", log: DebugMarkerLog | None = None
) -> Path:
    job = tmp_path / f"{stem}{RENDER_JOB_SUFFIX}"
    write_render_job(
        job,
        log_path=tmp_path / f"{stem}.log",
        spectator_config=_SPECTATOR,
        tick_count=120,
        warmup_ticks=66,
        video_size=(960, 540),
        debug_log=log if log is not None else _log(),
    )
    (tmp_path / f"{stem}.log").write_bytes(b"recording")
    return job


# ---------------------------------------------------------------------------
# The render job
# ---------------------------------------------------------------------------


def test_a_render_job_reads_back_as_it_was_written(tmp_path: Path) -> None:
    job = read_render_job(_write(tmp_path))
    assert job.log_path == tmp_path / "Run-2.log"
    assert job.mp4_path == tmp_path / "Run-2.mp4"
    assert job.stem == "Run-2"
    assert job.spectator_config == _SPECTATOR
    assert (job.tick_count, job.warmup_ticks, job.video_size) == (120, 66, (960, 540))


def test_the_markers_read_back_frame_by_frame_one_object_per_plan(
    tmp_path: Path,
) -> None:
    original = _log()
    job = read_render_job(_write(tmp_path, log=original))
    for frame in range(0, 10):
        before, after = original.at(frame), job.debug_log.at(frame)
        assert (before is None) == (after is None), frame
        if before is not None and after is not None:
            assert after.polylines == before.polylines
        # Held as one object for as long, so the replay draws it once.
        assert job.debug_log.span(frame) == original.span(frame), frame


def test_the_recording_is_named_relative_to_the_job(tmp_path: Path) -> None:
    data = json.loads(_write(tmp_path).read_text())
    assert data["log"] == "Run-2.log"


def test_a_job_of_another_version_is_refused(tmp_path: Path) -> None:
    job = _write(tmp_path)
    data = json.loads(job.read_text())
    data["version"] = 99
    job.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="version"):
        read_render_job(job)


def test_jobs_are_found_by_directory_job_or_recording(tmp_path: Path) -> None:
    first = _write(tmp_path, "A")
    (tmp_path / "sub").mkdir()
    second = _write(tmp_path / "sub", "B")
    assert find_render_jobs([tmp_path]) == [first, second]
    assert find_render_jobs([tmp_path / "A.log"]) == [first]
    assert find_render_jobs([first, tmp_path / "A.log"]) == [first]


def test_a_path_with_no_job_is_refused(tmp_path: Path) -> None:
    (tmp_path / "C.log").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="video.enabled"):
        find_render_jobs([tmp_path / "C.log"])
    with pytest.raises(FileNotFoundError):
        find_render_jobs([tmp_path])


# ---------------------------------------------------------------------------
# Rendering it: the same replay the run would have made
# ---------------------------------------------------------------------------


class _DrawingWorld:
    """A world that records what is drawn on it and when it ticks, in order."""

    def __init__(self) -> None:
        self.events: List[Any] = []
        self.debug = MagicMock()
        self.debug.draw_line.side_effect = lambda *a, **k: self.events.append(
            ("line", a[0].x, a[0].y, round(k["life_time"], 6))
        )
        self.debug.draw_point.side_effect = lambda *a, **k: self.events.append(
            ("point", a[0].x, a[0].y, round(k["life_time"], 6))
        )
        self.frame = 0

    def tick(self) -> None:
        self.frame += 1
        self.events.append(("tick", self.frame))

    def get_settings(self) -> MagicMock:
        return MagicMock()

    def apply_settings(self, settings: Any) -> None:
        del settings


def _replay(tmp_path: Path, log: DebugMarkerLog, warmup: int, ticks: int) -> list:
    world = _DrawingWorld()
    client = MagicMock()
    client.get_world.return_value = world
    ego = MagicMock()
    ego.get_location.side_effect = lambda: SimpleNamespace(
        x=float(world.frame), y=0.0, z=0.0
    )
    recorder = MagicMock()
    recorder.write_frame.side_effect = lambda caption: world.events.append(
        ("frame", caption)
    )
    with (
        patch("autoware_carla_scenario.scenario_runner._destroy_all_dynamic_actors"),
        patch(
            "autoware_carla_scenario.scenario_runner.find_actor_by_role_name",
            return_value=ego,
        ),
        patch(
            "autoware_carla_scenario.scenario_runner.CameraRecorder",
            return_value=recorder,
        ),
    ):
        assert render_video_from_recording(
            client,
            tmp_path / "r.log",
            tmp_path / "r.mp4",
            _SPECTATOR,
            ticks,
            warmup,
            "S",
            log,
        )
    return world.events


def test_a_video_rendered_later_is_the_one_the_run_would_have_rendered(
    tmp_path: Path,
) -> None:
    original = _log()
    job = read_render_job(_write(tmp_path, log=original))
    run = _replay(tmp_path, original, warmup=3, ticks=6)
    later = _replay(tmp_path, job.debug_log, warmup=3, ticks=6)
    assert later == run
    # Every frame of the recording: the warm-up's captioned, the scenario's not.
    frames = [e for e in run if e[0] == "frame"]
    assert len(frames) == 3 + 6 - 1
    assert [bool(e[1]) for e in frames] == [True, True] + [False] * 6


def test_render_job_loads_the_recordings_map_and_renders_beside_it(
    tmp_path: Path,
) -> None:
    job = read_render_job(_write(tmp_path))
    client = MagicMock()
    client.show_recorder_file_info.return_value = (
        "Version: 1\nMap: Town10HD_Opt\nDate: 10/18/26 08:49:02\n"
    )
    with patch(
        "autoware_carla_scenario.scenario_runner.render_video_from_recording",
        return_value=True,
    ) as render:
        assert render_job(client, job, settle_seconds=0.0)
    client.load_world.assert_called_once_with("Town10HD_Opt")
    args, kwargs = render.call_args
    assert args[2] == tmp_path / "Run-2.mp4"
    assert args[4:7] == (120, 66, "Run-2")
    assert (kwargs["video_width"], kwargs["video_height"]) == (960, 540)


def test_render_job_without_its_recording_fails(tmp_path: Path) -> None:
    job = read_render_job(_write(tmp_path))
    job.log_path.unlink()
    assert not render_job(MagicMock(), job)


def test_the_command_skips_a_video_that_is_there_without_a_server(
    tmp_path: Path,
) -> None:
    _write(tmp_path)
    (tmp_path / "Run-2.mp4").write_bytes(b"video")
    with patch("autoware_carla_scenario.server.CarlaServerManager") as server:
        assert main([str(tmp_path)]) == 0
    server.assert_not_called()


def test_the_command_refuses_a_path_without_a_job(tmp_path: Path, capsys) -> None:
    assert main([str(tmp_path / "missing.log")]) == 2
    assert "render job" in capsys.readouterr().err


def test_the_command_reports_a_video_it_could_not_render(
    tmp_path: Path, capsys
) -> None:
    _write(tmp_path)
    with (
        patch("autoware_carla_scenario.server.CarlaServerManager"),
        patch("typesafe_carla.carla.Client"),
        patch("autoware_carla_scenario.video_render.render_job", return_value=False),
    ):
        assert main([str(tmp_path)]) == 1
    assert "not rendered" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The runner's wiring: video.enabled
# ---------------------------------------------------------------------------


class _Ego:
    attaches_to_existing_actor = False
    carried_actor_ids: frozenset[int] = frozenset()
    actor = None
    use_autopilot = True
    is_initialized = True
    termination_requested = False

    def __init__(self) -> None:
        self.markers = _markers(1.0)
        for name in (
            "set_traffic_backend",
            "set_initial_speeds",
            "on_scenario_start",
            "on_scenario_end",
            "destroy",
        ):
            setattr(self, name, MagicMock())
        self.spawn = MagicMock(return_value=SimpleNamespace(id=7, type_id="vehicle.x"))

    def on_tick(self, world, elapsed) -> None:
        del world, elapsed

    def debug_markers(self) -> DebugMarkers:
        return self.markers


def _run(
    tmp_path: Path, **runner_options: Any
) -> tuple[ScenarioRunner, MagicMock, Any]:
    """Run a scenario of three ticks through run_scenario, on stand-ins."""
    with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
        runner = ScenarioRunner(MagicMock(), output_dir=tmp_path, **runner_options)
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
    scenario._spectator_camera_config = _SPECTATOR
    scenario._pass_conditions = []
    scenario._fail_conditions = []
    scenario._pre_tick_actions = []
    scenario._post_tick_actions = []
    scenario._pre_tick_callbacks = []
    scenario._post_tick_callbacks = []
    scenario.is_done.side_effect = [False, False, False, True]
    scenario.create_ego.return_value = _Ego()
    render = MagicMock()
    with (
        patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
        patch("autoware_carla_scenario.scenario_runner.CoverageCollector"),
        patch("autoware_carla_scenario.scenario_runner._write_result"),
        patch("autoware_carla_scenario.scenario_runner._record_run"),
        patch("autoware_carla_scenario.scenario_runner._ScenarioClock") as clock,
        patch("autoware_carla_scenario.scenario_runner.TrajectoryRecorder"),
        patch.object(ScenarioRunner, "_render_video_from_recording", render),
    ):
        clock.return_value.simulated = 0.1
        result = runner.run_scenario(scenario, reset_world=False)
    assert result is not None and result.passed
    return runner, render, scenario


def test_the_video_is_rendered_after_the_run_by_default(tmp_path: Path) -> None:
    _runner, render, _ = _run(tmp_path)
    render.assert_called_once()
    assert render.call_args.kwargs["tick_count"] == 3
    assert not list(tmp_path.glob(f"*{RENDER_JOB_SUFFIX}"))


def test_without_the_video_the_run_leaves_its_render_job(tmp_path: Path) -> None:
    _runner, render, _ = _run(
        tmp_path, video_enabled=False, video_width=960, video_height=540
    )
    render.assert_not_called()
    [path] = tmp_path.glob(f"*{RENDER_JOB_SUFFIX}")
    job = read_render_job(path)
    assert job.log_path.name == path.name.replace(RENDER_JOB_SUFFIX, ".log")
    assert (job.tick_count, job.warmup_ticks) == (3, 3)
    assert job.video_size == (960, 540)
    # The ego's markers on every tick of the loop (the recording's frames 4-6,
    # after the three of the warm-up), for the replay.
    assert all(job.debug_log.at(frame) is not None for frame in (4, 5, 6))
