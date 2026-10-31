"""Rerun recordings of the runs (``rerun.enabled``, #295): the extra being
absent, the background converter, strict mode, and the wiring through the
runner, the queue, the batch summary and the viewer."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario.conditions import ScenarioResult
from autoware_carla_scenario.rerun import (
    INSTALL_HINT,
    RerunConversion,
    RerunConverter,
    RerunUnavailableError,
    convert_outputs,
    main,
    recording_path,
    report,
    require_converter,
)

from .test_scenario_runner import _make_runner, _Stop
from .test_trajectory_recorder import _ActorSnapshot, _World


class _FakeConverter:
    """Stands in for ``autoware_carla_scenario_rerun``: records what it was
    asked, in the order it ran, and fails the logs it is told to."""

    def __init__(self, *, fail: tuple[str, ...] = (), delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.calls: list[tuple[str, str, bool]] = []
        self.threads: list[str] = []

    def convert(self, jsonl: str, out_rrd: str, *, strict: bool = False) -> dict:
        time.sleep(self.delay)
        self.calls.append((jsonl, out_rrd, strict))
        self.threads.append(threading.current_thread().name)
        if Path(jsonl).name in self.fail:
            raise ValueError(f"cannot read {Path(jsonl).name}")
        Path(out_rrd).write_bytes(b"RRD")
        return {"ok": True, "output": out_rrd, "ticks": 1}


@pytest.fixture
def no_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extra not installed: importing the converter fails."""
    monkeypatch.setitem(sys.modules, "autoware_carla_scenario_rerun", None)


def _log(directory: Path, stem: str) -> Path:
    """A real trajectory log of one tick and a verdict, as a run writes it."""
    from autoware_carla_scenario.trajectory_recorder import TrajectoryRecorder

    world = _World()
    recorder = TrajectoryRecorder(directory / f"{stem}.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    recorder.record(world, 0.0)
    recorder.record_result(ScenarioResult(True, "ok", 0.05))
    recorder.close()
    return recorder.path


# ---------------------------------------------------------------------------
# The extra being absent
# ---------------------------------------------------------------------------


def test_without_the_extra_the_converter_is_refused_saying_how_to_install_it(
    no_converter: None,
) -> None:
    with pytest.raises(RerunUnavailableError) as raised:
        require_converter()
    assert INSTALL_HINT in str(raised.value)
    assert "autoware-carla-scenario[rerun]" in str(raised.value)
    with pytest.raises(RerunUnavailableError):
        RerunConverter()


def test_a_run_asking_for_recordings_without_the_extra_is_refused_before_it_starts(
    no_converter: None,
) -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    assert run.build_rerun_converter(_shipped([])) is None
    with pytest.raises(RerunUnavailableError):
        run.build_rerun_converter(_shipped(["rerun.enabled=true"]))
    # Before anything is built: the scenario is never composed.
    with (
        patch.object(run, "build_scenario") as build,
        pytest.raises(RerunUnavailableError),
    ):
        run.run_scenario(_shipped(["rerun.enabled=true"]))
    build.assert_not_called()


def test_convert_outputs_without_the_extra_is_refused(
    no_converter: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(RerunUnavailableError):
        convert_outputs(tmp_path)
    assert main([str(tmp_path)]) == 2
    assert INSTALL_HINT in capsys.readouterr().err


def test_the_sweeper_refuses_a_sweep_asking_for_recordings_without_the_extra(
    no_converter: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        LaneletConstraintSweeper,
    )

    sweeper = LaneletConstraintSweeper()
    sweeper.config = OmegaConf.create({"rerun": {"enabled": True}})
    sweeper.launcher = MagicMock()
    with (
        patch(
            "autoware_carla_scenario.sweeper.lanelet_constraint_sweeper.resolve_map_paths"
        ) as resolve,
        pytest.raises(SystemExit) as exited,
    ):
        sweeper.sweep([])
    # Refused, as a single run is: exit status 2, saying how to install it.
    assert exited.value.code == 2
    assert INSTALL_HINT in capsys.readouterr().err
    resolve.assert_not_called()


def test_the_sweeper_refuses_a_negative_cooldown_max_retries() -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

    sweeper = mod.LaneletConstraintSweeper()
    sweeper.config = OmegaConf.create(
        {"sweep": {}, "server": {"cooldown_max_retries": -1}}
    )
    sweeper.launcher = MagicMock()
    with (
        patch.object(mod, "resolve_map_paths"),
        patch.object(mod, "load_map"),
        patch.object(mod, "expand_sweep", return_value=[("a=1",)]),
        pytest.raises(ValueError, match="cooldown_max_retries must be >= 0"),
    ):
        sweeper.sweep([])


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


def test_the_shipped_config_makes_no_recording() -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    cfg = _shipped([])
    assert cfg.rerun.enabled is False and cfg.rerun.strict is False
    assert run.build_rerun_converter(cfg) is None


def test_any_case_of_a_batch_asking_for_recordings_or_strictness_sets_it() -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    plain = _shipped([])
    enabled = _shipped(["rerun.enabled=true"])
    strict = _shipped(["rerun.strict=true"])
    with patch.object(run, "RerunConverter") as converter:
        assert run.build_rerun_converter(plain, strict) is None
        run.build_rerun_converter(plain, enabled)
        converter.assert_called_once_with(strict=False)
        converter.reset_mock()
        run.build_rerun_converter(enabled, strict)
        converter.assert_called_once_with(strict=True)


# ---------------------------------------------------------------------------
# The background converter: ordering, failures, waiting at the end
# ---------------------------------------------------------------------------


def test_the_recording_is_written_beside_the_log() -> None:
    assert recording_path(Path("/o/CutInScenario-3.trajectory.jsonl")) == Path(
        "/o/CutInScenario-3.rrd"
    )
    assert recording_path("/o/odd.jsonl") == Path("/o/odd.rrd")


def test_submit_returns_at_once_and_wait_waits_for_every_log(tmp_path: Path) -> None:
    fake = _FakeConverter(delay=0.05)
    logs = [tmp_path / f"S-{k}.trajectory.jsonl" for k in range(3)]
    with RerunConverter(converter=fake) as converter:
        started = time.monotonic()
        recordings = [converter.submit(log, log.name) for log in logs]
        # Three conversions of 50 ms each have not run by now.
        assert time.monotonic() - started < 0.1
        conversions = converter.wait()
    assert recordings == [recording_path(log) for log in logs]
    assert all(recording.is_file() for recording in recordings)
    assert [c.recording for c in conversions] == recordings
    assert all(c.ok for c in conversions)
    assert conversions[0].stats == {
        "ok": True,
        "output": str(recordings[0]),
        "ticks": 1,
    }


def test_conversions_run_off_the_calling_thread_in_the_order_the_runs_ended(
    tmp_path: Path,
) -> None:
    fake = _FakeConverter()
    logs = [tmp_path / f"S-{k}.trajectory.jsonl" for k in range(5)]
    with RerunConverter(converter=fake) as converter:
        for log in logs:
            converter.submit(log, log.name)
        conversions = converter.wait()
    assert [call[0] for call in fake.calls] == [str(log) for log in logs]
    assert [c.trajectory for c in conversions] == logs
    assert threading.current_thread().name not in fake.threads
    assert all(name.startswith("rerun-convert") for name in fake.threads)


def test_a_failure_is_reported_with_its_run_and_does_not_stop_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeConverter(fail=("B.trajectory.jsonl",))
    with (
        caplog.at_level(logging.ERROR, logger="autoware_carla_scenario.rerun"),
        RerunConverter(converter=fake) as converter,
    ):
        for stem in ("A", "B", "C"):
            converter.submit(tmp_path / f"{stem}.trajectory.jsonl", stem)
        conversions = converter.wait()
    assert [c.ok for c in conversions] == [True, False, True]
    assert conversions[1] == RerunConversion(
        "B",
        tmp_path / "B.trajectory.jsonl",
        tmp_path / "B.rrd",
        error="ValueError: cannot read B.trajectory.jsonl",
    )
    [record] = caplog.records
    assert "[B]" in record.getMessage()
    assert "cannot read B.trajectory.jsonl" in record.getMessage()


def test_wait_returns_every_conversion_so_far_and_close_ends_the_thread(
    tmp_path: Path,
) -> None:
    converter = RerunConverter(converter=_FakeConverter())
    converter.submit(tmp_path / "A.trajectory.jsonl", "A")
    assert [c.label for c in converter.wait()] == ["A"]
    converter.submit(tmp_path / "B.trajectory.jsonl", "B")
    assert [c.label for c in converter.close()] == ["A", "B"]
    with pytest.raises(RuntimeError, match="closed"):
        converter.submit(tmp_path / "C.trajectory.jsonl", "C")
    # Closing again is harmless.
    assert [c.label for c in converter.close()] == ["A", "B"]


def test_the_report_counts_the_recordings_and_names_each_failure(
    tmp_path: Path,
) -> None:
    conversions = [
        RerunConversion("A", tmp_path / "A.trajectory.jsonl", tmp_path / "A.rrd"),
        RerunConversion(
            "B", tmp_path / "B.trajectory.jsonl", tmp_path / "B.rrd", error="E: x"
        ),
    ]
    text = report(conversions)
    assert text.splitlines()[0].startswith("Rerun recordings: 1/2 written")
    assert "rerun <file.rrd>" in text
    assert "[FAILED] B: E: x" in text


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_strict_is_passed_on_to_every_conversion(tmp_path: Path) -> None:
    fake = _FakeConverter()
    with RerunConverter(strict=True, converter=fake) as converter:
        assert converter.strict
        converter.submit(tmp_path / "A.trajectory.jsonl", "A")
    assert fake.calls[0][2] is True


def test_a_strict_conversion_fails_on_a_field_it_does_not_know(
    tmp_path: Path,
) -> None:
    pytest.importorskip("autoware_carla_scenario_rerun")
    log = _log(tmp_path, "S")
    lines = log.read_text().splitlines()
    world = json.loads(lines[0])
    world["from_the_future"] = 1
    log.write_text("\n".join([json.dumps(world), *lines[1:]]) + "\n")

    with RerunConverter() as lenient:
        lenient.submit(log, "S")
    [ok] = lenient.close()
    assert ok.ok and ok.stats is not None
    # Besides it, only a field added within version 2 that a converter older
    # than the log may not know yet (a tick's ``angular_velocity``).
    unknown = dict(ok.stats["unknown_fields"])
    assert unknown.pop("world.from_the_future") == 1
    assert set(unknown) <= {"tick.angular_velocity"}

    with RerunConverter(strict=True) as strict:
        strict.submit(log, "S")
    [failed] = strict.close()
    assert not failed.ok
    assert failed.error is not None
    assert failed.error.startswith("UnknownFieldError:")


# ---------------------------------------------------------------------------
# With the real converter
# ---------------------------------------------------------------------------


def test_a_real_log_converts_into_a_recording(tmp_path: Path) -> None:
    pytest.importorskip("autoware_carla_scenario_rerun")
    log = _log(tmp_path, "CutInScenario")
    with RerunConverter() as converter:
        recording = converter.submit(log, "CutInScenario")
    [conversion] = converter.close()
    assert conversion.ok, conversion.error
    assert recording == tmp_path / "CutInScenario.rrd"
    assert recording.stat().st_size > 0
    assert conversion.stats is not None and conversion.stats["has_result"]


def test_convert_outputs_converts_a_finished_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("autoware_carla_scenario_rerun")
    for stem in ("A", "A-1"):
        _log(tmp_path / "2026-10-18" / "15-58-45", stem)
    broken = tmp_path / "2026-10-18" / "15-58-45" / "B.trajectory.jsonl"
    broken.write_text("not json\n")

    conversions = convert_outputs(tmp_path)
    by_label = {c.label: c for c in conversions}
    assert sorted(by_label) == ["A", "A-1", "B"]
    assert by_label["A"].ok and by_label["A"].recording.is_file()
    assert not by_label["B"].ok
    assert by_label["B"].recording == broken.with_name("B.rrd")

    # A single log, and the command line.
    single = _log(tmp_path / "single", "C")
    [one] = convert_outputs(single)
    assert one.ok and (tmp_path / "single" / "C.rrd").is_file()
    assert main([str(tmp_path / "single")]) == 0
    assert main([str(tmp_path)]) == 1
    assert "[FAILED] B:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Converting again keeps the run's sensors (#913)
# ---------------------------------------------------------------------------


def _real_sdk() -> None:
    pytest.importorskip("autoware_carla_scenario_rerun")
    pytest.importorskip("rerun")
    pytest.importorskip("rerun_cli")


def _record_sensors(log: Path, label: str) -> Path:
    """Write ``{label}.sensors.rrd`` beside *log* as a run with
    ``rerun.lidar`` and ``rerun.cameras`` does: a sweep and a camera frame."""
    import numpy as np

    from autoware_carla_scenario.rerun_lidar import CameraMount, SensorRecorder

    recorder = SensorRecorder(
        log.with_name(f"{label}.sensors.rrd"), label, 0, cameras=True
    )
    recorder.log_sweep("top", 1, np.ones((8, 3), np.float32), np.eye(3), np.zeros(3))
    recorder.log_camera_frame("front", 1, b"\xff\xd8\xff\xd9", CameraMount(4, 4, 90.0))
    return recorder.close()


def _entities(recording: Path) -> list[str]:
    """The entity of each chunk of *recording*, as ``rerun rrd print`` lists them."""
    import importlib.util
    import re
    import subprocess

    spec = importlib.util.find_spec("rerun_cli")
    assert spec is not None and spec.origin is not None
    printed = subprocess.run(
        [str(Path(spec.origin).parent / "rerun"), "rrd", "print", str(recording)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(re.findall(r"^Chunk\(.*? - (/\S*) - data columns", printed, re.M))


_SENSOR_CHUNKS = [
    "/world/ego/pose/cameras/front",
    "/world/ego/pose/cameras/front",
    "/world/lidar/top",
]


def _sensor_chunks(recording: Path) -> list[str]:
    return [
        e
        for e in _entities(recording)
        if e.startswith(("/world/lidar/", "/world/ego/pose/cameras/"))
    ]


def _in_run(directory: Path, stem: str) -> Path:
    """A run made with ``rerun.lidar`` and ``rerun.cameras``: its log, and its
    recording with the sensors merged in."""
    log = _log(directory, stem)
    _record_sensors(log, stem)
    with RerunConverter(lidar_max_points=0, cameras=True) as converter:
        converter.submit(log, stem)
    [conversion] = converter.close()
    assert conversion.ok, conversion.error
    assert not log.with_name(f"{stem}.sensors.rrd").exists()
    assert _sensor_chunks(recording_path(log)) == _SENSOR_CHUNKS
    return log


def test_converting_a_run_again_keeps_its_sensors(tmp_path: Path) -> None:
    _real_sdk()
    log = _in_run(tmp_path / "session", "S")
    recording = recording_path(log)
    trajectory_only = tmp_path / "plain.rrd"
    require_converter().convert(str(log), str(trajectory_only))
    expected = sorted(_entities(trajectory_only) + _SENSOR_CHUNKS)

    [conversion] = convert_outputs(tmp_path / "session")
    assert conversion.ok, conversion.error
    assert _sensor_chunks(recording) == _SENSOR_CHUNKS
    # Carried over once, beside a recording made anew.
    assert _entities(recording) == expected
    # Again, a single log, and from the command line: nothing piles up.
    [conversion] = convert_outputs(log)
    assert conversion.ok, conversion.error
    assert main([str(tmp_path / "session")]) == 0
    assert _entities(recording) == expected
    assert sorted(p.name for p in log.parent.iterdir()) == [
        "S.rrd",
        "S.trajectory.jsonl",
    ]


def test_a_sensor_recording_left_beside_a_log_is_merged_and_removed(
    tmp_path: Path,
) -> None:
    """A run whose merge failed keeps ``{stem}.sensors.rrd``; converting it
    again merges it, in a directory and a single log alike."""
    _real_sdk()
    for directory, convert in (
        (tmp_path / "dir", lambda log: convert_outputs(log.parent)),
        (tmp_path / "one", lambda log: convert_outputs(log)),
    ):
        log = _log(directory, "S")
        sensors = _record_sensors(log, "S")
        # The recording the run made before its merge failed.
        require_converter().convert(str(log), str(recording_path(log)))
        [conversion] = convert(log)
        assert conversion.ok, conversion.error
        assert not sensors.exists()
        assert _sensor_chunks(recording_path(log)) == _SENSOR_CHUNKS


def test_converting_a_run_without_sensors_again_is_unchanged(tmp_path: Path) -> None:
    _real_sdk()
    log = _log(tmp_path / "session", "S")
    plain = tmp_path / "plain.rrd"
    require_converter().convert(str(log), str(plain))
    for _ in range(2):
        [conversion] = convert_outputs(tmp_path / "session")
        assert conversion.ok, conversion.error
        assert _entities(recording_path(log)) == _entities(plain)
    assert sorted(p.name for p in log.parent.iterdir()) == [
        "S.rrd",
        "S.trajectory.jsonl",
    ]


def test_a_recording_the_sdk_cannot_read_is_kept_only_if_it_seems_to_hold_sensors(
    tmp_path: Path,
) -> None:
    """With the CLI, a recording it cannot read falls back to the byte scan:
    a broken recording of the trajectory alone is replaced (what converting
    again is for); one naming a sensor entity is kept."""
    _real_sdk()
    for stem, old in (("A", b"broken /world/lidar/top"), ("B", b"broken")):
        log = _log(tmp_path, stem)
        recording_path(log).write_bytes(old)
    a, b = convert_outputs(tmp_path)
    assert a.skipped and (tmp_path / "A.rrd").read_bytes() == b"broken /world/lidar/top"
    assert b.ok and _entities(tmp_path / "B.rrd")
    assert not list(tmp_path.glob(".*"))


def test_sensors_are_kept_when_the_cli_output_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rerun rrd print`` in a form :func:`extract_sensors` does not read
    (another SDK) is a failure to keep the sensors, not a recording without
    them: the recording is left as it was."""
    import re

    from autoware_carla_scenario import rerun_lidar

    _real_sdk()
    log = _in_run(tmp_path, "S")
    before = recording_path(log).read_bytes()
    monkeypatch.setattr(rerun_lidar, "_CHUNK_LINE", re.compile(r"^Chunk<(.*)>$"))
    with pytest.raises(RuntimeError, match="cannot read the entities"):
        rerun_lidar.extract_sensors(recording_path(log), tmp_path / "x.sensors.rrd")
    [conversion] = convert_outputs(log)
    assert conversion.skipped
    assert recording_path(log).read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "S.rrd",
        "S.trajectory.jsonl",
    ]


class _FakeModule(_FakeConverter):
    """The converter package, ``convert_dir`` included."""

    def convert_dir(
        self, output_dir: str, *, jobs: Any = None, strict: bool = False, out_dir: str
    ) -> list[dict]:
        entries = []
        for log in sorted(Path(output_dir).rglob("*.trajectory.jsonl")):
            out = Path(out_dir) / log.relative_to(output_dir)
            out = out.with_name(log.name[: -len(".trajectory.jsonl")] + ".rrd")
            out.parent.mkdir(parents=True, exist_ok=True)
            entries.append({**self.convert(str(log), str(out)), "input": str(log)})
        return entries


@pytest.fixture
def no_sdk_cli(monkeypatch: pytest.MonkeyPatch) -> _FakeModule:
    """A converter, but not the Rerun SDK's CLI: sensors cannot be carried over."""
    import importlib.util

    real = importlib.util.find_spec

    def find_spec(name: str, *args: Any) -> Any:
        return None if name == "rerun_cli" else real(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    fake = _FakeModule()
    monkeypatch.setitem(sys.modules, "autoware_carla_scenario_rerun", fake)
    return fake


def test_a_recording_whose_sensors_cannot_be_kept_is_left_as_it_was(
    tmp_path: Path, no_sdk_cli: _FakeModule, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.rerun_lidar import carries_sensors

    for stem in ("A", "B"):
        (tmp_path / f"{stem}.trajectory.jsonl").write_text("{}\n")
    old = b"RRD ... /world/lidar/top ..."
    (tmp_path / "A.rrd").write_bytes(old)
    (tmp_path / "B.rrd").write_bytes(b"RRD ... /world/actors ...")
    assert carries_sensors(tmp_path / "A.rrd")
    assert not carries_sensors(tmp_path / "B.rrd")

    a, b = convert_outputs(tmp_path)
    assert a.skipped and not a.ok and a.error is not None
    assert "A.rrd holds sensor data" in a.error and "--overwrite" in a.error
    assert (tmp_path / "A.rrd").read_bytes() == old
    # A recording without sensors is replaced, as before.
    assert b.ok and (tmp_path / "B.rrd").read_bytes() == b"RRD"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "A.rrd",
        "A.trajectory.jsonl",
        "B.rrd",
        "B.trajectory.jsonl",
    ]

    # The command line reports it, and fails.
    assert main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "Rerun recordings: 1/2 written" in out and "[SKIPPED] A:" in out
    [one] = convert_outputs(tmp_path / "A.trajectory.jsonl")
    assert one.skipped and (tmp_path / "A.rrd").read_bytes() == old

    # --overwrite replaces it.
    assert main([str(tmp_path / "A.trajectory.jsonl"), "--overwrite"]) == 0
    assert (tmp_path / "A.rrd").read_bytes() == b"RRD"


def test_a_failed_conversion_keeps_the_recording_it_would_replace(
    tmp_path: Path, no_sdk_cli: _FakeModule
) -> None:
    no_sdk_cli.fail = ("A.trajectory.jsonl",)
    log = tmp_path / "A.trajectory.jsonl"
    log.write_text("{}\n")
    (tmp_path / "A.rrd").write_bytes(b"OLD")
    [failed] = convert_outputs(log)
    assert not failed.ok and not failed.skipped
    assert failed.error == "ValueError: cannot read A.trajectory.jsonl"
    assert (tmp_path / "A.rrd").read_bytes() == b"OLD"


# ---------------------------------------------------------------------------
# The runner and the queue
# ---------------------------------------------------------------------------


class _FakeRecorder:
    """A trajectory log that starts, then ends the run (``_Stop``)."""

    def __init__(self, path: Path, *_: Any) -> None:
        self.path = path
        self.active = False
        self.closed = False

    def start(self, world: Any) -> None:
        self.active = True
        raise _Stop

    def close(self) -> None:
        self.closed = True


def test_the_runner_hands_each_closed_log_to_the_converter(tmp_path: Path) -> None:
    from .test_scenario_runner import TestRunScenarioRegistersTheDefaultTimeout

    runner = _make_runner(None)
    runner.output_dir = tmp_path
    runner._world = MagicMock()
    rerun = MagicMock()
    rerun.submit.side_effect = lambda path, label: recording_path(path)
    runner.rerun = rerun
    scenario = TestRunScenarioRegistersTheDefaultTimeout._scenario()
    type(scenario).__name__ = "CutInScenario"
    with (
        patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
        patch("autoware_carla_scenario.scenario_runner._ScenarioClock"),
        patch(
            "autoware_carla_scenario.scenario_runner.TrajectoryRecorder",
            _FakeRecorder,
        ),
        pytest.raises(_Stop),
    ):
        runner.run_scenario(scenario, reset_world=False)
    # Even a run that raised: its log is what shows what happened.
    [call] = rerun.submit.call_args_list
    path, label = call.args
    assert path.parent == tmp_path and path.name.endswith(".trajectory.jsonl")
    assert label == path.name[: -len(".trajectory.jsonl")]


def test_a_runner_without_a_converter_makes_no_recording() -> None:
    assert _make_runner(None).rerun is None


def test_the_queue_hands_its_converter_to_the_runner() -> None:
    from autoware_carla_scenario import ScenarioQueue

    rerun = MagicMock()
    queue = ScenarioQueue(server=MagicMock(), rerun=rerun)
    with (
        patch("autoware_carla_scenario.scenario_queue.ScenarioRunner") as runner_cls,
        patch("autoware_carla_scenario.scenario_queue.MapManager"),
    ):
        queue.start()
    assert runner_cls.return_value.rerun is rerun


# ---------------------------------------------------------------------------
# The result, the batch summary and the viewer
# ---------------------------------------------------------------------------


def test_the_result_names_its_recording_only_when_there_is_one() -> None:
    plain = ScenarioResult(True, "ok", 1.0)
    assert "rerun_recording" not in plain.to_dict()
    recorded = ScenarioResult(True, "ok", 1.0, rerun_recording="/o/S.rrd")
    assert recorded.to_dict()["rerun_recording"] == "/o/S.rrd"
    assert json.loads(recorded.to_json())["rerun_recording"] == "/o/S.rrd"


def test_the_batch_summary_lists_each_recording_and_every_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.examples.run import _print_summary, rerun_errors

    results = [
        ScenarioResult(True, "ok", 1.0, rerun_recording=str(tmp_path / "A.rrd")),
        ScenarioResult(False, "bad", 2.0, rerun_recording=str(tmp_path / "B.rrd")),
        ScenarioResult(False, "Errored", 0.0),
    ]
    conversions = [
        RerunConversion("A", tmp_path / "A.trajectory.jsonl", tmp_path / "A.rrd"),
        RerunConversion(
            "B",
            tmp_path / "B.trajectory.jsonl",
            tmp_path / "B.rrd",
            error="MalformedLineError: line 3",
        ),
    ]
    passed = _print_summary(
        ["A", "B", "C"],
        results,
        output_dir=tmp_path,
        errored=[2],
        rerun_errors=rerun_errors(results, conversions),
    )
    # The verdicts alone decide.
    assert passed is False
    out = capsys.readouterr().out
    assert f"Rerun: {tmp_path / 'A.rrd'}" in out
    assert "Rerun recordings: 1/2 written" in out
    assert "[FAILED] B: MalformedLineError: line 3" in out

    entries = json.loads((tmp_path / "batch_results.json").read_text())
    assert entries[0]["rerun_recording"] == str(tmp_path / "A.rrd")
    assert "rerun_error" not in entries[0]
    assert entries[1]["rerun_error"] == "MalformedLineError: line 3"
    assert "rerun_recording" not in entries[2]


def test_a_batch_without_recordings_says_nothing_of_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.examples.run import _print_summary

    _print_summary(["A"], [ScenarioResult(True, "ok", 1.0)], output_dir=tmp_path)
    assert "Rerun" not in capsys.readouterr().out


def test_a_single_run_waits_for_its_recording_and_reports_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    fake = _FakeConverter()
    seen: dict[str, Any] = {}

    def run_one(cfg: Any, rerun: Any, **_: Any) -> ScenarioResult:
        seen["rerun"] = rerun
        return ScenarioResult(True, "ok", 1.0)

    with (
        patch.object(
            run,
            "RerunConverter",
            side_effect=lambda strict: RerunConverter(strict=strict, converter=fake),
        ),
        patch.object(run, "_run_scenario", side_effect=run_one),
    ):
        result = run.run_scenario(_shipped(["rerun.enabled=true"]))
    assert result.passed
    assert isinstance(seen["rerun"], RerunConverter)
    with pytest.raises(RuntimeError, match="closed"):
        seen["rerun"].submit(Path("x.trajectory.jsonl"), "x")
    assert "Rerun recordings: 0/0 written" in capsys.readouterr().out


def test_the_viewer_shows_the_recording_rerun_enabled_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The viewer looks for a case's recording where ``rerun.enabled`` writes
    it, and shows that one rather than making another."""
    from autoware_carla_scenario.ui import recordings

    for name in ("CutInScenario.trajectory.jsonl", "CutInScenario-3.trajectory.jsonl"):
        log = tmp_path / name
        assert recordings.recording_path(log) == recording_path(log)
    log = tmp_path / "CutInScenario.trajectory.jsonl"
    log.write_text("{}\n")
    recording_path(log).write_bytes(b"RRD")

    def convert(trajectory: Path, out: Path) -> None:
        raise AssertionError(f"made {out} again")

    monkeypatch.setattr(recordings, "_convert", convert)
    assert recordings.ensure_recording(log) == recording_path(log)


def test_a_failed_conversion_is_found_by_the_case_it_belongs_to(tmp_path: Path) -> None:
    from autoware_carla_scenario.examples.run import rerun_errors

    results = [
        ScenarioResult(True, "ok", 1.0, rerun_recording=str(tmp_path / "A.rrd")),
        ScenarioResult(True, "ok", 1.0),
        ScenarioResult(True, "ok", 1.0, rerun_recording=str(tmp_path / "C.rrd")),
    ]
    conversions = [
        RerunConversion("A", tmp_path / "A.trajectory.jsonl", tmp_path / "A.rrd"),
        # A retried attempt's log: no case's result names it.
        RerunConversion(
            "X", tmp_path / "X.trajectory.jsonl", tmp_path / "X.rrd", error="E: x"
        ),
        RerunConversion(
            "C", tmp_path / "C.trajectory.jsonl", tmp_path / "C.rrd", error="E: c"
        ),
    ]
    assert rerun_errors(results, conversions) == {2: "E: c"}


def test_a_slurm_batch_reports_the_recordings_its_tasks_made(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``scenario-slurm collect`` reads each case's recording and its failure
    from the record its task wrote, and reports them as a local batch does."""
    from autoware_carla_scenario.batch_report import print_summary
    from autoware_carla_scenario.slurm.collect import RecordedResult

    results = [
        RecordedResult(
            {
                "passed": True,
                "message": "ok",
                "elapsed_seconds": 1.0,
                "condition_statuses": [],
                "rerun_recording": str(tmp_path / "A.rrd"),
            }
        ),
        RecordedResult(
            {
                "passed": True,
                "message": "ok",
                "elapsed_seconds": 1.0,
                "condition_statuses": [],
            }
        ),
    ]
    assert results[0].rerun_recording == str(tmp_path / "A.rrd")
    assert results[1].rerun_recording is None
    print_summary(["A", "B"], results, output_dir=tmp_path, rerun_errors={0: "E: a"})
    out = capsys.readouterr().out
    assert "Rerun recordings: 0/1 written" in out
    assert "[FAILED] A: E: a" in out
    entries = json.loads((tmp_path / "batch_results.json").read_text())
    assert entries[0]["rerun_error"] == "E: a"
    assert "rerun_error" not in entries[1]


def test_a_rerun_of_a_slurm_case_removes_its_old_recording() -> None:
    from autoware_carla_scenario.slurm.manifest import RUN_FILE_SUFFIXES

    assert ".rrd" in RUN_FILE_SUFFIXES
