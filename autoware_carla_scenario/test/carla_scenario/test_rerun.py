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
    no_converter: None,
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
        pytest.raises(RerunUnavailableError),
    ):
        sweeper.sweep([])
    resolve.assert_not_called()


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
    assert ok.stats["unknown_fields"] == {"world.from_the_future": 1}

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
