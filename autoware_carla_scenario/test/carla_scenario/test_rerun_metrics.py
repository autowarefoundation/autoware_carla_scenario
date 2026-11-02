"""A run's metrics in its Rerun recording (:mod:`autoware_carla_scenario.rerun_metrics`)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autoware_carla_scenario import rerun_metrics
from autoware_carla_scenario.rerun_metrics import merge_metrics, metric_records


def _log(path: Path, metrics: bool = True) -> Path:
    lines = [
        {
            "type": "world",
            "schema_version": 2,
            "map": "Carla/Maps/Town01",
            "episode": 1,
        },
        {
            "type": "actor",
            "id": 1,
            "type_id": "vehicle.lincoln.mkz",
            "role": "ego",
            "extent": [2.45, 1.07, 0.77],
            "center": [0.0, 0.0, 0.7],
        },
    ]
    for i in range(10):
        lines.append(
            {
                "type": "tick",
                "frame": 100 + i,
                "t": round(0.05 * i, 2),
                "actors": [[1, 0.05 * i, 0, 0, 0, 0, 0, 1, 0, 0]],
                "control": {"1": [0.5, 0, 0, 1]},
            }
        )
    if metrics:
        for i in range(10):
            lines.append(
                {
                    "type": "metric",
                    "frame": 100 + i,
                    "t": round(0.05 * i, 2),
                    "values": {"crime_ttc_s": 3.0 - 0.1 * i, "offroad": None},
                }
            )
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_the_metric_records_are_read_in_order(tmp_path: Path) -> None:
    records = metric_records(_log(tmp_path / "s.trajectory.jsonl"))
    assert [r["frame"] for r in records] == list(range(100, 110))
    assert metric_records(_log(tmp_path / "n.trajectory.jsonl", metrics=False)) == []


def test_a_log_without_metrics_leaves_the_recording(tmp_path: Path) -> None:
    trajectory = _log(tmp_path / "s.trajectory.jsonl", metrics=False)
    recording = tmp_path / "s.rrd"
    recording.write_bytes(b"untouched")
    assert not merge_metrics(trajectory, recording, "s")
    assert recording.read_bytes() == b"untouched"


def test_without_the_sdk_the_recording_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def missing() -> None:
        raise ImportError("no rerun")

    monkeypatch.setattr(rerun_metrics, "_sdk", missing)
    recording = tmp_path / "s.rrd"
    recording.write_bytes(b"untouched")
    assert not merge_metrics(_log(tmp_path / "s.trajectory.jsonl"), recording, "s")
    assert recording.read_bytes() == b"untouched"
    assert "Rerun SDK" in caplog.text


def test_the_metrics_are_merged_into_the_converted_recording(tmp_path: Path) -> None:
    converter = pytest.importorskip("autoware_carla_scenario_rerun")
    pytest.importorskip("rerun")
    from autoware_carla_scenario.rerun_lidar import _cli

    trajectory = _log(tmp_path / "s.trajectory.jsonl")
    recording = tmp_path / "s.rrd"
    converter.convert(str(trajectory), str(recording))

    assert merge_metrics(trajectory, recording, "s")

    printed = subprocess.run(
        [str(_cli()), "rrd", "print", "-vvv", str(recording)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "/metrics/crime_ttc_s" in printed
    # A value that reads nothing is not written.
    assert "/metrics/offroad" not in printed
    # The trajectory's own entities are still there, and the scratch file gone.
    assert "/world/ego/pose" in printed
    assert not (tmp_path / "s.metrics.rrd").exists()


def test_a_runs_conversion_puts_its_metrics_in(tmp_path: Path) -> None:
    pytest.importorskip("autoware_carla_scenario_rerun")
    pytest.importorskip("rerun")
    from autoware_carla_scenario.rerun import RerunConverter, convert_outputs
    from autoware_carla_scenario.rerun_lidar import _cli

    trajectory = _log(tmp_path / "s.trajectory.jsonl")

    def entities(recording: Path) -> str:
        return subprocess.run(
            [str(_cli()), "rrd", "print", str(recording)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    # As a run ends...
    with RerunConverter() as converter:
        recording = converter.submit(trajectory, "s")
    assert "/metrics/crime_ttc_s" in entities(recording)
    # ...and converting the log again afterwards.
    recording.unlink()
    [conversion] = convert_outputs(trajectory)
    assert conversion.ok
    assert "/metrics/crime_ttc_s" in entities(recording)


def test_a_converter_that_draws_metrics_is_left_to_it(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from autoware_carla_scenario.rerun_metrics import converter_draws_metrics

    knows = SimpleNamespace(RECORD_TYPES=["world", "tick", "metric", "result"])
    older = SimpleNamespace(RECORD_TYPES=["world", "tick", "result"])
    assert converter_draws_metrics(knows) and not converter_draws_metrics(older)
    recording = tmp_path / "s.rrd"
    recording.write_bytes(b"untouched")
    assert not merge_metrics(
        _log(tmp_path / "s.trajectory.jsonl"), recording, "s", converter=knows
    )
    assert recording.read_bytes() == b"untouched"
