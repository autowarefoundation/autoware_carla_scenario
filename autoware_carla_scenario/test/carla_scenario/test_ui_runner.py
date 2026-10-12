"""ui.runner's helpers, as spelled for the Codon check (#132)."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from autoware_carla_scenario.ui import runner


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 10, 12, 9, 5, 3),
        datetime(2026, 1, 2, 0, 0, 0),
        datetime(1999, 12, 31, 23, 59, 59, 999999),
        datetime(1000, 1, 1),
        datetime(9999, 12, 31, 23, 59, 59),
        datetime.now(),  # noqa: DTZ005
    ],
)
def test_the_multirun_dir_is_what_strftime_makes(now: datetime) -> None:
    # The f-string the runner used to build it with: the same string, with
    # zero-padded fields (years before 1000 differ between Pythons' strftime).
    assert runner._multirun_dir(now) == f"multirun/{now:%Y-%m-%d}/{now:%H-%M-%S}"


@pytest.mark.parametrize(
    ("stdout", "stderr", "written"),
    [
        ("out", "err", "out\n\nerr"),
        ("out", "", "out"),
        (None, "err", "err"),
        ("", None, None),
    ],
)
def test_the_raw_output_joins_the_streams_written(
    stdout: str | None, stderr: str | None, written: str | None, tmp_path: Path
) -> None:
    result = subprocess.CompletedProcess(["true"], 0, stdout, stderr)
    job_dir = tmp_path / "multirun" / "0"
    runner._save_raw_output(result, job_dir)
    log = job_dir / runner.RAW_OUTPUT_LOG
    if written is None:
        assert not log.exists()
    else:
        assert log.read_text(encoding="utf-8") == written


def test_a_run_appends_the_extra_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner._run_worker([["scenario=a"], ["scenario=b", "x=1"]], None, ["y=2"], False)
    assert commands == [
        ["uv", "run", "scenario", "scenario=a", "y=2"],
        ["uv", "run", "scenario", "scenario=b", "x=1", "y=2"],
    ]
    progress = runner.get_progress()
    assert progress is not None
    assert (progress.current, progress.total, progress.status) == (2, 2, "done")
    assert not runner.is_running()
