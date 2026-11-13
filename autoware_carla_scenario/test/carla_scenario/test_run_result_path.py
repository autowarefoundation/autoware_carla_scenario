"""The ``Result JSON:`` line a single run prints names the file the runner
wrote, ``X-1_result.json`` once an earlier run in the directory took ``X``
(issue #1217)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.conditions import ScenarioResult
from autoware_carla_scenario.examples import run
from autoware_carla_scenario.scenario_runner import _run_file_stem


class _Scenario:
    """Stands in for a scenario: only its class name names its files."""


class _FakeQueue:
    """A queue whose runner writes the result as the real one does: under the
    stem it is given, else the first free one (``_run_file_stem``)."""

    added: list[Optional[str]] = []

    def __init__(self, *, output_dir: Path, **_: Any) -> None:
        self._output_dir = output_dir
        self._stem: Optional[str] = None
        self._scenario: Any = None

    def add(self, scenario: Any, *, file_stem: Optional[str] = None, **_: Any) -> None:
        self._scenario = scenario
        self._stem = file_stem
        _FakeQueue.added.append(file_stem)

    def __enter__(self) -> _FakeQueue:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def run_all(self) -> list[ScenarioResult]:
        stem = self._stem
        if stem is None:
            stem = _run_file_stem(self._output_dir, type(self._scenario).__name__)
        (self._output_dir / f"{stem}_result.json").write_text("{}")
        return [ScenarioResult(True, "ok", 1.0)]


def _run(output_dir: Path) -> None:
    cfg = OmegaConf.create(
        {
            "server": {"host": "localhost", "port": 2000},
            "traffic_manager": {"port": 8000},
            "map": {},
        }
    )
    hydra_config = MagicMock()
    hydra_config.get.return_value.runtime.output_dir = str(output_dir)
    stubs: dict[str, Any] = {
        name: MagicMock()
        for name in (
            "check_odd",
            "_odd_spec",
            "resolve_map_paths",
            "run_timeout_seconds",
            "build_traffic_backend",
            "build_autoware_launcher",
            "_odd_case",
            "server_extra_args",
            "_map_source",
            "build_t4_recorder",
            "build_t4scene_exporter",
            "build_backend_pool",
        )
    }
    with (
        patch.multiple(run, **stubs),
        patch.object(run, "build_scenario", return_value=(None, _Scenario())),
        patch.object(run, "_queue_retries", return_value=0),
        patch.object(run, "world_reset_settings", return_value=("reload", 0)),
        patch.object(run, "HydraConfig", hydra_config),
        patch.object(run, "ScenarioQueue", _FakeQueue),
    ):
        run._run_scenario(cfg, None)


def _printed_path(out: str) -> Path:
    (line,) = [line for line in out.splitlines() if line.startswith("Result JSON:")]
    return Path(line.removeprefix("Result JSON: "))


def test_a_first_run_prints_its_class_named_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(tmp_path)
    printed = _printed_path(capsys.readouterr().out)
    assert printed == (tmp_path / "_Scenario_result.json").resolve()
    assert printed.is_file()


def test_a_run_after_an_earlier_one_prints_the_numbered_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An earlier run (a sweeper's retry, a reused run directory) took the stem.
    (tmp_path / "_Scenario_result.json").write_text("{}")
    (tmp_path / "_Scenario-1.trajectory.jsonl").write_text("")
    _run(tmp_path)
    printed = _printed_path(capsys.readouterr().out)
    assert printed == (tmp_path / "_Scenario-2_result.json").resolve()
    assert printed.is_file()


def test_the_stem_is_handed_to_the_queue(tmp_path: Path) -> None:
    (tmp_path / "_Scenario.log").write_text("")
    _FakeQueue.added = []
    _run(tmp_path)
    assert _FakeQueue.added == ["_Scenario-1"]


def test_a_result_not_written_is_said_so(tmp_path: Path) -> None:
    line = run._result_json_line(tmp_path, "X-1")
    assert line.startswith("Result JSON: not written")
    assert "X-1_result.json" in line


def test_the_result_line_names_the_written_result_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "X-1_result.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    resolved = (tmp_path / "out" / "X-1_result.json").resolve()
    assert run._result_json_line(Path("out"), "X-1") == f"Result JSON: {resolved}"
    assert run._result_json_line(Path("out"), "X-2") == (
        f"Result JSON: not written ({resolved.parent / 'X-2_result.json'})"
    )


def test_a_directory_named_as_the_result_is_not_a_written_result(
    tmp_path: Path,
) -> None:
    (tmp_path / "X_result.json").mkdir()
    assert run._result_json_line(tmp_path, "X") == (
        f"Result JSON: not written ({(tmp_path / 'X_result.json').resolve()})"
    )


@pytest.mark.parametrize(
    ("argv", "requested"),
    [
        (["run.py", "-m", "hydra/sweeper=lanelet_constraint"], True),
        (["run.py", "-m", "+hydra/sweeper=lanelet_constraint"], True),
        (["run.py", "++hydra/sweeper=lanelet_constraint"], True),
        (["run.py", "hydra/sweeper='lanelet_constraint'"], True),
        (["run.py", 'hydra/sweeper="lanelet_constraint"'], True),
        (["run.py", "hydra/sweeper= lanelet_constraint "], True),
        (["run.py", "hydra/sweeper=basic"], False),
        (["run.py", "hydra/sweeper"], False),
        (["run.py", "hydra/sweeper=lanelet_constraint2"], False),
        (["run.py", "~hydra/sweeper=lanelet_constraint"], False),
        (["run.py", "x/hydra/sweeper=lanelet_constraint"], False),
        (["run.py", "lanelet_constraint"], False),
        (["hydra/sweeper=lanelet_constraint"], True),
        ([], False),
    ],
)
def test_the_lanelet_sweeper_is_seen_on_the_command_line(
    argv: list[str], requested: bool
) -> None:
    assert run._lanelet_sweeper_requested_in_argv(argv) is requested


def test_the_helpers_are_the_checked_modules_under_their_old_names() -> None:
    # Moved out of examples.run, which the library check cannot compile, into
    # run_outputs and run_dirs, which it does (#2116); the old names are the
    # same objects, so a caller or a patch of either reaches the one function.
    from autoware_carla_scenario import run_dirs, run_outputs

    assert run._result_json_line is run_outputs.result_json_line
    assert (
        run._lanelet_sweeper_requested_in_argv
        is run_dirs.lanelet_sweeper_requested_in_argv
    )
    assert run.RESULT_SUFFIX == run_outputs.RESULT_SUFFIX
