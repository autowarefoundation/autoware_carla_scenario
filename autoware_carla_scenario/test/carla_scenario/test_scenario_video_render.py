"""``tools/scenario_video/render.py`` takes a run's result from the
``Result JSON:`` line the runner prints, and only when the result was written
(issue #1545): ``Result JSON: not written (...)`` is no result.

``chase.py`` beside it refuses a ``--port`` outside 1-65535 as a usage error
(issue #1781), and ``sumo_replay.py`` starts sumo-gui with the SUMO backend's
``start_sumo``, not ``traci.start`` (issue #2327)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import pytest

from autoware_carla_scenario.examples.run import _result_json_line

_TOOLS = Path(__file__).parents[2] / "tools" / "scenario_video"
_RENDER = _TOOLS / "render.py"
_CHASE = _TOOLS / "chase.py"
_SUMO_REPLAY = _TOOLS / "sumo_replay.py"


@pytest.fixture(scope="module")
def render() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scenario_video_render", _RENDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("directory", ["out", "out dir"])
def test_a_written_result_is_taken_whole(
    render: ModuleType, tmp_path: Path, directory: str
) -> None:
    output_dir = tmp_path / directory
    output_dir.mkdir()
    written = output_dir / "Case_result.json"
    written.write_text("{}")
    log = f"Result: SUCCESS\n{_result_json_line(output_dir, 'Case')}\nDone.\n"

    assert render._result_json_paths(log) == [str(written.resolve())]


def test_an_unwritten_result_is_none(render: ModuleType, tmp_path: Path) -> None:
    line = _result_json_line(tmp_path, "Case")
    assert line.startswith("Result JSON: not written (")

    assert render._result_json_paths(f"Result: FAILURE\n{line}\n") == []


def test_only_the_written_results_are_taken(render: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "A_result.json").write_text("{}")
    log = "\n".join(
        [
            _result_json_line(tmp_path, "A"),
            _result_json_line(tmp_path, "B"),
            "[Case] Result JSON written to: /elsewhere/C_result.json",
        ]
    )

    assert render._result_json_paths(log) == [
        str((tmp_path / "A_result.json").resolve())
    ]


@pytest.fixture(scope="module")
def chase() -> ModuleType:
    pytest.importorskip("typesafe_carla.carla")
    spec = importlib.util.spec_from_file_location("scenario_video_chase", _CHASE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", ["0", "-1", "70000", "67536", "two"])
def test_chase_refuses_a_port_outside_the_tcp_range(
    chase: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    def _no_client(*_: object) -> None:
        raise AssertionError("connected to CARLA with a bad port")

    monkeypatch.setattr(chase.carla, "Client", _no_client)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["chase.py", str(out), f"--port={value}"])

    with pytest.raises(SystemExit) as raised:
        chase.main()

    assert raised.value.code == 2
    assert "--port" in capsys.readouterr().err
    assert not out.exists()


@pytest.fixture(scope="module")
def sumo_replay() -> ModuleType:
    for name in ("sumo", "sumolib", "traci"):
        pytest.importorskip(name)
    spec = importlib.util.spec_from_file_location(
        "scenario_video_sumo_replay", _SUMO_REPLAY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """The processes Popen started; none is left running afterwards, and no
    TraCI connection is left registered."""
    import traci

    processes: list[Any] = []
    popen = subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    yield processes
    leaked = [p for p in processes if p.poll() is None]
    for process in leaked:
        process.kill()
        process.wait()
    registered = traci.connection.has("default")
    if registered:
        traci.connection.get("default").close(False)
    assert not leaked and not registered


def test_sumo_replay_fails_fast_with_the_stderr_of_a_sumo_that_exits(
    sumo_replay: ModuleType, tmp_path: Path, started: list[Any]
) -> None:
    """One sumo-gui, not one per traci.start try, reaped, and its own error
    in the message rather than traci's 'could not connect'."""
    from autoware_carla_scenario.traffic import TrafficBackendError

    script = tmp_path / "fake_sumo.py"
    script.write_text(
        "import sys\n"
        "sys.stderr.write(\"Error: Could not load net-file 'bad.net.xml'\\n\")\n"
        "sys.exit(1)\n"
    )
    began = time.monotonic()
    with pytest.raises(
        TrafficBackendError,
        match=(
            r"^SUMO \(the replay's sumo-gui\) did not start: "
            r"Error: Could not load net-file 'bad\.net\.xml'$"
        ),
    ):
        sumo_replay.start_replay([sys.executable, str(script)])
    assert time.monotonic() - began < 10.0
    (process,) = started
    assert process.returncode == 1


def test_sumo_replay_stop_reaps_a_sumo_that_does_not_exit(
    sumo_replay: ModuleType,
    tmp_path: Path,
    started: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no connection left to close, the replay's SUMO is still killed
    and reaped and its standard error closed."""
    import tempfile

    from autoware_carla_scenario.traffic.sumo import mesosim

    monkeypatch.setattr(mesosim, "_EXIT_TIMEOUT_S", 0.2)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    stderr = tempfile.TemporaryFile()
    sumo_replay.stop_replay(process, stderr)
    assert process.returncode == -9
    assert stderr.closed
