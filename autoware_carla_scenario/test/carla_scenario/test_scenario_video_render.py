"""``tools/scenario_video/render.py`` takes a run's result from the
``Result JSON:`` line the runner prints, and only when the result was written
(issue #1545): ``Result JSON: not written (...)`` is no result.

``chase.py`` beside it refuses a ``--port`` outside 1-65535 as a usage error
(issue #1781)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from autoware_carla_scenario.examples.run import _result_json_line

_TOOLS = Path(__file__).parents[2] / "tools" / "scenario_video"
_RENDER = _TOOLS / "render.py"
_CHASE = _TOOLS / "chase.py"


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
