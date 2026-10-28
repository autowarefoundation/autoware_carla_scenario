"""The command lines of ``viewer`` and ``scenario-editor`` (#712).

Both used to ignore their arguments: ``viewer --help`` or
``scenario-editor --port 8080`` started the server on ``0.0.0.0`` instead.
``uvicorn.run`` is patched, so nothing here serves.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from autoware_carla_scenario import editor, ui

_ENV = (
    "VIEWER_HOST",
    "VIEWER_PORT",
    "VIEWER_BASE_PATH",
    "SCENARIO_EDITOR_HOST",
    "SCENARIO_EDITOR_PORT",
    "SCENARIO_EDITOR_DRAFTS",
    "SCENARIO_EDITOR_EXPORT_DIR",
)

_MAINS: dict[str, Callable[[list[str] | None], None]] = {
    "viewer": ui.main,
    "scenario-editor": editor.main,
}
_PREFIX = {"viewer": "VIEWER", "scenario-editor": "SCENARIO_EDITOR"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The calls to ``uvicorn.run``, which serves nothing here."""
    import uvicorn

    calls: list[dict[str, Any]] = []

    def run(app: object, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(uvicorn, "run", run)
    return calls


@pytest.fixture
def base_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the viewer's BASE_PATH, which ``--base-path`` sets."""
    from autoware_carla_scenario.ui import app as app_module

    monkeypatch.setattr(app_module, "BASE_PATH", app_module.BASE_PATH)


@pytest.mark.parametrize("prog", sorted(_MAINS))
class TestBothCommands:
    def test_help_exits_without_serving(
        self,
        prog: str,
        served: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            _MAINS[prog](["--help"])
        assert exc.value.code == 0
        assert served == []
        out = capsys.readouterr().out
        assert f"usage: {prog}" in out
        assert "--host" in out and "--port" in out

    def test_an_unknown_argument_is_an_error(
        self, prog: str, served: list[dict[str, Any]]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            _MAINS[prog](["--no-such-option"])
        assert exc.value.code == 2
        assert served == []

    def test_host_and_port_reach_uvicorn(
        self, prog: str, served: list[dict[str, Any]], base_path: None
    ) -> None:
        _MAINS[prog](["--host", "127.0.0.2", "--port", "8080"])
        assert len(served) == 1
        assert served[0]["host"] == "127.0.0.2"
        assert served[0]["port"] == 8080

    def test_the_default_host_is_loopback(
        self, prog: str, served: list[dict[str, Any]], base_path: None
    ) -> None:
        _MAINS[prog]([])
        assert served[0]["host"] == "127.0.0.1"
        assert served[0]["port"] == (9000 if prog == "viewer" else 9100)

    def test_the_environment_still_configures_it(
        self,
        prog: str,
        served: list[dict[str, Any]],
        base_path: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(f"{_PREFIX[prog]}_HOST", "127.0.0.3")
        monkeypatch.setenv(f"{_PREFIX[prog]}_PORT", "8181")
        _MAINS[prog]([])
        assert (served[0]["host"], served[0]["port"]) == ("127.0.0.3", 8181)

    def test_a_flag_wins_over_the_environment(
        self,
        prog: str,
        served: list[dict[str, Any]],
        base_path: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(f"{_PREFIX[prog]}_HOST", "127.0.0.3")
        monkeypatch.setenv(f"{_PREFIX[prog]}_PORT", "8181")
        _MAINS[prog](["--host", "127.0.0.4", "--port", "8282"])
        assert (served[0]["host"], served[0]["port"]) == ("127.0.0.4", 8282)

    def test_a_malformed_port_is_an_error(
        self,
        prog: str,
        served: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(f"{_PREFIX[prog]}_PORT", "eighty")
        with pytest.raises(SystemExit) as exc:
            _MAINS[prog]([])
        assert exc.value.code == 2
        assert served == []

    def test_every_interface_is_served_with_a_warning(
        self,
        prog: str,
        served: list[dict[str, Any]],
        base_path: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _MAINS[prog](["--host", "0.0.0.0"])  # noqa: S104
        assert served[0]["host"] == "0.0.0.0"  # noqa: S104
        assert "WARNING" in capsys.readouterr().err

    def test_loopback_is_served_without_a_warning(
        self,
        prog: str,
        served: list[dict[str, Any]],
        base_path: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _MAINS[prog](["--host", "127.0.0.1"])
        assert "WARNING" not in capsys.readouterr().err


class TestViewer:
    def test_base_path_reaches_the_app(
        self, served: list[dict[str, Any]], base_path: None, tmp_path: Path
    ) -> None:
        from autoware_carla_scenario.ui import app as app_module

        ui.main(["--base-path", str(tmp_path)])
        assert app_module.BASE_PATH == tmp_path.resolve()

    def test_fetch_rerun_web_viewer_still_fetches_instead_of_serving(
        self, served: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetched: list[list[str]] = []
        monkeypatch.setattr(ui, "_fetch_web_viewer", fetched.append)
        ui.main(["--fetch-rerun-web-viewer", "--from", "x.tgz", "--version", "1"])
        assert fetched == [["--from", "x.tgz", "--version", "1"]]
        assert served == []

    def test_a_mistyped_fetch_flag_is_an_error(
        self, served: list[dict[str, Any]]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            ui.main(["--fetch-rerun-web-viewr"])
        assert exc.value.code == 2
        assert served == []


class TestEditor:
    def test_directories_reach_the_app(
        self,
        served: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from autoware_carla_scenario.editor import app as app_module

        made: list[dict[str, Any]] = []

        def create_app(**kwargs: Any) -> object:
            made.append(kwargs)
            return object()

        monkeypatch.setattr(app_module, "create_app", create_app)
        monkeypatch.setenv("SCENARIO_EDITOR_EXPORT_DIR", str(tmp_path / "env"))
        editor.main(["--drafts", str(tmp_path / "drafts")])
        assert made == [
            {"draft_dir": tmp_path / "drafts", "export_dir": tmp_path / "env"}
        ]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("prog", sorted(_MAINS))
def test_the_console_script_answers_help(prog: str) -> None:
    """The reproduction of #712, run as the console script runs it.

    Before the fix, ``--help`` started the server, which never exits; the
    port is a free one so that server could not collide with another.
    """
    module = "ui" if prog == "viewer" else "editor"
    env = {**os.environ, f"{_PREFIX[prog]}_PORT": str(_free_port())}
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.argv[0] = {prog!r}; "
                f"from autoware_carla_scenario.{module} import main; main()",
                "--help",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"`{prog} --help` served instead of printing its usage")
    assert result.returncode == 0, result.stderr
    assert f"usage: {prog}" in result.stdout
    assert "Uvicorn running" not in result.stderr
