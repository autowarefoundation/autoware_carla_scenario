"""Unit tests for CarlaServerManager."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from autoware_carla_scenario import CarlaServerManager
from autoware_carla_scenario.carla_install import HOME_ENV, LAUNCHER, install_dir


def test_the_installed_carla_is_the_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without CARLA_EXECUTABLE, the launcher scenario-setup installed is used."""
    monkeypatch.delenv(CarlaServerManager.ENV_VAR, raising=False)
    monkeypatch.setenv(HOME_ENV, str(tmp_path))
    assert CarlaServerManager.executable() is None
    launcher = install_dir() / LAUNCHER
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\n")
    assert CarlaServerManager.executable() == launcher
    monkeypatch.setenv(CarlaServerManager.ENV_VAR, "/opt/carla/CarlaUnreal.sh")
    assert CarlaServerManager.executable() == Path("/opt/carla/CarlaUnreal.sh")


def test_start_raises_without_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CarlaServerManager.start() raises RuntimeError when there is no launcher.

    reuse_if_running=False forces start() to attempt launching a new process
    (rather than reusing an already-running server), so the missing-env-var
    guard is always reached regardless of whether CARLA is running locally.
    """
    monkeypatch.delenv(CarlaServerManager.ENV_VAR, raising=False)
    monkeypatch.setenv(HOME_ENV, str(tmp_path))  # nothing installed there
    manager = CarlaServerManager(reuse_if_running=False)
    with pytest.raises(RuntimeError, match=CarlaServerManager.ENV_VAR):
        manager.start()


def test_is_alive_false_before_start() -> None:
    """is_alive() returns False when no process has been started."""
    manager = CarlaServerManager()
    assert manager.is_alive() is False


def test_stop_is_idempotent() -> None:
    """Calling stop() on an unstarted manager must not raise."""
    manager = CarlaServerManager()
    manager.stop()  # Should not raise


@pytest.mark.integration
class TestCarlaServerIntegration:
    """Integration tests that verify the session-scoped CARLA server.

    These tests reuse the server already started by the ``carla_queue``
    session fixture.  They do NOT start a new CarlaServerManager because
    doing so would attempt to bind the same port twice.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_carla(self, carla_queue) -> None:  # noqa: ANN001
        """Depend on the session fixture; skips automatically if CARLA is unavailable."""

    def test_server_is_alive(self, carla_queue) -> None:  # noqa: ANN001
        """The session server must be reachable during the test run."""
        assert carla_queue._server.is_alive()

    def test_server_process_is_running(self, carla_queue) -> None:  # noqa: ANN001
        """The server process is alive (owned or reused)."""
        server = carla_queue._server
        if server._reused:
            # Externally-managed server: no _process, but must be pingable.
            assert server._ping()
        else:
            assert server._process is not None
            assert server._process.poll() is None


def _launched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    port: int,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """The arguments start() launches CARLA with, the launcher exiting at once."""
    argv = tmp_path / "argv"
    launcher = tmp_path / "CarlaUnreal.sh"
    launcher.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {argv}\nexit 3\n")
    launcher.chmod(0o755)
    monkeypatch.setenv(CarlaServerManager.ENV_VAR, str(launcher))
    manager = CarlaServerManager(
        port=port, extra_args=extra_args, reuse_if_running=False
    )
    with pytest.raises(RuntimeError, match="exited with code 3"):
        manager.start()
    manager.stop()
    return argv.read_text().split()


def test_a_launched_server_listens_on_the_configured_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_port: int
) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    assert _launched(monkeypatch, tmp_path, free_port) == [
        f"-carla-rpc-port={free_port}"
    ]
    assert _launched(monkeypatch, tmp_path, free_port, ["-carla-rpc-port=3000"]) == [
        "-carla-rpc-port=3000"
    ]


def test_without_a_display_the_server_renders_off_screen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_port: int
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert "-RenderOffScreen" in _launched(monkeypatch, tmp_path, free_port)
