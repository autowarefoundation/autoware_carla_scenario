"""Unit tests for CarlaServerManager."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Iterator, Optional

import pytest

from autoware_carla_scenario import CarlaServerManager
from autoware_carla_scenario import server as server_module
from autoware_carla_scenario.carla_install import HOME_ENV, LAUNCHER, install_dir
from autoware_carla_scenario.carla_release import CARLA_REF_ENV


@pytest.fixture(autouse=True)
def _no_version_check(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fakes answer pings, not version queries; the check has its own tests."""
    if "version_check" not in request.keywords:
        monkeypatch.setattr(
            CarlaServerManager, "_check_client_matches", lambda self: None
        )


class _VersionedClient:
    """A fake client of one CARLA version, talking to a server of another."""

    client_version = "1.0"
    server_version = "1.0"

    def __init__(self, host: str, port: int) -> None:
        pass

    def set_timeout(self, seconds: float) -> None:
        pass

    def get_client_version(self) -> str:
        return self.client_version

    def get_server_version(self) -> str:
        return self.server_version


@pytest.mark.version_check
def test_a_reused_server_of_the_clients_branch_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module.carla, "Client", _VersionedClient)
    manager = CarlaServerManager(host="127.0.0.1", port=2000, reuse_if_running=True)
    monkeypatch.setattr(manager, "_ping", lambda: True)
    manager.start()
    assert manager.reused


@pytest.mark.version_check
def test_a_server_of_another_branch_names_the_client_to_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CARLA 0.10 server with the 1.0 client would fail on apply_settings."""

    class _Old(_VersionedClient):
        server_version = "0.10.0-185-g0a5ce0d5b"

    monkeypatch.setenv(CARLA_REF_ENV, "1.0")
    monkeypatch.setattr(server_module.carla, "Client", _Old)
    manager = CarlaServerManager(host="127.0.0.1", port=2000, reuse_if_running=True)
    monkeypatch.setattr(manager, "_ping", lambda: True)
    with pytest.raises(RuntimeError, match=f"{CARLA_REF_ENV}=ue5-dev"):
        manager.start()
    assert not manager.reused


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


def test_extra_args_are_kept_as_given_or_empty() -> None:
    """A list of extra arguments is kept as it is; None or [] is a new []."""
    given = ["-quality-level=Low"]
    assert CarlaServerManager(extra_args=given).extra_args is given
    assert CarlaServerManager().extra_args == []
    empty: list[str] = []
    kept = CarlaServerManager(extra_args=empty).extra_args
    assert kept == [] and kept is not empty


def test_a_closed_port_does_not_ping_without_asking_the_carla_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On arm64 the CARLA client can abort the process on a refused
    connection, so a closed port is answered before it is made."""

    def no_client(*_args: object) -> None:
        raise AssertionError("the CARLA client was made for a closed port")

    monkeypatch.setattr(server_module.carla, "Client", no_client)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    assert not CarlaServerManager(host="127.0.0.1", port=port)._ping()


def test_a_reused_server_warns_that_it_keeps_its_own_arguments(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A server already running was not started with the extra arguments."""
    args = ["-ExecCmds=carla.Weather.FogDensityScale 0.01"]
    manager = CarlaServerManager(extra_args=args)
    monkeypatch.setattr(manager, "_ping", lambda: True)
    with caplog.at_level("WARNING"):
        manager.start()
    assert "FogDensityScale 0.01" in caplog.text
    caplog.clear()
    plain = CarlaServerManager()
    monkeypatch.setattr(plain, "_ping", lambda: True)
    with caplog.at_level("WARNING"):
        plain.start()
    assert caplog.text == ""


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
    # A -carla-rpc-port= that names the server's port is passed on as it is.
    same = [f"-carla-rpc-port={free_port}"]
    assert _launched(monkeypatch, tmp_path, free_port, same) == same
    # Spelled otherwise, it is followed by the exact spelling, the marker the
    # SLURM cleanup finds the server's processes by.
    upper = [f"-CARLA-RPC-PORT=0{free_port}"]
    assert _launched(monkeypatch, tmp_path, free_port, upper) == upper + same


@pytest.mark.parametrize(
    "arg",
    [
        "-carla-rpc-port=3000",
        "-carla-rpc-port=",
        "-carla-rpc-port=2x",
        "-CARLA-RPC-PORT=3000",
        "--carla-rpc-port=3000",
    ],
)
def test_a_carla_rpc_port_other_than_the_port_is_refused_before_launching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_port: int, arg: str
) -> None:
    """CARLA would listen on the extra argument's port while the manager
    waited on its own until the timeout (#2300): refused, naming both, and
    nothing is launched."""
    mark = _marking_launcher(monkeypatch, tmp_path)
    manager = CarlaServerManager(
        port=free_port, extra_args=[arg], reuse_if_running=False
    )
    with pytest.raises(ValueError) as raised:
        manager.start()
    message = str(raised.value)
    assert repr(arg) in message and str(free_port) in message, message
    assert not mark.exists()
    assert manager._process is None


def test_without_a_display_the_server_renders_off_screen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_port: int
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert "-RenderOffScreen" in _launched(monkeypatch, tmp_path, free_port)


# ---------------------------------------------------------------------------
# A server is launched only for a local host (#1984)
# ---------------------------------------------------------------------------


def _marking_launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fake CARLA launcher that leaves a mark when it runs; the mark's path."""
    mark = tmp_path / "launched"
    launcher = tmp_path / "CarlaUnreal.sh"
    launcher.write_text(f"#!/bin/sh\ntouch {mark}\nexit 3\n")
    launcher.chmod(0o755)
    monkeypatch.setenv(CarlaServerManager.ENV_VAR, str(launcher))
    return mark


@pytest.mark.parametrize("reuse", [True, False])
@pytest.mark.parametrize("host", ["192.0.2.1", "2001:db8::1", "[2001:db8::1]"])
def test_a_remote_host_that_does_not_answer_launches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host: str, reuse: bool
) -> None:
    """An unreachable remote server.host is refused at once, naming it; no
    local CARLA is launched to be polled at the remote address."""
    mark = _marking_launcher(monkeypatch, tmp_path)
    manager = CarlaServerManager(
        host=host, port=2000, timeout=30.0, reuse_if_running=reuse
    )
    monkeypatch.setattr(manager, "_ping", lambda: False)
    with pytest.raises(RuntimeError) as raised:
        manager.start()
    message = str(raised.value)
    # An IPv6 address is named in brackets, however it was given.
    where = f"[{host.strip('[]')}]" if ":" in host else host
    assert message.startswith(f"No CARLA server reachable at {where}:2000"), message
    assert "only launches CARLA for a local host" in message
    assert not mark.exists()
    assert manager._process is None


@pytest.mark.parametrize("reuse", [True, False])
def test_a_remote_server_that_answers_is_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reuse: bool
) -> None:
    """A remote server is the one there: reused, never launched (nor stopped)."""
    mark = _marking_launcher(monkeypatch, tmp_path)
    manager = CarlaServerManager(host="192.0.2.1", reuse_if_running=reuse)
    monkeypatch.setattr(manager, "_ping", lambda: True)
    manager.start()
    assert manager.reused
    assert not mark.exists()
    manager.stop()


def test_a_local_host_that_does_not_answer_is_launched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_port: int
) -> None:
    mark = _marking_launcher(monkeypatch, tmp_path)
    manager = CarlaServerManager(host="127.0.0.1", port=free_port, timeout=30.0)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        manager.start()
    manager.stop()
    assert mark.exists()


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LocalHost.",
        "carla.localhost",
        "127.0.0.1",
        "127.10.0.2",
        "::1",
        "[::1]",
        "::ffff:127.0.0.1",
        "",
        "0.0.0.0",
    ],
)
def test_loopback_and_this_machines_addresses_are_local(host: str) -> None:
    from autoware_carla_scenario.server import _is_local_host

    assert _is_local_host(host)


def test_this_machines_name_is_local() -> None:
    from autoware_carla_scenario.server import _is_local_host

    assert _is_local_host(socket.gethostname())


@pytest.mark.parametrize(
    "host",
    ["192.0.2.1", "2001:db8::1", "[2001:db8::1]", "127.0.0.256", "128.0.0.1", "a" * 70],
)
def test_other_machines_are_not_local(host: str) -> None:
    from autoware_carla_scenario.server import _is_local_host

    assert not _is_local_host(host)


# ---------------------------------------------------------------------------
# A server is never adopted from another process (#2233)
# ---------------------------------------------------------------------------


@pytest.fixture
def listener() -> Iterator[int]:
    """A plain TCP listener on loopback, not a CARLA server; its port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield int(sock.getsockname()[1])


def test_a_launched_server_that_exits_fails_though_another_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_port: int
) -> None:
    """A launched CARLA that died is not taken for the server answering the
    port: start() fails with its exit code and points to its log."""
    launcher = tmp_path / "CarlaUnreal.sh"
    launcher.write_text("#!/bin/sh\nexit 1\n")
    launcher.chmod(0o755)
    monkeypatch.setenv(CarlaServerManager.ENV_VAR, str(launcher))
    manager = CarlaServerManager(
        host="127.0.0.1", port=free_port, timeout=30.0, reuse_if_running=False
    )

    def foreign_server_answers() -> bool:
        # Answers only once the launched process has exited, as a server
        # already on the port would after the new one died binding it.
        assert manager._process is not None
        manager._process.wait()
        return True

    monkeypatch.setattr(manager, "_ping", foreign_server_answers)
    with pytest.raises(RuntimeError) as raised:
        manager.start()
    message = str(raised.value)
    assert (
        f"exited with code 1 before it became reachable at 127.0.0.1:{free_port}"
        in (message)
    )
    assert str(tmp_path / "CarlaUnreal" / "Saved" / "Logs") in message
    assert manager._process is None
    assert not manager.reused


def test_a_taken_port_is_not_launched_onto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, listener: int
) -> None:
    """Without reuse_if_running, a port another process listens on is
    refused before anything is launched."""
    mark = _marking_launcher(monkeypatch, tmp_path)
    manager = CarlaServerManager(
        host="127.0.0.1", port=listener, timeout=30.0, reuse_if_running=False
    )
    monkeypatch.setattr(manager, "_ping", lambda: True)  # never asked
    with pytest.raises(RuntimeError) as raised:
        manager.start()
    message = str(raised.value)
    assert f"Port {listener} on 127.0.0.1 is in use by another process" in message
    assert "reuse_if_running=True" in message
    assert not mark.exists()
    assert manager._process is None


def test_a_busy_server_that_answers_late_is_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, listener: int
) -> None:
    """A CARLA that misses the first ping's deadline is pinged again and
    reused, not launched over."""
    mark = _marking_launcher(monkeypatch, tmp_path)
    manager = CarlaServerManager(host="127.0.0.1", port=listener, timeout=30.0)
    answers = iter([False, False, True])
    monkeypatch.setattr(manager, "_ping", lambda: next(answers))
    monkeypatch.setattr(server_module.time, "sleep", lambda _: None)
    manager.start()
    assert manager.reused
    assert not mark.exists()
    manager.stop()


def test_a_port_that_never_answers_as_carla_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, listener: int
) -> None:
    """With reuse_if_running, a process on the port that never answers as
    CARLA is neither reused nor launched over."""
    mark = _marking_launcher(monkeypatch, tmp_path)
    monkeypatch.setattr(CarlaServerManager, "REUSE_PING_S", 0.2)
    manager = CarlaServerManager(host="127.0.0.1", port=listener, timeout=30.0)
    monkeypatch.setattr(manager, "_ping", lambda: False)
    with pytest.raises(RuntimeError) as raised:
        manager.start()
    message = str(raised.value)
    assert f"Port {listener} on 127.0.0.1 is in use by a process" in message
    assert "did not answer as a CARLA server within 0.2s" in message
    assert not mark.exists()
    assert not manager.reused


def test_a_port_freed_while_it_is_pinged_is_launched_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_port: int
) -> None:
    """A port that stops accepting connections during the reuse wait (a
    server still shutting down, as a queue's restart stops one) is launched
    on, not refused."""
    mark = _marking_launcher(monkeypatch, tmp_path)
    manager = CarlaServerManager(host="127.0.0.1", port=free_port, timeout=30.0)
    connects = iter([True, False, False])
    monkeypatch.setattr(manager, "_tcp_connects", lambda: next(connects, False))
    monkeypatch.setattr(manager, "_ping", lambda: False)
    monkeypatch.setattr(server_module.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError) as raised:
        manager.start()
    # The marking launcher exits 3: it was launched, then failed as itself.
    assert "exited with code 3" in str(raised.value)
    assert mark.exists()


# ---------------------------------------------------------------------------
# A bracketed IPv6 host is connected to without its brackets (#2300)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["[::1]", "::1"])
def test_a_bracketed_ipv6_host_is_connected_to_unbracketed(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    """getaddrinfo() resolves no "[::1]": the TCP probe and the CARLA client
    are given "::1", so a server running there is found and reused."""
    connected: list[tuple[str, int]] = []
    clients: list[tuple[str, int]] = []

    class _Connection:
        def close(self) -> None:
            pass

    def create_connection(
        address: tuple[str, int], timeout: Optional[float] = None
    ) -> _Connection:
        connected.append(address)
        return _Connection()

    class _Client:
        def __init__(self, client_host: str, port: int) -> None:
            clients.append((client_host, port))

        def set_timeout(self, seconds: float) -> None:
            pass

        def get_server_version(self) -> str:
            return "0.10.0"

    monkeypatch.setattr(server_module.socket, "create_connection", create_connection)
    monkeypatch.setattr(server_module.carla, "Client", _Client)
    manager = CarlaServerManager(host=host, port=2000)
    assert manager.host == "::1"
    manager.start()
    assert manager.reused
    assert connected == [("::1", 2000)]
    assert clients == [("::1", 2000)]


def test_an_ipv6_host_is_named_in_brackets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = CarlaServerManager(host="[::1]", port=2000, timeout=0.0)
    monkeypatch.setattr(manager, "_ping", lambda: False)
    with pytest.raises(RuntimeError, match=r"at \[::1\]:2000"):
        manager._wait_until_ready()


# ---------------------------------------------------------------------------
# The working directory's .env is loaded (#2300)
# ---------------------------------------------------------------------------


def test_the_working_directorys_dotenv_is_loaded(tmp_path: Path) -> None:
    """The .env nearest the working directory is loaded on import, not the
    one nearest the package, and the environment wins over it."""
    import os
    import subprocess
    import sys

    run_root = tmp_path / "run"
    run_dir = run_root / "sub" / "dir"
    run_dir.mkdir(parents=True)
    (run_root / ".env").write_text(
        "ACS_DOTENV_FROM_CWD=loaded\nACS_DOTENV_SET=from-file\n"
    )
    # A script file, not `python -c`: python-dotenv searches from the working
    # directory anyway when __main__ has no __file__, which would hide the
    # search from the package's own directory this guards against.
    probe = tmp_path / "probe" / "probe.py"
    probe.parent.mkdir()
    probe.write_text(
        "import os\n"
        "import autoware_carla_scenario.server\n"
        "print(os.environ.get('ACS_DOTENV_FROM_CWD'), os.environ['ACS_DOTENV_SET'])\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("ACS_DOTENV_FROM_CWD", "PYTHON_DOTENV_DISABLED")
    }
    env["ACS_DOTENV_SET"] = "from-env"
    out = subprocess.run(
        [sys.executable, str(probe)],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["loaded", "from-env"]
