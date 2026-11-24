"""The SUMO backend starts SUMO through TraCI without leaking it (#2321).

``traci.start`` launched a new SUMO for each of its 60 tries, left every one
running when it failed and kept SUMO's standard error out of its message.  The
backend now starts one SUMO itself and connects with ``traci.init``: these
tests run it against fake SUMO executables (small Python scripts) and the real
``traci`` module, so they need neither SUMO nor libsumo.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from autoware_carla_scenario.traffic import TrafficBackendError
from autoware_carla_scenario.traffic.sumo import mesosim
from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
from autoware_carla_scenario.traffic.sumo.config import SumoBackendConfig

traci = pytest.importorskip("traci")


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """The processes Popen started; none is left running afterwards, and no
    TraCI connection is left registered."""
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


def _fake_sumo(tmp_path: Path, body: str) -> list[str]:
    """The command of a fake SUMO running *body*, ``port`` set to the value
    of its ``--remote-port``."""
    script = tmp_path / "fake_sumo.py"
    script.write_text(
        "import sys\n"
        "port = int(sys.argv[sys.argv.index('--remote-port') + 1])\n"
        + textwrap.dedent(body)
    )
    return [sys.executable, str(script)]


def _backend() -> SumoTrafficBackend:
    backend = SumoTrafficBackend(
        SumoBackendConfig(use_libsumo=False, traffic_light_authority="none")
    )
    backend._traci = traci
    return backend


def test_sumo_that_exits_at_once_fails_fast_with_its_stderr(
    tmp_path: Path, started: list[Any]
) -> None:
    cmd = _fake_sumo(
        tmp_path,
        """
        sys.stderr.write("Error: Could not load route file 'bad.rou.xml'\\n")
        sys.exit(1)
        """,
    )
    began = time.monotonic()
    with pytest.raises(
        TrafficBackendError,
        match=r"^SUMO did not start: Error: Could not load route file 'bad\.rou\.xml'$",
    ):
        _backend()._start_traci(cmd)
    assert time.monotonic() - began < 10.0  # not 60 launches a second apart
    (process,) = started  # one SUMO, not one per try
    assert process.returncode == 1  # reaped


def test_it_works_after_libsumo_replaced_traci_exceptions(
    tmp_path: Path, started: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing libsumo swaps ``traci.exceptions.TraCIException`` for its own
    class while traci's connection code raises the original: a SUMO that
    exits is still seen as one, not reported as a TraCI error."""

    class _LibsumoTraCIException(Exception):
        pass

    monkeypatch.setattr(traci.exceptions, "TraCIException", _LibsumoTraCIException)
    cmd = _fake_sumo(
        tmp_path,
        """
        import time
        time.sleep(0.3)
        sys.stderr.write("Error: no network\\n")
        sys.exit(1)
        """,
    )
    with pytest.raises(
        TrafficBackendError, match=r"^SUMO did not start: Error: no network$"
    ):
        _backend()._start_traci(cmd)
    (process,) = started
    assert process.returncode == 1


def test_sumo_that_never_listens_is_killed_at_the_deadline(
    tmp_path: Path, started: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mesosim, "_CONNECT_TIMEOUT_S", 0.5)
    cmd = _fake_sumo(
        tmp_path,
        """
        import time
        sys.stderr.write("Loading net-file ...\\n")
        sys.stderr.flush()
        time.sleep(60)
        """,
    )
    began = time.monotonic()
    with pytest.raises(
        TrafficBackendError,
        match=r"did not listen for TraCI within 0\.5 s: Loading net-file \.\.\.$",
    ):
        _backend()._start_traci(cmd)
    assert time.monotonic() - began < 10.0
    (process,) = started
    assert process.returncode == -9  # killed and reaped


def test_a_failed_handshake_leaves_no_connection_registered(
    tmp_path: Path, started: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SUMO accepts the connection and drops it before answering: each try's
    label is unregistered, so the next try is not refused as 'already
    active', and SUMO is killed at the deadline."""
    monkeypatch.setattr(mesosim, "_CONNECT_TIMEOUT_S", 1.0)
    cmd = _fake_sumo(
        tmp_path,
        """
        import socket
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", port))
        server.listen()
        while True:
            conn, _ = server.accept()
            conn.close()
        """,
    )
    with pytest.raises(TrafficBackendError, match="did not listen for TraCI"):
        _backend()._start_traci(cmd)
    assert not traci.connection.has("default")
    (process,) = started
    assert process.returncode == -9


def test_a_connection_left_registered_is_refused_before_sumo_starts(
    tmp_path: Path, started: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(traci, "getConnection", lambda label: object())
    with pytest.raises(TrafficBackendError, match="'default' is already active"):
        _backend()._start_traci(_fake_sumo(tmp_path, "sys.exit(0)\n"))
    assert started == []


def test_a_missing_sumo_binary_is_a_backend_error(
    tmp_path: Path, started: list[Any]
) -> None:
    with pytest.raises(TrafficBackendError, match="SUMO did not start"):
        _backend()._start_traci([str(tmp_path / "no-such-sumo")])


def test_a_port_taken_before_sumo_bound_it_is_tried_again(
    tmp_path: Path, started: list[Any]
) -> None:
    """A SUMO that exits because its port was taken meanwhile is started
    again on another port, as traci.start would, one SUMO at a time; any
    other early exit is not."""
    marker = tmp_path / "launched"
    cmd = _fake_sumo(
        tmp_path,
        f"""
        from pathlib import Path
        marker = Path({str(marker)!r})
        if not marker.exists():
            marker.write_text("")
            sys.stderr.write(
                "Error: tcpip::Socket::accept() Unable to create listening "
                "socket: Address already in use\\n"
            )
            sys.exit(1)
        sys.stderr.write("Error: no network\\n")
        sys.exit(1)
        """,
    )
    with pytest.raises(
        TrafficBackendError, match=r"^SUMO did not start: Error: no network$"
    ):
        _backend()._start_traci(cmd)
    first, second = started  # one after the other, each reaped
    assert first.returncode == 1 and second.returncode == 1


def test_a_port_always_taken_gives_up(tmp_path: Path, started: list[Any]) -> None:
    cmd = _fake_sumo(
        tmp_path,
        """
        sys.stderr.write("Error: Address already in use\\n")
        sys.exit(1)
        """,
    )
    with pytest.raises(TrafficBackendError, match="Address already in use$"):
        _backend()._start_traci(cmd)
    assert len(started) == mesosim._PORT_TRIES
    assert all(p.returncode == 1 for p in started)


def _running_sumo(tmp_path: Path, name: str) -> list[str]:
    """A fake SUMO named *name* that runs until it is killed."""
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    script.chmod(0o755)
    return [str(script)]


@pytest.mark.parametrize("name", ["sumo", "sumo-gui"])
def test_close_ends_a_headless_sumo_but_leaves_sumo_gui_open(
    tmp_path: Path,
    started: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """A headless SUMO that does not exit once its connection closes is
    killed; a sumo-gui, which stays open without --quit-on-end, is left for
    the user to close."""
    monkeypatch.setattr(mesosim, "_EXIT_TIMEOUT_S", 0.2)
    backend = _backend()
    monkeypatch.setattr(backend, "_traci", _Connected())
    backend._start_traci(_running_sumo(tmp_path, name))
    backend._connected = True
    (process,) = started
    stderr = backend._sumo_stderr
    backend.close()
    assert backend._sumo_process is None and stderr is not None and stderr.closed
    if name == "sumo-gui":
        assert process.poll() is None  # still open; the fixture ends it
        process.kill()
        process.wait()
    else:
        assert process.returncode == -9


class _Connected:
    """A traci whose init connects at once and whose close does nothing."""

    __name__ = "traci"

    def getConnection(self, label: str) -> Any:  # noqa: N802 - traci's name
        raise traci.exceptions.TraCIException(f"Connection {label!r} is not known.")

    def init(self, port: int, **kwargs: Any) -> None:
        pass

    def close(self, wait: bool = True) -> None:
        pass
