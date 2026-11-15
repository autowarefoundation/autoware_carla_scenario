"""SUMO's mesoscopic process is never leaked (#2174): not when the model fails
after connecting, not when SUMO never listens, and its standard error is a
file, not a pipe nobody drains, that still explains an early exit.

A fake process and a fake TraCI stand in for SUMO, so these run without the
``sumo`` extra.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import pytest

from autoware_carla_scenario.traffic.sumo import mesosim
from autoware_carla_scenario.traffic.sumo.mesosim import SumoMesoscopicTraffic


class _FatalTraCIError(Exception):
    pass


class _TraCIException(Exception):
    pass


class _Process:
    """A SUMO process: running until killed or told to exit with *exit_code*."""

    def __init__(
        self,
        args: list[str],
        *,
        stdout: Any,
        stderr: Any,
        exit_code: Optional[int] = None,
        error: bytes = b"",
    ) -> None:
        self.args = args
        self.stdout = stdout
        self.stderr = stderr
        self.pid = 4242
        self.returncode = exit_code
        self.killed = 0
        self.waited: list[Optional[float]] = []
        if error:
            stderr.write(error)
            stderr.flush()

    def poll(self) -> Optional[int]:
        return self.returncode

    def kill(self) -> None:
        self.killed += 1
        if self.returncode is None:
            self.returncode = -9

    def wait(self, timeout: Optional[float] = None) -> int:
        self.waited.append(timeout)
        if self.returncode is None:
            self.returncode = 0  # it exits once its connection closes
        return self.returncode


class _Conn:
    def __init__(self, types_error: Optional[BaseException] = None) -> None:
        self.closed: list[bool] = []
        self._types_error = types_error
        self.vehicletype = self

    def getIDList(self) -> tuple[str, ...]:  # noqa: N802 - TraCI's name
        if self._types_error is not None:
            raise self._types_error
        return ("DEFAULT_VEHTYPE",)

    def close(self, wait: bool = True) -> None:
        self.closed.append(wait)


class _Sumo:
    """What Popen and traci.connect were asked, and what they answered."""

    def __init__(self) -> None:
        self.processes: list[_Process] = []
        self.conn = _Conn()
        self.exit_code: Optional[int] = None
        self.error = b""
        self.listens = True

    def popen(self, args: list[str], **kwargs: Any) -> _Process:
        process = _Process(args, **kwargs, exit_code=self.exit_code, error=self.error)
        self.processes.append(process)
        return process

    def connect(self, port: int, numRetries: int, proc: Any) -> _Conn:  # noqa: N803 - TraCI's name
        if not self.listens:
            raise OSError("connection refused")
        return self.conn


@pytest.fixture
def sumo(monkeypatch: pytest.MonkeyPatch) -> _Sumo:
    fake = _Sumo()
    traci = ModuleType("traci")
    traci.connect = fake.connect  # type: ignore[attr-defined]
    exceptions = ModuleType("traci.exceptions")
    exceptions.FatalTraCIError = _FatalTraCIError  # type: ignore[attr-defined]
    exceptions.TraCIException = _TraCIException  # type: ignore[attr-defined]
    traci.exceptions = exceptions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "traci", traci)
    monkeypatch.setitem(sys.modules, "traci.exceptions", exceptions)
    monkeypatch.setattr(subprocess, "Popen", fake.popen)
    return fake


def _model(tmp_path: Path) -> SumoMesoscopicTraffic:
    return SumoMesoscopicTraffic(
        {}, net_file=tmp_path / "net.net.xml", binary="sumo", step_length=0.05, seed=1
    )


def test_its_stderr_is_a_file_not_an_undrained_pipe(
    sumo: _Sumo, tmp_path: Path
) -> None:
    model = _model(tmp_path)
    (process,) = sumo.processes
    assert process.stderr is not subprocess.PIPE
    assert process.stderr.writable()
    model.close()
    assert sumo.conn.closed == [False]
    assert process.waited and process.waited[0] is not None
    assert process.stderr.closed
    model.close()  # twice is harmless


def test_a_failure_after_connecting_stops_sumo(sumo: _Sumo, tmp_path: Path) -> None:
    sumo.conn = _Conn(types_error=RuntimeError("TraCI broke"))
    with pytest.raises(RuntimeError, match="TraCI broke"):
        _model(tmp_path)
    (process,) = sumo.processes
    assert sumo.conn.closed == [False]
    assert process.returncode is not None and process.waited
    assert process.stderr.closed


def test_sumo_that_never_listens_is_killed_and_reaped(
    sumo: _Sumo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mesosim, "_CONNECT_TIMEOUT_S", -1.0)
    sumo.listens = False
    with pytest.raises(RuntimeError, match="did not listen for TraCI"):
        _model(tmp_path)
    (process,) = sumo.processes
    assert process.killed == 1
    assert process.waited == [mesosim._EXIT_TIMEOUT_S]
    assert process.stderr.closed


def test_an_early_exit_says_what_sumo_wrote(sumo: _Sumo, tmp_path: Path) -> None:
    sumo.exit_code = 1
    sumo.error = b"Error: Could not load net.net.xml\n"
    with pytest.raises(
        RuntimeError,
        match=r"^SUMO \(mesoscopic\) did not start: Error: Could not load net\.net\.xml$",
    ):
        _model(tmp_path)
    (process,) = sumo.processes
    assert process.waited
    assert process.stderr.closed
