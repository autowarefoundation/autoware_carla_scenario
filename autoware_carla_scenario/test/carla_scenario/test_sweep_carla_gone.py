"""The next case of a sweep does not meet the CARLA a stopped case left (#717).

A case that times out or dies of a signal is ``SIGKILL``-ed with its process
group, but the CARLA server it launched is in a session of its own: its
watchdog stops it about a second later, and CARLA may take a while to go.
The sweeper waits until ``server.host:server.port`` stops answering before
the next attempt, so that attempt neither reuses the dying server nor starts
one next to it -- unless the server was there before the case (one the user
started), which no case stops and the sweeper does not wait for.

The CARLA here is a fake ``CarlaUnreal.sh`` that, like the real one, runs its
"UE5 binary" as a child: a Python script listening on the test's own free
port, which holds the port for a few seconds after ``SIGTERM`` (a server
tearing down).  Every process a test starts is killed when it ends.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from autoware_carla_scenario.server import _WATCHDOG_GRACE_S, CarlaServerManager
from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as lcs

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the sweeper forks its cases (Linux)"
)

#: How long the fake CARLA holds its port after ``SIGTERM``.
_LINGER_S = 3.0

#: The fake "UE5 binary": listens on ``-carla-rpc-port`` until ``SIGTERM``,
#: then holds the port for ``argv[2]`` s more.  Writes "pid pgid" to argv[1].
_FAKE_UE5 = """
import os, signal, socket, sys, time
from pathlib import Path

pid_file, linger = Path(sys.argv[1]), float(sys.argv[2])
port = next(
    int(arg.split("=", 1)[1]) for arg in sys.argv[3:]
    if arg.startswith("-carla-rpc-port=")
)
stopping = []
signal.signal(signal.SIGTERM, lambda signum, frame: stopping.append(signum))
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", port))
server.listen(16)
server.settimeout(0.1)
tmp = pid_file.with_suffix(".tmp")
tmp.write_text(f"{os.getpid()} {os.getpgid(0)}")
tmp.rename(pid_file)
while not stopping:
    try:
        connection, _ = server.accept()
        connection.close()
    except OSError:
        pass
time.sleep(linger)  # still tearing down, the port still held
"""

#: A ``@hydra.main`` app swept by LaneletConstraintSweeper, expanded to two
#: cases.  Each attempt notes whether the port answered when it started; the
#: first attempt of case 1 starts CARLA as a case's queue does
#: (``reuse_if_running``) and hangs past its timeout.
_SWEEP_APP = """
import json, os, time
from pathlib import Path

from autoware_carla_scenario import hydra_compat  # noqa: F401

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import autoware_carla_scenario.sweeper.lanelet_constraint_sweeper as lcs
from autoware_carla_scenario.server import CarlaServerManager

lcs.check_constraint_keys = lambda sweep: None
lcs.resolve_map_paths = lambda cfg: None
lcs.load_map = lambda paths: object()
lcs.scenario_controls = lambda cfg: None
lcs.expand_sweep = lambda sweep, lanelet_map, arguments, **_: [
    ["+case=1", *arguments],
    ["+case=2", *arguments],
]
# The fake answers TCP, not CARLA's RPC.
CarlaServerManager._ping = lambda self: lcs._carla_answers(self.host, self.port)


@hydra.main(version_base=None, config_path=None, config_name=None)
def task(cfg: DictConfig) -> None:
    work = Path(cfg.work)
    (work / f"case{os.getpid()}.pid").write_text(str(os.getpid()))
    counter = work / f"attempts{cfg.case}"
    attempt = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(attempt))
    busy = lcs._carla_answers(cfg.server.host, cfg.server.port)
    print(f"case {cfg.case} attempt {attempt}: port busy={busy}", flush=True)
    (work / f"busy{cfg.case}.{attempt}").write_text(str(busy))
    if cfg.case == 1 and attempt == 1:
        server = CarlaServerManager(host=cfg.server.host, port=cfg.server.port)
        server.start()
        (work / "started").write_text("reused" if server._reused else "launched")
        time.sleep(600)  # past the job timeout: SIGKILL-ed by the sweeper
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    (output_dir / "App_result.json").write_text(json.dumps({"passed": True}))


if __name__ == "__main__":
    task()
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


class _FakeCarla:
    """A fake ``CarlaUnreal.sh`` in *directory*, and the pids it writes."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.ue5 = directory / "fake_ue5.py"
        self.ue5.write_text(textwrap.dedent(_FAKE_UE5), encoding="utf-8")
        self.ue5_pid = directory / "ue5.pid"
        self.exe = directory / "CarlaUnreal.sh"
        # A child, not exec: the real launcher runs UE5 the same way.
        self.exe.write_text(
            "#!/bin/sh\n"
            f"'{sys.executable}' '{self.ue5}' '{self.ue5_pid}' {_LINGER_S} \"$@\"\n"
        )
        self.exe.chmod(0o755)

    def start_as_user(self, port: int) -> subprocess.Popen[bytes]:
        """A CARLA the user started, outside any case."""
        return subprocess.Popen(
            [
                sys.executable,
                str(self.ue5),
                str(self.ue5_pid),
                "0",
                f"-carla-rpc-port={port}",
            ],
            start_new_session=True,
        )

    def pid(self) -> int | None:
        try:
            return int(self.ue5_pid.read_text().split()[0])
        except (FileNotFoundError, ValueError, IndexError):
            return None

    def kill(self) -> None:
        """Kill the fake, its launcher and the watchdog that leads its group."""
        try:
            pid, pgid = (int(field) for field in self.ue5_pid.read_text().split())
        except (FileNotFoundError, ValueError):
            return
        for kill, target in ((os.killpg, pgid), (os.kill, pid)):
            try:
                kill(target, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


@pytest.fixture
def fake_carla(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeCarla]:
    carla = _FakeCarla(tmp_path)
    monkeypatch.setenv(CarlaServerManager.ENV_VAR, str(carla.exe))
    yield carla
    carla.kill()


def _sweep(tmp_path: Path, port: int) -> subprocess.CompletedProcess[str]:
    """Run the two-case sweep: case 1's first attempt times out, then a retry."""
    script = tmp_path / "app.py"
    script.write_text(textwrap.dedent(_SWEEP_APP), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "SWEEP_RESUME_FROM"}
    sweeper = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--multirun",
            "hydra/sweeper=lanelet_constraint",
            "+sweep.constraints={}",
            "+sweep.job_timeout_seconds=3",
            "+server.host=127.0.0.1",
            f"+server.port={port}",
            "+server.cooldown_seconds=0",
            "+server.cooldown_max_retries=1",
            f"+work={tmp_path}",
            f"hydra.sweep.dir={tmp_path / 'multirun'}",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = sweeper.communicate(timeout=120)
    finally:
        try:
            os.killpg(sweeper.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        sweeper.wait()
        for case in tmp_path.glob("case*.pid"):
            try:
                os.kill(int(case.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    return subprocess.CompletedProcess(sweeper.args, sweeper.returncode, output, "")


def test_the_retry_after_a_timed_out_case_waits_for_its_carla_to_go(
    fake_carla: _FakeCarla, free_port: int
) -> None:
    result = _sweep(fake_carla.directory, free_port)
    output = result.stdout
    assert result.returncode == 0, output
    assert (fake_carla.directory / "started").read_text() == "launched", output
    assert "TIMED OUT" in output, output
    assert "Waiting up to" in output, output
    # Neither the retry nor the next case found the dying server on the port.
    assert (fake_carla.directory / "busy1.1").read_text() == "False", output
    assert (fake_carla.directory / "busy1.2").read_text() == "False", output
    assert (fake_carla.directory / "busy2.1").read_text() == "False", output
    pid = fake_carla.pid()
    assert pid is not None and not _alive(pid), "the case's CARLA outlived the sweep"


def test_a_carla_the_user_started_is_not_waited_for(
    fake_carla: _FakeCarla, free_port: int
) -> None:
    user = fake_carla.start_as_user(free_port)
    try:
        deadline = time.monotonic() + 30
        while not lcs._carla_answers("127.0.0.1", free_port):
            assert time.monotonic() < deadline, "the user's CARLA never answered"
            assert user.poll() is None, "the user's CARLA exited"
            time.sleep(0.05)
        result = _sweep(fake_carla.directory, free_port)
        output = result.stdout
        assert result.returncode == 0, output
        assert (fake_carla.directory / "started").read_text() == "reused", output
        assert "TIMED OUT" in output, output
        assert "Waiting up to" not in output, output
        # The retry reused the user's server, which nothing stopped.
        assert (fake_carla.directory / "busy1.2").read_text() == "True", output
        assert user.poll() is None, "the user's CARLA was stopped"
    finally:
        user.kill()
        user.wait()


def test_wait_until_carla_gone_returns_once_the_port_is_released(
    free_port: int,
) -> None:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", free_port))
    listener.listen(16)
    closer = threading.Timer(0.5, listener.close)
    closer.start()
    try:
        started = time.monotonic()
        assert lcs._wait_until_carla_gone("127.0.0.1", free_port, timeout=10.0)
        assert 0.4 <= time.monotonic() - started < 5.0
    finally:
        closer.cancel()
        listener.close()


def test_wait_until_carla_gone_gives_up_after_its_timeout(free_port: int) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", free_port))
        listener.listen(16)
        started = time.monotonic()
        assert not lcs._wait_until_carla_gone("127.0.0.1", free_port, timeout=0.5)
        assert time.monotonic() - started < 5.0


def test_wait_until_carla_gone_returns_at_once_when_nothing_answers(
    free_port: int,
) -> None:
    started = time.monotonic()
    assert lcs._wait_until_carla_gone("127.0.0.1", free_port, timeout=10.0)
    assert time.monotonic() - started < 2.0


def test_the_wait_outlasts_the_watchdog_grace_period() -> None:
    assert lcs._CARLA_GONE_TIMEOUT_S > _WATCHDOG_GRACE_S
