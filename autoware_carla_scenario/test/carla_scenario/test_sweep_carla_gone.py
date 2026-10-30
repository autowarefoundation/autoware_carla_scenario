"""The next case of a sweep does not meet the CARLA a stopped case left (#717).

A case that times out or dies of a signal is ``SIGKILL``-ed with its process
group, but the CARLA server it launched is in a session of its own: its
watchdog stops it about a second later, and CARLA may take a while to go.
The sweeper waits until every CARLA the case launched has exited before the
next attempt, so that attempt neither reuses the dying server nor starts one
next to it -- even when the case was killed before its server answered.  A
server the case did not launch (one the user started) is not waited for.

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

from autoware_carla_scenario.server import (
    _WATCHDOG_GRACE_S,
    CarlaServerManager,
    _dying_with_this_process,
)
from autoware_carla_scenario.sweeper import isolation

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the sweeper forks its cases (Linux)"
)

#: How long the fake CARLA holds its port after ``SIGTERM``.
_LINGER_S = 3.0

#: The fake "UE5 binary": writes "pid pgid" to argv[1], waits
#: ``FAKE_UE5_BIND_DELAY`` s (a server still loading), then listens on
#: ``-carla-rpc-port`` until ``SIGTERM`` and holds the port ``argv[2]`` s more.
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
tmp = pid_file.with_suffix(".tmp")
tmp.write_text(f"{os.getpid()} {os.getpgid(0)}")
tmp.rename(pid_file)
time.sleep(float(os.environ.get("FAKE_UE5_BIND_DELAY", "0")))
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", port))
server.listen(16)
server.settimeout(0.1)
while not stopping:
    try:
        connection, _ = server.accept()
        connection.close()
    except OSError:
        pass
time.sleep(linger)  # still tearing down, the port still held
"""

#: A ``@hydra.main`` app swept by LaneletConstraintSweeper, expanded to two
#: cases.  Each attempt notes whether the port answered and whether anything
#: of the first attempt's CARLA still ran when it started; the first attempt
#: of case 1 starts CARLA as a case's queue does (``reuse_if_running``) and
#: then dies of ``SIGKILL``, as a case past its timeout does -- after its
#: server answers, or, with ``+kill_early=true``, while it still loads.
_SWEEP_APP = """
import json, os, signal, socket, threading, time
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


def answers(host, port):
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


# The fake answers TCP, not CARLA's RPC.
CarlaServerManager._ping = lambda self: answers(self.host, self.port)


def group_runs(pgid):
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as stat:
                fields = stat.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            return True
    return False


@hydra.main(version_base=None, config_path=None, config_name=None)
def task(cfg: DictConfig) -> None:
    work = Path(cfg.work)
    (work / f"case{os.getpid()}.pid").write_text(str(os.getpid()))
    counter = work / f"attempts{cfg.case}"
    attempt = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(attempt))
    busy = answers(cfg.server.host, cfg.server.port)
    print(f"case {cfg.case} attempt {attempt}: port busy={busy}", flush=True)
    (work / f"busy{cfg.case}.{attempt}").write_text(str(busy))
    ue5 = work / "ue5.pid"
    left = ue5.exists() and group_runs(int(ue5.read_text().split()[1]))
    (work / f"left{cfg.case}.{attempt}").write_text(str(left))
    if cfg.case == 1 and attempt == 1:
        server = CarlaServerManager(host=cfg.server.host, port=cfg.server.port)
        if cfg.get("kill_early", False):
            threading.Thread(target=server.start, daemon=True).start()
            while not ue5.exists():
                time.sleep(0.05)
            (work / "started").write_text("loading")
        else:
            server.start()
            (work / "started").write_text("reused" if server._reused else "launched")
        os.kill(os.getpid(), signal.SIGKILL)  # as the sweeper kills a hung case
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


def _group_runs(pgid: int) -> bool:
    """Whether a process of group *pgid* runs (zombies aside)."""
    return pgid in isolation._live_groups({pgid})


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

    def pids(self) -> tuple[int, int] | None:
        """The fake UE5's pid and process group, once it has written them."""
        try:
            pid, pgid = (int(field) for field in self.ue5_pid.read_text().split())
        except (FileNotFoundError, ValueError):
            return None
        return pid, pgid

    def kill(self) -> None:
        """Kill the fake, its launcher and the watchdog that leads its group."""
        pids = self.pids()
        if pids is None:
            return
        pid, pgid = pids
        for kill, target in ((os.killpg, pgid), (os.kill, pid)):
            try:
                kill(target, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


@pytest.fixture
def fake_carla(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeCarla]:
    carla = _FakeCarla(tmp_path)
    monkeypatch.setenv(CarlaServerManager.ENV_VAR, str(carla.exe))
    monkeypatch.delenv("FAKE_UE5_BIND_DELAY", raising=False)
    yield carla
    carla.kill()


def _sweep(tmp_path: Path, port: int, *extra: str) -> subprocess.CompletedProcess[str]:
    """Run the two-case sweep: case 1's first attempt dies, then a retry."""
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
            # A guard only: the case that hangs kills itself.
            "+sweep.job_timeout_seconds=120",
            "+server.host=127.0.0.1",
            f"+server.port={port}",
            "+server.cooldown_seconds=0",
            "+server.cooldown_max_retries=1",
            f"+work={tmp_path}",
            f"hydra.sweep.dir={tmp_path / 'multirun'}",
            *extra,
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = sweeper.communicate(timeout=240)
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


def _assert_retry_met_no_carla(directory: Path, output: str) -> None:
    # Neither the retry nor the next case found anything of the dead case's
    # CARLA still running, or its port taken.
    for attempt in ("1.2", "2.1"):
        assert (directory / f"left{attempt}").read_text() == "False", output
        assert (directory / f"busy{attempt}").read_text() == "False", output


def test_the_retry_after_a_killed_case_waits_for_its_carla_to_go(
    fake_carla: _FakeCarla, free_port: int
) -> None:
    result = _sweep(fake_carla.directory, free_port)
    output = result.stdout
    assert result.returncode == 0, output
    assert (fake_carla.directory / "started").read_text() == "launched", output
    assert "CRASHED (signal 9)" in output, output
    assert "Waiting up to" in output, output
    assert (fake_carla.directory / "busy1.1").read_text() == "False", output
    _assert_retry_met_no_carla(fake_carla.directory, output)
    pids = fake_carla.pids()
    assert pids is not None and not _alive(pids[0]), "the case's CARLA outlived it"


def test_a_case_killed_before_its_carla_answers_waits_for_it_too(
    fake_carla: _FakeCarla, free_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing answers on the port yet, but the server is there, loading."""
    monkeypatch.setenv("FAKE_UE5_BIND_DELAY", "2")
    result = _sweep(fake_carla.directory, free_port, "+kill_early=true")
    output = result.stdout
    assert result.returncode == 0, output
    assert (fake_carla.directory / "started").read_text() == "loading", output
    assert "Waiting up to" in output, output
    _assert_retry_met_no_carla(fake_carla.directory, output)


def test_a_carla_the_user_started_is_not_waited_for(
    fake_carla: _FakeCarla, free_port: int
) -> None:
    user = fake_carla.start_as_user(free_port)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                socket.create_connection(("127.0.0.1", free_port), timeout=1).close()
                break
            except OSError:
                pass
            assert time.monotonic() < deadline, "the user's CARLA never answered"
            assert user.poll() is None, "the user's CARLA exited"
            time.sleep(0.05)
        result = _sweep(fake_carla.directory, free_port)
        output = result.stdout
        assert result.returncode == 0, output
        assert (fake_carla.directory / "started").read_text() == "reused", output
        assert "CRASHED (signal 9)" in output, output
        assert "Waiting up to" not in output, output
        # The retry reused the user's server, which nothing stopped.
        assert (fake_carla.directory / "busy1.2").read_text() == "True", output
        assert user.poll() is None, "the user's CARLA was stopped"
    finally:
        user.kill()
        user.wait()


#: A stand-in for CARLA under the watchdog; ``argv[1]`` says whether it
#: ignores ``SIGTERM`` (a server that does not stop when told).
_STAND_IN = (
    "import signal, sys, time\n"
    "if sys.argv[1] == 'stubborn':\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "time.sleep(600)\n"
)


@pytest.fixture
def watched() -> Iterator[list[subprocess.Popen[bytes]]]:
    """Watchdogs a test starts with this process as their parent; killed after."""
    started: list[subprocess.Popen[bytes]] = []
    yield started
    for process in started:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _watchdog(started: list[subprocess.Popen[bytes]], mode: str) -> int:
    """Start the stand-in as CarlaServerManager does; return its group."""
    process = subprocess.Popen(
        _dying_with_this_process([sys.executable, "-c", _STAND_IN, mode]),
        start_new_session=True,
    )
    started.append(process)
    deadline = time.monotonic() + 30
    while not isolation._carla_groups_of(os.getpid()):
        assert time.monotonic() < deadline, "the watchdog never showed up"
        time.sleep(0.05)
    return process.pid


def test_the_carla_of_a_case_is_found_by_its_watchdog(
    watched: list[subprocess.Popen[bytes]],
) -> None:
    group = _watchdog(watched, "plain")
    assert isolation._carla_groups_of(os.getpid()) == {group}
    # Another process's CARLA is not this one's.
    assert isolation._carla_groups_of(os.getppid()) == set()


def test_wait_until_carla_gone_returns_once_the_server_exits(
    watched: list[subprocess.Popen[bytes]],
) -> None:
    group = _watchdog(watched, "plain")
    # The group is stopped as the watchdog does once its parent is gone.
    stopper = threading.Timer(0.5, os.killpg, (group, signal.SIGTERM))
    stopper.start()
    try:
        started = time.monotonic()
        assert isolation._wait_until_carla_gone(os.getpid(), timeout=30.0)
        assert time.monotonic() - started >= 0.4
        assert not _group_runs(group)
    finally:
        stopper.cancel()


def test_wait_until_carla_gone_kills_a_server_still_there_at_its_timeout(
    watched: list[subprocess.Popen[bytes]],
) -> None:
    group = _watchdog(watched, "stubborn")
    started = time.monotonic()
    assert not isolation._wait_until_carla_gone(os.getpid(), timeout=0.5)
    assert time.monotonic() - started < 10.0
    deadline = time.monotonic() + 10
    while _group_runs(group):
        assert time.monotonic() < deadline, "the stubborn server was not killed"
        time.sleep(0.05)


def test_wait_until_carla_gone_returns_at_once_when_the_case_left_none() -> None:
    started = time.monotonic()
    assert isolation._wait_until_carla_gone(os.getpid(), timeout=30.0)
    assert time.monotonic() - started < 2.0


def test_the_wait_outlasts_the_watchdog_grace_period() -> None:
    assert isolation._CARLA_GONE_TIMEOUT_S > _WATCHDOG_GRACE_S


def test_the_library_check_is_given_the_wait_python_computes() -> None:
    # sweeper.isolation hands the library check a literal in the
    # `if TYPE_CHECKING:` branch (Codon cannot read server's global at its top
    # level); it must stay the value Python computes in the `else` branch.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(isolation))
    branch = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
        and any(
            isinstance(stmt, ast.Assign)
            and [ast.unparse(t) for t in stmt.targets] == ["_CARLA_GONE_TIMEOUT_S"]
            for stmt in node.body
        )
    )
    (assign,) = [stmt for stmt in branch.body if isinstance(stmt, ast.Assign)]
    assert ast.literal_eval(assign.value) == isolation._CARLA_GONE_TIMEOUT_S
