"""A sweep's case runs its own ``atexit`` handlers, and none of the sweep's (#1568).

The sweeper forks each case (``sweeper/isolation.py``) and ends the child with
``os._exit``, which runs no ``atexit`` handler.  What a case registered one for
-- the directory of the Unix socket it serves its policy on
(``driver.transport=unix``), its Docker containers, its CommonRoad workers --
was left behind, once per case.  The child now runs the handlers the case
registered before it exits; those it inherited from the sweep (which would stop
the sweep's own servers) it never runs.

The sweep here is a fresh interpreter whose ``$TMPDIR`` is the test's, so
nothing it serves outlives the test.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "linux", reason="the sweeper forks its cases (Linux)"
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("carla_driver_interface.inprocess") is None,
        reason="carla-driver-interface < 2.2 has no in-process transports",
    ),
]

#: A sweep of two cases, each serving its policy on a Unix socket and
#: registering an exit handler of its own; one returns, one raises.  The sweep
#: registered one before forking.  Every handler that runs appends
#: "<who> <pid>" to argv[1]/exits.
_SWEEP = """
import atexit, os, sys
from pathlib import Path

from carla_driver_interface.driver import BaseDriver, DriveResult

from autoware_carla_scenario.driver import egodriver_client
from autoware_carla_scenario.driver.geometry import Trajectory
from autoware_carla_scenario.sweeper.isolation import _launch_isolated

out = Path(sys.argv[1])


class _Still(BaseDriver):
    name = "still"

    def drive(self, ctx):
        return DriveResult(trajectory_in_rig=Trajectory.empty())


egodriver_client.load_policy = lambda spec: _Still()


def note(who):
    with open(out / "exits", "a", encoding="utf-8") as exits:
        exits.write(f"{who} {os.getpid()}\\n")


atexit.register(note, "sweep")


def case(name, fails):
    def run_case():
        address = egodriver_client._serve_on_unix_socket("still")
        with open(out / "cases", "a", encoding="utf-8") as cases:
            cases.write(f"{name} {os.getpid()} {address}\\n")
        atexit.register(note, name)
        if fails:
            raise RuntimeError("the case failed")
        return 0

    return run_case


print(_launch_isolated(case("passes", False), 60), flush=True)
print(_launch_isolated(case("raises", True), 60), flush=True)
"""


@pytest.fixture
def tmpdir_short() -> Iterator[Path]:
    """A directory short enough for a socket path under it, removed after."""
    directory = Path(tempfile.mkdtemp(prefix="acs-iso-", dir="/tmp"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_a_case_runs_its_own_exit_handlers_and_none_of_the_sweeps(
    tmpdir_short: Path,
) -> None:
    out = tmpdir_short / "out"
    out.mkdir()
    env = dict(os.environ, TMPDIR=str(tmpdir_short))
    sweep = subprocess.run(
        [sys.executable, "-c", _SWEEP, str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert sweep.returncode == 0, sweep.stderr
    assert sweep.stdout.splitlines() == ["(True, 0)", "(False, 1)"], sweep.stderr

    cases = {}
    for line in (out / "cases").read_text(encoding="utf-8").splitlines():
        name, pid, address = line.split()
        cases[name] = (int(pid), Path(address.removeprefix("unix://")))
    assert set(cases) == {"passes", "raises"}
    # Each case served on a socket of its own, under the sweep's $TMPDIR...
    for _, socket in cases.values():
        assert socket.parent.parent == tmpdir_short
    assert cases["passes"][1] != cases["raises"][1]
    # ...and removed it as it exited, raising or not.
    assert sorted(p.name for p in tmpdir_short.glob("acs-policy-*")) == []

    exits = (out / "exits").read_text(encoding="utf-8").splitlines()
    # Each case ran its own handler, once, in its own process; the sweep's
    # ran once, in the sweep, at its exit -- never in a case.
    assert exits == [
        f"passes {cases['passes'][0]}",
        f"raises {cases['raises'][0]}",
        f"sweep {sweep_pid(exits)}",
    ]
    assert sweep_pid(exits) not in {pid for pid, _ in cases.values()}


def sweep_pid(exits: list[str]) -> int:
    """The pid the sweep's handler wrote, from the last of *exits*."""
    who, pid = exits[-1].split()
    assert who == "sweep"
    return int(pid)


def test_commonroad_registers_its_close_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A case that starts CommonRoad workers has them closed at its exit.

    Registered when this process starts its workers, not at import: a case
    forked from a sweep that imported the module drops the sweep's handlers.
    """
    from autoware_carla_scenario.commonroad import session

    registered: list[object] = []
    pid = [100]
    monkeypatch.setattr(session.atexit, "register", registered.append)
    monkeypatch.setattr(session.os, "getpid", lambda: pid[0])
    monkeypatch.setattr(session, "_CLOSE_AT_EXIT_PID", None)
    session._close_at_exit()
    session._close_at_exit()
    assert registered == [session.close_commonroad_workers]
    pid[0] = 101  # a forked case
    session._close_at_exit()
    assert registered == [session.close_commonroad_workers] * 2
