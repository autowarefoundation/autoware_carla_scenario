"""A forked case's ``SystemExit`` is read as the interpreter reads it (#2321):
``sys.exit()`` (code ``None``) is success, an int is the exit status, and
anything else (a message) is a failure.

The cases are forked from a fresh interpreter, not from the test's.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the sweeper forks its cases (Linux)"
)

_SWEEP = """
import sys

from autoware_carla_scenario.sweeper.isolation import _launch_isolated


def exits(*args):
    def run_case():
        sys.exit(*args)

    return run_case


for args in [(), (None,), (0,), (3,), ("a message",)]:
    print(_launch_isolated(exits(*args), 60), flush=True)
"""


def test_a_forked_case_exits_as_its_system_exit_says() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _SWEEP],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "(True, 0)",  # sys.exit()
        "(True, 0)",  # sys.exit(None)
        "(True, 0)",
        "(False, 3)",
        "(False, 1)",  # sys.exit("a message")
    ]
