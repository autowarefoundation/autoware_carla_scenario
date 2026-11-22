"""``scenario``'s command line (#2333): ``--help`` names the runner's own
flags, and Ctrl-C prints one line and still dies by ``SIGINT``."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from autoware_carla_scenario.examples import run

#: ``scenario`` with the arguments after ``-c``'s, as its entry point runs.
_SCENARIO = (
    "import sys\n"
    "from autoware_carla_scenario.examples import run\n"
    "sys.argv = ['scenario', *sys.argv[1:]]\n"
    "run.main()\n"
)

#: ``scenario`` whose run is stopped by Ctrl-C, with clean-up that prints.
_INTERRUPTED = """\
import atexit
from autoware_carla_scenario.examples import run

atexit.register(lambda: print("atexit ran", flush=True))


def interrupted() -> None:
    try:
        raise KeyboardInterrupt
    finally:
        print("finally ran", flush=True)


run._main = interrupted
run.main()
"""


def _python(tmp_path: Path, code: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "SWEEP_RESUME_FROM"}
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env=env,
        check=False,
    )


def test_help_names_the_runner_s_own_flags(tmp_path: Path) -> None:
    proc = _python(tmp_path, _SCENARIO, "--help")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert out.startswith("scenario: run a scenario")
    # Hydra's help is all there: the groups and the composed config.
    assert "== Configuration groups ==" in out
    assert "== Config ==" in out
    assert "== Runner flags ==" in out
    for flag in ("--resume-from N", "--resume-into DIR", "--no-wait"):
        assert f"\n  {flag} " in out, flag
    assert "Use --hydra-help to view Hydra specific help" in out
    assert not (tmp_path / "outputs").exists()


def test_hydra_help_is_still_hydra_s(tmp_path: Path) -> None:
    proc = _python(tmp_path, _SCENARIO, "--hydra-help")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("Hydra (")
    assert "--hydra-help : Hydra's help" in proc.stdout
    assert "== Runner flags ==" not in proc.stdout


def test_ctrl_c_prints_one_line_and_dies_by_sigint(tmp_path: Path) -> None:
    proc = _python(tmp_path, _INTERRUPTED)
    # A death by SIGINT, as Python's own for an uncaught Ctrl-C: a shell
    # reports 130, a parent sees the signal.
    assert proc.returncode == -signal.SIGINT, proc.stderr
    assert proc.stderr == "scenario: interrupted\n"
    # The clean-up ran first: the run's finally blocks, then atexit's.
    assert proc.stdout == "finally ran\natexit ran\n"


def test_ctrl_c_is_re_raised_with_only_its_traceback_silenced(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    seen: list[BaseException] = []

    def previous(kind: object, value: BaseException, tb: object) -> None:
        seen.append(value)

    monkeypatch.setattr(run, "_main", interrupted)
    monkeypatch.setattr(sys, "excepthook", previous)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            run.main()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    assert capsys.readouterr().err == "scenario: interrupted\n"
    sys.excepthook(KeyboardInterrupt, raised.value, None)
    assert seen == []
    other = RuntimeError("boom")
    sys.excepthook(RuntimeError, other, None)
    assert seen == [other]
