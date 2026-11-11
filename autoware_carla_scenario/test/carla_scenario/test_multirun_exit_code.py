"""A ``--multirun`` whose cases fail exits 1, one whose cases all pass exits 0.

As documented in docs/usage.md ("Exit Codes"), and as a glob batch
(``run_batch``) already does (#689).  Each sweep runs end to end in a
subprocess: the lanelet sweeper forks its cases, and Hydra turns what the
sweep raises into the process's exit status.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

#: A minimal ``@hydra.main`` app swept by LaneletConstraintSweeper, with the
#: map loading and the expansion stubbed to two cases (none with
#: ``+sweep.none=true``; a ``ValueError`` with ``+sweep.invalid=true``, a
#: ``RuntimeError`` with ``+sweep.broken=true``).  Its task fails when ``+fail=true`` is given, writing
#: the result JSON a scenario writes; as ``examples/run.py:_hydra_main`` does in
#: ``--multirun``, it neither raises nor ``sys.exit``s for a failed scenario.
#: With ``+raise_once=true`` each case raises the first time it runs, as a run
#: whose CARLA went away does.  With ``+raise_value=true`` every case raises a
#: ``ValueError``.  With ``+pass_then_raise=true`` each case, as
#: ``ScenarioRunner`` does, names its outputs after the first stem none has
#: taken (``App``, ``App-1`` ...); the first time it runs it writes a passing
#: result and raises after it (as a teardown that raises does), and the retry
#: fails.  With ``+hang=true`` every case sleeps past any job timeout, with
#: ``+die=true`` every case kills itself with ``SIGKILL``, and with
#: ``+kill_sweep=true`` the first case kills the sweep (its parent) with
#: ``SIGKILL``, as the OOM killer would.  With ``+gated=true`` each case, as
#: ``ScenarioRunner`` does, takes the first free stem and writes its coverage
#: file, its gate record (``gate_pass.json`` in the working directory, its
#: ``STEM`` replaced by the stem) and a passing result under it.  With
#: ``+service_logs_once=true`` each case appends to the logs services write
#: into its directory, takes the first free stem and writes a CARLA recorder
#: log and a result under it, and raises the first time it runs.  With
#: ``+unremovable=true`` the sweeper cannot remove a run's outputs.
_SWEEPER_APP = """
import json
import os
import sys
from pathlib import Path

from autoware_carla_scenario import hydra_compat  # noqa: F401

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import autoware_carla_scenario.sweeper.lanelet_constraint_sweeper as lcs
from autoware_carla_scenario.run_outputs import run_outputs_exist

if "+unremovable=true" in sys.argv:
    def remove_run_outputs(directory, stem):
        raise PermissionError(f"cannot remove {stem}")

    lcs.remove_run_outputs = remove_run_outputs

lcs.check_constraint_keys = lambda sweep: None
lcs.load_sweep_map = lambda map_cfg, **_: object()
lcs.scenario_controls = lambda cfg: None


def expand_sweep(sweep, lanelet_map, arguments, **_):
    if sweep.get("invalid"):
        # What expand_sweep raises for a sweep config it refuses.
        raise ValueError("OddSampler: ODD default has no module urban")
    if sweep.get("broken"):
        raise RuntimeError("an unexpected expansion bug")
    if sweep.get("none"):
        return []
    return [["+case=1", *arguments], ["+case=2", *arguments]]


lcs.expand_sweep = expand_sweep


@hydra.main(version_base=None, config_path=None, config_name=None)
def task(cfg: DictConfig) -> None:
    print(f"case {cfg.case}", flush=True)
    owns = os.environ.get("AUTOWARE_CARLA_SCENARIO_SWEEPER_OWNS_RETRIES")
    print(f"queue retries owned by the sweeper: {owns}", flush=True)
    raised = Path.cwd() / f"raised{cfg.case}"
    if cfg.get("raise_once", False) and not raised.exists():
        raised.write_text("yes")
        raise RuntimeError("CARLA went away")
    if cfg.get("pass_then_raise", False):
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        stem, n = "App", 0
        while run_outputs_exist(output_dir, stem):
            n += 1
            stem = f"App-{n}"
        first = not raised.exists()
        result = {"passed": first, "message": stem, "elapsed_seconds": 1.0}
        (output_dir / f"{stem}_result.json").write_text(json.dumps(result))
        (output_dir / f"{stem}_coverage.json").write_text("{}")
        (output_dir / f"{stem}.trajectory.jsonl").write_text("")
        if first:
            raised.write_text("yes")
            raise RuntimeError("the world reset after the result raised")
        return
    if cfg.get("service_logs_once", False):
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        for name in (
            "backend_sumo.log",
            "autoware-000-ego.log",
            "autoware-build.log",
            "raw_output.log",
        ):
            with (output_dir / name).open("a") as log:
                log.write("attempt\\n")
        stem, n = "App", 0
        while run_outputs_exist(output_dir, stem):
            n += 1
            stem = f"App-{n}"
        (output_dir / f"{stem}.log").write_text("")
        result = {"passed": True, "message": stem, "elapsed_seconds": 1.0}
        (output_dir / f"{stem}_result.json").write_text(json.dumps(result))
        if not raised.exists():
            raised.write_text("yes")
            raise RuntimeError("CARLA went away")
        return
    if cfg.get("gated", False):
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        stem, n = "App", 0
        while run_outputs_exist(output_dir, stem):
            n += 1
            stem = f"App-{n}"
        record = (Path.cwd() / "gate_pass.json").read_text()
        (output_dir / f"{stem}_coverage.json").write_text("{}")
        (output_dir / f"{stem}_gate.json").write_text(record.replace("STEM", stem))
        result = {"passed": True, "message": stem, "elapsed_seconds": 1.0}
        (output_dir / f"{stem}_result.json").write_text(json.dumps(result))
        return
    if cfg.get("raise_value", False):
        raise ValueError("some distinctive text")
    if cfg.get("hang", False):
        import time
        time.sleep(120)
    if cfg.get("die", False):
        import signal
        os.kill(os.getpid(), signal.SIGKILL)
    if cfg.get("kill_sweep", False):
        import signal
        os.kill(os.getppid(), signal.SIGKILL)
        os._exit(1)
    if cfg.get("refuse", False):
        # What examples/run.py:_hydra_main does for a refused scenario.
        raise SystemExit(2)
    passed = not cfg.get("fail", False)
    # Where ScenarioRunner writes it: the sweeper reads the verdict there.
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    (output_dir / "App_result.json").write_text(json.dumps({"passed": passed}))


if __name__ == "__main__":
    task()
"""

#: ``scenario --multirun`` through Hydra's basic sweeper, with the run of a
#: scenario faked: each case passes or fails as its ``+verdict`` says.
_BASIC_APP = """
import sys
from types import SimpleNamespace

from autoware_carla_scenario.examples import run


def fake_run_scenario(cfg, **_):
    if str(cfg.verdict) == "raise":
        print("case verdict=raise", flush=True)
        raise RuntimeError("CARLA went away")
    if str(cfg.verdict) == "refuse":
        print("case verdict=refuse", flush=True)
        raise run.ScenarioTypeError("does not type-check")
    if str(cfg.verdict) == "kill":
        # The sweep dies with its case: CARLA segfaulting it, the OOM killer.
        import os
        import signal
        os.kill(os.getpid(), signal.SIGKILL)
    passed = str(cfg.verdict) == "pass"
    print(f"case verdict={cfg.verdict}", flush=True)
    return SimpleNamespace(passed=passed)


run.run_scenario = fake_run_scenario
sys.argv = ["scenario", *sys.argv[1:]]
run.main()
"""


def _run(
    app: str,
    tmp_path: Path,
    *args: str,
    resume_from: int | str | None = None,
    mode: str = "--multirun",
    sweep_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "app.py"
    script.write_text(textwrap.dedent(app), encoding="utf-8")
    # A resume point would skip the cases.
    env = {k: v for k, v in os.environ.items() if k != "SWEEP_RESUME_FROM"}
    if resume_from is not None:
        # What `scenario --resume-from N` passes the sweeper.
        env["SWEEP_RESUME_FROM"] = str(resume_from)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            mode,
            *args,
            f"hydra.sweep.dir={sweep_dir or tmp_path / 'multirun'}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env=env,
        check=False,
    )


@pytest.mark.parametrize(
    ("fail", "summary", "returncode"),
    [
        ("true", "0/2 succeeded, 2 failed", 1),
        ("false", "2/2 succeeded, 0 failed", 0),
    ],
)
def test_lanelet_sweeper_exit_code(
    tmp_path: Path, fail: str, summary: str, returncode: int
) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        f"+fail={fail}",
    )
    output = proc.stdout + proc.stderr
    assert summary in output, output
    assert proc.returncode == returncode, output
    if returncode:
        assert "2 of 2 case(s) failed" in output, output


def test_lanelet_sweeper_does_not_retry_a_case_that_ran_and_failed(
    tmp_path: Path,
) -> None:
    """A FAILED verdict is final, as in ScenarioQueue: only a raise is retried (#709)."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=2",
        "+fail=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert output.count("case 1") == 1, output
    assert output.count("case 2") == 1, output
    assert "0/2 succeeded, 2 failed" in output, output


def test_lanelet_sweeper_retries_a_case_that_raised(tmp_path: Path) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=1",
        "+raise_once=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert output.count("case 1") == 2, output
    assert output.count("case 2") == 2, output
    assert "2/2 succeeded, 0 failed" in output, output


def test_a_retried_case_keeps_none_of_the_failed_attempts_outputs(
    tmp_path: Path,
) -> None:
    """Before a retry the sweeper removes what the failed attempt wrote in the
    job's directory, as the queue and the SLURM worker do (#1373).

    The first attempt wrote a passing result, then raised; the retry failed.
    The viewer, which shows a job's first stem, and the coverage merge, which
    reads every coverage file, must read the retry's alone, as the sweeper's
    verdict does.
    """
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _read_verdict,
    )
    from autoware_carla_scenario.ui import results, scanner

    date, time = "2026-11-04", "10-00-00"
    session = tmp_path / "multirun" / date / time
    # An earlier sweep's run in a directory named again is replaced (#1956);
    # a file that is not a run's output is not touched.
    (session / "1").mkdir(parents=True)
    (session / "1" / "AnEarlierRun_result.json").write_text('{"passed": true}')
    (session / "1" / "notes.txt").write_text("mine")
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=1",
        "+pass_then_raise=true",
        sweep_dir=session,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert output.count("case 1") == 2, output
    assert "0/2 succeeded, 2 failed" in output, output

    job = session / "0"
    assert sorted(p.name for p in job.iterdir() if p.name != ".hydra") == [
        "App.trajectory.jsonl",
        "App_coverage.json",
        "App_result.json",
        "app.log",  # Hydra's job log, not a CARLA recorder log
    ]
    assert _read_verdict(job) is False
    assert sorted(p.name for p in (session / "1").glob("*_result.json")) == [
        "App_result.json",
    ]
    assert (session / "1" / "notes.txt").read_text() == "mine"
    assert len(list(session.glob("*/*_coverage.json"))) == 2  # one per case

    # Job 0, whose directory held nothing before: the viewer reads the retry.
    scanner.clear_cache()
    try:
        items = scanner.load_session(tmp_path, "multirun", date, time)
        loaded = results.load_results(tmp_path, "multirun", date, time, metrics=False)
    finally:
        scanner.clear_cache()
    assert (items[0].passed, items[0].message) == (False, "App"), items
    assert (loaded[0].passed, loaded[0].message) == (False, "App"), loaded


@pytest.mark.parametrize(
    ("content", "verdict"),
    [
        ('{"passed": true}', True),
        ('{"passed": false}', False),
        ("{}", None),
        ('{"passed": "false"}', None),
        ('{"passed": 1}', None),
        ('{"passed": null}', None),
        ("[true]", None),
        ("1", None),
        ("not json", None),
    ],
)
def test_read_verdict_takes_only_a_boolean_passed(
    tmp_path: Path, content: str, verdict: bool | None
) -> None:
    """A result without a boolean ``passed`` is no verdict, so the case fails.

    ``"passed": "false"`` is truthy and a missing key used to read as a pass
    (#1711); the viewer and the gate read both as not passed.
    """
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _read_verdict,
    )

    (tmp_path / "App_result.json").write_text(content, encoding="utf-8")
    assert _read_verdict(tmp_path) is verdict


def test_the_stems_a_retry_removes_are_those_of_run_outputs(tmp_path: Path) -> None:
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _run_output_stems,
    )

    for name in (
        "X_result.json",
        "Y.sensors.rrd",  # a recording left half done: stem Y, not Y.sensors
        "Z.log",  # a CARLA recorder log
        "app.log",  # Hydra's job log
        "notes.txt",
    ):
        (tmp_path / name).write_text("")
    (tmp_path / "W.t4").mkdir()
    (tmp_path / ".hydra").mkdir()
    assert _run_output_stems(tmp_path, "app.log") == {"W", "X", "Y", "Z"}
    assert _run_output_stems(tmp_path / "missing", None) == set()


def test_lanelet_sweeper_turns_off_the_retries_of_its_cases_queue(
    tmp_path: Path,
) -> None:
    """The sweeper retries a case; its queue must not too, or retries stack."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=2",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert output.count("queue retries owned by the sweeper: 1") == 2, output


def test_lanelet_sweeper_that_raised_on_every_attempt_fails(tmp_path: Path) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+raise_once=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert "0/2 succeeded, 2 failed" in output, output


def test_lanelet_sweeper_logs_why_a_case_raised(tmp_path: Path) -> None:
    """Hydra stores what a case raised without logging it; the sweeper logs it (#1232).

    With its traceback, on the terminal and in the job's log file.
    """
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+raise_value=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert "0/2 succeeded, 2 failed" in output, output
    assert "ValueError: some distinctive text" in output, output
    # Once per case, not again by the child's own exception handler.
    assert output.count("Traceback (most recent call last)") == 2, output
    assert "Job raised an exception in child process" not in output, output
    for job in ("0", "1"):
        job_log = (tmp_path / "multirun" / job / "app.log").read_text(encoding="utf-8")
        assert "ValueError: some distinctive text" in job_log, job_log
        assert "Traceback (most recent call last)" in job_log, job_log


@pytest.mark.parametrize(
    ("return_value", "hydra_cfg", "configure_log_raises"),
    [
        (ValueError("boom"), None, False),  # no Hydra config: logged as is
        ("not an exception", None, False),
        (ValueError("boom"), {"hydra": {"job_logging": {}}}, True),
    ],
)
def test_logging_what_a_case_raised_never_changes_its_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    return_value: Any,
    hydra_cfg: Any,
    configure_log_raises: bool,
) -> None:
    """Whatever logging the exception does, a FAILED case exits 1."""
    from hydra.core.utils import JobReturn, JobStatus
    from omegaconf import OmegaConf

    import autoware_carla_scenario.sweeper.lanelet_constraint_sweeper as lcs

    def _raise(*_: Any) -> None:
        raise RuntimeError("configure_log failed")

    if configure_log_raises:
        monkeypatch.setattr(lcs, "configure_log", _raise)
    ret = JobReturn(
        overrides=["+case=1"],
        hydra_cfg=None if hydra_cfg is None else OmegaConf.create(hydra_cfg),
        status=JobStatus.FAILED,
    )
    ret.return_value = return_value
    assert lcs._job_exit_code([ret]) == 1


@pytest.mark.parametrize(
    ("args", "resume_from", "message"),
    [
        (("+sweep.none=true",), None, "expands to no case"),
        ((), 3, "--resume-from 3 is past the sweep's 2 case(s)"),
        ((), 9999, "--resume-from 9999 is past the sweep's 2 case(s)"),
    ],
)
def test_lanelet_sweeper_that_runs_no_case_exits_1(
    tmp_path: Path, args: tuple[str, ...], resume_from: int | None, message: str
) -> None:
    """As a glob batch with no case left to run does (#710)."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        *args,
        resume_from=resume_from,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert message in output, output
    assert "case 1" not in output, output


def test_lanelet_sweeper_resumes_at_its_last_case(tmp_path: Path) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        resume_from=2,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "case 1" not in output, output
    assert "case 2" in output, output


@pytest.mark.parametrize(
    ("config", "env", "first_case"),
    [
        # --resume-from (the variable) wins over sweep.resume_from (#1784).
        ("1", 2, "case 2"),
        ("2", 1, "case 1"),
        # The config alone.
        ("2", None, "case 2"),
        # 0 and null in the config: not set.
        ("0", None, "case 1"),
        ("null", None, "case 1"),
        # A variable of 0 or empty is not set either: the config stands.
        ("2", "0", "case 2"),
        ("2", "", "case 2"),
    ],
)
def test_lanelet_sweeper_resume_point_precedence(
    tmp_path: Path, config: str, env: int | str | None, first_case: str
) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        f"+sweep.resume_from={config}",
        resume_from=env,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert first_case in output, output
    if first_case == "case 2":
        assert "case 1" not in output, output


@pytest.mark.parametrize(
    ("config", "env", "message"),
    [
        ("true", None, "Error: sweep.resume_from must be a whole number of at least 1"),
        ("2.7", None, "Error: sweep.resume_from must be a whole number of at least 1"),
        ("3.0", None, "Error: sweep.resume_from must be a whole number of at least 1"),
        ("abc", None, "Error: sweep.resume_from must be a whole number of at least 1"),
        ("-3", None, "Error: sweep.resume_from must be a whole number of at least 1"),
        # Refused even when --resume-from overrides it.
        ("abc", 2, "Error: sweep.resume_from must be a whole number of at least 1"),
        # The variable is checked as strictly.
        (
            "null",
            "abc",
            "Error: SWEEP_RESUME_FROM must be a whole number of at least 1",
        ),
        ("null", "-1", "Error: SWEEP_RESUME_FROM must be a whole number of at least 1"),
        (
            "null",
            "2.5",
            "Error: SWEEP_RESUME_FROM must be a whole number of at least 1",
        ),
    ],
)
def test_lanelet_sweeper_refuses_a_bad_resume_point_before_any_case(
    tmp_path: Path, config: str, env: int | str | None, message: str
) -> None:
    """#1784: one line and exit 2, not a truncation, a run of every case or
    int()'s traceback."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        f"+sweep.resume_from={config}",
        resume_from=env,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2, output
    assert message in output, output
    assert "case 1" not in output, output
    assert "case 2" not in output, output


@pytest.mark.parametrize(
    ("config", "env", "expected"),
    [
        (None, None, 0),
        (0, None, 0),
        (3, None, 3),
        (None, "4", 4),
        (None, " 4 ", 4),
        (None, "", 0),
        (None, "0", 0),
        (5, "10", 10),
        (10, "5", 5),
        (5, "0", 5),
        (5, "", 5),
    ],
)
def test_resume_from_reads_the_variable_over_the_config(
    config: Any, env: str | None, expected: int
) -> None:
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _resume_from,
    )

    assert _resume_from(config, env) == expected


@pytest.mark.parametrize(
    ("config", "env", "match"),
    [
        (True, None, "sweep.resume_from must be a whole number"),
        (False, None, "sweep.resume_from must be a whole number"),
        (2.7, None, "sweep.resume_from must be a whole number"),
        (3.0, None, "sweep.resume_from must be a whole number"),
        ("3", None, "sweep.resume_from must be a whole number"),
        ("abc", None, "sweep.resume_from must be a whole number"),
        (-3, None, "sweep.resume_from must be a whole number"),
        ([1], None, "sweep.resume_from must be a whole number"),
        # A bad config value is refused even when the variable wins.
        (-3, "2", "sweep.resume_from must be a whole number"),
        (None, "abc", "SWEEP_RESUME_FROM must be a whole number"),
        (None, "-1", "SWEEP_RESUME_FROM must be a whole number"),
        (None, "+1", "SWEEP_RESUME_FROM must be a whole number"),
        (None, "2.5", "SWEEP_RESUME_FROM must be a whole number"),
        (None, "\u00b2", "SWEEP_RESUME_FROM must be a whole number"),
    ],
)
def test_resume_from_refuses_what_is_not_a_whole_number(
    config: Any, env: str | None, match: str
) -> None:
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _resume_from,
    )

    with pytest.raises(ValueError, match=match):
        _resume_from(config, env)


@pytest.mark.parametrize(
    ("verdicts", "returncode"),
    [("pass,fail,pass", 1), ("pass,pass", 0)],
)
def test_basic_sweeper_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdicts: str, returncode: int
) -> None:
    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    proc = _run(_BASIC_APP, tmp_path, f"+verdict={verdicts}")
    output = proc.stdout + proc.stderr
    # Every case ran: a failed one does not end the sweep.
    assert output.count("case verdict=") == len(verdicts.split(",")), output
    assert proc.returncode == returncode, output
    if returncode:
        assert "1 of 3 case(s) failed" in output, output


def test_lanelet_sweeper_stops_with_2_on_a_refused_case(tmp_path: Path) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+refuse=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2, output
    # The second case would be refused too: it does not run.
    assert "case 1" in output, output
    assert "case 2" not in output, output
    assert "REFUSED" in output, output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ("+t4scene.enabled=true", "+t4scene.step_s=0.07"),
            "t4scene.step_s",
        ),
        (("+server.world_reset=bogus",), "server.world_reset"),
        (("+scenario.timeout_seconds=-1",), "scenario.timeout_seconds"),
        (("+server.cooldown_max_retries=-1",), "server.cooldown_max_retries"),
    ],
)
def test_lanelet_sweeper_refuses_a_bad_config_value_before_any_case(
    tmp_path: Path, args: tuple[str, ...], message: str
) -> None:
    """#1233: a value every case's run would refuse is refused once, with 2,
    before any case is forked -- not by every case in turn, each retried."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        *args,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2, output
    assert f"Error: {message}" in output, output
    assert "case 1" not in output, output
    assert "Launching" not in output, output
    assert output.count(message) == 1, output


#: The stubs of :data:`_SWEEPER_APP` that stand in for the sweeper's checks
#: before it expands the sweep: without them the real ones run.
_PRE_EXPANSION_STUBS = (
    "lcs.check_constraint_keys = lambda sweep: None\n",
    "lcs.load_sweep_map = lambda map_cfg, **_: object()\n",
)


def _scenario_app(tmp_path: Path, *, real_checks: bool = False) -> str:
    """:data:`_SWEEPER_APP` with a ``scenario`` config group holding
    ``cut_in/left``, so ``+scenario=cut_in/left`` is the sweep's scenario
    choice, as ``scenario=cut_in/left`` is a real sweep's.  With
    *real_checks*, the sweep's constraint keys are checked and its map
    loaded by the sweeper's own functions, not stubs."""
    group = tmp_path / "conf" / "scenario" / "cut_in"
    group.mkdir(parents=True)
    (group / "left.yaml").write_text("{}\n", encoding="utf-8")
    old = "config_path=None"
    assert old in _SWEEPER_APP
    app = _SWEEPER_APP.replace(old, 'config_path="conf"')
    if real_checks:
        for stub in _PRE_EXPANSION_STUBS:
            assert stub in app
            app = app.replace(stub, "")
    return app


def test_lanelet_sweeper_refuses_an_invalid_sweep_with_one_line(
    tmp_path: Path,
) -> None:
    """#1705: a sweep config expand_sweep refuses prints one
    ``Error: <scenario>: <message>`` line, as scenario-expand does, and exits
    1 with no traceback and before any case runs."""
    proc = _run(
        _scenario_app(tmp_path),
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+scenario=cut_in/left",
        "+sweep.constraints={}",
        "+sweep.invalid=true",
    )
    output = proc.stdout + proc.stderr
    line = "Error: cut_in/left: OddSampler: ODD default has no module urban"
    assert proc.returncode == 1, output
    assert proc.stderr.splitlines() == [line], output
    assert "case 1" not in output, output
    assert "Launching" not in output, output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ("+sweep.constraints={a: [{}], b: [{}]}",),
            "sweep.constraints sweeps one parameter; got keys ['a', 'b']",
        ),
        (
            (
                "+sweep.constraints={ego.spawn_lanelet_id: [{}]}",
                "+map.directory=/nonexistent/no_map",
            ),
            "LaneletConstraintSweeper: Map directory /nonexistent/no_map "
            "does not exist",
        ),
    ],
    ids=["second-constraint-key", "missing-map"],
)
def test_lanelet_sweeper_refuses_a_sweep_before_expanding_it_with_one_line(
    tmp_path: Path, args: tuple[str, ...], message: str
) -> None:
    """#1705: the sweeper's own checks before it expands the sweep (the
    constraint keys, the map) refuse it in one line too, exit 1."""
    proc = _run(
        _scenario_app(tmp_path, real_checks=True),
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+scenario=cut_in/left",
        *args,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    lines = proc.stderr.splitlines()
    assert len(lines) == 1, output
    assert lines[0].startswith(f"Error: cut_in/left: {message}"), output
    assert "Launching" not in output, output


def test_lanelet_sweeper_refuses_an_invalid_sweep_without_a_scenario(
    tmp_path: Path,
) -> None:
    """Without a ``scenario`` choice the line names none."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+sweep.invalid=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert proc.stderr.splitlines() == [
        "Error: OddSampler: ODD default has no module urban"
    ], output


def test_lanelet_sweeper_keeps_the_traceback_of_an_unexpected_error(
    tmp_path: Path,
) -> None:
    """Only a ValueError is a refusal: any other error is a bug, shown with
    its traceback as Hydra shows it."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+sweep.broken=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert "Traceback" in output, output
    assert "RuntimeError: an unexpected expansion bug" in output, output
    assert "\nError: an unexpected" not in "\n" + output, output


@pytest.mark.parametrize(
    "value", ["0.5", "-1", "nan", "inf", "90.5", "abc", "1e10", "2147483648"]
)
def test_lanelet_sweeper_refuses_a_bad_job_timeout_before_any_case(
    tmp_path: Path, value: str
) -> None:
    """#1418: 0.5 truncated to 0 (no timeout), a negative one disabled it
    silently; each is refused before any case is forked."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        f"+sweep.job_timeout_seconds={value}",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2, output
    assert "Error: sweep.job_timeout_seconds must be a whole number" in output, output
    assert "Launching" not in output, output
    assert "succeeded" not in output, output


@pytest.mark.parametrize("value", ["null", "0", "30", "30.0"])
def test_lanelet_sweeper_runs_with_a_valid_or_null_job_timeout(
    tmp_path: Path, value: str
) -> None:
    """#1418: ``null`` is the default (it raised a TypeError), 0 no timeout."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        f"+sweep.job_timeout_seconds={value}",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "2/2 succeeded, 0 failed" in output, output


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 120), (0, 0), (30, 30), (30.0, 30), ("45", 45), (2**31 - 1, 2**31 - 1)],
)
def test_job_timeout_seconds_reads_a_whole_number_or_the_default(
    value: Any, expected: int
) -> None:
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _DEFAULT_JOB_TIMEOUT,
        _job_timeout_seconds,
    )

    assert _DEFAULT_JOB_TIMEOUT == 120
    assert _job_timeout_seconds(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        *(0.5, 90.5, -1, -0.5, float("nan"), float("inf"), True, "abc", [1], 2**31),
        1e10,
        pytest.param(10**400, id="huge"),  # too big for a float (#1652)
    ],
)
def test_job_timeout_seconds_refuses_what_is_not_a_whole_number_of_seconds(
    value: Any,
) -> None:
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _job_timeout_seconds,
    )

    with pytest.raises(ValueError, match="job_timeout_seconds must be a whole"):
        _job_timeout_seconds(value)


def test_lanelet_sweeper_runs_a_sweep_whose_config_values_are_valid(
    tmp_path: Path,
) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+t4scene.enabled=true",
        "+t4scene.step_s=0.1",
        "+scenario.timeout_seconds=30",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "2/2 succeeded, 0 failed" in output, output


@pytest.mark.parametrize("forked", [False, True])
def test_a_case_refused_for_a_config_value_exits_2_only_in_a_sweeper(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], forked: bool
) -> None:
    """The safety net: a value only a case's overrides set ends the sweep at
    that case (2), and is an ordinary config error (1) anywhere else."""
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.sweeper.isolation import SWEEPER_OWNS_RETRIES_ENV

    from .test_video_removed import _shipped

    if forked:
        monkeypatch.setenv(SWEEPER_OWNS_RETRIES_ENV, "1")
    else:
        monkeypatch.delenv(SWEEPER_OWNS_RETRIES_ENV, raising=False)
    cfg = _shipped(["t4scene.enabled=true", "t4scene.step_s=0.07"])
    monkeypatch.setattr(
        run, "build_scenario", lambda *_, **__: pytest.fail("built the scenario")
    )
    hydra_main = run._hydra_main.__wrapped__  # type: ignore[attr-defined]
    if forked:
        with pytest.raises(SystemExit) as exited:
            hydra_main(cfg)
        assert exited.value.code == 2
        assert "Error: t4scene.step_s" in capsys.readouterr().err
    else:
        with pytest.raises(run.RunConfigError, match="t4scene.step_s"):
            hydra_main(cfg)


@pytest.mark.parametrize(
    ("verdicts", "returncode"),
    [("pass,fail,pass", 1), ("pass,pass", 0)],
)
def test_a_sweep_set_by_hydra_mode_runs_every_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdicts: str, returncode: int
) -> None:
    """``hydra.mode=MULTIRUN`` sweeps as ``--multirun`` does: the first case
    does not end the sweep with its own verdict (#1034)."""
    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    proc = _run(
        _BASIC_APP, tmp_path, f"+verdict={verdicts}", mode="hydra.mode=MULTIRUN"
    )
    output = proc.stdout + proc.stderr
    assert output.count("case verdict=") == len(verdicts.split(",")), output
    assert proc.returncode == returncode, output


@pytest.mark.parametrize(
    ("argv", "multirun"),
    [
        (["--multirun"], True),
        (["-m"], True),
        (["hydra.mode=MULTIRUN"], True),
        (["++hydra.mode=MULTIRUN"], True),
        (["hydra.mode=RUN"], False),
        # Hydra refuses it: not a multirun.
        (["hydra.mode=multirun"], False),
        (["seed=1,2"], False),
    ],
)
def test_multirun_requested_in_argv(argv: list[str], multirun: bool) -> None:
    from autoware_carla_scenario.examples import run

    assert run._multirun_requested_in_argv(["scenario", *argv]) is multirun


@pytest.mark.parametrize("mode", ["RUN", "MULTIRUN"])
def test_is_multirun_inside_a_hydra_job_is_the_jobs_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """In a job, the mode Hydra settled on wins over argv: a config may set it."""
    from types import SimpleNamespace

    from hydra.types import RunMode

    from autoware_carla_scenario.examples import run

    monkeypatch.setattr(run.HydraConfig, "initialized", staticmethod(lambda: True))
    monkeypatch.setattr(
        run.HydraConfig,
        "get",
        staticmethod(lambda: SimpleNamespace(mode=RunMode[mode])),
    )
    monkeypatch.setattr(sys, "argv", ["scenario", "seed=1,2"])
    assert run._is_multirun() is (mode == "MULTIRUN")


def test_is_multirun_outside_a_hydra_job_reads_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.examples import run

    monkeypatch.setattr(run.HydraConfig, "initialized", staticmethod(lambda: False))
    monkeypatch.setattr(sys, "argv", ["scenario", "hydra.mode=MULTIRUN"])
    assert run._is_multirun()
    monkeypatch.setattr(sys, "argv", ["scenario"])
    assert not run._is_multirun()


def test_basic_sweeper_stops_with_2_on_a_refused_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    proc = _run(_BASIC_APP, tmp_path, "+verdict=refuse,pass")
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2, output
    assert "does not type-check" in output, output
    assert output.count("case verdict=") == 1, output


def test_main_does_not_exit_for_an_earlier_calls_failed_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The verdicts recorded are those of this main() call alone."""
    from autoware_carla_scenario.examples import run

    # main() reserves the sweep's multirun/ directory here (#1359).
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(run, "_multirun_failed", ["+verdict=fail"])
    monkeypatch.setattr(run, "_multirun_ran", ["+verdict=fail"])
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: False)
    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_hydra_main", lambda: None)
    monkeypatch.setattr(sys, "argv", ["scenario", "--multirun", "+verdict=pass"])
    run.main()  # no SystemExit


def _stub_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hydra_main: Any) -> Any:
    """``examples.run`` with main()'s single-run path calling *hydra_main*,
    in *tmp_path*, where main() reserves the run's directory (#1359)."""
    from autoware_carla_scenario.examples import run

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(run, "_multirun_failed", [])
    monkeypatch.setattr(run, "_multirun_ran", [])
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: False)
    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_hydra_main", hydra_main)
    return run


def test_resume_from_is_this_main_calls_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--resume-from reaches the sweeper of its own run, not a later one's (#797)."""
    seen: list[str | None] = []
    run = _stub_main(
        monkeypatch,
        tmp_path,
        lambda: seen.append(os.environ.get("SWEEP_RESUME_FROM")),
    )
    monkeypatch.delenv("SWEEP_RESUME_FROM", raising=False)
    argv = ["scenario", "--multirun", "scenario=a/b"]
    monkeypatch.setattr(sys, "argv", [*argv, "--resume-from", "3"])
    run.main()
    monkeypatch.setattr(sys, "argv", list(argv))
    run.main()
    assert seen == ["3", None]
    assert "SWEEP_RESUME_FROM" not in os.environ


def test_resume_from_puts_back_the_value_it_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A value the user set stands for the runs without the flag, and comes
    back after one with it, even one that exits."""

    def hydra_main() -> None:
        seen.append(os.environ.get("SWEEP_RESUME_FROM"))
        raise SystemExit(1)

    seen: list[str | None] = []
    run = _stub_main(monkeypatch, tmp_path, hydra_main)
    monkeypatch.setenv("SWEEP_RESUME_FROM", "5")
    argv = ["scenario", "--multirun", "scenario=a/b"]
    monkeypatch.setattr(sys, "argv", [*argv, "--resume-from", "3"])
    with pytest.raises(SystemExit):
        run.main()
    assert os.environ["SWEEP_RESUME_FROM"] == "5"
    monkeypatch.setattr(sys, "argv", list(argv))
    with pytest.raises(SystemExit):
        run.main()
    assert seen == ["3", "5"]


@pytest.mark.parametrize(("owned", "retries"), [(False, 2), (True, 0)])
def test_a_case_forked_by_the_sweeper_runs_its_queue_without_retries(
    monkeypatch: pytest.MonkeyPatch, owned: bool, retries: int
) -> None:
    """``run_scenario``'s queue leaves a sweeper case's retries to the sweeper."""
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.sweeper.isolation import SWEEPER_OWNS_RETRIES_ENV

    if owned:
        monkeypatch.setenv(SWEEPER_OWNS_RETRIES_ENV, "1")
    else:
        monkeypatch.delenv(SWEEPER_OWNS_RETRIES_ENV, raising=False)
    cfg = OmegaConf.create({"server": {"cooldown_max_retries": 2}})
    assert run._queue_retries(cfg) == retries


# ---------------------------------------------------------------------------
# The marker of a job given up: case_errored.json (#1830)
# ---------------------------------------------------------------------------


def _markers(root: Path) -> dict[str, dict[str, Any]]:
    """Every ``case_errored.json`` under *root*, by its job directory's name."""
    import json

    return {
        p.parent.name: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(root.rglob("case_errored.json"))
    }


@pytest.mark.parametrize(
    ("args", "exit_code", "reason"),
    [
        (("+raise_value=true",), 1, "exit code 1"),
        (("+die=true",), -9, "crashed (SIGKILL)"),
        (
            ("+hang=true", "+sweep.job_timeout_seconds=1"),
            None,
            "timed out after 1s",
        ),
    ],
)
def test_lanelet_sweeper_marks_a_job_given_up(
    tmp_path: Path, args: tuple[str, ...], exit_code: int | None, reason: str
) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=1",
        *args,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    markers = _markers(tmp_path / "multirun")
    assert sorted(markers) == ["0", "1"], output
    for job, marker in markers.items():
        assert marker["schema"] == "autoware_carla_scenario.case_errored/1"
        assert marker["exit_code"] == exit_code
        assert marker["attempts"] == 2
        assert marker["reason"] == reason
        assert f"+case={int(job) + 1}" in marker["overrides"]


def test_lanelet_sweeper_marks_no_job_that_failed_or_a_retry_passed(
    tmp_path: Path,
) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+fail=true",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    # A failed verdict wrote its record: it is judged, not marked.
    assert _markers(tmp_path / "multirun") == {}

    retried = tmp_path / "retried"
    retried.mkdir()
    proc = _run(
        _SWEEPER_APP,
        retried,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=1",
        "+raise_once=true",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _markers(retried / "multirun") == {}


def test_lanelet_sweeper_removes_a_stale_marker_before_a_job(tmp_path: Path) -> None:
    """A sweep run again in the same directory decides each job afresh."""
    session = tmp_path / "multirun"
    for job in ("0", "1"):
        (session / job).mkdir(parents=True)
        (session / job / "case_errored.json").write_text("{}")
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        sweep_dir=session,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _markers(session) == {}


def test_a_refused_job_is_marked(tmp_path: Path) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+refuse=true",
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    markers = _markers(tmp_path / "multirun")
    assert list(markers) == ["0"]
    assert markers["0"]["exit_code"] == 2


def test_basic_sweeper_marks_a_case_that_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    proc = _run(_BASIC_APP, tmp_path, "+verdict=pass,raise")
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    markers = _markers(tmp_path / "multirun")
    assert list(markers) == ["1"], output
    assert markers["1"]["reason"] == "raised RuntimeError: CARLA went away"
    assert markers["1"]["exit_code"] == 1
    assert "+verdict=raise" in markers["1"]["overrides"]


def test_basic_sweeper_marks_no_case_that_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    session = tmp_path / "multirun"
    # A marker an earlier run of the case left is removed as it starts.
    (session / "0").mkdir(parents=True)
    (session / "0" / "case_errored.json").write_text("{}")
    proc = _run(_BASIC_APP, tmp_path, "+verdict=pass,fail")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _markers(session) == {}


# ---------------------------------------------------------------------------
# A sweep killed mid-job: the unfinished marker and the plan (#1833)
# ---------------------------------------------------------------------------


def _plan(session: Path) -> list[tuple[str, list[str]]]:
    """The jobs of the sweep plan in *session*: each directory as written,
    and its overrides."""
    import json

    data = json.loads((session / "sweep_plan.json").read_text(encoding="utf-8"))
    assert data["schema"] == "autoware_carla_scenario.sweep_plan/1"
    return [(job["dir"], job["overrides"]) for job in data["jobs"]]


def test_lanelet_sweeper_writes_its_plan_before_the_first_job(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.gate.evaluate import (
        batch_read_errors,
        unrecorded_cases,
    )

    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    session = tmp_path / "multirun"
    plan = _plan(session)
    assert [d for d, _ in plan] == ["0", "1"]
    assert "+case=1" in plan[0][1] and "+case=2" in plan[1][1]
    # A complete sweep leaves no marker, and is judged as before.
    assert _markers(session) == {}
    assert unrecorded_cases(session) == []
    assert batch_read_errors(session) == []
    # The viewer lists the sweep's jobs, not the plan beside them.
    from autoware_carla_scenario.ui import scanner

    summary = scanner._scan_multirun_session(session, "d", "t")
    assert summary is not None
    assert (summary.passed_count, summary.total_count) == (2, 2)


def test_a_sweep_refused_before_it_runs_leaves_no_plan(tmp_path: Path) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+sweep.none=true",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not (tmp_path / "multirun" / "sweep_plan.json").exists()


def test_a_killed_lanelet_sweep_leaves_its_job_marked_and_the_rest_planned(
    tmp_path: Path,
) -> None:
    """The sweep dies in its first job (SIGKILL): that job leaves the marker
    written before its attempt, and the job it never reached is in the plan;
    scenario-gate refuses both."""
    from autoware_carla_scenario.gate.evaluate import main, unrecorded_cases

    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+kill_sweep=true",
    )
    session = tmp_path / "multirun"
    assert proc.returncode == -9, proc.stdout + proc.stderr
    markers = _markers(session)
    assert list(markers) == ["0"]
    assert markers["0"]["unfinished"] is True
    assert markers["0"]["exit_code"] is None
    assert markers["0"]["attempts"] == 1
    assert markers["0"]["reason"].startswith("did not finish")
    assert "+case=1" in markers["0"]["overrides"]
    assert [d for d, _ in _plan(session)] == ["0", "1"]
    assert not (session / "1").exists()

    errored, unstarted = unrecorded_cases(session)
    assert "1 job(s) of the sweep" in errored and "0 (did not finish" in errored
    assert "1 of 2 planned job(s) of the sweep never ran" in unstarted
    assert "(1)" in unstarted
    assert main([str(session)]) == 2


def test_rerunning_a_sweep_into_its_directory_clears_its_markers_and_keeps_its_plan(
    tmp_path: Path,
) -> None:
    """The killed sweep resumed at its first job into the same directory
    finishes every job; resumed at its second, it keeps the first job in the
    plan, which it skips."""
    from autoware_carla_scenario.gate.evaluate import unrecorded_cases

    session = tmp_path / "multirun"
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+kill_sweep=true",
        sweep_dir=session,
    )
    assert proc.returncode == -9, proc.stdout + proc.stderr
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        sweep_dir=session,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _markers(session) == {}
    assert unrecorded_cases(session) == []

    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+fail=true",
        resume_from=2,
        sweep_dir=session,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    plan = _plan(session)
    assert [d for d, _ in plan] == ["0", "1"]
    # Job 0 as the earlier sweep planned it; job 1 as this one did.
    assert "+fail=true" not in plan[0][1]
    assert "+fail=true" in plan[1][1]
    # A failed verdict is judged: no marker.
    assert _markers(session) == {}
    assert unrecorded_cases(session) == []


def _gate_template(tmp_path: Path) -> None:
    """``gate_pass.json``, the passing gate record ``+gated=true`` writes, its
    stem left as ``STEM``."""
    from autoware_carla_scenario.gate.record import GateCase, GateRecord
    from autoware_carla_scenario.gate.record import write_gate_record

    record = GateRecord(
        scenario="left_turn",
        title="Left turn",
        digest="d" * 64,
        gate={"min_success_rate": 1.0},
        case=GateCase(
            map="Town",
            index=0,
            lanelet_ids=(1, 2, 3),
            case_count=2,
            segment_kinds=("lane", "junction", "lane"),
            segment_ends=(40.0, 60.0, 100.0),
        ),
        passed=True,
        message="ok",
        elapsed_seconds=12.0,
        failure_category=None,
        ego=[(i * 0.1, float(i), 0.0, 10.0) for i in range(101)],
        files={"coverage": "STEM_coverage.json", "result": "STEM_result.json"},
    )
    write_gate_record(tmp_path / "gate_pass.json", record)


def test_a_job_resumed_into_its_directory_replaces_the_earlier_runs_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1956: a sweep killed in job 1 after the job wrote its (failing) gate
    record, result and coverage file is resumed at that job into the same
    ``hydra.sweep.dir``.  The job replaces the earlier run: the gate judges
    one run a job (2 of 2, not 2 of 3), the coverage merge reads one file a
    job, and the viewer and the sweeper's verdict read the same result."""
    import json

    from autoware_carla_scenario.gate.errored import write_case_unfinished
    from autoware_carla_scenario.gate.evaluate import main
    from autoware_carla_scenario.gate.record import read_gate_record
    from autoware_carla_scenario.gate.record import write_gate_record
    from autoware_carla_scenario.gate.sweep_plan import write_sweep_plan
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _read_verdict,
    )
    from autoware_carla_scenario.ui.case_paths import last_result_path, result_path

    _gate_template(tmp_path)
    passing = read_gate_record(tmp_path / "gate_pass.json")
    session = tmp_path / "multirun"
    jobs = [(session / "0", ["+case=1"]), (session / "1", ["+case=2"])]
    write_sweep_plan(session, jobs)
    for (job, overrides), passed in zip(jobs, (True, False)):
        job.mkdir(parents=True)
        record = passing
        record.passed = passed
        record.message = "ok" if passed else "collision"
        record.failure_category = None if passed else "collision"
        record.files = {"coverage": "App_coverage.json", "result": "App_result.json"}
        write_gate_record(job / "App_gate.json", record)
        (job / "App_coverage.json").write_text("{}")
        (job / "App_result.json").write_text(json.dumps({"passed": passed}))
    # Job 1 was killed after writing them (in the world reset, say).
    write_case_unfinished(session / "1", overrides=["+case=2"], attempt=1)
    (session / "1" / "notes.txt").write_text("mine")
    assert main([str(session)]) == 1
    assert "1 of 2 run(s) passed" in capsys.readouterr().out

    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+gated=true",
        resume_from=2,
        sweep_dir=session,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    job = session / "1"
    # The job took the stem again: the earlier run's outputs went first.
    assert sorted(p.name for p in job.glob("App*")) == [
        "App_coverage.json",
        "App_gate.json",
        "App_result.json",
    ]
    assert (job / "notes.txt").read_text() == "mine"
    assert not (job / "case_errored.json").exists()
    # Job 0, which this sweep skipped, keeps its run.
    assert (session / "0" / "App_gate.json").is_file()

    assert main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "2 of 2 run(s) passed" in out, out
    assert len(list(session.glob("*/*_coverage.json"))) == 2  # one per job
    assert result_path(job) == last_result_path(job) == job / "App_result.json"
    assert _read_verdict(job) is True


def test_jobs_sharing_a_directory_keep_each_others_outputs(tmp_path: Path) -> None:
    """A directory several jobs of the sweep share (one
    ``hydra.sweep.subdir``) is not cleaned: a later job keeps the earlier
    job's run and takes the next stem, and an earlier sweep's run there
    stays, as before #1956."""
    _gate_template(tmp_path)
    session = tmp_path / "multirun"
    (session / "shared").mkdir(parents=True)
    # Sorting before App: the sweeper's verdict reads the last stem.
    (session / "shared" / "Aaa_result.json").write_text('{"passed": false}')
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+gated=true",
        "hydra.sweep.subdir=shared",
        sweep_dir=session,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sorted(p.name for p in (session / "shared").glob("*_result.json")) == [
        "Aaa_result.json",
        "App-1_result.json",
        "App_result.json",
    ]


def test_resuming_into_a_shared_directory_keeps_the_skipped_jobs_runs(
    tmp_path: Path,
) -> None:
    """A directory every job maps to (one ``hydra.sweep.subdir``) holds the
    run of the job a ``--resume-from`` skips too: it is not cleaned, so that
    run stays (#1956)."""
    _gate_template(tmp_path)
    session = tmp_path / "multirun"
    common = (
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+gated=true",
        "hydra.sweep.subdir=shared",
    )
    proc = _run(_SWEEPER_APP, tmp_path, *common, sweep_dir=session)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc = _run(_SWEEPER_APP, tmp_path, *common, resume_from=2, sweep_dir=session)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Removing the outputs" not in proc.stdout + proc.stderr
    # Job 1's run (App) and both of job 2's: nothing removed.
    assert sorted(p.name for p in (session / "shared").glob("*_gate.json")) == [
        "App-1_gate.json",
        "App-2_gate.json",
        "App_gate.json",
    ]


def test_an_earlier_sweeps_service_logs_are_not_removed(tmp_path: Path) -> None:
    """The logs services write into a job directory (a containerised
    backend's, Autoware's) are not a run, though ``run_output_stems`` reads
    them as a CARLA recorder log's stem; a CARLA recorder log, or anything
    of a run, is."""
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _remove_earlier_outputs,
    )

    kept = [
        "backend_carla.log",
        "autoware-000-ego.log",
        "autoware-build.log",
        "autoware-deps.log",
        "raw_output.log",
        "run.log",
        "app.log",  # Hydra's job log
        "notes.txt",
    ]
    removed = ["Old.log", "Old_result.json", "autoware-x_result.json"]
    for name in kept + removed:
        (tmp_path / name).write_text("")
    (tmp_path / "Old.t4").mkdir()
    _remove_earlier_outputs((tmp_path, "app.log"), "1/1")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(kept)


def test_a_retry_keeps_the_failed_attempts_service_logs(tmp_path: Path) -> None:
    """Before a retry the sweeper removes the failed attempt's run (its
    CARLA recorder log, its result) but not the logs services append to in
    the job's directory: they explain the failed attempt (#2070)."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=1",
        "+service_logs_once=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "2/2 succeeded, 0 failed" in output, output
    for job in ("0", "1"):
        directory = tmp_path / "multirun" / job
        assert sorted(p.name for p in directory.iterdir() if p.name != ".hydra") == [
            "App.log",  # the retry's, under the stem the failed attempt took
            "App_result.json",
            "app.log",  # Hydra's job log
            "autoware-000-ego.log",
            "autoware-build.log",
            "backend_sumo.log",
            "raw_output.log",
        ]
        # Both attempts' lines: the failed attempt's were kept.
        for name in ("backend_sumo.log", "autoware-000-ego.log", "raw_output.log"):
            assert (directory / name).read_text() == "attempt\nattempt\n", name


def test_an_output_a_retry_cannot_remove_is_logged_and_the_retry_runs(
    tmp_path: Path,
) -> None:
    """An ``OSError`` removing the failed attempt's outputs is logged, as
    before a job's first attempt, and the retry runs under the next free
    stem: it neither aborts the sweep nor leaves the job unfinished (#2070)."""
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=1",
        "+service_logs_once=true",
        "+unremovable=true",
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert output.count("case 1") == 2, output
    assert output.count("case 2") == 2, output
    assert "2/2 succeeded, 0 failed" in output, output
    assert "Cannot remove the outputs of the failed attempt App" in output, output
    assert _markers(tmp_path / "multirun") == {}
    for job in ("0", "1"):
        directory = tmp_path / "multirun" / job
        assert sorted(p.name for p in directory.glob("App*_result.json")) == [
            "App-1_result.json",
            "App_result.json",
        ]


def test_only_a_directory_one_job_of_the_sweep_maps_to_is_owned(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import (
        _owned_dirs,
    )

    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    running: list[tuple[Path, str | None] | None] = [
        (a, "x.log"),
        (b, "x.log"),
        (b, "x.log"),
    ]
    assert _owned_dirs(running, []) == {a}
    # A job the resume skips shares a's directory.
    assert _owned_dirs(running, [(tmp_path, a, "x.log")]) == set()
    assert _owned_dirs(running, [(tmp_path, c, "x.log")]) == {a, c}
    # A job whose directory cannot be told may share any.
    assert _owned_dirs(running, [None]) == set()
    assert _owned_dirs([(a, None), None], []) == set()


def test_a_sweep_resumed_into_a_new_directory_plans_only_what_it_runs(
    tmp_path: Path,
) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        resume_from=2,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert [d for d, _ in _plan(tmp_path / "multirun")] == ["1"]


def test_a_retry_that_passes_clears_the_unfinished_marker(tmp_path: Path) -> None:
    proc = _run(
        _SWEEPER_APP,
        tmp_path,
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        "+server.cooldown_max_retries=1",
        "+raise_once=true",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _markers(tmp_path / "multirun") == {}


def test_a_killed_basic_sweep_leaves_its_case_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hydra's basic sweeper runs its cases in the sweep's process: the case
    killed with it leaves the marker written as it started.  It plans
    nothing: the cases it never reached leave no trace (docs/gate.md)."""
    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    proc = _run(_BASIC_APP, tmp_path, "+verdict=pass,kill,pass")
    session = tmp_path / "multirun"
    assert proc.returncode == -9, proc.stdout + proc.stderr
    markers = _markers(session)
    assert list(markers) == ["1"]
    assert markers["1"]["unfinished"] is True
    assert "+verdict=kill" in markers["1"]["overrides"]
    assert not (session / "2").exists()
    assert not (session / "sweep_plan.json").exists()
