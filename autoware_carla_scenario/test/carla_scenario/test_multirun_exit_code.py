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
#: ``+sweep.none=true``).  Its task fails when ``+fail=true`` is given, writing
#: the result JSON a scenario writes; as ``examples/run.py:_hydra_main`` does in
#: ``--multirun``, it neither raises nor ``sys.exit``s for a failed scenario.
#: With ``+raise_once=true`` each case raises the first time it runs, as a run
#: whose CARLA went away does.
_SWEEPER_APP = """
import json
import os
from pathlib import Path

from autoware_carla_scenario import hydra_compat  # noqa: F401

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import autoware_carla_scenario.sweeper.lanelet_constraint_sweeper as lcs

lcs.check_constraint_keys = lambda sweep: None
lcs.resolve_map_paths = lambda cfg: None
lcs.load_map = lambda paths: object()
lcs.scenario_controls = lambda cfg: None
lcs.expand_sweep = lambda sweep, lanelet_map, arguments, **_: (
    [] if sweep.get("none") else [["+case=1", *arguments], ["+case=2", *arguments]]
)


@hydra.main(version_base=None, config_path=None, config_name=None)
def task(cfg: DictConfig) -> None:
    print(f"case {cfg.case}", flush=True)
    owns = os.environ.get("AUTOWARE_CARLA_SCENARIO_SWEEPER_OWNS_RETRIES")
    print(f"queue retries owned by the sweeper: {owns}", flush=True)
    raised = Path.cwd() / f"raised{cfg.case}"
    if cfg.get("raise_once", False) and not raised.exists():
        raised.write_text("yes")
        raise RuntimeError("CARLA went away")
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
    if str(cfg.verdict) == "refuse":
        print("case verdict=refuse", flush=True)
        raise run.ScenarioTypeError("does not type-check")
    passed = str(cfg.verdict) == "pass"
    print(f"case verdict={cfg.verdict}", flush=True)
    return SimpleNamespace(passed=passed)


run.run_scenario = fake_run_scenario
sys.argv = ["scenario", *sys.argv[1:]]
run.main()
"""


def _run(
    app: str, tmp_path: Path, *args: str, resume_from: int | None = None
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
            "--multirun",
            *args,
            f"hydra.sweep.dir={tmp_path / 'multirun'}",
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verdicts recorded are those of this main() call alone."""
    from autoware_carla_scenario.examples import run

    monkeypatch.setattr(run, "_multirun_failed", ["+verdict=fail"])
    monkeypatch.setattr(run, "_multirun_ran", ["+verdict=fail"])
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: False)
    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_hydra_main", lambda: None)
    monkeypatch.setattr(sys, "argv", ["scenario", "--multirun", "+verdict=pass"])
    run.main()  # no SystemExit


def _stub_main(monkeypatch: pytest.MonkeyPatch, hydra_main: Any) -> Any:
    """``examples.run`` with main()'s single-run path calling *hydra_main*."""
    from autoware_carla_scenario.examples import run

    monkeypatch.setattr(run, "_multirun_failed", [])
    monkeypatch.setattr(run, "_multirun_ran", [])
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: False)
    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_hydra_main", hydra_main)
    return run


def test_resume_from_is_this_main_calls_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--resume-from reaches the sweeper of its own run, not a later one's (#797)."""
    seen: list[str | None] = []
    run = _stub_main(
        monkeypatch, lambda: seen.append(os.environ.get("SWEEP_RESUME_FROM"))
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value the user set stands for the runs without the flag, and comes
    back after one with it, even one that exits."""

    def hydra_main() -> None:
        seen.append(os.environ.get("SWEEP_RESUME_FROM"))
        raise SystemExit(1)

    seen: list[str | None] = []
    run = _stub_main(monkeypatch, hydra_main)
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
