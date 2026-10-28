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

import pytest

#: A minimal ``@hydra.main`` app swept by LaneletConstraintSweeper, with the
#: map loading and the expansion stubbed to two cases.  Its task fails when
#: ``+fail=true`` is given, writing the result JSON a scenario writes; as
#: ``examples/run.py:_hydra_main`` does in ``--multirun``, it neither raises
#: nor ``sys.exit``s for a failed scenario.
_SWEEPER_APP = """
import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import autoware_carla_scenario.sweeper.lanelet_constraint_sweeper as lcs

lcs.check_constraint_keys = lambda sweep: None
lcs.resolve_map_paths = lambda cfg: None
lcs.load_map = lambda paths: object()
lcs.scenario_controls = lambda cfg: None
lcs.expand_sweep = lambda sweep, lanelet_map, arguments, **_: [
    ["+case=1", *arguments],
    ["+case=2", *arguments],
]


@hydra.main(version_base=None, config_path=None, config_name=None)
def task(cfg: DictConfig) -> None:
    print(f"case {cfg.case}", flush=True)
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


def _run(app: str, tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "app.py"
    script.write_text(textwrap.dedent(app), encoding="utf-8")
    # A resume point would skip the cases.
    env = {k: v for k, v in os.environ.items() if k != "SWEEP_RESUME_FROM"}
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
