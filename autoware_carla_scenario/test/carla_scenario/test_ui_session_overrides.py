"""Multirun and single sessions match their cases by the sweep's overrides
(#2239).

A case's key leaves out the session's command line -- the overrides a CLI
multirun's ``multirun.yaml`` records, else those every job shares (the Run
panel's extra overrides) -- so two sessions that differ only there are
compared case by case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoware_carla_scenario.ui import cases, results

DATE = "2026-11-20"


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path


def _job(base: Path, time: str, index: int, overrides: list[str], passed: bool) -> Path:
    job = base / "multirun" / DATE / time / str(index)
    (job / ".hydra").mkdir(parents=True)
    (job / ".hydra" / "overrides.yaml").write_text(
        "".join(f"- {o}\n" for o in overrides)
    )
    (job / "X_result.json").write_text(
        json.dumps({"passed": passed, "elapsed_seconds": 1})
    )
    return job


def _changes(base: Path, *times: str) -> list[tuple[str, str]]:
    sessions = [
        (time, results.load_results(base, "multirun", DATE, time, metrics=False))
        for time in times
    ]
    return sorted((m.key, m.change()) for m in results.compare(sessions).cases)


def test_sessions_differing_in_a_session_override_match_case_by_case(
    base: Path,
) -> None:
    # As the Run panel groups them: its extra override is given to every job.
    _job(base, "10-00-00", 0, ["scenario=a/b", "ego.speed=10"], True)
    _job(base, "10-00-00", 1, ["scenario=a/b", "ego.speed=20"], True)
    _job(
        base,
        "11-00-00",
        0,
        ["scenario=a/b", "ego.speed=10", "autoware.planner=new"],
        False,
    )
    _job(
        base,
        "11-00-00",
        1,
        ["scenario=a/b", "ego.speed=20", "autoware.planner=new"],
        True,
    )
    assert _changes(base, "10-00-00", "11-00-00") == [
        ("a/b|ego.speed=10", "regression"),
        ("a/b|ego.speed=20", "same"),
    ]


def test_a_cli_multiruns_command_line_is_read_from_multirun_yaml(
    base: Path,
) -> None:
    # A one-job session: its job's overrides are all shared, but Hydra's
    # record says which of them the sweep set.
    for time, planner, passed in (
        ("10-00-00", "old", True),
        ("11-00-00", "new", False),
    ):
        _job(
            base,
            time,
            0,
            ["scenario=a/b", "ego.speed=10", f"autoware.planner={planner}"],
            passed,
        )
        (base / "multirun" / DATE / time / "multirun.yaml").write_text(
            "hydra:\n  overrides:\n    task:\n"
            "    - scenario=a/b\n    - ego.speed=10\n"
            f"    - autoware.planner={planner}\n"
        )
    # ego.speed=10 is a sweep of one value, given as a fixed override: both
    # are the command line, and the case is matched by its scenario.
    assert _changes(base, "10-00-00", "11-00-00") == [("a/b", "regression")]
    (base / "multirun" / DATE / "11-00-00" / "multirun.yaml").write_text(
        "hydra:\n  overrides:\n    task:\n"
        "    - scenario=a/b\n    - ego.speed=10,20\n    - autoware.planner=new\n"
    )
    (base / "multirun" / DATE / "10-00-00" / "multirun.yaml").write_text(
        "hydra:\n  overrides:\n    task:\n"
        "    - scenario=a/b\n    - ego.speed=10,20\n    - autoware.planner=old\n"
    )
    assert _changes(base, "10-00-00", "11-00-00") == [
        ("a/b|ego.speed=10", "regression")
    ]


def test_an_unreadable_multirun_yaml_falls_back_to_the_shared_overrides(
    base: Path,
) -> None:
    _job(base, "10-00-00", 0, ["scenario=a/b", "x=1", "y=1"], True)
    _job(base, "10-00-00", 1, ["scenario=a/b", "x=2", "y=1"], True)
    (base / "multirun" / DATE / "10-00-00" / "multirun.yaml").write_text("[: not yaml")
    found = results.load_results(base, "multirun", DATE, "10-00-00", metrics=False)
    assert [c.overrides for c in found] == [["x=1"], ["x=2"]]


def test_a_fixed_override_hydra_rewrote_is_the_command_line(base: Path) -> None:
    # Hydra's basic sweeper hands each job ``x=[1,2]`` for a command line's
    # ``x=[1, 2]``: it is the command line all the same.
    for time, planner in (("10-00-00", "old"), ("11-00-00", "new")):
        for index, speed in enumerate((10, 20)):
            _job(
                base,
                time,
                index,
                ["scenario=a/b", f"ego.speed={speed}", "x=[1,2]", f"p={planner}"],
                True,
            )
        (base / "multirun" / DATE / time / "multirun.yaml").write_text(
            "hydra:\n  overrides:\n    task:\n"
            "    - scenario=a/b\n    - ego.speed=range(10,30,10)\n"
            f"    - x=[1, 2]\n    - p= {planner}\n"
        )
    found = results.load_results(base, "multirun", DATE, "11-00-00", metrics=False)
    assert [c.overrides for c in found] == [["ego.speed=10"], ["ego.speed=20"]]
    assert _changes(base, "10-00-00", "11-00-00") == [
        ("a/b|ego.speed=10", "same"),
        ("a/b|ego.speed=20", "same"),
    ]


@pytest.mark.parametrize(
    "sweep",
    ["ego.speed=10,20", "ego.speed=range(10,30,10)", "ego.speed=choice(10,20)"],
)
def test_a_sweep_on_the_command_line_is_never_a_jobs_value(sweep: str) -> None:
    forms = cases._as_jobs_receive([sweep, "+g=glob(*)", "bad=(", "scenario=a/b"])
    assert forms == {sweep, "+g=glob(*)", "bad=(", "scenario=a/b"}


def test_a_job_without_overrides_leaves_the_others_shared(base: Path) -> None:
    # A job directory whose .hydra/overrides.yaml is missing (the job never
    # got that far) does not make every override a case's own.
    _job(base, "10-00-00", 0, ["scenario=a/b", "x=1", "y=1"], True)
    _job(base, "10-00-00", 1, ["scenario=a/b", "x=2", "y=1"], True)
    (base / "multirun" / DATE / "10-00-00" / "2").mkdir()
    found = results.load_results(base, "multirun", DATE, "10-00-00", metrics=False)
    assert [c.overrides for c in found] == [["x=1"], ["x=2"], []]
