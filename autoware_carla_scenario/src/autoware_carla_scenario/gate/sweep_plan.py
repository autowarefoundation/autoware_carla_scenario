"""The jobs a ``--multirun`` sweep is going to run: ``sweep_plan.json``.

A job that never started leaves nothing on disk: no gate record, no marker
(``case_errored.json``, :mod:`.errored`), often not even its directory.  A
20-job sweep killed in job 8 would leave jobs 1-7 for ``scenario-gate`` to
judge as if they were the whole sweep.  So the lanelet sweeper writes this
plan into its sweep directory (``hydra.sweep.dir``) before it launches the
first job: the directory and the overrides of every job it is about to run.
``scenario-gate`` counts a planned job that left neither a gate record, nor a
marker, nor a result as one that never ran
(:func:`~autoware_carla_scenario.gate.evaluate.unrecorded_cases`, #1833).

A sweep planned again into the same directory (``hydra.sweep.dir=`` naming
it again, as a ``--resume-from`` rerun does to finish an interrupted sweep)
merges its plan into the one there: a job it runs again is replaced, and the
jobs of the earlier plan it does not run (those a resume skips) are kept,
since their outputs are still there to be judged.

Each job's directory is written relative to the plan's when it lies under
it, so a sweep directory moved or copied whole keeps its plan.  The name
ends in neither a run output's suffix nor ``_result.json``, and the plan sits
beside the job directories, not in one: the viewer, which lists a sweep's
numbered job directories, and the coverage merge do not read it.

Hydra's basic sweeper creates its job directories one at a time and has no
plan: there, only the job running when the sweep died leaves a trace (its
marker).

Free of heavy imports: the sweeper's parent process writes it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..atomic_file import write_text_atomically

__all__ = [
    "SWEEP_PLAN",
    "SWEEP_PLAN_SCHEMA",
    "PlannedJob",
    "find_sweep_plans",
    "read_sweep_plan",
    "write_sweep_plan",
]

#: The plan's file name, in the sweep directory.
SWEEP_PLAN = "sweep_plan.json"

#: The format of the plan.
SWEEP_PLAN_SCHEMA = "autoware_carla_scenario.sweep_plan/1"

#: A planned job: its directory (absolute) and its overrides.
PlannedJob = tuple[Path, list[str]]


def _entry_dir(sweep_dir: Path, job_dir: Path) -> str:
    """*job_dir* as the plan in *sweep_dir* writes it: relative when it lies
    under *sweep_dir*, absolute otherwise."""
    job = Path(os.path.abspath(job_dir))
    base = Path(os.path.abspath(sweep_dir))
    try:
        return job.relative_to(base).as_posix()
    except ValueError:
        return str(job)


def _read_entries(path: Path) -> list[dict[str, Any]]:
    """The plan's entries at *path*, raw.

    Raises:
        OSError: When it cannot be read.
        ValueError: When it is not a plan (not JSON, not an object holding a
            ``jobs`` list of objects with a ``dir`` string).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except RecursionError as exc:
        raise ValueError(str(exc)) from None
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise ValueError("not a sweep plan (no list of jobs)")
    entries = []
    for job in data["jobs"]:
        if not isinstance(job, dict) or not isinstance(job.get("dir"), str):
            raise ValueError(f"a job without a directory ({job!r})")
        entries.append(job)
    return entries


def write_sweep_plan(sweep_dir: Path, jobs: list[PlannedJob]) -> Path:
    """Write the plan of *jobs* into *sweep_dir*, in one rename, merged into
    the plan already there, and return its path.

    An entry of the earlier plan for a directory among *jobs* is replaced;
    the others are kept, in their order, before the new ones.  An earlier
    plan that cannot be read is replaced whole.
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)
    path = sweep_dir / SWEEP_PLAN
    new: dict[str, dict[str, Any]] = {}
    for job_dir, overrides in jobs:
        key = _entry_dir(sweep_dir, job_dir)
        new[key] = {"dir": key, "overrides": [str(o) for o in overrides]}
    merged: list[dict[str, Any]] = []
    if path.is_file():
        try:
            earlier = _read_entries(path)
        except (OSError, ValueError):
            earlier = []
        merged = [e for e in earlier if e["dir"] not in new]
    merged.extend(new.values())
    data: dict[str, Any] = {"schema": SWEEP_PLAN_SCHEMA, "jobs": merged}
    write_text_atomically(path, json.dumps(data, indent=2, ensure_ascii=False))
    return path


def read_sweep_plan(path: Path) -> list[PlannedJob]:
    """The jobs of the plan at *path*, each directory resolved against the
    plan's.

    Raises:
        OSError: When it cannot be read.
        ValueError: When it is not a plan.
    """
    jobs: list[PlannedJob] = []
    for entry in _read_entries(path):
        overrides = entry.get("overrides")
        listed = [str(o) for o in overrides] if isinstance(overrides, list) else []
        jobs.append((path.parent / str(entry["dir"]), listed))
    return jobs


def find_sweep_plans(path: Path) -> list[Path]:
    """Every plan under the directory *path*, sorted; none for a file."""
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob(SWEEP_PLAN) if p.is_file())
