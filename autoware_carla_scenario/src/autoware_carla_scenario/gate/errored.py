"""The marker a ``--multirun`` job leaves when it is given up: ``case_errored.json``.

A case that raises, times out or crashes writes no result and so no gate
record (:mod:`.record`).  A batch lists such a case in its own summary
(``batch_results.json``, a SLURM batch's outcomes), but a ``--multirun``
sweep has no summary: its jobs are directories of their own.  So a sweep
writes this marker into the job's directory when it gives a job up for any
reason but a failed verdict -- the lanelet sweeper once every attempt has
failed, ``examples/run.py`` when a case of Hydra's basic sweeper raises --
and ``scenario-gate`` refuses a set of runs holding one
(:func:`~autoware_carla_scenario.gate.evaluate.unrecorded_cases`).

It is removed before the job's first attempt, so a job run again in the same
directory (``hydra.sweep.dir=`` naming it again) leaves no stale marker, and
it is written only once the job is given up, so a retry that passes leaves
none.  Its name ends in neither a run output's suffix
(``run_outputs.RUN_SUFFIXES``) nor ``_result.json``: the viewer, the retry's
clean-up and the coverage merge do not take it for a run's file.

Free of heavy imports: the sweeper's parent process writes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional  # noqa: UP035

from ..atomic_file import write_text_atomically

__all__ = [
    "CASE_ERRORED",
    "CASE_ERRORED_SCHEMA",
    "find_case_errored",
    "read_case_errored",
    "remove_case_errored",
    "write_case_errored",
]

#: The marker's file name, in the job's directory.
CASE_ERRORED = "case_errored.json"

#: The format of the marker.
CASE_ERRORED_SCHEMA = "autoware_carla_scenario.case_errored/1"


def write_case_errored(
    directory: Path,
    *,
    overrides: list[str],
    exit_code: Optional[int],
    attempts: int,
    reason: str,
) -> Path:
    """Write the marker of a job given up into *directory*, in one rename,
    and return its path.

    *exit_code* is the job's last exit status (``None`` when it timed out, a
    negative number for the signal it died of); *reason* says why in words.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CASE_ERRORED
    data: dict[str, Any] = {
        "schema": CASE_ERRORED_SCHEMA,
        "overrides": list(overrides),
        "exit_code": exit_code,
        "attempts": attempts,
        "reason": reason,
    }
    write_text_atomically(path, json.dumps(data, indent=2, ensure_ascii=False))
    return path


def remove_case_errored(directory: Path) -> bool:
    """Remove the marker from *directory*; whether there was one."""
    path = directory / CASE_ERRORED
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def read_case_errored(path: Path) -> Optional[dict[str, Any]]:
    """The marker at *path* as a JSON object; ``None`` when it cannot be read
    or is not one (it still marks its job as given up)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_case_errored(path: Path) -> list[Path]:
    """Every marker under the directory *path*, sorted; none for a file."""
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob(CASE_ERRORED) if p.is_file())
