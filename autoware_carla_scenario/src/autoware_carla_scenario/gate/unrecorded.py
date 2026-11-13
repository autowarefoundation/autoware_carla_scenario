"""What a set of runs left unrecorded, file by file: the scans behind
``scenario-gate``'s refusal (:func:`~.evaluate.unrecorded_cases`).

A case of a local batch that raised rather than ran, or that a batch cut
short never finished, is listed in its summary (``batch_results.json``); a
``--multirun`` job given up or whose process died with it leaves its marker
(``case_errored.json``, :mod:`.errored`); a job the lanelet sweeper planned
and never started is in its sweep's plan (``sweep_plan.json``,
:mod:`.sweep_plan`).  This module finds the batches under a directory, reads
a local batch's summary, and finds those jobs, each beside the gate records
and the runs' results (``*_result.json``) that tell a run that left a record
from one that did not.  A SLURM batch's cases are read from its manifest by
:mod:`.evaluate`, which joins the lines of them all.

Free of heavy imports, so the library check compiles it (docs/typecheck.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .errored import CASE_ERRORED, find_case_errored, read_case_errored
from .record import GATE_RECORD_SUFFIX
from .sweep_plan import find_sweep_plans, read_sweep_plan

__all__ = [
    "BATCH_RESULTS",
]

#: The summary a batch writes beside its runs' outputs (``batch_report``).
BATCH_RESULTS = "batch_results.json"

#: The suffix of a run's result, beside its gate record.
_RESULT_SUFFIX = "_result.json"

#: How many of a batch's unrecorded cases the refusal names.
_NAMED = 5


def _batch_dirs(path: Path) -> list[Path]:
    """The batches under *path*: each directory holding a summary
    (``batch_results.json``) or a SLURM batch's manifest, sorted."""
    found = {p.parent for p in path.rglob(BATCH_RESULTS)}
    found.update(p.parent.parent for p in path.rglob("slurm/manifest.json"))
    return sorted(found)


def _recorded_at_stem(directory: Path, entry: dict[str, Any]) -> bool:
    """Whether a gate record is at the stem *entry* names in *directory*: a
    file stem of its own (no separator), as a batch writes it since #1957."""
    stem = entry.get("stem")
    # `stem` is the str json.loads() read, so str(stem) is that str itself:
    # Codon holds it as a JSONValue, which compares with no str.
    if not isinstance(stem, str):
        return False
    name = str(stem)
    if name in ("", ".", "..") or "/" in name:
        return False
    if "\\" in name or "\0" in name:
        return False
    return (directory / f"{name}{GATE_RECORD_SUFFIX}").is_file()


def _summary_cases(directory: Path) -> tuple[list[str], int, Optional[str]]:
    """The names of a local batch's errored cases (``"error": true`` in its
    ``batch_results.json``), which raised and so, but for one that raised
    only once its run had ended, wrote no gate record, and of its pending
    ones (``"pending": true``: a batch cut short never finished them); how
    many cases it has; and ``None``.  Or no names, ``0`` and why the summary
    cannot be read, instead.  (:func:`~.evaluate._summary_unrecorded` returns
    the one or the other, a value of either type, which Codon cannot hold.)

    An errored case whose entry names its stem (``stem``, written since
    #1957) and has a gate record there is not counted: its run wrote one
    before it raised, and that record is judged, as a SLURM batch's is.  A
    pending case is counted whatever is at its stem: the batch never
    finished it.  An entry that is no JSON object raises AttributeError (its
    ``get()``)."""
    path = directory / BATCH_RESULTS
    # Each part of the result held in a local of its type, which Codon then
    # gives the tuple (an Optional of no str, the str of none).
    unrecorded: list[str] = []
    unreadable: Optional[str] = None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, ValueError) as exc:
        unreadable = f"{path}: the batch summary cannot be read ({exc})"
        return unrecorded, 0, unreadable
    if not isinstance(entries, list):
        unreadable = f"{path}: not a batch summary (not a JSON array)"
        return unrecorded, 0, unreadable
    # `x is True` is written `isinstance(x, bool) and bool(x)`, the same test
    # (bool has no subclass, and True is its one true instance: 1 and 1.0
    # are no bool), which Codon compiles for a JSON value; and
    # `str(x or i)` is written out, as Codon has no value of either type.
    for i, entry in enumerate(entries):
        if isinstance(entry, dict):
            # An object counts when it errored with no record at its stem,
            # or is pending; anything else counts (and raises below).
            error = entry.get("error")
            if not (
                isinstance(error, bool)
                and bool(error)
                and not _recorded_at_stem(directory, entry)
            ):
                pending = entry.get("pending")
                if not (isinstance(pending, bool) and bool(pending)):
                    continue
        scenario = entry.get("scenario")
        unrecorded.append(str(scenario) if scenario else str(i))
    return unrecorded, len(entries), unreadable


def _has_gate_record(directory: Path) -> bool:
    """Whether *directory* holds a gate record."""
    try:
        return any(
            p.name.endswith(GATE_RECORD_SUFFIX) and p.is_file()
            for p in directory.iterdir()
        )
    except OSError:
        return False


def _has_result(directory: Path) -> bool:
    """Whether *directory* holds a run's result (``*_result.json``)."""
    try:
        return any(
            p.name.endswith(_RESULT_SUFFIX) and p.is_file() for p in directory.iterdir()
        )
    except OSError:
        return False


def _has_result_since(marker: Path) -> bool:
    """Whether the job directory of *marker* holds a run's result
    (``*_result.json``) written no earlier than the marker: one of an
    earlier run in the same directory is not this attempt's.  ``False`` when
    either cannot be read."""
    try:
        since = marker.stat().st_mtime
        return any(
            p.name.endswith(_RESULT_SUFFIX)
            and p.is_file()
            and p.stat().st_mtime >= since
            for p in marker.parent.iterdir()
        )
    except OSError:
        return False


def _unstarted_jobs(path: Path) -> tuple[list[str], list[str]]:
    """One line per sweep under *path* whose plan (``sweep_plan.json``,
    :mod:`.sweep_plan`) lists jobs that never ran: their directory holds no
    gate record, no marker (``case_errored.json``, counted by
    :func:`_errored_jobs`) and no result -- a job of a scenario without a
    gate writes a result and no record, and is no concern of the gate.  And
    one line per plan that cannot be read, instead."""
    lines: list[str] = []
    unreadable: list[str] = []
    for plan in find_sweep_plans(path):
        try:
            jobs = read_sweep_plan(plan)
        except (OSError, ValueError) as exc:
            unreadable.append(f"{plan}: the sweep plan cannot be read ({exc})")
            continue
        names: list[str] = []
        for job, _overrides in jobs:
            if job.is_dir() and (
                (job / CASE_ERRORED).is_file()
                or _has_gate_record(job)
                or _has_result(job)
            ):
                continue
            try:
                names.append(job.relative_to(plan.parent).as_posix())
            except ValueError:
                names.append(str(job))
        if not names:
            continue
        shown = ", ".join(names[:_NAMED])
        more = f", and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
        lines.append(
            f"{plan.parent}: {len(names)} of {len(jobs)} planned job(s) of the "
            f"sweep never ran, and left no gate record ({shown}{more}); the "
            "gate would be judged over the runs that survived"
        )
    return lines, unreadable


def _errored_jobs(path: Path) -> list[str]:
    """One line per sweep under *path* whose ``--multirun`` jobs were given up
    or did not finish (``case_errored.json``, :mod:`.errored`) and left no
    gate record.

    A marker saying the job did not finish (``"unfinished": true``) beside a
    run's result (``*_result.json``) written no earlier than it is not
    counted either.  The runner writes a run's gate record before its
    result, so the run was of a scenario without a gate, which writes no
    record, and its process was killed in what follows (the world reset, the
    Rerun conversion) before the marker was removed (#1840).  A result older
    than the marker is an earlier run's, and the marker counts.  A marker of
    a job given up counts beside a result, which may be an earlier
    attempt's.  A marker that cannot be read counts too."""
    by_sweep: dict[Path, list[str]] = {}
    for marker in find_case_errored(path):
        job = marker.parent
        if _has_gate_record(job):
            # The run wrote its record before the job failed: it is judged.
            continue
        data = read_case_errored(marker)
        # `x is True` is written `isinstance(x, bool) and bool(x)`, the same
        # test, which Codon compiles for a JSON value (_summary_cases()).
        if data is not None:
            unfinished = data.get("unfinished")
            if (
                isinstance(unfinished, bool)
                and bool(unfinished)
                and _has_result_since(marker)
            ):
                # The run wrote its result and was killed after it: it ran a
                # scenario without a gate, no concern of the gate.
                continue
        # The reason is the str json.loads() read, so str(reason) is that
        # str itself: Codon holds it as a JSONValue.
        name = job.name
        if data is not None:
            reason = data.get("reason")
            if isinstance(reason, str) and str(reason):
                name = f"{job.name} ({str(reason)})"
        by_sweep.setdefault(job.parent, []).append(name)
    lines: list[str] = []
    for sweep, names in sorted(by_sweep.items()):
        shown = ", ".join(names[:_NAMED])
        more = f", and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
        lines.append(
            f"{sweep}: {len(names)} job(s) of the sweep raised, timed out, "
            f"crashed or did not finish, and left no gate record ({shown}{more}); the gate would "
            "be judged over the runs that survived"
        )
    return lines
