"""What a set of runs left unrecorded, file by file: the scans behind
``scenario-gate``'s refusal (:func:`~.evaluate.unrecorded_cases`).

A case of a local batch that raised rather than ran, or that a batch cut
short never finished, is listed in its summary (``batch_results.json``); a
``--multirun`` job given up or whose process died with it leaves its marker
(``case_errored.json``, :mod:`.errored`); a job the lanelet sweeper planned
and never started is in its sweep's plan (``sweep_plan.json``,
:mod:`.sweep_plan`).  This module finds the batches under a directory, reads
a local batch's summary and a SLURM batch's manifest and outcomes
(``slurm/``, :mod:`..slurm.manifest`), and finds those jobs, each beside the
gate records and the runs' results (``*_result.json``) that tell a run that
left a record from one that did not, and joins the lines of them all
(:func:`_scan_batches`, which :mod:`.evaluate` re-exports).

Free of heavy imports, so the library check compiles it (docs/typecheck.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# The summary's name and the check of a stem it names: batch_report's, which
# writes it (BATCH_RESULTS re-exported under the name it had here).
from ..batch_report import BATCH_RESULTS as BATCH_RESULTS
from ..batch_report import is_plain_stem

# The suffix of a run's result, beside its gate record: run_outputs's (#2198).
from ..run_outputs import RESULT_SUFFIX as _RESULT_SUFFIX
from .errored import (
    CASE_ERRORED,
    find_case_errored,
    marker_since,
    read_case_errored,
    written_since,
)
from .record import GATE_RECORD_SUFFIX
from .sweep_plan import find_sweep_plans, read_sweep_plan

__all__ = [
    "BATCH_RESULTS",
]

#: How many of a batch's unrecorded cases the refusal names.
_NAMED = 5


def _batch_dirs(path: Path) -> list[Path]:
    """The batches under *path*: each directory holding a summary
    (``batch_results.json``) or a SLURM batch's manifest, sorted."""
    found = {p.parent for p in path.rglob(BATCH_RESULTS)}
    found.update(p.parent.parent for p in path.rglob("slurm/manifest.json"))
    return sorted(found)


def _recorded_at(directory: Path, name: str) -> bool:
    """Whether a gate record is at the stem *name* in *directory*: a file
    stem of its own (``batch_report.is_plain_stem()``), as a batch writes it
    since #1957."""
    if not is_plain_stem(name):
        return False
    return (directory / f"{name}{GATE_RECORD_SUFFIX}").is_file()


def _recorded_at_stem(directory: Path, entry: dict[str, Any]) -> bool:
    """Whether a gate record is at the stem *entry* names in *directory*: a
    file stem of its own (``batch_report.is_plain_stem()``), as a batch
    writes it since #1957."""
    stem = entry.get("stem")
    # `stem` is the str json.loads() read, so str(stem) is that str itself:
    # Codon holds it as a JSONValue, which compares with no str.
    if not isinstance(stem, str):
        return False
    return _recorded_at(directory, str(stem))


def _summary_cases(directory: Path) -> tuple[list[str], int, Optional[str]]:
    """The names of a local batch's errored cases (``"error": true`` in its
    ``batch_results.json``), which raised and so, but for one that raised
    only once its run had ended, wrote no gate record, and of its pending
    ones (``"pending": true``: a batch cut short never finished them); how
    many cases it has; and ``None``.  Or no names, ``0`` and why the summary
    cannot be read, instead.  (:func:`_scan_batches` tells the one from the
    other: a value of either type, which Codon cannot hold.)

    An errored case whose entry names its stem (``stem``, written since
    #1957) and has a gate record there is not counted: its run wrote one
    before it raised, and that record is judged, as a SLURM batch's is.  A
    pending case is counted whatever is at its stem: the batch never
    finished it.  A summary with an entry that is no JSON object cannot be
    read, as ``batch_resume.read_resumed_batch()`` refuses it too."""
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
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        unreadable = f"{path}: not a batch summary (not a JSON array of objects)"
        return unrecorded, 0, unreadable
    # `x is True` is written `isinstance(x, bool) and bool(x)`, the same test
    # (bool has no subclass, and True is its one true instance: 1 and 1.0
    # are no bool), which Codon compiles for a JSON value; and
    # `str(x or i)` is written out, as Codon has no value of either type.
    for i, entry in enumerate(entries):
        # An entry counts when it errored with no record at its stem, or is
        # pending.
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


def _slurm_cases(directory: Path) -> tuple[list[str], int, Optional[str]]:
    """The labels of a SLURM batch's cases that errored, have no outcome or
    one holding no verdict, and left no gate record at their stem, and of
    those whose outcome is stale (:func:`..slurm.collect.stale_reason`):
    an earlier submission's, which the last one to give the case did not
    reach, so the gate record at its stem is that earlier run's and does not
    count (#2418); how many cases it has; and ``None``.  Or no labels,
    ``0`` and why the manifest cannot be read, instead
    (:func:`_summary_cases`'s triple).

    A case is named by its ``label``, or its ``scenario``, when true, else
    by its index (:meth:`.ManifestCase.name`).  Only a stem of its own
    counts (#2265), as a local batch's, and only a string:
    ``case.get("stem")`` as :func:`_recorded_at_stem` reads an entry
    (:meth:`.ManifestCase.string_stem`).  A manifest without ``cases``, or
    whose ``cases`` cannot be iterated, cannot be read
    (:meth:`.Manifest.cases`), nor can one with a case without an index
    that ``int()`` reads, which the error shows
    (:meth:`.ManifestCase.case_repr`), nor one whose submission records
    cannot be read to tell whether an outcome is stale (``chunks`` of the
    wrong type, :meth:`.SubmissionRecord.task_of`)."""
    from ..slurm.collect import stale_reason  # noqa: PLC0415
    from ..slurm.manifest import (  # noqa: PLC0415
        BatchLayout,
        CaseOutcome,
        recorded_verdict,
    )

    layout = BatchLayout(directory)
    # Each part of the result held in a local of its type (_summary_cases()).
    unrecorded: list[str] = []
    unreadable: Optional[str] = None
    try:
        cases = layout.read_manifest().cases()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        unreadable = f"{layout.manifest}: the SLURM manifest cannot be read ({exc})"
        return unrecorded, 0, unreadable
    for case in cases:
        try:
            index = case.index()
        except (KeyError, TypeError, ValueError):
            unreadable = (
                f"{layout.manifest}: a case without an index ({case.case_repr()})"
            )
            none: list[str] = []
            return none, 0, unreadable
        record = layout.read_result(index)
        stale = False
        if record is not None:
            # Typed through a local: Codon 0.19 keeps the Optional otherwise.
            outcome: CaseOutcome = record
            try:
                stale = stale_reason(layout, index, outcome) is not None
            except (OSError, TypeError, ValueError) as exc:
                # Named by their directory, not the manifest: the line
                # says to repair the file it names.
                records = layout.root / "submissions"
                unreadable = (
                    f"{records}: the batch's SLURM submission records "
                    f"cannot be read ({exc})"
                )
                none_read: list[str] = []
                return none_read, 0, unreadable
            if stale:
                # An earlier submission's outcome, and the gate record at
                # its stem that run's: the case was to run again and did
                # not (#2418).
                unrecorded.append(case.name(index))
                continue
            if recorded_verdict(outcome) is not None and not outcome.errored:
                continue
        # A stem of its own only, as a local batch's (#2265).
        stem = case.string_stem()
        if stem is not None and _recorded_at(directory, stem):
            continue
        unrecorded.append(case.name(index))
    return unrecorded, len(cases), unreadable


def _shown(names: list[str]) -> str:
    """The first :data:`_NAMED` of *names*, joined, and how many more there
    are (``a, b, c, d, e, and 2 more``): what a refusal's line names."""
    shown = ", ".join(names[:_NAMED])
    more = f", and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
    return f"{shown}{more}"


def _batch_line(directory: Path, names: list[str], total: int) -> str:
    """The refusal's line for a batch whose cases *names* (of *total*)
    raised or have no outcome and left no gate record, naming the first
    :data:`_NAMED` of them."""
    return (
        f"{directory}: {len(names)} of {total} case(s) raised or have no "
        f"outcome, and left no gate record ({_shown(names)}); the gate would "
        "be judged over the runs that survived"
    )


def _scan_batches(path: Path) -> tuple[list[str], list[str]]:
    """What :func:`~.evaluate.unrecorded_cases` and
    :func:`~.evaluate.batch_read_errors` return for *path*, read once."""
    if not path.is_dir():
        return [], []
    lines: list[str] = []
    unreadable: list[str] = []
    for directory in _batch_dirs(path):
        if (directory / "slurm" / "manifest.json").is_file():
            names, total, why = _slurm_cases(directory)
        else:
            names, total, why = _summary_cases(directory)
        if why is not None:
            unreadable.append(why)
            continue
        if not names:
            continue
        lines.append(_batch_line(directory, names, total))
    lines.extend(_errored_jobs(path))
    unstarted, unreadable_plans = _unstarted_jobs(path)
    lines.extend(unstarted)
    unreadable.extend(unreadable_plans)
    return lines, unreadable


def _has_file(directory: Path, suffix: str) -> bool:
    """Whether *directory* holds a file whose name ends in *suffix*;
    ``False`` when it cannot be listed."""
    try:
        return any(p.name.endswith(suffix) and p.is_file() for p in directory.iterdir())
    except OSError:
        return False


def _has_gate_record(directory: Path) -> bool:
    """Whether *directory* holds a gate record."""
    return _has_file(directory, GATE_RECORD_SUFFIX)


def _has_result(directory: Path) -> bool:
    """Whether *directory* holds a run's result (``*_result.json``)."""
    return _has_file(directory, _RESULT_SUFFIX)


def _has_file_since(directory: Path, suffix: str, since: float) -> bool:
    """Whether *directory* holds a file whose name ends in *suffix* written
    no earlier than *since* (:func:`.errored.written_since`): one of an
    earlier run in the same directory is not this attempt's.  ``False`` when
    the directory cannot be listed."""
    try:
        return any(
            p.name.endswith(suffix) and written_since(p, since)
            for p in directory.iterdir()
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
    except OSError:
        return False
    return _has_file_since(marker.parent, _RESULT_SUFFIX, since)


def _has_gate_record_since(marker: Path) -> bool:
    """Whether the job directory of *marker* holds a gate record written no
    earlier than the attempt the marker is of started
    (:func:`.errored.marker_since`): one of an earlier run in the same
    directory is not this attempt's (#2416).  Any gate record counts when
    that time cannot be read, as before."""
    since = marker_since(marker)
    if since is None:
        return _has_gate_record(marker.parent)
    return _has_file_since(marker.parent, GATE_RECORD_SUFFIX, since)


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
        lines.append(
            f"{plan.parent}: {len(names)} of {len(jobs)} planned job(s) of the "
            f"sweep never ran, and left no gate record ({_shown(names)}); the "
            "gate would be judged over the runs that survived"
        )
    return lines, unreadable


def _errored_jobs(path: Path) -> list[str]:
    """One line per sweep under *path* whose ``--multirun`` jobs were given up
    or did not finish (``case_errored.json``, :mod:`.errored`) and left no
    gate record since the attempt the marker is of started
    (:func:`_has_gate_record_since`): a gate record older than that is an
    earlier run's in the same directory, and the marker counts (#2416).

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
        if _has_gate_record_since(marker):
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
        lines.append(
            f"{sweep}: {len(names)} job(s) of the sweep raised, timed out, "
            f"crashed or did not finish, and left no gate record "
            f"({_shown(names)}); the gate would "
            "be judged over the runs that survived"
        )
    return lines
