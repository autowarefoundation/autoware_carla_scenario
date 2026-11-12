"""Judge logical scenarios' gates from their runs' records: ``scenario-gate``.

A concrete scenario passes or fails on its own conditions.  A logical
scenario's gate (:class:`~autoware_carla_scenario.authoring.models.GateSpec`)
is judged over every run of it, after they have run, from what each run
recorded (``{stem}_gate.json``, :mod:`.record`) and the coverage file beside
it.  Nothing is run again, so a gate can be re-judged -- against a document
whose thresholds have since changed (``--document``) -- at no cost.

The criteria, each judged only when the gate sets it:

* ``min_runs`` -- how many runs there were;
* ``min_success_rate`` -- the share that passed;
* ``min_case_coverage`` -- the share of the route search's cases (its matches,
  on each map the runs were on) that ran at least once;
* ``min_scenario_coverage``, ``min_odd_coverage``, ``coverage_items`` -- the
  grades of the runs' coverage files merged (openodd's merge, as
  ``scenario-coverage`` reports them);
* ``max_lateral_spread_m``, ``max_speed_spread_mps`` -- trajectory
  consistency: at stations along the pattern, each at the same fraction of its
  segment on every match, the standard deviation across runs of the ego's
  lateral offset from the route and of its speed; the largest is judged.

Given two sets of runs -- a baseline's and a candidate's, of the same
logical scenarios -- both are judged against one gate, and the report says
what regressed: a criterion that passed and fails now, and the cases every
baseline run of passed and a candidate run of failed.  The candidate's gate
decides the exit code; ``--fail-on-regression`` makes a regression fail it
too.

A case of a batch that raised rather than ran (``ERROR`` in its summary,
``batch_results.json``), that a batch cut short never finished (``pending``)
or, under SLURM, has no outcome, wrote no record; nor did a ``--multirun``
job given up or whose process died with it (``case_errored.json``,
:mod:`.errored`), nor one the lanelet sweeper planned and never started
(``sweep_plan.json``, :mod:`.sweep_plan`): the success rate would be judged
over the runs that survived.  A set of runs holding such a batch or
job is refused (:func:`unrecorded_cases`) unless ``--allow-unrecorded`` is
given.  A batch summary or manifest that cannot be read refuses the set
whatever ``--allow-unrecorded`` says (:func:`batch_read_errors`).

See ``docs/gate.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..authoring.models import GateSpec
from .criteria import (
    CaseChange,
    ConsistencyResult,
    CriterionChange,
    CriterionResult,
    _case_coverage,
    _criterion,
    _outcomes,
    _where,
    _written_at,
    consistency,
    find_gate_records,
    station_values,
)
from .errored import CASE_ERRORED, find_case_errored, read_case_errored
from .record import GATE_RECORD_SUFFIX, GateRecord, read_gate_record
from .sweep_plan import find_sweep_plans, read_sweep_plan
from .report import (
    case_changes,
    comparison_markdown,
    criterion_cell,
    criterion_changes,
    records_by_scenario,
    verdict_markdown,
)

__all__ = [
    "CaseChange",
    "ConsistencyResult",
    "CriterionChange",
    "GateComparison",
    "compare",
    "compare_all",
    "CriterionResult",
    "GateVerdict",
    "batch_read_errors",
    "consistency",
    "find_gate_records",
    "judge",
    "judge_all",
    "load_gate_records",
    "main",
    "station_values",
    "unrecorded_cases",
]

#: Exit codes of ``scenario-gate``.
EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_ERROR = 2


@dataclass
class GateVerdict:
    """A logical scenario's gate, judged over its runs.

    Attributes:
        scenario: The logical scenario's id.
        title: Its title.
        gate: The gate judged.
        records: The runs judged.
        criteria: Each criterion judged, in :data:`GATE_CRITERIA` order.
        warnings: What the judging noticed and went on past.
        errors: Files the judging needed and could not read (a run's
            coverage file); ``scenario-gate`` exits 2 on any.
    """

    scenario: str
    title: str
    gate: GateSpec
    records: list[GateRecord]
    criteria: list[CriterionResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every criterion holds."""
        return all(c.passed for c in self.criteria)

    @property
    def runs(self) -> int:
        return len(self.records)

    @property
    def passed_runs(self) -> int:
        return sum(1 for r in self.records if r.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "title": self.title,
            "passed": self.passed,
            "runs": self.runs,
            "passed_runs": self.passed_runs,
            "gate": self.gate.to_dict(),
            "criteria": [c.to_dict() for c in self.criteria],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "records": [str(r.path) for r in self.records if r.path is not None],
        }

    def to_markdown(self) -> str:
        return verdict_markdown(
            self.scenario,
            self.title,
            self.passed,
            self.passed_runs,
            self.runs,
            self.criteria,
            self.warnings,
            self.errors,
        )


def _gate_key(gate: Mapping[str, Any]) -> str:
    """*gate* (as a record stores it) in a form two gates compare equal in."""
    return json.dumps(gate, sort_keys=True)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def load_gate_records(paths: Iterable[Path]) -> tuple[list[GateRecord], list[str]]:
    """The records in *paths*, and why any file could not be read.

    A record whose gate is not a valid :class:`GateSpec` (a field it does not
    have, a value of the wrong type or out of range) cannot be read either:
    it is reported here, naming its file, rather than failing the judging of
    every record later.  It is refused even when a document's gate
    (``--document``) replaces the gates the records state: a record whose
    gate is malformed is corrupt, and judging it would hide that.
    """
    records: list[GateRecord] = []
    errors: list[str] = []
    for path in paths:
        try:
            record = read_gate_record(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        try:
            GateSpec.model_validate(record.gate)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            errors.append(f"{path}: not a valid gate ({exc})")
            continue
        records.append(record)
    return records, errors


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


def _slurm_unrecorded(directory: Path) -> tuple[list[str], int] | str:
    """The labels of a SLURM batch's cases that errored, have no outcome or
    one holding no verdict, and left no gate record at their stem; and how
    many cases it has.  Why the manifest cannot be read, instead, if it
    cannot."""
    from ..slurm.manifest import BatchLayout, recorded_verdict  # noqa: PLC0415

    layout = BatchLayout(directory)
    try:
        manifest = layout.read_manifest()
        cases = list(manifest["cases"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return f"{layout.manifest}: the SLURM manifest cannot be read ({exc})"
    unrecorded: list[str] = []
    for case in cases:
        try:
            index = int(case["index"])
        except (KeyError, TypeError, ValueError):
            return f"{layout.manifest}: a case without an index ({case!r})"
        record = layout.read_result(index)
        if (
            record is not None
            and recorded_verdict(record) is not None
            and not record.errored
        ):
            continue
        stem = case.get("stem")
        if stem and (directory / f"{stem}{GATE_RECORD_SUFFIX}").is_file():
            continue
        unrecorded.append(str(case.get("label") or case.get("scenario") or index))
    return unrecorded, len(cases)


def _recorded_at_stem(directory: Path, entry: dict[str, Any]) -> bool:
    """Whether a gate record is at the stem *entry* names in *directory*: a
    file stem of its own (no separator), as a batch writes it since #1957."""
    stem = entry.get("stem")
    if not isinstance(stem, str) or stem in ("", ".", "..") or "/" in stem:
        return False
    if "\\" in stem or "\0" in stem:
        return False
    return (directory / f"{stem}{GATE_RECORD_SUFFIX}").is_file()


def _summary_unrecorded(directory: Path) -> tuple[list[str], int] | str:
    """The names of a local batch's errored cases (``"error": true`` in its
    ``batch_results.json``), which raised and so, but for one that raised
    only once its run had ended, wrote no gate record, and of its pending
    ones (``"pending": true``: a batch cut short never finished them); and
    how many cases it has.  Why the summary cannot be read, instead.

    An errored case whose entry names its stem (``stem``, written since
    #1957) and has a gate record there is not counted: its run wrote one
    before it raised, and that record is judged, as a SLURM batch's is.  A
    pending case is counted whatever is at its stem: the batch never
    finished it."""
    path = directory / BATCH_RESULTS
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, ValueError) as exc:
        return f"{path}: the batch summary cannot be read ({exc})"
    if not isinstance(entries, list):
        return f"{path}: not a batch summary (not a JSON array)"
    unrecorded = [
        str(entry.get("scenario") or i)
        for i, entry in enumerate(entries)
        if not isinstance(entry, dict)
        or (entry.get("error") is True and not _recorded_at_stem(directory, entry))
        or entry.get("pending") is True
    ]
    return unrecorded, len(entries)


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
        if (
            data is not None
            and data.get("unfinished") is True
            and _has_result_since(marker)
        ):
            # The run wrote its result and was killed after it: it ran a
            # scenario without a gate, no concern of the gate.
            continue
        reason = data.get("reason") if data is not None else None
        name = (
            f"{job.name} ({reason})" if isinstance(reason, str) and reason else job.name
        )
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


def _scan_batches(path: Path) -> tuple[list[str], list[str]]:
    """What :func:`unrecorded_cases` and :func:`batch_read_errors` return for
    *path*, read once."""
    if not path.is_dir():
        return [], []
    lines: list[str] = []
    unreadable: list[str] = []
    for directory in _batch_dirs(path):
        if (directory / "slurm" / "manifest.json").is_file():
            found = _slurm_unrecorded(directory)
        else:
            found = _summary_unrecorded(directory)
        if isinstance(found, str):
            unreadable.append(found)
            continue
        names, total = found
        if not names:
            continue
        shown = ", ".join(names[:_NAMED])
        more = f", and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
        lines.append(
            f"{directory}: {len(names)} of {total} case(s) raised or have no "
            f"outcome, and left no gate record ({shown}{more}); the gate would "
            "be judged over the runs that survived"
        )
    lines.extend(_errored_jobs(path))
    unstarted, unreadable_plans = _unstarted_jobs(path)
    lines.extend(unstarted)
    unreadable.extend(unreadable_plans)
    return lines, unreadable


def unrecorded_cases(path: Path) -> list[str]:
    """Why the runs under *path* cannot be judged as they are: one line per
    batch under it whose cases raised rather than ran, were never finished
    or (SLURM) have no outcome, and left no gate record, one per
    ``--multirun`` sweep whose jobs were given up or did not finish, and one
    per sweep whose plan lists jobs that never ran.

    A run that raises writes no record, so judging the records there are
    would compute ``min_success_rate`` over the runs that survived.  A
    SLURM batch (``slurm/manifest.json``) is read from its tasks' outcomes,
    collected or not, and a case with a record at its stem is not counted; a
    local batch is read from its ``batch_results.json``, whose errored and
    pending entries count, but an errored one with a gate record at the stem
    its entry names (an entry of a summary written before #1957 names none,
    and is counted); a sweep is read from its jobs'
    markers (``case_errored.json``).  A job whose directory holds a gate
    record is not counted.  Nor is one whose marker says it did not finish
    (``"unfinished": true``) beside a result (``*_result.json``) written no
    earlier than the marker: a scenario without a gate, killed after
    writing its result (#1840).  A marker of a job given up, or one that
    cannot be read, counts whatever results are beside it.  A lanelet sweep
    is read from its plan (``sweep_plan.json``): a planned job whose
    directory holds no gate record, no marker and no result counts as one
    that never ran (#1833).
    Empty for a gate record (a file), and for a directory holding no batch,
    no marker and no plan.  A summary, manifest or plan that cannot be read
    is not reported here but by :func:`batch_read_errors`.
    """
    return _scan_batches(path)[0]


def batch_read_errors(path: Path) -> list[str]:
    """One line per batch under *path* whose summary (``batch_results.json``)
    or SLURM manifest cannot be read, and per sweep whose plan
    (``sweep_plan.json``) cannot be: what it left unrecorded cannot be told,
    and ``scenario-gate`` exits 2 whatever ``--allow-unrecorded`` says."""
    return _scan_batches(path)[1]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def _coverage_criteria(
    gate: GateSpec,
    records: Sequence[GateRecord],
    warnings: list[str],
    errors: list[str],
) -> list[CriterionResult]:
    """The coverage grades the gate asks for, from the records' coverage files
    merged.

    A coverage file a record names that cannot be read (openodd's loader
    skips it), or files that cannot be merged, go into *errors*: the grades
    are judged on what was read, and ``scenario-gate`` exits 2.
    """
    groups = [
        (name, group, threshold)
        for name, group, threshold in (
            ("min_scenario_coverage", "scenario", gate.min_scenario_coverage),
            ("min_odd_coverage", "odd", gate.min_odd_coverage),
        )
        if threshold is not None
    ]
    if not groups and not gate.coverage_items:
        return []
    from ..coverage.report import (  # noqa: PLC0415
        load_coverage_files,
        merge_coverage,
    )

    paths: list[Path] = []
    for record in records:
        path = record.file("coverage")
        if path is not None and path.is_file():
            paths.append(path)
        else:
            warnings.append(f"{record.path}: no coverage file beside it")
    report = None
    if paths:
        documents = load_coverage_files(paths)
        read = {p.resolve() for p, _ in documents}
        errors.extend(
            f"{p}: the coverage file cannot be read"
            for p in paths
            if p.resolve() not in read
        )
        try:
            report = merge_coverage([doc for _, doc in documents])
        except Exception as exc:  # noqa: BLE001 -- reported as the exit code says
            names = ", ".join(str(p) for p, _ in documents)
            errors.append(f"{names}: the coverage files cannot be merged ({exc})")
    merged_groups = report.groups() if report is not None else []
    detail = f"{len(paths)} coverage file(s)"
    out: list[CriterionResult] = []
    for name, group, threshold in groups:
        if report is None or group not in merged_groups:
            value, why = None, f"no {group} cover item in {detail}"
        else:
            value, why = report.group_grade(group), detail
        out.append(_criterion(name, value, ">=", threshold, why))
    for entry in gate.coverage_items:
        merged = report.entry(entry.item) if report is not None else None
        label = f"coverage_items[{entry.item}]"
        if merged is None:
            out.append(
                _criterion(label, None, ">=", entry.min, f"no such item in {detail}")
            )
        else:
            holes = f"{len(merged.holes)} hole(s)"
            out.append(_criterion(label, merged.grade, ">=", entry.min, holes))
    return out


def judge(
    records: Sequence[GateRecord], gate: Optional[GateSpec] = None
) -> GateVerdict:
    """Judge one logical scenario's gate over *records*, its runs.

    *gate* replaces the one the records state (a document's, edited since).
    Without one, the gate of the latest record (by file time) is judged, and
    a warning says so when the records state different ones.

    Raises:
        ValueError: If there is no record, or the records are of different
            scenarios.
    """
    if not records:
        raise ValueError("no gate record to judge")
    ids = {r.scenario for r in records}
    if len(ids) != 1:
        raise ValueError(f"records of more than one scenario: {sorted(ids)}")
    warnings: list[str] = []
    latest = max(records, key=_written_at)
    if gate is None:
        stated = {_gate_key(r.gate) for r in records}
        if len(stated) > 1:
            warnings.append(
                f"the runs state {len(stated)} different gates; judging the "
                f"latest one's ({latest.path})"
            )
        gate = GateSpec.model_validate(latest.gate)
    digests = {r.digest for r in records}
    if len(digests) > 1:
        warnings.append(f"the runs ran {len(digests)} versions of the document")
    verdict = GateVerdict(
        scenario=latest.scenario,
        title=latest.title,
        gate=gate,
        records=list(records),
        warnings=warnings,
    )
    runs, passed = verdict.runs, verdict.passed_runs
    out = verdict.criteria
    out.append(_criterion("min_runs", runs, ">=", gate.min_runs, f"{runs} run(s)"))
    if gate.min_success_rate is not None:
        rate = passed / runs
        detail = f"{passed} of {runs} passed"
        out.append(
            _criterion("min_success_rate", rate, ">=", gate.min_success_rate, detail)
        )
    if gate.min_case_coverage is not None:
        value, detail = _case_coverage(records)
        out.append(
            _criterion("min_case_coverage", value, ">=", gate.min_case_coverage, detail)
        )
    out.extend(_coverage_criteria(gate, records, warnings, verdict.errors))
    if gate.judges_consistency:
        spread = consistency(
            records, gate.consistency_stations, gate.consistency_min_runs
        )
        for name, threshold, value, at in (
            (
                "max_lateral_spread_m",
                gate.max_lateral_spread_m,
                spread.lateral_spread_m,
                spread.lateral_at,
            ),
            (
                "max_speed_spread_mps",
                gate.max_speed_spread_mps,
                spread.speed_spread_mps,
                spread.speed_at,
            ),
        ):
            if threshold is None:
                continue
            if spread.stations == 0:
                why = f"no station was passed by {gate.consistency_min_runs} run(s)"
            else:
                why = f"{_where(at)}, {spread.stations} station(s)"
            out.append(_criterion(name, value, "<=", threshold, why))
    return verdict


def judge_all(
    records: Sequence[GateRecord],
    gate: Optional[GateSpec] = None,
    scenario: Optional[str] = None,
) -> list[GateVerdict]:
    """Judge every logical scenario *records* are runs of, one verdict each,
    sorted by id; only *scenario* when it is given."""
    groups = _by_scenario(records, scenario)
    return [judge(groups[name], gate) for name in sorted(groups)]


# ---------------------------------------------------------------------------
# Regression: a baseline's runs against a candidate's
# ---------------------------------------------------------------------------


@dataclass
class GateComparison:
    """One logical scenario's gate on a baseline's runs and a candidate's.

    The candidate is judged against the same gate as the baseline, so what
    moved is the runs, not the thresholds.

    Attributes:
        scenario: The logical scenario's id.
        title: Its title.
        baseline: The gate on the baseline's runs; ``None`` when it has none.
        candidate: The gate on the candidate's runs; ``None`` when it has none.
        criteria: Each criterion, compared.
        newly_failing: Cases every baseline run of passed, and a candidate run
            of failed.
        fixed: Cases a baseline run of failed, and every candidate run of
            passed.
        warnings: What the comparison noticed and went on past.
    """

    scenario: str
    title: str
    baseline: Optional[GateVerdict]
    candidate: Optional[GateVerdict]
    criteria: list[CriterionChange] = field(default_factory=list)
    newly_failing: list[CaseChange] = field(default_factory=list)
    fixed: list[CaseChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        """Whether the candidate is worse off: the scenario did not run, a
        criterion that passed fails, or a case that passed fails."""
        if self.candidate is None:
            return True
        return bool(self.newly_failing) or any(
            c.change == "regressed" for c in self.criteria
        )

    @property
    def passed(self) -> bool:
        """Whether the candidate passes the gate.

        A regression inside the gate -- a case that broke while the success
        rate still clears it -- does not fail it; :meth:`ok` with *strict*
        does (``scenario-gate --fail-on-regression``).
        """
        return self.candidate is not None and self.candidate.passed

    @property
    def errors(self) -> list[str]:
        """What either side's judging could not read."""
        return [
            *(self.baseline.errors if self.baseline is not None else []),
            *(self.candidate.errors if self.candidate is not None else []),
        ]

    def ok(self, strict: bool = False) -> bool:
        """Whether the comparison passes: the candidate passes the gate, and
        with *strict* nothing regressed either."""
        return self.passed and not (strict and self.regressed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "title": self.title,
            "passed": self.passed,
            "regressed": self.regressed,
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "criteria": [c.to_dict() for c in self.criteria],
            "newly_failing": [c.to_dict() for c in self.newly_failing],
            "fixed": [c.to_dict() for c in self.fixed],
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        return comparison_markdown(
            self.scenario,
            self.title,
            _summary(self.baseline),
            _summary(self.candidate),
            self.regressed,
            self.criteria,
            self.newly_failing,
            self.fixed,
            self.warnings,
        )


def _summary(verdict: Optional[GateVerdict]) -> Optional[tuple[int, int, bool]]:
    """``(passed runs, runs, whether the gate passed)`` of *verdict*."""
    if verdict is None:
        return None
    return verdict.passed_runs, verdict.runs, verdict.passed


def _cell(result: Optional[CriterionResult]) -> str:
    return criterion_cell(result)


def compare(
    baseline: Sequence[GateRecord],
    candidate: Sequence[GateRecord],
    gate: Optional[GateSpec] = None,
) -> GateComparison:
    """Compare one logical scenario's runs: *baseline*'s against *candidate*'s.

    Both are judged against one gate: *gate*, else the one the candidate's
    latest record states (the baseline's when the candidate has no run).  A
    baseline whose runs stated another gate gets a warning: its runs are
    re-judged, not taken as they were judged then.

    Raises:
        ValueError: If both are empty, or they hold more than one scenario.
    """
    ids = {r.scenario for r in [*baseline, *candidate]}
    if len(ids) != 1:
        raise ValueError(f"expected the runs of one scenario, got {sorted(ids)}")
    warnings: list[str] = []
    after = judge(candidate, gate) if candidate else None
    used = gate if gate is not None else (after.gate if after is not None else None)
    before = judge(baseline, used) if baseline else None
    if gate is None and after is not None and before is not None:
        stated = {_gate_key(r.gate) for r in baseline}
        if stated != {_gate_key(after.gate.to_dict())}:
            warnings.append(
                "the baseline's runs stated another gate; both are judged "
                "against the candidate's"
            )
    if before is not None and after is not None:
        if {r.digest for r in baseline} != {r.digest for r in candidate}:
            warnings.append(
                "the baseline and the candidate ran different versions of the "
                "document"
            )
    # The candidate's order, then what only the baseline judged.
    criteria = criterion_changes(
        before.criteria if before else [], after.criteria if after else []
    )
    newly_failing, fixed, missing = case_changes(
        _outcomes(baseline), _outcomes(candidate)
    )
    if missing and after is not None:
        warnings.append(f"{missing} case(s) the baseline ran did not run")
    latest = after or before
    assert latest is not None  # noqa: S101 -- one side has runs
    return GateComparison(
        scenario=latest.scenario,
        title=latest.title,
        baseline=before,
        candidate=after,
        criteria=criteria,
        newly_failing=newly_failing,
        fixed=fixed,
        warnings=[*(before.warnings if before else []), *warnings],
    )


def compare_all(
    baseline: Sequence[GateRecord],
    candidate: Sequence[GateRecord],
    gate: Optional[GateSpec] = None,
    scenario: Optional[str] = None,
) -> list[GateComparison]:
    """Compare every logical scenario either side ran, one comparison each,
    sorted by id; only *scenario* when it is given."""
    before, after = _by_scenario(baseline, scenario), _by_scenario(candidate, scenario)
    return [
        compare(before.get(name, []), after.get(name, []), gate)
        for name in sorted(before.keys() | after.keys())
    ]


def _by_scenario(
    records: Sequence[GateRecord], scenario: Optional[str] = None
) -> dict[str, list[GateRecord]]:
    """*records* grouped by scenario id; only *scenario*'s when it is given."""
    return records_by_scenario(records, scenario)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser(prog: Optional[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Judge logical scenarios' gates from their runs' gate records "
            "({stem}_gate.json): minimum runs, success rate, case and cover "
            "coverage, and trajectory consistency (docs/gate.md). Given two "
            "sets of runs, judge both against one gate and report what "
            "regressed from the first (the baseline) to the second (the "
            "candidate)."
        ),
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        metavar="RUNS",
        help=(
            "a set of runs: a directory searched for *_gate.json, or one gate "
            "record. One set is judged; two are BASELINE CANDIDATE"
        ),
    )
    parser.add_argument(
        "--document",
        type=Path,
        help=(
            "judge against this document's gate instead of the one the runs "
            "recorded (its id selects the runs)"
        ),
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=(
            "with two sets: exit 1 on any regression (a criterion that passed "
            "and fails, a newly failing case), even when the candidate passes "
            "the gate. By default only the candidate's gate decides"
        ),
    )
    parser.add_argument(
        "--allow-unrecorded",
        action="store_true",
        help=(
            "judge the records there are even when a batch among the runs "
            "lists cases that raised (ERROR), were never finished (pending) "
            "or, under SLURM, have no outcome, or a --multirun job was given "
            "up or did not finish (case_errored.json) or never ran "
            "(sweep_plan.json), and left no gate record; warned of on "
            "stderr. By default such a set is refused (exit 2). A batch "
            "summary, manifest or sweep plan that cannot be read is refused "
            "either way"
        ),
    )
    parser.add_argument("--scenario", help="judge only the runs of this scenario id")
    parser.add_argument("--json", type=Path, help="also write the verdicts as JSON")
    parser.add_argument(
        "--markdown", type=Path, help="write the report to this file, not stdout"
    )
    return parser


def _load(path: Path, errors: list[str]) -> list[GateRecord]:
    records, failed = load_gate_records(find_gate_records([path]))
    errors.extend(failed)
    return records


def _write_output(path: Path, text: str, errors: list[str]) -> None:
    """Write ``text`` to ``path``; on failure, append why to ``errors``."""
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot write {path}: {exc.strerror or exc}")


def main(
    argv: Optional[Sequence[str]] = None, prog: Optional[str] = "scenario-gate"
) -> int:
    """``scenario-gate RUNS`` or ``scenario-gate BASELINE CANDIDATE``.

    0 when every gate passes (on the candidate, with two sets), 1 when one
    fails -- or, with ``--fail-on-regression``, something regressed -- and 2
    when there is nothing to judge, a file cannot be read (a batch summary,
    SLURM manifest or sweep plan among the runs too, :func:`batch_read_errors`, whatever
    ``--allow-unrecorded`` says), a batch or sweep among the runs lists cases
    that left no record (:func:`unrecorded_cases`; unless
    ``--allow-unrecorded``), or an output (``--markdown``, ``--json``)
    cannot be written.
    """
    parser = _parser(prog)
    args = parser.parse_args(argv)
    if len(args.runs) > 2:
        parser.error(
            "takes one set of runs, or two to compare (BASELINE CANDIDATE); "
            f"got {len(args.runs)}"
        )
    gate: Optional[GateSpec] = None
    scenario: Optional[str] = args.scenario
    if args.document is not None:
        from ..authoring.persistence import load_document  # noqa: PLC0415

        try:
            document = load_document(args.document)
        except Exception as exc:  # noqa: BLE001 -- reported as the exit code says
            print(f"scenario-gate: {args.document}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        if document.gate is None:
            print(
                f"scenario-gate: {args.document}: the document has no gate",
                file=sys.stderr,
            )
            return EXIT_ERROR
        gate = document.gate
        scenario = scenario or document.id
    scans = [_scan_batches(path) for path in args.runs]
    unrecorded = [line for found, _ in scans for line in found]
    unreadable = [line for _, failed in scans for line in failed]
    for line in unreadable:
        print(
            f"scenario-gate: {line}; the cases it left without a record cannot "
            "be told, so the runs are not judged (--allow-unrecorded does not "
            "apply). Repair the file, or run the batch or sweep again",
            file=sys.stderr,
        )
    if unrecorded and not args.allow_unrecorded:
        for line in unrecorded:
            print(
                f"scenario-gate: {line}. Run them again, or pass "
                "--allow-unrecorded to judge the records there are",
                file=sys.stderr,
            )
        return EXIT_ERROR
    if unreadable:
        return EXIT_ERROR
    for line in unrecorded:
        print(f"scenario-gate: warning: {line}", file=sys.stderr)
    errors: list[str] = []
    sets = [_load(path, errors) for path in args.runs]
    for error in errors:
        print(f"scenario-gate: {error}", file=sys.stderr)
    reports: Sequence[GateVerdict | GateComparison]
    strict = bool(args.fail_on_regression)
    if strict and len(sets) == 1:
        parser.error("--fail-on-regression needs two sets (BASELINE CANDIDATE)")
    if len(sets) == 1:
        reports = judge_all(sets[0], gate, scenario)
        title = "# Logical scenario gates"
        payload: dict[str, Any] = {"verdicts": [r.to_dict() for r in reports]}
    else:
        reports = compare_all(sets[0], sets[1], gate, scenario)
        title = f"# Logical scenario gates: {args.runs[1]} against {args.runs[0]}"
        payload = {
            "baseline": str(args.runs[0]),
            "candidate": str(args.runs[1]),
            "comparisons": [r.to_dict() for r in reports],
        }
    if not reports:
        print("scenario-gate: no gate record to judge", file=sys.stderr)
        return EXIT_ERROR
    markdown = title + "\n\n" + "\n\n".join(r.to_markdown() for r in reports)
    # Both outputs are attempted, so one line names every path that cannot
    # be written; the verdict printed to stdout is unchanged either way.
    unwritable: list[str] = []
    if args.markdown is not None:
        _write_output(args.markdown, markdown + "\n", unwritable)
    else:
        print(markdown)
    if args.json is not None:
        _write_output(args.json, json.dumps(payload, indent=2), unwritable)
    for error in unwritable:
        print(f"scenario-gate: {error}", file=sys.stderr)
    # What the judging could not read (a run's coverage file), as the records
    # main could not read: reported, and the exit code says so.
    unread = [e for r in reports for e in r.errors]
    for error in unread:
        print(f"scenario-gate: {error}", file=sys.stderr)
    if errors or unread or unwritable:
        return EXIT_ERROR
    passed = all(
        r.ok(strict) if isinstance(r, GateComparison) else r.passed for r in reports
    )
    return EXIT_PASSED if passed else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
