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
from .record import GateRecord, read_gate_record
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
    "consistency",
    "find_gate_records",
    "judge",
    "judge_all",
    "load_gate_records",
    "main",
    "station_values",
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
    when there is nothing to judge, a file cannot be read, or an output
    (``--markdown``, ``--json``) cannot be written.
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
