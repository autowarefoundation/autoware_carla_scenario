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
import dataclasses
import json
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..authoring.models import GateSpec
from .record import GATE_RECORD_SUFFIX, GateRecord, read_gate_record

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


@dataclass(frozen=True)
class CriterionResult:
    """One criterion of a gate, judged.

    Attributes:
        name: The criterion (a :class:`GateSpec` field; a coverage item's is
            ``coverage_items[<item>]``).
        passed: Whether it holds.
        value: What the runs measured; ``None`` when they could not measure it.
        threshold: What the gate asks.
        comparison: ``>=`` or ``<=``: how *value* is held to *threshold*.
        detail: Why, in words.
    """

    name: str
    passed: bool
    value: Optional[float]
    threshold: float
    comparison: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ConsistencyResult:
    """The spread of the ego's drive across runs, station by station.

    Attributes:
        stations: How many stations enough runs passed to be measured.
        lateral_spread_m: The largest standard deviation of the lateral
            offset at a measured station, or ``None`` when none was.
        lateral_at: Where: ``(segment, fraction of it)``.
        speed_spread_mps: The same for the speed.
        speed_at: Where.
    """

    stations: int = 0
    lateral_spread_m: Optional[float] = None
    lateral_at: Optional[tuple[int, float]] = None
    speed_spread_mps: Optional[float] = None
    speed_at: Optional[tuple[int, float]] = None


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
    """

    scenario: str
    title: str
    gate: GateSpec
    records: list[GateRecord]
    criteria: list[CriterionResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

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
            "records": [str(r.path) for r in self.records if r.path is not None],
        }

    def to_markdown(self) -> str:
        mark = "PASSED" if self.passed else "FAILED"
        lines = [
            f"## {self.title or self.scenario} (`{self.scenario}`): {mark}",
            "",
            f"{self.passed_runs} of {self.runs} run(s) passed.",
            "",
            "| Criterion | Measured | Required | Result |",
            "|---|---|---|---|",
        ]
        for c in self.criteria:
            value = "-" if c.value is None else _number(c.value)
            result = "pass" if c.passed else "**FAIL**"
            detail = f" ({c.detail})" if c.detail else ""
            lines.append(
                f"| {c.name} | {value}{detail} | {c.comparison} "
                f"{_number(c.threshold)} | {result} |"
            )
        lines += _warning_lines(self.warnings)
        return "\n".join(lines)


def _number(value: float) -> str:
    return f"{value:.4g}"


def _warning_lines(warnings: Sequence[str]) -> list[str]:
    """A report's warnings, after a blank line; none without any."""
    return ["", *(f"- warning: {w}" for w in warnings)] if warnings else []


def _gate_key(gate: Mapping[str, Any]) -> str:
    """*gate* (as a record stores it) in a form two gates compare equal in."""
    return json.dumps(gate, sort_keys=True)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def find_gate_records(paths: Iterable[Path]) -> list[Path]:
    """Every gate record *paths* name: a file as it is, a directory searched
    through (``*_gate.json``), sorted and without repeats."""
    found: set[Path] = set()
    for path in paths:
        if path.is_dir():
            found.update(path.rglob(f"*{GATE_RECORD_SUFFIX}"))
        else:
            found.add(path)
    return sorted(found)


def load_gate_records(paths: Iterable[Path]) -> tuple[list[GateRecord], list[str]]:
    """The records in *paths*, and why any file could not be read."""
    records: list[GateRecord] = []
    errors: list[str] = []
    for path in paths:
        try:
            records.append(read_gate_record(path))
        except ValueError as exc:
            errors.append(str(exc))
    return records, errors


# ---------------------------------------------------------------------------
# Trajectory consistency
# ---------------------------------------------------------------------------


def _stations(segment_ends: Sequence[float], per_segment: int) -> list[float]:
    """Route s of each station: *per_segment* in each segment, each in the
    middle of an equal share of it."""
    out: list[float] = []
    start = 0.0
    for end in segment_ends:
        for j in range(per_segment):
            out.append(start + (j + 0.5) / per_segment * (end - start))
        start = end
    return out


def station_values(
    record: GateRecord, per_segment: int
) -> dict[tuple[int, int], tuple[float, float]]:
    """``(lateral offset, speed)`` of *record*'s ego at each station it drove
    past, by ``(segment, station in it)``.

    A station is passed between two rows the first time the route s goes from
    below it to it or beyond (after the one before it); the values are
    interpolated there.  A station before the ego's first row -- behind its
    spawn -- or past its last one is not passed.
    """
    ends = record.case.segment_ends
    stations = _stations(ends, per_segment)
    rows = record.ego
    out: dict[tuple[int, int], tuple[float, float]] = {}
    first = 1
    for number, s in enumerate(stations):
        i = first
        while i < len(rows) and not (rows[i - 1][1] < s <= rows[i][1]):
            i += 1
        if i >= len(rows):
            continue  # not passed (behind the spawn, past the last row)
        first = i
        (_, s0, lat0, v0), (_, s1, lat1, v1) = rows[i - 1], rows[i]
        f = (s - s0) / (s1 - s0)
        out[(number // per_segment, number % per_segment)] = (
            lat0 + f * (lat1 - lat0),
            v0 + f * (v1 - v0),
        )
    return out


def consistency(
    records: Sequence[GateRecord], per_segment: int, min_runs: int
) -> ConsistencyResult:
    """The spread of the ego's lateral offset and speed across *records*.

    Each station counts once *min_runs* runs or more passed it; its spread is
    the population standard deviation of their values.
    """
    by_station: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for record in records:
        for key, at_station in station_values(record, per_segment).items():
            by_station.setdefault(key, []).append(at_station)
    # (lateral spread, speed spread, where) of each station enough runs passed.
    spreads = [
        (
            statistics.pstdev(v[0] for v in passed),
            statistics.pstdev(v[1] for v in passed),
            (segment, (j + 0.5) / per_segment),
        )
        for (segment, j), passed in sorted(by_station.items())
        if len(passed) >= min_runs
    ]
    if not spreads:
        return ConsistencyResult()
    # max() keeps the first of equals: the station furthest back.
    lateral = max(spreads, key=lambda row: row[0])
    speed = max(spreads, key=lambda row: row[1])
    return ConsistencyResult(len(spreads), lateral[0], lateral[2], speed[1], speed[2])


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def _criterion(
    name: str, value: Optional[float], comparison: str, threshold: float, detail: str
) -> CriterionResult:
    """*name* judged: *value* held ``>=`` or ``<=`` *threshold* (within
    1e-12), failing when the runs measured nothing."""
    if value is None:
        passed = False
    elif comparison == ">=":
        passed = value + 1e-12 >= threshold
    else:
        passed = value <= threshold + 1e-12
    return CriterionResult(name, passed, value, threshold, comparison, detail)


def _case_coverage(records: Sequence[GateRecord]) -> tuple[Optional[float], str]:
    """The share of the cases that ran, over every map the runs were on."""
    counts: dict[str, int] = {}
    run: dict[str, set[tuple[int, ...]]] = {}
    for record in records:
        case = record.case
        counts[case.map] = max(counts.get(case.map, 0), case.case_count)
        run.setdefault(case.map, set()).add(case.lanelet_ids)
    unknown = sorted(name or "?" for name, count in counts.items() if count <= 0)
    if unknown:
        return None, f"the case count on {', '.join(unknown)} is not known"
    total = sum(counts.values())
    covered = sum(min(len(run[name]), counts[name]) for name in counts)
    maps = len(counts)
    return covered / total, f"{covered} of {total} case(s) on {maps} map(s)"


def _coverage_criteria(
    gate: GateSpec, records: Sequence[GateRecord], warnings: list[str]
) -> list[CriterionResult]:
    """The coverage grades the gate asks for, from the records' coverage files
    merged."""
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
    from ..coverage.report import load_and_merge  # noqa: PLC0415

    paths: list[Path] = []
    for record in records:
        path = record.file("coverage")
        if path is not None and path.is_file():
            paths.append(path)
        else:
            warnings.append(f"{record.path}: no coverage file beside it")
    report = load_and_merge(paths) if paths else None
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


def _where(at: Optional[tuple[int, float]]) -> str:
    if at is None:
        return ""
    return f"segment {at[0]} at {at[1]:.0%}"


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
    out.extend(_coverage_criteria(gate, records, warnings))
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


def _written_at(record: GateRecord) -> float:
    try:
        return record.path.stat().st_mtime if record.path is not None else 0.0
    except OSError:
        return 0.0


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


@dataclass(frozen=True)
class CriterionChange:
    """One criterion, judged on the baseline's runs and on the candidate's."""

    name: str
    baseline: Optional[CriterionResult]
    candidate: Optional[CriterionResult]

    @property
    def change(self) -> str:
        """How the criterion changed from the baseline to the candidate.

        ``regressed`` / ``improved``: it passed and fails now, or the other
        way; ``worse`` / ``better`` / ``same``: its verdict held and its value
        moved (or not); ``new`` / ``dropped``: only the candidate / the
        baseline judged it.
        """
        before, after = self.baseline, self.candidate
        if after is None:
            return "dropped"
        if before is None:
            return "new"
        if before.passed and not after.passed:
            return "regressed"
        if after.passed and not before.passed:
            return "improved"
        delta = self.delta
        if delta is None:
            if before.value is None and after.value is None:
                return "same"
            return "worse" if after.value is None else "better"
        if abs(delta) <= 1e-12:
            return "same"
        # ">=": higher is better; "<=": lower is.
        higher_is_better = after.comparison == ">="
        return "better" if (delta > 0) == higher_is_better else "worse"

    @property
    def delta(self) -> Optional[float]:
        """The candidate's value less the baseline's, when both have one."""
        if self.baseline is None or self.candidate is None:
            return None
        if self.baseline.value is None or self.candidate.value is None:
            return None
        return self.candidate.value - self.baseline.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "change": self.change,
            "delta": self.delta,
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
        }


@dataclass(frozen=True)
class CaseChange:
    """A case whose outcome changed: its runs in the baseline and the candidate.

    Attributes:
        map: The case's map.
        lanelet_ids: Its route's lanelets.
        baseline: ``(passed runs, runs)`` in the baseline.
        candidate: ``(passed runs, runs)`` in the candidate.
    """

    map: str
    lanelet_ids: tuple[int, ...]
    baseline: tuple[int, int]
    candidate: tuple[int, int]

    def label(self) -> str:
        ids = ",".join(str(v) for v in self.lanelet_ids)
        return f"{self.map or '?'} [{ids}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map,
            "lanelet_ids": list(self.lanelet_ids),
            "baseline": {"passed": self.baseline[0], "runs": self.baseline[1]},
            "candidate": {"passed": self.candidate[0], "runs": self.candidate[1]},
        }


def _outcomes(
    records: Sequence[GateRecord],
) -> dict[tuple[str, tuple[int, ...]], tuple[int, int]]:
    """``(passed runs, runs)`` of each case."""
    out: dict[tuple[str, tuple[int, ...]], tuple[int, int]] = {}
    for record in records:
        passed, runs = out.get(record.case.key, (0, 0))
        out[record.case.key] = (passed + int(record.passed), runs + 1)
    return out


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
        if self.candidate is None:
            verdict = "NOT RUN"
        else:
            verdict = "PASSED" if self.candidate.passed else "FAILED"
        trend = "REGRESSED" if self.regressed else "no regression"
        lines = [
            f"## {self.title or self.scenario} (`{self.scenario}`): "
            f"{verdict}, {trend}",
            "",
        ]
        for name, verdict_of in (
            ("Baseline", self.baseline),
            ("Candidate", self.candidate),
        ):
            if verdict_of is None:
                lines.append(f"- {name}: no run")
            else:
                mark = "passed" if verdict_of.passed else "failed"
                lines.append(
                    f"- {name}: {verdict_of.passed_runs} of {verdict_of.runs} "
                    f"run(s) passed; gate {mark}"
                )
        lines += [
            "",
            "| Criterion | Required | Baseline | Candidate | Change |",
            "|---|---|---|---|---|",
        ]
        for c in self.criteria:
            ref = c.candidate or c.baseline
            assert ref is not None  # noqa: S101 -- one side judged it
            required = f"{ref.comparison} {_number(ref.threshold)}"
            change = c.change
            if change == "regressed":
                change = "**regressed**"
            lines.append(
                f"| {c.name} | {required} | {_cell(c.baseline)} | "
                f"{_cell(c.candidate)} | {change} |"
            )
        for title, cases in (
            ("Newly failing cases", self.newly_failing),
            ("Fixed cases", self.fixed),
        ):
            if cases:
                lines += ["", f"{title}:", ""]
                lines += [
                    f"- {case.label()}: baseline {case.baseline[0]}/"
                    f"{case.baseline[1]} passed, candidate {case.candidate[0]}/"
                    f"{case.candidate[1]}"
                    for case in cases
                ]
        lines += _warning_lines(self.warnings)
        return "\n".join(lines)


def _cell(result: Optional[CriterionResult]) -> str:
    if result is None:
        return "-"
    value = "n/a" if result.value is None else _number(result.value)
    return value if result.passed else f"{value} (fail)"


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
    before_by_name = {c.name: c for c in (before.criteria if before else [])}
    after_by_name = {c.name: c for c in (after.criteria if after else [])}
    # The candidate's order, then what only the baseline judged.
    order = [*after_by_name, *(n for n in before_by_name if n not in after_by_name)]
    criteria = [
        CriterionChange(name, before_by_name.get(name), after_by_name.get(name))
        for name in order
    ]
    newly_failing: list[CaseChange] = []
    fixed: list[CaseChange] = []
    before_cases, after_cases = _outcomes(baseline), _outcomes(candidate)
    for key in sorted(before_cases.keys() & after_cases.keys()):
        b, a = before_cases[key], after_cases[key]
        change = CaseChange(key[0], key[1], b, a)
        if b[0] == b[1] and a[0] < a[1]:
            newly_failing.append(change)
        elif b[0] < b[1] and a[0] == a[1]:
            fixed.append(change)
    missing = len(before_cases.keys() - after_cases.keys())
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
    groups: dict[str, list[GateRecord]] = {}
    for record in records:
        if scenario is None or record.scenario == scenario:
            groups.setdefault(record.scenario, []).append(record)
    return groups


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


def main(
    argv: Optional[Sequence[str]] = None, prog: Optional[str] = "scenario-gate"
) -> int:
    """``scenario-gate RUNS`` or ``scenario-gate BASELINE CANDIDATE``.

    0 when every gate passes (on the candidate, with two sets), 1 when one
    fails -- or, with ``--fail-on-regression``, something regressed -- and 2
    when there is nothing to judge or a file cannot be read.
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
    if args.markdown is not None:
        args.markdown.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if errors:
        return EXIT_ERROR
    passed = all(
        r.ok(strict) if isinstance(r, GateComparison) else r.passed for r in reports
    )
    return EXIT_PASSED if passed else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
