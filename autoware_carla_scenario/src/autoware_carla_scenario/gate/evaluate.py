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

See ``docs/gate.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..authoring.models import GateSpec
from .record import GATE_RECORD_SUFFIX, GateRecord, read_gate_record

__all__ = [
    "ConsistencyResult",
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
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "detail": self.detail,
        }


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
        if self.warnings:
            lines.append("")
            lines.extend(f"- warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def _number(value: float) -> str:
    return f"{value:.4g}"


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
    measured = 0
    lateral: Optional[float] = None
    lateral_at: Optional[tuple[int, float]] = None
    speed: Optional[float] = None
    speed_at: Optional[tuple[int, float]] = None
    for (segment, j), passed in sorted(by_station.items()):
        if len(passed) < min_runs:
            continue
        measured += 1
        where = (segment, (j + 0.5) / per_segment)
        lat = statistics.pstdev(v[0] for v in passed)
        spd = statistics.pstdev(v[1] for v in passed)
        if lateral is None or lat > lateral:
            lateral, lateral_at = lat, where
        if speed is None or spd > speed:
            speed, speed_at = spd, where
    return ConsistencyResult(measured, lateral, lateral_at, speed, speed_at)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def _at_least(
    name: str, value: Optional[float], threshold: float, detail: str
) -> CriterionResult:
    passed = value is not None and value + 1e-12 >= threshold
    return CriterionResult(name, passed, value, threshold, ">=", detail)


def _at_most(
    name: str, value: Optional[float], threshold: float, detail: str
) -> CriterionResult:
    passed = value is not None and value <= threshold + 1e-12
    return CriterionResult(name, passed, value, threshold, "<=", detail)


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
    asked = (
        gate.min_scenario_coverage is not None
        or gate.min_odd_coverage is not None
        or bool(gate.coverage_items)
    )
    if not asked:
        return []
    from openodd.report import load_and_merge  # noqa: PLC0415

    paths: list[Path] = []
    for record in records:
        path = record.file("coverage")
        if path is not None and path.is_file():
            paths.append(path)
        else:
            warnings.append(f"{record.path}: no coverage file beside it")
    report = load_and_merge(paths) if paths else None
    groups = report.groups() if report is not None else []
    detail = f"{len(paths)} coverage file(s)"
    out: list[CriterionResult] = []
    for name, group, threshold in (
        ("min_scenario_coverage", "scenario", gate.min_scenario_coverage),
        ("min_odd_coverage", "odd", gate.min_odd_coverage),
    ):
        if threshold is None:
            continue
        if report is None or group not in groups:
            out.append(
                _at_least(name, None, threshold, f"no {group} cover item in {detail}")
            )
        else:
            out.append(_at_least(name, report.group_grade(group), threshold, detail))
    for entry in gate.coverage_items:
        merged = report.entry(entry.item) if report is not None else None
        label = f"coverage_items[{entry.item}]"
        if merged is None:
            out.append(_at_least(label, None, entry.min, f"no such item in {detail}"))
        else:
            holes = len(merged.holes)
            out.append(_at_least(label, merged.grade, entry.min, f"{holes} hole(s)"))
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
        stated = {json.dumps(r.gate, sort_keys=True) for r in records}
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
    runs = len(records)
    passed = sum(1 for r in records if r.passed)
    out = verdict.criteria
    out.append(
        _at_least("min_runs", float(runs), float(gate.min_runs), f"{runs} run(s)")
    )
    if gate.min_success_rate is not None:
        out.append(
            _at_least(
                "min_success_rate",
                passed / runs,
                gate.min_success_rate,
                f"{passed} of {runs} passed",
            )
        )
    if gate.min_case_coverage is not None:
        value, detail = _case_coverage(records)
        out.append(
            _at_least("min_case_coverage", value, gate.min_case_coverage, detail)
        )
    out.extend(_coverage_criteria(gate, records, warnings))
    if gate.judges_consistency:
        spread = consistency(
            records, gate.consistency_stations, gate.consistency_min_runs
        )
        if spread.stations == 0:
            why = f"no station was passed by {gate.consistency_min_runs} run(s)"
        else:
            why = ""
        if gate.max_lateral_spread_m is not None:
            out.append(
                _at_most(
                    "max_lateral_spread_m",
                    spread.lateral_spread_m,
                    gate.max_lateral_spread_m,
                    why or f"{_where(spread.lateral_at)}, {spread.stations} station(s)",
                )
            )
        if gate.max_speed_spread_mps is not None:
            out.append(
                _at_most(
                    "max_speed_spread_mps",
                    spread.speed_spread_mps,
                    gate.max_speed_spread_mps,
                    why or f"{_where(spread.speed_at)}, {spread.stations} station(s)",
                )
            )
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
    groups: dict[str, list[GateRecord]] = {}
    for record in records:
        if scenario is None or record.scenario == scenario:
            groups.setdefault(record.scenario, []).append(record)
    return [judge(groups[name], gate) for name in sorted(groups)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser(prog: Optional[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Judge logical scenarios' gates from their runs' gate records "
            "({stem}_gate.json): minimum runs, success rate, case and cover "
            "coverage, and trajectory consistency (docs/gate.md)."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="gate records, or directories searched for *_gate.json",
    )
    parser.add_argument(
        "--document",
        type=Path,
        help=(
            "judge against this document's gate instead of the one the runs "
            "recorded (its id selects the runs)"
        ),
    )
    parser.add_argument("--scenario", help="judge only the runs of this scenario id")
    parser.add_argument("--json", type=Path, help="also write the verdicts as JSON")
    parser.add_argument(
        "--markdown", type=Path, help="write the report to this file, not stdout"
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None, prog: Optional[str] = "scenario-gate"
) -> int:
    """``scenario-gate``: 0 when every gate passes, 1 when one fails, 2 when
    there is nothing to judge or a file cannot be read."""
    args = _parser(prog).parse_args(argv)
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
    records, errors = load_gate_records(find_gate_records(args.paths))
    for error in errors:
        print(f"scenario-gate: {error}", file=sys.stderr)
    verdicts = judge_all(records, gate, scenario)
    if not verdicts:
        print("scenario-gate: no gate record to judge", file=sys.stderr)
        return EXIT_ERROR
    markdown = "# Logical scenario gates\n\n" + "\n\n".join(
        v.to_markdown() for v in verdicts
    )
    if args.markdown is not None:
        args.markdown.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    if args.json is not None:
        args.json.write_text(
            json.dumps({"verdicts": [v.to_dict() for v in verdicts]}, indent=2),
            encoding="utf-8",
        )
    if errors:
        return EXIT_ERROR
    return EXIT_PASSED if all(v.passed for v in verdicts) else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
