"""The typed core of a gate's criteria: what ``scenario-gate`` measures.

The parts of :mod:`.evaluate` that need neither pydantic (the gate's model)
nor openodd: a criterion judged and the spread of the ego's drive across
runs, the share of the cases run, finding the records, and how a criterion
and a case changed from a baseline's runs to a candidate's.  They read only
the per-run records (:mod:`.record`); :mod:`.evaluate` judges a
:class:`~autoware_carla_scenario.authoring.models.GateSpec` with them and
re-exports them.  See ``docs/gate.md``.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .record import GATE_RECORD_SUFFIX, GateRecord

__all__ = [
    "CaseChange",
    "ConsistencyResult",
    "CriterionChange",
    "CriterionResult",
    "consistency",
    "find_gate_records",
    "station_values",
]


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


def _number(value: float) -> str:
    return f"{value:.4g}"


def _warning_lines(warnings: Sequence[str]) -> list[str]:
    """A report's warnings, after a blank line; none without any."""
    return ["", *(f"- warning: {w}" for w in warnings)] if warnings else []


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
    # A station with a value that is not finite is not measured: the records
    # bound their values, so only a degenerate interpolation gives one.
    spreads: list[tuple[float, float, tuple[int, float]]] = []
    for (segment, j), passed in sorted(by_station.items()):
        if len(passed) < min_runs or not all(
            math.isfinite(x) for v in passed for x in v
        ):
            continue
        lateral_spread = statistics.pstdev([v[0] for v in passed])
        speed_spread = statistics.pstdev([v[1] for v in passed])
        if math.isfinite(lateral_spread) and math.isfinite(speed_spread):
            where = (segment, (j + 0.5) / per_segment)
            spreads.append((lateral_spread, speed_spread, where))
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
    # Typed: Codon types a returned tuple by its values, not the annotation.
    share: Optional[float] = covered / total
    return share, f"{covered} of {total} case(s) on {maps} map(s)"


def _where(at: Optional[tuple[int, float]]) -> str:
    if at is None:
        return ""
    return f"segment {at[0]} at {at[1]:.0%}"


def _written_at(record: GateRecord) -> float:
    try:
        return record.path.stat().st_mtime if record.path is not None else 0.0
    except OSError:
        return 0.0


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
