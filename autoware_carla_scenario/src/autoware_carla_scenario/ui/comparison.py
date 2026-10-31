"""Sessions compared case by case and ODD cell by cell: each case's change
from the first session that ran it to the last, its metric deltas, and the
pass rate of each cell of two ODD attributes, per session.

The cases come as :class:`ComparedCase` records, which ``ui.results`` builds
from what a case left on disk (``CaseResult.compared()``): the buckets it was
drawn in and the key it is matched by -- its overrides, else the ODD values it
was drawn at -- are read off its coverage document (JSON of any type) there.
Split out of ``ui.results``, which re-exports what it used of it, so that the
library check compiles the comparison (``docs/typecheck.md``, "Checking the
framework itself").
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "METRICS",
    "CellComparison",
    "ComparedCase",
    "Comparison",
    "MatchedCase",
    "compare",
    "default_axes",
    "varying_attributes",
]


def _metric(label: str, more_is_better: bool | None) -> tuple[str, bool | None]:
    """A metric's entry of :data:`METRICS` (a function, so that Codon types
    each entry's verdict ``Optional[bool]``, ``None`` or not)."""
    return label, more_is_better


#: How each metric reads: its label, and whether more is better (``None``:
#: neither), for the comparison's colours.
METRICS: dict[str, tuple[str, bool | None]] = {
    "elapsed_s": _metric("elapsed (s)", None),
    "min_gap_m": _metric("min gap (m)", True),
    "max_decel_ms2": _metric("max decel (m/s²)", False),
    "stop_time_s": _metric("stopped at (s)", None),
    "distance_m": _metric("distance (m)", None),
    "max_speed_kmh": _metric("max speed (km/h)", None),
    "final_speed_kmh": _metric("final speed (km/h)", None),
}


@dataclass
class ComparedCase:
    """One case of a session, as the comparison reads it."""

    index: int
    #: What the batch calls it (``oneplanner/cut_in#8``), or the scenario.
    label: str
    passed: bool | None
    errored: bool
    message: str
    #: What the case ran with beyond the session's command line.
    overrides: list[str]
    #: The ODD bucket of each attribute the case was drawn in.
    buckets: dict[str, str]
    metrics: dict[str, float]
    #: What the case ran, to match it in another session.
    key: str

    @property
    def scenario(self) -> str:
        """The label without its case number (``oneplanner/cut_in``)."""
        return self.label.split("#", 1)[0]

    @property
    def outcome(self) -> str:
        """``pass``, ``fail``, ``error`` (raised, or no result), or ``none``."""
        if self.errored:
            return "error"
        passed = self.passed
        if passed is None:
            return "none"
        return "pass" if passed else "fail"


def _distinct_buckets(cases: list[ComparedCase]) -> dict[str, set[str]]:
    """The buckets the cases were drawn in, per attribute, in the order the
    attributes are first seen."""
    seen: dict[str, set[str]] = {}
    for case in cases:
        for name, bucket in case.buckets.items():
            if name not in seen:
                seen[name] = set()
            seen[name].add(bucket)
    return seen


def varying_attributes(cases: list[ComparedCase]) -> list[str]:
    """The attributes the cases were drawn in different buckets of."""
    seen = _distinct_buckets(cases)
    return [name for name, buckets in seen.items() if len(buckets) > 1]


def default_axes(cases: list[ComparedCase]) -> tuple[str | None, str | None]:
    """The two attributes the cases were drawn over that tell them apart
    most (the most distinct buckets), else ``None`` (openodd's choice)."""
    seen = _distinct_buckets(cases)
    ranked = sorted([n for n in seen if len(seen[n]) > 1], key=lambda n: -len(seen[n]))
    x: str | None = ranked[0] if ranked else None
    y: str | None = ranked[1] if len(ranked) > 1 else None
    return x, y


@dataclass
class MatchedCase:
    """One case as each compared session ran it (``None``: did not)."""

    key: str
    runs: list[ComparedCase | None]

    @property
    def first(self) -> ComparedCase:
        for run in self.runs:
            if run is not None:
                return run
        raise StopIteration()

    @property
    def ran(self) -> list[ComparedCase]:
        """The runs of the sessions that ran it, oldest first."""
        out: list[ComparedCase] = []
        for run in self.runs:
            if run is not None:
                out.append(run)
        return out

    def change(self) -> str:
        """Between the first session that ran it and the last: ``regression``
        (passed, then not), ``fixed`` (the other way), ``flaky`` (passed at
        both ends and not in between, or the other way), ``same`` (a fail and
        an error are both not passing); ``new`` or
        ``gone`` when one session alone ran it (not the baseline, or only
        the baseline)."""
        ran = self.ran
        if len(ran) == 1:
            return "gone" if self.runs[0] is not None else "new"
        before, after = ran[0].outcome, ran[-1].outcome
        if before == "pass" and after != "pass":
            return "regression"
        if before != "pass" and after == "pass":
            return "fixed"
        # The same at both ends, and passed (or not) in between.
        if any((r.outcome == "pass") != (before == "pass") for r in ran):
            return "flaky"
        return "same"

    def delta(self, metric: str) -> float | None:
        """*metric* of the last session that ran it less the first's."""
        ran = [r for r in self.ran if metric in r.metrics]
        if len(ran) < 2:
            return None
        return round(ran[-1].metrics[metric] - ran[0].metrics[metric], 3)

    def significant(self, metric: str) -> bool:
        """Whether the delta is more than noise: over 5% of the first value
        and over 0.05."""
        delta = self.delta(metric)
        if delta is None:
            return False
        for run in self.ran:
            if metric in run.metrics:
                base = run.metrics[metric]
                return abs(delta) > max(0.05, 0.05 * abs(base))
        raise StopIteration()


@dataclass
class CellComparison:
    """An ODD cell (a bucket of *x*, and of *y*), per session."""

    x: str
    y: str
    #: Per session: (passed, runs) of its cases drawn in the cell.
    tallies: list[tuple[int, int]]
    keys: list[str]

    def rate(self, i: int) -> float | None:
        passed, runs = self.tallies[i]
        return None if runs == 0 else passed / runs

    @property
    def delta(self) -> float | None:
        """The pass rate of the last session with runs in the cell less the
        first's."""
        rates: list[float] = []
        for i in range(len(self.tallies)):
            rate = self.rate(i)
            if rate is not None:
                rates.append(rate)
        return None if len(rates) < 2 else rates[-1] - rates[0]


@dataclass
class Comparison:
    sessions: list[str]
    cases: list[MatchedCase]
    x: str | None
    y: str | None
    x_labels: list[str]
    y_labels: list[str]
    cells: dict[tuple[str, str], CellComparison]
    #: Per session: (passed, runs).
    totals: list[tuple[int, int]]

    def count(self, change: str) -> int:
        return sum(1 for c in self.cases if c.change() == change)

    def rate(self, i: int) -> float | None:
        passed, runs = self.totals[i]
        return None if runs == 0 else passed / runs

    @property
    def metrics(self) -> list[str]:
        present: set[str] = set()
        for case in self.cases:
            for run in case.runs:
                if run is not None:
                    for name in run.metrics:
                        present.add(name)
        return [m for m in METRICS if m in present]


def _bucket_key(label: str) -> tuple[int, float, str]:
    """A bucket's place in reading order: a numeric range by its lower
    bound, then the rest by name."""
    head = label.lstrip("[(").split(",", 1)[0].strip()
    try:
        return 0, float(head), label
    except ValueError:
        return 1, 0.0, label


def _bucket_order(labels: set[str]) -> list[str]:
    """Buckets in reading order: numeric ranges by their lower bound."""
    return sorted(labels, key=_bucket_key)


def _case_number(label: str) -> tuple[int, str]:
    number = label.rpartition("#")[2]
    return (int(number), label) if number.isdigit() else (0, label)


def _bucket_of(buckets: dict[str, str], attribute: str | None) -> str:
    """The bucket of *attribute* (``""`` without one, or without the
    attribute)."""
    if attribute is None or not attribute:
        return ""
    return buckets.get(attribute, "")


#: The order of the cases: what changed first.
_CHANGE_ORDER = {
    "regression": 0,
    "fixed": 1,
    "flaky": 2,
    "new": 3,
    "gone": 4,
    "same": 5,
}


def _case_order(case: MatchedCase) -> tuple[int, str, tuple[int, str]]:
    first = case.first
    return (_CHANGE_ORDER[case.change()], first.scenario, _case_number(first.label))


def compare(
    sessions: list[tuple[str, list[ComparedCase]]],
    *,
    x: str | None = None,
    y: str | None = None,
) -> Comparison:
    """Compare sessions (oldest first: the first is the baseline) case by case
    and ODD cell by cell, the cells along *x* and *y* (the attributes the
    cases were drawn over, :func:`default_axes` when not given)."""
    every: list[ComparedCase] = []
    for _, cases in sessions:
        every.extend(cases)
    if x is None and y is None:
        x, y = default_axes(every)
    count = len(sessions)
    matched: dict[str, MatchedCase] = {}
    for i, (_, cases) in enumerate(sessions):
        for result in cases:
            if result.key not in matched:
                runs: list[ComparedCase | None] = [None for _ in range(count)]
                matched[result.key] = MatchedCase(result.key, runs)
            # A key run twice in one session (a repeated case): the last run.
            matched[result.key].runs[i] = result
    ordered = sorted(matched.values(), key=_case_order)
    cells: dict[tuple[str, str], CellComparison] = {}
    x_labels: set[str] = set()
    y_labels: set[str] = set()
    for case in matched.values():
        buckets = case.first.buckets
        bx = _bucket_of(buckets, x)
        by = _bucket_of(buckets, y)
        if (x is not None and x and not bx) or (y is not None and y and not by):
            continue
        x_labels.add(bx)
        y_labels.add(by)
        if (bx, by) not in cells:
            tallies: list[tuple[int, int]] = [(0, 0) for _ in range(count)]
            cells[(bx, by)] = CellComparison(bx, by, tallies, [])
        cell = cells[(bx, by)]
        cell.keys.append(case.key)
        for i, run in enumerate(case.runs):
            if run is not None:
                passed, ran = cell.tallies[i]
                won = 1 if run.outcome == "pass" else 0
                cell.tallies[i] = (passed + won, ran + 1)
    totals: list[tuple[int, int]] = []
    for _, cases in sessions:
        totals.append((sum(1 for r in cases if r.outcome == "pass"), len(cases)))
    return Comparison(
        sessions=[name for name, _ in sessions],
        cases=ordered,
        x=x,
        y=y,
        x_labels=_bucket_order(x_labels),
        y_labels=_bucket_order(y_labels),
        cells=cells,
        totals=totals,
    )
