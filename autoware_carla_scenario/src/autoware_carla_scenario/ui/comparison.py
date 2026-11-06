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
from typing import Optional

__all__ = [
    "HYDRA_PREFIXES",
    "METRICS",
    "POSITIONAL_OVERRIDES",
    "CellComparison",
    "ComparedCase",
    "Comparison",
    "MatchedCase",
    "compare",
    "default_axes",
    "outcome_of",
    "override_key",
    "scenario_of",
    "varying_attributes",
]

#: An override that names the case by its position (the ODD sampler's
#: encoded case), not by what it runs: left out of a case's key.
POSITIONAL_OVERRIDES = ("odd_case=",)

#: Overrides of a run that are not the case (Hydra's own).
HYDRA_PREFIXES = ("hydra.", "hydra/")


def scenario_of(label: str) -> str:
    """A case's label without its case number (``oneplanner/cut_in``)."""
    return label.split("#", 1)[0]


def outcome_of(errored: bool, passed: Optional[bool]) -> str:
    """``pass``, ``fail``, ``error`` (raised, or no result), or ``none``."""
    if errored:
        return "error"
    if passed is None:
        return "none"
    return "pass" if passed else "fail"


def _own_override(override: str) -> bool:
    """Whether *override* says what the case runs: neither positional nor
    Hydra's own."""
    bare = override.lstrip("+")
    if any(bare.startswith(prefix) for prefix in POSITIONAL_OVERRIDES):
        return False
    return not any(override.startswith(prefix) for prefix in HYDRA_PREFIXES)


def override_key(scenario: str, overrides: list[str]) -> Optional[str]:
    """What a case of *scenario* ran, by its *overrides* (less the positional
    ones and Hydra's, without a leading ``+``, sorted), to match it in another
    session; ``None`` when none is left."""
    own = sorted([o.lstrip("+") for o in overrides if _own_override(o)])
    if not own:
        return None
    return f"{scenario}|" + "|".join(own)


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
    """One run of a case in a session, as the comparison reads it."""

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
        return scenario_of(self.label)

    @property
    def outcome(self) -> str:
        """``pass``, ``fail``, ``error`` (raised, or no result), or ``none``."""
        return outcome_of(self.errored, self.passed)


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
    """One case as each compared session ran it: every run of its key in each
    session, in run order (empty: the session did not run it). A session
    runs a key more than once when its cases repeat one (a sweep that draws
    the same ODD values twice, a case run again): every run is kept."""

    key: str
    runs: list[list[ComparedCase]]

    @property
    def first(self) -> ComparedCase:
        """The first run of the first session that ran it."""
        for session in self.runs:
            for run in session:
                return run
        raise StopIteration()

    @property
    def ran(self) -> list[ComparedCase]:
        """Every run, oldest session first, each session's in run order."""
        out: list[ComparedCase] = []
        for session in self.runs:
            out.extend(session)
        return out

    @property
    def latest(self) -> list[ComparedCase | None]:
        """Each session's last run of it (``None``: did not run it)."""
        out: list[ComparedCase | None] = []
        for session in self.runs:
            last: ComparedCase | None = session[-1] if session else None
            out.append(last)
        return out

    def tally(self, i: int) -> tuple[int, int]:
        """(passed, runs) of session *i*'s runs of it."""
        passed = sum(1 for r in self.runs[i] if r.outcome == "pass")
        return passed, len(self.runs[i])

    def verdict(self, i: int) -> str:
        """Session *i*'s verdict on it: ``pass`` (every run passed), ``fail``
        (none did: failed, errored or without a verdict), ``mixed`` (some
        did), ``""`` (did not run it)."""
        passed, runs = self.tally(i)
        if runs == 0:
            return ""
        if passed == runs:
            return "pass"
        return "fail" if passed == 0 else "mixed"

    def change(self) -> str:
        """How its verdict changed across the sessions that ran it, each
        session's verdict read off all of its runs (:meth:`verdict`):
        ``flaky`` when a session both passed and failed it, or when the first
        and last sessions agree but one between does not; else, between the
        first session that ran it and the last, ``regression`` (every run
        passed, then none), ``fixed`` (no run passed, then every one),
        ``same`` (a fail and an error are both not passing); ``new`` or
        ``gone`` when one session alone ran it (not the baseline, or only the
        baseline)."""
        verdicts: list[str] = []
        for i in range(len(self.runs)):
            verdict = self.verdict(i)
            if verdict:
                verdicts.append(verdict)
        if "mixed" in verdicts:
            return "flaky"
        if len(verdicts) == 1:
            return "gone" if self.runs[0] else "new"
        before, after = verdicts[0], verdicts[-1]
        if before == "pass" and after == "fail":
            return "regression"
        if before == "fail" and after == "pass":
            return "fixed"
        # The same at both ends, and passed (or not) in between.
        if any(v != before for v in verdicts):
            return "flaky"
        return "same"

    def _means(self, metric: str) -> list[float]:
        """*metric*'s mean over each session's runs that have it, oldest
        first (sessions without one left out)."""
        means: list[float] = []
        for session in self.runs:
            values = [r.metrics[metric] for r in session if metric in r.metrics]
            if values:
                means.append(sum(values) / len(values))
        return means

    def delta(self, metric: str) -> float | None:
        """*metric* of the last session that ran it less the first's, each
        the mean of the session's runs."""
        means = self._means(metric)
        if len(means) < 2:
            return None
        return round(means[-1] - means[0], 3)

    def significant(self, metric: str) -> bool:
        """Whether the delta is more than noise: over 5% of the first value
        and over 0.05."""
        delta = self.delta(metric)
        if delta is None:
            return False
        base = self._means(metric)[0]
        return abs(delta) > max(0.05, 0.05 * abs(base))


@dataclass
class CellComparison:
    """An ODD cell (a bucket of *x*, and of *y*), per session."""

    x: str
    y: str
    #: Per session: (passed, runs) of every run of the keys in the cell.
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
            for run in case.ran:
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
    return (int(number), label) if number.isascii() and number.isdigit() else (0, label)


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
                runs: list[list[ComparedCase]] = [[] for _ in range(count)]
                matched[result.key] = MatchedCase(result.key, runs)
            # A key run twice in one session (a repeated case): every run.
            matched[result.key].runs[i].append(result)
    ordered = sorted(matched.values(), key=_case_order)
    cells: dict[tuple[str, str], CellComparison] = {}
    x_labels: set[str] = set()
    y_labels: set[str] = set()
    # Every run of a key is tallied in the key's cell (where its first run
    # was drawn: a run that errored before it was drawn has no buckets), so
    # that the cells add up to the totals, less the keys drawn in no bucket
    # of x or y.
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
        for i in range(count):
            passed, ran = cell.tallies[i]
            won, played = case.tally(i)
            cell.tallies[i] = (passed + won, ran + played)
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
