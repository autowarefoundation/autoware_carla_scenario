"""A session's results on its ODD, and sessions compared case by case.

Each case of a session is read with what it left on disk (``ui.cases``): its
verdict (``batch_results.json`` or ``{stem}_result.json``), the case it ran (a
sweep's overrides; the ODD values and buckets it was drawn at, from
``{stem}_coverage.json``'s ``case``), its coverage document, and a few metrics
of how the ego drove, read off its trajectory log (:func:`case_metrics`).

The ODD view is openodd's own coverage component (``openodd.CoverageView``)
and its case plot (``openodd.CasePlot``), fed the sessions' coverage
documents. Comparing sessions matches their cases by what they ran -- the
case's overrides, else the ODD values it was drawn at -- not by position, so a
case keeps its match when a sweep's order or size changes; and tallies each
ODD cell (the buckets a case was drawn in, along two attributes) per session.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cases import CaseFiles, case_files, job_dir, session_dir

__all__ = [
    "CaseResult",
    "CellComparison",
    "Comparison",
    "MatchedCase",
    "case_metrics",
    "compare",
    "default_axes",
    "load_results",
    "cases_in_cell",
]

#: An override that names the case by its position (the ODD sampler's
#: encoded case), not by what it runs: left out of a case's key.
_POSITIONAL_OVERRIDES = ("odd_case=",)

#: Overrides of a run that are not the case (Hydra's own).
_HYDRA_PREFIXES = ("hydra.", "hydra/")


def _read_json(path: Path | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass
class CaseResult:
    """One case of a session, as the result views need it."""

    index: int
    #: What the batch calls it (``oneplanner/cut_in#8``), or the scenario.
    label: str
    passed: bool | None
    errored: bool
    elapsed_seconds: float | None
    message: str
    #: What the case ran with beyond the session's command line.
    overrides: list[str]
    files: CaseFiles | None
    coverage: dict[str, Any] | None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def scenario(self) -> str:
        """The label without its case number (``oneplanner/cut_in``)."""
        return self.label.split("#", 1)[0]

    @property
    def case(self) -> dict[str, Any]:
        """The coverage document's ``case``: the ODD ``values`` and
        ``buckets`` the case was drawn at (empty without one)."""
        case = (self.coverage or {}).get("case")
        return case if isinstance(case, dict) else {}

    @property
    def buckets(self) -> dict[str, str]:
        buckets = self.case.get("buckets")
        return (
            {str(k): str(v) for k, v in buckets.items()}
            if isinstance(buckets, dict)
            else {}
        )

    @property
    def values(self) -> dict[str, Any]:
        values = self.case.get("values")
        return dict(values) if isinstance(values, dict) else {}

    @property
    def key(self) -> str:
        """What the case ran, to match it in another session: its overrides
        (less the positional ones), else the ODD values it was drawn at, else
        its label."""
        own = sorted(
            o.lstrip("+")
            for o in self.overrides
            if not o.lstrip("+").startswith(_POSITIONAL_OVERRIDES)
            and not o.startswith(_HYDRA_PREFIXES)
        )
        if own:
            return f"{self.scenario}|" + "|".join(own)
        if self.values:
            return f"{self.scenario}|" + json.dumps(self.values, sort_keys=True)
        return self.label

    @property
    def outcome(self) -> str:
        """``pass``, ``fail``, ``error`` (raised, or no result), or ``none``."""
        if self.errored:
            return "error"
        if self.passed is None:
            return "none"
        return "pass" if self.passed else "fail"


# ── Metrics off the trajectory log ──────────────────────────────────────────


def _box(
    x: float, y: float, yaw: float, extent: tuple[float, float]
) -> list[tuple[float, float]]:
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    ex, ey = extent
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((ex, ey), (-ex, ey), (-ex, -ey), (ex, -ey))
    ]


def _separated(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * px + ny * py for px, py in a]
            pb = [nx * px + ny * py for px, py in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return True
    return False


def _point_segment(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length = dx * dx + dy * dy
    t = (
        0.0
        if length == 0
        else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length))
    )
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _box_distance(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """The distance between two boxes on the ground (0 when they overlap)."""
    if not _separated(a, b):
        return 0.0
    return min(
        _point_segment(p, q[i], q[(i + 1) % 4])
        for p_box, q in ((a, b), (b, a))
        for p in p_box
        for i in range(4)
    )


_metrics_cache: dict[str, tuple[float, dict[str, float]]] = {}

#: The span the hardest braking is measured over.
_DECEL_WINDOW_S = 0.5


def case_metrics(trajectory: Path) -> dict[str, float]:
    """How the ego drove, read off its trajectory log (format 1 or 2):

    * ``min_gap_m`` -- the least distance on the ground from the ego's box to
      another vehicle's or pedestrian's (0 touching);
    * ``distance_m`` -- how far it drove; ``max_speed_kmh``,
      ``final_speed_kmh``;
    * ``max_decel_ms2`` -- its hardest braking, its speed's drop over half a
      second;
    * ``stop_time_s`` -- when it first came to a stop (below 0.5 km/h after
      having moved), if it did.

    Empty when the log has no ego. Kept until the log changes.
    """
    try:
        stamp = trajectory.stat().st_mtime
    except OSError:
        return {}
    cached = _metrics_cache.get(str(trajectory))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    extents: dict[int, tuple[float, float]] = {}
    ego: int | None = None
    min_gap = math.inf
    distance = 0.0
    max_speed = 0.0
    final_speed: float | None = None
    max_decel = 0.0
    stop_time: float | None = None
    moved = False
    last: tuple[float, float, float, float] | None = None  # t, x, y, speed
    window: list[tuple[float, float]] = []  # (t, speed) over the last window
    try:
        with trajectory.open(encoding="utf-8") as stream:
            for line in stream:
                if '"tick"' not in line and '"actor"' not in line:
                    continue
                record = json.loads(line)
                if record.get("type") == "actor":
                    extent = record.get("extent") or [0.0, 0.0]
                    extents[int(record["id"])] = (float(extent[0]), float(extent[1]))
                    if str(record.get("role", "")).lower() == "ego":
                        ego = int(record["id"])
                    continue
                # The text test above lets through any record whose label or
                # role reads "tick" or "actor" (a spawn's, a despawn's, ...):
                # only ticks carry poses.
                if record.get("type") != "tick":
                    continue
                control = record.get("control") or {}
                if ego is None and control:
                    ego = int(next(iter(control)))
                if ego is None:
                    continue
                states = {int(a[0]): a for a in record.get("actors") or []}
                me = states.get(ego)
                if me is None:
                    continue
                t = float(record["t"])
                speed = math.hypot(float(me[7]), float(me[8])) * 3.6
                ego_box = _box(me[1], me[2], me[6], extents.get(ego, (2.4, 1.0)))
                for other, state in states.items():
                    if other == ego:
                        continue
                    gap = _box_distance(
                        ego_box,
                        _box(
                            state[1], state[2], state[6], extents.get(other, (0.3, 0.3))
                        ),
                    )
                    min_gap = min(min_gap, gap)
                if last is not None:
                    lt, lx, ly, ls = last
                    distance += math.hypot(me[1] - lx, me[2] - ly)
                # Braking over a window, not tick to tick: a single tick's
                # velocity jumps (a spawn, a teleport) would read as braking.
                window.append((t, speed))
                while len(window) > 1 and t - window[1][0] >= _DECEL_WINDOW_S:
                    window.pop(0)
                wt, ws = window[0]
                if t - wt >= _DECEL_WINDOW_S:
                    max_decel = max(max_decel, (ws - speed) / 3.6 / (t - wt))
                max_speed = max(max_speed, speed)
                moved = moved or speed > 5.0
                if moved and stop_time is None and speed < 0.5:
                    stop_time = t
                final_speed = speed
                last = (t, float(me[1]), float(me[2]), speed)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return {}
    metrics: dict[str, float] = {}
    if last is not None:
        metrics["distance_m"] = round(distance, 2)
        metrics["max_speed_kmh"] = round(max_speed, 2)
        metrics["final_speed_kmh"] = round(final_speed or 0.0, 2)
        metrics["max_decel_ms2"] = round(max_decel, 2)
        if math.isfinite(min_gap):
            metrics["min_gap_m"] = round(min_gap, 3)
        if stop_time is not None:
            metrics["stop_time_s"] = round(stop_time, 2)
    _metrics_cache[str(trajectory)] = (stamp, metrics)
    return metrics


#: How each metric reads: its label, and whether more is better (``None``:
#: neither), for the comparison's colours.
METRICS: dict[str, tuple[str, bool | None]] = {
    "elapsed_s": ("elapsed (s)", None),
    "min_gap_m": ("min gap (m)", True),
    "max_decel_ms2": ("max decel (m/s²)", False),
    "stop_time_s": ("stopped at (s)", None),
    "distance_m": ("distance (m)", None),
    "max_speed_kmh": ("max speed (km/h)", None),
    "final_speed_kmh": ("final speed (km/h)", None),
}


# ── A session's cases ───────────────────────────────────────────────────────


def _hydra_overrides(directory: Path) -> list[str]:
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415

        data = yaml.safe_load(
            (directory / ".hydra" / "overrides.yaml").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ImportError):
        return []
    except Exception:  # noqa: BLE001 -- yaml.YAMLError
        return []
    return [str(o) for o in data] if isinstance(data, list) else []


def load_results(
    base: Path, session_type: str, date: str, time: str, *, metrics: bool = True
) -> list[CaseResult]:
    """Every case of a session, with its files, case and (with *metrics*) the
    metrics of its trajectory log."""
    directory = session_dir(base, session_type, date, time)
    if directory is None or not directory.is_dir():
        return []
    out: list[CaseResult] = []
    if session_type == "batch":
        entries = _read_json(directory / "batch_results.json")
        if not isinstance(entries, list):
            return []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            files = case_files(base, session_type, date, time, index)
            out.append(
                CaseResult(
                    index=index,
                    label=str(entry.get("scenario") or "unknown"),
                    passed=entry.get("passed"),
                    errored=bool(entry.get("error")),
                    elapsed_seconds=entry.get("elapsed_seconds"),
                    message=str(entry.get("message") or ""),
                    overrides=[str(o) for o in entry.get("overrides") or []],
                    files=files,
                    coverage=_read_json(files.coverage) if files else None,
                )
            )
    else:
        indices = (
            sorted(
                int(p.name)
                for p in directory.iterdir()
                if p.is_dir() and p.name.isdigit()
            )
            if session_type == "multirun"
            else [0]
        )
        for index in indices:
            files = case_files(base, session_type, date, time, index)
            job = (
                job_dir(base, date, time, index)
                if session_type == "multirun"
                else directory
            )
            result = _read_json(files.result) if files else None
            result = result if isinstance(result, dict) else {}
            overrides = _hydra_overrides(job) if job is not None else []
            label = next(
                (o.split("=", 1)[1] for o in overrides if o.startswith("scenario=")),
                files.stem if files else "unknown",
            )
            out.append(
                CaseResult(
                    index=index,
                    label=label,
                    passed=result.get("passed"),
                    errored=files is None or not result,
                    elapsed_seconds=result.get("elapsed_seconds"),
                    message=str(result.get("message") or ""),
                    overrides=[o for o in overrides if not o.startswith("scenario=")],
                    files=files,
                    coverage=_read_json(files.coverage) if files else None,
                )
            )
    for case in out:
        if case.elapsed_seconds is not None:
            case.metrics["elapsed_s"] = round(float(case.elapsed_seconds), 2)
        if metrics and case.files is not None and case.files.trajectory.is_file():
            case.metrics.update(case_metrics(case.files.trajectory))
    return out


def varying_attributes(results: list[CaseResult]) -> list[str]:
    """The attributes the cases were drawn in different buckets of."""
    seen: dict[str, set[str]] = {}
    for result in results:
        for name, bucket in result.buckets.items():
            seen.setdefault(name, set()).add(bucket)
    return [name for name, buckets in seen.items() if len(buckets) > 1]


def coverage_documents(results: list[CaseResult]) -> list[dict[str, Any]]:
    """The cases' coverage documents, for openodd's views."""
    return [r.coverage for r in results if isinstance(r.coverage, dict)]


def default_axes(results: list[CaseResult]) -> tuple[str | None, str | None]:
    """The two attributes the cases were drawn over that tell them apart
    most (the most distinct buckets), else ``None`` (openodd's choice)."""
    seen: dict[str, set[str]] = {}
    for result in results:
        for name, bucket in result.buckets.items():
            seen.setdefault(name, set()).add(bucket)
    ranked = sorted((n for n in seen if len(seen[n]) > 1), key=lambda n: -len(seen[n]))
    x = ranked[0] if ranked else None
    y = ranked[1] if len(ranked) > 1 else None
    return x, y


def _seconds_in(coverage: dict[str, Any], attribute: str, bucket: str) -> float:
    """How long the run spent with *attribute* in *bucket* (its ``odd.``
    coverage item's seconds; a versioned item, ``@v2``, too)."""
    for item in coverage.get("items") or []:
        if not isinstance(item, dict) or item.get("attribute") != attribute:
            continue
        seconds = item.get("seconds") or {}
        if isinstance(seconds, dict) and bucket in seconds:
            return float(seconds[bucket] or 0.0)
    return 0.0


def cases_in_cell(
    results: list[CaseResult], conditions: list[tuple[str, str]]
) -> list[CaseResult]:
    """The cases that spent time in every bucket of *conditions* (the cell
    openodd's view was drilled into), as the view counts runs in a cell."""
    if not conditions:
        return list(results)
    return [
        r
        for r in results
        if r.coverage is not None
        and all(_seconds_in(r.coverage, a, b) > 0.0 for a, b in conditions)
    ]


# ── Comparing sessions ──────────────────────────────────────────────────────


@dataclass
class MatchedCase:
    """One case as each compared session ran it (``None``: did not)."""

    key: str
    runs: list[CaseResult | None]

    @property
    def first(self) -> CaseResult:
        return next(r for r in self.runs if r is not None)

    @property
    def ran(self) -> list[CaseResult]:
        """The runs of the sessions that ran it, oldest first."""
        return [r for r in self.runs if r is not None]

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
        base = next(r.metrics[metric] for r in self.ran if metric in r.metrics)
        return abs(delta) > max(0.05, 0.05 * abs(base))


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
        rates = [
            r for r in (self.rate(i) for i in range(len(self.tallies))) if r is not None
        ]
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
        present = {m for c in self.cases for r in c.runs if r for m in r.metrics}
        return [m for m in METRICS if m in present]


def _bucket_order(labels: set[str]) -> list[str]:
    """Buckets in reading order: numeric ranges by their lower bound."""

    def key(label: str) -> tuple[int, float, str]:
        head = label.lstrip("[(").split(",", 1)[0].strip()
        try:
            return 0, float(head), label
        except ValueError:
            return 1, 0.0, label

    return sorted(labels, key=key)


def _case_number(label: str) -> tuple[int, str]:
    number = label.rpartition("#")[2]
    return (int(number), label) if number.isdigit() else (0, label)


def compare(
    sessions: list[tuple[str, list[CaseResult]]],
    *,
    x: str | None = None,
    y: str | None = None,
) -> Comparison:
    """Compare sessions (oldest first: the first is the baseline) case by case
    and ODD cell by cell, the cells along *x* and *y* (the attributes the
    cases were drawn over, :func:`default_axes` when not given)."""
    every = [r for _, results in sessions for r in results]
    if x is None and y is None:
        x, y = default_axes(every)
    matched: dict[str, MatchedCase] = {}
    for i, (_, results) in enumerate(sessions):
        for result in results:
            case = matched.setdefault(
                result.key, MatchedCase(result.key, [None] * len(sessions))
            )
            # A key run twice in one session (a repeated case): the last run.
            case.runs[i] = result
    order = {"regression": 0, "fixed": 1, "flaky": 2, "new": 3, "gone": 4, "same": 5}
    cases = sorted(
        matched.values(),
        key=lambda c: (
            order[c.change()],
            c.first.scenario,
            _case_number(c.first.label),
        ),
    )
    cells: dict[tuple[str, str], CellComparison] = {}
    x_labels: set[str] = set()
    y_labels: set[str] = set()
    for case in matched.values():
        buckets = case.first.buckets
        bx = buckets.get(x, "") if x else ""
        by = buckets.get(y, "") if y else ""
        if (x and not bx) or (y and not by):
            continue
        x_labels.add(bx)
        y_labels.add(by)
        cell = cells.setdefault(
            (bx, by), CellComparison(bx, by, [(0, 0)] * len(sessions), [])
        )
        cell.keys.append(case.key)
        for i, run in enumerate(case.runs):
            if run is not None:
                passed, runs = cell.tallies[i]
                cell.tallies[i] = (passed + (run.outcome == "pass"), runs + 1)
    totals = [
        (sum(1 for r in results if r.outcome == "pass"), len(results))
        for _, results in sessions
    ]
    return Comparison(
        sessions=[name for name, _ in sessions],
        cases=cases,
        x=x,
        y=y,
        x_labels=_bucket_order(x_labels),
        y_labels=_bucket_order(y_labels),
        cells=cells,
        totals=totals,
    )
