"""A session's results on its ODD, and sessions compared case by case.

Each case of a session is read with what it left on disk (``ui.cases``): its
verdict (a batch's entries, ``ui.cases.batch_entries``: its
``batch_results.json``, or what a SLURM batch's tasks recorded while it is not
collected yet or was resubmitted since; else ``{stem}_result.json``), the case
it ran (a sweep's overrides; the ODD values and buckets it was drawn at, from
``{stem}_coverage.json``'s ``case``), its coverage document, and a few metrics
of how the ego drove, read off its trajectory log (:func:`case_metrics`, which
hands the ego's ticks to ``ui.drive_metrics``, the box geometry and metric
arithmetic the library check compiles).

The ODD view is openodd's own coverage component (``openodd.CoverageView``)
and its case plot (``openodd.CasePlot``), fed the sessions' coverage
documents -- those it can read: a document openodd rejects (a number too
long for a float, ``NaN``, #2012) is left out of every view, and the pages
name its file (:func:`skipped_coverage`). Comparing sessions matches their cases by what they ran -- the
case's overrides, else the ODD values it was drawn at -- not by position, so a
case keeps its match when a sweep's order or size changes; and tallies each
ODD cell (the buckets a case was drawn in, along two attributes) per session.
The comparison itself is ``ui.comparison``'s, which the library check
compiles: it reads each case as a ``ComparedCase``
(:meth:`CaseResult.compared`), whose buckets and key are read off the
coverage document here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from .case_paths import _job_indices as _job_indices
from .case_paths import _override_label as _override_label
from .cases import CaseFiles, batch_entries, case_files, job_dir, session_dir

# The comparison of sessions, compiled by the library check; the names it
# had here stay importable from here.
from .comparison import METRICS as METRICS
from .comparison import CellComparison as CellComparison
from .comparison import ComparedCase as ComparedCase
from .comparison import Comparison as Comparison
from .comparison import MatchedCase as MatchedCase
from .comparison import _bucket_order as _bucket_order
from .comparison import _case_number as _case_number
from .comparison import compare as _compare
from .comparison import outcome_of, override_key, scenario_of
from .comparison import default_axes as _default_axes
from .comparison import varying_attributes as _varying_attributes

# The box geometry and the running metrics of the ego's ticks, compiled by the
# library check; the names case_metrics used stay importable from here.
from .drive_metrics import DECEL_WINDOW_S as _DECEL_WINDOW_S  # noqa: F401
from .drive_metrics import DriveMetrics
from .drive_metrics import _point_segment as _point_segment
from .drive_metrics import _separated as _separated
from .drive_metrics import box as _box
from .drive_metrics import box_distance as _box_distance

# Below this height (m) a road user is parked under the map, out of the scene.
from ..utils.clearance import _HIDDEN_BELOW_M, _level_gap

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
    "skipped_coverage",
]


def _read_json(path: Path | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _passed(value: Any) -> bool | None:
    """A verdict: *value* when it is a boolean, else ``None``."""
    return value if isinstance(value, bool) else None


def _seconds(value: Any) -> float | None:
    """A duration: *value* when it is a number (not a boolean), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        # An integer too long for a float (#1726).
        return None


def _strings(value: Any) -> list[str]:
    """A list of overrides: each item as a string, empty unless a list."""
    return [str(o) for o in value] if isinstance(value, list) else []


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
    #: A SLURM batch's case without an outcome yet (``passed`` ``None``).
    pending: bool = False

    @cached_property
    def coverage_viewable(self) -> bool:
        """Whether openodd's views can read the case's coverage document:
        it has one and ``openodd.CoverageView`` accepts it alone. One that
        holds a number too long for a float or ``NaN`` (which Python's
        ``json`` reads) is refused there (#2012); it is left out of the views
        rather than failing the page. Probed once per loaded case."""
        if not isinstance(self.coverage, dict):
            return False
        import openodd  # noqa: PLC0415 -- a Rust extension, loaded on first use

        try:
            openodd.CoverageView([self.coverage])
        except (ValueError, TypeError, OverflowError):
            return False
        return True

    @property
    def scenario(self) -> str:
        """The label without its case number (``oneplanner/cut_in``)."""
        return scenario_of(self.label)

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
        (less the positional ones and Hydra's own), else the ODD values it was drawn at, else
        its label."""
        by_overrides = override_key(self.scenario, self.overrides)
        if by_overrides is not None:
            return by_overrides
        if self.values:
            return f"{self.scenario}|" + json.dumps(self.values, sort_keys=True)
        return self.label

    @property
    def outcome(self) -> str:
        """``pass``, ``fail``, ``error`` (raised, or no result), or ``none``."""
        return outcome_of(self.errored, self.passed)

    def compared(self) -> ComparedCase:
        """The case as the comparison reads it: its buckets and key read off
        the coverage document here (``ui.comparison`` compiles without
        one)."""
        return ComparedCase(
            index=self.index,
            label=self.label,
            passed=self.passed,
            errored=self.errored,
            message=self.message,
            overrides=self.overrides,
            buckets=self.buckets,
            metrics=self.metrics,
            key=self.key,
        )


# ── Metrics off the trajectory log ──────────────────────────────────────────


_metrics_cache: dict[str, tuple[float, dict[str, float]]] = {}


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

    Empty when the log has no ego or cannot be read. A line that is not
    JSON (a killed run's last tick, cut off mid-line) or a record that cannot
    be used is skipped, as ``rerun_metrics.metric_records`` skips it: the
    metrics of every good tick are kept. Kept until the log changes.
    """
    try:
        stamp = trajectory.stat().st_mtime
    except OSError:
        return {}
    cached = _metrics_cache.get(str(trajectory))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    reader = _TrajectoryMetrics()
    try:
        with trajectory.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if '"tick"' not in line and '"actor"' not in line:
                    continue
                try:
                    reader.read(json.loads(line))
                except (
                    ValueError,
                    TypeError,
                    IndexError,
                    KeyError,
                    AttributeError,
                    # A number too large for an int or a float (#1964).
                    OverflowError,
                ):
                    continue
    except OSError:
        return {}
    metrics = reader.drive.to_dict()
    _metrics_cache[str(trajectory)] = (stamp, metrics)
    return metrics


class _TrajectoryMetrics:
    """``case_metrics``'s pass over a trajectory log, a record at a time."""

    def __init__(self) -> None:
        self.extents: dict[int, tuple[float, float]] = {}
        # Each box's offset from its actor's origin (none in older logs).
        self.centers: dict[int, tuple[float, float]] = {}
        # Each box's height above its actor's origin and its half height
        # (m); none for an actor whose log records no ``extent`` z (an older
        # one), whose gaps are on the ground alone.
        self.heights: dict[int, tuple[float, float]] = {}
        self.ego: int | None = None
        self.drive = DriveMetrics()

    def read(self, record: Any) -> None:
        """Take in one *record*; raises ``TypeError``, ``ValueError``,
        ``IndexError``, ``KeyError`` or ``AttributeError`` for one it cannot
        use, before changing anything."""
        if record.get("type") == "actor":
            extent = record.get("extent") or [0.0, 0.0]
            center = record.get("center") or [0.0, 0.0]
            actor = int(record["id"])
            box = (float(extent[0]), float(extent[1]))
            offset = (float(center[0]), float(center[1]))
            # No z in an older log: its gaps are on the ground alone.
            height: tuple[float, float] | None = None
            if len(extent) > 2:
                lift = float(center[2]) if len(center) > 2 else 0.0
                height = (lift, float(extent[2]))
            self.extents[actor] = box
            self.centers[actor] = offset
            if height is None:
                self.heights.pop(actor, None)
            else:
                self.heights[actor] = height
            if str(record.get("role", "")).lower() == "ego":
                self.ego = actor
            return
        # The text test in case_metrics lets through any record whose label
        # or role reads "tick" or "actor" (a spawn's, a despawn's, ...): only
        # ticks carry poses.
        if record.get("type") != "tick":
            return
        ego = self.ego
        control = record.get("control") or {}
        if ego is None and control:
            ego = int(next(iter(control)))
            self.ego = ego
        if ego is None:
            return
        states = {int(a[0]): a for a in record.get("actors") or []}
        me = states.get(ego)
        if me is None:
            return
        t = float(record["t"])
        x, y = float(me[1]), float(me[2])
        speed = math.hypot(float(me[7]), float(me[8])) * 3.6
        ego_box = _box(
            me[1],
            me[2],
            me[6],
            self.extents.get(ego, (2.4, 1.0)),
            self.centers.get(ego, (0.0, 0.0)),
        )
        ego_span = self._span(ego, me)
        # Every gap first, so a tick that breaks halfway adds none of them.
        gaps: list[float] = []
        for other, state in states.items():
            # A road user parked under the map (hidden) is out of the scene.
            if other == ego or float(state[3]) < _HIDDEN_BELOW_M:
                continue
            # One on another level of road -- on a bridge over the ego, in an
            # underpass beneath it -- may overlap the ego's box on the ground,
            # but is as far from it as the height between them, as
            # ``metrics._Frame.distance`` measures it (#1584, #1598).
            span = self._span(other, state)
            level = 0.0
            if ego_span is not None and span is not None:
                level = _level_gap(ego_span, span)
            planar = _box_distance(
                ego_box,
                _box(
                    state[1],
                    state[2],
                    state[6],
                    self.extents.get(other, (0.3, 0.3)),
                    self.centers.get(other, (0.0, 0.0)),
                ),
            )
            gaps.append(math.hypot(planar, level))
        for gap in gaps:
            self.drive.gap(gap)
        self.drive.tick(t, x, y, speed)

    def _span(self, actor: int, state: Any) -> tuple[float, float] | None:
        """The bottom and top (m) of *actor*'s box at its tick *state*, as
        ``utils.clearance._vertical_span`` reads them off a live actor, pitch
        and roll ignored; ``None`` where its log records no height."""
        height = self.heights.get(actor)
        if height is None:
            return None
        centre = float(state[3]) + height[0]
        return centre - height[1], centre + height[1]


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


def _job(
    base: Path, directory: Path, session_type: str, date: str, time: str, index: int
) -> Path | None:
    """The directory of case *index* of a multirun or single session."""
    return job_dir(base, date, time, index) if session_type == "multirun" else directory


def session_scenarios(base: Path, session_type: str, date: str, time: str) -> set[str]:
    """The scenarios a multirun or single session ran, named as
    :func:`load_results` names them (each case's :attr:`CaseResult.scenario`),
    but read from the Hydra overrides alone: no result, coverage or trajectory
    file is read, and a case's files are looked up only when it has no
    ``scenario=`` override."""
    directory = session_dir(base, session_type, date, time)
    if session_type == "batch" or directory is None or not directory.is_dir():
        return set()
    out: set[str] = set()
    for index in _job_indices(directory, session_type):
        job = _job(base, directory, session_type, date, time, index)
        label = _override_label(_hydra_overrides(job)) if job is not None else None
        if label is None:
            files = case_files(base, session_type, date, time, index)
            label = files.stem if files else "unknown"
        out.add(label.split("#", 1)[0])
    return out


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
        # Its summary, or a SLURM batch's cases recorded so far: the entries
        # the session page and case_files() read.
        found = batch_entries(directory)
        if found is None:
            return []
        entries = found[0]
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            files = case_files(base, session_type, date, time, index)
            scenario = entry.get("scenario")
            out.append(
                CaseResult(
                    index=index,
                    label=scenario
                    if isinstance(scenario, str) and scenario
                    else "unknown",
                    passed=_passed(entry.get("passed")),
                    errored=bool(entry.get("error")),
                    elapsed_seconds=_seconds(entry.get("elapsed_seconds")),
                    message=str(entry.get("message") or ""),
                    overrides=_strings(entry.get("overrides")),
                    files=files,
                    coverage=_read_json(files.coverage) if files else None,
                    pending=bool(entry.get("pending")),
                )
            )
    else:
        for index in _job_indices(directory, session_type):
            files = case_files(base, session_type, date, time, index)
            job = _job(base, directory, session_type, date, time, index)
            result = _read_json(files.result) if files else None
            result = result if isinstance(result, dict) else {}
            overrides = _hydra_overrides(job) if job is not None else []
            label = _override_label(overrides) or (files.stem if files else "unknown")
            out.append(
                CaseResult(
                    index=index,
                    label=label,
                    passed=_passed(result.get("passed")),
                    errored=files is None or not result,
                    elapsed_seconds=_seconds(result.get("elapsed_seconds")),
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
    return _varying_attributes([r.compared() for r in results])


def coverage_documents(results: list[CaseResult]) -> list[dict[str, Any]]:
    """The cases' coverage documents, for openodd's views: those they can
    read (:attr:`CaseResult.coverage_viewable`); :func:`skipped_coverage`
    names the others."""
    return [
        r.coverage
        for r in results
        if isinstance(r.coverage, dict) and r.coverage_viewable
    ]


def skipped_coverage(results: list[CaseResult]) -> list[str]:
    """The coverage files the views leave out because openodd cannot read
    them (#2012), by file name (the case's label without one)."""
    return [
        r.files.coverage.name if r.files is not None else r.label
        for r in results
        if isinstance(r.coverage, dict) and not r.coverage_viewable
    ]


def default_axes(results: list[CaseResult]) -> tuple[str | None, str | None]:
    """The two attributes the cases were drawn over that tell them apart
    most (the most distinct buckets), else ``None`` (openodd's choice)."""
    return _default_axes([r.compared() for r in results])


def _odd_cells(coverage: Any) -> dict[str, Any] | None:
    """The run's ``odd.cells`` table (``{"attributes": [...], "rows":
    [[[label, ...], ticks, seconds, meters], ...]}``), else ``None``: a
    coverage file from before openodd recorded the cells, or one whose
    attributes are not all names, which openodd's view leaves out."""
    if not isinstance(coverage, dict):
        return None
    odd = coverage.get("odd")
    cells = odd.get("cells") if isinstance(odd, dict) else None
    if not isinstance(cells, dict):
        return None
    attributes = cells.get("attributes")
    if not isinstance(attributes, list) or not all(
        isinstance(a, str) for a in attributes
    ):
        return None
    return cells


def _counted(ticks: Any) -> bool:
    """Whether a cell row counts: its ticks are a JSON number above 0 (an
    integer too long for a float too, #1964), not text, ``true`` or
    ``null``."""
    return isinstance(ticks, (int, float)) and not isinstance(ticks, bool) and ticks > 0


def _cell_seconds(row: list[Any]) -> float:
    """A counted cell row's seconds: a JSON number, at least 0; else 0 (text,
    ``null``, missing; an integer too long for a float, #1964)."""
    value = row[2] if len(row) > 2 else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        return max(0.0, float(value))
    except OverflowError:
        return 0.0


def _in_cells(
    cells: dict[str, Any], conditions: list[tuple[str, str]], min_seconds: float
) -> bool:
    """Whether the run whose cells are *cells* is in the cell of
    *conditions* as openodd's ``CoverageView.matrix`` counts it: it has a
    row with every condition's bucket at once, counted (ticks above 0), and
    its counted rows there add up to at least *min_seconds*."""
    attributes = cells["attributes"]
    indices = []
    for attribute, bucket in conditions:
        if attribute not in attributes:
            # The view knows the attribute but this run did not record it.
            return False
        indices.append((attributes.index(attribute), bucket))
    found = False
    seconds = 0.0
    for row in cells.get("rows") or []:
        if not isinstance(row, list) or len(row) < 2:
            continue
        labels = row[0]
        if (
            not isinstance(labels, list)
            or len(labels) != len(attributes)
            or not all(isinstance(label, str) for label in labels)
            or any(labels[i] != bucket for i, bucket in indices)
            or not _counted(row[1])
        ):
            continue
        found = True
        seconds += _cell_seconds(row)
    return found and seconds >= min_seconds


def cases_in_cell(
    results: list[CaseResult],
    conditions: list[tuple[str, str]],
    *,
    min_seconds: float = 0.0,
) -> list[CaseResult]:
    """The cases openodd's ``CoverageView.matrix`` counts in the cell of
    *conditions* (the cell drilled into, ``[(attribute, bucket), ...]``):
    those that spent time with every condition's bucket *at once*, read off
    the joint ``odd.cells`` table of their coverage document -- not merely in
    each bucket at some moment. As the view does, a condition on an
    attribute no run's cells are of is dropped, a second one on the same
    attribute too, and *min_seconds* (the view's, 0 by default) is the least
    a run must stay in the cell.

    A case whose coverage document has no ``odd.cells`` (or one the view
    cannot read), or a document openodd rejects altogether
    (:attr:`CaseResult.coverage_viewable`, #2012), is in no cell, as the view
    leaves it out; with no
    *conditions* (the whole matrix) every case is listed."""
    if not conditions:
        return list(results)
    viewable = [r for r in results if r.coverage_viewable]
    known: set[str] = set()
    for r in viewable:
        cells = _odd_cells(r.coverage)
        if cells is not None:
            known.update(cells["attributes"])
    joint: list[tuple[str, str]] = []
    for attribute, bucket in conditions:
        if attribute in known and all(attribute != a for a, _ in joint):
            joint.append((attribute, bucket))
    found = []
    for r in viewable:
        cells = _odd_cells(r.coverage)
        if cells is not None and _in_cells(cells, joint, min_seconds):
            found.append(r)
    return found


# ── Comparing sessions ──────────────────────────────────────────────────────


def pending_count(found: list[CaseResult]) -> int:
    """How many of a session's cases are pending (a SLURM batch's cases
    without an outcome yet), which :func:`compare` leaves out."""
    return sum(1 for r in found if r.pending)


def compare(
    sessions: list[tuple[str, list[CaseResult]]],
    *,
    x: str | None = None,
    y: str | None = None,
) -> Comparison:
    """Compare sessions (oldest first: the first is the baseline) case by case
    and ODD cell by cell, the cells along *x* and *y* (the attributes the
    cases were drawn over, :func:`default_axes` when not given): the checked
    ``ui.comparison``'s, fed each case's :meth:`CaseResult.compared`.

    A pending case (a SLURM batch's case without an outcome yet) is left out,
    as if its session had not run it: it has no verdict to tally or to compare
    with, and counting it as not passing would report a regression and lower
    the pass rate before the case has finished. The compare page shows how
    many cases of each session are pending (:func:`pending_count`) beside its
    pass rate. A task that crashed recorded no outcome, so its case is pending
    until the batch is collected."""
    return _compare(
        [
            (name, [r.compared() for r in found if not r.pending])
            for name, found in sessions
        ],
        x=x,
        y=y,
    )
