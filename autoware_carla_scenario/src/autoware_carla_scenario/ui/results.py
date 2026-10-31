"""A session's results on its ODD, and sessions compared case by case.

Each case of a session is read with what it left on disk (``ui.cases``): its
verdict (``batch_results.json`` or ``{stem}_result.json``), the case it ran (a
sweep's overrides; the ODD values and buckets it was drawn at, from
``{stem}_coverage.json``'s ``case``), its coverage document, and a few metrics
of how the ego drove, read off its trajectory log (:func:`case_metrics`, which
hands the ego's ticks to ``ui.drive_metrics``, the box geometry and metric
arithmetic the library check compiles).

The ODD view is openodd's own coverage component (``openodd.CoverageView``)
and its case plot (``openodd.CasePlot``), fed the sessions' coverage
documents. Comparing sessions matches their cases by what they ran -- the
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
from pathlib import Path
from typing import Any

from .cases import CaseFiles, case_files, job_dir, session_dir

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
from ..utils.clearance import _HIDDEN_BELOW_M

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


def _passed(value: Any) -> bool | None:
    """A verdict: *value* when it is a boolean, else ``None``."""
    return value if isinstance(value, bool) else None


def _seconds(value: Any) -> float | None:
    """A duration: *value* when it is a number (not a boolean), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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
    drive = DriveMetrics()
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
                    # A road user parked under the map (hidden) is out of
                    # the scene.
                    if other == ego or float(state[3]) < _HIDDEN_BELOW_M:
                        continue
                    drive.gap(
                        _box_distance(
                            ego_box,
                            _box(
                                state[1],
                                state[2],
                                state[6],
                                extents.get(other, (0.3, 0.3)),
                            ),
                        )
                    )
                drive.tick(t, float(me[1]), float(me[2]), speed)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return {}
    metrics = drive.to_dict()
    _metrics_cache[str(trajectory)] = (stamp, metrics)
    return metrics


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
    """The cases' coverage documents, for openodd's views."""
    return [r.coverage for r in results if isinstance(r.coverage, dict)]


def default_axes(results: list[CaseResult]) -> tuple[str | None, str | None]:
    """The two attributes the cases were drawn over that tell them apart
    most (the most distinct buckets), else ``None`` (openodd's choice)."""
    return _default_axes([r.compared() for r in results])


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


def compare(
    sessions: list[tuple[str, list[CaseResult]]],
    *,
    x: str | None = None,
    y: str | None = None,
) -> Comparison:
    """Compare sessions (oldest first: the first is the baseline) case by case
    and ODD cell by cell, the cells along *x* and *y* (the attributes the
    cases were drawn over, :func:`default_axes` when not given): the checked
    ``ui.comparison``'s, fed each case's :meth:`CaseResult.compared`."""
    return _compare(
        [(name, [r.compared() for r in found]) for name, found in sessions],
        x=x,
        y=y,
    )
