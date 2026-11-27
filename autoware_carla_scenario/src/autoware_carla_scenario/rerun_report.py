"""How converting a run's trajectory logs into Rerun recordings went.

:class:`RerunConversion` is one log's conversion, :func:`latest` keeps the last
conversion of each recording, and :func:`report` is the summary a run prints of
them, in the lines :func:`summary_lines` makes, which ``batch_report`` prints
a batch's with too (#2582); :func:`_draw_radius_argument` is the converter's keyword for the radius
a recording draws in.  :mod:`.rerun`, which converts the logs, re-exports them
under the names they had there (#2124): free of the converter, the thread
pool and the Rerun SDK, they are compiled by the library check
(docs/typecheck.md).  :func:`strict_coverage_refusal` is why
:func:`.rerun.check_strict_coverage` refuses a strict conversion, and
:func:`rerun_errors` which cases' recordings failed, for ``examples.run``'s
summary (#2364).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "RerunConversion",
    "latest",
    "report",
    "rerun_errors",
    "strict_coverage_refusal",
    "summary_lines",
]


@dataclass(frozen=True)
class RerunConversion:
    """How converting one trajectory log went.

    Attributes:
        label: The run the log belongs to (its file stem, ``CutInScenario-3``).
        trajectory: The log.
        recording: The recording made of it, or that would have been.
        error: Why the conversion failed (``ExceptionType: message``), or
            ``None`` when it did not.
        stats: What the converter returned (``Stats``): ticks, actors, the
            record types and fields it skipped, ...; ``None`` on failure.
        skipped: The log converted, but the recording was not replaced
            (:func:`.rerun.convert_outputs`: it holds sensors that could not be
            kept); *error* says why.
    """

    label: str
    trajectory: Path
    recording: Path
    error: Optional[str] = None
    stats: Optional[dict[str, Any]] = field(default=None, compare=False)
    skipped: bool = False

    @property
    def ok(self) -> bool:
        """Whether the recording was written."""
        return self.error is None


def latest(conversions: list[RerunConversion]) -> list[RerunConversion]:
    """*conversions* with only the last of each recording's, in the order the
    recordings were first converted.

    A retried run is converted again into the recording of the attempt it
    replaces, so :meth:`.rerun.RerunConverter.wait` returns that recording twice;
    the earlier conversion no longer says how it went.
    """
    last: dict[str, RerunConversion] = {}
    for conversion in conversions:
        last[str(conversion.recording)] = conversion
    return list(last.values())


def summary_lines(
    written: int, total: int, failures: list[tuple[str, str, str]]
) -> list[str]:
    """The Rerun lines of a run's summary: *written* of *total* recordings
    written, then one line per failure in *failures*, each
    ``(mark, label, error)`` (*mark* ``FAILED`` or ``SKIPPED``).  What
    :func:`report` and ``batch_report.print_summary`` both print."""
    lines = [
        f"Rerun recordings: {written}/{total} "
        "written (open one with: rerun <file.rrd>)"
    ]
    for mark, label, error in failures:
        lines.append(f"  [{mark}] {label}: {error}")
    return lines


def report(conversions: list[RerunConversion]) -> str:
    """One line on *conversions*, then one per failure, for a run's summary.

    A recording converted more than once (a retried run) counts once, by its
    last conversion (:func:`latest`).
    """
    conversions = latest(conversions)
    failed = [c for c in conversions if not c.ok]
    failures = [
        ("SKIPPED" if c.skipped else "FAILED", c.label, f"{c.error}") for c in failed
    ]
    return "\n".join(
        summary_lines(len(conversions) - len(failed), len(conversions), failures)
    )


def _draw_radius_argument(draw_radius_m: Optional[float]) -> dict[str, float]:
    """The converter's keyword for *draw_radius_m*; none when ``None``, so that
    what does not ask for it goes on as it did."""
    argument: dict[str, float] = {}
    if draw_radius_m is not None:
        argument["draw_radius_m"] = draw_radius_m
    return argument


def strict_coverage_refusal(
    known: Optional[set[str]], written: list[str], version: Optional[str]
) -> Optional[str]:
    """Why a strict conversion by a converter that knows the record types
    *known* would fail every log; ``None`` when it would not.

    *written* is the record types the framework writes
    (:func:`.rerun.written_records`), *known* the converter's
    ``RECORD_TYPES`` (``None``: it has none, and is not checked) and
    *version* its ``__version__`` when that is text.
    """
    if known is None:
        return None
    missing = sorted([t for t in written if t not in known])
    if not missing:
        return None
    which = (
        f"autoware-carla-scenario-rerun {version}"
        if version is not None
        else "the converter"
    )
    return (
        f"rerun.strict is set, but {which} does not know "
        f"{', '.join(missing)}, which the framework writes, so every strict "
        "conversion would fail: set rerun.strict=false, or upgrade "
        "autoware-carla-scenario-rerun to a version that knows them"
    )


def rerun_errors(
    recordings: list[Optional[str]], conversions: list[RerunConversion]
) -> dict[int, str]:
    """Why each case's Rerun recording failed, by its position in
    *recordings* (each case's ``rerun_recording``): the cases whose recording
    is among the failed *conversions*.

    A recording converted more than once (a retried case, which writes under
    the stem of the attempt it replaces) goes by its last conversion."""
    failed: dict[str, str] = {}
    for conversion in latest(conversions):
        if conversion.error is not None:
            failed[str(conversion.recording)] = conversion.error
    errors: dict[int, str] = {}
    for i, recording in enumerate(recordings):
        if recording is not None and recording in failed:
            errors[i] = failed[recording]
    return errors
