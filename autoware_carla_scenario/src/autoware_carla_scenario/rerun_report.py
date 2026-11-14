"""How converting a run's trajectory logs into Rerun recordings went.

:class:`RerunConversion` is one log's conversion, :func:`latest` keeps the last
conversion of each recording, and :func:`report` is the summary a run prints of
them; :func:`_draw_radius_argument` is the converter's keyword for the radius
a recording draws in.  :mod:`.rerun`, which converts the logs, re-exports them
under the names they had there (#2124): free of the converter, the thread
pool and the Rerun SDK, they are compiled by the library check
(docs/typecheck.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "RerunConversion",
    "latest",
    "report",
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


def report(conversions: list[RerunConversion]) -> str:
    """One line on *conversions*, then one per failure, for a run's summary.

    A recording converted more than once (a retried run) counts once, by its
    last conversion (:func:`latest`).
    """
    conversions = latest(conversions)
    failed = [c for c in conversions if not c.ok]
    lines = [
        f"Rerun recordings: {len(conversions) - len(failed)}/{len(conversions)} "
        "written (open one with: rerun <file.rrd>)"
    ]
    lines.extend(
        f"  [{'SKIPPED' if c.skipped else 'FAILED'}] {c.label}: {c.error}"
        for c in failed
    )
    return "\n".join(lines)


def _draw_radius_argument(draw_radius_m: Optional[float]) -> dict[str, float]:
    """The converter's keyword for *draw_radius_m*; none when ``None``, so that
    what does not ask for it goes on as it did."""
    argument: dict[str, float] = {}
    if draw_radius_m is not None:
        argument["draw_radius_m"] = draw_radius_m
    return argument
