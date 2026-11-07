"""Resuming a local batch into its own directory: ``--resume-into DIR`` (#1957).

A batch resumed with ``--resume-from N`` alone runs its remaining cases in a
new directory, and the directory of the batch it resumes keeps the cases that
batch never finished as pending (``"pending": true`` in its
``batch_results.json``): ``scenario-gate`` refuses it, and over both
directories it would count a case killed after writing its gate record twice.
``--resume-into DIR`` runs them in *DIR* instead (``examples.run.run_batch``):
the cases before the resume point keep their entries, the batch's summary is
rewritten (atomically, as every summary is) from those and the cases run
now, and before the remaining cases run, the outputs their earlier run left
under their stems are removed, so each case is counted once.

This module reads and checks *DIR*'s summary: it must list the same cases
(each one's name and sweep case), name each case's file stem (``stem``, which
a batch has written since #1957), and hold a result for every case before
the resume point.  Nothing is removed from a directory that fails a check.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .failure import Failure

__all__ = [
    "BATCH_RESULTS",
    "KeptResult",
    "ResumeIntoError",
    "ResumedBatch",
    "read_resumed_batch",
]

#: A local batch's summary, which ``--resume-into`` reads and rewrites.
BATCH_RESULTS = "batch_results.json"


class ResumeIntoError(ValueError):
    """``--resume-into DIR`` names a directory the batch cannot resume into;
    the message says why.  Raised before anything is removed or run."""


def _plain_name(stem: object) -> bool:
    """Whether *stem* is a file stem in its directory: a non-empty string that
    names no other directory (no separator, not ``.`` or ``..``)."""
    return (
        isinstance(stem, str)
        and stem not in ("", ".", "..")
        and "/" not in stem
        and "\\" not in stem
        and "\0" not in stem
    )


@dataclass(frozen=True)
class ResumedBatch:
    """The batch ``--resume-into`` resumes: its directory and the entries of
    its ``batch_results.json``, checked against the batch's cases."""

    directory: Path
    entries: list[dict[str, Any]]

    @property
    def stems(self) -> list[str]:
        """Each case's file stem, as the summary records it."""
        return [str(entry["stem"]) for entry in self.entries]

    def first_pending(self) -> int | None:
        """The index of the first case the batch has no result for (a
        ``pending`` entry), or ``None`` when every case has one."""
        for i, entry in enumerate(self.entries):
            if entry.get("pending") is True:
                return i
        return None

    def check_kept(self, skip: int, labels: Sequence[str]) -> None:
        """Refuse a resume point (*skip* cases kept) past a case the batch has
        no result for: the summary would keep it pending."""
        for i in range(min(skip, len(self.entries))):
            if self.entries[i].get("pending") is True:
                raise ResumeIntoError(
                    f"{self.directory}: case {i + 1} ({labels[i]}) has no result "
                    f"there yet; resume from case {i + 1} or earlier "
                    f"(--resume-from {i + 1})"
                )

    def check_stems(self, skip: int, classes: Sequence[str]) -> None:
        """Refuse the cases from *skip* on unless each one's recorded stem is
        its scenario class's (``Class`` or ``Class-N``): a case named after
        another class is not the one that ran there.  *classes* are the
        scenario classes of the cases run now, from case *skip* on."""
        for offset, name in enumerate(classes):
            i = skip + offset
            stem = self.stems[i]
            if re.fullmatch(re.escape(name) + r"(?:-[1-9][0-9]*)?", stem) is None:
                raise ResumeIntoError(
                    f"{self.directory}: case {i + 1} wrote its outputs as "
                    f"{stem!r} there, which is not a run of {name}; resume "
                    "with the scenarios and overrides of the batch it ran"
                )


def read_resumed_batch(
    directory: Path, labels: Sequence[str], overrides: Sequence[list[str]] | None
) -> ResumedBatch:
    """Read and check the summary of the batch in *directory* that a batch of
    the cases *labels* (with their sweep cases, *overrides*, for a
    ``--multirun`` batch; ``None`` for one without) resumes.

    Raises:
        ResumeIntoError: *directory* is not a directory, is a SLURM batch's,
            holds no summary or one that cannot be read, lists other cases
            or sweep cases, or names no file stem (or an unsafe or repeated
            one) for a case.
    """
    shown = str(directory)
    if not directory.is_dir():
        raise ResumeIntoError(f"{shown}: no such directory")
    if (directory / "slurm" / "manifest.json").is_file():
        raise ResumeIntoError(
            f"{shown}: a SLURM batch's directory; --resume-into resumes a "
            "local batch (resubmit a SLURM batch's cases with scenario-slurm)"
        )
    path = directory / BATCH_RESULTS
    if not path.is_file():
        raise ResumeIntoError(
            f"{shown}: holds no {BATCH_RESULTS}, so it is not the directory of "
            "a local batch"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, ValueError) as exc:
        raise ResumeIntoError(
            f"{path}: the batch summary cannot be read ({exc})"
        ) from exc
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        raise ResumeIntoError(
            f"{path}: not a batch summary (not a JSON array of objects)"
        )
    entries: list[dict[str, Any]] = data
    if len(entries) != len(labels):
        raise ResumeIntoError(
            f"{path}: lists {len(entries)} case(s), this batch has "
            f"{len(labels)}; resume with the scenarios and overrides of the "
            "batch it ran"
        )
    for i, (entry, label) in enumerate(zip(entries, labels)):
        if entry.get("scenario") != label:
            raise ResumeIntoError(
                f"{path}: case {i + 1} is {entry.get('scenario')!r} there and "
                f"{label!r} here; resume with the scenarios and overrides of "
                "the batch it ran"
            )
        recorded = entry.get("overrides")
        expected = None if overrides is None else list(overrides[i])
        if recorded != expected:
            raise ResumeIntoError(
                f"{path}: case {i + 1} ({label}) ran with the sweep case "
                f"{recorded!r} there and {expected!r} here; resume with the "
                "scenarios, overrides and --multirun of the batch it ran"
            )
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        stem = entry.get("stem")
        if "stem" not in entry:
            raise ResumeIntoError(
                f"{path}: names no file stem for case {i + 1} ({labels[i]}) "
                "-- it was written before a batch recorded its cases' stems -- "
                "so the outputs the cases run again left there cannot be told "
                "from the others', and nothing is removed; resume it into a "
                "new directory (without --resume-into)"
            )
        if not _plain_name(stem) or stem in seen:
            raise ResumeIntoError(
                f"{path}: case {i + 1} ({labels[i]}) names the file stem "
                f"{stem!r}, which is not a file name of its own in the directory"
            )
        seen.add(str(stem))
    return ResumedBatch(directory, entries)


@dataclass(frozen=True)
class _KeptStatus:
    """A condition's status as a kept entry recorded it, for the summary's
    table."""

    label: str
    satisfied: bool
    message: str


class KeptResult:
    """The result of a case the resumed batch ran and ``--resume-into`` keeps:
    its entry of the summary, read as a result, so that the batch's summary
    (``batch_report``) lists it as it was.  :meth:`to_dict` gives the entry
    back without what the summary writes of its own (the case's name, sweep
    case, stem and whether it errored)."""

    def __init__(self, entry: dict[str, Any]) -> None:
        self._entry = entry

    @property
    def failure(self) -> Failure | None:
        """Why the case failed, as its entry recorded it."""
        failure = self._entry.get("failure")
        return Failure.from_dict(failure) if isinstance(failure, dict) else None

    @property
    def passed(self) -> bool:
        return self._entry.get("passed") is True

    @property
    def message(self) -> str:
        message = self._entry.get("message")
        return message if isinstance(message, str) else ""

    @property
    def elapsed_seconds(self) -> float:
        elapsed = self._entry.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            return float(elapsed)
        return 0.0

    @property
    def condition_statuses(self) -> list[_KeptStatus]:
        statuses = self._entry.get("condition_statuses")
        if not isinstance(statuses, list):
            return []
        return [
            _KeptStatus(
                str(status.get("label", "")),
                status.get("satisfied") is True,
                str(status.get("message", "")),
            )
            for status in statuses
            if isinstance(status, dict)
        ]

    @property
    def rerun_recording(self) -> str | None:
        recording = self._entry.get("rerun_recording")
        return recording if isinstance(recording, str) else None

    @property
    def rerun_error(self) -> str | None:
        """Why its Rerun recording failed to convert, if the entry says."""
        why = self._entry.get("rerun_error")
        return why if isinstance(why, str) and why else None

    @property
    def errored(self) -> bool:
        """Whether the case raised rather than ran (``"error": true``)."""
        return self._entry.get("error") is True

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self._entry.items()
            if key not in ("scenario", "overrides", "stem", "error")
        }
