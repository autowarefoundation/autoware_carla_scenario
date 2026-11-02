"""Where a session's cases are on disk: the viewer's path guard and case stems.

What the viewer builds a path with from a request value: the directory of a
session (:func:`session_dir`) or of a multirun job (:func:`job_dir`), both
through :func:`contained`, which refuses a path that resolves outside the
viewer's base; the files one case leaves (:class:`CaseFiles`), and the stems
of the cases in a directory, in the order a runner hands them out
(:func:`stem_order`), and the one case of a directory that holds one run
(whose result :func:`result_path` finds).

Split out of ``ui.cases``, which re-exports every name here and keeps what
reads the JSON documents of any type (``Any``) a batch leaves, so that the
library check compiles it (``docs/typecheck.md``, "Checking the framework
itself").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "SESSION_TYPES",
    "CaseFiles",
    "contained",
    "job_dir",
    "last_result_path",
    "result_path",
    "session_dir",
    "stem_order",
]

#: The kinds of session the viewer lists (``scanner.scan_sessions``).
SESSION_TYPES = ("multirun", "batch", "single")

#: A date or time directory name, or a job index: no separator, no ``..``,
#: and no longer than a file name may be (NAME_MAX), so that a stat of it
#: fails as "not found" rather than with ENAMETOOLONG.
_COMPONENT = re.compile(r"^[A-Za-z0-9_.:+-]{1,255}$")

_TRAJECTORY = ".trajectory.jsonl"
_RESULT = "_result.json"
_STEM_NUMBER = re.compile(r"^(?P<base>.*?)-(?P<n>\d+)$")


@dataclass(frozen=True)
class CaseFiles:
    """The files of one case: *directory*/*stem* plus each suffix."""

    directory: Path
    stem: str

    def path(self, suffix: str) -> Path:
        return self.directory / f"{self.stem}{suffix}"

    @property
    def trajectory(self) -> Path:
        return self.path(_TRAJECTORY)

    @property
    def result(self) -> Path:
        return self.path(_RESULT)

    @property
    def coverage(self) -> Path:
        return self.path("_coverage.json")

    @property
    def odd_archive(self) -> Path:
        return self.path("_odd.tar.gz")

    @property
    def environment(self) -> Path:
        return self.path("_environment.json")

    @property
    def recording(self) -> Path:
        """The Rerun recording made of the trajectory log (``X.rrd`` beside
        ``X.trajectory.jsonl``, where the converter writes it)."""
        return self.path(".rrd")


def _safe(component: str) -> bool:
    return bool(_COMPONENT.match(component)) and component not in (".", "..")


def contained(root: Path, *parts: str) -> Optional[Path]:
    """*root* joined with *parts*, or ``None`` when that leaves *root*.

    The joined path is resolved (``..`` and symlinks followed) and must be
    *root*, resolved, or under it: the one check every path the viewer builds
    from a request value goes through, so a URL cannot name a file outside the
    viewer's base.
    """
    path = root.joinpath(*parts)
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return path


def session_dir(base: Path, session_type: str, date: str, time: str) -> Optional[Path]:
    """The directory of a session, or ``None`` when the names do not make one
    under *base* (an unknown type, a name with a separator or ``..``, or one
    that resolves outside ``multirun/`` or ``outputs/``, as a symlink can)."""
    if session_type not in SESSION_TYPES or not (_safe(date) and _safe(time)):
        return None
    root = "multirun" if session_type == "multirun" else "outputs"
    return contained(base / root, date, time)


def job_dir(base: Path, date: str, time: str, index: int) -> Optional[Path]:
    """The directory of job *index* of multirun session *date*/*time*, or
    ``None`` when the names do not make one under *base*."""
    directory = session_dir(base, "multirun", date, time)
    if directory is None or index < 0:
        return None
    return contained(directory, str(index))


def stem_order(stem: str) -> tuple[str, int]:
    """Where *stem* comes in the order a runner hands stems out: ``X``, then
    ``X-1``, ``X-2``, ... ``X-10`` (by number, not by text)."""
    match = _STEM_NUMBER.match(stem)
    if match is None:
        return stem, 0
    return match["base"], int(match["n"])


def _stems_in(directory: Path) -> list[str]:
    """Every stem with a result or a trajectory log in *directory*, in
    :func:`stem_order`."""
    stems: set[str] = set()
    try:
        names = [p.name for p in directory.iterdir()]
    except OSError:
        return []
    for name in names:
        for suffix in (_RESULT, _TRAJECTORY):
            if name.endswith(suffix) and len(name) > len(suffix):
                stems.add(name[: -len(suffix)])
    return sorted(stems, key=stem_order)


def result_path(directory: Path) -> Optional[Path]:
    """The ``_result.json`` of the run a directory that holds one run (a
    multirun job, a single run) shows: the first, in :func:`stem_order`, so
    ``X``'s rather than ``X-1``'s, which a plain sort puts first (``-`` sorts
    before ``_``). ``None`` when it has none."""
    for stem in _stems_in(directory):
        path = directory / f"{stem}{_RESULT}"
        if path.is_file():
            return path
    return None


def last_result_path(directory: Path) -> Optional[Path]:
    """The ``_result.json`` of the last run in *directory*: the last stem in
    :func:`stem_order` (``X-1`` after ``X``, ``X-10`` after ``X-2``), which is
    the newest, since a run takes the first stem no earlier run has taken
    (``scenario_runner._run_file_stem``). ``None`` when that run left no
    result, even if an earlier one did: an earlier run's verdict is not the
    last run's."""
    stems = _stems_in(directory)
    if not stems:
        return None
    path = directory / f"{stems[-1]}{_RESULT}"
    return path if path.is_file() else None


def _only_case(directory: Path) -> CaseFiles | None:
    """The case of a directory that holds one run (a multirun job, a single
    run): the stem of its :func:`result_path`, else the first with a log."""
    result = result_path(directory)
    if result is not None:
        return CaseFiles(directory, result.name[: -len(_RESULT)])
    stems = _stems_in(directory)
    return CaseFiles(directory, stems[0]) if stems else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
