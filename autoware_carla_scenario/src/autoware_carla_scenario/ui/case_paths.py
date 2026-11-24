"""Where a session's cases are on disk: the viewer's path guard and case stems.

What the viewer builds a path with from a request value: the directory of a
session (:func:`session_dir`) or of a multirun job (:func:`job_dir`), both
through :func:`contained`, which refuses a path that resolves outside the
viewer's base; the files one case leaves (:class:`CaseFiles`), and the stems
of the cases in a directory, in the order a runner hands them out
(:func:`stem_order`), and the one case of a directory that holds one run
(whose result :func:`result_path` finds); what a batch's caches are kept by
(:func:`batch_stamp`) and whether its summary is current; and the job indices
of a session (a job directory's name read by :func:`job_index`, the one rule
every reader of a multirun session uses), whether a case's file is its own
(:func:`own_file`: not a symbolic link, which could name a file outside the
viewer's base) and the ``scenario=`` override among a case's overrides, or
the one a batch case's label gives (:func:`scenario_overrides`); and the
session page of a batch directory (:func:`batch_session_url`).

Split out of ``ui.cases`` and ``ui.results``, which re-export every name
moved here and keep what reads the JSON documents of any type (``Any``) a
batch leaves, so that the library check compiles it (``docs/typecheck.md``,
"Checking the framework itself").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# A case's trajectory log and its recording are named by the checked
# run_outputs's rule (#1678), as rerun and ui.web_viewer_paths name them; its
# result and coverage file by run_outputs's suffixes too (#1704), and its
# environment (#1713) and ODD archive (#1748).
from ..run_outputs import (
    COVERAGE_SUFFIX,
    ENVIRONMENT_SUFFIX,
    ODD_ARCHIVE_SUFFIX,
    RECORDING_SUFFIX,
    RESULT_SUFFIX,
    TRAJECTORY_SUFFIX,
)

__all__ = [
    "INDEX_DIGITS",
    "MAX_INDEX",
    "SESSION_TYPES",
    "CaseFiles",
    "is_index",
    "batch_session_url",
    "batch_stamp",
    "contained",
    "job_dir",
    "job_index",
    "own_file",
    "scenario_overrides",
    "last_result_path",
    "result_path",
    "session_dir",
    "session_order",
    "stem_order",
]

#: The kinds of session the viewer lists (``scanner.scan_sessions``).
SESSION_TYPES = ("multirun", "batch", "single")

#: A date or time directory name, or a job index: no separator, no ``..``,
#: and no longer than a file name may be (NAME_MAX), so that a stat of it
#: fails as "not found" rather than with ENAMETOOLONG.
_COMPONENT = re.compile(r"^[A-Za-z0-9_.:+-]{1,255}$")

#: The most digits a case index the viewer reads from a request has (its URL
#: converter's and ``/compare/runs``'s): more is no case, and a longer one
#: makes a file name longer than NAME_MAX (ENAMETOOLONG) or an integer longer
#: than Python converts (#1964).
INDEX_DIGITS = 18
#: The largest case index :func:`job_dir` makes a directory of.
MAX_INDEX = 999_999_999_999_999_999

_TRAJECTORY = TRAJECTORY_SUFFIX
_RESULT = RESULT_SUFFIX
_COVERAGE = COVERAGE_SUFFIX
_STEM_NUMBER = re.compile(r"^(?P<base>.*?)-(?P<n>\d+)$")
#: A session's time directory that ``run_dirs.reserve_timestamped_dir``
#: numbered: ``{HH-MM-SS}-{n}``, a run started in the same second as another.
_SESSION_NUMBER = re.compile(r"^(?P<base>[^-]+-[^-]+-[^-]+)-(?P<n>\d+)$")


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
        return self.path(_COVERAGE)

    @property
    def odd_archive(self) -> Path:
        return self.path(ODD_ARCHIVE_SUFFIX)

    @property
    def environment(self) -> Path:
        return self.path(ENVIRONMENT_SUFFIX)

    @property
    def recording(self) -> Path:
        """The Rerun recording made of the trajectory log (``X.rrd`` beside
        ``X.trajectory.jsonl``, where the converter writes it)."""
        return self.path(RECORDING_SUFFIX)


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


def is_index(text: str) -> bool:
    """Whether *text* is a case index: ASCII digits (``"²".isdigit()`` is
    true too, and ``int()`` refuses it), no more than :data:`INDEX_DIGITS`."""
    return text.isascii() and text.isdigit() and len(text) <= INDEX_DIGITS


def job_index(name: str) -> Optional[int]:
    """The case index a multirun job directory named *name* holds, or
    ``None`` when it is no job's: ASCII digits (:func:`is_index`), with no
    leading zero (``0`` itself aside). A job's directory is named by its
    index as :func:`job_dir` makes it (``str(index)``), so ``01`` is no job:
    it is neither counted nor opened, where it used to be counted as job 1
    and then not found."""
    if not is_index(name) or (len(name) > 1 and name.startswith("0")):
        return None
    return int(name)


def own_file(path: Path) -> bool:
    """Whether *path* is a file of its case's own: a file, and not a symbolic
    link. The case's directory was found under the viewer's base
    (:func:`contained`); a link in it may name any file on the machine (one
    in an unpacked results archive, say), so the viewer neither reads nor
    serves one."""
    return not path.is_symlink() and path.is_file()


def job_dir(base: Path, date: str, time: str, index: int) -> Optional[Path]:
    """The directory of job *index* of multirun session *date*/*time*, or
    ``None`` when the names do not make one under *base* (also for an index
    above :data:`MAX_INDEX`, #1964)."""
    directory = session_dir(base, "multirun", date, time)
    if directory is None or index < 0 or index > MAX_INDEX:
        return None
    return contained(directory, str(index))


def stem_order(stem: str) -> tuple[str, int]:
    """Where *stem* comes in the order a runner hands stems out: ``X``, then
    ``X-1``, ``X-2``, ... ``X-10`` (by number, not by text)."""
    match = _STEM_NUMBER.match(stem)
    if match is None:
        return stem, 0
    return match["base"], int(match["n"])


def session_order(time: str) -> tuple[str, int]:
    """Where a session's *time* directory comes among those of its date:
    ``HH-MM-SS``, then ``HH-MM-SS-1``, ``HH-MM-SS-2``, ... ``HH-MM-SS-10``
    (by number, not by text), the order ``run_dirs.reserve_timestamped_dir``
    hands them out in (#1845).  Only a number after the three fields of the
    time is one: ``12-00-01`` is not ``12-00`` numbered 1."""
    match = _SESSION_NUMBER.match(time)
    if match is None:
        return time, 0
    return match["base"], int(match["n"])


def _stems_in(directory: Path) -> list[str]:
    """Every stem with a result or a trajectory log in *directory*, in
    :func:`stem_order`; a symbolic link of either name is left out."""
    stems: set[str] = set()
    try:
        paths = list(directory.iterdir())
    except OSError:
        return []
    for path in paths:
        name = path.name
        for suffix in (_RESULT, _TRAJECTORY):
            # A symbolic link is no case's file (own_file()).
            if (
                name.endswith(suffix)
                and len(name) > len(suffix)
                and not path.is_symlink()
            ):
                stems.add(name[: -len(suffix)])
    return sorted(stems, key=stem_order)


def result_path(directory: Path) -> Optional[Path]:
    """The ``_result.json`` of the run a directory that holds one run (a
    multirun job, a single run) shows: the first, in :func:`stem_order`, so
    ``X``'s rather than ``X-1``'s, which a plain sort puts first (``-`` sorts
    before ``_``). ``None`` when it has none."""
    for stem in _stems_in(directory):
        path = directory / f"{stem}{_RESULT}"
        if own_file(path):
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
    return path if own_file(path) else None


def _only_case(directory: Path) -> CaseFiles | None:
    """The case of a directory that holds one run (a multirun job, a single
    run): the stem of its :func:`result_path`, else the first with a log."""
    result = result_path(directory)
    if result is not None:
        return CaseFiles(directory, result.name[: -len(_RESULT)])
    stems = _stems_in(directory)
    return CaseFiles(directory, stems[0]) if stems else None


def _shared_overrides(jobs: list[list[str]]) -> set[str]:
    """The overrides every one of *jobs* (each job's overrides) holds, leaving
    out a job whose overrides could not be read (an empty list); none when no
    job's could (``ui.cases.session_overrides``)."""
    read = [overrides for overrides in jobs if overrides]
    if not read:
        return set()
    shared = set(read[0])
    for overrides in read[1:]:
        shared &= set(overrides)
    return shared


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _stat(path: Path) -> tuple[int, int]:
    """*path*'s modification time (ns) and size, ``(0, -1)`` when missing."""
    try:
        info = path.stat()
    except OSError:
        return (0, -1)
    return (info.st_mtime_ns, info.st_size)


def batch_stamp(directory: Path) -> tuple[tuple[int, int], ...]:
    """What a batch's entries and case files are read from, cheaply: the
    modification time and size of its ``batch_results.json``, of a SLURM
    batch's manifest and ``collected.json``, and of the directories its
    submissions and case outcomes are written to, and of *directory* itself.

    A few ``stat`` calls, whatever the size of the batch: the launcher writes
    every record by renaming a new file over it (``slurm.manifest.write_json``)
    and a task removes a case's old outcome before it runs it again, so a
    record added, replaced or removed changes its directory's time.  The
    viewer's caches of a batch (``ui.cases.batch_case_files``, and the session and
    case pages' in ``ui.scanner``) are kept while this stays the same.
    """
    slurm = directory / "slurm"
    # tuple() of a list: the library check reads a variable-length tuple
    # returned as a list, which a tuple display is not.
    return tuple(
        [
            _stat(directory / "batch_results.json"),
            _stat(slurm / "manifest.json"),
            _stat(slurm / "collected.json"),
            _stat(slurm / "submissions"),
            _stat(slurm / "results"),
            _stat(directory),
        ]
    )


def _summary_is_current(directory: Path) -> bool:
    """Whether a batch's ``batch_results.json`` is its latest summary: not
    older than a SLURM batch's last submission or case outcome.

    A resubmission (``scenario-slurm resubmit``, the viewer's "Resubmit")
    leaves the summary of the earlier collect in place until its own collect
    rewrites it; until then the summary is stale, and the cases are read from
    what the tasks record (``slurm/results``).  A local batch has no
    ``slurm/``, so its summary is always current.
    """
    slurm = directory / "slurm"
    summary = _stat(directory / "batch_results.json")[0]
    collected = _stat(slurm / "collected.json")
    if collected[1] >= 0:
        # A collect writes the summary, then collected.json.
        summary = min(summary, collected[0])
    latest = max(_stat(slurm / "submissions")[0], _stat(slurm / "results")[0])
    return latest <= summary


def _job_indices(directory: Path, session_type: str) -> list[int]:
    """The case indices of a multirun or single session in *directory*."""
    if session_type != "multirun":
        return [0]
    indices: list[int] = []
    try:
        entries = list(directory.iterdir())
    except OSError:
        # A directory it cannot read (another user's, mode 0700) has no case
        # it can show, as _stems_in() finds none (#2183).
        return indices
    for p in entries:
        index = job_index(p.name)
        if index is not None and p.is_dir():
            indices.append(index)
    return sorted(indices)


def _override_label(overrides: list[str]) -> str | None:
    """The ``scenario=`` override among *overrides*, if any."""
    for o in overrides:
        if o.startswith("scenario="):
            return o.split("=", 1)[1]
    return None


def scenario_overrides(overrides: list[str], label: str) -> list[str]:
    """*overrides*, a batch case's, with a ``scenario=`` override for its
    batch *label* when they name none (``ui.scanner``, #2344).

    The scenario page's Re-run needs a ``scenario=`` override, which a batch
    case's overrides do not name, so one is added from its label: the batch's
    name for the case, less the ``#k`` it is numbered with when its sweep gave
    the scenario more than one case (``examples.argv._case_labels``).  None is
    added for an empty label.  The label is ``name`` or ``name#k`` alone (a
    local batch, a SLURM one, a glob's resolved names); a scenario's name has
    no ``#`` of its own, which Hydra's override grammar refuses unquoted.
    """
    if any(o.startswith("scenario=") for o in overrides):
        return overrides
    if not label:
        return overrides
    name, sep, number = label.rpartition("#")
    if not (sep and name and number.isascii() and number.isdigit()):
        name = label
    return [f"scenario={name}", *overrides]


def batch_session_url(batch_dir: str, base: Path) -> Optional[str]:
    """``/session/batch/<date>/<time>`` of a batch directory under *base*'s
    ``outputs/``; None for one anywhere else (``ui.app._session_of``)."""
    try:
        rel = Path(batch_dir).resolve().relative_to(base.resolve() / "outputs")
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) == 2:
        return f"/session/batch/{parts[0]}/{parts[1]}"
    return None
