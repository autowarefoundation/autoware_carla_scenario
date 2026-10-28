"""Which files on disk are a session's cases: the stem each case's files share.

A run names every file it writes after one stem (``scenario_runner``'s
``_run_file_stem``): ``{stem}_result.json``, ``{stem}.trajectory.jsonl``,
``{stem}_coverage.json``, ``{stem}_odd.tar.gz``, ``{stem}_environment.json``.
A multirun job directory and a single run's directory hold one case; a batch
directory holds all of its cases side by side (``CutInScenario``,
``CutInScenario-1``, ...), and ``batch_results.json`` lists their results
without naming the stems.  A SLURM batch's manifest names them
(``slurm/manifest.json``); for a local batch they are matched here: each entry
takes the first stem, in the order the runner hands them out, whose result is
the entry's (the same verdict, elapsed time and message) and whose recorded
case (its coverage file's ``case.overrides``) is among the entry's overrides.
A case that raised before it opened its trajectory log left its stem to the
next one, and one retried took a second stem, so a position alone is not
enough.

Free of heavy imports: the viewer reads these on every page.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SESSION_TYPES",
    "CaseFiles",
    "batch_case_files",
    "batch_entries",
    "case_files",
    "contained",
    "job_dir",
    "match_batch_stems",
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


def contained(root: Path, *parts: str) -> Path | None:
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


def session_dir(base: Path, session_type: str, date: str, time: str) -> Path | None:
    """The directory of a session, or ``None`` when the names do not make one
    under *base* (an unknown type, a name with a separator or ``..``, or one
    that resolves outside ``multirun/`` or ``outputs/``, as a symlink can)."""
    if session_type not in SESSION_TYPES or not (_safe(date) and _safe(time)):
        return None
    root = "multirun" if session_type == "multirun" else "outputs"
    return contained(base / root, date, time)


def job_dir(base: Path, date: str, time: str, index: int) -> Path | None:
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


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _same_result(entry: dict[str, Any], result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    return (
        bool(entry.get("passed")) == bool(result.get("passed"))
        and entry.get("elapsed_seconds") == result.get("elapsed_seconds")
        and (entry.get("message") or "") == (result.get("message") or "")
    )


def _same_case(entry: dict[str, Any], coverage: Any) -> bool:
    """Whether the case *coverage* recorded is among the entry's overrides
    (true when either side records none)."""
    overrides = entry.get("overrides")
    if not isinstance(overrides, list) or not isinstance(coverage, dict):
        return True
    case = coverage.get("case")
    own = case.get("overrides") if isinstance(case, dict) else None
    if not isinstance(own, list):
        return True
    given = {str(o).lstrip("+") for o in overrides}
    return all(str(o).lstrip("+") in given for o in own)


def match_batch_stems(
    directory: Path, entries: list[dict[str, Any]]
) -> list[str | None]:
    """The stem of each entry of a local batch's ``batch_results.json``.

    Each entry takes the first unused stem, in :func:`stem_order`, whose
    ``_result.json`` is the entry's result and whose coverage case is among
    the entry's overrides; ``None`` when none is (an errored case, which wrote
    no result).
    """
    stems = _stems_in(directory)
    results = {stem: _read_json(directory / f"{stem}{_RESULT}") for stem in stems}
    coverages: dict[str, Any] = {}
    used: set[str] = set()
    out: list[str | None] = []
    for entry in entries:
        chosen: str | None = None
        if isinstance(entry, dict) and not entry.get("error"):
            for stem in stems:
                if stem in used or not _same_result(entry, results.get(stem)):
                    continue
                if stem not in coverages:
                    coverages[stem] = _read_json(directory / f"{stem}_coverage.json")
                if _same_case(entry, coverages[stem]):
                    chosen = stem
                    break
        if chosen is not None:
            used.add(chosen)
        out.append(chosen)
    return out


def _manifest_stems(directory: Path, count: int) -> list[str | None] | None:
    """The stems a SLURM batch's manifest gives its cases, if it has one of
    *count* cases."""
    manifest = _read_json(directory / "slurm" / "manifest.json")
    if not isinstance(manifest, dict):
        return None
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != count:
        return None
    ordered = sorted(cases, key=lambda c: int(c.get("index", 0)))
    return [str(c["stem"]) if c.get("stem") else None for c in ordered]


def _slurm_entries(directory: Path) -> list[dict[str, Any]] | None:
    """The entries a SLURM batch's summary will have, from its manifest and
    the outcomes its tasks recorded (``slurm/results/<case>.json``), while it
    has no ``batch_results.json`` yet. A case without an outcome yet has
    ``passed`` ``None`` and ``pending`` true."""
    manifest = _read_json(directory / "slurm" / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        return None
    entries: list[dict[str, Any]] = []
    for case in sorted(manifest["cases"], key=lambda c: int(c.get("index", 0))):
        record = _read_json(
            directory / "slurm" / "results" / f"{int(case['index'])}.json"
        )
        entry: dict[str, Any] = {
            "scenario": case.get("label") or case.get("scenario") or "unknown",
            "overrides": list(case.get("case_overrides") or []),
        }
        if isinstance(record, dict) and isinstance(record.get("result"), dict):
            entry.update(record["result"])
            if record.get("errored"):
                entry["error"] = True
        else:
            entry.update(
                passed=None,
                message="No result yet (running under SLURM)",
                elapsed_seconds=None,
                pending=True,
            )
        entries.append(entry)
    return entries


def batch_entries(directory: Path) -> tuple[list[dict[str, Any]], bool] | None:
    """A batch's entries: its ``batch_results.json``, or, for a SLURM batch
    not collected yet, what its tasks have recorded so far; and whether they
    are final (the summary). ``None`` when *directory* holds no batch."""
    entries = _read_json(directory / "batch_results.json")
    if isinstance(entries, list):
        return entries, True
    slurm = _slurm_entries(directory)
    return (slurm, False) if slurm is not None else None


_batch_cache: dict[str, tuple[tuple[float, ...], list[CaseFiles | None]]] = {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def batch_case_files(directory: Path) -> list[CaseFiles | None]:
    """The files of each case of the batch in *directory*, in the order of its
    ``batch_results.json`` (``None`` for a case whose files are not found).

    Kept until ``batch_results.json``, the manifest or the directory change.
    """
    summary = directory / "batch_results.json"
    stamp = (
        _mtime(summary),
        _mtime(directory / "slurm" / "manifest.json"),
        _mtime(directory),
    )
    key = str(directory)
    cached = _batch_cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    found = batch_entries(directory)
    entries = found[0] if found is not None else []
    stems = _manifest_stems(directory, len(entries))
    if stems is None:
        stems = match_batch_stems(directory, entries)
    files = [None if s is None else CaseFiles(directory, s) for s in stems]
    _batch_cache[key] = (stamp, files)
    return files


def _only_case(directory: Path) -> CaseFiles | None:
    """The case of a directory that holds one run (a multirun job, a single
    run): the first stem with a result, else the first with a log."""
    stems = _stems_in(directory)
    for stem in stems:
        if (directory / f"{stem}{_RESULT}").is_file():
            return CaseFiles(directory, stem)
    return CaseFiles(directory, stems[0]) if stems else None


def case_files(
    base: Path, session_type: str, date: str, time: str, index: int
) -> CaseFiles | None:
    """The files of case *index* of a session, or ``None``."""
    directory = session_dir(base, session_type, date, time)
    if directory is None or index < 0:
        return None
    if session_type == "multirun":
        job = job_dir(base, date, time, index)
        return _only_case(job) if job is not None else None
    if session_type == "single":
        return _only_case(directory) if index == 0 else None
    files = batch_case_files(directory)
    return files[index] if index < len(files) else None
