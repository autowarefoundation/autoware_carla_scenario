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

The path guard every directory the viewer builds from a request goes
through (``contained``, ``session_dir``, ``job_dir``), ``CaseFiles`` and the
stems of a directory live in ``ui.case_paths``, which the library check
compiles, and are re-exported here; what reads the JSON documents a batch
leaves (of any type) stays here.

Free of heavy imports: the viewer reads these on every page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The path guard and the case stems, compiled by the library check; every
# name stays importable from here.
from .case_paths import _COMPONENT as _COMPONENT
from .case_paths import _mtime as _mtime
from .case_paths import _only_case as _only_case
from .case_paths import _RESULT as _RESULT
from .case_paths import _STEM_NUMBER as _STEM_NUMBER
from .case_paths import _TRAJECTORY as _TRAJECTORY
from .case_paths import SESSION_TYPES as SESSION_TYPES
from .case_paths import CaseFiles as CaseFiles
from .case_paths import _safe as _safe
from .case_paths import _stems_in as _stems_in
from .case_paths import contained as contained
from .case_paths import job_dir as job_dir
from .case_paths import session_dir as session_dir
from .case_paths import stem_order as stem_order

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
