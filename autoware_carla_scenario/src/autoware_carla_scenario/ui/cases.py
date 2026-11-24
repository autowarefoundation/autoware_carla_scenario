"""Which files on disk are a session's cases: the stem each case's files share.

A run names every file it writes after one stem (``scenario_runner``'s
``_run_file_stem``): ``{stem}_result.json``, ``{stem}.trajectory.jsonl``,
``{stem}_coverage.json``, ``{stem}_odd.tar.gz``, ``{stem}_environment.json``.
A multirun job directory and a single run's directory hold one case; a batch
directory holds all of its cases side by side (``CutInScenario``,
``CutInScenario-1``, ...), and ``batch_results.json`` lists their results.
Since #1957 a batch's summary names each case's stem (``stem``) too; this
module does not read it yet, and matches the stems as it does for a summary
written before, which names none.  A SLURM batch's manifest names them
(``slurm/manifest.json``); for a local batch they are matched here: each entry
takes the first stem, in the order the runner hands them out, whose result is
the entry's (the same verdict, elapsed time and message) and whose recorded
case (its coverage file's ``case.overrides``) is among the entry's overrides.
A case that raised before it opened its trajectory log left its stem to the
next one, and one retried took a second stem, so a position alone is not
enough.

The path guard every directory the viewer builds from a request goes
through (``contained``, ``session_dir``, ``job_dir``), ``CaseFiles``, the
stems of a directory and a batch's stamp (``batch_stamp``) live in
``ui.case_paths``, which the library check compiles, and are re-exported here;
what reads the JSON documents a batch leaves (of any type) stays here.

Free of heavy imports: the viewer reads these on every page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Whether a SLURM case's outcome is stale, compiled by the library check.
from ..slurm.collect import _stale as _stale
from ..slurm.collect import stale_reason as stale_reason
from ..slurm.manifest import BatchLayout, verdict_of_document

# The path guard, the case stems, a batch's stamp and the overrides a session's jobs
# share, compiled by the library check; every name stays importable from here.
from .case_paths import _COMPONENT as _COMPONENT
from .case_paths import _COVERAGE as _COVERAGE
from .case_paths import _mtime as _mtime
from .case_paths import _only_case as _only_case
from .case_paths import _RESULT as _RESULT
from .case_paths import _STEM_NUMBER as _STEM_NUMBER
from .case_paths import _TRAJECTORY as _TRAJECTORY
from .case_paths import SESSION_TYPES as SESSION_TYPES
from .case_paths import CaseFiles as CaseFiles
from .case_paths import _safe as _safe
from .case_paths import _shared_overrides as _shared_overrides
from .case_paths import _stat as _stat
from .case_paths import _stems_in as _stems_in
from .case_paths import _summary_is_current as _summary_is_current
from .case_paths import batch_stamp as batch_stamp
from .case_paths import contained as contained
from .case_paths import job_dir as job_dir
from .case_paths import session_dir as session_dir
from .case_paths import stem_order as stem_order

__all__ = [
    "SESSION_TYPES",
    "CaseFiles",
    "batch_case_files",
    "batch_entries",
    "batch_stamp",
    "case_files",
    "contained",
    "job_dir",
    "match_batch_stems",
    "session_dir",
    "session_overrides",
    "stem_order",
]


def _read_json(path: Path | None) -> Any:
    """The JSON document at *path*, ``None`` when there is none (or no path)
    or it cannot be read: ``ui.results`` and ``ui.scanner`` read with it too."""
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


def _hydra_overrides(directory: Path) -> list[str]:
    """The run's ``.hydra/overrides.yaml`` in *directory*, each item as a
    string.

    Empty when the file cannot be read or is no YAML list: one that is not
    UTF-8 (a ``UnicodeDecodeError``, a ``ValueError``), not YAML, or anything
    else that raises reading it must not break the session and case pages
    (#1631).  ``yaml`` is imported here, not at the top: the viewer reads the
    rest of this module on every page.
    """
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415

        data = yaml.safe_load(
            (directory / ".hydra" / "overrides.yaml").read_text(encoding="utf-8")
        )
    except Exception:  # noqa: BLE001 -- yaml.YAMLError, OSError, ValueError, ...
        return []
    return [str(o) for o in data] if isinstance(data, list) else []


def _multirun_task(directory: Path) -> list[str] | None:
    """The command line of the Hydra multirun in *directory*: the
    ``hydra.overrides.task`` its ``multirun.yaml`` records, each item as a
    string (a sweep as given, ``ego.speed=10,20``; the others as each job
    received them).  ``None`` when there is no such file or it records no
    list there -- a session the viewer's Run panel grouped writes none -- and
    for one that cannot be read, as :func:`_hydra_overrides` reads its
    file."""
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load((directory / "multirun.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- yaml.YAMLError, OSError, ValueError, ...
        return None
    hydra = data.get("hydra") if isinstance(data, dict) else None
    overrides = hydra.get("overrides") if isinstance(hydra, dict) else None
    task = overrides.get("task") if isinstance(overrides, dict) else None
    return [str(o) for o in task] if isinstance(task, list) else None


def _as_jobs_receive(overrides: list[str]) -> set[str]:
    """Each of *overrides*, as the command line gave it, and as a job of the
    multirun received it: Hydra's basic sweeper hands each job a fixed
    override re-written from its parsed value (``x=[1, 2]`` as ``x=[1,2]``,
    ``w= 3`` as ``w=3``), where this package's sweeper passes it on as given.
    A sweep (``ego.speed=10,20``, ``range(...)``, ``choice(...)``,
    ``glob(...)``) is only kept as given, which no job's value equals; so is
    an override Hydra's parser cannot read."""
    forms = set(overrides)
    try:
        from hydra.core.override_parser.overrides_parser import (  # noqa: PLC0415
            OverridesParser,
        )

        parser = OverridesParser.create()
    except Exception:  # noqa: BLE001 -- Hydra cannot be imported
        return forms
    for override in overrides:
        try:
            parsed = parser.parse_override(override)
            if not parsed.is_sweep_override():
                value = parsed.get_value_element_as_str()
                forms.add(f"{parsed.get_key_element()}={value}")
        except Exception:  # noqa: BLE001, S112 -- OverrideParseException, ...
            continue
    return forms


def session_overrides(directory: Path, jobs: list[list[str]]) -> set[str]:
    """The overrides of a multirun or single session's command line, which
    every case of it ran with and so are not what tells its cases apart:
    those its ``multirun.yaml`` records (:func:`_multirun_task`, each also
    as its jobs received it, :func:`_as_jobs_receive`), else those every one
    of its jobs' overrides (*jobs*) holds -- for a session of one job (a
    single run), all of them.  A job whose overrides could not be read (an
    empty list: no ``.hydra/overrides.yaml``) is left out of that, rather
    than leave no override shared."""
    recorded = _multirun_task(directory)
    if recorded is not None:
        return _as_jobs_receive(recorded)
    return _shared_overrides(jobs)


def _same_result(entry: dict[str, Any], result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    return (
        _passed(entry.get("passed")) == _passed(result.get("passed"))
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
    no result), and for a pending one (a case the batch has not run, #1831),
    which takes no stem: one its run is still writing belongs to no entry
    yet.
    """
    stems = _stems_in(directory)
    results = {stem: _read_json(directory / f"{stem}{_RESULT}") for stem in stems}
    coverages: dict[str, Any] = {}
    used: set[str] = set()
    out: list[str | None] = []
    for entry in entries:
        chosen: str | None = None
        if (
            isinstance(entry, dict)
            and entry.get("error") is not True
            and entry.get("pending") is not True
        ):
            for stem in stems:
                if stem in used or not _same_result(entry, results.get(stem)):
                    continue
                if stem not in coverages:
                    coverages[stem] = _read_json(directory / f"{stem}{_COVERAGE}")
                if _same_case(entry, coverages[stem]):
                    chosen = stem
                    break
        if chosen is not None:
            used.add(chosen)
        out.append(chosen)
    return out


def _manifest_cases(directory: Path) -> list[dict[str, Any]] | None:
    """A SLURM batch's manifest cases, by index; ``None`` when it has no
    manifest or one that is not what the launcher writes: ``cases`` not a
    list, or a case that is not an object with a whole-number ``index`` of
    its own (0 or more).  One bad file must not break the session list
    (#1726, #1964)."""
    manifest = _read_json(directory / "slurm" / "manifest.json")
    if not isinstance(manifest, dict):
        return None
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return None
    indices: set[int] = set()
    for case in cases:
        if not isinstance(case, dict):
            return None
        index = case.get("index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in indices
        ):
            return None
        indices.add(index)
    return sorted(cases, key=lambda c: int(c["index"]))


def _manifest_stems(directory: Path, count: int) -> list[str | None] | None:
    """The stems a SLURM batch's manifest gives its cases, if it has one of
    *count* cases; ``None`` too when a stem is not a single safe path
    component (``..``, a separator: :func:`_safe`), which would name files
    outside the batch's directory, and the stems are matched instead."""
    cases = _manifest_cases(directory)
    if cases is None or len(cases) != count:
        return None
    stems: list[str | None] = []
    for case in cases:
        stem = case.get("stem")
        if not stem:
            stems.append(None)
        elif isinstance(stem, str) and _safe(stem):
            stems.append(stem)
        else:
            return None
    return stems


def _slurm_entries(directory: Path) -> list[dict[str, Any]] | None:
    """The entries a SLURM batch's summary will have, from its manifest and
    the outcomes its tasks recorded (``slurm/results/<case>.json``), while it
    has no ``batch_results.json`` yet. A case without an outcome yet has
    ``passed`` ``None`` and ``pending`` true; one whose outcome holds no
    verdict, or a result that cannot be reported (``verdict_of_document``,
    #1718, #1726), did not pass: ``error`` true, as ``collect`` reports it.
    A stale outcome, an earlier submission's that the last one to give the
    case stopped before replacing (``slurm.collect.stale_reason``, #2247),
    is not the case's: it is pending too, with a message saying why, as
    ``status`` and the SLURM panel count it, never its old verdict (#2285)."""
    cases = _manifest_cases(directory)
    if cases is None:
        return None
    layout = BatchLayout(directory)
    entries: list[dict[str, Any]] = []
    for case in cases:
        overrides = case.get("case_overrides")
        record = _read_json(
            directory / "slurm" / "results" / f"{int(case['index'])}.json"
        )
        entry: dict[str, Any] = {
            "scenario": case.get("label") or case.get("scenario") or "unknown",
            "overrides": list(overrides) if isinstance(overrides, list) else [],
        }
        stale = _stale(layout, int(case["index"])) if isinstance(record, dict) else None
        if stale is not None:
            entry.update(
                passed=None,
                message=f"Stale result: {stale}",
                elapsed_seconds=None,
                pending=True,
            )
        elif isinstance(record, dict) and isinstance(record.get("result"), dict):
            entry.update(record["result"])
            if record.get("errored"):
                entry["error"] = True
            if verdict_of_document(record) is None:
                path = directory / "slurm" / "results" / f"{int(case['index'])}.json"
                entry.update(
                    passed=False,
                    error=True,
                    elapsed_seconds=None,
                    message=f"No verdict: {path} holds no result that can be reported",
                )
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
    not collected yet or submitted again since it was collected, what its
    tasks have recorded so far; and whether they are final (the summary).
    ``None`` when *directory* holds no batch."""
    entries = _read_json(directory / "batch_results.json")
    if isinstance(entries, list) and _summary_is_current(directory):
        return entries, True
    slurm = _slurm_entries(directory)
    if slurm is not None:
        return slurm, False
    return (entries, True) if isinstance(entries, list) else None


_batch_cache: dict[str, tuple[tuple[tuple[int, int], ...], list[CaseFiles | None]]] = {}


def batch_case_files(directory: Path) -> list[CaseFiles | None]:
    """The files of each case of the batch in *directory*, in the order of its
    ``batch_results.json`` (``None`` for a case whose files are not found).

    Kept until the batch's :func:`batch_stamp` changes.
    """
    stamp = batch_stamp(directory)
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
