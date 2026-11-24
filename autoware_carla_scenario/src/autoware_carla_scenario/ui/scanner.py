"""Scan multirun/ and outputs/ directories for scenario test results."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..registry import get_conf_dirs, load_scenario_plugins
from ..run_outputs import RESULT_SUFFIX
from ..scenario_glob import scenario_names
from . import cases
from .case_paths import (
    MAX_INDEX,
    _stat,
    job_index,
    own_file,
    result_path,
    scenario_overrides,
    session_order,
)
from .cases import _hydra_overrides as _read_overrides
from .cases import _passed as _passed
from .cases import _read_json
from .cases import _seconds as _seconds
from .cases import batch_entries, case_files
from .models import ConditionNode, ScenarioResultView, SessionItem, SessionSummary
from .runner import _read_raw_log as _read_raw_log

logger = logging.getLogger(__name__)

# Module-level cache: cleared by the Update button.  Touched on the event
# loop only, never from a worker thread.
_cache: dict[str, Any] = {}


def _entries_stamp(directory: Path) -> tuple[Any, ...]:
    """The name, modification time (ns) and size of every entry of
    *directory*, and its own: what a directory that holds one run (a
    multirun job, a single run) is read from.  One ``scandir`` and a ``stat``
    per entry, of which such a directory has a handful; a file created,
    removed, renamed over or written to changes it."""
    try:
        with os.scandir(directory) as found:
            entries = sorted(found, key=lambda e: e.name)
            return (
                _stat(directory),
                tuple((e.name, *_stat_entry(e)) for e in entries),
            )
    except OSError:
        return (_stat(directory),)


def _stat_entry(entry: os.DirEntry[str]) -> tuple[int, int]:
    try:
        info = entry.stat()
    except OSError:
        return (0, -1)
    return (info.st_mtime_ns, info.st_size)


def _jobs_stamp(directory: Path) -> tuple[Any, ...]:
    """The modification time (ns) of a multirun session's directory and of
    each of its job directories: a job added changes the first, a file
    created, removed or renamed in a job (its result, its log) the job's.
    One ``scandir`` and a ``stat`` per job, not per file: a result still
    being written in place changes no directory's time, which is why
    :func:`load_session` keeps a session only once every job has a result
    it could read."""
    try:
        with os.scandir(directory) as found:
            jobs = sorted(
                (e for e in found if job_index(e.name) is not None),
                key=lambda e: e.name,
            )
            return (
                _stat(directory),
                tuple((e.name, _stat_entry(e)[0]) for e in jobs),
            )
    except OSError:
        return (_stat(directory),)


def _stamp(directory: Path, session_type: str) -> tuple[Any, ...]:
    """What a cached session page of the session in *directory* is kept for:
    a batch's :func:`~autoware_carla_scenario.ui.cases.batch_stamp` (a SLURM
    batch resubmitted and collected again changes it), a multirun session's
    :func:`_jobs_stamp`, a single run's :func:`_entries_stamp`.  A run still
    in progress writes its files after the page was first read (#1964)."""
    if session_type == "batch":
        return cases.batch_stamp(directory)
    if session_type == "multirun":
        return _jobs_stamp(directory)
    return _entries_stamp(directory)


def _cached(cache_key: str, stamp: tuple[Any, ...]) -> tuple[bool, Any]:
    """``(True, value)`` when *cache_key* holds a value cached at *stamp*."""
    entry = _cache.get(cache_key)
    if entry is not None and entry[0] == stamp:
        return True, entry[1]
    return False, None


# The built-in Hydra configs (``examples/conf``), resolved here rather than
# taken from ``examples.run``, which registers it on import but pulls in CARLA.
BUILTIN_CONF_DIR = Path(__file__).resolve().parent.parent / "examples" / "conf"


def scenario_conf_dirs() -> list[Path]:
    """Every Hydra config directory the viewer's scenarios come from.

    The built-in ``examples/conf`` first -- the primary config directory the
    viewer composes with, as the ``scenario`` CLI does -- then each directory
    an external scenario package registered (``register_conf_dir``), in
    registration order and without duplicates.  The packages are loaded
    first (:func:`~autoware_carla_scenario.registry.load_scenario_plugins`,
    which loads them once per process).
    """
    load_scenario_plugins()
    dirs = [BUILTIN_CONF_DIR]
    for conf_dir in get_conf_dirs():
        if conf_dir not in dirs:
            dirs.append(conf_dir)
    return dirs


def list_scenario_configs() -> list[str]:
    """Return sorted list of available scenario config names.

    Scans ``scenario/`` under every config directory
    (:func:`scenario_conf_dirs`: the built-in examples and those of the
    installed scenario packages) for YAML files and returns relative paths
    without the ``.yaml`` suffix (e.g. ``"intersection_passing/left_turn"``),
    each name once.  This is the viewer's scenario list, ordered by path
    component; patterns are not matched against it but resolved as the
    ``scenario`` CLI resolves them
    (:func:`~autoware_carla_scenario.scenario_glob.resolve_scenario_glob`).
    The scan is :func:`~autoware_carla_scenario.scenario_glob.scenario_names`.
    """
    return scenario_names(scenario_conf_dirs())


def clear_cache() -> None:
    """Clear all cached scan results."""
    _cache.clear()


def _read_result_json(path: Path) -> dict[str, Any] | None:
    """Read and parse a ``*_result.json`` file, returning ``None`` on error.

    ``None`` too when the file is not a JSON object: a result cut off
    mid-character (not UTF-8, a ``ValueError``) or of another type must not
    break the scan of every other session. Read as ``ui.cases`` reads every
    JSON document (:func:`~.cases._read_json`): a document that is JSON
    ``null`` is one that cannot be read here.
    """
    data = _read_json(path)
    if data is None:
        logger.warning("Failed to read result JSON: %s", path)
        return None
    if not isinstance(data, dict):
        logger.warning("Result JSON is not an object: %s", path)
        return None
    return data


def _dict_entries(entries: list[Any]) -> list[tuple[int, dict[str, Any]]]:
    """A batch's entries that are JSON objects, each with its position in the
    batch (which its case's files are found by), the others left out."""
    return [(i, e) for i, e in enumerate(entries) if isinstance(e, dict)]


def _name(value: Any, fallback: str) -> str:
    """*value* when it is a non-empty string, else *fallback*."""
    return value if isinstance(value, str) and value else fallback


def _text(value: Any) -> str:
    """A message: *value* as a string, empty for ``None``."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _entry_overrides(entry: dict[str, Any]) -> list[str]:
    """The overrides a batch case ran with, as its ``batch_results.json``
    entry (or a SLURM batch's case recorded so far) gives them: its sweep
    case, each item as a string (as :mod:`.results` reads them).

    A batch directory has no ``.hydra/overrides.yaml``: each case carries its
    own (#2344).  An entry names no ``scenario=`` override, which the scenario
    page's Re-run needs, so one is added from the entry's ``scenario``: the
    batch's label for the case, less the ``#k`` it is numbered with when its
    sweep gave the scenario more than one case
    (``examples.argv._case_labels``).  None is added for an entry that names
    no scenario (the checked :func:`.case_paths.scenario_overrides`, #2354).

    They are not all the case ran with: a batch records no command line, so
    the overrides it gave every case are not among them, and the scenario
    page says so beside its Re-run.
    """
    value = entry.get("overrides")
    overrides = [str(o) for o in value] if isinstance(value, list) else []
    label = entry.get("scenario")
    if not isinstance(label, str):
        return overrides
    return scenario_overrides(overrides, label)


def _listing(directory: Path) -> list[Path] | None:
    """The entries of *directory*, ``None`` (with a warning) when it cannot be
    read: a session directory of another user's, mode 0700, must not break
    the session list or the session pages (#2183), as one bad file does not
    (#1726)."""
    try:
        return list(directory.iterdir())
    except OSError as exc:
        logger.warning("Cannot list %s: %s", directory, exc)
        return None


def _extract_scenario_name(result: dict[str, Any] | None, job_dir: Path) -> str:
    """Extract scenario name from result JSON or, when it names none (a
    missing, ``null`` or empty ``scenario``), its filename."""
    name = result.get("scenario") if result else None
    if isinstance(name, str) and name:
        return name
    # Derive from the *_result.json filename
    path = result_path(job_dir)
    if path is not None:
        return path.name[: -len(RESULT_SUFFIX)]
    return "unknown"


def _scan_multirun_session(
    session_dir: Path, date: str, time: str
) -> SessionSummary | None:
    """Scan a single multirun session directory."""
    # Multirun directories contain numbered subdirectories (0, 1, 2, ...)
    entries = _listing(session_dir)
    if entries is None:
        return None
    job_dirs = sorted(
        (d for d in entries if job_index(d.name) is not None and d.is_dir()),
        key=lambda d: int(d.name),
    )
    if not job_dirs:
        return None

    passed_count = 0
    pending_count = 0
    total_count = len(job_dirs)
    scenario_names: set[str] = set()

    for job_dir in job_dirs:
        result_file = result_path(job_dir)
        data = _read_result_json(result_file) if result_file is not None else None
        if result_file is not None:
            scenario_names.add(_extract_scenario_name(data, job_dir))
        if not data:
            # A missing or unreadable result: without a result, as the
            # session page counts it (#2285).
            pending_count += 1
        elif _passed(data.get("passed")):
            passed_count += 1

    scenario_label = ", ".join(sorted(scenario_names)) if scenario_names else "unknown"

    return SessionSummary(
        date=date,
        time=time,
        session_type="multirun",
        scenario_name=scenario_label,
        passed_count=passed_count,
        total_count=total_count,
        pending_count=pending_count,
    )


def _scan_outputs_session(
    session_dir: Path, date: str, time: str
) -> SessionSummary | None:
    """Scan a single outputs session directory (batch or single)."""
    # A batch: its summary, or a SLURM batch's cases recorded so far.
    found = batch_entries(session_dir)
    if found is not None:
        entries = [entry for _, entry in _dict_entries(found[0])]
        if not entries:
            return None
        passed_count = sum(1 for item in entries if _passed(item.get("passed")))
        pending_count = sum(1 for item in entries if item.get("pending") is True)
        scenario_names = {_name(item.get("scenario"), "unknown") for item in entries}
        return SessionSummary(
            date=date,
            time=time,
            session_type="batch",
            scenario_name=", ".join(sorted(scenario_names)),
            passed_count=passed_count,
            total_count=len(entries),
            pending_count=pending_count,
        )

    # Single run: look for *_result.json
    result_file = result_path(session_dir)
    if result_file is None:
        return None  # Skip directories with no result

    data = _read_result_json(result_file)
    passed = bool(_passed(data.get("passed"))) if data else False
    scenario_name = _extract_scenario_name(data, session_dir)
    return SessionSummary(
        date=date,
        time=time,
        session_type="single",
        scenario_name=scenario_name,
        passed_count=1 if passed else 0,
        total_count=1,
        # An unreadable result: without a result, as the session page
        # counts it (#2285).
        pending_count=0 if data else 1,
    )


def scan_sessions(base_path: Path) -> list[SessionSummary]:
    """Scan ``multirun/`` and ``outputs/`` under *base_path* for sessions.

    Returns a list of :class:`SessionSummary` sorted by date/time descending.
    """
    cache_key = f"sessions:{base_path}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    sessions: list[SessionSummary] = []

    # Scan multirun/
    multirun_root = base_path / "multirun"
    if multirun_root.is_dir():
        for date_dir in sorted(_listing(multirun_root) or [], reverse=True):
            if not date_dir.is_dir():
                continue
            for time_dir in sorted(_listing(date_dir) or [], reverse=True):
                if not time_dir.is_dir() or (
                    cases.session_dir(
                        base_path, "multirun", date_dir.name, time_dir.name
                    )
                    is None
                ):
                    continue
                summary = _scan_multirun_session(time_dir, date_dir.name, time_dir.name)
                if summary:
                    sessions.append(summary)

    # Scan outputs/
    outputs_root = base_path / "outputs"
    if outputs_root.is_dir():
        for date_dir in sorted(_listing(outputs_root) or [], reverse=True):
            if not date_dir.is_dir():
                continue
            for time_dir in sorted(_listing(date_dir) or [], reverse=True):
                # "single" and "batch" share outputs/: either type checks it.
                if not time_dir.is_dir() or (
                    cases.session_dir(base_path, "single", date_dir.name, time_dir.name)
                    is None
                ):
                    continue
                summary = _scan_outputs_session(time_dir, date_dir.name, time_dir.name)
                if summary:
                    sessions.append(summary)

    # Sort by date+time descending; ``-10`` after ``-9`` (#1845).
    sessions.sort(key=lambda s: (s.date, *session_order(s.time)), reverse=True)

    _cache[cache_key] = sessions
    return sessions


def load_session(
    base_path: Path, session_type: str, date: str, time: str
) -> list[SessionItem]:
    """Load all scenario items within a session.

    Returns a list of :class:`SessionItem` ordered by index, empty when the
    names do not make a session under *base_path*
    (:func:`~autoware_carla_scenario.ui.cases.session_dir`).
    """
    items: list[SessionItem] = []
    directory = cases.session_dir(base_path, session_type, date, time)
    if directory is None:
        return items

    cache_key = f"session:{base_path}:{session_type}:{date}:{time}"
    stamp = _stamp(directory, session_type)
    hit, value = _cached(cache_key, stamp)
    if hit:
        return value  # type: ignore[no-any-return]

    final = True

    if session_type == "multirun":
        if not directory.is_dir():
            return items
        entries = _listing(directory)
        if entries is None:
            # Not cached: it is read again once it can be.
            return items
        indices = sorted(
            int(d.name) for d in entries if job_index(d.name) is not None and d.is_dir()
        )
        for index in indices:
            job = cases.job_dir(base_path, date, time, index)
            if job is None:
                continue
            overrides = _read_overrides(job)
            result_file = result_path(job)
            if result_file is not None:
                data = _read_result_json(result_file)
                if data:
                    items.append(
                        SessionItem(
                            index=index,
                            scenario_name=_extract_scenario_name(data, job),
                            passed=_passed(data.get("passed")),
                            elapsed_seconds=_seconds(data.get("elapsed_seconds")),
                            overrides=overrides,
                            message=_text(data.get("message")),
                        )
                    )
                else:
                    final = False
                    items.append(
                        SessionItem(
                            index=index,
                            scenario_name="unknown",
                            passed=None,
                            elapsed_seconds=None,
                            overrides=overrides,
                            message="Failed to read result JSON",
                            pending=True,
                        )
                    )
            else:
                final = False
                items.append(
                    SessionItem(
                        index=index,
                        scenario_name=_extract_scenario_name(None, job),
                        passed=None,
                        elapsed_seconds=None,
                        overrides=overrides,
                        message="No result file found",
                        pending=True,
                    )
                )

    elif session_type == "batch":
        found = batch_entries(directory)
        if found is None:
            return items
        data_list, final = found
        # An entry that is not an object is left out; the others keep their
        # position, which load_scenario and case_files index the batch by.
        for i, entry in _dict_entries(data_list):
            items.append(
                SessionItem(
                    index=i,
                    scenario_name=_name(entry.get("scenario"), "unknown"),
                    passed=_passed(entry.get("passed")),
                    elapsed_seconds=_seconds(entry.get("elapsed_seconds")),
                    overrides=_entry_overrides(entry),
                    message=_text(entry.get("message")),
                    pending=entry.get("pending") is True,
                )
            )

    elif session_type == "single":
        if not directory.is_dir():
            return items
        overrides = _read_overrides(directory)
        result_file = result_path(directory)
        if result_file is not None:
            data = _read_result_json(result_file)
            if data:
                items.append(
                    SessionItem(
                        index=0,
                        scenario_name=_extract_scenario_name(data, directory),
                        passed=_passed(data.get("passed")),
                        elapsed_seconds=_seconds(data.get("elapsed_seconds")),
                        overrides=overrides,
                        message=_text(data.get("message")),
                    )
                )
            else:
                # Listed, as a multirun job's is, so that the page counts it
                # as the session list does, never "All passed" (#2285);
                # read again until it can be.
                final = False
                items.append(
                    SessionItem(
                        index=0,
                        scenario_name=_extract_scenario_name(None, directory),
                        passed=None,
                        elapsed_seconds=None,
                        overrides=overrides,
                        message="Failed to read result JSON",
                        pending=True,
                    )
                )

    # A SLURM batch still running is read again each time; a final one is
    # kept until its files change (a resubmission, a new collect).  So is a
    # multirun session with a job that has no result it could read yet (a
    # result being written in place changes no directory's time, #1964).
    if final:
        _cache[cache_key] = (stamp, items)
    return items


def _build_condition_tree(details: dict[str, Any]) -> list[ConditionNode]:
    """Recursively extract child condition nodes from a details dict."""
    children: list[ConditionNode] = []

    # Composite conditions (AndCondition, OrCondition) have "children"
    if "children" in details and isinstance(details["children"], list):
        for child_detail in details["children"]:
            node = _detail_to_condition_node(child_detail)
            if node:
                children.append(node)

    # Wrapper conditions (StickyCondition, PersistentCondition, NotCondition) have "child"
    if "child" in details and isinstance(details["child"], dict):
        node = _detail_to_condition_node(details["child"])
        if node:
            children.append(node)

    return children


_STRUCTURAL_KEYS = {
    "label",
    "condition_type",
    "satisfied",
    "message",
    "children",
    "child",
    "operator",
    "wrapper",
}


def _detail_to_condition_node(detail: dict[str, Any]) -> ConditionNode | None:
    """Convert a detail dict (from nested condition tree) to a ConditionNode."""
    if not isinstance(detail, dict):
        return None

    label = _text(detail.get("label"))
    condition_type = _text(detail.get("condition_type"))
    satisfied = bool(detail.get("satisfied", False))
    message = _text(detail.get("message"))

    # Build children recursively
    children = _build_condition_tree(detail)

    return ConditionNode(
        label=label,
        satisfied=satisfied,
        message=message,
        condition_type=condition_type,
        role="",
        details={k: v for k, v in detail.items() if k not in _STRUCTURAL_KEYS},
        children=children,
    )


def load_scenario(
    base_path: Path, session_type: str, date: str, time: str, index: int
) -> ScenarioResultView | None:
    """Load the full result detail for a single scenario, or ``None`` (also
    when the names do not make a session under *base_path*,
    :func:`~autoware_carla_scenario.ui.cases.session_dir`)."""
    directory = cases.session_dir(base_path, session_type, date, time)
    # No case has an index above MAX_INDEX, which a cache key can spell (#1964).
    if directory is None or not 0 <= index <= MAX_INDEX:
        return None
    cache_key = f"scenario:{base_path}:{session_type}:{date}:{time}:{index}"
    if session_type == "multirun":
        # The case's own directory: the job's files, not its siblings'.
        job = cases.job_dir(base_path, date, time, index)
        stamp = _entries_stamp(job) if job is not None else ()
    else:
        stamp = _stamp(directory, session_type)
    hit, value = _cached(cache_key, stamp)
    if hit:
        return value  # type: ignore[no-any-return]

    data: dict[str, Any] | None = None
    overrides: list[str] = []
    job_path: Path | None = None
    cacheable = True
    # What a case without a result says (a batch's pending case says why).
    no_result = "No result data available (scenario may have failed to start)"

    if session_type == "multirun":
        job = cases.job_dir(base_path, date, time, index)
        if job is None or not job.is_dir():
            return None
        job_path = job
        overrides = _read_overrides(job_path)
        result_file = result_path(job_path)
        if result_file is not None:
            data = _read_result_json(result_file)

    elif session_type == "batch":
        job_path = directory
        found = batch_entries(job_path)
        if found is not None:
            data_list, final = found
            entry = data_list[index] if 0 <= index < len(data_list) else None
            if isinstance(entry, dict):
                # The case's own overrides: a batch directory has no
                # .hydra/overrides.yaml (#2344).
                overrides = _entry_overrides(entry)
                if entry.get("pending") is not True:
                    data = entry
                elif isinstance(entry.get("message"), str) and entry["message"]:
                    no_result = entry["message"]
            if not final:
                cacheable = False

    elif session_type == "single":
        job_path = directory
        overrides = _read_overrides(job_path)
        result_file = result_path(job_path)
        if result_file is not None and index == 0:
            data = _read_result_json(result_file)

    if data is None:
        # No result JSON (e.g. spawn failed), but the job directory exists.
        # Return a partial view with only the log and overrides so the viewer
        # can still display useful information instead of returning a 500 error.
        if job_path is None:
            return None
        raw_log = _read_raw_log(job_path)
        if not raw_log and not overrides:
            return None
        result = ScenarioResultView(
            passed=None,
            message=no_result,
            elapsed_seconds=None,
            condition_statuses=[],
            overrides=overrides,
            raw_log=raw_log,
        )
        if cacheable:
            _cache[cache_key] = (stamp, result)
        return result

    # Build condition tree
    condition_statuses: list[ConditionNode] = []
    statuses = data.get("condition_statuses")
    for cs in statuses if isinstance(statuses, list) else []:
        if not isinstance(cs, dict):
            continue
        details = cs.get("details")
        if not isinstance(details, dict):
            details = {}
        children = _build_condition_tree(details)
        condition_statuses.append(
            ConditionNode(
                label=_text(cs.get("label")),
                satisfied=bool(cs.get("satisfied", False)),
                message=_text(cs.get("message")),
                condition_type=_text(cs.get("condition_type")),
                role=_text(cs.get("role")),
                details={k: v for k, v in details.items() if k not in _STRUCTURAL_KEYS},
                children=children,
            )
        )

    raw_log = _read_raw_log(job_path) if job_path else ""

    # The run's trajectory log, which a Rerun recording of it is made from: a
    # batch's cases share a directory, each under a stem of its own.
    trajectory_path: str | None = None
    files = case_files(base_path, session_type, date, time, index)
    if files is not None and own_file(files.trajectory):
        trajectory_path = str(files.trajectory.resolve())

    result = ScenarioResultView(
        passed=_passed(data.get("passed")),
        message=_text(data.get("message")),
        elapsed_seconds=_seconds(data.get("elapsed_seconds")),
        condition_statuses=condition_statuses,
        overrides=overrides,
        raw_log=raw_log,
        trajectory_path=trajectory_path,
    )
    if cacheable:
        _cache[cache_key] = (stamp, result)
    return result
