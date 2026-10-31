"""Scan multirun/ and outputs/ directories for scenario test results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..registry import get_conf_dirs, load_scenario_plugins
from ..scenario_glob import scenario_names
from . import cases
from .cases import batch_entries, case_files
from .models import ConditionNode, ScenarioResultView, SessionItem, SessionSummary
from .runner import RAW_OUTPUT_LOG

logger = logging.getLogger(__name__)

# Module-level cache: cleared by the Update button.
_cache: dict[str, Any] = {}


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
    break the scan of every other session.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
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


def _passed(value: Any) -> bool | None:
    """A verdict: ``passed`` when it is a boolean, else ``None``."""
    return value if isinstance(value, bool) else None


def _seconds(value: Any) -> float | None:
    """A duration: *value* when it is a number (not a boolean), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _text(value: Any) -> str:
    """A message: *value* as a string, empty for ``None``."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _read_overrides(job_dir: Path) -> list[str]:
    """Read ``.hydra/overrides.yaml`` from a job directory."""
    try:
        raw = (job_dir / ".hydra" / "overrides.yaml").read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        return data if isinstance(data, list) else []
    except (yaml.YAMLError, OSError):
        return []


def _extract_scenario_name(result: dict[str, Any] | None, job_dir: Path) -> str:
    """Extract scenario name from result JSON or, when it names none (a
    missing, ``null`` or empty ``scenario``), its filename."""
    name = result.get("scenario") if result else None
    if isinstance(name, str) and name:
        return name
    # Derive from *_result.json filename
    for f in sorted(job_dir.glob("*_result.json")):
        name = f.stem
        if name.endswith("_result"):
            return name[: -len("_result")]
    return "unknown"


def _scan_multirun_session(
    session_dir: Path, date: str, time: str
) -> SessionSummary | None:
    """Scan a single multirun session directory."""
    # Multirun directories contain numbered subdirectories (0, 1, 2, ...)
    job_dirs = sorted(
        (d for d in session_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not job_dirs:
        return None

    passed_count = 0
    total_count = len(job_dirs)
    scenario_names: set[str] = set()

    for job_dir in job_dirs:
        result_files = sorted(job_dir.glob("*_result.json"))
        if result_files:
            data = _read_result_json(result_files[0])
            if data and _passed(data.get("passed")):
                passed_count += 1
            scenario_names.add(_extract_scenario_name(data, job_dir))
        # Missing result counts as failed (no increment)

    scenario_label = ", ".join(sorted(scenario_names)) if scenario_names else "unknown"

    return SessionSummary(
        date=date,
        time=time,
        session_type="multirun",
        scenario_name=scenario_label,
        passed_count=passed_count,
        total_count=total_count,
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
        scenario_names = {_name(item.get("scenario"), "unknown") for item in entries}
        return SessionSummary(
            date=date,
            time=time,
            session_type="batch",
            scenario_name=", ".join(sorted(scenario_names)),
            passed_count=passed_count,
            total_count=len(entries),
        )

    # Single run: look for *_result.json
    result_files = sorted(session_dir.glob("*_result.json"))
    if not result_files:
        return None  # Skip directories with no result

    data = _read_result_json(result_files[0])
    passed = bool(_passed(data.get("passed"))) if data else False
    scenario_name = _extract_scenario_name(data, session_dir)
    return SessionSummary(
        date=date,
        time=time,
        session_type="single",
        scenario_name=scenario_name,
        passed_count=1 if passed else 0,
        total_count=1,
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
        for date_dir in sorted(multirun_root.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            for time_dir in sorted(date_dir.iterdir(), reverse=True):
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
        for date_dir in sorted(outputs_root.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            for time_dir in sorted(date_dir.iterdir(), reverse=True):
                # "single" and "batch" share outputs/: either type checks it.
                if not time_dir.is_dir() or (
                    cases.session_dir(base_path, "single", date_dir.name, time_dir.name)
                    is None
                ):
                    continue
                summary = _scan_outputs_session(time_dir, date_dir.name, time_dir.name)
                if summary:
                    sessions.append(summary)

    # Sort by date+time descending
    sessions.sort(key=lambda s: (s.date, s.time), reverse=True)

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
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    final = True

    if session_type == "multirun":
        if not directory.is_dir():
            return items
        indices = sorted(
            int(d.name) for d in directory.iterdir() if d.is_dir() and d.name.isdigit()
        )
        for index in indices:
            job = cases.job_dir(base_path, date, time, index)
            if job is None:
                continue
            overrides = _read_overrides(job)
            result_files = sorted(job.glob("*_result.json"))
            if result_files:
                data = _read_result_json(result_files[0])
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
                    items.append(
                        SessionItem(
                            index=index,
                            scenario_name="unknown",
                            passed=None,
                            elapsed_seconds=None,
                            overrides=overrides,
                            message="Failed to read result JSON",
                        )
                    )
            else:
                items.append(
                    SessionItem(
                        index=index,
                        scenario_name=_extract_scenario_name(None, job),
                        passed=None,
                        elapsed_seconds=None,
                        overrides=overrides,
                        message="No result file found",
                    )
                )

    elif session_type == "batch":
        found = batch_entries(directory)
        if found is None:
            return items
        data_list, final = found
        overrides = _read_overrides(directory)
        # An entry that is not an object is left out; the others keep their
        # position, which load_scenario and case_files index the batch by.
        for i, entry in _dict_entries(data_list):
            items.append(
                SessionItem(
                    index=i,
                    scenario_name=_name(entry.get("scenario"), "unknown"),
                    passed=_passed(entry.get("passed")),
                    elapsed_seconds=_seconds(entry.get("elapsed_seconds")),
                    overrides=overrides,
                    message=_text(entry.get("message")),
                )
            )

    elif session_type == "single":
        if not directory.is_dir():
            return items
        overrides = _read_overrides(directory)
        result_files = sorted(directory.glob("*_result.json"))
        if result_files:
            data = _read_result_json(result_files[0])
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

    # A SLURM batch still running is read again each time.
    if session_type != "batch" or final:
        _cache[cache_key] = items
    return items


_RAW_LOG_MAX_BYTES = 256 * 1024  # 256 KB cap per log


def _read_raw_log(job_dir: Path) -> str:
    """Read raw terminal log from a job directory.

    Tries ``raw_output.log`` first (subprocess stdout/stderr captured by
    the runner), then falls back to ``run.log`` (Python logging output).
    Returns an empty string when neither file exists.  Large files are
    truncated to the last ``_RAW_LOG_MAX_BYTES`` bytes.
    """
    for name in (RAW_OUTPUT_LOG, "run.log"):
        try:
            log_path = job_dir / name
            size = log_path.stat().st_size
            if size <= _RAW_LOG_MAX_BYTES:
                return log_path.read_text(encoding="utf-8", errors="replace")
            with log_path.open(encoding="utf-8", errors="replace") as f:
                f.seek(size - _RAW_LOG_MAX_BYTES)
                f.readline()  # discard partial first line
                tail = f.read()
            return f"... (truncated, showing last {len(tail)} chars) ...\n{tail}"
        except OSError:
            continue
    return ""


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
    if directory is None or index < 0:
        return None
    cache_key = f"scenario:{base_path}:{session_type}:{date}:{time}:{index}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    data: dict[str, Any] | None = None
    overrides: list[str] = []
    job_path: Path | None = None
    cacheable = True

    if session_type == "multirun":
        job = cases.job_dir(base_path, date, time, index)
        if job is None or not job.is_dir():
            return None
        job_path = job
        overrides = _read_overrides(job_path)
        result_files = sorted(job_path.glob("*_result.json"))
        if result_files:
            data = _read_result_json(result_files[0])

    elif session_type == "batch":
        job_path = directory
        overrides = _read_overrides(job_path)
        found = batch_entries(job_path)
        if found is not None:
            data_list, final = found
            entry = data_list[index] if 0 <= index < len(data_list) else None
            if isinstance(entry, dict) and not entry.get("pending"):
                data = entry
            if not final:
                cacheable = False

    elif session_type == "single":
        job_path = directory
        overrides = _read_overrides(job_path)
        result_files = sorted(job_path.glob("*_result.json"))
        if result_files and index == 0:
            data = _read_result_json(result_files[0])

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
            message="No result data available (scenario may have failed to start)",
            elapsed_seconds=None,
            condition_statuses=[],
            overrides=overrides,
            raw_log=raw_log,
        )
        if cacheable:
            _cache[cache_key] = result
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
    if files is not None and files.trajectory.is_file():
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
        _cache[cache_key] = result
    return result
