"""What ``scenario-gate`` finds a set of runs left unrecorded, pinned.

Pins, against verbatim copies of them as they were in ``gate.evaluate`` (the
``_old_*`` oracles below), what the helpers that scan a set of runs return or
raise: the batches under a directory (``_batch_dirs``), whether a directory
holds a gate record or a run's result (``_has_gate_record``,
``_has_result``, ``_has_result_since``), the gate record at a summary entry's
stem (``_recorded_at_stem``), a local batch's unrecorded cases
(``_summary_unrecorded``), a sweep's given-up jobs (``_errored_jobs``) and
its jobs that never started (``_unstarted_jobs``), and what
``unrecorded_cases()`` and ``batch_read_errors()`` make of them.

Each runs, old and new, on the same directory: random sets of batches,
markers and plans, and summary entries whose flags are ``true``, ``1``,
``1.0``, ``"true"`` or ``false`` (only ``true`` is ``is True``), whose
names are false values of every JSON type, and whose stems are any JSON
value or a path that is no file name of its own.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import pytest

from autoware_carla_scenario.gate import evaluate
from autoware_carla_scenario.gate.errored import (
    CASE_ERRORED,
    find_case_errored,
    read_case_errored,
    write_case_errored,
    write_case_unfinished,
)
from autoware_carla_scenario.gate.record import GATE_RECORD_SUFFIX
from autoware_carla_scenario.gate.sweep_plan import (
    SWEEP_PLAN,
    find_sweep_plans,
    read_sweep_plan,
    write_sweep_plan,
)

BATCH_RESULTS = "batch_results.json"
_RESULT_SUFFIX = "_result.json"
_NAMED = 5


# ---------------------------------------------------------------------------
# The oracles: gate.evaluate's helpers as they were, verbatim but for the
# names of the helpers they call.
# ---------------------------------------------------------------------------


def _old_batch_dirs(path: Path) -> list[Path]:
    """The batches under *path*: each directory holding a summary
    (``batch_results.json``) or a SLURM batch's manifest, sorted."""
    found = {p.parent for p in path.rglob(BATCH_RESULTS)}
    found.update(p.parent.parent for p in path.rglob("slurm/manifest.json"))
    return sorted(found)


def _old_recorded_at_stem(directory: Path, entry: dict[str, Any]) -> bool:
    """Whether a gate record is at the stem *entry* names in *directory*: a
    file stem of its own (no separator), as a batch writes it since #1957."""
    stem = entry.get("stem")
    if not isinstance(stem, str) or stem in ("", ".", "..") or "/" in stem:
        return False
    if "\\" in stem or "\0" in stem:
        return False
    return (directory / f"{stem}{GATE_RECORD_SUFFIX}").is_file()


def _old_summary_unrecorded(directory: Path) -> tuple[list[str], int] | str:
    path = directory / BATCH_RESULTS
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, ValueError) as exc:
        return f"{path}: the batch summary cannot be read ({exc})"
    if not isinstance(entries, list):
        return f"{path}: not a batch summary (not a JSON array)"
    unrecorded = [
        str(entry.get("scenario") or i)
        for i, entry in enumerate(entries)
        if not isinstance(entry, dict)
        or (entry.get("error") is True and not _old_recorded_at_stem(directory, entry))
        or entry.get("pending") is True
    ]
    return unrecorded, len(entries)


def _old_has_gate_record(directory: Path) -> bool:
    """Whether *directory* holds a gate record."""
    try:
        return any(
            p.name.endswith(GATE_RECORD_SUFFIX) and p.is_file()
            for p in directory.iterdir()
        )
    except OSError:
        return False


def _old_has_result(directory: Path) -> bool:
    """Whether *directory* holds a run's result (``*_result.json``)."""
    try:
        return any(
            p.name.endswith(_RESULT_SUFFIX) and p.is_file() for p in directory.iterdir()
        )
    except OSError:
        return False


def _old_has_result_since(marker: Path) -> bool:
    try:
        since = marker.stat().st_mtime
        return any(
            p.name.endswith(_RESULT_SUFFIX)
            and p.is_file()
            and p.stat().st_mtime >= since
            for p in marker.parent.iterdir()
        )
    except OSError:
        return False


def _old_unstarted_jobs(path: Path) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    unreadable: list[str] = []
    for plan in find_sweep_plans(path):
        try:
            jobs = read_sweep_plan(plan)
        except (OSError, ValueError) as exc:
            unreadable.append(f"{plan}: the sweep plan cannot be read ({exc})")
            continue
        names: list[str] = []
        for job, _overrides in jobs:
            if job.is_dir() and (
                (job / CASE_ERRORED).is_file()
                or _old_has_gate_record(job)
                or _old_has_result(job)
            ):
                continue
            try:
                names.append(job.relative_to(plan.parent).as_posix())
            except ValueError:
                names.append(str(job))
        if not names:
            continue
        shown = ", ".join(names[:_NAMED])
        more = f", and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
        lines.append(
            f"{plan.parent}: {len(names)} of {len(jobs)} planned job(s) of the "
            f"sweep never ran, and left no gate record ({shown}{more}); the "
            "gate would be judged over the runs that survived"
        )
    return lines, unreadable


def _old_errored_jobs(path: Path) -> list[str]:
    by_sweep: dict[Path, list[str]] = {}
    for marker in find_case_errored(path):
        job = marker.parent
        if _old_has_gate_record(job):
            # The run wrote its record before the job failed: it is judged.
            continue
        data = read_case_errored(marker)
        if (
            data is not None
            and data.get("unfinished") is True
            and _old_has_result_since(marker)
        ):
            # The run wrote its result and was killed after it: it ran a
            # scenario without a gate, no concern of the gate.
            continue
        reason = data.get("reason") if data is not None else None
        name = (
            f"{job.name} ({reason})" if isinstance(reason, str) and reason else job.name
        )
        by_sweep.setdefault(job.parent, []).append(name)
    lines: list[str] = []
    for sweep, names in sorted(by_sweep.items()):
        shown = ", ".join(names[:_NAMED])
        more = f", and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
        lines.append(
            f"{sweep}: {len(names)} job(s) of the sweep raised, timed out, "
            f"crashed or did not finish, and left no gate record ({shown}{more}); the gate would "
            "be judged over the runs that survived"
        )
    return lines


def _old_scan_batches(path: Path) -> tuple[list[str], list[str]]:
    if not path.is_dir():
        return [], []
    lines: list[str] = []
    unreadable: list[str] = []
    for directory in _old_batch_dirs(path):
        if (directory / "slurm" / "manifest.json").is_file():
            found = evaluate._slurm_unrecorded(directory)
        else:
            found = _old_summary_unrecorded(directory)
        if isinstance(found, str):
            unreadable.append(found)
            continue
        names, total = found
        if not names:
            continue
        shown = ", ".join(names[:_NAMED])
        more = f", and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
        lines.append(
            f"{directory}: {len(names)} of {total} case(s) raised or have no "
            f"outcome, and left no gate record ({shown}{more}); the gate would "
            "be judged over the runs that survived"
        )
    lines.extend(_old_errored_jobs(path))
    unstarted, unreadable_plans = _old_unstarted_jobs(path)
    lines.extend(unstarted)
    unreadable.extend(unreadable_plans)
    return lines, unreadable


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


def _outcome(function: Callable[..., Any], *args: Any) -> tuple[Any, ...]:
    try:
        value = function(*args)
    except Exception as exc:  # noqa: BLE001 - the oracle's exceptions are compared
        return ("raised", type(exc), exc.args)
    return ("returned", type(value), value)


def _same(old: Callable[..., Any], new: Callable[..., Any], *args: Any) -> None:
    assert _outcome(new, *args) == _outcome(old, *args), args


def test_the_constants_are_the_oracles() -> None:
    assert evaluate.BATCH_RESULTS == BATCH_RESULTS
    assert evaluate._RESULT_SUFFIX == _RESULT_SUFFIX
    assert evaluate._NAMED == _NAMED


# Values a summary entry's flags, names and stems take: every JSON type, the
# ones Python's `==` takes for True or a str, and false ones.
_FLAGS: list[Any] = [True, False, 1, 0, 1.0, "true", "True", None, [], {}, [True]]
_NAMES: list[Any] = [
    "case",
    "",
    None,
    0,
    0.0,
    False,
    True,
    [],
    {},
    [0],
    {"a": 1},
    7,
    -1.5,
    "ünïcode",
    "a\nb",
]
_STEMS: list[Any] = [
    "run",
    "run2",
    "",
    ".",
    "..",
    "a/b",
    "/abs",
    "a\\b",
    "a\0b",
    "trailing/",
    None,
    1,
    True,
    ["run"],
    {"stem": "run"},
    "missing",
    "ünï",
]


def _random_entry(rng: random.Random) -> Any:
    if rng.random() < 0.02:
        return rng.choice([None, 1, "x", [], ["error"], True])
    entry: dict[str, Any] = {}
    for key, values in (
        ("scenario", _NAMES),
        ("error", _FLAGS),
        ("pending", _FLAGS),
        ("stem", _STEMS),
    ):
        if rng.random() < 0.75:
            entry[key] = rng.choice(values)
    return entry


def _write_summary(rng: random.Random, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    roll = rng.random()
    path = directory / BATCH_RESULTS
    if roll < 0.05:
        path.write_bytes(b"\xff\xfe not utf-8")
    elif roll < 0.1:
        path.write_text("{not json", encoding="utf-8")
    elif roll < 0.13:
        path.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    elif roll < 0.18:
        path.write_text(json.dumps(rng.choice([{}, 1, "x", None])), encoding="utf-8")
    elif roll < 0.2:
        path.mkdir()
    else:
        count = rng.choice([0, 1, 3, 5, 6, 7, 12])
        entries = [_random_entry(rng) for _ in range(count)]
        path.write_text(json.dumps(entries), encoding="utf-8")
    for stem in ("run", "run2", "ünï", "a\\b"):
        if rng.random() < 0.5:
            record = directory / f"{stem}{GATE_RECORD_SUFFIX}"
            if rng.random() < 0.15:
                record.mkdir(exist_ok=True)
            else:
                record.write_text("{}", encoding="utf-8")


def _write_job(rng: random.Random, job: Path) -> None:
    job.mkdir(parents=True, exist_ok=True)
    roll = rng.random()
    marker = job / CASE_ERRORED
    if roll < 0.35:
        write_case_unfinished(job, overrides=["a=1"], attempt=1)
    elif roll < 0.6:
        write_case_errored(
            job,
            overrides=[],
            exit_code=rng.choice([None, 1, -9]),
            attempts=2,
            reason=rng.choice(["timed out", "", "crashed (π)"]),
            unfinished=rng.random() < 0.3,
        )
    elif roll < 0.75:
        data: dict[str, Any] = {}
        if rng.random() < 0.7:
            data["unfinished"] = rng.choice(_FLAGS)
        if rng.random() < 0.7:
            data["reason"] = rng.choice(_NAMES)
        marker.write_text(json.dumps(data), encoding="utf-8")
    elif roll < 0.82:
        marker.write_text(rng.choice(["not json", "[1, 2]", "null", "7"]), "utf-8")
    elif roll < 0.85:
        marker.write_bytes(b"\xff")
    elif roll < 0.88:
        marker.mkdir()
    if rng.random() < 0.25:
        (job / f"s{GATE_RECORD_SUFFIX}").write_text("{}", encoding="utf-8")
    if rng.random() < 0.5:
        result = job / f"s{_RESULT_SUFFIX}"
        if rng.random() < 0.1:
            result.mkdir()
        else:
            result.write_text("{}", encoding="utf-8")
            if marker.exists():
                # Older than the marker, as old as it, or newer.
                since = marker.stat().st_mtime
                at = since + rng.choice([-100.0, 0.0, 100.0])
                os.utime(result, (at, at))


def _random_tree(rng: random.Random, root: Path) -> None:
    root.mkdir(parents=True)
    for b in range(rng.randrange(0, 4)):
        _write_summary(rng, root / f"batch{b}" / rng.choice(["", "deep/er"]))
    for s in range(rng.randrange(0, 3)):
        sweep = root / f"sweep{s}"
        jobs = [sweep / str(j) for j in range(rng.choice([1, 2, 6, 8]))]
        for job in jobs:
            if rng.random() < 0.7:
                _write_job(rng, job)
        planned = rng.random()
        if planned < 0.5:
            outside = [root.parent / f"elsewhere-{root.name}"]
            write_sweep_plan(
                sweep,
                [(job, ["x=1"]) for job in jobs]
                + [(o, []) for o in outside if rng.random() < 0.3],
            )
        elif planned < 0.6:
            sweep.mkdir(parents=True, exist_ok=True)
            (sweep / SWEEP_PLAN).write_text(
                rng.choice(["{}", "[]", '{"jobs": [1]}', "x"]), encoding="utf-8"
            )
    if rng.random() < 0.1:
        (root / "loose").write_text("", encoding="utf-8")


def _compare_tree(root: Path) -> None:
    paths = [root, root / "loose", root / "missing", *sorted(root.rglob("*"))]
    for path in paths:
        _same(_old_batch_dirs, evaluate._batch_dirs, path)
        _same(_old_has_gate_record, evaluate._has_gate_record, path)
        _same(_old_has_result, evaluate._has_result, path)
        _same(_old_summary_unrecorded, evaluate._summary_unrecorded, path)
        _same(_old_errored_jobs, evaluate._errored_jobs, path)
        _same(_old_unstarted_jobs, evaluate._unstarted_jobs, path)
        _same(_old_scan_batches, evaluate._scan_batches, path)
        _same(lambda p: _old_scan_batches(p)[0], evaluate.unrecorded_cases, path)
        _same(lambda p: _old_scan_batches(p)[1], evaluate.batch_read_errors, path)
        if path.name == CASE_ERRORED or path.is_file():
            _same(_old_has_result_since, evaluate._has_result_since, path)


@pytest.mark.parametrize("seed", range(40))
def test_random_runs_are_scanned_as_before(tmp_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    root = tmp_path / "runs"
    _random_tree(rng, root)
    _compare_tree(root)


def test_every_summary_entry_is_read_as_before(tmp_path: Path) -> None:
    # Every flag, name and stem, alone and in pairs, against the records at
    # the stems.
    directory = tmp_path / "batch"
    directory.mkdir()
    for stem in ("run", "ünï", "a\\b", "..", "."):
        if stem not in (".", ".."):
            (directory / f"{stem}{GATE_RECORD_SUFFIX}").write_text("{}", "utf-8")
    entries: list[Any] = [{}, None, 1, "x", [], True]
    for flag in _FLAGS:
        for stem in _STEMS:
            entries.append({"error": flag, "stem": stem, "scenario": "e"})
        entries.append({"pending": flag, "stem": "run"})
        entries.append({"pending": flag, "error": flag})
    for name in _NAMES:
        entries.append({"error": True, "scenario": name})
        entries.append({"pending": True, "scenario": name})
    (directory / BATCH_RESULTS).write_text(json.dumps(entries), encoding="utf-8")
    # An entry that is no object raises AttributeError (its `.get()`), as it
    # did.
    _same(_old_summary_unrecorded, evaluate._summary_unrecorded, directory)
    with pytest.raises(AttributeError):
        evaluate._summary_unrecorded(directory)
    _same(_old_scan_batches, evaluate._scan_batches, tmp_path)
    objects = [e for e in entries if isinstance(e, dict)]
    (directory / BATCH_RESULTS).write_text(json.dumps(objects), encoding="utf-8")
    _same(_old_summary_unrecorded, evaluate._summary_unrecorded, directory)
    found = evaluate._summary_unrecorded(directory)
    assert not isinstance(found, str)
    # The flags count only when `true`, and a false name is the case's index.
    assert (
        "e" in found[0]
        and str(objects.index({"error": True, "scenario": ""})) in found[0]
    )
    for entry in entries:
        if isinstance(entry, dict):
            _same(_old_recorded_at_stem, evaluate._recorded_at_stem, directory, entry)
    _same(_old_scan_batches, evaluate._scan_batches, tmp_path)


def test_a_marker_beside_results_is_read_as_before(tmp_path: Path) -> None:
    # Each marker's unfinished flag and reason, beside a result older than it,
    # as old as it and newer, and with a gate record.
    sweep = tmp_path / "sweep"
    n = 0
    for flag in _FLAGS:
        for reason in _NAMES:
            for offset in (None, -1.0, 0.0, 1.0):
                job = sweep / f"job{n}"
                n += 1
                job.mkdir(parents=True)
                marker = job / CASE_ERRORED
                marker.write_text(
                    json.dumps({"unfinished": flag, "reason": reason}), "utf-8"
                )
                if offset is not None:
                    result = job / f"x{_RESULT_SUFFIX}"
                    result.write_text("{}", encoding="utf-8")
                    at = marker.stat().st_mtime + offset
                    os.utime(result, (at, at))
                _same(_old_has_result_since, evaluate._has_result_since, marker)
    _same(_old_errored_jobs, evaluate._errored_jobs, tmp_path)
    _same(_old_errored_jobs, evaluate._errored_jobs, sweep)
    _same(_old_scan_batches, evaluate._scan_batches, tmp_path)
    lines = evaluate._errored_jobs(tmp_path)
    assert len(lines) == 1 and ", and " in lines[0]


def test_a_marker_that_cannot_be_read_is_read_as_before(tmp_path: Path) -> None:
    # A marker gone between finding it and reading it, or whose job's
    # directory cannot be listed.
    job = tmp_path / "sweep" / "0"
    job.mkdir(parents=True)
    _same(_old_has_result_since, evaluate._has_result_since, job / CASE_ERRORED)
    _same(_old_has_gate_record, evaluate._has_gate_record, job / "nothing")
    _same(_old_has_result, evaluate._has_result, job / "nothing")
    (job / CASE_ERRORED).write_text("{}", encoding="utf-8")
    _same(_old_has_result_since, evaluate._has_result_since, job / CASE_ERRORED)
    _same(_old_has_result_since, evaluate._has_result_since, job / CASE_ERRORED / "x")


def test_a_plan_with_many_jobs_is_read_as_before(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    jobs = [sweep / str(j) for j in range(9)] + [tmp_path / "outside"]
    write_sweep_plan(sweep, [(job, []) for job in jobs])
    (sweep / "3").mkdir()
    (sweep / "4").mkdir()
    (sweep / "4" / f"a{_RESULT_SUFFIX}").write_text("{}", encoding="utf-8")
    _same(_old_unstarted_jobs, evaluate._unstarted_jobs, tmp_path)
    lines, unreadable = evaluate._unstarted_jobs(tmp_path)
    assert unreadable == [] and ", and 4 more" in lines[0]


def test_evaluate_re_exports_the_checked_scans() -> None:
    # gate.evaluate's names are gate.unrecorded's objects (#2124), and its
    # summary reader returns what _summary_cases() read.
    from autoware_carla_scenario.gate import unrecorded

    for name in (
        "BATCH_RESULTS",
        "_NAMED",
        "_RESULT_SUFFIX",
        "_batch_dirs",
        "_errored_jobs",
        "_has_gate_record",
        "_has_result",
        "_has_result_since",
        "_recorded_at_stem",
        "_unstarted_jobs",
    ):
        assert getattr(evaluate, name) is getattr(unrecorded, name), name
    assert evaluate._summary_cases is unrecorded._summary_cases
