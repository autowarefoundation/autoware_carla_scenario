"""What ``scenario-gate`` finds a set of runs left unrecorded, pinned.

Pins, against verbatim copies of them as they were in ``gate.evaluate`` (the
``_old_*`` oracles below), what the helpers that scan a set of runs return or
raise: the batches under a directory (``_batch_dirs``), whether a directory
holds a gate record or a run's result (``_has_gate_record``,
``_has_result``, ``_has_result_since``), the gate record at a summary entry's
stem (``_recorded_at_stem``), a local batch's unrecorded cases
(``_summary_cases``), a SLURM batch's (``_slurm_unrecorded``, read from its
manifest as a ``dict`` reads it; now ``gate.unrecorded``'s ``_slurm_cases``
over the typed ``Manifest.cases()``, #2394), a sweep's given-up jobs
(``_errored_jobs``) and its jobs that never started (``_unstarted_jobs``),
and what ``_scan_batches()``, ``unrecorded_cases()`` and
``batch_read_errors()`` make of them.

Each runs, old and new, on the same directory: random sets of batches,
markers and plans, and summary entries whose flags are ``true``, ``1``,
``1.0``, ``"true"`` or ``false`` (only ``true`` is ``is True``), whose
names are false values of every JSON type, and whose stems are any JSON
value or a path that is no file name of its own; and SLURM batches whose
manifests cannot be read, whose cases have no index ``int()`` reads, labels,
scenarios and stems of every JSON type, and outcomes that passed, failed,
errored, hold no verdict or are missing.

One change is pinned in the oracle: a summary holding an entry that is no
JSON object raised ``AttributeError`` out of ``scenario-gate``; it cannot be
read (``not a JSON array of objects``), as ``batch_resume`` refuses it
(#2231).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import pytest

from autoware_carla_scenario.gate import evaluate, unrecorded
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
from autoware_carla_scenario.slurm.manifest import (
    BatchLayout,
    ManifestCase,
    manifest_of_text,
    recorded_verdict,
)
from autoware_carla_scenario.slurm.records import JsonMembers

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
    # #2231: a summary with an entry that is no object cannot be read, as
    # batch_resume.read_resumed_batch() refuses it (it raised AttributeError).
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return f"{path}: not a batch summary (not a JSON array of objects)"
    unrecorded = [
        str(entry.get("scenario") or i)
        for i, entry in enumerate(entries)
        if (entry.get("error") is True and not _old_recorded_at_stem(directory, entry))
        or entry.get("pending") is True
    ]
    return unrecorded, len(entries)


def _summary_unrecorded(directory: Path) -> tuple[list[str], int] | str:
    """What evaluate._scan_batches() reads of a local batch's summary
    (``_summary_cases``): its unrecorded cases and how many it has, or why it
    cannot be read."""
    names, total, why = evaluate._summary_cases(directory)
    return (names, total) if why is None else why


def _old_slurm_unrecorded(directory: Path) -> tuple[list[str], int] | str:
    """The labels of a SLURM batch's cases that errored, have no outcome or
    one holding no verdict, and left no gate record at their stem; and how
    many cases it has.  Why the manifest cannot be read, instead, if it
    cannot."""
    layout = BatchLayout(directory)
    try:
        manifest = layout.read_manifest()
        cases = list(manifest["cases"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return f"{layout.manifest}: the SLURM manifest cannot be read ({exc})"
    unrecorded: list[str] = []
    for case in cases:
        try:
            index = int(case["index"])
        except (KeyError, TypeError, ValueError):
            return f"{layout.manifest}: a case without an index ({case!r})"
        record = layout.read_result(index)
        if (
            record is not None
            and recorded_verdict(record) is not None
            and not record.errored
        ):
            continue
        # A stem of its own only, as a local batch's (#2265).
        if _old_recorded_at_stem(directory, case):
            continue
        unrecorded.append(str(case.get("label") or case.get("scenario") or index))
    return unrecorded, len(cases)


def old_slurm_cases(directory: Path) -> tuple[list[str], int, str | None]:
    """:func:`_old_slurm_unrecorded` as ``gate.unrecorded._slurm_cases``
    returns it: a triple, for the readers that patch it in
    (test_slurm_manifest_record's ``_old_reading``)."""
    found = _old_slurm_unrecorded(directory)
    if isinstance(found, str):
        return [], 0, found
    return found[0], found[1], None


def _slurm_cases(directory: Path) -> tuple[list[str], int] | str:
    """What evaluate._scan_batches() reads of a SLURM batch
    (``_slurm_cases``): its unrecorded cases and how many it has, or why its
    manifest cannot be read."""
    names, total, why = unrecorded._slurm_cases(directory)
    return (names, total) if why is None else why


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
            found = _old_slurm_unrecorded(directory)
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


# Values a SLURM case's index takes: what int() reads, what it refuses
# (ValueError, TypeError), and an infinity, whose OverflowError the scan let
# through.
_INDICES: list[Any] = [
    0,
    1,
    2,
    3,
    "1",
    " 2 ",
    "x",
    "1e3",
    1.5,
    -1,
    True,
    False,
    None,
    [],
    {},
    [0],
    float("nan"),
    float("inf"),
    10**30,
]

#: The members of a case a test leaves out: absent from the case.
_MISSING = object()


def _outcome_text(rng: random.Random, index: int) -> str | None:
    """A SLURM case's outcome (``results/<i>.json``): one that passed,
    failed, errored or holds no verdict, one that cannot be read, or none."""
    result: dict[str, Any] = {
        "passed": rng.choice([True, False]),
        "message": "m",
        "elapsed_seconds": 1.5,
        "condition_statuses": [],
    }
    roll = rng.random()
    if roll < 0.2:
        return None
    if roll < 0.3:
        return rng.choice(["not json", "[1]", "null", '"\\udcff"'])
    if roll < 0.45:
        # No verdict: `passed` of no boolean, or a result that cannot be
        # reported.
        result[rng.choice(["passed", "message", "condition_statuses"])] = rng.choice(
            ["true", 1, None]
        )
    record: dict[str, Any] = {"index": index, "result": result}
    if rng.random() < 0.6:
        record["errored"] = rng.choice(_FLAGS)
    if rng.random() < 0.1:
        record["result"] = rng.choice([None, [], "x"])
    return json.dumps(record)


def _random_case(rng: random.Random, indices: list[Any]) -> Any:
    if rng.random() < 0.03:
        return rng.choice([None, 1, "x", [], ["index"], True, 1.5, {"a": 1}])
    case: dict[str, Any] = {}
    for key, values in (
        ("index", indices),
        ("label", _NAMES),
        ("scenario", _NAMES),
        ("stem", _STEMS),
        ("config", [{"k": [1, {"x": None}]}]),
    ):
        if rng.random() < (0.97 if key == "index" else 0.75):
            case[key] = rng.choice(values)
    return case


def _write_slurm(rng: random.Random, directory: Path) -> None:
    """A SLURM batch: its manifest (or one that cannot be read), its cases'
    outcomes and the gate records at stems its cases may name."""
    layout = BatchLayout(directory)
    layout.root.mkdir(parents=True, exist_ok=True)
    roll = rng.random()
    document: Any
    if roll < 0.03:
        layout.manifest.write_bytes(b"\xff\xfe not utf-8")
        document = None
    elif roll < 0.06:
        layout.manifest.write_text("{not json", encoding="utf-8")
        document = None
    elif roll < 0.08:
        layout.manifest.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
        document = None
    elif roll < 0.1:
        layout.manifest.mkdir()
        document = None
    elif roll < 0.15:
        document = rng.choice([[], 1, None, "x"])
    elif roll < 0.2:
        document = {"cases": []} if rng.random() < 0.5 else {"version": 3}
    elif roll < 0.25:
        document = {"version": rng.choice([1, 2, 1.0])}
    elif roll < 0.33:
        document = {
            "version": 1,
            "cases": rng.choice([1, True, None, 1.5, "ab", {"a": 1}, "", {}]),
        }
    else:
        # Mostly cases with a good index, so that most batches are read.
        good = _INDICES[:4] + ["1", 1.5, True]
        indices = good if rng.random() < 0.8 else _INDICES
        count = rng.choice([0, 1, 3, 6, 8])
        document = {
            "version": rng.choice([1, 2]),
            "cases": [_random_case(rng, indices) for _ in range(count)],
        }
    if document is not None:
        layout.manifest.write_text(json.dumps(document), encoding="utf-8")
    for index in range(4):
        text = _outcome_text(rng, index)
        if text is None:
            continue
        layout.result(index).parent.mkdir(parents=True, exist_ok=True)
        if rng.random() < 0.05:
            layout.result(index).mkdir()
        else:
            layout.result(index).write_text(text, encoding="utf-8")
    for stem in ("run", "run2", "ünï", "a\\b"):
        if rng.random() < 0.5:
            record = directory / f"{stem}{GATE_RECORD_SUFFIX}"
            if rng.random() < 0.15:
                record.mkdir(exist_ok=True)
            else:
                record.write_text("{}", encoding="utf-8")


def _random_tree(rng: random.Random, root: Path) -> None:
    root.mkdir(parents=True)
    for b in range(rng.randrange(0, 4)):
        _write_summary(rng, root / f"batch{b}" / rng.choice(["", "deep/er"]))
    for b in range(rng.randrange(0, 4)):
        _write_slurm(rng, root / f"slurm{b}" / rng.choice(["", "deep/er"]))
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
        _same(_old_summary_unrecorded, _summary_unrecorded, path)
        _same(_old_slurm_unrecorded, evaluate._slurm_unrecorded, path)
        _same(_old_slurm_unrecorded, unrecorded._slurm_unrecorded, path)
        _same(_old_slurm_unrecorded, _slurm_cases, path)
        _same(_old_errored_jobs, evaluate._errored_jobs, path)
        _same(_old_unstarted_jobs, evaluate._unstarted_jobs, path)
        _same(_old_scan_batches, evaluate._scan_batches, path)
        _same(_old_scan_batches, unrecorded._scan_batches, path)
        _same(lambda p: _old_scan_batches(p)[0], evaluate.unrecorded_cases, path)
        _same(lambda p: _old_scan_batches(p)[1], evaluate.batch_read_errors, path)
        if path.name == CASE_ERRORED or path.is_file():
            _same(_old_has_result_since, evaluate._has_result_since, path)


@pytest.mark.parametrize("seed", range(80))
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
    # An entry that is no object makes the summary unreadable (#2231): it
    # raised AttributeError (its `.get()`).
    _same(_old_summary_unrecorded, _summary_unrecorded, directory)
    assert _summary_unrecorded(directory) == (
        f"{directory / BATCH_RESULTS}: not a batch summary (not a JSON array of "
        "objects)"
    )
    _same(_old_scan_batches, evaluate._scan_batches, tmp_path)
    objects = [e for e in entries if isinstance(e, dict)]
    (directory / BATCH_RESULTS).write_text(json.dumps(objects), encoding="utf-8")
    _same(_old_summary_unrecorded, _summary_unrecorded, directory)
    found = _summary_unrecorded(directory)
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
    # summary reader returns what _summary_cases() read; so are its SLURM
    # scan and the scan of a set of runs (#2394).
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
        "_scan_batches",
        "_slurm_cases",
        "_slurm_unrecorded",
    ):
        assert getattr(evaluate, name) is getattr(unrecorded, name), name
    assert evaluate._summary_cases is unrecorded._summary_cases


def _old_batch_line(directory: Path, names: list[str], total: int) -> str:
    """_scan_batches()'s line for a batch, as it was before #2386."""
    shown = ", ".join(names[:_NAMED])
    more = f", and {len(names) - _NAMED} more" if len(names) > _NAMED else ""
    return (
        f"{directory}: {len(names)} of {total} case(s) raised or have no "
        f"outcome, and left no gate record ({shown}{more}); the gate would "
        "be judged over the runs that survived"
    )


def test_a_batch_s_line_is_written_as_before() -> None:
    # The line _scan_batches() writes per batch is the checked
    # gate.unrecorded's _batch_line() (#2386).
    from autoware_carla_scenario.gate import unrecorded

    rng = random.Random(2386)
    pool = ["a#0", "", "ü", "x, y", "7", "pkg/cut_in#12"]
    for _ in range(1000):
        names = [rng.choice(pool) for _ in range(rng.randrange(0, 14))]
        total = rng.choice([0, len(names), len(names) + rng.randrange(0, 50), -1])
        directory = Path(rng.choice(["/runs/b", "rel", ".", "/ü dir/x"]))
        assert unrecorded._batch_line(directory, names, total) == _old_batch_line(
            directory, names, total
        )


# ---------------------------------------------------------------------------
# A SLURM batch's cases (#2394)
# ---------------------------------------------------------------------------


def _slurm_batch(directory: Path, cases: Any, outcomes: dict[int, str]) -> Path:
    layout = BatchLayout(directory)
    layout.root.mkdir(parents=True)
    layout.manifest.write_text(
        json.dumps({"version": 1, "cases": cases}), encoding="utf-8"
    )
    for index, text in outcomes.items():
        layout.result(index).parent.mkdir(parents=True, exist_ok=True)
        layout.result(index).write_text(text, encoding="utf-8")
    return directory


_PASSED = json.dumps(
    {
        "index": 0,
        "result": {
            "passed": True,
            "message": "ok",
            "elapsed_seconds": 1.0,
            "condition_statuses": [],
        },
    }
)


def _compare_slurm(directory: Path) -> None:
    _same(_old_slurm_unrecorded, evaluate._slurm_unrecorded, directory)
    _same(_old_slurm_unrecorded, unrecorded._slurm_unrecorded, directory)
    _same(_old_slurm_unrecorded, _slurm_cases, directory)
    _same(_old_scan_batches, evaluate._scan_batches, directory.parent)


def test_every_slurm_case_is_read_as_before(tmp_path: Path) -> None:
    # Every label, scenario and stem, alone and in pairs, against the
    # records at the stems, with and without an outcome that passed.
    cases: list[Any] = []
    outcomes: dict[int, str] = {}
    for label in [*_NAMES, _MISSING]:
        for scenario in [*_NAMES, _MISSING]:
            case: dict[str, Any] = {"index": len(cases)}
            if label is not _MISSING:
                case["label"] = label
            if scenario is not _MISSING:
                case["scenario"] = scenario
            cases.append(case)
    for stem in _STEMS:
        cases.append({"index": len(cases), "stem": stem, "label": "s"})
        cases.append({"stem": stem, "index": str(len(cases))})
    for index in range(0, len(cases), 3):
        outcomes[index] = _PASSED
    directory = _slurm_batch(tmp_path / "runs" / "batch", cases, outcomes)
    for stem in ("run", "ünï", "a\\b", "missing"):
        if stem != "missing":
            (directory / f"{stem}{GATE_RECORD_SUFFIX}").write_text("{}", "utf-8")
    _compare_slurm(directory)
    found = evaluate._slurm_unrecorded(directory)
    assert not isinstance(found, str)
    # A false label falls back to the scenario, and a false scenario to the
    # index; a stem names a record only as a string of its own.
    nameless = next(
        c["index"] for c in cases if c.get("label") == "" and c.get("scenario") == 0
    )
    assert "case" in found[0] and str(nameless) in found[0]
    assert "{'a': 1}" in found[0]
    assert found[1] == len(cases)


@pytest.mark.parametrize("index", range(len(_INDICES)))
def test_each_index_of_a_slurm_case_is_read_as_before(
    tmp_path: Path, index: int
) -> None:
    # A case without an index int() reads makes the manifest unreadable,
    # named with the whole case; an infinity's OverflowError goes through.
    value = _INDICES[index]
    cases = [
        {"index": 0, "label": "first"},
        {"index": value, "label": "x", "config": {"k": [1, None]}},
        {"label": "last"},
    ]
    directory = _slurm_batch(tmp_path / "runs" / "batch", cases, {})
    _compare_slurm(directory)


@pytest.mark.parametrize(
    "case",
    [None, 1, 1.5, True, "x", "", [], ["index"], {}, {"label": "no index"}],
    ids=repr,
)
def test_a_slurm_case_without_an_index_is_named_as_before(
    tmp_path: Path, case: Any
) -> None:
    directory = _slurm_batch(tmp_path / "runs" / "batch", [{"index": 0}, case], {})
    _compare_slurm(directory)
    manifest = BatchLayout(directory).manifest
    assert evaluate._slurm_unrecorded(directory) == (
        f"{manifest}: a case without an index ({case!r})"
    )


@pytest.mark.parametrize(
    "text",
    [
        '{"version": 1}',
        '{"version": 1, "cases": 1}',
        '{"version": 1, "cases": 1.5}',
        '{"version": 1, "cases": true}',
        '{"version": 1, "cases": null}',
        '{"version": 1, "cases": "ab"}',
        '{"version": 1, "cases": {"a": 1}}',
        '{"version": 1, "cases": []}',
        '{"version": 3, "cases": []}',
        '{"cases": []}',
        "[]",
        "{",
    ],
)
def test_a_slurm_manifest_that_cannot_be_read_is_reported_as_before(
    tmp_path: Path, text: str
) -> None:
    directory = tmp_path / "runs" / "batch"
    layout = BatchLayout(directory)
    layout.root.mkdir(parents=True)
    layout.manifest.write_text(text, encoding="utf-8")
    _compare_slurm(directory)


def test_a_slurm_batch_without_a_manifest_is_reported_as_before(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "runs" / "batch"
    directory.mkdir(parents=True)
    _compare_slurm(directory)
    (directory / "slurm" / "manifest.json").mkdir(parents=True)
    _compare_slurm(directory)


# ---------------------------------------------------------------------------
# What ManifestCase gives the gate (#2394)
# ---------------------------------------------------------------------------


def _raw_cases() -> list[Any]:
    cases: list[Any] = [None, 1, -1.5, True, "x", "", [], [1, [2]], {}]
    for name in [*_NAMES, _MISSING]:
        for stem in [*_STEMS, _MISSING]:
            case: dict[str, Any] = {"index": 0, "config": {"k": [1, None]}}
            if name is not _MISSING:
                case["label"] = name
                case["scenario"] = name
            if stem is not _MISSING:
                case["stem"] = stem
            cases.append(case)
    for name in _NAMES:
        cases.append({"label": "", "scenario": name})
        cases.append({"scenario": name, "label": None, "z": 1})
    return cases


def _old_string_stem(raw: Any) -> str | None:
    stem = raw.get("stem")
    return stem if isinstance(stem, str) else None


def test_a_case_s_stem_name_and_repr_are_as_the_dict_gave_them() -> None:
    cases = _raw_cases()
    manifest = manifest_of_text(json.dumps({"version": 1, "cases": cases}))
    assert manifest is not None
    read = manifest.cases()
    assert len(read) == len(cases)
    for case, raw in zip(read, cases):
        _same(partial_of(_old_string_stem, raw), case.string_stem)
        for index in (0, 7):
            _same(
                partial_of(
                    lambda raw, index: str(
                        raw.get("label") or raw.get("scenario") or index
                    ),
                    raw,
                    index,
                ),
                partial_of(case.name, index),
            )
        assert case.case_repr() == repr(raw)
        # The members the checked readers read stay what they were: the
        # scenario is not one of them.
        if isinstance(raw, dict):
            assert "scenario" not in case.members.keys()


def partial_of(function: Callable[..., Any], *args: Any) -> Callable[[], Any]:
    return lambda: function(*args)


def test_a_case_not_read_from_a_manifest_shows_what_it_holds() -> None:
    members = JsonMembers()
    members.put("index", "3")
    members.put("label", '""')
    members.put("stem", '"run"')
    case = ManifestCase(members, "", '"demo"')
    assert case.name(3) == "demo"
    assert case.string_stem() == "run"
    assert case.case_repr() == repr({"index": 3, "label": "", "stem": "run"})
    assert ManifestCase(members).name(3) == "3"
    members.put("stem", "1")
    assert case.string_stem() is None
    for shape, value in (("[]", []), ('""', ""), ("0", 0), ("null", None)):
        typed = ManifestCase(JsonMembers(), shape)
        assert typed.case_repr() == repr(value)
        for call in (typed.string_stem, lambda: typed.name(0)):
            with pytest.raises(AttributeError) as error:
                call()
            with pytest.raises(AttributeError) as expected:
                value.get("stem")  # type: ignore[union-attr]
            assert str(error.value) == str(expected.value)
