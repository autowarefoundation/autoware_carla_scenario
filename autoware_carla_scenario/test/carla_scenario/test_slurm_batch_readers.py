"""What ``scenario-slurm status``, ``cancel`` and a resubmit's busy check
print, and the viewer's batch status, of a SLURM batch's files.

Pinned against the code that read a batch before its readers moved into the
library check (the ``_old_*`` oracles below, copied from ``slurm/cli.py``
and ``ui/slurm_launch.py`` as they were), over batches of every shape:
normal ones, records of every JSON type, keys named twice, chunks, cases and
statuses that cannot be iterated or measured, files missing, empty, not
JSON, truncated, not UTF-8, not objects or directories, unreadable records
fresh and stale, submissions with and without job ids (in flight, timed
out, stale), and values nested deep.  ``squeue`` and ``scancel`` are fakes
on ``PATH``, and the clock is fixed, so that the oracles and the readers ask
the same questions and get the same answers; each command's exit status (or
what it raises), what it prints to stdout and stderr, what it logs, the jobs
it cancels and the files it writes are compared.
"""

from __future__ import annotations

import json
import logging
import os
import random
import stat
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.slurm import cli
from autoware_carla_scenario.slurm import queue as queue_mod
from autoware_carla_scenario.slurm.collect import unfinished_cases
from autoware_carla_scenario.slurm.manifest import (
    MANIFEST_VERSION,
    SBATCH_IN_FLIGHT_SECONDS,
    BatchLayout,
    TaskStatus,
    no_job_note,
    recorded_verdict,
    sbatch_in_flight,
    sbatch_timed_out,
)
from autoware_carla_scenario.slurm.queue import ACTIVE_STATES
from autoware_carla_scenario.slurm.records import JsonMembers
from autoware_carla_scenario.slurm.settings import SlurmSettings
from autoware_carla_scenario.ui import slurm_launch
from autoware_carla_scenario.ui.slurm_case_state import case_state, no_job_case_state

#: The fixed clock both the oracles and the readers read.
_NOW = 1_800_000_000.25

_old_logger = logging.getLogger("autoware_carla_scenario.slurm.cli")


def _squeue_states(job: str) -> dict[int, str] | None:
    """The readers' squeue, looked up when it is asked (the fake on PATH)."""
    return queue_mod.squeue_states(job)


def _cancel_jobs(jobs: list[str]) -> bool:
    return queue_mod.cancel_jobs(jobs)


# -- The oracles: the readers as they were ------------------------------------------


def _old_active_jobs(
    layout: BatchLayout, below: int | None = None
) -> tuple[list[str], list[str], list[int], list[int], list[int]]:
    active: list[str] = []
    unknown: list[str] = []
    in_flight: list[int] = []
    unreadable: list[int] = []
    timed_out: list[int] = []
    for index in layout.submissions():
        if below is not None and index >= below:
            continue
        path = layout.submission(index)
        try:
            submission = layout.read_submission(index)
        except FileNotFoundError:
            continue  # removed since: its sbatch failed
        except (OSError, ValueError):
            submission = None
        if submission is None:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < SBATCH_IN_FLIGHT_SECONDS:
                unreadable.append(index)
            else:
                _old_logger.warning(
                    "Ignoring %s: it cannot be read as a submission record, "
                    "and was last modified %.0f s ago",
                    path,
                    age,
                )
            continue
        job = submission.get("job_id")
        if not job:
            if sbatch_in_flight(submission):
                if sbatch_timed_out(submission):
                    timed_out.append(index)
                else:
                    in_flight.append(index)
            continue
        states = _squeue_states(str(job))
        if states is None:
            unknown.append(str(job))
        elif any(state in ACTIVE_STATES for state in states.values()):
            active.append(str(job))
    return active, unknown, in_flight, unreadable, timed_out


def _old_busy(layout: BatchLayout, below: int | None = None) -> str | None:
    active, unknown, in_flight, unreadable, timed_out = _old_active_jobs(layout, below)
    if active:
        return (
            f"job(s) {', '.join(active)} of this batch are still queued or "
            "running; wait for them (scenario-slurm wait), cancel them "
            "(scenario-slurm cancel), or pass --force"
        )
    if in_flight:
        return (
            f"submission(s) {', '.join(str(i) for i in in_flight)} of this batch "
            "are being submitted (sbatch has not returned yet), by another "
            "resubmit or the viewer; try again once it has, or pass --force. "
            "A record left by a submitter that died is ignored "
            f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s after it was made"
        )
    if timed_out:
        return (
            f"the sbatch of submission(s) {', '.join(str(i) for i in timed_out)} "
            "of this batch timed out, and SLURM may have taken the job all the "
            "same: check `squeue --me` (scancel the job if it is there and you "
            "want its cases run again), then pass --force, or try again once "
            f"the record is ignored, {SBATCH_IN_FLIGHT_SECONDS:.0f} s after it "
            "was made"
        )
    if unreadable:
        files = ", ".join(str(layout.submission(i)) for i in unreadable)
        return (
            f"the submission record(s) {files} of this batch are unreadable "
            "(another submitter may still be writing them); try again shortly, "
            "or pass --force. An unreadable record is ignored "
            f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s after it was last modified"
        )
    if unknown:
        return (
            f"squeue could not tell whether job(s) {', '.join(unknown)} of "
            "this batch are still queued or running; check with squeue, or pass "
            "--force if they are not"
        )
    return None


def _old_status(layout: BatchLayout) -> int:
    manifest = layout.read_manifest()
    cases = manifest["cases"]
    done = passed = errored = 0
    for case in cases:
        record = layout.read_result(int(case["index"]))
        if record is None:
            continue
        done += 1
        passed += recorded_verdict(record) is True
        errored += record.errored
    print(f"Batch {layout.batch_dir}: {len(cases)} case(s)")  # noqa: T201
    print(  # noqa: T201
        f"  {done} with a result ({passed} passed, {errored} errored), "
        f"{len(cases) - done} without"
    )
    for index in layout.submissions():
        submission = layout.submission_record(index)
        if submission is None:
            path = layout.submission(index)
            if path.exists():
                print(  # noqa: T201
                    f"  submission {index}: unreadable record {path} "
                    "(empty, not JSON, or not a JSON object)"
                )
            continue
        job = submission.get("job_id")
        chunks = submission.get("chunks") or []
        size = f"{len(chunks)} task(s), {len(submission.get('cases') or [])} case(s)"
        if job:
            states = _squeue_states(str(job))
            queued = (
                "squeue unavailable"
                if states is None
                else ", ".join(f"{t}:{s}" for t, s in sorted(states.items())) or "ended"
            )
            print(f"  submission {index}: job {job}, {size}; in the queue: {queued}")  # noqa: T201
        else:
            print(f"  submission {index}: {size}; {no_job_note(submission)}")  # noqa: T201
        for task, chunk in enumerate(chunks):
            status = layout.read_task(index, task) or TaskStatus(JsonMembers())
            ports = status.get("ports") or {}
            rss = status.get("peak_rss_mb")
            host = status.get("host")
            error = status.get("error")
            print(  # noqa: T201
                f"    task {task}: {status.state_text('not started')}"
                f"{' on ' + host if host else ''}"
                f"{' port ' + str(ports['rpc']) if ports.get('rpc') else ''}"
                f", {len(status.get('done') or [])}/{len(chunk)} case(s)"
                f"{f', peak RSS {rss} MB' if rss else ''}"
                f"{' -- ' + error if error else ''}"
            )
    return 0


def _old_cancel(layout: BatchLayout) -> int:
    jobs: list[str] = []
    for index in layout.submissions():
        submission = layout.submission_record(index)
        if submission is None:
            path = layout.submission(index)
            if path.exists():
                print(  # noqa: T201
                    f"Skipping {path}: it cannot be read as a submission record; "
                    "if it named a job, cancel that with scancel",
                    file=sys.stderr,
                )
            continue
        for key in ("job_id", "collect_job_id"):
            if submission.get(key):
                jobs.append(str(submission[key]))
    if not jobs:
        print("The batch has no job to cancel.")  # noqa: T201
        return 0
    print(f"Cancelling job(s) {', '.join(jobs)}")  # noqa: T201
    return 0 if _cancel_jobs(jobs) else 1


def _old_resubmit_dry_run(layout: BatchLayout, failed: bool, force: bool) -> int:
    """``resubmit --dry-run`` (of every unfinished case) as it was."""
    manifest = layout.read_manifest()
    reason = None if force else _old_busy(layout)
    if reason is not None:
        print(f"Error: {reason}", file=sys.stderr)  # noqa: T201
        return 1
    chosen = unfinished_cases(layout, failed=failed)
    if not chosen:
        print("Nothing to resubmit: every case has a verdict.")  # noqa: T201
        return 0
    SlurmSettings.from_mapping(manifest["launcher"]).replace(
        array_parallelism=None, cases_per_task=None, wait=None
    )
    labels_of = [manifest["cases"][i]["label"] for i in chosen]
    print(f"Resubmitting {len(chosen)} case(s): {', '.join(labels_of)}")  # noqa: T201
    return 0


def _old_main(call: Callable[[], int]) -> int:
    """*call* as ``scenario-slurm``'s main ran a command."""
    try:
        return call()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        return 1


_ACTIVE = tuple(sorted(ACTIVE_STATES))


def _old_case_states(
    layout: BatchLayout,
    manifest: Any,
    squeue: dict[int, dict[int, str] | None],
) -> list[dict[str, Any]]:
    given: dict[int, tuple[int, int, Any]] = {}
    for index in layout.submissions():
        sub = layout.submission_record(index)
        if sub is None:
            continue
        for task, chunk in enumerate(sub.get("chunks") or []):
            for case in chunk:
                given[int(case)] = (index, task, sub)
    out = []
    for case in manifest.get("cases") or []:
        index = int(case["index"])
        record = layout.read_result(index)
        entry: dict[str, Any] = {
            "index": index,
            "label": case.get("label"),
            "stem": case.get("stem"),
            "overrides": case.get("case_overrides") or [],
        }
        result: dict[str, Any] = {}
        errored = passed = False
        if record is not None:
            recorded = record.get("result")
            result = recorded if isinstance(recorded, dict) else {}
            errored = record.errored
            passed = not errored and recorded_verdict(record) is True
        queued: str | None = None
        squeue_known = at_case = False
        task, job_id, task_state, task_error = 0, "", "", None
        no_job: tuple[str, str] | None = None
        if record is None and index in given and not given[index][2].get("job_id"):
            sub_index, _task, sub = given[index]
            no_job = no_job_case_state(
                sub_index,
                sbatch_timed_out(sub),
                sbatch_in_flight(sub),
                f"{sub.get('sbatch_timed_out_at') or ''}",
            )
        elif record is None and index in given:
            sub_index, task, sub = given[index]
            states = squeue.get(sub_index)
            squeue_known = states is not None
            queued = None if states is None else states.get(task)
            status = layout.read_task(sub_index, task) or TaskStatus(JsonMembers())
            job_id = f"{sub.get('job_id')}"
            if queued in ACTIVE_STATES:
                at_case = queued == "RUNNING" and status.is_at(index)
            elif squeue_known:
                task_state = status.state_text("never started")
                task_error = status.error
        state, message = no_job or case_state(
            record is not None,
            errored,
            passed,
            index in given,
            queued,
            squeue_known,
            at_case,
            task,
            job_id,
            task_state,
            task_error,
            _ACTIVE,
        )
        if record is not None:
            entry.update(
                state=state,
                message=result.get("message", ""),
                elapsed_seconds=result.get("elapsed_seconds"),
                host=record.get("host"),
                job_id=record.get("job_id"),
            )
        else:
            entry.update(state=state, message=message)
        out.append(entry)
    return out


def _old_batch_status(batch_dir: Path) -> dict[str, Any] | None:
    layout = BatchLayout(batch_dir)
    try:
        manifest = layout.read_manifest()
    except (FileNotFoundError, ValueError, OSError):
        return None
    squeue: dict[int, dict[int, str] | None] = {}
    submissions = []
    for index in layout.submissions():
        sub = layout.submission_record(index)
        if sub is None:
            path = layout.submission(index)
            if path.exists():
                submissions.append(
                    {
                        "index": index,
                        "unreadable": True,
                        "note": f"Its record {path} cannot be read (empty, not "
                        "JSON, or not a JSON object): its job is not shown and "
                        "its cases count as not submitted by it.",
                        "job_id": None,
                        "collect_job_id": None,
                        "collect_error": None,
                        "cases": 0,
                        "active": False,
                        "squeue_unavailable": False,
                        "tasks": [],
                    }
                )
            continue
        job = sub.get("job_id")
        states = _squeue_states(str(job)) if job else {}
        squeue[index] = states
        timed_out = not job and sbatch_timed_out(sub)
        in_flight = not job and not timed_out and sbatch_in_flight(sub)
        tasks = []
        for task, chunk in enumerate(sub.get("chunks") or []):
            status = layout.read_task(index, task) or TaskStatus(JsonMembers())
            tasks.append(
                {
                    "task": task,
                    "cases": chunk,
                    "squeue": None if states is None else states.get(task),
                    "state": status.get("state")
                    if status.has("state")
                    else "not started",
                    "host": status.get("host"),
                    "done": len(status.get("done") or []),
                    "current": status.get("current"),
                    "error": status.get("error"),
                    "peak_rss_mb": status.get("peak_rss_mb"),
                }
            )
        submissions.append(
            {
                "index": index,
                "job_id": job,
                "collect_job_id": sub.get("collect_job_id"),
                "collect_error": sub.get("collect_error"),
                "cases": len(sub.get("cases") or []),
                "active": in_flight
                or (
                    states is not None
                    and any(s in ACTIVE_STATES for s in states.values())
                ),
                "squeue_unavailable": states is None,
                "tasks": tasks,
                "unreadable": False,
                "note": None if job else no_job_note(sub),
                "sbatch_timed_out": bool(timed_out),
                "in_flight": bool(in_flight),
            }
        )
    cases = _old_case_states(layout, manifest, squeue)
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["state"]] = counts.get(case["state"], 0) + 1
    collected = layout.root / "collected.json"
    return {
        "batch_dir": str(batch_dir),
        "created": manifest.get("created"),
        "scenario_names": manifest.get("scenario_names") or [],
        "launcher": manifest.get("launcher") or {},
        "cases": cases,
        "counts": counts,
        "total": len(cases),
        "submissions": submissions,
        "active": any(s["active"] for s in submissions),
        "collected": collected.is_file(),
        "summary": (batch_dir / "batch_results.json").is_file(),
    }


# -- The fakes: squeue, scancel and the clock ---------------------------------------

_SQUEUE = """\
import json, os, sys
job = sys.argv[sys.argv.index("-j") + 1]
answers = json.load(open(os.environ["FAKE_SLURM_SQUEUE"], encoding="utf-8"))
answer = answers.get(job, "invalid")
if answer == "fail":
    sys.stderr.write("slurm_load_jobs error: Socket timed out\\n")
    sys.exit(1)
if answer == "invalid":
    sys.stderr.write("slurm_load_jobs error: Invalid job id specified\\n")
    sys.exit(1)
sys.stdout.write(answer)
"""

_SCANCEL = """\
import json, os, sys
with open(os.environ["FAKE_SLURM_SCANCEL"], "a", encoding="utf-8") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(1 if os.path.exists(os.environ["FAKE_SLURM_SCANCEL"] + ".fail") else 0)
"""


def install_fake_slurm(directory: Path) -> dict[str, str]:
    """Write fake ``squeue`` and ``scancel`` executables to *directory*; the
    environment that puts them first on ``PATH`` (and where they read their
    answers and log the jobs cancelled)."""
    bin_dir = directory / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, body in (("squeue", _SQUEUE), ("scancel", _SCANCEL)):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_SLURM_SQUEUE": str(directory / "squeue.json"),
        "FAKE_SLURM_SCANCEL": str(directory / "scancel.log"),
    }


# -- The corpus ---------------------------------------------------------------------

#: Values of every JSON type, as their text in a file.
_VALUES: list[str] = [
    "null",
    "true",
    "false",
    "0",
    "1",
    "2",
    "-1",
    "1.0",
    "1.5",
    "-0.0",
    "NaN",
    "Infinity",
    "1e400",
    "123456789012345678901234567890",
    '""',
    '"text"',
    '"1"',
    '"n\\u00fc \\u2603"',
    '"\\ud800"',
    "[]",
    "[1]",
    '["a", "b"]',
    "[[0, 1], [2]]",
    "{}",
    '{"rpc": 2000}',
    '{"k": "v"}',
]

#: Files that hold no record.
_BROKEN: list[bytes] = [
    b"",
    b"{",
    b'{"job_id": "1"',
    b"not json",
    b"\xff\xfe{}",
    b"\xef\xbb\xbf{}",
    b"[]",
    b"3",
    b'"text"',
    b"null",
]


#: A file that holds no record, or a directory in its place.
_BROKEN_OR_DIR: list[bytes | str] = [*_BROKEN, "dir"]


def nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def _value(rng: random.Random, *common: str) -> str:
    """A value: one of *common* mostly, else one of every JSON type."""
    if common and rng.random() < 0.7:
        return rng.choice(common)
    return rng.choice(_VALUES)


def _object(members: list[tuple[str, str]]) -> str:
    return "{" + ", ".join(f"{json.dumps(k)}: {v}" for k, v in members) + "}"


def _maybe(rng: random.Random, members: list[tuple[str, str]], p: float = 0.9) -> list:
    """*members*, each kept with probability *p*, now and then one named twice."""
    kept = [m for m in members if rng.random() < p]
    if kept and rng.random() < 0.05:
        key, _ = rng.choice(kept)
        kept.append((key, _value(rng)))
    return kept


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8", errors="surrogatepass")
    else:
        path.write_bytes(content)


def write_batch(batch_dir: Path, rng: random.Random, now: float = _NOW) -> dict:
    """Write a batch of *rng*'s making to *batch_dir*; squeue's answers to
    its jobs (``{job: "fail" | "invalid" | output}``)."""
    root = batch_dir / "slurm"
    deep = rng.random() < 0.05
    count = rng.choice([0, 1, 2, 3, 5, 8])
    cases: list[str] = []
    for index in range(count):
        if rng.random() < 0.03:
            cases.append(rng.choice(["[]", '"case"', "3", "null", "[1, 2]"]))
            continue
        label = _value(rng, json.dumps(f"case {index}"), json.dumps(f"c{index}"))
        if deep and rng.random() < 0.5:
            label = nested(rng.choice([50, 400]))
        members = [
            ("index", _value(rng, str(index), str(index), json.dumps(str(index)))),
            ("label", label),
            ("stem", _value(rng, json.dumps(f"s{index}"))),
            ("case_overrides", _value(rng, "[]", json.dumps([f"x={index}"]))),
            ("config", '{"a": 1}'),
        ]
        cases.append(_object(_maybe(rng, members, 0.95)))
    cases_text = "[" + ", ".join(cases) + "]"
    if rng.random() < 0.04:
        cases_text = rng.choice(
            ["null", "0", "5", "true", '"ab"', "{}", '{"a": 1}', "[]"]
        )
    manifest = [
        ("version", str(MANIFEST_VERSION) if rng.random() < 0.98 else "9"),
        ("created", _value(rng, '"2026-11-12T00:00:00"')),
        ("batch_dir", json.dumps(str(batch_dir))),
        ("scenario_names", _value(rng, '["a", "b"]')),
        ("launcher", '{"cases_per_task": 1}' if rng.random() < 0.97 else '"x"'),
        ("cases", cases_text),
    ]
    manifest_text = _object(_maybe(rng, manifest, 0.99))
    if rng.random() < 0.03:
        manifest_text = rng.choice(_BROKEN).decode("utf-8", "replace")
    if rng.random() < 0.01:
        root.mkdir(parents=True, exist_ok=True)  # no manifest at all
    else:
        _write(root / "manifest.json", manifest_text)
    squeue: dict[str, str] = {}
    for sub_index in range(rng.choice([0, 1, 1, 2, 3])):
        path = root / "submissions" / f"{sub_index}.json"
        if rng.random() < 0.08:
            broken = rng.choice(_BROKEN_OR_DIR)
            if broken == "dir":
                path.mkdir(parents=True)
            else:
                _write(path, broken)
                age = rng.choice([5.0, SBATCH_IN_FLIGHT_SECONDS + 100.4])
                os.utime(path, (now - age, now - age))
            continue
        pool = list(range(count + 1))
        rng.shuffle(pool)
        pool = pool[: rng.randint(0, count + 1)]
        chunks: list[str] = []
        while pool:
            size = rng.randint(1, 3)
            chunk, pool = pool[:size], pool[size:]
            items = [
                str(c) if rng.random() < 0.95 else rng.choice(['"1"', "1.0", "true"])
                for c in chunk
            ]
            chunks.append("[" + ", ".join(items) + "]")
        chunks_text = "[" + ", ".join(chunks) + "]"
        if rng.random() < 0.06:
            chunks_text = rng.choice(
                [
                    "null",
                    "0",
                    "7",
                    "true",
                    '"ab"',
                    '{"a": [0]}',
                    "[3]",
                    '[[0], "x"]',
                    "[[0], null]",
                ]
            )
        job = _value(rng, '"1000"', "1001", '"1002"', "null")
        kind = rng.random()
        members = [
            ("index", str(sub_index)),
            ("created", '"2026-11-12T00:00:00"'),
            (
                "submitted_at",
                _value(
                    rng,
                    repr(now - 10),
                    repr(now - SBATCH_IN_FLIGHT_SECONDS - 50),
                    str(int(now)),
                ),
            ),
            ("cases", _value(rng, "[0, 1]", "[]")),
            ("chunks", chunks_text),
            ("launcher", "{}"),
            ("job_id", job),
            ("collect_job_id", _value(rng, "null", '"2000"', "2001")),
            ("sbatch", '["sbatch"]'),
        ]
        if kind < 0.2:
            members += [
                ("sbatch_timed_out", _value(rng, "true", "false")),
                ("sbatch_timed_out_at", _value(rng, '"2026-11-12T00:10:00"', "null")),
            ]
        elif kind < 0.5:
            members += [
                ("collect_sbatch", '["sbatch"]'),
                ("collect_error", _value(rng, "null", '"boom"')),
            ]
        if deep and rng.random() < 0.3:
            members.append(("deep", nested(rng.choice([50, 400]))))
        _write(path, _object(_maybe(rng, members, 0.95)))
        try:
            job_value = json.loads(job)
        except ValueError:
            job_value = None
        squeue[str(job_value)] = _squeue_answer(rng, str(job_value), len(chunks))
        if rng.random() < 0.2:
            squeue[str(job_value)] = rng.choice(["fail", "invalid"])
        for task in range(len(chunks) + 1):
            task_path = root / "tasks" / f"{sub_index}-{task}.json"
            if rng.random() < 0.2:
                continue
            if rng.random() < 0.06:
                broken = rng.choice(_BROKEN_OR_DIR)
                if broken == "dir":
                    task_path.mkdir(parents=True)
                else:
                    _write(task_path, broken)
                continue
            status = [
                ("submission", str(sub_index)),
                ("task", str(task)),
                ("host", _value(rng, '"node1"', '"n\\u00fc"', "null")),
                ("state", _value(rng, '"running"', '"done"', '"failed"', '"starting"')),
                ("done", _value(rng, "[0]", "[]", "[0, 1]")),
                ("current", _value(rng, "0", "1", "null")),
                ("ports", _value(rng, '{"rpc": 2000}', '{"rpc": null}', "{}")),
                ("error", _value(rng, '"boom"', "null")),
                ("peak_rss_mb", _value(rng, "12.5", "null", "0")),
            ]
            if deep and rng.random() < 0.3:
                status.append(("deep", nested(rng.choice([50, 400]))))
            _write(task_path, _object(_maybe(rng, status, 0.9)))
    for index in range(count):
        path = root / "results" / f"{index}.json"
        if rng.random() < 0.4:
            continue
        if rng.random() < 0.06:
            _write(path, rng.choice(_BROKEN))
            continue
        result = [
            ("passed", _value(rng, "true", "false")),
            ("message", _value(rng, '"ok"', '""')),
            ("elapsed_seconds", _value(rng, "1.5", "12")),
        ]
        result_text = _object(_maybe(rng, result, 0.95))
        if rng.random() < 0.05:
            result_text = _value(rng)
        record = [
            ("index", str(index)),
            ("result", result_text),
            ("errored", _value(rng, "false", "true")),
            ("host", _value(rng, '"node1"')),
            ("job_id", _value(rng, '"1000"', "1001")),
        ]
        _write(path, _object(_maybe(rng, record, 0.95)))
    return squeue


def _squeue_answer(rng: random.Random, job: str, tasks: int) -> str:
    states = [*sorted(ACTIVE_STATES), "COMPLETED", "FAILED", "TIMEOUT", "running"]
    lines = [
        f"{job}_{task} {rng.choice(states)}"
        for task in range(tasks)
        if rng.random() < 0.6
    ]
    if tasks > 1 and rng.random() < 0.1:
        lines.append(f"{job}_[0-{tasks - 1}] PENDING")
    return "".join(f"{line}\n" for line in lines)


# -- Running a reader ---------------------------------------------------------------


def _text(value: Any) -> str:
    """*value* as JSON text (repr for what JSON cannot write): NaN equal to
    NaN, -0.0 told from 0.0, 1 from 1.0 and from true."""
    try:
        return json.dumps(value, indent=1, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return repr(value)


def _outcome(call: Callable[[], Any]) -> tuple:
    try:
        return ("returns", _text(call()))
    except Exception as exc:  # noqa: BLE001
        return ("raises", type(exc).__name__, str(exc))


class _Seen:
    """What a reader returns or raises, prints, logs, cancels and writes."""

    def __init__(
        self,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
        env: dict[str, str],
        batch_dir: Path,
    ) -> None:
        self.capsys = capsys
        self.caplog = caplog
        self.scancel = Path(env["FAKE_SLURM_SCANCEL"])
        self.batch_dir = batch_dir

    def files(self) -> list[tuple[str, bytes | None]]:
        return sorted(
            (
                str(p.relative_to(self.batch_dir)),
                p.read_bytes() if p.is_file() else None,
            )
            for p in self.batch_dir.rglob("*")
        )

    def __call__(self, call: Callable[[], Any]) -> tuple:
        self.capsys.readouterr()
        self.caplog.clear()
        self.scancel.unlink(missing_ok=True)
        with self.caplog.at_level(logging.DEBUG):
            outcome = _outcome(call)
        captured = self.capsys.readouterr()
        cancelled = (
            self.scancel.read_text(encoding="utf-8") if self.scancel.exists() else ""
        )
        logs = [(r.name, r.levelno, r.getMessage()) for r in self.caplog.records]
        return outcome, captured.out, captured.err, logs, cancelled, self.files()


@pytest.fixture
def fake_slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    env = install_fake_slurm(tmp_path / "fake-slurm")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(time, "time", lambda: _NOW)
    return env


def _readers(batch_dir: Path) -> dict[str, tuple[Callable[[], Any], Callable[[], Any]]]:
    """Each reader of the batch, as (the reader, its oracle)."""
    layout = BatchLayout(batch_dir)
    batch = str(batch_dir)
    count = len(layout.submissions())
    readers: dict[str, tuple[Callable[[], Any], Callable[[], Any]]] = {
        "status": (
            lambda: cli.main(["status", batch]),
            lambda: _old_main(lambda: _old_status(BatchLayout(batch_dir.resolve()))),
        ),
        "cancel": (
            lambda: cli.main(["cancel", batch]),
            lambda: _old_main(lambda: _old_cancel(BatchLayout(batch_dir.resolve()))),
        ),
        "resubmit-dry-run": (
            lambda: cli.main(["resubmit", batch, "--dry-run"]),
            lambda: _old_main(
                lambda: _old_resubmit_dry_run(
                    BatchLayout(batch_dir.resolve()), False, False
                )
            ),
        ),
        "resubmit-failed-forced-dry-run": (
            lambda: cli.main(["resubmit", batch, "--dry-run", "--failed", "--force"]),
            lambda: _old_main(
                lambda: _old_resubmit_dry_run(
                    BatchLayout(batch_dir.resolve()), True, True
                )
            ),
        ),
        "active-jobs": (
            lambda: cli._active_jobs(layout),
            lambda: _old_active_jobs(layout),
        ),
        "busy": (lambda: cli._busy(layout), lambda: _old_busy(layout)),
        "batch-status": (
            lambda: slurm_launch.batch_status(batch_dir),
            lambda: _old_batch_status(batch_dir),
        ),
    }
    for below in range(count + 1):
        readers[f"busy-below-{below}"] = (
            partial(cli._busy, layout, below=below),
            partial(_old_busy, layout, below=below),
        )
    return readers


def _compare(
    batch_dir: Path,
    seen: _Seen,
) -> None:
    for name, (reader, oracle) in _readers(batch_dir).items():
        expected = seen(oracle)
        actual = seen(reader)
        assert actual == expected, name


@pytest.mark.parametrize("seed", range(30))
def test_every_reader_matches_the_oracle(
    tmp_path: Path,
    fake_slurm: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    seed: int,
) -> None:
    rng = random.Random(seed)
    for number in range(8):
        batch_dir = tmp_path / f"batch-{number}"
        squeue = write_batch(batch_dir, rng)
        Path(fake_slurm["FAKE_SLURM_SQUEUE"]).write_text(
            json.dumps(squeue), encoding="utf-8"
        )
        _compare(batch_dir, _Seen(capsys, caplog, fake_slurm, batch_dir))


def test_the_corpus_covers_what_the_readers_tell_apart(
    tmp_path: Path,
    fake_slurm: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The corpus reaches every branch worth pinning: each exit status, the
    errors raised, every note and every case state."""
    outcomes: set[str] = set()
    texts: list[str] = []
    for seed in range(30):
        rng = random.Random(seed)
        for number in range(8):
            batch_dir = tmp_path / f"s{seed}" / f"batch-{number}"
            squeue = write_batch(batch_dir, rng)
            Path(fake_slurm["FAKE_SLURM_SQUEUE"]).write_text(
                json.dumps(squeue), encoding="utf-8"
            )
            seen = _Seen(capsys, caplog, fake_slurm, batch_dir)
            for name, (_, oracle) in _readers(batch_dir).items():
                outcome, out, err, logs, cancelled, _ = seen(oracle)
                outcomes.add(f"{name}:{outcome[0]}:{outcome[1]}")
                texts.append(
                    str(outcome[1])
                    + out
                    + err
                    + "".join(m for _, _, m in logs)
                    + cancelled
                )
    joined = "\n".join(texts)
    for phrase in (
        "in the queue: ended",
        "in the queue: squeue unavailable",
        "RUNNING",
        "no job id yet (sbatch in flight)",
        "no job id: sbatch timed out",
        "(a stale record)",
        "unreadable record",
        "Skipping",
        "Cancelling job(s)",
        "The batch has no job to cancel.",
        "still queued or running",
        "are being submitted",
        "timed out, and SLURM may have taken",
        "are unreadable",
        "squeue could not tell",
        "was last modified",
        "Nothing to resubmit",
        "Resubmitting",
        "Error: ",
        " port 2000",
        "peak RSS",
        " -- boom",
        '"running"',
        '"pending"',
        '"missing"',
        '"unknown"',
        '"passed"',
        '"failed"',
        '"errored"',
    ):
        assert phrase in joined, phrase
    raised = {o.split(":")[2] for o in outcomes if o.split(":")[1] == "raises"}
    assert {"TypeError", "AttributeError", "KeyError"} <= raised, raised


@pytest.mark.parametrize("where", ["label", "submission", "task", "result"])
@pytest.mark.parametrize("depth", [800, 5000, 100_000])
def test_a_batch_nested_deep(
    tmp_path: Path,
    fake_slurm: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    where: str,
    depth: int,
) -> None:
    """A value nested deeper than ``json``'s encoder written in Python lays
    out is read wherever ``json.loads()`` reads it, and what it raises
    where it does not is raised alike."""
    batch_dir = tmp_path / "batch"
    squeue: dict[str, str] = {}
    root = batch_dir / "slurm"
    value = nested(depth)
    manifest = {
        "version": MANIFEST_VERSION,
        "launcher": {},
        "cases": [{"index": 0, "label": "a"}, {"index": 1, "label": "b"}],
    }
    text = json.dumps(manifest)
    if where == "label":
        text = text.replace('"label": "a"', f'"label": {value}')
    _write(root / "manifest.json", text)
    sub = '{"job_id": "1000", "chunks": [[0, 1]], "cases": [0, 1]'
    _write(
        root / "submissions" / "0.json",
        sub + (f', "deep": {value}' if where == "submission" else "") + "}",
    )
    _write(
        root / "tasks" / "0-0.json",
        '{"state": "running", "current": 1'
        + (f', "deep": {value}' if where == "task" else "")
        + "}",
    )
    _write(
        root / "results" / "0.json",
        '{"errored": false, "result": {"passed": true, "message": "ok", '
        '"elapsed_seconds": 1.0'
        + (f', "deep": {value}' if where == "result" else "")
        + "}}",
    )
    squeue["1000"] = "1000_0 RUNNING\n"
    Path(fake_slurm["FAKE_SLURM_SQUEUE"]).write_text(
        json.dumps(squeue), encoding="utf-8"
    )
    _compare(batch_dir, _Seen(capsys, caplog, fake_slurm, batch_dir))


def test_status_of_a_batch_as_it_reads(
    tmp_path: Path,
    fake_slurm: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``scenario-slurm status`` of a batch midway, word for word."""
    batch_dir = tmp_path / "batch"
    root = batch_dir / "slurm"
    manifest = {
        "version": MANIFEST_VERSION,
        "launcher": {},
        "cases": [{"index": i, "label": f"c{i}"} for i in range(4)],
    }
    _write(root / "manifest.json", json.dumps(manifest))
    _write(
        root / "submissions" / "0.json",
        json.dumps(
            {"job_id": "1000", "chunks": [[0, 1], [2, 3]], "cases": [0, 1, 2, 3]}
        ),
    )
    _write(
        root / "submissions" / "1.json",
        json.dumps({"job_id": None, "chunks": [[3]], "submitted_at": _NOW - 5}),
    )
    _write(root / "submissions" / "2.json", "{")
    _write(
        root / "tasks" / "0-0.json",
        json.dumps(
            {
                "state": "running",
                "host": "node1",
                "ports": {"rpc": 2000},
                "done": [0],
                "current": 1,
                "peak_rss_mb": 12.5,
            }
        ),
    )
    _write(
        root / "tasks" / "0-1.json", json.dumps({"state": "failed", "error": "boom"})
    )
    _write(
        root / "results" / "0.json",
        json.dumps(
            {
                "errored": False,
                "result": {"passed": True, "message": "ok", "elapsed_seconds": 1.0},
            }
        ),
    )
    Path(fake_slurm["FAKE_SLURM_SQUEUE"]).write_text(
        json.dumps({"1000": "1000_0 RUNNING\n"}), encoding="utf-8"
    )
    assert cli.main(["status", str(batch_dir)]) == 0
    resolved = batch_dir.resolve()
    assert capsys.readouterr().out == (
        f"Batch {resolved}: 4 case(s)\n"
        "  1 with a result (1 passed, 0 errored), 3 without\n"
        "  submission 0: job 1000, 2 task(s), 4 case(s); in the queue: 0:RUNNING\n"
        "    task 0: running on node1 port 2000, 1/2 case(s), peak RSS 12.5 MB\n"
        "    task 1: failed, 0/2 case(s) -- boom\n"
        "  submission 1: 1 task(s), 0 case(s); no job id yet (sbatch in flight)\n"
        "    task 0: not started, 0/1 case(s)\n"
        f"  submission 2: unreadable record {resolved}/slurm/submissions/2.json "
        "(empty, not JSON, or not a JSON object)\n"
    )
    assert cli.main(["cancel", str(batch_dir)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Cancelling job(s) 1000\n"
    assert captured.err == (
        f"Skipping {resolved}/slurm/submissions/2.json: it cannot be read as a "
        "submission record; if it named a job, cancel that with scancel\n"
    )
    assert Path(fake_slurm["FAKE_SLURM_SCANCEL"]).read_text() == '["1000"]\n'
    assert cli.main(["resubmit", str(batch_dir), "--dry-run"]) == 1
    assert capsys.readouterr().err == (
        "Error: job(s) 1000 of this batch are still queued or running; wait for "
        "them (scenario-slurm wait), cancel them (scenario-slurm cancel), or "
        "pass --force\n"
    )
    batch = slurm_launch.batch_status(batch_dir)
    assert batch is not None
    assert [(c["state"], c["message"]) for c in batch["cases"]] == [
        ("passed", "ok"),
        ("running", "task 0 of job 1000: running"),
        ("missing", "no result: task 1 of job 1000 failed: boom"),
        (
            "pending",
            "no job id yet: submission 1 is being submitted (sbatch in flight)",
        ),
    ]
    assert list(batch["cases"][0]) == [
        "index",
        "label",
        "stem",
        "overrides",
        "state",
        "message",
        "elapsed_seconds",
        "host",
        "job_id",
    ]
    assert list(batch["cases"][1]) == [
        "index",
        "label",
        "stem",
        "overrides",
        "state",
        "message",
    ]


def test_the_fake_squeue_answers_as_squeue(fake_slurm: dict[str, str]) -> None:
    Path(fake_slurm["FAKE_SLURM_SQUEUE"]).write_text(
        json.dumps({"1": "1_0 RUNNING\n1_[1-2] PENDING\n", "2": "fail"}),
        encoding="utf-8",
    )
    assert queue_mod.squeue_states("1") == {0: "RUNNING", 1: "PENDING", 2: "PENDING"}
    assert queue_mod.squeue_states("2") is None
    assert queue_mod.squeue_states("3") == {}
    assert _cancel_jobs(["1", "2"]) is True
    assert Path(fake_slurm["FAKE_SLURM_SCANCEL"]).read_text() == '["1", "2"]\n'
