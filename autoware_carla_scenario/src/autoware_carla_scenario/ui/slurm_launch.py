"""Running a batch from the viewer through SLURM (``launcher=slurm``,
docs/slurm.md), and following it.

The viewer submits the batch as ``scenario`` itself would be asked to on a
submit host -- ``scenario [--multirun] scenario=... launcher=slurm
launcher.<key>=... --no-wait`` -- so the batch is expanded, checked and laid
out exactly as one submitted from a shell, and the viewer's process does not
wait for it: a collect job (``launcher.collect_job``) writes the batch's
summary once the array ends. Everything after that is read from the batch's
directory (its ``slurm/`` manifest, submissions, task and case records) and
from ``squeue``, so a batch submitted from a shell is followed the same way,
and the viewer can be restarted while one runs. Resubmitting, collecting and
cancelling are ``scenario-slurm``'s commands, run on the batch's directory.

Nothing here works without SLURM's ``sbatch`` on ``PATH``; :func:`availability`
says why, and the viewer offers the local launcher only.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..slurm.manifest import (
    BatchLayout,
    Manifest,
    SubmissionRecord,
    TaskStatus,
    no_job_note,
    recorded_verdict,
    sbatch_in_flight,
    sbatch_timed_out,
)
from ..slurm.queue import ACTIVE_STATES, squeue_states
from ..slurm.records import JsonMembers
from .case_paths import session_order
from .slurm_case_state import case_state, no_job_case_state

# SLURM's command-line tools are run in the checked slurm.queue (#1634):
# whether SLURM is here, and its partitions, are re-exported from there.
from ..slurm.queue import availability as availability
from ..slurm.queue import partitions as partitions

# Checked by the library check, and re-exported: what each field may hold, the
# overrides the options become and how the command line is put together.
from .slurm_options import (  # noqa: F401
    _BEFORE_QUOTE,
    _ENV_NAME,
    _NODE_PATH,
    _NOT_A_STRING,
    _SIMPLE,
    _TIME,
    _quote,
    _quote_always,
    check_at_least,
    check_env,
    check_node_path,
    check_sbatch_value,
    check_setup_line,
    check_time,
)

# The options and the command line, imported by name: Codon imports no module
# as a value (docs/typecheck.md).
from .slurm_options import SlurmOptions as _SlurmOptions
from .slurm_options import build_command as _build_command

__all__ = [
    "SlurmOptions",
    "Submission",
    "availability",
    "batch_status",
    "build_command",
    "partitions",
    "run_cli",
    "run_collect",
    "scenario_command",
    "start_submit",
    "submission",
]

#: The command a run starts with (``uv run scenario``, as the local launcher's),
#: or ``VIEWER_SCENARIO_COMMAND`` (``.venv/bin/scenario``, for a viewer that
#: must not have uv sync an environment).
_ENV_COMMAND = "VIEWER_SCENARIO_COMMAND"

#: The batch directory ``slurm/submit.py`` prints at the end of its line
#: (``...; outputs in <batch_dir>``): the rest of the line, as the directory
#: may contain spaces.
_OUTPUTS_IN = re.compile(r"outputs in (?P<dir>.+?)\s*$", re.MULTILINE)


def scenario_command() -> list[str]:
    """``scenario``, as this viewer starts it."""
    configured = os.environ.get(_ENV_COMMAND, "").strip()
    return shlex.split(configured) if configured else ["uv", "run", "scenario"]


#: The run of backslashes that ends a value, which ui.slurm_options doubles
#: when it quotes one (its tests compare that with this pattern's ``sub``).
_AT_END = re.compile(r"(\\+)\Z")


class SlurmOptions(_SlurmOptions):
    """The ``launcher.*`` keys the viewer sets (the rest are the config's):
    ui.slurm_options' :class:`~autoware_carla_scenario.ui.slurm_options.SlurmOptions`,
    read off the run panel's fields."""

    @classmethod
    def from_request(cls, data: dict[str, Any] | None) -> SlurmOptions:
        """From the run panel's fields.

        Raises:
            ValueError: A field is not what ``launcher.<key>`` takes.
        """
        data = data or {}

        def text(key: str) -> str | None:
            value = str(data.get(key) or "").strip()
            return value or None

        def number(key: str, kind: type) -> Any:
            value = text(key)
            if value is None:
                return None
            try:
                parsed = kind(value)
            except ValueError as exc:
                raise ValueError(f"{key} must be a number, not {value!r}") from exc
            check_at_least(key, parsed, 0 if kind is float else 1)
            return parsed

        env: dict[str, str] = {}
        raw_env = data.get("env") or {}
        if isinstance(raw_env, str):
            lines = [line.strip() for line in raw_env.splitlines() if line.strip()]
            raw_env = dict(line.split("=", 1) for line in lines if "=" in line)
            if any("=" not in line for line in lines):
                raise ValueError("each extra env line is NAME=value")
        for name, value in dict(raw_env).items():
            name = str(name).strip()
            check_env(name, str(value), repr(value))
            env[name] = str(value)
        time = text("time")
        check_time(time)
        for key in ("partition", "mem", "gres"):
            check_sbatch_value(key, text(key))
        for key in ("remote_cwd", "python"):
            check_node_path(key, text(key))
        raw_setup = data.get("setup") or []
        if isinstance(raw_setup, str):
            raw_setup = raw_setup.splitlines()
        setup = [str(line).strip() for line in raw_setup if str(line).strip()]
        for line in setup:
            check_setup_line(line)
        return cls(
            partition=text("partition"),
            array_parallelism=number("array_parallelism", int),
            cases_per_task=number("cases_per_task", int),
            time=time,
            mem=text("mem"),
            gres=text("gres"),
            start_stagger_seconds=number("start_stagger_seconds", float),
            remote_cwd=text("remote_cwd"),
            python=text("python"),
            setup=setup,
            env=env,
        )


def build_command(
    scenario: str,
    extra_overrides: list[str],
    sweeper: str,
    options: _SlurmOptions,
) -> list[str]:
    """The ``scenario`` command that submits the batch and exits (``--no-wait``),
    started as :func:`scenario_command` says.

    With a sweeper the batch runs every case of each scenario's sweep
    (``--multirun``, expanded by the batch itself); without one each scenario
    the pattern matches runs once.
    """
    return _build_command(
        scenario_command(), scenario, extra_overrides, sweeper, options
    )


# ── The submission the viewer started ──────────────────────────────────────


def _now_iso() -> str:
    """Now, in UTC to the second, as ISO 8601 (``2026-01-01T00:00:00+00:00``):
    when a :class:`Submission` started."""
    if TYPE_CHECKING:
        # Codon's datetime has no timezone and no isoformat(): the check
        # compiles a str in its place (docs/typecheck.md).
        return ""
    else:
        from datetime import datetime, timezone  # noqa: PLC0415

        return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Submission:
    """The last batch the viewer submitted: ``submitting``, then
    ``submitted`` (with its directory) or ``failed`` (with the output)."""

    command: list[str]
    state: str = "submitting"
    batch_dir: str | None = None
    output: str = ""
    returncode: int | None = None
    started: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": " ".join(shlex.quote(c) for c in self.command),
            "state": self.state,
            "batch_dir": self.batch_dir,
            "output": self.output[-8000:],
            "returncode": self.returncode,
            "started": self.started,
        }


_lock = threading.Lock()
_submission: Submission | None = None


def submission() -> Submission | None:
    with _lock:
        return _submission


def _new_manifests(base: Path, before: set[Path]) -> list[Path]:
    """The batch manifests written since *before*, oldest session first
    (``outputs/{date}/{time}/slurm/manifest.json``; ``-10`` after ``-9``,
    #1845)."""
    return sorted(
        set((base / "outputs").glob("*/*/slurm/manifest.json")) - before,
        key=lambda p: (p.parents[2].name, *session_order(p.parents[1].name)),
    )


#: How long the ``scenario ... --no-wait`` a submit runs may take (#1786).
#: It expands and checks the batch before it calls sbatch -- the Codon check
#: alone may take up to ``typecheck.check.DEFAULT_TIMEOUT_SECONDS`` (600 s),
#: and ``uv run`` may sync the environment first -- so it is given minutes;
#: past them the submit is recorded as failed (:data:`TIMED_OUT`), and the
#: viewer takes a new one.
SUBMIT_TIMEOUT_SECONDS = 1800.0


def _kill_group(process: subprocess.Popen[str]) -> None:
    """SIGKILL *process*'s group (its own session's, ``start_new_session``):
    the ``scenario`` it ran, the ``uv run`` in front of it and the sbatch or
    checks it started."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # Every process of the group has ended already.


def _joined(stdout: str | None, stderr: str | None) -> str:
    """What a process wrote: *stdout* and *stderr*, those that are not empty,
    one after the other on a line of its own."""
    # A list of the two, not a generator of the Optional[str]s: Codon joins
    # only str (docs/typecheck.md).
    written: list[str] = []
    if stdout:
        written.append(stdout)
    if stderr:
        written.append(stderr)
    return "\n".join(written)


def _run_submit(command: list[str], base: Path) -> tuple[int | None, str]:
    """Run the submitting *command* in *base*, as :func:`run_cli` runs
    ``scenario-slurm``: its exit status and output; :data:`TIMED_OUT`, with
    why, when it did not end within :data:`SUBMIT_TIMEOUT_SECONDS`, its whole
    process group killed; ``None`` when it could not be started."""
    try:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=str(base),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:  # ValueError: a NUL in an argument
        return None, f"Could not run {command[0]}: {exc}"
    timeout = SUBMIT_TIMEOUT_SECONDS
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        # Bounded: what the killed group wrote, once its pipes close.
        try:
            stdout, stderr = process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            # A process that left the group holds the pipes: reap ours.
            process.wait()
            stdout, stderr = "", ""
        output = _joined(stdout, stderr)
        reason = (
            f"{command[0]} did not end within {timeout:g} s and was stopped: "
            "the batch's checks or sbatch hung (is the SLURM controller "
            "answering?). Stopped during sbatch, it keeps the batch's "
            "submission record, as the controller may have taken the job: if "
            "`squeue --me` lists it, it runs and the batch can be collected "
            "once it ends (or scancel it), and resubmits of that batch are "
            "refused until 600 s after the record was made. A new submit (a "
            "new batch) is taken now."
        )
        # Typed as the function returns it: Codon's tuples do not widen.
        timed_out: int | None = TIMED_OUT
        return timed_out, f"{output}\n{reason}" if output else reason
    except BaseException:
        _kill_group(process)
        process.wait()
        raise
    return process.returncode, _joined(stdout, stderr)


def _submit(command: list[str], base: Path, record: Submission) -> None:
    before = set((base / "outputs").glob("*/*/slurm/manifest.json"))
    returncode, output = _run_submit(command, base)
    # The directory the last match names, read at each match: Codon refuses
    # an Optional variable as a loop's variable, which is a Match.
    batch_dir: str | None = None
    for match in _OUTPUTS_IN.finditer(output):
        batch_dir = match["dir"]
    if batch_dir is None:
        found = _new_manifests(base, before)
        batch_dir = str(found[-1].parent.parent) if found else None
    with _lock:
        record.output = output
        record.returncode = returncode
        record.batch_dir = batch_dir
        record.state = "submitted" if returncode == 0 and batch_dir else "failed"


def start_submit(command: list[str], base: Path) -> Submission:
    """Submit in a background thread; the record it fills in.

    Raises:
        RuntimeError: A submission is still being made.
    """
    global _submission  # noqa: PLW0603
    with _lock:
        if _submission is not None and _submission.state == "submitting":
            raise RuntimeError("A batch is still being submitted")
        record = Submission(command=list(command))
        _submission = record
    threading.Thread(target=_submit, args=(command, base, record), daemon=True).start()
    return record


# ── Following a batch ──────────────────────────────────────────────────────

#: ACTIVE_STATES as the typed tuple ui.slurm_case_state takes.
_ACTIVE_STATES: tuple[str, ...] = tuple(sorted(ACTIVE_STATES))


def _member_json(members: JsonMembers, key: str, default: str = "null") -> str:
    """The JSON text of the member *key* of *members*, *default* when there
    is none: what ``dict.get(key, default)`` gave of the object."""
    return members.value_text(key) if members.has(key) else default


@dataclass
class CaseState:
    """A case of a batch as the viewer shows it (:func:`_case_states`): its
    ``index``; its ``label``, ``stem`` and ``case_overrides`` as the manifest
    holds them (``[]`` for false overrides), each as its JSON text; its
    ``state`` (ui.slurm_case_state's); and the ``message`` that says why,
    as its JSON text.  A case whose outcome is recorded also shows its
    result's ``elapsed_seconds`` and the ``host`` and ``job_id`` that ran
    it, as the record holds them, and its result's ``message`` (``""``
    when it has none).

    The values are held as JSON text, written with the C encoder as
    :class:`~autoware_carla_scenario.slurm.records.JsonMembers` holds a
    value, so that :func:`batch_status` gives the viewer's API the very
    values ``json.loads()`` read of the batch's files, of whatever type they
    are.
    """

    index: int
    label_json: str
    stem_json: str
    overrides_json: str
    state: str
    message_json: str
    recorded: bool = False
    elapsed_seconds_json: str = "null"
    host_json: str = "null"
    job_id_json: str = "null"


def _case_states(
    layout: BatchLayout,
    manifest: Manifest,
    squeue: dict[int, dict[int, str] | None],
) -> list[CaseState]:
    """Each case: its record's outcome, or where its task is."""
    # The last submission (and task) each case was given to.
    given: dict[int, tuple[int, int, SubmissionRecord]] = {}
    for index in layout.submissions():
        read = layout.submission_record(index)
        if read is None:
            continue  # batch_status notes it; it gave no case (#1356)
        # Typed through a local: Codon 0.19 keeps the Optional of the read
        # record in the tuple.
        submitted: SubmissionRecord = read
        for task, chunk in enumerate(submitted.chunk_list()):
            for item in chunk:
                given[int(item)] = (index, task, submitted)
    out: list[CaseState] = []
    for case in manifest.listed_cases():
        index = case.index()
        record = layout.read_result(index)
        label_json = case.value_json("label")
        stem_json = case.value_json("stem")
        overrides_json = case.value_json("case_overrides")
        if not bool(json.loads(overrides_json)):
            overrides_json = "[]"
        # A result that is no object reads as an empty one, and only a
        # boolean ``passed`` as a verdict (#1718).
        result = JsonMembers()
        errored = passed = False
        if record is not None:
            # Annotated: Codon 0.19 refuses the call's value for the local
            # without.
            recorded: JsonMembers | None = record.result()
            if recorded is not None:
                result = recorded
            errored = record.errored
            if not errored:
                verdict = recorded_verdict(record)
                passed = verdict is not None and verdict
        queued: str | None = None
        squeue_known = at_case = False
        task, job_id, task_state = 0, "", ""
        task_error: str | None = None
        no_job: tuple[str, str] | None = None
        if record is None and index in given and not given[index][2].has_job:
            # No job to ask squeue about, nor to name: never "job None" and
            # never missing while its sbatch is in flight or timed out (#1791).
            sub_index, _task, sub = given[index]
            at = sub.timed_out_at_text()
            no_job = no_job_case_state(
                sub_index,
                sbatch_timed_out(sub),
                sbatch_in_flight(sub),
                "" if at is None else at,
            )
        elif record is None and index in given:
            sub_index, task, sub = given[index]
            # squeue.get(sub_index), spelled out: Codon's get() of a dict of
            # Optional values gives an Optional of an Optional.
            states: dict[int, str] | None = None
            if sub_index in squeue:
                states = squeue[sub_index]
            squeue_known = states is not None
            queued = None if states is None else states.get(task)
            task_read = layout.read_task(sub_index, task)
            status = TaskStatus(JsonMembers()) if task_read is None else task_read
            job_id = sub.job_text()
            # Only what case_state's branch reads of the status; a task file
            # that holds no JSON object reads as no status at all (#1630).
            if queued is not None and queued in ACTIVE_STATES:
                at_case = queued == "RUNNING" and status.is_at(index)
            elif squeue_known:
                task_state = status.state_text("never started")
                task_error = status.error
        if no_job is not None:
            state, message = no_job
        else:
            state, message = case_state(
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
                _ACTIVE_STATES,
            )
        if record is not None:
            out.append(
                CaseState(
                    index,
                    label_json,
                    stem_json,
                    overrides_json,
                    state,
                    _member_json(result, "message", '""'),
                    True,
                    _member_json(result, "elapsed_seconds"),
                    _member_json(record.members, "host"),
                    _member_json(record.members, "job_id"),
                )
            )
        else:
            out.append(
                CaseState(
                    index,
                    label_json,
                    stem_json,
                    overrides_json,
                    state,
                    json.dumps(message, ensure_ascii=False),
                )
            )
    return out


def batch_status(batch_dir: Path) -> dict[str, Any] | None:
    """A SLURM batch's progress, or ``None`` when *batch_dir* holds none:
    its cases (``passed``, ``failed``, ``errored``, ``running``, ``pending``,
    ``missing`` or ``unknown``), its submissions with their jobs' and tasks'
    states (``squeue``), and whether it has been collected.  A submission
    whose record cannot be read is listed ``unreadable`` with a ``note``, as
    is one with no job id, which says why (``in_flight`` or
    ``sbatch_timed_out``, else a stale record; #1791)."""
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
            if path.exists():  # else removed since: its sbatch failed
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
        states = squeue_states(str(job)) if job else {}
        squeue[index] = states
        # Its sbatch has not returned yet; one that timed out has (#1791).
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
                # One being submitted is followed as a queued one is.
                "active": in_flight
                or (
                    states is not None
                    and any(s in ACTIVE_STATES for s in states.values())
                ),
                "squeue_unavailable": states is None,
                "tasks": tasks,
                "unreadable": False,
                # Why it has no job id: in flight, timed out or stale (#1791).
                "note": None if job else no_job_note(sub),
                "sbatch_timed_out": bool(timed_out),
                "in_flight": bool(in_flight),
            }
        )

    def case_entry(case: CaseState) -> dict[str, Any]:
        """The case as the viewer's API sends it, its keys in their order."""
        entry: dict[str, Any] = {
            "index": case.index,
            "label": json.loads(case.label_json),
            "stem": json.loads(case.stem_json),
            "overrides": json.loads(case.overrides_json),
            "state": case.state,
            "message": json.loads(case.message_json),
        }
        if case.recorded:
            entry["elapsed_seconds"] = json.loads(case.elapsed_seconds_json)
            entry["host"] = json.loads(case.host_json)
            entry["job_id"] = json.loads(case.job_id_json)
        return entry

    case_states = _case_states(layout, manifest, squeue)
    counts: dict[str, int] = {}
    for entry in case_states:
        counts[entry.state] = counts.get(entry.state, 0) + 1
    cases = [case_entry(entry) for entry in case_states]
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


#: What :func:`run_cli` returns when ``scenario-slurm`` could not be started
#: (an ``OSError``) or did not end in time, as a shell does: never 0 or 1, so
#: no action reads either as done, nor collect as "collected, a case failed"
#: (#1746).
NOT_STARTED = 127
TIMED_OUT = 124


def run_cli(args: list[str], timeout: float = 300.0) -> tuple[int, str]:
    """``scenario-slurm <args>``, run by this viewer's interpreter: its exit
    status and output; :data:`NOT_STARTED` or :data:`TIMED_OUT`, with why,
    when it could not be started or did not end within *timeout* seconds."""
    command = [sys.executable, "-m", "autoware_carla_scenario.slurm.cli", *args]
    try:
        done = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return TIMED_OUT, str(exc)
    except OSError as exc:
        return NOT_STARTED, str(exc)
    return done.returncode, _joined(done.stdout, done.stderr)


def _stat_key(path: Path) -> tuple[int, int, int] | None:
    """What tells one ``collected.json`` from the next: ``write_json`` renames
    a new file over the old one, so its inode changes on every write."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


def run_collect(batch_dir: Path, timeout: float = 300.0) -> tuple[bool, int, str]:
    """``scenario-slurm collect <batch_dir>``: whether it collected the batch,
    its exit status and output.

    Collected is exit 0 (every case passed), or exit 1 with
    ``slurm/collected.json`` written by this run: collect writes it last, so
    it is there only once the summary and ``batch_results.json`` are (#1746).
    Exit 1 alone is no evidence: ``scenario-slurm`` also exits 1 on an
    ``Error:`` (an unreadable manifest, an unwritable batch directory) and on
    a traceback.
    """
    collected = BatchLayout(batch_dir).root / "collected.json"
    before = _stat_key(collected)
    code, output = run_cli(["collect", str(batch_dir)], timeout)
    after = _stat_key(collected)
    ok = code == 0 or (code == 1 and after is not None and after != before)
    return ok, code, output
