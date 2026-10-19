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

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..slurm.manifest import BatchLayout
from ..slurm.submit import ACTIVE_STATES, squeue_states

__all__ = [
    "SlurmOptions",
    "Submission",
    "availability",
    "batch_status",
    "build_command",
    "partitions",
    "run_cli",
    "scenario_command",
    "start_submit",
    "submission",
]

#: The command a run starts with (``uv run scenario``, as the local launcher's),
#: or ``VIEWER_SCENARIO_COMMAND`` (``.venv/bin/scenario``, for a viewer that
#: must not have uv sync an environment).
_ENV_COMMAND = "VIEWER_SCENARIO_COMMAND"

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TIME = re.compile(r"^(\d+-)?\d{1,2}(:\d{2}){0,2}$|^\d+$")
_SIMPLE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")
#: A path on the nodes (``launcher.remote_cwd``, ``launcher.python``).
_NODE_PATH = re.compile(r"^~?[A-Za-z0-9_.:/@+-]*$")
_OUTPUTS_IN = re.compile(r"outputs in (?P<dir>\S+)")


def scenario_command() -> list[str]:
    """``scenario``, as this viewer starts it."""
    configured = os.environ.get(_ENV_COMMAND, "").strip()
    return shlex.split(configured) if configured else ["uv", "run", "scenario"]


def availability() -> tuple[bool, str]:
    """Whether batches can be submitted from here, and if not, why."""
    missing = [tool for tool in ("sbatch", "squeue") if shutil.which(tool) is None]
    if missing:
        return False, (
            f"SLURM's {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
            "not on this viewer's PATH: run the viewer on a SLURM submit (login) "
            "host to launch batches through SLURM (docs/slurm.md)."
        )
    return True, ""


def partitions() -> list[str]:
    """The partitions ``sinfo`` lists (the default one first), if it can."""
    try:
        done = subprocess.run(
            ["sinfo", "-h", "-o", "%P"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    names = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    default = [n.rstrip("*") for n in names if n.endswith("*")]
    return default + sorted({n.rstrip("*") for n in names} - set(default))


def _quote(value: str) -> str:
    """*value* as a Hydra override's value: as it is when plain, else quoted."""
    if _SIMPLE.match(value):
        return value
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


@dataclass
class SlurmOptions:
    """The ``launcher.*`` keys the viewer sets (the rest are the config's)."""

    partition: str | None = None
    array_parallelism: int | None = None
    cases_per_task: int | None = None
    time: str | None = None
    mem: str | None = None
    gres: str | None = None
    start_stagger_seconds: float | None = None
    #: Where the jobs work on the nodes, for a viewer whose host sees their
    #: file system at another path (docs/slurm.md).
    remote_cwd: str | None = None
    python: str | None = None
    #: ``launcher.setup``: shell lines a job runs first; empty: the config's.
    setup: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

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
            if parsed < (0 if kind is float else 1):
                raise ValueError(f"{key} must be at least {0 if kind is float else 1}")
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
            if not _ENV_NAME.match(name):
                raise ValueError(f"{name!r} is not an environment variable name")
            env[name] = str(value)
        time = text("time")
        if time is not None and not _TIME.match(time):
            raise ValueError(
                f"time is sbatch's --time ([D-]HH:MM:SS or minutes), not {time!r}"
            )
        for key in ("partition", "mem", "gres"):
            value = text(key)
            if value is not None and not _SIMPLE.match(value):
                raise ValueError(f"{key} {value!r} has characters sbatch does not take")
        for key in ("remote_cwd", "python"):
            value = text(key)
            if value is not None and (
                not _NODE_PATH.match(value)
                or (value.startswith("~") and value != "~" and value[1] != "/")
            ):
                raise ValueError(
                    f"{key} {value!r} is not a path the launcher takes (letters, "
                    "digits, _ . : / @ + -, and ~ or ~/ first)"
                )
        raw_setup = data.get("setup") or []
        if isinstance(raw_setup, str):
            raw_setup = raw_setup.splitlines()
        setup = [str(line).strip() for line in raw_setup if str(line).strip()]
        for line in setup:
            # A Hydra override's value, then an OmegaConf string: a backslash
            # or an interpolation would not reach the job as written.
            if "\\" in line or "${" in line:
                raise ValueError(
                    f"setup line {line!r}: write $NAME rather than ${{NAME}}, and "
                    "no backslash (or set launcher.setup in a config)"
                )
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

    def overrides(self) -> list[str]:
        """``launcher=slurm`` and the keys set, as Hydra overrides."""
        out = ["launcher=slurm"]
        for key in (
            "partition",
            "array_parallelism",
            "cases_per_task",
            "time",
            "mem",
            "gres",
            "start_stagger_seconds",
            "remote_cwd",
            "python",
        ):
            value = getattr(self, key)
            if value is not None:
                out.append(f"launcher.{key}={_quote(str(value))}")
        if self.setup:
            lines = ",".join(
                "'" + line.replace("'", "\\'") + "'" for line in self.setup
            )
            out.append(f"launcher.setup=[{lines}]")
        # `env` is an empty mapping in the config: a new key is added (++).
        out.extend(
            f"++launcher.env.{k}={_quote(v)}" for k, v in sorted(self.env.items())
        )
        return out


def build_command(
    scenario: str,
    extra_overrides: list[str],
    sweeper: str,
    options: SlurmOptions,
) -> list[str]:
    """The ``scenario`` command that submits the batch and exits (``--no-wait``).

    With a sweeper the batch runs every case of each scenario's sweep
    (``--multirun``, expanded by the batch itself); without one each scenario
    the pattern matches runs once.
    """
    cmd = scenario_command()
    if sweeper:
        cmd.append("--multirun")
    cmd.append(f"scenario={scenario}")
    cmd.extend(extra_overrides)
    cmd.extend(options.overrides())
    cmd.append("--no-wait")
    return cmd


# ── The submission the viewer started ──────────────────────────────────────


@dataclass
class Submission:
    """The last batch the viewer submitted: ``submitting``, then
    ``submitted`` (with its directory) or ``failed`` (with the output)."""

    command: list[str]
    state: str = "submitting"
    batch_dir: str | None = None
    output: str = ""
    returncode: int | None = None
    started: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

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
    return sorted(set((base / "outputs").glob("*/*/slurm/manifest.json")) - before)


def _submit(command: list[str], base: Path, record: Submission) -> None:
    before = set((base / "outputs").glob("*/*/slurm/manifest.json"))
    try:
        done = subprocess.run(  # noqa: S603
            command,
            cwd=str(base),
            capture_output=True,
            text=True,
            check=False,
        )
        output = "\n".join(s for s in (done.stdout, done.stderr) if s)
        returncode = done.returncode
    except OSError as exc:
        output, returncode = f"Could not run {command[0]}: {exc}", None
    match = None
    for match in _OUTPUTS_IN.finditer(output):
        pass
    batch_dir: str | None = match["dir"] if match else None
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


def _case_states(
    layout: BatchLayout,
    manifest: dict[str, Any],
    squeue: dict[int, dict[int, str] | None],
) -> list[dict[str, Any]]:
    """Each case: its record's outcome, or where its task is."""
    # The last submission (and task) each case was given to.
    given: dict[int, tuple[int, int, dict[str, Any]]] = {}
    for index in layout.submissions():
        sub = layout.read_submission(index)
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
        if record is not None:
            result = record.get("result") or {}
            entry.update(
                state="errored"
                if record.get("errored")
                else ("passed" if result.get("passed") else "failed"),
                message=result.get("message", ""),
                elapsed_seconds=result.get("elapsed_seconds"),
                host=record.get("host"),
                job_id=record.get("job_id"),
            )
        elif index not in given:
            entry.update(state="missing", message="never submitted")
        else:
            sub_index, task, sub = given[index]
            states = squeue.get(sub_index)
            queued = None if states is None else states.get(task)
            status = layout.read_task(sub_index, task) or {}
            if queued in ACTIVE_STATES:
                running = queued == "RUNNING" and status.get("current") == index
                entry.update(
                    state="running" if running else "pending",
                    message=f"task {task} of job {sub.get('job_id')}: {queued.lower()}",
                )
            elif states is None:
                entry.update(state="unknown", message="squeue could not be asked")
            else:
                why = status.get("state", "never started")
                if status.get("error"):
                    why += f": {status['error']}"
                entry.update(
                    state="missing",
                    message=f"no result: task {task} of job {sub.get('job_id')} {why}",
                )
        out.append(entry)
    return out


def batch_status(batch_dir: Path) -> dict[str, Any] | None:
    """A SLURM batch's progress, or ``None`` when *batch_dir* holds none:
    its cases (``passed``, ``failed``, ``errored``, ``running``, ``pending``,
    ``missing`` or ``unknown``), its submissions with their jobs' and tasks'
    states (``squeue``), and whether it has been collected."""
    layout = BatchLayout(batch_dir)
    try:
        manifest = layout.read_manifest()
    except (FileNotFoundError, ValueError, OSError):
        return None
    squeue: dict[int, dict[int, str] | None] = {}
    submissions = []
    for index in layout.submissions():
        sub = layout.read_submission(index)
        job = sub.get("job_id")
        states = squeue_states(str(job)) if job else {}
        squeue[index] = states
        tasks = []
        for task, chunk in enumerate(sub.get("chunks") or []):
            status = layout.read_task(index, task) or {}
            tasks.append(
                {
                    "task": task,
                    "cases": chunk,
                    "squeue": None if states is None else states.get(task),
                    "state": status.get("state", "not started"),
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
                "active": states is not None
                and any(s in ACTIVE_STATES for s in states.values()),
                "squeue_unavailable": states is None,
                "tasks": tasks,
            }
        )
    cases = _case_states(layout, manifest, squeue)
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


def run_cli(args: list[str], timeout: float = 300.0) -> tuple[int, str]:
    """``scenario-slurm <args>``, run by this viewer's interpreter: its exit
    status and output."""
    command = [sys.executable, "-m", "autoware_carla_scenario.slurm.cli", *args]
    try:
        done = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return done.returncode, "\n".join(s for s in (done.stdout, done.stderr) if s)
