"""Submitting a batch's cases as a SLURM array, and following it (docs/slurm.md).

The submitting side of ``launcher=slurm``: :func:`submit_batch` is what
``scenario --multirun ... launcher=slurm`` ends in, once the batch is expanded,
labelled, built and checked as a local batch is.  It writes the manifest,
submits the array (:func:`submit_cases`) and, unless told not to wait, follows
it until it ends and collects it (:func:`follow`).  ``scenario-slurm
resubmit`` submits the cases that did not finish again the same way.

Only the standard library here (and OmegaConf, inside :func:`submit_batch`):
the ``scenario-slurm`` commands import this without CARLA.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .manifest import (
    MANIFEST_VERSION,
    BatchLayout,
    config_files,
    sequential_stems,
    write_json,
)
from .settings import SlurmSettings, require_sbatch, snapshot_environment

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = [
    "callable_path",
    "collect_script",
    "follow",
    "job_script",
    "relative_batch_dir",
    "sbatch_command",
    "squeue_states",
    "submit_batch",
    "submit_cases",
]

logger = logging.getLogger(__name__)

#: How the worker is started in the job.
WORKER_MODULE = "autoware_carla_scenario.slurm.worker"
#: The light command line the collect job runs.
CLI_MODULE = "autoware_carla_scenario.slurm.cli"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def callable_path(fn: Callable[..., Any]) -> str:
    """``module:qualname`` of *fn*, which a worker imports to build its cases.

    Raises:
        ValueError: *fn* cannot be imported by name (a lambda, a nested
            function, something defined in ``__main__``).
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname or module == "__main__" or "<" in qualname:
        raise ValueError(
            "launcher=slurm runs each case in a SLURM task, which imports "
            f"build_scenario_fn by name, and {fn!r} has none a task could import "
            "(define it at module level in an importable module)"
        )
    return f"{module}:{qualname}"


def _resolved(cfg: "DictConfig") -> dict[str, Any] | None:
    """*cfg* resolved to plain data, or ``None`` when it does not resolve here."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    try:
        container = OmegaConf.to_container(cfg, resolve=True)
    except Exception:  # noqa: BLE001 - only the worker's cross-check is lost
        logger.warning("A case's config does not resolve here", exc_info=True)
        return None
    return cast(dict[str, Any], container) if isinstance(container, dict) else None


def submit_batch(
    *,
    scenario_names: list[str],
    cases: Sequence[tuple[str, list[str]]],
    labels: list[str],
    case_overrides: list[list[str]],
    configs: Sequence["DictConfig"],
    scenario_classes: list[str],
    overrides: list[str],
    expand: bool,
    output_dir: Path,
    build_scenario_fn: Callable[..., Any] | None = None,
    wait: bool | None = None,
) -> int:
    """Submit a built and checked batch as a SLURM array; the exit status.

    *cases*, *labels*, *case_overrides* and *configs* are what ``run_batch``
    made of the command line, one per case, and *scenario_classes* the class
    of the scenario each built.  The outputs go to *output_dir* as a local
    batch's would.  *wait* overrides ``launcher.wait``.
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    launchers = [
        OmegaConf.to_container(cfg.launcher, resolve=True) if "launcher" in cfg else {}
        for cfg in configs
    ]
    if any(launcher != launchers[0] for launcher in launchers):
        raise ValueError(
            "batch scenarios must share their launcher settings; set launcher.* "
            "on the command line rather than in a scenario's config"
        )
    raw = launchers[0]
    settings = SlurmSettings.from_mapping(
        cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    )
    if wait is not None:
        settings = settings.replace(wait=wait)

    layout = BatchLayout(output_dir)
    stems = sequential_stems(scenario_classes)
    resolved = [_resolved(cfg) for cfg in configs]
    digests: dict[str, str] = {}
    cwd = os.getcwd()
    remote: dict[str, Any] = {}
    if settings.remote_cwd is not None:
        remote["batch_dir_relative"] = relative_batch_dir(output_dir, cwd)
    manifest: dict[str, Any] = {
        # Version 1 unless the jobs need what version 2 adds (remote_cwd).
        "version": MANIFEST_VERSION if remote else 1,
        "created": _now(),
        "batch_dir": str(output_dir),
        **remote,
        "cwd": cwd,
        "python": sys.executable,
        "argv": list(sys.argv),
        "scenario_names": list(scenario_names),
        "overrides": list(overrides),
        "expand": bool(expand),
        "build_scenario_fn": (
            None if build_scenario_fn is None else callable_path(build_scenario_fn)
        ),
        # The submitting process checked every case (once a config shape); the
        # tasks build the same configs and do not check them again.
        "typechecked": True,
        "launcher": settings.to_dict(),
        "env": snapshot_environment(
            settings.env_passthrough, explicit=settings.env.keys()
        ),
        "cases": [
            {
                "index": index,
                "label": label,
                "scenario": name,
                "overrides": list(case),
                "case_overrides": list(own),
                "scenario_class": scenario_class,
                "stem": stem,
                "config": config,
                # The files the config names, by content: a node may see them
                # at other paths (worker._check_same_config).
                "files": config_files(config, digests),
            }
            for index, ((name, case), label, own, scenario_class, stem, config) in (
                enumerate(
                    zip(
                        cases, labels, case_overrides, scenario_classes, stems, resolved
                    )
                )
            )
        ],
    }
    # Only the user's: it holds the recorded environment (write_json, 0600).
    write_json(layout.manifest, manifest)
    index = submit_cases(layout, manifest, settings, list(range(len(labels))))
    return after_submit(layout, index, settings)


def relative_batch_dir(batch_dir: Path, cwd: str) -> str:
    """*batch_dir* relative to *cwd*, as the jobs of a ``remote_cwd`` batch
    find it under their working directory.

    Raises:
        ValueError: *batch_dir* is not under *cwd*.
    """
    base = Path(cwd).resolve()
    target = (base / batch_dir).resolve()
    try:
        relative = target.relative_to(base)
    except ValueError:
        raise ValueError(
            f"launcher.remote_cwd: the batch's output directory {target} is not "
            f"under the directory scenario runs in ({base}), which the nodes see "
            "at launcher.remote_cwd; run scenario from that directory (the "
            "submit host's mount of it) with the outputs under it "
            "(docs/slurm.md)"
        ) from None
    if relative == Path("."):
        raise ValueError(
            "launcher.remote_cwd: the batch's output directory is the directory "
            f"scenario runs in ({base}); it must be a directory under it"
        )
    return relative.as_posix()


def _shell_path(path: str) -> str:
    """*path* as one shell word, its leading ``~`` the job user's ``$HOME``."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        rest = path[2:].lstrip("/")
        return '"$HOME"/' + shlex.quote(rest) if rest else '"$HOME"'
    return shlex.quote(path)


def _remote_dir(remote_cwd: str) -> str:
    """``launcher.remote_cwd`` as one shell word: relative to ``$HOME``."""
    if remote_cwd.startswith("/") or remote_cwd.startswith("~"):
        return _shell_path(remote_cwd)
    return '"$HOME"/' + shlex.quote(remote_cwd)


#: Where sbatch puts a ``remote_cwd`` job's output until the job script sends
#: it to the batch's ``slurm/logs/``: the batch's path on the node is not known
#: on the submit host, and ``/tmp`` is a directory every node has.  Only what
#: comes before the job reaches its working directory, and what SLURM itself
#: writes (a time limit, a cancel), is left there.
REMOTE_EARLY_OUTPUT_DIR = "/tmp"


def _remote_prologue(
    manifest: dict[str, Any], settings: SlurmSettings, log: str
) -> list[str]:
    """The lines a ``remote_cwd`` job starts with: to its working directory
    (``$WORKDIR``) on the node, the batch's (``$BATCH_DIR``) under it, and its
    output to *log* in the batch's ``slurm/logs/``."""
    relative = manifest.get("batch_dir_relative")
    if not relative or settings.remote_cwd is None:
        raise ValueError(
            "launcher.remote_cwd is set, but the batch's manifest has no "
            "batch_dir_relative: the batch was submitted without remote_cwd"
        )
    workdir = _remote_dir(settings.remote_cwd)
    return [
        "# launcher.remote_cwd: the submit host sees this file system at other",
        "# paths, so every path here is the node's.  The job user's home on",
        "# this node, from its passwd database: --export may have handed over",
        "# the submit host's $HOME.",
        'node_home=$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)',
        'if [ -n "$node_home" ]; then export HOME="$node_home"; fi',
        f"cd {workdir} || "
        f'{{ echo "launcher.remote_cwd: no directory "{workdir} >&2; exit 1; }}',
        "WORKDIR=$PWD",
        f'BATCH_DIR="$WORKDIR"/{shlex.quote(str(relative))}',
        "# The job's output from here on: the batch's logs.",
        f'exec >>"$BATCH_DIR"/slurm/logs/{log} 2>&1',
    ]


def _setup_lines(settings: SlurmSettings) -> list[str]:
    """``launcher.setup``; with ``remote_cwd`` free to read unset variables
    (an environment script on the node, run with little of an environment)."""
    if settings.remote_cwd is None:
        return list(settings.setup)
    return ["set +u", *settings.setup, "set -u"]


def after_submit(layout: BatchLayout, index: int, settings: SlurmSettings) -> int:
    """Follow submission *index* to its end and collect (``wait``), or say
    how to; the exit status."""
    submission = layout.read_submission(index)
    if settings.wait:
        return follow(layout, index, settings)
    if submission.get("collect_error"):
        return 1  # submit_cases said what failed, and how to collect
    cli = f"{shlex.quote(sys.executable)} -m {CLI_MODULE}"
    target = shlex.quote(str(layout.batch_dir))
    print(f"Submitted; not waiting (--no-wait). Job {submission['job_id']}.")  # noqa: T201
    if submission.get("collect_job_id"):
        print(  # noqa: T201
            f"Collect job {submission['collect_job_id']} writes "
            f"{layout.batch_dir / 'batch_results.json'} once the array ends."
        )
    print(f"  follow:   scenario-slurm wait {target}")  # noqa: T201
    print(f"  status:   scenario-slurm status {target}")  # noqa: T201
    print(f"  collect:  scenario-slurm collect {target}")  # noqa: T201
    print(f"  (or {cli} <command> {target})")  # noqa: T201
    return 0


def _chunks(indices: list[int], size: int) -> list[list[int]]:
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def job_script(
    layout: BatchLayout,
    manifest: dict[str, Any],
    settings: SlurmSettings,
    submission: int,
) -> str:
    """The shell script every task of submission *submission* runs."""
    python = _shell_path(settings.python or manifest["python"])
    remote = settings.remote_cwd is not None
    batch = (
        f"{manifest.get('batch_dir_relative')} under launcher.remote_cwd "
        f"{settings.remote_cwd}"
        if remote
        else str(layout.batch_dir)
    )
    lines = [
        "#!/bin/bash",
        "# autoware_carla_scenario: one task of a batch submitted with "
        "launcher=slurm (docs/slurm.md).",
        f"# Batch: {batch}  submission: {submission}",
        "set -u",
        'TASK="${SLURM_ARRAY_TASK_ID:?this script runs as a SLURM array task}"',
    ]
    if remote:
        lines += [
            "",
            *_remote_prologue(
                manifest, settings, '"${SLURM_ARRAY_JOB_ID}_${TASK}.out"'
            ),
        ]
    lines += [
        "",
        "# The submitting shell's environment, as it was (launcher.env_passthrough).",
    ]
    lines += [
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(manifest.get("env", {}).items())
    ]
    if settings.headless:
        lines += [
            "",
            "# CARLA renders off-screen (launcher.headless).",
            "unset DISPLAY WAYLAND_DISPLAY",
        ]
    if settings.setup:
        lines += ["", "# launcher.setup"]
        lines += _setup_lines(settings)
    if settings.env:
        lines += ["", "# launcher.env"]
        lines += [
            f"export {name}={shlex.quote(value)}"
            for name, value in sorted(settings.env.items())
        ]
    if remote:
        port_file = f'"$BATCH_DIR"/slurm/tasks/{submission}-${{TASK}}.port'
        manifest_path = '"$BATCH_DIR"/slurm/manifest.json'
        workdir = '"$WORKDIR"'
    else:
        tasks_dir = layout.root / "tasks"
        port_file = f"{shlex.quote(str(tasks_dir))}/{submission}-${{TASK}}.port"
        manifest_path = shlex.quote(str(layout.manifest))
        workdir = shlex.quote(manifest["cwd"])
    lines += [
        "",
        f"cd {workdir} || exit 1",
        f"PORT_FILE={port_file}",
        "",
        "# The worker runs in the background so that the signal SLURM sends",
        "# before the time limit (--signal=B:TERM) reaches it through this trap.",
        f"{python} -u -m {WORKER_MODULE} " f'{manifest_path} {submission} "$TASK" &',
        "child=$!",
        "trap 'kill -TERM \"$child\" 2>/dev/null' TERM INT USR1 USR2",
        "rc=0",
        "while :; do",
        '  wait "$child"; rc=$?',
        '  kill -0 "$child" 2>/dev/null || break',
        "done",
        "",
        "# The worker stops its CARLA however it ends, and removes the port file",
        "# once it has; if it was killed outright, its CARLA is stopped here.",
        'if [ -s "$PORT_FILE" ]; then',
        '  port=$(cat "$PORT_FILE")',
        '  if pkill -TERM -f -- "-carla-rpc-port=${port}( |$)"; then',
        "    sleep 5",
        '    pkill -KILL -f -- "-carla-rpc-port=${port}( |$)"',
        "  fi",
        '  rm -f "$PORT_FILE"',
        "fi",
        'exit "$rc"',
        "",
    ]
    return "\n".join(lines)


def _job_name(layout: BatchLayout, settings: SlurmSettings) -> str:
    if settings.job_name:
        return settings.job_name
    batch = layout.batch_dir
    return f"acs-{batch.parent.name}-{batch.name}"


def _output_options(
    layout: BatchLayout, settings: SlurmSettings, name: str
) -> list[str]:
    """``sbatch --output`` (and ``--chdir``) of a job whose log is *name*.

    With ``remote_cwd`` neither the batch's path nor the job's working
    directory on the node is known here: the job starts in ``/tmp`` with its
    output there, and its script sends the output on to the batch's logs.
    """
    if settings.remote_cwd is None:
        return [f"--output={layout.logs}/{name}"]
    early = REMOTE_EARLY_OUTPUT_DIR
    return [f"--chdir={early}", f"--output={early}/acs-slurm-%u-{name}"]


def sbatch_command(
    layout: BatchLayout,
    settings: SlurmSettings,
    script: Path,
    tasks: int,
) -> list[str]:
    """The ``sbatch`` command line submitting *tasks* tasks of *script*."""
    array = f"0-{tasks - 1}"
    if settings.array_parallelism is not None:
        array += f"%{settings.array_parallelism}"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={_job_name(layout, settings)}",
        f"--array={array}",
        *_output_options(layout, settings, "%A_%a.out"),
        f"--export={settings.export}",
        f"--signal=B:TERM@{settings.signal_grace_seconds}",
    ]
    for flag, value in (
        ("partition", settings.partition),
        ("account", settings.account),
        ("qos", settings.qos),
        ("time", settings.time),
        ("mem", settings.mem),
        ("cpus-per-task", settings.cpus_per_task),
        ("gres", settings.gres),
    ):
        if value is not None:
            command.append(f"--{flag}={value}")
    command += list(settings.extra_sbatch_args)
    command.append(str(script))
    return command


def collect_script(manifest: dict[str, Any], settings: SlurmSettings) -> str:
    """The script of the job collecting a ``remote_cwd`` batch, on a node."""
    python = _shell_path(settings.python or manifest["python"])
    lines = [
        "#!/bin/bash",
        "# autoware_carla_scenario: collects a batch submitted with "
        "launcher=slurm once its array has ended (docs/slurm.md).",
        f"# Batch: {manifest.get('batch_dir_relative')} under launcher.remote_cwd "
        f"{settings.remote_cwd}",
        "set -u",
        "",
        *_remote_prologue(manifest, settings, '"collect-${SLURM_JOB_ID}.out"'),
    ]
    if manifest.get("env"):
        lines += ["", "# launcher.env_passthrough"]
        lines += [
            f"export {name}={shlex.quote(value)}"
            for name, value in sorted(manifest["env"].items())
        ]
    if settings.setup:
        lines += ["", "# launcher.setup", *_setup_lines(settings)]
    if settings.env:
        lines += ["", "# launcher.env"]
        lines += [
            f"export {name}={shlex.quote(value)}"
            for name, value in sorted(settings.env.items())
        ]
    lines += [
        "",
        'cd "$WORKDIR" || exit 1',
        f'exec {python} -m {CLI_MODULE} collect "$BATCH_DIR"',
        "",
    ]
    return "\n".join(lines)


def _collect_command(
    layout: BatchLayout,
    manifest: dict[str, Any],
    settings: SlurmSettings,
    job: str,
    script: Path | None = None,
) -> list[str]:
    """The ``sbatch`` command of the job collecting the batch after *job*:
    with ``remote_cwd``, the job runs *script* (:func:`collect_script`)."""
    if settings.remote_cwd is not None:
        if script is None:
            raise ValueError("a remote_cwd batch's collect job runs a script")
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={_job_name(layout, settings)}-collect",
            f"--dependency=afterany:{job}",
            *_output_options(layout, settings, "collect-%j.out"),
            f"--export={settings.export}",
            "--cpus-per-task=1",
        ]
        for flag, value in (
            ("partition", settings.partition),
            ("account", settings.account),
            ("qos", settings.qos),
            ("mem", settings.collect_mem),
        ):
            if value is not None:
                command.append(f"--{flag}={value}")
        command.append(str(script))
        return command
    python = settings.python or manifest["python"]
    wrap = " ".join(
        [
            _shell_path(python),
            "-m",
            CLI_MODULE,
            "collect",
            shlex.quote(str(layout.batch_dir)),
        ]
    )
    exports = " ".join(
        f"{name}={shlex.quote(value)}"
        for name, value in sorted(manifest.get("env", {}).items())
        if name in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "VIRTUAL_ENV")
    )
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={_job_name(layout, settings)}-collect",
        f"--dependency=afterany:{job}",
        f"--output={layout.logs}/collect-%j.out",
        f"--export={settings.export}",
        "--cpus-per-task=1",
        f"--chdir={manifest['cwd']}",
    ]
    for flag, value in (
        ("partition", settings.partition),
        ("account", settings.account),
        ("qos", settings.qos),
        ("mem", settings.collect_mem),
    ):
        if value is not None:
            command.append(f"--{flag}={value}")
    command.append(f"--wrap={'env ' + exports + ' ' if exports else ''}{wrap}")
    return command


def _run_sbatch(command: list[str]) -> str:
    """Run *command* (an ``sbatch --parsable``); the job id it printed."""
    command = [require_sbatch(), *command[1:]]
    logger.info("Submitting: %s", shlex.join(command))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"sbatch failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    # --parsable: "<job id>" or "<job id>;<cluster>".
    job = completed.stdout.strip().splitlines()[-1].split(";")[0].strip()
    if not job:
        raise RuntimeError(f"sbatch printed no job id: {completed.stdout!r}")
    return job


def _write_script(path: Path, text: str) -> None:
    """Write the job script *path*.  Only the user's: it exports the recorded
    environment.  Created so, never readable by others even for a moment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    path.chmod(0o700)


def submit_cases(
    layout: BatchLayout,
    manifest: dict[str, Any],
    settings: SlurmSettings,
    case_indices: list[int],
) -> int:
    """Submit *case_indices* of the batch as one array; the submission's index.

    The cases are split into tasks of ``launcher.cases_per_task`` (in order),
    which SLURM runs at most ``launcher.array_parallelism`` at a time.
    """
    if not case_indices:
        raise ValueError("no case to submit")
    index = layout.next_submission()
    chunks = _chunks(list(case_indices), settings.chunk_size(len(case_indices)))
    script = layout.job_script(index)
    _write_script(script, job_script(layout, manifest, settings, index))
    layout.logs.mkdir(parents=True, exist_ok=True)
    (layout.root / "tasks").mkdir(parents=True, exist_ok=True)
    (layout.root / "results").mkdir(parents=True, exist_ok=True)

    submission: dict[str, Any] = {
        "index": index,
        "created": _now(),
        "submitted_at": time.time(),
        "cases": list(case_indices),
        "chunks": chunks,
        "launcher": settings.to_dict(),
        "job_id": None,
        "collect_job_id": None,
    }
    # Written before sbatch: a task that starts at once reads its chunk here.
    write_json(layout.submission(index), submission)
    command = sbatch_command(layout, settings, script, len(chunks))
    submission["sbatch"] = command
    try:
        submission["job_id"] = _run_sbatch(command)
    except BaseException:
        layout.submission(index).unlink(missing_ok=True)
        raise
    write_json(layout.submission(index), submission)
    print(  # noqa: T201
        f"SLURM job {submission['job_id']}: {len(case_indices)} case(s) in "
        f"{len(chunks)} task(s) of up to {len(chunks[0])} case(s), "
        + (
            f"{settings.array_parallelism} at a time"
            if settings.array_parallelism is not None
            else "as many at a time as SLURM runs"
        )
        + f"; outputs in {layout.batch_dir}",
        flush=True,
    )
    if not settings.wait and settings.collect_job:
        collect_path: Path | None = None
        if settings.remote_cwd is not None:
            collect_path = layout.root / "jobs" / f"{index}-collect.sh"
            _write_script(collect_path, collect_script(manifest, settings))
        collect = _collect_command(
            layout, manifest, settings, submission["job_id"], collect_path
        )
        submission["collect_sbatch"] = collect
        try:
            submission["collect_job_id"] = _run_sbatch(collect)
        except (RuntimeError, OSError) as exc:
            # The array runs all the same: say how to collect it by hand.
            submission["collect_error"] = str(exc)
            target = shlex.quote(str(layout.batch_dir))
            print(  # noqa: T201
                f"Error: SLURM job {submission['job_id']} (the batch's array) was "
                f"submitted, but the job collecting it was not: {exc}\n"
                f"  follow and collect it: scenario-slurm wait {target}\n"
                f"  or once it ends:       scenario-slurm collect {target}",
                file=sys.stderr,
                flush=True,
            )
        write_json(layout.submission(index), submission)
    return index


#: The states of a task SLURM has not finished with.
ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "RUNNING",
        "COMPLETING",
        "SUSPENDED",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
    }
)

#: After a Ctrl-C's ``scancel``, how long :func:`follow` waits for the tasks to
#: leave the queue (stopping their CARLAs, recording their state) before it
#: collects.
CANCEL_WAIT_SECONDS = 60.0

_TASK_RANGE = re.compile(r"^(\d+)_\[([^\]]*)\]$")


def _expand_ranges(spec: str) -> list[int]:
    """``0-3,5`` (``%K`` dropped) as its task ids."""
    spec = spec.split("%", 1)[0]
    tasks: list[int] = []
    for part in spec.split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        tasks.extend(range(int(first), int(last or first) + 1))
    return tasks


def squeue_states(job: str) -> dict[int, str] | None:
    """The state of each task of array *job* that SLURM still lists.

    ``{}`` once the job is gone from the queue (or never was); ``None`` when
    ``squeue`` cannot be asked (not on PATH, or it failed for another reason).
    """
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-r", "-j", str(job), "-o", "%i %T"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        if "Invalid job id" in completed.stderr:
            return {}
        return None
    states: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        job_task, state = parts[0], parts[1]
        ranged = _TASK_RANGE.match(job_task)
        if ranged:
            for task in _expand_ranges(ranged.group(2)):
                states[task] = state
            continue
        head, _, task_id = job_task.partition("_")
        if head == str(job) and task_id.isdigit():
            states[int(task_id)] = state
    return states


def follow(layout: BatchLayout, index: int, settings: SlurmSettings) -> int:
    """Follow submission *index* until SLURM lists none of its tasks, then
    collect the batch; the exit status (0 when every case passed)."""
    from .collect import collect  # noqa: PLC0415

    submission = layout.read_submission(index)
    job = str(submission["job_id"])
    cases = [case for chunk in submission["chunks"] for case in chunk]
    last = ""
    unknown_since: float | None = None
    try:
        while True:
            states = squeue_states(job)
            done = passed = 0
            for case in cases:
                record = layout.read_result(case)
                if record is not None and record.get("submission") == index:
                    done += 1
                    passed += bool(record["result"]["passed"])
            running = sum(1 for s in (states or {}).values() if s == "RUNNING")
            pending = sum(1 for s in (states or {}).values() if s == "PENDING")
            line = (
                f"[slurm] job {job}: {done}/{len(cases)} case(s) done "
                f"({passed} passed); {running} task(s) running, {pending} pending"
            )
            if line != last:
                print(line, flush=True)  # noqa: T201
                last = line
            if states is not None and not any(
                state in ACTIVE_STATES for state in states.values()
            ):
                break
            if states is None:
                # squeue failed: give it a few polls before giving up on it.
                unknown_since = unknown_since or time.monotonic()
                if time.monotonic() - unknown_since > 10 * settings.poll_seconds:
                    print(  # noqa: T201
                        "[slurm] squeue keeps failing; stopped following. Collect "
                        f"later with: scenario-slurm collect {layout.batch_dir}",
                        flush=True,
                    )
                    return 1
            else:
                unknown_since = None
            time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        if settings.cancel_on_interrupt:
            cancel_jobs([job])
            print(  # noqa: T201
                f"\n[slurm] interrupted: cancelled job {job}. Run the cases that "
                f"did not finish with: scenario-slurm resubmit {layout.batch_dir}",
                flush=True,
            )
            _wait_until_gone(job, CANCEL_WAIT_SECONDS, settings.poll_seconds)
        else:
            print(  # noqa: T201
                f"\n[slurm] stopped following job {job}, which goes on. Follow it "
                f"again with: scenario-slurm wait {layout.batch_dir}",
                flush=True,
            )
            return 130
    return collect(layout)


def _wait_until_gone(job: str, timeout: float, poll_seconds: float) -> bool:
    """Wait, at most *timeout* s, until SLURM lists no active task of *job*:
    collecting sooner would find the cancelled tasks' results half written.
    Whether they left; a second Ctrl-C stops waiting."""
    deadline = time.monotonic() + timeout
    try:
        while True:
            states = squeue_states(job)
            if states is not None and not any(
                state in ACTIVE_STATES for state in states.values()
            ):
                return True
            if time.monotonic() >= deadline:
                break
            time.sleep(min(poll_seconds, 1.0))
    except KeyboardInterrupt:
        pass
    print(  # noqa: T201
        f"[slurm] job {job} is still in the queue; collecting what there is",
        flush=True,
    )
    return False


def cancel_jobs(jobs: Sequence[str]) -> bool:
    """``scancel`` *jobs* (this user's own); whether scancel succeeded."""
    jobs = [str(job) for job in jobs if job]
    if not jobs:
        return True
    try:
        completed = subprocess.run(
            ["scancel", *jobs], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        print(f"[slurm] could not run scancel: {exc}", file=sys.stderr)  # noqa: T201
        return False
    if completed.returncode != 0:
        print(  # noqa: T201
            f"[slurm] scancel {' '.join(jobs)} failed: {completed.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True
