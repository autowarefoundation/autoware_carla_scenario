"""The text a SLURM batch hands to SLURM, and reads back from it (docs/slurm.md).

The job script every task of a submission runs (:func:`job_script`), the
``sbatch`` command line submitting it (:func:`sbatch_command`), and the
script and command line of the job collecting the batch after its array
(:func:`collect_script`, :func:`_collect_command`); and the state of each
task ``squeue`` lists (:func:`squeue_task_states`).  Plain text from the
launcher's settings and what they read of the batch's manifest
(:class:`ScriptInputs`), with nothing of the file system, of SLURM or of the
manifest's JSON: :mod:`.submit` writes and runs them, and the library check
compiles this module (docs/typecheck.md).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

from .settings import SlurmSettings

__all__ = [
    "CLI_MODULE",
    "REMOTE_EARLY_OUTPUT_DIR",
    "WORKER_MODULE",
    "BatchPaths",
    "ScriptInputs",
    "collect_script",
    "job_script",
    "relative_batch_dir",
    "sbatch_command",
    "squeue_task_states",
]

#: How the worker is started in the job.
WORKER_MODULE = "autoware_carla_scenario.slurm.worker"
#: The light command line the collect job runs.
CLI_MODULE = "autoware_carla_scenario.slurm.cli"

#: Where sbatch puts a ``remote_cwd`` job's output until the job script sends
#: it to the batch's ``slurm/logs/``: the batch's path on the node is not known
#: on the submit host, and ``/tmp`` is a directory every node has.  Only what
#: comes before the job reaches its working directory, and what SLURM itself
#: writes (a time limit, a cancel), is left there.
REMOTE_EARLY_OUTPUT_DIR = "/tmp"


@dataclass(frozen=True)
class BatchPaths:
    """Where the launcher's files of the batch in *batch_dir* are.

    Only the paths: :class:`~autoware_carla_scenario.slurm.manifest.BatchLayout`
    derives from it, and reads the files.
    """

    batch_dir: Path

    @property
    def root(self) -> Path:
        return self.batch_dir / "slurm"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def submission(self, index: int) -> Path:
        return self.root / "submissions" / f"{index}.json"

    def job_script(self, index: int) -> Path:
        return self.root / "jobs" / f"{index}.sh"

    def task(self, submission: int, task: int) -> Path:
        return self.root / "tasks" / f"{submission}-{task}.json"

    def task_port(self, submission: int, task: int) -> Path:
        """The CARLA RPC port task *task* holds (the job's leak guard reads it)."""
        return self.root / "tasks" / f"{submission}-{task}.port"

    @property
    def start_lock(self) -> Path:
        """When the batch's last task started (``launcher.start_stagger_seconds``)."""
        return self.root / "start.lock"

    def result(self, case: int) -> Path:
        return self.root / "results" / f"{case}.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


class ScriptInputs(NamedTuple):
    """What the scripts read of the batch's manifest (``slurm/manifest.json``).

    :mod:`.submit` reads it off the manifest.
    """

    #: The submitting interpreter (``launcher.python`` overrides it).
    python: str
    #: The directory the batch was submitted from.
    cwd: str
    #: The submitting shell's variables the jobs set again
    #: (``launcher.env_passthrough``).
    env: dict[str, str]
    #: The batch's directory under ``launcher.remote_cwd``; ``None`` for a
    #: batch submitted without it.
    batch_dir_relative: Optional[str]


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


def _remote_prologue(
    manifest: ScriptInputs, settings: SlurmSettings, log: str
) -> list[str]:
    """The lines a ``remote_cwd`` job starts with: to its working directory
    (``$WORKDIR``) on the node, the batch's (``$BATCH_DIR``) under it, and its
    output to *log* in the batch's ``slurm/logs/``."""
    relative = manifest.batch_dir_relative
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


def job_script(
    layout: BatchPaths,
    manifest: ScriptInputs,
    settings: SlurmSettings,
    submission: int,
) -> str:
    """The shell script every task of submission *submission* runs."""
    python = _shell_path(settings.python or manifest.python)
    remote = settings.remote_cwd is not None
    batch = (
        f"{manifest.batch_dir_relative} under launcher.remote_cwd "
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
        for name, value in sorted(manifest.env.items())
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
    if len(settings.env) > 0:
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
        workdir = shlex.quote(manifest.cwd)
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


def _job_name(layout: BatchPaths, settings: SlurmSettings) -> str:
    if settings.job_name:
        return settings.job_name
    batch = layout.batch_dir
    return f"acs-{batch.parent.name}-{batch.name}"


def _output_options(
    layout: BatchPaths, settings: SlurmSettings, name: str
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
    layout: BatchPaths,
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


def collect_script(manifest: ScriptInputs, settings: SlurmSettings) -> str:
    """The script of the job collecting a ``remote_cwd`` batch, on a node."""
    python = _shell_path(settings.python or manifest.python)
    lines = [
        "#!/bin/bash",
        "# autoware_carla_scenario: collects a batch submitted with "
        "launcher=slurm once its array has ended (docs/slurm.md).",
        f"# Batch: {manifest.batch_dir_relative} under launcher.remote_cwd "
        f"{settings.remote_cwd}",
        "set -u",
        "",
        *_remote_prologue(manifest, settings, '"collect-${SLURM_JOB_ID}.out"'),
    ]
    if len(manifest.env) > 0:
        lines += ["", "# launcher.env_passthrough"]
        lines += [
            f"export {name}={shlex.quote(value)}"
            for name, value in sorted(manifest.env.items())
        ]
    if settings.setup:
        lines += ["", "# launcher.setup", *_setup_lines(settings)]
    if len(settings.env) > 0:
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
    layout: BatchPaths,
    manifest: ScriptInputs,
    settings: SlurmSettings,
    job: str,
    script: Optional[Path] = None,
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
    python = settings.python or manifest.python
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
        for name, value in sorted(manifest.env.items())
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
        f"--chdir={manifest.cwd}",
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


def squeue_task_states(job: str, output: str) -> dict[int, str]:
    """The state of each task of array *job* in *output*, what
    ``squeue -h -r -j <job> -o "%i %T"`` printed: ``<job>_<task> <state>``
    a line, or ``<job>_[<ranges>] <state>`` for pending tasks squeue did not
    split."""
    states: dict[int, str] = {}
    for line in output.splitlines():
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
