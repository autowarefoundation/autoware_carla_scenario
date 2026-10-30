"""The ``launcher`` config: who runs a batch's cases (docs/slurm.md).

``launcher.type`` is ``local`` -- the batch runs its cases one after another in
this process, on one CARLA, which is what a config with no ``launcher`` at all
means -- or ``slurm``, which runs them as the tasks of a SLURM array job.  The
``launcher`` group is not part of the default config: nothing changes unless a
run selects it (``launcher=slurm``, or ``launcher.type: slurm`` in a config).

Free of heavy imports, so the light ``scenario-slurm`` commands can read it.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import math
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_ENV_PASSTHROUGH",
    "LAUNCHER_TYPES",
    "SENSITIVE_NAME_PATTERNS",
    "SlurmSettings",
    "launcher_requested_in_argv",
    "launcher_type",
    "require_sbatch",
    "snapshot_environment",
]

#: What ``launcher.type`` may be.
LAUNCHER_TYPES = ("local", "slurm")

#: The variables of the submitting shell a job sets again, as they were when
#: the batch was submitted, whatever the shell a resubmit runs from has: the
#: interpreter's and the libraries' paths, CARLA's, the framework's own, and the
#: ones a driver policy reads (OnePlanner's ``ONEPLANNER_*``).  ``*`` matches
#: any run of characters.  No token or credential variable is on it: the job
#: script holding them is a file in the batch's output directory, and a name
#: :data:`SENSITIVE_NAME_PATTERNS` matches is left out even when a pattern
#: here matches it (:func:`snapshot_environment`).
#: ``tuple([...])``, not a tuple display: the library check reads it as the
#: ``List[str]`` the ``env_passthrough`` field it defaults is checked as
#: (docs/typecheck.md).
DEFAULT_ENV_PASSTHROUGH: tuple[str, ...] = tuple(
    [
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CARLA_*",
        "AUTOWARE_*",
        "ONEPLANNER_*",
        "TYPESAFE_CARLA_*",
        "GLIBC_TUNABLES",
        "MALLOC_*",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDAToolkit_ROOT",
    ]
)


#: Names never recorded through a pattern (case-insensitive): a secret's value
#: would be written into the manifest and the job scripts.  A variable named
#: exactly -- not through a ``*`` -- on ``env_passthrough``, or in
#: ``launcher.env``, is recorded all the same.  ``tuple([...])``, as
#: :data:`DEFAULT_ENV_PASSTHROUGH` is.
SENSITIVE_NAME_PATTERNS: tuple[str, ...] = tuple(
    [
        "*TOKEN*",
        "*SECRET*",
        "*KEY*",
        "*PASSWORD*",
        "*CREDENTIAL*",
    ]
)


def _sensitive(name: str) -> bool:
    upper = name.upper()
    return any(fnmatch.fnmatchcase(upper, p) for p in SENSITIVE_NAME_PATTERNS)


def launcher_type(cfg: Mapping[str, Any] | None) -> str:
    """``launcher.type`` of *cfg*: ``local`` when it has no ``launcher``.

    Raises:
        ValueError: The type is not one of :data:`LAUNCHER_TYPES`.
    """
    launcher = None if cfg is None else cfg.get("launcher")
    if launcher is None:
        return "local"
    kind = str(launcher.get("type") or "local")
    if kind not in LAUNCHER_TYPES:
        raise ValueError(
            f"Unknown launcher.type: {kind!r}. Expected one of: "
            + ", ".join(repr(t) for t in LAUNCHER_TYPES)
            + "."
        )
    return kind


def launcher_requested_in_argv(argv: Iterable[str]) -> str | None:
    """The launcher a command line selects (``launcher=slurm``,
    ``launcher.type=slurm``, with or without ``+``), or ``None``."""
    selected: str | None = None
    for arg in argv:
        name, sep, value = arg.lstrip("+").partition("=")
        if sep and name in ("launcher", "launcher.type"):
            selected = value.strip().strip("'\"")
    return selected


def require_sbatch() -> str:
    """The path of ``sbatch``.

    Raises:
        RuntimeError: There is no ``sbatch`` on ``PATH``.
    """
    path = shutil.which("sbatch")
    if path is None:
        raise RuntimeError(
            "launcher=slurm needs SLURM's sbatch, which is not on PATH: run the "
            "batch on a SLURM submit (login) host, or add SLURM's bin directory "
            "to PATH. Without launcher=slurm the batch runs here, case after "
            "case (docs/slurm.md)."
        )
    return path


def snapshot_environment(
    patterns: Iterable[str],
    environ: Mapping[str, str] | None = None,
    *,
    explicit: Iterable[str] = (),
) -> dict[str, str]:
    """The variables of *environ* (``os.environ``) whose names match *patterns*.

    A name :data:`SENSITIVE_NAME_PATTERNS` matches is left out, unless one of
    *patterns* is that very name or it is in *explicit* (``launcher.env``).
    """
    env = os.environ if environ is None else environ
    patterns = tuple(patterns)
    named = set(explicit) | {p for p in patterns if not any(c in p for c in "*?[")}
    return {
        name: value
        for name, value in sorted(env.items())
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        and (name in named or not _sensitive(name))
    }


def _optional_str(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


def _check_remote_path(value: str, name: str) -> None:
    """Refuse a path the job script cannot expand on the node: ``~user/``."""
    if "\n" in value or "\0" in value:
        raise ValueError(f"launcher.{name} must be one line, not {value!r}")
    if value.startswith("~") and value != "~" and not value.startswith("~/"):
        raise ValueError(
            f"launcher.{name} {value!r}: only ~ and ~/... (the job user's own "
            "home on the node) are expanded, not ~user"
        )


def _optional_int(value: Any, name: str, minimum: int = 1) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < minimum:
        raise ValueError(f"launcher.{name} must be at least {minimum} (got {number})")
    return number


@dataclass(frozen=True)
class SlurmSettings:
    """``launcher.*`` of a ``launcher.type: slurm`` run (conf/launcher/slurm.yaml).

    Every field is documented in docs/slurm.md; the ones given to ``sbatch``
    are left out of its command line when ``None``.
    """

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    #: ``sbatch --time`` (``HH:MM:SS``, ``D-HH:MM``, minutes).
    time: str | None = None
    #: ``sbatch --mem`` (``12G``); only scheduling where SLURM has no cgroups.
    mem: str | None = None
    cpus_per_task: int | None = None
    #: ``sbatch --gres`` (``gpu:1``, or ``shard:1`` to share a GPU).
    gres: str | None = None
    #: At most this many tasks at once (``--array=...%K``); ``None``: no cap.
    array_parallelism: int | None = None
    #: Cases one task runs on its CARLA; ``None``: spread the cases evenly over
    #: ``array_parallelism`` tasks, or one case a task without a cap.
    cases_per_task: int | None = None
    job_name: str | None = None
    extra_sbatch_args: tuple[str, ...] = ()
    #: ``sbatch --export``: ``ALL`` passes the submitting shell's environment
    #: (``NONE`` by default with ``remote_cwd``).
    export: str = "ALL"
    #: Set in the job before the worker starts, after ``setup``.
    env: Mapping[str, str] = field(default_factory=dict)
    #: The submitting shell's variables the job sets again (fnmatch patterns;
    #: none by default with ``remote_cwd``).
    env_passthrough: tuple[str, ...] = DEFAULT_ENV_PASSTHROUGH
    #: Shell lines the job runs before the worker (``module load``, ``source``).
    setup: tuple[str, ...] = ()
    #: The interpreter the jobs (the worker's and the collect job) run; a
    #: relative path is the job's working directory's, a bare name is looked
    #: up on ``PATH``, ``~/`` is the home on the node; ``None``: this one.
    python: str | None = None
    #: Where every job works on the nodes, when the submit host sees their
    #: file system at another path (an sshfs or NFS mount): relative to the
    #: job user's home on the node, ``~/...``, or absolute.  Expanded on the
    #: node.  ``None``: the submitting directory, which the nodes see too.
    remote_cwd: str | None = None
    #: Follow the batch until it ends, then collect; ``False``: submit and exit.
    wait: bool = True
    poll_seconds: float = 15.0
    #: SIGTERM this long before ``time`` runs out (``--signal=B:TERM@N``).
    signal_grace_seconds: int = 120
    #: Each task's ports: a slot of ``PORT_SLOT_WIDTH`` from ``port_base``.
    port_base: int = 20000
    port_slots: int = 1000
    #: Where the slot locks are; ``None``: a node-wide directory
    #: (``/dev/shm/acs-slurm-ports-<uid>``, else ``/tmp``), not a job's
    #: private ``/tmp``.
    port_lock_dir: str | None = None
    #: Unset ``DISPLAY`` in the job, so CARLA renders off-screen.
    headless: bool = True
    #: No task of the batch starts sooner than this after another one did
    #: (the first wave, the tasks that follow as slots free up, a resubmit's),
    #: so that two never fill a cache they share -- a driver policy's
    #: engines -- at once; 0: they start as SLURM starts them.
    start_stagger_seconds: float = 0.0
    #: With ``wait: false``: a job, after the array, that collects the batch.
    collect_job: bool = True
    collect_mem: str | None = "2G"
    #: On Ctrl-C while waiting, cancel the batch's jobs (else leave them run).
    cancel_on_interrupt: bool = True

    @classmethod
    def from_mapping(cls, launcher: Mapping[str, Any] | None) -> "SlurmSettings":
        """Read ``launcher`` (a plain mapping, resolved); unknown keys refused.

        Raises:
            ValueError: A key no field has, or a value out of range.
        """
        data = dict(launcher or {})
        data.pop("type", None)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown launcher setting(s) {unknown} for launcher.type=slurm; "
                f"known: {sorted(known)}"
            )
        options: dict[str, Any] = {}
        for key in ("partition", "account", "qos", "time", "mem", "gres"):
            if key in data:
                options[key] = _optional_str(data[key])
        for key in ("job_name", "python", "collect_mem", "port_lock_dir", "remote_cwd"):
            if key in data:
                options[key] = _optional_str(data[key])
        for key in ("cpus_per_task", "array_parallelism", "cases_per_task"):
            if key in data:
                options[key] = _optional_int(data[key], key)
        for key in ("extra_sbatch_args", "env_passthrough", "setup"):
            if key in data:
                value = data[key]
                if value is None and key == "env_passthrough":
                    options[key] = None  # the mode's default, below
                    continue
                if isinstance(value, str):
                    value = [value]
                options[key] = tuple(str(v) for v in (value or ()))
        if "env" in data:
            options["env"] = {str(k): str(v) for k, v in (data["env"] or {}).items()}
        remote = options.get("remote_cwd") is not None
        if remote:
            _check_remote_path(options["remote_cwd"], "remote_cwd")
            if options.get("python") is None:
                raise ValueError(
                    "launcher.remote_cwd needs launcher.python, the interpreter "
                    "on the nodes: a path relative to remote_cwd "
                    "(.venv/bin/python), ~/..., absolute, or a command on the "
                    "node's PATH (python3, with launcher.setup activating its "
                    "environment); the submitting one's path is this host's"
                )
        if options.get("python") is not None:
            _check_remote_path(options["python"], "python")
        # null: the default of the mode.  With remote_cwd nothing of the
        # submitting host's environment is the nodes' (docs/slurm.md).
        if data.get("export") is not None and data.get("export") != "":
            options["export"] = str(data["export"])
        elif remote:
            options["export"] = "NONE"
        if options.get("env_passthrough") is None:
            options.pop("env_passthrough", None)
            if remote:
                options["env_passthrough"] = ()
        for key in ("wait", "headless", "collect_job", "cancel_on_interrupt"):
            if key in data:
                options[key] = bool(data[key])
        if "start_stagger_seconds" in data:
            options["start_stagger_seconds"] = max(
                0.0, float(data["start_stagger_seconds"] or 0.0)
            )
        if "poll_seconds" in data:
            options["poll_seconds"] = max(0.1, float(data["poll_seconds"]))
        for key in ("signal_grace_seconds", "port_base", "port_slots"):
            if key in data:
                options[key] = int(data[key])
        settings = cls(**options)
        if settings.port_slots < 1:
            raise ValueError(
                f"launcher.port_slots must be at least 1 (got {settings.port_slots}): "
                "each task takes a slot of ports"
            )
        if (
            settings.port_base < 1024
            or settings.port_base + (settings.port_slots * 10) > 65535
        ):
            raise ValueError(
                "launcher.port_base/port_slots must keep every task's ports within "
                f"1024-65535 (got base {settings.port_base}, {settings.port_slots} slots)"
            )
        return settings

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable copy, as the manifest stores it."""
        data = dataclasses.asdict(self)
        data["env"] = dict(self.env)
        for key in ("extra_sbatch_args", "env_passthrough", "setup"):
            data[key] = list(data[key])
        return data

    def replace(self, **changes: Any) -> "SlurmSettings":
        """A copy with *changes* (only the ones that are not ``None``)."""
        data = {**self.to_dict(), **{k: v for k, v in changes.items() if v is not None}}
        return SlurmSettings.from_mapping(data)

    def chunk_size(self, cases: int) -> int:
        """How many cases one task runs, for a batch of *cases*."""
        if self.cases_per_task is not None:
            return self.cases_per_task
        if self.array_parallelism is None or cases <= 0:
            return 1
        return max(1, math.ceil(cases / self.array_parallelism))
