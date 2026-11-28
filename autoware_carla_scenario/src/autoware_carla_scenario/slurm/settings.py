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
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..config_flags import check_flag
from ..seed import check_whole_number
from .port_slots import PORT_SLOT_WIDTH, SlotPorts

__all__ = [
    "DEFAULT_ENV_PASSTHROUGH",
    "LAUNCHER_TYPES",
    "MAX_SIGNAL_GRACE_SECONDS",
    "SENSITIVE_NAME_PATTERNS",
    "SlurmSettings",
    "launcher_requested_in_argv",
    "launcher_type",
    "require_sbatch",
    "snapshot_environment",
]

#: The most ``launcher.signal_grace_seconds`` may be: sbatch's ``--signal``
#: takes a ``sig_time`` of 0 to 65535.
MAX_SIGNAL_GRACE_SECONDS = 65535

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
        "*PASSWD*",
        "*CREDENTIAL*",
        "*DSN*",
    ]
)


def _sensitive(name: str) -> bool:
    upper = name.upper()
    return any(fnmatch.fnmatchcase(upper, p) for p in SENSITIVE_NAME_PATTERNS)


def launcher_type(cfg: Mapping[str, Any] | None) -> str:
    """``launcher.type`` of *cfg*: ``local`` when it has no ``launcher``, or
    its ``type`` is missing or null.

    Raises:
        ValueError: The type is not one of :data:`LAUNCHER_TYPES`: ``false``,
            ``0`` or ``""`` is no launcher, not ``local``.
    """
    launcher = None if cfg is None else cfg.get("launcher")
    if launcher is None:
        return "local"
    kind = launcher.get("type")
    if kind is None:
        return "local"
    if not isinstance(kind, str) or kind not in LAUNCHER_TYPES:
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


#: SLURM's default ``MaxArraySize``: an array's task ids run from 0 to one less
#: than it, so an array of at most this many tasks (``--array=0-1000``).
DEFAULT_MAX_ARRAY_SIZE = 1001


def parse_max_array_size(config: str) -> int | None:
    """``MaxArraySize`` in *config* (what ``scontrol show config`` prints), or
    ``None`` when it names none, or none of at least 1."""
    for line in config.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "MaxArraySize":
            text = value.strip()
            if text.isascii() and text.isdigit() and int(text) >= 1:
                return int(text)
            return None
    return None


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


#: A name a job script can ``export``: what the shell takes as a variable name
#: (#1891).  The run panel checks the same (``ui.slurm_options``).
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_env_name(name: str) -> bool:
    """Whether *name* is one a job script can ``export``."""
    return _ENV_NAME_PATTERN.fullmatch(name) is not None


def snapshot_environment(
    patterns: Iterable[str],
    environ: Mapping[str, str] | None = None,
    *,
    explicit: Iterable[str] = (),
) -> dict[str, str]:
    """The variables of *environ* (``os.environ``) whose names match *patterns*.

    A name :data:`SENSITIVE_NAME_PATTERNS` matches is left out, unless one of
    *patterns* is that very name or it is in *explicit* (``launcher.env``).
    So is one no shell takes as a variable name (``BASH_FUNC_f%%``, which a
    pattern can match), which the job script could not set again.
    """
    env = os.environ if environ is None else environ
    patterns = tuple(patterns)
    named = set(explicit) | {p for p in patterns if not any(c in p for c in "*?[")}
    return {
        name: value
        for name, value in sorted(env.items())
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        and _is_env_name(name)
        and (name in named or not _sensitive(name))
    }


def _optional_str(value: Any, name: str) -> str | None:
    """A text setting: ``None`` when null or blank; a number is its text.

    A boolean (``partition: true``), a list or a mapping is no text an
    ``sbatch`` option takes, so it is refused rather than written as
    ``True`` or ``['a']``.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"launcher.{name} must be text, got {value!r}")
    return str(value)


def _check_time_is_text(value: Any) -> None:
    """Refuse a number as ``launcher.time``.

    YAML 1.1, which the config is read as, takes an unquoted ``12:00:00`` as
    the base-60 number 43200 (and ``120:00`` as 7200), which sbatch would
    read as minutes: 30 days, not 12 hours.  Such a number cannot be told
    from one written as minutes, so a number is refused (#2483).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    raise ValueError(
        f"launcher.time must be text, got the number {value!r}: YAML reads an "
        "unquoted 12:00:00 as the number 43200 (seconds, in base 60), which "
        'sbatch would take as minutes. Quote it: time: "12:00:00" (or '
        'time: "90" for 90 minutes; on the command line '
        "launcher.time=\"'90'\")"
    )


#: sbatch's ``--time`` forms: ``minutes``, ``minutes:seconds``,
#: ``hours:minutes:seconds`` (the leading field of any width: ``168:00:00``),
#: ``days-hours``, ``days-hours:minutes`` and ``days-hours:minutes:seconds``;
#: each field of any number of ASCII digits, as sbatch reads them
#: (``time_str2secs``: ``1:2`` is a minute and two seconds), so that nothing
#: sbatch takes is refused here.
_TIME_PATTERN = re.compile(r"^[0-9]+(-[0-9]+)?(:[0-9]+){0,2}$")
#: The ``--time`` that sets no limit, in any case (``unlimited``), and
#: sbatch's ``-1``.
_TIME_NO_LIMIT: tuple[str, ...] = ("UNLIMITED", "INFINITE", "-1")


def _check_time_format(time: str | None, name: str) -> None:
    """Refuse a *time* that is not sbatch's ``--time``, if set, naming it
    *name* (``launcher.time``; the run panel's ``time``): before the batch's
    manifest and job script are written, not by sbatch (#2601).

    Raises:
        ValueError: It is not.
    """
    if time is None or time.upper() in _TIME_NO_LIMIT:
        return
    if _TIME_PATTERN.fullmatch(time) is None:
        raise ValueError(
            f"{name} is sbatch's --time (minutes, minutes:seconds, "
            "hours:minutes:seconds, days-hours, days-hours:minutes, "
            f"days-hours:minutes:seconds or UNLIMITED), not {time!r}"
        )


def _ends_in_backslash(line: str) -> bool:
    """Whether *line* ends in an unescaped backslash: the job script holds a
    ``launcher.setup`` line as written, so it would run on into the script's
    next line (#1893)."""
    return (len(line) - len(line.rstrip("\\"))) % 2 == 1


def _str_list(value: Any, name: str) -> tuple[str, ...]:
    """A list setting: a list of text, or one string as a list of one.

    A null or blank item is refused: it would land as an empty ``sbatch``
    argument (or an empty line), not as no item.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        items: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"launcher.{name} must be a list, got {value!r}")
    texts: list[str] = []
    for item in items:
        text = _optional_str(item, name)
        if text is None or not text.strip():
            raise ValueError(
                f"launcher.{name} must be a list of non-blank text, "
                f"got the item {item!r}"
            )
        texts.append(text)
    return tuple(texts)


#: sbatch's short options that take no argument: in a group of short options
#: (``-vM x``, ``-vMx``) any other letter takes the rest of the group, or the
#: next argument, as its value (``-k``'s value is optional: ``-kM`` gives it M).
_SBATCH_SHORT_FLAGS: str = "HhOQsuVvW"


def _names_clusters(arg: str) -> bool:
    """Whether *arg* is ``--clusters``/``--cluster`` or a group of short
    options holding ``-M`` (``-Mx``, ``-vMx``, ``-v -M x``'s ``-vM``)."""
    name = arg.split("=", 1)[0]
    if name in ("--clusters", "--cluster"):
        return True
    if not arg.startswith("-") or arg.startswith("--"):
        return False
    for letter in arg[1:]:
        if letter == "M":
            return True
        if letter not in _SBATCH_SHORT_FLAGS:
            return False  # the rest of the group is this option's value
    return False


def _check_no_clusters(args: tuple[str, ...]) -> None:
    """Refuse ``--clusters``/``--cluster``/``-M`` in ``extra_sbatch_args``.

    The batch would be submitted to another cluster, while ``squeue``,
    ``scancel`` and the collect job's dependency ask the default one: follow
    would collect at once, ``resubmit`` submit duplicates while the array runs
    and ``cancel`` miss it (#2247).
    """
    for arg in args:
        if _names_clusters(arg):
            raise ValueError(
                f"launcher.extra_sbatch_args: {arg!r} submits to another "
                "cluster (sbatch --clusters/-M), which launcher=slurm does not "
                "support: squeue, scancel and the collect job would look for "
                "the batch on the default cluster. Submit from a host of the "
                "cluster the batch is to run on"
            )


def _check_remote_path(value: str, name: str) -> None:
    """Refuse a path the job script cannot expand on the node: ``~user/``."""
    if "\n" in value or "\0" in value:
        raise ValueError(f"launcher.{name} must be one line, not {value!r}")
    if value.startswith("~") and value != "~" and not value.startswith("~/"):
        raise ValueError(
            f"launcher.{name} {value!r}: only ~ and ~/... (the job user's own "
            "home on the node) are expanded, not ~user"
        )


def _whole_number(value: Any, name: str, minimum: int = 1) -> int:
    """A whole-number setting, at least *minimum*.

    ``2``, ``2.0`` or the text ``"2"`` (seed.check_whole_number, as every
    whole-number setting of a run is read): ``int()`` alone took ``2.7`` as
    2 and ``true`` as 1.  A boolean, a fraction, NaN, infinity, a blank or
    other text, or null is refused.
    """
    key = f"launcher.{name}"
    number = value
    if isinstance(value, str):
        text = value.strip()
        try:
            number = int(text)
        except ValueError:
            try:
                number = float(text)
            except ValueError:
                raise ValueError(f"{key} must be a whole number, got {value!r}")
    whole = check_whole_number(key, number)
    if whole < minimum:
        raise ValueError(f"{key} must be at least {minimum} (got {value!r})")
    return whole


def _optional_int(value: Any, name: str, minimum: int = 1) -> int | None:
    """:func:`_whole_number`, or ``None`` when null."""
    if value is None:
        return None
    return _whole_number(value, name, minimum)


def _seconds(value: Any, name: str) -> float:
    """A number of seconds: finite, at least 0; a number or its text.

    A boolean is no number of seconds (``float(True)`` is 1.0); inf would
    make a task wait (or the poll sleep) forever, NaN would read as 0
    (#784).
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(
            f"launcher.{name} must be a finite number of seconds, "
            f"at least 0 (got {value!r})"
        )
    try:
        seconds = float(value)
    except ValueError:
        seconds = math.nan
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            f"launcher.{name} must be a finite number of seconds, "
            f"at least 0 (got {value!r})"
        )
    return seconds


def _flag(value: Any, name: str) -> bool:
    """A true/false setting: a boolean, or its spelling (``"false"``, ``0``).

    ``bool()`` read ``"false"`` as true; anything that spells neither (a
    float, a list, null, ``"ture"``) is refused (config_flags.check_flag,
    as every switch of a run is read).
    """
    return check_flag(f"launcher.{name}", value)


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
            ValueError: A key no field has, or a value of the wrong type or
                out of range, naming it (``launcher.<key>``).
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
        if "time" in data:
            _check_time_is_text(data["time"])
        for key in ("partition", "account", "qos", "time", "mem", "gres"):
            if key in data:
                options[key] = _optional_str(data[key], key)
        _check_time_format(options.get("time"), "launcher.time")
        for key in ("job_name", "python", "collect_mem", "port_lock_dir", "remote_cwd"):
            if key in data:
                options[key] = _optional_str(data[key], key)
        for key in ("cpus_per_task", "array_parallelism", "cases_per_task"):
            if key in data:
                options[key] = _optional_int(data[key], key)
        for key in ("extra_sbatch_args", "env_passthrough", "setup"):
            if key in data:
                value = data[key]
                if value is None and key == "env_passthrough":
                    options[key] = None  # the mode's default, below
                    continue
                options[key] = _str_list(value, key)
        _check_no_clusters(options.get("extra_sbatch_args", ()))
        for line in options.get("setup", ()):
            # The job script holds each line as written: one ending in an
            # unescaped backslash would run on into the script's next line
            # (`set -u` with remote_cwd), as the run panel refuses (#1893).
            if _ends_in_backslash(line):
                raise ValueError(
                    f"launcher.setup line {line!r} ends in a backslash: write "
                    "each command on a line of its own"
                )
        if "env" in data:
            # Only null is no variables: [], 0 or "" is no mapping.
            env = {} if data["env"] is None else data["env"]
            if not isinstance(env, Mapping):
                raise ValueError(f"launcher.env must be a mapping, got {env!r}")
            values: dict[str, str] = {}
            for k, v in env.items():
                # Text or a number, as _optional_str takes it; "" is a value
                # (export NAME=''), null is not "None".
                if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                    raise ValueError(f"launcher.env.{k} must be text, got {v!r}")
                values[str(k)] = str(v)
            options["env"] = values
            for name in options["env"]:
                # Written into the job script as `export NAME=...`: a name no
                # shell takes is an error there, or a command (#1891).
                if not _is_env_name(name):
                    raise ValueError(
                        f"launcher.env: {name!r} is not an environment variable name"
                    )
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
        export = _optional_str(data.get("export"), "export")
        if export is not None:
            options["export"] = export
        elif remote:
            options["export"] = "NONE"
        if options.get("env_passthrough") is None:
            options.pop("env_passthrough", None)
            if remote:
                options["env_passthrough"] = ()
        for key in ("wait", "headless", "collect_job", "cancel_on_interrupt"):
            if key in data:
                options[key] = _flag(data[key], key)
        for key in ("start_stagger_seconds", "poll_seconds"):
            if key not in data:
                continue
            raw = data[key]
            if raw is None and key == "start_stagger_seconds":
                raw = 0.0  # null: no stagger
            seconds = _seconds(raw, key)
            # A poll that never sleeps would spin: at least 0.1 s.
            options[key] = max(0.1, seconds) if key == "poll_seconds" else seconds
        # SIGTERM 0 s before the end is allowed (sbatch takes @0); a
        # negative grace, or one above the 65535 of --signal's sig_time, only
        # sbatch refused, after the batch was written (#2483).
        if "signal_grace_seconds" in data:
            grace = _whole_number(
                data["signal_grace_seconds"], "signal_grace_seconds", 0
            )
            if grace > MAX_SIGNAL_GRACE_SECONDS:
                raise ValueError(
                    "launcher.signal_grace_seconds must be at most "
                    f"{MAX_SIGNAL_GRACE_SECONDS}, what sbatch --signal takes "
                    f"(got {data['signal_grace_seconds']!r})"
                )
            options["signal_grace_seconds"] = grace
        if "port_base" in data:
            options["port_base"] = _whole_number(data["port_base"], "port_base", 1024)
        if "port_slots" in data:
            # At least 1: each task takes a slot of ports.
            options["port_slots"] = _whole_number(data["port_slots"], "port_slots")
        settings = cls(**options)
        # The last slot starts PORT_SLOT_WIDTH * (slots - 1) above the base
        # and uses the first ports of its width (port_slots.SlotPorts).
        last = settings.port_base + PORT_SLOT_WIDTH * (settings.port_slots - 1)
        highest = max(SlotPorts(settings.port_slots - 1, last).ports())
        if settings.port_base < 1024 or highest > 65535:
            raise ValueError(
                "launcher.port_base/port_slots must keep every task's ports within "
                f"1024-65535 (got base {settings.port_base}, {settings.port_slots} "
                f"slots: ports up to {highest})"
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

    def chunk_size(self, cases: int, max_tasks: int = DEFAULT_MAX_ARRAY_SIZE) -> int:
        """How many cases one task runs, for a batch of *cases*.

        Without ``cases_per_task``, never fewer than it takes to keep the
        array within *max_tasks* tasks (SLURM's ``MaxArraySize``, #1347).
        ``cases_per_task`` is taken as it is, whatever the array's size.
        """
        if self.cases_per_task is not None:
            return self.cases_per_task
        if cases <= 0:
            return 1
        size = 1
        if self.array_parallelism is not None:
            size = max(1, math.ceil(cases / self.array_parallelism))
        if max_tasks >= 1:
            size = max(size, math.ceil(cases / max_tasks))
        return size
