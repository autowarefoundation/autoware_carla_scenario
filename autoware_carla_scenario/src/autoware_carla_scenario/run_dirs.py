"""What a run's Hydra command line asks for, and the run directory it gets.

Whether an argv asks Hydra for a multirun (:func:`multirun_requested_in_argv`),
for no job at all (:func:`asks_for_no_job`) or names a directory itself
(:func:`sets_hydra_key`), and the Hydra-style timestamped directory reserved
for a run, a run's alone (:func:`reserve_timestamped_dir`, #1223, #1359).

Shared by ``examples.run``, which reserves the directory of a ``scenario``
run, sweep or batch before Hydra starts, and ``ui.runner``, which reserves
the ``multirun/...`` directory of the viewer's grouped runs.  Split out of
``examples.run``, which needs Hydra and omegaconf, so that the library check
compiles it (``docs/typecheck.md``, "Checking the framework itself").  The
callers read the clock (``datetime.now()``) and pass it in.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

__all__ = [
    "HYDRA_NO_JOB_FLAGS",
    "asks_for_no_job",
    "is_multirun_mode_override",
    "multirun_dir",
    "multirun_requested_in_argv",
    "reserve_timestamped_dir",
    "sets_hydra_key",
    "timestamped_dir",
]


#: Hydra's command-line flags that print something and run no job: a run
#: directory reserved for them would be left empty.
HYDRA_NO_JOB_FLAGS = (
    "--help",
    "-h",
    "--hydra-help",
    "--version",
    "--cfg",
    "-c",
    "--info",
    "-i",
    "--shell-completion",
    "-sc",
)


def is_multirun_mode_override(arg: str) -> bool:
    """Return ``True`` for a ``hydra.mode=MULTIRUN`` override.

    Hydra's documented alternative to ``--multirun``.  The value is matched as
    Hydra matches it, by the enum member's name: Hydra refuses
    ``hydra.mode=multirun``.
    """
    return arg.lstrip("+") == "hydra.mode=MULTIRUN"


def multirun_requested_in_argv(argv: list[str]) -> bool:
    """Return ``True`` when *argv* asks Hydra for a multirun.

    For use before Hydra composes, so a ``hydra.mode: MULTIRUN`` set only in
    a config file is not seen here.
    """
    return any(
        arg in ("--multirun", "-m") or is_multirun_mode_override(arg) for arg in argv
    )


def asks_for_no_job(argv: list[str]) -> bool:
    """Return ``True`` when *argv* (a full ``sys.argv``: its first item, the
    program, is skipped) asks Hydra for something other than a job
    (``--help``, ``--cfg``, ... :data:`HYDRA_NO_JOB_FLAGS`, alone or as
    ``--flag=value``)."""
    return any(arg.split("=", 1)[0] in HYDRA_NO_JOB_FLAGS for arg in argv[1:])


def sets_hydra_key(argv: list[str], key: str) -> bool:
    """Return ``True`` when an override in *argv* sets *key* (``key=``,
    ``+key=``, ``++key=`` or ``~key``)."""
    for arg in argv:
        name = arg.lstrip("+~").split("=", 1)[0]
        if name == key:
            return True
    return False


def timestamped_dir(root: str, now: datetime) -> str:
    """``{root}/{date}/{time}`` of *now*: ``f"{root}/{now:%Y-%m-%d}/{now:%H-%M-%S}"``.

    Spelled out from the fields, as Codon formats no datetime; the same string
    for every year from 1000 on (strftime's ``%Y`` pads years before 1000
    only on some Pythons).
    """
    date = f"{now.year:04d}-{now.month:02d}-{now.day:02d}"
    time = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    return f"{root}/{date}/{time}"


def multirun_dir(now: datetime) -> str:
    """``multirun/{date}/{time}`` of *now* (:func:`timestamped_dir`)."""
    return timestamped_dir("multirun", now)


def reserve_timestamped_dir(
    base: Path, root: str, now: datetime, dangling_links_taken: bool = True
) -> str:
    """Create a ``{root}/{date}/{time}`` of *now* under *base* that is this
    run's alone, and return it relative to *base*.

    Created with ``mkdir``, which fails when it exists, so a run started in
    the same second as another (a CLI run or sweep, a batch, another viewer's
    run) never shares it (#1223, #1359): the first gets
    :func:`timestamped_dir`'s ``{time}``, the next ``{time}-1``, then
    ``{time}-2``, and so on.

    Relative, as Hydra's own ``outputs/...`` and ``multirun/...`` defaults
    are, so it can be handed to Hydra as an override whatever characters the
    working directory's path holds.

    A name is taken when ``mkdir`` fails and the path exists; with
    *dangling_links_taken* (the default), a symbolic link that resolves to
    nothing takes it too, as a ``FileExistsError`` says, which Codon does not
    name.  ``ui.runner`` passes ``False``: its copy raised there.  Any other
    failure (permissions, a full disk) is raised.
    """
    name = timestamped_dir(root, now)
    (base / name).parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        candidate = name if suffix == 0 else f"{name}-{suffix}"
        path = base / candidate
        try:
            path.mkdir()
        except OSError:
            if not (path.exists() or (dangling_links_taken and path.is_symlink())):
                raise
            suffix += 1
            continue
        return candidate
