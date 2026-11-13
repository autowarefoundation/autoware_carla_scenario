"""Submitting a batch's cases as a SLURM array, and following it (docs/slurm.md).

The submitting side of ``launcher=slurm``: :func:`submit_batch` is what
``scenario --multirun ... launcher=slurm`` ends in, once the batch is expanded,
labelled, built and checked as a local batch is.  It writes the manifest,
submits the array (:func:`submit_cases`) and, unless told not to wait, follows
it until it ends and collects it (:func:`follow`, :mod:`.follow`'s, as
:func:`after_submit` is).  ``scenario-slurm resubmit`` submits the cases
that did not finish again the same way.

Only the standard library here (and OmegaConf, inside :func:`submit_batch`):
the ``scenario-slurm`` commands import this without CARLA.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from . import scripts

# Following a submission, and what a submitter says of it once submitted,
# are slurm.follow's, which the library check compiles; these are named here
# as they were before, and called here as deep in the stack as they were.
from .follow import after_submit as after_submit
from .follow import follow as follow
from .manifest import (
    MANIFEST_VERSION,
    BatchLayout,
    Manifest,
    config_files,
    sequential_stems,
    write_json,
)

# SLURM's command-line tools are run in slurm.queue (#1634), which the library
# check compiles; these are named here as they were before.
from .queue import ACTIVE_STATES as ACTIVE_STATES
from .queue import CANCEL_WAIT_SECONDS as CANCEL_WAIT_SECONDS
from .queue import SbatchTimedOut as SbatchTimedOut
from .queue import _array_refused as _array_refused
from .queue import _chunks as _chunks
from .queue import _run_sbatch as _run_sbatch
from .queue import _wait_until_gone as _wait_until_gone
from .queue import cancel_jobs as cancel_jobs
from .queue import max_array_size as max_array_size
from .queue import squeue_states as squeue_states

# The job scripts and sbatch command lines are built in slurm.scripts (#732),
# which the library check compiles; these are named here as they were before.
from .scripts import CLI_MODULE as CLI_MODULE
from .scripts import REMOTE_EARLY_OUTPUT_DIR as REMOTE_EARLY_OUTPUT_DIR
from .scripts import WORKER_MODULE as WORKER_MODULE
from .scripts import ScriptInputs
from .scripts import _expand_ranges as _expand_ranges
from .scripts import _shell_path as _shell_path
from .scripts import relative_batch_dir as relative_batch_dir
from .scripts import sbatch_command as sbatch_command
from .scripts import squeue_task_states as squeue_task_states
from .settings import (
    DEFAULT_MAX_ARRAY_SIZE,
    SlurmSettings,
    snapshot_environment,
)
from .settings import parse_max_array_size as parse_max_array_size
from .settings import require_sbatch as require_sbatch

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
    batch_settings: dict[str, Any] | None = None,
) -> int:
    """Submit a built and checked batch as a SLURM array; the exit status.

    *cases*, *labels*, *case_overrides* and *configs* are what ``run_batch``
    made of the command line, one per case, and *scenario_classes* the class
    of the scenario each built.  The outputs go to *output_dir* as a local
    batch's would.  *wait* overrides ``launcher.wait``.  *batch_settings* is
    which cases decide the batch-level settings
    (``examples.run.batch_settings_cases``): every task decides them from
    those, as a local batch does, whichever cases it runs.
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    layout = BatchLayout(output_dir)
    if layout.manifest.exists():
        # A fresh batch only: a resubmit adds a submission to the existing
        # manifest (submit_cases), and never comes here.  Writing over another
        # batch's manifest would hand its tasks this batch's cases (#1223).
        raise FileExistsError(
            f"{output_dir} already holds a SLURM batch ({layout.manifest}); "
            "a new batch needs a directory of its own (to run some of that "
            f"batch's cases again: scenario-slurm resubmit {output_dir})"
        )

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
        # Which cases decide the batch's queue, Rerun converter, canvas and
        # metrics (examples.run.batch_settings_cases).  An older worker
        # ignores it, and decides them from its own cases as it always did.
        **({} if batch_settings is None else {"batch_settings": batch_settings}),
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


def _script_inputs(
    manifest: dict[str, Any] | Manifest, *, python: bool, cwd: bool, env: Any
) -> ScriptInputs:
    """What the scripts read of *manifest*, as they read it before #732.

    The manifest's values are handed over as they are, so that a script
    fails, or not, as it did when it read the manifest itself: *python* and
    *cwd* say whether the script indexes that key (a ``KeyError`` here when
    the manifest lacks it), *env* is what it read as the environment.
    """
    return ScriptInputs(
        python=manifest["python"] if python else manifest.get("python", ""),
        cwd=manifest["cwd"] if cwd else manifest.get("cwd", ""),
        env=env,
        batch_dir_relative=manifest.get("batch_dir_relative"),
    )


def job_script(
    layout: BatchLayout,
    manifest: dict[str, Any] | Manifest,
    settings: SlurmSettings,
    submission: int,
) -> str:
    """The shell script every task of submission *submission* runs
    (:func:`.scripts.job_script`)."""
    inputs = _script_inputs(
        manifest,
        python=not settings.python,
        cwd=settings.remote_cwd is None,
        env=manifest.get("env", {}),
    )
    return scripts.job_script(layout, inputs, settings, submission)


def collect_script(
    manifest: dict[str, Any] | Manifest,
    settings: SlurmSettings,
    layout: BatchLayout | None = None,
) -> str:
    """The script of the job collecting the batch once its array has ended
    (:func:`.scripts.collect_script`); *layout* is the batch's, which a batch
    submitted without ``remote_cwd`` needs."""
    inputs = _script_inputs(
        manifest,
        python=not settings.python,
        cwd=settings.remote_cwd is None,
        env=manifest.get("env") or {},
    )
    return scripts.collect_script(inputs, settings, layout)


def _collect_command(
    layout: BatchLayout,
    manifest: dict[str, Any] | Manifest,
    settings: SlurmSettings,
    job: str,
    script: Path | None = None,
) -> list[str]:
    """The ``sbatch`` command of the job collecting the batch after *job*,
    running *script* (:func:`.scripts._collect_command`)."""
    inputs = _script_inputs(
        manifest, python=False, cwd=settings.remote_cwd is None, env={}
    )
    return scripts._collect_command(layout, inputs, settings, job, script)


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
    manifest: dict[str, Any] | Manifest,
    settings: SlurmSettings,
    case_indices: list[int],
    before_sbatch: Callable[[int], None] | None = None,
) -> int:
    """Submit *case_indices* of the batch as one array; the submission's index.

    The submission's index is reserved first, its record created exclusively
    (:meth:`BatchLayout.reserve_submission`), so that two submitters of one
    batch at once each get their own (#1352).  *before_sbatch*, given that
    index, runs next; should it raise, nothing is submitted.  A resubmit
    checks there that no other submission of the batch is in flight
    (``scenario-slurm resubmit``).  Whatever fails before the job id is
    known removes the record, and with it the reservation.

    The cases are split into tasks of ``launcher.cases_per_task`` (in order),
    which SLURM runs at most ``launcher.array_parallelism`` at a time.  Without
    ``cases_per_task``, into no more tasks than the cluster's ``MaxArraySize``
    (:func:`max_array_size`).

    Raises:
        RuntimeError: ``launcher.cases_per_task`` makes more tasks than
            ``MaxArraySize`` allows (nothing is submitted), or sbatch refused
            the array (its submission record removed).
        SbatchTimedOut: sbatch did not return in time (a RuntimeError); the
            submission record is kept, as the job may have been taken.
    """
    if not case_indices:
        raise ValueError("no case to submit")
    limit = max_array_size()
    size = settings.chunk_size(len(case_indices), limit)
    chunks = _chunks(list(case_indices), size)
    if len(chunks) > limit:
        least = -(-len(case_indices) // limit)
        raise RuntimeError(
            f"{len(case_indices)} case(s) at launcher.cases_per_task={size} make "
            f"{len(chunks)} array tasks, more than the cluster's MaxArraySize "
            f"({limit}) allows: set launcher.cases_per_task to at least {least}, "
            "or unset it to have the cases spread within the limit (docs/slurm.md)"
        )
    submission: dict[str, Any] = {
        "index": None,
        "created": _now(),
        "submitted_at": time.time(),
        "cases": list(case_indices),
        "chunks": chunks,
        "launcher": settings.to_dict(),
        "job_id": None,
        "collect_job_id": None,
    }
    # Written before sbatch: a task that starts at once reads its chunk here.
    # Created exclusively, which reserves the index; job_id None until
    # sbatch returns marks it in flight (manifest.sbatch_in_flight).
    index = layout.reserve_submission(submission)
    try:
        script = layout.job_script(index)
        _write_script(script, job_script(layout, manifest, settings, index))
        layout.logs.mkdir(parents=True, exist_ok=True)
        (layout.root / "tasks").mkdir(parents=True, exist_ok=True)
        (layout.root / "results").mkdir(parents=True, exist_ok=True)
        if before_sbatch is not None:
            before_sbatch(index)
        command = sbatch_command(layout, settings, script, len(chunks))
        submission["sbatch"] = command
        submission["job_id"] = _run_sbatch(command)
    except SbatchTimedOut:
        # The job may have been taken (#1786): keep the record its tasks
        # read, job_id None, so it counts as in flight (refusing resubmits)
        # until SBATCH_IN_FLIGHT_SECONDS, and its index is not reused.  Say
        # in it that sbatch timed out (#1791), so status, wait and the viewer
        # tell it from one whose sbatch has not returned yet; atomically, so
        # a task reading its chunk never sees half a record.
        submission["sbatch_timed_out"] = True
        submission["sbatch_timed_out_at"] = _now()
        try:
            write_json(layout.submission(index), submission)
        except OSError:
            logger.warning(
                "Could not record in %s that sbatch timed out",
                layout.submission(index),
                exc_info=True,
            )
        raise
    except RuntimeError as exc:
        layout.submission(index).unlink(missing_ok=True)
        if not _array_refused(str(exc)):
            raise
        # The limit scontrol did not tell (#1347): say what to change.
        raise RuntimeError(
            f"{exc}\n  The array has {len(chunks)} task(s) of up to {size} "
            f"case(s); a cluster's MaxArraySize (`scontrol show config`, "
            f"{DEFAULT_MAX_ARRAY_SIZE} by default) caps an array's tasks: set "
            "launcher.cases_per_task so that fewer tasks run the cases "
            "(docs/slurm.md)"
        ) from exc
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
        collect_path = layout.root / "jobs" / f"{index}-collect.sh"
        try:
            # Inside the try: the array is submitted already, so a script
            # that cannot be written is reported like a refused sbatch.
            _write_script(collect_path, collect_script(manifest, settings, layout))
            collect = _collect_command(
                layout, manifest, settings, submission["job_id"], collect_path
            )
            submission["collect_sbatch"] = collect
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
