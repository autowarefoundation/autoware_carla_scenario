"""A SLURM task of a ``launcher=slurm`` batch: its cases, on a CARLA of its own.

Run by the job script, not by hand::

    python -m autoware_carla_scenario.slurm.worker <manifest> <submission> <task>

It takes a slot of ports (:mod:`.ports`), composes each of its cases' configs
from the scenario and overrides the manifest records -- and refuses to run one
that no longer resolves to the config the batch was submitted with -- builds
them, and runs them one after another on one CARLA through the same queue a
local batch uses, writing their outputs into the batch's directory under the
names a local batch gives them.  Each case's outcome is recorded as soon as it
has run (``slurm/results/<i>.json``); the task's progress is in
``slurm/tasks/<submission>-<task>.json``.

SIGTERM (``scancel``, or ``--signal`` before the time limit) ends the case
running and stops CARLA; if the worker is killed before it has (the SIGKILL
after the scheduler's grace period), :class:`.ports.CarlaWatchdog` stops it.
The cases not run by then have no outcome and are run again by
``scenario-slurm resubmit``.
"""

from __future__ import annotations

import contextlib
import errno
import importlib
import json
import logging
import math
import os
import signal
import socket
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .manifest import (
    RUN_FILE_SUFFIXES,
    BatchLayout,
    config_key,
    file_sha256,
    write_json,
)
from .ports import (
    CarlaWatchdog,
    PortSlot,
    acquire_slot,
    first_slot,
    kill_carla_on_port,
)
from .settings import SlurmSettings
from .task_timing import (
    rss_report,
    start_delay,
    status_fields,
    tree_rss_kb,
)
from .task_timing import with_port as _with_port

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from ..rerun import RerunConverter

__all__ = ["main", "wait_for_start_turn", "worker_overrides"]

logger = logging.getLogger("autoware_carla_scenario.slurm.worker")

#: How often the resident memory of the task's processes is sampled.
_RSS_PERIOD_S = 5.0


def _now() -> str:
    from datetime import datetime  # noqa: PLC0415

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _process_tree_rss_kb(root: int) -> tuple[int, int]:
    """(this process's RSS, its descendants' RSS) in KiB, from /proc."""
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        pid = int(entry.name)
        ppid, vmrss = status_fields(status)
        if ppid is not None:
            parents[pid] = ppid
        if vmrss is not None:
            rss[pid] = vmrss
    return tree_rss_kb(parents, rss, root)


class _RssSampler(threading.Thread):
    """Samples the task's resident memory (worker, and CARLA under it)."""

    def __init__(self) -> None:
        super().__init__(name="slurm-rss-sampler", daemon=True)
        self.peak_worker_kb = 0
        self.peak_children_kb = 0
        self.peak_total_kb = 0
        self._stop_event = threading.Event()

    def run(self) -> None:
        pid = os.getpid()
        while not self._stop_event.is_set():
            try:
                own, children = _process_tree_rss_kb(pid)
            except Exception:  # noqa: BLE001 - a measurement, nothing depends on it
                own = children = 0
            self.peak_worker_kb = max(self.peak_worker_kb, own)
            self.peak_children_kb = max(self.peak_children_kb, children)
            self.peak_total_kb = max(self.peak_total_kb, own + children)
            self._stop_event.wait(_RSS_PERIOD_S)

    def stop(self) -> None:
        self._stop_event.set()

    def to_dict(self) -> dict[str, float]:
        return rss_report(
            self.peak_worker_kb, self.peak_children_kb, self.peak_total_kb
        )


def worker_overrides(cfg: "DictConfig", slot: PortSlot) -> dict[str, Any]:
    """Point *cfg* at the task's ports (in place); what it changed.

    The CARLA server and its TrafficManager, and the bridges the run itself
    hosts.  ``driver.address`` is left alone: a policy the run serves itself
    (``driver.policy``) listens on a port the OS picks, and one that is not
    is a server someone else runs, which every task dials.
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    changes: dict[str, Any] = {
        "server.port": slot.rpc,
        "traffic_manager.port": slot.traffic_manager,
    }
    autoware = cfg.get("autoware")
    if autoware is not None and autoware.get("address"):
        changes["autoware.address"] = _with_port(autoware.address, slot.autoware_bridge)
    driver = cfg.get("driver")
    bridge = None if driver is None else driver.get("bridge")
    if bridge is not None and bridge.get("address"):
        changes["driver.bridge.address"] = _with_port(
            bridge.address, slot.driver_bridge
        )
    for key, value in changes.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return changes


class _SameFiles:
    """Whether a string of a case's config that differs here names the same
    thing: the submitting directory's path seen from this job's working
    directory, or a file with the content it had where it was submitted."""

    def __init__(self, files: dict[str, str], submit_cwd: str | None) -> None:
        self.files = files
        self.prefixes: list[tuple[str, str]] = []
        if submit_cwd:
            here = {os.getcwd(), os.environ.get("PWD") or os.getcwd()}
            for directory in sorted(here):
                if directory != submit_cwd:
                    self.prefixes.append((submit_cwd.rstrip("/") + "/", directory))
        self._digests: dict[str, str] = {}

    def why_not(self, key: str, expected: str, actual: str) -> str | None:
        """``None`` when *actual* names what *expected* did; else why not."""
        for prefix, directory in self.prefixes:
            if expected.startswith(prefix):
                moved = os.path.join(directory, expected[len(prefix) :])
                if moved == actual:
                    return None
        digest = self.files.get(key)
        if digest is None:
            return None if expected == actual else ""
        if not Path(actual).is_file():
            return f"{expected} there, {actual} here: missing on the node"
        if actual not in self._digests:
            self._digests[actual] = file_sha256(Path(actual))
        if self._digests[actual] != digest:
            return f"{expected} there, {actual} here: content differs"
        return None


def _differences(
    expected: Any,
    actual: Any,
    prefix: str = "",
    same: _SameFiles | None = None,
) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = sorted(set(expected) | set(actual), key=str)
        out: list[str] = []
        for key in keys:
            name = config_key(prefix, key)
            if key not in expected or key not in actual:
                out.append(name)
            else:
                out.extend(_differences(expected[key], actual[key], name, same))
        return out
    if (
        isinstance(expected, list)
        and isinstance(actual, list)
        and len(expected) == len(actual)
    ):
        out = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            out.extend(_differences(left, right, config_key(prefix, index), same))
        return out
    name = prefix or "<root>"
    if same is not None and isinstance(expected, str) and isinstance(actual, str):
        if expected == actual:
            return []
        why = same.why_not(name, expected, actual)
        if why is None:
            return []
        return [f"{name} ({why})" if why else name]
    equal = json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True)
    return [] if equal else [name]


def _check_same_config(
    case: dict[str, Any], cfg: "DictConfig", submit_cwd: str | None = None
) -> None:
    """Refuse a case whose config here is not the one it was submitted with.

    A path that differs counts as the same when it is the submitting
    directory's (*submit_cwd*) seen from here, or a file with the content it
    had where it was submitted (the manifest's ``files``, by SHA-256).

    Raises:
        RuntimeError: They differ (the environment or the installed scenario
            packages are not the submitting process's).
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    expected = case.get("config")
    if expected is None:
        return  # it did not resolve where it was submitted: nothing to compare
    actual = OmegaConf.to_container(cfg, resolve=True)
    same = _SameFiles(dict(case.get("files") or {}), submit_cwd)
    differing = _differences(expected, actual, same=same)
    if differing:
        shown = ", ".join(differing[:10]) + (" ..." if len(differing) > 10 else "")
        raise RuntimeError(
            f"case {case['label']}: its config here differs from the one it was "
            f"submitted with, at {shown}. The job does not run in the submitting "
            "environment: check launcher.export, launcher.env and launcher.setup "
            "(docs/slurm.md)"
        )


def _remove_outputs(batch_dir: Path, stem: str) -> None:
    """Remove what an earlier run of the case left under *stem*."""
    for suffix in RUN_FILE_SUFFIXES:
        (batch_dir / f"{stem}{suffix}").unlink(missing_ok=True)


def _raise_interrupt(signum: int, _frame: object) -> None:
    """End the task: the case running, then CARLA (``with queue``).

    Once: SLURM signals every process of the job, and the job script passes
    its own signal on, so a second SIGTERM follows the first; it must not cut
    the clean-up short.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    raise KeyboardInterrupt(f"signal {signal.Signals(signum).name}")


#: The errors of a ``flock`` the file system does not support (some NFS and
#: Lustre mounts): the start lock is then taken with an ``O_EXCL`` file.
_NO_FLOCK_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOLCK, errno.EINVAL}
)


@contextlib.contextmanager
def _exclusive_file(path: Path, stale_after: float) -> Iterator[None]:
    """Hold *path* created with ``O_EXCL``, retrying until no one else does.

    A file older than *stale_after* seconds is the leftover of a task killed
    while it held it, and is removed.
    """
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_after:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.05)
            continue
        os.close(fd)
        break
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@contextlib.contextmanager
def _start_lock(stream: Any, lock: Path, spacing: float) -> Iterator[None]:
    """Hold *lock* (open as *stream*): a ``flock``, or where the file system
    has none, an ``O_EXCL`` file beside it."""
    import fcntl  # noqa: PLC0415

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)  # released when closed
    except OSError as exc:
        if exc.errno not in _NO_FLOCK_ERRNOS:
            raise
        logger.info("No flock on %s (%s): using a lock file", lock.parent, exc)
        excl = lock.with_name(lock.name + ".excl")
        with _exclusive_file(excl, stale_after=spacing + 60.0):
            yield
        return
    yield


def wait_for_start_turn(lock: Path, spacing: float) -> float:
    """Start no sooner than *spacing* seconds after the batch's last task did.

    *lock* holds when that was; it is held (``flock``, or an ``O_EXCL`` file
    where the shared file system has no ``flock``) while this waits, so the
    tasks that start together start one after another, *spacing* apart, and
    so do the ones started later as slots free up.  How long it waited: at
    most *spacing*, however far ahead the clock of the node that wrote *lock*
    runs; a recorded time that is no finite number counts as no start.

    Raises:
        ValueError: *spacing* is not a finite number of seconds, at least 0.
    """
    if not math.isfinite(spacing) or spacing < 0:
        raise ValueError(
            "launcher.start_stagger_seconds must be a finite number of seconds, "
            f"at least 0 (got {spacing!r})"
        )
    lock.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(lock, "a+", encoding="utf-8") as stream,
        _start_lock(stream, lock, spacing),
    ):
        stream.seek(0)
        # At most *spacing*: a lock written by a node whose clock runs ahead
        # would otherwise hold this task back by the skew as well (#784).
        waited = start_delay(stream.read(), spacing, time.time())
        if waited > 0:
            logger.info("Starting in %.0f s (launcher.start_stagger_seconds)", waited)
            time.sleep(waited)
        stream.seek(0)
        stream.truncate()
        # Full precision: a start rounded up would make the next task wait
        # out the rounding after a full spacing (#304).
        stream.write(f"{time.time()!r}\n")
        stream.flush()
    return waited


def _build_fn(path: str | None) -> Any:
    if path is None:
        return None
    module, _, qualname = path.partition(":")
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


class _Task:
    """Task *task* of submission *submission*: its status file."""

    def __init__(self, layout: BatchLayout, submission: int, task: int) -> None:
        self.layout = layout
        self.submission = submission
        self.task = task
        self.status: dict[str, Any] = {
            "submission": submission,
            "task": task,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "state": "starting",
            "started": _now(),
            "cases": [],
            "done": [],
            "current": None,
        }

    def update(self, **changes: Any) -> None:
        self.status.update(changes)
        self.status["updated"] = _now()
        write_json(self.layout.task(self.submission, self.task), self.status)


def run_task(manifest_path: Path, submission_index: int, task_index: int) -> int:
    """Run task *task_index* of submission *submission_index*; its exit status."""
    layout = BatchLayout(manifest_path.resolve().parent.parent)
    manifest = layout.read_manifest()
    submission = layout.read_submission(submission_index)
    chunks = submission["chunks"]
    if not 0 <= task_index < len(chunks):
        print(  # noqa: T201
            f"Error: submission {submission_index} has {len(chunks)} task(s), "
            f"not a task {task_index}",
            file=sys.stderr,
        )
        return 2
    chunk = [int(i) for i in chunks[task_index]]
    cases = [manifest["cases"][i] for i in chunk]
    settings = SlurmSettings.from_mapping(manifest["launcher"])
    batch_dir = layout.batch_dir

    task = _Task(layout, submission_index, task_index)
    task.update(cases=chunk)

    if manifest.get("typechecked"):
        # Checked by the submitting process, once a config shape: building the
        # same configs here does not compile them again.
        from ..typecheck.mode import TYPECHECK_ENV  # noqa: PLC0415

        os.environ[TYPECHECK_ENV] = "off"
    if settings.headless:
        for name in ("DISPLAY", "WAYLAND_DISPLAY"):
            os.environ.pop(name, None)

    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGUSR1, _raise_interrupt)

    slot: PortSlot | None = None
    watchdog: CarlaWatchdog | None = None
    # Converts each case's trajectory log into a Rerun recording as the case
    # ends (rerun.enabled): each task converts its own cases.
    rerun: RerunConverter | None = None
    # Bound before the try: layout.task_port() can raise once the slot is held.
    port_file: Path | None = None
    sampler = _RssSampler()
    exit_status = 0
    try:
        from ..examples import run  # noqa: PLC0415 - CARLA, Hydra, the scenarios

        # Before any thread of this process starts (see _cap_malloc_arenas).
        run._cap_malloc_arenas()
        sampler.start()
        run.load_scenario_plugins()

        if settings.start_stagger_seconds > 0:
            task.update(state="waiting")
            wait_for_start_turn(layout.start_lock, settings.start_stagger_seconds)
        slot = acquire_slot(
            settings.port_base,
            settings.port_slots,
            start=first_slot(os.environ, settings.port_slots),
            lock_dir=(
                Path(settings.port_lock_dir).expanduser()
                if settings.port_lock_dir
                else None
            ),
        )
        port_file = layout.task_port(submission_index, task_index)
        try:
            watchdog = CarlaWatchdog(slot, port_file)
        except OSError as exc:
            logger.warning("No CARLA watchdog (%s): a SIGKILL would leave CARLA", exc)
        port_file.write_text(f"{slot.rpc}\n", encoding="utf-8")
        logger.info(
            "Task %d of submission %d: %d case(s) %s, ports %s",
            task_index,
            submission_index,
            len(chunk),
            [c["label"] for c in cases],
            slot.to_dict(),
        )
        task.update(state="building", ports=slot.to_dict())

        build_fn = _build_fn(manifest.get("build_scenario_fn"))
        configs = []
        for case in cases:
            cfg = run._compose_config(case["scenario"], list(case["overrides"]))
            _check_same_config(case, cfg, manifest.get("cwd"))
            worker_overrides(cfg, slot)
            configs.append(cfg)
        scenarios = []
        for i, (case, cfg) in enumerate(zip(cases, configs), 1):
            logger.info("Building scenario [%d/%d]: %s", i, len(cases), case["label"])
            _ego, scenario = run.build_scenario(cfg, build_scenario_fn=build_fn)
            scenarios.append(scenario)
            del scenario
        for case, scenario_built in zip(cases, scenarios):
            if type(scenario_built).__name__ != case["scenario_class"]:
                raise RuntimeError(
                    f"case {case['label']} built a {type(scenario_built).__name__}, "
                    f"not the {case['scenario_class']} it was submitted as"
                )

        first_cfg = configs[0]
        map_paths = run.resolve_map_paths(first_cfg.map)
        # The submitting process already refused a batch without the extra.
        rerun = run.build_rerun_converter(*configs)
        run.apply_canvas(*configs)
        queue = run._batch_queue(
            first_cfg,
            map_paths,
            batch_dir,
            extra_server_args=[
                f"-carla-streaming-port={slot.streaming}",
                f"-carla-secondary-port={slot.secondary}",
            ],
            rerun=rerun,
        )
        cooldown = float(first_cfg.server.get("cooldown_seconds", 0.0))

        task.update(state="running")
        with queue:
            for position, (case, cfg) in enumerate(zip(cases, configs)):
                if position > 0 and cooldown > 0:
                    # Between consecutive cases, as a local batch's queue waits.
                    time.sleep(cooldown)
                index = int(case["index"])
                stem = str(case["stem"])
                layout.result(index).unlink(missing_ok=True)
                _remove_outputs(batch_dir, stem)
                task.update(current=index)
                logger.info(
                    "[%d/%d] case %s (%s)",
                    position + 1,
                    len(cases),
                    case["label"],
                    stem,
                )
                queue.add(
                    scenarios.pop(0),
                    timeout_seconds=float(cfg.scenario.get("timeout_seconds", 60.0)),
                    case=run._odd_case(cfg),
                    file_stem=stem,
                )
                started = time.time()
                (result,) = queue.run_all(release=True, continue_on_error=True)
                # The case's recording is a few milliseconds' work, done while
                # the world was reset: wait for it, so its outcome is in the
                # case's record for `collect`.
                rerun_error = (
                    run.rerun_errors([result], rerun.wait()).get(0)
                    if rerun is not None
                    else None
                )
                write_json(
                    layout.result(index),
                    {
                        "index": index,
                        "label": case["label"],
                        "stem": stem,
                        "result": result.to_dict(),
                        "errored": bool(queue.errored_indices()),
                        "submission": submission_index,
                        "task": task_index,
                        "job_id": task.status["job_id"],
                        "host": task.status["host"],
                        "wall_seconds": round(time.time() - started, 3),
                        "finished": _now(),
                        **({"rerun_error": rerun_error} if rerun_error else {}),
                    },
                )
                task.update(done=[*task.status["done"], index], current=None)
        task.update(state="done", finished=_now(), **sampler.to_dict())
    except KeyboardInterrupt as exc:
        logger.warning("Task interrupted (%s): stopping", exc)
        task.update(
            state="cancelled", finished=_now(), error=str(exc), **sampler.to_dict()
        )
        exit_status = 143
    except BaseException as exc:  # noqa: BLE001 - recorded, then the task fails
        logger.error("Task failed: %s", exc, exc_info=True)
        task.update(
            state="failed",
            finished=_now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            **sampler.to_dict(),
        )
        exit_status = 1
    finally:
        # Ignore further signals while cleaning up: SLURM sends SIGTERM to
        # every process of the job, and a second one must not cut this short.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        sampler.stop()
        if rerun is not None:
            from ..rerun import report  # noqa: PLC0415

            logger.info("%s", report(rerun.close()))
        if slot is not None:
            left = kill_carla_on_port(slot.rpc)
            if left:
                logger.error("CARLA on port %d still runs: %s", slot.rpc, left)
            else:
                # The file (the job script's pointer to CARLA) goes only once
                # CARLA is gone; a worker cut short here leaves it to the
                # watchdog, which removes it after stopping CARLA.
                if port_file is not None:
                    port_file.unlink(missing_ok=True)
                if watchdog is not None:
                    watchdog.finish()
            # With CARLA left, the watchdog ends it when this process does,
            # and keeps the slot locked until then.
            slot.release(keep_lock=watchdog is not None and bool(left))
    return exit_status


def main(argv: list[str] | None = None) -> int:
    """``python -m autoware_carla_scenario.slurm.worker <manifest> <k> <task>``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(  # noqa: T201
            "usage: python -m autoware_carla_scenario.slurm.worker "
            "<manifest> <submission> <task>",
            file=sys.stderr,
        )
        return 2
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    return run_task(Path(args[0]), int(args[1]), int(args[2]))


if __name__ == "__main__":
    sys.exit(main())
