"""Hydra Sweeper plugin that enumerates lanelets matching declarative constraints.

When activated via ``--multirun hydra/sweeper=lanelet_constraint``, this
sweeper:

1. Loads the Lanelet2 map (lightweight, no CARLA required).
2. Parses constraints from ``sweep.constraints`` and finds matching lanelets.
3. For each match, resolves bindings from ``sweep.bindings`` to compute
   derived parameters (e.g. ``ego.spawn_s``).
4. Constructs per-lanelet Hydra override batches and launches them.

Once every case has run, a sweep with a failed (or timed-out) case raises
:class:`SweepFailedError`, which Hydra reports and turns into exit status 1.
A refused case (one exiting 2, as ``examples/run.py`` does for a scenario that
does not type-check or an extra that is not installed) ends the sweep at once
with exit status 2: every case after it would be refused for the same reason.
A sweep whose config sets a value its runs refuse (``t4scene.step_s`` that is
not a multiple of the tick, a negative ``server.cooldown_max_retries``, ...),
or asks for an extra that is not installed, exits 2 before any case is forked
(:func:`_refuse_before_forking`).
A sweep with no case to run (nothing expanded, or ``--resume-from`` past the
last case) raises :class:`SweepFailedError` too, before anything runs.

Each case runs in a forked child of its own process group.  A case whose
scenario ran and failed exits :data:`_FAILED_VERDICT_EXIT_CODE` and is final;
only a case that raised, timed out or crashed is retried
(``server.cooldown_max_retries``).  Ctrl-C or ``SIGTERM`` stops the running
case (``SIGTERM``, then ``SIGKILL`` after
:data:`.isolation._STOP_GRACE_SECONDS`) and ends the sweep with 130 or 143.

A case that timed out or died of a signal had no chance to stop the CARLA
server it launched; the server's watchdog stops it instead, within about a
second and then a grace period.  Before the next attempt the sweeper waits,
at most :data:`.isolation._CARLA_GONE_TIMEOUT_S`, until every CARLA the case
launched has exited (:func:`.isolation._wait_until_carla_gone`), so the next
case neither reuses the dying server nor launches one next to it (#717).  A
server the case did not launch (one the user started, which it reused) is not
waited for.

The forking, signalling and waiting is :mod:`.isolation`'s, which the library
check compiles (docs/typecheck.md); this module adds what a case is (a Hydra
launch, and the exit status its result reads as).
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, JobStatus, configure_log
from hydra.errors import CompactHydraException
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from .. import hydra_compat  # noqa: F401  (Python 3.14 argparse)
from ..rerun import RerunUnavailableError
from ..run_outputs import RUN_DIR_SUFFIXES, RUN_FILE_SUFFIXES, remove_run_outputs
from ..ui.case_paths import last_result_path
from .expand import (
    EMPTY_SWEEP_MESSAGE,
    _config_seed,
    check_constraint_keys,
    expand_sweep,
    load_sweep_map,
    scenario_controls,
)
from .isolation import (  # noqa: F401  (SWEEPER_OWNS_RETRIES_ENV: re-exported)
    SWEEPER_OWNS_RETRIES_ENV,
    _interrupt_signal,
    _launch_isolated,
    _raise_interrupted,
)

logger = logging.getLogger(__name__)

# Default hard timeout per job (seconds).  Can be overridden via
# ``sweep.job_timeout_seconds`` in the scenario YAML.
_DEFAULT_JOB_TIMEOUT = 120

#: The largest ``sweep.job_timeout_seconds``: ``signal.alarm`` takes a C
#: ``int`` and raises ``OverflowError`` above it (about 68 years).
_MAX_JOB_TIMEOUT = 2**31 - 1

#: The exit status of a refused run (docs/usage.md, "Exit Codes").
_REFUSED_EXIT_CODE = 2

#: The exit status of a forked case whose scenario ran to a FAILED verdict
#: (its result JSON says ``passed: false``).  Final, as in
#: :class:`~autoware_carla_scenario.scenario_queue.ScenarioQueue`, which
#: retries a run that raised but not one that completed: only a case exiting
#: 1 (it raised, or wrote no result), timing out or dying of a signal is
#: retried (``server.cooldown_max_retries``).
_FAILED_VERDICT_EXIT_CODE = 3


class SweepFailedError(CompactHydraException):
    """One or more cases of a sweep failed, crashed or timed out.

    Raised by :meth:`LaneletConstraintSweeper.sweep` once every case has run
    and the summary is logged, or before any runs when the sweep has no case
    to run.  Hydra returns nothing of a ``--multirun`` to its caller, so
    raising is how a sweep reports failure: Hydra prints the message and exits
    with status 1 (docs/usage.md, "Exit Codes"), as a glob batch does.  Being
    a :class:`~hydra.errors.CompactHydraException`, it is printed without a
    traceback.
    """


def _job_timeout_seconds(value: Any) -> int:
    """``sweep.job_timeout_seconds``, checked: the hard timeout per job, in
    whole seconds, :data:`_DEFAULT_JOB_TIMEOUT` when *value* is unset
    (``None``, as ``job_timeout_seconds: null`` reads), 0 for none.

    A case is stopped by ``signal.alarm``, which counts whole seconds, so
    *value* must be a whole number (``900`` or ``900.0``): a fraction would
    be truncated, and ``0.5`` would become 0, no timeout at all.  A negative,
    NaN or infinite value is refused too, rather than silently disabling the
    timeout, and so is one above :data:`_MAX_JOB_TIMEOUT`, which
    ``signal.alarm`` cannot take, rather than failing after the first fork.

    Raises:
        ValueError: *value* is not a whole number of seconds from 0 to
            :data:`_MAX_JOB_TIMEOUT`.
    """
    if value is None:
        return _DEFAULT_JOB_TIMEOUT
    message = (
        "sweep.job_timeout_seconds must be a whole number of seconds from 0 "
        f"(no timeout) to {_MAX_JOB_TIMEOUT}, got {value!r}"
    )
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if (
        not math.isfinite(seconds)
        or not 0 <= seconds <= _MAX_JOB_TIMEOUT
        or seconds != math.floor(seconds)
    ):
        raise ValueError(message)
    return int(seconds)


def _read_verdict(output_dir: Path) -> bool | None:
    """The verdict of the scenario result JSON the case just launched wrote in
    its job's output directory: the last run's
    (:func:`..ui.case_paths.last_result_path`).  A retry
    (``server.cooldown_max_retries``) runs the case again in the same job
    directory; the sweeper removes the failed attempt's outputs first (#1373),
    so the retry takes the same stem, but a directory that held a run before
    the case started (one ``hydra.sweep.dir`` names again) keeps it, and the
    case takes the next stem (``X-1`` after ``X``): its verdict is the one
    that counts.

    Returns:
        ``True`` if the scenario passed, ``False`` if it ran and failed, or
        ``None`` if there is no verdict: no result JSON (the run raised
        before :class:`ScenarioRunner` could write it) or one that cannot be
        read.
    """
    result_file = last_result_path(output_dir)
    if result_file is None:
        return None
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError: not JSON, or not UTF-8
        return None
    if not isinstance(data, dict):
        return None
    return bool(data.get("passed", True))


#: Every suffix of a run output (:mod:`..run_outputs`), longest first, so
#: that ``X.sensors.rrd`` reads as stem ``X``, not ``X.sensors``.
_RUN_SUFFIXES: tuple[str, ...] = tuple(
    sorted(RUN_FILE_SUFFIXES + RUN_DIR_SUFFIXES, key=len, reverse=True)
)


def _run_output_stems(directory: Path, job_log: str | None) -> set[str]:
    """The stems of the runs that left an output (:mod:`..run_outputs`) in
    *directory*: what ``scenario_runner._run_file_stem`` skips over.

    *job_log* is the file name of Hydra's job log (``<job name>.log``), which
    is not a run's CARLA recorder log though it ends as one does.
    """
    try:
        names = [path.name for path in directory.iterdir()]
    except OSError:
        return set()
    stems: set[str] = set()
    for name in names:
        if name == job_log:
            continue
        for suffix in _RUN_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                stems.add(name[: -len(suffix)])
                break
    return stems


def _get_job_output_dir(ret: JobReturn) -> str | None:
    """Return the Hydra output directory for a completed job.

    ``JobReturn.working_dir`` is the *current working directory* at
    execution time, which equals the project root when
    ``hydra.job.chdir=False`` (the default since Hydra 1.2).
    The actual output directory where :class:`ScenarioRunner` writes
    result JSONs is ``hydra.runtime.output_dir`` inside the job's
    ``hydra_cfg``.
    """
    hydra_cfg = getattr(ret, "hydra_cfg", None)
    if hydra_cfg is not None:
        try:
            return str(OmegaConf.select(hydra_cfg, "hydra.runtime.output_dir"))
        except Exception:
            pass
    # Fallback: working_dir (correct only when hydra.job.chdir=True).
    return getattr(ret, "working_dir", None)


def _log_job_exception(ret: JobReturn) -> None:
    """Log, with its traceback, the exception the case's task function raised.

    Hydra's ``run_job`` catches it and stores it in the ``JobReturn``
    (``_return_value``; the ``return_value`` property re-raises it) without
    logging it, so without this a case that raised shows only ``Job FAILED
    (exit code 1)`` (#1232).  It is logged where the task's own log records
    went: with the job's logging (``hydra.job_logging``: the terminal and the
    job's ``<job name>.log`` in its output directory) put back for it, then
    Hydra's own (``hydra.hydra_logging``) again, as Hydra's launcher leaves
    it after a job.  Without the job's Hydra config, the sweeper's logging as
    it stands.

    Never raises: the case's exit status is 1 whatever happens here.
    ``_return_value`` is private to Hydra (the public ``return_value``
    re-raises it and writes to stderr), so it is read with ``getattr``.
    """
    try:
        _log_job_exception_unguarded(ret)
    except Exception:  # noqa: BLE001, S110 - logging must not change the exit status
        pass


def _log_job_exception_unguarded(ret: JobReturn) -> None:
    """:func:`_log_job_exception`, which may raise."""
    exc = getattr(ret, "_return_value", None)
    if not isinstance(exc, BaseException):
        return
    hydra_cfg = getattr(ret, "hydra_cfg", None)
    job_logging = None
    verbose: Any = False
    if hydra_cfg is not None:
        try:
            job_logging = OmegaConf.select(hydra_cfg, "hydra.job_logging")
            verbose = OmegaConf.select(hydra_cfg, "hydra.verbose", default=False)
            if job_logging is not None:
                configure_log(job_logging, verbose)
        except Exception:  # noqa: BLE001 - log it anyway, as things stand
            job_logging = None
    try:
        logger.error(
            "Case %s raised %s: %s",
            " ".join(ret.overrides or ()),
            type(exc).__name__,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    finally:
        if job_logging is not None and hydra_cfg is not None:
            try:
                hydra_logging = OmegaConf.select(hydra_cfg, "hydra.hydra_logging")
                if hydra_logging is not None:
                    configure_log(hydra_logging, verbose)
            except Exception:  # noqa: BLE001, S110 - the record is out
                pass


def _job_exit_code(job_returns: Any) -> int:
    """The exit status of a forked case, from what ``launcher.launch`` returned.

    ``launcher.launch()`` completes without error even when the scenario
    *fails* (in --multirun mode ``_hydra_main`` does not ``sys.exit``), and
    Hydra catches what the task function raises and stores it in the
    ``JobReturn`` with ``status=FAILED``.  So:

    * 1 -- the task raised (logged here, with its traceback:
      :func:`_log_job_exception`), or completed without a readable result
      JSON;
    * :data:`_FAILED_VERDICT_EXIT_CODE` -- the scenario ran and failed;
    * 0 -- it passed (or there is no output directory to read a verdict from).
    """
    if not job_returns:
        return 0
    ret = job_returns[0]
    if ret.status != JobStatus.COMPLETED:
        if ret.status == JobStatus.FAILED:
            _log_job_exception(ret)
        return 1
    # hydra_cfg.hydra.runtime.output_dir -- NOT working_dir, which equals the
    # cwd when hydra.job.chdir=False (the default since Hydra 1.2).
    output_dir = _get_job_output_dir(ret)
    if not output_dir or not Path(output_dir).is_dir():
        return 0
    verdict = _read_verdict(Path(output_dir))
    if verdict is None:
        return 1
    return 0 if verdict else _FAILED_VERDICT_EXIT_CODE


def _launch_job_isolated(
    launcher: Any,
    batch: tuple[str, ...],
    idx: int,
    timeout: int,
) -> tuple[bool, int | None]:
    """Run one case in a forked child of its own (:func:`.isolation._launch_isolated`).

    Args:
        launcher: Hydra job launcher instance.
        batch: Override tuple for this job.
        idx: 0-based job index.
        timeout: Hard timeout in seconds (0 = no timeout).

    Returns:
        ``(succeeded, exit_code)`` where *exit_code* is 0 on success,
        :data:`_FAILED_VERDICT_EXIT_CODE` when the scenario ran and failed,
        another positive int when the run raised (1) or was refused (2), a
        negative int (``-signal_number``) on signal death, or ``None`` on
        timeout.
    """

    def run_case() -> int:
        return _job_exit_code(launcher.launch([batch], initial_job_idx=idx))

    return _launch_isolated(run_case, timeout)


def _refuse_before_forking(cfg: DictConfig) -> None:
    """Exit :data:`_REFUSED_EXIT_CODE`, saying why, when every case of the
    sweep of *cfg* would be refused before it starts: for a config value its
    run refuses (:func:`examples.run.check_run_config`: the queue's server,
    traffic and Autoware launcher settings, the run timeout, the seeds
    (``seed``, ``sweep.odd_sample.seed`` and ``.count``), the
    ``background_traffic`` and ``environment`` settings, a ``carla_driver``
    ego's ``driver`` settings, ``odd_case``,
    ``rerun.lidar_max_points``, the ``t4`` and ``driver`` settings and the
    ``t4scene`` settings), or for an extra it asks for that is not installed
    (:func:`examples.run.check_run_extras`).  Builds nothing that starts, and
    needs no map.

    *cfg* is the sweep's base config, with the command line's overrides: what
    a case adds to it -- the ego's spawn and goal and the values its bindings
    derive, a route's ``scenario.route.*``, the ODD's environment knobs and
    the scenario's controls, and its own ``odd_case``, which the sweep encodes
    -- are not what these checks read, so a value refused here is refused in
    every case.  One only a case's overrides set is refused by that case,
    which exits :data:`_REFUSED_EXIT_CODE` too (``examples/run.py``'s
    ``_hydra_main``), and the sweep stops there.
    """
    # The run's own checks; imported here, as examples.run imports this
    # package's modules inside its functions.
    from ..examples.run import (  # noqa: PLC0415
        RunConfigError,
        check_run_config,
        check_run_extras,
    )
    from ..t4 import T4UnavailableError  # noqa: PLC0415

    try:
        check_run_config(cfg)
        check_run_extras(cfg)
    except RunConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(_REFUSED_EXIT_CODE)
    except (RerunUnavailableError, T4UnavailableError) as exc:
        print(exc, file=sys.stderr)  # noqa: T201
        sys.exit(_REFUSED_EXIT_CODE)


class LaneletConstraintSweeper(Sweeper):
    """Sweep over lanelets that satisfy declarative constraints."""

    def __init__(self) -> None:
        super().__init__()
        self.config: DictConfig | None = None
        self.hydra_context: HydraContext | None = None
        self.launcher: Any = None
        #: The 1-based case running (or about to), for an interrupted sweep's
        #: message; ``None`` before the first.
        self._current_case: int | None = None

    def setup(
        self,
        *,
        config: DictConfig,
        hydra_context: HydraContext,
        task_function: Any,
    ) -> None:
        """Store references and instantiate the job launcher."""
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )

    def _job_output_dir(
        self, batch: tuple[str, ...], idx: int
    ) -> tuple[Path, str | None] | None:
        """Where job *idx* runs *batch*, and the file name of its Hydra job log.

        The directory is the one Hydra's launcher gives the job
        (``hydra.sweep.dir``/``hydra.sweep.subdir`` of its config, with
        ``hydra.job.num`` = *idx*), computed as ``BasicLauncher.launch`` and
        ``run_job`` compute it.  ``None``, logged, when it cannot be: the
        retry then runs beside what the failed attempt left.
        """
        assert self.config is not None
        try:
            if self.hydra_context is None:
                raise RuntimeError("the sweeper was not set up")
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config, list(batch)
            )
            with open_dict(sweep_config):
                sweep_config.hydra.job.id = idx
                sweep_config.hydra.job.num = idx
            hydra_config = HydraConfig.instance()
            previous = hydra_config.cfg
            # As run_job does: hydra.sweep.subdir may read ${hydra:...}.
            hydra_config.set_config(sweep_config)
            try:
                directory = str(OmegaConf.select(sweep_config, "hydra.sweep.dir"))
                subdir = OmegaConf.select(sweep_config, "hydra.sweep.subdir")
                if subdir is not None:
                    directory = os.path.join(directory, str(subdir))
                job_name = OmegaConf.select(sweep_config, "hydra.job.name")
            finally:
                hydra_config.cfg = previous
        except Exception as exc:  # noqa: BLE001 - the case runs anyway
            logger.warning(
                "Cannot tell the output directory of job %d (%s: %s); a retry "
                "of it keeps what the failed attempt left there.",
                idx,
                type(exc).__name__,
                exc,
            )
            return None
        job_log = f"{job_name}.log" if job_name is not None else None
        return Path(os.path.abspath(directory)), job_log

    def sweep(self, arguments: list[str]) -> Any:
        """Execute the constraint-based sweep.

        While it runs, ``SIGTERM`` is turned into a ``KeyboardInterrupt``, as
        Ctrl-C is.  Either stops the case running (``SIGTERM`` to it, then
        ``SIGKILL`` after :data:`.isolation._STOP_GRACE_SECONDS`) and ends the
        sweep.

        Args:
            arguments: Additional CLI overrides passed alongside ``--multirun``.

        Returns:
            The aggregated results from all launched jobs.

        Raises:
            SweepFailedError: After every case has run, when one or more of
                them failed or timed out; or at once, when the sweep has no
                case to run (none expanded, or ``--resume-from`` past the
                last).
            SystemExit: With status 2, at once, when the sweep sets a config
                value its runs refuse or asks for Rerun recordings or T4
                datasets without the extra, or when a case is refused;
                with status 130 (``SIGINT``) or 143 (``SIGTERM``) when the
                sweep is interrupted.
            ValueError: At once, before any case runs, when the sweep is
                invalid -- among others, when its Lanelet2 map is not named,
                is missing, or cannot be resolved, fetched or read
                (:func:`.expand.load_sweep_map`), the message naming it.
        """
        self._current_case = None
        try:
            previous_sigterm = signal.signal(signal.SIGTERM, _raise_interrupted)
        except ValueError:
            # Not the main thread: signals are not this sweep's to handle.
            previous_sigterm = None
        try:
            return self._sweep(arguments)
        except KeyboardInterrupt as exc:
            signum = _interrupt_signal(exc)
            name = signal.Signals(signum).name
            where = (
                f" at case {self._current_case}; resume with "
                f"--resume-from {self._current_case}"
                if self._current_case is not None
                else ""
            )
            logger.error("Sweep interrupted (%s)%s.", name, where)
            print(f"Sweep interrupted ({name}){where}.", file=sys.stderr)  # noqa: T201
            sys.exit(128 + signum)
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)

    def _sweep(self, arguments: list[str]) -> Any:
        """:meth:`sweep`, but for turning an interrupt into the exit status."""
        assert self.config is not None
        assert self.launcher is not None

        cfg = self.config

        # -- 0. Refuse at once a sweep every case of which would be refused
        # before it starts, rather than fork each to fail (and be retried):
        # a config value a run refuses (#1233), or Rerun recordings or T4
        # datasets asked for without the extra.  Each case converts its own
        # log and writes its own dataset (examples/run.py).
        _refuse_before_forking(cfg)

        # -- 1. Read the sweep section, before any map is fetched --------
        sweep_cfg = OmegaConf.select(cfg, "sweep")
        if sweep_cfg is None:
            raise ValueError(
                "No 'sweep' section found in config. "
                "LaneletConstraintSweeper requires sweep.constraints, sweep.route "
                "or sweep.odd_sample."
            )
        sweep_dict = OmegaConf.to_container(sweep_cfg, resolve=True)
        assert isinstance(sweep_dict, dict)
        check_constraint_keys(sweep_dict)
        # Checked here, before any map is fetched or case runs, and refused
        # as step 0 refuses a config value: exit 2, naming the key.
        try:
            job_timeout = _job_timeout_seconds(sweep_dict.get("job_timeout_seconds"))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
            sys.exit(_REFUSED_EXIT_CODE)

        # -- 2-3. Resolve map paths and load the Lanelet2 map --------------
        # A map named by `map.source` is cloned (or found already downloaded)
        # here, so a sweep runs against a published HD map without anyone
        # having to name a local file for it first; `map.directory` names a
        # map on this machine and is read in place.  A map that cannot be
        # resolved, fetched or read is refused as an invalid sweep, a
        # ValueError naming it, as scenario-expand and a batch refuse it
        # (#1447).
        lanelet_map = load_sweep_map(
            OmegaConf.select(cfg, "map"), prefix="LaneletConstraintSweeper: "
        )

        # -- 4-6. Expand: matching lanelets -> one override batch each -----
        batches: list[tuple[str, ...]] = [
            tuple(case)
            for case in expand_sweep(
                sweep_dict,
                lanelet_map,
                arguments,
                odd=OmegaConf.select(cfg, "odd"),
                controls=scenario_controls(cfg),
                seed=_config_seed(cfg),
            )
        ]
        if not batches:
            # Exit 0 means every case passed: a sweep that tests nothing
            # (a misspelt constraint, the wrong map) must not look like one.
            raise SweepFailedError(f"Sweep failed: {EMPTY_SWEEP_MESSAGE}.")

        logger.info(
            "Sweeping %d case(s): %s",
            len(batches),
            [b[0] if b else "" for b in batches],
        )

        # -- 7. Resolve cooldown, retry count, and resume ---
        # (job_timeout: read and checked at step 1.)
        cooldown: float = float(
            OmegaConf.select(cfg, "server.cooldown_seconds", default=0.0)
        )
        # Refused at step 0 already (check_run_config); read as the run
        # reads it, for a caller that reaches here without it.
        from ..examples.run import cooldown_max_retries  # noqa: PLC0415

        try:
            max_retries = cooldown_max_retries(cfg)
        except ValueError as exc:
            raise ValueError(f"LaneletConstraintSweeper: {exc}.") from None
        max_attempts = 1 + max_retries
        resume_from: int = int(
            sweep_dict.get("resume_from", 0) or os.environ.get("SWEEP_RESUME_FROM", "0")
        )

        # -- 8. Apply resume: skip jobs before the resume point (1-indexed) --
        if resume_from > len(batches):
            # As a glob batch with no case left to run: exit 1, not 0.
            raise SweepFailedError(
                f"Sweep failed: --resume-from {resume_from} is past the sweep's "
                f"{len(batches)} case(s); no case left to run."
            )
        if resume_from > 1:
            skip_count = resume_from - 1
            logger.info(
                "Resuming from job %d — skipping %d/%d job(s).",
                resume_from,
                skip_count,
                len(batches),
            )
            batches = batches[skip_count:]
            # Offset indices so logs still show the original 1-based position.
            idx_offset = skip_count
        else:
            idx_offset = 0

        # -- 9. Launch batches one-by-one via fork isolation --
        all_returns: list[Any] = []
        failed_count = 0
        timed_out_count = 0
        first_job = True

        for local_idx, batch in enumerate(batches):
            idx = local_idx + idx_offset
            self._current_case = idx + 1

            succeeded = False
            exit_code: int | None = None
            # The job's directory and the stems already in it before its
            # first attempt (none, unless hydra.sweep.dir names a directory
            # an earlier sweep wrote in): a retry removes only what the
            # failed attempts of this job left (#1373).
            job_dir = self._job_output_dir(batch, idx) if max_attempts > 1 else None
            stems_before = (
                _run_output_stems(job_dir[0], job_dir[1])
                if job_dir is not None
                else set()
            )
            for attempt in range(max_attempts):
                # Cooldown in the parent process (no CARLA state, SIGSEGV-safe).
                # Skip cooldown before the very first job of this run.
                if (not first_job or attempt > 0) and cooldown > 0:
                    _wait_with_progress(
                        cooldown,
                        desc="CARLA cooldown"
                        if attempt == 0
                        else "CARLA cooldown (retry)",
                    )

                if attempt > 0 and job_dir is not None:
                    # As the queue and the SLURM worker do before a retry:
                    # what the failed attempt wrote (a result, a coverage
                    # file, a trajectory log ...) makes way for the retry's,
                    # which then takes the same stem.  Left behind, the
                    # viewer, which shows a job's first stem, and the
                    # coverage merge, which reads every coverage file, would
                    # read the failed attempt's (#1373).
                    directory, job_log = job_dir
                    for stem in sorted(
                        _run_output_stems(directory, job_log) - stems_before
                    ):
                        logger.info(
                            "[%d/%d] Removing the outputs of the failed "
                            "attempt %s in %s before the retry.",
                            idx + 1,
                            idx_offset + len(batches),
                            stem,
                            directory,
                        )
                        remove_run_outputs(directory, stem)

                logger.info(
                    "[%d/%d] Launching (attempt %d/%d): %s",
                    idx + 1,
                    idx_offset + len(batches),
                    attempt + 1,
                    max_attempts,
                    batch,
                )

                ok, exit_code = _launch_job_isolated(
                    self.launcher, batch, idx, job_timeout
                )
                if ok:
                    succeeded = True
                    break
                if exit_code == _FAILED_VERDICT_EXIT_CODE:
                    # The scenario ran and failed: final, as in the queue,
                    # which retries only a run that raised.
                    logger.error(
                        "[%d/%d] Scenario FAILED (attempt %d/%d) for overrides "
                        "%s -- not retried.",
                        idx + 1,
                        idx_offset + len(batches),
                        attempt + 1,
                        max_attempts,
                        batch,
                    )
                    break
                if exit_code == _REFUSED_EXIT_CODE:
                    # The case printed why; the rest would be refused too.
                    logger.error(
                        "[%d/%d] Job REFUSED for overrides %s — stopping the " "sweep.",
                        idx + 1,
                        idx_offset + len(batches),
                        batch,
                    )
                    sys.exit(_REFUSED_EXIT_CODE)

                # Log based on exit code.
                if exit_code is None:
                    logger.error(
                        "[%d/%d] Job TIMED OUT after %ds (attempt %d/%d) "
                        "for overrides %s",
                        idx + 1,
                        idx_offset + len(batches),
                        job_timeout,
                        attempt + 1,
                        max_attempts,
                        batch,
                    )
                elif exit_code < 0:
                    logger.error(
                        "[%d/%d] Job CRASHED (signal %d) (attempt %d/%d) "
                        "for overrides %s",
                        idx + 1,
                        idx_offset + len(batches),
                        -exit_code,
                        attempt + 1,
                        max_attempts,
                        batch,
                    )
                else:
                    logger.error(
                        "[%d/%d] Job FAILED (exit code %d) (attempt %d/%d) "
                        "for overrides %s",
                        idx + 1,
                        idx_offset + len(batches),
                        exit_code,
                        attempt + 1,
                        max_attempts,
                        batch,
                    )

            first_job = False
            # Result JSON is written to disk by the child; parent only
            # tracks success/failure.
            all_returns.append(None)
            if not succeeded:
                # Distinguish timeouts from other failures for the summary.
                if exit_code is None:
                    timed_out_count += 1
                failed_count += 1
                logger.error(
                    "[%d/%d] Job FAILED after %d attempt(s) for overrides %s "
                    "— continuing.",
                    idx + 1,
                    idx_offset + len(batches),
                    attempt + 1,
                    batch,
                )

        executed = len(batches)
        logger.info(
            "Sweep complete: %d/%d succeeded, %d failed (%d timed out).",
            executed - failed_count,
            executed,
            failed_count,
            timed_out_count,
        )
        if failed_count:
            raise SweepFailedError(
                f"Sweep failed: {failed_count} of {executed} case(s) failed "
                f"({timed_out_count} timed out)."
            )
        return all_returns


def _wait_with_progress(seconds: float, *, desc: str = "Waiting") -> None:
    """Sleep for *seconds* with a tqdm progress bar (0.1 s resolution)."""
    steps = max(1, int(seconds * 10))
    interval = seconds / steps
    for _ in tqdm(
        range(steps),
        desc=desc,
        bar_format="{desc}: {bar}| {elapsed}<{remaining}",
    ):
        time.sleep(interval)
