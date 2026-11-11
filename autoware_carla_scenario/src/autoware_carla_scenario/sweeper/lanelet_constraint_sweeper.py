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
(``server.cooldown_max_retries``).  A case given up for any reason but a
failed verdict leaves ``case_errored.json`` in its job directory
(:mod:`..gate.errored`), which ``scenario-gate`` refuses a set of runs for: it
wrote no gate record.  The marker is written before each attempt too, saying
the case did not finish, and removed once the case ends with a verdict, so a
case whose sweep was killed with it (a segfault, the OOM killer, ``SIGKILL``)
leaves it as well; none is left for a case that passed, a retry included, or
failed.  Before a case's first attempt, the run outputs an earlier sweep
left in its job directory (one ``hydra.sweep.dir`` names again, a
``--resume-from`` rerun) are removed: the case replaces that run (#1956);
a directory several cases of the sweep share is left as it is.
Before the first case is launched, the sweep writes its plan
(``sweep_plan.json``, :mod:`..gate.sweep_plan`) into its sweep directory: the
directory and overrides of every case it is about to run, so that
``scenario-gate`` also sees the cases a killed sweep never started (#1833).
Ctrl-C or ``SIGTERM`` stops the running
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

import fnmatch
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, JobStatus, configure_log
from hydra.errors import CompactHydraException
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from .. import hydra_compat  # noqa: F401  (Python 3.14 argparse)
from ..gate.errored import (
    remove_case_errored,
    write_case_errored,
    write_case_unfinished,
)
from ..gate.sweep_plan import write_sweep_plan
from ..rerun import RerunUnavailableError
from ..run_outputs import RUN_SUFFIXES as _RUN_SUFFIXES  # noqa: F401
from ..run_outputs import remove_run_outputs, run_output_paths
from ..run_outputs import run_output_stems as _run_output_stems
from ..ui.case_paths import last_result_path
from ..value_text import shown
from .expand import (
    EMPTY_SWEEP_MESSAGE,
    _config_seed,
    check_constraint_keys,
    environment_rules,
    expand_sweep,
    load_sweep_map,
    scenario_controls,
    scenario_rules,
)
from .isolation import (  # noqa: F401  (SWEEPER_OWNS_RETRIES_ENV: re-exported)
    SWEEPER_OWNS_RETRIES_ENV,
    _interrupt_signal,
    _launch_isolated,
    _raise_interrupted,
)
from .job_status import (
    DEFAULT_JOB_TIMEOUT,
    FAILED_VERDICT_EXIT_CODE,
    MAX_JOB_TIMEOUT,
    REFUSED_EXIT_CODE,
    verdict_exit_code,
    whole_job_seconds,
)

logger = logging.getLogger(__name__)

# The case's timeout and exit statuses are the checked :mod:`.job_status`'s;
# the old private names stay, for what imports them from here.
_DEFAULT_JOB_TIMEOUT = DEFAULT_JOB_TIMEOUT
_MAX_JOB_TIMEOUT = MAX_JOB_TIMEOUT
_REFUSED_EXIT_CODE = REFUSED_EXIT_CODE
_FAILED_VERDICT_EXIT_CODE = FAILED_VERDICT_EXIT_CODE


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
        f"(no timeout) to {_MAX_JOB_TIMEOUT}, got {shown(value)}"
    )
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: an int too big for a float (10**400) (#1652).
        raise ValueError(message) from None
    whole = whole_job_seconds(seconds)
    if whole is None:
        raise ValueError(message)
    return whole


#: The environment variable ``scenario --resume-from N`` passes the sweeper
#: *N* in (``examples.run.RESUME_FROM_ENV``): Hydra's parser would reject the
#: flag.
_RESUME_FROM_ENV = "SWEEP_RESUME_FROM"


def _resume_from(config_value: Any, env_value: str | None) -> int:
    """The 1-based case the sweep starts at, 0 to run every case.

    *env_value* is :data:`_RESUME_FROM_ENV` (``--resume-from N``, or the
    variable set by hand), *config_value* ``sweep.resume_from``.  Each is a
    whole number of at least 1, or unset: ``None`` (``null``, absent) or 0 in
    the config, an empty or unset variable or ``0``.  The variable, when set,
    wins over the config: ``--resume-from 10`` resumes at case 10 whatever
    ``sweep.resume_from`` says.  Both are checked, so a bad config value is
    refused even when the command line overrides it (#1784).

    A bool, a float (``2.7`` or ``3.0``), a string or a negative number in the
    config is refused rather than truncated, cast or read as "run every case";
    so is a variable that is not a whole number of at least 0.

    Raises:
        ValueError: Naming the key or variable, when either is invalid.
    """
    expected = (
        "a whole number of at least 1 (the 1-indexed case to start at; "
        "0 or unset for none)"
    )
    if config_value is None:
        from_config = 0
    elif (
        isinstance(config_value, int)
        and not isinstance(config_value, bool)
        and config_value >= 0
    ):
        from_config = config_value
    else:
        raise ValueError(
            f"sweep.resume_from must be {expected}, got {shown(config_value)}"
        )
    text = (env_value or "").strip()
    if not text:
        return from_config
    if not (text.isascii() and text.isdigit()):
        # No sign, fraction, exponent or non-ASCII digit.
        raise ValueError(f"{_RESUME_FROM_ENV} must be {expected}, got {text!r}")
    from_env = int(text)
    return from_env if from_env >= 1 else from_config


def _read_verdict(output_dir: Path) -> bool | None:
    """The verdict of the scenario result JSON the case just launched wrote in
    its job's output directory: the last run's
    (:func:`..ui.case_paths.last_result_path`).  A retry
    (``server.cooldown_max_retries``) runs the case again in the same job
    directory; the sweeper removes the failed attempt's outputs first (#1373),
    so the retry takes the same stem, and before the case's first attempt it
    removes those of a run an earlier sweep left there (one
    ``hydra.sweep.dir`` names again, #1956).  Should a run still be there
    (its outputs could not be removed, or the directory is shared by several
    jobs of the sweep), the case takes the next stem (``X-1`` after ``X``): its
    verdict is the one that counts.

    Returns:
        ``True`` if the scenario passed, ``False`` if it ran and failed, or
        ``None`` if there is no verdict: no result JSON (the run raised
        before :class:`ScenarioRunner` could write it), one that cannot be
        read, or one whose ``passed`` is missing or not a JSON boolean
        (``"false"`` and ``1`` are no verdict, not a pass, as the viewer and
        the gate read them).
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
    passed = data.get("passed")
    return passed if isinstance(passed, bool) else None


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
    return verdict_exit_code(_read_verdict(Path(output_dir)))


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


def _read_sweep_section(cfg: DictConfig) -> dict[str, Any]:
    """The config's ``sweep`` section, resolved, its constraint keys checked.

    Raises:
        ValueError: When the config has no ``sweep`` section, or its
            constraints name more than one key (:func:`.expand.check_constraint_keys`).
    """
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
    return {str(k): v for k, v in sweep_dict.items()}


def _refuse_invalid_sweep(cfg: DictConfig, exc: ValueError) -> NoReturn:
    """Report *exc*, the sweep's own refusal of its config, and exit 1.

    One line on stderr, ``Error: <scenario>: <message>``, as
    ``scenario-expand`` and a ``--multirun`` batch print it
    (``examples/expand.py``), and exit 1: a configuration error
    (docs/usage.md, "Exit Codes").  The scenario is the config's
    ``scenario`` choice; without one the line names none.  With
    ``HYDRA_FULL_ERROR=1`` *exc* is raised again, traceback and all, as
    ``scenario-expand`` does.

    Raises:
        ValueError: *exc*, with ``HYDRA_FULL_ERROR=1``.
        SystemExit: With status 1, otherwise.
    """
    if os.environ.get("HYDRA_FULL_ERROR") == "1":
        raise exc
    name = OmegaConf.select(cfg, "hydra.runtime.choices.scenario", default=None)
    where = f"{name}: " if name else ""
    print(f"Error: {where}{exc}", file=sys.stderr)  # noqa: T201
    sys.exit(1)


def _read_controls(cfg: DictConfig) -> dict[str, Any]:
    """The config's ``controls`` (:func:`.expand.scenario_controls`), or exit 1.

    Invalid ``controls`` are reported as ``scenario-expand`` and a
    ``--multirun`` batch report them (``examples/expand.py``): one line on
    stderr per problem, ``<scenario>: <path>: <why>``
    (:func:`~..odd.controls.format_diagnostics`), and exit 1, a
    configuration error (docs/usage.md, "Exit Codes").  The scenario is the
    config's ``scenario`` choice; without one the lines name none.  Unlike
    ``scenario-expand``, with ``HYDRA_FULL_ERROR=1`` the
    :class:`~..odd.controls.ControlsError` is raised instead, traceback and
    all.

    Raises:
        ControlsError: With ``HYDRA_FULL_ERROR=1``, when they are invalid.
        SystemExit: With status 1, when they are invalid, otherwise.
    """
    from ..odd.controls import ControlsError, format_diagnostics  # noqa: PLC0415

    try:
        return scenario_controls(cfg)
    except ControlsError as exc:
        if os.environ.get("HYDRA_FULL_ERROR") == "1":
            raise
        name = OmegaConf.select(cfg, "hydra.runtime.choices.scenario", default=None)
        for line in format_diagnostics(exc.diagnostics, name or ""):
            print(line, file=sys.stderr)  # noqa: T201
        sys.exit(1)


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
        retry then runs beside what the failed attempt left, and a job given
        up leaves no error marker (``case_errored.json``).
        """
        paths = self._job_paths(batch, idx)
        return None if paths is None else (paths[1], paths[2])

    def _job_paths(
        self, batch: tuple[str, ...], idx: int
    ) -> tuple[Path, Path, str | None] | None:
        """:meth:`_job_output_dir`, with the sweep directory
        (``hydra.sweep.dir``, absolute) first: where the sweep's plan goes."""
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
                sweep_dir = directory
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
        return (
            Path(os.path.abspath(sweep_dir)),
            Path(os.path.abspath(directory)),
            job_log,
        )

    def _plan_jobs(
        self, batches: list[tuple[str, ...]], idx_offset: int
    ) -> list[tuple[Path, str | None] | None]:
        """The directory and job log of each of *batches* (job
        ``idx_offset + i``), as :meth:`_job_output_dir` gives them, after
        writing the sweep's plan of them (``sweep_plan.json``,
        :mod:`..gate.sweep_plan`) into its sweep directory, merged into any
        plan an earlier sweep left there (a ``--resume-from`` rerun into the
        same ``hydra.sweep.dir`` keeps the jobs it skips).

        A job whose directory cannot be told is left out of the plan; a plan
        that cannot be written is logged, and the sweep runs anyway.
        """
        found = [
            self._job_paths(batch, local_idx + idx_offset)
            for local_idx, batch in enumerate(batches)
        ]
        plans: dict[Path, list[tuple[Path, list[str]]]] = {}
        for batch, paths in zip(batches, found):
            if paths is not None:
                plans.setdefault(paths[0], []).append((paths[1], list(batch)))
        unplanned = sum(1 for paths in found if paths is None)
        if unplanned:
            logger.warning(
                "%d job(s) of the sweep are left out of its plan: their "
                "output directory cannot be told; scenario-gate will not see "
                "them if they never run.",
                unplanned,
            )
        for sweep_dir, jobs in plans.items():
            try:
                path = write_sweep_plan(sweep_dir, jobs)
            except OSError as exc:
                logger.warning(
                    "Cannot write the sweep's plan in %s (%s); scenario-gate "
                    "will not see the jobs it never runs.",
                    sweep_dir,
                    exc,
                )
            else:
                logger.info("Planned %d job(s) in %s.", len(jobs), path)
        return [None if paths is None else (paths[1], paths[2]) for paths in found]

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
                value its runs refuse, a ``sweep.job_timeout_seconds`` or
                ``sweep.resume_from`` (or ``SWEEP_RESUME_FROM``) it refuses,
                or asks for Rerun recordings or T4 datasets without the
                extra, or when a case is refused;
                with status 130 (``SIGINT``) or 143 (``SIGTERM``) when the
                sweep is interrupted.
            SystemExit: With status 1, at once, before any case runs, when
                the sweep is invalid -- it has no ``sweep`` section, its
                constraints name more than one key, its Lanelet2 map is not
                named, is missing, or cannot be resolved, fetched or read
                (:func:`.expand.load_sweep_map`, the message naming it), or
                :func:`.expand.expand_sweep` refuses it -- after printing
                ``Error: <scenario>: <message>`` to stderr, as
                ``scenario-expand`` does; or when its ``controls`` are
                invalid, before its map is loaded, after printing each
                problem as ``<scenario>: <path>: <why>``, one line each.
            ValueError: Instead of that exit, with ``HYDRA_FULL_ERROR=1``
                (for invalid ``controls``, a :class:`~..odd.controls.ControlsError`).
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
        # Each refusal of the sweep's own config -- here, loading its map and
        # expanding it -- is one line, `Error: <scenario>: <why>`, exit 1, as
        # scenario-expand reports it, not Hydra's traceback (#1705).
        try:
            sweep_dict = _read_sweep_section(cfg)
        except ValueError as exc:
            _refuse_invalid_sweep(cfg, exc)
        # Its `controls` too: they do not depend on the map, and invalid ones
        # are refused one line per problem, exit 1, before a `map.source` is
        # cloned (#1710).
        controls = _read_controls(cfg)
        # Checked here, before any map is fetched or case runs, and refused
        # as step 0 refuses a config value: exit 2, naming the key.
        try:
            job_timeout = _job_timeout_seconds(sweep_dict.get("job_timeout_seconds"))
            # `--resume-from` (the variable) wins over `sweep.resume_from`;
            # a bad value of either is refused here, not truncated or
            # raised from int() mid-sweep (#1784).
            resume_from = _resume_from(
                sweep_dict.get("resume_from"), os.environ.get(_RESUME_FROM_ENV)
            )
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
        # (#1447), in one line (#1705).
        try:
            lanelet_map = load_sweep_map(
                OmegaConf.select(cfg, "map"), prefix="LaneletConstraintSweeper: "
            )
        except ValueError as exc:
            _refuse_invalid_sweep(cfg, exc)

        # -- 4-6. Expand: matching lanelets -> one override batch each -----
        odd = OmegaConf.select(cfg, "odd")
        seed = _config_seed(cfg)
        try:
            batches: list[tuple[str, ...]] = [
                tuple(case)
                for case in expand_sweep(
                    sweep_dict,
                    lanelet_map,
                    arguments,
                    odd=odd,
                    controls=controls,
                    seed=seed,
                    rules=scenario_rules(cfg),
                    environment_rules=environment_rules(cfg),
                )
            ]
        except ValueError as exc:
            # An invalid sweep (a weights file that does not parse or names a
            # module the ODD does not have, a constraint that does not parse,
            # a case_index past its cases, ...).  Other errors keep their
            # traceback.
            _refuse_invalid_sweep(cfg, exc)
        if not batches:
            # Exit 0 means every case passed: a sweep that tests nothing
            # (a misspelt constraint, the wrong map) must not look like one.
            raise SweepFailedError(f"Sweep failed: {EMPTY_SWEEP_MESSAGE}.")

        logger.info(
            "Sweeping %d case(s): %s",
            len(batches),
            [b[0] if b else "" for b in batches],
        )

        # -- 7. Resolve cooldown and retry count ---
        # (job_timeout and resume_from: read and checked at step 1.)
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
            # Where the jobs it skips write: a directory one of them shares
            # with a job it runs keeps their runs (#1956).
            skipped_dirs = [
                self._job_paths(batch, i)
                for i, batch in enumerate(batches[:skip_count])
            ]
            batches = batches[skip_count:]
            # Offset indices so logs still show the original 1-based position.
            idx_offset = skip_count
        else:
            skipped_dirs = []
            idx_offset = 0

        # -- 9. Plan the jobs, so that scenario-gate sees those a sweep killed
        # before it reached them never ran (#1833).  After every refusal
        # above: a sweep refused before it runs leaves no plan.
        job_dirs = self._plan_jobs(batches, idx_offset)

        # -- 10. Launch batches one-by-one via fork isolation --
        all_returns: list[Any] = []
        failed_count = 0
        timed_out_count = 0
        first_job = True
        # The job directories one job of the whole sweep owns: only there is
        # what an earlier sweep left replaced before the job's first attempt
        # (#1956).  A directory more than one job maps to (one
        # hydra.sweep.subdir for all) also holds the runs of the jobs that
        # share it, those a --resume-from skips included, so it is left as
        # it was.
        owned_dirs = _owned_dirs(job_dirs, skipped_dirs)

        for local_idx, batch in enumerate(batches):
            idx = local_idx + idx_offset
            self._current_case = idx + 1

            succeeded = False
            exit_code: int | None = None
            # The job's directory, where a job given up leaves its marker
            # (#1830).
            job_dir = job_dirs[local_idx]
            stems_before: set[str] = set()
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

                if attempt > 0 and job_dir is not None and max_attempts > 1:
                    # As the queue and the SLURM worker do before a retry:
                    # what the failed attempt wrote (a result, a coverage
                    # file, a trajectory log ...) makes way for the retry's,
                    # which then takes the same stem.  Left behind, the
                    # viewer, which shows a job's first stem, and the
                    # coverage merge, which reads every coverage file, would
                    # read the failed attempt's (#1373).  The services'
                    # logs, appended to, stay to explain the failed attempt,
                    # and an output that cannot be removed is logged, not raised,
                    # as before the first attempt (#2070).
                    directory, job_log = job_dir
                    for stem in sorted(
                        _run_output_stems(directory, job_log) - stems_before
                    ):
                        if _is_service_log(directory, stem):
                            continue
                        logger.info(
                            "[%d/%d] Removing the outputs of the failed "
                            "attempt %s in %s before the retry.",
                            idx + 1,
                            idx_offset + len(batches),
                            stem,
                            directory,
                        )
                        try:
                            remove_run_outputs(directory, stem)
                        except OSError as exc:
                            logger.warning(
                                "Cannot remove the outputs of the failed "
                                "attempt %s in %s (%s).",
                                stem,
                                directory,
                                exc,
                            )

                if attempt == 0 and job_dir is not None:
                    if job_dir[0] in owned_dirs:
                        # A job run again into its directory (hydra.sweep.dir
                        # naming one an earlier sweep wrote in, a
                        # --resume-from rerun) replaces the earlier run, as
                        # its plan entry does: what that run left (its
                        # result, gate record, coverage file ...) goes, so
                        # that the gate, the coverage merge, the viewer and
                        # the verdict read this run's alone (#1956).  Right
                        # before the marker is written, after the cooldown,
                        # so that a sweep killed in between rarely leaves
                        # the directory emptied and unmarked.
                        _remove_earlier_outputs(
                            job_dir, f"{idx + 1}/{idx_offset + len(batches)}"
                        )
                    if max_attempts > 1:
                        # The stems already there before the first attempt
                        # (none, unless the directory is shared or the
                        # outputs above could not be removed): a retry
                        # removes only what the failed attempts of this job
                        # left (#1373).
                        stems_before = _run_output_stems(job_dir[0], job_dir[1])

                # The job did not finish, until it ends with a verdict: a
                # sweep killed with it (a segfault, the OOM killer, SIGKILL)
                # leaves this marker, which scenario-gate refuses the sweep
                # for (#1833).  It replaces any an earlier sweep left in the
                # same directory, so this run of the job decides afresh.
                _write_unfinished_marker(job_dir, batch, attempt + 1)
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
                    _write_marker(job_dir, batch, exit_code, attempt + 1)
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
            if (succeeded or exit_code == _FAILED_VERDICT_EXIT_CODE) and (
                job_dir is not None
            ):
                # Ended with a verdict, passed or failed: its run wrote its
                # result (and gate record), and is judged.
                _remove_marker(job_dir[0])
            if not succeeded:
                # Distinguish timeouts from other failures for the summary.
                if exit_code is None:
                    timed_out_count += 1
                failed_count += 1
                if exit_code != _FAILED_VERDICT_EXIT_CODE:
                    # Raised, timed out or crashed on every attempt: no
                    # result, so no gate record; the marker says so to
                    # scenario-gate (#1830).  A failed verdict wrote its
                    # record, and is judged.
                    _write_marker(job_dir, batch, exit_code, attempt + 1, job_timeout)
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


def _marker_reason(exit_code: int | None, timeout: int | None = None) -> str:
    """Why a job was given up, in words, from its last exit status."""
    if exit_code is None:
        return "timed out" if not timeout else f"timed out after {timeout}s"
    if exit_code < 0:
        try:
            name = signal.Signals(-exit_code).name
        except ValueError:
            name = f"signal {-exit_code}"
        return f"crashed ({name})"
    if exit_code == _REFUSED_EXIT_CODE:
        return f"refused (exit code {exit_code})"
    return f"exit code {exit_code}"


def _owned_dirs(
    job_dirs: list[tuple[Path, str | None] | None],
    skipped: list[tuple[Path, Path, str | None] | None],
) -> set[Path]:
    """The directories of *job_dirs* (the jobs a sweep runs) that no other
    job of the sweep maps to, *skipped* (those a ``--resume-from`` skips,
    as :meth:`LaneletConstraintSweeper._job_paths` gives them) included.
    None at all when the directory of a job cannot be told: it may be any."""
    counts: dict[Path, int] = {}
    for job_dir in job_dirs:
        if job_dir is None:
            return set()
        counts[job_dir[0]] = counts.get(job_dir[0], 0) + 1
    for paths in skipped:
        if paths is None:
            return set()
        counts[paths[1]] = counts.get(paths[1], 0) + 1
    return {directory for directory, count in counts.items() if count == 1}


#: The logs that services a run starts write into its directory under names
#: of their own, which :func:`..run_outputs.run_output_stems` reads as the
#: CARLA recorder log of a run (``X.log``): a containerised backend's
#: (``backend_<name>.log``, appended to), Autoware's (``autoware-<n>-<name>.log``,
#: ``autoware-build.log``, ``autoware-deps.log``) and the viewer runner's.
#: An earlier sweep's are not a run of the job, and stay.
_SERVICE_LOGS: tuple[str, ...] = (
    "backend_*.log",
    "autoware-*.log",
    "raw_output.log",
    "run.log",
)


def _is_service_log(directory: Path, stem: str) -> bool:
    """Whether all a run named *stem* has in *directory* is a log one of
    :data:`_SERVICE_LOGS` names: a service's log, not a run."""
    found = [path for path in run_output_paths(directory, stem) if path.exists()]
    return (
        len(found) == 1
        and found[0].name == f"{stem}.log"
        and any(
            fnmatch.fnmatchcase(found[0].name, pattern) for pattern in _SERVICE_LOGS
        )
    )


def _remove_earlier_outputs(job_dir: tuple[Path, str | None], label: str) -> None:
    """Remove the run outputs an earlier sweep left in *job_dir* (its
    directory and Hydra's job log), before the job's first attempt in this
    sweep: every stem :func:`..run_outputs.run_output_stems` finds there,
    with :func:`..run_outputs.remove_run_outputs`, as a retry removes the
    failed attempt's (#1956).  Hydra's job log, its ``.hydra`` directory, the
    job's marker (``case_errored.json``, replaced as the attempt starts), the
    logs of the services runs start (:data:`_SERVICE_LOGS`) and any other
    file that is not a run's output are left alone.  Logged, never
    raised, when one cannot be removed: the job runs anyway, under the next
    free stem."""
    directory, job_log = job_dir
    for stem in sorted(_run_output_stems(directory, job_log)):
        if _is_service_log(directory, stem):
            continue
        logger.info(
            "[%s] Removing the outputs of the earlier run %s in %s: the job "
            "runs again and replaces it.",
            label,
            stem,
            directory,
        )
        try:
            remove_run_outputs(directory, stem)
        except OSError as exc:
            logger.warning(
                "Cannot remove the outputs of the earlier run %s in %s (%s).",
                stem,
                directory,
                exc,
            )


def _remove_marker(directory: Path) -> None:
    """Remove the marker (``case_errored.json``) of a job that ended with a
    verdict from *directory*; logged, never raised, when it cannot be."""
    try:
        remove_case_errored(directory)
    except OSError as exc:
        logger.warning("Cannot remove the error marker in %s (%s).", directory, exc)


def _write_unfinished_marker(
    job_dir: tuple[Path, str | None] | None, batch: tuple[str, ...], attempt: int
) -> None:
    """Give a job the marker saying it did not finish
    (:func:`..gate.errored.write_case_unfinished`) before its attempt
    *attempt* (from 1); logged, never raised, when it cannot be.  Nothing
    when its directory is not known (:meth:`LaneletConstraintSweeper._job_output_dir`
    logged why)."""
    if job_dir is None:
        return
    try:
        write_case_unfinished(job_dir[0], overrides=list(batch), attempt=attempt)
    except OSError as exc:
        logger.warning(
            "Cannot write the unfinished-job marker in %s (%s).", job_dir[0], exc
        )


def _write_marker(
    job_dir: tuple[Path, str | None] | None,
    batch: tuple[str, ...],
    exit_code: int | None,
    attempts: int,
    timeout: int | None = None,
) -> None:
    """Leave the marker of a job given up (``case_errored.json``,
    :mod:`..gate.errored`) in its directory; logged, never raised, when it
    cannot be, or the directory is not known."""
    if job_dir is None:
        logger.warning(
            "Cannot tell the output directory of the job given up for "
            "overrides %s; it leaves no error marker, and scenario-gate will "
            "not see it.",
            batch,
        )
        return
    try:
        write_case_errored(
            job_dir[0],
            overrides=list(batch),
            exit_code=exit_code,
            attempts=attempts,
            reason=_marker_reason(exit_code, timeout),
        )
    except OSError as exc:
        logger.warning("Cannot write the error marker in %s (%s).", job_dir[0], exc)


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
