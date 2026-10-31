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
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, JobStatus
from hydra.errors import CompactHydraException
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .. import hydra_compat  # noqa: F401  (Python 3.14 argparse)
from ..maps import resolve_map_paths
from ..rerun import RerunUnavailableError, require_converter
from .expand import check_constraint_keys, expand_sweep, scenario_controls
from .isolation import (  # noqa: F401  (SWEEPER_OWNS_RETRIES_ENV: re-exported)
    SWEEPER_OWNS_RETRIES_ENV,
    _interrupt_signal,
    _launch_isolated,
    _raise_interrupted,
)
from .map_loader import load_map

logger = logging.getLogger(__name__)

# Default hard timeout per job (seconds).  Can be overridden via
# ``sweep.job_timeout_seconds`` in the scenario YAML.
_DEFAULT_JOB_TIMEOUT = 120

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


def _read_verdict(output_dir: Path) -> bool | None:
    """The verdict of the scenario result JSON in a job's output directory.

    Returns:
        ``True`` if the scenario passed, ``False`` if it ran and failed, or
        ``None`` if there is no verdict: no result JSON (the run raised
        before :class:`ScenarioRunner` could write it) or one that cannot be
        read.
    """
    result_files = sorted(output_dir.glob("*_result.json"))
    if not result_files:
        return None
    try:
        data = json.loads(result_files[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError: not JSON, or not UTF-8
        return None
    if not isinstance(data, dict):
        return None
    return bool(data.get("passed", True))


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


def _job_exit_code(job_returns: Any) -> int:
    """The exit status of a forked case, from what ``launcher.launch`` returned.

    ``launcher.launch()`` completes without error even when the scenario
    *fails* (in --multirun mode ``_hydra_main`` does not ``sys.exit``), and
    Hydra catches what the task function raises and stores it in the
    ``JobReturn`` with ``status=FAILED``.  So:

    * 1 -- the task raised, or completed without a readable result JSON;
    * :data:`_FAILED_VERDICT_EXIT_CODE` -- the scenario ran and failed;
    * 0 -- it passed (or there is no output directory to read a verdict from).
    """
    if not job_returns:
        return 0
    ret = job_returns[0]
    if ret.status != JobStatus.COMPLETED:
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
            SystemExit: With status 2, at once, when the sweep asks for
                Rerun recordings without the extra or a case is refused;
                with status 130 (``SIGINT``) or 143 (``SIGTERM``) when the
                sweep is interrupted.
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

        # -- 0. Refuse at once a sweep asking for Rerun recordings without
        # the extra, rather than fail every case it forks.  Each case
        # converts its own log (examples/run.py).
        if OmegaConf.select(cfg, "rerun.enabled", default=False):
            try:
                require_converter()
            except RerunUnavailableError as exc:
                print(exc, file=sys.stderr)  # noqa: T201
                sys.exit(_REFUSED_EXIT_CODE)

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

        # -- 2. Resolve map paths ------------------------------------------
        # A map named by `map.source` is cloned (or found already downloaded)
        # here, so a sweep runs against a published HD map without anyone
        # having to name a local file for it first; `map.directory` names a
        # map on this machine and is read in place.
        paths = resolve_map_paths(OmegaConf.select(cfg, "map"))

        # -- 3. Load the Lanelet2 map (lightweight) ------------------------
        try:
            lanelet_map = load_map(paths)
        except FileNotFoundError as exc:
            raise ValueError(f"LaneletConstraintSweeper: {exc}") from exc

        # -- 4-6. Expand: matching lanelets -> one override batch each -----
        batches: list[tuple[str, ...]] = [
            tuple(case)
            for case in expand_sweep(
                sweep_dict,
                lanelet_map,
                arguments,
                odd=OmegaConf.select(cfg, "odd"),
                controls=scenario_controls(cfg),
            )
        ]
        if not batches:
            # Exit 0 means every case passed: a sweep that tests nothing
            # (a misspelt constraint, the wrong map) must not look like one.
            raise SweepFailedError(
                "Sweep failed: the sweep expands to no case (its constraints, "
                "route or ODD sample match nothing on this map)."
            )

        logger.info(
            "Sweeping %d case(s): %s",
            len(batches),
            [b[0] if b else "" for b in batches],
        )

        # -- 7. Resolve per-job timeout, cooldown, retry count, and resume ---
        job_timeout: int = int(
            sweep_dict.get("job_timeout_seconds", _DEFAULT_JOB_TIMEOUT)
        )
        cooldown: float = float(
            OmegaConf.select(cfg, "server.cooldown_seconds", default=0.0)
        )
        max_retries: int = int(
            OmegaConf.select(cfg, "server.cooldown_max_retries", default=0)
        )
        if max_retries < 0:
            raise ValueError(
                "LaneletConstraintSweeper: server.cooldown_max_retries must be "
                f">= 0, got {max_retries}."
            )
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
