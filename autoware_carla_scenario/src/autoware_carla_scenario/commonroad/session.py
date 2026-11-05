"""The CommonRoad metrics of a run: its recorded frames, evaluated by workers
running at once.

When a run that reads a CommonRoad metric is over,
:func:`~autoware_carla_scenario.metric_batch.evaluate_metrics` hands
:func:`evaluate_commonroad` the frames every :attr:`CommonRoadSettings.period_s`
-- every :meth:`CommonRoadSettings.period_for` a metric, for the slow ones
(:attr:`CommonRoadSettings.metric_periods_s`).  The heaviest
(:data:`~autoware_carla_scenario.heavy_metrics.HEAVY_METRICS`) can be turned
off altogether (:attr:`CommonRoadSettings.heavy_metrics`).
It converts the world's map to CommonRoad (:mod:`.map`, once per machine),
starts :attr:`CommonRoadSettings.workers` workers on it (:mod:`.worker`, kept
for the next run on the map), and shares the frames out among them, each
worker evaluating its share in turn while the others do theirs.

A run that reads a CommonRoad metric needs the workers: when they cannot be
started, :func:`evaluate_commonroad` raises rather than let a fail condition
on a metric that reads nothing pass the run.
"""

from __future__ import annotations

import atexit
import logging
import operator
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# Re-exported under the names they had: the heavy metrics' settings live in
# the checked heavy_metrics (#1689).
from ..heavy_metrics import (  # noqa: F401
    DEFAULT_METRIC_PERIODS_S,
    HEAVY_METRICS,
    HEAVY_METRICS_DECIDED_ENV,
    HEAVY_METRICS_ENV,
    _check_period,
    _heavy_metrics_default,
    _heavy_metrics_from_user,
    _heavy_metrics_on,
    evaluates,
)
from ..metrics import COMMONROAD_METRICS
from .frame import read_frame
from .map import commonroad_map
from .worker import CommonRoadWorker, CommonRoadWorkerError, worker_command


if TYPE_CHECKING:
    from ..metric_batch import Frame

__all__ = [
    "CommonRoadSettings",
    "close_commonroad_workers",
    "commonroad_settings",
    "configure_commonroad_metrics",
    "evaluate_commonroad",
]

logger = logging.getLogger(__name__)


def _default_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _check_workers(workers: Any) -> int:
    """*workers* as an ``int``: an integer (a NumPy one too, not a bool) of at
    least 1."""
    try:
        count = None if isinstance(workers, bool) else operator.index(workers)
    except TypeError:
        count = None
    if count is None or count < 1:
        raise ValueError(f"workers must be an integer of at least 1, got {workers!r}")
    return count


@dataclass(frozen=True)
class CommonRoadSettings:
    """How the CommonRoad metrics are evaluated.

    Attributes:
        period_s: Simulated seconds between two frames evaluated.
        metric_periods_s: The same, for each metric evaluated at a period of
            its own (a slow one less often); the others, every
            :attr:`period_s`.  By default :data:`DEFAULT_METRIC_PERIODS_S`.
        heavy_metrics: Whether the heavy metrics (:data:`HEAVY_METRICS`) are
            evaluated at all; off, a metric condition on one is not judged.
            On unless ``$AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS`` is ``0``
            (or the ``scenario`` command decided otherwise,
            :data:`HEAVY_METRICS_DECIDED_ENV`).
        workers: How many workers evaluate frames at once; at least 1.
        dt: The time step of the CommonRoad scenario a frame becomes (s);
            positive and finite.
        horizon_s: How far ahead the road users' trajectories run on, and the
            reachable set is computed (s); positive and finite.
        radius_m: How far from the ego road users and lanelets are taken in;
            positive and finite.
        reach_settings: Overrides of CommonRoad-Reach's configuration, by
            section and key.
        cache_dir: Where the converted maps are kept; ``None`` for
            :func:`~.map.default_cache_dir`.
        command: The command starting a worker; ``None`` finds it
            (:func:`~.worker.worker_command`).
    """

    period_s: float = 0.1
    metric_periods_s: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_METRIC_PERIODS_S)
    )
    heavy_metrics: bool = field(default_factory=_heavy_metrics_default)
    workers: int = field(default_factory=_default_workers)
    dt: float = 0.1
    horizon_s: float = 3.0
    radius_m: float = 100.0
    reach_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_dir: Optional[Path] = None
    command: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        _check_period("period_s", self.period_s)
        for key, period in self.metric_periods_s.items():
            if key not in COMMONROAD_METRICS:
                raise ValueError(
                    f"metric_periods_s: {key!r} is not a CommonRoad metric "
                    f"({', '.join(COMMONROAD_METRICS)})"
                )
            _check_period(f"metric_periods_s[{key!r}]", period)
        object.__setattr__(self, "workers", _check_workers(self.workers))
        _check_period("dt", self.dt)
        _check_period("horizon_s", self.horizon_s)
        _check_period("radius_m", self.radius_m)

    def evaluates(self, key: str) -> bool:
        """Whether the metric *key* is evaluated: not when it is heavy and the
        heavy metrics are off (:func:`~autoware_carla_scenario.heavy_metrics.evaluates`)."""
        return evaluates(self.heavy_metrics, key)

    def period_for(self, key: str) -> float:
        """Simulated seconds between two frames the metric *key* is evaluated
        on."""
        return self.metric_periods_s.get(key, self.period_s)

    def worker_settings(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "horizon_steps": max(1, round(self.horizon_s / self.dt)),
            "radius_m": self.radius_m,
            "reach_settings": self.reach_settings,
        }


class _Pool:
    settings: CommonRoadSettings
    workers: list[CommonRoadWorker]
    #: The name of the map the workers read, so a new map is noticed.
    map_name: Optional[str]

    def __init__(self) -> None:
        self.settings = CommonRoadSettings()
        self.workers = []
        self.map_name = None


_POOL = _Pool()
#: Held while the pool's workers are started, swapped or closed.
_POOL_LOCK = threading.RLock()


def configure_commonroad_metrics(**changes: Any) -> CommonRoadSettings:
    """Change how the CommonRoad metrics are evaluated
    (:class:`CommonRoadSettings`' fields); the settings now.  A change to how
    the workers evaluate, or to how many there are, restarts them.

    A field given replaces the one set, ``metric_periods_s`` too: to change
    one metric's period and keep the others', start from those set
    (``metric_periods_s={**commonroad_settings().metric_periods_s,
    "crime_wttr_s": 0.3}``).
    """
    with _POOL_LOCK:
        settings = replace(_POOL.settings, **changes)
        old = _POOL.settings
        if (
            settings.worker_settings() != old.worker_settings()
            or settings.command != old.command
            or settings.workers != old.workers
        ):
            close_commonroad_workers()
        _POOL.settings = settings
        return settings


def commonroad_settings() -> CommonRoadSettings:
    """The settings the CommonRoad metrics are evaluated with."""
    return _POOL.settings


def _workers_for(carla_map: Any) -> list[CommonRoadWorker]:
    """The workers on *carla_map*, started (and the map converted) if need be:
    started afresh when one of those pooled has exited."""
    with _POOL_LOCK:
        name = str(carla_map.name)
        workers = _POOL.workers
        if (
            workers
            and _POOL.map_name == name
            and all(worker.running() for worker in workers)
        ):
            return workers
        close_commonroad_workers()
        started = _start_workers(carla_map, name)
        _POOL.workers, _POOL.map_name = started, name
        _close_at_exit()
        return started


#: The process that registered :func:`close_commonroad_workers` with
#: ``atexit``, if any.
_CLOSE_AT_EXIT_PID: Optional[int] = None


def _close_at_exit() -> None:
    """Have this process close its workers at exit, once.

    Per process, not at import: a sweep's case, forked
    (``sweeper/isolation.py``), runs none of the ``atexit`` handlers it
    inherited from the sweep, only those it registered itself (#1568).
    """
    global _CLOSE_AT_EXIT_PID
    if _CLOSE_AT_EXIT_PID != os.getpid():
        _CLOSE_AT_EXIT_PID = os.getpid()
        atexit.register(close_commonroad_workers)


def _start_workers(carla_map: Any, name: str) -> list[CommonRoadWorker]:
    """:attr:`CommonRoadSettings.workers` workers on *carla_map*: all of them
    or, when one cannot be started, none (those started are closed)."""
    settings = _POOL.settings
    try:
        map_path = commonroad_map(carla_map.to_opendrive(), settings.cache_dir)
    except ImportError as exc:
        raise CommonRoadWorkerError(str(exc)) from exc
    except Exception as exc:  # roadgen raises ValueError on a map it refuses
        raise CommonRoadWorkerError(
            f"roadgen could not convert {name} to CommonRoad: {exc}"
        ) from exc
    command = list(settings.command) if settings.command else worker_command()
    logger.info(
        "starting %d CommonRoad workers: %s", settings.workers, " ".join(command)
    )
    # Started together: each spends its first seconds importing CommonRoad.
    with ThreadPoolExecutor(settings.workers) as starting:
        futures: list[Future[CommonRoadWorker]] = [
            starting.submit(
                CommonRoadWorker, command, map_path, settings.worker_settings()
            )
            for _ in range(settings.workers)
        ]
    started = [f.result() for f in futures if f.exception() is None]
    failed = [f.exception() for f in futures if f.exception() is not None]
    if failed:
        _close_all(started)
        raise failed[0]  # type: ignore[misc]
    return started


def evaluate_commonroad(
    frames: list["Frame"], carla_map: Any, keys: list[str]
) -> dict[str, list[Optional[float]]]:
    """The CommonRoad metrics *keys* on each of *frames*, on the run's
    *carla_map*: for each key, a value per frame (``None`` where it reads
    nothing).

    Raises:
        CommonRoadWorkerError: The workers cannot be started, or one failed.
    """
    from ..metric_batch import ReplayWorld  # noqa: PLC0415

    values: dict[str, list[Optional[float]]] = {
        key: [None] * len(frames) for key in keys
    }
    if not frames or not keys:
        return values
    workers = _workers_for(carla_map)
    settings = _POOL.settings
    world = ReplayWorld(carla_map)
    requests: list[Optional[tuple[dict[str, Any], list[dict[str, Any]]]]] = []
    for frame in frames:
        world.show(frame)
        read = read_frame(world, settings.radius_m)  # type: ignore[arg-type]
        requests.append(None if read is None else (read.ego, read.others))
    reported: set[str] = set()

    def share(k: int) -> list[tuple[int, dict[str, Optional[float]], dict[str, str]]]:
        worker = workers[k]
        done = []
        for i in range(k, len(frames), len(workers)):
            request = requests[i]
            if request is None:
                continue
            got, errors = worker.evaluate(keys, request[0], request[1])
            done.append((i, got, errors))
        return done

    try:
        with ThreadPoolExecutor(len(workers)) as pool:
            shares = list(pool.map(share, range(len(workers))))
    except BaseException:
        # A worker that failed, answered out of protocol or was interrupted
        # mid-request is not reused: the next run starts the pool afresh.
        with _POOL_LOCK:
            if _POOL.workers is workers:
                close_commonroad_workers()
        raise
    for done in shares:
        for i, got, errors in done:
            for key in keys:
                values[key][i] = got.get(key)
            for key, error in errors.items():
                if key not in reported:
                    reported.add(key)
                    logger.warning("CommonRoad metric %s reads nothing: %s", key, error)
    return values


def close_commonroad_workers() -> None:
    """Stop the workers, if any run."""
    with _POOL_LOCK:
        workers = _POOL.workers
        _POOL.workers, _POOL.map_name = [], None
    _close_all(workers)


def _close_all(workers: list[CommonRoadWorker]) -> None:
    """Close each of *workers*: one that cannot be closed neither stops the
    others' closing nor masks the error being raised, if any."""
    for worker in workers:
        try:
            worker.close()
        except Exception:  # noqa: BLE001
            logger.warning("could not close a CommonRoad worker", exc_info=True)
