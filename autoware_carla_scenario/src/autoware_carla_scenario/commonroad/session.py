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
on a metric that reads nothing pass the run.  So it does when the workers do
not know a metric (a worker older than the framework, #1953), when one of
them does not answer a frame within :attr:`CommonRoadSettings.request_timeout_s`
(#1954), and when a metric fails in every frame it is evaluated on: a metric
that fails only now and then reads nothing in those frames, as one with no
one to measure from does.
"""

from __future__ import annotations

import atexit
import logging
import operator
import os
import threading
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
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
from .worker import WORKER_ENV, CommonRoadWorker, CommonRoadWorkerError, worker_command


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


def _check_timeout(name: str, timeout: Any) -> Optional[float]:
    """*timeout*, ``None`` or a period (:func:`_check_period`; not a
    bool), as a ``float``."""
    if timeout is None:
        return None
    if isinstance(timeout, bool) or (
        getattr(getattr(timeout, "dtype", None), "kind", None) == "b"
    ):
        raise ValueError(
            f"{name} must be positive and finite, or None, got {timeout!r}"
        )
    _check_period(name, timeout)
    return float(timeout)


def _check_workers(workers: Any) -> int:
    """*workers* as an ``int``: an integer (a NumPy one too, not a bool) of at
    least 1."""
    # numpy's bool_ too: the numpy that runs on Python 3.10 still takes it
    # as an index, with a warning (#1900).
    is_bool = isinstance(workers, bool) or (
        getattr(getattr(workers, "dtype", None), "kind", None) == "b"
    )
    try:
        count = None if is_bool else operator.index(workers)
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
        request_timeout_s: How long a worker may take to answer a frame (all
            the metrics due on it), in wall-clock seconds; one that has not
            answered by then is killed and the run fails with
            :class:`~.worker.CommonRoadWorkerError`.  ``None`` waits as long
            as it takes.  The default leaves a frame many times the 2 s TTR
            takes on one with a collision ahead.
        start_timeout_s: The same, for a worker to start and read the map
            (the first ``uv run`` builds its environment meanwhile).
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
    request_timeout_s: Optional[float] = 120.0
    start_timeout_s: Optional[float] = 1800.0

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
        object.__setattr__(
            self,
            "request_timeout_s",
            _check_timeout("request_timeout_s", self.request_timeout_s),
        )
        object.__setattr__(
            self,
            "start_timeout_s",
            _check_timeout("start_timeout_s", self.start_timeout_s),
        )

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
    # Every worker whose process runs, its map given or not, so that an
    # interrupt closes it: in a session of its own, Ctrl-C does not reach it,
    # and the pool (closed at exit) does not hold it yet (#1780).
    spawned: list[CommonRoadWorker] = []
    spawned_lock = threading.Lock()
    interrupted = False

    def on_start(worker: CommonRoadWorker) -> None:
        with spawned_lock:
            if interrupted:
                # Started after the snapshot below: the worker closes itself.
                raise CommonRoadWorkerError("interrupted while starting")
            spawned.append(worker)

    # Started together: each spends its first seconds importing CommonRoad.
    starting = ThreadPoolExecutor(settings.workers)
    futures: list[Future[CommonRoadWorker]] = [
        starting.submit(
            CommonRoadWorker,
            command,
            map_path,
            settings.worker_settings(),
            on_start=on_start,
            timeout=settings.start_timeout_s,
        )
        for _ in range(settings.workers)
    ]
    try:
        wait(futures)
    except BaseException:
        # Not ``with``: its exit would wait for every worker to take the map.
        starting.shutdown(wait=False, cancel_futures=True)
        with spawned_lock:
            interrupted = True
            closing = list(spawned)
        # A worker still waiting for its init answer reads the end of its
        # output once closed, and raises in its own thread.
        _close_all(closing)
        raise
    starting.shutdown()
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
        CommonRoadWorkerError: The workers cannot be started, do not know a
            key, or one failed (exited, answered out of protocol, or did not
            answer a frame within :attr:`CommonRoadSettings.request_timeout_s`);
            or a key failed in every frame evaluated.
    """
    from ..metric_batch import ReplayWorld  # noqa: PLC0415

    values: dict[str, list[Optional[float]]] = {
        key: [None] * len(frames) for key in keys
    }
    if not frames or not keys:
        return values
    workers = _workers_for(carla_map)
    settings = _POOL.settings
    for worker in workers:
        _check_measures(worker, keys)
    world = ReplayWorld(carla_map)
    requests: list[Optional[tuple[dict[str, Any], list[dict[str, Any]]]]] = []
    for frame in frames:
        world.show(frame)
        read = read_frame(world, settings.radius_m)  # type: ignore[arg-type]
        requests.append(None if read is None else (read.ego, read.others))
    reported: set[str] = set()
    # Set when the evaluation is given up: the shares stop at their next
    # frame rather than evaluate the rest of theirs (#1780).
    stop = threading.Event()

    def share(k: int) -> list[tuple[int, dict[str, Optional[float]], dict[str, str]]]:
        worker = workers[k]
        done = []
        for i in range(k, len(frames), len(workers)):
            if stop.is_set():
                break
            request = requests[i]
            if request is None:
                continue
            got, errors = worker.evaluate(
                keys, request[0], request[1], timeout=settings.request_timeout_s
            )
            _check_answer(worker, keys, got, errors)
            done.append((i, got, errors))
        return done

    # Not ``with``: on an interrupt its exit would wait for every share to
    # evaluate all of its frames before the workers are closed (#1780).
    pool = ThreadPoolExecutor(len(workers))
    futures = [pool.submit(share, k) for k in range(len(workers))]
    try:
        wait(futures, return_when=FIRST_EXCEPTION)
        for future in futures:
            if future.done() and future.exception() is not None:
                raise future.exception()  # type: ignore[misc]
        shares = [future.result() for future in futures]
    except BaseException:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        # A worker that failed, answered out of protocol or was interrupted
        # mid-request is not reused: the next run starts the pool afresh.
        # Closing it also ends the requests of the shares still running:
        # they read the end of its output, and raise in their own threads.
        with _POOL_LOCK:
            if _POOL.workers is workers:
                close_commonroad_workers()
        # Bounded: a share is not waited for longer than a worker's closing.
        # Not those cancelled before they ran, which wait() never sees done.
        wait(
            [future for future in futures if not future.cancelled()],
            timeout=_SHARES_JOIN_S,
        )
        raise
    pool.shutdown()
    evaluated = 0
    failed: dict[str, int] = {}
    first_error: dict[str, str] = {}
    for done in shares:
        for i, got, errors in done:
            evaluated += 1
            for key in keys:
                values[key][i] = got.get(key)
            for key, error in errors.items():
                failed[key] = failed.get(key, 0) + 1
                first_error.setdefault(key, error)
                if key not in reported:
                    reported.add(key)
                    logger.warning("CommonRoad metric %s reads nothing: %s", key, error)
    # A metric that failed in every frame read nothing at all: a fail
    # condition on it would pass the run without having judged it.
    always = [key for key in keys if evaluated and failed.get(key, 0) == evaluated]
    if always:
        raise CommonRoadWorkerError(
            "the CommonRoad metrics "
            + ", ".join(f"{key} (failed: {first_error[key]})" for key in always)
            + f" failed in every frame evaluated ({evaluated}), so read nothing"
        )
    return values


def _check_measures(worker: CommonRoadWorker, keys: list[str]) -> None:
    """Raise when *worker* does not know one of *keys* (its ``init`` answer
    says which it knows): an older worker than the framework, found on the
    ``PATH``, would read nothing in every frame."""
    unknown = [key for key in keys if key not in worker.measures]
    if unknown:
        raise CommonRoadWorkerError(
            f"the CommonRoad worker {worker.command} does not know the metrics "
            f"{', '.join(unknown)} (it knows {', '.join(worker.measures) or 'none'}): "
            "update the commonroad_metrics project it runs, or name another "
            f"command in ${WORKER_ENV}"
        )


def _check_answer(
    worker: CommonRoadWorker,
    keys: list[str],
    got: dict[str, Optional[float]],
    errors: dict[str, str],
) -> None:
    """Raise when *worker*'s answer to a frame asked *keys* leaves one out
    (neither a value nor an error) or calls it unknown: out of protocol."""
    missing = [key for key in keys if key not in got and key not in errors]
    unknown = [key for key in keys if errors.get(key) == _UNKNOWN_MEASURE]
    if missing or unknown:
        raise CommonRoadWorkerError(
            f"the CommonRoad worker {worker.command} answered 'evaluate' out "
            "of protocol: "
            + "; ".join(
                part
                for part in (
                    f"no value nor error for {', '.join(missing)}" if missing else "",
                    f"{', '.join(unknown)} unknown" if unknown else "",
                )
                if part
            )
        )


#: The error the worker gives a measure it does not know.
_UNKNOWN_MEASURE = "unknown measure"


#: How long :func:`evaluate_commonroad`, given up, waits for its shares to
#: stop once their workers are closed, in seconds.
_SHARES_JOIN_S = 5.0


def close_commonroad_workers() -> None:
    """Stop the workers, if any run."""
    with _POOL_LOCK:
        workers = _POOL.workers
        _POOL.workers, _POOL.map_name = [], None
    _close_all(workers)


def _close_all(workers: list[CommonRoadWorker]) -> None:
    """Close each of *workers*, all at once: one that cannot be closed neither
    stops the others' closing nor masks the error being raised, if any.

    At once, so that stuck workers take one close timeout, not one each.
    Each close is bounded, so the threads are joined without a bound.
    """

    def close(worker: CommonRoadWorker) -> None:
        try:
            worker.close()
        except Exception:  # noqa: BLE001
            logger.warning("could not close a CommonRoad worker", exc_info=True)

    if len(workers) == 1:
        close(workers[0])
        return
    threads = [
        threading.Thread(
            target=close, args=(worker,), name="commonroad-close", daemon=False
        )
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
