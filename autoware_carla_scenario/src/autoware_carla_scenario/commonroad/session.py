"""The CommonRoad metrics of a run: its recorded frames, evaluated by workers
running at once.

When a run that reads a CommonRoad metric is over,
:func:`~autoware_carla_scenario.metric_batch.evaluate_metrics` hands
:func:`evaluate_commonroad` the frames every :attr:`CommonRoadSettings.period_s`
-- every :meth:`CommonRoadSettings.period_for` a metric, for the slow ones
(:attr:`CommonRoadSettings.metric_periods_s`).
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
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..metrics import COMMONROAD_METRICS, CRIME_TTR_S
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


#: The metrics evaluated less often than every :attr:`CommonRoadSettings.period_s`
#: by default, and how often (simulated seconds): CriMe's TTR simulates the
#: ego's evasive manoeuvres, about 2 s a frame with a collision ahead.
DEFAULT_METRIC_PERIODS_S: dict[str, float] = {CRIME_TTR_S: 0.5}


@dataclass(frozen=True)
class CommonRoadSettings:
    """How the CommonRoad metrics are evaluated.

    Attributes:
        period_s: Simulated seconds between two frames evaluated.
        metric_periods_s: The same, for each metric evaluated at a period of
            its own (a slow one less often); the others, every
            :attr:`period_s`.  By default :data:`DEFAULT_METRIC_PERIODS_S`.
        workers: How many workers evaluate frames at once.
        dt: The time step of the CommonRoad scenario a frame becomes (s).
        horizon_s: How far ahead the road users' trajectories run on, and the
            reachable set is computed (s).
        radius_m: How far from the ego road users and lanelets are taken in.
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
    workers: int = field(default_factory=_default_workers)
    dt: float = 0.1
    horizon_s: float = 3.0
    radius_m: float = 100.0
    reach_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_dir: Optional[Path] = None
    command: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        periods = {"period_s": self.period_s}
        for key, period in self.metric_periods_s.items():
            if key not in COMMONROAD_METRICS:
                raise ValueError(
                    f"metric_periods_s: {key!r} is not a CommonRoad metric "
                    f"({', '.join(COMMONROAD_METRICS)})"
                )
            periods[f"metric_periods_s[{key!r}]"] = period
        for name, period in periods.items():
            if not (math.isfinite(period) and period > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {period}")

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


def configure_commonroad_metrics(**changes: Any) -> CommonRoadSettings:
    """Change how the CommonRoad metrics are evaluated
    (:class:`CommonRoadSettings`' fields); the settings now.  A change to how
    the workers evaluate, or to how many there are, restarts them.

    ``metric_periods_s`` is merged into the periods set so far: a metric
    given ``None`` goes back to :attr:`CommonRoadSettings.period_s`
    (``configure_commonroad_metrics(metric_periods_s={"crime_ttr_s": 1.0,
    "crime_wttr_s": 0.3})``).
    """
    if "metric_periods_s" in changes:
        periods = dict(_POOL.settings.metric_periods_s)
        for key, period in dict(changes["metric_periods_s"]).items():
            if period is None:
                periods.pop(key, None)
            else:
                periods[key] = float(period)
        changes = {**changes, "metric_periods_s": periods}
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
    """The workers on *carla_map*, started (and the map converted) if need be."""
    name = str(carla_map.name)
    if _POOL.workers and _POOL.map_name == name:
        return _POOL.workers
    close_commonroad_workers()
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
        started = list(
            starting.map(
                lambda _: CommonRoadWorker(
                    command, map_path, settings.worker_settings()
                ),
                range(settings.workers),
            )
        )
    _POOL.workers, _POOL.map_name = started, name
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

    with ThreadPoolExecutor(len(workers)) as pool:
        for done in pool.map(share, range(len(workers))):
            for i, got, errors in done:
                for key in keys:
                    values[key][i] = got.get(key)
                for key, error in errors.items():
                    if key not in reported:
                        reported.add(key)
                        logger.warning(
                            "CommonRoad metric %s reads nothing: %s", key, error
                        )
    return values


def close_commonroad_workers() -> None:
    """Stop the workers, if any run."""
    workers = _POOL.workers
    _POOL.workers, _POOL.map_name = [], None
    for worker in workers:
        worker.close()


atexit.register(close_commonroad_workers)
