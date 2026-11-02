"""The CommonRoad metrics of a run: its recorded frames, evaluated by workers
running at once.

When a run that reads a CommonRoad metric is over,
:func:`~autoware_carla_scenario.metric_batch.evaluate_metrics` hands
:func:`evaluate_commonroad` the frames every :attr:`CommonRoadSettings.period_s`.
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
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

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


@dataclass(frozen=True)
class CommonRoadSettings:
    """How the CommonRoad metrics are evaluated.

    Attributes:
        period_s: Simulated seconds between two frames evaluated.
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
    workers: int = field(default_factory=_default_workers)
    dt: float = 0.1
    horizon_s: float = 3.0
    radius_m: float = 100.0
    reach_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_dir: Optional[Path] = None
    command: Optional[tuple[str, ...]] = None

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
    the workers evaluate, or to how many there are, restarts them."""
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
