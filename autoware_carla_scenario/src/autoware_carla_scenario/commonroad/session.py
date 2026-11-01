"""The run's CommonRoad metrics: evaluated by the worker every tick, published
to :mod:`autoware_carla_scenario.metrics`.

The runner calls :func:`track_commonroad_metrics` every tick, before the
conditions are checked.  It does nothing until a CommonRoad metric is read in
the run (:func:`~autoware_carla_scenario.metrics.commonroad_metrics_wanted`: a
:class:`~autoware_carla_scenario.conditions.MetricCondition` on one, or a
:func:`~autoware_carla_scenario.metrics.read_metric`).  Then it converts the
world's map to CommonRoad (:mod:`.map`, once per machine), starts the worker
(:mod:`.worker`, once per process and map), and every
:attr:`CommonRoadSettings.period_s` of simulated time sends it the frame
(:mod:`.frame`) and publishes what it answers
(:func:`~autoware_carla_scenario.metrics.set_commonroad_values`); between two
evaluations the metrics hold their last values.

A run that reads a CommonRoad metric needs the worker: when it cannot be
started, :func:`track_commonroad_metrics` raises rather than let a fail
condition on a metric that reads nothing pass the run.
"""

from __future__ import annotations

import atexit
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..metrics import commonroad_metrics_wanted, set_commonroad_values
from .frame import read_frame
from .map import commonroad_map
from .worker import CommonRoadWorker, CommonRoadWorkerError, worker_command

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "CommonRoadSettings",
    "close_commonroad_worker",
    "commonroad_settings",
    "configure_commonroad_metrics",
    "end_commonroad_run",
    "track_commonroad_metrics",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommonRoadSettings:
    """How the CommonRoad metrics are evaluated.

    Attributes:
        period_s: Simulated seconds between two evaluations; the metrics hold
            their values between.  ``0`` evaluates every tick.
        dt: The time step of the CommonRoad scenario a frame becomes (s).
        horizon_s: How far ahead the road users' trajectories run on, and the
            reachable set is computed (s).
        radius_m: How far from the ego road users and lanelets are taken in.
        reach_settings: Overrides of CommonRoad-Reach's configuration, by
            section and key (``{"planning": {"size_grid": 0.3}}``).
        cache_dir: Where the converted maps are kept; ``None`` for
            :func:`~.map.default_cache_dir`.
        command: The command starting the worker; ``None`` finds it
            (:func:`~.worker.worker_command`).
    """

    period_s: float = 0.2
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


class _Session:
    settings: CommonRoadSettings
    worker: Optional[CommonRoadWorker]
    map_path: Optional[Path]
    #: The name of the map the worker reads, so a new map is noticed.
    map_name: Optional[str]
    last: Optional[float]
    reported: set[str]

    def __init__(self) -> None:
        self.settings = CommonRoadSettings()
        self.worker = None
        self.map_path = None
        self.map_name = None
        self.last = None
        self.reported = set()


_SESSION = _Session()


def configure_commonroad_metrics(**changes: Any) -> CommonRoadSettings:
    """Change how the CommonRoad metrics are evaluated
    (:class:`CommonRoadSettings`' fields); the settings now.  A change to how
    the worker evaluates restarts it."""
    settings = replace(_SESSION.settings, **changes)
    if settings.worker_settings() != _SESSION.settings.worker_settings() or (
        settings.command != _SESSION.settings.command
    ):
        close_commonroad_worker()
    _SESSION.settings = settings
    return settings


def commonroad_settings() -> CommonRoadSettings:
    """The settings the CommonRoad metrics are evaluated with."""
    return _SESSION.settings


def _worker_for(world: "carla.World") -> CommonRoadWorker:
    """The worker on *world*'s map, started (and the map converted) if need be."""
    carla_map = world.get_map()
    name = str(carla_map.name)
    worker = _SESSION.worker
    if worker is not None and _SESSION.map_name == name:
        return worker
    close_commonroad_worker()
    settings = _SESSION.settings
    try:
        map_path = commonroad_map(carla_map.to_opendrive(), settings.cache_dir)
    except ImportError as exc:
        raise CommonRoadWorkerError(str(exc)) from exc
    except Exception as exc:  # roadgen raises ValueError on a map it refuses
        raise CommonRoadWorkerError(
            f"roadgen could not convert {name} to CommonRoad: {exc}"
        ) from exc
    command = list(settings.command) if settings.command else worker_command()
    logger.info("starting the CommonRoad worker: %s", " ".join(command))
    worker = CommonRoadWorker(command, map_path, settings.worker_settings())
    _SESSION.worker, _SESSION.map_path, _SESSION.map_name = worker, map_path, name
    return worker


def track_commonroad_metrics(world: "carla.World", elapsed: float) -> None:
    """Evaluate the CommonRoad metrics read in the run on *world*'s frame at
    *elapsed* simulated seconds, and publish them; nothing while none is read,
    or before :attr:`CommonRoadSettings.period_s` has passed since the last.

    Raises:
        CommonRoadWorkerError: A metric is read and the worker cannot be
            started, or failed.
    """
    wanted = commonroad_metrics_wanted()
    if not wanted:
        return
    last = _SESSION.last
    if last is not None and 0.0 <= elapsed - last < _SESSION.settings.period_s:
        return
    worker = _worker_for(world)
    frame = read_frame(world, _SESSION.settings.radius_m)
    _SESSION.last = elapsed
    if frame is None:
        set_commonroad_values({})
        return
    values, errors = worker.evaluate(wanted, frame.ego, frame.others)
    for key, error in errors.items():
        if key not in _SESSION.reported:
            _SESSION.reported.add(key)
            logger.warning("CommonRoad metric %s reads nothing: %s", key, error)
    set_commonroad_values(values)


def end_commonroad_run() -> None:
    """Forget the run's evaluations; the worker stays up for the next run."""
    _SESSION.last = None
    _SESSION.reported = set()


def close_commonroad_worker() -> None:
    """Stop the worker, if one runs."""
    worker = _SESSION.worker
    _SESSION.worker, _SESSION.map_path, _SESSION.map_name = None, None, None
    _SESSION.last = None
    if worker is not None:
        worker.close()


atexit.register(close_commonroad_worker)
