"""The CommonRoad metrics: CriMe's criticality measures and CommonRoad-Reach's
drivable area, read off the world as built-in metrics
(:data:`autoware_carla_scenario.metrics.COMMONROAD_METRICS`).

CommonRoad's packages run in a worker process of their own (:mod:`.worker`) on
the world's map, which roadgen converts to CommonRoad (:mod:`.map`); the
runner sends it the frame every tick a CommonRoad metric is read in the run
(:mod:`.session`).  ``docs/metrics.md`` ("CommonRoad metrics") says how to set
it up.
"""

from .map import commonroad_map
from .session import (
    CommonRoadSettings,
    close_commonroad_worker,
    commonroad_settings,
    configure_commonroad_metrics,
    end_commonroad_run,
    track_commonroad_metrics,
)
from .worker import WORKER_ENV, CommonRoadWorker, CommonRoadWorkerError, worker_command

__all__ = [
    "WORKER_ENV",
    "CommonRoadSettings",
    "CommonRoadWorker",
    "CommonRoadWorkerError",
    "close_commonroad_worker",
    "commonroad_map",
    "commonroad_settings",
    "configure_commonroad_metrics",
    "end_commonroad_run",
    "track_commonroad_metrics",
    "worker_command",
]
