"""The CommonRoad metrics: CriMe's criticality measures and CommonRoad-Reach's
drivable area (:data:`autoware_carla_scenario.metrics.COMMONROAD_METRICS`),
computed once a run is over.

CommonRoad's packages run in worker processes of their own (:mod:`.worker`)
on the world's map, which roadgen converts to CommonRoad (:mod:`.map`); the
run's recorded frames are shared out among them (:mod:`.session`).
``docs/metrics.md`` ("CommonRoad metrics") says how to set them up.
"""

from .map import commonroad_map
from .session import (
    CommonRoadSettings,
    close_commonroad_workers,
    commonroad_settings,
    configure_commonroad_metrics,
    evaluate_commonroad,
)
from .worker import WORKER_ENV, CommonRoadWorker, CommonRoadWorkerError, worker_command

__all__ = [
    "WORKER_ENV",
    "CommonRoadSettings",
    "CommonRoadWorker",
    "CommonRoadWorkerError",
    "close_commonroad_workers",
    "commonroad_map",
    "commonroad_settings",
    "configure_commonroad_metrics",
    "evaluate_commonroad",
    "worker_command",
]
