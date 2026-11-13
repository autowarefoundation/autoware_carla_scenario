"""The CommonRoad metrics: CriMe's criticality measures and CommonRoad-Reach's
drivable area (:data:`autoware_carla_scenario.metrics.COMMONROAD_METRICS`),
computed once a run is over.

CommonRoad's packages run in worker processes of their own (:mod:`.worker`)
on the world's map, which roadgen converts to CommonRoad (:mod:`.map`); the
run's recorded frames are shared out among them (:mod:`.session`).
``docs/metrics.md`` ("CommonRoad metrics") says how to set them up.
"""

# The heavy metrics' settings are the checked heavy_metrics' (#1689), which
# commonroad.session re-exports too: the same objects, imported from where they
# are defined.
from ..heavy_metrics import (
    DEFAULT_METRIC_PERIODS_S,
    HEAVY_METRICS,
    HEAVY_METRICS_DECIDED_ENV,
    HEAVY_METRICS_ENV,
)
from .map import commonroad_map
from .session import (
    CommonRoadSettings,
    close_commonroad_workers,
    commonroad_settings,
    configure_commonroad_metrics,
    evaluate_commonroad,
)
from .worker import CommonRoadWorker

# The command that starts the worker and its error are the checked
# worker_start's (#2116), which commonroad.worker re-exports too: the same
# objects, imported from where they are defined.
from .worker_start import WORKER_ENV, CommonRoadWorkerError, worker_command

__all__ = [
    "DEFAULT_METRIC_PERIODS_S",
    "HEAVY_METRICS",
    "HEAVY_METRICS_DECIDED_ENV",
    "HEAVY_METRICS_ENV",
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
