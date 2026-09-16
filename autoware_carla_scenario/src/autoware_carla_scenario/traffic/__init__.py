"""traffic -- what drives the vehicles a scenario did not author.

The seam that lets a run's traffic come from somewhere other than CARLA's
TrafficManager: a microscopic traffic simulator, a replay, or nothing at all.

Selecting one is a name::

    uv run scenario traffic.backend=none

and implementing one is a subclass of
:class:`~autoware_carla_scenario.traffic.base.TrafficBackend` registered under a
name of its own -- from this package, or from any package that advertises the
``autoware_carla_scenario.traffic_backends`` entry point.

Built-in backends:

``traffic_manager``
    CARLA's TrafficManager.  The default, and what every scenario written before
    this seam existed gets.
``none``
    No traffic model at all: only the ego and whatever the scenario drives
    itself move.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import (
    LaneChangeDirection,
    LaneChanging,
    NullTrafficBackend,
    TrafficBackend,
    TrafficBackendError,
    TrafficBackendUnavailable,
    TrafficContext,
    TurnDirection,
    TurningAtJunctions,
)
from .config import TrafficConfig, TrafficManagerBackendConfig
from .traffic_manager import TrafficManagerBackend
from .registry import (
    TRAFFIC_BACKEND_ENTRY_POINT_GROUP,
    TrafficBackendFactory,
    available_backends,
    build_backend,
    get_backend_factory,
    load_traffic_backend_plugins,
    register_backend,
    unregister_backend,
)

__all__ = [
    "LaneChangeDirection",
    "LaneChanging",
    "NullTrafficBackend",
    "TrafficBackend",
    "TrafficBackendError",
    "TrafficBackendFactory",
    "TrafficBackendUnavailable",
    "TrafficConfig",
    "TrafficContext",
    "TrafficManagerBackend",
    "TrafficManagerBackendConfig",
    "TurnDirection",
    "build_traffic_manager",
    "build_none",
    "register_builtin_backends",
    "TurningAtJunctions",
    "TRAFFIC_BACKEND_ENTRY_POINT_GROUP",
    "available_backends",
    "build_backend",
    "get_backend_factory",
    "load_traffic_backend_plugins",
    "register_backend",
    "unregister_backend",
]


def build_traffic_manager(options: Mapping[str, Any]) -> TrafficBackend:
    """Build the TrafficManager backend from its options mapping."""
    return TrafficManagerBackend(TrafficManagerBackendConfig.from_mapping(options))


def build_none(options: Mapping[str, Any]) -> TrafficBackend:
    """Build the no-traffic backend, which takes no options."""
    if options:
        raise ValueError(
            f"The 'none' traffic backend takes no options; got {sorted(options)}."
        )
    return NullTrafficBackend()


def register_builtin_backends() -> None:
    """Register the backends this package ships.

    Called at import.  Public because a test that overrides a built-in needs a
    way to put it back, and reaching into the module for a private factory is a
    worse way to say that.
    """
    register_backend("traffic_manager", build_traffic_manager)
    register_backend("none", build_none)


register_builtin_backends()
