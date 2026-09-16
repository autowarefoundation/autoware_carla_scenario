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
    "TurningAtJunctions",
    "TRAFFIC_BACKEND_ENTRY_POINT_GROUP",
    "available_backends",
    "build_backend",
    "get_backend_factory",
    "load_traffic_backend_plugins",
    "register_backend",
    "unregister_backend",
]


def _build_traffic_manager(options: Mapping[str, Any]) -> TrafficBackend:
    """Build the TrafficManager backend, importing it only when one is asked for."""
    from .traffic_manager import TrafficManagerBackend  # noqa: PLC0415

    return TrafficManagerBackend(TrafficManagerBackendConfig.from_mapping(options))


def _build_none(options: Mapping[str, Any]) -> TrafficBackend:
    """Build the no-traffic backend, which takes no options."""
    if options:
        raise ValueError(
            f"The 'none' traffic backend takes no options; got {sorted(options)}."
        )
    return NullTrafficBackend()


register_backend("traffic_manager", _build_traffic_manager)
register_backend("none", _build_none)


def __getattr__(name: str) -> Any:
    """Expose ``TrafficManagerBackend`` without importing it at package import.

    Keeps ``from autoware_carla_scenario.traffic import TrafficConfig`` as cheap
    as the config layer needs it to be, while
    ``traffic.TrafficManagerBackend`` still resolves for the code that wants the
    class itself.
    """
    if name == "TrafficManagerBackend":
        from .traffic_manager import TrafficManagerBackend  # noqa: PLC0415

        return TrafficManagerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
