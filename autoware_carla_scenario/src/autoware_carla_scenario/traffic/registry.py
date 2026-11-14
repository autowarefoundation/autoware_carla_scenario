"""Traffic backends, addressable by name.

The counterpart of :mod:`autoware_carla_scenario.registry` for traffic: that one
lets a scenario live outside this package, this one lets a *traffic model* live
outside it.  A package that implements a backend registers it at import time and
advertises the entry point::

    [project.entry-points."autoware_carla_scenario.traffic_backends"]
    my_simulator = "my_package:register_traffic"

and ``traffic.backend=my_simulator`` then selects it exactly as a built-in name
does.

The module is deliberately free of heavy imports: the built-in backends are
registered through factories that import their own module lazily, so naming a
backend costs nothing until one is built.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

from .base import TrafficBackend
from .parameters import BicycleParameters, PedestrianParameters, VehicleParameters

logger = logging.getLogger(__name__)

__all__ = [
    "TrafficBackendFactory",
    "TRAFFIC_BACKEND_ENTRY_POINT_GROUP",
    "register_backend",
    "unregister_backend",
    "available_backends",
    "get_backend_factory",
    "build_backend",
    "tm_port_option",
    "load_traffic_backend_plugins",
]

#: Entry-point group third-party packages advertise a backend through.
TRAFFIC_BACKEND_ENTRY_POINT_GROUP = "autoware_carla_scenario.traffic_backends"

#: Builds a backend from its own options node.
TrafficBackendFactory = Callable[[Mapping[str, Any]], TrafficBackend]

_BACKENDS: dict[str, TrafficBackendFactory] = {}

#: Name -> the option of a backend that drives through a TrafficManager its
#: port is read from (:func:`tm_port_option`).
_TM_PORT_OPTIONS: dict[str, str] = {}

#: Whether the entry-point walk has already run in this process.  Set only
#: once the walk is over (#2037).
_plugins_loaded = False

#: Whether the walk is running right now, on the thread holding the plugin
#: lock (a plugin's import calling load_traffic_backend_plugins again).
_plugins_loading = False


def register_backend(
    name: str,
    factory: TrafficBackendFactory,
    *,
    tm_port_option: Optional[str] = None,
) -> None:
    """Make *factory* selectable as ``traffic.backend=<name>``.

    Re-registering a name replaces it, which is what a package overriding a
    built-in backend means to do; :func:`load_traffic_backend_plugins` still
    logs a warning naming the plugin that did.

    Args:
        name: The name the config selects this backend by.
        factory: Callable taking the backend's options mapping and returning a
            :class:`~autoware_carla_scenario.traffic.base.TrafficBackend`.
        tm_port_option: For a backend that drives through a CARLA
            TrafficManager, the option of its own (``traffic.options.<it>``)
            the TrafficManager's port is read from: where a run reports the
            port and a SLURM task sets its own.  ``None``: none.
    """
    _BACKENDS[name] = factory
    if tm_port_option is None:
        _TM_PORT_OPTIONS.pop(name, None)
    else:
        _TM_PORT_OPTIONS[name] = tm_port_option


def unregister_backend(name: str) -> None:
    """Forget the backend registered under *name*, if any.

    Exists for tests, which register a fake backend and must not leak it into
    the next test.
    """
    _BACKENDS.pop(name, None)
    _TM_PORT_OPTIONS.pop(name, None)


def tm_port_option(name: str) -> Optional[str]:
    """The option the backend *name* reads its TrafficManager's port from
    (``port``, ``tm_port``), or ``None`` for one without a TrafficManager or
    unknown; backends of installed packages included."""
    load_traffic_backend_plugins()
    return _TM_PORT_OPTIONS.get(name)


def available_backends() -> list[str]:
    """Return every registered backend name, sorted."""
    return sorted(_BACKENDS)


def get_backend_factory(name: str) -> TrafficBackendFactory:
    """Return the factory registered under *name*.

    Raises:
        ValueError: If no backend answers to *name*.  The message lists what is
            registered, because the usual cause is a typo and the usual next
            question is "what could I have meant?".
    """
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown traffic backend: {name!r}. "
            f"Registered backends: {available_backends()}. "
            "A backend from another package has to be installed and advertised "
            f"through the {TRAFFIC_BACKEND_ENTRY_POINT_GROUP!r} entry-point group."
        ) from None


def build_backend(
    name: str,
    options: Mapping[str, Any] | None = None,
    vehicles: VehicleParameters | None = None,
    pedestrians: PedestrianParameters | None = None,
    bicycles: BicycleParameters | None = None,
) -> TrafficBackend:
    """Build the backend *name* from its *options*, with its road users'
    parameters.

    Args:
        name: Registered backend name.
        options: The backend's own options node; ``None`` means its defaults.
        vehicles: How its vehicles drive
            (:class:`~autoware_carla_scenario.traffic.parameters.VehicleParameters`);
            ``None`` means the backend's defaults.
        pedestrians: How its pedestrians walk
            (:class:`~autoware_carla_scenario.traffic.parameters.PedestrianParameters`);
            ``None`` means the backend's defaults.
        bicycles: How its bicycles ride
            (:class:`~autoware_carla_scenario.traffic.parameters.BicycleParameters`);
            ``None`` means the backend's defaults.

    Raises:
        ValueError: If no backend answers to *name*.
        TrafficParameterError: If *vehicles*, *pedestrians* or *bicycles* sets a field the
            backend does not support; the message names the backends that do.
    """
    backend = get_backend_factory(name)(dict(options or {}))
    if vehicles is not None:
        backend.set_vehicle_parameters(vehicles)
    if pedestrians is not None:
        backend.set_pedestrian_parameters(pedestrians)
    if bicycles is not None:
        backend.set_bicycle_parameters(bicycles)
    return backend


def load_traffic_backend_plugins() -> None:
    """Import every package advertising a backend entry point.

    The entry points load sorted by distribution and entry-point name.  A
    plugin that overrides a backend registered before them (a built-in) is
    logged as a warning, and so are two plugins registering the same name, of
    which the one loaded last wins.

    Loading is idempotent: subsequent calls are no-ops within a single process,
    which matters because a Hydra ``--multirun`` sweep builds a backend once per
    job in one process, and re-running a third-party package's registration code
    once per job is unbounded work this package does not control.

    Loading is thread-safe (#2037), as
    :func:`~autoware_carla_scenario.registry.load_scenario_plugins` is, and
    under the same lock: concurrent first callers wait for the one that walks,
    a call a plugin makes while it is being loaded returns at once, and a walk
    that fails outright is not retried.
    """
    global _plugins_loaded, _plugins_loading
    if _plugins_loaded:
        return

    from ..registry import _PLUGIN_LOAD_LOCK  # noqa: PLC0415

    with _PLUGIN_LOAD_LOCK:
        if _plugins_loaded or _plugins_loading:
            return
        _plugins_loading = True
        try:
            _walk_traffic_backend_plugins()
        finally:
            _plugins_loaded = True
            _plugins_loading = False


def _walk_traffic_backend_plugins() -> None:
    """Load every traffic backend plugin (under the plugin lock)."""
    from ..registry import load_entry_point_plugins  # noqa: PLC0415

    # A plugin that overrides a built-in backend means to (register_backend),
    # but it is still reported, as is one that takes another plugin's name:
    # the last one loaded wins, in the fixed order load_entry_point_plugins
    # walks the entry points in (#2034).
    load_entry_point_plugins(
        TRAFFIC_BACKEND_ENTRY_POINT_GROUP,
        "traffic backend",
        snapshot=lambda: dict(_BACKENDS),
        override_intended=True,
    )
