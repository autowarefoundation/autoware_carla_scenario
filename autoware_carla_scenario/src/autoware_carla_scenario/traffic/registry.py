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
from typing import Any, Callable, Mapping

from .base import TrafficBackend

logger = logging.getLogger(__name__)

__all__ = [
    "TrafficBackendFactory",
    "TRAFFIC_BACKEND_ENTRY_POINT_GROUP",
    "register_backend",
    "unregister_backend",
    "available_backends",
    "get_backend_factory",
    "build_backend",
    "load_traffic_backend_plugins",
]

#: Entry-point group third-party packages advertise a backend through.
TRAFFIC_BACKEND_ENTRY_POINT_GROUP = "autoware_carla_scenario.traffic_backends"

#: Builds a backend from its own options node.
TrafficBackendFactory = Callable[[Mapping[str, Any]], TrafficBackend]

_BACKENDS: dict[str, TrafficBackendFactory] = {}


def register_backend(name: str, factory: TrafficBackendFactory) -> None:
    """Make *factory* selectable as ``traffic.backend=<name>``.

    Re-registering a name replaces it, which is what a package overriding a
    built-in backend means to do.

    Args:
        name: The name the config selects this backend by.
        factory: Callable taking the backend's options mapping and returning a
            :class:`~autoware_carla_scenario.traffic.base.TrafficBackend`.
    """
    _BACKENDS[name] = factory


def unregister_backend(name: str) -> None:
    """Forget the backend registered under *name*, if any.

    Exists for tests, which register a fake backend and must not leak it into
    the next test.
    """
    _BACKENDS.pop(name, None)


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
    name: str, options: Mapping[str, Any] | None = None
) -> TrafficBackend:
    """Build the backend *name* from its *options*.

    Args:
        name: Registered backend name.
        options: The backend's own options node; ``None`` means its defaults.

    Raises:
        ValueError: If no backend answers to *name*.
    """
    return get_backend_factory(name)(dict(options or {}))


def load_traffic_backend_plugins() -> None:
    """Import every package advertising a backend entry point.

    Each entry point resolves to a zero-argument callable that registers the
    package's backends.  A plugin that fails to import is logged and skipped:
    one broken third-party package must not stop a run that does not use it.
    """
    from importlib.metadata import entry_points  # noqa: PLC0415

    for entry_point in entry_points(group=TRAFFIC_BACKEND_ENTRY_POINT_GROUP):
        try:
            entry_point.load()()
        except Exception:
            logger.warning(
                "Traffic backend plugin %r failed to load; it is skipped",
                entry_point.name,
                exc_info=True,
            )
        else:
            logger.info("Loaded traffic backend plugin: %s", entry_point.name)
