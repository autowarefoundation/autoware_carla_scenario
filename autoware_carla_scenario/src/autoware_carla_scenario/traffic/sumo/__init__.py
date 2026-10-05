"""SUMO as a traffic backend (``traffic.backend=sumo``).

See :mod:`.backend`.  Importing this package does not import SUMO: the
simulator is loaded when a run is prepared, so a missing ``sumo`` extra is
reported there, before anything is spawned.
"""

from __future__ import annotations

from .config import SumoBackendConfig

__all__ = ["SumoBackendConfig", "SumoTrafficBackend"]


def __getattr__(name: str) -> object:
    if name == "SumoTrafficBackend":
        from .backend import SumoTrafficBackend  # noqa: PLC0415

        return SumoTrafficBackend
    raise AttributeError(name)
