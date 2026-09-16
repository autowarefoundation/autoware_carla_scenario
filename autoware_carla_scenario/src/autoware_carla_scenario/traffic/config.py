"""Configuration for traffic backends.

One shared node says *which* backend drives a run; everything else is the
backend's own and travels as a plain mapping::

    traffic:
      backend: traffic_manager
      options:
        port: 8100

The split is what keeps a third-party backend from needing an edit here: a
backend ships its own options dataclass and reads ``options`` with its own
:meth:`from_mapping`, so the shared config never grows a field per simulator.

Like :mod:`autoware_carla_scenario.scenario_config`, this module has no heavy
dependencies -- no CARLA, no simulator client -- so the editor, the sweeper and
the config layer can import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..constants import DEFAULT_TM_PORT
from ..utils.config import checked_options

__all__ = [
    "TrafficConfig",
    "TrafficManagerBackendConfig",
]


@dataclass
class TrafficManagerBackendConfig:
    """Options of the CARLA TrafficManager backend."""

    #: Port the TrafficManager is reached on.  The CARLA default (8000) often
    #: collides with other services, which is why the framework's default is
    #: :data:`~autoware_carla_scenario.constants.DEFAULT_TM_PORT`.
    port: int = DEFAULT_TM_PORT

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrafficManagerBackendConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        return cls(**checked_options(cls, mapping))


@dataclass
class TrafficConfig:
    """Which backend drives a run's traffic, and with what options.

    Attributes:
        backend: The registered name of the backend.  ``traffic_manager`` is
            the default and is what every scenario written before this config
            existed gets.
        options: The backend's own options, passed to its factory verbatim.
    """

    backend: str = "traffic_manager"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrafficConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        checked = checked_options(cls, mapping)
        options = checked.get("options") or {}
        return cls(
            backend=str(checked.get("backend", "traffic_manager")),
            options=dict(options),
        )
