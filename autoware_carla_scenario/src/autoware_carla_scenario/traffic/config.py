"""Configuration for traffic backends.

One shared node says *which* backend drives a run and how its road users
behave; everything else is the backend's own and travels as a plain mapping::

    traffic:
      backend: traffic_manager
      options:
        port: 8100
      vehicles:
        speed_factor: 0.9
      pedestrians: {}
      bicycles: {}

The split is what keeps a third-party backend from needing an edit here: a
backend ships its own options dataclass and reads ``options`` with its own
:meth:`from_mapping`, so the shared config never grows a field per simulator.
``vehicles``, ``pedestrians`` and ``bicycles`` are the typed parts every
backend reads alike
(:class:`~autoware_carla_scenario.traffic.parameters.VehicleParameters`,
:class:`~autoware_carla_scenario.traffic.parameters.PedestrianParameters`,
:class:`~autoware_carla_scenario.traffic.parameters.BicycleParameters`), and
a backend refuses the fields it cannot honour.

Like :mod:`autoware_carla_scenario.scenario_config`, this module has no heavy
dependencies -- no CARLA, no simulator client -- so the editor, the sweeper and
the config layer can import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..constants import DEFAULT_TM_PORT
from .parameters import BicycleParameters, PedestrianParameters, VehicleParameters

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

    def __post_init__(self) -> None:
        # Refused here, where a run's config is checked, rather than by
        # client.get_trafficmanager() once CARLA is up: a bool is an int to
        # Python, and 0 or 70000 is no TCP port (the SUMO backend's tm_port
        # and hils.port have the same rule).
        port = self.port
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise ValueError(
                "traffic.options.port (traffic_manager.port) must be an integer "
                f"within 1-65535 (got {port!r})"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrafficManagerBackendConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        # Imported here: utils.config reads a dataclass's fields from its class,
        # which Codon cannot express, and the library check compiles no import
        # in a function it does not call (typecheck/library.py, UNCALLED).
        from ..utils.config import checked_options  # noqa: PLC0415

        return cls(**checked_options(cls, mapping))


@dataclass
class TrafficConfig:
    """Which backend drives a run's traffic, and with what options.

    Attributes:
        backend: The registered name of the backend.  ``traffic_manager`` is
            the default and is what every scenario written before this config
            existed gets.
        options: The backend's own options, passed to its factory verbatim.
        vehicles: How the vehicles drive, in the terms every backend shares;
            a field the backend does not support is an error.
        pedestrians: How the pedestrians walk, likewise.
        bicycles: How the bicycles ride, likewise.
    """

    backend: str = "traffic_manager"
    options: dict[str, Any] = field(default_factory=dict)
    vehicles: VehicleParameters = field(default_factory=VehicleParameters)
    pedestrians: PedestrianParameters = field(default_factory=PedestrianParameters)
    bicycles: BicycleParameters = field(default_factory=BicycleParameters)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrafficConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node).

        ``options``, ``vehicles``, ``pedestrians`` and ``bicycles`` are
        mappings; only an absent or null one is empty (the defaults), while
        ``false``, ``0``, ``""`` or a list is refused, naming the key, rather
        than read as the defaults (#2143).
        """
        from ..utils.config import checked_options  # noqa: PLC0415

        if "parameters" in mapping:
            raise ValueError(
                "traffic.parameters has been split: vehicle parameters are "
                "traffic.vehicles, pedestrian parameters traffic.pedestrians"
            )
        checked = checked_options(cls, mapping)

        def section(key: str) -> Mapping[str, Any]:
            value = checked.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ValueError(f"traffic.{key} must be a mapping, got {value!r}")
            return value

        return cls(
            backend=str(checked.get("backend", "traffic_manager")),
            options=dict(section("options")),
            vehicles=VehicleParameters.from_mapping(section("vehicles")),
            pedestrians=PedestrianParameters.from_mapping(section("pedestrians")),
            bicycles=BicycleParameters.from_mapping(section("bicycles")),
        )
