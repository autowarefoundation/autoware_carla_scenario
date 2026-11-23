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
from ..number_settings import check_port as check_port  # re-exported (#2398)
from .parameters import BicycleParameters, PedestrianParameters, VehicleParameters

__all__ = [
    "TrafficConfig",
    "TrafficManagerBackendConfig",
    "check_port",
    "traffic_backend_name",
]

#: The backend a ``traffic`` group without one (or with a null one) selects.
DEFAULT_TRAFFIC_BACKEND = "traffic_manager"


def traffic_backend_name(value: Any) -> str:
    """The backend name the ``traffic.backend`` setting *value* selects.

    An absent or null one (*value* ``None``) is ``traffic_manager``; a
    non-empty string is itself; anything else is refused, naming the key
    (#2156).  The one reader of the setting, so the backend built
    (:meth:`TrafficConfig.from_mapping`) and the TrafficManager port a
    SLURM task is given agree (#2160).

    Raises:
        ValueError: If *value* is neither ``None`` nor a non-empty string.
    """
    if value is None:
        return DEFAULT_TRAFFIC_BACKEND
    if not isinstance(value, str) or not value:
        raise ValueError(f"traffic.backend must be a backend's name, got {value!r}")
    return value


@dataclass
class TrafficManagerBackendConfig:
    """Options of the CARLA TrafficManager backend."""

    #: Port the TrafficManager is reached on.  The CARLA default (8000) often
    #: collides with other services, which is why the framework's default is
    #: :data:`~autoware_carla_scenario.constants.DEFAULT_TM_PORT`.
    port: int = DEFAULT_TM_PORT

    def __post_init__(self) -> None:
        check_port("traffic.options.port (traffic_manager.port)", self.port)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrafficManagerBackendConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node).

        A null ``port`` -- what the YAML groups' ``${oc.select:
        traffic_manager.port,null}`` gives a run with no ``traffic_manager``
        group -- is the default, as every null setting of the ``sumo`` and
        ``hybrid`` backends that does not take null is.
        """
        # Imported here: utils.config reads a dataclass's fields from its class,
        # which Codon cannot express, and the library check compiles no import
        # in a function it does not call (typecheck/library.py, UNCALLED).
        from ..utils.config import checked_options, null_as_default  # noqa: PLC0415

        return cls(**null_as_default(cls, checked_options(cls, mapping)))


@dataclass
class TrafficConfig:
    """Which backend drives a run's traffic, and with what options.

    Attributes:
        backend: The registered name of the backend.  ``traffic_manager`` is
            the default and is what every scenario written before this config
            existed gets.
        options: The backend's own options, passed to its factory verbatim
            -- but for the nulls, which a run drops unless the backend is
            registered with ``takes_null_options`` (#2178).
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
        than read as the defaults (#2143). ``backend`` is a non-empty
        string; an absent or null one is ``traffic_manager``, while any
        other value is refused, naming the key (#2156).
        """
        from ..utils.config import checked_options, checked_section  # noqa: PLC0415

        if "parameters" in mapping:
            raise ValueError(
                "traffic.parameters has been split: vehicle parameters are "
                "traffic.vehicles, pedestrian parameters traffic.pedestrians"
            )
        checked = checked_options(cls, mapping)

        def section(key: str) -> Mapping[str, Any]:
            value = checked_section(f"traffic.{key}", checked.get(key))
            return {} if value is None else value

        return cls(
            backend=traffic_backend_name(checked.get("backend")),
            options=dict(section("options")),
            vehicles=VehicleParameters.from_mapping(section("vehicles")),
            pedestrians=PedestrianParameters.from_mapping(section("pedestrians")),
            bicycles=BicycleParameters.from_mapping(section("bicycles")),
        )
