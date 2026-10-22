"""Change the traffic's parameters during a run.

The run's traffic parameters (``traffic.vehicles``, ``traffic.pedestrians``,
:mod:`autoware_carla_scenario.traffic.parameters`) are what its road users
draw their behaviour from when they join it.  :class:`TrafficParametersAction`
changes them from the tick it fires on: the fields it sets replace the run's,
the rest stay as they were, and every road user draws its value of the
changed fields again -- or, given an *entity_name*, only that vehicle does::

    # From 20 s on, drivers take their time and keep their distance.
    scenario.register_post_tick(
        TrafficParametersAction(
            vehicles=VehicleParameters(
                speed_factor=Spread(0.8, 0.05), min_gap_m=Spread(4.0, 1.0)
            ),
            condition=ElapsedTimeCondition(20.0, label="after_20s"),
        )
    )

    # The NPC ahead becomes a reckless driver.
    TrafficParametersAction(
        vehicles=VehicleParameters(speed_factor=1.3, ignore_lights=1.0),
        entity_name="npc1",
    )

A field the run's backend cannot change during a run is refused when the
action is registered on the scenario, before the run starts, naming the
backends that can (:class:`~autoware_carla_scenario.traffic.parameters.TrafficParameterError`):
the TrafficManager drives no pedestrians, and SUMO sets a pedestrian's
red-light rule on its type, before it starts.  The change ends with the run:
the next run of a queue starts from its own config.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..conditions.base import BaseCondition
from ..entity.registry import find_vehicle_entity
from ..traffic.base import DrivenEntity, TrafficBackend
from ..traffic.driven import BackendDriven
from ..traffic.parameters import (
    PEDESTRIAN_PARAMETER_BACKENDS,
    VEHICLE_PARAMETER_BACKENDS,
    PedestrianParameters,
    VehicleParameters,
    check_supported,
    record_run_parameters,
)
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["TrafficParametersAction"]


class TrafficParametersAction(BaseAction):
    """Override the run's traffic parameters, or one vehicle's, when it fires.

    Args:
        vehicles: The vehicle parameters to change; the fields it leaves unset
            keep their values.
        pedestrians: The pedestrian parameters to change, likewise.  Not with
            *entity_name*: the entity here is a vehicle.
        entity_name: ``role_name`` of the one vehicle to change; ``None``
            (default) changes the whole run's traffic.
        condition: Trigger condition (see :class:`BaseCondition`); fires on
            the first tick by default.
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.

    Raises:
        ValueError: If it changes nothing, or changes pedestrians of an
            entity.
    """

    _vehicles: Optional[VehicleParameters]
    _pedestrians: Optional[PedestrianParameters]
    _entity_name: Optional[str]
    _backend: Optional[TrafficBackend]

    def __init__(
        self,
        vehicles: Optional[VehicleParameters] = None,
        pedestrians: Optional[PedestrianParameters] = None,
        *,
        entity_name: Optional[str] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.POST_TICK,
        label: str = "TrafficParameters",
        once: bool = True,
    ) -> None:
        super().__init__(label, condition, timing, once=once)
        changes_vehicles = vehicles is not None and len(vehicles.set_fields()) > 0
        changes_pedestrians = (
            pedestrians is not None and len(pedestrians.set_fields()) > 0
        )
        if not changes_vehicles and not changes_pedestrians:
            raise ValueError(
                f"{type(self).__name__} '{label}': sets no vehicle or pedestrian "
                "parameter"
            )
        if entity_name is not None and changes_pedestrians:
            raise ValueError(
                f"{type(self).__name__} '{label}': an entity is a vehicle, so "
                "pedestrian parameters apply to the run's traffic only (leave "
                "entity_name unset)"
            )
        self._vehicles = vehicles if changes_vehicles else None
        self._pedestrians = pedestrians if changes_pedestrians else None
        self._entity_name = entity_name
        self._backend = None

    def set_traffic_backend(self, backend: TrafficBackend) -> None:
        """Take the run's traffic backend, and refuse a field it cannot change
        during a run; the scenario calls this on registration.

        Raises:
            TrafficParameterError: Naming each such field and the backends that
                can change it.
        """
        vehicles = self._vehicles
        if vehicles is not None:
            check_supported(
                "traffic.vehicles",
                backend.name,
                vehicles.set_fields(),
                backend.supported_vehicle_overrides(),
                VEHICLE_PARAMETER_BACKENDS,
                during_run=True,
                before_run=backend.supported_vehicle_parameters(),
            )
        pedestrians = self._pedestrians
        if pedestrians is not None:
            check_supported(
                "traffic.pedestrians",
                backend.name,
                pedestrians.set_fields(),
                backend.supported_pedestrian_overrides(),
                PEDESTRIAN_PARAMETER_BACKENDS,
                during_run=True,
                before_run=backend.supported_pedestrian_parameters(),
            )
        self._backend = backend

    def execute(self, world: "carla.World") -> None:
        """Hand the overrides to the backend."""
        if self._backend is None:
            logger.warning(
                "%s '%s': no traffic backend; register it on a scenario",
                type(self).__name__,
                self.label,
            )
            return
        backend: TrafficBackend = self._backend
        vehicles = self._vehicles
        if self._entity_name is not None:
            name: str = self._entity_name
            entity = find_vehicle_entity(name)
            if entity is None:
                logger.warning(
                    "%s '%s': vehicle '%s' not found",
                    type(self).__name__,
                    self.label,
                    name,
                )
                return
            if vehicles is not None:
                # A typed local: Codon unwraps the Optional here.
                one: VehicleParameters = vehicles
                # As the base class the backend takes: Codon passes a subclass
                # to a parameter of its base only through a local of it.
                found: BackendDriven = entity
                driven: DrivenEntity = found
                backend.override_entity_parameters(world, one, driven)
            logger.info("%s: '%s' draws its parameters anew", self.label, name)
            return
        if vehicles is not None:
            changed: VehicleParameters = vehicles
            backend.override_vehicle_parameters(world, changed)
        pedestrians = self._pedestrians
        if pedestrians is not None:
            walkers: PedestrianParameters = pedestrians
            backend.override_pedestrian_parameters(world, walkers)
        # What the ODD's probes read from now on (odd.traffic_probes).
        record_run_parameters(backend.vehicle_parameters, backend.pedestrian_parameters)
        logger.info("%s: the traffic draws its parameters anew", self.label)
