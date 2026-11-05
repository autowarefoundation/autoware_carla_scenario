"""Emergency action: put an emergency vehicle on a call, or take it off one."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..conditions.base import BaseCondition
from ..control_domain import ControlDomain
from ..entity.registry import find_vehicle_entity
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["EmergencyAction"]


class EmergencyAction(BaseAction):
    """Put an emergency vehicle on a call (*active*, the default) or off duty.

    Calls :meth:`~autoware_carla_scenario.entity.EmergencyVehicleEntity.set_emergency`
    on the :class:`~autoware_carla_scenario.entity.EmergencyVehicleEntity`
    named: on a call its siren lights (CARLA's ``VehicleLightState.Special1``)
    are on and the run's traffic backend drives it as one -- through red
    lights and stop signs, and, under SUMO, with SUMO's vehicles making way;
    off duty it is lit and driven as any other vehicle.  It stays so until set
    again.  A vehicle that is not an emergency vehicle is left as it is, with
    a warning.

    Args:
        entity_name: ``role_name`` of the emergency vehicle.
        active: ``True`` to put it on a call, ``False`` to take it off one.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        overrides: Take over a control domain of the entity another action
            holds (:attr:`~BaseAction.overrides`).
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _active: bool

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        active: bool = True,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "emergency",
        once: bool = True,
        overrides: bool = False,
    ) -> None:
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            overrides=overrides,
        )
        self._entity_name = str(entity_name)
        self._active = active

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The vehicle's siren lights: its lighting domain."""
        return [(str(self._entity_name), ControlDomain.LIGHTING)]

    def execute(self, world: "carla.World") -> None:
        """Put the named emergency vehicle on a call, or take it off one."""
        entity = find_vehicle_entity(self._entity_name)
        if entity is None:
            logger.warning("EmergencyAction: entity '%s' not found", self._entity_name)
            return
        entity.set_emergency(self._active)
        logger.info(
            "EmergencyAction: '%s' %s",
            self._entity_name,
            "on a call" if self._active else "off duty",
        )
