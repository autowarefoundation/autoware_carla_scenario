"""Lane-change action: force a lane change via TrafficManager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

from typing import Optional as _Optional

from ..conditions import BaseCondition
from ..entity.registry import find_entity_by_role_name
from ..entity.tm_driving import LaneChangeDirection, LaneChanging
from ..constants import (
    DEFAULT_TM_PORT,
)
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)


class LaneChangeAction(BaseAction):
    """Force a lane change via TrafficManager.

    When the associated condition is satisfied, this action:

    1. Locates the target vehicle by its ``role_name``
    2. Calls ``TrafficManager.force_lane_change(actor, direction)`` to
       command an immediate lane change

    ``force_lane_change`` only queues the manoeuvre, so the action keeps
    watching the vehicle afterwards (see :meth:`is_finished`) and stays
    :attr:`~autoware_carla_scenario.actions.base.ActionState.RUNNING` until the
    vehicle has actually settled onto the next lane.  That is the signal another
    actor can react to; ``done`` would fire while the car is still straddling
    the line.

    Args:
        entity_name: ``role_name`` of the vehicle actor to control.
        direction: :class:`LaneChangeDirection` — ``LEFT`` or ``RIGHT``.
        client: A ``carla.Client`` used to obtain the TrafficManager.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        once: If ``True`` (default) the action fires at most once.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        direction: LaneChangeDirection,
        client: "carla.Client",
        condition: _Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "lane_change",
        once: bool = True,
        tm_port: int = DEFAULT_TM_PORT,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = entity_name
        self._direction = direction
        # Kept for backwards compatibility: the TrafficManager is now reached
        # by the entity, which the runner gives a client of its own.  Passing
        # them here is harmless and keeps every existing call site working.
        self._client = client
        self._tm_port = tm_port
        #: The entity resolved in :meth:`execute`, asked each tick whether the
        #: manoeuvre has settled.  Looked up once rather than per tick: the
        #: registry is cheap, but the answer cannot change mid-manoeuvre.
        self._entity: _Optional[LaneChanging] = None

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def execute(self, world: "carla.World") -> None:
        """Ask the named entity to change lane."""
        entity = find_entity_by_role_name(self._entity_name)
        if entity is None:
            logger.warning(
                "LaneChangeAction: entity '%s' not found", str(self._entity_name)
            )
            self._entity = None
            return

        self._entity = entity
        entity.change_lane(world, self._direction)

    def is_finished(self, world: "carla.World", running_for: float) -> bool:
        """Whether the entity reports the manoeuvre settled.

        Asked of the entity rather than measured here: what "finished" means
        depends on what performed the change.  A TrafficManager-driven vehicle
        has settled once it is centred on the lane it was sent to; a stack that
        plans its own manoeuvres would answer from its own state.

        A manoeuvre that never happens simply never finishes, and the action
        stays :attr:`~autoware_carla_scenario.action_state.ActionState.RUNNING`.
        That is OpenSCENARIO's behaviour, and it keeps ``completeState`` from
        being reached by a lane change that did not happen; ending the run on a
        timer is the scenario timeout's job, not this action's.

        Args:
            world: The CARLA world instance.
            running_for: Seconds since the command was issued.
        """
        if self._entity is None:
            return False
        finished = self._entity.lane_change_finished(world)
        if finished:
            logger.info(
                "LaneChangeAction: '%s' settled onto its new lane after %.1fs",
                self._entity_name,
                running_for,
            )
        return finished
