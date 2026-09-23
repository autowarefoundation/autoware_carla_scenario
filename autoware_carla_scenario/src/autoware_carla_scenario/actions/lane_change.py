"""Lane-change action: force a lane change via TrafficManager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

from typing import Optional as _Optional

from ..conditions import BaseCondition, LaneChangeSettledCondition
from ..entity.registry import find_entity_by_role_name
from ..traffic import LaneChangeDirection
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
    watching the vehicle afterwards and stays
    :attr:`~autoware_carla_scenario.actions.base.ActionState.RUNNING` until the
    vehicle has actually settled onto the next lane.  That is the signal another
    actor can react to; ``done`` would fire while the car is still straddling
    the line.

    The waiting is a :class:`LaneChangeSettledCondition` supplied as the run's
    *until*, so the end of the manoeuvre is written in the same vocabulary as
    the trigger that started it.  The command itself is **not** reissued: one
    ``force_lane_change`` queues the manoeuvre, and sending it again every tick
    would restart it.

    Args:
        entity_name: ``role_name`` of the vehicle actor to control.
        direction: :class:`LaneChangeDirection` — ``LEFT`` or ``RIGHT``.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        once: If ``True`` (default) the action fires at most once.
        until: Overrides what counts as the manoeuvre having settled.  Defaults
            to :class:`LaneChangeSettledCondition` on *entity_name*.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        direction: LaneChangeDirection,
        condition: _Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "lane_change",
        once: bool = True,
        until: _Optional[BaseCondition] = None,
    ) -> None:
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            until=(
                until
                if until is not None
                else LaneChangeSettledCondition(entity_name, label=f"{label}_settled")
            ),
        )
        self._entity_name = entity_name
        self._direction = direction

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
            return

        entity.change_lane(world, self._direction)
