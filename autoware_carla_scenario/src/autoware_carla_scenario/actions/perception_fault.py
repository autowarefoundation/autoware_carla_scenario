"""Perception fault injection: report one road user to the ego's driver as
something it is not."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..conditions.base import BaseCondition, find_actor_in_list
from ..constants import EGO_ROLE_NAME
from ..entity.externally_driven import ExternallyDrivenEgo
from ..entity.registry import find_vehicle_entity
from ..entity_role import EntityRole
from ..perception_fault import (
    PerceivedCategory,
    PerceptionFault,
    set_perception_fault,
)
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["PerceptionFaultInjectionAction"]


class PerceptionFaultInjectionAction(BaseAction):
    """Report one road user to the ego's driver policy as something it is not.

    From the tick it fires on, the road user *entity_name* is reported with
    the category, the box size or the visibility given here, in place of its
    own; whatever is left at its default stays as it is.  Only that road user:
    every other one is reported as it is.  It stays so until another of these
    actions sets that road user again (one with every argument at its default
    reports it as it is) or the run ends.

    What changes is what the policy is told -- the ``renderer_data`` actors
    and the live T4Scene's boxes (``ego.entity=carla_driver``).  The world is
    not touched, nor what the run records: the trajectory log, the T4
    dataset, the metrics and the conditions keep CARLA's ground truth, so a
    collision with the road user the policy was told is not there is still a
    collision.

    An ego driven by a stack that perceives the road users itself
    (``ego.entity=autoware``, ROS 2, HILS) is not told of them by the
    framework, so there is nothing to alter: the action raises
    :class:`NotImplementedError` as it fires.

    Args:
        entity_name: ``role_name`` of the road user to misreport (not the ego).
        category: The category to report it as; ``ACTUAL`` keeps its own.
        length_m: The box's length to report, m; 0 keeps its own.
        width_m: The box's width to report, m; 0 keeps its own.
        height_m: The box's height to report, m; 0 keeps its own.
        visible: ``False`` leaves it out of what the policy is told.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.

    Raises:
        ValueError: If a size is negative or not finite, or *entity_name*
            is the ego.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _fault: PerceptionFault

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        category: PerceivedCategory = PerceivedCategory.ACTUAL,
        length_m: float = 0.0,
        width_m: float = 0.0,
        height_m: float = 0.0,
        visible: bool = True,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "perception_fault",
        once: bool = True,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = str(entity_name)
        if self._entity_name == str(EGO_ROLE_NAME):
            raise ValueError(
                "PerceptionFaultInjectionAction: the ego is not reported to its "
                "own driver; name another road user"
            )
        self._fault = PerceptionFault(
            category=category,
            length_m=float(length_m),
            width_m=float(width_m),
            height_m=float(height_m),
            visible=bool(visible),
        )

    def execute(self, world: "carla.World") -> None:
        """Report the named road user as the fault says from now on.

        Raises:
            NotImplementedError: If the ego perceives the road users itself.
        """
        ego = find_vehicle_entity(EGO_ROLE_NAME)
        if isinstance(ego, ExternallyDrivenEgo) and not ego.road_users_from_framework:
            raise NotImplementedError(
                "PerceptionFaultInjectionAction is not implemented yet for an ego "
                f"driven by {ego.stack_name}: that stack perceives the road users "
                "itself, so the framework has no report of them to alter"
            )
        actor = find_actor_in_list(world.get_actors(), self._entity_name)
        if actor is None:
            logger.warning(
                "PerceptionFaultInjectionAction: entity '%s' not found",
                self._entity_name,
            )
            return
        set_perception_fault(int(actor.id), self._fault)
        logger.info(
            "PerceptionFaultInjectionAction: '%s' reported as %s",
            self._entity_name,
            self._fault,
        )
