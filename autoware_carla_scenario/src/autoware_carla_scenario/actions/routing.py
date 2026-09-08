"""Tell an entity where to go and let it plan the route.

An entity driven by a full autonomy stack is not told how to steer; it is told
*where to go* and works the trajectory out itself.  This action says where and
when; *how* the destination is delivered belongs to the entity, so this calls
:meth:`~autoware_carla_scenario.entity.ego.EgoVehicle.route_to` and nothing
else.  One stack takes a map-frame pose over a bridge, another might take a
lane sequence or nothing at all, and none of that belongs in a timeline.

**When it runs is the caller's choice, with one hard constraint.**  The ego
needs a goal at least once during initialization: the runner waits for it to
report ready *before* the tick loop starts, and an Autoware ego only becomes
ready once it has localized, routed and engaged, so a first goal delivered from
inside the loop would never arrive.  That one is registered with
:meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_init`.

Nothing else is bound to initialization.  Re-routing part-way through a run, or
routing another entity once something has happened, is an ordinary tick-loop
action: register it with ``register_pre_tick`` / ``register_post_tick`` and
give it a condition like any other.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose
from ..entity.registry import find_entity_by_role_name
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

    from ..conditions import BaseCondition

logger = logging.getLogger(__name__)


class RoutingAction(BaseAction):
    """Give an entity a goal to plan a route to, and a pose to localize at.

    Names the entity rather than holding it, as every other action does: the
    lookup happens when the action runs, so an NPC registered during ``setup()``
    or an ego swapped in after the action was built is still found.  Where
    :class:`~autoware_carla_scenario.actions.lane_change.LaneChangeAction` and
    friends resolve the CARLA *actor* a role names, this resolves the *entity*
    -- a route is not something that can be applied to an actor.

    Only an entity that plans its own route has a mission to set; for any other
    :meth:`~autoware_carla_scenario.entity.ego.EgoVehicle.route_to` is a no-op,
    so a scenario can register this unconditionally and let the entity decide
    whether it means anything.

    Args:
        entity_name: Role name of the entity to route.
        goal: Lanelet2 pose the entity is routed to.
        condition: Condition gating the action.  Evaluated once during
            initialization for an init action, or each tick for one on the loop.
        timing: Which tick phase to run on, when registered on the loop.
        label: Action label.
        once: Whether the action fires at most once.  A re-routing action that
            should fire every time its condition holds passes ``False``.
        initial_pose: CARLA world pose the entity is expected to start at, used
            to initialize localization.  ``None`` leaves it to the entity, which
            reads the attached actor instead.  Only meaningful for the first
            routing; a re-route leaves localization alone.
        ground_projection: Settings used to snap the goal to the road surface.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        goal: Lanelet2Pose,
        condition: Optional["BaseCondition"] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "routing",
        once: bool = True,
        initial_pose: Optional[CarlaWorldPose] = None,
        ground_projection: Optional[GroundProjectionConfig] = None,
    ) -> None:
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
        )
        self._entity_name = entity_name
        self._goal = goal
        self._initial_pose = initial_pose
        self._ground_projection = ground_projection

    @property
    def goal(self) -> Lanelet2Pose:
        """The Lanelet2 pose the entity is routed to."""
        return self._goal

    @property
    def entity_name(self) -> Union[EntityRole, str]:
        """Role name of the entity this routes."""
        return self._entity_name

    def execute(self, world: "carla.World") -> None:
        """Hand the goal to the named entity."""
        entity = find_entity_by_role_name(self._entity_name)
        if entity is None:
            logger.warning(
                "RoutingAction: entity '%s' not found", str(self._entity_name)
            )
            return

        entity.route_to(
            world,
            self._goal,
            initial_pose=self._initial_pose,
            ground_projection=self._ground_projection,
        )
