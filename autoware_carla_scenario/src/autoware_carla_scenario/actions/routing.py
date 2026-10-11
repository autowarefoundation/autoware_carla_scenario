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
from typing import TYPE_CHECKING, Optional, Sequence, Union

from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose
from ..entity.pedestrian_entity import PedestrianEntity
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
from ..entity_role import EntityRole
from ..traffic.driven import BackendDriven
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..conditions.base import BaseCondition

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
        waypoints: Lanelet2 poses the route must pass through, in order.  Empty
            leaves the way to the goal to the entity's own planner.
    """

    _entity_name: Union[EntityRole, str]
    _goal: Lanelet2Pose
    _initial_pose: Optional[CarlaWorldPose]
    _ground_projection: Optional[GroundProjectionConfig]
    _waypoints: tuple[Lanelet2Pose, ...]

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
        waypoints: Sequence[Lanelet2Pose] = (),
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
        self._waypoints = tuple(waypoints)

    @property
    def waypoints(self) -> tuple[Lanelet2Pose, ...]:
        """The Lanelet2 poses the route must pass through, in order."""
        return self._waypoints

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
        # The entity of any kind the role names, as find_entity_by_role_name
        # finds it: a vehicle first, then a pedestrian.  Looked up by kind, so
        # the static check knows its type (docs/typecheck.md).
        vehicle = find_vehicle_entity(self._entity_name)
        if vehicle is not None:
            entity: BackendDriven = vehicle
            self._route(world, entity)
            return
        pedestrian = find_pedestrian_entity(self._entity_name)
        if pedestrian is None:
            logger.warning(
                "RoutingAction: entity '%s' not found", str(self._entity_name)
            )
            return
        walker: PedestrianEntity = pedestrian
        self._route(world, walker)

    def _route(
        self, world: "carla.World", entity: Union[BackendDriven, PedestrianEntity]
    ) -> None:
        if hasattr(entity, "route_to"):
            # Waypoints only when there are some: an entity that overrides
            # route_to() with the signature it had before waypoints existed
            # keeps working for every run that names none.  Two calls rather
            # than `**extra`: Codon unpacks no dict into keyword arguments.
            if self._waypoints:
                entity.route_to(
                    world,
                    self._goal,
                    initial_pose=self._initial_pose,
                    ground_projection=self._ground_projection,
                    waypoints=self._waypoints,
                )
            else:
                entity.route_to(
                    world,
                    self._goal,
                    initial_pose=self._initial_pose,
                    ground_projection=self._ground_projection,
                )
            return
        # An entity without route_to() (an NPC vehicle, a pedestrian): the call
        # raises the AttributeError it always did.  Codon, which reads
        # TYPE_CHECKING as true, decides hasattr() by the static type, which
        # never has route_to (only EgoVehicle declares it), and compiles none
        # of the calls.
        if TYPE_CHECKING:
            raise AttributeError("route_to")
        else:
            entity.route_to(
                world,
                self._goal,
                initial_pose=self._initial_pose,
                ground_projection=self._ground_projection,
            )
