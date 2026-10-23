"""Tell an entity where to go and let it plan the route.

An entity driven by a full autonomy stack is not told how to steer; it is told
*where to go* and works the trajectory out itself.  This action says where and
when; *how* the destination is delivered belongs to the entity, so this calls
:meth:`~autoware_carla_scenario.traffic.driven.BackendDriven.route_to` and
nothing else.  One stack takes a map-frame pose over a bridge, a traffic
backend a path to follow, an external policy nothing at all, and none of that
belongs in a timeline.

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

from ..action_state import ActionState
from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose
from ..entity.pedestrian_entity import PedestrianEntity, plan_walk
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
from ..entity_role import EntityRole
from ..traffic.driven import BackendDriven
from ..trajectory.model import Trajectory, TrajectoryFollowingMode, TrajectoryVertex
from .base import BaseAction, TickTiming
from .follow_trajectory import FollowTrajectoryAction, _TrajectoryEnd

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..conditions.base import BaseCondition

#: A pedestrian's walking speed along a route, in m/s, unless the action says.
DEFAULT_WALKING_SPEED_MPS: float = 1.4

logger = logging.getLogger(__name__)


class RoutingAction(BaseAction):
    """Give an entity a goal to plan a route to, and a pose to localize at.

    Names the entity rather than holding it, as every other action does: the
    lookup happens when the action runs, so an NPC registered during ``setup()``
    or an ego swapped in after the action was built is still found.  Where
    :class:`~autoware_carla_scenario.actions.lane_change.LaneChangeAction` and
    friends resolve the CARLA *actor* a role names, this resolves the *entity*
    -- a route is not something that can be applied to an actor.

    An entity that plans its own route is given a mission; a vehicle a traffic
    backend drives, the ego or an NPC, is handed the route as a path
    (:meth:`~autoware_carla_scenario.traffic.driven.BackendDriven.route_to`);
    for an ego an external policy drives it is a no-op.  So a scenario can
    register this unconditionally and let the entity decide what it means.

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
        walking_speed: How fast a pedestrian walks the route, in m/s.  A
            vehicle goes at the speed whatever drives it chooses.

    **A pedestrian** is walked by the action itself, along the way
    :func:`~autoware_carla_scenario.entity.pedestrian_entity.plan_walk` lays
    out, with the walking :class:`~autoware_carla_scenario.actions.FollowTrajectoryAction`
    does in ``FOLLOW`` mode: so on a pedestrian the action runs on the tick
    loop until the walker has arrived, and has to be registered there.
    """

    _entity_name: Union[EntityRole, str]
    _goal: Lanelet2Pose
    _initial_pose: Optional[CarlaWorldPose]
    _ground_projection: Optional[GroundProjectionConfig]
    _waypoints: tuple[Lanelet2Pose, ...]
    _walking_speed: float
    _walk: Optional[FollowTrajectoryAction]
    _elapsed: float

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
        walking_speed: float = DEFAULT_WALKING_SPEED_MPS,
    ) -> None:
        if not walking_speed > 0.0:
            raise ValueError("walking_speed must be positive")
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
        self._walking_speed = float(walking_speed)
        self._walk = None
        self._elapsed = 0.0

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

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Run the action's tick, keeping *elapsed* for a walk it steps."""
        self._elapsed = elapsed
        super().tick(world, elapsed)

    def _reissues_by_default(self) -> bool:
        """A walk is stepped every tick; a vehicle's route is handed over once."""
        return self._walk is not None

    def execute(self, world: "carla.World") -> None:
        """Hand the goal to the named entity, or walk the named pedestrian."""
        if self.state is ActionState.STANDBY:
            # A new run: a walk is planned again, from where the walker is now.
            self._walk = None
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
        self._step_walk(world, walker)

    def _step_walk(self, world: "carla.World", walker: PedestrianEntity) -> None:
        """Walk *walker* one tick along its route, planning it on the first."""
        if self._walk is None:
            points = plan_walk(walker, self._goal, self._waypoints)
            if len(points) < 2:
                logger.warning(
                    "RoutingAction: no walk for '%s' to follow", str(self._entity_name)
                )
                return
            vertices: list[TrajectoryVertex] = []
            for point in points:
                vertices.append(TrajectoryVertex(point))
            walk = FollowTrajectoryAction(
                self._entity_name,
                Trajectory(f"{self.label} walk", vertices),
                following_mode=TrajectoryFollowingMode.FOLLOW,
                label=f"{self.label}_walk",
                speed=self._walking_speed,
            )
            self._walk = walk
            # The run ends when the walk does.
            ending: BaseCondition = _TrajectoryEnd(walk, self.label)
            self._until = ending
        stepping: FollowTrajectoryAction = self._walk
        stepping.tick(world, self._elapsed)

    def _route(self, world: "carla.World", entity: BackendDriven) -> None:
        # Waypoints only when there are some: an entity that overrides
        # route_to() with the signature it had before waypoints existed keeps
        # working for every run that names none.  Two calls rather than
        # `**extra`: Codon unpacks no dict into keyword arguments.  And each
        # through the class in the check: Codon calls a method a subclass
        # overrides (ExternallyDrivenEgo's, BridgedEgoEntity's) through a
        # dispatch thunk, which takes no keyword argument (docs/typecheck.md);
        # the check calls BackendDriven's own, and each override through its
        # class.
        if self._waypoints:
            if TYPE_CHECKING:
                BackendDriven.route_to(
                    entity,
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
                    waypoints=self._waypoints,
                )
        elif TYPE_CHECKING:
            BackendDriven.route_to(
                entity,
                world,
                self._goal,
                initial_pose=self._initial_pose,
                ground_projection=self._ground_projection,
            )
        else:
            entity.route_to(
                world,
                self._goal,
                initial_pose=self._initial_pose,
                ground_projection=self._ground_projection,
            )
