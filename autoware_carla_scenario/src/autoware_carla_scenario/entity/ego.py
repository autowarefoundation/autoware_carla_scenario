"""Ego vehicle spawning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, FrozenSet, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose
    from ..scenario_base import EgoConfig

from ..constants import EGO_ROLE_NAME
from ..debug_markers import DebugMarkers
from ._spawn import spawn_vehicle_actor
from ..traffic.driven import BackendDriven

logger = logging.getLogger(__name__)

#: Metres between the points of the route a backend-driven ego is handed
#: (:meth:`EgoVehicle.route_to`).
ROUTE_POINT_STEP_M: float = 5.0


class EgoVehicle(BackendDriven):
    """Manages the ego vehicle actor.

    Beyond spawning and destroying the actor, this class defines the lifecycle hooks
    :class:`ScenarioRunner` calls on the ego: :meth:`on_scenario_start` after warm-up,
    :meth:`on_tick` on every simulation tick, and :meth:`on_scenario_end` during
    teardown.  They are no-ops here so that TrafficManager-driven egos cost nothing;
    subclasses that drive the vehicle themselves (see
    :class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`)
    override them.
    """

    #: When ``True`` (default), :class:`ScenarioRunner` enables
    #: TrafficManager autopilot on this actor after warm-up.
    use_autopilot: bool = True

    #: Whether this entity cannot start without a goal.  An ego that plans its
    #: own route -- see
    #: :class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`
    #: -- has nowhere to go without one and never reports ready, so a scenario
    #: that names none is refused.  One that is driven for it (TrafficManager,
    #: an external driver policy) reads no goal, and a scenario is free to leave
    #: it unsaid.
    requires_goal: bool = False

    #: When ``True``, the actor is not this entity's to create or destroy: it
    #: belongs to something outside the scenario (an ``autoware_carla_interface``
    #: node, say) and :meth:`spawn` only attaches to it.  :class:`ScenarioRunner`
    #: reads this to leave that actor out of the cleanup it does before a run.
    attaches_to_existing_actor: bool = False

    # Declared for the static check (docs/typecheck.md): Codon types an
    # attribute from its declaration; to Python it is a bare annotation.
    _vehicle: Optional["carla.Actor"]

    def __init__(self) -> None:
        self._vehicle: Optional["carla.Actor"] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def actor(self) -> Optional["carla.Actor"]:
        """Return the spawned CARLA actor, or ``None`` before :meth:`spawn`."""
        return self._vehicle

    @property
    def is_initialized(self) -> bool:
        """Whether the entity is ready for the scenario to be judged.

        An entity that needs a stack to come up first -- localization, routing
        and engagement, minutes of it -- returns ``False`` until it has, and
        :class:`ScenarioRunner` holds the scenario clock and its conditions
        until then.  An entity that is ready the moment its actor exists (the
        default) says so.
        """
        return True

    @property
    def carried_actor_ids(self) -> FrozenSet[int]:
        """The vehicles the entity itself moves while it is not yet initialized.

        :class:`ScenarioRunner` holds every vehicle on its brakes through the wait
        for :attr:`is_initialized`; an entity that carries vehicles somewhere in
        that time (a run-up onto the first frame) names them, and the hold
        spares them.
        """
        return frozenset()

    def set_initial_speeds(self, speeds_mps: Mapping[int, float]) -> None:
        """Be told the speed each vehicle starts the scenario at, by actor id.

        Called by :class:`ScenarioRunner` just before :meth:`on_scenario_start`.
        Ignored here; an entity that plays the run-up to the first frame reads it.
        """

    @property
    def termination_requested(self) -> bool:
        """Whether this entity has asked to end the scenario early.

        :class:`ScenarioRunner` checks this after the pass and fail conditions, so a
        condition that fires on the same tick still decides the outcome.
        """
        return False

    @property
    def termination_reason(self) -> Optional[str]:
        """Why this entity asked to end the scenario, when it says.

        Part of the result's message when :attr:`termination_requested` ends a
        run; ``None`` leaves the message generic.
        """
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_to(
        self,
        world: "carla.World",
        goal: "Lanelet2Pose",
        *,
        initial_pose: "Optional[CarlaWorldPose]" = None,
        ground_projection: "Optional[GroundProjectionConfig]" = None,
        waypoints: "Sequence[Lanelet2Pose]" = (),
    ) -> None:
        """Send this ego to *goal* through *waypoints*, by the lanelets between.

        The ego's traffic backend drives it, and a backend left to itself picks
        its own way at every junction.  So the way is worked out here: the
        Lanelet2 map's shortest path from where the ego is through each
        waypoint in turn to the goal
        (:func:`~autoware_carla_scenario.odd.route_plan.route_poses`), its
        centrelines a point every :data:`ROUTE_POINT_STEP_M` metres, handed to
        the backend as one path (:meth:`follow_route`).  A route that cannot be
        planned is logged, and the ego left to the backend.

        Routing lives on the entity rather than in the action that asks for it
        because *how* a destination is delivered is the entity's business: one
        stack takes a map-frame pose over a bridge, a traffic backend a path
        of points, an external control policy nothing at all (see
        :class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`
        and :class:`~autoware_carla_scenario.entity.externally_driven.ExternallyDrivenEgo`,
        which override this).  The action only says where and when.

        Args:
            world: The CARLA world.
            goal: Lanelet2 pose to route to.
            initial_pose: Where the ego starts, which the route starts from.
                ``None`` reads the attached actor.
            ground_projection: Not used: the route is laid on the lanelets'
                centrelines.
            waypoints: Lanelet2 poses the route passes through, in order.
        """
        path = self._route_path(goal, initial_pose, waypoints)
        if path:
            # Through a local of the type that declares it: Codon calls a
            # method a subclass overrides through a dispatch thunk, which does
            # not upcast (docs/typecheck.md).
            driven: BackendDriven = self
            driven.follow_route(world, path)

    def _route_path(
        self,
        goal: "Lanelet2Pose",
        initial_pose: "Optional[CarlaWorldPose]",
        waypoints: "Sequence[Lanelet2Pose]",
    ) -> List["carla.Location"]:
        """The CARLA locations :meth:`route_to` hands the backend, or none."""
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..coordinate.poses import CarlaWorldPose  # noqa: PLC0415
        from ..coordinate.transform import to_carla_location, to_lanelet2  # noqa: PLC0415
        from ..odd.route_plan import PlannedRoute, RouteError, route_poses  # noqa: PLC0415

        located: List["carla.Location"] = []
        start_pose = initial_pose
        if start_pose is None:
            actor = self._vehicle
            if actor is None:
                logger.warning("Ego: a route was asked for before the actor exists")
                return located
            # A typed local: Codon unwraps the Optional here.
            vehicle: "carla.Actor" = actor
            transform = vehicle.get_transform()
            start_pose = CarlaWorldPose(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z,
                yaw=transform.rotation.yaw,
            )
        start: CarlaWorldPose = start_pose
        mm = MapManager.get_instance()
        try:
            poses = route_poses(
                PlannedRoute(start=to_lanelet2(start), goal=goal, via=tuple(waypoints)),
                mm.lanelet_map,
                mm.routing_graph,
                ROUTE_POINT_STEP_M,
            )
        except RouteError as error:
            logger.warning("Ego: no route to follow: %s", error)
            return located
        for pose in poses:
            located.append(to_carla_location(pose))
        return located

    def spawn(self, world: "carla.World", config: EgoConfig) -> "carla.Actor":
        """Spawn the ego vehicle.

        Args:
            world: The CARLA world instance.
            config: Ego vehicle spawn configuration.

        Returns:
            The spawned vehicle actor.

        Raises:
            ValueError: If the vehicle blueprint is not found or spawn index
                is out of range.
            RuntimeError: If the vehicle could not be spawned at the
                requested location.
        """
        self._vehicle = spawn_vehicle_actor(
            world,
            config.vehicle_type,
            str(EGO_ROLE_NAME),
            config.spawn_location,
            od_pose=config.od_pose,
            spawn_retry_max_count=config.spawn_retry_max_count,
            spawn_retry_t_step=config.spawn_retry_t_step,
            spawn_retry_z_step=config.spawn_retry_z_step,
            ground_projection=config.ground_projection,
        )
        return self._vehicle

    # ------------------------------------------------------------------
    # Lifecycle hooks (no-ops by default)
    # ------------------------------------------------------------------

    def on_scenario_start(self, world: "carla.World") -> None:
        """Called once after the warm-up ticks and initial speeds are applied.

        Args:
            world: The CARLA world instance.
        """

    def on_tick(self, world: "carla.World", elapsed: float) -> None:
        """Called on every simulation tick, right after ``world.tick()``.

        Args:
            world: The CARLA world instance.
            elapsed: Simulated seconds since the run began.
        """

    def debug_markers(self) -> DebugMarkers:
        """What this entity shows on the current tick: the plan it follows.

        :class:`ScenarioRunner` asks after every tick of the scenario and
        writes the answer into the run's trajectory log when it changes (a
        ``markers`` record, docs/trajectory_log.md), where a viewer of the run
        draws it (:mod:`autoware_carla_scenario.debug_markers`).  Nothing here;
        an entity that follows a plan of its own shows it.
        """
        return DebugMarkers()

    def on_scenario_end(self, world: "carla.World") -> None:
        """Called during teardown, before the actor is destroyed.

        Args:
            world: The CARLA world instance.
        """

    def destroy(self) -> None:
        """Destroy the vehicle actor."""
        if self._vehicle is not None:
            self._vehicle.destroy()
            self._vehicle = None
