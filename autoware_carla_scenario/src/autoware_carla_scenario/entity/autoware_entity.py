"""Autoware ego vehicles.

Two entities live here:

* :class:`AutowareEntity` - a bare placeholder that spawns the ego, opts out of
  TrafficManager, and leaves the actor standing still for an external stack to
  control out of band.  Kept for backwards compatibility.

* :class:`AutowareEgoEntity` - the closed-loop entity.  Autoware (via the
  ``autoware_carla_interface`` ROS 2 node) reads CARLA sensors and applies
  control to the ego **directly**, so unlike
  :class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`
  the framework is *not* in the control loop.  This entity **spawns** the ego at
  the scenario's own spawn pose, hands Autoware the scenario's initial pose and
  goal, and waits for Autoware to become ready.  The whole startup sequence
  (localization init, routing, engage) is owned by the Autoware side; see
  :mod:`autoware_carla_scenario.autoware_bridge`.

  The scenario places the ego because the scenario is what knows where the run
  starts.  When the interface node placed it instead, the scenario's pose
  reached it afterwards, as an initial pose it moved the ego onto -- a teleport
  every other node could observe, and one that left the ego at the interface's
  own spawn point whenever the pose went missing.  The interface node must
  therefore be launched to attach to this ego rather than spawn its own.

Tick ownership: the scenario framework remains the tick master; the interface
node runs as a non-ticking, asynchronous I/O bridge (``sync_mode:=false``).

Role name: the framework identifies the ego by
:data:`~autoware_carla_scenario.constants.EGO_ROLE_NAME` (``"Ego"``).  Launch
the interface node with ``ego_vehicle_role_name:=Ego`` so it looks for the
actor this entity spawns under that name.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import carla

    from ..autoware_bridge.base import AutowareBridge, BridgePose
    from ..coordinate import GroundProjectionConfig, Lanelet2Pose
    from ..scenario_base import EgoConfig

from ..autoware_bridge.base import AutowareBridgeConfig
from ..constants import EGO_ROLE_NAME
from ..coordinate.poses import CarlaWorldPose
from ..coordinate.transform import to_map_frame
from ._spawn import spawn_vehicle_actor
from .ego import EgoVehicle

logger = logging.getLogger(__name__)

#: How far, horizontally, the ego may be from where the scenario expected it
#: before that disagreement is worth a warning.  Snapping a spawn onto the road
#: surface moves it by centimetres; a spawn_point that does not match moves it
#: by metres.
_INITIAL_POSE_TOLERANCE_M: float = 1.0


class AutowareEntity(EgoVehicle):
    """Ego vehicle controlled by Autoware instead of TrafficManager.

    After spawning, the :class:`ScenarioRunner` reads
    :attr:`EgoVehicle.use_autopilot` and skips ``set_autopilot(True)``
    for this actor, leaving it free for external (Autoware) control.

    The lifecycle hooks inherited from :class:`EgoVehicle` stay no-ops, so the
    vehicle stands still unless something outside the scenario drives it.  For a
    closed loop that waits for Autoware to become ready, use
    :class:`AutowareEgoEntity`.
    """

    use_autopilot: bool = False


class AutowareEgoEntity(EgoVehicle):
    """Ego vehicle spawned at the scenario's spawn pose and driven by Autoware.

    The :class:`ScenarioRunner` reads :attr:`EgoVehicle.use_autopilot` (``False``
    here) and skips ``set_autopilot(True)`` for this actor, leaving it under
    Autoware's control.

    Two things about a run are different because Autoware drives:

    * The actor is the scenario's, like any other entity's: this entity spawns
      it and destroys it.  Autoware drives it in between.
    * The scenario cannot be judged until Autoware is driving.  The runner holds
      the scenario clock and its conditions until :attr:`is_initialized`, so a
      condition that is already true near the initial pose -- standing still,
      say -- cannot record a result while Autoware is still localizing.

    Pose feedback to the scenario is read directly from the CARLA actor, not the
    bridge: because :meth:`spawn` leaves this entity's
    :attr:`~EgoVehicle.actor` set to the ego, every existing condition
    (``EntityLanePositionCondition``,
    ``WaypointCondition``, ``CollisionCondition`` ...) works against the Autoware
    ego exactly as for a TrafficManager or driver ego, via
    ``actor.get_transform()``.

    The ``bridge`` is a required keyword argument: the live gRPC transport (the
    server the interface node dials as a client, splatsim-consistent) is a
    follow-up (see ``proto/autoware_bridge/v0/autoware_bridge.proto``), so callers
    pass a bridge explicitly today (e.g. ``FakeAutowareBridge`` in tests).

    Args:
        config: Bridge connection settings.  ``None`` uses
            :class:`~autoware_carla_scenario.autoware_bridge.base.AutowareBridgeConfig`
            defaults.
        bridge: The bridge to the interface node.
        initial_pose: Map-frame pose Autoware initializes localization at.
            Required before :meth:`on_scenario_start`.
        goal_pose: Map-frame goal pose Autoware plans the route to.  Required
            before :meth:`on_scenario_start`.
    """

    #: Autoware drives; TrafficManager must keep its hands off this actor.
    use_autopilot: bool = False

    #: This entity spawns the ego, so it is the scenario's to clean up.
    attaches_to_existing_actor: bool = False

    #: Autoware plans a route from the initial pose to a goal and only then
    #: engages: without one it never reports ready.
    requires_goal: bool = True

    def __init__(
        self,
        config: Optional["AutowareBridgeConfig"] = None,
        *,
        bridge: "AutowareBridge",
        initial_pose: Optional["BridgePose"] = None,
        goal_pose: Optional["BridgePose"] = None,
    ) -> None:
        super().__init__()
        self._config = config or AutowareBridgeConfig()
        self._bridge = bridge
        self._initial_pose = initial_pose
        self._goal_pose = goal_pose
        self._waypoint_poses: tuple = ()
        self._configured: bool = False
        self._ready: bool = False
        self._ready_ticks: int = 0
        self._termination_requested: bool = False

    # ------------------------------------------------------------------
    # Mission
    # ------------------------------------------------------------------

    def set_mission(
        self,
        initial_pose: Optional["BridgePose"],
        goal_pose: "BridgePose",
        waypoints: Sequence["BridgePose"] = (),
    ) -> None:
        """Set the mission before :meth:`on_scenario_start` hands it over.

        ``ScenarioRunner`` calls ``BaseScenario.create_ego()`` *before*
        ``setup()``, so a scenario whose goal comes from the live map -- snapped
        onto the road surface, say -- cannot pass it to the constructor.  It
        builds the entity first and calls this from ``setup()``.

        The goal is what a scenario knows then.  The initial pose is not: the
        ego actor appears after ``setup()`` and its pose is chosen by whoever
        spawned it, so ``None`` here means "wherever the ego turns out to be",
        read off the attached actor when the scenario starts.  Passing one
        anyway states where the ego is *expected* to be, and a disagreement is
        reported rather than silently localized away.

        The built-in scenarios reach this through
        :class:`~autoware_carla_scenario.actions.routing.RoutingAction`, which
        ``_setup_ego_spawn()`` registers as an init action from the snapped
        spawn and the scenario's ``goal_pose``; calling this directly is for a
        scenario that derives its mission some other way.

        Args:
            initial_pose: Map-frame pose Autoware initializes localization at,
                or ``None`` to take it from the attached ego actor.
            goal_pose: Map-frame goal pose Autoware plans the route to.
            waypoints: Map-frame poses the route must pass through, in order.
        """
        self._initial_pose = initial_pose
        self._goal_pose = goal_pose
        self._waypoint_poses = tuple(waypoints)

    def route_to(
        self,
        world: "carla.World",
        goal: "Lanelet2Pose",
        *,
        initial_pose: Optional["CarlaWorldPose"] = None,
        ground_projection: Optional["GroundProjectionConfig"] = None,
        waypoints: Sequence["Lanelet2Pose"] = (),
    ) -> None:
        """Send Autoware to *goal*: snap it, put it in the map frame, hand it over.

        The whole of "how a destination is delivered to this stack" lives here
        rather than in the action that asks for it.  The goal arrives as a
        Lanelet2 pose -- what a scenario author writes -- and Autoware needs a
        6-DoF pose in its own ``map`` frame, resolved against the road surface
        the vehicle will actually drive on, so both steps happen here where the
        stack's requirements are known.

        Before the run starts this is the mission :meth:`on_scenario_start`
        hands over; after it, it is a re-route, and the initial pose is left
        alone because localization is already running.

        Args:
            world: The CARLA world, used to snap the goal onto the road.
            goal: Lanelet2 pose to route to.
            initial_pose: CARLA world pose to initialize localization at.
                ``None`` leaves it to :meth:`_resolve_initial_pose`, which
                reads the attached actor.
            ground_projection: Settings used to snap the goal to the road
                surface.  Defaults to :class:`GroundProjectionConfig`.
            waypoints: Lanelet2 poses the route must pass through, in order.
                Snapped exactly as the goal is, and for the same reason: they are
                what the scenario author wrote, and Autoware needs map-frame poses
                on the road it will actually drive.
        """
        from ..coordinate import (  # noqa: PLC0415
            GroundProjectionConfig,
            snap_to_carla_road,
        )

        # The goal is snapped as the Lanelet2 pose it was written as, not as its
        # OpenDRIVE projection: the snap takes its heading from the frame it is
        # given, and only the Lanelet2 frame carries the lanelet's direction of
        # travel. On a left-hand-traffic map converted from Lanelet2 the road's
        # reference line runs the other way, so a goal snapped as an OpenDRIVE
        # pose faces back down its lane; the mission planner then measures the
        # goal against its lanelet's angle, finds ~180 degrees against a 45
        # degree threshold, and answers "Goal is not valid!" -- every route
        # request comes back "The planned route is empty" and the scenario never
        # becomes ready. The OpenDRIVE round trip would also move the position
        # (see snap_to_carla_road), and the goal needs no OpenDRIVE metadata.
        projection = ground_projection or GroundProjectionConfig()

        def _snap(pose: "Lanelet2Pose", what: str) -> "CarlaWorldPose":
            snapped = snap_to_carla_road(pose, world, ground_projection=projection)
            logger.info(
                "%s lanelet %d s=%.1f -> CARLA (%.1f, %.1f, %.1f) yaw=%.1f",
                what,
                pose.lanelet_id,
                pose.s,
                snapped.x,
                snapped.y,
                snapped.z,
                snapped.yaw,
            )
            return snapped

        snapped_waypoints = [_snap(pose, "Routing via") for pose in waypoints]
        snapped_goal = _snap(goal, "Routing to")
        self.set_mission(
            None if initial_pose is None else to_map_frame(initial_pose),
            to_map_frame(snapped_goal),
            [to_map_frame(pose) for pose in snapped_waypoints],
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> "AutowareBridgeConfig":
        """Return the bridge connection settings."""
        return self._config

    @property
    def bridge(self) -> "AutowareBridge":
        """Return the bridge used to communicate with the interface node."""
        return self._bridge

    @property
    def termination_requested(self) -> bool:
        """Whether Autoware failed to become ready in time and the run should end."""
        return self._termination_requested

    @property
    def is_initialized(self) -> bool:
        """``True`` once Autoware is ready (initialized, routed, engaged, driving)."""
        return self._ready

    # ------------------------------------------------------------------
    # Manoeuvres
    # ------------------------------------------------------------------

    def change_lane(self, world: "carla.World", direction) -> None:  # noqa: ANN001
        """Refuse a TrafficManager manoeuvre: nothing here is driven by it.

        Autoware plans and executes its own manoeuvres; a lane change is
        something its planner decides, not something the scenario forces on it.
        Send it somewhere with :meth:`route_to` and let it work out the lanes.
        """
        del world
        logger.warning(
            "%s: a lane change was asked for, but this entity is not driven by "
            "the TrafficManager, so forcing one there would do nothing. Route it instead.",
            type(self).__name__,
        )

    def set_desired_speed(self, world: "carla.World", speed_kmh: float) -> None:
        """Refuse a TrafficManager speed: nothing here is driven by it.

        Autoware decides its own speed from its planner and the map's limits.
        A desired speed set on the TrafficManager would be ignored -- the
        runner passes this ego in ``skip_actor_ids`` precisely so the
        TrafficManager does not touch it -- while the action reported progress
        and completion, which is worse than refusing.
        """
        del world, speed_kmh
        logger.warning(
            "%s: a speed was set, but this entity is not driven by the "
            "TrafficManager, so the command would go to an actor it does not "
            "control. Autoware chooses its own speed.",
            type(self).__name__,
        )

    def turn_at_junction(self, world: "carla.World", direction, **kwargs) -> None:  # noqa: ANN001, ANN003
        """Refuse a TrafficManager route: nothing here is driven by it."""
        del world, direction, kwargs
        logger.warning(
            "%s: a turn was asked for, but this entity is not driven by the "
            "TrafficManager, so setting a route there would do nothing.",
            type(self).__name__,
        )

    # ------------------------------------------------------------------
    # Actor lifecycle (attach, not spawn)
    # ------------------------------------------------------------------

    def spawn(self, world: "carla.World", config: "EgoConfig") -> "carla.Actor":
        """Place the ego where the scenario says the run starts.

        The same :func:`spawn_vehicle_actor` call :class:`EgoVehicle` makes, at
        the spawn pose ``setup()`` already resolved and wrote into *config* --
        so the retries and ground projection behave as they do for any other
        vehicle.  Autoware finds the actor afterwards by its ``role_name``.

        Args:
            world: The CARLA world instance.
            config: Ego configuration; its spawn location is where the ego goes.

        Returns:
            The spawned ego vehicle actor.

        Raises:
            RuntimeError: If the ego could not be spawned.
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
        logger.info(
            "Spawned the Autoware ego: id=%d role_name=%s",
            self._vehicle.id,
            EGO_ROLE_NAME,
        )
        return self._vehicle

    def destroy(self) -> None:
        """Destroy the ego actor this entity spawned."""
        if self._vehicle is not None:
            self._vehicle.destroy()
            self._vehicle = None

    # ------------------------------------------------------------------
    # Lifecycle hooks (driven by ScenarioRunner)
    # ------------------------------------------------------------------

    def on_scenario_start(self, world: "carla.World") -> None:
        """Hand Autoware the mission; readiness is awaited in ticks.

        Where the ego actually is is checked against where the scenario said it
        would be: the pose is chosen by the ``spawn_point`` the interface node
        was launched with, on the other side of the run, and the two agreeing is
        otherwise left to whoever typed them.  A scenario that hands over no
        initial pose gets the actor's.

        Raises:
            RuntimeError: If the ego actor has not been attached yet.
            ValueError: If the goal pose is missing.
        """
        del world
        if self.actor is None:
            raise RuntimeError(
                "AutowareEgoEntity.on_scenario_start called before spawn()/attach"
            )
        if self._goal_pose is None:
            raise ValueError(
                "AutowareEgoEntity has no goal: Autoware needs one to plan a "
                "route. A scenario that calls BaseScenario._setup_ego_spawn() "
                "gets it from its goal_pose (ego.goal_lanelet_id in the config); "
                "one that does not must call set_mission() from its setup(), "
                "where poses snapped onto the live map exist, or pass it to the "
                "constructor."
            )
        # The scenario's poses are where the *actor* is or is to be -- its origin,
        # the vehicle's centre -- while Autoware localizes and plans for
        # ``base_link``, the rear axle.  Handed over unmoved, localization starts
        # half a wheelbase ahead of where the sensors say the car is, and NDT,
        # with little but a flat road to align against while standing still,
        # settles metres off: the ego then steers for the lane it thinks it has
        # left.  The goal moves with it, so the actor still stops at the goal.
        offset = self._base_link_offset()
        initial_pose = self._resolve_initial_pose().moved_forward(offset)
        goal_pose = self._goal_pose.moved_forward(offset)
        waypoint_poses = tuple(
            pose.moved_forward(offset) for pose in self._waypoint_poses
        )
        logger.info(
            "Autoware's base_link is %.3f m along the ego from its origin; "
            "initial pose (%.2f, %.2f), goal (%.2f, %.2f) in the map frame",
            offset,
            initial_pose.position.x,
            initial_pose.position.y,
            goal_pose.position.x,
            goal_pose.position.y,
        )
        # The transport is brought up here rather than at construction: a batch
        # of scenarios is built before the first one runs, and two bridges
        # cannot hold the same address at once.
        self._bridge.start()
        # Waypoints only when there are some, so a bridge implementing the
        # two-argument configure() it was written against keeps working for
        # every mission that names none.
        if waypoint_poses:
            self._bridge.configure(initial_pose, goal_pose, waypoint_poses)
        else:
            self._bridge.configure(initial_pose, goal_pose)
        self._configured = True

    def _base_link_offset(self) -> float:
        """Metres along the ego from the actor's origin to Autoware's ``base_link``."""
        from ..driver.observation import rear_axle_offset  # noqa: PLC0415

        assert self.actor is not None  # noqa: S101 - checked by the caller
        return rear_axle_offset(self.actor, self._config.base_link_offset_m)

    def _resolve_initial_pose(self) -> "BridgePose":
        """Return the pose to initialize localization at, checked against the ego.

        The scenario's pose wins when it has one: it is a spawn snapped onto the
        road surface, the height ``base_link`` is at, while the actor's
        transform is the actor's own origin -- a metre and a half above the road
        on some vehicles.  Both are the vehicle's centre along its length; the
        caller moves the result back to the rear axle.  What the actor is good for is saying whether the ego
        is *where the scenario thinks*, which is a different question and the one
        that goes wrong silently.

        The comparison is horizontal for the same reason: the heights are
        measured from different places and would disagree on every run.
        """
        assert self.actor is not None  # noqa: S101 - checked by the caller
        actual = to_map_frame(
            CarlaWorldPose.from_carla_transform(self.actor.get_transform())
        )
        expected = self._initial_pose
        if expected is None:
            logger.info(
                "No initial pose was set; localizing at the ego's own pose "
                "(%.2f, %.2f). Autoware fits the height to the map.",
                actual.position.x,
                actual.position.y,
            )
            return actual
        offset = math.dist(
            (expected.position.x, expected.position.y),
            (actual.position.x, actual.position.y),
        )
        if offset > _INITIAL_POSE_TOLERANCE_M:
            logger.warning(
                "The ego is %.2f m from where the scenario expected it "
                "(%.2f, %.2f) -- it is at (%.2f, %.2f). The interface node's "
                "spawn_point and the scenario's spawn disagree; localization is "
                "initialized at the scenario's pose, which is the one its goal "
                "and conditions were written against.",
                offset,
                expected.position.x,
                expected.position.y,
                actual.position.x,
                actual.position.y,
            )
        return expected

    def on_tick(self, world: "carla.World", elapsed: float) -> None:
        """Poll Autoware's readiness each tick until it is ready (or times out).

        Autoware only makes progress while simulation time advances, so readiness
        is awaited here rather than blocking in :meth:`on_scenario_start`.  Once
        ready this is a no-op.  If Autoware is not ready within
        ``ready_timeout_ticks``, the entity requests early termination.
        """
        del world, elapsed
        if self._ready or self._termination_requested or not self._configured:
            return
        if self._bridge.is_ready():
            self._ready = True
            logger.info("Autoware is ready; scenario may proceed")
            return
        self._ready_ticks += 1
        if self._ready_ticks >= self._config.ready_timeout_ticks:
            logger.warning(
                "Autoware not ready after %d ticks - requesting termination",
                self._ready_ticks,
            )
            self._termination_requested = True

    def on_scenario_end(self, world: "carla.World") -> None:
        """Close the bridge transport.  Never raises."""
        del world
        try:
            self._bridge.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.warning("AutowareEgoEntity: bridge.close() failed", exc_info=True)
