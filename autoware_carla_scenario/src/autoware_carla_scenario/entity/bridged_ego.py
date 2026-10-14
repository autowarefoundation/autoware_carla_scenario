"""Egos whose driving stack reaches the scenario over the :class:`AutowareBridge`.

The bridge is the splatsim-consistent handshake of
:mod:`autoware_carla_scenario.autoware_bridge`: the scenario library hosts the
``AutowareBridge`` gRPC server and the stack's interface node dials it as a
client, pulls the mission (initial pose, goal, waypoints) and reports a single
readiness flag.  Nothing in that contract is particular to Autoware -- it is how
any stack that plans from a mission and needs time to come up is told where to
go and waited for -- so it lives here, in :class:`BridgeHandshake`, which any
ego can hold.  :class:`BridgedEgoEntity` is the ego that is nothing but that
handshake -- the stack drives -- and
:class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity` is the
Autoware stack on top of it; an ego the framework drives itself, such as
:class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`,
holds one alongside its own loop.  It is held rather than inherited because the
static check takes single inheritance only (docs/typecheck.md).

What the handshake does around the bridge:

* resolves the mission (:meth:`BridgeHandshake.route_to`,
  :meth:`BridgeHandshake.set_mission`) into the stack's ``map`` frame and moves
  it to the stack's ``base_link``;
* hands it over and polls readiness each tick (:attr:`BridgeHandshake.ready`),
  which the ego holding it reports as not yet initialized, so the runner holds
  the scenario clock until the stack is driving;
* given a :class:`~autoware_carla_scenario.autoware_stack.AutowareLauncher`,
  starts a fresh stack for its scenario once the mission is being served and
  removes it when the scenario ends.

Tick ownership: the scenario framework remains the tick master; the interface
node runs as a non-ticking, asynchronous I/O bridge.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..autoware_bridge.base import AutowareBridge, BridgePose
    from ..autoware_stack.launcher import AutowareLauncher
    from ..coordinate import GroundProjectionConfig, Lanelet2Pose

from ..autoware_bridge.base import AutowareBridgeConfig
from ..constants import EGO_ROLE_NAME
from ..coordinate.poses import CarlaWorldPose
from ..coordinate.transform import to_map_frame
from .externally_driven import ExternallyDrivenEgo

logger = logging.getLogger(__name__)

__all__ = ["BridgeHandshake", "BridgedEgoEntity"]

#: How far, horizontally, the ego may be from where the scenario expected it
#: before that disagreement is worth a warning.  Snapping a spawn onto the road
#: surface moves it by centimetres; a spawn_point that does not match moves it
#: by metres.
_INITIAL_POSE_TOLERANCE_M: float = 1.0


class BridgeHandshake:
    """The mission and readiness handshake with a stack, over the bridge.

    Held by the ego entity whose stack it talks to, which spawns the actor and
    hands it over in :meth:`start`.  Nothing here ends a run itself: a stack that
    fails is reported by :meth:`poll` for the ego to end it.

    Args:
        config: Bridge connection settings.  ``None`` uses
            :class:`~autoware_carla_scenario.autoware_bridge.base.AutowareBridgeConfig`
            defaults.
        bridge: The bridge to the interface node.
        stack_name: The stack, as messages name it ("the Autoware stack").
        initial_pose: Map-frame pose the stack initializes localization at.
            ``None`` takes it from the ego when the scenario starts.
        goal_pose: Map-frame goal pose the stack plans the route to.  Required
            before :meth:`start`.
        launcher: Starts the stack when the scenario starts and stops it when
            it ends.  ``None`` leaves the stack to whoever started it.
        episode_name: The scenario's name, for the launcher's logs.
    """

    _config: AutowareBridgeConfig
    _bridge: AutowareBridge
    _stack_name: str
    _launcher: Optional[AutowareLauncher]
    _episode_name: str
    _stack_running: bool
    _boot_started: Optional[float]
    _initial_pose: Optional[BridgePose]
    _goal_pose: Optional[BridgePose]
    _waypoint_poses: list[BridgePose]
    _configured: bool
    _ready: bool
    _failed: bool
    _ready_ticks: int

    def __init__(
        self,
        config: Optional["AutowareBridgeConfig"],
        bridge: "AutowareBridge",
        stack_name: str,
        initial_pose: Optional["BridgePose"] = None,
        goal_pose: Optional["BridgePose"] = None,
        launcher: Optional["AutowareLauncher"] = None,
        episode_name: str = "",
    ) -> None:
        self._config = config or AutowareBridgeConfig()
        self._bridge = bridge
        self._stack_name = stack_name
        self._launcher = launcher
        self._episode_name = episode_name
        self._stack_running: bool = False
        self._boot_started: Optional[float] = None
        self._initial_pose = initial_pose
        self._goal_pose = goal_pose
        self._waypoint_poses = []
        self._configured: bool = False
        self._ready: bool = False
        self._failed: bool = False
        self._ready_ticks: int = 0

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
    def launcher(self) -> Optional["AutowareLauncher"]:
        """The launcher that starts the stack for this scenario, if any."""
        return self._launcher

    @property
    def ready(self) -> bool:
        """``True`` once the stack is ready (initialized, routed, engaged, driving)."""
        return self._ready

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
        ego actor appears after ``setup()``, so ``None`` here means "wherever
        the ego turns out to be", read off the actor when the scenario starts.
        Passing one anyway states where the ego is *expected* to be, and a
        disagreement is reported rather than silently localized away.

        The built-in scenarios reach this through
        :class:`~autoware_carla_scenario.actions.routing.RoutingAction`, which
        ``_setup_ego_spawn()`` registers as an init action from the snapped
        spawn and the scenario's ``goal_pose``; calling this directly is for a
        scenario that derives its mission some other way.

        Args:
            initial_pose: Map-frame pose the stack initializes localization at,
                or ``None`` to take it from the ego actor.
            goal_pose: Map-frame goal pose the stack plans the route to.
            waypoints: Map-frame poses the route must pass through, in order.
        """
        self._initial_pose = initial_pose
        self._goal_pose = goal_pose
        self._waypoint_poses = list(waypoints)

    def route_to(
        self,
        world: "carla.World",
        goal: "Lanelet2Pose",
        initial_pose: Optional["CarlaWorldPose"] = None,
        ground_projection: Optional["GroundProjectionConfig"] = None,
        waypoints: Sequence["Lanelet2Pose"] = (),
    ) -> None:
        """Send the stack to *goal*: snap it, put it in the map frame, hand it over.

        The whole of "how a destination is delivered to this stack" lives here
        rather than in the action that asks for it.  The goal arrives as a
        Lanelet2 pose -- what a scenario author writes -- and the stack needs a
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
                reads the actor.
            ground_projection: Settings used to snap the goal to the road
                surface.  Defaults to :class:`GroundProjectionConfig`.
            waypoints: Lanelet2 poses the route must pass through, in order.
                Snapped exactly as the goal is, and for the same reason: they are
                what the scenario author wrote, and the stack needs map-frame
                poses on the road it will actually drive.
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
        # pose faces back down its lane; Autoware's mission planner then
        # measures the goal against its lanelet's angle, finds ~180 degrees
        # against a 45 degree threshold, and answers "Goal is not valid!" --
        # every route request comes back "The planned route is empty" and the
        # scenario never becomes ready. The OpenDRIVE round trip would also move
        # the position (see snap_to_carla_road), and the goal needs no OpenDRIVE
        # metadata.
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
        # An if statement, not a conditional expression: Codon narrows no
        # Optional, and unwraps one assigned to an annotated local.
        initial: Optional["BridgePose"] = None
        if initial_pose is not None:
            known: CarlaWorldPose = initial_pose
            initial = to_map_frame(known)
        self.set_mission(
            initial,
            to_map_frame(snapped_goal),
            [to_map_frame(pose) for pose in snapped_waypoints],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, actor: "carla.Actor", owner: str) -> None:
        """Hand the stack the mission for the ego *actor*; readiness is polled.

        Where the ego actually is is checked against where the scenario said it
        would be, and a scenario that hands over no initial pose gets the
        actor's.  A retried scenario starts again from nothing: no readiness the
        attempt before reached carries over.

        Args:
            actor: The ego, already spawned where the run starts.
            owner: The entity's class name, for its messages.

        Raises:
            ValueError: If the goal pose is missing.
        """
        logger.info(
            "The %s stack is to attach to the ego: id=%d role_name=%s",
            self._stack_name,
            actor.id,
            EGO_ROLE_NAME,
        )
        self._configured = False
        self._ready = False
        self._failed = False
        self._ready_ticks = 0
        self._stack_running = False
        self._boot_started = None
        if self._goal_pose is None:
            raise ValueError(
                f"{owner} has no goal: the {self._stack_name} stack "
                "needs one to plan a route. A scenario that calls "
                "BaseScenario._setup_ego_spawn() gets it from its goal_pose "
                "(ego.goal_lanelet_id in the config); one that does not must call "
                "set_mission() from its setup(), where poses snapped onto the "
                "live map exist, or pass it to the constructor."
            )
        # The scenario's poses are where the *actor* is or is to be -- its origin,
        # the vehicle's centre -- while the stack localizes and plans for
        # ``base_link``, the rear axle.  Handed over unmoved, localization starts
        # half a wheelbase ahead of where the sensors say the car is, and NDT,
        # with little but a flat road to align against while standing still,
        # settles metres off: the ego then steers for the lane it thinks it has
        # left.  The goal moves with it, so the actor still stops at the goal.
        offset = self._base_link_offset(actor)
        initial_pose = self._resolve_initial_pose(actor).moved_forward(offset)
        goal_pose = self._goal_pose.moved_forward(offset)
        waypoint_poses = [pose.moved_forward(offset) for pose in self._waypoint_poses]
        logger.info(
            "The %s stack's base_link is %.3f m along the ego from its origin; "
            "initial pose (%.2f, %.2f), goal (%.2f, %.2f) in the map frame",
            self._stack_name,
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
        # Codon calls configure(), which bridges override, through a dispatch
        # thunk, which fills in no default and does not wrap a list in the
        # Optional it takes (docs/typecheck.md): the check passes an Optional.
        if waypoint_poses:
            waypoints: Optional[list["BridgePose"]] = waypoint_poses
            self._bridge.configure(initial_pose, goal_pose, waypoints)
        elif TYPE_CHECKING:
            no_waypoints: Optional[list["BridgePose"]] = None
            self._bridge.configure(initial_pose, goal_pose, no_waypoints)
        else:
            self._bridge.configure(initial_pose, goal_pose)
        self._configured = True
        if self._launcher is not None:
            self._start_stack()

    def _start_stack(self) -> None:
        """Start this scenario's stack, now that the mission is being served.

        After the ego exists, so the interface node finds the actor it is to
        attach to, and after :meth:`configure`, so the first ``GetMission`` the
        stack sends is answered with the mission.
        """
        assert self._launcher is not None  # noqa: S101 - checked by the caller
        address = self._bridge.client_address or self._config.address
        episode = self._launcher.start(
            name=self._episode_name or "scenario", bridge_address=address
        )
        self._stack_running = True
        self._boot_started = time.monotonic()
        logger.info(
            "Started %s stack #%d for %s; it is to dial the bridge at %s",
            self._stack_name,
            episode.index,
            episode.name,
            address,
        )

    def _base_link_offset(self, actor: "carla.Actor") -> float:
        """Metres along the ego from the actor's origin to the stack's ``base_link``."""
        from ..driver.observation import rear_axle_offset  # noqa: PLC0415

        return rear_axle_offset(actor, self._config.base_link_offset_m)

    def _resolve_initial_pose(self, actor: "carla.Actor") -> "BridgePose":
        """Return the pose to initialize localization at, checked against the ego.

        The scenario's pose wins when it has one: it is a spawn snapped onto the
        road surface, the height ``base_link`` is at, while the actor's
        transform is the actor's own origin -- a metre and a half above the road
        on some vehicles.  Both are the vehicle's centre along its length; the
        caller moves the result back to the rear axle.  What the actor is good
        for is saying whether the ego is *where the scenario thinks*, which is a
        different question and the one that goes wrong silently.

        The comparison is horizontal for the same reason: the heights are
        measured from different places and would disagree on every run.
        """
        actual = to_map_frame(
            CarlaWorldPose.from_carla_transform(actor.get_transform())
        )
        expected = self._initial_pose
        if expected is None:
            logger.info(
                "No initial pose was set; localizing at the ego's own pose "
                "(%.2f, %.2f). The %s stack fits the height to the map.",
                actual.position.x,
                actual.position.y,
                self._stack_name,
            )
            return actual
        # math.hypot of the differences, which is math.dist's own computation:
        # Codon's math has no dist().
        offset = math.hypot(
            expected.position.x - actual.position.x,
            expected.position.y - actual.position.y,
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

    def poll(self) -> Optional[str]:
        """Poll the stack's readiness, once a tick until it is ready.

        The stack only makes progress while simulation time advances, so
        readiness is awaited here rather than blocking in :meth:`start`.  Once
        ready this only watches the launched stack.

        Returns:
            Why the run is to end -- the stack exited, did not reach the bridge
            in time, or was not ready within ``ready_timeout_ticks`` -- the
            first time that happens; otherwise ``None``.
        """
        if self._failed or not self._configured:
            return None
        if self._stack_running:
            assert self._launcher is not None  # noqa: S101 - set when it started
            code = self._launcher.poll()
            if code is not None:
                self._stack_running = False
                return self._fail(
                    f"The {self._stack_name} stack exited with code {code}"
                )
        if self._ready:
            return None
        if self._stack_running and not self._bridge.client_connected:
            assert self._boot_started is not None  # noqa: S101 - set when it started
            booting_s = time.monotonic() - self._boot_started
            if booting_s > self._config.boot_timeout_s:
                return self._fail(
                    f"The {self._stack_name} stack did not reach the bridge within "
                    f"{self._config.boot_timeout_s:.0f} s of being started"
                )
            if self._config.boot_tick_period_s > 0:
                time.sleep(self._config.boot_tick_period_s)
            return None
        if self._bridge.is_ready():
            self._ready = True
            logger.info("The %s stack is ready; scenario may proceed", self._stack_name)
            return None
        self._ready_ticks += 1
        if self._ready_ticks >= self._config.ready_timeout_ticks:
            return self._fail(
                f"The {self._stack_name} stack not ready after {self._ready_ticks} ticks"
            )
        return None

    def _fail(self, reason: str) -> str:
        """Stop polling, and say why."""
        self._failed = True
        return reason

    def close(self, owner: str) -> None:
        """Stop this scenario's stack, then close the bridge.  Never raises.

        Args:
            owner: The entity's class name, for its messages.
        """
        if self._launcher is not None:
            try:
                self._launcher.stop()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning(
                    "%s: stopping the %s stack failed",
                    owner,
                    self._stack_name,
                    exc_info=True,
                )
            self._stack_running = False
        try:
            self._bridge.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.warning("%s: bridge.close() failed", owner, exc_info=True)


class BridgedEgoEntity(ExternallyDrivenEgo):
    """Ego vehicle driven by a stack that takes its mission over the bridge.

    The :class:`ScenarioRunner` reads :attr:`EgoVehicle.use_autopilot` (``False``
    here) and skips ``set_autopilot(True)`` for this actor, leaving it under the
    stack's control.

    Two things about a run are different because the stack drives:

    * The actor is the scenario's, like any other entity's: this entity spawns
      it and destroys it.  The stack drives it in between.
    * The scenario cannot be judged until the stack is driving.  The runner
      holds the scenario clock and its conditions until :attr:`is_initialized`,
      so a condition that is already true near the initial pose -- standing
      still, say -- cannot record a result while the stack is still localizing.

    Pose feedback to the scenario is read directly from the CARLA actor, not the
    bridge: because :meth:`spawn` leaves this entity's
    :attr:`~EgoVehicle.actor` set to the ego, every existing condition
    (``EntityLanePositionCondition``, ``WaypointCondition``,
    ``CollisionCondition`` ...) works against it exactly as for a TrafficManager
    or driver ego, via ``actor.get_transform()``.

    Everything about the handshake is the :class:`BridgeHandshake` it holds;
    subclasses name the stack (:attr:`~ExternallyDrivenEgo.stack_name`).

    Args:
        config: Bridge connection settings.  ``None`` uses
            :class:`~autoware_carla_scenario.autoware_bridge.base.AutowareBridgeConfig`
            defaults.
        bridge: The bridge to the interface node.
        initial_pose: Map-frame pose the stack initializes localization at.
            ``None`` takes it from the ego when the scenario starts.
        goal_pose: Map-frame goal pose the stack plans the route to.  Required
            before :meth:`on_scenario_start`.
        launcher: Starts the stack when the scenario starts and stops it when
            it ends.  ``None`` leaves the stack to whoever started it.
        episode_name: The scenario's name, for the launcher's logs.
    """

    stack_name: str = "bridged"

    #: The stack plans a route from the initial pose to a goal and only then
    #: engages: without one it never reports ready.
    requires_goal: bool = True

    _handshake: BridgeHandshake

    def __init__(
        self,
        config: Optional["AutowareBridgeConfig"] = None,
        *,
        bridge: "AutowareBridge",
        initial_pose: Optional["BridgePose"] = None,
        goal_pose: Optional["BridgePose"] = None,
        launcher: Optional["AutowareLauncher"] = None,
        episode_name: str = "",
    ) -> None:
        super().__init__()
        self._handshake = BridgeHandshake(
            config,
            bridge,
            self.stack_name,
            initial_pose,
            goal_pose,
            launcher,
            episode_name,
        )

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

        See :meth:`BridgeHandshake.set_mission`.
        """
        self._handshake.set_mission(initial_pose, goal_pose, waypoints)

    def route_to(
        self,
        world: "carla.World",
        goal: "Lanelet2Pose",
        *,
        initial_pose: Optional["CarlaWorldPose"] = None,
        ground_projection: Optional["GroundProjectionConfig"] = None,
        waypoints: Sequence["Lanelet2Pose"] = (),
    ) -> None:
        """Send the stack to *goal*.  See :meth:`BridgeHandshake.route_to`."""
        self._handshake.route_to(
            world, goal, initial_pose, ground_projection, waypoints
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def handshake(self) -> BridgeHandshake:
        """The handshake with the stack."""
        return self._handshake

    @property
    def config(self) -> "AutowareBridgeConfig":
        """Return the bridge connection settings."""
        return self._handshake.config

    @property
    def bridge(self) -> "AutowareBridge":
        """Return the bridge used to communicate with the interface node."""
        return self._handshake.bridge

    @property
    def launcher(self) -> Optional["AutowareLauncher"]:
        """The launcher that starts the stack for this scenario, if any."""
        return self._handshake.launcher

    @property
    def is_initialized(self) -> bool:
        """``True`` once the stack is ready (initialized, routed, engaged, driving)."""
        return self._handshake.ready

    # ------------------------------------------------------------------
    # Lifecycle hooks (driven by ScenarioRunner)
    # ------------------------------------------------------------------

    def on_scenario_start(self, world: "carla.World") -> None:
        """Hand the stack the mission; readiness is awaited in ticks.

        Raises:
            RuntimeError: If the ego actor has not been spawned yet.
            ValueError: If the goal pose is missing.
        """
        if self.actor is None:
            raise RuntimeError(
                f"{type(self).__name__}.on_scenario_start called before spawn()"
            )
        actor: "carla.Actor" = self.actor
        self._reset_termination()
        self._handshake.start(actor, type(self).__name__)

    def on_tick(self, world: "carla.World", elapsed: float) -> None:
        """Poll the stack's readiness; end the run if the stack fails."""
        reason = self._handshake.poll()
        if reason is not None:
            known: str = reason
            self._terminate(known)

    def on_scenario_end(self, world: "carla.World") -> None:
        """Stop this scenario's stack, then close the bridge.  Never raises."""
        self._handshake.close(type(self).__name__)
