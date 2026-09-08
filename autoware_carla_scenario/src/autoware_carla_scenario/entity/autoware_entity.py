"""Autoware ego vehicles.

Two entities live here:

* :class:`AutowareEntity` - a bare placeholder that spawns the ego, opts out of
  TrafficManager, and leaves the actor standing still for an external stack to
  control out of band.  Kept for backwards compatibility.

* :class:`AutowareEgoEntity` - the closed-loop entity.  Autoware (via the
  ``autoware_carla_interface`` ROS 2 node) reads CARLA sensors and applies
  control to the ego **directly**, so unlike
  :class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`
  the framework is *not* in the control loop.  This entity **attaches** to the
  ego actor spawned by the interface node, hands Autoware the scenario's initial
  pose and goal, and waits for Autoware to become ready.  The whole startup
  sequence (localization init, routing, engage) is owned by the Autoware side;
  see :mod:`autoware_carla_scenario.autoware_bridge`.

Tick ownership: the scenario framework remains the tick master; the interface
node runs as a non-ticking, asynchronous I/O bridge (``sync_mode:=false``).

Role name: the framework identifies the ego by
:data:`~autoware_carla_scenario.constants.EGO_ROLE_NAME` (``"Ego"``).  Launch
the interface node with ``ego_vehicle_role_name:=Ego`` so the spawned actor
matches.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import carla

    from ..autoware_bridge.base import AutowareBridge, BridgePose
    from ..coordinate import GroundProjectionConfig, Lanelet2Pose
    from ..scenario_base import EgoConfig

from ..autoware_bridge.base import AutowareBridgeConfig
from ..conditions.base import find_actor_by_role_name
from ..constants import EGO_ROLE_NAME
from ..coordinate.poses import CarlaWorldPose
from ..coordinate.transform import to_map_frame
from .ego import EgoVehicle

logger = logging.getLogger(__name__)

#: Polling interval while waiting for the interface node to spawn the ego actor.
_ATTACH_POLL_INTERVAL_S: float = 0.5

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
    """Ego vehicle spawned by ``autoware_carla_interface`` and driven by Autoware.

    The :class:`ScenarioRunner` reads :attr:`EgoVehicle.use_autopilot` (``False``
    here) and skips ``set_autopilot(True)`` for this actor, leaving it under
    Autoware's control.

    Two things about a run are different because the ego is someone else's:

    * The actor is not the scenario's to destroy.
      :attr:`~EgoVehicle.attaches_to_existing_actor` is ``True``, which keeps it
      (and its sensors) out of the cleanup :class:`ScenarioRunner` does before a
      run -- otherwise the interface-spawned ego would be destroyed and
      :meth:`spawn` would only expire at ``attach_timeout``.
    * The scenario cannot be judged until Autoware is driving.  The runner holds
      the scenario clock and its conditions until :attr:`is_initialized`, so a
      condition that is already true near the initial pose -- standing still,
      say -- cannot record a result while Autoware is still localizing.

    Pose feedback to the scenario is read directly from the CARLA actor, not the
    bridge: because :meth:`spawn` attaches this entity's
    :attr:`~EgoVehicle.actor` by ``role_name``, every existing condition
    (``EntityLanePositionCondition``,
    ``WaypointCondition``, ``CollisionCondition`` ...) works against the Autoware
    ego exactly as for a TrafficManager or driver ego, via
    ``find_actor_by_role_name(world, EGO_ROLE_NAME).get_transform()``.

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

    #: The interface node spawns the ego and owns its lifecycle.
    attaches_to_existing_actor: bool = True

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
        self._configured: bool = False
        self._ready: bool = False
        self._ready_ticks: int = 0
        self._termination_requested: bool = False

    # ------------------------------------------------------------------
    # Mission
    # ------------------------------------------------------------------

    def set_mission(
        self, initial_pose: Optional["BridgePose"], goal_pose: "BridgePose"
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
        """
        self._initial_pose = initial_pose
        self._goal_pose = goal_pose

    def route_to(
        self,
        world: "carla.World",
        goal: "Lanelet2Pose",
        *,
        initial_pose: Optional["CarlaWorldPose"] = None,
        ground_projection: Optional["GroundProjectionConfig"] = None,
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
        """
        from ..coordinate import (  # noqa: PLC0415
            GroundProjectionConfig,
            snap_to_carla_road,
            to_opendrive,
        )

        snapped = snap_to_carla_road(
            to_opendrive(goal),
            world,
            ground_projection=ground_projection or GroundProjectionConfig(),
        )
        logger.info(
            "Routing to lanelet %d s=%.1f -> CARLA (%.1f, %.1f, %.1f)",
            goal.lanelet_id,
            goal.s,
            snapped.x,
            snapped.y,
            snapped.z,
        )
        self.set_mission(
            None if initial_pose is None else to_map_frame(initial_pose),
            to_map_frame(snapped),
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
        """Attach to the ego actor already spawned by the interface node.

        This does **not** create a new actor.  It polls the world for an actor
        whose ``role_name`` matches :data:`EGO_ROLE_NAME` until one appears or
        :attr:`AutowareBridgeConfig.attach_timeout` elapses.

        Args:
            world: The CARLA world instance.
            config: Ego configuration (accepted for API compatibility with
                :class:`EgoVehicle`; the spawn location is owned by the interface
                node and is not used here).

        Returns:
            The attached ego vehicle actor.

        Raises:
            RuntimeError: If no matching ego actor appears within the timeout.
        """
        del config  # Spawn is owned by the interface node; config is unused.

        deadline = time.monotonic() + self._config.attach_timeout
        while True:
            actor = find_actor_by_role_name(world, EGO_ROLE_NAME)
            if actor is not None:
                self._vehicle = actor
                logger.info(
                    "Attached to Autoware ego actor: id=%d role_name=%s",
                    actor.id,
                    EGO_ROLE_NAME,
                )
                return actor
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"No ego actor with role_name={str(EGO_ROLE_NAME)!r} appeared "
                    f"within {self._config.attach_timeout:.1f}s. Ensure "
                    "autoware_carla_interface is running and launched with "
                    "ego_vehicle_role_name:=Ego."
                )
            time.sleep(_ATTACH_POLL_INTERVAL_S)

    def destroy(self) -> None:
        """Detach from the ego actor without destroying it.

        The interface node owns the ego actor's lifecycle, so this only clears
        the local reference; it never calls ``actor.destroy()``.
        """
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
        initial_pose = self._resolve_initial_pose()
        # The transport is brought up here rather than at construction: a batch
        # of scenarios is built before the first one runs, and two bridges
        # cannot hold the same address at once.
        self._bridge.start()
        self._bridge.configure(initial_pose, self._goal_pose)
        self._configured = True

    def _resolve_initial_pose(self) -> "BridgePose":
        """Return the pose to initialize localization at, checked against the ego.

        The scenario's pose wins when it has one: it is a spawn snapped onto the
        road surface, which is what ``base_link`` means, while the actor's
        transform is the actor's own origin -- a metre and a half above the road
        on some vehicles.  What the actor is good for is saying whether the ego
        is *where the scenario thinks*, which is a different question and the one
        that goes wrong silently.

        The comparison is horizontal for the same reason: the heights are
        measured from different places and would disagree on every run.
        """
        assert self.actor is not None  # noqa: S101 - checked by the caller
        transform = self.actor.get_transform()
        actual = to_map_frame(
            CarlaWorldPose(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z,
                roll=transform.rotation.roll,
                pitch=transform.rotation.pitch,
                yaw=transform.rotation.yaw,
            )
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
