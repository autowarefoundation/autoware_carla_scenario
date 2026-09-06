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
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import carla

    from ..autoware_bridge.base import AutowareBridge, BridgePose
    from ..scenario_base import EgoConfig

from ..autoware_bridge.base import AutowareBridgeConfig
from ..conditions.base import find_actor_by_role_name
from ..constants import EGO_ROLE_NAME
from .ego import EgoVehicle

logger = logging.getLogger(__name__)

#: Polling interval while waiting for the interface node to spawn the ego actor.
_ATTACH_POLL_INTERVAL_S: float = 0.5


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

    def set_mission(self, initial_pose: "BridgePose", goal_pose: "BridgePose") -> None:
        """Set the mission before :meth:`on_scenario_start` hands it over.

        ``ScenarioRunner`` calls ``BaseScenario.create_ego()`` *before*
        ``setup()``, so a scenario whose poses come from the live world -- a
        spawn snapped onto the road surface, say -- cannot pass them to the
        constructor.  It builds the entity first and calls this from ``setup()``.

        The built-in scenarios reach this through
        :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.configure_autoware_mission`,
        which ``_setup_ego_spawn()`` calls with the snapped spawn and the
        scenario's ``goal_pose``; calling this directly is for a scenario that
        derives its mission some other way.

        Args:
            initial_pose: Map-frame pose Autoware initializes localization at.
            goal_pose: Map-frame goal pose Autoware plans the route to.
        """
        self._initial_pose = initial_pose
        self._goal_pose = goal_pose

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
        """Hand Autoware the initial pose and goal; readiness is awaited in ticks.

        Raises:
            RuntimeError: If the ego actor has not been attached yet.
            ValueError: If the initial pose or goal pose is missing.
        """
        del world
        if self.actor is None:
            raise RuntimeError(
                "AutowareEgoEntity.on_scenario_start called before spawn()/attach"
            )
        if self._initial_pose is None or self._goal_pose is None:
            raise ValueError(
                "AutowareEgoEntity has no mission: Autoware needs an initial pose "
                "and a goal pose to localize and route. A scenario that calls "
                "BaseScenario._setup_ego_spawn() gets this from its goal_pose "
                "(ego.goal_lanelet_id in the config); one that does not must call "
                "set_mission() from its setup(), where poses snapped onto the live "
                "map exist, or pass them to the constructor."
            )
        # The transport is brought up here rather than at construction: a batch
        # of scenarios is built before the first one runs, and two bridges
        # cannot hold the same address at once.
        self._bridge.start()
        self._bridge.configure(self._initial_pose, self._goal_pose)
        self._configured = True

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
