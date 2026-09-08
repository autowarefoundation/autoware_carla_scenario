"""Ego vehicle spawning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import carla

    from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose
    from ..scenario_base import EgoConfig

from ..constants import EGO_ROLE_NAME
from ._spawn import spawn_vehicle_actor
from .tm_driving import TrafficManagerDriven


class EgoVehicle(TrafficManagerDriven):
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

    #: When ``True``, the actor is not this entity's to create or destroy: it
    #: belongs to something outside the scenario (an ``autoware_carla_interface``
    #: node, say) and :meth:`spawn` only attaches to it.  :class:`ScenarioRunner`
    #: reads this to leave that actor out of the cleanup it does before a run.
    attaches_to_existing_actor: bool = False

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
    def termination_requested(self) -> bool:
        """Whether this entity has asked to end the scenario early.

        :class:`ScenarioRunner` checks this after the pass and fail conditions, so a
        condition that fires on the same tick still decides the outcome.
        """
        return False

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
    ) -> None:
        """Send this entity to *goal*, if it is the kind that plans a route.

        A no-op here.  An entity driven by something that takes no destination
        -- the TrafficManager, an external control policy -- has nothing to do
        with a goal, and saying so costs nothing; an entity that runs its own
        planner (see
        :class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`)
        overrides this.

        Routing lives on the entity rather than in the action that asks for it
        because *how* a destination is delivered is the entity's business: one
        stack takes a map-frame pose over a bridge, another might take a lane
        sequence or nothing at all.  The action only says where and when.

        Args:
            world: The CARLA world, for resolving the goal against the road.
            goal: Lanelet2 pose to route to.
            initial_pose: Pose to initialize localization at, for a stack that
                needs one.  ``None`` leaves it to the entity.
            ground_projection: Settings used to snap the goal to the road.
        """
        del world, goal, initial_pose, ground_projection

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
            elapsed: Wall-clock seconds since the tick loop started.
        """

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
