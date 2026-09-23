"""Pedestrian entity for CARLA scenarios."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import carla

from ..entity_role import EntityRole

if TYPE_CHECKING:
    from ._spawn import SpawnTransform

logger = logging.getLogger(__name__)

__all__ = ["PedestrianEntity", "PedestrianEntityConfig"]

#: Metres the spawn point is raised before the walker is placed.  A pose taken
#: off the map sits exactly on the surface, and a walker spawned flush with it
#: is rejected as colliding with the ground.
_SPAWN_LIFT_M = 0.3


@dataclass
class PedestrianEntityConfig:
    """Configuration for spawning a pedestrian.

    Attributes:
        role_name: The role the pedestrian answers to.
        spawn_location: Where to place it, as a transform.
        walker_type: CARLA blueprint id; must be a ``walker.*``.
    """

    role_name: Union[EntityRole, str]
    spawn_location: "SpawnTransform"
    walker_type: str = "walker.pedestrian.0001"


class PedestrianEntity:
    """Manages the lifecycle of a pedestrian actor in CARLA.

    Deliberately **not** a :class:`~autoware_carla_scenario.traffic.driven.BackendDriven`
    entity.  A traffic backend drives vehicles; nothing in this framework
    drives a walker, and inheriting the manoeuvres would offer ``change_lane``
    on something that has no lane.  A pedestrian moves because a
    :class:`~autoware_carla_scenario.actions.WalkStraightAction` tells it to,
    and stands still otherwise.

    Walking is commanded with :class:`carla.WalkerControl` rather than through
    a ``controller.ai.walker``.  The AI controller exists to send a walker *to
    a destination*, which is a different scenario primitive and one nothing
    here asks for yet; using it to walk straight would mean inventing a
    destination far enough away to look like "straight", and the walker would
    then route around obstacles the scenario put there on purpose.
    """

    def __init__(self, config: PedestrianEntityConfig) -> None:
        self._config = config
        self._walker: Optional["carla.Actor"] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def role_name(self) -> Union[EntityRole, str]:
        """Return the role name that identifies this entity."""
        return self._config.role_name

    @property
    def walker_type(self) -> str:
        """Return the CARLA blueprint id for this pedestrian."""
        return self._config.walker_type

    @property
    def actor(self) -> Optional["carla.Actor"]:
        """Return the underlying CARLA actor, or ``None`` if not spawned."""
        return self._walker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def spawn(self, world: "carla.World") -> "carla.Actor":
        """Spawn the pedestrian in the CARLA world.

        Raises:
            ValueError: If the blueprint is not available.
            RuntimeError: If CARLA refused the spawn point.
        """
        blueprint_library = world.get_blueprint_library()
        try:
            blueprint = blueprint_library.find(self._config.walker_type)
        except (IndexError, RuntimeError) as exc:
            raise ValueError(
                f"pedestrian blueprint {self._config.walker_type!r} is not "
                f"available in this CARLA build"
            ) from exc

        # Not guarded on the attribute existing.  Every entity-based condition
        # -- existence, speed, distance, TTC, waypoint -- finds its actor by
        # `role_name`, so a walker spawned without one is live in the world and
        # invisible to the scenario: each of those conditions would read it as
        # absent and never evaluate.  A blueprint that cannot carry the name is
        # therefore unusable here, and saying so beats spawning a pedestrian
        # nothing can assert about.
        if not blueprint.has_attribute("role_name"):
            raise ValueError(
                f"pedestrian blueprint {self._config.walker_type!r} has no "
                f"role_name attribute, so no condition could find it; pick a "
                f"walker blueprint that has one"
            )
        blueprint.set_attribute("role_name", str(self._config.role_name))
        # A walker that another actor can push is a walker the scenario no
        # longer controls; every pedestrian here is scripted.
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")

        transform = self._lifted_transform()
        walker = world.try_spawn_actor(blueprint, transform)
        if walker is None:
            raise RuntimeError(
                f"could not spawn pedestrian {self._config.role_name!r} at "
                f"{transform.location}"
            )
        self._walker = walker
        logger.info(
            "Spawned pedestrian '%s' (%s) at %s",
            self._config.role_name,
            self._config.walker_type,
            transform.location,
        )
        return walker

    def destroy(self) -> None:
        """Destroy the pedestrian actor and release resources."""
        if self._walker is not None:
            self._walker.destroy()
            self._walker = None

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def walk_straight(self, speed_ms: float) -> None:
        """Walk forward at *speed_ms*, holding the current heading.

        Args:
            speed_ms: Walking speed in metres per second.  Zero stops the
                walker where it stands.
        """
        if self._walker is None:
            logger.warning(
                "PedestrianEntity '%s': not spawned, cannot walk",
                self._config.role_name,
            )
            return
        forward = self._walker.get_transform().get_forward_vector()
        self._walker.apply_control(
            carla.WalkerControl(direction=forward, speed=speed_ms)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lifted_transform(self) -> "carla.Transform":
        """Return the spawn transform raised clear of the ground."""
        transform = self._config.spawn_location.value
        return carla.Transform(
            carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + _SPAWN_LIFT_M,
            ),
            transform.rotation,
        )
