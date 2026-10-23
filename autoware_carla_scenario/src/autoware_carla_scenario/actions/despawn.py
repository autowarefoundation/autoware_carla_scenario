"""Take a road user out of the world for good.

A road user transcribed from a recording leaves the scene somewhere -- turns
off, drives out of the sensors' range -- and an NPC whose route has ended
would otherwise drive on wherever its traffic backend takes it.
:class:`DespawnAction` removes it: typically triggered when the entity's own
:class:`~.routing.RoutingAction` completes (an ``action_state`` condition on
it), so the NPC leaves where its route ends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..conditions.base import BaseCondition
from ..constants import EGO_ROLE_NAME
from ..entity.registry import (
    find_pedestrian_entity,
    find_vehicle_entity,
    unregister_entity,
)
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["DespawnAction", "despawn_entity"]


class DespawnAction(BaseAction):
    """Remove a vehicle or a pedestrian from the world (not the ego).

    When it fires the entity is handed back by its traffic backend (the
    TrafficManager or SUMO stops driving it and forgets it), the collision
    log stops listening to it, its actor is destroyed and the entity is
    unregistered, so every later lookup of its role finds nothing: an action
    still naming it logs that once and does nothing, a condition on it reads
    it as absent.  Instantaneous.

    Args:
        entity_name: ``role_name`` of the vehicle or pedestrian.
        condition: When it leaves; ``None`` at once.
        timing: Tick phase.
        label: Human-readable identifier.
        once: Whether it fires at most once (the default; a second run finds
            nothing to remove).

    Raises:
        ValueError: For the ego, which a run is measured on.
    """

    _entity_name: str
    _despawned: list[tuple[str, int]]

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "despawn",
        once: bool = True,
    ) -> None:
        if str(entity_name) == str(EGO_ROLE_NAME):
            raise ValueError(f"{label}: the ego cannot be despawned")
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = str(entity_name)
        self._despawned = []

    @property
    def entity_name(self) -> str:
        """``role_name`` of the entity removed."""
        return self._entity_name

    def take_despawns(self) -> list[tuple[str, int]]:
        """The ``(role, actor id)`` removed since the last call, for the log."""
        out = self._despawned
        self._despawned = []
        return out

    def execute(self, world: "carla.World") -> None:
        """Hand the entity back, stop hearing it, destroy it, forget it."""
        actor_id = despawn_entity(world, self._entity_name, self.label)
        if actor_id is not None:
            gone: int = actor_id
            self._despawned.append((self._entity_name, gone))


def despawn_entity(world: "carla.World", name: str, label: str) -> Optional[int]:
    """Take the vehicle or pedestrian *name* out of the world: its traffic
    backend lets go of it, the collision log stops hearing it, its actor is
    destroyed and the entity unregistered.  Its actor's id (``-1`` if it had
    none), or ``None`` when no such entity is in the world (logged)."""
    actor: Optional["carla.Actor"] = None
    vehicle = find_vehicle_entity(name)
    if vehicle is not None:
        if hasattr(vehicle, "actor"):
            actor = vehicle.actor
        vehicle.release_from_traffic(world)
        _forget(actor)
        # BackendDriven, the mixin, declares no destroy(); the entities it
        # is mixed into do (docs/typecheck.md).
        if hasattr(vehicle, "destroy"):
            try:
                vehicle.destroy()
            except RuntimeError:
                _already_gone(label, name)
    else:
        pedestrian = find_pedestrian_entity(name)
        if pedestrian is None:
            logger.warning("%s: '%s' is not in the world", label, name)
            return None
        actor = pedestrian.actor
        pedestrian.release_from_traffic(world)
        _forget(actor)
        try:
            pedestrian.destroy()
        except RuntimeError:
            _already_gone(label, name)
    unregister_entity(name)
    actor_id = -1
    if actor is not None:
        gone: "carla.Actor" = actor
        actor_id = int(gone.id)
    logger.info("%s: '%s' despawned", label, name)
    return actor_id


def _already_gone(label: str, name: str) -> None:
    """Its actor was destroyed before (by CARLA, a backend): nothing to do."""
    logger.info("%s: '%s' was already gone from the world", label, name)


def _forget(actor: Optional["carla.Actor"]) -> None:
    """Have the collision log let go of *actor* before it is destroyed."""
    if actor is None:
        return
    found: "carla.Actor" = actor
    if TYPE_CHECKING:
        # The collision log is not compiled (npc_collisions, EXCLUDED).
        pass
    else:
        _release_collision_sensor(int(found.id))


def _release_collision_sensor(actor_id: int) -> None:
    """Stop the run's collision log listening to *actor_id*."""
    from ..npc_collisions import release_actor  # noqa: PLC0415

    release_actor(actor_id)
