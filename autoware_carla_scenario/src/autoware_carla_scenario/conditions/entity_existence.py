"""Fail condition that triggers when a named actor does not exist."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from ..coordinate.poses import CarlaWorldPose
from ..entity_role import EntityRole
from .base import BaseCondition, ScenarioResult, find_actor_by_role_name
from ..failure import FailureCategory

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

#: An actor this far down (m) is parked under the map, out of the scene.
HIDDEN_BELOW_M = -100.0


def find_present_actor_by_role_name(
    world: "carla.World", role_name: Union[EntityRole, str]
) -> Optional["carla.Actor"]:
    """Find the actor with *role_name*, or ``None`` when it is not in the scene.

    Like :func:`~.base.find_actor_by_role_name`, but an actor parked under the
    map (below :data:`HIDDEN_BELOW_M`: spawned out of sight and not yet brought
    in, or taken out again) counts as absent.  What a condition measures
    against such an actor -- a gap, a closing speed, a heading -- is measured
    against a point under the road, not against anything a driver could see.
    The guard of a composition condition reads its own entity this way
    (:class:`EntityExistenceCondition` with ``quiet=True``); a condition that
    measures against a second entity (a target, a reference) reads that one
    this way too, so swapping the two gives the same answer.

    A trigger meant as "the ego is near where a hidden entity will appear"
    therefore never fires on that entity: name the place instead, with a
    position condition
    (:class:`~.composition.entity_position_distance.EntityPositionDistanceCondition`).
    """
    actor = find_actor_by_role_name(world, role_name)
    if actor is None or float(actor.get_location().z) < HIDDEN_BELOW_M:
        return None
    return actor


class ReferenceAbsentError(ValueError):
    """The entity a relative place is written against is not in the scene
    (:meth:`ReferencePoses.pose`): a place that waits for it, rather than one
    that cannot be placed at all."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ReferencePoses:
    """Where the entities relative places are written against are now, in
    *world*: the ``reference`` of
    :func:`~autoware_carla_scenario.trajectory.resolve.resolve_position`.

    A :class:`~autoware_carla_scenario.trajectory.model.RelativeLanePose` is
    placed against its ``entity_ref``, and a route pose without an anchor
    against the ego (:func:`~autoware_carla_scenario.trajectory.model.placed_against_entity`).

    Args:
        world: The CARLA world the entities are looked up in.
        present_only: Count an actor parked under the map (spawned hidden, not
            yet brought in) as absent, as a condition reads a second entity
            (:func:`find_present_actor_by_role_name`).  Off, the actor is
            read wherever it is, as :class:`~autoware_carla_scenario.actions.SpawnAction`
            reads it.
    """

    # Declared for the static check (docs/typecheck.md).
    _world: "carla.World"
    _present_only: bool

    def __init__(self, world: "carla.World", present_only: bool = False) -> None:
        self._world = world
        self._present_only = present_only

    def pose(self, entity_ref: Optional[Union[EntityRole, str]]) -> CarlaWorldPose:
        """Where *entity_ref* is now.

        Raises:
            ValueError: For no entity (``None``: a place relative to "the
                entity the action moves", which a lookup by name has not got).
            ReferenceAbsentError: When the entity is not in the scene.
        """
        if entity_ref is None:
            raise ValueError("a place relative to no entity")
        name = str(entity_ref)
        if self._present_only:
            actor = find_present_actor_by_role_name(self._world, name)
        else:
            actor = find_actor_by_role_name(self._world, name)
        if actor is None:
            raise ReferenceAbsentError(f"'{entity_ref}' is not in the world")
        found: "carla.Actor" = actor
        return CarlaWorldPose.from_carla_transform(found.get_transform())


class EntityExistenceCondition(BaseCondition):
    """Fail condition that triggers whenever a named actor is absent from the world.

    On every call to :meth:`check`, the condition searches for an actor with
    the given ``role_name``.  If the actor is **not found** — whether it has
    never spawned or has since been destroyed — the condition returns a failure
    result immediately.

    This condition is typically registered as a **fail condition** so that the
    scenario terminates immediately when the ego (or any critical actor)
    is missing.

    Args:
        entity_name: The ``role_name`` attribute of the actor to monitor.
            Accepts both :class:`EntityRole` and plain ``str``.
        quiet: An absent actor is expected, not an error: a guard (a
            composition condition's on its entity, which waits until the
            entity is there and reads one that is gone as nothing to check)
            says so at debug level only, once.  For a guard an actor parked
            under the map (:data:`HIDDEN_BELOW_M`: spawned out of sight, not
            yet brought in) is not there either -- how far it is from the ego,
            or that it stands still, says nothing of the scene.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _quiet: bool
    _said_absent: bool

    def __init__(
        self, entity_name: Union[EntityRole, str], *, label: str, quiet: bool = False
    ) -> None:
        super().__init__(label=label)
        # Kept as its string, the one thing every use of it reads.
        self._entity_name = str(entity_name)
        self._quiet = quiet
        self._said_absent = False

    def reset(self) -> None:
        """Say an absence again in the next run (:meth:`BaseCondition.reset`)."""
        super().reset()
        self._said_absent = False

    def failure_category(self) -> Optional[FailureCategory]:
        """An entity that vanished: :attr:`~FailureCategory.INFRASTRUCTURE`."""
        return FailureCategory.INFRASTRUCTURE

    def get_details(self) -> dict[str, Any]:
        return {"entity_name": str(self._entity_name)}

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a fail result if the actor does not exist.

        The condition triggers immediately whenever the actor cannot be found,
        including before it has ever been spawned.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            :class:`ScenarioResult` with ``passed=False`` if the actor
            is not found, ``None`` otherwise.
        """
        actor = (
            find_present_actor_by_role_name(world, self._entity_name)
            if self._quiet
            else find_actor_by_role_name(world, self._entity_name)
        )

        if actor is not None:
            self._said_absent = False
            return None

        msg = (
            f"Actor '{self._entity_name}' not found in the world " f"at {elapsed:.2f}s"
        )
        if not self._quiet:
            logger.error(msg)
        elif not self._said_absent:
            self._said_absent = True
            logger.debug(msg)

        return ScenarioResult(
            passed=False,
            message=msg,
            elapsed_seconds=elapsed,
        )
