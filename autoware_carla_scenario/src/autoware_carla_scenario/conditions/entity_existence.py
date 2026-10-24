"""Fail condition that triggers when a named actor does not exist."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from ..entity_role import EntityRole
from .base import BaseCondition, ScenarioResult, find_actor_by_role_name

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

#: An actor this far down (m) is parked under the map, out of the scene.
HIDDEN_BELOW_M = -100.0


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
        actor = find_actor_by_role_name(world, self._entity_name)

        if actor is not None and not (
            self._quiet and float(actor.get_location().z) < HIDDEN_BELOW_M
        ):
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
