"""The live entities of a run, addressable by role name.

Actions name the thing they act on rather than holding it:
``LaneChangeAction("Ego", ...)`` says *who*, and finds the CARLA actor with
:func:`~autoware_carla_scenario.conditions.base.find_actor_by_role_name` when it
runs.  The world is the registry there -- every actor carries its ``role_name``
as an attribute -- so nothing else was needed.

That stops working the moment an action needs the *entity* rather than its
actor.  A route is not something you can apply to an actor: an
:class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`
delivers it over the bridge it owns, and the CARLA world knows nothing about
that.  This module is the missing half -- the same lookup, for entities -- so
that an action which calls a method on an entity can still name it rather than
be handed it.

The registry is per-run.  :class:`~autoware_carla_scenario.ScenarioRunner`
clears it before each scenario, because a batch runs several against one world
and an entity from the previous scenario answering to ``"npc1"`` would be worse
than no entity at all.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from ..entity_role import EntityRole

logger = logging.getLogger(__name__)

#: Role name -> the live entity answering to it, for the current scenario.
_ENTITIES: dict[str, Any] = {}


def register_entity(role_name: Union[EntityRole, str], entity: Any) -> None:
    """Make *entity* findable by its role name.

    Re-registering a role replaces it, which is what a scenario that respawns
    an entity means.

    Args:
        role_name: The role the entity answers to.
        entity: The entity object.
    """
    _ENTITIES[str(role_name)] = entity


def unregister_entity(role_name: Union[EntityRole, str]) -> None:
    """Forget the entity registered under *role_name*, if any."""
    _ENTITIES.pop(str(role_name), None)


def find_entity_by_role_name(role_name: Union[EntityRole, str]) -> Optional[Any]:
    """Return the live entity answering to *role_name*, or ``None``.

    The entity counterpart of
    :func:`~autoware_carla_scenario.conditions.base.find_actor_by_role_name`:
    that one reaches the actor a role names, this one reaches the object
    driving it.

    Args:
        role_name: The role to look up.  Accepts :class:`EntityRole` and
            plain ``str``.

    Returns:
        The entity, or ``None`` when no entity has that role in this run.
    """
    return _ENTITIES.get(str(role_name))


def clear_entities() -> None:
    """Drop every registration.  Called by the runner before each scenario."""
    _ENTITIES.clear()
