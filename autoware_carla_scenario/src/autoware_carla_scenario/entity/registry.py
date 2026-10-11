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

Entities are kept by kind, so that an action can look up the kind it acts on
with a type (docs/typecheck.md): :func:`find_vehicle_entity` returns what a
traffic backend drives, the ego included, and :func:`find_pedestrian_entity` a
pedestrian.  An object of neither kind -- a test double, a duck-typed entity of
a scenario package -- is kept as both, so every lookup returns it, as
:func:`find_entity_by_role_name` returns an entity of any kind.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from ..entity_role import EntityRole
from ..traffic.driven import BackendDriven
from .pedestrian_entity import PedestrianEntity

logger = logging.getLogger(__name__)

#: Role name -> the live entity answering to it, for the current scenario: the
#: vehicles (whatever a traffic backend drives, the ego included) ...
_VEHICLES: dict[str, BackendDriven] = {}
#: ... and the pedestrians.  An entity of neither kind is in both.
_PEDESTRIANS: dict[str, PedestrianEntity] = {}


def register_entity(role_name: Union[EntityRole, str], entity: Any) -> None:
    """Make *entity* findable by its role name.

    Re-registering a role replaces it, which is what a scenario that respawns
    an entity means.

    Args:
        role_name: The role the entity answers to.
        entity: The entity object.
    """
    key = str(role_name)
    unregister_entity(key)
    if isinstance(entity, BackendDriven):
        _VEHICLES[key] = entity
    elif isinstance(entity, PedestrianEntity):
        _PEDESTRIANS[key] = entity
    else:
        # Of neither kind: every lookup returns it, and it answers what it can.
        _VEHICLES[key] = entity
        _PEDESTRIANS[key] = entity


def unregister_entity(role_name: Union[EntityRole, str]) -> None:
    """Forget the entity registered under *role_name*, if any."""
    key = str(role_name)
    if key in _VEHICLES:
        del _VEHICLES[key]
    if key in _PEDESTRIANS:
        del _PEDESTRIANS[key]


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
    key = str(role_name)
    if key in _VEHICLES:
        return _VEHICLES[key]
    if key in _PEDESTRIANS:
        return _PEDESTRIANS[key]
    return None


def find_vehicle_entity(role_name: Union[EntityRole, str]) -> Optional[BackendDriven]:
    """Return the vehicle answering to *role_name*, or ``None``.

    A vehicle is what a traffic backend drives: an NPC vehicle or the ego.

    Args:
        role_name: The role to look up.

    Returns:
        The vehicle, or ``None`` when no entity has that role in this run or
        the one that has is a pedestrian.
    """
    key = str(role_name)
    if key in _VEHICLES:
        return _VEHICLES[key]
    return None


def find_pedestrian_entity(
    role_name: Union[EntityRole, str],
) -> Optional[PedestrianEntity]:
    """Return the pedestrian answering to *role_name*, or ``None``.

    Args:
        role_name: The role to look up.

    Returns:
        The pedestrian, or ``None`` when no entity has that role in this run
        or the one that has is a vehicle.
    """
    key = str(role_name)
    if key in _PEDESTRIANS:
        return _PEDESTRIANS[key]
    return None


def clear_entities() -> None:
    """Drop every registration.  Called by the runner before each scenario."""
    _VEHICLES.clear()
    _PEDESTRIANS.clear()
