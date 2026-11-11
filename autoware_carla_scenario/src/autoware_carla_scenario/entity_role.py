"""Validated entity role names for CARLA actors.

CARLA uses the ``role_name`` blueprint attribute to identify actors.  This
module provides a type-safe wrapper that validates role names at construction
time, preventing silent typos and enforcing a consistent naming convention.

Valid role names:
- ``"Ego"`` -- the ego vehicle (fixed string).
- ``<category><N>`` -- a lowercase alphabetic category followed by a positive
  integer, e.g. ``npc1``, ``vehicle2``, ``pedestrian3``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


_PATTERN = re.compile(r"^(Ego|[a-z]+[1-9]\d*)$")

#: The role names CARLA's native Autoware interface serves (CARLA 1.0,
#: ``ros2_autoware_control``): it drives only a vehicle registered under one of
#: them.  An actor with one of them, in any case, *is* the ego to the framework
#: -- :func:`canonical_role_name` reads it as ``"Ego"`` -- so an ego spawned
#: for that interface (#2038) is found by ``EGO_ROLE_NAME`` like any other.
#: No :class:`EntityRole` can be one of them: the pattern has no ``ego`` or
#: ``hero`` (an NPC's name ends in its number).
EGO_ROLE_ALIASES: tuple[str, str] = ("ego", "hero")

#: The role name an ego registered with CARLA's native Autoware interface is
#: spawned under.
NATIVE_AUTOWARE_EGO_ROLE_NAME: str = "ego"


def canonical_role_name(role_name: str) -> str:
    """*role_name* as the framework names it: ``"Ego"`` for the ego.

    The ego is ``"Ego"`` (``EGO_ROLE_NAME``) and, spawned for CARLA's native
    Autoware interface, ``ego`` or ``hero`` (:data:`EGO_ROLE_ALIASES`, matched
    in any case); every one of them reads as ``"Ego"``.  Any other role name
    is returned as it is.
    """
    lowered = role_name.lower()
    for alias in EGO_ROLE_ALIASES:
        if lowered == alias:
            return "Ego"
    return role_name


def _refuse_ego_role_name(role_name: str, what: str, allow_ego: bool = False) -> None:
    """Refuse *role_name* for *what* (``"an NPC vehicle"``, say) if
    :func:`canonical_role_name` reads it as the ego's.

    A name given as a plain string skips :class:`EntityRole`'s pattern, so
    an NPC could be named ``ego`` or ``hero`` (:data:`EGO_ROLE_ALIASES`, in
    any case) -- and then be taken for the ego by every lookup, condition,
    metric and the end-of-run sweep, with no warning (#2046).  With
    *allow_ego*, ``"Ego"`` itself passes, as :class:`EntityRole` lets it: a
    config the ego's own is built on (``VehicleEntityConfig``) names the ego
    that way; every alias is still refused.

    Raises:
        ValueError: naming *what* and the role name.
    """
    if canonical_role_name(role_name) != "Ego":
        return
    if allow_ego and role_name == "Ego":
        return
    raise ValueError(
        f"Invalid role name {role_name!r} for {what}: the framework reads "
        "'Ego', 'ego' and 'hero', in any case, as the ego. Name it "
        "<lowercase-category><positive-integer> (e.g. 'npc1', 'vehicle2')."
    )


class EntityRole:
    """Validated CARLA actor role name.

    The role name must match either the fixed string ``"Ego"`` or the pattern
    ``<category><N>`` where *category* is one or more lowercase ASCII letters
    and *N* is a positive integer (no leading zeros).

    Use the factory methods :meth:`ego` and :meth:`npc` for convenient,
    discoverable construction.

    Args:
        value: Raw role-name string to validate.

    Raises:
        ValueError: If *value* does not match the allowed pattern.
    """

    __slots__ = ("_value",)
    # Declared for the static check (docs/typecheck.md): Codon types an
    # attribute from its declaration; to Python it is a bare annotation.
    _value: str

    def __init__(self, value: str) -> None:
        if not _PATTERN.match(value):
            raise ValueError(
                f"Invalid entity role name {value!r}. "
                "Must be 'Ego' or <lowercase-category><positive-integer> "
                "(e.g. 'npc1', 'vehicle2')."
            )
        self._value = value

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def ego(cls) -> EntityRole:
        """Return the role for the ego vehicle (``"Ego"``)."""
        return cls("Ego")

    @classmethod
    def npc(cls, n: int) -> EntityRole:
        """Return a numbered NPC role (e.g. ``npc1``, ``npc2``).

        Args:
            n: Positive integer suffix.

        Raises:
            ValueError: If *n* is not a positive integer.
        """
        if n < 1:
            raise ValueError(f"NPC number must be a positive integer, got {n}")
        return cls(f"npc{n}")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return the raw role-name string for CARLA API compatibility."""
        return self._value

    def __repr__(self) -> str:
        return f"EntityRole({self._value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EntityRole):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


def _role_name(actor: "carla.Actor") -> Optional[str]:
    """The ``role_name`` attribute of *actor*, or ``None`` if it has none.

    As :func:`canonical_role_name` reads it: the ego's is ``"Ego"`` whether
    it was spawned as ``Ego`` or, for CARLA's native Autoware interface, as
    ``ego``.
    """
    # Not attributes.get(): Codon's dict.get() requires a default of the
    # value type, and no string means "absent".
    attributes = actor.attributes
    if "role_name" not in attributes:
        return None
    return canonical_role_name(attributes["role_name"])


def find_actor_in_list(
    actors: Union[carla.ActorList, list[carla.Actor]],
    role_name: Union["EntityRole", str],
) -> Optional["carla.Actor"]:
    """Find a single actor by its ``role_name`` in a pre-fetched actor list.

    Args:
        actors: Pre-fetched actors (e.g. ``world.get_actors()``, or a list).
        role_name: The ``role_name`` attribute value to search for.
            Accepts both :class:`EntityRole` and plain ``str``.

    Returns:
        The matching actor, or ``None`` if no actor with that role name exists.
        The ego is found by ``"Ego"`` under any of its role names
        (:func:`canonical_role_name`).
    """
    name = canonical_role_name(str(role_name))
    return next(
        (a for a in actors if _role_name(a) == name),
        None,
    )


#: The id of the actor last found for a role name, by ``(world id, role
#: name)``. ``world.id`` is CARLA's episode id, which a map load or a world
#: reload changes, so an entry never outlives the world it was found in.
_ACTOR_IDS: dict[tuple[int, str], int] = {}


def find_actor_in_world(
    world: "carla.World", role_name: Union["EntityRole", str]
) -> Optional["carla.Actor"]:
    """Find a single actor of *world* by its ``role_name``.

    The same answer as ``find_actor_in_list(world.get_actors(), role_name)``,
    without walking every actor on the map each time: the id found for a role
    name is remembered, and a later call looks that one actor up by id
    (``world.get_actor()``, served from the client's actor cache). The entry
    is used only while that actor is alive and still has this role name;
    otherwise the world is scanned again.

    Args:
        world: The CARLA world instance.
        role_name: The ``role_name`` attribute value to search for.
            Accepts both :class:`EntityRole` and plain ``str``.

    Returns:
        The matching actor, or ``None`` if no actor with that role name exists.
    """
    name = canonical_role_name(str(role_name))
    key = (world.id, name)
    if key in _ACTOR_IDS:
        # get_actor() can still return an actor destroyed in this episode
        # (with is_alive False): LibCarla clears its actor cache only when an
        # episode starts.
        cached = world.get_actor(_ACTOR_IDS[key])
        if cached is not None and cached.is_alive and _role_name(cached) == name:
            return cached
        del _ACTOR_IDS[key]
    actor = find_actor_in_list(world.get_actors(), name)
    if actor is not None:
        _ACTOR_IDS[key] = actor.id
    return actor
