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
    """The ``role_name`` attribute of *actor*, or ``None`` if it has none."""
    # Not attributes.get(): Codon's dict.get() requires a default of the
    # value type, and no string means "absent".
    attributes = actor.attributes
    return attributes["role_name"] if "role_name" in attributes else None


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
    """
    name = str(role_name)
    return next(
        (a for a in actors if _role_name(a) == name),
        None,
    )
