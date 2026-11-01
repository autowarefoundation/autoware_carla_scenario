"""Condition that holds while a road user is carrying out an adversity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from ..adversity.triggers import AdversityKind
from ..adversity_activity import ActiveAdversity, active_adversities
from .base import BaseCondition, ScenarioResult
from .entity_filter import EntityFilter

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["AdversityActiveCondition", "adversity_kinds_from_field"]


class AdversityActiveCondition(BaseCondition):
    """Holds while a road user that *entities* takes is carrying out an
    adversity of one of *kinds* -- through its own adversarial action, or
    because NADE sampled it, a background vehicle as much as a named NPC.

    "A cut-in within 30 m of the ego" is
    ``AdversityActiveCondition(EntityFilter(max_distance_m=30.0),
    [AdversityKind.CUT_IN])``; with no filter and no kinds, any adversity
    anywhere.  It holds for as long as the road user drives the adversity's
    path, until it is handed back to traffic: under a ``StickyCondition``,
    it says the adversity happened.

    Args:
        entities: Which road users count (:class:`~.entity_filter.EntityFilter`);
            ``None``: every one.
        kinds: The adversities that count; empty: every kind.
        include_responses: Count a victim's response to another's adversity
            too (NADE's avoid or accept), as the adversity it responds to.
        label: Human-readable identifier.
    """

    # Declared for the static check (docs/typecheck.md).
    _entities: EntityFilter
    _kinds: list[str]
    _include_responses: bool
    _active: list[ActiveAdversity]

    def __init__(
        self,
        entities: Optional[EntityFilter] = None,
        kinds: Optional[Sequence[AdversityKind]] = None,
        include_responses: bool = False,
        *,
        label: str,
    ) -> None:
        super().__init__(label=label)
        chosen = EntityFilter()
        if entities is not None:
            given: EntityFilter = entities
            given.validate()
            chosen = given
        self._entities = chosen
        values: list[str] = []
        if kinds is not None:
            for kind in kinds:
                values.append(kind.value)
        self._kinds = values
        self._include_responses = include_responses
        self._active = []

    @property
    def entities(self) -> EntityFilter:
        """Which road users count."""
        return self._entities

    @property
    def active(self) -> list[ActiveAdversity]:
        """The adversities the last check found under way, of the road users
        it takes."""
        return list(self._active)

    def get_details(self) -> dict[str, Any]:
        return {
            "entities": self._entities.describe(),
            "kinds": list(self._kinds) if self._kinds else "any",
            "include_responses": self._include_responses,
            "active": [f"{a.agent}: {a.kind}" for a in self._active],
        }

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Hold while one of the road users taken carries out a kind counted."""
        under_way: list[ActiveAdversity] = []
        for adversity in active_adversities():
            if adversity.responding_to and not self._include_responses:
                continue
            if self._kinds and adversity.kind not in self._kinds:
                continue
            under_way.append(adversity)
        if not under_way:
            self._active = []
            return None
        taken: set[str] = set()
        for entity in self._entities.select(world):
            taken.add(entity.name)
        found: list[ActiveAdversity] = []
        for adversity in under_way:
            if adversity.agent in taken:
                found.append(adversity)
        self._active = found
        if not found:
            return None
        said: list[str] = []
        for adversity in found:
            what = adversity.kind.replace("_", " ")
            if adversity.responding_to:
                what = f"responding to {adversity.responding_to}'s {what}"
            said.append(f"'{adversity.agent}' {what} since {adversity.since:.2f}s")
        return ScenarioResult(
            passed=True,
            message=f"{'; '.join(said)} ({self._entities.describe()})",
            elapsed_seconds=elapsed,
        )


def adversity_kinds_from_field(kind: str = "ANY") -> list[AdversityKind]:
    """The kinds a scenario document's field names: one ``AdversityKind`` by
    name, or ``ANY`` (none listed: every kind).

    Raises:
        ValueError: An unknown kind.
    """
    wanted = str(kind).upper()
    if wanted == "ANY":
        return []
    for member in AdversityKind:
        if member.name == wanted:
            return [member]
    raise ValueError(f"no adversity kind {kind!r}")
