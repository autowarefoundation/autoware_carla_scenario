"""What the framework's actor lookups need of a fake CARLA world."""

from __future__ import annotations

import fnmatch
from typing import Any, Optional


class ActorsById:
    """Mixin for a fake world whose ``get_actors()`` lists its live actors.

    ``find_actor_in_world()`` keys what it found by ``world.id`` and looks it
    up again with ``world.get_actor()``. A fake world's actors are all alive,
    so a fake's actors need ``is_alive`` too.
    """

    @property
    def id(self) -> int:
        return id(self)

    def get_actor(self, actor_id: int) -> Optional[Any]:
        return next((a for a in self.get_actors() if a.id == actor_id), None)  # type: ignore[attr-defined]


class FakeActorList(list):
    """A list with ``ActorList.filter()``: a wildcard on ``type_id``."""

    def filter(self, pattern: str) -> "FakeActorList":
        return FakeActorList(a for a in self if fnmatch.fnmatchcase(a.type_id, pattern))
