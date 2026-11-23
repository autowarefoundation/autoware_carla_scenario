"""The id a new entity gets, moved from editor.service into the checked
editor.naming (#2364).

``_old_unique_entity_id`` is editor.service._unique_entity_id() before
#2364, verbatim but for the name and the ids in use, which it read off the
document and now takes.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

from autoware_carla_scenario.editor import naming, service


def _old_unique_entity_id(document: Any, stem: str) -> str:
    taken = {e.id for e in document.entities}
    if stem == "ego":
        if stem not in taken:
            return stem
    index = 1
    while f"{stem}{index}" in taken:
        index += 1
    return f"{stem}{index}"


_IDS = ("ego", "ego1", "ego2", "npc1", "npc2", "npc3", "ped1", "ped3", "npc", "x")
_STEMS = ("ego", "npc", "ped", "bike", "")


def test_the_entity_id_is_as_before() -> None:
    rng = random.Random(2364)
    for _ in range(5000):
        ids = rng.sample(_IDS, rng.randrange(0, len(_IDS) + 1))
        document = SimpleNamespace(entities=[SimpleNamespace(id=i) for i in ids])
        stem = rng.choice(_STEMS)
        expected = _old_unique_entity_id(document, stem)
        assert naming.unique_entity_id(set(ids), stem) == expected
        assert service._unique_entity_id(document, stem) == expected  # type: ignore[arg-type]


def test_the_ego_is_unnumbered_until_taken() -> None:
    assert naming.unique_entity_id(set(), "ego") == "ego"
    assert naming.unique_entity_id({"ego"}, "ego") == "ego1"
    assert naming.unique_entity_id(set(), "npc") == "npc1"
    assert naming.unique_entity_id({"npc1", "npc3"}, "npc") == "npc2"
