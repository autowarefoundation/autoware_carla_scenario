"""The ids of a WOSAC rollout and the counts its writers take (docs/wosac.md).

A run's agents are WOSAC objects by role name (:func:`object_id_of`), and its
rollout belongs to the scenario its stem names (:func:`scenario_id_of`), so
the runs of one scenario are rollouts of one scenario whatever CARLA actor
ids they got.  :func:`checked_count` checks the counts :mod:`.convert` and
:mod:`.pack` take (rows, rollouts, shards, object ids).

Pure: it reads no log and builds no message, so :mod:`.pack` takes its counts
from here without importing the trajectory log's reader.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Optional

from ..entity_role import canonical_role_name
from ..seed import check_whole_number

__all__ = [
    "checked_count",
    "object_id_of",
    "scenario_id_of",
]

#: The run counter ``scenario_runner`` appends to a stem taken in the
#: directory already (``CutInScenario-1``).
_RUN_COUNTER = re.compile(r"-\d+$")
#: The largest WOSAC ``object_id`` (an int32).
_INT32_MAX = 2**31 - 1


def scenario_id_of(stem: str) -> str:
    """The scenario a run of *stem* is a rollout of, by default: the stem
    without the run counter a directory's later runs of one scenario get
    (``CutInScenario-2`` is a rollout of ``CutInScenario``)."""
    return _RUN_COUNTER.sub("", stem)


def object_id_of(role: str) -> int:
    """The ``object_id`` of an agent of *role*: the first 31 bits of the
    SHA-256 of its canonical role name (``ego`` and ``hero`` are ``Ego``).

    A run's actor ids are CARLA's, which differ from run to run; a role name
    is the scenario's, so the same entity has the same id in every rollout.
    """
    # The first 4 bytes of the digest, big-endian: its first 8 hex digits.
    digest = hashlib.sha256(canonical_role_name(role).encode()).hexdigest()
    return int(digest[:8], 16) & _INT32_MAX


def checked_count(
    value: int, key: str, minimum: int, maximum: Optional[int] = None
) -> int:
    """*value* as an int: a whole number (:func:`seed.check_whole_number`)
    from *minimum* to *maximum* (no limit when ``None``).

    Raises:
        ValueError: naming *key*.
    """
    number = check_whole_number(key, value)
    if number < minimum or (maximum is not None and number > maximum):
        bound = f">= {minimum}" if maximum is None else f"from {minimum} to {maximum}"
        raise ValueError(f"{key} must be a whole number {bound}, not {value!r}")
    return number


def _checked_object_ids(object_ids: Mapping[str, int]) -> dict[str, int]:
    """*object_ids* keyed by canonical role name, every id an int32 >= 0."""
    checked: dict[str, int] = {}
    for role, object_id in object_ids.items():
        checked[canonical_role_name(str(role))] = checked_count(
            object_id, f"the object id of {role!r}", 0, _INT32_MAX
        )
    return checked
