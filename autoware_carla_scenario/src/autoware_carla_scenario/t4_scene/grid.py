"""The pure pieces of a T4Scene's time grid: the step in ticks, the scene's
stem, who the ego is, who is on the scene, the ego's turn indicator and how
far from it a road user is still one of the scene's objects.

t4_scene.convert (the offline converter), t4_scene.live (the scenes written
as a run goes) and the ``scenario-t4scene`` command share these, so they
agree on what a step, a stem and a row mean.  Nothing here reads a log or
builds a message.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from ..constants import EGO_ROLES, tick_multiple
from ..rerun import TRAJECTORY_SUFFIX

__all__ = [
    "checked_max_distance",
    "checked_step",
    "indicator_at",
    "is_ego",
    "on_scene",
    "scene_stem",
    "step_frames",
]


def step_frames(step_s: float) -> int:
    """The ticks in one step of *step_s* seconds.

    Raises:
        ValueError: If *step_s* is a bool or not a number
            (:func:`checked_step`), or not a positive multiple of the tick.
    """
    return tick_multiple(checked_step(step_s), "step_s")


def checked_step(step_s: float, key: str = "step_s") -> float:
    """*step_s* as a float, which must be a number and a positive multiple of
    the tick (:func:`~autoware_carla_scenario.constants.tick_multiple`).

    A bool is refused: Hydra reads ``step_s=true`` as one, and ``float()``
    would take it for 1 s, a valid step, so every scene would be written at
    1 s (#1502).  A string is refused too, rather than parsed, and anything
    else ``float()`` cannot read (``None``, a list) with an error naming *key*.

    Args:
        step_s: The step, seconds.
        key: The setting's name, which the error names (``t4scene.step_s``).

    Raises:
        ValueError: If *step_s* is a bool or not a number, or not a positive
            multiple of the tick.
    """
    error = ValueError(f"{key} must be a number of seconds, not {step_s!r}")
    if isinstance(step_s, bool) or isinstance(step_s, str):
        raise error
    try:
        value = float(step_s)
    except TypeError:
        raise error
    tick_multiple(value, key)
    return value


def checked_max_distance(
    max_distance_m: Optional[float],
    key: str = "max_distance_m",
    unlimited: str = "None",
) -> Optional[float]:
    """*max_distance_m* as a float, which must be positive and finite;
    ``None`` (every road user, however far) stays ``None``.

    0, a negative distance or NaN would keep no road user and infinity is
    what ``None`` says, so each is refused before a scene is written rather
    than leaving it silently empty.  A bool is refused too: ``true`` is not
    a distance, though ``float()`` would read it as 1 m, and so is a string
    or anything else that is not a number, with an error naming *key*.

    Args:
        max_distance_m: The distance, metres.
        key: The setting's name, which the error names (``t4scene.max_distance_m``).
        unlimited: What the error says takes every road user instead
            (``null``, ``--all-objects``).

    Raises:
        ValueError: If *max_distance_m* is a bool, not a number, or not
            positive and finite.
    """
    if max_distance_m is None:
        return None
    error = ValueError(
        f"{key} must be positive and finite ({unlimited} for no limit), "
        f"not {max_distance_m}"
    )
    if isinstance(max_distance_m, bool) or isinstance(max_distance_m, str):
        raise error
    try:
        value: float = float(max_distance_m)
    except TypeError:
        raise error
    if not (math.isfinite(value) and value > 0.0):
        raise error
    return value


def scene_stem(name: str) -> str:
    """The run's stem of a trajectory log's file *name*: the name without
    ``.trajectory.jsonl``, or without its last suffix for any other name."""
    if name.endswith(TRAJECTORY_SUFFIX):
        return name[: -len(TRAJECTORY_SUFFIX)]
    return Path(name).stem


def is_ego(role: str) -> bool:
    """Whether an actor of *role* is the ego (any case)."""
    return role.lower() in EGO_ROLES


def on_scene(spawned: Optional[int], despawned: Optional[int], frame: int) -> bool:
    """Whether a road user is on the scene at *frame*, not parked out of
    sight up to its ``SpawnAction``'s frame *spawned* (whose tick may have
    been logged before the action moved it) or from its ``DespawnAction``'s,
    *despawned*; ``None``: it has no such action."""
    if spawned is not None:
        first: int = spawned
        if frame <= first:
            return False
    if despawned is not None:
        last: int = despawned
        if frame >= last:
            return False
    return True


def indicator_at(frames: list[int], states: list[str], frame: int) -> Optional[str]:
    """The turn indicator's state at *frame*: the state of the last change
    at or before it, the changes being on *frames* (in order) with
    *states*; ``None`` before the first."""
    low = 0
    high = len(frames)
    while low < high:
        middle = (low + high) // 2
        if frame < frames[middle]:
            high = middle
        else:
            low = middle + 1
    if low == 0:
        return None
    return states[low - 1]
