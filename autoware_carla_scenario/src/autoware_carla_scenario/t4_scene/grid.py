"""The pure pieces of a T4Scene's time grid: the step in ticks, the scene's
stem, who the ego is, who is on the scene and the ego's turn indicator.

t4_scene.convert (the offline converter), t4_scene.live (the scenes written
as a run goes) and the ``scenario-t4scene`` command share these, so they
agree on what a step, a stem and a row mean.  Nothing here reads a log or
builds a message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..constants import EGO_ROLES, tick_multiple
from ..rerun import TRAJECTORY_SUFFIX

__all__ = ["indicator_at", "is_ego", "on_scene", "scene_stem", "step_frames"]


def step_frames(step_s: float) -> int:
    """The ticks in one step of *step_s* seconds.

    Raises:
        ValueError: If *step_s* is not a positive multiple of the tick.
    """
    return tick_multiple(step_s, "step_s")


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
