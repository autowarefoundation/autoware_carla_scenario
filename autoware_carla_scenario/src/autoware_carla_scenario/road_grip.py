"""How well the road holds a tyre: the grip a run's weather leaves.

CARLA draws rain but never drives on it, so
:class:`~autoware_carla_scenario.EnvironmentAction` scales every vehicle's
wheel friction to the share of the dry road's grip the road's water leaves
(:func:`~autoware_carla_scenario.actions.environment.apply_tire_friction`).
That reaches the vehicles CARLA's physics moves, and nothing else: a traffic
simulator plans its vehicles' braking with what it was told, and a driver who
knows the road is wet brakes earlier.  So the grip is also recorded here, for
the world it was set in (:func:`set_world_grip`), as the visibility is
(:mod:`autoware_carla_scenario.visibility`), and a traffic backend reads it
every tick (:func:`world_grip`):

* the ``sumo`` backend gives every vehicle SUMO drives the acceleration and
  braking the grip leaves it (:func:`grip_braking`), so its traffic stops
  where a car on that road would;
* the stopping distance a driver keeps within what they see is worked out
  from that braking, by both backends that follow the visibility
  (``traffic.vehicles.follow_visibility``).

The runner forgets it at the start of every run (:func:`forget_grip`).

Like :mod:`autoware_carla_scenario.visibility`, this module imports neither
CARLA nor any traffic simulator.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "forget_grip",
    "grip_braking",
    "set_world_grip",
    "world_grip",
]

#: World (episode) id -> the share of the dry road's grip set in it.
_GRIP: dict[int, float] = {}


def set_world_grip(world_id: int, grip: float) -> None:
    """Record that tyres in the world *world_id* keep *grip* of the dry
    road's grip (1.0 is CARLA's own).

    Replaces what was set there before; another world's is dropped (an
    episode once left is not come back to).

    Raises:
        ValueError: If *grip* is not positive.
    """
    if not grip > 0.0:
        raise ValueError(f"grip must be positive, got {grip}")
    for old in [w for w in _GRIP if w != world_id]:
        del _GRIP[old]
    _GRIP[world_id] = float(grip)


def world_grip(world_id: Optional[int]) -> Optional[float]:
    """The grip set in the world *world_id*, or ``None`` where none was.

    *world_id* is what :func:`~autoware_carla_scenario.world_reset.episode_of`
    reads off a world: ``None``, a world that names none, has none set.
    """
    if world_id is None:
        return None
    if world_id in _GRIP:
        return _GRIP[world_id]
    return None


def grip_braking(value: float, grip: Optional[float]) -> float:
    """*value* -- an acceleration or a braking, m/s² -- as the road's *grip*
    leaves it.

    A road that grips less than a dry one (*grip* under 1) scales it down in
    proportion: the tyre's force, and with it the deceleration, is the
    friction's.  One that grips more leaves it as it is: a driver brakes as
    hard as they mean to, not harder because the road would allow it.
    ``None``, no grip set, leaves it as it is too.
    """
    if grip is None:
        return value
    return value * min(float(grip), 1.0)


def forget_grip() -> None:
    """Forget every world's grip; the runner calls this for every run."""
    _GRIP.clear()
