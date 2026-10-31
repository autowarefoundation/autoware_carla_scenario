"""A walker's velocity, measured from where it stands.

CARLA's walker controller (``controller.ai.walker``) moves its walker by
placing it, not by driving it, so ``get_velocity()`` reads 0 for a walker
walking on its controller.  :class:`WalkerVelocities` measures a walker's
velocity instead from where it stood when last asked and how much time has
passed since: everything that reads how fast a walker goes -- the speed
conditions, the pedestrian measures, a walker watching for a threat -- reads
it through one, and every other actor keeps what CARLA reports.

Each reader keeps its own :class:`WalkerVelocities` (a condition its own, the
measures theirs), so what one asks does not change what another measures,
and clears it when its run ends.  A clock that has gone back -- a new run --
starts over too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .vector import Vector3

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["WalkerVelocities", "is_walker", "simulated_seconds"]


def is_walker(actor: "carla.Actor") -> bool:
    """Whether *actor* is a walker (``walker.*``)."""
    return str(actor.type_id).startswith("walker.")


def simulated_seconds(world: "carla.World") -> Optional[float]:
    """*world*'s simulated time (s), or ``None`` where it does not say."""
    try:
        return float(world.get_snapshot().timestamp.elapsed_seconds)
    except Exception:
        return None


class WalkerVelocities:
    """Walkers' velocities, each measured from where it stood when last
    asked.

    The first time a walker is asked about, and again once the clock has gone
    back (a new run), it reads what CARLA reports; asked again at the same
    time, the same as before.
    """

    #: Each walker's last place and the velocity measured to it, by actor
    #: id: ``(elapsed, x, y, z, vx, vy, vz)``.
    _placed: dict[int, tuple[float, float, float, float, float, float, float]]

    def __init__(self) -> None:
        self._placed = {}

    def clear(self) -> None:
        """Forget every walker (a run ended)."""
        self._placed.clear()

    def forget(self, actor_id: int) -> None:
        """Forget walker *actor_id* (it is gone)."""
        self._placed.pop(int(actor_id), None)

    def walker(
        self,
        walker: "carla.Actor",
        elapsed: Optional[float],
        location: Optional["carla.Location"] = None,
    ) -> Vector3:
        """*walker*'s velocity, measured from where it stood when last asked
        to where it stands now (*location*, else asked), *elapsed* the time
        now.  Without a time, what CARLA reports."""
        reported = Vector3.from_carla_vector3d(walker.get_velocity())
        if elapsed is None:
            return reported
        now: float = elapsed
        if location is None:
            location = walker.get_location()
        key = int(walker.id)
        velocity = reported
        if key in self._placed:
            when, x, y, z, vx, vy, vz = self._placed[key]
            span = now - when
            if span == 0.0:
                return Vector3(vx, vy, vz)  # asked again at the same time
            if span > 0.0:
                velocity = Vector3(
                    (location.x - x) / span,
                    (location.y - y) / span,
                    (location.z - z) / span,
                )
        self._placed[key] = (
            now,
            float(location.x),
            float(location.y),
            float(location.z),
            velocity.x,
            velocity.y,
            velocity.z,
        )
        return velocity

    def velocity(self, actor: "carla.Actor", elapsed: Optional[float]) -> Vector3:
        """*actor*'s velocity: a walker's measured (:meth:`walker`), any
        other actor's what CARLA reports."""
        if is_walker(actor):
            return self.walker(actor, elapsed)
        return Vector3.from_carla_vector3d(actor.get_velocity())
