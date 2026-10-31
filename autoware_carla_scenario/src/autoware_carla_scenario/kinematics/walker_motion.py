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
starts over too, and so does a walker teleported (put where its walk starts,
brought in from under the map): that is no walk to measure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .vector import Vector3

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "HIDDEN_BELOW_M",
    "TELEPORT_MPS",
    "WalkerVelocities",
    "is_walker",
    "simulated_seconds",
]


def is_walker(actor: "carla.Actor") -> bool:
    """Whether *actor* is a walker (``walker.*``)."""
    return str(actor.type_id).startswith("walker.")


def simulated_seconds(world: "carla.World") -> Optional[float]:
    """*world*'s simulated time (s), or ``None`` where it does not say."""
    try:
        return float(world.get_snapshot().timestamp.elapsed_seconds)
    except Exception:
        return None


#: A walker this far down (m) is out of the world (parked under the map, not
#: yet brought in): a move from or to there is a teleport, not a walk.
HIDDEN_BELOW_M: float = -100.0
#: Faster than this (m/s) no walker walks: a move that fast between two reads
#: is a teleport (put where its walk starts, brought in), not a walk.
TELEPORT_MPS: float = 15.0
#: Slower than this (m/s) a walker reports standing (see :meth:`velocity`).
_STILL_MPS: float = 1e-6


class WalkerVelocities:
    """Walkers' velocities, each measured from where it stood when last
    asked.

    The first time a walker is asked about, again once the clock has gone
    back (a new run), and where it has been teleported since (moved faster
    than :data:`TELEPORT_MPS`, or out of the world below
    :data:`HIDDEN_BELOW_M`), there is nothing to measure from: it reads what
    CARLA reports.  Asked again at the same time, the same as before.
    """

    #: Each walker's last place and the velocity read there, by actor id:
    #: ``(elapsed, x, y, z, vx, vy, vz, measured)``, *measured* whether the
    #: velocity was measured rather than reported.
    _placed: dict[int, tuple[float, float, float, float, float, float, float, bool]]

    def __init__(self) -> None:
        self._placed = {}

    def clear(self) -> None:
        """Forget every walker (a run ended)."""
        self._placed.clear()

    def forget(self, actor_id: int) -> None:
        """Forget walker *actor_id* (it is gone)."""
        self._placed.pop(int(actor_id), None)

    def _read(
        self,
        walker: "carla.Actor",
        elapsed: Optional[float],
        location: Optional["carla.Location"],
    ) -> tuple[Vector3, bool]:
        """*walker*'s velocity, and whether it was measured (else reported)."""
        reported = Vector3.from_carla_vector3d(walker.get_velocity())
        if elapsed is None:
            return reported, False
        now: float = elapsed
        if location is None:
            location = walker.get_location()
        key = int(walker.id)
        velocity = reported
        measured = False
        if key in self._placed:
            when, x, y, z, vx, vy, vz, was = self._placed[key]
            span = now - when
            if span == 0.0:
                # Asked again at the same time.
                return Vector3(vx, vy, vz), was
            if (
                span > 0.0
                and z >= HIDDEN_BELOW_M
                and float(location.z) >= HIDDEN_BELOW_M
            ):
                moved = Vector3(
                    (location.x - x) / span,
                    (location.y - y) / span,
                    (location.z - z) / span,
                )
                if moved.magnitude() <= TELEPORT_MPS:
                    velocity = moved
                    measured = True
        self._placed[key] = (
            now,
            float(location.x),
            float(location.y),
            float(location.z),
            velocity.x,
            velocity.y,
            velocity.z,
            measured,
        )
        return velocity, measured

    def walker(
        self,
        walker: "carla.Actor",
        elapsed: Optional[float],
        location: Optional["carla.Location"] = None,
    ) -> Vector3:
        """*walker*'s velocity, measured from where it stood when last asked
        to where it stands now (*location*, else asked), *elapsed* the time
        now.  Without a time, or with nothing to measure from, what CARLA
        reports."""
        velocity, _measured = self._read(walker, elapsed, location)
        return velocity

    def velocity(
        self, actor: "carla.Actor", elapsed: Optional[float]
    ) -> Optional[Vector3]:
        """*actor*'s velocity: a walker's measured (:meth:`walker`), any
        other actor's what CARLA reports.

        ``None`` -- not known yet -- for a walker with nothing to measure from
        (:meth:`walker`) that reports standing: a walker on CARLA's controller
        reports that walking too, so a condition read first while it walks
        would otherwise see it stand.  It is known from the next read.
        """
        if not is_walker(actor):
            return Vector3.from_carla_vector3d(actor.get_velocity())
        velocity, measured = self._read(actor, elapsed, None)
        if not measured and elapsed is not None and velocity.magnitude() < _STILL_MPS:
            return None
        return velocity
