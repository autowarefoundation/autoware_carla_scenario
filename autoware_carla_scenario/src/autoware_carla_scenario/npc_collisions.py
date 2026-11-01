"""Log the collisions of the road users other than the ego, and walkers that fall.

The ego's collisions fail a run (:class:`~autoware_carla_scenario.conditions.CollisionCondition`);
everyone else's go unremarked, though a scenario whose NPCs crash into each
other, or knock a pedestrian over, is not the scenario that was written.
:class:`NpcCollisionMonitor` makes them visible without judging them: each is
logged as a warning and written to the trajectory log (``collision`` and
``walker_fallen`` records), and nothing is removed or put right.

**Collisions** are heard the way the ego's condition hears its own: a
``sensor.other.collision`` on each vehicle other than the ego, attached as
the vehicle is first seen (the world's actors are listed once a second, not
every tick).  A vehicle's sensor hears it hit another vehicle, a walker, or
the ego; walker-to-walker contact and contact with map geometry are not
recorded.  The two vehicles of one crash each report it, and a contact keeps
reporting while it lasts: one pair is recorded once per
:data:`REPEAT_S` seconds.

**A fallen walker** is told from one world snapshot every few ticks: a walker
tipped over more than :data:`FALLEN_TILT_DEG`, or more than half its height
below the lowest it stood over the last :data:`_STANDING_LOOKS` looks while
barely moving sideways (a ragdoll goes down; a walker going downhill or
teleported does not look like that, nor one pushed off its feet a moment that
lands where it stood).  It is
recorded once per walker.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from .constants import EGO_ROLE_NAME

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = [
    "FALLEN_TILT_DEG",
    "release_actor",
    "NpcCollision",
    "NpcCollisionMonitor",
    "REPEAT_S",
    "WalkerFallen",
]

#: How often (ticks) the world's actors are listed for new vehicles to hear.
SCAN_EVERY_TICKS = 20
#: How often (ticks) the walkers are looked at for a fall.
FALL_CHECK_EVERY_TICKS = 5
#: How long (s) without an event from a pair ends its contact.
REPEAT_S = 1.0
#: How long (s) a walker lies fallen before it is despawned (``None``: never).
FALLEN_DESPAWN_S: Optional[float] = 5.0
#: A walker tilted further than this (deg) has fallen.
FALLEN_TILT_DEG = 50.0
#: A walker this far below where it stood, or deeper, is out of the world
#: (spawned hidden, parked under the map), not fallen.
_HIDDEN_BELOW_M = 100.0
#: Looks a walker has to have been in the world for before a drop counts:
#: one that has just appeared settles onto the ground first.
_SETTLE_LOOKS = 8
#: Sideways (m) between two looks beyond which a drop is a teleport.
_TELEPORT_M = 2.0
#: Looks back a walker's standing height is the lowest of: a walker bumped
#: into the air comes down to where it stood, which is no fall.
_STANDING_LOOKS = 4


@dataclass(frozen=True)
class NpcCollision:
    """A collision of a vehicle other than the ego, as the log records it.

    Attributes:
        frame: The world frame it happened on.
        t: Scenario time (s) it was taken up at.
        actors: The ``role_name`` of each side (its type id when it has none).
        ids: The two actors' ids.
        impulse: The normal impulse's magnitude (N·s); at the end of a
            contact, the strongest during it.
        ego: Whether one side is the ego (whose own condition fails the run).
        phase: ``"start"`` when the contact began, ``"end"`` when it ended.
        duration: How long (s) the contact lasted, at its end; ``None`` at
            its start.
    """

    frame: int
    t: float
    actors: list[str]
    ids: list[int]
    impulse: float
    ego: bool
    phase: str = "start"
    duration: Optional[float] = None


class _Contact:
    """A contact between two actors under way: since when, and how hard."""

    frame: int
    start: float
    last: float
    names: list[str]
    ids: list[int]
    impulse: float

    def __init__(
        self, frame: int, start: float, names: list[str], ids: list[int], impulse: float
    ) -> None:
        self.frame = frame
        self.start = start
        self.last = start
        self.names = names
        self.ids = ids
        self.impulse = impulse

    def record(
        self,
        frame: int,
        t: float,
        phase: str,
        duration: Optional[float],
        ego_role: str,
    ) -> NpcCollision:
        return NpcCollision(
            frame=frame,
            t=t,
            actors=self.names,
            ids=self.ids,
            impulse=self.impulse,
            ego=ego_role in self.names,
            phase=phase,
            duration=duration,
        )


@dataclass(frozen=True)
class WalkerFallen:
    """A walker that fell over, as the log records it.

    Attributes:
        frame: The world frame it was seen down on.
        t: Scenario time (s).
        id: The walker's actor id.
        role: Its ``role_name`` (its type id when it has none).
        drop: How far (m) it dropped since the look before.
        tilt: How far (deg) it is tipped over.
    """

    frame: int
    t: float
    id: int
    role: str
    drop: float
    tilt: float


class _Sensed:
    """A vehicle heard: its sensor, and what to call it."""

    sensor: "carla.Actor"
    name: str

    def __init__(self, sensor: "carla.Actor", name: str) -> None:
        self.sensor = sensor
        self.name = name


def release_actor(actor_id: int) -> None:
    """Stop the run's monitor listening to *actor_id*, about to be destroyed
    (:class:`~autoware_carla_scenario.actions.despawn.DespawnAction`)."""
    if _ACTIVE is not None:
        _ACTIVE.release(actor_id)


class NpcCollisionMonitor:
    """Hears the collisions of every vehicle but the ego; sees walkers fall.

    Call :meth:`tick` once per world tick and :meth:`close` before the world
    goes, as the runner does for the conditions' sensors: a sensor is only
    stopped through the handle that started it listening.
    """

    _ego_role: str
    _ticks: int
    _sensed: dict[int, _Sensed]
    _names: dict[int, str]
    _lock: threading.Lock
    _heard: list[tuple[int, int, int, str, float]]
    _contacts: dict[tuple[int, int], _Contact]
    _down: dict[int, float]
    _despawn_fallen_after: Optional[float]
    _walkers: dict[int, tuple[float, float, float, float]]
    _heights: dict[int, list[float]]
    _settled: dict[int, int]
    _gone: set[int]
    _fallen: set[int]
    _broken: bool

    def __init__(
        self,
        ego_role: Optional[str] = None,
        despawn_fallen_after_s: Optional[float] = FALLEN_DESPAWN_S,
    ) -> None:
        self._ego_role = str(EGO_ROLE_NAME) if ego_role is None else ego_role
        self._ticks = 0
        self._sensed: dict[int, _Sensed] = {}
        self._names: dict[int, str] = {}
        self._lock = threading.Lock()
        self._heard: list[tuple[int, int, int, str, float]] = []
        self._contacts: dict[tuple[int, int], _Contact] = {}
        #: When each walker seen fallen went down, until it is despawned.
        self._down: dict[int, float] = {}
        self._despawn_fallen_after = despawn_fallen_after_s
        self._walkers: dict[int, tuple[float, float, float, float]] = {}
        #: Each walker's height (m) at its last looks, the latest last.
        self._heights: dict[int, list[float]] = {}
        #: How many looks in a row each walker has been in the world for.
        self._settled: dict[int, int] = {}
        #: Actors released before they are destroyed: not heard again.
        self._gone: set[int] = set()
        self._fallen: set[int] = set()
        self._broken = False
        global _ACTIVE
        _ACTIVE = self

    # ------------------------------------------------------------------

    def tick(
        self, world: "carla.World", elapsed: float
    ) -> tuple[list[NpcCollision], list[WalkerFallen], list[tuple[str, int]]]:
        """What happened since the last tick: contacts begun and ended, falls,
        and the fallen walkers despawned (role, actor id), logged."""
        if self._broken:
            return [], [], []
        try:
            return self._tick(world, elapsed)
        except Exception:  # noqa: BLE001 - a log aid never fails the run
            logger.warning(
                "NPC collision logging stopped after an error", exc_info=True
            )
            self._broken = True
            return [], [], []

    def release(self, actor_id: int) -> None:
        """Stop hearing *actor_id* and forget it: it is leaving the world."""
        sensed = self._sensed.pop(actor_id, None)
        if sensed is not None:
            _release(sensed.sensor)
        self._walkers.pop(actor_id, None)
        self._heights.pop(actor_id, None)
        self._gone.add(actor_id)

    def close(self) -> None:
        """Stop and destroy every sensor this attached, through its own handle."""
        global _ACTIVE
        for sensed in self._sensed.values():
            _release(sensed.sensor)
        self._sensed.clear()
        if _ACTIVE is self:
            _ACTIVE = None

    # ------------------------------------------------------------------

    def _tick(
        self, world: "carla.World", elapsed: float
    ) -> tuple[list[NpcCollision], list[WalkerFallen], list[tuple[str, int]]]:
        if self._ticks % SCAN_EVERY_TICKS == 0:
            self._scan(world)
        fallen: list[WalkerFallen] = []
        if self._ticks % FALL_CHECK_EVERY_TICKS == 0 and len(self._walkers) > 0:
            fallen = self._look_for_falls(world, elapsed)
            for fall in fallen:
                self._down[fall.id] = elapsed
        self._ticks += 1
        return self._collisions(elapsed), fallen, self._despawn_fallen(world, elapsed)

    def _despawn_fallen(
        self, world: "carla.World", elapsed: float
    ) -> list[tuple[str, int]]:
        """Despawn the walkers down :data:`FALLEN_DESPAWN_S` (never the ego)."""
        after = self._despawn_fallen_after
        if after is None or len(self._down) == 0:
            return []
        from .actions.despawn import despawn_entity  # noqa: PLC0415

        out: list[tuple[str, int]] = []
        for walker_id, since in list(self._down.items()):
            # Within a tolerance: the scenario clock is a running sum of the
            # tick, so a whole number of ticks comes out a hair short.
            if elapsed - since < after - 1e-9:
                continue
            del self._down[walker_id]
            role = self._names.get(walker_id, "")
            if not role or role == self._ego_role:
                continue
            gone = despawn_entity(world, role, "fallen walker")
            if gone is not None:
                out.append((role, int(gone)))
        return out

    def _scan(self, world: "carla.World") -> None:
        """Hear new vehicles, let go of gone ones, note the walkers."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        actors = world.get_actors()
        present: set[int] = set()
        blueprint: Optional["carla.ActorBlueprint"] = None
        for actor in actors:
            type_id = str(actor.type_id)
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
                continue
            if actor.id in self._gone:
                continue
            present.add(actor.id)
            name = _name_of(actor)
            self._names[actor.id] = name
            if type_id.startswith("walker."):
                if actor.id not in self._walkers:
                    extent = float(actor.bounding_box.extent.z)
                    self._walkers[actor.id] = (math.nan, math.nan, math.nan, extent)
                continue
            if name == self._ego_role or actor.id in self._sensed:
                continue
            if blueprint is None:
                blueprint = world.get_blueprint_library().find("sensor.other.collision")
            try:
                sensor = world.spawn_actor(
                    blueprint, carla.Transform(), attach_to=actor
                )
            except RuntimeError:
                # Gone between the listing and now: a backend took it out of
                # CARLA (SUMO's mesoscopic far traffic does, every few ticks).
                # Not a reason to stop logging the others; a vehicle still
                # there is sensed on the next tick.
                present.discard(actor.id)
                continue
            parent = actor.id
            sensor.listen(_Listener(self, parent))
            self._sensed[parent] = _Sensed(sensor, name)
        for gone in [i for i in self._sensed if i not in present]:
            _release(self._sensed.pop(gone).sensor)
        for gone in [i for i in self._walkers if i not in present]:
            del self._walkers[gone]
            self._heights.pop(gone, None)

    def _on_sensor_data(
        self, parent: int, data: Union["carla.CollisionEvent", "carla.SensorData"]
    ) -> None:
        """*parent*'s sensor's listener (:class:`_Listener`), on CARLA's thread.

        CARLA's Python API, and typesafe_carla's CPython package, call it with
        the ``carla.CollisionEvent`` itself.  typesafe_carla's Codon library
        calls it with the ``carla.SensorData``, which ``as_collision()`` turns
        into the event; the static check sees that branch, and Python, whose
        ``CollisionEvent`` has no ``as_collision``, never takes it
        (:meth:`~autoware_carla_scenario.conditions.CollisionCondition._on_sensor_data`
        does the same).
        """
        if hasattr(data, "as_collision"):
            self._on_event(parent, data.as_collision())
        else:
            self._on_event(parent, data)

    def _on_event(self, parent: int, event: "carla.CollisionEvent") -> None:
        """Queue the collision *parent*'s sensor heard (on CARLA's thread)."""
        other = event.other_actor
        if other is None:
            return  # map geometry
        type_id = str(other.type_id)
        if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
            return
        impulse = event.normal_impulse
        magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        with self._lock:
            self._heard.append(
                (int(event.frame), parent, int(other.id), _name_of(other), magnitude)
            )

    def _collisions(self, elapsed: float) -> list[NpcCollision]:
        """A record for each contact that began, and for each that ended
        (no event from the pair for :data:`REPEAT_S`), with how long it
        lasted; a warning when one begins."""
        with self._lock:
            heard, self._heard = self._heard, []
        out: list[NpcCollision] = []
        for frame, parent, other, other_name, impulse in heard:
            self._names.setdefault(other, other_name)
            pair = (min(parent, other), max(parent, other))
            contact = self._contacts.get(pair)
            if contact is not None:
                contact.last = elapsed
                contact.impulse = max(contact.impulse, impulse)
                continue
            names = [self._names.get(parent, str(parent)), self._names[other]]
            contact = _Contact(frame, elapsed, names, [parent, other], impulse)
            self._contacts[pair] = contact
            logger.warning(
                "Collision at %.2fs between '%s' and '%s' (impulse %.0f N*s)",
                elapsed,
                names[0],
                names[1],
                impulse,
            )
            out.append(contact.record(frame, elapsed, "start", None, self._ego_role))
        for pair, contact in list(self._contacts.items()):
            if elapsed - contact.last <= REPEAT_S:
                continue
            del self._contacts[pair]
            out.append(
                contact.record(
                    contact.frame,
                    elapsed,
                    "end",
                    contact.last - contact.start,
                    self._ego_role,
                )
            )
        return out

    def _look_for_falls(
        self, world: "carla.World", elapsed: float
    ) -> list[WalkerFallen]:
        """The walkers seen down for the first time since the last look."""
        snapshot = world.get_snapshot()
        out: list[WalkerFallen] = []
        for walker_id, (x0, y0, z0, half) in list(self._walkers.items()):
            seen: Optional["carla.ActorSnapshot"] = snapshot.find(walker_id)
            if seen is None:
                continue
            transform = seen.get_transform()
            x, y, z = transform.location.x, transform.location.y, transform.location.z
            self._walkers[walker_id] = (x, y, z, half)
            if z < -_HIDDEN_BELOW_M:
                self._settled[walker_id] = 0
                self._heights.pop(walker_id, None)
                continue
            heights = self._heights.setdefault(walker_id, [])
            standing = min(heights) if len(heights) > 0 else z
            heights.append(z)
            del heights[:-_STANDING_LOOKS]
            looks = self._settled.get(walker_id, 0) + 1
            self._settled[walker_id] = looks
            if walker_id in self._fallen or looks <= _SETTLE_LOOKS:
                # Just brought in: it may still be settling onto the ground.
                continue
            tilt = max(abs(transform.rotation.pitch), abs(transform.rotation.roll))
            drop = 0.0
            if not math.isnan(z0) and z0 > -_HIDDEN_BELOW_M:
                if math.hypot(x - x0, y - y0) < _TELEPORT_M:
                    drop = standing - z
            if tilt < FALLEN_TILT_DEG and drop < 0.5 * max(half, 0.4):
                continue
            self._fallen.add(walker_id)
            role = self._names.get(walker_id, str(walker_id))
            logger.warning(
                "Walker '%s' fell at %.2fs (dropped %.2f m, tipped %.0f deg); "
                "left where it is",
                role,
                elapsed,
                drop,
                tilt,
            )
            out.append(
                WalkerFallen(
                    frame=int(snapshot.frame),
                    t=elapsed,
                    id=walker_id,
                    role=role,
                    drop=drop,
                    tilt=tilt,
                )
            )
        return out


#: The run's monitor, for :func:`release_actor`; ``None`` between runs.
_ACTIVE: Optional[NpcCollisionMonitor] = None


def _name_of(actor: "carla.Actor") -> str:
    """*actor*'s ``role_name``, or its type id when it has none."""
    attributes = actor.attributes
    role = attributes.get("role_name", "") if hasattr(attributes, "get") else ""
    return str(role) if role else str(actor.type_id)


def _release(sensor: "carla.Actor") -> None:
    """Stop and destroy *sensor*; one already gone is let go."""
    try:
        sensor.stop()
    except RuntimeError:
        pass
    try:
        sensor.destroy()
    except RuntimeError:
        pass


class _Listener:
    """A vehicle's collision sensor's listener: hands *parent*'s events to
    *monitor*.

    An object rather than a lambda with a default argument (``parent=parent``),
    which the static check (docs/typecheck.md) cannot compile; ``listen()``
    takes any callable.
    """

    monitor: NpcCollisionMonitor
    parent: int

    def __init__(self, monitor: NpcCollisionMonitor, parent: int) -> None:
        self.monitor = monitor
        self.parent = parent

    def __call__(self, data: Union["carla.CollisionEvent", "carla.SensorData"]) -> None:
        self.monitor._on_sensor_data(self.parent, data)
