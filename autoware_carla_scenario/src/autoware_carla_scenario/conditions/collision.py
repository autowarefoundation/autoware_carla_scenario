"""Collision-based scenario fail condition using CARLA sensor.other.collision."""

from __future__ import annotations

import enum
import math
import threading
from typing import Any, Optional, Union

import typesafe_carla.carla as carla

from ..constants import EGO_ROLE_NAME
from ..entity_role import EntityRole
from .base import BaseCondition, ScenarioResult
from ..failure import FailureCategory

__all__ = ["CollisionCondition", "CollisionTargetType"]

#: What the details and the message name when the other side of the collision
#: is not an actor (``CollisionEvent.other_actor`` is ``None``).
_NOT_AN_ACTOR = "not an actor"


class CollisionTargetType(enum.Enum):
    """A class of object a collision may be with.

    OpenSCENARIO's ``ByObjectType``, narrowed to the kinds CARLA distinguishes.
    Each member carries the prefix of the blueprint ids it covers, because that
    is the one identifier every CARLA version spells the same way: semantic
    tags are integers whose meaning has moved between releases, and a scenario
    pinned to one would quietly start matching something else.

    Attributes:
        ANY: No restriction -- any collision counts.  The default, and what
            the condition meant before it could be told otherwise.
        VEHICLE: Cars, vans, motorcycles.
        PEDESTRIAN: Walkers.
        STATIC: Street furniture, barriers, props: actors whose blueprint id
            starts ``static.``.  Map geometry (a wall, a kerb) is no actor,
            so it is not of this class, nor of any but ``ANY``.
    """

    ANY = ""
    VEHICLE = "vehicle."
    PEDESTRIAN = "walker."
    STATIC = "static."

    def matches(self, type_id: Optional[str]) -> bool:
        """Whether a CARLA blueprint id belongs to this class."""
        if self is CollisionTargetType.ANY:
            return True
        if type_id is None:
            return False
        return type_id.startswith(self.value)


class CollisionCondition(BaseCondition):
    """Fail condition that triggers when the ego vehicle collides with anything.

    Uses the CARLA ``sensor.other.collision`` sensor attached to the ego vehicle
    (identified by ``role_name == EGO_ROLE_NAME``).  The sensor is attached lazily on the
    first call to :meth:`check` so that it works even when the ego vehicle is
    spawned after the condition is registered.

    By default it fires on a collision with *anything*, which is the honest
    reading of "did the ego crash" but a poor pass/fail assertion in a scenario
    with several actors: "fail if the ego hits the pedestrian" then cannot be
    told apart from "fail if the ego clips a kerb".  Naming a *target* -- one
    entity, or a class of object -- narrows it to the collision the scenario is
    actually about.

    A collision with something that is not an actor -- map geometry such as a
    wall or a kerb, for which CARLA's ``CollisionEvent.other_actor`` is
    ``None`` -- counts when the condition is not targeted, and is ignored by a
    targeted one: it is neither the entity named nor of a type named.  Its
    details then say ``other_actor_type: "not an actor"``.

    Args:
        min_impulse: Minimum collision impulse magnitude (in N·s) required to
            trigger the condition.  Collisions below this threshold are ignored.
            Defaults to ``0.0`` (all collisions trigger).
        target: ``role_name`` of the entity the ego must hit.  ``None`` (the
            default) accepts a collision with anything.
        target_type: A class of object the ego must hit.  Defaults to
            :attr:`CollisionTargetType.ANY`.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If both *target* and a *target_type* other than
            :attr:`~CollisionTargetType.ANY` are given.  OpenSCENARIO's
            ``CollisionCondition`` takes an ``EntityRef`` *or* a ``ByType``,
            and a condition that said both would be asserting one thing twice
            in two vocabularies -- most likely because the author expected one
            of them to be ignored.
    """

    # Minimum interval between ego vehicle search attempts (seconds).
    _ATTACH_RETRY_INTERVAL: float = 1.0

    # Declared for the static check (docs/typecheck.md).
    _min_impulse: float
    _target: Optional[str]
    _target_type: CollisionTargetType
    _sensor: Optional["carla.Actor"]
    _listening: bool
    _closed: bool
    _lock: threading.Lock
    _collided: bool
    _other_type_id: Optional[str]
    _other_is_actor: bool
    _last_attach_attempt: float
    _cached_result: Optional[ScenarioResult]

    def __init__(
        self,
        min_impulse: float = 0.0,
        target: Union[EntityRole, str, None] = None,
        target_type: CollisionTargetType = CollisionTargetType.ANY,
        *,
        label: str,
    ) -> None:
        if target is not None and target_type is not CollisionTargetType.ANY:
            raise ValueError(
                "give a collision condition a target entity or a target type, "
                "not both"
            )
        super().__init__(label=label)
        self._min_impulse = min_impulse
        self._target = str(target) if target is not None else None
        self._target_type = target_type
        self._sensor: Optional["carla.Actor"] = None
        # Whether the sensor's stream is subscribed: CARLA's own flag is
        # spelled differently across its releases (a property, a method).
        self._listening = False
        self._closed = False
        self._lock = threading.Lock()
        self._collided = False
        self._other_type_id: Optional[str] = None
        self._other_is_actor = True
        self._last_attach_attempt: float = -math.inf
        self._cached_result: Optional[ScenarioResult] = None

    def failure_category(self) -> Optional[FailureCategory]:
        """A collision: :attr:`~FailureCategory.COLLISION`."""
        return FailureCategory.COLLISION

    def get_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {"min_impulse": self._min_impulse}
        if self._target is not None:
            details["target"] = self._target
        if self._target_type is not CollisionTargetType.ANY:
            details["target_type"] = self._target_type.name
        if self._collided:
            details["other_actor_type"] = self._other_description()
            if self._cached_result is not None:
                details["collision_elapsed_seconds"] = (
                    self._cached_result.elapsed_seconds
                )
        return details

    def _other_description(self) -> str:
        """What the latched collision was with, for the details and the message."""
        if not self._other_is_actor:
            return _NOT_AN_ACTOR
        return self._other_type_id or "unknown"

    def _wanted(self, other: "carla.Actor") -> bool:
        """Whether a collision with *other* is the one this condition is about.

        Filtering here rather than at the verdict is what makes a targeted
        condition usable: the first collision is latched, so a collision that
        is not the one named must not be recorded at all -- otherwise clipping
        a kerb on the way to the junction would consume the latch and the
        collision the scenario is about would never be seen.
        """
        if self._target is not None:
            attributes = other.attributes
            # Codon's dict.get() takes a default (docs/typecheck.md).
            return "role_name" in attributes and attributes["role_name"] == self._target
        return self._target_type.matches(other.type_id)

    def _on_collision(self, event: "carla.CollisionEvent") -> None:
        """Callback invoked by CARLA when a collision event occurs."""
        impulse = event.normal_impulse
        magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        if magnitude < self._min_impulse:
            return
        # None when the other side is not an actor (map geometry): only an
        # untargeted condition is about that collision (#277).
        other = event.other_actor
        type_id: Optional[str] = None
        if other is None:
            if self._target is not None:
                return
            if self._target_type is not CollisionTargetType.ANY:
                return
        else:
            # Annotated, so Codon unwraps its Optional (docs/typecheck.md).
            actor: "carla.Actor" = other
            if not self._wanted(actor):
                return
            type_id = actor.type_id
        with self._lock:
            if not self._collided:
                self._collided = True
                self._other_is_actor = other is not None
                self._other_type_id = type_id

    def _on_sensor_data(
        self, data: Union["carla.CollisionEvent", "carla.SensorData"]
    ) -> None:
        """The sensor's listener: hands the collision event to :meth:`_on_collision`.

        CARLA's Python API, and typesafe_carla's CPython package, call it with
        the ``carla.CollisionEvent`` itself.  typesafe_carla's Codon library
        calls it with the ``carla.SensorData``, which ``as_collision()`` turns
        into the event; the static check sees that branch, and Python, whose
        ``CollisionEvent`` has no ``as_collision``, never takes it.
        """
        if hasattr(data, "as_collision"):
            self._on_collision(data.as_collision())
        else:
            self._on_collision(data)

    def _try_attach_sensor(self, world: "carla.World", elapsed: float) -> None:
        """Search for the ego vehicle and attach a collision sensor to it.

        Rate-limited by ``_ATTACH_RETRY_INTERVAL`` to avoid calling the
        expensive ``world.get_actors()`` API on every tick during startup.
        Returns without doing anything if the ego is not yet available.
        """
        if elapsed - self._last_attach_attempt < self._ATTACH_RETRY_INTERVAL:
            return
        self._last_attach_attempt = elapsed

        actors = world.get_actors().filter("vehicle.*")
        ego_name = str(EGO_ROLE_NAME)
        # The first vehicle with the ego's role name; Codon's dict.get() takes
        # a default (docs/typecheck.md).
        found = [
            a
            for a in actors
            if "role_name" in a.attributes and a.attributes["role_name"] == ego_name
        ]
        if not found:
            return
        ego = found[0]

        blueprint_library = world.get_blueprint_library()
        sensor_bp = blueprint_library.find("sensor.other.collision")
        sensor = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=ego)
        self._sensor = sensor
        sensor.listen(self._on_sensor_data)
        self._listening = True

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a failure result if the ego vehicle has collided.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            ScenarioResult with passed=False if a collision occurred, None otherwise.
        """
        # Fast path: return cached result after first collision without acquiring the lock.
        if self._cached_result is not None:
            return self._cached_result

        if self._sensor is None and not self._closed:
            self._try_attach_sensor(world, elapsed)

        with self._lock:
            if self._collided:
                if self._other_is_actor:
                    other = f"'{self._other_description()}'"
                else:
                    other = "something that is not an actor (map geometry)"
                self._cached_result = ScenarioResult(
                    passed=False,
                    message=f"Ego vehicle collided with {other} at {elapsed:.2f}s",
                    elapsed_seconds=elapsed,
                )
                # Stop the listener — no further callbacks are needed.
                if self._sensor is not None and self._listening:
                    self._sensor.stop()
                    self._listening = False
                return self._cached_result
        return None

    def close(self) -> None:
        """Stop the collision sensor and destroy it, through this handle.

        Only the handle that started a sensor listening can stop it: LibCarla
        keeps whether a sensor listens on the client's handle, not on the
        actor, and a handle of the same sensor taken from
        ``world.get_actors()`` destroys the actor without unsubscribing its
        stream.  The client then keeps that stream's socket, and its streaming
        threads keep reconnecting it to a stream the server no longer has --
        which slowed every later tick (#290), and races the socket's close
        against the client's own until one of them fails with ``close: Bad
        file descriptor`` and the process aborts (#371).  So the runner closes the
        condition at the end of the run, before anything else destroys the
        sensor or reloads the world.

        A sensor already gone (destroyed by someone else, its episode
        reloaded) is let go: LibCarla raises ``RuntimeError`` for it, and
        there is nothing left to release.  The condition attaches no sensor
        after this, and closing it again does nothing.
        """
        self._closed = True
        if self._sensor is None:
            return
        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        sensor: "carla.Actor" = self._sensor
        self._sensor = None
        if self._listening:
            self._listening = False
            try:
                sensor.stop()
            except RuntimeError:
                pass
        try:
            sensor.destroy()
        except RuntimeError:
            pass
