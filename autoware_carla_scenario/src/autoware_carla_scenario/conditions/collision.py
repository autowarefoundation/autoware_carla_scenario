"""Collision-based scenario fail condition using CARLA sensor.other.collision."""

from __future__ import annotations

import enum
import math
import threading
from typing import Any, Optional, Union

import carla

from ..constants import EGO_ROLE_NAME
from ..entity_role import EntityRole
from .base import BaseCondition, ScenarioResult

__all__ = ["CollisionCondition", "CollisionTargetType"]


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
        STATIC: Street furniture, barriers, props.
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
    """Fail condition that triggers when the ego vehicle collides with any actor.

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

    Args:
        min_impulse: Minimum collision impulse magnitude (in N·s) required to
            trigger the condition.  Collisions below this threshold are ignored.
            Defaults to ``0.0`` (all collisions trigger).
        target: ``role_name`` of the entity the ego must hit.  ``None`` (the
            default) accepts any actor.
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
        self._lock = threading.Lock()
        self._collided = False
        self._other_type_id: Optional[str] = None
        self._last_attach_attempt: float = -math.inf
        self._cached_result: Optional[ScenarioResult] = None

    def get_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {"min_impulse": self._min_impulse}
        if self._target is not None:
            details["target"] = self._target
        if self._target_type is not CollisionTargetType.ANY:
            details["target_type"] = self._target_type.name
        if self._collided:
            details["other_actor_type"] = self._other_type_id or "unknown"
            if self._cached_result is not None:
                details["collision_elapsed_seconds"] = (
                    self._cached_result.elapsed_seconds
                )
        return details

    def _wanted(self, other: "carla.Actor") -> bool:
        """Whether a collision with *other* is the one this condition is about.

        Filtering here rather than at the verdict is what makes a targeted
        condition usable: the first collision is latched, so a collision that
        is not the one named must not be recorded at all -- otherwise clipping
        a kerb on the way to the junction would consume the latch and the
        collision the scenario is about would never be seen.
        """
        if self._target is not None:
            return other.attributes.get("role_name") == self._target
        return self._target_type.matches(other.type_id)

    def _on_collision(self, event: "carla.CollisionEvent") -> None:
        """Callback invoked by CARLA when a collision event occurs."""
        impulse = event.normal_impulse
        magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        if magnitude < self._min_impulse:
            return
        if not self._wanted(event.other_actor):
            return
        with self._lock:
            if not self._collided:
                self._collided = True
                self._other_type_id = event.other_actor.type_id

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
        ego = next(
            (a for a in actors if a.attributes.get("role_name") == str(EGO_ROLE_NAME)),
            None,
        )
        if ego is None:
            return

        blueprint_library = world.get_blueprint_library()
        sensor_bp = blueprint_library.find("sensor.other.collision")
        self._sensor = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=ego)
        self._sensor.listen(self._on_collision)

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

        if self._sensor is None:
            self._try_attach_sensor(world, elapsed)

        with self._lock:
            if self._collided:
                other = self._other_type_id or "unknown"
                self._cached_result = ScenarioResult(
                    passed=False,
                    message=f"Ego vehicle collided with '{other}' at {elapsed:.2f}s",
                    elapsed_seconds=elapsed,
                )
                # Stop the listener — no further callbacks are needed.
                if self._sensor is not None:
                    self._sensor.stop()
                return self._cached_result
        return None
