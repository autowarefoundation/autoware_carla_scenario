"""Whether an entity may carry out an adversity now (#392).

:class:`AdversityFeasibleCondition` is TeraSim's trigger as a condition: it
holds on the tick an adversity of its kind may fire for the entity -- the
entity at the location (:mod:`~..adversity.location`) and the kind's trigger
met (:mod:`~..adversity.triggers`).  It is the default trigger of every
adversarial action, which reads the command it found (:attr:`command`); a
scenario can also compose it with others (``AndCondition``) or use it on its
own, to assert that a situation came about.

Each check takes a snapshot of every vehicle and walker in the world (in the
Lanelet2 map frame, :mod:`~..adversity.snapshot`) and of the traffic lights
the map names, and asks the trigger.  Needs the
:class:`~..coordinate.map_manager.MapManager` to hold the Lanelet2 map.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Optional

from ..adversity.command import AdversarialCommand
from ..adversity.location import AdversityLocation
from ..adversity.observation import AgentSnapshot, Scene
from ..adversity.snapshot import SignalReader, snapshot_actor
from ..adversity.triggers import (
    DEFAULT_THRESHOLDS,
    AdversityKind,
    TriggerThresholds,
    adversity_command,
)
from ..coordinate.map_manager import MapManager
from ..entity_role import EntityRole, _role_name
from ..route.geometry import MapFeatures
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["AdversityFeasibleCondition"]


class AdversityFeasibleCondition(BaseCondition):
    """Holds when *entity_name* may carry out a *kind* adversity at *location*.

    Args:
        entity_name: ``role_name`` of the vehicle (or, for JAYWALKING, the
            pedestrian).
        kind: The adversity.
        location: Where it may fire; :attr:`AdversityLocation.ANY` anywhere.
        label: Human-readable identifier.
        seed: Seeds the random draws a trigger makes (a jaywalker's speed).
        thresholds: The triggers' numbers; TeraSim's by default.
    """

    _entity_name: str
    _kind: AdversityKind
    _location: AdversityLocation
    _rng: random.Random
    _thresholds: TriggerThresholds
    _features: Optional[MapFeatures]
    _signals: Optional[SignalReader]
    _command: Optional[AdversarialCommand]
    _scene: Optional[Scene]

    def __init__(
        self,
        entity_name: EntityRole | str,
        kind: AdversityKind,
        location: AdversityLocation,
        *,
        label: str,
        seed: int = 0,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        super().__init__(label=label)
        self._entity_name = str(entity_name)
        self._kind = kind
        self._location = location
        self._rng = random.Random(seed)
        self._thresholds = thresholds
        self._features = None
        self._signals = None
        self._command = None
        self._scene = None

    @property
    def kind(self) -> AdversityKind:
        """The adversity it asks about."""
        return self._kind

    @property
    def location(self) -> AdversityLocation:
        """Where the adversity may fire."""
        return self._location

    @property
    def command(self) -> Optional[AdversarialCommand]:
        """The command the last check that held found, else None."""
        return self._command

    @property
    def scene(self) -> Optional[Scene]:
        """The scene the last check that held saw, else None."""
        return self._scene

    def get_details(self) -> dict[str, Any]:
        return {
            "entity_name": self._entity_name,
            "kind": self._kind.value,
            "location": self._location.value,
        }

    def _map(self) -> MapFeatures:
        if self._features is None:
            manager = MapManager.get_instance()
            features = MapFeatures(manager.lanelet_map, manager.routing_graph)
            self._features = features
            return features
        return self._features

    def _read_signals(self, world: carla.World, features: MapFeatures) -> SignalReader:
        if self._signals is None:
            reader = SignalReader(world, features.traffic_light_ids())
            self._signals = reader
            return reader
        return self._signals

    def check(self, world: carla.World, elapsed: float) -> Optional[ScenarioResult]:
        """Hold, with the command kept, when the adversity may fire now."""
        features = self._map()
        agents: list[AgentSnapshot] = []
        for actor in world.get_actors():
            type_id = str(actor.type_id)
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
                continue
            role = _role_name(actor)
            name = str(actor.id) if role is None else str(role)
            agents.append(snapshot_actor(actor, name))
        scene = Scene(features, agents, self._read_signals(world, features).read())
        if self._entity_name not in scene.agents:
            return None
        command = adversity_command(
            self._kind,
            scene,
            self._entity_name,
            self._location,
            self._rng,
            self._thresholds,
        )
        if command is None:
            return None
        self._command = command
        self._scene = scene
        return ScenarioResult(
            passed=True,
            message=(
                f"'{self._entity_name}' may {self._kind.value.replace('_', ' ')} "
                f"({self._location.value})"
            ),
            elapsed_seconds=elapsed,
        )
