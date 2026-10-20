"""Whether an entity may carry out an adversity now (#392).

:class:`AdversityFeasibleCondition` is TeraSim's trigger as a condition: it
holds on the tick an adversity of its kind may fire for the entity -- the
entity at the location (:mod:`~..adversity.location`) and the kind's trigger
met (:mod:`~..adversity.triggers`).  It is the default trigger of every
adversarial action, which reads the command it found (:attr:`command`); a
scenario can also compose it with others (``AndCondition``) or use it on its
own, to assert that a situation came about.

Each check asks the trigger of the tick's scene: the world's vehicles,
walkers and traffic lights in the Lanelet2 map frame, read once a frame for
every condition and action that asks (:func:`~..adversity.snapshot.scene_at`).
Needs the :class:`~..coordinate.map_manager.MapManager` to hold the Lanelet2
map.  Not re-exported from :mod:`autoware_carla_scenario.conditions`, which
every scenario imports, because it pulls in the Lanelet2 map layer: import it
from here or from the package's top level.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Optional, Union

from ..adversity.command import AdversarialCommand
from ..adversity.location import AdversityLocation
from ..adversity.observation import Scene
from ..adversity.snapshot import scene_at
from ..adversity.triggers import (
    DEFAULT_THRESHOLDS,
    AdversityKind,
    TriggerThresholds,
    adversity_command,
)
from ..entity_role import EntityRole
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["AdversityFeasibleCondition"]


class AdversityFeasibleCondition(BaseCondition):
    """Holds when *entity_name* may carry out a *kind* adversity at *location*.

    Args:
        entity_name: ``role_name`` of the vehicle (or, for a kind of
            ``PEDESTRIAN_KINDS``, the pedestrian).
        kind: The adversity.
        location: Where it may fire; :attr:`AdversityLocation.ANY` anywhere.
        label: Human-readable identifier.
        seed: Seeds the random draws a trigger makes (a jaywalker's speed,
            a stopped crosser's wait).
        thresholds: The triggers' numbers; TeraSim's by default.
    """

    _entity_name: str
    _kind: AdversityKind
    _location: AdversityLocation
    _rng: random.Random
    _thresholds: TriggerThresholds
    _command: Optional[AdversarialCommand]
    _scene: Optional[Scene]

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
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

    def check(self, world: carla.World, elapsed: float) -> Optional[ScenarioResult]:
        """Hold, with the command kept, when the adversity may fire now."""
        scene = scene_at(world)
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
