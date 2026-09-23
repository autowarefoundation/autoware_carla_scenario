"""Walk-straight action: send a pedestrian forward with no destination."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..conditions import BaseCondition
from ..entity.registry import find_entity_by_role_name
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

    from ..entity.pedestrian_entity import PedestrianEntity

logger = logging.getLogger(__name__)

__all__ = ["WalkStraightAction"]

#: A brisk but ordinary walking pace.  `scenario_simulator_v2`'s own
#: `WalkStraightAction` takes no speed at all, so a scenario transpiled from it
#: needs a defensible default rather than a zero that would leave the
#: pedestrian standing in the road.
DEFAULT_WALKING_SPEED_MS = 1.4


class WalkStraightAction(BaseAction):
    """Make a pedestrian walk forward, holding its current heading.

    `scenario_simulator_v2`'s built-in command of the same name.  There is no
    destination: the walker goes the way it is facing until something else
    stops it, which is what makes it usable for a crossing -- the pedestrian
    steps off the kerb on a trigger and keeps going, rather than routing
    around the vehicle the scenario is about.

    Setting *speed_ms* to zero stops the walker where it stands, so a scenario
    can start one walking and stop it later with a second card.

    Args:
        entity_name: ``role_name`` of the pedestrian to command.
        speed_ms: Walking speed in metres per second.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.

    Raises:
        ValueError: If *speed_ms* is negative.  A negative speed would be read
            by CARLA as a direction rather than refused, and a pedestrian
            walking backwards through a crossing is not what anyone wrote.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        speed_ms: float = DEFAULT_WALKING_SPEED_MS,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "walk_straight",
        once: bool = True,
    ) -> None:
        if speed_ms < 0:
            raise ValueError("speed_ms must not be negative")
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = entity_name
        self._speed_ms = speed_ms

    def execute(self, world: "carla.World") -> None:
        """Tell the named pedestrian to start walking."""
        del world
        entity: Optional[PedestrianEntity] = find_entity_by_role_name(self._entity_name)
        if entity is None:
            logger.warning(
                "WalkStraightAction: entity '%s' not found", str(self._entity_name)
            )
            return
        walk = getattr(entity, "walk_straight", None)
        if walk is None:
            logger.warning(
                "WalkStraightAction: '%s' is not a pedestrian and cannot walk",
                str(self._entity_name),
            )
            return
        walk(self._speed_ms)
        logger.info(
            "WalkStraightAction: '%s' walking at %.2f m/s",
            self._entity_name,
            self._speed_ms,
        )
