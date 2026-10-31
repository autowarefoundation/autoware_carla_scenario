"""Turn indicator action: light a vehicle's turn indicators."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..conditions.base import BaseCondition, find_actor_in_list
from ..entity_role import EntityRole
from ..utils.turn_indicator import TurnIndicatorState, is_vehicle, set_turn_indicators
from ..control_domain import ControlDomain
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["TurnIndicatorAction"]


class TurnIndicatorAction(BaseAction):
    """Show *state* on a vehicle's turn indicators.

    Sets the blinker bits of the CARLA vehicle's light state and leaves its
    other lights as they are; the indicators stay so until something sets them
    again -- another of these actions, or, on an ego driven by a policy
    (``ego.entity=carla_driver``), the policy's next turn-indicator command.
    What a camera sees, and what
    :class:`~autoware_carla_scenario.conditions.TurnIndicatorCondition` reads.

    Args:
        entity_name: ``role_name`` of the vehicle whose indicators to set.
        state: The indicators to show: ``OFF``, ``LEFT``, ``RIGHT`` or ``HAZARD``.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _state: TurnIndicatorState

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        state: TurnIndicatorState,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "turn_indicator",
        once: bool = True,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = str(entity_name)
        self._state = state

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The vehicle's indicators: its lighting domain."""
        return [(str(self._entity_name), ControlDomain.LIGHTING)]

    def execute(self, world: "carla.World") -> None:
        """Set the named vehicle's turn indicators."""
        actor = find_actor_in_list(world.get_actors(), self._entity_name)
        if actor is None:
            logger.warning(
                "TurnIndicatorAction: entity '%s' not found", self._entity_name
            )
            return
        if not is_vehicle(actor):
            logger.warning(
                "TurnIndicatorAction: entity '%s' is not a vehicle; it has no turn "
                "indicators",
                self._entity_name,
            )
            return
        set_turn_indicators(actor, self._state)
        logger.info(
            "TurnIndicatorAction: '%s' turn indicators %s",
            self._entity_name,
            self._state.name,
        )
