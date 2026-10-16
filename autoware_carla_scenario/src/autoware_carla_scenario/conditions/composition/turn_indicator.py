"""Turn indicator condition: which of a vehicle's turn indicators are lit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...utils.turn_indicator import TurnIndicatorState, is_vehicle, turn_indicators_of
from ..base import ScenarioResult, find_actor_by_role_name
from .base import CompositionCondition

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["TurnIndicatorCondition", "TurnIndicatorState"]


class TurnIndicatorCondition(CompositionCondition):
    """Pass while a vehicle's turn indicators are in *state*.

    Reads the CARLA vehicle's light state, so it holds whatever lit them: a
    driver policy's command (``ego.entity=carla_driver``), a
    :class:`~autoware_carla_scenario.actions.TurnIndicatorAction`, or the
    simulator itself.  An entity that is not a vehicle never passes.

    Args:
        entity_name: The ``role_name`` attribute of the vehicle to evaluate.
        state: The turn indicators expected, e.g. :attr:`TurnIndicatorState.LEFT`.
        label: Human-readable identifier for this condition.
    """

    # Declared for the static check (docs/typecheck.md).
    _state: TurnIndicatorState

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        state: TurnIndicatorState,
        *,
        label: str,
    ) -> None:
        super().__init__(entity_name=entity_name, label=label)
        self._state = state

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update({"state": self._state.name})
        return details

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result if the vehicle shows the expected indicators.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            :class:`ScenarioResult` with ``passed=True`` if the indicators
            match, ``None`` otherwise.
        """
        assert self._entity_name is not None
        entity = find_actor_by_role_name(world, self._entity_name)
        if entity is None or not is_vehicle(entity):
            return None
        if turn_indicators_of(entity) != self._state:
            return None
        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' turn indicators are"
                f" {self._state.name.lower()}"
            ),
            elapsed_seconds=elapsed,
        )
