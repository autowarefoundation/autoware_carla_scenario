"""Turn indicator condition: which of a vehicle's turn indicators are lit."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ..base import ScenarioResult, find_actor_in_list
from .base import CompositionCondition

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


class TurnIndicatorState(Enum):
    """Which turn indicators a vehicle shows.

    Attributes:
        OFF: Neither.
        LEFT: The left one only.
        RIGHT: The right one only.
        HAZARD: Both, as hazard lights show them.
    """

    OFF = auto()
    LEFT = auto()
    RIGHT = auto()
    HAZARD = auto()


def turn_indicator_state(light_state: int) -> TurnIndicatorState:
    """The turn indicators a ``carla.VehicleLightState`` bit set shows."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    left = (light_state & int(carla.VehicleLightState.LeftBlinker)) != 0
    right = (light_state & int(carla.VehicleLightState.RightBlinker)) != 0
    if left and right:
        return TurnIndicatorState.HAZARD
    if left:
        return TurnIndicatorState.LEFT
    if right:
        return TurnIndicatorState.RIGHT
    return TurnIndicatorState.OFF


class TurnIndicatorCondition(CompositionCondition):
    """Pass while a vehicle's turn indicators are in *state*.

    Reads the CARLA vehicle's light state, so it holds whatever lit them: a
    driver policy's command (``ego.entity=carla_driver``), a scenario, or the
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
        entity = find_actor_in_list(world.get_actors(), self._entity_name)
        if entity is None or not str(entity.type_id).startswith("vehicle."):
            return None
        shown = turn_indicator_state(int(_as_vehicle(entity).get_light_state()))
        if shown != self._state:
            return None
        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' turn indicators are"
                f" {self._state.name.lower()}"
            ),
            elapsed_seconds=elapsed,
        )


def _as_vehicle(actor: "carla.Actor") -> "carla.Vehicle":
    """*actor* as the vehicle its light state is read from.

    At run time that is *actor* itself, which typesafe_carla's Python API takes
    as it is; its Codon library has the light state on a ``carla.Vehicle``
    only, which the check gets with the checked conversion (Codon reads
    ``TYPE_CHECKING`` as a compile-time true, docs/typecheck.md).
    """
    if TYPE_CHECKING:
        return actor.as_vehicle()
    else:
        return actor
