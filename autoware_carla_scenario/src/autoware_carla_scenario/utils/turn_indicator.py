"""A vehicle's turn indicators, as the blinker bits of its CARLA light state.

The one place that knows which ``carla.VehicleLightState`` bits are the turn
indicators: :class:`~autoware_carla_scenario.conditions.TurnIndicatorCondition`
reads them; :class:`~autoware_carla_scenario.actions.TurnIndicatorAction`, a
driver policy's command (``ego.entity=carla_driver``) and Autoware's
``TurnIndicatorsCommand`` (``ego.entity=autoware`` and ``ros2``) set them.
Setting them leaves every other light as it is.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

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


def _blinker_bits() -> tuple[int, int]:
    """CARLA's ``(LeftBlinker, RightBlinker)`` light bits."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    return (
        int(carla.VehicleLightState.LeftBlinker),
        int(carla.VehicleLightState.RightBlinker),
    )


def turn_indicator_state(light_state: int) -> TurnIndicatorState:
    """The turn indicators a ``carla.VehicleLightState`` bit set shows."""
    left_bit, right_bit = _blinker_bits()
    left = (light_state & left_bit) != 0
    right = (light_state & right_bit) != 0
    if left and right:
        return TurnIndicatorState.HAZARD
    if left:
        return TurnIndicatorState.LEFT
    if right:
        return TurnIndicatorState.RIGHT
    return TurnIndicatorState.OFF


def with_turn_indicators(light_state: int, state: TurnIndicatorState) -> int:
    """*light_state* with its blinker bits showing *state*, every other bit kept."""
    left_bit, right_bit = _blinker_bits()
    others = light_state & ~(left_bit | right_bit)
    if state == TurnIndicatorState.LEFT:
        return others | left_bit
    if state == TurnIndicatorState.RIGHT:
        return others | right_bit
    if state == TurnIndicatorState.HAZARD:
        return others | left_bit | right_bit
    return others


#: ``autoware_vehicle_msgs/msg/TurnIndicatorsCommand`` values, which
#: ``TurnIndicatorsReport`` shares (it has no NO_COMMAND) and carla-driver-interface's
#: ``TurnIndicator`` too.
AUTOWARE_NO_COMMAND: int = 0
AUTOWARE_DISABLE: int = 1
AUTOWARE_ENABLE_LEFT: int = 2
AUTOWARE_ENABLE_RIGHT: int = 3


def from_autoware_command(command: int) -> Optional[TurnIndicatorState]:
    """The indicators an Autoware ``TurnIndicatorsCommand`` value asks for.

    ``None`` for ``NO_COMMAND``, which leaves the indicators as they are.

    Raises:
        ValueError: If *command* is none of the message's values.
    """
    if command == AUTOWARE_NO_COMMAND:
        return None
    if command == AUTOWARE_DISABLE:
        return TurnIndicatorState.OFF
    if command == AUTOWARE_ENABLE_LEFT:
        return TurnIndicatorState.LEFT
    if command == AUTOWARE_ENABLE_RIGHT:
        return TurnIndicatorState.RIGHT
    raise ValueError(f"{command} is not a TurnIndicatorsCommand value")


def autoware_report(state: TurnIndicatorState) -> int:
    """The ``TurnIndicatorsReport`` value of *state*.

    The report has no hazard: Autoware reports hazard lights on their own
    message, so both indicators lit report as ``DISABLE``, as a vehicle
    interface reports them.
    """
    if state == TurnIndicatorState.LEFT:
        return AUTOWARE_ENABLE_LEFT
    if state == TurnIndicatorState.RIGHT:
        return AUTOWARE_ENABLE_RIGHT
    return AUTOWARE_DISABLE


def is_vehicle(actor: "carla.Actor") -> bool:
    """Whether *actor* is a vehicle, the one kind of actor with turn indicators."""
    return str(actor.type_id).startswith("vehicle.")


def turn_indicators_of(actor: "carla.Actor") -> TurnIndicatorState:
    """The turn indicators vehicle *actor* shows."""
    return turn_indicator_state(int(_as_vehicle(actor).get_light_state()))


def set_turn_indicators(actor: "carla.Actor", state: TurnIndicatorState) -> None:
    """Show *state* on vehicle *actor*'s turn indicators, its other lights as they are."""
    vehicle = _as_vehicle(actor)
    vehicle.set_light_state(with_turn_indicators(int(vehicle.get_light_state()), state))


def show_autoware_command(actor: "carla.Actor", command: int) -> bool:
    """Light vehicle *actor*'s indicators as an Autoware ``TurnIndicatorsCommand`` asks.

    ``NO_COMMAND``, or the indicators already shown, changes nothing; only the
    blinker bits are set.  Returns whether the indicators changed.

    Raises:
        ValueError: If *command* is none of the message's values.
    """
    state = from_autoware_command(command)
    if state is None:
        return False
    wanted: TurnIndicatorState = state
    if turn_indicators_of(actor) == wanted:
        return False
    set_turn_indicators(actor, wanted)
    return True


def _as_vehicle(actor: "carla.Actor") -> "carla.Vehicle":
    """*actor* as the vehicle its light state is read from and set on.

    At run time that is *actor* itself, which typesafe_carla's Python API takes
    as it is; its Codon library has the light state on a ``carla.Vehicle``
    only, which the check gets with the checked conversion (Codon reads
    ``TYPE_CHECKING`` as a compile-time true, docs/typecheck.md).
    """
    if TYPE_CHECKING:
        return actor.as_vehicle()
    else:
        return actor
