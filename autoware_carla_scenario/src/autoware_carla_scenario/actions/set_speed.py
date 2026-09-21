"""Set-speed action: command a vehicle's target speed during a run."""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Optional, Union

from ..conditions import BaseCondition
from ..entity.registry import find_entity_by_role_name
from ..entity_role import EntityRole
from ..traffic import SettingSpeed
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

__all__ = ["SetSpeedAction", "SpeedTransition"]


class SpeedTransition(enum.Enum):
    """How the target speed gets from where it was to where it is going.

    Mirrors the two OpenSCENARIO ``dynamicsShape`` values that can be honoured
    here.  ``sinusoidal`` and ``cubic`` are not offered: approximating them
    with a straight line would keep the scenario's number while changing what
    it describes, which is worse than refusing.

    Attributes:
        STEP: The new target applies on the tick the action fires.
        LINEAR: The target is interpolated over ``duration`` seconds.
    """

    STEP = "step"
    LINEAR = "linear"


class SetSpeedAction(BaseAction):
    """Command a vehicle to drive at a new speed.

    The framework could say what speed a vehicle *started* at and nothing
    else, so "the car ahead brakes to 3 m/s once the ego is within 30 m" had no
    way of being written.  This is that missing half.

    What the command reaches depends on what drives the vehicle: the intent
    goes to the entity, which carries it to its traffic backend.  For a
    TrafficManager-driven vehicle that is ``set_desired_speed``.

    **The target is a target.** Whatever drives the vehicle gets it there under
    its own acceleration limits, so even a :attr:`SpeedTransition.STEP` change
    is not an instant change of velocity -- and a :attr:`SpeedTransition.LINEAR`
    transition ramps the *target*, which is what OpenSCENARIO's
    ``dynamicsShape: linear`` with ``dynamicsDimension: time`` describes.

    A ramping action stays
    :attr:`~autoware_carla_scenario.action_state.ActionState.RUNNING` for the
    whole of *duration* and only then reports ``completeState``, so an
    ``action_state`` condition can wait for the manoeuvre rather than for the
    command.

    Args:
        entity_name: ``role_name`` of the vehicle to command.
        target_speed_kmh: The speed to hold, in km/h.
        transition: Step or linear.  Defaults to :attr:`SpeedTransition.STEP`.
        duration: Seconds the linear ramp takes.  Ignored, and required to be
            zero, for a step.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.

    Raises:
        ValueError: If *target_speed_kmh* is negative, or if *duration* does not
            match *transition* -- a linear ramp over zero seconds is a step
            written the long way round, and saying both is a mistake worth
            reporting rather than quietly resolving.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        target_speed_kmh: float,
        transition: SpeedTransition = SpeedTransition.STEP,
        duration: float = 0.0,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "set_speed",
        once: bool = True,
    ) -> None:
        if target_speed_kmh < 0:
            raise ValueError("target_speed_kmh must not be negative")
        if transition is SpeedTransition.LINEAR and duration <= 0:
            raise ValueError("a linear speed transition needs a positive duration")
        if transition is SpeedTransition.STEP and duration:
            raise ValueError(
                "a step speed transition takes no duration; use "
                "SpeedTransition.LINEAR to ramp"
            )
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = entity_name
        self._target_speed_kmh = target_speed_kmh
        self._transition = transition
        self._duration = duration
        #: The entity resolved in :meth:`execute`, commanded again on each tick
        #: of a ramp.  Looked up once: the answer cannot change mid-manoeuvre.
        self._entity: Optional[SettingSpeed] = None
        #: The speed the vehicle was asked to hold when the ramp began, which
        #: is where the interpolation starts from.
        self._start_speed_kmh: float = 0.0

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def execute(self, world: "carla.World") -> None:
        """Command the new speed, or begin ramping towards it."""
        entity = find_entity_by_role_name(self._entity_name)
        if entity is None:
            logger.warning(
                "SetSpeedAction: entity '%s' not found", str(self._entity_name)
            )
            self._entity = None
            return

        self._entity = entity
        self._start_speed_kmh = _current_speed_kmh(entity)

        if self._transition is SpeedTransition.STEP:
            entity.set_speed(world, self._target_speed_kmh)
            logger.info(
                "SetSpeedAction: '%s' commanded to %.1f km/h",
                self._entity_name,
                self._target_speed_kmh,
            )
            return

        logger.info(
            "SetSpeedAction: '%s' ramping %.1f -> %.1f km/h over %.1fs",
            self._entity_name,
            self._start_speed_kmh,
            self._target_speed_kmh,
            self._duration,
        )

    def on_running(self, world: "carla.World", running_for: float) -> None:
        """Apply this tick's point on the ramp."""
        if self._entity is None or self._transition is SpeedTransition.STEP:
            return
        self._entity.set_speed(world, self._speed_at(running_for))

    def is_finished(self, world: "carla.World", running_for: float) -> bool:
        """Whether the commanded change is over.

        A step is over the moment it is commanded -- what happens afterwards is
        the vehicle obeying a target, not the action still working.  A ramp is
        over when its duration has elapsed.

        An action whose entity was not found never ramps and is finished at
        once: there is nothing to wait for, and holding it ``RUNNING`` would
        make a missing NPC look like a manoeuvre still under way.
        """
        del world
        if self._entity is None or self._transition is SpeedTransition.STEP:
            return True
        return running_for >= self._duration

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _speed_at(self, running_for: float) -> float:
        """Return the ramp's target speed *running_for* seconds in."""
        if running_for >= self._duration:
            return self._target_speed_kmh
        fraction = running_for / self._duration
        span = self._target_speed_kmh - self._start_speed_kmh
        return self._start_speed_kmh + span * fraction


def _current_speed_kmh(entity: object) -> float:
    """Return *entity*'s present speed in km/h, or 0.0 when unknowable.

    The ramp has to start somewhere, and starting it at the vehicle's actual
    speed is what makes a linear transition look linear.  An entity with no
    actor yet -- or one whose actor has gone -- ramps from zero, which is the
    only honest answer available and never worse than refusing to ramp at all.
    """
    actor = getattr(entity, "actor", None)
    if actor is None:
        return 0.0
    velocity = actor.get_velocity()
    speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
    return speed_ms * 3.6
