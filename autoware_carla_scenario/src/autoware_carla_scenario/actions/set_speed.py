"""Set-speed action: command a vehicle's target speed during a run."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..conditions import BaseCondition, ComparisonRule, SpeedCondition
from ..entity.registry import find_entity_by_role_name
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

__all__ = ["SetSpeedAction", "ARRIVAL_TOLERANCE_KMH"]

#: How close to the target counts as having arrived, in km/h.
#:
#: The run ends on the *vehicle's* speed rather than on a stopwatch, so it needs
#: a band: whatever drives the vehicle is chasing a target under its own control
#: law and will sit near it rather than exactly on it.
ARRIVAL_TOLERANCE_KMH: float = 0.5

_KMH_PER_MS: float = 3.6


class SetSpeedAction(BaseAction):
    """Command a vehicle to drive at a new speed.

    The framework could say what speed a vehicle *started* at and nothing
    else, so "the car ahead brakes to 3 m/s once the ego is within 30 m" had no
    way of being written.  This is that missing half.

    What the command reaches depends on what drives the vehicle: the intent
    goes to the entity, which carries it to its traffic backend.  For a
    TrafficManager-driven vehicle that is ``set_desired_speed``.

    **The target is a target.** Whatever drives the vehicle gets it there under
    its own acceleration limits, so commanding a speed is not commanding a
    velocity, and an immediate change of target is not an immediate change of
    speed.

    Giving *rate_kmh_s* asks for the change to be made no faster than that.  It
    is honoured by walking the commanded target up or down from the vehicle's
    **actual** speed, one tick at a time, which is why the number means
    something: the command can never run ahead of what the vehicle achieved, so
    a vehicle held up by traffic or a speed limit simply takes longer rather
    than being chased by a target it never reached.  OpenSCENARIO's
    ``dynamicsShape: linear`` with ``dynamicsDimension: rate`` is this.

    ``duration`` is deliberately not offered.  A transition stated as a time
    would have to be interpolated open-loop, and a TrafficManager that could
    not keep up would leave the commanded value describing a manoeuvre that
    never happened -- the scenario's number kept while its meaning changed.

    A rate-limited change holds the action
    :attr:`~autoware_carla_scenario.action_state.ActionState.RUNNING` until the
    vehicle is within :data:`ARRIVAL_TOLERANCE_KMH` of the target, so
    ``completeState`` means the vehicle got there rather than that the commands
    stopped.  A vehicle that never gets there never completes, which is the
    same answer a lane change that does not happen gives.

    Args:
        entity_name: ``role_name`` of the vehicle to command.
        target_speed_kmh: The speed to hold, in km/h.
        rate_kmh_s: Most the commanded speed may change per second.  ``None``
            (default) commands the target at once.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        until: Overrides what counts as having arrived.  Defaults to the
            vehicle being within :data:`ARRIVAL_TOLERANCE_KMH` of the target
            for a rate-limited change, and to nothing at all for an immediate
            one -- there is no manoeuvre to wait for when the target is simply
            handed over.
        reissue: Overrides whether the command is re-sent every tick.  The
            default follows *rate_kmh_s*, which is right for a backend that
            holds only the last target it was given; a backend whose command
            carries the rate itself needs no repeat, and says so through this.

    Raises:
        ValueError: If *target_speed_kmh* is negative, or *rate_kmh_s* is not
            positive.  A rate of zero is a change that never arrives, which is
            worth reporting rather than running.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        target_speed_kmh: float,
        rate_kmh_s: Optional[float] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "set_speed",
        once: bool = True,
        until: Optional[BaseCondition] = None,
        reissue: Optional[bool] = None,
    ) -> None:
        if target_speed_kmh < 0:
            raise ValueError("target_speed_kmh must not be negative")
        if rate_kmh_s is not None and rate_kmh_s <= 0:
            raise ValueError("rate_kmh_s must be positive")

        if until is None and rate_kmh_s is not None:
            until = SpeedCondition(
                entity_name=entity_name,
                value=target_speed_kmh / _KMH_PER_MS,
                rule=ComparisonRule.EQUAL_TO,
                tolerance=ARRIVAL_TOLERANCE_KMH / _KMH_PER_MS,
                label=f"{label}_arrived",
            )

        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            until=until,
            reissue=reissue,
        )
        self._entity_name = entity_name
        self._target_speed_kmh = target_speed_kmh
        self._rate_kmh_s = rate_kmh_s

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def _reissues_by_default(self) -> bool:
        """A rate is walked towards, so its target has to be moved every tick.

        An immediate change is one call: ``set_desired_speed`` holds what it was
        given, and repeating it would say the same thing again.
        """
        return self._rate_kmh_s is not None

    def execute(self, world: "carla.World") -> None:
        """Command the target, or this tick's step towards it."""
        entity = find_entity_by_role_name(self._entity_name)
        if entity is None:
            logger.warning(
                "SetSpeedAction: entity '%s' not found", str(self._entity_name)
            )
            return

        if self._rate_kmh_s is None:
            entity.set_speed(world, self._target_speed_kmh)
            logger.info(
                "SetSpeedAction: '%s' commanded to %.1f km/h",
                self._entity_name,
                self._target_speed_kmh,
            )
            return

        step = self._rate_kmh_s * _tick_seconds(world)
        entity.set_speed(
            world,
            _toward(_current_speed_kmh(entity), self._target_speed_kmh, step),
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _toward(now_kmh: float, target_kmh: float, step_kmh: float) -> float:
    """Return *now_kmh* moved at most *step_kmh* towards *target_kmh*."""
    if target_kmh > now_kmh:
        return min(now_kmh + step_kmh, target_kmh)
    return max(now_kmh - step_kmh, target_kmh)


def _tick_seconds(world: "carla.World") -> float:
    """Return how much simulated time the last tick covered.

    Read from the world rather than from ``fixed_delta_seconds`` so the rate
    follows what the server actually did.
    """
    return float(world.get_snapshot().timestamp.delta_seconds)


def _current_speed_kmh(entity: object) -> float:
    """Return *entity*'s present speed in km/h, or 0.0 when unknowable.

    The step is taken from the vehicle's own speed, which is what keeps the
    commanded target from running ahead of it.  An entity with no actor yet --
    or one whose actor has gone -- reads as stopped, which is the only honest
    answer available and never worse than refusing to command anything.
    """
    actor = getattr(entity, "actor", None)
    if actor is None:
        return 0.0
    velocity = actor.get_velocity()
    speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
    return speed_ms * _KMH_PER_MS
