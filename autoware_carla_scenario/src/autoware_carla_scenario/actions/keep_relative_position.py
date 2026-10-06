"""Keep-relative-position action: hold a vehicle at a gap from another one."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional, Union

from ..conditions import BaseCondition, ScenarioResult
from ..conditions.base import find_actor_pair
from ..coordinate.lane_distance import lane_gap
from ..coordinate.poses import CarlaWorldPose
from ..entity.registry import find_entity_by_role_name
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["KeepRelativePositionAction"]

_KMH_PER_MS: float = 3.6


class _HeldFor(BaseCondition):
    """Fires *seconds* after it is first asked: how long a hold lasts."""

    def __init__(self, seconds: float, label: str) -> None:
        super().__init__(label=label)
        self._seconds = seconds
        self._since: Optional[float] = None

    def get_details(self) -> dict[str, Any]:
        return {"seconds": self._seconds}

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        del world
        if self._since is None:
            self._since = elapsed
        if elapsed - self._since < self._seconds:
            return None
        return ScenarioResult(
            passed=True,
            message=f"held for {self._seconds:.1f} s",
            elapsed_seconds=elapsed,
        )


class KeepRelativePositionAction(BaseAction):
    """Hold a vehicle *gap_m* metres ahead of (or behind) another, every tick.

    A scenario that places road users relative to the ego -- "a car 20 m ahead
    in the next lane" -- only describes the moment they spawn.  The ego is
    driven by its own stack and the others by theirs, so a few seconds later
    the relationship the scenario was about is gone, and whatever was meant to
    happen in it (a cut-in at 20 m) happens somewhere else or not at all.

    This closes that loop.  On every tick the vehicle's gap to *target* is
    measured along the road (the target's direction of travel, through
    :func:`~autoware_carla_scenario.coordinate.lane_distance.lane_gap`), and the
    vehicle is commanded the target's own speed plus a correction proportional
    to the error::

        command = v_target + gain * (gap_m - gap)

    so a vehicle that has fallen behind where it should be speeds up, one that
    has run ahead slows, and one in place matches the target.  The command is
    clamped to ``[0, max_speed_kmh]``.  Where no road joins the two the
    straight-line offset projected on the target's heading stands in.

    Only the longitudinal gap is held: the lane is the one the vehicle spawned
    in, and a lane change changes it.

    Args:
        entity_name: ``role_name`` of the vehicle to command.
        target: ``role_name`` of the vehicle the gap is measured from.
        gap_m: Where to hold it, along the road from *target*: positive ahead,
            negative behind.
        gain: Correction per metre of error, in 1/s.
        max_speed_kmh: The most the vehicle is commanded.
        hold_seconds: How long to hold, in seconds, from when it fires; ``0``
            holds until the run ends.  The action completes when the hold
            does, so a manoeuvre can wait on ``action_state`` for it.
        condition: Trigger condition.
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.

    Raises:
        ValueError: If *gain* is not positive, or *max_speed_kmh* or
            *hold_seconds* is negative.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        target: Union[EntityRole, str],
        gap_m: float,
        gain: float = 0.5,
        max_speed_kmh: float = 80.0,
        hold_seconds: float = 0.0,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "keep_relative_position",
        once: bool = True,
    ) -> None:
        if gain <= 0:
            raise ValueError("gain must be positive")
        if max_speed_kmh < 0:
            raise ValueError("max_speed_kmh must not be negative")
        if hold_seconds < 0:
            raise ValueError("hold_seconds must not be negative")
        # A hold always has an `until`: without one the run would end on the
        # tick it began, after a single correction.
        until = _HeldFor(
            hold_seconds if hold_seconds > 0 else math.inf, label=f"{label}_held"
        )
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            until=until,
            reissue=True,
        )
        self._entity_name = entity_name
        self._target = target
        self._gap_m = gap_m
        self._gain = gain
        self._max_speed_kmh = max_speed_kmh

    def gap(self, world: "carla.World") -> Optional[tuple[float, float]]:
        """Return ``(gap, target speed along its heading)`` in m and m/s."""
        vehicle, target = find_actor_pair(
            world.get_actors(), self._entity_name, self._target
        )
        if vehicle is None or target is None:
            return None
        transform = target.get_transform()
        forward = transform.get_forward_vector()
        fx, fy = forward.x, forward.y
        here, there = vehicle.get_location(), target.get_location()
        ahead = lane_gap(
            CarlaWorldPose(x=there.x, y=there.y, z=there.z, yaw=0.0),
            fx,
            fy,
            CarlaWorldPose(x=here.x, y=here.y, z=here.z, yaw=0.0),
        )
        if ahead is None:
            ahead = (here.x - there.x) * fx + (here.y - there.y) * fy
        velocity = target.get_velocity()
        return ahead, velocity.x * fx + velocity.y * fy

    def command_kmh(self, gap: float, target_speed_ms: float) -> float:
        """The speed that closes the error between *gap* and :attr:`gap_m`."""
        command_ms = target_speed_ms + self._gain * (self._gap_m - gap)
        return min(max(command_ms * _KMH_PER_MS, 0.0), self._max_speed_kmh)

    def execute(self, world: "carla.World") -> None:
        """Command this tick's correction."""
        entity = find_entity_by_role_name(self._entity_name)
        measured = self.gap(world)
        if entity is None or measured is None:
            logger.debug(
                "KeepRelativePositionAction: '%s' or '%s' not found",
                self._entity_name,
                self._target,
            )
            return
        entity.set_desired_speed(world, self.command_kmh(*measured))
