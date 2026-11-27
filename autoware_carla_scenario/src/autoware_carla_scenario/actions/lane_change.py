"""Lane-change action: ask a vehicle's traffic backend for a lane change."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Union

from ..action_state import ActionState
from ..conditions.base import BaseCondition
from ..conditions.lane_change_settled import LaneChangeSettledCondition
from ..entity.registry import fail_if_pedestrian, find_vehicle_entity
from ..traffic import LaneChangeDirection
from ..entity_role import EntityRole
from ..control_domain import ControlDomain
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


class LaneChangeAction(BaseAction):
    """Ask a vehicle to change one lane, through whatever drives it.

    When the associated condition is satisfied, this action:

    1. Locates the target vehicle by its ``role_name``
    2. Calls ``entity.change_lane(world, direction)`` on it

    The action does not name a mechanism: the entity carries the request out
    itself, and what happens depends on what drives it.

    * A vehicle the run's traffic backend drives (an NPC, or a plain
      :class:`~autoware_carla_scenario.entity.ego.EgoVehicle`; see
      :class:`~autoware_carla_scenario.traffic.driven.BackendDriven`) hands
      the request to that backend's ``change_lane``:

      * the TrafficManager backend (the default) calls
        ``TrafficManager.force_lane_change(actor, direction)``;
      * the SUMO backend calls ``traci.vehicle.changeLane`` towards the
        adjacent lane (logging a warning, but still asking, when the marking
        forbids the change).  A vehicle SUMO simulates mesoscopically has no
        lanes, so it logs a warning and does nothing; a scenario vehicle SUMO
        leaves to the TrafficManager gets ``force_lane_change`` as above;
      * a backend that cannot change lane (the ``none`` backend, or SUMO for a
        vehicle it does not drive) logs a warning and does nothing.

    * An ego driven by a stack of its own (an
      :class:`~autoware_carla_scenario.entity.externally_driven.ExternallyDrivenEgo`:
      Autoware, an external driver policy, ROS 2 nodes) refuses: the stack
      plans its own manoeuvres, so it logs a warning and does nothing.

    The command only starts the manoeuvre, so the action keeps watching the
    vehicle afterwards and stays
    :attr:`~autoware_carla_scenario.actions.base.ActionState.RUNNING` until the
    vehicle has actually settled onto the next lane.  That is the signal another
    actor can react to; ``done`` would fire while the car is still straddling
    the line.

    The waiting is a :class:`LaneChangeSettledCondition` supplied as the run's
    *until*, so the end of the manoeuvre is written in the same vocabulary as
    the trigger that started it.  The command itself is **not** reissued: one
    request starts the manoeuvre, and sending it again every tick would restart
    it.

    By default the manoeuvre's place and shape are the backend's own.
    *start_after_m* and *lane_change_length_m* ask for them, in metres the
    vehicle drives, so a slower vehicle takes longer over the same geometry and
    a stopped one waits.  The vehicle then also slows for what is ahead of it
    between it and the lane it goes to, on the TrafficManager and SUMO backends
    (SUMO's stretched crossing needs its sublane model).

    Args:
        entity_name: ``role_name`` of the vehicle actor to control.
        direction: :class:`LaneChangeDirection` — ``LEFT`` or ``RIGHT``.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        overrides: Take over a control domain of the entity another action
            holds (:attr:`~BaseAction.overrides`).
        until: Overrides what counts as the manoeuvre having settled.  Defaults
            to :class:`LaneChangeSettledCondition` on *entity_name*.  One that
            ends the run before the vehicle has settled stops the lane change
            still to be carried out, as an override does (:meth:`tick`).
        start_after_m: How far to drive after the trigger before starting to
            cross, m.  ``0.0`` (default): at once.
        lane_change_length_m: How far to drive while crossing, m.  ``0.0``
            (default): as far as the backend takes.

    Raises:
        ValueError: *start_after_m* or *lane_change_length_m* is a boolean,
            negative or not finite.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _direction: LaneChangeDirection
    _start_after_m: float
    _lane_change_length_m: float
    _settled: LaneChangeSettledCondition

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        direction: LaneChangeDirection,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "lane_change",
        once: bool = True,
        overrides: bool = False,
        until: Optional[BaseCondition] = None,
        start_after_m: float = 0.0,
        lane_change_length_m: float = 0.0,
    ) -> None:
        # #2077: NaN passed `< 0` and changed lanes at once; infinity held
        # the vehicle (and its lateral domain) for the rest of the run.
        # A boolean would read as 0 or 1 m (#2273).
        if (
            isinstance(start_after_m, bool)
            or not math.isfinite(start_after_m)
            or start_after_m < 0.0
        ):
            raise ValueError(
                f"start_after_m must be finite and not negative, not {start_after_m!r}"
            )
        if (
            isinstance(lane_change_length_m, bool)
            or not math.isfinite(lane_change_length_m)
            or lane_change_length_m < 0.0
        ):
            raise ValueError(
                "lane_change_length_m must be finite and not negative, "
                f"not {lane_change_length_m!r}"
            )
        if until is None:
            settled: BaseCondition = LaneChangeSettledCondition(
                entity_name, label=f"{label}_settled"
            )
            until = settled
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            overrides=overrides,
            until=until,
        )
        self._entity_name = str(entity_name)
        self._direction = direction
        self._start_after_m = start_after_m
        self._lane_change_length_m = lane_change_length_m
        # Whether the vehicle has settled, asked as a run ends: one its
        # *until* ended before then is stopped (:meth:`tick`).
        self._settled = LaneChangeSettledCondition(
            entity_name, label=f"{label}_settled"
        )

    @property
    def needs_tick_loop(self) -> bool:
        """Always: the run waits on its *until*, the vehicle settling onto the
        next lane, which only a tick can see."""
        return True

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Step the run (:meth:`BaseAction.tick`); one its *until* ends before
        the vehicle has settled onto the next lane is a stop, as an overridden
        one is (:meth:`_stopped`): the lane change still to be carried out is
        dropped, so it does not happen on top of a later lateral action."""
        super().tick(world, elapsed)
        if (
            self.state is ActionState.END_TRANSITION
            and self._settled.check(world, elapsed) is None
        ):
            self._stopped(world)

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The vehicle's lane: its lateral domain."""
        return [(str(self._entity_name), ControlDomain.LATERAL)]

    def execute(self, world: "carla.World") -> None:
        """Ask the named entity to change lane."""
        entity = find_vehicle_entity(self._entity_name)
        if entity is None:
            fail_if_pedestrian(self._entity_name, "change_lane")
            if self._report_missing():
                logger.warning(
                    "LaneChangeAction: entity '%s' not found", str(self._entity_name)
                )
            return

        entity.change_lane(
            world, self._direction, self._start_after_m, self._lane_change_length_m
        )

    def _stopped(self, world: "carla.World") -> None:
        """Overridden mid-run (:meth:`BaseAction._stop_overridden`), or the
        run ended by its *until* before the vehicle settled (:meth:`tick`):
        the lane change the backend is still to carry out -- one waiting for its
        *start_after_m*, or being stretched over its *lane_change_length_m* --
        is dropped, so it does not happen on top of what took the lane over
        (#2430).  One that is done is left as it is
        (:meth:`~autoware_carla_scenario.traffic.TrafficBackend.cancel_lane_change`)."""
        entity = find_vehicle_entity(self._entity_name)
        if entity is None:
            return
        entity.cancel_lane_change(world)
