"""Lane-change action: ask a vehicle's traffic backend for a lane change."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..conditions.base import BaseCondition
from ..conditions.lane_change_settled import LaneChangeSettledCondition
from ..entity.pedestrian_entity import PedestrianEntity
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
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

    Args:
        entity_name: ``role_name`` of the vehicle actor to control.
        direction: :class:`LaneChangeDirection` — ``LEFT`` or ``RIGHT``.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        once: If ``True`` (default) the action fires at most once.
        overrides: Take over a control domain of the entity another action
            holds (:attr:`~BaseAction.overrides`).
        until: Overrides what counts as the manoeuvre having settled.  Defaults
            to :class:`LaneChangeSettledCondition` on *entity_name*.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _direction: LaneChangeDirection

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
    ) -> None:
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

    @property
    def needs_tick_loop(self) -> bool:
        """Always: the run waits on its *until*, the vehicle settling onto the
        next lane, which only a tick can see."""
        return True

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
            _fail_if_pedestrian(self._entity_name)
            logger.warning(
                "LaneChangeAction: entity '%s' not found", str(self._entity_name)
            )
            return

        entity.change_lane(world, self._direction)


def _fail_if_pedestrian(entity_name: str) -> None:
    """Raise :class:`AttributeError` when *entity_name* is a pedestrian.

    A pedestrian cannot change lane. The vehicle lookup does not find one, but
    the action used to look up an entity of any kind and call
    ``change_lane`` on it, which raised; it still raises, rather than reading
    as an entity that is not there.
    """
    found = find_pedestrian_entity(entity_name)
    if found is None:
        return
    walker: PedestrianEntity = found
    raise AttributeError(
        f"'{type(walker).__name__}' object has no attribute 'change_lane'"
    )
