"""Traffic signal controller action: put a junction into a named phase."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import carla

from ..conditions import BaseCondition
from ..coordinate.traffic_light import (
    find_traffic_lights_for_lanelet2_id,
    junction_group_of,
)
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["TrafficSignalControllerAction"]


class TrafficSignalControllerAction(BaseAction):
    """Put a junction's signals into the phase that gives one approach green.

    The difference from :class:`~autoware_carla_scenario.TrafficSignalAction`
    is not bureaucratic.  Setting lights one at a time can leave two
    conflicting approaches green -- a state no real junction reaches -- so a
    scenario built that way tests the ego against a road that cannot exist.
    A *phase* is the unit that keeps a junction consistent, and this action
    applies one: the named approach green, every other light of its group red.

    A phase is addressed by the Lanelet2 regulatory element whose signals it
    makes green, rather than by a phase table on the document.  OpenSCENARIO
    declares phases under ``RoadNetwork/TrafficSignals``, which a
    ``ScenarioDocument`` has no place for; naming the phase by its leading
    signal needs no such place and no new IR.

    Args:
        green_lanelet2_id: Lanelet2 regulatory element id of the approach that
            gets green.  Every other light in the same CARLA group goes red.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        freeze: Freeze the group so the TrafficManager does not resume cycling
            it.  On by default: a phase a scenario set and the simulator then
            moved on from is not a phase the scenario can assert about.
    """

    def __init__(
        self,
        green_lanelet2_id: int,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "traffic_signal_controller",
        once: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._green_lanelet2_id = green_lanelet2_id
        self._freeze = freeze

    def execute(self, world: "carla.World") -> None:
        """Set the named approach green and the rest of its junction red."""
        green = find_traffic_lights_for_lanelet2_id(world, self._green_lanelet2_id)
        if not green:
            logger.warning(
                "TrafficSignalControllerAction [%s]: no traffic light found "
                "for Lanelet2 regulatory element ID %d",
                self.label,
                self._green_lanelet2_id,
            )
            return

        green_ids = {light.id for light in green}
        group = junction_group_of(green)
        for light in group:
            light.set_state(
                carla.TrafficLightState.Green
                if light.id in green_ids
                else carla.TrafficLightState.Red
            )
            light.freeze(self._freeze)

        logger.info(
            "TrafficSignalControllerAction [%s]: lanelet2 %d green, %d other "
            "light(s) in its junction red (freeze=%s)",
            self.label,
            self._green_lanelet2_id,
            len(group) - len(green_ids),
            self._freeze,
        )
