"""Traffic signal controller condition: is a junction in a named phase?"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import carla

from ..coordinate.traffic_light import (
    find_traffic_lights_for_lanelet2_id,
    junction_group_of,
)
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["TrafficSignalControllerCondition"]


class TrafficSignalControllerCondition(BaseCondition):
    """Fire when a junction is in the phase that gives one approach green.

    The counterpart of
    :class:`~autoware_carla_scenario.TrafficSignalControllerAction`, and the
    question it answers is about the whole junction rather than one light:
    *is the named approach the one that is going?*  A single-signal check
    cannot tell "green for us" from "green for us and green for the crossing
    traffic as well", and the second is a junction state the ego should never
    be tested against by accident.

    Args:
        lanelet2_regulatory_element_id: The approach the phase gives green,
            named by the Lanelet2 regulatory element its signals belong to --
            the same id, and the same name for it, that
            :class:`~autoware_carla_scenario.TrafficSignalAction` takes for one
            light.
        label: Human-readable identifier for this condition.
    """

    def __init__(self, lanelet2_regulatory_element_id: int, *, label: str) -> None:
        super().__init__(label=label)
        self._lanelet2_regulatory_element_id = lanelet2_regulatory_element_id
        #: Whether the id failing to resolve has already been reported.  It is
        #: checked every tick and the answer does not change between them, so
        #: an unguarded warning would print at the tick rate for a whole run.
        self._warned_unresolved = False

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result while the junction holds the named phase.

        ``None`` covers two different situations, and they are logged
        differently because only one of them is the scenario's own answer:

        * **The id resolves to nothing.**  A setup problem -- a mistyped
          regulatory element, or a map that is not loaded -- and reported once
          as a warning, because a condition that never fires for a whole run
          should say why rather than leave the reader to guess between this
          and the phase simply never arriving.
        * **The junction is in some other phase.**  The ordinary answer, true
          on most ticks of most runs, so it is logged at debug and names the
          lights that disagreed: which light is wrong is the first thing
          anyone asks.

        Both still return ``None`` rather than a failing result.  An action
        treats any non-``None`` result as its trigger having fired, so a
        ``passed=False`` here would start the action it is meant to hold back.
        """
        green = find_traffic_lights_for_lanelet2_id(
            world, self._lanelet2_regulatory_element_id
        )
        if not green:
            if not self._warned_unresolved:
                self._warned_unresolved = True
                logger.warning(
                    "TrafficSignalControllerCondition [%s]: Lanelet2 "
                    "regulatory element %d resolves to no traffic light, so "
                    "this condition cannot fire. Check the id, or that the "
                    "map is loaded.",
                    self.label,
                    self._lanelet2_regulatory_element_id,
                )
            return None

        green_ids = {light.id for light in green}
        wrong: list[str] = []
        for light in junction_group_of(green):
            expected = (
                carla.TrafficLightState.Green
                if light.id in green_ids
                else carla.TrafficLightState.Red
            )
            state = light.get_state()
            if state != expected:
                wrong.append(f"{light.get_opendrive_id()}={state}")

        if wrong:
            logger.debug(
                "TrafficSignalControllerCondition [%s]: not the phase of "
                "lanelet2 %d -- %s",
                self.label,
                self._lanelet2_regulatory_element_id,
                ", ".join(wrong),
            )
            return None

        return ScenarioResult(
            passed=True,
            message=(
                f"Junction of lanelet2 {self._lanelet2_regulatory_element_id} is in its "
                f"phase: that approach green, the rest red"
            ),
            elapsed_seconds=elapsed,
        )

    def get_details(self) -> dict[str, Any]:
        return {"lanelet2_regulatory_element_id": self._lanelet2_regulatory_element_id}
