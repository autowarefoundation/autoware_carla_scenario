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
        green_lanelet2_id: Lanelet2 regulatory element id of the approach that
            the phase gives green.
        label: Human-readable identifier for this condition.
    """

    def __init__(self, green_lanelet2_id: int, *, label: str) -> None:
        super().__init__(label=label)
        self._green_lanelet2_id = green_lanelet2_id

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result while the junction holds the named phase.

        Returns ``None`` while the id resolves to nothing -- the map may not be
        loaded yet, and that is not the same as the phase being wrong.
        """
        green = find_traffic_lights_for_lanelet2_id(world, self._green_lanelet2_id)
        if not green:
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
            return None

        return ScenarioResult(
            passed=True,
            message=(
                f"Junction of lanelet2 {self._green_lanelet2_id} is in its "
                f"phase: that approach green, the rest red"
            ),
            elapsed_seconds=elapsed,
        )

    def get_details(self) -> dict[str, Any]:
        return {"green_lanelet2_id": self._green_lanelet2_id}
