"""Traffic signal controller action: put a junction into a named phase."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..conditions import BaseCondition
from ..signals.registry import find_signal_controller
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

__all__ = ["TrafficSignalControllerAction"]


class TrafficSignalControllerAction(BaseAction):
    """Jump a junction's controller to one of its declared phases.

    The difference from :class:`~autoware_carla_scenario.TrafficSignalAction`
    is not bureaucratic.  That one sets the lights it is given to the colour it
    is given, so a junction has to be assembled a light at a time -- and a
    scenario that forgets one, or that is interrupted between two of them,
    leaves two conflicting approaches green, a state no real road reaches.  A
    *phase* is the unit that keeps a junction consistent: it names every signal
    its controller drives, so applying it puts the whole junction into a known
    state at once.

    The phases themselves are declared on the map (``map.traffic_signal_
    controllers`` in the document), not here, because a junction's cycle is a
    property of the road network -- which is where OpenSCENARIO keeps it too.
    This action only says *which* of them to show, by name.

    The cycle carries on from the phase this jumps to.  A junction forced green
    does not therefore stay green: that would be a second decision, and a
    scenario that wants it makes it by naming a phase whose duration is long
    enough, or by declaring a controller with one phase.

    Args:
        controller: Name of the controller, as declared on the map.
        phase: Name of the phase to show.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
    """

    def __init__(
        self,
        controller: str,
        phase: str,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "traffic_signal_controller",
        once: bool = True,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._controller = controller
        self._phase = phase

    def execute(self, world: "carla.World") -> None:
        """Show the named phase, and let the cycle continue from it."""
        controller = find_signal_controller(self._controller)
        if controller is None:
            logger.warning(
                "TrafficSignalControllerAction [%s]: no controller named '%s' "
                "is running; the map declares none by that name",
                self.label,
                self._controller,
            )
            return

        if controller.change_phase_to(self._phase, world):
            logger.info(
                "TrafficSignalControllerAction [%s]: '%s' now in phase '%s'",
                self.label,
                self._controller,
                self._phase,
            )
