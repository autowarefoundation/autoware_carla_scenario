"""Traffic signal controller condition: is a junction showing a named phase?"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..signals.registry import find_signal_controller
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

__all__ = ["TrafficSignalControllerCondition"]


class TrafficSignalControllerCondition(BaseCondition):
    """Fire while a junction's controller is showing a named phase.

    The counterpart of
    :class:`~autoware_carla_scenario.TrafficSignalControllerAction`, and the
    question it answers is about the whole junction rather than one light:
    *is this the phase that is running?*  A single-signal check cannot tell
    "green for us" from "green for us and green for the crossing traffic as
    well", and it cannot see an amber interval as a thing with a name at all.

    Read off the controller rather than off the lights.  The controller is what
    decided the lights, so asking it is asking the source: a check that
    re-derived the phase by reading every light back would also report a phase
    when some other actor had happened to set the same colours, which is not
    the same statement.

    Args:
        controller: Name of the controller, as declared on the map.
        phase: Name of the phase to wait for.
        label: Human-readable identifier for this condition.
    """

    def __init__(self, controller: str, phase: str, *, label: str) -> None:
        super().__init__(label=label)
        self._controller = controller
        self._phase = phase
        #: Whether the controller being absent has already been reported.  The
        #: condition is checked every tick and the answer does not change
        #: between them, so an unguarded warning would print at the tick rate.
        self._warned_missing = False

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result while the named phase is the one showing.

        ``None`` covers three situations, and they are logged differently
        because only one of them is the scenario's own answer:

        * **No such controller.**  A setup problem -- a mistyped name, or a map
          that declares no controllers -- warned once, because a condition that
          never fires for a whole run should say why.
        * **The cycle has not started.**  True for the first ticks of a
          controller that waits on another one's delay.  Silent: it is about to
          change by itself.
        * **Some other phase is showing.**  The ordinary answer, true for most
          of every cycle, and logged at debug naming the phase that *is*
          showing -- which is the first thing anyone asks.

        All three return ``None`` rather than a failing result: an action
        treats any non-``None`` result as its trigger having fired, so a
        ``passed=False`` here would start the action it is meant to hold back.
        """
        controller = find_signal_controller(self._controller)
        if controller is None:
            if not self._warned_missing:
                self._warned_missing = True
                logger.warning(
                    "TrafficSignalControllerCondition [%s]: no controller "
                    "named '%s' is running, so this condition cannot fire. "
                    "Check the name against the map's declared controllers.",
                    self.label,
                    self._controller,
                )
            return None

        current = controller.current_phase
        if current is None:
            return None

        if current.name != self._phase:
            logger.debug(
                "TrafficSignalControllerCondition [%s]: '%s' is in phase "
                "'%s', not '%s'",
                self.label,
                self._controller,
                current.name,
                self._phase,
            )
            return None

        return ScenarioResult(
            passed=True,
            message=(f"Controller '{self._controller}' is in phase '{self._phase}'"),
            elapsed_seconds=elapsed,
        )

    def get_details(self) -> dict[str, Any]:
        return {"controller": self._controller, "phase": self._phase}
