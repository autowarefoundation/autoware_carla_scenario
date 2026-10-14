"""The base of every ego that something other than the TrafficManager drives.

The ego entities split along one line: who drives.
:class:`~autoware_carla_scenario.entity.ego.EgoVehicle` itself is driven by the
run's traffic backend (the TrafficManager by default); every other ego is driven
by a stack of its own -- Autoware, an external driver policy, ROS 2 nodes -- and
those share everything that follows from that, which lives here:

* :attr:`use_autopilot` is ``False``, so the runner never puts the actor under
  the TrafficManager;
* a TrafficManager manoeuvre asked of it (a lane change, a desired speed, a
  turn) is refused and logged rather than delegated to a backend that is not
  driving it;
* it can ask to end the run (:meth:`_terminate`), with a reason.

The hierarchy::

    EgoVehicle                      driven by the traffic backend
    └── ExternallyDrivenEgo         driven by a stack of its own
        ├── AutowareEntity          (nothing; left standing for an outside stack)
        ├── CarlaDriverEntity       (an egodriver policy over gRPC)
        ├── Ros2EgoEntity           (ROS 2 nodes over DDS)
        └── BridgedEgoEntity        (a stack that reaches the AutowareBridge)
            └── AutowareEgoEntity   (Autoware through autoware_carla_interface)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..traffic.base import LaneChangeDirection, TurnDirection

from .ego import EgoVehicle

logger = logging.getLogger(__name__)

__all__ = ["ExternallyDrivenEgo"]


class ExternallyDrivenEgo(EgoVehicle):
    """Ego vehicle driven by a stack of its own rather than the TrafficManager.

    Subclasses set :attr:`stack_name` to name that stack in what they log, and
    call :meth:`_terminate` when it fails in a way the run cannot recover from.
    """

    #: Something else drives; TrafficManager must keep its hands off this actor.
    use_autopilot: bool = False

    #: The stack that drives this ego, as its messages name it: "the
    #: Autoware stack", "the driver policy stack".
    stack_name: str = "external"

    _termination_requested: bool
    _termination_reason: Optional[str]

    def __init__(self) -> None:
        super().__init__()
        self._termination_requested = False
        self._termination_reason = None

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    @property
    def termination_requested(self) -> bool:
        """Whether the stack driving this ego asked, or failed, to end the run."""
        return self._termination_requested

    @property
    def termination_reason(self) -> Optional[str]:
        """Why the run is to end, once :attr:`termination_requested`."""
        return self._termination_reason

    def _terminate(self, reason: str) -> None:
        """Ask the runner to end the scenario, for *reason*."""
        logger.warning("%s - requesting termination", reason)
        self._termination_reason = reason
        self._termination_requested = True

    def _reset_termination(self) -> None:
        """Forget a request from an earlier attempt of a retried scenario."""
        self._termination_requested = False
        self._termination_reason = None

    # ------------------------------------------------------------------
    # Manoeuvres
    # ------------------------------------------------------------------

    def change_lane(
        self, world: "carla.World", direction: "LaneChangeDirection"
    ) -> None:
        """Refuse a TrafficManager manoeuvre: nothing here is driven by it.

        The stack driving this ego plans its own manoeuvres; one forced through
        the TrafficManager would do nothing, or be overwritten by the next
        control the stack applies.
        """
        self._refuse(
            "a lane change was asked for", "forcing one there would do nothing"
        )

    def set_desired_speed(self, world: "carla.World", speed_kmh: float) -> None:
        """Refuse a TrafficManager speed: nothing here is driven by it.

        A desired speed set on the TrafficManager would be ignored -- the
        runner passes this ego in ``skip_actor_ids`` precisely so the
        TrafficManager does not touch it -- while the action reported progress
        and completion, which is worse than refusing.
        """
        self._refuse(
            "a speed was set",
            "the command would go to an actor it does not control",
        )

    def turn_at_junction(
        self, world: "carla.World", direction: "TurnDirection", **kwargs: Any
    ) -> None:
        """Refuse a TrafficManager route: nothing here is driven by it."""
        self._refuse("a turn was asked for", "setting a route there would do nothing")

    def _refuse(self, request: str, consequence: str) -> None:
        """Log that *request* was refused, and why it would have done nothing."""
        logger.warning(
            "%s: %s, but this entity is not driven by the TrafficManager, so %s. "
            "That is left to the %s stack.",
            type(self).__name__,
            request,
            consequence,
            self.stack_name,
        )
