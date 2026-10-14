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

    Subclasses set :attr:`driven_by` to name that stack in the refusals, and
    call :meth:`_terminate` when it fails in a way the run cannot recover from.
    """

    #: Something else drives; TrafficManager must keep its hands off this actor.
    use_autopilot: bool = False

    #: What drives this ego, as the refusals name it ("Autoware", "The
    #: policy"): the subject of "... decides its own speed".
    driven_by: str = "The stack driving it"

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
        logger.warning(
            "%s: a lane change was asked for, but this entity is not driven by "
            "the TrafficManager, so forcing one there would do nothing. "
            "%s decides its own manoeuvres.",
            type(self).__name__,
            self.driven_by,
        )

    def set_desired_speed(self, world: "carla.World", speed_kmh: float) -> None:
        """Refuse a TrafficManager speed: nothing here is driven by it.

        A desired speed set on the TrafficManager would be ignored -- the
        runner passes this ego in ``skip_actor_ids`` precisely so the
        TrafficManager does not touch it -- while the action reported progress
        and completion, which is worse than refusing.
        """
        logger.warning(
            "%s: a speed was set, but this entity is not driven by the "
            "TrafficManager, so the command would go to an actor it does not "
            "control. %s decides its own speed.",
            type(self).__name__,
            self.driven_by,
        )

    def turn_at_junction(
        self, world: "carla.World", direction: "TurnDirection", **kwargs: Any
    ) -> None:
        """Refuse a TrafficManager route: nothing here is driven by it."""
        logger.warning(
            "%s: a turn was asked for, but this entity is not driven by the "
            "TrafficManager, so setting a route there would do nothing.",
            type(self).__name__,
        )
