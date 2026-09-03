"""Base class for conditional actions executed during scenario tick loops."""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from ..action_state import ActionState
from ..conditions import BaseCondition
from ..conditions.always_true import AlwaysTrueCondition

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

#: Re-exported so ``from .base import ActionState`` keeps working for callers
#: that already have the action in hand.
__all__ = ["ActionState", "BaseAction", "TickTiming"]


class TickTiming(enum.Enum):
    """When the action is evaluated within the tick loop."""

    PRE_TICK = "pre_tick"
    POST_TICK = "post_tick"


class BaseAction(ABC):
    """Abstract base for actions that fire when a condition is met.

    Subclasses must implement :meth:`execute`.  The action is registered on a
    :class:`~autoware_carla_scenario.BaseScenario` via
    :meth:`~autoware_carla_scenario.BaseScenario.register_pre_tick` or
    :meth:`~autoware_carla_scenario.BaseScenario.register_post_tick` depending
    on *timing*.

    Args:
        condition: A :class:`BaseCondition` whose :meth:`check` is called each
            tick.  When ``check`` returns a non-``None`` result the action's
            :meth:`execute` is invoked.  Defaults to
            :class:`AlwaysTrueCondition` (fires unconditionally).
        timing: Whether to run on the pre-tick or post-tick phase.
        once: If ``True`` (default), the action fires at most once.  After
            ``execute`` has been called the condition is no longer evaluated.

    Firing and finishing are two different moments.  :meth:`execute` only
    *starts* the work -- ``force_lane_change`` returns long before the vehicle
    is in the next lane -- so an action stays observable afterwards through
    :meth:`is_finished`, and :attr:`state` reports where it is in the
    :class:`ActionState` lifecycle.  Anything reacting to a manoeuvre having
    finished has to wait for :attr:`ActionState.COMPLETE`; :attr:`done` only
    says the command went out.
    """

    def __init__(
        self,
        label: str,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.POST_TICK,
        *,
        once: bool = True,
    ) -> None:
        if not label:
            raise ValueError(
                f"{type(self).__name__}: label must not be empty. "
                "Provide a non-empty string to identify this action."
            )
        self.label = label
        self._condition = condition if condition is not None else AlwaysTrueCondition()
        self._timing = timing
        self._once = once
        self._done = False
        self._lifecycle = ActionState.STANDBY
        #: Elapsed time at which the current run started, or ``None`` in standby.
        self._running_since: Optional[float] = None

    @property
    def timing(self) -> TickTiming:
        """The tick phase this action is bound to."""
        return self._timing

    @property
    def done(self) -> bool:
        """Whether this action has already fired (relevant when *once=True*)."""
        return self._done

    @property
    def state(self) -> ActionState:
        """Where this action is in its :class:`ActionState` lifecycle.

        Stored as ``_lifecycle`` rather than ``_state`` because subclasses
        already use ``_state`` for their own payload -- ``TrafficSignalAction``
        holds the light state it sets there.
        """
        return self._lifecycle

    def is_finished(self, world: "carla.World", running_for: float) -> bool:
        """Whether the work :meth:`execute` started has met its completion criteria.

        Called every tick while the action is
        :attr:`~ActionState.RUNNING`, so a manoeuvre that takes seconds to play
        out can say when it is genuinely over.

        Args:
            world: The CARLA world instance.
            running_for: Seconds since :meth:`execute` was called.

        Returns:
            ``True`` once the work is complete, ``False`` while it is still
            under way.

        The default is instantaneous: an action with nothing to observe after
        :meth:`execute` -- setting a traffic light, attaching a sensor -- is
        complete the moment it has run, which is what every action did before
        this hook existed.
        """
        return True

    @abstractmethod
    def execute(self, world: "carla.World") -> None:
        """Perform the action.

        Called when the condition is satisfied.

        Args:
            world: The CARLA world instance.
        """
        ...

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Evaluate the condition and, if met, run :meth:`execute`.

        Called by the scenario runner's tick loop with the current elapsed
        time so that time-based conditions work correctly.

        Args:
            world: The CARLA world instance.
            elapsed: Seconds elapsed since the tick loop started.
        """
        # Transitions are resolved before the trigger is looked at, so a
        # repeating action is back in standby in time to fire again on this
        # tick -- `once=False` has always meant "every tick the condition
        # holds", and going through the lifecycle must not slow that down.
        just_started = False
        if self._lifecycle is ActionState.START_TRANSITION:
            self._lifecycle = ActionState.RUNNING
            just_started = True

        if self._lifecycle is ActionState.RUNNING:
            # Compared against None, not truth-tested: an action triggered at
            # elapsed 0.0 has a falsy start time and would otherwise measure
            # its own runtime as zero forever.
            started = self._running_since
            running_for = elapsed if started is None else elapsed - started
            if not self.is_finished(world, running_for):
                # The trigger is deliberately not re-evaluated while running,
                # so a repeating action cannot start a second run on top of one
                # that has not finished.
                return
            self._lifecycle = ActionState.END_TRANSITION
            if not just_started:
                # A manoeuvre that genuinely took time holds `endTransition`
                # for a tick, so a condition can watch for it.  One that
                # finished the moment it was commanded never ran, and pausing
                # on a transition it passed straight through would only make
                # every instantaneous action a tick slower to repeat.
                return

        if self._lifecycle is ActionState.END_TRANSITION:
            # A repeating action goes back to standby to be triggered again,
            # which is what OpenSCENARIO does for an element with a maximum
            # execution count above one.
            self._running_since = None
            self._lifecycle = (
                ActionState.COMPLETE if self._once else ActionState.STANDBY
            )

        if self._lifecycle is not ActionState.STANDBY:
            return

        result = self._condition.check(world, elapsed)
        if result is None:
            return
        logger.info(
            "%s triggered: %s",
            type(self).__name__,
            result.message,
        )
        self.execute(world)
        self._done = True
        self._running_since = elapsed
        # Held for this whole tick, so a condition watching `startTransition`
        # has a tick on which to see it.
        self._lifecycle = ActionState.START_TRANSITION
