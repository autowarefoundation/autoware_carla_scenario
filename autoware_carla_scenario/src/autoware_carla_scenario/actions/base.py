"""Base class for conditional actions executed during scenario tick loops."""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional

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
        until: A :class:`BaseCondition` that ends the run, on the first tick
            it returns a non-``None`` result -- the same "a result means it
            fired" convention the trigger uses, so a failing result ends the
            run too.  ``None`` (default) leaves the action instantaneous unless
            it overrides :meth:`is_finished`.
        reissue: Whether :meth:`execute` is called again on every tick of the
            run, overriding what the action would decide for itself.  ``None``
            (default) leaves that decision to
            :meth:`~BaseAction.reissues_while_running`.

    Firing and finishing are two different moments.  :meth:`execute` only
    *starts* the work -- ``force_lane_change`` returns long before the vehicle
    is in the next lane -- so an action stays observable afterwards through
    :meth:`is_finished`, and :attr:`state` reports where it is in the
    :class:`ActionState` lifecycle.  Anything reacting to a manoeuvre having
    finished has to wait for :attr:`ActionState.COMPLETE`; :attr:`done` only
    says the command went out.

    A run has two independent questions, and they are asked separately because
    the answers come from different places:

    * **When does it end?**  *until* (a condition), or :meth:`is_finished` (a
      predicate) when no *until* is given.  This is about the world.
    * **Does :meth:`execute` repeat while it runs?**
      :meth:`reissues_while_running`.  This is about the *command*, not the
      manoeuvre: whether the thing the action called holds the intent by itself
      or has to be re-sent.  A TrafficManager target persists until it is
      changed; a command that walks a target towards a goal has to be re-sent
      on every tick to move it at all.

    Keeping them apart is what lets both shapes exist.  An action that hands
    work to the simulator and then watches for it to land -- ``force_lane_change``
    returns long before the vehicle is in the next lane -- says when it is over
    without ever repeating its command.  An action that keeps acting says both.

    Whether a command has to be reissued is a property of **whatever is driving
    the entity**, not of the scenario, so it has an injection point rather than
    being fixed by the action's class: see :meth:`reissues_while_running`.

    *until* and *reissue* are both independent of *once*, which says whether
    another run may begin once this one has ended.

    An action that reissues has an :meth:`execute` that runs many times, and
    must be written for it -- idempotent, or advancing the work by one tick.
    When *until* is given it decides the run on its own and :meth:`is_finished`
    is not consulted.
    """

    #: What :meth:`reissues_while_running` answers when nothing was injected
    #: and the action does not decide for itself.  Overriding
    #: :meth:`_reissues_by_default` is the way to decide per instance.
    REISSUES_BY_DEFAULT: ClassVar[bool] = False

    def __init__(
        self,
        label: str,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.POST_TICK,
        *,
        once: bool = True,
        until: Optional[BaseCondition] = None,
        reissue: Optional[bool] = None,
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
        self._until = until
        self._reissue = reissue
        self._done = False
        self._lifecycle = ActionState.STANDBY
        #: Elapsed time at which the current run started.  Only read while
        #: running, and always written on the way in, so there is no "not
        #: started" value to confuse with an action triggered at 0.0.
        self._running_since: float = 0.0

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

    @property
    def reissues_while_running(self) -> bool:
        """Whether :meth:`execute` is called again on every tick of the run.

        Resolved in the order the knowledge is available:

        1. the *reissue* argument, when one was given.  That is the injection
           point for whoever knows what is driving the entity -- the same
           action against a backend whose command persists needs no repeat,
           and against one that holds only the last value needs one every tick.
        2. :meth:`_reissues_by_default`, which is the action's own answer.
        3. :attr:`REISSUES_BY_DEFAULT`, which that returns unless overridden.

        Not an override point itself: overriding it would lose step 1, and
        with it the ability to say from outside what a backend needs.
        """
        if self._reissue is not None:
            return self._reissue
        return self._reissues_by_default()

    def _reissues_by_default(self) -> bool:
        """The action's own answer for :meth:`reissues_while_running`.

        Override to decide per instance -- on how the action was configured, or
        by asking the entity what is driving it.  A class with one answer can
        set :attr:`REISSUES_BY_DEFAULT` instead.
        """
        return self.REISSUES_BY_DEFAULT

    def is_finished(self, world: "carla.World", running_for: float) -> bool:
        """Whether the work :meth:`execute` started has met its completion criteria.

        Called every tick while the action is
        :attr:`~ActionState.RUNNING`, so a manoeuvre that takes seconds to play
        out can say when it is genuinely over.

        Args:
            world: The CARLA world instance.
            running_for: Simulated seconds since :meth:`execute` was called.

        Returns:
            ``True`` once the work is complete, ``False`` while it is still
            under way.

        Not consulted at all when the action was given an *until* condition,
        which decides the run by itself.

        The default is instantaneous: an action with nothing to observe after
        :meth:`execute` -- setting a traffic light, attaching a sensor -- is
        complete the moment it has run, which is what every action did before
        this hook existed.
        """
        return True

    @abstractmethod
    def execute(self, world: "carla.World") -> None:
        """Perform the action.

        Called when the condition is satisfied, and then again on every tick
        of the run when :meth:`reissues_while_running` is true -- see
        :class:`BaseAction` for what that asks of an implementation.

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
            elapsed: Simulated seconds since the run began.
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
            # Read once: an override may ask the entity what drives it, and the
            # answer has to be the same for both uses below.
            reissues = self.reissues_while_running
            if reissues:
                # Every tick of the run, this one included.  The trigger's
                # `execute` ran on the *previous* tick -- entering
                # `runningState` is what this tick did -- so standing down here
                # would leave a frame with no command in it.
                self.execute(world)
            # Asked second, and separately: what ends the run is a question
            # about the world, not about whether the command had to be re-sent.
            if self._until is not None:
                finished = self._until.check(world, elapsed) is not None
            else:
                finished = self.is_finished(world, elapsed - self._running_since)
            if not finished:
                # The trigger is deliberately not re-evaluated while running,
                # so a repeating action cannot start a second run on top of one
                # that has not finished.
                return
            self._lifecycle = ActionState.END_TRANSITION
            if not just_started or self._until is not None or reissues:
                # A manoeuvre that genuinely took time holds `endTransition`
                # for a tick, so a condition can watch for it.  An action given
                # an `until` condition, or one that reissues, is never the
                # instantaneous case: the first was waiting on the world and
                # the second was acting.  Holding the transition is also what
                # stops a reissuing action from beginning its next run in the
                # tick it ended this one, which would put two commands in one
                # frame.  One that finished the moment it was commanded never
                # ran, and pausing on a transition it passed straight through
                # would only make every instantaneous action a tick slower to
                # repeat.
                return

        if self._lifecycle is ActionState.END_TRANSITION:
            # A repeating action goes back to standby to be triggered again,
            # which is what OpenSCENARIO does for an element with a maximum
            # execution count above one.
            self._running_since = 0.0
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
