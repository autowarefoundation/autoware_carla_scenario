"""Base class for conditional actions executed during scenario tick loops."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional

from ..action_state import ActionState
from ..conditions.base import BaseCondition
from ..conditions.always_true import AlwaysTrueCondition
from ..adversity_event import AdversityEvent
from ..control_domain import ControlConflictError, ControlDomain
from ..waypoint_arrival import WaypointArrival

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

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
            run too.  ``None`` (default) makes the action instantaneous: it is
            over on the tick after the one it was commanded on.
        reissue: Whether :meth:`execute` is called again on every tick of the
            run, overriding what the action would decide for itself.  ``None``
            (default) leaves that decision to
            :meth:`~BaseAction.reissues_while_running`.
        overrides: Whether the action, firing to command a control domain of
            an entity another action holds, takes it over, stopping that
            action, rather than raise
            :class:`~autoware_carla_scenario.control_domain.ControlConflictError`
            (default) -- :attr:`overrides`.

    Firing and finishing are two different moments.  :meth:`execute` only
    *starts* the work -- ``force_lane_change`` returns long before the vehicle
    is in the next lane -- so an action stays observable afterwards through
    *until*, and :attr:`state` reports where it is in the :class:`ActionState`
    lifecycle.  Anything reacting to a manoeuvre having finished has to wait for
    :attr:`ActionState.COMPLETE`; :attr:`done` only says the command went out.

    A run has two independent questions, and they are asked separately because
    the answers come from different places:

    * **When does it end?**  *until*, a condition -- the same kind the trigger
      is written in, which is what lets the two be read together.  This is
      about the world.
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
    """

    #: What :meth:`reissues_while_running` answers when nothing was injected
    #: and the action does not decide for itself.  Overriding
    #: :meth:`_reissues_by_default` is the way to decide per instance.
    REISSUES_BY_DEFAULT: ClassVar[bool] = False

    label: str
    _condition: BaseCondition
    _timing: TickTiming
    _once: bool
    _until: Optional[BaseCondition]
    _reissue: Optional[bool]
    _done: bool
    _lifecycle: ActionState
    _overrides: bool
    _claims_control: bool

    def __init__(
        self,
        label: str,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.POST_TICK,
        *,
        once: bool = True,
        until: Optional[BaseCondition] = None,
        reissue: Optional[bool] = None,
        overrides: bool = False,
    ) -> None:
        if not label:
            raise ValueError(
                f"{type(self).__name__}: label must not be empty. "
                "Provide a non-empty string to identify this action."
            )
        self.label = label
        if condition is not None:
            self._condition = condition
        else:
            self._condition = AlwaysTrueCondition()
        self._timing = timing
        self._once = once
        self._until = until
        self._reissue = reissue
        self._done = False
        self._lifecycle = ActionState.STANDBY
        self._overrides = overrides
        # False for an action another drives as a part of its own run (an
        # adversity's path), whose domains that one holds.
        self._claims_control = True

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
    def importance_sampling_weight(self) -> float:
        """The likelihood ratio the action has put on the run: ``1`` unless it
        samples what happens importance-sampled (NADE's
        :class:`~autoware_carla_scenario.AdversarialTrafficAction`).  A
        scenario's weight is the product of its actions'."""
        return 1.0

    @property
    def importance_sampled(self) -> bool:
        """Whether the action samples what happens importance-sampled, so that
        its run's outcome is recorded with :attr:`importance_sampling_weight`;
        ``False`` but for NADE's
        :class:`~autoware_carla_scenario.AdversarialTrafficAction`."""
        return False

    def take_adversity_events(self) -> list[AdversityEvent]:
        """The adversities the action sampled since it was last asked, for
        the run's log; none unless it samples them."""
        none: list[AdversityEvent] = []
        return none

    def take_waypoint_arrivals(self) -> list[WaypointArrival]:
        """The timed waypoints reached since the last call, for the run's
        log; none unless the action paces a route."""
        none: list[WaypointArrival] = []
        return none

    def take_spawns(self) -> list[tuple[str, int]]:
        """The ``(role, actor id)`` brought in since the last call, for the
        run's log; none unless the action brings entities in."""
        none: list[tuple[str, int]] = []
        return none

    def take_despawns(self) -> list[tuple[str, int]]:
        """The ``(role, actor id)`` removed since the last call, for the
        run's log; none unless the action removes entities."""
        none: list[tuple[str, int]] = []
        return none

    @property
    def sets_tire_friction(self) -> bool:
        """Whether the action sets the vehicles' grip (an
        :class:`~autoware_carla_scenario.EnvironmentAction` on a wet road):
        a scenario registering one has the vehicles spawned later follow it."""
        return False

    @property
    def needs_tick_loop(self) -> bool:
        """Whether the action has to run on the tick loop, because it keeps
        acting: one registered for initialization is put on the loop instead
        (:meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_init`).
        ``False`` unless the action says."""
        return False

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

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The ``(entity, domain)`` the action commands when it fires
        (:mod:`~autoware_carla_scenario.control_domain`); none unless it
        moves, steers, lights or animates an entity.  Asked as it fires, so
        the answer may depend on what the entity is."""
        none: list[tuple[str, ControlDomain]] = []
        return none

    def _take_control(self, world: "carla.World", at: float) -> None:
        """Take the domains the action commands as it fires at *at*: for its
        run if it runs on, else for this tick.

        Raises:
            ControlConflictError: If another action holds one of them.
        """
        if not self._claims_control:
            return
        claims = self._control_claims()
        if not claims:
            return
        runs = self._until is not None or self.reissues_while_running
        for stopped in _SLOTS.take(self, claims, at, runs):
            stopped._stop_overridden(world, self._control_name)

    @property
    def _control_name(self) -> str:
        """How a control conflict names the action: its class and label."""
        return f"{type(self).__name__} '{self.label}'"

    @property
    def overrides(self) -> bool:
        """Whether the action, firing to command a control domain another
        action holds, takes it over -- stopping that action, as OpenSCENARIO
        has every action do -- rather than raise
        :class:`~autoware_carla_scenario.control_domain.ControlConflictError`
        (the default).  Read as the action fires."""
        return self._overrides

    def _stop_overridden(self, world: "carla.World", by: str) -> None:
        """Stop the run, overridden by the action *by*: OpenSCENARIO's stop
        transition, to ``completeState``; what it held is given back."""
        logger.info("%s overridden by %s", self._control_name, by)
        self._lifecycle = ActionState.COMPLETE
        self._stopped(world)

    def _stopped(self, world: "carla.World") -> None:
        """Undo what the run left in place, now that it is stopped before its
        end -- an entity it took from its driver, given back.  Nothing unless
        the action takes one."""
        return

    def _release_control(self) -> None:
        """Give back the domains the action took, as its run ends."""
        _SLOTS.release(self)

    def close(self) -> None:
        """Close the action's trigger and its *until* condition, at the run's
        end (:meth:`BaseCondition.close`), and give back what it controls."""
        self._release_control()
        self._condition.close()
        if self._until is not None:
            # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
            until: BaseCondition = self._until
            until.close()

    def reset(self) -> None:
        """Back to standby, never fired, before a run begins, and reset the
        trigger and the *until* condition (:meth:`BaseCondition.reset`).

        An action a scenario registers in ``__init__`` is the same object on
        every run of it -- a retry, or a second ``run_all()``
        (:meth:`~autoware_carla_scenario.BaseScenario._reset_registrations`)
        -- so the runner resets it before each one: a ``once`` action that
        fired in an earlier run fires again in this one.  An action that
        keeps state of its own across ticks clears it here too.
        """
        self._done = False
        self._lifecycle = ActionState.STANDBY
        self._condition.reset()
        if self._until is not None:
            # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
            until: BaseCondition = self._until
            until.reset()

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

    def _settle_start(self, world: "carla.World", elapsed: float) -> None:
        """Step one that fired out of ``startTransition`` without running it
        again or looking at its trigger: under way, or -- one that ended the
        moment it was commanded and does not reissue -- complete (standby for
        a repeating one).  For the init phase, which has no ticks for an
        action's lifecycle to step through."""
        if self._lifecycle is not ActionState.START_TRANSITION:
            return
        # Nothing ticks it again: what it commanded is left to the entity's
        # driver (an action that keeps acting is put on the tick loop
        # instead), so it controls nothing from here.
        self._release_control()
        self._lifecycle = ActionState.RUNNING
        if self.reissues_while_running:
            return
        if self._until is None or self._until.check(world, elapsed) is not None:
            self._lifecycle = (
                ActionState.COMPLETE if self._once else ActionState.STANDBY
            )

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
            # No `until` means there was nothing to wait for.
            finished = (
                self._until is None or self._until.check(world, elapsed) is not None
            )
            if not finished:
                # The trigger is deliberately not re-evaluated while running,
                # so a repeating action cannot start a second run on top of one
                # that has not finished.
                return
            self._lifecycle = ActionState.END_TRANSITION
            self._release_control()
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
        # Before it commands anything: a conflict raises with the entity
        # untouched.
        self._take_control(world, elapsed)
        self.execute(world)
        self._done = True
        # Held for this whole tick, so a condition watching `startTransition`
        # has a tick on which to see it.
        self._lifecycle = ActionState.START_TRANSITION


@dataclass
class _Hold:
    """A control domain taken by an action."""

    owner: BaseAction
    at: float
    runs: bool


class _ControlSlots:
    """Who holds each control domain of each entity
    (:mod:`~autoware_carla_scenario.control_domain`)."""

    _holds: dict[tuple[str, str], _Hold]

    def __init__(self) -> None:
        self._holds = {}

    def take(
        self,
        owner: BaseAction,
        claims: list[tuple[str, ControlDomain]],
        at: float,
        runs: bool,
    ) -> list[BaseAction]:
        """Take each ``(entity, domain)`` of *claims* for *owner*, fired at
        scenario time *at*: until it is :meth:`released <release>` if it
        *runs* on, else for that tick alone.

        Returns:
            The actions it took a domain over from (*owner* overrides), each
            released of everything it held, to be stopped.

        Raises:
            ControlConflictError: If another action holds one of them and
                *owner* does not override; none is taken then.
        """
        overridden: list[BaseAction] = []
        for entity, domain in claims:
            held = self.holder(entity, domain, at)
            if held is None:
                continue
            holder: BaseAction = held
            if not owner.overrides:
                raise ControlConflictError(
                    entity, domain, holder._control_name, owner._control_name
                )
            if not any(stopped is holder for stopped in overridden):
                overridden.append(holder)
        for stopped in overridden:
            self.release(stopped)
        for entity, domain in claims:
            self._holds[(entity, domain.value)] = _Hold(owner, at, runs)
        return overridden

    def release(self, owner: BaseAction) -> None:
        """Give back what *owner* took."""
        for key in [k for k, hold in self._holds.items() if hold.owner is owner]:
            del self._holds[key]

    def holder(
        self, entity: str, domain: ControlDomain, at: float
    ) -> Optional[BaseAction]:
        """The action holding *domain* of *entity* at scenario time *at*, or
        ``None``."""
        key = (entity, domain.value)
        if key not in self._holds:
            return None
        hold = self._holds[key]
        if hold.runs or hold.at == at:
            return hold.owner
        return None

    def clear(self) -> None:
        """Forget every hold."""
        self._holds = {}


#: Who controls which entity: process-wide, as the entity registry is.
_SLOTS = _ControlSlots()


def control_holder(entity: str, domain: ControlDomain, at: float) -> Optional[str]:
    """The :attr:`~BaseAction._control_name` of the action holding *domain* of
    *entity* at scenario time *at*, or ``None``."""
    held = _SLOTS.holder(entity, domain, at)
    if held is None:
        return None
    holder: BaseAction = held
    return holder._control_name


def reset_control_slots() -> None:
    """Forget who controls which entity, as a run begins: what an action of a
    previous run that never finished held is not this run's."""
    _SLOTS.clear()
