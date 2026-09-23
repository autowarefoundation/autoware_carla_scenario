"""``until`` and *reissue*: the two questions a run asks, kept apart.

*until* says **when the run ends**, in the same vocabulary the trigger is
written in.  *reissue* says **whether ``execute`` repeats** while it runs,
which is a fact about the command rather than about the manoeuvre -- a
TrafficManager target persists until it is changed, a command that walks a
target towards a goal has to be re-sent to move at all.

They are separate because the answers come from different places, and because
both shapes have to exist: an action that hands work to the simulator and
watches for it to land says only the first, and one that keeps acting says
both.
"""

from __future__ import annotations

from typing import Optional

from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions.base import BaseAction, TickTiming
from autoware_carla_scenario.conditions.always_true import AlwaysTrueCondition
from autoware_carla_scenario.conditions.base import BaseCondition, ScenarioResult


class _AfterNChecks(BaseCondition):
    """Fires on the *n*-th time it is checked, and every time after."""

    def __init__(self, n: int, *, passed: bool = True) -> None:
        super().__init__(label=f"after_{n}")
        self._n = n
        self._passed = passed
        self.checks = 0

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        self.checks += 1
        if self.checks < self._n:
            return None
        return ScenarioResult(
            passed=self._passed, message="fired", elapsed_seconds=elapsed
        )


class _CountingAction(BaseAction):
    """Records how many times it was executed, and what it was told about."""

    def __init__(
        self,
        until: Optional[BaseCondition] = None,
        condition: Optional[BaseCondition] = None,
        *,
        once: bool = True,
        reissue: Optional[bool] = None,
    ) -> None:
        super().__init__(
            label="counting",
            condition=condition if condition is not None else AlwaysTrueCondition(),
            timing=TickTiming.PRE_TICK,
            once=once,
            until=until,
            reissue=reissue,
        )
        self.executed = 0

    def execute(self, world: object) -> None:
        self.executed += 1


def _tick(action: BaseAction, elapsed: float) -> ActionState:
    action.tick(object(), elapsed)
    return action.state


class TestWithoutUntil:
    """No condition means nothing to wait for."""

    def test_an_instantaneous_action_is_unchanged(self) -> None:
        action = _CountingAction()

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.COMPLETE
        assert action.executed == 1


class TestReissuing:
    def test_execute_runs_once_on_every_tick_of_the_run(self) -> None:
        """One command per tick, with no frame left empty.

        The trigger's `execute` ran on the tick *before* the action reached
        `runningState`, so the first running tick has to reissue rather than
        count the previous one -- a rate-limited command that skipped a frame
        would stall for it.
        """
        action = _CountingAction(until=_AfterNChecks(4), reissue=True)

        _tick(action, 0.0)  # triggered
        assert action.executed == 1

        assert _tick(action, 0.1) is ActionState.RUNNING
        assert action.executed == 2
        assert _tick(action, 0.2) is ActionState.RUNNING
        assert action.executed == 3
        assert _tick(action, 0.3) is ActionState.RUNNING
        assert action.executed == 4

    def test_the_run_ends_on_the_tick_until_fires(self) -> None:
        action = _CountingAction(until=_AfterNChecks(3), reissue=True)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.RUNNING  # until check 1
        assert _tick(action, 0.2) is ActionState.RUNNING  # until check 2
        assert _tick(action, 0.3) is ActionState.END_TRANSITION  # check 3 fires
        assert _tick(action, 0.4) is ActionState.COMPLETE

    def test_a_run_ends_on_its_first_running_tick_when_until_is_satisfied(
        self,
    ) -> None:
        """The shortest possible run still gets a visible `endTransition`.

        An action that acts over ticks is never the instantaneous case the
        lifecycle passes straight through, so the transition is held for a tick
        whatever `until` decides on.
        """
        action = _CountingAction(until=_AfterNChecks(1), reissue=True)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.END_TRANSITION
        assert _tick(action, 0.2) is ActionState.COMPLETE
        assert action.executed == 2

    def test_a_failing_result_ends_the_run_too(self) -> None:
        """A result means the condition fired -- the same rule the trigger uses.

        It is what makes a timeout usable as an *until*: giving up is a way of
        being finished, and an action left running because its deadline
        *failed* would be the opposite of what the author asked for.
        """
        action = _CountingAction(until=_AfterNChecks(2, passed=False), reissue=True)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.RUNNING
        assert _tick(action, 0.2) is ActionState.END_TRANSITION


class TestTheTwoAxesAreIndependent:
    """Four combinations, and each one is a shape something needs.

    Tying them together is what would force an action that must not repeat its
    command to keep a predicate of its own instead of naming its end as a
    condition.
    """

    def test_until_alone_watches_without_repeating_the_command(self) -> None:
        """The shape `LaneChangeAction` needs.

        `force_lane_change` must not be re-sent every tick, but the manoeuvre
        it starts still takes time to land -- so the run needs an end condition
        and no repeat.
        """
        action = _CountingAction(until=_AfterNChecks(3))

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.RUNNING
        assert _tick(action, 0.2) is ActionState.RUNNING
        assert _tick(action, 0.3) is ActionState.END_TRANSITION

        # One command, at the trigger.  The rest of the run was watching.
        assert action.executed == 1

    def test_reissue_alone_repeats_until_the_trigger_stops_it(self) -> None:
        """Reissuing without an end condition keeps going for one tick.

        With nothing to wait for the run is over as soon as it is looked at, so
        the command goes out on the trigger's tick and once more on the tick
        the run is found to be finished.
        """
        action = _CountingAction(reissue=True)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert action.executed == 1
        assert _tick(action, 0.1) is ActionState.END_TRANSITION
        assert action.executed == 2
        assert _tick(action, 0.2) is ActionState.COMPLETE
        assert action.executed == 2

    def test_neither_is_the_instantaneous_action_unchanged(self) -> None:
        action = _CountingAction()

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.COMPLETE
        assert action.executed == 1

    def test_both_acts_every_tick_until_the_condition_fires(self) -> None:
        action = _CountingAction(until=_AfterNChecks(3), reissue=True)

        _tick(action, 0.0)
        _tick(action, 0.1)
        _tick(action, 0.2)
        assert _tick(action, 0.3) is ActionState.END_TRANSITION
        assert action.executed == 4


class TestWhoDecidesReissuing:
    """The answer depends on what drives the entity, so it can be injected."""

    def test_the_class_default_applies_when_nothing_is_injected(self) -> None:
        action = _CountingAction(until=_AfterNChecks(9))

        _tick(action, 0.0)
        _tick(action, 0.1)
        assert action.reissues_while_running is False
        assert action.executed == 1

    def test_an_action_can_decide_for_itself(self) -> None:
        """Where a subclass reads its own configuration, or asks its backend."""

        class _SelfDeciding(_CountingAction):
            def _reissues_by_default(self) -> bool:
                return True

        action = _SelfDeciding(until=_AfterNChecks(9))

        assert action.reissues_while_running is True
        _tick(action, 0.0)
        _tick(action, 0.1)
        assert action.executed == 2

    def test_an_injected_answer_wins_over_the_action_s_own(self) -> None:
        """The backend is what really knows, and it is outside the action."""

        class _SelfDeciding(_CountingAction):
            def _reissues_by_default(self) -> bool:
                return True

        action = _SelfDeciding(until=_AfterNChecks(9), reissue=False)

        assert action.reissues_while_running is False
        _tick(action, 0.0)
        _tick(action, 0.1)
        assert action.executed == 1

    def test_injecting_true_turns_repeating_on(self) -> None:
        action = _CountingAction(reissue=True)

        assert action.reissues_while_running is True


class TestUntilIsIndependentOfOnce:
    """*until* says when this run ends; *once* says whether another begins."""

    def test_a_repeating_action_runs_again_after_its_until_fired(self) -> None:
        until = _AfterNChecks(2)
        action = _CountingAction(until=until, once=False, reissue=True)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert action.executed == 1
        assert _tick(action, 0.1) is ActionState.RUNNING
        assert _tick(action, 0.2) is ActionState.END_TRANSITION

        # Three ticks, three commands: the action acts first and is asked
        # afterwards whether that was the last time.
        assert action.executed == 3

        # Back to standby, and the trigger fires it a second time -- on its own
        # tick, not on the one that ended the first run.  The `until` condition
        # is the same object and stays fired, so the second run is a short one,
        # which is the condition's business and not the lifecycle's.
        assert _tick(action, 0.3) is ActionState.START_TRANSITION
        assert action.executed == 4

    def test_a_repeating_action_never_commands_twice_in_one_tick(self) -> None:
        """No tick both ends a run and begins the next one.

        Collapsing them would put two commands in a single frame, which for a
        rate-limited command means a step of twice the rate.
        """
        action = _CountingAction(until=_AfterNChecks(2), once=False, reissue=True)

        for tick_index in range(6):
            before = action.executed
            _tick(action, tick_index * 0.1)
            assert action.executed - before == 1

    def test_a_one_shot_action_completes_and_stays_complete(self) -> None:
        action = _CountingAction(until=_AfterNChecks(2), once=True, reissue=True)

        _tick(action, 0.0)
        _tick(action, 0.1)
        assert _tick(action, 0.2) is ActionState.END_TRANSITION
        assert _tick(action, 0.3) is ActionState.COMPLETE
        assert _tick(action, 0.4) is ActionState.COMPLETE
        # Nothing runs after the run ended: the count stopped at the tick
        # `until` fired.
        assert action.executed == 3
