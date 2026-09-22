"""The ``until`` condition: an action that keeps acting while it runs.

``is_finished`` answers "has the work I handed the simulator landed yet?".
``until`` answers a different question -- "is the thing I am *still doing*
done?" -- and is the only one of the two that runs ``execute`` again, so an
action whose command has to be reissued every tick needs no hook beside the
one it already implements.
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
        finished: bool = True,
    ) -> None:
        super().__init__(
            label="counting",
            condition=condition if condition is not None else AlwaysTrueCondition(),
            timing=TickTiming.PRE_TICK,
            once=once,
            until=until,
        )
        self.executed = 0
        self.is_finished_calls = 0
        self._finished = finished

    def execute(self, world: object) -> None:
        self.executed += 1

    def is_finished(self, world: object, running_for: float) -> bool:
        self.is_finished_calls += 1
        return self._finished


def _tick(action: BaseAction, elapsed: float) -> ActionState:
    action.tick(object(), elapsed)
    return action.state


class TestWithoutUntil:
    """Every action that existed before ``until`` keeps its exact behaviour."""

    def test_an_instantaneous_action_is_unchanged(self) -> None:
        action = _CountingAction()

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.COMPLETE
        assert action.executed == 1

    def test_is_finished_still_decides_a_watching_action(self) -> None:
        action = _CountingAction(finished=False)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.RUNNING
        assert _tick(action, 0.2) is ActionState.RUNNING
        # Watching is not acting: `execute` ran once, when the trigger fired.
        assert action.executed == 1
        assert action.is_finished_calls == 2


class TestUntilRunsExecute:
    def test_execute_runs_again_on_every_tick_of_the_run(self) -> None:
        action = _CountingAction(until=_AfterNChecks(4))

        _tick(action, 0.0)  # triggered: execute #1
        assert action.executed == 1

        assert _tick(action, 0.1) is ActionState.RUNNING
        # The tick the run begins is the tick `execute` already ran on, so it
        # is not run twice for one command.
        assert action.executed == 1

        assert _tick(action, 0.2) is ActionState.RUNNING
        assert action.executed == 2
        assert _tick(action, 0.3) is ActionState.RUNNING
        assert action.executed == 3

    def test_the_run_ends_on_the_tick_until_fires(self) -> None:
        action = _CountingAction(until=_AfterNChecks(3))

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.RUNNING  # until check 1
        assert _tick(action, 0.2) is ActionState.RUNNING  # until check 2
        assert _tick(action, 0.3) is ActionState.END_TRANSITION  # check 3 fires
        assert _tick(action, 0.4) is ActionState.COMPLETE

    def test_a_run_whose_until_is_already_satisfied_ends_at_once(self) -> None:
        """One command goes out, and the action does not linger."""
        action = _CountingAction(until=_AfterNChecks(1))

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.COMPLETE
        assert action.executed == 1

    def test_a_failing_result_ends_the_run_too(self) -> None:
        """A result means the condition fired -- the same rule the trigger uses.

        It is what makes a timeout usable as an *until*: giving up is a way of
        being finished, and an action left running because its deadline
        *failed* would be the opposite of what the author asked for.
        """
        action = _CountingAction(until=_AfterNChecks(2, passed=False))

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.RUNNING
        assert _tick(action, 0.2) is ActionState.END_TRANSITION


class TestUntilTakesPrecedence:
    def test_is_finished_is_not_consulted(self) -> None:
        """Two answers to one question would be a silent contradiction.

        The action below says it is finished; its *until* says it is not.  The
        run continues, and `is_finished` is never asked -- an action written
        for `until` is not also asked to keep a stale predicate honest.
        """
        action = _CountingAction(until=_AfterNChecks(3), finished=True)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.RUNNING
        assert _tick(action, 0.2) is ActionState.RUNNING
        assert action.is_finished_calls == 0


class TestUntilIsIndependentOfOnce:
    """*until* says when this run ends; *once* says whether another begins."""

    def test_a_repeating_action_runs_again_after_its_until_fired(self) -> None:
        until = _AfterNChecks(2)
        action = _CountingAction(until=until, once=False)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert action.executed == 1
        assert _tick(action, 0.1) is ActionState.RUNNING
        assert _tick(action, 0.2) is ActionState.END_TRANSITION

        # Two commands so far: the one that started the run, and the one on
        # the tick it ended -- an action acts first and is asked afterwards
        # whether that was the last time.
        assert action.executed == 2

        # Back to standby, and the trigger fires it a second time.  The `until`
        # condition is the same object and stays fired, so the second run is a
        # short one -- which is the condition's business, not the lifecycle's.
        assert _tick(action, 0.3) is ActionState.START_TRANSITION
        assert action.executed == 3

    def test_a_one_shot_action_completes_and_stays_complete(self) -> None:
        action = _CountingAction(until=_AfterNChecks(2), once=True)

        _tick(action, 0.0)
        _tick(action, 0.1)
        assert _tick(action, 0.2) is ActionState.END_TRANSITION
        assert _tick(action, 0.3) is ActionState.COMPLETE
        assert _tick(action, 0.4) is ActionState.COMPLETE
        # Nothing runs after the run ended: the count stopped at the tick
        # `until` fired.
        assert action.executed == 2
