"""The action lifecycle, and the condition that queries it.

Covers what ``done`` could never express: an action is *running* between the
tick its command went out and the tick its own completion criteria are met, and
only then does it reach ``completeState``.
"""

from __future__ import annotations

from typing import Optional

import pytest

from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions.base import BaseAction, TickTiming
from autoware_carla_scenario.conditions import ActionStateCondition
from autoware_carla_scenario.conditions.always_true import AlwaysTrueCondition
from autoware_carla_scenario.conditions.base import BaseCondition, ScenarioResult


class _NeverCondition(BaseCondition):
    """A trigger that is never satisfied."""

    def __init__(self) -> None:
        super().__init__(label="never")

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        return None


class _SlowAction(BaseAction):
    """An action whose work takes *duration* seconds after it is commanded."""

    def __init__(
        self,
        duration: float,
        condition: Optional[BaseCondition] = None,
        *,
        once: bool = True,
    ) -> None:
        super().__init__(
            label="slow",
            condition=condition if condition is not None else AlwaysTrueCondition(),
            timing=TickTiming.PRE_TICK,
            once=once,
        )
        self.executed = 0
        self._duration = duration

    def execute(self, world: object) -> None:
        self.executed += 1

    def is_finished(self, world: object, running_for: float) -> bool:
        return running_for >= self._duration


def _tick(action: BaseAction, elapsed: float) -> ActionState:
    action.tick(object(), elapsed)
    return action.state


class TestLifecycle:
    def test_an_untriggered_action_stays_in_standby(self) -> None:
        action = _SlowAction(duration=0.0, condition=_NeverCondition())
        assert _tick(action, 0.0) is ActionState.STANDBY
        assert action.executed == 0

    def test_the_states_run_in_openscenario_order(self) -> None:
        action = _SlowAction(duration=1.0)

        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert action.executed == 1
        # `done` is true from the moment the command went out, which is exactly
        # why it cannot stand in for completion.
        assert action.done

        assert _tick(action, 0.1) is ActionState.RUNNING
        assert _tick(action, 0.5) is ActionState.RUNNING
        assert _tick(action, 1.0) is ActionState.END_TRANSITION
        assert _tick(action, 1.1) is ActionState.COMPLETE
        assert _tick(action, 1.2) is ActionState.COMPLETE

    def test_an_instantaneous_action_does_not_linger_in_transitions(self) -> None:
        """An action with no work to do passes straight through to complete.

        Holding `endTransition` for a tick is what lets a real manoeuvre be
        watched for; an action that finished the moment it was commanded never
        ran, and pausing there would only make it a tick slower to repeat.
        """
        action = _SlowAction(duration=0.0)
        assert _tick(action, 0.0) is ActionState.START_TRANSITION
        assert _tick(action, 0.1) is ActionState.COMPLETE

    def test_work_in_flight_is_never_restarted(self) -> None:
        """A repeating action must not fire again on top of an unfinished run."""
        action = _SlowAction(duration=1.0, once=False)
        for elapsed in (0.0, 0.1, 0.5):
            _tick(action, elapsed)
        assert action.executed == 1

    def test_a_repeating_action_still_fires_on_every_tick(self) -> None:
        """`once=False` has always meant "every tick the condition holds".

        The lifecycle must not cost it ticks: an instantaneous action returns to
        standby and re-triggers within the same tick.
        """
        action = _SlowAction(duration=0.0, once=False)
        for index, elapsed in enumerate((0.0, 0.1, 0.2, 0.3), start=1):
            _tick(action, elapsed)
            assert action.executed == index, elapsed


class TestActionStateCondition:
    @pytest.fixture
    def action(self) -> _SlowAction:
        return _SlowAction(duration=1.0)

    def test_completion_is_not_satisfied_while_the_work_is_in_flight(
        self, action: _SlowAction
    ) -> None:
        actions = {"a1": action}
        condition = ActionStateCondition("a1", ActionState.COMPLETE, actions)

        _tick(action, 0.0)
        assert condition.check(object(), 0.0) is None, "fired is not finished"
        _tick(action, 0.5)
        assert condition.check(object(), 0.5) is None, "still running"

        _tick(action, 1.0)  # end transition
        _tick(action, 1.1)  # complete
        assert condition.check(object(), 1.1) is not None

    def test_a_state_name_is_accepted_as_written_in_the_document(
        self, action: _SlowAction
    ) -> None:
        condition = ActionStateCondition("a1", "startTransition", {"a1": action})
        _tick(action, 0.0)
        assert condition.check(object(), 0.0) is not None

    def test_an_action_built_later_still_resolves(self) -> None:
        """The mapping is shared and read late, so trigger order does not matter."""
        actions: dict[str, BaseAction] = {}
        condition = ActionStateCondition("a1", ActionState.COMPLETE, actions)
        assert condition.check(object(), 0.0) is None

        action = _SlowAction(duration=0.0)
        actions["a1"] = action
        for elapsed in (0.0, 0.1, 0.2):
            _tick(action, elapsed)
        assert condition.check(object(), 0.3) is not None
