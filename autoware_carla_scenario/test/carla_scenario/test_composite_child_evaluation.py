"""Regression tests: composites check every child on every tick (issue #663).

A composite that stops at the first pending (or first firing) child leaves the
children after it unchecked on that tick.  A stateful child -- a
:class:`PersistentCondition` timer above all -- then misses the ticks on which
it did not hold, and reports that it held "continuously" across them.  The
verdict also comes to depend on the order of the children.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from autoware_carla_scenario import (
    AndCondition,
    BaseCondition,
    OrCondition,
    PersistentCondition,
    ScenarioResult,
    StandstillCondition,
)
from autoware_carla_scenario.conditions.composition.base import CompositionCondition

_TICKS = (1.0, 2.0, 3.0, 4.0, 5.0)


class _Scripted(BaseCondition):
    """Return ``passed=True`` at the times in *true_at*, ``None`` otherwise."""

    def __init__(self, true_at: set[float], label: str) -> None:
        super().__init__(label=label)
        self.true_at = true_at
        self.calls: list[float] = []

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        self.calls.append(elapsed)
        if elapsed in self.true_at:
            return ScenarioResult(
                passed=True, message=self.label, elapsed_seconds=elapsed
            )
        return None


class _Failing(BaseCondition):
    """Return ``passed=False`` on every tick."""

    def __init__(self, label: str = "failing") -> None:
        super().__init__(label=label)

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        return ScenarioResult(passed=False, message=self.label, elapsed_seconds=elapsed)


# ---------------------------------------------------------------------------
# AndCondition
# ---------------------------------------------------------------------------


class TestAndChecksEveryChild:
    @pytest.mark.parametrize("persistent_first", [False, True])
    def test_persistent_child_does_not_count_unchecked_ticks(
        self, persistent_first: bool
    ) -> None:
        """``stopped`` held at t=1 and t=5 only, so a 2 s hold never happened."""
        near = _Scripted({1.0, 5.0}, "near_stop_line")
        stopped = _Scripted({1.0, 5.0}, "stopped")
        persistent = PersistentCondition(stopped, 2.0)
        children: list[BaseCondition] = (
            [persistent, near] if persistent_first else [near, persistent]
        )
        cond = AndCondition(children)

        out = [cond.check(None, t) for t in _TICKS]  # type: ignore[arg-type]

        assert stopped.calls == list(_TICKS)
        assert out == [None] * len(_TICKS)

    def test_child_after_a_pending_child_is_checked(self) -> None:
        spy = MagicMock(spec=BaseCondition)
        spy.label = "spy"
        spy.check.return_value = None
        cond = AndCondition([_Scripted(set(), "pending"), spy])

        assert cond.check(MagicMock(), elapsed=1.0) is None
        spy.check.assert_called_once()

    def test_child_after_a_failing_child_is_checked(self) -> None:
        spy = MagicMock(spec=BaseCondition)
        spy.label = "spy"
        spy.check.return_value = None
        cond = AndCondition([_Failing(), spy])

        result = cond.check(MagicMock(), elapsed=1.0)

        assert result is not None and result.passed is False
        spy.check.assert_called_once()

    @pytest.mark.parametrize("failing_first", [False, True])
    def test_a_failure_wins_over_a_pending_child_in_either_order(
        self, failing_first: bool
    ) -> None:
        pending = _Scripted(set(), "pending")
        failing = _Failing()
        children: list[BaseCondition] = (
            [failing, pending] if failing_first else [pending, failing]
        )

        result = AndCondition(children).check(None, 1.0)  # type: ignore[arg-type]

        assert result is not None and result.passed is False
        assert result.message == "failing"

    def test_first_failure_in_order_is_returned(self) -> None:
        cond = AndCondition([_Failing("first"), _Failing("second")])

        result = cond.check(None, 1.0)  # type: ignore[arg-type]

        assert result is not None and result.message == "first"


# ---------------------------------------------------------------------------
# OrCondition
# ---------------------------------------------------------------------------


class TestOrChecksEveryChild:
    def test_persistent_child_does_not_count_ticks_another_child_fired_on(
        self,
    ) -> None:
        """Nested in an AND that is still waiting, an OR keeps every child current.

        ``other`` fires at t=2..4, which used to stop the OR before the
        persistent child.  ``held`` did not hold at t=2..4, so at t=5 it has
        held for 0 s, not 4 s.
        """
        gate = _Scripted({5.0}, "gate")
        other = _Scripted({2.0, 3.0, 4.0}, "other")
        held = _Scripted({1.0, 5.0}, "held")
        cond = AndCondition(
            [gate, OrCondition([other, PersistentCondition(held, 2.0)])]
        )

        out = [cond.check(None, t) for t in _TICKS]  # type: ignore[arg-type]

        assert held.calls == list(_TICKS)
        assert out == [None] * len(_TICKS)

    def test_child_after_a_firing_child_is_checked(self) -> None:
        spy = MagicMock(spec=BaseCondition)
        spy.label = "spy"
        spy.check.return_value = None
        cond = OrCondition([_Scripted({1.0}, "fires"), spy])

        result = cond.check(MagicMock(), elapsed=1.0)

        assert result is not None and result.message == "fires"
        spy.check.assert_called_once()

    def test_first_result_in_order_is_returned(self) -> None:
        cond = OrCondition([_Scripted({1.0}, "first"), _Failing("second")])

        result = cond.check(None, 1.0)  # type: ignore[arg-type]

        assert result is not None and result.passed is True
        assert result.message == "first"


# ---------------------------------------------------------------------------
# CompositionCondition entity guard
# ---------------------------------------------------------------------------


class _GuardedComposition(CompositionCondition):
    def _check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        return ScenarioResult(passed=True, message=self.label, elapsed_seconds=elapsed)


def _world(role_name: Optional[str], *, speed: float = 0.0) -> MagicMock:
    """Return a world holding one stopped actor, or no actor at all."""
    world = MagicMock()
    if role_name is None:
        world.get_actors.return_value = []
        return world
    actor = MagicMock()
    actor.attributes = {"role_name": role_name}
    velocity = MagicMock()
    velocity.x, velocity.y, velocity.z = speed, 0.0, 0.0
    actor.get_velocity.return_value = velocity
    world.get_actors.return_value = [actor]
    return world


class TestCompositionChecksChildWhileEntityAbsent:
    def test_child_is_checked_while_the_entity_is_absent(self) -> None:
        child = _Scripted(set(), "child")
        cond = _GuardedComposition(child=child, entity_name="ego", label="guarded")

        assert cond.check(_world(None), 1.0) is None
        assert child.calls == [1.0]

    def test_result_is_none_while_the_entity_is_absent(self) -> None:
        child = _Scripted({1.0}, "child")
        cond = _GuardedComposition(child=child, entity_name="ego", label="guarded")

        assert cond.check(_world(None), 1.0) is None

    def test_standstill_timer_restarts_after_the_entity_was_absent(self) -> None:
        """Stopped at t=1, gone at t=2..4, stopped again at t=5: no 2 s hold."""
        cond = StandstillCondition("ego", duration=2.0, label="standstill")
        present, absent = _world("ego"), _world(None)
        worlds = [present, absent, absent, absent, present]

        out = [cond.check(w, t) for w, t in zip(worlds, _TICKS)]

        assert out == [None] * len(_TICKS)
