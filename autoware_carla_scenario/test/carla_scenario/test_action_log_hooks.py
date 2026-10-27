"""What an action hands the runner for the trajectory log (#655).

``BaseAction`` declares each hook the runner reads off every action, as an
empty list: ``take_adversity_events()``, ``take_waypoint_arrivals()`` and
``take_despawns()``. The actions that have something to log override them,
and the runner calls them directly, with no ``getattr``, so that the
library check compiles the logging.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest

from autoware_carla_scenario import DespawnAction, RoutingAction
from autoware_carla_scenario.actions.adversarial_traffic import (
    AdversarialTrafficAction,
)
from autoware_carla_scenario.actions.base import BaseAction
from autoware_carla_scenario.scenario_runner import ScenarioRunner
from autoware_carla_scenario.waypoint_arrival import WaypointArrival

#: Each hook the runner reads off every action, and the action that has
#: something to hand it.
_HOOKS = {
    "take_adversity_events": AdversarialTrafficAction,
    "take_waypoint_arrivals": RoutingAction,
    "take_despawns": DespawnAction,
}


class _Plain(BaseAction):
    """An action with nothing to log."""

    def execute(self, world: Any) -> None:
        pass


@pytest.mark.parametrize("hook", sorted(_HOOKS))
def test_an_action_hands_the_log_nothing_by_default(hook: str) -> None:
    action = _Plain(label="plain")
    first = getattr(action, hook)()
    assert first == []
    # A fresh list each call: nothing appended to one leaks into the next.
    first.append(object())
    assert getattr(action, hook)() == []


@pytest.mark.parametrize(("hook", "owner"), sorted(_HOOKS.items()))
def test_the_action_with_something_to_log_overrides_the_hook(
    hook: str, owner: type[BaseAction]
) -> None:
    assert getattr(owner, hook) is not getattr(BaseAction, hook)
    # The same signature as the base's: the runner calls it as the base's.
    assert inspect.signature(getattr(owner, hook)) == inspect.signature(
        getattr(BaseAction, hook)
    )


def test_a_despawn_is_handed_over_once() -> None:
    action = DespawnAction("npc1")
    action._despawned.append(("npc1", 42))
    assert action.take_despawns() == [("npc1", 42)]
    assert action.take_despawns() == []


def test_a_waypoint_arrival_is_a_dataclass_the_recorder_reads() -> None:
    arrival = WaypointArrival(
        entity="npc1",
        vertex=2,
        kind="time",
        mode="driver",
        target_t=3.0,
        actual_t=4.5,
    )
    assert arrival.error_s == pytest.approx(1.5)


def _run_scenario() -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(ScenarioRunner.run_scenario))
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


@pytest.mark.parametrize("hook", sorted(_HOOKS))
def test_the_runner_calls_each_hook_off_the_action_directly(hook: str) -> None:
    run = _run_scenario()
    calls = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == hook
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "action"
    ]
    assert len(calls) == 1, f"run_scenario() calls action.{hook}() {len(calls)} times"
    # Not looked up by name: getattr() hides the call from the library check.
    looked_up = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and any(
            isinstance(arg, ast.Constant) and arg.value == hook for arg in node.args
        )
    ]
    assert not looked_up, f"run_scenario() looks action.{hook} up with getattr()"
