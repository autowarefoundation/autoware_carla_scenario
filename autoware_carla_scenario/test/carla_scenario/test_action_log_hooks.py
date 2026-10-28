"""What an action hands the runner for the trajectory log (#655).

``BaseAction`` declares each hook the runner reads off every action, as an
empty list: ``take_adversity_events()``, ``take_waypoint_arrivals()``,
``take_spawns()`` (#658) and ``take_despawns()``. The actions that have something to log override them,
and the runner calls them directly, with no ``getattr``, so that the
library check compiles the logging.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario import DespawnAction, RoutingAction, SpawnAction
from autoware_carla_scenario.actions import _trajectory_run
from autoware_carla_scenario.actions import spawn as spawn_module
from autoware_carla_scenario.actions.adversarial_traffic import (
    AdversarialTrafficAction,
)
from autoware_carla_scenario.actions.base import BaseAction
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose
from autoware_carla_scenario.scenario_runner import ScenarioRunner
from autoware_carla_scenario.waypoint_arrival import WaypointArrival

#: Each hook the runner reads off every action, and the action that has
#: something to hand it.
_HOOKS = {
    "take_adversity_events": AdversarialTrafficAction,
    "take_waypoint_arrivals": RoutingAction,
    "take_spawns": SpawnAction,
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


class _Parked:
    """A hidden entity's actor, parked under the map."""

    id = 42
    bounding_box = SimpleNamespace(extent=SimpleNamespace(x=2.0, y=1.0, z=0.8))

    def get_transform(self) -> Any:
        return SimpleNamespace(
            location=SimpleNamespace(x=1.0, y=2.0, z=-500.0),
            rotation=SimpleNamespace(yaw=0.0),
        )


def _bring_in_with(
    monkeypatch: pytest.MonkeyPatch, actor: Any
) -> list[tuple[Any, float, float, float]]:
    """Have a SpawnAction find *actor* for its entity; the places it is put."""
    put: list[tuple[Any, float, float, float]] = []
    monkeypatch.setattr(_trajectory_run, "_entity_actor", lambda name: actor)
    monkeypatch.setattr(_trajectory_run, "_is_walker", lambda found: False)
    monkeypatch.setattr(
        spawn_module,
        "_put",
        lambda world, found, x, y, z, yaw, speed: put.append((found, x, y, z)),
    )
    return put


def test_a_spawn_is_handed_over_once(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _Parked()
    put = _bring_in_with(monkeypatch, actor)
    action = SpawnAction("npc1")
    action.tick(object(), 0.0)  # type: ignore[arg-type]
    assert [p[0] for p in put] == [actor]
    assert action.take_spawns() == [("npc1", 42)]
    assert action.take_spawns() == []
    # Fired once: a later tick brings nothing in again.
    action.tick(object(), 0.05)  # type: ignore[arg-type]
    assert action.take_spawns() == []


def test_a_spawn_that_finds_no_entity_hands_over_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    put = _bring_in_with(monkeypatch, None)
    action = SpawnAction("npc1")
    assert not action.apply(object())  # type: ignore[arg-type]
    assert put == []
    assert action.take_spawns() == []


def test_a_prop_is_not_a_spawn_of_an_entity() -> None:
    action = SpawnAction(name="cone", position=CarlaWorldPose(1.0, 2.0, 0.0, 0.0))
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(filter=lambda name: [name]),
        try_spawn_actor=lambda blueprint, transform: SimpleNamespace(
            destroy=lambda: None
        ),
    )
    try:
        assert action.apply(world, ground=lambda x, y: 0.0)  # type: ignore[arg-type]
    finally:
        spawn_module.remove_prop("cone")
    assert action.take_spawns() == []


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


def test_the_runner_records_a_spawn_as_it_records_a_despawn() -> None:
    """Each pair the hook hands over goes to the recorder under the action's
    label (#658)."""
    run = _run_scenario()
    recorded: dict[str, str] = {}
    for node in ast.walk(run):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.attr in ("take_spawns", "take_despawns")
        ):
            (body,) = node.body
            assert isinstance(body, ast.Expr) and isinstance(body.value, ast.Call)
            call = body.value
            assert isinstance(call.func, ast.Attribute)
            assert ast.unparse(call.args[0]) == "action.label"
            recorded[node.iter.func.attr] = call.func.attr
    assert recorded == {
        "take_spawns": "record_spawn",
        "take_despawns": "record_despawn",
    }
