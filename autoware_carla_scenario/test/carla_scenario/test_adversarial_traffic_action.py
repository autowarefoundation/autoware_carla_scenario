"""AdversarialTrafficAction, NADE as an action, on a fake CARLA world (#395).

The world and its actors are the fakes of
:mod:`.test_adversity_feasible_condition` (the synthetic crossroads, CARLA's
frame its own with y flipped).  Entities are looked up through a stand-in
registry, and a sampled adversity is driven by a stand-in execution that
finishes after three ticks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions import _adversarial_execution as executions
from autoware_carla_scenario.actions import adversarial_traffic
from autoware_carla_scenario.actions.adversarial_traffic import AdversarialTrafficAction
from autoware_carla_scenario.adversity import AdversityKind, AdversityLocation
from autoware_carla_scenario.adversity.nade import NadeAdversity, NadeSettings
from autoware_carla_scenario.trajectory.model import MapPose
from autoware_carla_scenario.utils.turn_indicator import TurnIndicatorState

from .test_adversity_feasible_condition import INNER, _actor, _world
from .test_adversity_feasible_condition import crossroads as crossroads  # noqa: F401

EGO = "Ego"
BRAKE = NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, 0.01)
ALWAYS = NadeSettings(fixed_is_probability=1.0)

#: Every stand-in execution made in the test, in order.
made: list["_Execution"] = []


class _Execution:
    """Stands in for AdversarialExecution: done on its third tick."""

    def __init__(
        self,
        name: str,
        path: Any,
        mode: Any,
        label: str,
        actor: Any = None,
        backend: Any = None,
        box_ahead: float = 0.0,
        box_left: float = 0.0,
    ) -> None:
        self.name, self.path, self.label = name, path, label
        self.entity_name = name
        self.actor, self.backend = actor, backend
        self.box_ahead, self.box_left = box_ahead, box_left
        self.ticks = 0
        self.closed = False
        made.append(self)

    @property
    def done(self) -> bool:
        return self.ticks >= 3

    def tick(self, world: Any, elapsed: float) -> None:
        self.ticks += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    made.clear()
    ego_lights = {"state": TurnIndicatorState.OFF}
    monkeypatch.setattr(executions, "AdversarialExecution", _Execution)
    monkeypatch.setattr(
        adversarial_traffic, "drivable", lambda name: name in ("npc1", "npc2")
    )
    monkeypatch.setattr(
        adversarial_traffic,
        "find_actor_by_role_name",
        lambda world, name: SimpleNamespace(type_id="vehicle.ego"),
    )
    monkeypatch.setattr(
        adversarial_traffic, "turn_indicators_of", lambda actor: ego_lights["state"]
    )
    return ego_lights


def _leading(frame: int, *, npc: str = "npc1", box_x: float = 0.0) -> Any:
    # npc1 15 m ahead of the ego, both at 10 m/s: it may brake-check the ego.
    return _world(
        _actor(1, EGO, -55.0, INNER, 10.0),
        _actor(2, npc, -40.0, INNER, 10.0, box_x=box_x),
        frame=frame,
    )


def _run(action: Any, count: int, start: int = 1, **kw: Any) -> list[ActionState]:
    states = []
    for i in range(count):
        action.tick(_leading(start + i, **kw), 0.1 * (start + i))
        states.append(action.state)
    return states


def test_a_challenging_adversity_is_sampled_and_driven() -> None:
    action = AdversarialTrafficAction([BRAKE], settings=ALWAYS)
    states = _run(action, 4)
    # Started at once, and runs for the rest of the run.
    assert states[0] is ActionState.START_TRANSITION
    assert states[1:] == [ActionState.RUNNING] * 3
    (run,) = made
    assert run.name == "npc1" and run.label == "adversarial_traffic:npc1"
    assert run.path[0].t == 0.0 and run.path[-1].speed < 10.0
    assert action.weight == pytest.approx(0.01)
    assert [c.name for c in action.choices] == ["npc1"]


def test_one_adversity_at_a_time_and_none_in_the_cooldown() -> None:
    settings = NadeSettings(fixed_is_probability=1.0, cooldown=1.0)
    action = AdversarialTrafficAction([BRAKE], settings=settings)
    _run(action, 5)  # t = 0.1 .. 0.5: sampled at 0.1, driven to 0.3
    assert len(made) == 1
    _run(action, 12, start=6)  # past 1.1 s: the next
    assert len(made) == 2
    assert action.weight == pytest.approx(0.01**2)


def test_only_a_registered_entity_is_driven() -> None:
    action = AdversarialTrafficAction([BRAKE], settings=ALWAYS)
    _run(action, 3, npc="background_7")
    assert made == [] and action.weight == 1.0


class _Backend:
    """Offers one background vehicle, as a traffic backend does (#417)."""

    def __init__(self, actors: dict[str, Any]) -> None:
        self.actors = actors

    def background_actors(self, world: Any) -> dict[str, Any]:
        return dict(self.actors)


def test_a_background_vehicle_is_driven_by_its_actor_without_an_entity() -> None:
    from autoware_carla_scenario.entity.registry import find_vehicle_entity

    actor = SimpleNamespace(id=7)
    backend = _Backend({"background_7": actor})
    action = AdversarialTrafficAction([BRAKE], settings=ALWAYS)
    action.set_traffic_backend(backend)  # type: ignore[arg-type]
    _run(action, 1, npc="background_7")
    (run,) = made
    assert run.name == "background_7"
    assert run.actor is actor and run.backend is backend
    assert find_vehicle_entity("background_7") is None  # no entity made


def test_a_background_vehicle_s_origin_is_put_on_the_path() -> None:
    # Its box's centre 1.3 m ahead of its origin at x = -40 (#889).
    actor = SimpleNamespace(id=7)
    backend = _Backend({"background_7": actor})
    action = AdversarialTrafficAction([BRAKE], settings=ALWAYS)
    action.set_traffic_backend(backend)  # type: ignore[arg-type]
    _run(action, 1, npc="background_7", box_x=1.3)
    (run,) = made
    assert run.path[0].x == pytest.approx(-38.7)
    assert (run.box_ahead, run.box_left) == pytest.approx((1.3, 0.0))
    trajectory = executions.path_trajectory(
        run.label, run.path, run.box_ahead, run.box_left
    )
    first = trajectory.vertices[0].position
    assert isinstance(first, MapPose)
    assert (first.x, first.y) == pytest.approx((-40.0, INNER))


def test_background_vehicles_can_be_left_alone() -> None:
    action = AdversarialTrafficAction([BRAKE], settings=ALWAYS, drive_background=False)
    action.set_traffic_backend(_Backend({"background_7": SimpleNamespace(id=7)}))  # type: ignore[arg-type]
    _run(action, 3, npc="background_7")
    assert made == []
