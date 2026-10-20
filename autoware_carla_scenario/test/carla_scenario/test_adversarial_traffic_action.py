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

import autoware_carla_scenario as acs
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions import _adversarial_execution as executions
from autoware_carla_scenario.actions import adversarial_traffic
from autoware_carla_scenario.actions.adversarial_traffic import AdversarialTrafficAction
from autoware_carla_scenario.adversity import AdversityKind, AdversityLocation
from autoware_carla_scenario.adversity.nade import NadeAdversity, NadeSettings
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

    def __init__(self, name: str, path: Any, mode: Any, label: str) -> None:
        self.name, self.path, self.label = name, path, label
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


def _leading(frame: int, *, npc: str = "npc1") -> Any:
    # npc1 15 m ahead of the ego, both at 10 m/s: it may brake-check the ego.
    return _world(
        _actor(1, EGO, -55.0, INNER, 10.0),
        _actor(2, npc, -40.0, INNER, 10.0),
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


def test_no_ego_no_adversity() -> None:
    action = AdversarialTrafficAction([BRAKE], settings=ALWAYS)
    world = _world(
        _actor(2, "npc1", -40.0, INNER, 10.0), _actor(3, "npc2", -55.0, INNER, 10.0)
    )
    action.tick(world, 0.0)
    assert made == []


def test_the_ego_s_indicator_is_its_predicted_lane_change(
    stand_ins: dict[str, Any],
) -> None:
    # Signalling right, the ego is predicted out of npc1's way: no challenge.
    stand_ins["state"] = TurnIndicatorState.RIGHT
    action = AdversarialTrafficAction([BRAKE], settings=ALWAYS)
    _run(action, 2)
    assert made == [] and action.weight == 1.0


def test_closing_closes_the_adversity_under_way() -> None:
    action = AdversarialTrafficAction([BRAKE], settings=ALWAYS)
    _run(action, 1)
    action.close()
    assert made[0].closed


def test_a_probability_is_a_probability() -> None:
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        AdversarialTrafficAction(
            [NadeAdversity(AdversityKind.CUT_IN, AdversityLocation.ANY, 1.5)]
        )


def test_it_is_exported_with_its_list_entry() -> None:
    assert acs.AdversarialTrafficAction is AdversarialTrafficAction
    assert acs.NadeAdversity is NadeAdversity and acs.NadeSettings is NadeSettings


def test_the_editor_s_card_builds_the_action() -> None:
    from autoware_carla_scenario.authoring import _builders_generated as built
    from autoware_carla_scenario.authoring.compiler import coerce_params
    from autoware_carla_scenario.authoring.registry import get_action_spec

    spec = get_action_spec("adversarial_traffic")
    assert spec is not None and spec.scope == "environment"
    params = coerce_params(
        spec.fields,
        {
            "adversities": [
                {"kind": "cut_in", "location": "highway", "probability": "0.002"},
                {"kind": "brake_check", "location": "any", "probability": 0.01},
            ],
            "fixed_is_probability": "0.2",
            "seed": "7",
        },
    )
    compiled: Any = SimpleNamespace(params=params, label="nade")
    no_context: Any = None
    action = built.build_adversarial_traffic_action(
        compiled, None, acs.TickTiming.PRE_TICK, no_context
    )
    assert isinstance(action, AdversarialTrafficAction)
    assert action._adversities == [
        NadeAdversity(AdversityKind.CUT_IN, AdversityLocation.HIGHWAY, 0.002),
        NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, 0.01),
    ]
    assert action._sampler.settings == NadeSettings(fixed_is_probability=0.2)
