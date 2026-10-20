"""Avoidability and the victims' response, alone and inside NADE (#396).

On the synthetic crossroads of :mod:`._route_maps`: W_IN2 (x -60..-10,
y = -1.75) eastbound.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.adversity import (
    AdversarialCommand,
    AdversityKind,
    AdversityLocation,
    AgentFootprint,
    PredictedState,
)
from autoware_carla_scenario.adversity.avoidance import (
    ACCEPT,
    AVOID,
    AvoidanceSettings,
    Response,
    accept_command,
    avoid_command,
    avoidable,
    respond,
    victims,
)
from autoware_carla_scenario.adversity.nade import (
    ImportanceSampler,
    NadeAdversity,
    NadeSettings,
    is_probability,
)
from autoware_carla_scenario.adversity.observation import AgentSnapshot, Scene
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

CAR = AgentFootprint(5.0, 2.0)
INNER = -1.75
BRAKE = NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, 0.01)
ALWAYS = NadeSettings(fixed_is_probability=1.0)


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    lanelet_map = rm.crossroads()
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


def _car(name: str, x: float, speed: float = 10.0) -> AgentSnapshot:
    return AgentSnapshot(name, x, INNER, 0.0, speed, 0.0, CAR)


def _line(x: float, speed: float, steps: int = 11) -> list[PredictedState]:
    return [
        PredictedState(x + speed * 0.5 * i, 0.0, 0.0, speed, 0.5 * i)
        for i in range(steps)
    ]


def _draw(value: float) -> Any:
    return SimpleNamespace(random=lambda: value)


def _step(sampler: ImportanceSampler, scene: Scene, *names: str) -> Any:
    return sampler.step(scene, "ego", "", set(names), [BRAKE], 0.0)


# -- the pieces -----------------------------------------------------------------


def test_the_victims_are_those_whose_normal_future_it_conflicts_with() -> None:
    adversary = _line(0.0, 0.0)  # stands at x = 0
    normal = {"behind": _line(-20.0, 10.0), "away": _line(10.0, 10.0)}
    footprints = {"behind": CAR, "away": CAR}
    assert victims(adversary, CAR, normal, footprints) == ["behind"]


def test_a_victim_that_can_brake_out_of_it_makes_it_avoidable() -> None:
    adversary = _line(0.0, 0.0)
    far = [PredictedState(-20.0, 0.0, 0.0, 0.0, 0.5 * i) for i in range(11)]
    near = [PredictedState(-7.0, 0.0, 0.0, 0.0, 0.5 * i) for i in range(11)]
    assert avoidable(adversary, CAR, {"v": far}, {"v": CAR})
    # 7 m apart: clear of each other, but not of the 0.5 m buffer.
    assert not avoidable(adversary, CAR, {"v": near}, {"v": CAR})
    assert avoidable(
        adversary, CAR, {"v": near}, {"v": CAR}, AvoidanceSettings(buffer=0.0)
    )


def test_the_victims_commands() -> None:
    assert avoid_command(9.0) == AdversarialCommand.accelerate(-9.0, 3.0)
    assert accept_command() == AdversarialCommand.accelerate(0.0, 2.0)


@pytest.mark.parametrize(
    ("ego_is_victim", "can_avoid", "draw", "expected"),
    [
        (True, True, 0.0, Response("", 1.0)),  # the ego drives itself
        (False, False, 0.0, Response(ACCEPT, 1.0)),  # nothing to avoid with
        (False, True, 0.1, Response(AVOID, 0.99 / 0.2)),
        (False, True, 0.5, Response(ACCEPT, 0.01 / 0.8)),
    ],
)
def test_the_response(
    ego_is_victim: bool, can_avoid: bool, draw: float, expected: Response
) -> None:
    got = respond(_draw(draw), ego_is_victim, can_avoid)
    assert got.mode == expected.mode
    assert got.weight == pytest.approx(expected.weight)


def test_an_unavoidable_adversity_s_magnitude_is_multiplied() -> None:
    settings = NadeSettings()
    assert is_probability(1e-5, settings) == pytest.approx(2e-4)
    assert is_probability(1e-5, settings, avoidable=False) == pytest.approx(8e-3)


# -- in NADE --------------------------------------------------------------------


def test_an_ego_that_can_brake_in_time_keeps_p(features: MapFeatures) -> None:
    # 10 m behind npc1 at 10 m/s: braking at 9 m/s², the ego stops in time.
    scene = Scene(features, [_car("ego", -55.0), _car("npc1", -40.0)], {})
    sampler = ImportanceSampler(settings=ALWAYS)
    choice = _step(sampler, scene, "npc1")
    assert choice is not None and choice.avoidable
    assert choice.victims == [] and choice.response == ""  # the ego's only
    assert sampler.weight == pytest.approx(0.01)


def test_an_unavoidable_adversity_is_that_much_less_likely(
    features: MapFeatures,
) -> None:
    # 1 m behind npc1 at 15 m/s: no braking keeps the ego clear of it.
    scene = Scene(features, [_car("ego", -50.0, 15.0), _car("npc1", -44.0, 15.0)], {})
    sampler = ImportanceSampler(settings=ALWAYS)
    choice = _step(sampler, scene, "npc1")
    assert choice is not None and not choice.avoidable
    assert sampler.weight == pytest.approx(0.01 * 3e-4)


def test_npc_victims_respond_when_npcs_are_challenged(features: MapFeatures) -> None:
    # npc1 brake-checks npc2 behind it; the ego is well ahead of both.
    scene = Scene(
        features, [_car("ego", -20.0), _car("npc1", -45.0), _car("npc2", -58.0)], {}
    )
    alone = ImportanceSampler(settings=ALWAYS)
    assert _step(alone, scene, "npc1", "npc2") is None  # the ego is clear
    npcs = ImportanceSampler(
        settings=NadeSettings(fixed_is_probability=1.0, challenge_npcs=True)
    )
    choice = _step(npcs, scene, "npc1", "npc2")
    assert choice is not None and choice.name == "npc1"
    assert choice.victims == ["npc2"] and choice.response in (AVOID, ACCEPT)
    factor = 0.99 / 0.2 if choice.response == AVOID else 0.01 / 0.8
    assert npcs.weight == pytest.approx(0.01 * factor)
