"""TeraSim's NADE sampling, on the synthetic crossroads (#395).

The eastbound W approach is W_IN2 (x -60..-10, y = -1.75) with the outer lane
W_OUT_LANE2 at y = -5.25; the ego drives east on it.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from autoware_carla_scenario.adversity import (
    AdversarialCommand,
    AdversityKind,
    AdversityLocation,
    AgentFootprint,
    predict_command,
)
from autoware_carla_scenario.adversity.nade import (
    ImportanceSampler,
    NadeAdversity,
    NadeSettings,
    candidates,
    ego_command,
    is_probability,
)
from autoware_carla_scenario.adversity.observation import AgentSnapshot, Scene
from autoware_carla_scenario.adversity.geometry import PredictedState
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

CAR = AgentFootprint(5.0, 2.0)
PERSON = AgentFootprint(0.5, 0.5, is_vehicle=False)
INNER = -1.75
OUTER = -5.25
BRAKE = NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, 0.01)


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    lanelet_map = rm.crossroads()
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


def _car(name: str, x: float, y: float = INNER, speed: float = 10.0) -> AgentSnapshot:
    return AgentSnapshot(name, x, y, 0.0, speed, 0.0, CAR)


def _scene(features: MapFeatures, *agents: AgentSnapshot) -> Scene:
    return Scene(features, list(agents), {})


def _leading(features: MapFeatures) -> Scene:
    # npc1 15 m ahead of the ego, both at 10 m/s: npc1 may brake-check it.
    return _scene(features, _car("ego", -55.0), _car("npc1", -40.0))


ALWAYS = NadeSettings(fixed_is_probability=1.0)


def _step(
    sampler: ImportanceSampler,
    scene: Scene,
    elapsed: float = 0.0,
    controllable: tuple[str, ...] = ("npc1",),
    adversities: tuple[NadeAdversity, ...] = (BRAKE,),
) -> Any:
    return sampler.step(scene, "ego", "", set(controllable), list(adversities), elapsed)


# -- candidates and q ---------------------------------------------------------


def test_the_candidates_are_the_nearest_vehicles_then_the_pedestrians(
    features: MapFeatures,
) -> None:
    scene = _scene(
        features,
        _car("ego", -40.0),
        _car("far", 30.0),  # 70 m away
        _car("b", -30.0),
        _car("a", -35.0),
        _car("c", -20.0),
        _car("unregistered", -38.0),
        AgentSnapshot("p", -40.0, -12.0, 0.0, 0.0, 0.0, PERSON),
    )
    controllable = {"far", "a", "b", "c", "p"}
    assert candidates(scene, "ego", controllable) == ["a", "b", "c", "p"]
    two = NadeSettings(max_controlled=2)
    assert candidates(scene, "ego", controllable, two) == ["a", "b", "p"]


@pytest.mark.parametrize(
    ("p", "settings", "q"),
    [
        (0.001, NadeSettings(), 0.02),  # p x 20
        (0.01, NadeSettings(), 0.05),  # clipped
        (0.01, NadeSettings(magnitude=2.0), 0.02),
        (0.01, NadeSettings(fixed_is_probability=0.3), 0.3),  # fixed, whatever p
    ],
)
def test_the_is_probability(p: float, settings: NadeSettings, q: float) -> None:
    assert is_probability(p, settings) == pytest.approx(q)


# -- sampling -----------------------------------------------------------------


def test_a_challenging_adversity_sampled_weighs_p_over_q(features: MapFeatures) -> None:
    sampler = ImportanceSampler(settings=ALWAYS)
    choice = _step(sampler, _leading(features))
    assert choice is not None
    assert (choice.name, choice.adversity, choice.is_probability) == (
        "npc1",
        BRAKE,
        1.0,
    )
    assert choice.command == AdversarialCommand.accelerate(-9.0, 2.0)
    assert sampler.weight == pytest.approx(0.01)


def test_one_not_sampled_weighs_one_minus_p_over_one_minus_q(
    features: MapFeatures,
) -> None:
    sampler = ImportanceSampler(settings=NadeSettings(fixed_is_probability=1e-12))
    assert _step(sampler, _leading(features)) is None
    assert sampler.weight == pytest.approx((1 - 0.01) / (1 - 1e-12))


def test_an_adversity_that_challenges_no_one_weighs_nothing(
    features: MapFeatures,
) -> None:
    # npc1 brake-checks npc2 behind it, both behind the ego: the ego is clear.
    scene = _scene(
        features, _car("ego", -20.0), _car("npc1", -45.0), _car("npc2", -58.0)
    )
    sampler = ImportanceSampler(settings=ALWAYS)
    assert _step(sampler, scene, 0.0) is None
    assert sampler.weight == 1.0


def test_the_first_adversity_that_fires_is_the_agent_s(features: MapFeatures) -> None:
    leader = NadeAdversity(AdversityKind.IGNORE_LEADER, AdversityLocation.ANY, 0.5)
    gated = NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.HIGHWAY, 0.5)
    sampler = ImportanceSampler(settings=ALWAYS)
    choice = _step(sampler, _leading(features), adversities=(leader, gated, BRAKE))
    # No leader to ignore, and no highway here: the third.
    assert choice is not None and choice.adversity is BRAKE


def test_no_adversity_after_one_for_the_cooldown(features: MapFeatures) -> None:
    sampler = ImportanceSampler(settings=ALWAYS)
    scene = _leading(features)
    assert _step(sampler, scene, 1.0) is not None
    assert sampler.cooling(6.0)
    assert _step(sampler, scene, 6.0) is None
    assert sampler.weight == pytest.approx(0.01)
    assert _step(sampler, scene, 6.5) is not None


def test_one_adversity_a_tick_the_first_challenger(features: MapFeatures) -> None:
    # npc1 right ahead of the ego, npc2 ahead of npc1: both may brake-check
    # into the ego, and only the first decided is.
    scene = _scene(
        features, _car("ego", -55.0), _car("npc1", -40.0), _car("npc2", -27.0)
    )
    sampler = ImportanceSampler(settings=ALWAYS)
    choice = _step(sampler, scene, controllable=("npc1", "npc2"))
    assert choice is not None and choice.name == "npc1"
    assert sampler.weight == pytest.approx(0.01)


def test_an_agent_s_adversity_is_its_first_that_fires_even_unchallenging(
    features: MapFeatures,
) -> None:
    # npc1 may cut right as well as brake-check; the cut-in, listed first,
    # is its adversity -- and leaves the ego alone -- as in TeraSim, where
    # the first adversity that fires gives the agent's distribution.
    scene = _scene(
        features,
        _car("ego", -55.0),
        _car("npc1", -40.0),
        _car("npc2", -46.0, OUTER, speed=12.0),
    )
    cut = NadeAdversity(AdversityKind.CUT_IN, AdversityLocation.ANY, 0.02)
    sampler = ImportanceSampler(settings=ALWAYS)
    assert _step(sampler, scene, adversities=(cut, BRAKE)) is None
    assert sampler.weight == 1.0


def test_the_draws_are_seeded(features: MapFeatures) -> None:
    def run(seed: int) -> list[bool]:
        sampler = ImportanceSampler(
            seed, NadeSettings(fixed_is_probability=0.5, cooldown=0.0)
        )
        scene = _leading(features)
        return [_step(sampler, scene, float(t)) is not None for t in range(20)]

    assert run(3) == run(3) != run(4)


def test_an_ego_off_the_lanes_is_challenged_by_nothing(features: MapFeatures) -> None:
    scene = _scene(features, _car("ego", -40.0, -30.0), _car("npc1", -25.0))
    sampler = ImportanceSampler(settings=ALWAYS)
    assert _step(sampler, scene, 0.0) is None


# -- prediction -----------------------------------------------------------------


def _seen(features: MapFeatures, x: float = -50.0, speed: float = 10.0) -> Any:
    return _scene(features, _car("npc1", x, speed=speed)).vehicle("npc1")


def test_a_predicted_acceleration_is_held_to_the_horizon(features: MapFeatures) -> None:
    path = predict_command(
        features, _seen(features), AdversarialCommand.accelerate(-4.0, 2.0), 0.5, 5.0
    )
    assert [p.t for p in path] == pytest.approx([i * 0.5 for i in range(11)])
    assert path[-1].speed == 0.0  # braked to a stop, not for 2 s only


def test_a_predicted_lane_change_keeps_to_the_new_lane(features: MapFeatures) -> None:
    path = predict_command(
        features, _seen(features), AdversarialCommand.change_lane("right"), 0.5, 5.0
    )
    assert path[-1].t == pytest.approx(5.0)
    assert all(p.y == pytest.approx(OUTER) for p in path if p.t >= 1.0)
    assert all(abs(p.yaw) < 1e-6 for p in path if p.t > 1.0)


def test_a_predicted_path_is_resampled_and_cut(features: MapFeatures) -> None:
    walk = [PredictedState(0.0, 0.1 * i, math.pi / 2, 1.0, 0.1 * i) for i in range(71)]
    path = predict_command(
        features, _seen(features), AdversarialCommand.follow(walk), 0.5, 5.0
    )
    assert [p.t for p in path] == pytest.approx([i * 0.5 for i in range(11)])
    assert path[-1].y == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("indicator", "expected"),
    [
        ("right", AdversarialCommand.change_lane("right")),
        ("left", AdversarialCommand.accelerate(0.0, 5.0)),  # no lane on the left
        ("", AdversarialCommand.accelerate(0.0, 5.0)),
    ],
)
def test_the_ego_is_predicted_from_its_indicator(
    features: MapFeatures, indicator: str, expected: AdversarialCommand
) -> None:
    scene = _scene(features, _car("ego", -40.0))
    assert ego_command(scene, "ego", indicator) == expected
