"""An action that keeps acting is put on the tick loop by ``register_init`` (#1011).

The initialization phase performs an action once and never ticks it again, so
one that reissues, steps something along, or waits on an ``until`` would be
left running with nothing driving it.  ``BaseAction.needs_tick_loop`` is how
an action says it has to run on the loop instead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    BaseScenario,
    CarlaWorldPose,
    EgoConfig,
    LaneChangeAction,
    PlayMotionAction,
    SetSpeedAction,
    SpawnTransform,
    StalledObjectAction,
    WalkStraightAction,
)
from autoware_carla_scenario.actions._trajectory_run import TrajectoryRun
from autoware_carla_scenario.actions.adversarial import (
    AdversarialBrakeAction,
    JaywalkAction,
)
from autoware_carla_scenario.actions.adversarial_traffic import AdversarialTrafficAction
from autoware_carla_scenario.actions.background_traffic import (
    TrafficSinkAction,
    TrafficSourceAction,
)
from autoware_carla_scenario.actions.base import BaseAction
from autoware_carla_scenario.actions.construction_zone import ConstructionZoneAction
from autoware_carla_scenario.actions.lane_change import LaneChangeDirection
from autoware_carla_scenario.actions.routing import RoutingAction
from autoware_carla_scenario.actions.smart_pedestrian import (
    SmartPedestrianControllerAction,
)
from autoware_carla_scenario.adversity import AdversityKind, AdversityLocation
from autoware_carla_scenario.adversity.nade import NadeAdversity
from autoware_carla_scenario.conditions import AlwaysTrueCondition
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity

from .test_play_motion import RAISED, _write_motion
from .test_routing_trajectory_action import _DT, _Actor, _Entity, _World

_NOT_JUNCTION = [{"type": "not", "constraint": {"type": "is_junction"}}]
_AT = CarlaWorldPose(30.0, 4.0, 0.0, yaw=90.0)


class _Scenario(BaseScenario):
    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return False


def _scenario() -> _Scenario:
    return _Scenario(
        EgoConfig(
            spawn_location=SpawnTransform(carla.Transform(carla.Location())),
            vehicle_type="vehicle.mini.cooper",
        )
    )


@pytest.fixture
def stalled_entity() -> Iterator[_Entity]:
    entity = _Entity(_Actor())
    register_entity("stalled1", entity)
    yield entity
    unregister_entity("stalled1")


def test_a_stalled_object_registered_for_init_appears_on_the_loop(
    stalled_entity: _Entity,
) -> None:
    """The issue's case: registered for init, it never appeared at all."""
    scenario = _scenario()
    action = StalledObjectAction("stalled1", _AT, start_time=1.0, end_time=3.0)
    scenario.register_init(action)
    assert action in scenario._pre_tick_actions
    assert action not in scenario._init_actions

    world = _World()
    scenario.run_init(world)  # type: ignore[arg-type]
    actor = stalled_entity.actor
    shown: list[int] = []
    t = 0.0
    for tick in range(80):
        for a in scenario._pre_tick_actions:
            a.tick(world, t)  # type: ignore[arg-type]
        actor.step(_DT, driven=False)
        if actor.physics:
            shown.append(tick)
        t += _DT
    assert shown == list(range(20, 60))


Make = Callable[[Path], BaseAction]

#: Every action that keeps acting after its first tick, or waits on an until.
_ON_THE_LOOP: dict[str, Make] = {
    "trajectory_run": lambda _: StalledObjectAction("v", _AT, 1.0, 3.0),
    "walk_straight_turning_back": lambda _: WalkStraightAction(
        "p", 1.4, turn_back_distance_m=10.0
    ),
    "walk_straight_turning_back_standing": lambda _: WalkStraightAction(
        "p", 0.0, turn_back_ttc_s=2.0
    ),
    "play_motion": lambda tmp: PlayMotionAction(
        "p", _write_motion(tmp / "wave.npz", [{}, RAISED])
    ),
    "smart_pedestrian": lambda _: SmartPedestrianControllerAction(
        "p", "walks on", duration_s=2.0
    ),
    "adversarial_brake": lambda _: AdversarialBrakeAction("v", AdversityLocation.ANY),
    "jaywalk": lambda _: JaywalkAction("p", AdversityLocation.ANY),
    "adversarial_traffic": lambda _: AdversarialTrafficAction(
        [NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, 0.01)]
    ),
    "set_speed_with_rate": lambda _: SetSpeedAction("v", 30.0, rate_kmh_s=5.0),
    "set_speed_with_until": lambda _: SetSpeedAction(
        "v", 30.0, until=AlwaysTrueCondition()
    ),
    "lane_change": lambda _: LaneChangeAction("v", LaneChangeDirection.LEFT),
    "construction_zone": lambda _: ConstructionZoneAction(
        1, 20.0, 70.0, start_time=1.0, end_time=3.0
    ),
    "construction_zone_appearing": lambda _: ConstructionZoneAction(
        1, 20.0, 70.0, start_time=1.0
    ),
    "construction_zone_cleared": lambda _: ConstructionZoneAction(
        1, 20.0, 70.0, end_time=3.0
    ),
}

#: Every action that is over once commanded, or is meant to act once at init.
_IN_THE_PHASE: dict[str, Make] = {
    "walk_straight": lambda _: WalkStraightAction("p", 1.4),
    "set_speed": lambda _: SetSpeedAction("v", 30.0),
    "traffic_source": lambda _: TrafficSourceAction(
        _NOT_JUNCTION, initial_vehicles=5, vehicles_per_minute=6.0
    ),
    "traffic_sink": lambda _: TrafficSinkAction(_NOT_JUNCTION),
    "construction_zone_standing": lambda _: ConstructionZoneAction(1, 20.0, 70.0),
}


@pytest.mark.parametrize("make", _ON_THE_LOOP.values(), ids=_ON_THE_LOOP.keys())
def test_an_action_that_keeps_acting_needs_the_tick_loop(
    make: Make, tmp_path: Path
) -> None:
    action = make(tmp_path)
    assert action.needs_tick_loop
    scenario = _scenario()
    scenario.register_init(action)
    assert action in scenario._pre_tick_actions
    assert action not in scenario._init_actions


@pytest.mark.parametrize("make", _IN_THE_PHASE.values(), ids=_IN_THE_PHASE.keys())
def test_an_action_that_acts_once_stays_in_the_init_phase(
    make: Make, tmp_path: Path
) -> None:
    action = make(tmp_path)
    assert not action.needs_tick_loop
    scenario = _scenario()
    scenario.register_init(action)
    assert action in scenario._init_actions
    assert action not in scenario._pre_tick_actions


def test_routing_keeps_its_own_answer_over_the_trajectory_run() -> None:
    """RoutingAction derives TrajectoryRun, whose answer is always True, and
    keeps its own: a vehicle handed an untimed goal stays in the phase
    (test_routing_driver_mode.py)."""
    routing = RoutingAction.__dict__["needs_tick_loop"]
    assert routing is not TrajectoryRun.__dict__["needs_tick_loop"]
    assert RoutingAction.needs_tick_loop is routing
