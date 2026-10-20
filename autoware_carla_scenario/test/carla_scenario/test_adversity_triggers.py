"""When the adversities fire (#392, #398).

On the synthetic crossroads of :mod:`._route_maps`: the eastbound W approach
is W_IN2 (x -60..-10, y = -1.75), signalised with its stop line at x = -12,
the outer lane W_OUT_LANE2 at y = -5.25 on its right; the junction begins at
x = -10.
"""

from __future__ import annotations

import math
import random

import pytest

from autoware_carla_scenario.adversity import (
    AdversarialCommand,
    AdversityKind,
    AdversityLocation,
    AgentFootprint,
    CommandKind,
    adversity_command,
)
from autoware_carla_scenario.adversity.observation import (
    AgentSnapshot,
    Scene,
    SignalState,
)
from autoware_carla_scenario.adversity.triggers import (
    TriggerThresholds,
    brake_check,
    cut_in,
    head_on,
    ignore_leader,
    jaywalking,
    lane_change_abort,
    traffic_rule,
)
from autoware_carla_scenario.route.geometry import MapFeatures

from . import _route_maps as rm

CAR = AgentFootprint(5.0, 2.0)
PERSON = AgentFootprint(0.5, 0.5, is_vehicle=False)
INNER = -1.75
OUTER = -5.25


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    lanelet_map = rm.crossroads()
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


def _car(
    name: str,
    x: float,
    y: float = INNER,
    speed: float = 10.0,
    acceleration: float = 0.0,
) -> AgentSnapshot:
    return AgentSnapshot(name, x, y, 0.0, speed, acceleration, CAR)


def _scene(features: MapFeatures, *agents: AgentSnapshot, red: bool = False) -> Scene:
    signals = {rm.TRAFFIC_LIGHT: SignalState.RED} if red else {}
    return Scene(features, list(agents), signals)


# -- BRAKE_CHECK ------------------------------------------------------------------


def test_a_brake_check_on_a_follower_that_keeps_coming(features: MapFeatures) -> None:
    seen = _scene(features, _car("ego", -40.0), _car("behind", -55.0)).vehicle("ego")
    command = brake_check(seen)
    assert command == AdversarialCommand.accelerate(-9.0, 2.0)


@pytest.mark.parametrize(
    ("ego", "behind"),
    [
        (_car("ego", -40.0, acceleration=1.0), _car("behind", -55.0)),  # speeding up
        (_car("ego", -40.0), _car("behind", -55.0, acceleration=-1.0)),  # it slows
        (
            _car("ego", -40.0, speed=0.0),
            _car("behind", -55.0, speed=0.0),
        ),  # both stopped
    ],
)
def test_no_brake_check(
    features: MapFeatures, ego: AgentSnapshot, behind: AgentSnapshot
) -> None:
    assert brake_check(_scene(features, ego, behind).vehicle("ego")) is None


def test_no_brake_check_without_a_follower(features: MapFeatures) -> None:
    assert brake_check(_scene(features, _car("ego", -40.0)).vehicle("ego")) is None


# -- IGNORE_LEADER ----------------------------------------------------------------


def test_a_leader_close_and_slow_is_ignored(features: MapFeatures) -> None:
    seen = _scene(
        features, _car("ego", -40.0), _car("ahead", -25.0, speed=5.0)
    ).vehicle("ego")
    command = ignore_leader(seen)
    assert command is not None and command.kind is CommandKind.ACCELERATION
    assert command.acceleration > 0.0 and command.duration == 2.0


def test_a_leader_at_the_same_speed_is_followed(features: MapFeatures) -> None:
    seen = _scene(
        features, _car("ego", -40.0), _car("ahead", -25.0, speed=9.0)
    ).vehicle("ego")
    assert ignore_leader(seen) is None


def test_a_far_leader_holds_nothing_back(features: MapFeatures) -> None:
    seen = _scene(
        features, _car("ego", -60.0, speed=5.0), _car("ahead", -25.0, speed=10.0)
    ).vehicle("ego")
    assert ignore_leader(seen) is None


# -- CUT_IN -----------------------------------------------------------------------


def test_a_cut_in_in_front_of_the_vehicle_too_close_behind(
    features: MapFeatures,
) -> None:
    seen = _scene(features, _car("ego", -40.0), _car("side", -46.0, OUTER)).vehicle(
        "ego"
    )
    assert cut_in(seen) == AdversarialCommand.change_lane("right", 1.0)


@pytest.mark.parametrize(
    "others",
    [
        [_car("side", -70.0, OUTER)],  # far enough behind: a plain lane change
        [_car("side", -46.0, OUTER), _car("front", -33.0, OUTER)],  # blocked ahead too
        [_car("side", -38.0, OUTER)],  # already abreast, beyond the overlap
    ],
)
def test_no_cut_in(features: MapFeatures, others: list[AgentSnapshot]) -> None:
    assert cut_in(_scene(features, _car("ego", -40.0), *others).vehicle("ego")) is None


# -- LANE_CHANGE_ABORT ------------------------------------------------------------


def test_a_lane_change_abort_drifts_half_a_lane_and_back(
    features: MapFeatures,
) -> None:
    seen = _scene(features, _car("ego", -40.0), _car("side", -46.0, OUTER)).vehicle(
        "ego"
    )
    command = lane_change_abort(seen, features)
    assert command is not None and command.kind is CommandKind.TRAJECTORY
    assert command.duration == pytest.approx(1.0)
    path = command.path
    assert len(path) == 11
    # Toward the outer lane on the right, half the 3.5 m to it, and back.
    middle, last = path[5], path[-1]
    assert (middle.x, middle.y) == pytest.approx((-35.0, INNER - 1.75))
    assert (last.x, last.y) == pytest.approx((-30.0, INNER))
    assert all(p.speed == 10.0 and p.yaw == pytest.approx(0.0) for p in path)


def test_no_lane_change_abort_where_its_lanelet_ends_too_soon(
    features: MapFeatures,
) -> None:
    # Blocked beside as for a cut-in, but W_IN2 ends within the second.
    seen = _scene(features, _car("ego", -18.0), _car("side", -24.0, OUTER)).vehicle(
        "ego"
    )
    assert cut_in(seen) is not None
    assert lane_change_abort(seen, features) is None


def test_no_lane_change_abort_below_the_moving_speed(features: MapFeatures) -> None:
    seen = _scene(features, _car("ego", -40.0), _car("side", -46.0, OUTER)).vehicle(
        "ego"
    )
    slow = TriggerThresholds(moving_speed=10.0)
    assert lane_change_abort(seen, features, slow) is None


def test_no_lane_change_abort_when_the_lane_beside_is_free(
    features: MapFeatures,
) -> None:
    seen = _scene(features, _car("ego", -40.0)).vehicle("ego")
    assert lane_change_abort(seen, features) is None


# -- HEAD_ON ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def hairpin() -> MapFeatures:
    """Eastbound A (x 0..30, y = 0), a U-turn B, then westbound C (y = 10)."""
    b = rm._Builder()
    b.lanelet(1, rm._line((0.0, 0.0), (30.0, 0.0)))
    b.lanelet(
        2,
        rm._arc((30.0, 5.0), 5.0, -math.pi / 2, math.pi / 2),
        ends=(0.0, math.pi),
    )
    b.lanelet(3, rm._line((30.0, 10.0), (0.0, 10.0)))
    return MapFeatures(b.map, rm.routing_graph(b.map))


def _towards(name: str, x: float, y: float, yaw: float, speed: float) -> AgentSnapshot:
    return AgentSnapshot(name, x, y, yaw, speed, 0.0, CAR)


def test_a_head_on_at_a_leader_coming_the_other_way(hairpin: MapFeatures) -> None:
    scene = _scene(
        hairpin,
        _towards("ego", 20.0, 0.0, 0.0, 10.0),
        _towards("ahead", 25.0, 10.0, math.pi, 5.0),
    )
    seen = scene.vehicle("ego")
    assert seen.leader is not None and seen.leader.name == "ahead"
    command = head_on(seen, scene.lane_of("ahead"))
    assert command is not None and command == ignore_leader(seen)
    assert (
        adversity_command(
            AdversityKind.HEAD_ON, scene, "ego", AdversityLocation.ANY, random.Random()
        )
        == command
    )


def test_no_head_on_at_a_leader_going_the_same_way(features: MapFeatures) -> None:
    scene = _scene(features, _car("ego", -40.0), _car("ahead", -25.0, speed=5.0))
    seen = scene.vehicle("ego")
    assert ignore_leader(seen) is not None  # it would ignore it, but not head-on
    assert head_on(seen, scene.lane_of("ahead")) is None


# -- TRAFFIC_RULE -----------------------------------------------------------------


def test_running_the_red_it_is_braking_for(features: MapFeatures) -> None:
    seen = _scene(features, _car("ego", -40.0, acceleration=-2.0), red=True).vehicle(
        "ego"
    )
    command = traffic_rule(seen)
    assert command is not None and command.acceleration > 0.0


def test_a_green_light_is_no_rule_to_break(features: MapFeatures) -> None:
    seen = _scene(features, _car("ego", -40.0, acceleration=-2.0)).vehicle("ego")
    assert seen.signal is not None and seen.signal.state is SignalState.UNKNOWN
    assert traffic_rule(seen) is None


def test_a_vehicle_held_back_by_its_leader_breaks_no_rule(
    features: MapFeatures,
) -> None:
    seen = _scene(
        features,
        _car("ego", -40.0, acceleration=-2.0),
        _car("ahead", -32.0, speed=0.0),
        red=True,
    ).vehicle("ego")
    assert traffic_rule(seen) is None


# -- JAYWALKING -------------------------------------------------------------------


def test_a_jaywalker_walks_straight_across(features: MapFeatures) -> None:
    person = AgentSnapshot("p", -40.0, -12.0, math.pi / 2, 0.0, 0.0, PERSON)
    seen = _scene(features, person).pedestrian("p")
    command = jaywalking(seen, random.Random(1))
    assert command is not None and command.kind is CommandKind.TRAJECTORY
    speed = 1.4 * random.Random(1).uniform(0.5, 2.0)
    end = command.path[-1]
    assert command.duration == pytest.approx(5.0)
    assert (end.x, end.y) == pytest.approx((-40.0, -12.0 + speed * 5.0))
    assert end.yaw == pytest.approx(math.pi / 2)


def test_no_jaywalking_on_a_crosswalk_or_far_from_a_road(features: MapFeatures) -> None:
    on = AgentSnapshot("p", -15.0, 5.0, 0.0, 1.0, 0.0, PERSON)
    far = AgentSnapshot("q", -40.0, -40.0, 0.0, 1.0, 0.0, PERSON)
    scene = _scene(features, on, far)
    assert jaywalking(scene.pedestrian("p"), random.Random(0)) is None
    assert jaywalking(scene.pedestrian("q"), random.Random(0)) is None


# -- the location gate and the thresholds ---------------------------------------


@pytest.mark.parametrize(
    ("ego_x", "location", "fires"),
    [
        # 30 m from the junction: not at the intersection yet; 10 m: there.
        (-40.0, AdversityLocation.INTERSECTION, False),
        (-20.0, AdversityLocation.INTERSECTION, True),
        (-40.0, AdversityLocation.ANY, True),
        (-20.0, AdversityLocation.HIGHWAY, False),
    ],
)
def test_the_location_gate_comes_first(
    features: MapFeatures, ego_x: float, location: AdversityLocation, fires: bool
) -> None:
    scene = _scene(features, _car("ego", ego_x), _car("behind", ego_x - 15.0))
    command = adversity_command(
        AdversityKind.BRAKE_CHECK, scene, "ego", location, random.Random(0)
    )
    assert (command is not None) is fires


def test_a_pedestrian_is_located_by_the_road_nearest_it(features: MapFeatures) -> None:
    person = AgentSnapshot("p", -40.0, -12.0, 0.0, 0.0, 0.0, PERSON)
    scene = _scene(features, person)
    rng = random.Random(0)
    assert (
        adversity_command(
            AdversityKind.JAYWALKING, scene, "p", AdversityLocation.ANY, rng
        )
        is not None
    )
    assert (
        adversity_command(
            AdversityKind.JAYWALKING, scene, "p", AdversityLocation.INTERSECTION, rng
        )
        is None
    )


def test_the_thresholds_can_be_overridden(features: MapFeatures) -> None:
    seen = _scene(features, _car("ego", -40.0), _car("behind", -55.0)).vehicle("ego")
    longer = brake_check(seen, TriggerThresholds(duration=3.0))
    assert longer is not None and longer.duration == 3.0
    seen = _scene(
        features, _car("ego", -40.0), _car("ahead", -25.0, speed=5.0)
    ).vehicle("ego")
    assert ignore_leader(seen, TriggerThresholds(acceleration_gain=50.0)) is None
