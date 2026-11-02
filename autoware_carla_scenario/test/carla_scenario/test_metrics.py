"""Metrics (:mod:`autoware_carla_scenario.metrics`) and the condition on them.

AlpaSim's per-timestep metrics, read off a stand-in world: a straight road
along +x with driving lanes side by side, and boxes on it.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import BaseScenario, EgoConfig
from autoware_carla_scenario.authoring.registry import (
    METRIC_OPTIONS,
    get_condition_spec,
)
from autoware_carla_scenario.conditions import (
    AndCondition,
    ComparisonRule,
    MetricCondition,
    NotCondition,
    OrCondition,
    PersistentCondition,
    StickyCondition,
)
from autoware_carla_scenario.metric_batch import FrameRecorder, evaluate_metrics
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.entity._spawn import SpawnPointIndex
from autoware_carla_scenario import metrics
from autoware_carla_scenario.metrics import (
    COLLISION_ANY,
    COLLISION_AT_FAULT,
    COLLISION_FRONT,
    COLLISION_LATERAL,
    COLLISION_REAR,
    BUILT_IN_METRICS,
    LATERAL_REVERSALS,
    LATERAL_WOBBLE_M,
    MIN_DISTANCE_TO_LANE_BOUNDARY_M,
    MIN_DISTANCE_TO_OBSTACLE_M,
    OFFROAD,
    OFFROAD_OR_COLLISION,
    OFFROAD_OR_COLLISION_AT_FAULT,
    STEERING_REVERSALS,
    TURN_INDICATOR_CHANGES,
    WRONG_LANE,
    is_metric,
    read_metric,
    reset_ego_motion,
    track_ego_motion,
)
from autoware_carla_scenario.measures import BUILT_IN_MEASURES
from autoware_carla_scenario.odd import load_odd_binding, scenario_measure

from ._fake_world import ActorsById

#: A car's half-extents: 4.6 m by 2 m.
_CAR = (2.3, 1.0)
_LANE_WIDTH = 3.5


def _v(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


class _Box:
    def __init__(self, extent: tuple[float, float]) -> None:
        self.extent = _v(extent[0], extent[1], 0.8)
        self.location = _v(0.0, 0.0, 0.8)
        self.rotation = SimpleNamespace(yaw=0.0)


class _Actor:
    def __init__(
        self,
        actor_id: int,
        type_id: str,
        at: tuple[float, float],
        yaw: float = 0.0,
        extent: tuple[float, float] = _CAR,
        role: str = "",
    ) -> None:
        self.id = actor_id
        self.is_alive = True
        self.type_id = type_id
        self.attributes = {"role_name": role}
        self.bounding_box = _Box(extent)
        self._at = _v(at[0], at[1])
        self._yaw = yaw
        self.velocity = _v()
        self.steer = 0.0
        self.lights = 0

    def get_velocity(self) -> Any:
        return self.velocity

    def get_control(self) -> Any:
        return SimpleNamespace(steer=self.steer)

    def get_light_state(self) -> int:
        return self.lights

    def get_transform(self) -> Any:
        return SimpleNamespace(
            location=self._at, rotation=SimpleNamespace(yaw=self._yaw)
        )


class _Lane:
    """A straight driving lane along x, centred on *y*, heading *yaw*."""

    def __init__(
        self, lane_id: int, y: float, yaw: float, junction: bool = False
    ) -> None:
        self.lane_id = lane_id
        self.y = y
        self.yaw = yaw
        self.junction = junction


class _Waypoint:
    def __init__(self, road: "_Road", lane: _Lane, x: float) -> None:
        self._road = road
        self._lane = lane
        self._x = x
        self.lane_id = lane.lane_id
        self.lane_width = _LANE_WIDTH
        self.lane_type = int(carla.LaneType.Driving)
        self.is_junction = lane.junction
        self.transform = SimpleNamespace(
            location=_v(x, lane.y), rotation=SimpleNamespace(yaw=lane.yaw)
        )

    def _beside(self, towards_plus_y: bool) -> Optional["_Waypoint"]:
        step = _LANE_WIDTH if towards_plus_y else -_LANE_WIDTH
        for lane in self._road.lanes:
            if abs(lane.y - (self._lane.y + step)) < 1e-6:
                return _Waypoint(self._road, lane, self._x)
        return None

    # CARLA's frame is left-handed: heading +x, right is +y.
    def get_right_lane(self) -> Optional["_Waypoint"]:
        return self._beside(self._lane.yaw == 0.0)

    def get_left_lane(self) -> Optional["_Waypoint"]:
        return self._beside(self._lane.yaw != 0.0)


class _Road:
    def __init__(self, lanes: list[_Lane]) -> None:
        self.lanes = lanes

    def get_waypoint(
        self, location: Any, project_to_road: bool = True, lane_type: int = 2
    ) -> Optional[_Waypoint]:
        assert not project_to_road
        for lane in self.lanes:
            if abs(location.y - lane.y) <= _LANE_WIDTH / 2:
                return _Waypoint(self, lane, location.x)
        return None


#: Two lanes heading +x, at y = 0 (the ego's) and y = 3.5 (to its right), and
#: one heading -x to its left, at y = -3.5.
_TWO_WAY = _Road([_Lane(-1, 0.0, 0.0), _Lane(-2, 3.5, 0.0), _Lane(1, -3.5, 180.0)])
#: One lane heading +x, at y = 0.
_ONE_LANE = _Road([_Lane(-1, 0.0, 0.0)])

_FRAMES = iter(range(1, 1_000_000))


class _World(ActorsById):
    def __init__(self, actors: list[_Actor], road: Optional[_Road]) -> None:
        self._actors = actors
        self._road = road
        self.frame = next(_FRAMES)

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=self.frame)

    def get_actors(self) -> list[_Actor]:
        return self._actors

    def get_map(self) -> _Road:
        if self._road is None:
            raise RuntimeError("no map")
        return self._road


def _world(
    *others: _Actor,
    at: tuple[float, float] = (0.0, 0.0),
    yaw: float = 0.0,
    road: Optional[_Road] = _TWO_WAY,
) -> _World:
    ego = _Actor(1, "vehicle.ego", at, yaw=yaw, role=str(EGO_ROLE_NAME))
    return _World([ego, *others], road)


def _car(actor_id: int, at: tuple[float, float], yaw: float = 0.0) -> _Actor:
    return _Actor(actor_id, "vehicle.npc", at, yaw=yaw)


def _read(key: str, world: _World) -> Optional[float]:
    return read_metric(key, world)  # type: ignore[arg-type]


def _collisions(world: _World) -> dict[str, Optional[float]]:
    return {
        key: _read(key, world)
        for key in (
            COLLISION_ANY,
            COLLISION_FRONT,
            COLLISION_REAR,
            COLLISION_LATERAL,
            COLLISION_AT_FAULT,
        )
    }


# ---------------------------------------------------------------------------
# Built in, under AlpaSim's names
# ---------------------------------------------------------------------------


def test_the_metrics_are_alpasim_s() -> None:
    assert set(BUILT_IN_METRICS) == {
        "collision_any",
        "collision_front",
        "collision_rear",
        "collision_lateral",
        "collision_at_fault",
        "offroad",
        "offroad_or_collision",
        "offroad_or_collision_at_fault",
        "wrong_lane",
        "min_distance_to_obstacle_m",
        "min_distance_to_lane_boundary_m",
        # Not AlpaSim's: NADE's weight, its last value the run's (#397).
        "importance_sampling_weight",
        # Not AlpaSim's: the ego's weaving, of the run so far.
        "lateral_reversals",
        "lateral_wobble_m",
        "steering_reversals",
        "turn_indicator_changes",
        # Not AlpaSim's: CriMe's and CommonRoad-Reach's, by the worker.
        *metrics.COMMONROAD_METRICS,
    }
    weight = BUILT_IN_METRICS["importance_sampling_weight"]
    assert (weight.unit, weight.aggregation) == ("ratio", "last")
    wobble = BUILT_IN_METRICS["lateral_wobble_m"]
    assert (wobble.unit, wobble.aggregation) == ("m", "max")
    for key in ("lateral_reversals", "steering_reversals", "turn_indicator_changes"):
        assert (BUILT_IN_METRICS[key].unit, BUILT_IN_METRICS[key].aggregation) == (
            "count",
            "last",
        )
    for key, metric in BUILT_IN_METRICS.items():
        assert metric.key == key
        if key in metrics.COMMONROAD_METRICS:
            continue  # test_commonroad_metrics.py
        if metric.unit not in ("ratio", "count") and metric is not wobble:
            assert metric.aggregation == ("min" if metric.unit == "m" else "max")
    assert read_metric("no_such_metric", _world()) is None  # type: ignore[arg-type]


def test_the_importance_sampling_weight_is_the_one_the_runner_publishes() -> None:
    from autoware_carla_scenario.metrics import set_importance_sampling_weight

    assert read_metric("importance_sampling_weight", _world()) is None  # no run
    set_importance_sampling_weight(0.125)
    try:
        assert read_metric("importance_sampling_weight", _world()) == 0.125  # type: ignore[arg-type]
    finally:
        set_importance_sampling_weight(None)


def test_a_metric_is_not_a_measure() -> None:
    assert not set(BUILT_IN_METRICS) & set(BUILT_IN_MEASURES)
    assert is_metric(OFFROAD)
    assert not is_metric("vehicle_ahead_gap_m")


def test_an_odd_cannot_map_onto_a_metric() -> None:
    with pytest.raises(ValueError, match="is a metric"):
        scenario_measure(COLLISION_AT_FAULT)


def test_a_binding_file_cannot_map_onto_a_metric(tmp_path: Path) -> None:
    (tmp_path / "taxonomy.yml").write_text(
        "TAXONOMY:\n  dynamic:\n    clearance: float length\n"
    )
    binding = tmp_path / "quality.yaml"
    binding.write_text(
        "openodd: [taxonomy.yml]\n"
        "probes:\n"
        "  clearance: {measure: min_distance_to_obstacle_m}\n"
    )

    with pytest.raises(ValueError, match="is a metric"):
        load_odd_binding(binding)


class _Scenario(BaseScenario):
    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return True


def test_a_scenario_cannot_register_a_metric_as_a_measure() -> None:
    scenario = _Scenario(
        EgoConfig(vehicle_type="vehicle.x", spawn_location=SpawnPointIndex(0))
    )

    with pytest.raises(ValueError, match="metric"):
        scenario.register_measure(OFFROAD, lambda world: 0.0)


def test_nothing_is_read_without_an_ego() -> None:
    world = _World([_car(2, (10.0, 0.0))], _TWO_WAY)

    for key in BUILT_IN_METRICS:
        assert _read(key, world) is None


def test_the_map_metrics_are_not_read_without_a_map() -> None:
    world = _world(road=None)

    assert _read(OFFROAD, world) is None
    assert _read(WRONG_LANE, world) is None
    assert _read(MIN_DISTANCE_TO_LANE_BOUNDARY_M, world) is None
    assert _read(OFFROAD_OR_COLLISION, world) is None
    # The collisions need no map.
    assert _read(COLLISION_ANY, world) == 0.0


# ---------------------------------------------------------------------------
# Collisions
# ---------------------------------------------------------------------------


def test_no_collision_when_apart() -> None:
    assert set(_collisions(_world(_car(2, (10.0, 0.0)))).values()) == {0.0}


def test_running_into_a_vehicle_ahead_is_a_front_collision_at_fault() -> None:
    found = _collisions(_world(_car(2, (4.5, 0.0))))

    assert found == {
        COLLISION_ANY: 1.0,
        COLLISION_FRONT: 1.0,
        COLLISION_REAR: 0.0,
        COLLISION_LATERAL: 0.0,
        COLLISION_AT_FAULT: 1.0,
    }


def test_being_run_into_from_behind_is_not_at_fault() -> None:
    found = _collisions(_world(_car(2, (-4.5, 0.0))))

    assert found[COLLISION_REAR] == 1.0
    assert found[COLLISION_FRONT] == 0.0
    assert found[COLLISION_ANY] == 1.0
    assert found[COLLISION_AT_FAULT] == 0.0


def test_a_side_swipe_is_lateral() -> None:
    # Shorter than the ego, alongside: it meets neither the front nor the rear.
    alongside = _Actor(2, "vehicle.npc", (0.0, 1.9), extent=(1.5, 1.0))
    found = _collisions(_world(alongside))

    assert found[COLLISION_LATERAL] == 1.0
    assert found[COLLISION_FRONT] == 0.0
    assert found[COLLISION_REAR] == 0.0
    assert found[COLLISION_AT_FAULT] == 1.0


def test_a_pedestrian_counts() -> None:
    walker = _Actor(2, "walker.pedestrian.1", (2.5, 0.0), extent=(0.3, 0.3))

    assert _read(COLLISION_FRONT, _world(walker)) == 1.0


def test_a_turned_box_is_turned() -> None:
    # Unturned, it would reach back to x = 1.1 and meet the ego; turned
    # across the lane it is 2 m long, from x = 2.4.
    ahead = _car(2, (3.4, 0.0), yaw=90.0)

    assert _read(COLLISION_ANY, _world(ahead)) == 0.0
    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, _world(ahead)) == pytest.approx(0.1)


def test_a_road_user_parked_under_the_map_is_out_of_the_scene() -> None:
    """Hidden road users are parked 500 m under the map, the ego may drive
    right over one (#844): no collision, and no obstacle."""
    parked = _car(2, (3.0, 0.0))
    parked._at.z = -500.0
    walker = _Actor(3, "walker.pedestrian.0001", (0.0, 1.0), extent=(0.3, 0.3))
    walker._at.z = -499.0

    assert set(_collisions(_world(parked, walker)).values()) == {0.0}
    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, _world(parked, walker)) is None
    # Above the line it is a road user again.
    parked._at.z = -50.0
    assert _read(COLLISION_ANY, _world(parked)) == 1.0


def test_props_are_not_road_users() -> None:
    prop = _Actor(2, "static.prop.barrel", (1.0, 0.0))

    assert _read(COLLISION_ANY, _world(prop)) == 0.0
    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, _world(prop)) is None


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------


def test_the_distance_is_box_to_box_to_the_nearest() -> None:
    world = _world(_car(2, (20.0, 0.0)), _car(3, (0.0, 5.0)))

    # Side to side: 5 m between centres less two half widths.
    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, world) == pytest.approx(3.0)


def test_the_distance_to_a_corner() -> None:
    world = _world(_car(2, (2.3 + 2.3 + 3.0, 1.0 + 1.0 + 4.0)))

    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, world) == pytest.approx(5.0)


def test_touching_is_no_distance() -> None:
    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, _world(_car(2, (4.6, 0.0)))) == 0.0


def test_alone_there_is_no_distance() -> None:
    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, _world()) is None


# ---------------------------------------------------------------------------
# The road
# ---------------------------------------------------------------------------


def test_on_the_lanes_is_on_the_road() -> None:
    world = _world()

    assert _read(OFFROAD, world) == 0.0
    assert _read(OFFROAD_OR_COLLISION, world) == 0.0


def test_straddling_two_lanes_is_on_the_road() -> None:
    assert _read(OFFROAD, _world(at=(0.0, 1.75))) == 0.0


def test_a_corner_off_the_lanes_is_off_the_road() -> None:
    # The right lane's edge is at y = 5.25; the ego's right side at 4.5 + 1.
    world = _world(at=(0.0, 4.5))

    assert _read(OFFROAD, world) == 1.0
    assert _read(OFFROAD_OR_COLLISION, world) == 1.0
    assert _read(OFFROAD_OR_COLLISION_AT_FAULT, world) == 1.0
    assert _read(MIN_DISTANCE_TO_LANE_BOUNDARY_M, world) == 0.0


def test_a_rear_collision_is_not_an_incident_at_fault() -> None:
    world = _world(_car(2, (-4.5, 0.0)))

    assert _read(OFFROAD_OR_COLLISION, world) == 1.0
    assert _read(OFFROAD_OR_COLLISION_AT_FAULT, world) == 0.0


def test_along_the_lane_is_the_right_way() -> None:
    assert _read(WRONG_LANE, _world(yaw=10.0)) == 0.0


def test_against_the_lane_is_the_wrong_way() -> None:
    assert _read(WRONG_LANE, _world(yaw=180.0)) == 1.0
    assert _read(WRONG_LANE, _world(yaw=-150.0)) == 1.0
    # In the oncoming lane, heading +x.
    assert _read(WRONG_LANE, _world(at=(0.0, -3.5))) == 1.0
    assert _read(WRONG_LANE, _world(at=(0.0, -3.5), yaw=180.0)) == 0.0


def test_no_lane_no_way() -> None:
    assert _read(WRONG_LANE, _world(at=(0.0, 20.0))) is None


def test_the_lane_boundary_is_the_edge_of_the_driving_lanes() -> None:
    # Lanes from y = -5.25 (the oncoming lane's left edge) to 5.25: the ego,
    # centred on y = 0 and 1 m to either side, is 4.25 m from both.
    assert _read(MIN_DISTANCE_TO_LANE_BOUNDARY_M, _world()) == pytest.approx(4.25)
    # 1 m to the right: 3.25 m from the right edge.
    assert _read(
        MIN_DISTANCE_TO_LANE_BOUNDARY_M, _world(at=(0.0, 1.0))
    ) == pytest.approx(3.25)


def test_on_one_lane_the_boundary_is_its_own() -> None:
    world = _world(at=(0.0, 0.25), road=_ONE_LANE)

    assert _read(MIN_DISTANCE_TO_LANE_BOUNDARY_M, world) == pytest.approx(0.5)


def test_a_turned_ego_reaches_further_across() -> None:
    world = _world(yaw=90.0, road=_ONE_LANE)

    # Across a 3.5 m lane, a 4.6 m car is off it.
    assert _read(OFFROAD, world) == 1.0
    assert _read(MIN_DISTANCE_TO_LANE_BOUNDARY_M, world) == 0.0


def test_lanes_beyond_the_search_radius_do_not_count() -> None:
    wide = _Road([_Lane(-1 - i, 3.5 * (i - 4), 0.0) for i in range(9)])

    # Nine lanes, the ego on the middle one.  To either side, the lanes
    # beside it start 1.75 and 5.25 m from its centre and count; the third,
    # 8.75 m away, does not: the area ends 8.75 m out, 7.75 m from its sides.
    assert _read(MIN_DISTANCE_TO_LANE_BOUNDARY_M, _world(road=wide)) == pytest.approx(
        7.75
    )


def test_a_frame_is_read_once() -> None:
    world = _world(_car(2, (10.0, 0.0)))
    first = _read(MIN_DISTANCE_TO_OBSTACLE_M, world)
    world._actors[1] = _car(2, (20.0, 0.0))

    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, world) == first
    world.frame = next(_FRAMES)
    assert _read(MIN_DISTANCE_TO_OBSTACLE_M, world) == pytest.approx(20.0 - 4.6)


def test_headings_wrap() -> None:
    assert _read(WRONG_LANE, _world(yaw=350.0)) == 0.0
    assert _read(WRONG_LANE, _world(yaw=-179.0)) == 1.0


# ---------------------------------------------------------------------------
# Weaving, of the run so far
# ---------------------------------------------------------------------------

#: The runner's tick: 20 Hz.
_DT = 0.05


def _drive(
    across: Any,
    seconds: float,
    *,
    road: _Road = _TWO_WAY,
    heading: float = 0.0,
    steer: Any = None,
    lights: Any = None,
) -> _World:
    """Drives the ego at 10 m/s along *heading* (degrees) for *seconds*,
    *across(t)* m/s to its right, steering *steer(t)*, every tick; the world of
    the last tick."""
    reset_ego_motion()
    world = _world(road=road)
    ego = world._actors[0]
    yaw = math.radians(heading)
    for i in range(round(seconds / _DT) + 1):
        t = i * _DT
        side = across(t)
        ego.velocity = _v(
            10.0 * math.cos(yaw) - side * math.sin(yaw),
            10.0 * math.sin(yaw) + side * math.cos(yaw),
        )
        ego.steer = 0.0 if steer is None else steer(t)
        ego.lights = 0 if lights is None else lights(t)
        world.frame = next(_FRAMES)
        track_ego_motion(world, t)  # type: ignore[arg-type]
    return world


def _weave(amplitude: float, hz: float) -> Any:
    """The sideways speed of a drift of *amplitude* m at *hz*."""
    w = 2.0 * math.pi * hz
    return lambda t: amplitude * w * math.cos(w * t)


def test_a_weave_turns_back_twice_a_cycle() -> None:
    # 0.3 m each way at 0.5 Hz for 10 s: five cycles, turning back at 0.5,
    # 1.5, ... 9.5 s.
    world = _drive(_weave(0.3, 0.5), 10.0)

    assert _read(LATERAL_REVERSALS, world) == 10.0
    # About a sine's RMS, 0.3 / sqrt(2), over the last two cycles: the line
    # through them takes a little.
    assert _read(LATERAL_WOBBLE_M, world) == pytest.approx(0.3 / math.sqrt(2), rel=0.1)


def test_holding_the_lane_is_not_weaving() -> None:
    # 5 cm each way: under the reversal gap.
    world = _drive(_weave(0.05, 0.5), 10.0)

    assert _read(LATERAL_REVERSALS, world) == 0.0
    assert _read(LATERAL_WOBBLE_M, world) == pytest.approx(0.05 / math.sqrt(2), rel=0.1)


def test_a_lane_change_is_a_drift_not_a_reversal() -> None:
    # A lane's width to the right over 4 s, smoothly, then straight on.
    def across(t: float) -> float:
        s = min(t / 4.0, 1.0)
        return _LANE_WIDTH * 6.0 * s * (1.0 - s) / 4.0

    reset_ego_motion()
    lane_change = _drive(across, 6.0)
    assert _read(LATERAL_REVERSALS, lane_change) == 0.0
    # Mostly the course: a tenth of the lane's width at the most.
    worst = 0.0
    for seconds in (2.0, 3.0, 4.0, 5.0, 6.0):
        wobble = _read(LATERAL_WOBBLE_M, _drive(across, seconds))
        assert wobble is not None
        worst = max(worst, wobble)
    assert worst < 0.1 * _LANE_WIDTH


def test_a_junction_holds_the_drift() -> None:
    _drive(lambda t: 1.0, 1.0, road=_Road([_Lane(-1, 0.0, 0.0, junction=True)]))

    assert metrics._MOTION[0].lateral == 0.0  # noqa: SLF001
    reset_ego_motion()


def test_a_lane_the_other_way_is_crossed_the_ego_s_way() -> None:
    _drive(lambda t: 1.0, 1.0, road=_Road([_Lane(1, 0.0, 180.0)]))

    # To the ego's right, as on a lane its own way.
    assert metrics._MOTION[0].lateral == pytest.approx(1.0, abs=0.1)  # noqa: SLF001
    reset_ego_motion()


def test_following_a_bend_is_not_a_drift() -> None:
    # The lane heads 30 degrees, and so does the ego.
    bend = _Road([_Lane(-1, 0.0, 30.0)])
    world = _drive(lambda t: 0.0, 4.0, road=bend, heading=30.0)

    assert _read(LATERAL_WOBBLE_M, world) == pytest.approx(0.0, abs=1e-9)
    assert _read(LATERAL_REVERSALS, world) == 0.0


def test_steering_back_and_forth_is_counted() -> None:
    def sawing(t: float) -> float:
        return 0.05 if int(t) % 2 else -0.05

    world = _drive(lambda t: 0.0, 5.0, steer=sawing)
    # Turns at 1, 2, 3, 4 and 5 s, the first from where it began.
    assert _read(STEERING_REVERSALS, world) == 4.0

    # A controller holding its course moves the steer by a few hundredths.
    jitter = _drive(lambda t: 0.0, 5.0, steer=lambda t: 0.02 * (-1) ** round(t / _DT))
    assert _read(STEERING_REVERSALS, jitter) == 0.0


_LEFT = int(carla.VehicleLightState.LeftBlinker)
_RIGHT = int(carla.VehicleLightState.RightBlinker)
_LOW_BEAM = int(carla.VehicleLightState.LowBeam)


def test_a_turn_signalled_and_cancelled_is_two_changes() -> None:
    def signal(t: float) -> int:
        return _LOW_BEAM | (_LEFT if 1.0 <= t < 4.0 else 0)

    world = _drive(lambda t: 0.0, 5.0, lights=signal)
    # The other lights are not the indicators.
    assert _read(TURN_INDICATOR_CHANGES, world) == 2.0


def test_a_flickering_indicator_is_counted_apart_from_the_steering() -> None:
    def flicker(t: float) -> int:
        return _LEFT if int(t * 4) % 2 else 0

    # Steady steering, an indicator on and off every quarter second, from 0.25 s
    # to 1.75 s.
    world = _drive(lambda t: 0.0, 1.9, lights=flicker, steer=lambda t: 0.1)
    assert _read(TURN_INDICATOR_CHANGES, world) == 7.0
    assert _read(STEERING_REVERSALS, world) == 0.0


def test_left_to_right_and_hazard_are_changes() -> None:
    def lights(t: float) -> int:
        return [_LEFT, _RIGHT, _LEFT | _RIGHT, 0][min(int(t), 3)]

    world = _drive(lambda t: 0.0, 3.5, lights=lights)
    assert _read(TURN_INDICATOR_CHANGES, world) == 3.0


def test_weaving_reads_nothing_until_the_ego_is_tracked() -> None:
    reset_ego_motion()
    world = _world()
    for key in (
        LATERAL_REVERSALS,
        LATERAL_WOBBLE_M,
        STEERING_REVERSALS,
        TURN_INDICATOR_CHANGES,
    ):
        assert _read(key, world) is None
    track_ego_motion(world, 0.0)  # type: ignore[arg-type]
    assert _read(LATERAL_REVERSALS, world) == 0.0
    # Three frames before there is a course to stray from.
    assert _read(LATERAL_WOBBLE_M, world) is None
    assert _read(LATERAL_REVERSALS, _world(road=_TWO_WAY)) is not None
    assert _read(LATERAL_REVERSALS, _World([], _TWO_WAY)) is None
    reset_ego_motion()


def test_a_tick_seen_twice_adds_nothing() -> None:
    world = _drive(_weave(0.3, 0.5), 2.0)
    before = _read(LATERAL_WOBBLE_M, world)
    world._actors[0].velocity = _v(10.0, 50.0)
    world.frame = next(_FRAMES)
    track_ego_motion(world, 2.0)  # type: ignore[arg-type]

    assert _read(LATERAL_WOBBLE_M, world) == before
    reset_ego_motion()


def test_the_runner_tracks_the_ego_every_tick() -> None:
    import inspect

    from autoware_carla_scenario import scenario_runner

    source = inspect.getsource(scenario_runner)
    assert "track_ego_motion(world, elapsed)" in source
    assert source.count("reset_ego_motion()") == 2


# ---------------------------------------------------------------------------
# Watching a metric
# ---------------------------------------------------------------------------


def _series(world_at: list[tuple[float, _World]], key: str) -> Any:
    """*key* over the frames of *world_at*, recorded and replayed on their
    map as the runner does once a run is over."""
    recorder = FrameRecorder()
    for t, world in world_at:
        recorder.record(world, t, None)  # type: ignore[arg-type]
    road = world_at[0][1]._road
    return evaluate_metrics(recorder.frames, road, [key])[key]


def test_an_event_condition_fails_the_run_at_its_first_frame() -> None:
    condition = MetricCondition(COLLISION_AT_FAULT, label="collision at fault")
    apart, hit = _car(2, (10.0, 0.0)), _car(2, (4.5, 0.0))
    series = _series(
        [
            (0.0, _world(apart)),
            (1.0, _world(apart)),
            (2.0, _world(hit)),
            (3.0, _world(apart)),
        ],
        COLLISION_AT_FAULT,
    )

    assert series.values == [0.0, 0.0, 1.0, 0.0]
    result = condition.judge(series.times, series.values)
    assert result is not None and result.passed
    assert result.elapsed_seconds == 2.0
    assert "collision_at_fault" in result.message
    details = condition.get_details()
    assert (details["ng_seconds"], details["first_ng_at"], details["worst"]) == (
        1.0,
        2.0,
        1.0,
    )


def test_a_distance_condition_compares_in_metres() -> None:
    condition = MetricCondition(
        MIN_DISTANCE_TO_OBSTACLE_M,
        ComparisonRule.LESS_THAN,
        1.0,
        label="closer than a metre",
    )
    series = _series(
        [(0.0, _world(_car(2, (5.7, 0.0)))), (0.5, _world(_car(2, (5.0, 0.0))))],
        MIN_DISTANCE_TO_OBSTACLE_M,
    )

    result = condition.judge(series.times, series.values)
    assert result is not None and "m less than" in result.message
    assert result.elapsed_seconds == 0.5
    assert condition.get_details() == {
        "metric": MIN_DISTANCE_TO_OBSTACLE_M,
        "rule": "LESS_THAN",
        "value": 1.0,
        "unit": "m",
        "tolerance": 1e-6,
        "max_ng_seconds": 0.0,
        "ng_seconds": 0.0,
        "first_ng_at": 0.5,
        "worst": pytest.approx(0.4),
    }


def test_a_condition_fails_once_its_metric_was_ng_long_enough() -> None:
    condition = MetricCondition(
        MIN_DISTANCE_TO_OBSTACLE_M,
        ComparisonRule.LESS_THAN,
        1.0,
        max_ng_seconds=0.25,
        label="near",
    )
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # NG for 0.2 s, clear, then NG again: 0.25 s are up 0.05 s into it.
    values: list[Optional[float]] = [0.5, 0.4, 2.0, 0.8, 0.3, 0.9, 2.0]
    result = condition.judge(times, values)
    assert result is not None
    assert result.elapsed_seconds == pytest.approx(0.35)
    assert condition.get_details()["ng_seconds"] == pytest.approx(0.5)
    assert condition.get_details()["worst"] == 0.3
    # Under the limit: it holds.
    assert condition.judge(times[:4], values[:4]) is None
    assert condition.get_details()["ng_seconds"] == pytest.approx(0.2)


def test_nothing_read_is_not_ng() -> None:
    condition = MetricCondition(
        MIN_DISTANCE_TO_OBSTACLE_M, ComparisonRule.LESS_THAN, 1.0, label="near"
    )
    # Alone: no obstacle to be near.
    series = _series([(0.0, _world()), (0.1, _world())], MIN_DISTANCE_TO_OBSTACLE_M)
    assert series.values == [None, None]
    assert condition.judge(series.times, series.values) is None
    assert condition.check(_world(), 0.0) is None  # type: ignore[arg-type]


def test_the_replay_reads_what_the_world_read() -> None:
    worlds = [
        _world(_car(2, (10.0, 0.0))),
        _world(_car(2, (4.5, 0.0))),
        _world(_car(2, (5.0, 3.5)), at=(0.0, 1.5)),
        _world(_car(2, (8.0, -3.5), yaw=180.0), yaw=170.0),
        _world(_car(2, (20.0, 0.0)), at=(0.0, 9.0)),
    ]
    keys = [k for k in BUILT_IN_METRICS if k not in metrics.COMMONROAD_METRICS]
    recorder = FrameRecorder()
    for i, world in enumerate(worlds):
        recorder.record(world, float(i), None)  # type: ignore[arg-type]
    replayed = evaluate_metrics(recorder.frames, _TWO_WAY, keys)
    for i, world in enumerate(worlds):
        for key in (
            "collision_any",
            "collision_at_fault",
            "offroad",
            "wrong_lane",
            "min_distance_to_obstacle_m",
            "min_distance_to_lane_boundary_m",
        ):
            assert replayed[key].values[i] == _read(key, world), (i, key)


def test_a_metric_condition_composes_with_nothing() -> None:
    condition = MetricCondition(OFFROAD, label="off the road")
    with pytest.raises(ValueError, match="judged once the run is over"):
        PersistentCondition(condition, 1.0)
    with pytest.raises(ValueError, match="judged once the run is over"):
        NotCondition(condition)
    with pytest.raises(ValueError, match="judged once the run is over"):
        AndCondition([condition, MetricCondition(WRONG_LANE, label="wrong")])
    with pytest.raises(ValueError, match="judged once the run is over"):
        OrCondition([MetricCondition(WRONG_LANE, label="wrong"), condition])
    with pytest.raises(ValueError, match="judged once the run is over"):
        StickyCondition(condition)
    scenario = _Scenario(
        EgoConfig(vehicle_type="vehicle.x", spawn_location=SpawnPointIndex(0))
    )
    with pytest.raises(ValueError, match="register_fail_condition"):
        scenario.register_pass_condition(condition)
    scenario.register_fail_condition(condition)
    with pytest.raises(ValueError, match="max_ng_seconds"):
        MetricCondition(OFFROAD, max_ng_seconds=-1.0, label="x")


def test_an_unknown_metric_is_refused() -> None:
    with pytest.raises(ValueError, match="not a metric"):
        MetricCondition("vehicle_ahead_gap_m", label="x")


def test_the_editor_offers_every_metric() -> None:
    assert [(o.value, o.label) for o in METRIC_OPTIONS] == [
        (m.key, m.title) for m in BUILT_IN_METRICS.values()
    ]
    assert all(m.title for m in BUILT_IN_METRICS.values())
    spec = get_condition_spec("metric")
    assert spec is not None
    assert spec.target == "..conditions:MetricCondition"


# ---------------------------------------------------------------------------
# Judged once the run is over
# ---------------------------------------------------------------------------


def _judged_run(
    *conditions: Any,
    result: Any,
    worlds: list[tuple[float, _World]],
    trajectory: Any = None,
) -> Any:
    from autoware_carla_scenario.scenario_runner import _judge_metrics

    scenario = _Scenario(
        EgoConfig(vehicle_type="vehicle.x", spawn_location=SpawnPointIndex(0))
    )
    for condition in conditions:
        scenario.register_fail_condition(condition)
    recorder = FrameRecorder()
    for t, world in worlds:
        recorder.record(world, t, None)  # type: ignore[arg-type]
    return _judge_metrics(
        scenario,
        recorder,
        worlds[-1][1],
        result,
        "test",
        trajectory,  # type: ignore[arg-type]
    )


def _hit_at_two() -> list[tuple[float, _World]]:
    apart, hit = _car(2, (10.0, 0.0)), _car(2, (4.5, 0.0))
    return [
        (0.0, _world(apart)),
        (1.0, _world(apart)),
        (2.0, _world(hit)),
        (3.0, _world(hit)),
    ]


def test_a_metric_condition_fails_a_run_that_passed() -> None:
    from autoware_carla_scenario.conditions import ScenarioResult
    from autoware_carla_scenario.failure import FailureCategory

    passed = ScenarioResult(passed=True, message="done", elapsed_seconds=3.0)
    result = _judged_run(
        MetricCondition(WRONG_LANE, label="wrong lane"),
        MetricCondition(COLLISION_AT_FAULT, max_ng_seconds=0.5, label="hit"),
        result=passed,
        worlds=_hit_at_two(),
    )

    assert not result.passed
    assert result.elapsed_seconds == 2.5
    assert result.failure.condition == "fail[1](hit)"
    assert result.failure.category is FailureCategory.COLLISION
    statuses = {s.label: s for s in result.condition_statuses}
    assert statuses["fail[1](hit)"].satisfied
    # NG at 2 s and 3 s: the last frame is the run's end, and counts no time.
    assert statuses["fail[1](hit)"].details["ng_seconds"] == 1.0
    assert statuses["fail[0](wrong lane)"].message == "not triggered"


def test_an_earlier_failure_stands() -> None:
    from autoware_carla_scenario.conditions import ScenarioResult
    from autoware_carla_scenario.failure import Failure, FailureCategory

    failed = ScenarioResult(
        passed=False,
        message="timed out",
        elapsed_seconds=1.5,
        failure=Failure("fail[1](timeout)", FailureCategory.PROGRESS, 1.5, "timed out"),
    )
    result = _judged_run(
        MetricCondition(COLLISION_AT_FAULT, label="hit"),
        result=failed,
        worlds=_hit_at_two(),
    )

    assert result is failed and failed.failure is not None
    assert failed.failure.condition == "fail[1](timeout)"
    # What the metric found is in the run's statuses, and among what else held.
    assert failed.failure.also_triggered == ["fail[0](hit)"]
    assert [s.label for s in result.condition_statuses] == ["fail[0](hit)"]


def test_the_runner_records_frames_and_judges_after_the_loop() -> None:
    import inspect

    from autoware_carla_scenario.scenario_runner import ScenarioRunner

    source = inspect.getsource(ScenarioRunner.run_scenario)
    assert "metric_recorder.record(" in source
    assert source.index("_judge_metrics(") < source.index("coverage.end(world")
    assert "if is_post_run(condition):" in source


class _Log:
    """What a run's trajectory log is told once the run is over."""

    def __init__(self) -> None:
        self.metrics: list[tuple[int, float, dict[str, Optional[float]]]] = []
        self.conditions: list[tuple[int, float, str, int, str, bool]] = []

    def record_metric(
        self, frame: int, t: float, values: dict[str, Optional[float]]
    ) -> None:
        self.metrics.append((frame, t, values))

    def record_condition_at(
        self, frame: int, t: float, role: str, index: int, label: str, satisfied: bool
    ) -> None:
        self.conditions.append((frame, t, role, index, label, satisfied))


def test_the_metrics_and_their_ng_frames_go_into_the_log() -> None:
    from autoware_carla_scenario.conditions import ScenarioResult

    log = _Log()
    _judged_run(
        MetricCondition(COLLISION_AT_FAULT, label="hit"),
        MetricCondition(WRONG_LANE, label="wrong lane"),
        result=ScenarioResult(passed=True, message="done", elapsed_seconds=3.0),
        worlds=_hit_at_two(),
        trajectory=log,
    )

    # A record a frame, every metric judged in it, in frame order.
    assert [(t, values) for _, t, values in log.metrics] == [
        (0.0, {COLLISION_AT_FAULT: 0.0, WRONG_LANE: 0.0}),
        (1.0, {COLLISION_AT_FAULT: 0.0, WRONG_LANE: 0.0}),
        (2.0, {COLLISION_AT_FAULT: 1.0, WRONG_LANE: 0.0}),
        (3.0, {COLLISION_AT_FAULT: 1.0, WRONG_LANE: 0.0}),
    ]
    frames = [frame for frame, _, _ in log.metrics]
    assert frames == sorted(frames) and len(set(frames)) == 4
    # Each condition on its first frame and whenever it changed.
    assert [(t, role, i, label, ng) for _, t, role, i, label, ng in log.conditions] == [
        (0.0, "fail", 0, "hit", False),
        (2.0, "fail", 0, "hit", True),
        (0.0, "fail", 1, "wrong lane", False),
    ]
