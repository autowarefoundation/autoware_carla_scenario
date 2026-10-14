"""Metrics (:mod:`autoware_carla_scenario.metrics`) and the condition on them.

AlpaSim's per-timestep metrics, read off a stand-in world: a straight road
along +x with driving lanes side by side, and boxes on it.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import BaseScenario, EgoConfig
from autoware_carla_scenario.authoring.registry import (
    METRIC_OPTIONS,
    get_condition_spec,
)
from autoware_carla_scenario.conditions import ComparisonRule, MetricCondition
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.entity._spawn import SpawnPointIndex
from autoware_carla_scenario.metrics import (
    COLLISION_ANY,
    COLLISION_AT_FAULT,
    COLLISION_FRONT,
    COLLISION_LATERAL,
    COLLISION_REAR,
    BUILT_IN_METRICS,
    MIN_DISTANCE_TO_LANE_BOUNDARY_M,
    MIN_DISTANCE_TO_OBSTACLE_M,
    OFFROAD,
    OFFROAD_OR_COLLISION,
    OFFROAD_OR_COLLISION_AT_FAULT,
    WRONG_LANE,
    is_metric,
    read_metric,
    reset_metrics,
)
from autoware_carla_scenario.measures import BUILT_IN_MEASURES, read_measure
from autoware_carla_scenario.odd import load_odd_binding, scenario_measure

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
        self.type_id = type_id
        self.attributes = {"role_name": role}
        self.bounding_box = _Box(extent)
        self._at = _v(at[0], at[1])
        self._yaw = yaw

    def get_location(self) -> SimpleNamespace:
        return self._at

    def get_velocity(self) -> SimpleNamespace:
        return _v()

    def get_transform(self) -> Any:
        return SimpleNamespace(
            location=self._at, rotation=SimpleNamespace(yaw=self._yaw)
        )


class _Lane:
    """A straight driving lane along x, centred on *y*, heading *yaw*."""

    def __init__(self, lane_id: int, y: float, yaw: float) -> None:
        self.lane_id = lane_id
        self.y = y
        self.yaw = yaw


class _Waypoint:
    def __init__(self, road: "_Road", lane: _Lane, x: float) -> None:
        self._road = road
        self._lane = lane
        self._x = x
        self.lane_id = lane.lane_id
        self.lane_width = _LANE_WIDTH
        self.lane_type = int(carla.LaneType.Driving)
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


class _World:
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


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    # A stand-in world is collected after its test, and the next one can take
    # its id(): what was read for that id must not carry over.
    reset_metrics()
    yield
    reset_metrics()


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
    }
    for key, metric in BUILT_IN_METRICS.items():
        assert metric.key == key
        assert metric.aggregation == ("min" if metric.unit == "m" else "max")
    assert read_metric("no_such_metric", _world()) is None  # type: ignore[arg-type]


def test_a_metric_is_not_a_measure() -> None:
    assert not set(BUILT_IN_METRICS) & set(BUILT_IN_MEASURES)
    assert is_metric(OFFROAD)
    assert not is_metric("vehicle_ahead_gap_m")
    assert read_measure(OFFROAD, _world()) is None  # type: ignore[arg-type]


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


def test_the_map_measures_are_not_read_without_a_map() -> None:
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
# Watching a metric
# ---------------------------------------------------------------------------


def test_an_event_condition_holds_while_it_happens() -> None:
    condition = MetricCondition(COLLISION_AT_FAULT, label="collision at fault")

    apart = _world(_car(2, (10.0, 0.0)))
    assert condition.check(apart, 1.0) is None  # type: ignore[arg-type]
    hit = _world(_car(2, (4.5, 0.0)))
    result = condition.check(hit, 2.0)  # type: ignore[arg-type]
    assert result is not None
    assert result.passed
    assert result.elapsed_seconds == 2.0
    assert "collision_at_fault" in result.message


def test_a_distance_condition_compares_in_metres() -> None:
    condition = MetricCondition(
        MIN_DISTANCE_TO_OBSTACLE_M,
        ComparisonRule.LESS_THAN,
        1.0,
        label="closer than a metre",
    )

    clear = _world(_car(2, (5.7, 0.0)))
    assert condition.check(clear, 0.0) is None  # type: ignore[arg-type]
    close = _world(_car(2, (5.0, 0.0)))
    result = condition.check(close, 0.0)  # type: ignore[arg-type]
    assert result is not None and "m less than" in result.message
    assert condition.get_details() == {
        "metric": MIN_DISTANCE_TO_OBSTACLE_M,
        "rule": "LESS_THAN",
        "value": 1.0,
        "unit": "m",
        "tolerance": 1e-6,
    }


def test_nothing_read_does_not_hold() -> None:
    condition = MetricCondition(
        MIN_DISTANCE_TO_OBSTACLE_M, ComparisonRule.LESS_THAN, 1.0, label="near"
    )

    # Alone: no obstacle to be near.
    assert condition.check(_world(), 0.0) is None  # type: ignore[arg-type]


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
