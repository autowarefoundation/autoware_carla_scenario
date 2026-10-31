"""A walker's speed, from where it stands, wherever it is read (#900).

CARLA's walker controller places its walker, so ``get_velocity()`` reads 0
for one walking on it.  Each walker below does that: it reads 0 and moves
0.07 m every 0.05 s, 1.4 m/s.  Every vehicle keeps what CARLA reports.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest

from autoware_carla_scenario.conditions.comparison import ComparisonRule
from autoware_carla_scenario.conditions.composition import speed as speed_module
from autoware_carla_scenario.conditions.composition.relative_speed import (
    RelativeSpeedCondition,
)
from autoware_carla_scenario.conditions.composition.time_headway import (
    TimeHeadwayCondition,
)
from autoware_carla_scenario.conditions.composition.time_to_collision import (
    TimeToCollisionCondition,
)
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.measures import (
    CROSSING_PEDESTRIAN_GAP_M,
    CROSSING_PEDESTRIAN_SPEED_MS,
    read_measure,
    set_measured_scenario,
)

from ._fake_world import ActorsById, FakeActorList

#: How far a walker on its controller moves a tick, and the tick.
_STEP_M = 0.07
_TICK_S = 0.05
_WALKING_MS = _STEP_M / _TICK_S  # 1.4


def _velocities() -> Any:
    """A fresh :class:`WalkerVelocities` (imported here, so the condition and
    measure tests below also run, and fail, where the module is missing)."""
    from autoware_carla_scenario.kinematics.walker_motion import WalkerVelocities

    return WalkerVelocities()


def _walker(actor_id: int, where: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        id=actor_id,
        type_id="walker.pedestrian.0001",
        get_location=lambda: where,
        get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )


class _Actor:
    """An actor that is where it is and reports the velocity it is given."""

    def __init__(
        self,
        actor_id: int,
        type_id: str,
        at: tuple[float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
        role: str = "",
        yaw: float = 0.0,
    ) -> None:
        self.id = actor_id
        self.is_alive = True
        self.type_id = type_id
        self.attributes = {"role_name": role}
        self.at = SimpleNamespace(x=at[0], y=at[1], z=0.0)
        self.v = SimpleNamespace(x=velocity[0], y=velocity[1], z=0.0)
        self.yaw = yaw

    def get_location(self) -> SimpleNamespace:
        return self.at

    def get_velocity(self) -> SimpleNamespace:
        return self.v

    def get_transform(self) -> Any:
        return SimpleNamespace(location=self.at, rotation=SimpleNamespace(yaw=self.yaw))


class _World(ActorsById):
    """A world with a clock: its frame and simulated time step together."""

    def __init__(self, actors: list[_Actor]) -> None:
        self.actors = actors
        self.frame = 1
        self.elapsed = 10.0

    def tick(self) -> None:
        self.frame += 1
        self.elapsed += _TICK_S

    def get_snapshot(self) -> Any:
        return SimpleNamespace(
            frame=self.frame,
            timestamp=SimpleNamespace(elapsed_seconds=self.elapsed),
        )

    def get_actors(self) -> FakeActorList:
        return FakeActorList(self.actors)


# ---------------------------------------------------------------------------
# The velocity itself
# ---------------------------------------------------------------------------


def test_a_walker_on_its_controller_is_measured_by_where_it_stands() -> None:
    """CARLA's walker controller places its walker: get_velocity() reads 0."""
    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    walker = _walker(9101, where)
    velocities = _velocities()
    assert velocities.walker(walker, 1.0).magnitude() == 0.0
    where.x = 0.6
    assert velocities.walker(walker, 1.5).magnitude() == pytest.approx(1.2)
    # Asked again the same tick: the same.
    assert velocities.walker(walker, 1.5).magnitude() == pytest.approx(1.2)
    assert velocities.walker(walker, 2.0).magnitude() == 0.0  # stood
    # The clock gone back (a new run): what CARLA says.
    where.x = 50.0
    assert velocities.walker(walker, 0.1).magnitude() == 0.0
    # No clock: what CARLA says.
    where.x = 60.0
    assert velocities.walker(walker, None).magnitude() == 0.0


def test_each_reader_measures_on_its_own_and_forgets() -> None:
    """One reader asking does not change what another measures, and a run's
    end (clear) or a walker gone (forget) starts it over."""
    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    walker = _walker(9103, where)
    one, other = _velocities(), _velocities()
    one.walker(walker, 1.0)
    other.walker(walker, 1.0)
    where.x = 0.5
    assert one.walker(walker, 1.5).x == pytest.approx(1.0)
    where.x = 1.5
    # The other last asked at 1.0, at 0: 1.5 m over a second.
    assert other.walker(walker, 2.0).x == pytest.approx(1.5)
    one.clear()
    assert one.walker(walker, 3.0).x == 0.0
    other.forget(9103)
    assert other.walker(walker, 3.0).x == 0.0


def test_a_vehicle_reads_what_carla_reports() -> None:
    velocities = _velocities()
    car = _Actor(5, "vehicle.tesla.model3", (0.0, 0.0), (5.0, 0.0))
    assert velocities.velocity(car, 1.0).x == 5.0  # type: ignore[arg-type]
    car.at.x = 100.0  # placed far away: still what CARLA says
    assert velocities.velocity(car, 2.0).x == 5.0  # type: ignore[arg-type]


def test_simulated_seconds_reads_the_snapshot() -> None:
    from autoware_carla_scenario.kinematics.walker_motion import simulated_seconds

    world = _World([])
    assert simulated_seconds(world) == pytest.approx(10.0)  # type: ignore[arg-type]
    assert simulated_seconds(SimpleNamespace()) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The conditions
# ---------------------------------------------------------------------------


def test_a_walkers_standstill_reads_its_steps() -> None:
    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    walker = _walker(9102, where)
    condition = speed_module.SpeedCondition(
        entity_name="w", value=0.1, rule=ComparisonRule.LESS_THAN_OR_EQUAL, label="s"
    )
    # Read first, it has nothing to measure from: not known, not standing.
    assert condition._extract_speed_component(walker, None, 1.0) is None  # type: ignore[arg-type]
    where.y = 1.0
    moving = condition._extract_speed_component(walker, None, 2.0)  # type: ignore[arg-type]
    assert moving == pytest.approx(1.0)
    stood = condition._extract_speed_component(walker, None, 3.0)  # type: ignore[arg-type]
    assert stood == 0.0


def test_a_walker_first_read_walking_on_its_controller_does_not_stand() -> None:
    """A condition gated until its walker is already walking (on CARLA's
    controller, reading 0) is not satisfied by a standstill on its first
    check: it waits for a measured speed."""
    walker = _Actor(9104, "walker.pedestrian.0001", (0.0, 0.0), role="w")
    world = _World([walker])
    condition = speed_module.SpeedCondition(
        entity_name="w", value=0.1, rule=ComparisonRule.LESS_THAN_OR_EQUAL, label="s"
    )
    assert condition._check(world, world.elapsed) is None  # type: ignore[arg-type]
    _step(world, walker)
    assert condition._check(world, world.elapsed) is None  # type: ignore[arg-type]
    world.tick()  # it stops
    assert condition._check(world, world.elapsed) is not None  # type: ignore[arg-type]


def test_a_walker_walked_by_its_control_reads_carlas_velocity_at_first() -> None:
    """A walker on a WalkerControl reports its velocity: read first, that,
    and measured after -- the two agree."""
    walker = _Actor(9105, "walker.pedestrian.0001", (0.0, 0.0), (1.4, 0.0))
    velocities = _velocities()
    first = velocities.velocity(walker, 1.0)  # type: ignore[arg-type]
    assert first is not None and first.x == pytest.approx(1.4)
    walker.at.x = _STEP_M
    later = velocities.velocity(walker, 1.0 + _TICK_S)  # type: ignore[arg-type]
    assert later is not None and later.x == pytest.approx(1.4)


def test_a_teleported_walker_is_not_a_sprint() -> None:
    """A walker put where its walk starts, or brought in from under the map,
    reads what CARLA reports for that read, not a jump over a tick."""
    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    walker = _walker(9106, where)
    velocities = _velocities()
    velocities.walker(walker, 1.0)
    where.x = 30.0  # put 30 m on in a tick
    assert velocities.walker(walker, 1.05).magnitude() == 0.0
    assert velocities.velocity(walker, 1.05) is None  # type: ignore[arg-type]
    where.x += _STEP_M  # and walks on from there
    assert velocities.walker(walker, 1.1).x == pytest.approx(_WALKING_MS)
    # Parked under the map, then brought in where it was parked, a while on.
    where.z = -500.0
    velocities.walker(walker, 2.0)
    where.z = 0.0
    assert velocities.walker(walker, 100.0).magnitude() == 0.0


def _crossing() -> tuple[_World, _Actor, _Actor]:
    """An ego at 10 m/s along +x, and a walker 20 m ahead walking across."""
    ego = _Actor(1, "vehicle.ego", (0.0, 0.0), (10.0, 0.0), role=str(EGO_ROLE_NAME))
    walker = _Actor(2, "walker.pedestrian.0001", (20.0, -3.0), role="walker")
    return _World([ego, walker]), ego, walker


def _step(world: _World, walker: _Actor, dx: float = 0.0, dy: float = _STEP_M) -> None:
    world.tick()
    walker.at.x += dx
    walker.at.y += dy


def test_relative_speed_reads_a_walkers_steps() -> None:
    world, _ego, walker = _crossing()
    condition = RelativeSpeedCondition(
        "walker", EGO_ROLE_NAME, value=-9.0, rule=ComparisonRule.GREATER_THAN, label="r"
    )
    assert condition._check(world, world.elapsed) is None  # type: ignore[arg-type]
    _step(world, walker)
    assert condition._check(world, world.elapsed) is not None  # type: ignore[arg-type]
    relative = condition._measure(world, world.elapsed)  # type: ignore[arg-type]
    assert relative == pytest.approx(_WALKING_MS - 10.0)  # not -10.0


def test_time_to_collision_reads_a_walkers_steps() -> None:
    """A walker walking towards a parked car: a finite TTC, not none."""
    car = _Actor(3, "vehicle.parked", (0.0, 5.0), role="car")
    walker = _Actor(2, "walker.pedestrian.0001", (0.0, -5.0), role="walker")
    world = _World([car, walker])
    condition = TimeToCollisionCondition(
        "walker", "car", value=10.0, rule=ComparisonRule.LESS_THAN, label="t"
    )
    assert condition._check(world, world.elapsed) is None  # type: ignore[arg-type]
    _step(world, walker)
    assert condition._check(world, world.elapsed) is not None  # type: ignore[arg-type]
    ttc = condition._measure(world, world.elapsed)  # type: ignore[arg-type]
    assert ttc == pytest.approx((10.0 - _STEP_M) / _WALKING_MS)


def test_time_to_collision_keeps_a_vehicles_reported_velocity() -> None:
    """A vehicle's velocity is what CARLA reports, wherever it is put."""
    car = _Actor(3, "vehicle.car", (0.0, 0.0), (5.0, 0.0), role="car")
    place = _Actor(4, "vehicle.parked", (50.0, 0.0), role="parked")
    world = _World([car, place])
    condition = TimeToCollisionCondition(
        "car", "parked", value=10.0, rule=ComparisonRule.LESS_THAN, label="t"
    )
    assert condition._check(world, world.elapsed) is None  # type: ignore[arg-type]
    world.tick()
    car.at.x = 1.0  # teleported 1 m in a tick: still 5 m/s, 9.8 s
    assert condition._check(world, world.elapsed) is not None  # type: ignore[arg-type]
    car.at.x = 0.0
    assert condition._check(world, world.elapsed) is None  # type: ignore[arg-type]


def test_time_headway_reads_a_walkers_steps() -> None:
    car = _Actor(3, "vehicle.parked", (14.0, 0.0), role="car")
    walker = _Actor(2, "walker.pedestrian.0001", (0.0, 0.0), role="walker")
    world = _World([car, walker])
    condition = TimeHeadwayCondition(
        "walker", "car", value=20.0, rule=ComparisonRule.LESS_THAN, label="h"
    )
    assert condition._check(world, world.elapsed) is None  # type: ignore[arg-type]
    _step(world, walker, dx=_STEP_M, dy=0.0)
    assert condition._check(world, world.elapsed) is not None  # type: ignore[arg-type]
    headway = condition._measure(world, world.elapsed)  # type: ignore[arg-type]
    assert headway == pytest.approx((14.0 - _STEP_M) / _WALKING_MS)


def test_a_condition_registered_once_measures_each_run_afresh() -> None:
    """A condition registered once is checked again next run, after the
    runner closes it: what it measured last run is forgotten."""
    world, _ego, walker = _crossing()
    condition = RelativeSpeedCondition(
        "walker", EGO_ROLE_NAME, value=-9.0, rule=ComparisonRule.GREATER_THAN, label="r"
    )
    condition._check(world, 1.0)  # type: ignore[arg-type]
    condition.close()  # the run is over
    # Next run, the same walker id, first checked later than the last read.
    walker.at.y += 5.0
    assert condition._measure(world, 30.0) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The crossing-pedestrian measures
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_scenario() -> Iterator[None]:
    set_measured_scenario(None)
    yield
    set_measured_scenario(None)


@pytest.mark.usefixtures("_no_scenario")
def test_the_crossing_pedestrian_measures_read_a_walkers_steps() -> None:
    world, _ego, walker = _crossing()
    # The first frame has nothing to measure from: standing, as CARLA says.
    assert read_measure(CROSSING_PEDESTRIAN_SPEED_MS, world) is None  # type: ignore[arg-type]
    _step(world, walker)
    speed = read_measure(CROSSING_PEDESTRIAN_SPEED_MS, world)  # type: ignore[arg-type]
    assert speed == pytest.approx(_WALKING_MS)
    assert read_measure(CROSSING_PEDESTRIAN_GAP_M, world) == pytest.approx(20.0)  # type: ignore[arg-type]
    # Asked again the same frame: the same.
    assert read_measure(CROSSING_PEDESTRIAN_SPEED_MS, world) == pytest.approx(  # type: ignore[arg-type]
        _WALKING_MS
    )


@pytest.mark.usefixtures("_no_scenario")
def test_a_new_run_measures_its_walkers_afresh() -> None:
    world, _ego, walker = _crossing()
    read_measure(CROSSING_PEDESTRIAN_SPEED_MS, world)  # type: ignore[arg-type]
    set_measured_scenario(None)  # a run ended
    # Far from where the old run saw it, a moment later: not a sprint.
    _step(world, walker, dy=1.0)
    assert read_measure(CROSSING_PEDESTRIAN_SPEED_MS, world) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A walker on its controller watching for a threat off the kerb
# ---------------------------------------------------------------------------


def test_a_turn_back_threat_is_read_at_the_walkers_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.actions import _road_threat
    from autoware_carla_scenario.entity import _walker_ai

    asked: list[tuple[float, float]] = []

    def road_threat(
        world: Any, location: Any, wx: float, wy: float, *rest: Any
    ) -> Optional[str]:
        asked.append((wx, wy))
        return None

    monkeypatch.setattr(_road_threat, "road_threat", road_threat)
    world = _World([])
    walker = SimpleNamespace(
        id=9201,
        get_world=lambda: world,
        get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    at = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    try:
        _walker_ai._threat(walker, at, (6.0, 3.0, 1.0), standing=False)  # type: ignore[arg-type]
        world.tick()
        at = SimpleNamespace(x=0.0, y=_STEP_M, z=0.0)
        _walker_ai._threat(walker, at, (6.0, 3.0, 1.0), standing=False)  # type: ignore[arg-type]
    finally:
        _walker_ai.release(9201)
    assert asked[0] == (0.0, 0.0)
    assert asked[1][0] == pytest.approx(0.0)
    assert asked[1][1] == pytest.approx(_WALKING_MS)
    assert math.isclose(math.hypot(*asked[1]), _WALKING_MS)
