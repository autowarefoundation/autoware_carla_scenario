"""Unit tests for the policy warm-up's run-ups and the carried ego."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.entity._policy_warmup import (
    PolicyWarmup,
    _CarriedVehicle,
    _CarriedWalker,
    _LanePath,
)


def _walker(yaw: float) -> MagicMock:
    actor = MagicMock()
    actor.get_transform.return_value = carla.Transform(
        carla.Location(x=10.0, y=20.0, z=1.0), carla.Rotation(yaw=yaw)
    )
    return actor


@pytest.mark.parametrize("yaw", [0.0, 30.0, 90.0, -135.0])
def test_a_walker_is_moved_back_along_its_heading(yaw: float) -> None:
    """At tick t it stands (ticks - t) steps of its velocity behind its pose."""
    actor = _walker(yaw)
    walker = _CarriedWalker(actor, 1.5, ticks=10, step_s=0.05)
    heading = math.radians(yaw)
    for tick in (0, 4, 10):
        walker.place(tick)
        placed = actor.set_transform.call_args.args[0]
        back_s = (10 - tick) * 0.05
        assert placed.location.x == pytest.approx(
            10.0 - 1.5 * math.cos(heading) * back_s
        )
        assert placed.location.y == pytest.approx(
            20.0 - 1.5 * math.sin(heading) * back_s
        )
        assert placed.location.z == pytest.approx(1.0)
        assert placed.rotation.yaw == pytest.approx(yaw)


def test_carried_actor_lives_with_the_run_up() -> None:
    """``_CarriedActor`` moved into the checked ``_policy_warmup`` (#898) and
    stays importable from ``carla_driver_entity`` as the same class."""
    from autoware_carla_scenario.entity import _policy_warmup, carla_driver_entity

    assert carla_driver_entity._CarriedActor is _policy_warmup._CarriedActor
    assert _policy_warmup._CarriedActor.__module__ == _policy_warmup.__name__


_STEP_S = 0.05
_TICKS = 20
_SPEED = 10.0  # a 10 m run-up over the 20 ticks


class _Waypoint:
    """A waypoint on a straight lane along +x that starts at ``-lane_back_m``."""

    def __init__(self, x: float, lane_back_m: float) -> None:
        self.x = x
        self.lane_back_m = lane_back_m
        self.transform = carla.Transform(carla.Location(x=x), carla.Rotation())

    def previous(self, distance: float) -> list:
        x = self.x - distance
        return [_Waypoint(x, self.lane_back_m)] if x >= -self.lane_back_m - 1e-9 else []


class _Map:
    def __init__(self, lane_back_m: float) -> None:
        self.lane_back_m = lane_back_m

    def get_waypoint(self, location: carla.Location, project_to_road: bool = True):
        return _Waypoint(location.x, self.lane_back_m)


def _vehicle(lane_back_m: float, speed: float = _SPEED) -> _CarriedVehicle:
    """The ego, carried at ``speed`` along a lane ``lane_back_m`` long behind it."""
    actor = MagicMock()
    start = carla.Transform(carla.Location(), carla.Rotation())
    actor.get_transform.return_value = start
    actor.get_velocity.return_value = carla.Vector3D()
    actor.get_angular_velocity.return_value = carla.Vector3D()
    actor.get_control.return_value.gear = 1
    vehicle = _CarriedVehicle.__new__(_CarriedVehicle)
    vehicle.actor = actor
    vehicle.speed = speed
    vehicle._ticks = _TICKS
    vehicle._step_s = _STEP_S
    vehicle.start = start
    vehicle.path = _LanePath(
        _Map(lane_back_m), start, speed * _TICKS * _STEP_S, "vehicle.test"
    )
    vehicle._powertrain = MagicMock()
    vehicle._powertrain.pedals.return_value = (0.3, 0.0)
    return vehicle


def _warmup(vehicle: _CarriedVehicle) -> PolicyWarmup:
    warmup = PolicyWarmup.__new__(PolicyWarmup)
    warmup._ticks = _TICKS
    warmup._tick = 0
    warmup._step_s = _STEP_S
    warmup._ego = vehicle
    warmup._vehicles = [vehicle]
    warmup._walkers = []
    warmup._frozen = []
    return warmup


def _reported_speeds(warmup: PolicyWarmup) -> list:
    speeds = []
    while not warmup.done:
        velocity = warmup.velocity()
        speeds.append(math.hypot(velocity.x, velocity.y))
        warmup.advance()
    return speeds


def test_an_ego_waiting_at_a_short_lanes_end_reports_standing_still() -> None:
    """A lane 5 m long holds the ego at its end for the first half of a 10 m
    run-up: it reports zero speed while it waits, its speed once it moves (#1305)."""
    vehicle = _vehicle(lane_back_m=5.0)
    assert vehicle.path.length_m == pytest.approx(5.0)
    speeds = _reported_speeds(_warmup(vehicle))
    assert speeds[:10] == pytest.approx([0.0] * 10)
    assert speeds[10:] == pytest.approx([_SPEED] * 10)
    # What the policy is told agrees with where the path has the ego.
    for tick, speed in enumerate(speeds):
        here = vehicle.path.at(vehicle.back_m(tick))
        ahead = vehicle.path.at(vehicle.back_m(tick + 1))
        moved = ahead.location.x - here.location.x
        assert speed == pytest.approx(moved / _STEP_S)


def test_an_ego_setting_off_mid_tick_reports_the_share_of_the_step() -> None:
    """A lane ending between two ticks' path points (3.5 m, with 0.2 m a tick):
    the tick it sets off on reports the distance it covers in it."""
    vehicle = _vehicle(lane_back_m=3.75, speed=4.0)
    assert vehicle.path.length_m == pytest.approx(3.5)
    speeds = _reported_speeds(_warmup(vehicle))
    assert speeds[:2] == pytest.approx([0.0, 0.0])
    assert speeds[2] == pytest.approx(2.0)
    assert speeds[3:] == pytest.approx([4.0] * 17)


def test_an_ego_on_a_long_enough_lane_reports_its_speed_throughout() -> None:
    vehicle = _vehicle(lane_back_m=50.0)
    assert vehicle.path.length_m == pytest.approx(_SPEED * _TICKS * _STEP_S)
    assert _reported_speeds(_warmup(vehicle)) == pytest.approx([_SPEED] * _TICKS)


@pytest.mark.parametrize("lane_back_m", [5.0, 50.0])
def test_the_ego_is_released_at_its_initial_speed(lane_back_m: float) -> None:
    """The last tick of the run-up reports, and arrival commands, the initial speed."""
    vehicle = _vehicle(lane_back_m)
    warmup = _warmup(vehicle)
    warmup._tick = _TICKS - 1
    velocity = warmup.velocity()
    assert (velocity.x, velocity.y) == pytest.approx((_SPEED, 0.0))
    warmup.advance()
    assert warmup.done
    commanded = vehicle.actor.set_target_velocity.call_args.args[0]
    assert (commanded.x, commanded.y) == pytest.approx((_SPEED, 0.0))
