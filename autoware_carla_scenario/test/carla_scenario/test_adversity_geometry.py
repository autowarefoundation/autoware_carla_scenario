"""The adversity math ported from TeraSim (#387)."""

from __future__ import annotations

import math

import pytest

from autoware_carla_scenario.adversity import (
    AgentFootprint,
    IdmParameters,
    PredictedState,
    car_following_acceleration,
    circle_centers,
    circle_offsets,
    circle_radius,
    collision_time,
    free_flow_acceleration,
    interpolate_trajectory,
    lane_change_longitudinal_speed,
    predict_distance_velocity,
    segments_intersect,
    trajectories_intersect,
)
from autoware_carla_scenario.kinematics import normalize_angle_rad

CAR = AgentFootprint(length=5.0, width=2.0)
PEDESTRIAN = AgentFootprint(length=0.5, width=0.5, is_vehicle=False)


def _at(x: float, y: float, t: float = 0.0, speed: float = 0.0) -> PredictedState:
    return PredictedState(x, y, 0.0, speed, t)


def _straight(
    x: float, y: float, yaw: float, speed: float, steps: int = 11, dt: float = 0.5
) -> list[PredictedState]:
    return [
        PredictedState(
            x + speed * i * dt * math.cos(yaw),
            y + speed * i * dt * math.sin(yaw),
            yaw,
            speed,
            i * dt,
        )
        for i in range(steps)
    ]


# -- geometry -----------------------------------------------------------------


def test_a_car_is_four_circles_straddling_its_centre() -> None:
    # ceil(5 / 2) + 1 = 4 circles, 1.25 m apart, centred on the car.
    centers = circle_centers(_at(10.0, 0.0), circle_offsets(CAR))
    assert [round(x, 6) for x, _ in centers] == [8.125, 9.375, 10.625, 11.875]
    assert all(y == 0.0 for _, y in centers)
    assert circle_radius(CAR) == pytest.approx(math.hypot(0.625, 1.0))


def test_an_odd_count_puts_a_circle_on_the_centre_along_the_heading() -> None:
    # ceil(4 / 2) + 1 = 3 circles; heading north lines them up along y.
    footprint = AgentFootprint(length=4.0, width=2.0)
    centers = circle_centers(
        PredictedState(0.0, 0.0, math.pi / 2, 0.0, 0.0), circle_offsets(footprint)
    )
    assert [y for _, y in centers] == pytest.approx([-4.0 / 3, 0.0, 4.0 / 3])
    assert all(abs(x) < 1e-9 for x, _ in centers)


def test_a_pedestrian_is_one_circle_of_its_larger_dimension() -> None:
    point = PredictedState(1.0, 2.0, 0.3, 0.0, 0.0)
    assert circle_centers(point, circle_offsets(PEDESTRIAN)) == [(1.0, 2.0)]
    assert circle_radius(AgentFootprint(0.4, 0.8, is_vehicle=False)) == 0.4


@pytest.mark.parametrize(
    ("angle", "wrapped"),
    [
        (0.0, 0.0),
        (math.pi, -math.pi),
        (3 * math.pi / 2, -math.pi / 2),
        (-7.0, -7.0 + 2 * math.pi),
    ],
)
def test_normalize_angle_rad(angle: float, wrapped: float) -> None:
    assert normalize_angle_rad(angle) == pytest.approx(wrapped)


# -- collision ----------------------------------------------------------------


def test_two_cars_closing_head_on_collide_when_they_meet() -> None:
    east = _straight(0.0, 0.0, 0.0, 10.0)
    west = _straight(40.0, 0.0, math.pi, 10.0)
    # Centres meet at t = 2 s; at 1.5 s the front circles are still 6.25 m
    # apart, so 2 s is the first 0.5 s sample to overlap.
    assert collision_time(east, CAR, west, CAR) == 2.0
    assert trajectories_intersect(east, CAR, west, CAR)


def test_cars_in_neighbouring_lanes_do_not_collide_unless_buffered() -> None:
    lane1 = _straight(0.0, 0.0, 0.0, 10.0)
    lane2 = _straight(0.0, 3.5, 0.0, 10.0)
    assert collision_time(lane1, CAR, lane2, CAR) is None
    assert not trajectories_intersect(lane1, CAR, lane2, CAR)
    # 2 * 1.18 m of radius leaves ~1.14 m; a 0.6 m buffer on each closes it.
    assert collision_time(lane1, CAR, lane2, CAR, buffer=0.6) == 0.0


def test_paths_crossing_in_the_same_step_conflict_without_overlapping() -> None:
    # Two pedestrians swap places within one 1 s step: no sample overlaps,
    # but the two segments of that step cross.
    walker1 = [
        _at(0.0, 0.0, 0.0, 4.0),
        _at(4.0, 4.0, 1.0, 4.0),
    ]
    walker2 = [
        _at(4.0, 0.0, 0.0, 4.0),
        _at(0.0, 4.0, 1.0, 4.0),
    ]
    assert collision_time(walker1, PEDESTRIAN, walker2, PEDESTRIAN) is None
    assert trajectories_intersect(walker1, PEDESTRIAN, walker2, PEDESTRIAN)


def test_paths_crossing_at_different_times_do_not_conflict() -> None:
    north = [
        _at(5.0, -10.0, 0.0),
        _at(5.0, -5.0, 1.0),
        _at(5.0, 0.0, 2.0),
        _at(5.0, 5.0, 3.0),
    ]
    east = [
        _at(-20.0, 0.0, 0.0),
        _at(-15.0, 0.0, 1.0),
        _at(-10.0, 0.0, 2.0),
        _at(-5.0, 0.0, 3.0),
    ]
    assert not trajectories_intersect(north, PEDESTRIAN, east, PEDESTRIAN)


def test_agents_that_start_far_apart_are_not_checked() -> None:
    east = _straight(0.0, 0.0, 0.0, 10.0)
    west = _straight(60.0, 0.0, math.pi, 20.0)
    assert collision_time(east, CAR, west, CAR) is not None
    assert not trajectories_intersect(east, CAR, west, CAR)
    assert trajectories_intersect(east, CAR, west, CAR, distance_threshold=70.0)
    assert not trajectories_intersect([], CAR, west, CAR)


@pytest.mark.parametrize(
    ("c", "d", "crosses"),
    [
        ((0.0, 1.0), (1.0, -1.0), True),  # proper crossing
        ((1.0, 0.0), (1.0, 1.0), True),  # touching the end point
        ((0.5, 0.0), (1.5, 0.0), True),  # collinear overlap
        ((1.5, 0.0), (2.0, 0.0), False),  # collinear, apart
        ((0.0, 1.0), (1.0, 1.0), False),  # parallel
    ],
)
def test_segments_intersect(
    c: tuple[float, float], d: tuple[float, float], crosses: bool
) -> None:
    assert segments_intersect((0.0, 0.0), (1.0, 0.0), c, d) is crosses


# -- prediction ---------------------------------------------------------------


def test_a_braking_agent_stops_instead_of_reversing() -> None:
    distances, velocities = predict_distance_velocity(
        10.0, -4.0, [0.0, 1.0, 2.0, 3.0, 4.0], 30.0
    )
    assert velocities == [10.0, 6.0, 2.0, 0.0, 0.0]
    assert distances == [0.0, 8.0, 12.0, 13.0, 13.0]


def test_an_accelerating_agent_is_held_at_its_maximum_speed() -> None:
    distances, velocities = predict_distance_velocity(10.0, 2.0, [0.0, 1.0, 2.0], 11.0)
    assert velocities == [10.0, 11.0, 11.0]
    assert distances == [0.0, 10.5, 21.5]


def test_a_lane_change_takes_the_lateral_speed_out_of_the_speed() -> None:
    assert lane_change_longitudinal_speed(5.0, 3.0) == pytest.approx(4.0)
    assert lane_change_longitudinal_speed(2.0, 3.0) == 0.0
    assert lane_change_longitudinal_speed(5.0, 3.0, duration=2.0) == pytest.approx(
        math.sqrt(25 - 2.25)
    )


def test_interpolation_resamples_on_the_resolution_and_turns_the_short_way() -> None:
    trajectory = [
        PredictedState(0.0, 0.0, math.radians(170.0), 0.0, 0.0),
        PredictedState(10.0, 0.0, math.radians(-170.0), 10.0, 1.0),
    ]
    resampled = interpolate_trajectory(trajectory, 0.25)
    assert [p.t for p in resampled] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [p.x for p in resampled] == [0.0, 2.5, 5.0, 7.5, 10.0]
    # Halfway between 170 and -170 degrees the short way is 180, not 0.
    assert abs(abs(resampled[2].yaw) - math.pi) < 1e-9
    assert resampled[2].speed == 5.0


def test_interpolation_stops_at_the_last_whole_step() -> None:
    trajectory = _straight(0.0, 0.0, 0.0, 1.0, steps=3, dt=0.5)
    assert [p.t for p in interpolate_trajectory(trajectory, 0.4)] == pytest.approx(
        [0.0, 0.4, 0.8]
    )
    assert interpolate_trajectory(trajectory[:1], 0.1) == trajectory[:1]


# -- IDM ----------------------------------------------------------------------


def test_free_flow_accelerates_towards_the_desired_speed() -> None:
    assert free_flow_acceleration(0.0, 13.89) == pytest.approx(2.6)
    assert free_flow_acceleration(13.89, 13.89) == pytest.approx(0.0)
    assert free_flow_acceleration(20.0, 13.89) < 0.0
    assert free_flow_acceleration(0.0, 0.0) == 0.0


def test_a_close_leader_holds_the_follower_back() -> None:
    # TeraSim's IgnoreLeader fires when free flow beats car following by 1.5.
    ff = free_flow_acceleration(10.0, 13.89)
    cf = car_following_acceleration(10.0, 13.89, gap=10.0, leader_speed=5.0)
    assert ff - cf > 1.5
    far = car_following_acceleration(10.0, 13.89, gap=3000.0, leader_speed=10.0)
    assert far == pytest.approx(ff, abs=1e-3)


def test_car_following_never_brakes_harder_than_the_emergency_deceleration() -> None:
    params = IdmParameters(emergency_deceleration=7.0)
    assert car_following_acceleration(20.0, 20.0, 1.0, 0.0, params) == -7.0
    assert car_following_acceleration(20.0, 20.0, 0.0, 0.0, params) == -7.0
