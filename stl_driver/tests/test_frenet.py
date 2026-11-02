# SPDX-License-Identifier: Apache-2.0
"""The route coordinate system.

Most of this is sign conventions, which is exactly why it is worth testing:
a flipped lateral sign produces a policy that steers confidently in the wrong
direction, and nothing else in the stack would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from stl_driver.frenet import build_route_table, distance_behind_start, project_to_frenet
from tests.conftest import curved_route, straight_route


def table_for(route: np.ndarray, ds: float = 1.0, length_m: float = 200.0):
    return build_route_table(route, ds=ds, length_m=length_m)


def test_the_origin_projects_to_the_start_of_a_route_that_starts_there():
    table = table_for(straight_route())
    station, offset = project_to_frenet(table, np.zeros((1, 2)))
    assert station[0] == pytest.approx(0.0, abs=1e-6)
    assert offset[0] == pytest.approx(0.0, abs=1e-6)


def test_station_measures_distance_along_the_route():
    table = table_for(straight_route())
    station, _ = project_to_frenet(table, np.array([[37.0, 0.0]]))
    assert station[0] == pytest.approx(37.0, abs=1e-6)


def test_offset_is_positive_to_the_left():
    """The sign convention, pinned.

    The rig frame is x forward, y left. A point at +y is to the left of a
    route running along +x, so its offset must be positive.
    """
    table = table_for(straight_route())
    _, offset = project_to_frenet(table, np.array([[10.0, 1.5]]))
    assert offset[0] == pytest.approx(1.5, abs=1e-6)


def test_offset_is_negative_to_the_right():
    table = table_for(straight_route())
    _, offset = project_to_frenet(table, np.array([[10.0, -1.5]]))
    assert offset[0] == pytest.approx(-1.5, abs=1e-6)


def test_a_route_shifted_left_puts_the_ego_to_its_right():
    """The case the policy actually sees: the ego is not on the centreline.

    If the route runs 1 m to our left, our offset from it is -1 m. Getting
    this backwards makes the policy correct a lateral error by doubling it.
    """
    route = straight_route()
    route[:, 1] += 1.0
    table = table_for(route)
    _, offset = project_to_frenet(table, np.zeros((1, 2)))
    assert offset[0] == pytest.approx(-1.0, abs=1e-3)


def test_projection_round_trips_on_a_straight_route():
    table = table_for(straight_route())
    points = np.array([[12.0, 0.8], [45.0, -1.1], [90.0, 0.0]])
    station, offset = project_to_frenet(table, points)
    for index, point in enumerate(points):
        recovered = table.frenet_to_xy(float(station[index]), float(offset[index]))
        assert np.allclose(recovered, point, atol=1e-3)


def test_projection_round_trips_on_a_curve():
    """Two centimetres over a 60 m-radius bend, which the 1 m grid bounds."""
    table = table_for(curved_route(), ds=1.0, length_m=120.0)
    stations = np.array([5.0, 20.0, 40.0, 60.0])
    offsets = np.array([0.0, 0.9, -0.7, 0.4])
    points = np.stack(
        [table.frenet_to_xy(float(s), float(d)) for s, d in zip(stations, offsets, strict=True)]
    )
    recovered_s, recovered_d = project_to_frenet(table, points)
    assert np.allclose(recovered_s, stations, atol=0.02)
    assert np.allclose(recovered_d, offsets, atol=0.02)


def test_the_table_is_uniformly_spaced():
    """What makes the differentiable lookup index arithmetic."""
    table = table_for(straight_route(), ds=0.5, length_m=50.0)
    spacing = np.linalg.norm(np.diff(table.xy, axis=0), axis=1)
    assert np.allclose(spacing, 0.5, atol=1e-6)


def test_the_table_extends_past_the_published_route():
    """A fast plan overruns the published horizon on an ordinary step.

    Extrapolating rather than clamping keeps the lookup smooth there; the
    `stay_on_route` rule is what discourages planning that far.
    """
    route = straight_route(length_m=80.0)
    table = table_for(route, ds=1.0, length_m=200.0)
    assert table.published_length_m == pytest.approx(80.0)
    assert table.table_length_m >= 200.0
    # Extrapolation continues along the final heading, not back to the origin.
    assert table.position_at(150.0)[0] == pytest.approx(150.0, abs=1e-3)


def test_tangents_are_unit_length():
    table = table_for(curved_route(), ds=1.0, length_m=120.0)
    assert np.allclose(np.linalg.norm(table.tangent, axis=1), 1.0, atol=1e-6)


def test_a_degenerate_route_is_rejected_rather_than_producing_nonsense():
    with pytest.raises(ValueError, match="at least two points"):
        table_for(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="zero length"):
        table_for(np.zeros((4, 3)))


def test_projection_handles_points_behind_the_route_start():
    """The ego can sit slightly behind the first waypoint after a step."""
    table = table_for(straight_route())
    station, offset = project_to_frenet(table, np.array([[-2.0, 0.0]]))
    assert station[0] == pytest.approx(0.0, abs=1e-6)
    assert offset[0] == pytest.approx(0.0, abs=1e-6)


def _rotated(route: np.ndarray, yaw: float, shift: np.ndarray) -> np.ndarray:
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    out = route.copy()
    out[:, :2] = route[:, :2] @ rotation.T + shift
    return out


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.5])
@pytest.mark.parametrize("ds", [1.0, 2.0])
def test_a_table_reaching_back_gives_a_point_behind_the_route_start_its_station(ds, yaw):
    """A route starting 1.4 m ahead of the ego, as one walked from the actor's
    centre does: the ego is 1.4 m behind its start, not on it (#1141)."""
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    route = _rotated(straight_route(), yaw, 1.4 * heading)
    behind = distance_behind_start(route, np.zeros(2))
    assert behind == pytest.approx(1.4)

    table = build_route_table(route, ds=ds, length_m=200.0, behind_m=behind)
    station, offset = project_to_frenet(table, np.zeros((1, 2)))

    assert station[0] - table.route_start_m == pytest.approx(-1.4, abs=1e-6)
    assert offset[0] == pytest.approx(0.0, abs=1e-6)
    assert table.published_length_m - table.route_start_m == pytest.approx(300.0)
    # Ahead of the route's start, nothing moves.
    plain = build_route_table(route, ds=ds, length_m=200.0)
    start = round(table.route_start_m / ds)
    assert np.allclose(table.xy[start:], plain.xy, atol=1e-9)
    assert np.allclose(table.curvature[start + 2 :], plain.curvature[2:], atol=1e-9)
    point = np.array([[37.0 * heading[0], 37.0 * heading[1]]])
    ahead, _ = project_to_frenet(table, point)
    plain_ahead, _ = project_to_frenet(plain, point)
    assert ahead[0] - table.route_start_m == pytest.approx(plain_ahead[0], abs=1e-6)


def test_the_ego_on_or_ahead_of_the_route_start_extends_nothing():
    route = straight_route()
    assert distance_behind_start(route, np.zeros(2)) == 0.0
    assert distance_behind_start(route, np.array([5.0, 3.0])) == 0.0
    table = build_route_table(route, ds=1.0, length_m=200.0, behind_m=0.0)
    assert table.route_start_m == 0.0
    assert table.published_length_m == pytest.approx(300.0)


def test_the_table_reaches_back_along_the_first_segment_of_a_curve():
    """Behind its start a curve is continued along its first segment, as it is
    past its end along its last."""
    route = curved_route()
    shifted = route.copy()
    shifted[:, 0] += 2.5
    table = build_route_table(
        shifted, ds=1.0, length_m=200.0, behind_m=distance_behind_start(shifted, np.zeros(2))
    )
    points = np.array([[0.0, 0.0], [1.0, 0.2]])
    station, offset = project_to_frenet(table, points)

    first = shifted[1, :2] - shifted[0, :2]
    first /= np.linalg.norm(first)
    relative = points - shifted[0, :2]
    assert station - table.route_start_m == pytest.approx(relative @ first, abs=1e-6)
    assert offset == pytest.approx(first[0] * relative[:, 1] - first[1] * relative[:, 0], abs=1e-6)
    assert station - table.route_start_m == pytest.approx([-2.5, -1.5], abs=1e-2)


@pytest.mark.parametrize("ds", [1.0, 2.0])
def test_station_is_exact_between_grid_points_on_a_straight_route(ds):
    """Every point of a segment, not only its front half (issue #1134).

    Projecting only onto the segment ahead of the nearest vertex clamped
    every point in the back half of a segment onto that vertex, so 36.7 m
    came back as 37 m -- an error of up to ds/2 with a jump at each midpoint.
    """
    table = table_for(straight_route(), ds=ds)
    true_station = np.arange(0.0, 90.0, 0.05)
    lateral = np.where(np.arange(len(true_station)) % 3 == 0, 0.8, -0.5)
    station, offset = project_to_frenet(table, np.stack([true_station, lateral], axis=1))
    assert np.allclose(station, true_station, atol=1e-6)
    assert np.allclose(offset, lateral, atol=1e-6)


def test_station_is_exact_between_grid_points_on_a_curve():
    """Centreline points of a tight bend project back onto their own station."""
    table = table_for(curved_route(radius_m=20.0, sweep_rad=1.5), ds=1.0, length_m=40.0)
    true_station = np.arange(0.0, 29.0, 0.1)
    points = table.frenet_to_xy_many(true_station, np.zeros_like(true_station))[0]
    station, offset = project_to_frenet(table, points)
    assert np.allclose(station, true_station, atol=1e-6)
    assert np.allclose(offset, 0.0, atol=1e-6)


def test_projection_round_trips_off_grid_on_a_curve():
    """Off the centreline and off the grid, the bend's own geometry bounds it.

    The residual error is the angle between the chord and the interpolated
    normal -- a few centimetres at 1.5 m on a 20 m bend -- not the half
    segment that clamping onto the vertex used to cost.
    """
    table = table_for(curved_route(radius_m=20.0, sweep_rad=1.5), ds=1.0, length_m=40.0)
    stations = np.array([10.7, 10.7, 10.9, 15.55, 20.95])
    offsets = np.array([0.0, 1.5, -1.5, 0.9, -0.7])
    points = table.frenet_to_xy_many(stations, offsets)[0]
    recovered_s, recovered_d = project_to_frenet(table, points)
    assert np.allclose(recovered_s, stations, atol=0.05)
    assert np.allclose(recovered_d, offsets, atol=0.01)


def test_projection_handles_points_beyond_the_table_end():
    table = table_for(straight_route(), ds=1.0, length_m=50.0)
    station, offset = project_to_frenet(table, np.array([[table.table_length_m + 3.0, 0.4]]))
    assert station[0] == pytest.approx(table.table_length_m, abs=1e-6)
    assert offset[0] == pytest.approx(0.4, abs=1e-6)
