# SPDX-License-Identifier: Apache-2.0
"""The emergency plan starts at the ego, wherever the route table starts.

A route published slightly ahead of the rig -- on a bend, where stepping back
the rig's ``x`` along the arc leaves the route a millimetre ahead, or at the
start of the road network, where it starts at the actor's centre -- makes
`build_route_table` reach back behind it, and the ego then sits at a station
of a metre or two, not at the table's first point. The plan has to be laid
out from that station and that offset; measured from the first point, every
pose lands that far behind the ego.
"""

from __future__ import annotations

import numpy as np
import pytest

from stl_driver.config import STLDriverConfig
from stl_driver.fallback import emergency_brake_plan, hold_plan
from stl_driver.frenet import build_route_table, distance_behind_start, project_to_frenet
from tests.conftest import curved_route, straight_route

TIME_NOW_US = 1_700_000_000_000_000
CONFIG = STLDriverConfig()


def yaws_of(plan) -> np.ndarray:
    return np.array([pose.yaw for pose in plan.poses])


def braking_distance(speed_mps: float) -> float:
    """The distance the discrete profile covers before it stops, worked out
    independently of the code under test: each step loses ``a dt`` of speed
    and then travels ``v dt``."""
    horizon = CONFIG.horizon
    dt = horizon.step_s
    steps = np.arange(1, horizon.n_steps + 1)
    speeds = np.maximum(0.0, speed_mps - CONFIG.limits.emergency_decel_mps2 * dt * steps)
    return float(speeds.sum() * dt)


def straight_ahead(lead_m: float, lateral_m: float = 0.0) -> np.ndarray:
    """A straight route along ``+x``, starting ``lead_m`` ahead of the rig."""
    route = straight_route()
    route[:, 0] += lead_m
    route[:, 1] += lateral_m
    return route


def bend_ahead(lead_m: float, lateral_m: float = 0.0) -> np.ndarray:
    """A 60 m left-hand bend, starting ``lead_m`` ahead of the rig."""
    route = curved_route()
    route[:, 0] += lead_m
    route[:, 1] += lateral_m
    return route


ROUTES = {
    # The issue's case: about a centimetre ahead puts the ego two metres in.
    "straight, 1 cm ahead": straight_ahead(0.01),
    # The start of the road network: the actor's centre, 1.4 m ahead.
    "straight, 1.4 m ahead": straight_ahead(1.4),
    # A bend whose start is a millimetre ahead of the rig.
    "bend, 1 mm ahead": bend_ahead(0.001),
    # Off the centreline too, so the held offset is visible.
    "bend, 1.4 m ahead and 0.4 m left": bend_ahead(1.4, lateral_m=0.4),
}


def locate(route: np.ndarray):
    """The table and the ego's ``(station, offset)``, as the policy builds them."""
    table = build_route_table(
        route,
        ds=CONFIG.route_table_ds_m,
        length_m=CONFIG.route_table_length_m(),
        behind_m=distance_behind_start(route, np.zeros(2)),
    )
    station, offset = project_to_frenet(table, np.zeros((1, 2)))
    return table, float(station[0]), float(offset[0])


@pytest.fixture(params=sorted(ROUTES), ids=sorted(ROUTES))
def located(request):
    return locate(ROUTES[request.param])


@pytest.mark.parametrize("speed_mps", [2.0, 10.0])
def test_the_emergency_plan_starts_at_the_ego(located, speed_mps):
    table, station0, offset0 = located
    # These routes all start ahead: the case this file is about.
    assert station0 > 0.5

    plan = emergency_brake_plan(
        TIME_NOW_US, speed_mps, CONFIG, route=table, station0_m=station0, offset0_m=offset0
    )
    first = plan.positions[0, :2]
    # One step of braking from the ego, no further -- and not behind it.
    first_step = (speed_mps - CONFIG.limits.emergency_decel_mps2 * CONFIG.horizon.step_s) * (
        CONFIG.horizon.step_s
    )
    assert np.linalg.norm(first) == pytest.approx(first_step, abs=0.02)
    assert first[0] > 0.0


@pytest.mark.parametrize("speed_mps", [2.0, 10.0])
def test_the_emergency_plan_only_moves_forward(located, speed_mps):
    table, station0, offset0 = located
    plan = emergency_brake_plan(
        TIME_NOW_US, speed_mps, CONFIG, route=table, station0_m=station0, offset0_m=offset0
    )
    positions = np.concatenate([np.zeros((1, 2)), plan.positions[:, :2]])
    stations, _ = project_to_frenet(table, positions)
    assert np.all(np.diff(stations) >= -1e-6)
    assert stations[1] > stations[0]

    # And forward in the direction each pose faces, not merely along the
    # route: no step runs against its own heading.
    yaws = yaws_of(plan)
    headings = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)
    steps = np.diff(positions, axis=0)
    assert np.all(np.einsum("nk,nk->n", steps, headings) >= -1e-6)


@pytest.mark.parametrize("speed_mps", [2.0, 10.0])
def test_the_emergency_plan_stops_at_the_braking_distance(located, speed_mps):
    table, station0, offset0 = located
    plan = emergency_brake_plan(
        TIME_NOW_US, speed_mps, CONFIG, route=table, station0_m=station0, offset0_m=offset0
    )
    stations, offsets = project_to_frenet(table, plan.positions[:, :2])
    # Millimetres: off the centreline of a bend, the plan is laid out along
    # interpolated normals and projected back onto polyline segments, and the
    # two disagree that much. The bug this guards against is metres.
    assert stations[-1] - station0 == pytest.approx(braking_distance(speed_mps), abs=5e-3)
    # The offset is held: braking must not also swerve.
    assert np.allclose(offsets, offset0, atol=5e-3)


@pytest.mark.parametrize("speed_mps", [2.0, 10.0])
def test_on_a_straight_route_the_stop_is_that_far_ahead_of_the_rig(speed_mps):
    table, station0, offset0 = locate(straight_ahead(0.01))
    plan = emergency_brake_plan(
        TIME_NOW_US, speed_mps, CONFIG, route=table, station0_m=station0, offset0_m=offset0
    )
    assert plan.positions[-1, 0] == pytest.approx(braking_distance(speed_mps), abs=1e-6)
    assert np.allclose(plan.positions[:, 1], 0.0, atol=1e-9)
    assert np.allclose(yaws_of(plan), 0.0, atol=1e-9)


def test_the_emergency_plan_faces_along_the_bend_and_keeps_that_at_rest():
    """Heading is the route's tangent where each pose is; once stopped, the
    poses stay put and so does the heading -- as `plan_xy` holds a normal
    plan's heading at rest -- rather than snapping back to the rig's own."""
    route = bend_ahead(0.001)
    table, station0, offset0 = locate(route)
    plan = emergency_brake_plan(
        TIME_NOW_US, 10.0, CONFIG, route=table, station0_m=station0, offset0_m=offset0
    )
    stations, _ = project_to_frenet(table, plan.positions[:, :2])
    tangent_yaw = np.arctan2(*table.frenet_to_xy_many(stations, np.zeros_like(stations))[1].T[::-1])
    assert np.allclose(yaws_of(plan), tangent_yaw, atol=1e-3)

    # The bend turns left, and the stop is a few metres into it.
    assert yaws_of(plan)[-1] > 0.05
    at_rest = np.flatnonzero(np.diff(stations) < 1e-9)
    assert len(at_rest) > 10
    assert np.allclose(yaws_of(plan)[at_rest[0] :], yaws_of(plan)[-1], atol=1e-9)


def test_without_a_route_the_emergency_plan_brakes_straight_ahead():
    plan = emergency_brake_plan(TIME_NOW_US, 10.0, CONFIG)
    assert plan.positions[-1, 0] == pytest.approx(braking_distance(10.0), abs=1e-6)
    assert np.allclose(plan.positions[:, 1], 0.0)


def test_the_hold_plan_starts_at_the_rig():
    """The other fallback has no route to be offset along: the rig frame's
    origin is the ego, so it starts there."""
    plan = hold_plan(TIME_NOW_US, 5.0, CONFIG)
    assert 0.0 < plan.positions[0, 0] < 5.0 * CONFIG.horizon.step_s + 1e-9
    assert np.all(np.diff(plan.positions[:, 0]) >= 0.0)
    assert np.allclose(plan.positions[:, 1], 0.0)
    assert np.all(np.asarray(plan.timestamps_us) > TIME_NOW_US)
