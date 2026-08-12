# SPDX-License-Identifier: Apache-2.0
"""Reading the world.

The frame transform is the risky part, so it is checked against the *runtime's
own* composition rule (``anchor @ rig_pose == local_pose``) rather than
against a second copy of my arithmetic.  If both were mine, they would agree
about a mistake.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import (
    AABB,
    CarlaActorState,
    CarlaRendererData,
    DynamicState,
    TrafficLightState,
    Vec3,
)
from carla_driver_interface.grpc_api import Pose as PoseProto
from carla_driver_interface.grpc_api import Quat as QuatProto

from stl_driver.config import STLDriverConfig
from stl_driver.frenet import build_route_table
from stl_driver.perception import forecast_actors, read_traffic_light
from tests.conftest import straight_route

CONFIG = STLDriverConfig()


def pose_proto(x: float, y: float, yaw: float = 0.0) -> PoseProto:
    pose = Pose.from_xyz_yaw(x, y, 0.0, yaw)
    quat = pose.quat_xyzw
    return PoseProto(
        vec=Vec3(x=x, y=y, z=0.0),
        quat=QuatProto(w=quat[3], x=quat[0], y=quat[1], z=quat[2]),
    )


def actor(
    x: float,
    y: float,
    *,
    yaw: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    length: float = 4.5,
    width: float = 2.0,
    track_id: str = "a",
) -> CarlaActorState:
    return CarlaActorState(
        track_id=track_id,
        type_id="vehicle.tesla.model3",
        pose_local_to_aabb=pose_proto(x, y, yaw),
        aabb=AABB(size_x=length, size_y=width, size_z=1.6),
        dynamic_state=DynamicState(linear_velocity=Vec3(x=vx, y=vy, z=0.0)),
    )


def renderer(
    *actors: CarlaActorState,
    light: int = TrafficLightState.TRAFFIC_LIGHT_STATE_NONE,
    light_distance_m: float = -1.0,
    speed_limit_mps: float = 0.0,
) -> CarlaRendererData:
    return CarlaRendererData(
        ego_traffic_light=light,
        ego_traffic_light_distance_m=light_distance_m,
        speed_limit_mps=speed_limit_mps,
        actors=list(actors),
    )


def table():
    return build_route_table(
        straight_route(), ds=CONFIG.route_table_ds_m, length_m=CONFIG.route_table_length_m()
    )


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def test_actors_are_brought_into_the_rig_frame():
    """Checked against the driver service's own composition rule.

    The service publishes a plan as ``anchor @ rig_pose``. So a point whose
    local position is ``anchor @ p_rig`` must come back as ``p_rig``.
    """
    anchor = Pose.from_xyz_yaw(120.0, -45.0, 0.0, 0.9)
    expected_rig = np.array([18.0, 2.5, 0.0])
    local = anchor.transform_points(expected_rig.reshape(1, 3))[0]

    forecast = forecast_actors(
        renderer(actor(float(local[0]), float(local[1]))),
        anchor,
        table(),
        station0_m=0.0,
        config=CONFIG,
    )

    # The two discs straddle the actor centre, so their mean is the centre.
    centre_now = forecast.centres[0].mean(axis=0)
    assert centre_now == pytest.approx(expected_rig[:2], abs=1e-3)


def test_an_identity_anchor_leaves_positions_untouched():
    forecast = forecast_actors(renderer(actor(30.0, 1.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert forecast.centres[0].mean(axis=0) == pytest.approx([30.0, 1.0], abs=1e-3)


def test_actor_velocity_is_rotated_but_not_translated():
    """A velocity is a free vector: the anchor's rotation applies, not its offset.

    Translating it would make every distant actor appear to be hurtling
    towards us.
    """
    anchor = Pose.from_xyz_yaw(500.0, 300.0, 0.0, math.pi / 2.0)
    # Moving along +x in local; the rig is rotated 90 degrees, so in rig terms
    # that is -y.
    local = anchor.transform_points(np.array([[25.0, 0.0, 0.0]]))[0]
    forecast = forecast_actors(
        renderer(actor(float(local[0]), float(local[1]), vx=10.0)),
        anchor,
        table(),
        0.0,
        CONFIG,
    )
    displacement = forecast.centres[9].mean(axis=0) - forecast.centres[0].mean(axis=0)
    # Nine steps of 0.1 s at 10 m/s is 9 m, along rig -y.
    assert displacement[0] == pytest.approx(0.0, abs=1e-2)
    assert displacement[1] == pytest.approx(-9.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_a_stationary_actor_stays_put():
    forecast = forecast_actors(renderer(actor(25.0, 0.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert np.allclose(forecast.centres[0], forecast.centres[-1], atol=1e-5)


def test_a_moving_actor_is_extrapolated_at_constant_velocity():
    forecast = forecast_actors(
        renderer(actor(25.0, 0.0, vx=6.0)), Pose.identity(), table(), 0.0, CONFIG
    )
    dt = CONFIG.horizon.step_s
    for index in (0, 9, 39):
        expected_x = 25.0 + 6.0 * (index + 1) * dt
        assert forecast.centres[index].mean(axis=0)[0] == pytest.approx(expected_x, abs=1e-3)


def test_an_actor_becomes_two_discs_spanning_its_length():
    forecast = forecast_actors(
        renderer(actor(20.0, 0.0, length=4.8, width=2.0)), Pose.identity(), table(), 0.0, CONFIG
    )
    assert forecast.n_discs == 2
    assert forecast.n_actors == 1
    separation = np.linalg.norm(forecast.centres[0, 0] - forecast.centres[0, 1])
    assert separation == pytest.approx(4.8 / 2.0, abs=1e-3)
    # Radius covers a quarter-length by a half-width.
    assert forecast.radii[0] == pytest.approx(math.hypot(1.2, 1.0), abs=1e-3)


def test_discs_follow_the_actors_heading():
    """A crossing vehicle's discs must span across our path, not along it."""
    forecast = forecast_actors(
        renderer(actor(20.0, 0.0, yaw=math.pi / 2.0, length=4.8)),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    span = forecast.centres[0, 0] - forecast.centres[0, 1]
    assert abs(span[0]) < 1e-3
    assert abs(span[1]) == pytest.approx(2.4, abs=1e-3)


# ---------------------------------------------------------------------------
# Lead vehicle
# ---------------------------------------------------------------------------


def test_an_in_lane_vehicle_ahead_becomes_the_lead():
    forecast = forecast_actors(renderer(actor(30.0, 0.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert np.isfinite(forecast.lead_station_m).all()
    # Reported at the vehicle's rear, so short of its centre.
    assert forecast.lead_station_m[0] < 30.0
    assert forecast.lead_station_m[0] > 24.0


def test_a_vehicle_in_the_next_lane_is_not_the_lead():
    """Still an obstacle for `collision_free`, but not something to follow."""
    forecast = forecast_actors(renderer(actor(30.0, 3.6)), Pose.identity(), table(), 0.0, CONFIG)
    assert not np.isfinite(forecast.lead_station_m).any()
    assert forecast.n_discs == 2  # it still counts for clearance


def test_a_vehicle_behind_us_is_not_the_lead():
    forecast = forecast_actors(renderer(actor(-15.0, 0.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert not np.isfinite(forecast.lead_station_m).any()


def test_the_nearest_of_several_in_lane_vehicles_wins():
    forecast = forecast_actors(
        renderer(
            actor(60.0, 0.0, track_id="far"),
            actor(25.0, 0.0, track_id="near"),
            actor(40.0, 0.0, track_id="middle"),
        ),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    assert forecast.lead_station_m[0] < 25.0
    assert forecast.lead_station_m[0] > 20.0


def test_the_lead_moves_away_as_it_drives_off():
    forecast = forecast_actors(
        renderer(actor(25.0, 0.0, vx=8.0)), Pose.identity(), table(), 0.0, CONFIG
    )
    assert forecast.lead_station_m[-1] > forecast.lead_station_m[0] + 2.0


# ---------------------------------------------------------------------------
# Traffic light
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [TrafficLightState.TRAFFIC_LIGHT_STATE_RED, TrafficLightState.TRAFFIC_LIGHT_STATE_YELLOW],
)
def test_a_stopping_light_yields_a_stop_station(state):
    light = read_traffic_light(
        renderer(light=state, light_distance_m=22.0), station0_m=5.0, config=CONFIG
    )
    assert light.active
    assert light.stop_station_m == pytest.approx(5.0 + 22.0 - CONFIG.limits.stop_line_margin_m)


@pytest.mark.parametrize(
    "state",
    [
        TrafficLightState.TRAFFIC_LIGHT_STATE_GREEN,
        TrafficLightState.TRAFFIC_LIGHT_STATE_NONE,
        TrafficLightState.TRAFFIC_LIGHT_STATE_OFF,
        TrafficLightState.TRAFFIC_LIGHT_STATE_UNKNOWN,
    ],
)
def test_a_non_stopping_light_imposes_nothing(state):
    light = read_traffic_light(
        renderer(light=state, light_distance_m=22.0), station0_m=0.0, config=CONFIG
    )
    assert not light.active
    assert math.isinf(light.stop_station_m)


def test_a_negative_distance_means_no_light_applies():
    """The proto's own convention, which a state check alone would miss."""
    light = read_traffic_light(
        renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=-1.0),
        station0_m=0.0,
        config=CONFIG,
    )
    assert not light.active


# ---------------------------------------------------------------------------
# Absent payload
# ---------------------------------------------------------------------------


def test_no_renderer_data_is_handled_rather_than_assumed_away():
    """`renderer_data` is None under an upstream alpasim runtime."""
    forecast = forecast_actors(None, Pose.identity(), table(), 0.0, CONFIG)
    assert forecast.n_discs == 0
    assert forecast.n_actors == 0
    assert forecast.centres.shape == (CONFIG.horizon.n_steps, 0, 2)
    assert not np.isfinite(forecast.lead_station_m).any()

    light = read_traffic_light(None, 0.0, CONFIG)
    assert not light.active
    assert math.isinf(light.stop_station_m)


def test_an_empty_actor_list_is_handled():
    forecast = forecast_actors(renderer(), Pose.identity(), table(), 0.0, CONFIG)
    assert forecast.n_discs == 0


def test_a_zero_sized_actor_does_not_produce_a_degenerate_disc():
    """CARLA occasionally reports a zero extent; a zero radius would vanish."""
    forecast = forecast_actors(
        renderer(actor(20.0, 0.0, length=0.0, width=0.0)),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    assert (forecast.radii > 0.0).all()


# ---------------------------------------------------------------------------
# The deadlock the CARLA run exposed
# ---------------------------------------------------------------------------


def test_a_stop_line_inside_the_margin_is_still_satisfiable():
    """The regression test for a policy that stopped and never restarted.

    Within `stop_line_margin_m` of the line, subtracting the margin puts the
    required stop station *behind* the vehicle. No plan can travel backwards,
    so `stop_on_red` is violated by every candidate, the emergency fallback
    engages, and -- because the vehicle then sits inside the light's trigger
    volume -- it is re-reported every step thereafter. The vehicle never moves
    again, and only because it stopped correctly in the first place.
    """
    station0 = 12.0
    light = read_traffic_light(
        renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=1.5),
        station0_m=station0,
        config=CONFIG,
    )
    assert light.active
    assert light.stop_station_m >= station0


def test_a_stop_line_beyond_the_margin_keeps_its_margin():
    light = read_traffic_light(
        renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=30.0),
        station0_m=5.0,
        config=CONFIG,
    )
    assert light.stop_station_m == pytest.approx(5.0 + 30.0 - CONFIG.limits.stop_line_margin_m)


def test_a_line_already_crossed_stops_governing_us():
    """With the runtime reporting a signed distance, past the line reads negative.

    Freezing inside an intersection is worse than clearing it, so the rule has
    to stop applying rather than become unsatisfiable.
    """
    light = read_traffic_light(
        renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=-2.0),
        station0_m=40.0,
        config=CONFIG,
    )
    assert not light.active
    assert math.isinf(light.stop_station_m)
