# SPDX-License-Identifier: Apache-2.0
"""Reading the world.

The frame transform is the risky part, so it is checked against the *runtime's
own* composition rule (``anchor @ rig_pose == local_pose``) rather than
against a second copy of my arithmetic.  If both were mine, they would agree
about a mistake.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from carla_driver_interface.geometry import Pose
from carla_driver_interface.protocol import (
    AABB,
    CarlaActorState,
    CarlaRendererData,
    DynamicState,
    TrafficLightState,
    Vec3,
)
from carla_driver_interface.protocol import Pose as PoseProto
from carla_driver_interface.protocol import Quat as QuatProto

from stl_driver.config import STLDriverConfig
from stl_driver.frenet import build_route_table, project_to_frenet
from stl_driver.perception import SAME_STOP_LINE_M, forecast_actors, read_traffic_light
from tests.conftest import curved_route, straight_route

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
    # Measured against the straight-line model, so this isolates the frame
    # transform from whatever the prediction does with the road geometry.
    straight = replace(CONFIG, limits=replace(CONFIG.limits, frenet_prediction=False))
    anchor = Pose.from_xyz_yaw(500.0, 300.0, 0.0, math.pi / 2.0)
    # Moving along +x in local; the rig is rotated 90 degrees, so in rig terms
    # that is -y.
    local = anchor.transform_points(np.array([[25.0, 0.0, 0.0]]))[0]
    forecast = forecast_actors(
        renderer(actor(float(local[0]), float(local[1]), vx=10.0)),
        anchor,
        table(),
        0.0,
        straight,
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


def test_an_actor_is_covered_by_discs_that_span_its_length():
    length, width = 4.8, 2.0
    forecast = forecast_actors(
        renderer(actor(20.0, 0.0, length=length, width=width)),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    assert forecast.n_actors == 1
    assert forecast.n_discs >= 2

    # The discs span the body: outermost centres one half-spacing inside each end.
    xs = forecast.centres[0, :, 0]
    spacing = length / forecast.n_discs
    assert xs.min() == pytest.approx(20.0 - length / 2 + spacing / 2, abs=1e-3)
    assert xs.max() == pytest.approx(20.0 + length / 2 - spacing / 2, abs=1e-3)
    # And reach the corners.
    assert forecast.radii[0] == pytest.approx(math.hypot(spacing / 2, width / 2), abs=1e-3)


def test_the_footprint_is_not_inflated_into_the_next_lane():
    """The bug that made the policy emergency-brake on an empty road.

    Two discs per car turn a 2.0 m body into a 3.0 m one, which leaves almost
    nothing between the ego and a vehicle one lane over -- so an ordinary lane
    offset read as an imminent collision.
    """
    from stl_driver.perception import footprint_discs

    tolerance = CONFIG.limits.footprint_max_inflation_m
    for length, width in [(4.5, 2.0), (2.2, 0.9), (12.0, 2.6), (0.4, 0.4)]:
        _, radius = footprint_discs(length, width, tolerance)
        assert radius >= width / 2  # still covers the body
        assert radius - max(width, 0.5) / 2 <= tolerance + 1e-6

    offsets, radius = footprint_discs(4.5, 2.0, tolerance)
    # Every corner of the box is inside some disc.
    for corner in [(2.25, 1.0), (2.25, -1.0), (-2.25, 1.0), (-2.25, -1.0)]:
        assert min(math.hypot(corner[0] - o, corner[1]) for o in offsets) <= radius + 1e-6


def test_discs_follow_the_actors_heading():
    """A crossing vehicle's discs must span across our path, not along it."""
    length = 4.8
    forecast = forecast_actors(
        renderer(actor(20.0, 0.0, yaw=math.pi / 2.0, length=length)),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    xs, ys = forecast.centres[0, :, 0], forecast.centres[0, :, 1]
    # Laid out along +y (the actor's heading), so x is unchanged.
    assert np.ptp(xs) < 1e-3
    spacing = length / forecast.n_discs
    assert np.ptp(ys) == pytest.approx(length - spacing, abs=1e-3)


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
    assert forecast.n_discs >= 2  # it still counts for clearance


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


def headway_slack_now(forecast, speed0_mps: float = 0.0, lead_speed_mps: float = 0.0) -> float:
    """`headway_slack` at t = 0 for the lead this forecast found."""
    import torch

    from stl_driver.rollout import N_PARAMS, rollout
    from stl_driver.signals import Col
    from tests.conftest import make_inputs

    inputs = make_inputs(
        CONFIG,
        speed0_mps=speed0_mps,
        lead_station_m=float(forecast.lead_station_m[0]),
        lead_speed_mps=lead_speed_mps,
    )
    signal = rollout(torch.zeros(N_PARAMS, dtype=torch.float32), inputs, CONFIG)
    return float(signal[0, Col.HEADWAY_SLACK])


def ego_front_m() -> float:
    """Rear axle to front bumper, written out rather than asked of the config."""
    limits = CONFIG.limits
    return limits.ego_rear_axle_to_centre_m + limits.ego_length_m / 2.0


def test_a_stopped_car_half_a_metre_off_the_bumper_violates_safe_headway():
    """Issue #1140: the gap was measured from the rear axle, not the bumper.

    The lead's station is its rear edge, the ego's is its rear axle, and the
    RSS requirement is bumper to bumper. With nothing taking off the 3.75 m
    from axle to bumper, a stopped car 0.5 m off the bumper read as +1.2 m of
    slack against a 2.5 m standstill gap.
    """
    limits = CONFIG.limits
    length = 4.5
    for bumper_gap in (0.5, 1.0, 2.0):
        centre = ego_front_m() + bumper_gap + length / 2.0
        forecast = forecast_actors(
            renderer(actor(centre, 0.0, length=length)), Pose.identity(), table(), 0.0, CONFIG
        )
        slack = headway_slack_now(forecast)
        # The lead's rear is its box's (#1151), so the gap is exact.
        assert slack == pytest.approx(bumper_gap - limits.standstill_gap_m, abs=1e-3)
        assert slack < 0.0, f"a stopped car {bumper_gap} m off the bumper must violate"


def test_the_following_distance_is_kept_from_the_front_bumper():
    """At speed the required RSS distance is bumper to bumper, too."""
    limits = CONFIG.limits
    length = 4.5
    speed = 8.0
    rho = limits.rss_response_s
    required = (
        speed * rho
        + 0.5 * limits.rss_response_accel_mps2 * rho * rho
        + (speed + rho * limits.rss_response_accel_mps2) ** 2 / (2 * limits.rss_brake_mps2)
        - speed**2 / (2 * limits.rss_lead_brake_mps2)
    )
    # Exactly the required distance between bumpers: on the edge of the rule.
    centre = ego_front_m() + required + length / 2.0
    forecast = forecast_actors(
        renderer(actor(centre, 0.0, vx=speed, length=length)),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    slack = headway_slack_now(forecast, speed0_mps=speed, lead_speed_mps=speed)
    # The first sample is one step in: the ego has covered `speed * step_s`,
    # and `lead_station_m[0]`, the lead's forecast one step in, has moved on
    # by as much. With the lead's rear taken from its box (#1151) the gap is
    # then exactly the required one. Measured from the axle instead, this
    # read about +2.9 m; from the disc cover's rear, 0.6 m short.
    assert slack == pytest.approx(0.0, abs=1e-3)


def test_the_lead_is_reported_at_its_rear_bumper():
    """Issue #1151: the rear is the box's, not the disc cover's.

    A 4.5 x 2.0 m car centred 20 m ahead ends at 17.75 m. Taken as the rear
    disc's centre less its radius, it read 17.165 m -- the radius is mostly
    half the car's *width* -- and every following gap 0.6 m short.
    """
    forecast = forecast_actors(
        renderer(actor(20.0, 0.0, length=4.5, width=2.0)), Pose.identity(), table(), 0.0, CONFIG
    )
    assert forecast.lead_station_m[0] == pytest.approx(20.0 - 4.5 / 2.0, abs=1e-3)


def test_a_long_vehicle_elsewhere_does_not_move_the_lead():
    """The disc count is shared, so a bus anywhere changed every lead's rear.

    It read 16.95 m with a 12 m bus 30 m off to the side, against 17.165 m
    without one.
    """
    car = actor(20.0, 0.0, length=4.5, width=2.0, track_id="car")
    bus = actor(40.0, 30.0, length=12.0, width=2.5, track_id="bus")
    alone = forecast_actors(renderer(car), Pose.identity(), table(), 0.0, CONFIG)
    with_bus = forecast_actors(renderer(car, bus), Pose.identity(), table(), 0.0, CONFIG)
    assert with_bus.lead_station_m == pytest.approx(alone.lead_station_m, abs=1e-6)
    assert with_bus.lead_station_m[0] == pytest.approx(17.75, abs=1e-3)


def test_a_lead_at_an_angle_reaches_back_by_its_corner():
    """Turned to the road, the nearest part of the box is a rear corner."""
    yaw = 0.3
    forecast = forecast_actors(
        renderer(actor(20.0, 0.0, yaw=yaw, length=4.5, width=2.0)),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    corner = 20.0 - 2.25 * math.cos(yaw) - 1.0 * math.sin(yaw)
    assert forecast.lead_station_m[0] == pytest.approx(corner, abs=1e-3)


def test_a_stopped_queue_closes_to_the_standstill_gap():
    """At rest, `safe_headway` holds exactly down to `standstill_gap_m`.

    Behind a stopped car the rule's boundary is where the ego queues. With the
    disc cover's rear it sat 0.6 m further back than the gap the rule states
    (0.8 m with a long vehicle in the scene); now it is the stated gap.
    """
    limits = CONFIG.limits
    length = 4.5
    centre = ego_front_m() + limits.standstill_gap_m + length / 2.0
    bus = actor(40.0, 30.0, length=12.0, width=2.5, track_id="bus")
    for scene in ((), (bus,)):
        forecast = forecast_actors(
            renderer(actor(centre, 0.0, length=length), *scene),
            Pose.identity(),
            table(),
            0.0,
            CONFIG,
        )
        assert headway_slack_now(forecast) == pytest.approx(0.0, abs=1e-3)
    # And a decimetre closer is a violation, not slack.
    closer = forecast_actors(
        renderer(actor(centre - 0.1, 0.0, length=length)), Pose.identity(), table(), 0.0, CONFIG
    )
    assert headway_slack_now(closer) == pytest.approx(-0.1, abs=1e-3)


def test_a_vehicle_cutting_in_beside_our_front_half_is_the_lead():
    """What counts as ahead starts at the middle of the ego, not at its bumper.

    A car moving into our lane level with our front half -- its rear still
    beside us -- is one to let in. Followed, its gap is negative and
    `safe_headway` asks the policy to drop back; ignored until it cleared our
    nose, the policy would follow whatever was beyond it and close up on the
    merger instead.
    """
    limits = CONFIG.limits
    centre = limits.ego_rear_axle_to_centre_m + 1.1
    forecast = forecast_actors(
        renderer(actor(centre, 1.2, track_id="merger"), actor(40.0, 0.0, track_id="far")),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    assert forecast.lead_station_m[0] < ego_front_m()
    assert headway_slack_now(forecast) < -limits.standstill_gap_m


def test_a_vehicle_reaching_into_our_front_half_is_followed():
    """Its centre is behind ours, but its nose is past the middle of the ego."""
    # A 4.5 m car centred on the rear axle: its front bumper is 2.25 m ahead
    # of the axle, past the ego's middle (1.4 m).
    forecast = forecast_actors(renderer(actor(0.0, 0.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert np.isfinite(forecast.lead_station_m).all()


def test_a_vehicle_level_with_our_back_half_is_not_the_lead():
    """Beside our rear, it is the merger's turn to yield, not ours.

    Its front bumper is 1.25 m ahead of the rear axle, short of the ego's
    middle (1.4 m), but past the old mark of axle + 0.5 m, which made it the
    lead and had the policy brake for a car that was behind its own middle.
    """
    forecast = forecast_actors(renderer(actor(-1.0, 1.2)), Pose.identity(), table(), 0.0, CONFIG)
    assert not np.isfinite(forecast.lead_station_m).any()


# ---------------------------------------------------------------------------
# Traffic light
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [TrafficLightState.TRAFFIC_LIGHT_STATE_RED, TrafficLightState.TRAFFIC_LIGHT_STATE_YELLOW],
)
def test_a_stopping_light_yields_a_stop_station(state):
    light = read_traffic_light(
        renderer(light=state, light_distance_m=22.0), station0_m=5.0, config=CONFIG, speed_mps=0.0
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
        renderer(light=state, light_distance_m=22.0), station0_m=0.0, config=CONFIG, speed_mps=0.0
    )
    assert not light.active
    assert math.isinf(light.stop_station_m)


def test_a_negative_distance_means_no_light_applies():
    """The proto's own convention, which a state check alone would miss."""
    light = read_traffic_light(
        renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=-1.0),
        station0_m=0.0,
        config=CONFIG,
        speed_mps=0.0,
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

    light = read_traffic_light(None, 0.0, CONFIG, speed_mps=0.0)
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
        speed_mps=0.0,
    )
    assert light.active
    assert light.stop_station_m >= station0


def test_a_stop_line_beyond_the_margin_keeps_its_margin():
    light = read_traffic_light(
        renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=30.0),
        station0_m=5.0,
        config=CONFIG,
        speed_mps=0.0,
    )
    assert light.stop_station_m == pytest.approx(5.0 + 30.0 - CONFIG.limits.stop_line_margin_m)


def test_a_margin_that_gave_way_is_held_for_the_same_line():
    """Read afresh each step, a margin that had to shrink grew back (#1147).

    Once braking, the room to stop shrinks faster than the distance, so the
    stop station slid back to the very edge of what could still be reached,
    and the first plan short of the hardest allowed stop violated the hard
    rule. For the same line it is held; a different line, or a light that
    stopped governing us, starts afresh.
    """
    limits = CONFIG.limits
    red = TrafficLightState.TRAFFIC_LIGHT_STATE_RED
    speed = 6.5
    distance = limits.stop_room_m(speed) + 1.5  # 1.5 m of the margin left
    first = read_traffic_light(
        renderer(light=red, light_distance_m=distance),
        0.0,
        CONFIG,
        speed,
        anchor=Pose.identity(),
    )
    assert first.margin_m == pytest.approx(1.5)
    assert first.line_xy == pytest.approx([distance, 0.0])

    # A step later, braking hard and two metres on: the same line, 2 m nearer.
    moved = Pose.from_xyz_yaw(2.0, 0.0, 0.0, 0.0)
    lowest, _ = limits.plan_accel_envelope()

    def later(previous):
        return read_traffic_light(
            renderer(light=red, light_distance_m=distance - 2.0),
            2.0,
            CONFIG,
            5.0,
            lowest,
            anchor=moved,
            previous=previous,
        )

    afresh = later(None)
    assert afresh.margin_m > first.margin_m + 0.5  # what read afresh would have done
    held = later(first)
    assert held.margin_m == pytest.approx(first.margin_m)
    assert held.stop_station_m == pytest.approx(first.stop_station_m)

    # A line elsewhere is a different light.
    elsewhere = Pose.from_xyz_yaw(2.0, 2.0 * SAME_STOP_LINE_M, 0.0, 0.0)
    other = read_traffic_light(
        renderer(light=red, light_distance_m=distance - 2.0),
        2.0,
        CONFIG,
        5.0,
        lowest,
        anchor=elsewhere,
        previous=first,
    )
    assert other.margin_m == pytest.approx(afresh.margin_m)

    # And a light that went green in between starts afresh too.
    green = read_traffic_light(
        renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_GREEN, light_distance_m=distance),
        0.0,
        CONFIG,
        speed,
        anchor=Pose.identity(),
        previous=first,
    )
    assert not green.active
    assert later(green).margin_m == pytest.approx(afresh.margin_m)


def test_a_line_already_crossed_stops_governing_us():
    """With the runtime reporting a signed distance, past the line reads negative.

    Freezing inside an intersection is worse than clearing it, so the rule has
    to stop applying rather than become unsatisfiable.
    """
    light = read_traffic_light(
        renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=-2.0),
        station0_m=40.0,
        config=CONFIG,
        speed_mps=0.0,
    )
    assert not light.active
    assert math.isinf(light.stop_station_m)


def test_a_vehicle_queuing_behind_us_is_not_a_collision():
    """The dominant cause of spurious emergency stops in city traffic.

    Our plan only ever moves forward, so it cannot drive into something behind
    the rear bumper -- avoiding that is the following driver's problem. Left
    in, a car stopping close behind us at a red light violates
    `collision_free`, which is a hard rule, so the policy emergency-brakes
    while already stationary and correctly waiting.
    """
    behind = forecast_actors(renderer(actor(-4.0, 0.0)), Pose.identity(), table(), 0.0, CONFIG)
    # `_drop_traffic_behind` moves such discs far away rather than removing
    # them, since it works per timestep and the actor may yet come past us.
    # This one never does, so nothing it leaves behind is within reach and
    # `_drop_traffic_out_of_reach` then drops it outright.
    assert behind.centres.shape[1] == 0
    assert behind.radii.shape == (0,)

    ahead = forecast_actors(renderer(actor(8.0, 0.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert np.linalg.norm(ahead.centres[0], axis=-1).min() < 20.0


def test_a_vehicle_alongside_us_still_counts():
    """Only strictly behind is dropped; abreast is very much a collision risk."""
    forecast = forecast_actors(renderer(actor(0.5, 3.2)), Pose.identity(), table(), 0.0, CONFIG)
    assert np.linalg.norm(forecast.centres[0], axis=-1).min() < 20.0


def test_a_vehicle_beside_the_road_does_not_decide_the_clearance():
    """`collision_free` is a minimum, so anything present decides it.

    A vehicle two lanes over is the nearest thing about for most of a drive,
    and the objective pays for clearance -- which made it a standing reason to
    slow down for something that could not be hit.
    """
    beside = forecast_actors(renderer(actor(12.0, 9.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert beside.centres.shape[1] == 0

    # The corridor a plan may occupy still counts: this one is inside it.
    within = forecast_actors(renderer(actor(12.0, 1.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert within.centres.shape[1] > 0


def test_traffic_crossing_into_our_lane_is_kept_from_the_start():
    """The case a lane filter must not get wrong.

    Something crossing a junction is far to one side now and across our path
    two seconds from now. Judged on where it is, it would be dropped exactly
    when it matters most; judged on where it is predicted to be, it is kept.
    """
    crossing = actor(20.0, -14.0, yaw=math.pi / 2, vy=4.0)
    forecast = forecast_actors(renderer(crossing), Pose.identity(), table(), 0.0, CONFIG)
    assert forecast.centres.shape[1] > 0
    # It really does start outside the corridor -- otherwise the test proves
    # nothing about prediction.
    assert abs(forecast.centres[0, :, 1]).min() > CONFIG.limits.lane_half_width_m * 2.0


def test_a_vehicle_too_far_to_reach_is_not_forecast_at_all():
    """Most of what the simulator reports could never touch us.

    Carrying it is not free: the discs are padded to a fixed width and every
    candidate plan is scored against all of it.
    """
    far = CONFIG.limits.actor_reach_m + 30.0
    forecast = forecast_actors(renderer(actor(far, 0.0)), Pose.identity(), table(), 0.0, CONFIG)
    assert forecast.centres.shape[1] == 0
    # Still counted as seen -- it is dropped from the collision check, not
    # from the report of what is out there.
    assert forecast.n_actors == 1


def test_a_distant_vehicle_closing_on_us_is_kept():
    """The test is closest approach over the horizon, not distance now.

    The discs already carry the other vehicle's predicted motion, so the one
    case a range gate must not get wrong -- something far away and arriving --
    is the case it reads as near.
    """
    far = CONFIG.limits.actor_reach_m + 30.0
    closing = actor(far, 0.0, yaw=math.pi, vx=-20.0)
    forecast = forecast_actors(renderer(closing), Pose.identity(), table(), 0.0, CONFIG)
    assert forecast.centres.shape[1] > 0
    assert np.linalg.norm(forecast.centres, axis=-1).min() < CONFIG.limits.actor_reach_m


def test_a_vehicle_overtaking_from_behind_reappears_once_it_is_ahead():
    """The test is per timestep, so a faster follower is not ignored forever."""
    forecast = forecast_actors(
        renderer(actor(-6.0, 0.0, vx=12.0)), Pose.identity(), table(), 0.0, CONFIG
    )
    assert np.linalg.norm(forecast.centres[0], axis=-1).min() > 1000.0  # behind now
    assert np.linalg.norm(forecast.centres[-1], axis=-1).min() < 100.0  # ahead later


def test_the_stop_margin_never_demands_more_room_than_there_is():
    """CARLA reports a light only from inside its trigger volume.

    On Town10HD that first sighting can be four metres out, and a light can
    turn red just as close. A fixed four-metre margin would ask a moving
    vehicle to stop where it stands, which no plan can do, so every candidate
    violates the rule and the fallback engages on a perfectly stoppable
    approach. Wherever the line itself can still be stopped short of, the stop
    station leaves at least the room `stop_on_red` will ask for -- and where
    it cannot, the line, not a point past it.
    """
    limits = CONFIG.limits
    for speed in (0.0, 2.0, 4.0, 6.0, 8.0):
        for accel in (-3.0, 0.0, 1.5):
            room = limits.stop_room_m(speed, accel)
            for distance in (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0):
                light = read_traffic_light(
                    renderer(
                        light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=distance
                    ),
                    station0_m=0.0,
                    config=CONFIG,
                    speed_mps=speed,
                    accel_mps2=accel,
                )
                allowed = light.stop_station_m
                case = (speed, accel, distance, allowed, room)
                assert allowed <= distance, case  # never past the line
                assert allowed >= distance - limits.stop_line_margin_m - 1e-9, case
                if distance >= room:
                    assert allowed >= room - 1e-9, case
                else:
                    assert allowed == pytest.approx(distance), case  # the line itself

    # Far out, the full margin applies at any of these speeds.
    for speed in (0.0, 8.0):
        far = read_traffic_light(
            renderer(light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=40.0),
            station0_m=0.0,
            config=CONFIG,
            speed_mps=speed,
        )
        assert far.stop_station_m == pytest.approx(40.0 - limits.stop_line_margin_m)


def test_the_room_to_stop_is_what_the_rule_charges_at_its_peak():
    """`stop_room_m` against `stopping_distance_m`, which it bounds from above.

    Equal at rest and when already braking at the plan's bound -- nothing of
    the ramp is left to charge twice -- and larger from a cruise, by more the
    faster the cruise.
    """
    limits = CONFIG.limits
    lowest, _ = limits.plan_accel_envelope()
    assert limits.stop_room_m(0.0) == 0.0
    previous_excess = 0.0
    for speed in (2.0, 4.0, 6.0, 8.0):
        stopping = limits.stopping_distance_m(speed)
        assert limits.stop_room_m(speed, lowest) == pytest.approx(stopping)
        excess = limits.stop_room_m(speed) - stopping
        assert excess > previous_excess
        previous_excess = excess
        assert limits.stop_room_m(speed, 1.0) > limits.stop_room_m(speed)


@pytest.mark.parametrize(
    "state",
    [TrafficLightState.TRAFFIC_LIGHT_STATE_RED, TrafficLightState.TRAFFIC_LIGHT_STATE_YELLOW],
)
def test_the_stop_station_does_not_creep_forward_with_the_vehicle(state):
    """The regression test for #1147.

    The margin used to be ``min(margin, 0.5 * distance)`` of the *live*
    distance, so within two margins of the line the stop station sat halfway
    to it and moved half a metre on for every metre crept: closed loop, a red
    light 30 m ahead stopped the rear axle at 28.1 m instead of 26 m.

    Read from rest at every point on the way in, the stop station has to be
    the same place in the world -- the line less the full margin -- until the
    vehicle is inside the margin, where it is held where it stands.
    """
    line = 30.0
    target = line - CONFIG.limits.stop_line_margin_m
    for station0 in np.arange(18.0, 29.75, 0.25):
        light = read_traffic_light(
            renderer(light=state, light_distance_m=line - station0),
            station0_m=float(station0),
            config=CONFIG,
            speed_mps=0.0,
        )
        assert light.active
        assert light.stop_station_m == pytest.approx(max(target, station0)), station0


def test_a_braking_approach_keeps_the_full_margin():
    """Moving, the margin gives way only to what can no longer be stopped short of.

    A vehicle with just the room to stop at ``line - margin`` -- on its
    braking curve, or still cruising towards it -- is asked to stop exactly
    there, at every speed down to rest.
    """
    limits = CONFIG.limits
    lowest, _ = limits.plan_accel_envelope()
    line = 30.0
    for accel in (lowest, 0.0):
        for speed in np.linspace(8.0, 0.0, 17):
            room = limits.stop_room_m(float(speed), accel)
            distance = limits.stop_line_margin_m + room + 0.01
            light = read_traffic_light(
                renderer(
                    light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED, light_distance_m=distance
                ),
                station0_m=line - distance,
                config=CONFIG,
                speed_mps=float(speed),
                accel_mps2=accel,
            )
            expected = line - limits.stop_line_margin_m
            assert light.stop_station_m == pytest.approx(expected), (accel, speed)


# ---------------------------------------------------------------------------
# Predicting along the road
# ---------------------------------------------------------------------------


def curved_table():
    return build_route_table(
        curved_route(radius_m=40.0),
        ds=CONFIG.route_table_ds_m,
        length_m=CONFIG.route_table_length_m(),
    )


def test_a_vehicle_ahead_on_a_bend_stays_on_the_bend():
    """The error a straight line makes, and the reason to resolve into the road.

    Extrapolating tangentially sends the car we are following out of our lane
    within a second or two, so it stops being our lead exactly when we most
    need it to be.
    """
    table_c = curved_table()
    start = table_c.frenet_to_xy(20.0, 0.0)
    heading = math.atan2(*(table_c.tangent[20][::-1]))
    speed = 8.0

    along_road = forecast_actors(
        renderer(
            actor(
                float(start[0]),
                float(start[1]),
                yaw=heading,
                vx=speed * math.cos(heading),
                vy=speed * math.sin(heading),
            )
        ),
        Pose.identity(),
        table_c,
        0.0,
        CONFIG,
    )
    straight = replace(CONFIG, limits=replace(CONFIG.limits, frenet_prediction=False))
    tangentially = forecast_actors(
        renderer(
            actor(
                float(start[0]),
                float(start[1]),
                yaw=heading,
                vx=speed * math.cos(heading),
                vy=speed * math.sin(heading),
            )
        ),
        Pose.identity(),
        table_c,
        0.0,
        straight,
    )

    def lane_offset(forecast, index):
        _, offset = project_to_frenet(forecast_table := curved_table(), forecast.centres[index])
        del forecast_table
        return float(np.abs(offset).max())

    # Four seconds later the straight line has left the lane; the road-aware
    # prediction is still in it.
    assert lane_offset(along_road, -1) < 1.0
    assert lane_offset(tangentially, -1) > 3.0


def test_a_vehicle_behind_the_route_start_keeps_the_straight_line_model():
    """Frenet cannot express "behind" when the route begins at the ego.

    Projection clamps, so a follower would come back at station zero -- on our
    front bumper. Mixing the two models per actor is what avoids that.
    """
    forecast = forecast_actors(
        renderer(actor(-8.0, 0.0, vx=2.0)), Pose.identity(), table(), 0.0, CONFIG
    )
    # Still behind us, hence dropped from the collision check entirely.
    assert np.abs(forecast.centres[0]).min() > 1000.0


def test_the_lateral_rate_decays_rather_than_integrating_forever():
    """A vehicle's sideways drift relative to our road is not persistent."""
    drifting = forecast_actors(
        renderer(actor(30.0, 1.0, vx=6.0, vy=1.5)), Pose.identity(), table(), 0.0, CONFIG
    )
    offsets = project_to_frenet(table(), drifting.centres.reshape(-1, 2))[1]
    offsets = offsets.reshape(drifting.centres.shape[:2])
    early = float(np.abs(offsets[4]).max())
    late = float(np.abs(offsets[-1]).max())
    # It moves across early on, then settles rather than running away.
    assert early > 1.0
    assert late < 1.0 + 1.5 * CONFIG.limits.prediction_lateral_decay_s + 0.5


def test_a_vehicle_overtaking_from_behind_never_becomes_the_lead():
    """The bug that produced most of `safe_headway`'s violations.

    The forecast cannot know where the plan will put the ego, so the only
    reference it has for "ahead" is the ego's station at t = 0. Deciding the
    lead afresh at each step of the horizon against that fixed mark means a
    faster vehicle passing us crosses it and is adopted as the vehicle we are
    following -- at a gap measured from a position the ego left seconds ago,
    and so negative.

    On the road the lead station read a steady 74.6 m for most of the horizon,
    dipped to 0.4 m for three or four steps as someone passed, and returned.
    `safe_headway` reported -16 m against a clear road and the policy braked
    from 7 m/s to a standstill.
    """
    overtaking = actor(-12.0, 0.0, vx=14.0)  # behind us, and much quicker
    forecast = forecast_actors(renderer(overtaking), Pose.identity(), table(), 0.0, CONFIG)
    assert not np.isfinite(forecast.lead_station_m).any(), (
        "a vehicle that starts behind us must never be adopted as the lead, "
        "however far forward it travels within the horizon"
    )


def test_the_lead_stays_the_same_vehicle_across_the_horizon():
    """Following is an identity, not a per-step argmin.

    With a slow vehicle far ahead and a quick one passing, the gap to the one
    we follow must decrease smoothly -- never jump to the passer and back.
    """
    forecast = forecast_actors(
        renderer(actor(60.0, 0.0, vx=2.0), actor(-10.0, 0.0, vx=16.0)),
        Pose.identity(),
        table(),
        0.0,
        CONFIG,
    )
    lead = forecast.lead_station_m
    assert np.isfinite(lead).all()
    # Monotone: the vehicle we follow is moving forward, so its station only
    # grows. A switch to the overtaking car would show up as a step down.
    assert np.all(np.diff(lead) >= -1e-6)
    assert lead.min() > 50.0


def test_padding_discs_leaves_the_clearance_unchanged():
    """A fixed width must not change what `collision_free` sees.

    The padding exists so there is one graph shape rather than four; it earns
    that only if it is invisible to the rules.
    """
    import torch

    from stl_driver.perception import pad_actor_discs
    from stl_driver.rollout import N_PARAMS, rollout
    from stl_driver.signals import Col
    from tests.conftest import make_inputs, stationary_actor

    centres, radii = stationary_actor(CONFIG, ahead_m=18.0, lateral_m=0.4)
    unpadded = make_inputs(CONFIG, speed0_mps=7.0, actor_centres=centres, actor_radii=radii)

    padded_centres, padded_radii = pad_actor_discs(
        centres.numpy(), radii.numpy(), CONFIG.limits.max_actor_discs
    )
    assert padded_radii.shape == (CONFIG.limits.max_actor_discs,)
    padded = make_inputs(
        CONFIG,
        speed0_mps=7.0,
        actor_centres=torch.from_numpy(padded_centres),
        actor_radii=torch.from_numpy(padded_radii),
    )

    params = torch.tensor([0.3, -0.5, 0.2, 0.1, 0.4, -0.2])
    before = rollout(params, unpadded, CONFIG)[:, Col.CLEARANCE]
    after = rollout(params, padded, CONFIG)[:, Col.CLEARANCE]
    assert torch.allclose(before, after, atol=1e-4)
    assert params.shape == (N_PARAMS,)


def test_padding_an_empty_scene_reads_as_an_empty_scene():
    """All-padding must give the same "nothing there" as no discs at all."""
    import torch

    from stl_driver.perception import pad_actor_discs
    from stl_driver.rollout import rollout
    from stl_driver.signals import Col
    from tests.conftest import make_inputs

    steps = CONFIG.horizon.n_steps
    empty = make_inputs(CONFIG, speed0_mps=7.0)
    centres, radii = pad_actor_discs(
        np.zeros((steps, 0, 2), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        CONFIG.limits.max_actor_discs,
    )
    all_padding = make_inputs(
        CONFIG,
        speed0_mps=7.0,
        actor_centres=torch.from_numpy(centres),
        actor_radii=torch.from_numpy(radii),
    )
    params = torch.zeros(6)
    assert torch.allclose(
        rollout(params, empty, CONFIG)[:, Col.CLEARANCE],
        rollout(params, all_padding, CONFIG)[:, Col.CLEARANCE],
        atol=1e-4,
    )


def test_too_many_discs_drop_the_farthest():
    """Only the nearest can decide a minimum, so those are the ones to keep."""
    from stl_driver.perception import pad_actor_discs

    steps = CONFIG.horizon.n_steps
    distances = np.array([50.0, 5.0, 200.0, 12.0], dtype=np.float32)
    centres = np.zeros((steps, 4, 2), dtype=np.float32)
    centres[:, :, 0] = distances
    radii = np.arange(4, dtype=np.float32)

    kept_centres, kept_radii = pad_actor_discs(centres, radii, 2)
    assert kept_centres.shape == (steps, 2, 2)
    assert sorted(kept_centres[0, :, 0].tolist()) == [5.0, 12.0]
    # The radii must follow their own discs, not be re-paired with others.
    assert sorted(kept_radii.tolist()) == [1.0, 3.0]
