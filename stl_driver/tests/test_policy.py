# SPDX-License-Identifier: Apache-2.0
"""The policy's contract with the driver service.

Mostly about the shape of what it returns.  The service silently discards
plan waypoints at or before ``time_now_us`` and the controller coasts on a
plan shorter than two points, so a policy can be doing excellent planning and
still drive badly by getting the envelope wrong -- and nothing else in the
suite would notice.
"""

from __future__ import annotations

import numpy as np
import pytest
from carla_driver_interface.driver.base import DriveContext, SessionState
from carla_driver_interface.geometry import Pose, Trajectory
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
from stl_driver.policy import EgoPolicy
from tests.conftest import curved_route, straight_route

TIME_NOW_US = 1_700_000_000_000_000
CONFIG = STLDriverConfig()


@pytest.fixture(scope="module")
def policy() -> EgoPolicy:
    driver = EgoPolicy()
    driver.on_session_start(SessionState(uuid="s", seed=0, scene_id="t", cameras={}))
    return driver


def context(
    *,
    route: np.ndarray | None = None,
    speed_mps: float = 6.0,
    accel_mps2: float = 0.0,
    renderer_data: CarlaRendererData | None = None,
    time_now_us: int = TIME_NOW_US,
) -> DriveContext:
    session = SessionState(uuid="s", seed=0, scene_id="t", cameras={})
    if route is not None:
        session.route_waypoints_in_rig = np.asarray(route, dtype=np.float64)
    session.ego_trajectory = Trajectory([time_now_us], [Pose.identity()])
    session.dynamic_states = [
        DynamicState(
            linear_velocity=Vec3(x=speed_mps, y=0.0, z=0.0),
            linear_acceleration=Vec3(x=accel_mps2, y=0.0, z=0.0),
        )
    ]
    return DriveContext(
        session=session,
        time_now_us=time_now_us,
        time_query_us=time_now_us + 100_000,
        renderer_data=renderer_data,
    )


# ---------------------------------------------------------------------------
# The plan envelope the service requires
# ---------------------------------------------------------------------------


def test_the_plan_has_enough_points_for_the_controller(policy):
    """Below two points the runtime's tracker coasts with the brake on."""
    result = policy.drive(context(route=straight_route()))
    assert len(result.trajectory_in_rig) >= 2


def test_every_plan_timestamp_is_strictly_after_now(policy):
    """The service drops anything at or before `time_now_us`, silently."""
    result = policy.drive(context(route=straight_route()))
    assert all(ts > TIME_NOW_US for ts in result.trajectory_in_rig.timestamps_us)


def test_plan_timestamps_strictly_increase(policy):
    """`Trajectory.append` rejects a non-increasing stamp outright."""
    stamps = policy.drive(context(route=straight_route())).trajectory_in_rig.timestamps_us
    assert all(later > earlier for earlier, later in zip(stamps, stamps[1:], strict=False))


def test_the_plan_spans_the_configured_horizon(policy):
    result = policy.drive(context(route=straight_route()))
    stamps = result.trajectory_in_rig.timestamps_us
    span_s = (stamps[-1] - TIME_NOW_US) / 1e6
    assert span_s == pytest.approx(policy.config.horizon.duration_s, abs=0.05)


def test_the_plan_is_long_enough_for_the_pure_pursuit_lookahead(policy):
    """The tracker looks 4-20 m ahead; a shorter plan clamps to its own end."""
    result = policy.drive(context(route=straight_route(), speed_mps=8.0))
    positions = result.trajectory_in_rig.positions
    length = float(np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)))
    assert length > 20.0


def test_the_plan_starts_near_the_vehicle_not_at_the_route_origin(policy):
    """The plan is in the rig frame, anchored on where we are."""
    result = policy.drive(context(route=straight_route(), speed_mps=6.0))
    first = result.trajectory_in_rig.positions[0]
    assert np.linalg.norm(first[:2]) < 2.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_every_rule_is_reported_as_a_debug_scalar(policy):
    """The channel that makes the behaviour explainable after the fact."""
    result = policy.drive(context(route=straight_route()))
    for name in policy.rules.names:
        assert f"stl_rule_{name}" in result.debug_scalars
    assert "stl_robustness" in result.debug_scalars


def test_alternatives_are_reported_for_inspection(policy):
    result = policy.drive(context(route=straight_route()))
    assert len(result.sampled_trajectories_in_rig) == (
        policy.config.optimizer.n_reported_candidates
    )
    assert all(len(t) >= 2 for t in result.sampled_trajectories_in_rig)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_no_route_yields_a_hold_plan_rather_than_an_error(policy):
    """The route has simply not arrived yet; this is not a failure."""
    result = policy.drive(context(route=None))
    assert result.debug_scalars["stl_no_route"] == 1.0
    assert len(result.trajectory_in_rig) >= 2


def test_a_degenerate_route_does_not_fail_the_rollout(policy):
    """An exception from `drive` fails the whole rollout upstream."""
    result = policy.drive(context(route=np.zeros((6, 3))))
    assert result.debug_scalars["stl_bad_route"] == 1.0
    assert len(result.trajectory_in_rig) >= 2


def test_no_renderer_data_is_handled(policy):
    """`renderer_data` is absent under an upstream alpasim runtime."""
    result = policy.drive(context(route=straight_route(), renderer_data=None))
    assert "stl_robustness" in result.debug_scalars


def test_a_curved_route_is_planned_on(policy):
    result = policy.drive(context(route=curved_route(radius_m=40.0), speed_mps=6.0))
    positions = result.trajectory_in_rig.positions
    assert np.abs(positions[:, 1]).max() > 0.5  # it actually turns


# ---------------------------------------------------------------------------
# Sensor noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("measured", [-40.0, 40.0, float("nan"), float("inf")])
def test_an_implausible_measured_acceleration_is_confined(policy, measured):
    """CARLA finite-differences velocity, so this reading is noisy.

    The plan's profile is anchored on it, which makes the opening jerk the
    difference between the first knot and this number. Left unbounded, a
    sensor transient puts `comfort` out of reach of every candidate and the
    policy brakes for no reason.
    """
    result = policy.drive(context(route=straight_route(), accel_mps2=measured))
    assert np.isfinite(result.debug_scalars["stl_robustness"])
    assert result.debug_scalars["stl_rule_comfort"] > -5.0


def test_the_acceleration_clamp_keeps_ordinary_readings_intact(policy):
    for reading in (1.2, -3.0):
        policy._anchor_accel_mps2 = None  # as at the start of a session
        assert policy._plausible_acceleration(reading) == pytest.approx(reading)
    policy._anchor_accel_mps2 = None
    assert policy._plausible_acceleration(float("nan")) == 0.0


def test_the_anchor_moves_no_faster_than_the_vehicle_could(policy):
    """What the ride feels is the change, and nothing else was bounding it.

    Each plan is jerk-limited inside itself, so nothing within one is harsh.
    What the vehicle gets is the first step of a new plan every 100 ms, and
    consecutive plans agree only as well as their anchors do. CARLA's
    acceleration is a finite difference of velocity: measured over a drive it
    moved by up to 5.9 m/s^2 between steps -- 59 m/s^3 against a comfort bound
    of 4 -- and the commanded acceleration followed it at a correlation of
    0.99, exceeding the bound on 42% of steps while every rule read as
    satisfied.
    """
    limits = CONFIG.limits
    step = limits.plan_jerk_limit() * CONFIG.horizon.step_s

    policy._anchor_accel_mps2 = None
    settled = policy._plausible_acceleration(0.0)
    assert settled == pytest.approx(0.0)

    # A reading that jumps the whole envelope in one step is not acceleration.
    jumped = policy._plausible_acceleration(-3.9)
    assert jumped == pytest.approx(-step, abs=1e-6)

    # But a real change is followed, at the rate a real vehicle would make it.
    for _ in range(40):
        following = policy._plausible_acceleration(-3.9)
    assert following == pytest.approx(-3.9, abs=0.05)

    # And the ramp never exceeds what the plan itself may command.
    policy._anchor_accel_mps2 = None
    previous = policy._plausible_acceleration(0.0)
    for reading in (3.0, -3.0, 3.0, -3.0):
        for _ in range(3):
            current = policy._plausible_acceleration(reading)
            assert abs(current - previous) <= step + 1e-6
            previous = current


# ---------------------------------------------------------------------------
# Traffic lights, end to end through the policy
# ---------------------------------------------------------------------------


def red_light(distance_m: float) -> CarlaRendererData:
    return CarlaRendererData(
        ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
        ego_traffic_light_distance_m=distance_m,
        speed_limit_mps=13.9,
    )


def test_a_red_light_just_ahead_does_not_deadlock_the_policy(policy):
    """The failure seen against real CARLA, at the level it surfaced.

    Stopped a metre short of the line, every plan that moves violates
    `stop_on_red`; if the required stop station sits behind the vehicle then
    even standing still does, the fallback latches, and the vehicle never
    moves again.
    """
    result = policy.drive(
        context(route=straight_route(), speed_mps=0.0, renderer_data=red_light(1.0))
    )
    assert result.debug_scalars["stl_rule_stop_on_red"] >= 0.0
    assert result.debug_scalars["stl_fallback"] == 0.0


def test_a_line_already_crossed_lets_the_policy_move_on(policy):
    """With the fixed runtime the distance goes negative once past the line."""
    result = policy.drive(
        context(route=straight_route(), speed_mps=2.0, renderer_data=red_light(-1.5))
    )
    assert result.debug_scalars["light_active"] == 0.0
    assert result.debug_scalars["stl_fallback"] == 0.0


def test_a_distant_red_light_still_governs(policy):
    result = policy.drive(
        context(route=straight_route(), speed_mps=6.0, renderer_data=red_light(40.0))
    )
    assert result.debug_scalars["light_active"] == 1.0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_it_is_constructible_the_way_the_upstream_cli_does_it():
    """`_cmd_serve` calls `policy_cls(cruise_speed_mps=...)`, or with nothing."""
    assert EgoPolicy().config.cruise_speed_mps == STLDriverConfig().cruise_speed_mps
    assert EgoPolicy(cruise_speed_mps=5.0).config.cruise_speed_mps == 5.0


def test_an_explicit_config_is_respected():
    config = STLDriverConfig(cruise_speed_mps=11.0)
    assert EgoPolicy(config=config).config.cruise_speed_mps == 11.0
    # And an explicit speed still wins over the config's.
    assert EgoPolicy(cruise_speed_mps=4.0, config=config).config.cruise_speed_mps == 4.0


def test_it_can_be_registered_with_the_upstream_cli():
    from carla_driver_interface.driver.policies import POLICY_REGISTRY

    from stl_driver.policy import register_with_upstream_cli

    register_with_upstream_cli()
    assert POLICY_REGISTRY[EgoPolicy.name] is EgoPolicy


def test_a_new_session_clears_the_warm_start(policy):
    policy.drive(context(route=straight_route()))
    assert policy._warm_start is not None
    policy.on_session_start(SessionState(uuid="s2", seed=1, scene_id="t", cameras={}))
    assert policy._warm_start is None


def test_actors_reach_the_policy(policy):
    """A sanity check that the perception path is wired into the step."""
    data = CarlaRendererData(
        ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_NONE,
        ego_traffic_light_distance_m=-1.0,
        speed_limit_mps=13.9,
        actors=[
            CarlaActorState(
                track_id="a",
                type_id="vehicle.tesla.model3",
                pose_local_to_aabb=PoseProto(
                    vec=Vec3(x=25.0, y=0.0, z=0.0),
                    quat=QuatProto(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
                aabb=AABB(size_x=4.5, size_y=2.0, size_z=1.6),
                dynamic_state=DynamicState(linear_velocity=Vec3(x=0.0, y=0.0, z=0.0)),
            )
        ],
    )
    result = policy.drive(context(route=straight_route(), renderer_data=data))
    assert result.debug_scalars["n_actor_discs"] >= 2.0
    assert result.debug_scalars["min_clearance_m"] < 50.0
