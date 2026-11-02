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
from carla_driver_interface.driver import DriveContext, SessionState
from carla_driver_interface.geometry import Pose, Trajectory
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
from stl_driver.policy import EgoPolicy, _RolloutMemory
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
    uuid: str = "s",
) -> DriveContext:
    session = SessionState(uuid=uuid, seed=0, scene_id="t", cameras={})
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
        memory = _RolloutMemory()  # as at the start of a session
        assert policy._plausible_acceleration(reading, memory) == pytest.approx(reading)
    assert policy._plausible_acceleration(float("nan"), _RolloutMemory()) == 0.0


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

    memory = _RolloutMemory()
    settled = policy._plausible_acceleration(0.0, memory)
    assert settled == pytest.approx(0.0)

    # A reading that jumps the whole envelope in one step is not acceleration.
    jumped = policy._plausible_acceleration(-3.9, memory)
    assert jumped == pytest.approx(-step, abs=1e-6)

    # But a real change is followed, at the rate a real vehicle would make it.
    for _ in range(40):
        following = policy._plausible_acceleration(-3.9, memory)
    assert following == pytest.approx(-3.9, abs=0.05)

    # And the ramp never exceeds what the plan itself may command.
    memory = _RolloutMemory()
    previous = policy._plausible_acceleration(0.0, memory)
    for reading in (3.0, -3.0, 3.0, -3.0):
        for _ in range(3):
            current = policy._plausible_acceleration(reading, memory)
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
# A route that starts ahead of the ego (#1141)
# ---------------------------------------------------------------------------

#: How far ahead of the rig a route walked from the actor's centre starts.
LEAD_M = 1.4


def route_ahead_of_the_ego() -> np.ndarray:
    route = straight_route()
    route[:, 0] += LEAD_M
    return route


def inputs_for(policy, route, *, renderer_data=None, speed_mps=0.0):
    return policy._build_inputs(
        route_in_rig=route,
        anchor=Pose.identity(),
        renderer_data=renderer_data,
        speed_mps=speed_mps,
        accel_mps2=0.0,
        memory=_RolloutMemory(),
    )


def centreline_x(inputs, station_m: float) -> float:
    """The table's ``x`` at one station; the route runs along ``x``."""
    xy = inputs.route_xy.double().numpy()
    return float(np.interp(station_m / inputs.route_ds, np.arange(len(xy)), xy[:, 0]))


def test_the_ego_behind_the_route_start_gets_its_own_station(policy):
    inputs = inputs_for(policy, route_ahead_of_the_ego())
    assert centreline_x(inputs, float(inputs.station0_m)) == pytest.approx(0.0, abs=1e-5)
    # The route's first point stays where it was published.
    assert centreline_x(inputs, float(inputs.station0_m) + LEAD_M) == pytest.approx(
        LEAD_M, abs=1e-5
    )


def test_the_red_light_stop_is_measured_from_the_ego_not_the_route_start(policy):
    """The light is 10 m ahead of the rig: the rule must hold the rear axle
    ``stop_line_margin_m`` short of it, not 1.4 m further on."""
    inputs = inputs_for(policy, route_ahead_of_the_ego(), renderer_data=red_light(10.0))
    expected = 10.0 - CONFIG.limits.stop_line_margin_m
    assert centreline_x(inputs, float(inputs.stop_station_m)) == pytest.approx(expected, abs=1e-4)


def test_a_plan_at_rest_starts_at_the_ego(policy):
    import torch

    from stl_driver.rollout import plan_xy

    inputs = inputs_for(policy, route_ahead_of_the_ego(), renderer_data=red_light(10.0))
    hold = torch.tensor([-5.0, -5.0, -5.0, -5.0, 0.0, 0.0])
    positions, _ = plan_xy(hold, inputs, CONFIG)
    assert np.allclose(positions.detach().numpy(), 0.0, atol=1e-4)

    result = policy.drive(
        context(route=route_ahead_of_the_ego(), speed_mps=0.0, renderer_data=red_light(10.0))
    )
    assert np.linalg.norm(result.trajectory_in_rig.positions[0, :2]) < 0.3


def stopped_car_ahead(x_m: float) -> CarlaRendererData:
    return CarlaRendererData(
        ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_NONE,
        ego_traffic_light_distance_m=-1.0,
        speed_limit_mps=13.9,
        actors=[
            CarlaActorState(
                track_id="a",
                type_id="vehicle.tesla.model3",
                pose_local_to_aabb=PoseProto(
                    vec=Vec3(x=x_m, y=0.0, z=0.0),
                    quat=QuatProto(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
                aabb=AABB(size_x=4.5, size_y=2.0, size_z=1.6),
                dynamic_state=DynamicState(linear_velocity=Vec3(x=0.0, y=0.0, z=0.0)),
            )
        ],
    )


@pytest.mark.parametrize("speed_mps", [2.0, 3.0])
def test_an_emergency_stop_starts_at_the_ego(policy, speed_mps):
    """A car stopped 5 m ahead leaves no plan that keeps clear, so the policy
    brakes. With the route starting 1.4 m ahead of the rig, the table reaches
    back behind the ego; the stop is laid out from the ego's station, not
    from the table's first point, which would put every pose behind it."""
    result = policy.drive(
        context(
            route=route_ahead_of_the_ego(),
            speed_mps=speed_mps,
            renderer_data=stopped_car_ahead(5.0),
        )
    )
    assert result.debug_scalars["stl_fallback"] == 1.0
    positions = result.trajectory_in_rig.positions
    dt = CONFIG.horizon.step_s
    first_step = (speed_mps - CONFIG.limits.emergency_decel_mps2 * dt) * dt
    assert positions[0, 0] == pytest.approx(first_step, abs=1e-6)
    assert np.all(np.diff(positions[:, 0]) >= 0.0)
    assert positions[-1, 0] == pytest.approx(
        speed_mps**2 / (2.0 * CONFIG.limits.emergency_decel_mps2), abs=speed_mps * dt
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_it_is_constructible_with_a_cruise_speed_or_without():
    """`--policy-arg cruise_speed_mps=...` upstream, or nothing."""
    assert EgoPolicy().config.cruise_speed_mps == STLDriverConfig().cruise_speed_mps
    assert EgoPolicy(cruise_speed_mps=5.0).config.cruise_speed_mps == 5.0


def test_an_explicit_config_is_respected():
    config = STLDriverConfig(cruise_speed_mps=11.0)
    assert EgoPolicy(config=config).config.cruise_speed_mps == 11.0
    # And an explicit speed still wins over the config's.
    assert EgoPolicy(cruise_speed_mps=4.0, config=config).config.cruise_speed_mps == 4.0


def test_it_loads_compiled_by_the_spec_a_scenario_names():
    """`driver.policy=stl_driver.policy:warmed_up_policy` in autoware_carla_scenario
    (and `carla-driver-interface serve --policy` upstream) loads it this way, and it
    compiles while it loads, not inside the first `start_session` and its deadline."""
    from carla_driver_interface.policies import load_policy

    policy = load_policy("stl_driver.policy:warmed_up_policy", cruise_speed_mps=5.0)
    assert isinstance(policy, EgoPolicy)
    assert policy.config.cruise_speed_mps == 5.0
    assert policy.optimizer._compiled_batch is not None


def test_a_new_session_does_not_compile_again(policy):
    compiled = policy.optimizer._compiled_batch
    assert compiled is not None
    policy.on_session_start(SessionState(uuid="again", seed=0, scene_id="t", cameras={}))
    assert policy.optimizer._compiled_batch is compiled


def test_a_new_session_clears_the_warm_start(policy):
    policy.drive(context(route=straight_route()))
    assert policy._memory("s").warm_start is not None
    policy.on_session_start(SessionState(uuid="s", seed=1, scene_id="t", cameras={}))
    assert policy._memory("s").warm_start is None


# ---------------------------------------------------------------------------
# Concurrent sessions on one policy (#1158)
#
# The driver service serves every session on one policy, `stl-driver serve`
# with eight workers. Step-to-step state is the rollout's, not the policy's.
# ---------------------------------------------------------------------------


def session(uuid: str) -> SessionState:
    return SessionState(uuid=uuid, seed=0, scene_id="t", cameras={})


def test_interleaved_sessions_keep_their_own_acceleration_anchor(policy):
    """A cruising rollout must not be anchored at another one's braking."""
    policy.on_session_start(session("anchor-a"))
    policy.on_session_start(session("anchor-b"))
    for _ in range(20):
        braking = policy._plausible_acceleration(-2.5, policy._memory("anchor-a"))
    assert braking == pytest.approx(-2.5)

    # B's first reading is its own, within the envelope: nothing to rate-limit
    # it from, as at the start of any session.
    cruising = policy._plausible_acceleration(1.0, policy._memory("anchor-b"))
    assert cruising == pytest.approx(1.0)
    # And A carries on from where it was, not from B.
    assert policy._plausible_acceleration(-2.5, policy._memory("anchor-a")) == pytest.approx(-2.5)


def test_interleaved_drives_keep_their_own_warm_start_and_light(policy):
    policy.on_session_start(session("drive-a"))
    policy.on_session_start(session("drive-b"))

    policy.drive(
        context(
            route=straight_route(), speed_mps=6.0, renderer_data=red_light(40.0), uuid="drive-a"
        )
    )
    held = policy._memory("drive-a")
    assert held.light is not None and held.light.active
    a_warm_start = held.warm_start
    assert a_warm_start is not None
    # B has not driven: it starts as a fresh session does, A's state unseen.
    fresh = policy._memory("drive-b")
    assert fresh.warm_start is None
    assert fresh.light is None
    assert fresh.anchor_accel_mps2 is None

    policy.drive(context(route=straight_route(), speed_mps=3.0, uuid="drive-b"))
    other = policy._memory("drive-b")
    assert other.light is not None and not other.light.active
    assert other.warm_start is not None
    # B's step changed nothing of A's.
    assert policy._memory("drive-a").light is held.light
    assert policy._memory("drive-a").warm_start is a_warm_start


def test_closing_or_starting_one_session_leaves_the_other_alone(policy):
    policy.on_session_start(session("keep"))
    policy.drive(
        context(
            route=straight_route(),
            speed_mps=6.0,
            accel_mps2=-1.0,
            renderer_data=red_light(40.0),
            uuid="keep",
        )
    )
    kept = policy._memory("keep")
    before = (kept.warm_start, kept.anchor_accel_mps2, kept.light)
    assert all(value is not None for value in before)

    policy.on_session_start(session("other"))
    policy.drive(context(route=straight_route(), accel_mps2=1.0, uuid="other"))
    policy.on_session_close(session("other"))
    assert "other" not in policy._rollouts

    after = policy._memory("keep")
    assert after is kept
    assert (after.warm_start, after.anchor_accel_mps2, after.light) == before


def test_closing_a_session_forgets_it_and_the_memory_is_bounded():
    driver = EgoPolicy()
    driver.max_remembered_rollouts = 3
    for index in range(5):
        driver.on_session_start(session(f"never-closed-{index}"))
    # The least recently driven go first.
    assert list(driver._rollouts) == [f"never-closed-{index}" for index in (2, 3, 4)]
    driver._memory("never-closed-2")  # driving one makes it the most recent
    driver.on_session_start(session("newest"))
    assert list(driver._rollouts) == ["never-closed-4", "never-closed-2", "newest"]

    driver.on_session_close(session("newest"))
    driver.on_session_close(session("unknown"))  # closing twice is harmless
    assert list(driver._rollouts) == ["never-closed-4", "never-closed-2"]


def test_sessions_on_concurrent_threads_follow_their_own_readings(policy):
    """The service drives sessions from concurrent gRPC handler threads."""
    import threading

    readings = {f"thread-{index}": value for index, value in enumerate((-3.0, -1.0, 0.5, 1.5))}
    for uuid in readings:
        policy.on_session_start(session(uuid))
    barrier = threading.Barrier(len(readings))
    failures: list[str] = []

    def run(uuid: str, reading: float) -> None:
        barrier.wait()
        previous = None
        for _ in range(400):
            anchored = policy._plausible_acceleration(reading, policy._memory(uuid))
            if previous is None:
                if anchored != pytest.approx(reading):
                    failures.append(f"{uuid}: first anchor {anchored} != {reading}")
            elif abs(anchored - previous) > 1e-9:
                failures.append(f"{uuid}: anchor moved {previous} -> {anchored}")
            previous = anchored

    threads = [threading.Thread(target=run, args=item) for item in readings.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures[:5]


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
