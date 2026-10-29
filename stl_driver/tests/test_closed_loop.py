# SPDX-License-Identifier: Apache-2.0
"""The whole loop, over real gRPC, without a simulator.

Everything else in this suite tests a piece.  These tests drive the policy the
way the runtime does -- session start, observations, ``drive``, repeat -- and
judge it on where the vehicle ends up.

The stock ``FakeWorld`` reports no traffic lights and no actors, so two of the
most interesting rules would never fire against it.  The subclasses here inject
them.  Since an injected actor is not physically simulated, they assert on the
*ego's own trajectory* rather than on a collision counter that would read zero
either way.
"""

from __future__ import annotations

import functools

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

from stl_driver.policy import EgoPolicy, warmed_up_policy
from tests.fake_world import FakeWorld, Rollout

pytestmark = pytest.mark.slow


@functools.cache
def shared_policy() -> EgoPolicy:
    """One policy for the module, warmed up as a runtime should load it.

    On a cold torch cache the compile takes longer than the loop's deadline
    for ``start_session``, and it is paid per instance; every session resets
    the per-rollout state, as it would for a runtime's scenarios.
    """
    return warmed_up_policy()


def run_rollout(world: FakeWorld | None = None, *, steps: int = 200) -> Rollout:
    """Serve the policy and drive one rollout against it."""
    return (world or FakeWorld()).run(shared_policy(), steps)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_it_drives_the_route_without_incident():
    rollout = run_rollout()

    assert not rollout.terminated_by_policy
    assert rollout.distance_travelled_m > 40.0
    assert rollout.route_completion > 0.5


def test_it_tracks_the_route_closely():
    """Lateral error against the route, measured by the loop rather than the policy."""
    assert run_rollout().max_route_error_m < 1.5


def test_it_keeps_below_the_reported_speed_limit():
    """`FakeWorld.speed_limit_mps` is what it publishes as the limit."""
    rollout = run_rollout(FakeWorld(speed_limit_mps=6.0))
    assert rollout.speeds.max() < 6.5


def test_it_pulls_away_from_a_standstill():
    """The anti-parking property, end to end rather than in one solve.

    Asserted as "clearly moving" rather than against a particular speed. The
    search degrades to fewer refinement steps when the machine is busy -- by
    design -- and plans a little more conservatively as a result, so a tight
    speed threshold here would be measuring the load on the test runner rather
    than the policy.
    """
    rollout = run_rollout(steps=80)
    speeds = rollout.speeds
    positions = rollout.positions
    assert speeds[0] < 0.5
    assert speeds[-20:].mean() > 1.5
    assert np.linalg.norm(positions[-1] - positions[0]) > 8.0


# ---------------------------------------------------------------------------
# Rules the stock fake world never exercises
# ---------------------------------------------------------------------------


class RedLightWorld(FakeWorld):
    """A fake world with a red light at a fixed station along the route.

    Stock ``FakeWorld`` always reports ``TRAFFIC_LIGHT_STATE_NONE``, so
    ``stop_on_red`` would never be evaluated against it.
    """

    def __init__(self, *args, light_station_m: float = 30.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.light_station_m = light_station_m

    def environment(self, pose: Pose, now_us: int) -> CarlaRendererData:
        data = super().environment(pose, now_us)
        travelled = float(np.linalg.norm(pose.position[:2] - self.route[0, :2]))
        remaining = self.light_station_m - travelled
        if remaining > 0.0:
            data.ego_traffic_light = TrafficLightState.TRAFFIC_LIGHT_STATE_RED
            data.ego_traffic_light_distance_m = remaining
        return data


class ObstacleWorld(FakeWorld):
    """A fake world with one stationary vehicle parked on the route.

    Not physically simulated, so the ego can drive through it -- which is
    exactly why the test asserts on where the ego stops instead.
    """

    def __init__(self, *args, obstacle_station_m: float = 35.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        index = int(np.searchsorted(self._arc, obstacle_station_m))
        self.obstacle_xy = self.route[min(index, len(self.route) - 1), :2]

    def environment(self, pose: Pose, now_us: int) -> CarlaRendererData:
        data = super().environment(pose, now_us)
        data.actors.append(
            CarlaActorState(
                track_id="parked",
                type_id="vehicle.tesla.model3",
                pose_local_to_aabb=PoseProto(
                    vec=Vec3(x=float(self.obstacle_xy[0]), y=float(self.obstacle_xy[1]), z=0.0),
                    quat=QuatProto(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
                aabb=AABB(size_x=4.5, size_y=2.0, size_z=1.6),
                dynamic_state=DynamicState(linear_velocity=Vec3(x=0.0, y=0.0, z=0.0)),
            )
        )
        return data


def test_it_stops_for_a_red_light():
    """The ego must come to rest short of the line and stay there."""
    rollout = run_rollout(RedLightWorld(light_station_m=30.0), steps=220)
    positions = rollout.positions
    travelled = np.linalg.norm(positions - positions[0], axis=1)

    assert travelled.max() < 30.0  # never crossed the line
    assert rollout.speeds[-10:].mean() < 0.5  # and settled there


def test_it_stops_behind_a_parked_vehicle():
    """Stopping is judged by where the ego ends up, not by a collision count.

    The obstacle is injected into the observation stream but not into the
    fake world, so nothing would stop the ego driving through it -- which
    makes the ego's own trajectory the only honest evidence.
    """
    world = ObstacleWorld(obstacle_station_m=35.0)
    rollout = run_rollout(world, steps=220)
    closest = np.linalg.norm(rollout.positions - world.obstacle_xy, axis=1).min()

    # Ego discs reach 3.5 m ahead of the rear axle and the parked car is
    # 4.5 m long, so stopping cleanly leaves several metres between origins.
    assert closest > 3.0
    assert rollout.speeds[-10:].mean() < 1.5


def test_a_green_light_does_not_stop_it():
    """The control for `test_it_stops_for_a_red_light`."""

    class GreenLightWorld(FakeWorld):
        def environment(self, pose: Pose, now_us: int) -> CarlaRendererData:
            data = super().environment(pose, now_us)
            data.ego_traffic_light = TrafficLightState.TRAFFIC_LIGHT_STATE_GREEN
            data.ego_traffic_light_distance_m = 20.0
            return data

    rollout = run_rollout(GreenLightWorld(), steps=200)
    positions = rollout.positions
    assert np.linalg.norm(positions - positions[0], axis=1).max() > 40.0


# ---------------------------------------------------------------------------
# What the policy reports
# ---------------------------------------------------------------------------


def test_the_policy_reports_a_rule_breakdown_every_step(monkeypatch):
    """The debug channel is the thing that makes behaviour explainable."""
    policy = shared_policy()
    seen: list[dict[str, float]] = []
    original = policy.drive

    def recording(ctx):
        result = original(ctx)
        seen.append(dict(result.debug_scalars))
        return result

    monkeypatch.setattr(policy, "drive", recording)
    run_rollout(steps=40)

    assert seen
    # Once the route arrives, every rule is reported.
    with_rules = [scalars for scalars in seen if "stl_robustness" in scalars]
    assert with_rules
    for name in policy.rules.names:
        assert all(f"stl_rule_{name}" in scalars for scalars in with_rules)


def test_the_fallback_is_not_needed_on_an_ordinary_rollout(monkeypatch):
    """If this starts failing, the specification has become unsatisfiable."""
    policy = shared_policy()
    fallbacks: list[float] = []
    original = policy.drive

    def recording(ctx):
        result = original(ctx)
        fallbacks.append(result.debug_scalars.get("stl_fallback", 0.0))
        return result

    monkeypatch.setattr(policy, "drive", recording)
    run_rollout(steps=150)
    assert sum(fallbacks) == 0.0
