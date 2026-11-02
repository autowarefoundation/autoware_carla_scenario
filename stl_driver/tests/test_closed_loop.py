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
from tests.fake_world import FakeWorld, Rollout, straight_then_curve_route

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


def test_it_stops_for_a_red_light_on_a_route_starting_ahead_of_it():
    """A route published from the actor's centre, 1.4 m ahead of the rig, as
    the runtime sends where no lane leads in (#1141).

    Planned from the route's start rather than from the ego, every plan began
    1.4 m ahead of it: each step carried the ego that far on, from rest, and
    it ran the light.
    """
    rollout = run_rollout(RedLightWorld(light_station_m=30.0, route_lead_m=1.4), steps=220)
    positions = rollout.positions
    travelled = np.linalg.norm(positions - positions[0], axis=1)

    assert travelled.max() < 30.0
    assert rollout.speeds[-10:].mean() < 0.5


class StopLineWorld(FakeWorld):
    """A light on a long straight, at a station measured along the route.

    Red from the start, or -- with ``red_with_room_m`` -- green until the
    first step at which the room left once the ego has stopped comfortably,
    the distance to the line less the policy's own `Limits.stop_room_m` for
    the ego's speed and acceleration, falls under ``red_with_room_m``. That
    puts the change where the policy cannot keep its full margin whatever
    speed the search happens to have settled at, which is what makes it a
    dilemma rather than an ordinary approach.
    """

    def __init__(
        self,
        *,
        light_station_m: float,
        red_with_room_m: float | None = None,
        state: TrafficLightState = TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
        **kwargs,
    ) -> None:
        kwargs.setdefault("route", straight_then_curve_route(straight_m=150.0))
        super().__init__(**kwargs)
        self.light_station_m = light_station_m
        self.red_with_room_m = red_with_room_m
        self.state = state
        self.room_at_change_m: float | None = None
        self.speed_at_change_mps: float | None = None
        self._speed = self._accel = 0.0

    def _observe(self, stub, session, now_us, pose, speed, accel, station) -> None:
        self._speed, self._accel = speed, accel
        super()._observe(stub, session, now_us, pose, speed, accel, station)

    def environment(self, pose: Pose, now_us: int) -> CarlaRendererData:
        data = super().environment(pose, now_us)
        station, _ = self.station_of(pose.position[:2])
        remaining = self.light_station_m - station
        if self.red_with_room_m is not None and self.room_at_change_m is None:
            limits = shared_policy().config.limits
            room = remaining - limits.stop_room_m(self._speed, self._accel)
            if room >= self.red_with_room_m:
                data.ego_traffic_light = TrafficLightState.TRAFFIC_LIGHT_STATE_GREEN
                data.ego_traffic_light_distance_m = remaining
                return data
            self.room_at_change_m = room
            self.speed_at_change_mps = self._speed
        data.ego_traffic_light = self.state
        data.ego_traffic_light_distance_m = remaining
        return data


def _fallbacks_while(monkeypatch, world: FakeWorld, steps: int) -> tuple[Rollout, list[float]]:
    """Run ``world`` and record, per step, whether the emergency fallback drove."""
    policy = shared_policy()
    fallbacks: list[float] = []
    original = policy.drive

    def recording(ctx):
        result = original(ctx)
        fallbacks.append(result.debug_scalars.get("stl_fallback", 0.0))
        return result

    monkeypatch.setattr(policy, "drive", recording)
    return run_rollout(world, steps=steps), fallbacks


def _assert_stopped_at_the_margin(rollout: Rollout, light_station_m: float) -> None:
    """At rest, the rear axle within reach of ``line - stop_line_margin_m``.

    Never past it by more than a quarter metre, and short of it by no more
    than `liveness_stop_gap_m` -- the room `makes_progress` lets a vehicle at
    rest leave -- plus the same quarter metre.
    """
    limits = shared_policy().config.limits
    target = light_station_m - limits.stop_line_margin_m
    final = rollout.stations_m[-1]
    assert rollout.speeds[-10:].mean() < 0.1
    assert max(rollout.stations_m) <= target + 0.25, (max(rollout.stations_m), target)
    assert final >= target - limits.liveness_stop_gap_m - 0.25, (final, target)


def test_it_stops_at_the_margin_on_approach():
    """The stop station must not slide forward as the vehicle closes in (#1147).

    Within two margins of the line the margin used to be half the distance
    left, so every metre crept moved the stop half a metre on: a red light
    30 m ahead stopped the rear axle at 28.1 m rather than 26 m -- the front
    bumper, 3.75 m ahead of it, nearly two metres over the line.
    """
    rollout = run_rollout(RedLightWorld(light_station_m=30.0), steps=220)
    _assert_stopped_at_the_margin(rollout, 30.0)


def test_it_stops_at_the_margin_from_cruising_speed():
    world = StopLineWorld(light_station_m=70.0, speed_limit_mps=8.0)
    rollout = run_rollout(world, steps=260)
    assert rollout.speeds.max() > 6.0  # it was at speed when it braked
    _assert_stopped_at_the_margin(rollout, 70.0)


def test_it_stays_at_the_margin_from_rest():
    """At rest eight metres out, there is four metres to cover and no more."""
    world = StopLineWorld(light_station_m=8.0)
    rollout = run_rollout(world, steps=120)
    _assert_stopped_at_the_margin(rollout, 8.0)


def test_a_light_turning_red_inside_the_margin_leaves_a_feasible_stop(monkeypatch):
    """A light that turns red where the full margin is out of reach.

    The vehicle can still stop comfortably short of the line, just not a full
    margin short of it, so the stop station has to come forward to where a
    comfortable stop ends. Asking for the whole margin -- or, as the halving
    rule did, for whatever half the distance happened to leave -- made every
    candidate violate `stop_on_red` and handed the step to the emergency
    fallback on a perfectly stoppable approach.
    """
    world = StopLineWorld(light_station_m=80.0, red_with_room_m=2.0, speed_limit_mps=8.0)
    rollout, fallbacks = _fallbacks_while(monkeypatch, world, steps=240)

    assert world.speed_at_change_mps is not None and world.speed_at_change_mps > 6.0
    assert 0.0 <= world.room_at_change_m < shared_policy().config.limits.stop_line_margin_m
    assert sum(fallbacks) == 0.0
    assert max(rollout.stations_m) < 80.0  # the rear axle never reached the line
    assert rollout.speeds[-10:].mean() < 0.1


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
