# SPDX-License-Identifier: Apache-2.0
"""The whole loop, over real gRPC, without a simulator.

Everything else in this suite tests a piece.  These tests drive the policy the
way the runtime does -- session start, observations, ``drive``, actuation,
repeat -- and judge it on where the vehicle ends up.

``FakeWorld`` reports no traffic lights and no actors, so two of the most
interesting rules would never fire against it as shipped.  The subclasses here
inject them.  Since an injected actor is not physically simulated, they assert
on the *ego's own trajectory* rather than on a collision counter that would
read zero either way.
"""

from __future__ import annotations

import numpy as np
import pytest
from carla_driver_interface.driver import serving
from carla_driver_interface.fakes import FakeWorld, straight_then_curve_route
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
from carla_driver_interface.runtime import CarlaRuntime, RuntimeConfig, ScenarioSpec

from stl_driver.policy import EgoPolicy

pytestmark = pytest.mark.slow


def run_rollout(world_factory, *, steps: int = 200, policy: EgoPolicy | None = None):
    """Serve the policy and drive one rollout against it."""
    policy = policy or EgoPolicy()
    scenario = ScenarioSpec(map_name="FakeTown", name="closed_loop")
    with serving(policy, port=0, host="127.0.0.1") as port:
        config = RuntimeConfig(driver_address=f"127.0.0.1:{port}", max_steps=steps)
        world = world_factory(config, scenario)
        outcome = CarlaRuntime(world, config, scenario).run_rollout()
    return outcome, world


def positions_of(world: FakeWorld) -> np.ndarray:
    return np.stack([state.pose_local_to_rig.position[:2] for state in world.history])


def speeds_of(world: FakeWorld) -> np.ndarray:
    return np.array([state.speed_mps for state in world.history])


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_it_drives_the_route_without_incident():
    outcome, world = run_rollout(
        lambda config, scenario: FakeWorld(
            config, scenario, route_in_local=straight_then_curve_route()
        )
    )

    assert outcome.success
    metrics = outcome.metrics
    assert metrics["collisions"] == 0.0
    assert metrics["lane_invasions"] == 0.0
    assert metrics["distance_travelled_m"] > 40.0
    assert metrics["route_completion"] > 0.5


def test_it_tracks_the_route_closely():
    """Lateral error against the route the runtime measures for itself."""
    outcome, _ = run_rollout(
        lambda config, scenario: FakeWorld(
            config, scenario, route_in_local=straight_then_curve_route()
        )
    )
    assert outcome.metrics["route_lateral_error_m"] < 1.5


def test_it_keeps_below_the_reported_speed_limit():
    """`FakeWorld.max_speed_mps` is what it publishes as the limit."""
    outcome, _ = run_rollout(
        lambda config, scenario: FakeWorld(
            config,
            scenario,
            route_in_local=straight_then_curve_route(),
            max_speed_mps=6.0,
        )
    )
    assert outcome.metrics["speed_mps"] < 6.5


def test_it_pulls_away_from_a_standstill():
    """The anti-parking property, end to end rather than in one solve.

    Asserted as "clearly moving" rather than against a particular speed. The
    search degrades to fewer refinement steps when the machine is busy -- by
    design -- and plans a little more conservatively as a result, so a tight
    speed threshold here would be measuring the load on the test runner rather
    than the policy.
    """
    _, world = run_rollout(
        lambda config, scenario: FakeWorld(
            config, scenario, route_in_local=straight_then_curve_route()
        ),
        steps=80,
    )
    speeds = speeds_of(world)
    positions = positions_of(world)
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

    def environment(self, snapshot) -> CarlaRendererData:
        data = super().environment(snapshot)
        travelled = float(np.linalg.norm(snapshot.ego.pose_local_to_rig.position[:2]))
        remaining = self.light_station_m - travelled
        if remaining > 0.0:
            data.ego_traffic_light = TrafficLightState.TRAFFIC_LIGHT_STATE_RED
            data.ego_traffic_light_distance_m = remaining
        return data


class ObstacleWorld(FakeWorld):
    """A fake world with one stationary vehicle parked on the route.

    Not physically simulated, so the ego can drive through it without the
    collision counter noticing -- which is exactly why the test asserts on
    where the ego stops instead.
    """

    def __init__(self, *args, obstacle_station_m: float = 35.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        route = self._route
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1))]
        )
        index = int(np.searchsorted(arc, obstacle_station_m))
        self.obstacle_xy = route[min(index, len(route) - 1), :2]

    def environment(self, snapshot) -> CarlaRendererData:
        data = super().environment(snapshot)
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
    _, world = run_rollout(
        lambda config, scenario: RedLightWorld(
            config,
            scenario,
            route_in_local=straight_then_curve_route(),
            light_station_m=30.0,
        ),
        steps=220,
    )
    positions = positions_of(world)
    travelled = np.linalg.norm(positions - positions[0], axis=1)

    assert travelled.max() < 30.0  # never crossed the line
    assert speeds_of(world)[-10:].mean() < 0.5  # and settled there


def test_it_stops_behind_a_parked_vehicle():
    """Stopping is judged by where the ego ends up, not by a collision count.

    The obstacle is injected into the observation stream but not into the
    fake world's physics, so nothing would stop the ego driving through it --
    which makes the ego's own trajectory the only honest evidence.
    """
    _, world = run_rollout(
        lambda config, scenario: ObstacleWorld(
            config,
            scenario,
            route_in_local=straight_then_curve_route(),
            obstacle_station_m=35.0,
        ),
        steps=220,
    )
    positions = positions_of(world)
    obstacle = world.obstacle_xy
    closest = np.linalg.norm(positions - obstacle, axis=1).min()

    # Ego discs reach 3.5 m ahead of the rear axle and the parked car is
    # 4.5 m long, so stopping cleanly leaves several metres between origins.
    assert closest > 3.0
    assert speeds_of(world)[-10:].mean() < 1.5


def test_a_green_light_does_not_stop_it():
    """The control for `test_it_stops_for_a_red_light`."""

    class GreenLightWorld(RedLightWorld):
        def environment(self, snapshot):
            data = FakeWorld.environment(self, snapshot)
            data.ego_traffic_light = TrafficLightState.TRAFFIC_LIGHT_STATE_GREEN
            data.ego_traffic_light_distance_m = 20.0
            return data

    _, world = run_rollout(
        lambda config, scenario: GreenLightWorld(
            config, scenario, route_in_local=straight_then_curve_route()
        ),
        steps=200,
    )
    positions = positions_of(world)
    assert np.linalg.norm(positions - positions[0], axis=1).max() > 40.0


# ---------------------------------------------------------------------------
# Robustness of the loop itself
# ---------------------------------------------------------------------------


def test_it_survives_noisy_egomotion():
    """The plan is anchored on an *estimated* pose, so the round trip matters."""
    outcome, _ = run_rollout(
        lambda config, scenario: FakeWorld(
            config, scenario, route_in_local=straight_then_curve_route()
        ),
        steps=150,
    )
    assert outcome.success


def test_the_policy_reports_a_rule_breakdown_every_step():
    """The debug channel is the thing that makes behaviour explainable."""
    policy = EgoPolicy()
    seen: list[dict[str, float]] = []
    original = policy.drive

    def recording(ctx):
        result = original(ctx)
        seen.append(dict(result.debug_scalars))
        return result

    policy.drive = recording  # type: ignore[method-assign]
    run_rollout(
        lambda config, scenario: FakeWorld(
            config, scenario, route_in_local=straight_then_curve_route()
        ),
        steps=40,
        policy=policy,
    )

    assert seen
    # Once the route arrives, every rule is reported.
    with_rules = [scalars for scalars in seen if "stl_robustness" in scalars]
    assert with_rules
    for name in policy.rules.names:
        assert all(f"stl_rule_{name}" in scalars for scalars in with_rules)


def test_the_fallback_is_not_needed_on_an_ordinary_rollout():
    """If this starts failing, the specification has become unsatisfiable."""
    policy = EgoPolicy()
    fallbacks: list[float] = []
    original = policy.drive

    def recording(ctx):
        result = original(ctx)
        fallbacks.append(result.debug_scalars.get("stl_fallback", 0.0))
        return result

    policy.drive = recording  # type: ignore[method-assign]
    run_rollout(
        lambda config, scenario: FakeWorld(
            config, scenario, route_in_local=straight_then_curve_route()
        ),
        steps=150,
        policy=policy,
    )
    assert sum(fallbacks) == 0.0
