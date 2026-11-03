# SPDX-License-Identifier: Apache-2.0
"""How long a step actually takes.

The runtime calls ``drive`` every 100 ms.  By default the search takes a fixed
number of refinement steps and nothing cuts it short (``time_budget_s`` is
``None``), so a slow step is late rather than shallower; a deployment that
opts into a budget gets fewer steps instead, and must still get some.  This
is not a correctness test -- but a policy that quietly stopped refining
would look fine everywhere else while driving noticeably worse, and that is
worth noticing.

Measured, not estimated.  The thresholds are loose enough to survive a busy
machine and tight enough to catch a real regression.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from carla_driver_interface.driver import DriveContext, SessionState
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.protocol import DynamicState, Vec3

from stl_driver.config import STLDriverConfig
from stl_driver.policy import EgoPolicy
from tests.conftest import curved_route

pytestmark = [pytest.mark.slow, pytest.mark.latency]

N_MEASUREMENTS = 40
#: The budget a deployment holding the 100 ms step would opt into.
OPT_IN_BUDGET_S = 0.060


def make_context(route: np.ndarray, speed_mps: float, time_now_us: int) -> DriveContext:
    """A DriveContext as the service would build it, without the gRPC."""
    session = SessionState(uuid="latency", seed=0, scene_id="bench", cameras={})
    session.route_waypoints_in_rig = np.asarray(route, dtype=np.float64)
    session.ego_trajectory = Trajectory([time_now_us], [Pose.identity()])
    session.dynamic_states = [DynamicState(linear_velocity=Vec3(x=speed_mps, y=0.0, z=0.0))]
    return DriveContext(
        session=session,
        time_now_us=time_now_us,
        time_query_us=time_now_us + 100_000,
        renderer_data=None,
    )


@pytest.fixture(scope="module")
def warm_policy() -> EgoPolicy:
    policy = EgoPolicy()
    # The same warm-up the service performs before the first observation.
    policy.on_session_start(SessionState(uuid="warm", seed=0, scene_id="warm", cameras={}))
    return policy


def measure(policy: EgoPolicy, route: np.ndarray) -> list[float]:
    durations = []
    for index in range(N_MEASUREMENTS):
        # Vary the speed so every step solves a genuinely different problem
        # rather than repeatedly hitting the same warm start.
        ctx = make_context(route, speed_mps=2.0 + 0.2 * index, time_now_us=index * 100_000)
        started = time.perf_counter()
        policy.drive(ctx)
        durations.append(time.perf_counter() - started)
    return durations


def test_a_step_fits_inside_the_policy_timestep(warm_policy):
    durations = measure(warm_policy, curved_route(radius_m=60.0))
    median_ms = statistics.median(durations) * 1000.0
    p95_ms = sorted(durations)[int(0.95 * len(durations))] * 1000.0

    print(f"\n  median {median_ms:.1f} ms, p95 {p95_ms:.1f} ms (budget 100 ms/step)")
    assert median_ms < 100.0
    assert p95_ms < 150.0


def test_the_search_fits_inside_the_policy_step(warm_policy):
    """What actually has to hold: a plan every 100 ms.

    There is no wall-clock cutoff by default -- the search is a fixed number
    of steps so that the same situation gives the same plan -- so this is the
    requirement the fixed count has to be chosen against, rather than a check
    that a cutoff fired.
    """
    ctx = make_context(curved_route(radius_m=60.0), speed_mps=7.0, time_now_us=0)
    result = warm_policy.drive(ctx)
    assert warm_policy.config.optimizer.time_budget_s is None
    assert result.debug_scalars["stl_solve_seconds"] < 0.100


def test_refinement_still_runs_under_an_opt_in_budget():
    """A deployment's budget must leave room to refine, not just to sample.

    `time_budget_s` is off by default, so this sets the 60 ms a deployment
    holding the 100 ms step would choose. A search whose sampling stage alone
    grew to fill it would still meet the deadline -- by refining nothing,
    a silent downgrade nothing else would notice.
    """
    base = STLDriverConfig()
    config = replace(base, optimizer=replace(base.optimizer, time_budget_s=OPT_IN_BUDGET_S))
    policy = EgoPolicy(config=config)
    policy.on_session_start(SessionState(uuid="budget", seed=0, scene_id="budget", cameras={}))

    ctx = make_context(curved_route(radius_m=60.0), speed_mps=7.0, time_now_us=0)
    result = policy.drive(ctx)
    assert policy.config.optimizer.time_budget_s == OPT_IN_BUDGET_S
    assert result.debug_scalars["stl_refine_steps"] >= 2.0


def test_the_first_step_after_a_session_starts_is_a_full_search_in_time():
    """The warm-up exists so that torch's start-up cost is already paid.

    Its effect cannot be observed by timing a second policy in the same
    process -- torch initialises once per process, so by the time any test
    runs it is long since warm. What *is* observable, and is what actually
    matters, is that the first real step of a session takes every refinement
    step and still lands inside the 100 ms step rather than spending it on
    start-up. (Under an opt-in budget the warm-up must also reach every
    refinement step; `test_optimizer` checks that without a clock.)
    """
    policy = EgoPolicy()
    policy.on_session_start(SessionState(uuid="w", seed=0, scene_id="w", cameras={}))

    started = time.perf_counter()
    result = policy.drive(make_context(curved_route(radius_m=60.0), 5.0, 0))
    first_ms = (time.perf_counter() - started) * 1000.0

    print(f"\n  first step after session start: {first_ms:.1f} ms")
    assert first_ms < 100.0
    # And it was the whole search: with no budget nothing may cut it short,
    # so a step that fits only by refining less is a regression too.
    assert policy.config.optimizer.time_budget_s is None
    assert result.debug_scalars["stl_refine_steps"] == policy.config.optimizer.n_refine_steps


def test_the_batched_path_is_live(warm_policy):
    """The loop fallback would roughly double the sampling stage."""
    ctx = make_context(curved_route(radius_m=60.0), speed_mps=7.0, time_now_us=0)
    result = warm_policy.drive(ctx)
    assert result.debug_scalars["stl_vmap"] == 1.0


def test_thread_count_is_pinned(warm_policy):
    """Small tensors: more threads costs more in synchronisation than it saves."""
    assert torch.get_num_threads() == warm_policy.config.optimizer.torch_threads
