"""Unit tests for Autoware's lateral chain (driver/autoware_lateral.py): the config,
the MPC, the TSO, the publish delay and the localization lag."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from autoware_carla_scenario.driver.autoware_lateral import (
    DEFAULT_MPC_PARAM_FILE,
    DEFAULT_TSO_PARAM_FILE,
    AutowareLateralChain,
    LateralChainConfig,
    MpcLateralController,
    MpcReference,
    load_mpc_params,
    load_tso_params,
    tso_optimized_points,
)
from autoware_carla_scenario.driver.base import DriverClientConfig
from autoware_carla_scenario.driver.geometry import Pose, Trajectory
from autoware_carla_scenario.scenario_config import DriverLateralSpec

_L = 2.75


def _straight(speed: float = 5.0, seconds: float = 6.0, dt: float = 0.1) -> np.ndarray:
    """A node trajectory along +x: x, y, yaw, v, a, t, steer, yaw rate."""
    t = dt * np.arange(round(seconds / dt) + 1)
    pts = np.zeros((len(t), 8))
    pts[:, 0] = speed * t
    pts[:, 3] = speed
    pts[:, 5] = t
    return pts


def _arc(
    radius: float, speed: float = 5.0, seconds: float = 6.0, dt: float = 0.1
) -> np.ndarray:
    """A node trajectory turning left on a circle of *radius* from the origin, heading +x."""
    t = dt * np.arange(round(seconds / dt) + 1)
    phi = speed * t / radius
    pts = np.zeros((len(t), 8))
    pts[:, 0] = radius * np.sin(phi)
    pts[:, 1] = radius * (1.0 - np.cos(phi))
    pts[:, 2] = phi
    pts[:, 3] = speed
    pts[:, 5] = t
    return pts


def _plan(pts: np.ndarray, stamp_us: int) -> Trajectory:
    """*pts* as a policy's plan stamped *stamp_us*, the anchor at the stamp first."""
    plan = Trajectory.empty()
    for row in pts:
        plan.append(
            stamp_us + int(round(row[5] * 1e6)),
            Pose.from_xyz_yaw(float(row[0]), float(row[1]), 0.0, float(row[2])),
        )
    return plan


# -- config -----------------------------------------------------------------------


def test_the_default_is_pure_pursuit() -> None:
    config = LateralChainConfig.from_mapping(None)
    assert config == LateralChainConfig()
    assert not config.enabled


def test_the_schema_mirrors_the_config() -> None:
    spec = {f.name: f.default for f in dataclasses.fields(DriverLateralSpec)}
    runtime = {f.name: f.default for f in dataclasses.fields(LateralChainConfig)}
    assert spec == runtime


def test_the_config_names_the_key_it_refuses() -> None:
    with pytest.raises(ValueError, match="^controller must be one of"):
        LateralChainConfig.from_mapping({"controller": "stanley"})
    with pytest.raises(ValueError, match="^publish_delay_s must be zero or more"):
        LateralChainConfig.from_mapping({"publish_delay_s": -0.1})
    with pytest.raises(ValueError, match="^wheelbase_m must be a number"):
        LateralChainConfig.from_mapping({"wheelbase_m": True})
    with pytest.raises(ValueError, match="^lookahead: Unknown LateralChainConfig"):
        LateralChainConfig.from_mapping({"lookahead": 1.0})
    with pytest.raises(ValueError, match="^must be a mapping"):
        LateralChainConfig.from_mapping("mpc")  # type: ignore[arg-type]


def test_an_int_is_taken_as_a_float() -> None:
    config = LateralChainConfig.from_mapping(
        {"controller": "mpc", "publish_delay_s": 0}
    )
    assert config.enabled
    assert config.publish_delay_s == 0.0
    assert isinstance(config.publish_delay_s, float)


def test_the_driver_client_ignores_the_lateral_section() -> None:
    DriverClientConfig.from_mapping({"lateral": {"controller": "mpc"}})


def test_the_bundled_parameter_files_load() -> None:
    mpc, stop = load_mpc_params(DEFAULT_MPC_PARAM_FILE)
    assert (mpc.N, mpc.dt, mpc.input_delay) == (30, 0.1, 0.3)
    assert mpc.low_k == 0.03
    assert stop.slew
    tso = load_tso_params(DEFAULT_TSO_PARAM_FILE)
    assert (tso.w_lon, tso.w_dr, tso.a_lat_max) == (0.1, 1.0, 1.0)
    assert tso.tc is not None and tso.tc["lat"] == 2.0


# -- MPC --------------------------------------------------------------------------


def test_the_mpc_steers_back_onto_the_path() -> None:
    params, _ = load_mpc_params(DEFAULT_MPC_PARAM_FILE)
    reference = MpcReference(_straight(), 0.0, params)
    left = MpcLateralController(params)
    left.reset(0.0)
    right = MpcLateralController(params)
    right.reset(0.0)
    # 0.5 m left of the path steers right, 0.5 m right steers left.
    assert left.step(reference, 0.1, (0.5, 0.5, 0.0), 0.0) < 0.0
    assert right.step(reference, 0.1, (0.5, -0.5, 0.0), 0.0) > 0.0
    assert left.last["lat"] == pytest.approx(0.5)


def test_the_mpc_feeds_forward_a_curve() -> None:
    params, _ = load_mpc_params(DEFAULT_MPC_PARAM_FILE)
    radius = 20.0
    reference = MpcReference(_arc(radius), 0.0, params)
    mpc = MpcLateralController(params)
    steer = math.atan(_L / radius)
    mpc.reset(steer)
    command = steer
    for _ in range(20):
        command = mpc.step(reference, 0.0, (0.0, 0.0, 0.0), steer)
    assert command == pytest.approx(steer, abs=0.03)


# -- TSO --------------------------------------------------------------------------


def test_the_tso_puts_the_curve_s_steering_on_the_plan() -> None:
    q = load_tso_params(DEFAULT_TSO_PARAM_FILE)
    q.bounds = True
    q.upstream_short = True
    q.v0_from_ego = True
    radius = 30.0  # 5 m/s: 0.83 m/s^2, inside the 1.0 m/s^2 bound
    steer = math.atan(_L / radius)
    out, _, _, status = tso_optimized_points(q, _arc(radius), 5.0, steer, 0.0, _L)
    assert status > 0
    assert out.shape[1] == 8
    assert out[0, 5] == pytest.approx(0.1)
    assert np.allclose(out[5:40, 6], steer, atol=0.01)
    assert np.allclose(out[:, 3], 5.0, atol=0.2)


# -- the chain --------------------------------------------------------------------


@pytest.mark.parametrize("optimizer", ["none", "tso"])
def test_a_plan_is_steered_once_it_is_published(optimizer: str) -> None:
    config = LateralChainConfig(controller="mpc", trajectory_optimizer=optimizer)
    chain = AutowareLateralChain(config)
    stamp = 1_000_000
    pose = Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0)
    chain.submit(_plan(_arc(20.0), stamp), stamp, pose, 5.0, 0.0, 0.0)
    # Published 78 ms after its stamp: not at the next 50 ms tick, at the one after.
    assert chain.step(stamp + 50_000, pose, 5.0, 0.0) is None
    assert chain.published(stamp + 50_000) is None
    command = chain.step(stamp + 100_000, pose, 5.0, 0.0)
    assert command is not None and command > 0.0
    published = chain.published(stamp + 100_000)
    assert published is not None
    assert published.trajectory.timestamps_us[0] == stamp
    if optimizer == "tso":
        assert published.velocities is not None
        assert published.points.shape[1] == 8
        assert np.any(published.points[:, 6] > 0.05)
    else:
        assert published.velocities is None


def test_the_mpc_ticks_on_its_own_clock() -> None:
    config = LateralChainConfig(
        controller="mpc", trajectory_optimizer="none", publish_delay_s=0.0
    )
    chain = AutowareLateralChain(config)
    pose = Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0)
    chain.submit(_plan(_straight(), 0), 0, pose, 5.0, 0.0, 0.0)
    for k in range(21):  # 1 s of 50 ms ticks
        chain.step(k * 50_000, pose, 5.0, 0.0)
    # 0, 0.03, ..., 0.99: 34 MPC ticks in a second of CARLA's 21.
    assert len(chain.ticks) == 34


def test_the_localization_trails_the_vehicle() -> None:
    config = LateralChainConfig(controller="mpc", localization_lag_s=0.03)
    chain = AutowareLateralChain(config)
    chain._localization(0.0, Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0))
    x, y, yaw = chain._localization(0.05, Pose.from_xyz_yaw(1.0, 0.5, 0.0, 0.1))
    assert (x, y, yaw) == pytest.approx((0.4, 0.2, 0.04))


def test_reset_forgets_the_plans() -> None:
    chain = AutowareLateralChain(LateralChainConfig(controller="mpc"))
    pose = Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0)
    chain.submit(_plan(_straight(), 0), 0, pose, 5.0, 0.0, 0.0)
    chain.reset()
    assert chain.step(500_000, pose, 5.0, 0.0) is None
