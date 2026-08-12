# SPDX-License-Identifier: Apache-2.0
"""Scenario fixtures shared across the tests.

The builders here produce :class:`RolloutInputs` for hand-described
situations -- a straight empty road, a stopped vehicle ahead, a red light --
so that a test can state the scenario in one line and assert on behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from stl_driver.config import STLDriverConfig
from stl_driver.frenet import build_route_table, project_to_frenet
from stl_driver.rollout import RolloutInputs


@pytest.fixture(scope="session")
def config() -> STLDriverConfig:
    return STLDriverConfig()


def straight_route(length_m: float = 300.0, spacing_m: float = 2.0) -> np.ndarray:
    """A straight route ahead in the rig frame, as the runtime would publish."""
    stations = np.arange(0.0, length_m + spacing_m, spacing_m)
    return np.stack([stations, np.zeros_like(stations), np.zeros_like(stations)], axis=1)


def curved_route(radius_m: float = 60.0, sweep_rad: float = 1.2) -> np.ndarray:
    """A constant-radius left turn starting at the rig origin."""
    angles = np.linspace(0.0, sweep_rad, 120)
    x = radius_m * np.sin(angles)
    y = radius_m * (1.0 - np.cos(angles))
    return np.stack([x, y, np.zeros_like(x)], axis=1)


def make_inputs(
    config: STLDriverConfig,
    *,
    route: np.ndarray | None = None,
    speed0_mps: float = 6.0,
    accel0_mps2: float = 0.0,
    offset0_m: float | None = None,
    actor_centres: torch.Tensor | None = None,
    actor_radii: torch.Tensor | None = None,
    lead_station_m: float = math.inf,
    lead_speed_mps: float = 0.0,
    stop_station_m: float = math.inf,
    light_active: bool = False,
    speed_limit_mps: float | None = None,
    published_length_m: float | None = None,
) -> RolloutInputs:
    """Assemble rollout inputs for a described scenario.

    Defaults describe an empty straight road at a steady 6 m/s, so a test only
    states what makes its scenario distinctive.
    """
    route = straight_route() if route is None else route
    table = build_route_table(
        route,
        ds=config.route_table_ds_m,
        length_m=config.route_table_length_m(),
    )

    if offset0_m is None:
        station, offset = project_to_frenet(table, np.zeros((1, 2)))
        station0, offset0 = float(station[0]), float(offset[0])
    else:
        station0, offset0 = 0.0, float(offset0_m)

    n_steps = config.horizon.n_steps
    if actor_centres is None:
        actor_centres = torch.zeros((n_steps, 0, 2), dtype=torch.float32)
        actor_radii = torch.zeros((0,), dtype=torch.float32)
    if actor_radii is None:
        raise ValueError("actor_radii must accompany actor_centres")

    limit = config.limits.default_speed_limit_mps if speed_limit_mps is None else speed_limit_mps
    gate = config.limits.gate_magnitude_m
    target = config.target_speed_mps(limit)

    ego_offsets, ego_radius = config.ego_footprint()

    def scalar(value: float) -> torch.Tensor:
        return torch.as_tensor(float(value), dtype=torch.float32)

    return RolloutInputs(
        route_xy=torch.from_numpy(table.xy).to(torch.float32),
        route_tangent=torch.from_numpy(table.tangent).to(torch.float32),
        route_curvature=torch.from_numpy(table.curvature).to(torch.float32),
        route_ds=table.ds,
        published_length_m=scalar(
            table.published_length_m if published_length_m is None else published_length_m
        ),
        station0_m=scalar(station0),
        offset0_m=scalar(offset0),
        speed0_mps=scalar(speed0_mps),
        accel0_mps2=scalar(accel0_mps2),
        actor_centres=actor_centres,
        actor_radii=actor_radii,
        n_actors=int(actor_radii.shape[0]) // 2,
        ego_disc_offsets=torch.tensor(ego_offsets, dtype=torch.float32),
        ego_disc_radius=ego_radius,
        lead_station_m=torch.full((n_steps,), float(lead_station_m), dtype=torch.float32),
        lead_speed_mps=torch.full((n_steps,), float(lead_speed_mps), dtype=torch.float32),
        stop_station_m=scalar(stop_station_m),
        light_gate=scalar(gate if light_active else -gate),
        speed_limit_mps=scalar(limit),
        progress_target_m=scalar(config.progress_target_m(target, speed0_mps)),
    )


def stationary_actor(
    config: STLDriverConfig,
    *,
    ahead_m: float,
    lateral_m: float = 0.0,
    radius_m: float = 1.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One stationary obstacle, as a single disc held over the horizon."""
    n_steps = config.horizon.n_steps
    centre = torch.tensor([[ahead_m, lateral_m]], dtype=torch.float32)
    return centre.expand(n_steps, 1, 2).contiguous(), torch.tensor([radius_m], dtype=torch.float32)
